"""CTIM-G -- Algorithm 2's community decomposition, scored on the GLOBAL graph.

This is a *proposed variant*, not part of the paper.  ``ctim/ctim.py`` stays the
faithful transcription of Algorithm 2; this module changes exactly one thing
about it and reports the consequences.  Cf. ``DEVIATIONS.md`` Section 4.7, which
already lists ``ctim/ea_dp.py``, ``ctim/ris_imm.py`` and ``ctim/local_search.py``
as non-paper modules shipping inside the package; this is a fourth one of the
same kind.  Nothing in the results path imports it.

The blind spot being repaired
-----------------------------
Algorithm 2 line 34 reads

    dI_m = max_{u in c_m} ( I_m(S u {u}) - I_m(S) )

where ``I_m`` is Eq (18) evaluated on community ``m``'s **node-induced
subgraph** -- ``MIA(..., nodes=members)``, which keeps only arcs with *both*
endpoints inside ``c_m``.  Every arc leaving the community is deleted before the
candidate is scored.  Therefore **all cross-community influence is invisible to
seed selection**, and a bridge node -- weak inside its own community, strong
globally -- is systematically undervalued.  Measured on Digg: the best node of
community c52 scores 34.16 on its induced subgraph and 237.25 on the whole
graph, so line 34 sees 14% of that node's real influence.  Eq (19)'s partition
cuts 74.3% of the *live* arcs on Digg, which is how much of the graph line 34
is throwing away.

The fix is not to abandon the decomposition -- the decomposition is what makes
CTIM fast (measured x12.9 vs GlobalGreedy on Digg, x14.7 on Yelp) -- but to keep
the decomposition and delete only the truncation:

  * partition with Eq (19) exactly as Algorithm 2 does, so the candidate pools
    and the per-community budget allocation are unchanged;
  * score every candidate against ONE global ``MIA(n, out_adj, in_adj, pp, h)``;
  * allocate the budget across communities with the exact ``O(C K^2)``
    resource-allocation DP of ``ctim.ea_dp.allocate_exact``.

What breaks, and how it is handled (the honest part)
----------------------------------------------------
Algorithm 2's decomposition is only sound because ``I_m`` and ``I_m'`` are
functions of *disjoint* subgraphs: a seed in community A cannot change the
marginal gain of a candidate in community B, so the per-community curves are
independent and a DP over them is meaningful.  Scoring globally destroys that:
``ap(v | S)`` of Eq (17) is shared, so the curves overlap and the DP is now
optimising an *upper bound*

    sum_m I(S_m)  >=  I( union_m S_m )                                # Eq (18)

(submodularity; equality iff no two communities' arborescences overlap).

**The design implemented here is (a), two-pass**, chosen over (b) interleaved:

  1. *curve build* -- community ``m``'s curve ``I_m(j)`` is built by a lazy
     (CELF) greedy restricted to ``c_m`` but evaluated on the global MIA,
     **always against an empty seed set**.  The curves are then independent *by
     construction* -- each is a function of ``c_m`` alone -- which is exactly the
     property ``allocate_exact`` needs.  They are optimistic, not wrong.
  2. *DP* -- ``allocate_exact`` turns the curves into a budget ``alloc[m]``.
  3. *realisation* -- one global lazy greedy over the union of the allocated
     communities' members, with ``alloc[m]`` as a hard per-community quota, so
     the seeds are chosen against the true shared ``ap(v|S)`` and the returned
     value is a real ``I(S)``, not a sum of curve points.

Note that step 3 *is* design (b): a single global greedy constrained by the DP's
quotas.  (b) alone cannot exist -- it needs an allocation, which needs curves --
so (a) subsumes it, and implementing (a) yields (b)'s realisation for free.  The
one thing (a) buys on top is a *measurement*: ``stats["independence_gap"]``
compares the DP's predicted value against the realised ``I(S)``, which is
precisely the amount by which the independence assumption was violated.  A naive
implementation would have returned the concatenated curve sets and silently
reported the optimistic number; both are computed here and the better set is
returned, with ``stats["realisation_fallback"]`` recording which won.

Cost.  The decomposition's only reason to exist is speed, so the per-phase
wall-clock (``detect``/``solo``/``curve``/``dp``/``realise``/``repair``) is
reported unconditionally.  If CTIM-G ends up as slow as GlobalGreedy it has no
reason to exist, and the numbers must say so.

``repair=True`` additionally runs ``ctim.local_search.local_search`` on the
result with the same global MIA, so a benchmark can separate "fixing line 34 at
the source" from "fixing its output after the fact".

Standard library only.  Python 3.9 compatible.  All randomness flows through the
explicit ``random.Random`` passed in; selection itself is fully deterministic --
``rng`` reaches only the repair phase, where it permutes scan order and hence
breaks ties only.
"""

from __future__ import annotations

import heapq
import os
import sys
import time

if __name__ == "__main__" and __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctim.ctim import detect_communities  # noqa: E402
from ctim.ea_dp import CommunityCurve, allocate_exact  # noqa: E402
from ctim.influence import MIA  # noqa: E402
from ctim.local_search import local_search, solo_influence_all, top_candidates  # noqa: E402

__all__ = [
    "celf_select",
    "build_global_curves",
    "realise_quota_greedy",
    "select_seeds_global",
]

# Tolerance for deciding whether a lazily-cached heap key is stale.  Same value
# and same role as ``ctim.ctim._EPS``.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# lazy (CELF) greedy on a *global* MIA, optionally under per-community quotas
# ---------------------------------------------------------------------------


def celf_select(mia, candidates, budget, solo, comm=None, quota=None):
    """Lazy greedy maximisation of Eq (18) over `candidates`, on the whole graph.

    Returns ``(picked, gains, n_gain_evals, n_quota_blocks)`` where ``gains[j]``
    is the exact marginal gain of ``picked[j]`` given ``picked[:j]``, so
    ``sum(gains[:j])`` is exactly ``I(picked[:j])`` (Eq (18)).

    This replaces ``MIA.greedy_incremental``'s eagerly-maintained ``IncInf``
    table with the lazy-evaluation variant, for one reason: on the *global* MIA
    the eager initialisation costs a full pass over every node's MIIA, and this
    module needs a greedy per community.  The lazy form initialises from
    ``solo[u] = I({u})``, which ``ctim.local_search.solo_influence_all``
    computes once for all of V in closed form (one forward Dijkstra per node, no
    MIIA construction) and which this module then shares across every community.

    ``solo[u]`` is not a heuristic key: it is the *exact* marginal gain of ``u``
    at ``S = empty``, and by submodularity of Eq (18) a valid upper bound on
    ``u``'s gain at any later ``S``.  A popped entry is accepted only when its
    key equals a freshly recomputed exact gain, and ties are broken by the node
    id (the heap's secondary key) -- byte-identical to the rule in
    ``MIA.greedy_incremental`` and ``_CommunitySeedState.best()``, which is what
    lets the self-test assert exact agreement with the former.

    ``quota`` (with ``comm``) caps how many seeds each community may contribute:
    this is the realisation pass of design (b), the DP allocation used as a hard
    constraint on one single global greedy.
    """
    heap = [(-solo[u], u) for u in candidates]
    heapq.heapify(heap)

    picked = []
    S_set = set()
    gains = []
    used = {}
    cache = {}          # u -> exact gain at the CURRENT S; invalidated on accept
    n_evals = 0
    n_blocked = 0

    while len(picked) < budget and heap:
        negk, u = heap[0]

        if quota is not None:
            cu = comm[u]
            if used.get(cu, 0) >= quota.get(cu, 0):
                heapq.heappop(heap)   # community full: u can never be selected
                n_blocked += 1
                continue

        g = cache.get(u)
        if g is None:
            # Eq (17)/(18) restricted to MIOA(u,h) -- the only nodes u can move.
            g = mia.marginal_gain(S_set, u) if S_set else solo[u]
            cache[u] = g
            n_evals += 1
        if -negk > g + _EPS or -negk < g - _EPS:
            heapq.heapreplace(heap, (-g, u))   # stale bound: correct it in place
            continue

        if g <= 0.0 and picked:
            break   # nothing left to add (same stopping rule as greedy_incremental)

        heapq.heappop(heap)
        picked.append(u)
        S_set.add(u)
        gains.append(g)
        if quota is not None:
            used[comm[u]] = used.get(comm[u], 0) + 1
        cache = {}

    return picked, gains, n_evals, n_blocked


# ---------------------------------------------------------------------------
# phase 1 -- per-community curves from GLOBAL marginal gains
# ---------------------------------------------------------------------------


def build_global_curves(comm, ds, pp, K, h, mia=None, solo=None, verbose=False):
    """Curve ``I_m(j)`` per community, every gain measured on the whole graph.

    Structurally this is ``ctim.ea_dp.build_curves`` with one line changed: the
    evaluator is the single global ``MIA`` instead of a fresh
    ``MIA(..., nodes=members)`` per community.  The candidate pool is still
    exactly ``c_m`` (Algorithm 2 line 34's ``u in c_m``), so the decomposition
    -- which is what makes the method fast -- is untouched; only the truncation
    of the *objective* to the induced subgraph is removed.

    Every curve is built against an empty seed set, so ``I_m`` depends on ``c_m``
    alone and the curves are mutually independent, which is the precondition
    ``allocate_exact`` needs.  They are jointly optimistic -- see the module
    docstring -- and ``select_seeds_global`` measures by how much.

    Returns ``(curves, mia, solo, stats)``; `mia` and `solo` are returned so the
    caller can reuse them for the realisation and repair phases.
    """
    n_comm = (max(comm) + 1) if len(comm) else 0
    members = [[] for _ in range(n_comm)]
    n = min(len(comm), ds.n_users)
    for v in range(n):
        members[comm[v]].append(v)

    t0 = time.perf_counter()
    if mia is None:
        # ONE evaluator for the whole run -- the entire point of the module.
        mia = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)
    t_mia = time.perf_counter() - t0

    t0 = time.perf_counter()
    if solo is None:
        solo = solo_influence_all(mia)   # I({u}) for every u, Eq (18) closed form
    t_solo = time.perf_counter() - t0

    t0 = time.perf_counter()
    curves = []
    n_evals = 0
    for m in range(n_comm):
        mem = members[m]
        if not mem:
            continue
        picked, gains, ev, _blocked = celf_select(
            mia, mem, min(K, len(mem)), solo)   # Algorithm 2 line 34, global I
        n_evals += ev
        cur = CommunityCurve(m, mem, mia)
        total = 0.0
        run = []
        for j, u in enumerate(picked):
            run.append(u)
            total += gains[j]      # exact marginal gains -> I_m(S_j) on G
            cur.I.append(total)
            cur.sets.append(list(run))
        cur.repair_monotone()
        curves.append(cur)
    t_curve = time.perf_counter() - t0

    if verbose:
        print("    [curves] %d non-empty communities  mia %.2fs  solo %.2fs  "
              "greedy %.2fs  (%d gain evals)"
              % (len(curves), t_mia, t_solo, t_curve, n_evals))
    return curves, mia, solo, {
        "mia_seconds": t_mia,
        "solo_seconds": t_solo,
        "curve_seconds": t_curve,
        "curve_gain_evals": n_evals,
        "n_communities": len(curves),
    }


# ---------------------------------------------------------------------------
# phase 3 -- realisation under the DP's quotas
# ---------------------------------------------------------------------------


def realise_quota_greedy(mia, comm, alloc, K, solo, members_of=None):
    """One global greedy over the allocated communities, capped by ``alloc``.

    This is where the curve-independence fiction is paid back: the seeds are
    chosen against the *shared* ``ap(v|S)`` of Eq (17), so overlaps between
    communities are counted once, not twice.  The per-community budgets are
    still exactly what the DP decided, so the community decomposition still
    determines the shape of the answer.

    Returns ``(seeds, gains, n_gain_evals, n_quota_blocks)``.
    """
    if not alloc:
        return [], [], 0, 0
    if members_of is None:
        members_of = {}
        for v in range(len(comm)):
            members_of.setdefault(comm[v], []).append(v)
    cand = []
    for m in sorted(alloc):
        if alloc[m] > 0:
            cand.extend(members_of.get(m, ()))
    budget = min(int(K), sum(alloc.values()))
    return celf_select(mia, cand, budget, solo, comm=comm, quota=alloc)


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


def select_seeds_global(model, ds, pp, K, rng, h=0.1, comm=None,
                        edge_weights=None, item=None, repair=False,
                        mia=None, solo=None, repair_pool=0, repair_rounds=25,
                        repair_budget=None, verbose=False):
    """CTIM-G: Eq (19) communities + global gains + exact DP + quota realisation.

    Returns ``(seeds, stats)``.

    Parameters
    ----------
    model         a `ctim.gibbs.Model`; only ``.pi`` is read, and only when
                  `comm` is not supplied (Eq (19))
    ds            a `ctim.dataset.Dataset` (``.n_users``, ``.out_adj``, ``.in_adj``)
    pp            ``{(u,v): P(v|i,u)}`` per Eq (12); pass ``None`` together with
                  `edge_weights` and `item` to have it built here
    K             seed-set size
    rng           explicit `random.Random`.  Selection is deterministic and does
                  not consume it; it is forwarded to the repair phase, where it
                  permutes scan order and therefore breaks ties only.
    comm          optional injected partition ``comm[v] in [0,C)``; default is
                  Eq (19) ``argmax_c pi[v][c]``, i.e. exactly Algorithm 2's
    repair        also run `ctim.local_search.local_search` on the result, with
                  this same global MIA (never returns a worse set, by that
                  function's unconditional guarantee)
    mia, solo     optional pre-built global evaluator and ``{u: I({u})}`` table,
                  to share them across several calls
    repair_pool   0 (default) = offer every node to the repair, which is what
                  makes its "certified 1-swap local optimum" claim complete;
                  ``n > 0`` = only the top ``n`` by ``I({u})``

    ``stats`` carries a per-phase wall-clock breakdown and the independence
    diagnostics described in the module docstring.
    """
    t_start = time.perf_counter()
    K = int(K)

    stats = {
        "method": "ctim-global",
        "design": "two-pass (a): independent global curves -> exact DP -> "
                  "quota-constrained global greedy (which is design (b))",
        "K": K,
        "h": h,
        "weights_seconds": 0.0,
        "detect_seconds": 0.0,
        "mia_seconds": 0.0,
        "solo_seconds": 0.0,
        "curve_seconds": 0.0,
        "dp_seconds": 0.0,
        "realise_seconds": 0.0,
        "repair_seconds": 0.0,
        "eval_seconds": 0.0,
        "total_seconds": 0.0,
        "n_communities": 0,
        "n_communities_used": 0,
        "allocation": {},
        "curve_gain_evals": 0,
        "realise_gain_evals": 0,
        "quota_blocks": 0,
        "dp_value": 0.0,
        "concat_value": 0.0,
        "realised_value": 0.0,
        "value": 0.0,
        "independence_gap": 0.0,
        "independence_gap_pct": 0.0,
        "independence_violated": False,
        "realisation_fallback": False,
        "realisation_gain_over_concat": 0.0,
        "repaired": False,
        "repair_swaps": 0,
        "repair_gain": 0.0,
        "n_seeds": 0,
    }

    if pp is None:
        if edge_weights is None or item is None:
            raise ValueError("pass `pp`, or both `edge_weights` and `item`")
        t0 = time.perf_counter()
        pp = edge_weights.for_item(item)  # Eq (12)
        stats["weights_seconds"] = time.perf_counter() - t0
    if K <= 0:
        stats["total_seconds"] = time.perf_counter() - t_start
        return [], stats

    # ------------------------------------------------------------ Eq (19)
    t0 = time.perf_counter()
    if comm is None:
        comm = detect_communities(model.pi)  # Eq (19), Algorithm 2 lines 22-24
    stats["detect_seconds"] = time.perf_counter() - t0

    # ------------------------------------------------- phase 1: the curves
    curves, mia, solo, cstats = build_global_curves(
        comm, ds, pp, K, h, mia=mia, solo=solo, verbose=verbose)
    for key in ("mia_seconds", "solo_seconds", "curve_seconds",
                "curve_gain_evals", "n_communities"):
        stats[key] = cstats[key]

    # ---------------------------------------------------- phase 2: the DP
    t0 = time.perf_counter()
    concat, dp_value, alloc = allocate_exact(curves, K)   # exact O(C K^2)
    stats["dp_seconds"] = time.perf_counter() - t0
    stats["dp_value"] = dp_value
    stats["allocation"] = dict(sorted(alloc.items()))
    stats["n_communities_used"] = len(alloc)

    # -------------------------------------------- phase 3: the realisation
    t0 = time.perf_counter()
    members_of = {}
    for cur in curves:
        members_of[cur.m] = cur.members
    seeds, _gains, ev, blocked = realise_quota_greedy(
        mia, comm, alloc, K, solo, members_of=members_of)
    stats["realise_seconds"] = time.perf_counter() - t0
    stats["realise_gain_evals"] = ev
    stats["quota_blocks"] = blocked

    # The DP's value is a sum over curves built independently; the realised
    # value is one Eq (18) evaluation of the actual union.  Their difference IS
    # the violation of the independence assumption -- report it, do not hide it.
    t0 = time.perf_counter()
    realised = mia.influence(seeds)          # Eq (18), global
    concat_value = mia.influence(concat)     # what a naive implementation returns
    stats["eval_seconds"] = time.perf_counter() - t0
    stats["realised_value"] = realised
    stats["concat_value"] = concat_value
    stats["realisation_gain_over_concat"] = realised - concat_value

    if concat_value > realised + 1e-12:
        # The quota greedy is itself only greedy, so it is not guaranteed to beat
        # the concatenated curve sets.  Keep the better of the two so CTIM-G can
        # never score below the naive reading of its own DP.
        seeds = list(concat)
        value = concat_value
        stats["realisation_fallback"] = True
    else:
        value = realised

    gap = dp_value - value
    stats["independence_gap"] = gap
    stats["independence_gap_pct"] = (100.0 * gap / dp_value) if dp_value > 0 else 0.0
    stats["independence_violated"] = gap > 1e-9

    # ------------------------------------------------------ optional repair
    if repair:
        t0 = time.perf_counter()
        pool = (list(mia.nodes) if repair_pool <= 0
                else top_candidates(mia, repair_pool, solo=solo))
        S2, v2, ls = local_search(mia, seeds, pool, rng,
                                  max_rounds=repair_rounds, solo=solo,
                                  time_budget=repair_budget, verbose=verbose)
        stats["repair_seconds"] = time.perf_counter() - t0
        stats["repaired"] = True
        stats["repair_swaps"] = ls["swaps_accepted"]
        stats["repair_gain"] = v2 - value
        stats["repair_certified"] = ls["certified"]
        stats["repair_stats"] = ls
        if v2 >= value:      # local_search guarantees this; assert it by using max
            seeds, value = S2, v2

    stats["value"] = value
    stats["n_seeds"] = len(seeds)
    stats["total_seconds"] = time.perf_counter() - t_start
    return seeds, stats


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    failures = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name
              + ((" -- " + detail) if detail else ""))
        if not cond:
            failures.append(name)

    class _M(object):
        """Minimal stand-in for ctim.gibbs.Model (same attribute names)."""

        def __init__(self, pi):
            self.pi = pi
            self.eta = [[1.0] * len(pi[0]) for _ in range(len(pi[0]))]
            self.theta = None
            self.p_z_given_i = None

    class _DS(object):
        """Minimal stand-in for ctim.dataset.Dataset."""

        def __init__(self, n_users, out_adj, in_adj):
            self.n_users = n_users
            self.out_adj = out_adj
            self.in_adj = in_adj
            self.edges = [(u, v) for u in range(n_users) for v in out_adj[u]]

    def build(n, arcs):
        out = [[] for _ in range(n)]
        inn = [[] for _ in range(n)]
        pp = {}
        for (a, b, p) in arcs:
            out[a].append(b)
            inn[b].append(a)
            pp[(a, b)] = p
        return out, inn, pp

    def onehot(c, C):
        return [1.0 if k == c else 0.0 for k in range(C)]

    # ------------------------------------------------------------------ [1]
    print("[1] the bridge graph: what line 34's induced subgraph cannot see")
    # Community A = {0,1,2,3}, community B = {4,5,6,7}.  All pp = 0.9, h = 0.1.
    #   node 1 has two intra-A arcs      -> I_A({1}) = 1 + .9 + .9 = 2.8
    #   node 0 has NO intra-A arc but four arcs into B
    #                                    -> I_A({0}) = 1  but  I({0}) = 1 + 4*.9 = 4.6
    # Algorithm 2 line 34 scores 0 at 1.0 and picks node 1; CTIM-G scores it at
    # 4.6 and picks node 0.  Hand-checkable, and it is exactly the c52 situation.
    arcs = [(1, 2, 0.9), (1, 3, 0.9),
            (0, 4, 0.9), (0, 5, 0.9), (0, 6, 0.9), (0, 7, 0.9)]
    out_b, in_b, pp_b = build(8, arcs)
    ds_b = _DS(8, out_b, in_b)
    comm_b = [0, 0, 0, 0, 1, 1, 1, 1]
    model_b = _M([onehot(c, 2) for c in comm_b])
    mia_b = MIA(8, out_b, in_b, pp_b, h=0.1)

    check("hand values: I_induced({0}) = 1.0 but I_global({0}) = 4.6",
          abs(MIA(8, out_b, in_b, pp_b, h=0.1,
                  nodes=[0, 1, 2, 3]).influence([0]) - 1.0) < 1e-12
          and abs(mia_b.influence([0]) - 4.6) < 1e-12,
          "induced %.4f global %.4f"
          % (MIA(8, out_b, in_b, pp_b, h=0.1, nodes=[0, 1, 2, 3]).influence([0]),
             mia_b.influence([0])))

    sg, stg = select_seeds_global(model_b, ds_b, pp_b, 1, random.Random(1),
                                  h=0.1, comm=comm_b)
    from ctim.ea_dp import select_seeds_ea_dp  # noqa: E402  (comparison only)
    s_ind, _st_ind = select_seeds_ea_dp(model_b, ds_b, pp_b, 1, random.Random(1),
                                        h=0.1, use_ea=False, comm=comm_b)
    check("CTIM-G picks the bridge node 0 (I = 4.6); induced-I_m picks 1 (I = 2.8)",
          sg == [0] and s_ind == [1]
          and abs(mia_b.influence(sg) - 4.6) < 1e-12
          and abs(mia_b.influence(s_ind) - 2.8) < 1e-12,
          "CTIM-G %s I=%.4f  vs  induced %s I=%.4f"
          % (sg, mia_b.influence(sg), s_ind, mia_b.influence(s_ind)))
    check("Eq (19) partition is honoured (seed lies in an allocated community)",
          stg["allocation"] == {0: 1} and comm_b[sg[0]] == 0,
          "allocation=%s" % (stg["allocation"],))

    # ------------------------------------------------------------------ [2]
    print("[2] (i) one community holding every node == MIA.greedy_incremental(K)")
    rng = random.Random(20190408)
    n = 30
    eset = set()
    while len(eset) < 95:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            eset.add((a, b))
    arcs3 = [(a, b, 0.05 + 0.6 * rng.random()) for (a, b) in sorted(eset)]
    out3, in3, pp3 = build(n, arcs3)
    ds3 = _DS(n, out3, in3)
    mia3 = MIA(n, out3, in3, pp3, h=0.1)
    comm_one = [0] * n
    model_one = _M([[1.0] for _ in range(n)])

    ok_i, detail_i = True, ""
    for K in (1, 3, 5, 8):
        ref = mia3.greedy_incremental(K)
        got, st = select_seeds_global(model_one, ds3, pp3, K, random.Random(3),
                                      h=0.1, comm=comm_one)
        if got != ref:
            ok_i = False
            detail_i = "K=%d: %s vs greedy %s" % (K, got, ref)
            break
        if abs(st["value"] - mia3.influence(ref)) > 1e-9:
            ok_i = False
            detail_i = "K=%d: value %.9f vs %.9f" % (K, st["value"],
                                                     mia3.influence(ref))
            break
    check("identical seed sequence at K = 1,3,5,8 (order included)", ok_i, detail_i)
    got5, st5 = select_seeds_global(model_one, ds3, pp3, 5, random.Random(3),
                                    h=0.1, comm=comm_one)
    check("with one community the DP predicts the realised value exactly",
          not st5["independence_violated"]
          and abs(st5["dp_value"] - st5["realised_value"]) < 1e-9,
          "dp %.9f realised %.9f gap %.3e"
          % (st5["dp_value"], st5["realised_value"], st5["independence_gap"]))

    # ------------------------------------------------------------------ [3]
    print("[3] (ii) determinism under a fixed seed")
    comm3 = [rng.randrange(4) for _ in range(n)]
    model3 = _M([onehot(c, 4) for c in comm3])
    runs = [select_seeds_global(model3, ds3, pp3, 6, random.Random(99), h=0.1)
            for _ in range(3)]
    check("three runs with random.Random(99) return identical seeds and value",
          runs[0][0] == runs[1][0] == runs[2][0]
          and runs[0][1]["value"] == runs[1][1]["value"] == runs[2][1]["value"],
          "%s / %s / %s" % (runs[0][0], runs[1][0], runs[2][0]))
    alt = select_seeds_global(model3, ds3, pp3, 6, random.Random(4242), h=0.1)
    check("a different rng seed changes nothing (selection is deterministic)",
          alt[0] == runs[0][0], "%s vs %s" % (alt[0], runs[0][0]))
    check("Eq (19) is used when comm is not injected",
          runs[0][0] == select_seeds_global(model3, ds3, pp3, 6,
                                            random.Random(99), h=0.1,
                                            comm=detect_communities(model3.pi))[0])

    # ------------------------------------------------------------------ [4]
    print("[4] (iii) repair=True never scores below repair=False")
    # A max-cover instance (pp = 1) where greedy is provably suboptimal:
    #   0 -> {3,4,5,6}, 1 -> {3,4,7}, 2 -> {5,6,8};  K = 2
    #   greedy takes 0 then 1  -> I = 7;  the optimum is {1,2} -> I = 8
    cover = [(0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0), (0, 6, 1.0),
             (1, 3, 1.0), (1, 4, 1.0), (1, 7, 1.0),
             (2, 5, 1.0), (2, 6, 1.0), (2, 8, 1.0)]
    out4, in4, pp4 = build(9, cover)
    ds4 = _DS(9, out4, in4)
    mia4 = MIA(9, out4, in4, pp4, h=0.1)
    comm4 = [0] * 9
    model4 = _M([[1.0] for _ in range(9)])
    s_no, st_no = select_seeds_global(model4, ds4, pp4, 2, random.Random(7),
                                      h=0.1, comm=comm4)
    s_yes, st_yes = select_seeds_global(model4, ds4, pp4, 2, random.Random(7),
                                        h=0.1, comm=comm4, repair=True)
    check("on the max-cover instance repair lifts 7.0 -> 8.0 (the optimum)",
          abs(st_no["value"] - 7.0) < 1e-12 and abs(st_yes["value"] - 8.0) < 1e-12,
          "no-repair %s I=%.4f -> repair %s I=%.4f (%d swap(s), %.4fs)"
          % (s_no, st_no["value"], s_yes, st_yes["value"],
             st_yes["repair_swaps"], st_yes["repair_seconds"]))

    ok_iii, detail_iii = True, ""
    for K in (2, 4, 6):
        for cm, md in ((comm3, model3), (comm_one, model_one)):
            a = select_seeds_global(md, ds3, pp3, K, random.Random(11), h=0.1,
                                    comm=cm)[1]["value"]
            b = select_seeds_global(md, ds3, pp3, K, random.Random(11), h=0.1,
                                    comm=cm, repair=True)[1]["value"]
            if b < a - 1e-12:
                ok_iii = False
                detail_iii = "K=%d: repair %.9f < no-repair %.9f" % (K, b, a)
    check("repair >= no-repair on 6 (K, partition) combinations", ok_iii,
          detail_iii)

    # ------------------------------------------------------------------ [5]
    print("[5] (iv) the returned seed count is min(K, |V|)")
    ok_iv, detail_iv = True, ""
    for K in (1, 2, 5, 9, 15, 40):
        s, st = select_seeds_global(model_b, ds_b, pp_b, K, random.Random(5),
                                    h=0.1, comm=comm_b)
        want = min(K, ds_b.n_users)
        if len(s) != want or len(set(s)) != want:
            ok_iv = False
            detail_iv = "K=%d: got %d seeds %s (want %d)" % (K, len(s), s, want)
            break
        if st["n_seeds"] != len(s):
            ok_iv = False
            detail_iv = "K=%d: stats n_seeds=%d but %d seeds" % (K, st["n_seeds"],
                                                                 len(s))
            break
    check("|S| = min(K, |V|) and seeds are distinct for K in 1..40", ok_iv,
          detail_iv)
    for K in (3, 7, 12, 25):
        s, _ = select_seeds_global(model3, ds3, pp3, K, random.Random(5), h=0.1)
        if len(s) != min(K, n):
            ok_iv = False
            detail_iv = "4-community partition, K=%d: %d seeds" % (K, len(s))
    check("same on the 4-community partition (K = 3,7,12,25)", ok_iv, detail_iv)

    # ------------------------------------------------------------------ [6]
    print("[6] the independence violation is measured, not assumed away")
    # Two communities whose arborescences overlap: both 0 (in A) and 4 (in B)
    # point at the same targets, so the DP's sum of independent curves must
    # over-count and stats must say so.
    ov = [(0, 8, 0.9), (0, 9, 0.9), (0, 10, 0.9), (0, 11, 0.9),
          (4, 8, 0.9), (4, 9, 0.9), (4, 10, 0.9), (4, 11, 0.9),
          (1, 12, 0.5), (5, 13, 0.5)]
    out6, in6, pp6 = build(14, ov)
    ds6 = _DS(14, out6, in6)
    comm6 = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1]
    model6 = _M([onehot(c, 2) for c in comm6])
    s6, st6 = select_seeds_global(model6, ds6, pp6, 2, random.Random(2), h=0.1,
                                  comm=comm6)
    mia6 = MIA(14, out6, in6, pp6, h=0.1)
    check("DP over-predicts here and the gap is reported",
          st6["independence_violated"] and st6["independence_gap"] > 1e-6,
          "dp %.4f  realised %.4f  gap %.4f (%.2f%%)  seeds %s"
          % (st6["dp_value"], st6["realised_value"], st6["independence_gap"],
             st6["independence_gap_pct"], s6))
    check("the reported value is a real Eq (18) evaluation of the returned set",
          abs(st6["value"] - mia6.influence(s6)) < 1e-9,
          "stats %.9f vs recomputed %.9f" % (st6["value"], mia6.influence(s6)))
    check("the quota realisation is at least as good as concatenating the "
          "DP's curve sets",
          st6["value"] >= st6["concat_value"] - 1e-12,
          "realised %.4f vs concat %.4f (fallback=%s)"
          % (st6["realised_value"], st6["concat_value"],
             st6["realisation_fallback"]))

    # ------------------------------------------------------------------ [7]
    print("[7] stats carry a per-phase wall-clock breakdown")
    s7, st7 = select_seeds_global(model3, ds3, pp3, 5, random.Random(8), h=0.1,
                                  repair=True)
    phases = ("detect_seconds", "mia_seconds", "solo_seconds", "curve_seconds",
              "dp_seconds", "realise_seconds", "eval_seconds", "repair_seconds")
    check("every phase timer is present and non-negative",
          all(k in st7 and st7[k] >= 0.0 for k in phases),
          " ".join("%s=%.4fs" % (k, st7[k]) for k in phases))
    check("the phases sum to no more than total_seconds",
          sum(st7[k] for k in phases) <= st7["total_seconds"] + 1e-6,
          "sum=%.6f total=%.6f" % (sum(st7[k] for k in phases),
                                   st7["total_seconds"]))
    check("the reported value is a real Eq (18) evaluation of the returned set",
          abs(st7["value"] - mia3.influence(s7)) < 1e-9,
          "stats %.9f vs recomputed %.9f" % (st7["value"], mia3.influence(s7)))
    # Informational, NOT a check: scoring globally is the right objective, but
    # greedy is still greedy, so beating the induced-subgraph variant on every
    # instance is not a theorem and must not be asserted as one.
    s_ind7 = select_seeds_ea_dp(model3, ds3, pp3, 5, random.Random(8), h=0.1,
                                use_ea=False, comm=comm3)[0]
    print("      [info] CTIM-G I=%.6f %s  induced-I_m CTIM I=%.6f  (same Eq (18) "
          "evaluator, h=0.1)"
          % (st7["value"],
             ">=" if st7["value"] >= mia3.influence(s_ind7) else "< ",
             mia3.influence(s_ind7)))

    print("")
    if failures:
        print("FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("ALL CHECKS PASSED")
