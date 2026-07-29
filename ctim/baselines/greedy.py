"""Greedy baseline -- Kempe, Kleinberg & Tardos [3], Monte-Carlo IC.

SPEC.md Section 7 (Baselines, Table 2)::

    Greedy -- the original greedy algorithm of Kempe et al. [3], Monte-Carlo IC.
              Topic-blind and community-blind: no Z, no C.

The algorithm is the classical hill-climb: starting from ``S = {}``, repeat K
times

    u_k = argmax_{u not in S}  sigma(S u {u}) - sigma(S)
    S  <- S u {u_k}

where ``sigma(.)`` is the expected number of activated nodes under the
**Independent Cascade** model, estimated by ``n_mc`` Monte-Carlo simulations.
``sigma`` is monotone and submodular, so the hill-climb carries the classical
(1 - 1/e) guarantee and admits Leskovec et al.'s CELF lazy-forward acceleration.

Topic-blindness
---------------
``greedy_select`` takes a **single** ``pp`` dict mapping a directed edge
``(u, v)`` to one topic-independent activation probability.  It has no notion of
topics (``Z``) or communities (``C``) -- that is exactly the ablation Table 2
draws.  The experiment harness is expected to pass the **topic-averaged learned
probabilities**, i.e. the mean over the evaluated test items of the CTIM edge
weights from Eq (12)::

    pp[(u,v)] = (1/|I|) * sum_{i in I} P(v | i, u)

so that Greedy sees the same learned diffusion strengths as the topic-aware
methods, just collapsed to a single item-independent number.  The helper
:func:`topic_averaged_pp` builds exactly that dict from a
``ctim.influence.EdgeWeights``.  A uniform constant ``p`` (the other convention
in the IM literature) works equally well; whichever is used is recorded in the
result under ``extra["pp_source"]`` by the caller.

Determinism
-----------
All randomness flows through the explicit ``random.Random`` instance passed in;
the module never touches ``random.*``.  Within a greedy round the Monte-Carlo
oracle uses **common random numbers** (``crn=True``, the default): one round
seed is drawn from ``rng``, and every candidate in that round is simulated
against the same ``n_mc`` derived streams.  This is a standard variance
reduction -- it removes the sampling noise from the *comparison* between
candidates -- and it makes the oracle a deterministic function of
``(S, round)``, which is what makes CELF's lazy re-evaluations return exactly
the values plain greedy would have seen in that round.

Note that CELF is only *exactly* equivalent to plain greedy when the spread
oracle is deterministic and submodular across rounds.  Common random numbers
give that within a round but not across rounds (independent Monte-Carlo noise
between rounds can make a stale bound too small), so under the MC oracle the
two may occasionally diverge.  The self-test therefore pins the equivalence
assertion with an exact, deterministic oracle -- ``MIA.influence`` from
``ctim.influence`` (Eq (17)/(18)) -- passed through the ``oracle`` argument.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import heapq
import random
import time

try:
    from ctim.baselines import RunResult
except ImportError:  # pragma: no cover - direct `python3 ctim/baselines/greedy.py`
    import os as _os
    import sys as _sys

    _sys.path.insert(
        0,
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
    )
    from ctim.baselines import RunResult

__all__ = [
    "ic_simulate",
    "greedy_select",
    "select_seeds",
    "topic_averaged_pp",
    "ICSimulator",
    "RunResult",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _out_adj_of(ds_or_adj):
    """Accept a ``ctim.dataset.Dataset`` or a bare ``out_adj`` list of lists."""
    adj = getattr(ds_or_adj, "out_adj", None)
    if adj is not None:
        return adj
    return ds_or_adj


def topic_averaged_pp(edge_weights, items) -> dict:
    """Collapse the topic-aware Eq (12) weights to one topic-blind ``pp`` dict.

    ``edge_weights`` is a ``ctim.influence.EdgeWeights``; ``items`` an iterable
    of item ids.  Returns ``{(u, v): mean_i P(v|i,u)}`` over those items, which
    is the topic-independent probability map documented above for
    :func:`greedy_select`.
    """
    items = list(items)
    if not items:
        raise ValueError("topic_averaged_pp needs at least one item")
    acc = {}
    for i in items:
        for edge, p in edge_weights.for_item(i).items():  # Eq (12)
            acc[edge] = acc.get(edge, 0.0) + p
    inv = 1.0 / len(items)
    return {edge: total * inv for edge, total in acc.items()}


# ---------------------------------------------------------------------------
# Independent Cascade Monte-Carlo simulation
# ---------------------------------------------------------------------------


class ICSimulator:
    """Independent Cascade simulator with a reusable visited-stamp array.

    The adjacency is flattened once into ``nbrs[u] = [(v, p), ...]`` (edges with
    ``p <= 0`` dropped, since they can never fire), so the inner loop does no
    dict lookups at all.  ``stamp`` is an int array reused across simulations: a
    node is active in the current cascade iff ``stamp[v] == cur``, and a new
    cascade just bumps ``cur``.  Nothing is reallocated per simulation.

    Probabilities are clamped into ``[0, 1]``; ``n_clamped`` counts how often
    that was necessary, so a caller handing over malformed learned weights sees
    the violation rather than silently getting a broken cascade.
    """

    def __init__(self, ds_or_adj, pp):
        out_adj = _out_adj_of(ds_or_adj)
        self.n_users = len(out_adj)
        self.n_clamped = 0

        nbrs = []
        for u in range(self.n_users):
            row = []
            for v in out_adj[u]:
                p = pp.get((u, v), 0.0)
                if p > 1.0:
                    p = 1.0
                    self.n_clamped += 1
                elif p < 0.0:
                    p = 0.0
                    self.n_clamped += 1
                if p > 0.0:
                    row.append((v, p))
            nbrs.append(row)
        self.nbrs = nbrs

        self._stamp = [0] * self.n_users
        self._cur = 0
        self._crn_rng = random.Random(0)
        self._seed_cache_key = None
        self._seed_cache = None

    # -- one cascade --------------------------------------------------------

    def _one_cascade(self, seeds, random_fn) -> int:
        """One IC realisation from ``seeds``; returns the number of active nodes.

        Breadth-first over the newly-activated frontier: when ``u`` first becomes
        active it gets exactly one chance to activate each out-neighbour ``v``,
        which is the Independent Cascade rule.  Already-active ``v`` are skipped
        without drawing a coin -- a further success could not change the outcome,
        so the distribution of the active set is unaffected.
        """
        self._cur += 1
        cur = self._cur
        stamp = self._stamp
        nbrs = self.nbrs

        frontier = []
        count = 0
        for s in seeds:
            if stamp[s] != cur:
                stamp[s] = cur
                frontier.append(s)
                count += 1

        while frontier:
            nxt = []
            for u in frontier:
                for (v, p) in nbrs[u]:
                    if stamp[v] != cur and random_fn() < p:
                        stamp[v] = cur
                        nxt.append(v)
                        count += 1
            frontier = nxt
        return count

    # -- expected spread ----------------------------------------------------

    def spread(self, S, n_mc, rng) -> float:
        """Eq (18) estimated by Monte Carlo: mean |active set| over ``n_mc`` runs.

        Consumes ``rng`` sequentially.  ``rng`` must be a ``random.Random``.
        """
        seeds = list(S)
        if not seeds or n_mc <= 0:
            return float(len(set(seeds))) if seeds else 0.0
        random_fn = rng.random
        total = 0
        for _ in range(n_mc):
            total += self._one_cascade(seeds, random_fn)  # Eq (18), MC estimate
        return total / float(n_mc)

    def _crn_seeds(self, n_mc, base_seed):
        """The ``n_mc`` per-simulation seeds derived from one round seed."""
        key = (base_seed, n_mc)
        if self._seed_cache_key == key:
            return self._seed_cache
        gen = random.Random(base_seed)
        seeds = [gen.getrandbits(64) for _ in range(n_mc)]
        self._seed_cache_key = key
        self._seed_cache = seeds
        return seeds

    def spread_crn(self, S, n_mc, base_seed) -> float:
        """Eq (18) estimated with **common random numbers**.

        Every call with the same ``(n_mc, base_seed)`` replays the same ``n_mc``
        random streams, so the estimate is a deterministic function of
        ``(S, n_mc, base_seed)``.  Comparing candidates under a shared stream
        cancels most of the Monte-Carlo noise out of the *difference*, which is
        the quantity greedy actually ranks on.
        """
        seeds = list(S)
        if not seeds or n_mc <= 0:
            return float(len(set(seeds))) if seeds else 0.0
        gen = self._crn_rng
        total = 0
        for s in self._crn_seeds(n_mc, base_seed):
            gen.seed(s)
            total += self._one_cascade(seeds, gen.random)  # Eq (18), MC estimate
        return total / float(n_mc)


def ic_simulate(ds_or_adj, S, pp, n_mc, rng) -> float:
    """Expected spread of ``S`` under Independent Cascade, by Monte Carlo.

    Eq (18) ``I(S) = sum_v ap(v|S)`` estimated as the mean number of activated
    nodes over ``n_mc`` cascades.  Exposed for reuse by the other baselines.

    Parameters
    ----------
    ds_or_adj : a ``ctim.dataset.Dataset`` or a bare ``out_adj`` list of lists
    S         : iterable of seed node ids
    pp        : ``{(u, v): probability}`` for the directed edges of G
    n_mc      : number of Monte-Carlo simulations
    rng       : explicit ``random.Random`` instance (never module-level random)

    Building the simulator is O(|E|); if you are going to call this many times
    on the same graph, construct an :class:`ICSimulator` once and call
    ``.spread()`` instead, which reuses the flattened adjacency and the
    visited-stamp array.
    """
    if rng is None:
        raise ValueError("ic_simulate requires an explicit random.Random instance")
    return ICSimulator(ds_or_adj, pp).spread(S, n_mc, rng)


# ---------------------------------------------------------------------------
# Greedy (Kempe et al. [3]) with optional CELF lazy-forward
# ---------------------------------------------------------------------------


def greedy_select(ds, pp, K, n_mc, rng, use_celf=True,
                  candidates=None, oracle=None, crn=True) -> RunResult:
    """Kempe et al. [3] greedy influence maximization under Independent Cascade.

    K rounds; each round adds the node with the largest marginal gain in
    expected spread, estimated by ``n_mc`` Monte-Carlo IC simulations.

    Parameters
    ----------
    ds : ``ctim.dataset.Dataset`` (or a bare ``out_adj`` list of lists)
    pp : ``{(u, v): p}`` -- a **single, topic-independent** probability per
        directed edge.  Greedy is topic-blind and community-blind (SPEC.md
        Table 2): it gets no ``Z`` and no ``C``.  The experiment harness passes
        the topic-averaged learned probabilities, ``mean_i P(v|i,u)`` over the
        evaluated test items (see :func:`topic_averaged_pp`); a uniform constant
        ``p`` is the other accepted convention.
    K : seed-set size
    n_mc : Monte-Carlo simulations per spread evaluation
    rng : explicit ``random.Random`` -- the only source of randomness
    use_celf : enable Leskovec et al.'s CELF lazy-forward.  A max-heap of
        marginal gains is kept; each round the top entry is popped and, if its
        gain is stale, re-evaluated at the current ``S`` and reinserted, until a
        freshly-evaluated entry surfaces.  Submodularity makes every stale gain
        an upper bound on the true one, so that entry is the true argmax.
        Recorded in ``extra["use_celf"]``.
    candidates : optional restriction of which nodes may be selected (defaults
        to every node).  Never restricts which nodes are *counted* in the spread.
    oracle : optional ``f(S) -> float`` replacing the Monte-Carlo estimator with
        a deterministic exact spread (e.g. ``MIA.influence``, Eq (17)/(18)).
        Under such an oracle CELF and plain greedy are provably identical; the
        self-test asserts exactly that.
    crn : use common random numbers within each round (default).  Ignored when
        ``oracle`` is supplied.

    Returns
    -------
    RunResult with ``seeds`` in selection order, ``spread`` = the oracle's value
    for that set, ``seconds`` = wall clock, and ``extra`` carrying at least
    ``use_celf`` and the spread-evaluation counts.
    """
    if rng is None:
        raise ValueError("greedy_select requires an explicit random.Random instance")
    t0 = time.perf_counter()

    out_adj = _out_adj_of(ds)
    n_users = len(out_adj)

    if candidates is None:
        cand = list(range(n_users))
    else:
        cand = sorted(set(candidates))
    K = max(0, min(int(K), len(cand)))

    sim = None if oracle is not None else ICSimulator(out_adj, pp)

    n_evals = [0]
    round_seed = [0]

    def spread_of(S):
        """The greedy objective sigma(S), counted."""
        n_evals[0] += 1
        if oracle is not None:
            return float(oracle(S))
        if crn:
            return sim.spread_crn(S, n_mc, round_seed[0])
        return sim.spread(S, n_mc, rng)

    def new_round():
        """Draw this round's common-random-number seed from `rng`."""
        if oracle is None and crn:
            round_seed[0] = rng.getrandbits(63)

    S = []
    S_set = set()

    if K == 0 or not cand:
        seconds = time.perf_counter() - t0
        return RunResult(
            seeds=[], spread=0.0, seconds=seconds,
            extra=_extra(use_celf, n_mc, oracle, crn, cand, 0, 0, sim),
        )

    if use_celf:
        # ---- CELF (Leskovec et al.) lazy-forward --------------------------
        # `heap` holds exactly one entry (-gain, u) per unselected candidate;
        # `fresh[u]` is the round in which that gain was computed.  Ordering by
        # (-gain, u) breaks ties towards the lowest node id, which is precisely
        # the tie-break plain greedy makes by scanning `cand` in ascending order.
        new_round()
        base = 0.0  # sigma({}) = 0
        heap = []
        fresh = {}
        for u in cand:
            g = spread_of([u]) - base
            heap.append((-g, u))
            fresh[u] = 0
        heapq.heapify(heap)

        for k in range(K):
            if k > 0:
                new_round()
                base = spread_of(S)
            while True:
                neg_gain, u = heapq.heappop(heap)
                if fresh[u] == k:
                    chosen = u
                    break
                # stale: re-evaluate at the current S and reinsert
                g = spread_of(S + [u]) - base
                fresh[u] = k
                heapq.heappush(heap, (-g, u))
            S.append(chosen)
            S_set.add(chosen)
    else:
        # ---- plain greedy: rescan every candidate every round -------------
        for k in range(K):
            new_round()
            base = 0.0 if not S else spread_of(S)
            best_u = None
            best_g = None
            for u in cand:  # ascending -> ties resolve to the lowest node id
                if u in S_set:
                    continue
                g = spread_of(S + [u]) - base
                if best_g is None or g > best_g:
                    best_u, best_g = u, g
            if best_u is None:
                break
            S.append(best_u)
            S_set.add(best_u)

    n_selection_evals = n_evals[0]

    # Final spread of the returned seed set, under the same oracle.
    new_round()
    spread = spread_of(S)  # Eq (18)

    seconds = time.perf_counter() - t0
    return RunResult(
        seeds=S,
        spread=spread,
        seconds=seconds,
        extra=_extra(use_celf, n_mc, oracle, crn, cand,
                     n_selection_evals, n_evals[0], sim),
    )


def _extra(use_celf, n_mc, oracle, crn, cand, n_sel_evals, n_total_evals, sim):
    extra = {
        "method": "greedy",
        "reference": "Kempe et al. [3]",
        "topic_blind": True,
        "community_blind": True,
        "use_celf": bool(use_celf),
        "celf": bool(use_celf),  # alias, some harnesses look for this name
        "n_mc": int(n_mc),
        "oracle": "exact" if oracle is not None else "monte-carlo-ic",
        "common_random_numbers": bool(crn) and oracle is None,
        "n_candidates": len(cand),
        # spread evaluations performed during seed selection ...
        "n_selection_evals": int(n_sel_evals),
        # ... and in total, including the final evaluation of the returned set.
        "n_spread_evals": int(n_total_evals),
    }
    if sim is not None:
        extra["n_pp_clamped"] = sim.n_clamped
    return extra


# API.md: every baseline module exposes `select_seeds(...) -> RunResult`.
select_seeds = greedy_select


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import sys

    from ctim.influence import MIA

    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("  PASS  %s%s" % (name, (" -- " + detail) if detail else ""))
        else:
            print("  FAIL  %s%s" % (name, (" -- " + detail) if detail else ""))
            failures.append(name)

    # ------------------------------------------------------------------ [1]
    print("[1] ic_simulate on a 4-node chain vs the exact expectation")
    # 0 -> 1 -> 2 -> 3, every edge p = 0.5, S = {0}:
    #   E|active| = 1 + 0.5 + 0.25 + 0.125 = 1.875
    chain_out = [[1], [2], [3], []]
    chain_in = [[], [0], [1], [2]]
    chain_pp = {(0, 1): 0.5, (1, 2): 0.5, (2, 3): 0.5}
    est = ic_simulate(chain_out, [0], chain_pp, 200000, random.Random(1))
    check("MC spread %.4f == 1.875 within 1%%" % est,
          abs(est - 1.875) / 1.875 <= 0.01, "rel err %.4f%%" % (100 * abs(est - 1.875) / 1.875))

    a = ic_simulate(chain_out, [0], chain_pp, 5000, random.Random(7))
    b = ic_simulate(chain_out, [0], chain_pp, 5000, random.Random(7))
    c = ic_simulate(chain_out, [0], chain_pp, 5000, random.Random(8))
    check("ic_simulate is deterministic for a fixed seed", a == b, "%.6f" % a)
    check("ic_simulate actually depends on the seed", a != c, "%.6f vs %.6f" % (a, c))
    check("a seed set always activates at least itself",
          ic_simulate(chain_out, [0, 2], chain_pp, 200, random.Random(3)) >= 2.0)
    check("zero-probability graph spreads to exactly |S|",
          ic_simulate(chain_out, [0], {}, 100, random.Random(3)) == 1.0)
    check("ic_simulate accepts a Dataset-like object with .out_adj",
          ic_simulate(type("D", (), {"out_adj": chain_out})(), [0], chain_pp,
                      2000, random.Random(5)) > 1.0)

    # ------------------------------------------------------------------ [2]
    print("[2] CELF == plain greedy under an exact deterministic oracle")
    # CELF is exact only for a deterministic, monotone, submodular oracle.
    # MIA.influence (Eq (17)/(18), Chen et al. [19]) is exactly that, so we pin
    # the equivalence against it rather than against the noisy MC estimator.
    rng = random.Random(20190408)
    n = 45
    eset = set()
    while len(eset) < 170:
        x = rng.randrange(n)
        y = rng.randrange(n)
        if x != y:
            eset.add((x, y))
    out_adj = [[] for _ in range(n)]
    in_adj = [[] for _ in range(n)]
    pp = {}
    for (x, y) in sorted(eset):
        out_adj[x].append(y)
        in_adj[y].append(x)
        pp[(x, y)] = 0.05 + 0.55 * rng.random()

    mia = MIA(n, out_adj, in_adj, pp, h=0.05)
    oracle = mia.influence  # Eq (18), exact

    K = 6
    r_celf = greedy_select(out_adj, pp, K, 0, random.Random(42),
                           use_celf=True, oracle=oracle)
    r_plain = greedy_select(out_adj, pp, K, 0, random.Random(42),
                            use_celf=False, oracle=oracle)

    check("CELF and plain greedy return the SAME seed set",
          r_celf.seeds == r_plain.seeds,
          "%s vs %s" % (r_celf.seeds, r_plain.seeds))
    check("...and the same spread",
          abs(r_celf.spread - r_plain.spread) <= 1e-12,
          "%.9f vs %.9f" % (r_celf.spread, r_plain.spread))
    check("seed set has K distinct nodes",
          len(r_celf.seeds) == K and len(set(r_celf.seeds)) == K)
    check("CELF needs strictly fewer spread evaluations",
          r_celf.extra["n_selection_evals"] < r_plain.extra["n_selection_evals"],
          "%d vs %d" % (r_celf.extra["n_selection_evals"],
                        r_plain.extra["n_selection_evals"]))
    check("extra records whether CELF was used",
          r_celf.extra["use_celf"] is True and r_plain.extra["use_celf"] is False)
    check("extra records the oracle kind",
          r_celf.extra["oracle"] == "exact")

    # Independent brute-force greedy, to be sure both are really the hill-climb.
    def brute(K_):
        S = []
        for _ in range(K_):
            best, best_g = None, None
            base = oracle(S) if S else 0.0
            for u in range(n):
                if u in S:
                    continue
                g = oracle(S + [u]) - base
                if best_g is None or g > best_g:
                    best, best_g = u, g
            S.append(best)
        return S

    check("both match an independent brute-force hill-climb",
          r_celf.seeds == brute(K), "%s vs %s" % (r_celf.seeds, brute(K)))

    # ------------------------------------------------------------------ [3]
    print("[3] candidate restriction")
    cands = sorted(rng.sample(range(n), 15))
    r_c = greedy_select(out_adj, pp, 4, 0, random.Random(1),
                        use_celf=True, candidates=cands, oracle=oracle)
    r_p = greedy_select(out_adj, pp, 4, 0, random.Random(1),
                        use_celf=False, candidates=cands, oracle=oracle)
    check("selection stays inside the candidate set", set(r_c.seeds) <= set(cands))
    check("CELF == plain greedy on a restricted candidate set",
          r_c.seeds == r_p.seeds, "%s vs %s" % (r_c.seeds, r_p.seeds))
    check("extra records the candidate count", r_c.extra["n_candidates"] == len(cands))

    # ------------------------------------------------------------------ [4]
    print("[4] Monte-Carlo greedy: determinism and sanity")
    r1 = greedy_select(out_adj, pp, 5, 400, random.Random(11), use_celf=True)
    r2 = greedy_select(out_adj, pp, 5, 400, random.Random(11), use_celf=True)
    r3 = greedy_select(out_adj, pp, 5, 400, random.Random(12), use_celf=True)
    check("MC greedy is deterministic for a fixed seed",
          r1.seeds == r2.seeds and r1.spread == r2.spread, "%s" % (r1.seeds,))
    check("MC greedy consumes the seed (different seed -> different run)",
          r1.spread != r3.spread or r1.seeds != r3.seeds)
    check("MC greedy returns K distinct seeds",
          len(r1.seeds) == 5 and len(set(r1.seeds)) == 5)
    check("MC spread >= |S| and <= n_users",
          5.0 <= r1.spread <= n, "%.4f" % r1.spread)
    check("extra records n_mc and the MC oracle",
          r1.extra["n_mc"] == 400 and r1.extra["oracle"] == "monte-carlo-ic")
    check("common random numbers are on by default",
          r1.extra["common_random_numbers"] is True)
    check("wall-clock seconds recorded", r1.seconds > 0.0, "%.4fs" % r1.seconds)

    r_seq = greedy_select(out_adj, pp, 3, 200, random.Random(11),
                          use_celf=True, crn=False)
    check("crn=False path runs and is flagged",
          len(r_seq.seeds) == 3 and r_seq.extra["common_random_numbers"] is False)

    # Informational only: CELF vs plain greedy under the *noisy* MC oracle is
    # not guaranteed to agree (stale bounds are only submodular in expectation).
    r_mc_plain = greedy_select(out_adj, pp, 5, 400, random.Random(11), use_celf=False)
    print("      [info] MC CELF %s | MC plain %s | agree=%s"
          % (r1.seeds, r_mc_plain.seeds, r1.seeds == r_mc_plain.seeds))
    print("      [info] evals: CELF %d, plain %d"
          % (r1.extra["n_selection_evals"], r_mc_plain.extra["n_selection_evals"]))

    # ------------------------------------------------------------------ [5]
    print("[5] edge cases and RunResult shape")
    r0 = greedy_select(out_adj, pp, 0, 100, random.Random(1))
    check("K=0 returns an empty result", r0.seeds == [] and r0.spread == 0.0)
    r_over = greedy_select(out_adj, pp, 3, 50, random.Random(1), candidates=[7])
    check("K larger than the candidate set is clipped", r_over.seeds == [7])
    for fld in ("seeds", "spread", "seconds", "extra"):
        check("RunResult exposes .%s" % fld, hasattr(r1, fld))
    check("pp outside [0,1] is clamped and counted",
          ICSimulator(chain_out, {(0, 1): 1.4, (1, 2): -0.2, (2, 3): 0.5}).n_clamped == 2)
    check("select_seeds is the module entry point required by API.md",
          select_seeds is greedy_select)

    # ------------------------------------------------------------------ [6]
    print("[6] topic_averaged_pp collapses Eq (12) weights to one map")

    class _EW(object):
        def for_item(self, i):
            return {(0, 1): 0.2 * (i + 1), (1, 2): 0.1}

    avg = topic_averaged_pp(_EW(), [0, 1, 2])
    check("mean over items is correct",
          abs(avg[(0, 1)] - 0.4) < 1e-12 and abs(avg[(1, 2)] - 0.1) < 1e-12,
          "%s" % avg)
    try:
        topic_averaged_pp(_EW(), [])
        check("empty item list rejected", False)
    except ValueError:
        check("empty item list rejected", True)

    print("")
    if failures:
        print("FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("ctim/baselines/greedy.py self-test: ALL CHECKS PASSED")


if __name__ == "__main__":
    _self_test()
