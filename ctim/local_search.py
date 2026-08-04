"""1-swap local search over a *finished* seed set -- a non-paper addition.

Why this module exists
----------------------
Huang et al. (2019) never revisit a seed once Algorithm 2 line 45 has committed
it, and neither does plain MIA greedy.  Both stop at a set carrying only the
``(1 - 1/e)`` submodular-greedy guarantee, which is a *worst-case* bound, not a
certificate of optimality.  Nothing in the repository has so far asked the
obvious follow-up question: is the delivered seed set even a local optimum?

This is cheap and *verifiable* here for one specific reason.  ``MIA.influence``
(Eq (17)/(18)) is exact, closed form and deterministic -- there is no Monte-Carlo
fitness noise.  So a strictly improving swap found here is a fact about the
objective, not a sampling fluctuation, and conversely "no improving swap exists"
is a proof rather than a failure to sample one.  That is the whole appeal: a
0.00% result from this module is *informative*, because it upgrades greedy's
output from "has a 1-1/e bound" to "is a certified 1-swap local optimum".

This is **not** part of the paper.  Cf. ``DEVIATIONS.md`` Section 4.7, which
already lists ``ctim/ea_dp.py`` and ``ctim/ris_imm.py`` as non-paper modules
shipping inside the package; this is a third one of the same kind.  It is
reachable only from ``scripts/run_local_search.py``; no results path imports it,
and ``ctim/ctim.py`` is untouched.

Making the naive neighbourhood affordable
-----------------------------------------
The 1-swap neighbourhood of ``S`` is ``|S| x |V|`` pairs, and re-evaluating
Eq (18) from scratch on each would be ``20 x 30358`` full influence
computations per round on Digg.  Two exact reductions make it tractable, and
neither approximates anything:

**(1) A completeness-preserving candidate prune.**  For a swap ``u -> w``,

    delta(u,w) = I(S - u + w) - I(S)
               = [ I(S-u+w) - I(S-u) ] - [ I(S) - I(S-u) ]
               = gain_w(S \\ u) - loss_u

and by submodularity of Eq (18) under the MIA model (Chen et al. [19], who prove
``I`` submodular on the fixed arborescences) ``gain_w(S \\ u) <= I({w})``.  Hence

    delta(u,w) <= I({w}) - loss_u <= I({w}) - min_{u in S} loss_u.          (*)

So every ``w`` with ``I({w}) <= min_u loss_u`` is provably unable to improve
``S`` and may be discarded **without losing any improving swap**.  This is a
prune, not a heuristic restriction: when the supplied candidate pool is the
top-N by ``I({w})`` and the N-th value already falls below ``min_u loss_u``, the
resulting local optimum is a local optimum over *all* of ``V``, and
``stats["certified"]`` records that.

``I({w})`` for every node is itself cheap in closed form -- see
``solo_influence`` -- so ranking all of ``V`` costs one forward Dijkstra per
node and no MIIA construction at all.

**(2) An exact O(|MIOA(u) & MIOA(w)|) swap delta.**  Removing ``u`` and adding
``w`` can only change ``ap(v | .)`` for ``v in MIOA(u) | MIOA(w)`` (Eq (15)/(16)
are the same condition read from the two ends).  Splitting that union and
noting that outside the intersection each half is independent of the *other*
endpoint gives, with ``L_u = loss_u``, ``G_w = gain_w(S)`` and
``J = MIOA(u) & MIOA(w)``:

    delta(u,w) = G_w - L_u
                 - sum_{v in J} [ ap(v|S+w) + ap(v|S-u) - ap(v|S-u+w) - ap(v|S) ]

``L_u`` is computed once per seed per round, ``G_w`` once per candidate per
round, and the correction term is empty for every pair whose arborescences do
not overlap.  On a near-additive objective -- which is exactly what Digg at
``h = 0.1`` is -- almost every pair falls in that case, so the round costs
``O(|S| + |cand|)`` arborescence evaluations rather than ``O(|S| * |cand|)``.

The unconditional guarantee
---------------------------
Whatever the pruning does, ``local_search`` re-evaluates the returned set with a
real ``MIA.influence`` call and reverts to the input set if it did not strictly
improve.  The result can therefore **never** score below the input, even if the
submodularity assumption behind (*) were violated by some pathological ``pp``.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random``; the RNG only permutes scan order (and therefore only
breaks ties), never the accept/reject decision.
"""

from __future__ import annotations

import os
import sys
import time

if __name__ == "__main__" and __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = [
    "solo_influence",
    "solo_influence_all",
    "top_candidates",
    "swap_delta",
    "local_search",
]


# ---------------------------------------------------------------------------
# single-node spread, in closed form
# ---------------------------------------------------------------------------


def solo_influence(mia, w) -> float:
    """``I({w})``, Eq (18) specialised to a one-element seed set.

    With ``S = {w}`` the recursion of Eq (17) collapses: inside ``MIIA(v,h)``
    only the single branch leading to ``w`` carries non-zero activation
    probability, so ``ap(v|{w})`` is the product of ``pp`` along ``MIP(w,v)``,
    i.e. exactly ``pp(MIP(w,v))`` of Eq (13)/(14).  Summing over Eq (16):

        I({w}) = sum_{v in MIOA(w,h)} pp(MIP(w,v))                    # Eq (18)

    This needs one forward Dijkstra (the cached ``mioa``) and *no* MIIA
    construction, which is what makes ranking all of V affordable.  The
    self-test checks it against ``MIA.influence({w})`` exhaustively.
    """
    nodes, parent = mia.mioa(w)  # Eq (16)
    pp = mia.pp
    prob = {w: 1.0}
    for x in nodes:
        if x in prob:
            continue
        chain = []
        y = x
        while y not in prob:
            chain.append(y)
            y = parent[y]
        val = prob[y]
        for z in reversed(chain):
            val *= pp[(parent[z], z)]  # Eq (13): pp(P) is a product along P
            prob[z] = val
    return sum(prob.values())  # Eq (18)


def solo_influence_all(mia, nodes=None) -> dict:
    """``{w: I({w})}`` for every node of `mia` (or of `nodes`)."""
    it = mia.nodes if nodes is None else sorted(set(nodes))
    return {w: solo_influence(mia, w) for w in it}


def top_candidates(mia, n, solo=None, exclude=()) -> list:
    """The `n` nodes with the largest ``I({w})``, ties broken by node id.

    Ranking by single-node spread is the right prune here because of bound (*)
    in the module docstring: ``I({w})`` is an exact upper bound on what ``w``
    can contribute to *any* seed set, so nodes are being ordered by the only
    quantity that can license discarding them.
    """
    if solo is None:
        solo = solo_influence_all(mia)
    ex = set(exclude)
    ranked = sorted((w for w in solo if w not in ex),
                    key=lambda w: (-solo[w], w))
    return ranked[: max(0, int(n))]


# ---------------------------------------------------------------------------
# exact swap delta
# ---------------------------------------------------------------------------


def swap_delta(mia, S_set, u, w) -> float:
    """``I(S - u + w) - I(S)``, exact, evaluated only where it can differ.

    Reference implementation used by the self-test and by the slow path; the
    round loop of `local_search` uses the cached decomposition instead (same
    value, fewer Eq (17) evaluations).
    """
    if u not in S_set or w in S_set:
        raise ValueError("swap_delta expects u in S and w outside S")
    S2 = set(S_set)
    S2.discard(u)
    S2.add(w)
    affected = set(mia.mioa(u)[0])
    affected.update(mia.mioa(w)[0])
    d = 0.0
    for v in affected:
        d += mia.ap(v, S2) - mia.ap(v, S_set)  # Eq (17), summed as in Eq (18)
    return d


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


def local_search(mia, S, candidates, rng, max_rounds=25, strategy="best",
                 tol=1e-9, solo=None, time_budget=None, verbose=False):
    """1-swap local search on the full graph.  Returns ``(S_best, value, stats)``.

    Parameters
    ----------
    mia          a `ctim.influence.MIA` -- the *shared* Eq (18) evaluator, so
                 the result is directly comparable with everything scored by it
    S            the starting seed set (any iterable; duplicates dropped)
    candidates   the pool of nodes that may be swapped *in*.  Nodes already in
                 `S` are ignored.  See `top_candidates` for how to build one
                 that keeps the local-optimum claim complete over all of V.
    rng          explicit `random.Random`; permutes scan order only
    max_rounds   hard cap on passes (a pass that accepts nothing terminates)
    strategy     ``"best"``  -- scan the whole neighbourhood, apply the single
                 best improving swap (steepest ascent); or
                 ``"first"`` -- apply the first improving swap found
    tol          a swap must gain more than this to be accepted
    solo         optional ``{w: I({w})}``; enables the bound-(*) prune and the
                 ``certified`` flag
    time_budget  optional wall-clock second budget; the search stops cleanly at
                 the end of a round once exceeded (never mid-swap)

    The returned value is a real ``mia.influence`` evaluation of ``S_best``, and
    ``S_best`` is the *input* set whenever no strict improvement was proven.
    """
    if strategy not in ("best", "first"):
        raise ValueError("strategy must be 'best' or 'first', got %r" % (strategy,))

    t_start = time.time()
    S_in = list(dict.fromkeys(S))
    cur = list(S_in)
    cur_set = set(cur)

    stats = {
        "rounds": 0,
        "swaps_accepted": 0,
        "influence_calls": 0,
        "ap_calls": 0,
        "pairs_evaluated": 0,
        "pairs_pruned": 0,
        "naive_influence_calls": 0,
        "candidates": 0,
        "live_candidates": 0,
        "final_min_loss": 0.0,
        "certified": False,
        "certificate_reason": "no `solo` supplied",
        "converged": False,
        "bound_violations": 0,
        "worst_bound_slack": 0.0,
        "min_corr": 0.0,
        "max_corr": 0.0,
        "swaps": [],
        "seconds": 0.0,
        "start_value": 0.0,
        "final_value": 0.0,
        "timed_out": False,
    }

    start_value = mia.influence(cur)  # Eq (18)
    stats["influence_calls"] += 1
    stats["start_value"] = start_value
    if not cur:
        stats["final_value"] = start_value
        stats["seconds"] = time.time() - t_start
        return list(cur), start_value, stats

    pool = [w for w in dict.fromkeys(candidates) if w not in cur_set]
    stats["candidates"] = len(pool)
    if not pool:
        stats["final_value"] = start_value
        stats["seconds"] = time.time() - t_start
        return list(cur), start_value, stats

    mioa_cache = {}

    def mioa_set(x):
        got = mioa_cache.get(x)
        if got is None:
            got = frozenset(mia.mioa(x)[0])  # Eq (16)
            mioa_cache[x] = got
        return got

    value = start_value
    min_corr = 0.0
    max_corr = 0.0
    converged = False
    cert_reason = ("every 1-swap over the supplied pool was evaluated or "
                   "provably pruned by bound (*)"
                   if solo is not None else "no `solo` supplied")

    for _round in range(int(max_rounds)):
        stats["rounds"] += 1
        base = {}   # v -> ap(v | S), reused across the whole pass

        def ap_base(v):
            got = base.get(v)
            if got is None:
                got = mia.ap(v, cur_set)  # Eq (17)
                base[v] = got
                stats["ap_calls"] += 1
            return got

        # -- per-seed removal losses L_u --------------------------------------
        loss = {}
        ap_minus = {}
        for u in cur:
            Mu = mioa_set(u)
            S_minus = set(cur_set)
            S_minus.discard(u)
            am = {}
            tot = 0.0
            for v in Mu:
                a = mia.ap(v, S_minus)  # Eq (17)
                stats["ap_calls"] += 1
                am[v] = a
                tot += ap_base(v) - a
            loss[u] = tot                      # L_u = I(S) - I(S - u)
            ap_minus[u] = am

        thresh = min(loss.values())
        stats["final_min_loss"] = thresh

        # -- bound (*): discard candidates that provably cannot improve -------
        if solo is not None:
            live = [w for w in pool if solo[w] > thresh + tol]
            stats["pairs_pruned"] += (len(pool) - len(live)) * len(cur)
        else:
            live = list(pool)
        stats["live_candidates"] = len(live)

        # -- per-candidate gains G_w -----------------------------------------
        gain = {}
        ap_plus = {}
        for w in live:
            Mw = mioa_set(w)
            S_plus = set(cur_set)
            S_plus.add(w)
            apn = {}
            tot = 0.0
            for v in Mw:
                a = mia.ap(v, S_plus)  # Eq (17)
                stats["ap_calls"] += 1
                apn[v] = a
                tot += a - ap_base(v)
            gain[w] = tot                      # G_w = I(S + w) - I(S)
            ap_plus[w] = apn

        # -- scan the neighbourhood ------------------------------------------
        order_u = list(cur)
        rng.shuffle(order_u)                   # ties only
        order_w = list(live)
        rng.shuffle(order_w)

        best_delta = tol
        best_pair = None
        stop_scan = False
        for u in order_u:
            Mu = mioa_set(u)
            L_u = loss[u]
            am = ap_minus[u]
            for w in order_w:
                if solo is not None and solo[w] - L_u <= best_delta:
                    # bound (*) again, now against the incumbent best swap
                    stats["pairs_pruned"] += 1
                    continue
                stats["pairs_evaluated"] += 1
                Mw = mioa_set(w)
                J = Mu & Mw
                corr = 0.0
                if J:
                    S2 = set(cur_set)
                    S2.discard(u)
                    S2.add(w)
                    apn = ap_plus[w]
                    for v in J:
                        a2 = mia.ap(v, S2)  # Eq (17)
                        stats["ap_calls"] += 1
                        corr += apn[v] + am[v] - a2 - ap_base(v)
                    if corr < min_corr:
                        min_corr = corr
                    if corr > max_corr:
                        max_corr = corr
                delta = gain[w] - L_u - corr
                if solo is not None:
                    # Direct empirical audit of bound (*): delta must never
                    # exceed I({w}) - L_u.  A violation would mean Eq (18) is
                    # not submodular on this instance and would void the
                    # completeness of the prune -- so it is recorded, not
                    # assumed away.
                    slack = delta - (solo[w] - L_u)
                    if slack > stats["worst_bound_slack"]:
                        stats["worst_bound_slack"] = slack
                    if slack > 1e-9:
                        stats["bound_violations"] += 1
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (u, w)
                    if strategy == "first":
                        stop_scan = True
                        break
            if stop_scan:
                break

        stats["naive_influence_calls"] += len(cur) * len(pool)

        if best_pair is None:
            converged = True   # a full clean pass: 1-swap local optimum
            break

        u, w = best_pair
        cur[cur.index(u)] = w
        cur_set.discard(u)
        cur_set.add(w)
        value += best_delta
        stats["swaps_accepted"] += 1
        stats["swaps"].append({"round": stats["rounds"], "out": u, "in": w,
                               "delta": best_delta})
        if verbose:
            print("    round %2d: swap out %d in %d  delta=+%.6f  -> %.6f"
                  % (stats["rounds"], u, w, best_delta, value))

        if time_budget is not None and (time.time() - t_start) > time_budget:
            stats["timed_out"] = True
            break

    # -- unconditional guarantee: verify, and revert if not strictly better ---
    final = mia.influence(cur)  # Eq (18)
    stats["influence_calls"] += 1
    if final <= start_value:
        cur = list(S_in)
        final = start_value
        stats["reverted"] = stats["swaps_accepted"] > 0
    else:
        stats["reverted"] = False

    stats["min_corr"] = min_corr
    stats["max_corr"] = max_corr
    stats["converged"] = converged
    certified = converged and solo is not None and stats["bound_violations"] == 0
    if not converged:
        cert_reason = ("no clean pass completed (%s)"
                       % ("time budget" if stats["timed_out"] else "max_rounds"))
    elif stats["bound_violations"]:
        cert_reason = ("bound (*) was violated on %d evaluated pair(s) by up to "
                       "%.3e -- Eq (18) is not submodular here, so the prune is "
                       "not provably complete"
                       % (stats["bound_violations"], stats["worst_bound_slack"]))
    stats["certified"] = certified
    stats["certificate_reason"] = cert_reason
    stats["final_value"] = final
    stats["seconds"] = time.time() - t_start
    return cur, final, stats


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import itertools
    import random

    from ctim.influence import MIA  # noqa: E402

    failures = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name
              + ((" -- " + detail) if detail else ""))
        if not cond:
            failures.append(name)

    def build(n, arcs):
        out = [[] for _ in range(n)]
        inn = [[] for _ in range(n)]
        pp = {}
        for (a, b, p) in arcs:
            out[a].append(b)
            inn[b].append(a)
            pp[(a, b)] = p
        return out, inn, pp

    # ------------------------------------------------------------------ [1]
    print("[1] hand-checkable max-cover instance where greedy is suboptimal")
    # Three "seed" nodes 0,1,2 pointing at item nodes 3..8 with pp = 1.0:
    #   node 0 -> {3,4,5,6}      (4 items)
    #   node 1 -> {3,4,7}        (3 items)
    #   node 2 -> {5,6,8}        (3 items)
    # Eq (18) with pp = 1 is |S| + |covered|, so for K = 2:
    #   greedy takes 0 first (I = 5), then 1 or 2 (adds one new item) -> 7
    #   the optimum is {1,2}: covers {3,4,5,6,7,8} -> 2 + 6 = 8
    # A single swap 0 -> 2 (or 0 -> 1) repairs greedy's mistake.
    arcs = [(0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0), (0, 6, 1.0),
            (1, 3, 1.0), (1, 4, 1.0), (1, 7, 1.0),
            (2, 5, 1.0), (2, 6, 1.0), (2, 8, 1.0)]
    out, inn, pp = build(9, arcs)
    m1 = MIA(9, out, inn, pp, h=0.1)

    greedy = m1.greedy_incremental(2)
    v_greedy = m1.influence(greedy)
    best_set, best_val = None, -1.0
    for comb in itertools.combinations(range(9), 2):
        val = m1.influence(list(comb))
        if val > best_val + 1e-12:
            best_set, best_val = comb, val
    check("brute force optimum is {1,2} with I = 8.0",
          set(best_set) == {1, 2} and abs(best_val - 8.0) < 1e-12,
          "got %s I=%.6f" % (sorted(best_set), best_val))
    check("greedy is strictly suboptimal here (I = 7.0)",
          abs(v_greedy - 7.0) < 1e-12,
          "greedy=%s I=%.6f" % (greedy, v_greedy))

    solo1 = solo_influence_all(m1)
    S1, v1, st1 = local_search(m1, greedy, list(range(9)),
                               random.Random(7), solo=solo1)
    check("(c) local search reaches the brute-force optimum",
          abs(v1 - best_val) < 1e-12 and set(S1) == set(best_set),
          "got %s I=%.6f in %d round(s), %d swap(s)"
          % (sorted(S1), v1, st1["rounds"], st1["swaps_accepted"]))
    check("stats report influence() calls and swaps",
          st1["influence_calls"] >= 2 and st1["swaps_accepted"] == 1,
          "influence_calls=%d swaps=%d ap_calls=%d"
          % (st1["influence_calls"], st1["swaps_accepted"], st1["ap_calls"]))

    # ------------------------------------------------------------------ [2]
    print("[2] solo_influence closed form == MIA.influence({w})")
    rng = random.Random(20190408)
    worst = 0.0
    for trial in range(6):
        n = 26
        eset = set()
        while len(eset) < 70:
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                eset.add((a, b))
        arcs = [(a, b, 0.05 + 0.9 * rng.random()) for (a, b) in sorted(eset)]
        out, inn, pp = build(n, arcs)
        mm = MIA(n, out, inn, pp, h=0.1)
        for w in range(n):
            ref = mm.influence({w})
            got = solo_influence(mm, w)
            worst = max(worst, abs(ref - got))
    check("closed form agrees on 6 x 26 nodes", worst <= 1e-9,
          "max abs err = %.3e" % worst)

    # ------------------------------------------------------------------ [3]
    print("[3] cached swap delta == full recomputation (exhaustive)")
    n = 30
    eset = set()
    while len(eset) < 95:
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            eset.add((a, b))
    arcs = [(a, b, 0.05 + 0.6 * rng.random()) for (a, b) in sorted(eset)]
    out3, inn3, pp3 = build(n, arcs)
    m3 = MIA(n, out3, inn3, pp3, h=0.1)
    solo3 = solo_influence_all(m3)

    S0 = m3.greedy_incremental(4)
    S0set = set(S0)
    base_I = m3.influence(S0)
    worst_d = 0.0
    for u in S0:
        for w in range(n):
            if w in S0set:
                continue
            ref = m3.influence([x for x in S0 if x != u] + [w]) - base_I
            got = swap_delta(m3, S0set, u, w)
            worst_d = max(worst_d, abs(ref - got))
    check("swap_delta exact over the whole neighbourhood", worst_d <= 1e-9,
          "max abs err = %.3e" % worst_d)

    # ------------------------------------------------------------------ [4]
    print("[4] (a) the result never scores below the input")
    ok_a = True
    detail_a = ""
    for trial in range(25):
        K = rng.randrange(2, 6)
        S_start = rng.sample(range(n), K)
        v_in = m3.influence(S_start)
        S_out, v_out, st = local_search(m3, S_start, list(range(n)),
                                        random.Random(1000 + trial), solo=solo3)
        v_check = m3.influence(S_out)
        if v_out < v_in - 1e-12 or abs(v_check - v_out) > 1e-9 or len(S_out) != K:
            ok_a = False
            detail_a = "trial %d: in %.6f -> out %.6f (recheck %.6f)" % (
                trial, v_in, v_out, v_check)
            break
    check("25 random starts, output >= input and |S| preserved", ok_a, detail_a)

    # ------------------------------------------------------------------ [5]
    print("[5] (b) determinism under a fixed seed")
    S_start = rng.sample(range(n), 4)
    runs = [local_search(m3, S_start, list(range(n)), random.Random(99),
                         solo=solo3) for _ in range(3)]
    check("three runs with random.Random(99) are byte-identical",
          runs[0][0] == runs[1][0] == runs[2][0]
          and runs[0][1] == runs[1][1] == runs[2][1],
          "%s / %s / %s" % (runs[0][0], runs[1][0], runs[2][0]))
    alt = local_search(m3, S_start, list(range(n)), random.Random(12345),
                       solo=solo3)
    check("a different seed still reaches a local optimum of equal value",
          abs(alt[1] - runs[0][1]) <= 1e-9,
          "%.6f vs %.6f (sets %s / %s)"
          % (alt[1], runs[0][1], sorted(alt[0]), sorted(runs[0][0])))

    # ------------------------------------------------------------------ [6]
    print("[6] the returned set really has no improving 1-swap left")
    S_end, v_end, st_end = local_search(m3, m3.greedy_incremental(4),
                                        list(range(n)), random.Random(5),
                                        solo=solo3)
    S_end_set = set(S_end)
    best_left = 0.0
    for u in S_end:
        for w in range(n):
            if w in S_end_set:
                continue
            best_left = max(best_left, swap_delta(m3, S_end_set, u, w))
    check("exhaustive check finds no improving swap", best_left <= 1e-9,
          "best remaining delta = %.3e" % best_left)
    check("the run reports itself certified, and bound (*) held everywhere",
          st_end["certified"] and st_end["bound_violations"] == 0
          and st_end["max_corr"] <= 1e-9,
          "certified=%s (%s); corr in [%.3e, %.3e] (submodularity predicts "
          "corr <= 0)" % (st_end["certified"], st_end["certificate_reason"],
                          st_end["min_corr"], st_end["max_corr"]))

    # ------------------------------------------------------------------ [7]
    print("[7] the bound-(*) prune loses nothing (pruned vs full pool)")
    S_start = m3.greedy_incremental(5)
    full = local_search(m3, S_start, list(range(n)), random.Random(3))
    pruned = local_search(m3, S_start, top_candidates(m3, 12, solo=solo3),
                          random.Random(3), solo=solo3)
    check("top-12 pool reaches the same value as the full pool",
          abs(full[1] - pruned[1]) <= 1e-9,
          "full %.6f (%d pairs) vs pruned %.6f (%d pairs)"
          % (full[1], full[2]["pairs_evaluated"],
             pruned[1], pruned[2]["pairs_evaluated"]))
    check("'first' strategy also never scores below the input",
          local_search(m3, S_start, list(range(n)), random.Random(3),
                       strategy="first", solo=solo3)[1]
          >= m3.influence(S_start) - 1e-12)

    print("")
    if failures:
        print("FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("ALL CHECKS PASSED")
