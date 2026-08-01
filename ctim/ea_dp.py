"""CTIM seed selection with EA-refined community curves and an EXACT allocation DP.

This is a *proposed variant*, not part of the paper.  ``ctim/ctim.py`` stays the
faithful transcription of Algorithm 2; this module explores two independent
changes to its lines 32-45 and reports them separately so each one's
contribution is visible on its own.

Change 1 -- exact allocation DP (``full-dp``)
---------------------------------------------
Algorithm 2's DP only ever considers adding **one** seed at a time:

    I[m,k] = max( I[m-1,k], I[m,k-1] + dI_m )            # line 35

That is a greedy-flavoured recurrence.  The textbook resource-allocation DP,
given each community's whole spread curve ``I_m(j)``, is

    I[m,k] = max_{0 <= j <= min(k, cap_m)} ( I[m-1, k-j] + I_m(j) )

which is *exactly optimal* for the given curves in ``O(C K^2)``.  Algorithm 2's
version coincides with it only when every ``I_m`` is concave; it is not in
general, because ``I_m`` is submodular set-wise but its *greedy* curve need not
be concave once the community saturates.

Change 2 -- EA-refined curves (``ea``)
--------------------------------------
``I_m(j)`` above is whatever within-community optimiser produced it.  Greedy
carries the ``(1 - 1/e)`` bound; an evolutionary algorithm carries none, so it
is only admissible if it *cannot lose*: the greedy solution is injected into the
initial population and elitism is unconditional, making the EA curve pointwise
``>= `` the greedy curve by construction.

The EA is affordable here for one specific reason: ``I_m`` is
``MIA.influence`` -- exact, closed form, deterministic (Eq (17)/(18)).  Most
GA-for-influence-maximization work fights Monte-Carlo fitness noise; there is
none to fight here, so a small population and few generations suffice.

Because the DP consumes curves, not marginal gains, a non-concave EA curve is
handled correctly -- which is precisely the case where change 1 pays off.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random``.
"""

from __future__ import annotations

import os
import sys
import time

if __name__ == "__main__" and __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctim.ctim import _CommunitySeedState, detect_communities  # noqa: E402
from ctim.influence import MIA  # noqa: E402

__all__ = [
    "CommunityCurve",
    "build_curves",
    "allocate_exact",
    "ea_refine",
    "select_seeds_ea_dp",
]


# ---------------------------------------------------------------------------
# per-community spread curves
# ---------------------------------------------------------------------------


class CommunityCurve:
    """Community ``m``'s spread curve and the seed set realising each point.

    ``I[j]`` is the within-community spread of ``sets[j]`` (``|sets[j]| == j``),
    with ``I[0] = 0`` and ``sets[0] = []``.  ``I`` is repaired to be
    non-decreasing: the DP below assumes more budget never hurts, and a noisy
    optimiser could otherwise report ``I[4] < I[3]``.
    """

    __slots__ = ("m", "members", "mia", "I", "sets", "n_fitness_evals")

    def __init__(self, m, members, mia):
        self.m = m
        self.members = members
        self.mia = mia
        self.I = [0.0]
        self.sets = [[]]
        self.n_fitness_evals = 0

    @property
    def cap(self):
        return len(self.I) - 1

    def repair_monotone(self):
        """Force ``I`` non-decreasing, carrying the better set forward."""
        for j in range(1, len(self.I)):
            if self.I[j] < self.I[j - 1]:
                self.I[j] = self.I[j - 1]
                self.sets[j] = list(self.sets[j - 1])


def build_curves(comm, ds, pp, K, h, verbose=False):
    """Greedy curve ``I_m(j)``, ``j = 0..min(K, |c_m|)``, for every community.

    Uses the very same ``_CommunitySeedState`` machinery Algorithm 2 line 34
    uses, so the greedy curve here is exactly the sequence of marginal gains
    CTIM would have consumed -- the comparison downstream is apples to apples.
    ``I_m(S_j)`` is the running sum of the exact marginal gains.
    """
    n_comm = (max(comm) + 1) if len(comm) else 0
    members = [[] for _ in range(n_comm)]
    n = min(len(comm), ds.n_users)
    for v in range(n):
        members[comm[v]].append(v)

    curves = []
    t_mia = 0.0
    t_greedy = 0.0
    for m in range(n_comm):
        mem = members[m]
        if not mem:
            continue
        t0 = time.perf_counter()
        # I_m on community m's node-induced subgraph (Algorithm 2, note 2)
        mia_m = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h, nodes=mem)
        t_mia += time.perf_counter() - t0

        t0 = time.perf_counter()
        cur = CommunityCurve(m, mem, mia_m)
        state = _CommunitySeedState(m + 1, mia_m, mem)
        total = 0.0
        picked = []
        for _ in range(min(K, len(mem))):
            u, gain = state.best()  # Algorithm 2, line 34
            if u is None:
                break
            state.add(u)
            picked.append(u)
            total += gain  # exact marginal gain -> running I_m(S_j)
            cur.I.append(total)
            cur.sets.append(list(picked))
        t_greedy += time.perf_counter() - t0
        cur.repair_monotone()
        curves.append(cur)

    if verbose:
        print("    [curves] %d non-empty communities  MIA %.2fs  greedy %.2fs"
              % (len(curves), t_mia, t_greedy))
    return curves, {"mia_seconds": t_mia, "greedy_seconds": t_greedy}


# ---------------------------------------------------------------------------
# EA refinement of one curve point
# ---------------------------------------------------------------------------


def ea_refine(curve, j, rng, pop_size=24, generations=15, tournament=3,
              mutation_rate=0.3, warm=None):
    """Search for a size-``j`` seed set in community ``m`` beating greedy's.

    Returns ``(best_set, best_value, n_evals)``.  ``best_value`` is guaranteed
    ``>= `` the greedy value for the same ``j``: greedy's set seeds the initial
    population and the elite is carried through every generation unconditionally.

    Fitness is ``MIA.influence`` on the induced subgraph -- exact Eq (18), no
    Monte Carlo -- memoised per set, so re-evaluating a survivor is free.
    """
    mia = curve.mia
    members = curve.members
    if j <= 0 or j >= len(members):
        return list(curve.sets[j]) if j < len(curve.sets) else list(members), \
            (curve.I[j] if j < len(curve.I) else 0.0), 0

    cache = {}
    n_evals = [0]

    def fitness(sset):
        key = frozenset(sset)
        got = cache.get(key)
        if got is None:
            got = mia.influence(list(key))  # Eq (18)
            cache[key] = got
            n_evals[0] += 1
        return got

    # ---- initial population -------------------------------------------
    pop = []
    seen = set()

    def push(cand):
        key = frozenset(cand)
        if len(key) == j and key not in seen:
            seen.add(key)
            pop.append(sorted(key))

    push(curve.sets[j])                      # greedy -- the floor we cannot lose to
    if warm:
        push(warm)
    if j >= 1 and j - 1 < len(curve.sets):
        # greedy's (j-1)-set plus one random member: a cheap distinct neighbour
        base = curve.sets[j - 1]
        pool = [u for u in members if u not in base]
        for _ in range(3):
            if pool:
                push(list(base) + [rng.choice(pool)])
    while len(pop) < pop_size:
        push(rng.sample(members, j))
        if len(seen) >= _max_distinct(len(members), j):
            break

    scored = [(fitness(s), s) for s in pop]
    scored.sort(key=lambda t: (-t[0], t[1]))
    best_val, best_set = scored[0]

    # ---- generations ---------------------------------------------------
    for _ in range(generations):
        nxt = [best_set]  # unconditional elitism
        nxt_seen = {frozenset(best_set)}
        while len(nxt) < pop_size:
            pa = _tournament(scored, rng, tournament)
            pb = _tournament(scored, rng, tournament)
            child = _crossover(pa, pb, j, rng)
            if rng.random() < mutation_rate:
                child = _mutate(child, members, rng)
            key = frozenset(child)
            if len(key) == j and key not in nxt_seen:
                nxt_seen.add(key)
                nxt.append(sorted(key))
            elif len(nxt_seen) >= _max_distinct(len(members), j):
                break
        scored = [(fitness(s), s) for s in nxt]
        scored.sort(key=lambda t: (-t[0], t[1]))
        if scored[0][0] > best_val:
            best_val, best_set = scored[0]

    return list(best_set), best_val, n_evals[0]


def _max_distinct(n, j):
    """A cheap ceiling on how many distinct j-subsets exist (avoids infinite loops)."""
    if j <= 0 or j > n:
        return 1
    out = 1
    for t in range(j):
        out = out * (n - t) // (t + 1)
        if out > 100000:
            return 100000
    return out


def _tournament(scored, rng, k):
    """Pick the best of ``k`` random individuals."""
    best = None
    for _ in range(k):
        cand = scored[rng.randrange(len(scored))]
        if best is None or cand[0] > best[0]:
            best = cand
    return best[1]


def _crossover(pa, pb, j, rng):
    """Draw ``j`` distinct nodes from the union of both parents."""
    union = list(dict.fromkeys(list(pa) + list(pb)))
    if len(union) <= j:
        return list(union)
    return rng.sample(union, j)


def _mutate(child, members, rng):
    """Swap one node out for a member not currently in the set."""
    if not child:
        return child
    out = list(child)
    idx = rng.randrange(len(out))
    for _ in range(8):
        cand = rng.choice(members)
        if cand not in out:
            out[idx] = cand
            return out
    return out


def refine_curves(curves, K, rng, top_t=0, pop_size=24, generations=15,
                  verbose=False):
    """Run the EA over the most promising communities' curve points.

    ``top_t`` communities are refined, ranked by their greedy curve's endpoint
    (the spread they could contribute if handed the whole budget).  ``top_t=0``
    refines every community.  Ranking by realised greedy spread strictly
    dominates ranking by member count: a large sparse community contributes
    less than a small dense one, which is the whole reason Algorithm 2 has a DP
    rather than a proportional split.
    """
    ranked = sorted(curves, key=lambda c: (-c.I[c.cap], c.m))
    targets = ranked if top_t <= 0 else ranked[:top_t]

    t0 = time.perf_counter()
    n_evals = 0
    n_improved = 0
    total_gain = 0.0
    for cur in targets:
        warm = None
        for j in range(1, cur.cap + 1):
            before = cur.I[j]
            best_set, best_val, ev = ea_refine(
                cur, j, rng, pop_size=pop_size, generations=generations, warm=warm)
            n_evals += ev
            if best_val > before + 1e-12:
                cur.I[j] = best_val
                cur.sets[j] = best_set
                n_improved += 1
                total_gain += best_val - before
            warm = list(cur.sets[j]) + [u for u in cur.members
                                        if u not in cur.sets[j]][:1]
        cur.repair_monotone()
        cur.n_fitness_evals = n_evals

    secs = time.perf_counter() - t0
    if verbose:
        print("    [ea] refined %d/%d communities, %d fitness evals, "
              "%d curve points improved (total +%.4f) in %.2fs"
              % (len(targets), len(curves), n_evals, n_improved, total_gain, secs))
    return {"ea_seconds": secs, "n_fitness_evals": n_evals,
            "n_points_improved": n_improved, "ea_curve_gain": total_gain,
            "n_communities_refined": len(targets)}


# ---------------------------------------------------------------------------
# exact allocation DP
# ---------------------------------------------------------------------------


def allocate_exact(curves, K):
    """Exact resource-allocation DP over the community curves.

        I[m][k] = max_{0 <= j <= min(k, cap_m)} ( I[m-1][k-j] + I_m(j) )

    ``O(C K^2)``.  Returns ``(seeds, dp_value, alloc)`` where ``alloc[m]`` is the
    budget community ``m`` received.  Ties go to the smaller ``j`` (fewer seeds
    for the same value) and then to the lower community index, so the result is
    deterministic.
    """
    C = len(curves)
    NEG = float("-inf")
    I = [[NEG] * (K + 1) for _ in range(C + 1)]
    take = [[0] * (K + 1) for _ in range(C + 1)]
    for k in range(K + 1):
        I[0][k] = 0.0

    for m in range(1, C + 1):
        cur = curves[m - 1]
        cap = cur.cap
        Im = cur.I
        for k in range(K + 1):
            best_v = NEG
            best_j = 0
            top = cap if cap < k else k
            for j in range(top + 1):
                prev = I[m - 1][k - j]
                if prev == NEG:
                    continue
                v = prev + Im[j]
                if v > best_v + 1e-15:
                    best_v = v
                    best_j = j
            I[m][k] = best_v
            take[m][k] = best_j

    seeds = []
    alloc = {}
    k = K
    for m in range(C, 0, -1):
        j = take[m][k]
        if j > 0:
            alloc[curves[m - 1].m] = j
            seeds.extend(curves[m - 1].sets[j])
            k -= j
    seeds.reverse()
    return seeds, I[C][K], alloc


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------


def select_seeds_ea_dp(model, ds, pp, K, rng, h=0.1, use_ea=True, top_t=10,
                       pop_size=24, generations=15, comm=None, verbose=False):
    """Full pipeline: Eq (19) communities -> curves -> (EA) -> exact DP.

    Returns ``(seeds, stats)``.  ``stats`` carries a per-phase wall-clock
    breakdown so the cost of each change can be reported next to its benefit.
    """
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    if comm is None:
        comm = detect_communities(model.pi)  # Eq (19)
    t_detect = time.perf_counter() - t0

    curves, curve_stats = build_curves(comm, ds, pp, K, h, verbose=verbose)

    ea_stats = {"ea_seconds": 0.0, "n_fitness_evals": 0, "n_points_improved": 0,
                "ea_curve_gain": 0.0, "n_communities_refined": 0}
    if use_ea:
        ea_stats = refine_curves(curves, K, rng, top_t=top_t, pop_size=pop_size,
                                 generations=generations, verbose=verbose)

    t0 = time.perf_counter()
    seeds, dp_value, alloc = allocate_exact(curves, K)
    t_dp = time.perf_counter() - t0

    stats = {
        "detect_seconds": t_detect,
        "dp_seconds": t_dp,
        "total_seconds": time.perf_counter() - t_start,
        "dp_value": dp_value,
        "n_communities": len(curves),
        "n_communities_used": len(alloc),
        "allocation": dict(sorted(alloc.items())),
        "K_selected": len(seeds),
    }
    stats.update(curve_stats)
    stats.update(ea_stats)
    return seeds, stats
