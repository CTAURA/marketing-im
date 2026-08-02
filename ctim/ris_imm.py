"""IMM -- influence maximization via martingales, on top of reverse influence sampling.

This is a *proposed alternative* to Algorithm 2's seed selection, not part of the
paper.  ``ctim/ctim.py`` stays the faithful transcription; this module is an
independent selector scored with the same shared Eq (18) evaluator, so the two
can be compared on equal terms.

    [B14] C. Borgs, M. Brautbar, J. Chayes, B. Lucier.  "Maximizing social
          influence in nearly optimal time."  SODA 2014.
    [T15] Y. Tang, Y. Shi, X. Xiao.  "Influence maximization in near-linear time:
          a martingale approach."  SIGMOD 2015.

Why it is worth trying here
---------------------------
CTIM's influence computation model is MIA (Eq (13)-(18)), which approximates the
Independent Cascade spread by truncating every propagation path at
``pp(P) >= h``.  On the Digg benchmark that threshold admits **only single-hop
paths** (measured: mean ``|MIIA(v,h)| - 1 = 1.08``, max depth 1 at ``h = 0.1``),
so MIA is not evaluating a cascade at all.

RIS needs no threshold.  A *random reverse-reachable* (RR) set is built by
picking a node ``v`` uniformly at random and collecting everything that reaches
``v`` in one live-edge sample of ``G``:

    R_v = { u : u reaches v when each edge (u,x) is kept with probability pp(u,x) }

Borgs et al. show ``sigma(S) = n * Pr[ S intersects R_v ]``, so maximising the
*fraction of RR sets hit* is exactly maximising IC spread.  Greedy max-coverage
over the sampled RR sets therefore inherits the ``(1 - 1/e)`` guarantee, and IMM
supplies the number of samples ``theta`` needed to make the whole procedure
``(1 - 1/e - eps)``-approximate with probability ``1 - n^-ell``.

Trade-off, stated plainly: MIA is exact-but-truncated and deterministic; IMM is
untruncated-but-sampled and randomised.  Neither dominates.  The point of this
module is to *measure* which matters more on a given dataset.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random``; there is no module-level ``random.*`` use.
"""

from __future__ import annotations

import heapq
import math
import os
import sys
import time

if __name__ == "__main__" and __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = [
    "generate_rr_set",
    "node_selection",
    "imm_select_seeds",
    "ic_simulate",
]


# ---------------------------------------------------------------------------
# RR set generation
# ---------------------------------------------------------------------------


def generate_rr_set(root, in_adj, pp, rng, n_users):
    """One random reverse-reachable set rooted at ``root``.

    Backward BFS: from each node ``x`` already in the set, every in-edge
    ``(u, x)`` is traversed with probability ``pp(u, x)`` -- one live-edge
    sample restricted to the part that matters for ``root``.  Coin flips are
    drawn lazily, which is what keeps a single RR set cheap even on a dense
    graph (Borgs et al. [B14]).
    """
    seen = {root}
    stack = [root]
    rand = rng.random
    while stack:
        x = stack.pop()
        ins = in_adj[x] if x < len(in_adj) else ()
        for u in ins:
            if u in seen:
                continue
            p = pp.get((u, x), 0.0)
            if p > 0.0 and rand() < p:
                seen.add(u)
                stack.append(u)
    return seen


def _grow_rr(rr, occ, target, in_adj, pp, rng, n_users):
    """Extend ``rr`` (and its inverted index ``occ``) up to ``target`` sets."""
    while len(rr) < target:
        idx = len(rr)
        root = rng.randrange(n_users)
        s = generate_rr_set(root, in_adj, pp, rng, n_users)
        rr.append(s)
        for u in s:
            lst = occ.get(u)
            if lst is None:
                occ[u] = [idx]
            else:
                lst.append(idx)


# ---------------------------------------------------------------------------
# Greedy max-coverage over the RR sets  (IMM's NodeSelection)
# ---------------------------------------------------------------------------


def node_selection(rr, occ, k, n_users):
    """Greedy max-coverage: pick ``k`` nodes hitting the most RR sets.

    Returns ``(seeds, n_covered)``.  ``n_covered / len(rr)`` is the unbiased
    estimate of ``sigma(S) / n``.

    Lazy-forward heap: coverage is submodular, so a stale count is an upper
    bound and only has to be recomputed when it surfaces at the top.  Ties go to
    the lowest node id, which keeps the result deterministic given the RNG.
    """
    if not rr or k <= 0:
        return [], 0

    counts = {u: len(lst) for u, lst in occ.items()}
    heap = [(-c, u) for u, c in counts.items()]
    heapq.heapify(heap)

    covered = bytearray(len(rr))
    seeds = []
    n_covered = 0
    chosen = set()

    while heap and len(seeds) < k:
        neg, u = heapq.heappop(heap)
        if u in chosen:
            continue
        cur = counts[u]
        if -neg > cur:            # stale upper bound -- reprice and requeue
            heapq.heappush(heap, (-cur, u))
            continue
        if cur <= 0:
            break                 # nothing left to cover
        chosen.add(u)
        seeds.append(u)
        for idx in occ[u]:
            if covered[idx]:
                continue
            covered[idx] = 1
            n_covered += 1
            for w in rr[idx]:     # this RR set is now accounted for
                if w in counts:
                    counts[w] -= 1
    return seeds, n_covered


# ---------------------------------------------------------------------------
# IMM
# ---------------------------------------------------------------------------


def _log_n_choose_k(n, k):
    """ln C(n,k) via lgamma -- exact enough and never overflows."""
    if k <= 0 or k >= n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def imm_select_seeds(n_users, in_adj, pp, k, rng, eps=0.5, ell=1.0,
                     max_rr=2_000_000, verbose=False):
    """IMM [T15]: sampling phase to fix ``theta``, then greedy max-coverage.

    ``eps``   approximation slack; the guarantee is ``(1 - 1/e - eps)``.
    ``ell``   failure probability exponent: succeeds w.p. ``1 - n^-ell``.
    ``max_rr`` hard ceiling on RR sets, so a pathological ``theta`` cannot hang
              the run.  If it binds, ``stats["theta_capped"]`` is True and the
              guarantee no longer holds -- the caller is told rather than
              silently given a weaker result.

    Returns ``(seeds, stats)``.
    """
    t0 = time.perf_counter()
    n = int(n_users)
    if n <= 0 or k <= 0:
        return [], {"theta": 0, "n_rr": 0, "seconds": 0.0}

    # [T15] inflates ell to pay for the union bound over the estimation phase.
    ell_adj = ell * (1.0 + math.log(2.0) / math.log(n)) if n > 1 else ell
    log_nk = _log_n_choose_k(n, k)
    log_n = math.log(n) if n > 1 else 1.0

    # ---- phase 1: estimate a lower bound LB on OPT -------------------------
    eps_p = math.sqrt(2.0) * eps
    lam_p = ((2.0 + 2.0 * eps_p / 3.0)
             * (log_nk + ell_adj * log_n + math.log(max(2.0, math.log2(n))))
             * n / (eps_p * eps_p))

    rr = []
    occ = {}
    LB = 1.0
    t_sampling = time.perf_counter()
    n_rounds = int(math.log2(n)) if n > 2 else 1
    for i in range(1, n_rounds):
        x = n / (2.0 ** i)
        theta_i = int(lam_p / x) + 1
        if theta_i > max_rr:
            theta_i = max_rr
        _grow_rr(rr, occ, theta_i, in_adj, pp, rng, n)
        _s, cov = node_selection(rr, occ, k, n)
        frac = cov / float(len(rr))
        if verbose:
            print("    [imm] round %2d  theta_i=%-9d  n*F=%.2f  vs (1+eps')x=%.2f"
                  % (i, theta_i, n * frac, (1.0 + eps_p) * x))
        if n * frac >= (1.0 + eps_p) * x:
            LB = n * frac / (1.0 + eps_p)
            break
        if len(rr) >= max_rr:
            break
    t_sampling = time.perf_counter() - t_sampling

    # ---- phase 2: theta from LB, then the final selection ------------------
    alpha = math.sqrt(ell_adj * log_n + math.log(2.0))
    beta = math.sqrt((1.0 - 1.0 / math.e) * (log_nk + ell_adj * log_n + math.log(2.0)))
    lam_star = 2.0 * n * ((1.0 - 1.0 / math.e) * alpha + beta) ** 2 / (eps * eps)
    theta = int(lam_star / max(LB, 1e-12)) + 1

    capped = theta > max_rr
    if capped:
        theta = max_rr

    t_grow = time.perf_counter()
    _grow_rr(rr, occ, theta, in_adj, pp, rng, n)
    t_grow = time.perf_counter() - t_grow

    t_sel = time.perf_counter()
    seeds, cov = node_selection(rr, occ, k, n)
    t_sel = time.perf_counter() - t_sel

    stats = {
        "theta": theta,
        "theta_capped": capped,
        "n_rr": len(rr),
        "LB": LB,
        "eps": eps,
        "ell": ell,
        "rr_estimate": n * cov / float(len(rr)) if rr else 0.0,
        "mean_rr_size": (sum(len(s) for s in rr) / float(len(rr))) if rr else 0.0,
        "sampling_seconds": t_sampling,
        "grow_seconds": t_grow,
        "select_seconds": t_sel,
        "seconds": time.perf_counter() - t0,
    }
    return seeds, stats


# ---------------------------------------------------------------------------
# An independent Monte-Carlo IC oracle (neither MIA's nor IMM's own objective)
# ---------------------------------------------------------------------------


def ic_simulate(out_adj, pp, S, n_mc, rng):
    """Plain Independent Cascade Monte-Carlo estimate of sigma(S).

    Deliberately shares no machinery with MIA or with the RR sampler: it is the
    neutral referee when those two disagree.
    """
    if not S:
        return 0.0
    S = list(S)
    rand = rng.random
    total = 0
    for _ in range(n_mc):
        active = set(S)
        frontier = list(S)
        while frontier:
            x = frontier.pop()
            outs = out_adj[x] if x < len(out_adj) else ()
            for v in outs:
                if v in active:
                    continue
                p = pp.get((x, v), 0.0)
                if p > 0.0 and rand() < p:
                    active.add(v)
                    frontier.append(v)
        total += len(active)
    return total / float(n_mc)


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

    print("[1] RR sets on a hand-checked graph")
    # 0 -> 1 -> 2, all edges certain: RR(2) must be {0,1,2}, RR(0) must be {0}.
    out_adj = [[1], [2], []]
    in_adj = [[], [0], [1]]
    pp = {(0, 1): 1.0, (1, 2): 1.0}
    rng = random.Random(1)
    check("RR(2) = {0,1,2} with certain edges",
          generate_rr_set(2, in_adj, pp, rng, 3) == {0, 1, 2})
    check("RR(0) = {0} (no in-edges)",
          generate_rr_set(0, in_adj, pp, rng, 3) == {0})
    pp0 = {(0, 1): 0.0, (1, 2): 0.0}
    check("dead edges give singleton RR sets",
          generate_rr_set(2, in_adj, pp0, rng, 3) == {2})

    print("[2] RIS estimate matches Monte-Carlo IC on a random graph")
    rng = random.Random(7)
    N = 60
    out_adj = [[] for _ in range(N)]
    in_adj = [[] for _ in range(N)]
    pp = {}
    for _ in range(300):
        a, b = rng.randrange(N), rng.randrange(N)
        if a != b and (a, b) not in pp:
            pp[(a, b)] = 0.05 + 0.25 * rng.random()
            out_adj[a].append(b)
            in_adj[b].append(a)

    S = [0, 1, 2]
    rr, occ = [], {}
    _grow_rr(rr, occ, 20000, in_adj, pp, random.Random(11), N)
    hit = sum(1 for s in rr if s & set(S))
    ris_est = N * hit / float(len(rr))
    mc_est = ic_simulate(out_adj, pp, S, 20000, random.Random(12))
    check("RIS estimate == MC IC estimate within 3%%",
          abs(ris_est - mc_est) / max(mc_est, 1e-9) < 0.03,
          "RIS %.3f vs MC %.3f" % (ris_est, mc_est))

    print("[3] IMM returns k distinct seeds and beats random")
    seeds, st = imm_select_seeds(N, in_adj, pp, 5, random.Random(3), eps=0.3)
    check("k distinct seeds in range",
          len(seeds) == 5 and len(set(seeds)) == 5
          and all(0 <= u < N for u in seeds), "%s" % seeds)
    imm_mc = ic_simulate(out_adj, pp, seeds, 4000, random.Random(4))
    rr_rng = random.Random(5)
    rand_mc = max(ic_simulate(out_adj, pp, rr_rng.sample(range(N), 5), 4000,
                              random.Random(6)) for _ in range(8))
    check("IMM beats the best of 8 random seed sets",
          imm_mc > rand_mc, "IMM %.3f vs random-best %.3f" % (imm_mc, rand_mc))
    check("theta and RR bookkeeping reported",
          st["theta"] > 0 and st["n_rr"] >= st["theta"],
          "theta=%d n_rr=%d mean|RR|=%.2f" % (st["theta"], st["n_rr"],
                                              st["mean_rr_size"]))

    print("[4] determinism")
    a, _ = imm_select_seeds(N, in_adj, pp, 5, random.Random(99), eps=0.3)
    b, _ = imm_select_seeds(N, in_adj, pp, 5, random.Random(99), eps=0.3)
    check("same seed -> same seeds", a == b, "%s vs %s" % (a, b))

    print("[5] edge cases")
    check("k=0 returns nothing", imm_select_seeds(N, in_adj, pp, 0,
                                                  random.Random(1))[0] == [])
    empty_in = [[] for _ in range(5)]
    s5, _ = imm_select_seeds(5, empty_in, {}, 3, random.Random(1), eps=0.5)
    check("graph with no edges still returns k seeds",
          len(s5) == 3 and len(set(s5)) == 3, "%s" % s5)

    print("")
    if failures:
        print("FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("ALL CHECKS PASSED")
