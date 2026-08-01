#!/usr/bin/env python3
"""Sanity-check the EA before trusting a '0 improvements' result.

A search that reports no improvement is indistinguishable from a search that
never searched.  This script separates the two:

  A. the EA really does explore -- it evaluates many DISTINCT sets whose
     fitness genuinely varies (so the population is not collapsing onto the
     greedy seed);
  B. on small instances where the optimum is computable by brute force,
     greedy / EA / optimum are compared directly.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import itertools
import os
import pickle
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.ctim import detect_communities          # noqa: E402
from ctim.dataset import load_dataset             # noqa: E402
from ctim.ea_dp import build_curves, ea_refine    # noqa: E402
from ctim.influence import EdgeWeights            # noqa: E402


def main():
    t_all = time.perf_counter()
    ds = load_dataset(os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    with open(os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"), "rb") as fh:
        blob = pickle.load(fh)
    model, item = blob["model"], blob["test_items"][0]
    ew = EdgeWeights(model, ds)
    t0 = time.perf_counter()
    pp = ew.for_item(item)
    print("[setup] Eq (12) weights %.2fs" % (time.perf_counter() - t0))

    comm = detect_communities(model.pi)
    curves, _ = build_curves(comm, ds, pp, 20, 0.1)
    curves.sort(key=lambda c: (-c.I[c.cap], c.m))
    print("[setup] %d curves; top-5 by endpoint: %s"
          % (len(curves), [(c.m, len(c.members), round(c.I[c.cap], 3))
                           for c in curves[:5]]))
    print("        total setup %.2fs\n" % (time.perf_counter() - t_all))

    rng = random.Random(7)

    # ---- A. does the EA actually explore? ------------------------------
    print("=" * 72)
    print("A. exploration check -- instrumented ea_refine on the top community")
    print("=" * 72)
    cur = curves[0]
    for j in (2, 5, 10):
        if j > cur.cap:
            continue
        mia = cur.mia
        seen = {}

        orig_influence = mia.influence

        def traced(S, _seen=seen, _f=orig_influence):
            key = frozenset(S)
            v = _f(S)
            _seen[key] = v
            return v

        mia.influence = traced
        t0 = time.perf_counter()
        best_set, best_val, n_ev = ea_refine(cur, j, random.Random(7),
                                             pop_size=24, generations=15)
        secs = time.perf_counter() - t0
        mia.influence = orig_influence

        vals = sorted(seen.values())
        greedy_val = cur.I[j]
        print("  community %d (|c|=%d), j=%d   %.3fs" % (cur.m, len(cur.members), j, secs))
        print("    distinct sets evaluated : %d" % len(seen))
        print("    fitness min/med/max     : %.4f / %.4f / %.4f"
              % (vals[0], vals[len(vals) // 2], vals[-1]))
        print("    greedy value            : %.4f" % greedy_val)
        print("    EA best                 : %.4f  (%s)"
              % (best_val, "improved" if best_val > greedy_val + 1e-12 else "no gain"))
        print("    sets strictly WORSE than greedy: %d / %d"
              % (sum(1 for v in vals if v < greedy_val - 1e-12), len(vals)))

    # ---- B. how many communities can propagate at all? ------------------
    print("\n" + "=" * 72)
    print("B. do the communities carry any internal propagation?")
    print("=" * 72)
    # I_m(j) == j means every seed activates only itself: zero propagation.
    dead = [c for c in curves if c.I[c.cap] <= c.cap + 1e-9]
    print("  communities whose curve is exactly I_m(j) = j (no propagation at all):")
    print("    %d / %d" % (len(dead), len(curves)))
    live = [c for c in curves if c.I[c.cap] > c.cap + 1e-9]
    print("  communities with ANY propagation: %d -> %s"
          % (len(live), [(c.m, len(c.members), round(c.I[c.cap], 3)) for c in live]))

    # ---- C. restricted brute force on the communities that matter -------
    print("\n" + "=" * 72)
    print("C. restricted brute-force optimum vs greedy vs EA")
    print("=" * 72)
    print("  Full brute force is impossible (|c| in the thousands), so the search")
    print("  is restricted to the TOP-N single-node influencers of the community.")
    print("  Greedy's picks are always inside that pool, so this is a genuine")
    print("  (if partial) optimality test on the instances that carry the budget.")
    tested = n_greedy_opt = n_ea_opt = 0
    for cur in live[:2]:
        singles = sorted(((cur.mia.influence([u]), u) for u in cur.members),
                         key=lambda t: (-t[0], t[1]))
        for topn, j in ((30, 3), (20, 4)):
            pool = [u for (_v, u) in singles[:topn]]
            missing = [u for u in cur.sets[j] if u not in pool] if j < len(cur.sets) else []
            pool = pool + missing  # keep greedy's own picks in the search space
            if j > cur.cap:
                continue
            t0 = time.perf_counter()
            best_v = -1.0
            n_sets = 0
            for combo in itertools.combinations(pool, j):
                v = cur.mia.influence(list(combo))  # Eq (18)
                n_sets += 1
                if v > best_v:
                    best_v = v
            t_bf = time.perf_counter() - t0
            g = cur.I[j]
            _es, ev, _n = ea_refine(cur, j, random.Random(7),
                                    pop_size=24, generations=15)
            tested += 1
            g_opt = g >= best_v - 1e-9
            e_opt = ev >= best_v - 1e-9
            n_greedy_opt += g_opt
            n_ea_opt += e_opt
            print("  m=%-4d |c|=%-5d j=%d  pool=%-3d  restricted-opt %.4f  "
                  "greedy %.4f%s  EA %.4f%s   [%d sets, %.2fs]"
                  % (cur.m, len(cur.members), j, len(pool), best_v,
                     g, " OPT" if g_opt else " ***", ev, " OPT" if e_opt else " ***",
                     n_sets, t_bf))
    if tested:
        print("\n  greedy matched the restricted optimum on %d/%d instances"
              % (n_greedy_opt, tested))
        print("  EA     matched the restricted optimum on %d/%d instances"
              % (n_ea_opt, tested))

    print("\nTOTAL %.2fs" % (time.perf_counter() - t_all))


def _n_choose(n, k):
    out = 1
    for t in range(k):
        out = out * (n - t) // (t + 1)
    return out


if __name__ == "__main__":
    main()
