#!/usr/bin/env python3
"""Compare CTIM's Algorithm 2 selection against the exact-DP and EA variants.

Four selectors, ONE evaluator
-----------------------------
Every variant's seed set is scored with the same exact Eq (18) MIA influence on
the **full** graph, so the numbers are directly comparable (API.md: one shared
evaluator for every method).

  1. ``CTIM``            Algorithm 2 lines 32-45 verbatim (``ctim/ctim.py``)
  2. ``CTIM+fullDP``     greedy community curves + the exact O(C K^2)
                         resource-allocation DP
  3. ``CTIM+EA+fullDP``  EA-refined community curves + the same exact DP
  4. ``GlobalGreedy``    MIA greedy on the whole graph, no community
                         decomposition -- an upper reference that measures what
                         the community decomposition itself costs in spread

Variant 4 matters: CTIM optimises the *within-community* spread ``I_m`` but is
scored on the *global* ``I``.  The gap between 1-3 and 4 is that objective
mismatch, and no amount of within-community search can close it.

Every phase prints wall-clock seconds.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.ctim import ctim_select_seeds, detect_communities  # noqa: E402
from ctim.dataset import load_dataset                        # noqa: E402
from ctim.ea_dp import select_seeds_ea_dp                    # noqa: E402
from ctim.influence import MIA, EdgeWeights                  # noqa: E402


def fmt(s):
    return "%.3fs" % s


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-t", type=int, default=15,
                   help="communities the EA refines (0 = all)")
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--check-topic", action="store_true",
                   help="also compare Eq (12) weights across two items")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("=" * 78)
    print("EA / exact-DP variants vs Algorithm 2   K=%d  h=%g  seed=%d"
          % (args.K, args.h, args.seed))
    print("=" * 78)

    if not os.path.exists(args.cache):
        print("ERROR: no cached model at %s -- run scripts/calibrate_h.py first."
              % args.cache)
        return 1

    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model = blob["model"]
    test_items = blob["test_items"]
    item = test_items[0]
    ew = EdgeWeights(model, ds)  # Eq (10)+(11)
    print("[setup] dataset + cached model + EdgeWeights      %s"
          % fmt(time.perf_counter() - t0))

    t0 = time.perf_counter()
    pp = ew.for_item(item)  # Eq (12)
    t_pp = time.perf_counter() - t0
    print("[setup] Eq (12) weights for item %d               %s" % (item, fmt(t_pp)))

    # -- optional: is the model actually topic-aware? -----------------------
    if args.check_topic and len(test_items) > 1:
        t0 = time.perf_counter()
        pp2 = ew.for_item(test_items[1])
        keys = list(pp.keys())
        diffs = [abs(pp[k] - pp2.get(k, 0.0)) for k in keys]
        rels = [d / pp[k] for k, d in zip(keys, diffs) if pp[k] > 0]
        print("[setup] item %d vs item %d weight difference:     %s"
              % (item, test_items[1], fmt(time.perf_counter() - t0)))
        print("        max abs %.3e   mean abs %.3e   mean rel %.4f%%"
              % (max(diffs), sum(diffs) / len(diffs),
                 100.0 * sum(rels) / len(rels)))

    # -- the one shared evaluator ------------------------------------------
    t0 = time.perf_counter()
    evaluator = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)
    print("[setup] shared Eq (18) evaluator built            %s"
          % fmt(time.perf_counter() - t0))

    comm = detect_communities(model.pi)  # Eq (19), shared by variants 1-3
    results = []

    # ---------------------------------------------------------------- 1
    print("\n--- 1. CTIM (Algorithm 2 lines 32-45, as published) ---")
    t0 = time.perf_counter()
    seeds1 = ctim_select_seeds(model, ds, item, args.K, h=args.h, edge_weights=ew)
    t1 = time.perf_counter() - t0
    st = getattr(ctim_select_seeds, "last_stats", {})
    print("    select %s   (weights %.2fs  detect %.2fs  subgraphs %.2fs  "
          "dp %.3fs  select %.3fs)"
          % (fmt(t1), st.get("weights", 0), st.get("detect", 0),
             st.get("subgraphs", 0), st.get("dp", 0), st.get("select", 0)))
    results.append(("CTIM", seeds1, t1, st.get("dp", 0.0)))

    # ---------------------------------------------------------------- 2
    print("\n--- 2. CTIM + exact O(C K^2) allocation DP (no EA) ---")
    t0 = time.perf_counter()
    seeds2, st2 = select_seeds_ea_dp(model, ds, pp, args.K, random.Random(args.seed),
                                     h=args.h, use_ea=False, comm=comm, verbose=True)
    t2 = time.perf_counter() - t0
    print("    select %s   (curves: MIA %.2fs + greedy %.2fs, dp %.4fs)"
          % (fmt(t2), st2["mia_seconds"], st2["greedy_seconds"], st2["dp_seconds"]))
    print("    communities used: %d   allocation: %s"
          % (st2["n_communities_used"],
             dict(list(st2["allocation"].items())[:12])))
    results.append(("CTIM+fullDP", seeds2, t2, st2["dp_seconds"]))

    # ---------------------------------------------------------------- 3
    print("\n--- 3. CTIM + EA-refined curves + exact DP ---")
    t0 = time.perf_counter()
    seeds3, st3 = select_seeds_ea_dp(model, ds, pp, args.K, random.Random(args.seed),
                                     h=args.h, use_ea=True, top_t=args.top_t,
                                     pop_size=args.pop, generations=args.gens,
                                     comm=comm, verbose=True)
    t3 = time.perf_counter() - t0
    print("    select %s   (curves %.2fs, EA %.2fs, dp %.4fs)"
          % (fmt(t3), st3["mia_seconds"] + st3["greedy_seconds"],
             st3["ea_seconds"], st3["dp_seconds"]))
    print("    EA: %d communities refined, %d fitness evals, %d curve points "
          "improved, total curve gain +%.4f"
          % (st3["n_communities_refined"], st3["n_fitness_evals"],
             st3["n_points_improved"], st3["ea_curve_gain"]))
    results.append(("CTIM+EA+fullDP", seeds3, t3, st3["dp_seconds"]))

    # ---------------------------------------------------------------- 4
    print("\n--- 4. GlobalGreedy (MIA greedy on the full graph, no communities) ---")
    t0 = time.perf_counter()
    seeds4 = evaluator.greedy_incremental(args.K)
    t4 = time.perf_counter() - t0
    print("    select %s" % fmt(t4))
    results.append(("GlobalGreedy", seeds4, t4, 0.0))

    # ---------------------------------------------------------------- score
    print("\n" + "=" * 78)
    print("RESULTS -- all scored with the SAME exact Eq (18) evaluator (full graph)")
    print("=" * 78)
    print("%-18s %10s %12s %12s %10s %s"
          % ("variant", "I(S)", "vs CTIM", "select (s)", "DP (s)", "|S|"))
    base = None
    scored = []
    for (name, seeds, secs, dp_s) in results:
        t0 = time.perf_counter()
        val = evaluator.influence(list(seeds))  # Eq (18)
        t_eval = time.perf_counter() - t0
        if base is None:
            base = val
        scored.append((name, val, secs, dp_s, len(seeds), t_eval))
        print("%-18s %10.3f %11.2f%% %12.3f %10.4f %d"
              % (name, val, 100.0 * (val / base - 1.0) if base else 0.0,
                 secs, dp_s, len(seeds)))

    print("")
    for (name, _v, _s, _d, _n, t_eval) in scored:
        print("    evaluation cost for %-18s %s" % (name, fmt(t_eval)))

    print("\n    seed overlap with CTIM:")
    s1 = set(results[0][1])
    for (name, seeds, _s, _d) in results:
        ov = len(s1 & set(seeds))
        print("      %-18s %2d / %d shared" % (name, ov, len(s1)))

    print("\n" + "=" * 78)
    print("TOTAL %s" % fmt(time.perf_counter() - t_all))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
