#!/usr/bin/env python3
"""Compare CTIM's Algorithm 2 selection against IMM (RIS), under TWO evaluators.

Why two evaluators
------------------
The repo's shared evaluator is MIA, Eq (18), at ``h = 0.1``.  On Digg that
threshold admits only single-hop paths, so MIA is not measuring a cascade -- and
it is *the objective CTIM optimises*.  Scoring IMM with it alone would grade a
cascade-optimising method on a no-cascade metric.

So every seed set here is scored twice:

  1. **MIA Eq (18)** -- the repo's shared evaluator (API.md).  Comparable with
     every number in ``results/``.
  2. **Monte-Carlo Independent Cascade** -- a neutral referee that shares no
     machinery with MIA or with the RR sampler, and applies no threshold.

If the two disagree on the ordering, that disagreement is the finding: it means
``h = 0.1`` is deciding the comparison rather than the algorithms are.

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

from ctim.ctim import ctim_select_seeds              # noqa: E402
from ctim.dataset import load_dataset                # noqa: E402
from ctim.influence import MIA, EdgeWeights          # noqa: E402
from ctim.ris_imm import ic_simulate, imm_select_seeds  # noqa: E402


def fmt(s):
    return "%.2fs" % s


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=0.5,
                   help="IMM approximation slack; larger = fewer RR sets = faster")
    p.add_argument("--ell", type=float, default=1.0,
                   help="IMM succeeds with probability 1 - n^-ell")
    p.add_argument("--max-rr", type=int, default=400000,
                   help="hard ceiling on RR sets (guarantee voided if it binds)")
    p.add_argument("--mc", type=int, default=2000,
                   help="Monte-Carlo IC simulations for the neutral referee")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("=" * 76)
    print("IMM (RIS) vs CTIM Algorithm 2   K=%d  h=%g  eps=%g  ell=%g  seed=%d"
          % (args.K, args.h, args.eps, args.ell, args.seed))
    print("=" * 76)

    if not os.path.exists(args.cache):
        print("ERROR: no cached model at %s -- run scripts/calibrate_h.py first."
              % args.cache)
        return 1

    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model, item = blob["model"], blob["test_items"][0]
    ew = EdgeWeights(model, ds)
    print("[setup] dataset + cached model                 %s" % fmt(time.perf_counter() - t0))

    t0 = time.perf_counter()
    pp = ew.for_item(item)                    # Eq (12)
    print("[setup] Eq (12) weights for item %-6d        %s" % (item, fmt(time.perf_counter() - t0)))

    t0 = time.perf_counter()
    evaluator = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)
    print("[setup] shared Eq (18) evaluator               %s" % fmt(time.perf_counter() - t0))

    results = []

    # ------------------------------------------------------------------ CTIM
    print("\n--- CTIM (Algorithm 2, dp_tiebreak=paper-true) ---")
    t0 = time.perf_counter()
    seeds_ctim = ctim_select_seeds(model, ds, item, args.K, h=args.h, edge_weights=ew)
    t_ctim = time.perf_counter() - t0
    print("    select %s" % fmt(t_ctim))
    results.append(("CTIM", seeds_ctim, t_ctim, {}))

    # ------------------------------------------------------------------- IMM
    print("\n--- IMM (Tang et al. 2015) ---")
    t0 = time.perf_counter()
    seeds_imm, st = imm_select_seeds(ds.n_users, ds.in_adj, pp, args.K,
                                     random.Random(args.seed), eps=args.eps,
                                     ell=args.ell, max_rr=args.max_rr, verbose=True)
    t_imm = time.perf_counter() - t0
    print("    select %s   (sampling %s, grow %s, greedy %s)"
          % (fmt(t_imm), fmt(st["sampling_seconds"]), fmt(st["grow_seconds"]),
             fmt(st["select_seconds"])))
    print("    theta=%d  n_rr=%d  LB=%.2f  mean|RR|=%.2f  IMM's own estimate=%.2f"
          % (st["theta"], st["n_rr"], st["LB"], st["mean_rr_size"], st["rr_estimate"]))
    if st["theta_capped"]:
        print("    !! theta hit --max-rr=%d: the (1-1/e-eps) guarantee is VOID for"
              " this run" % args.max_rr)
    results.append(("IMM", seeds_imm, t_imm, st))

    # ---------------------------------------------------------- GlobalGreedy
    print("\n--- GlobalGreedy (MIA greedy, full graph) ---")
    t0 = time.perf_counter()
    seeds_gg = evaluator.greedy_incremental(args.K)
    t_gg = time.perf_counter() - t0
    print("    select %s" % fmt(t_gg))
    results.append(("GlobalGreedy", seeds_gg, t_gg, {}))

    # ------------------------------------------------------------- scoring
    print("\n" + "=" * 76)
    print("RESULTS -- two independent evaluators")
    print("=" * 76)
    print("%-14s %12s %14s %12s %s"
          % ("method", "MIA Eq(18)", "MC-IC (n=%d)" % args.mc, "select (s)", "|S|"))
    rows = []
    for (name, seeds, secs, _st) in results:
        mia_val = evaluator.influence(list(seeds))                 # Eq (18)
        mc_val = ic_simulate(ds.out_adj, pp, seeds, args.mc,
                             random.Random(args.seed + 1))         # neutral referee
        rows.append((name, mia_val, mc_val, secs, len(seeds)))
        print("%-14s %12.3f %14.3f %12.2f %d"
              % (name, mia_val, mc_val, secs, len(seeds)))

    base = rows[0]
    print("\nrelative to CTIM:")
    for (name, mia_val, mc_val, _s, _n) in rows:
        print("  %-14s MIA %+7.2f%%   MC-IC %+7.2f%%"
              % (name,
                 100.0 * (mia_val / base[1] - 1.0) if base[1] else 0.0,
                 100.0 * (mc_val / base[2] - 1.0) if base[2] else 0.0))

    order_mia = [r[0] for r in sorted(rows, key=lambda r: -r[1])]
    order_mc = [r[0] for r in sorted(rows, key=lambda r: -r[2])]
    print("\n  ranking by MIA Eq(18) : %s" % " > ".join(order_mia))
    print("  ranking by MC-IC      : %s" % " > ".join(order_mc))
    if order_mia != order_mc:
        print("  ** THE TWO EVALUATORS DISAGREE -- h=%g is deciding the comparison,"
              " not the algorithms **" % args.h)
    else:
        print("  the two evaluators agree on the ordering")

    s_ctim = set(seeds_ctim)
    print("\n  seed overlap with CTIM:")
    for (name, seeds, _s, _st) in results:
        print("    %-14s %2d / %d shared" % (name, len(s_ctim & set(seeds)), len(s_ctim)))

    print("\n" + "=" * 76)
    print("TOTAL %s" % fmt(time.perf_counter() - t_all))
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
