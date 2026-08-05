#!/usr/bin/env python3
"""Run CTIM-G (globally-scored community selection) and compare it, honestly.

CTIM-G is a *proposed variant*, not part of the paper -- ``ctim/ctim.py`` stays
the faithful transcription of Algorithm 2.  The one thing it changes is the
graph the candidate gains are measured on:

    Algorithm 2, line 34   IncInf_m(u) on the NODE-INDUCED subgraph of c_m
    CTIM-G                 IncInf(u)   on the FULL graph, candidates still
                           drawn per community, budget still split by the DP

Measured motivation (scripts/diagnose_blindness.py): the induced subgraph shows
only ~23% of the true influence of the seeds CTIM itself picks, and ~77% of a
seed's ap() mass lands outside its own community.

Grading -- the part this project got wrong twice
------------------------------------------------
Scoring a seed set with the same truncated MIA used to *select* it is circular,
and on this data it inverts orderings.  So every set here is scored with ONE
shared ``MIA(h_eval)`` object, plus a threshold-free Monte-Carlo IC referee.
The historical ``h_sel`` number is printed too, in its own column, clearly
labelled -- it is what earlier results/ tables reported, and it is not a valid
cross-method comparison.

``h_sel`` is a free parameter of every selector here, not a constant: measured,
GlobalGreedy gains ~2.4% by moving from 0.1 to 0.05.  Compare methods only at
the SAME ``--h``.

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

from ctim.ctim import ctim_select_seeds                # noqa: E402
from ctim.ctim_global import select_seeds_global       # noqa: E402
from ctim.dataset import load_dataset                  # noqa: E402
from ctim.influence import MIA, EdgeWeights            # noqa: E402
from ctim.ris_imm import ic_simulate                   # noqa: E402

ALL_METHODS = ("CTIM", "CTIM-G", "CTIM-G+repair", "GlobalGreedy")


def fmt(s):
    return "%.2fs" % s


def mc_batches(out_adj, pp, seeds, total, n_batches, seed):
    """MC-IC mean and standard error over ``n_batches`` paired batches.

    Every seed set is given the SAME (seed, batch) streams, so the batch means
    are paired across methods and the SE below is of the batch mean, not of one
    simulation.  With few batches the degrees of freedom are small -- a gap
    inside 2 SE is not a gap.
    """
    per = max(1, total // max(1, n_batches))
    means = []
    for b in range(n_batches):
        means.append(ic_simulate(out_adj, pp, seeds, per, random.Random(seed + 1000 * b)))
    m = sum(means) / len(means)
    if len(means) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in means) / (len(means) - 1)
    return m, (var / len(means)) ** 0.5


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--h", type=float, default=0.05,
                   help="h_sel -- the SELECTION threshold, shared by every method")
    p.add_argument("--h-eval", dest="h_eval", type=float, default=0.001,
                   help="h_eval -- the fixed grading threshold; must differ from --h")
    p.add_argument("--mc", type=int, default=1000, help="MC-IC simulations (0 = skip)")
    p.add_argument("--mc-batches", dest="mc_batches", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--item-index", dest="item_index", type=int, default=0)
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    args = p.parse_args(argv)

    if abs(args.h - args.h_eval) < 1e-12:
        print("ERROR: --h-eval equals --h; that is the circular comparison this "
              "script exists to avoid.  Pick a finer --h-eval.")
        return 2
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    t_all = time.perf_counter()
    print("=" * 84)
    print("CTIM-G   K=%d  h_sel=%g  h_eval=%g  seed=%d" % (args.K, args.h, args.h_eval, args.seed))
    print("=" * 84)

    if not os.path.exists(args.cache):
        print("ERROR: no model cache at %s\n"
              "Build one with:\n"
              "  python -u scripts/calibrate_h.py --dataset %s --cache %s"
              % (args.cache, args.dataset, args.cache))
        return 1

    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model = blob["model"]
    item = blob["test_items"][args.item_index]
    ew = EdgeWeights(model, ds)
    print("[setup] dataset + model + EdgeWeights            %s" % fmt(time.perf_counter() - t0))

    t0 = time.perf_counter()
    pp = ew.for_item(item)                                    # Eq (12)
    print("[setup] Eq (12) weights for item %-8d        %s  (|pp|=%d)"
          % (item, fmt(time.perf_counter() - t0), len(pp)))

    # ONE referee for every seed set, and one coarse evaluator for the historical column.
    referee = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h_eval)   # Eq (18)
    coarse = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)         # Eq (18), circular
    sel_mia = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)

    rows = []
    for name in methods:
        print("\n--- %s ---" % name)
        rng = random.Random(args.seed)
        t0 = time.perf_counter()
        st = {}
        if name == "CTIM":
            seeds = ctim_select_seeds(model, ds, item, args.K, h=args.h,
                                      dp_tiebreak="paper-true", edge_weights=ew)
        elif name == "CTIM-G":
            seeds, st = select_seeds_global(model, ds, pp, args.K, rng, h=args.h)
        elif name == "CTIM-G+repair":
            seeds, st = select_seeds_global(model, ds, pp, args.K, rng, h=args.h, repair=True)
        elif name == "GlobalGreedy":
            seeds = sel_mia.greedy_incremental(args.K)
        else:
            print("    unknown method %r -- skipped" % name)
            continue
        secs = time.perf_counter() - t0
        print("    select %s   |S|=%d" % (fmt(secs), len(seeds)))
        if st:
            print("    phases: detect %s  curves %s  dp %s  realise %s  repair %s"
                  % (fmt(st.get("detect_seconds", 0.0)), fmt(st.get("curve_seconds", 0.0)),
                     fmt(st.get("dp_seconds", 0.0)), fmt(st.get("realise_seconds", 0.0)),
                     fmt(st.get("repair_seconds", 0.0))))
            print("    communities used: %d / %d   allocation: %s"
                  % (st.get("n_communities_used", 0), st.get("n_communities", 0),
                     st.get("allocation", {})))
            dpv, val = st.get("dp_value", 0.0), st.get("value", 0.0)
            if dpv and val:
                print("    dp_value %.4f is an UPPER BOUND (sum_m I(S_m) >= I(union)); "
                      "realised %.4f, gap %.2f%%" % (dpv, val, 100.0 * (dpv / val - 1.0)))
        rows.append((name, list(seeds), secs))

    # ---- scoring: dedupe identical seed sets, h_eval is expensive -----------
    print("\n[scoring] one shared MIA(h_eval=%g) for every set ..." % args.h_eval)
    cache_fine, cache_coarse, cache_mc = {}, {}, {}
    scored = []
    t0 = time.perf_counter()
    for (name, seeds, secs) in rows:
        key = tuple(sorted(seeds))
        if key not in cache_fine:
            cache_fine[key] = referee.influence(list(seeds))     # Eq (18) at h_eval
            cache_coarse[key] = coarse.influence(list(seeds))    # Eq (18) at h_sel
            if args.mc > 0:
                cache_mc[key] = mc_batches(ds.out_adj, pp, seeds, args.mc,
                                           args.mc_batches, args.seed + 1)
            else:
                cache_mc[key] = (float("nan"), 0.0)
        m, se = cache_mc[key]
        scored.append((name, seeds, secs, cache_fine[key], cache_coarse[key], m, se))
    print("[scoring] done in %s  (%d distinct seed sets)" % (fmt(time.perf_counter() - t0), len(cache_fine)))

    base = scored[0]
    print("\n" + "=" * 84)
    print("RESULTS -- one pp, one item (%d), one h_eval.  Comparable across rows." % item)
    print("=" * 84)
    print("%-16s %14s %9s %16s %11s  %s"
          % ("method", "I_h_eval", "vs base", "MC-IC +- SE", "select(s)", "|S|"))
    for (name, seeds, secs, fine, _c, m, se) in scored:
        rel = 100.0 * (fine / base[3] - 1.0) if base[3] else 0.0
        mc = "%9.3f +-%5.2f" % (m, se) if m == m else "        --      "
        print("%-16s %14.4f %8.2f%% %16s %11.2f  %d" % (name, fine, rel, mc, secs, len(seeds)))

    print("\n  historical column -- I at h_sel=%g, i.e. graded with the SAME" % args.h)
    print("  threshold used to select.  This is the circular number earlier")
    print("  results/ tables reported; it is NOT a valid cross-method comparison.")
    for (name, _s, _t, _f, c, _m, _se) in scored:
        print("      %-16s %12.4f" % (name, c))

    s0 = set(base[1])
    print("\n  seed overlap with %s:" % base[0])
    for (name, seeds, _t, _f, _c, _m, _se) in scored:
        shared = len(s0 & set(seeds))
        tag = "  <-- IDENTICAL SET" if shared == len(s0) == len(seeds) and name != base[0] else ""
        print("      %-16s %2d / %d%s" % (name, shared, len(s0), tag))

    print("\n" + "=" * 84)
    print("TOTAL %s" % fmt(time.perf_counter() - t_all))
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
