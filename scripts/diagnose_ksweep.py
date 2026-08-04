#!/usr/bin/env python3
"""Does the "community decomposition costs nothing" result survive K and items?

Why this script exists
----------------------
Every number in this repo so far was measured at a single operating point:
``K = 20``, ``test_items[0]``, ``h = 0.1``.  Two of the conclusions drawn there
are budget-sensitive in principle:

  1. "GlobalGreedy beats CTIM by only +0.33%" -- a gap measured at one budget
     says nothing about whether the gap opens up as K grows.
  2. "the EA inside communities gains 0.00%" -- the standing suspicion is that
     this is an *artifact of the budget being thin*: K=20 spread over C=100
     communities could leave every community with 0 or 1 seed, and "choose the
     best j-subset of a community" is vacuous for j <= 1.  A within-community
     subset optimiser can only matter once some community is handed j >= 3
     seeds (j=2 is already searched exhaustively by two greedy steps in most
     practical senses; j >= 3 is where greedy can genuinely be beaten).

So this script sweeps K and reports, per cell, the per-community *allocation* --
not just the spread.  It answers "at what K does any community receive >= 3
seeds?" exactly, by running the allocation DP at every K in 1..K_max.

METRIC VALIDITY
---------------
Nothing here retrains the model.  The cached model, the item, ``pp`` (Eq (12))
and ``h`` are held fixed across the whole K sweep, so every MIA Eq (18) number
in the K table is **same-pp, same-h** and the spreads are directly comparable
to each other and to every other MIA number in the repo.

The per-item table at the end is different and is flagged as such: a different
item means a different ``pp``, hence a different weighted graph.  Spreads
*across* items are NOT comparable; only the within-item CTIM-vs-GlobalGreedy
percentage gap is.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random``.
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

from ctim.ctim import ctim_select_seeds, detect_communities   # noqa: E402
from ctim.dataset import load_dataset                         # noqa: E402
from ctim.ea_dp import allocate_exact, build_curves, select_seeds_ea_dp  # noqa: E402
from ctim.influence import MIA, EdgeWeights                   # noqa: E402


def fmt(s):
    return "%.2fs" % s


def alloc_of(seeds, comm):
    """Histogram of seeds per community, using Eq (19)'s assignment comm[v]."""
    hist = {}
    for v in seeds:
        if v < len(comm):
            c = comm[v]
            hist[c] = hist.get(c, 0) + 1
    return dict(sorted(hist.items()))


def alloc_summary(hist):
    """(#communities used, max seeds in any one community)."""
    if not hist:
        return 0, 0
    return len(hist), max(hist.values())


def pct_gap(a, b):
    """(a - b) / b as a percentage; b is the reference (CTIM)."""
    if b <= 0.0:
        return float("nan")
    return 100.0 * (a - b) / b


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--Ks", default="1,5,10,20,30,50,75,100",
                   help="comma-separated budgets for the K sweep")
    p.add_argument("--item-K", type=int, default=20,
                   help="budget at which the per-item check is repeated")
    p.add_argument("--n-items", type=int, default=3,
                   help="how many of test_items[0..] to cross-check")
    args = p.parse_args(argv)

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    K_max = max(Ks)
    rng = random.Random(args.seed)

    t_all = time.perf_counter()
    print("=" * 100)
    print("K SWEEP + ITEM SWEEP   h=%g  Ks=%s  seed=%d" % (args.h, Ks, args.seed))
    print("=" * 100)

    if not os.path.exists(args.cache):
        print("ERROR: no cached model at %s -- run scripts/calibrate_h.py first." % args.cache)
        return 1

    # ------------------------------------------------------------------ setup
    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model = blob["model"]
    test_items = blob["test_items"]
    item = test_items[0]
    ew = EdgeWeights(model, ds)
    print("[setup] dataset + cached model                 %s   n_users=%d  test_items=%d"
          % (fmt(time.perf_counter() - t0), ds.n_users, len(test_items)))

    t0 = time.perf_counter()
    pp = ew.for_item(item)                                    # Eq (12)
    t_pp = time.perf_counter() - t0
    print("[setup] Eq (12) weights for item %-8d      %s   |live edges|=%d"
          % (item, fmt(t_pp), len(pp)))

    t0 = time.perf_counter()
    evaluator = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)   # Eq (13)-(18)
    print("[setup] shared Eq (18) evaluator               %s" % fmt(time.perf_counter() - t0))

    t0 = time.perf_counter()
    comm = detect_communities(model.pi)                       # Eq (19)
    n_comm = max(comm) + 1
    print("[setup] Eq (19) communities                    %s   C=%d"
          % (fmt(time.perf_counter() - t0), n_comm))

    # Greedy community curves I_m(j), j = 0..min(K_max,|c_m|), built ONCE.
    # The greedy curve is a nested sequence, so the prefix of the K_max curve IS
    # the curve build_curves(..., K) would have produced for any K <= K_max.
    # allocate_exact(curves, K) therefore reproduces select_seeds_ea_dp(..., K,
    # use_ea=False) exactly -- asserted below at K = item-K.
    t0 = time.perf_counter()
    curves, curve_stats = build_curves(comm, ds, pp, K_max, args.h, verbose=True)
    t_curves = time.perf_counter() - t0
    print("[setup] curves I_m(j) up to j=%-4d             %s" % (K_max, fmt(t_curves)))

    t0 = time.perf_counter()
    _sd, _st = select_seeds_ea_dp(model, ds, pp, args.item_K, random.Random(args.seed),
                                  h=args.h, use_ea=False, comm=comm)
    t_verify = time.perf_counter() - t0
    _sd_fast, _dpv, _al_fast = allocate_exact(curves, args.item_K)
    same = (sorted(_sd) == sorted(_sd_fast)) and (_st["allocation"] == dict(sorted(_al_fast.items())))
    print("[check] shared-curves DP == select_seeds_ea_dp(K=%d, use_ea=False): %s   (%s)"
          % (args.item_K, "IDENTICAL" if same else "*** DIFFERS ***", fmt(t_verify)))

    print("\nSetup total %s\n" % fmt(time.perf_counter() - t_all))

    # ------------------------------------------------------------ the K sweep
    print("=" * 100)
    print("K SWEEP -- item %d, h=%g, SAME pp / SAME h throughout: all I_h(S) directly comparable"
          % (item, args.h))
    print("=" * 100)
    hdr = ("%4s | %10s %8s | %10s %8s | %10s %8s | %7s | %s"
           % ("K", "CTIM", "t", "ExactDP", "t", "GlobGrdy", "t", "gap%", "allocation (#comms used, max/comm)"))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for K in Ks:
        # -- CTIM Algorithm 2 (lines 1-45), paper-true DP tie-break
        t0 = time.perf_counter()
        s_ctim = ctim_select_seeds(model, ds, item, K, h=args.h,
                                   dp_tiebreak="paper-true", edge_weights=ew)
        t_ctim = time.perf_counter() - t0
        I_ctim = evaluator.influence(s_ctim)              # Eq (18)

        # -- exact O(C K^2) allocation DP over the same greedy curves
        t0 = time.perf_counter()
        s_dp, _dpv, al_dp = allocate_exact(curves, K)
        t_dp = time.perf_counter() - t0
        I_dp = evaluator.influence(s_dp)                  # Eq (18)

        # -- GlobalGreedy: no community decomposition at all
        t0 = time.perf_counter()
        s_gg = evaluator.greedy_incremental(K)
        t_gg = time.perf_counter() - t0
        I_gg = evaluator.influence(s_gg)                  # Eq (18)

        h_ctim = alloc_of(s_ctim, comm)
        h_dp = dict(sorted(al_dp.items()))
        h_gg = alloc_of(s_gg, comm)
        nc_ctim, mx_ctim = alloc_summary(h_ctim)
        nc_dp, mx_dp = alloc_summary(h_dp)
        nc_gg, mx_gg = alloc_summary(h_gg)

        print("%4d | %10.4f %8s | %10.4f %8s | %10.4f %8s | %+6.2f%% | CTIM(%d,%d) DP(%d,%d) GG(%d,%d)"
              % (K, I_ctim, fmt(t_ctim), I_dp, fmt(t_dp), I_gg, fmt(t_gg),
                 pct_gap(I_gg, I_ctim), nc_ctim, mx_ctim, nc_dp, mx_dp, nc_gg, mx_gg))
        rows.append((K, I_ctim, t_ctim, I_dp, t_dp, I_gg, t_gg, h_ctim, h_dp, h_gg))

    print("\nper-community allocations in full (community_id: n_seeds)")
    for (K, _a, _b, _c, _d, _e, _f, h_ctim, h_dp, h_gg) in rows:
        print("  K=%-4d CTIM %s" % (K, h_ctim))
        print("        DP   %s" % (h_dp,))
        print("        GG   %s" % (h_gg,))

    # ------------------------------ at which K does a community get >= 3 seeds?
    # The DP is cheap (0.005s), so scan EVERY K rather than only the grid.
    print("\n" + "=" * 100)
    print("WHEN CAN A WITHIN-COMMUNITY SUBSET OPTIMISER MATTER?")
    print("=" * 100)
    t0 = time.perf_counter()
    first = {2: None, 3: None, 4: None, 5: None}
    scan = []
    for K in range(1, K_max + 1):
        _s, _v, al = allocate_exact(curves, K)
        nc, mx = alloc_summary(dict(al))
        scan.append((K, nc, mx))
        for j in first:
            if first[j] is None and mx >= j:
                first[j] = K
    print("scanned K=1..%d with the exact DP in %s" % (K_max, fmt(time.perf_counter() - t0)))
    for j in sorted(first):
        print("  first K at which some community receives >= %d seeds : %s"
              % (j, first[j] if first[j] is not None else "never (K<=%d)" % K_max))
    print("  max seeds in any one community, by K:")
    line = []
    for (K, nc, mx) in scan:
        if K in (1, 2, 3, 4, 5) or K % 10 == 0:
            line.append("K=%d:%d(in %d comms)" % (K, mx, nc))
    print("    " + "  ".join(line))

    # how many communities can carry ANY propagation at all
    flat = [c for c in curves if all(abs(c.I[j] - j) < 1e-9 for j in range(len(c.I)))]
    print("  communities whose curve is I_m(j) = j exactly (no live intra-community edge): %d / %d"
          % (len(flat), len(curves)))
    live = sorted([c for c in curves if c not in flat], key=lambda c: -c.I[c.cap])
    for c in live:
        print("    community %-4d |c|=%-6d cap=%-4d I_m(cap)=%.4f  (slope over j: %.4f)"
              % (c.m, len(c.members), c.cap, c.I[c.cap],
                 c.I[c.cap] / c.cap if c.cap else 0.0))

    # -------------------------------------------------------- the item sweep
    print("\n" + "=" * 100)
    print("ITEM SWEEP at K=%d -- NOT same-pp: each item has its own Eq (12) weights," % args.item_K)
    print("so I_h(S) is NOT comparable ACROSS rows; only the within-row gap%% is.")
    print("=" * 100)
    ihdr = ("%8s | %8s | %10s %8s | %10s %8s | %10s | %7s | %s"
            % ("item", "|pp|", "CTIM", "t", "GlobGrdy", "t", "ExactDP", "gap%", "(#comms,max/comm) CTIM/DP/GG"))
    print(ihdr)
    print("-" * len(ihdr))

    for idx in range(min(args.n_items, len(test_items))):
        it = test_items[idx]
        if idx == 0:
            pp_i, ev_i = pp, evaluator
            curves_i = curves
        else:
            pp_i = ew.for_item(it)                                     # Eq (12)
            ev_i = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp_i, h=args.h)
            curves_i, _ = build_curves(comm, ds, pp_i, args.item_K, args.h)

        t0 = time.perf_counter()
        s_ctim = ctim_select_seeds(model, ds, it, args.item_K, h=args.h,
                                   dp_tiebreak="paper-true", edge_weights=ew)
        t_ctim = time.perf_counter() - t0
        I_ctim = ev_i.influence(s_ctim)                                # Eq (18)

        t0 = time.perf_counter()
        s_gg = ev_i.greedy_incremental(args.item_K)
        t_gg = time.perf_counter() - t0
        I_gg = ev_i.influence(s_gg)                                    # Eq (18)

        s_dp, _v, al_dp = allocate_exact(curves_i, args.item_K)
        I_dp = ev_i.influence(s_dp)                                    # Eq (18)

        nc_c, mx_c = alloc_summary(alloc_of(s_ctim, comm))
        nc_d, mx_d = alloc_summary(dict(al_dp))
        nc_g, mx_g = alloc_summary(alloc_of(s_gg, comm))
        print("%8d | %8d | %10.4f %8s | %10.4f %8s | %10.4f | %+6.2f%% | (%d,%d)/(%d,%d)/(%d,%d)"
              % (it, len(pp_i), I_ctim, fmt(t_ctim), I_gg, fmt(t_gg), I_dp,
                 pct_gap(I_gg, I_ctim), nc_c, mx_c, nc_d, mx_d, nc_g, mx_g))

    print("\nWALL-CLOCK RUNTIME (whole script): %s" % fmt(time.perf_counter() - t_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
