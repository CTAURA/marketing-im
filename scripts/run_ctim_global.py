#!/usr/bin/env python3
"""Run CTIM / CTIM-G / CTIM-G+repair / GlobalGreedy and report result + complexity.

CTIM-G is a *proposed variant*, not part of the paper -- ``ctim/ctim.py`` stays
the faithful transcription of Algorithm 2.  The one thing it changes is the
graph the candidate gains are measured on:

    Algorithm 2, line 34   IncInf_m(u) on the NODE-INDUCED subgraph of c_m
    CTIM-G                 IncInf(u)   on the FULL graph, candidates still
                           drawn per community, budget still split by the DP

Measured motivation (scripts/diagnose_blindness.py, Yelp): the induced subgraph
shows only 1.68% of the true influence of the seeds CTIM itself picks, and the
attenuation factor varies 4.9x BETWEEN communities (0.74%..3.61%) -- so the DP
of lines 32-41 compares ``I_m`` values that live on different scales, and
under-funds whole communities.  On Yelp 4 of the 5 seeds CTIM missed relative to
GlobalGreedy were their community's ``I_m``-rank-1 node in a community that got
zero budget.

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

Complexity
----------
Section [C] reports, per method, the textbook cost AND the measured primitive
that dominates it: the number of MIIA/MIOA arborescences built and their mean
size.  Each arborescence is a Dijkstra on ``-ln pp`` costing ``O(t log t)`` for
``t = |MIIA(v,h)|``, so ``(count, mean size)`` is the honest empirical
complexity, and wall-clock alone is not.

That pair is what explains the whole speed story: CTIM and GlobalGreedy build
roughly the SAME number of arborescences; CTIM's are drastically smaller,
because it builds them on induced subgraphs.  On Yelp that is the difference
between 47s and 3962s.

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

from ctim.ctim import ctim_select_seeds, detect_communities   # noqa: E402
from ctim.ctim_global import select_seeds_global              # noqa: E402
from ctim.dataset import load_dataset                         # noqa: E402
from ctim.influence import MIA, EdgeWeights                   # noqa: E402
from ctim.ris_imm import ic_simulate                          # noqa: E402

ALL_METHODS = ("CTIM", "CTIM-G", "CTIM-G+repair", "GlobalGreedy")

# Textbook cost of each selector.  n = |V|, m = |E|, K = budget, C = communities,
# t = mean |MIIA(v,h)| on the graph the selector actually scores on, n_c = |c|.
COMPLEXITY = {
    "CTIM": (
        "O(nC)  Eq (19)  +  sum_c O(n_c * t_c log t_c)  init  +  O(CK) DP cells",
        "n arborescences, each on an INDUCED subgraph -> t_c is small"),
    "CTIM-G": (
        "O(n * t log t)  global init  +  sum_c O(n_c * t log t)  curves  +  "
        "O(C K^2) DP  +  O(K * t^2) realisation",
        "n arborescences, each on the FULL graph -> t is the global mean"),
    "CTIM-G+repair": (
        "CTIM-G  +  O(rounds * K * |pool| ) swap evaluations, pruned by bound (*)",
        "adds one 1-swap local-search pass over the global MIA"),
    "CTIM-G-lazy": (
        "O(n * t log t)  global init  +  O(K log C)  community heap  +  "
        "curve points built only on demand  +  O(K * t^2) realisation",
        "same objective and split as CTIM-G; builds ~K curve points, not ~C*K"),
    "GlobalGreedy": (
        "O(n * t log t)  eager IncInf init  +  O(K * t^2)  updates",
        "n arborescences on the FULL graph; the init dominates and does not "
        "shrink with K"),
}


def fmt(s):
    return "%.2fs" % s


def parse_seed_list(spec):
    """``--gg-seeds`` is either a comma-separated list or a path to one-per-line."""
    if not spec:
        return None
    if os.path.exists(spec):
        out = []
        with open(spec, "r") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append(int(line.split(",")[0]))
        return out
    return [int(x) for x in spec.replace(" ", "").split(",") if x]


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


def arborescence_stats(mia):
    """(count, mean size) of the MIIA/MIOA trees ``mia`` actually built.

    Read off the memo dicts rather than by instrumenting ``ctim/influence.py``:
    a cache entry exists iff that arborescence was computed, so the size of the
    cache IS the number of Dijkstra runs paid for.
    """
    n = 0
    tot = 0
    for cache in (getattr(mia, "_miia_cache", {}), getattr(mia, "_mioa_cache", {})):
        for got in cache.values():
            n += 1
            try:
                tot += len(got[0])
            except (TypeError, IndexError):
                pass
    return n, (tot / n if n else 0.0)


def probe_arborescence_sizes(ds, pp, comm, h, n_sample, rng):
    """Mean |MIIA(v,h)| on the FULL graph vs on v's own induced subgraph.

    This is the ratio that explains the CTIM/GlobalGreedy cost gap: both build
    ~n arborescences, but CTIM's live on induced subgraphs.  Sampled, because on
    Yelp there are 366k nodes.
    """
    members = {}
    for v in range(min(len(comm), ds.n_users)):
        members.setdefault(comm[v], []).append(v)

    full = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)
    pool = [v for v in range(ds.n_users)]
    sample = rng.sample(pool, min(n_sample, len(pool)))

    t_full = 0
    for v in sample:
        t_full += len(full.miia(v)[0])

    # group the sample by community so each induced MIA is built once
    by_c = {}
    for v in sample:
        by_c.setdefault(comm[v], []).append(v)
    t_ind = 0
    for c, vs in by_c.items():
        mia_c = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h, nodes=members[c])
        for v in vs:
            t_ind += len(mia_c.miia(v)[0])

    n = len(sample)
    return (t_full / n if n else 0.0), (t_ind / n if n else 0.0), n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--h", type=float, default=0.1,
                   help="h_sel -- the SELECTION threshold, shared by every method")
    p.add_argument("--h-eval", dest="h_eval", type=float, default=0.001,
                   help="h_eval -- the fixed grading threshold; must differ from --h")
    p.add_argument("--mc", type=int, default=1000, help="MC-IC simulations (0 = skip)")
    p.add_argument("--mc-batches", dest="mc_batches", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--item-index", dest="item_index", type=int, default=0,
                   help="single item (kept for compatibility); --items overrides")
    p.add_argument("--items", default="",
                   help="comma-separated indices into test_items, e.g. 0,1,2. "
                        "Each item needs its own Eq (12) pp (~257s on full Yelp) "
                        "and its own run of every method.  'all' uses them all.")
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--gg-seeds", dest="gg_seeds", default="",
                   help="precomputed GlobalGreedy seeds (comma list or file); "
                        "skips its selection, which costs ~66 min on full Yelp")
    p.add_argument("--probe-sample", dest="probe_sample", type=int, default=200,
                   help="nodes sampled for the arborescence-size probe (0 = skip)")
    p.add_argument("--paper-eval", dest="paper_eval", action="store_true",
                   help="grade at h_sel, i.e. reproduce the paper's own protocol "
                        "(one h for both selection and Eq (18)).  This is the "
                        "circular comparison; allowed only when asked for by name.")
    args = p.parse_args(argv)

    if args.paper_eval:
        args.h_eval = args.h
    if abs(args.h - args.h_eval) < 1e-12 and not args.paper_eval:
        print("ERROR: --h-eval equals --h; that is the circular comparison this "
              "script exists to avoid.  Pick a finer --h-eval, or pass "
              "--paper-eval to ask for the paper's protocol on purpose.")
        return 2
    if args.paper_eval:
        print("!" * 88)
        print("! --paper-eval: grading at h_sel=%g, the SAME threshold used to select."
              % args.h)
        print("! This is the paper's protocol.  It is valid for the paper's own")
        print("! baselines, none of which maximise Eq (18) directly.  It is NOT a")
        print("! valid comparison for CTIM-G / GlobalGreedy, which do.")
        print("!" * 88)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    gg_seeds = parse_seed_list(args.gg_seeds)

    t_all = time.perf_counter()
    print("=" * 88)
    print("CTIM-G   K=%d  h_sel=%g  h_eval=%g  seed=%d" % (args.K, args.h, args.h_eval, args.seed))
    print("=" * 88)

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
    test_items = blob["test_items"]
    ew = EdgeWeights(model, ds)
    print("[setup] dataset + model + EdgeWeights            %s" % fmt(time.perf_counter() - t0))
    print("        U=%d  E=%d  C=%d  test_items=%s"
          % (ds.n_users, ds.n_links, len(model.pi[0]), test_items))

    if args.items.strip().lower() == "all":
        idxs = list(range(len(test_items)))
    elif args.items.strip():
        idxs = [int(x) for x in args.items.replace(" ", "").split(",") if x]
    else:
        idxs = [args.item_index]
    idxs = [i for i in idxs if 0 <= i < len(test_items)]
    if not idxs:
        print("ERROR: no valid item index in --items/--item-index")
        return 2

    per_item = []
    for run_i, idx in enumerate(idxs):
        item = test_items[idx]
        print("\n" + "#" * 88)
        print("# ITEM %d of %d -- test_items[%d] = %d" % (run_i + 1, len(idxs), idx, item))
        print("#" * 88)
        res = run_one_item(args, ds, model, ew, item,
                           probe=(run_i == 0 and args.probe_sample > 0))
        per_item.append((idx, item, res))

    if len(per_item) > 1:
        report_across_items(per_item)

    print("\n" + "=" * 88)
    print("TOTAL %s   for %d item(s)" % (fmt(time.perf_counter() - t_all), len(per_item)))
    print("=" * 88)
    return 0


def report_across_items(per_item):
    """Aggregate over items -- on RELATIVE gaps only.

    ``I_h_eval`` is NOT comparable across items: each item has its own Eq (12)
    ``pp``, hence its own diffusion graph.  Averaging raw spreads across items
    would be the same class of error as grading at ``h_sel``.  What does
    transfer is the WITHIN-item relative gap between methods, so that is what is
    averaged here, with its spread across items.
    """
    print("\n" + "=" * 88)
    print("[A] ACROSS ITEMS -- relative gaps only")
    print("=" * 88)
    print("  I_h_eval is not comparable across items (different pp = different")
    print("  graph).  Only the within-item gap vs the first method transfers.\n")

    names = [r[0] for r in per_item[0][2]]
    print("  %-16s %s" % ("method", "".join("%14s" % ("item %d" % it) for (_i, it, _r) in per_item)))
    for j, nm in enumerate(names):
        cells = []
        for (_idx, _item, rows) in per_item:
            cells.append("%13.4f " % rows[j][1])
        print("  %-16s %s" % (nm, "".join(cells)))

    print("\n  %-16s %10s %10s %10s %10s" % ("method", "mean gap%", "sd", "min", "max"))
    for j, nm in enumerate(names):
        gaps = []
        for (_idx, _item, rows) in per_item:
            base = rows[0][1]
            gaps.append(100.0 * (rows[j][1] / base - 1.0) if base else 0.0)
        m = sum(gaps) / len(gaps)
        sd = (sum((g - m) ** 2 for g in gaps) / (len(gaps) - 1)) ** 0.5 if len(gaps) > 1 else 0.0
        print("  %-16s %10.2f %10.2f %10.2f %10.2f" % (nm, m, sd, min(gaps), max(gaps)))
    print("\n  A gap whose sd across items is comparable to its mean is not")
    print("  established -- report the range, not just the mean.")


def run_one_item(args, ds, model, ew, item, probe=True):
    """One item end to end.  Returns [(method, I_h_eval, mc, se, secs, n_seeds), ...]."""
    t_item = time.perf_counter()
    t0 = time.perf_counter()
    pp = ew.for_item(item)                                    # Eq (12)
    t_pp = time.perf_counter() - t0
    print("[setup] Eq (12) weights                          %s  (|pp|=%d)" % (fmt(t_pp), len(pp)))
    gg_seeds = parse_seed_list(args.gg_seeds)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # ONE referee for every seed set, and one coarse evaluator for the historical column.
    referee = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h_eval)   # Eq (18)
    if abs(args.h - args.h_eval) < 1e-12:
        coarse = referee          # same threshold -> same trees; don't pay twice
    else:
        coarse = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)     # Eq (18), circular

    rows = []
    for name in methods:
        print("\n--- %s ---" % name)
        rng = random.Random(args.seed)
        st = {}
        counted = None
        t0 = time.perf_counter()
        eq12_inside = 0.0
        if name == "CTIM":
            seeds = ctim_select_seeds(model, ds, item, args.K, h=args.h,
                                      dp_tiebreak="paper-true", edge_weights=ew)
            # ctim_select_seeds has no `pp` parameter, so it rebuilds Eq (12)
            # internally from `edge_weights` -- EdgeWeights.for_item does NOT
            # cache (influence.py:254).  CTIM-G is handed the prebuilt `pp` and
            # never pays that cost, so leaving it in would inflate CTIM by
            # 40-50% and understate the CTIM-G / CTIM ratio by ~2x.  Subtract it
            # and compare selection against selection.
            eq12_inside = ctim_select_seeds.last_stats.get("weights", 0.0)
            print("    [timing] Eq (12) rebuilt inside CTIM: %s -- excluded, "
                  "CTIM-G is handed pp prebuilt" % fmt(eq12_inside))
        elif name in ("CTIM-G", "CTIM-G+repair", "CTIM-G-lazy"):
            # own the MIA object so its memo dicts can be read back as counters
            counted = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)
            seeds, st = select_seeds_global(model, ds, pp, args.K, rng, h=args.h,
                                            mia=counted,
                                            repair=(name == "CTIM-G+repair"),
                                            lazy=(name == "CTIM-G-lazy"))
        elif name == "GlobalGreedy":
            if gg_seeds is not None:
                seeds = gg_seeds[:args.K]
                print("    [injected] %d precomputed seeds -- selection NOT timed"
                      % len(seeds))
                t0 = None
            else:
                counted = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)
                seeds = counted.greedy_incremental(args.K)
        else:
            print("    unknown method %r -- skipped" % name)
            continue
        secs = ((time.perf_counter() - t0 - eq12_inside) if t0 is not None
                else float("nan"))
        print("    select %s   |S|=%d" % (fmt(secs) if secs == secs else "(injected)", len(seeds)))
        if st:
            print("    phases: mia %s  solo %s  curves %s  dp %s  realise %s  repair %s"
                  % (fmt(st.get("mia_seconds", 0.0)), fmt(st.get("solo_seconds", 0.0)),
                     fmt(st.get("curve_seconds", 0.0)), fmt(st.get("dp_seconds", 0.0)),
                     fmt(st.get("realise_seconds", 0.0)), fmt(st.get("repair_seconds", 0.0))))
            print("    communities used: %d / %d   quota blocks: %d"
                  % (st.get("n_communities_used", 0), st.get("n_communities", 0),
                     st.get("quota_blocks", 0)))
            print("    allocation: %s" % (st.get("allocation", {}),))
            dpv, val = st.get("dp_value", 0.0), st.get("value", 0.0)
            if dpv and val:
                print("    dp_value %.4f is an UPPER BOUND (sum_m I(S_m) >= I(union)); "
                      "realised %.4f, independence gap %.2f%%"
                      % (dpv, val, 100.0 * (dpv / val - 1.0)))
        rows.append((name, list(seeds), secs, st, counted))

    # ---- scoring: dedupe identical seed sets, h_eval is expensive -----------
    print("\n[scoring] one shared MIA(h_eval=%g) for every set ..." % args.h_eval)
    fine, crude, mcv = {}, {}, {}
    scored = []
    t0 = time.perf_counter()
    for (name, seeds, secs, st, counted) in rows:
        key = tuple(sorted(seeds))
        if key not in fine:
            fine[key] = referee.influence(list(seeds))          # Eq (18) at h_eval
            crude[key] = coarse.influence(list(seeds))          # Eq (18) at h_sel
            mcv[key] = (mc_batches(ds.out_adj, pp, seeds, args.mc, args.mc_batches,
                                   args.seed + 1) if args.mc > 0 else (float("nan"), 0.0))
        m, se = mcv[key]
        scored.append((name, seeds, secs, st, counted, fine[key], crude[key], m, se))
    print("[scoring] done in %s  (%d distinct seed sets for %d methods)"
          % (fmt(time.perf_counter() - t0), len(fine), len(scored)))

    base = scored[0]
    print("\n" + "=" * 88)
    print("[R] RESULTS -- one pp, one item (%d), one h_eval.  Comparable across rows." % item)
    print("=" * 88)
    print("%-16s %14s %9s %17s %11s  %s"
          % ("method", "I_h_eval", "vs base", "MC-IC +- SE", "select(s)", "|S|"))
    for (name, seeds, secs, _st, _pr, f, _c, m, se) in scored:
        rel = 100.0 * (f / base[5] - 1.0) if base[5] else 0.0
        mc = "%10.3f +-%5.2f" % (m, se) if m == m else "         --      "
        ts = "%11.2f" % secs if secs == secs else "  (injected)"
        print("%-16s %14.4f %8.2f%% %17s %s  %d" % (name, f, rel, mc, ts, len(seeds)))

    print("\n  historical column -- I at h_sel=%g, graded with the SAME threshold" % args.h)
    print("  used to select.  NOT a valid cross-method comparison; shown because")
    print("  it is what earlier results/ tables reported.")
    for (name, _s, _t, _st, _pr, _f, c, _m, _se) in scored:
        print("      %-16s %12.4f" % (name, c))

    # ---- pairwise overlap, so "identical set" is visible directly ----------
    print("\n  pairwise seed overlap:")
    names = [r[0] for r in scored]
    print("      %-16s %s" % ("", "".join("%16s" % n for n in names)))
    for (n1, s1, _t, _st, _pr, _f, _c, _m, _se) in scored:
        cells = []
        for (n2, s2, _t2, _st2, _pr2, _f2, _c2, _m2, _se2) in scored:
            k = len(set(s1) & set(s2))
            cells.append("%16s" % (("%d/%d" % (k, len(s1))) +
                                   ("*" if n1 != n2 and k == len(s1) == len(s2) else "")))
        print("      %-16s %s" % (n1, "".join(cells)))
    print("      (* = identical set)")

    # ---- complexity --------------------------------------------------------
    print("\n" + "=" * 88)
    print("[C] COMPLEXITY -- textbook cost, and the primitive that dominates it")
    print("=" * 88)
    print("  n=|V|=%d  m=|E|=%d  K=%d  C=%d  h_sel=%g"
          % (ds.n_users, ds.n_links, args.K, len(model.pi[0]), args.h))
    print("  t = mean |MIIA(v,h)|; each arborescence is a Dijkstra on -ln pp,")
    print("  costing O(t log t).  So (count, mean size) IS the empirical cost.\n")
    for (name, _s, secs, st, counted, _f, _c, _m, _se) in scored:
        cost, note = COMPLEXITY.get(name, ("?", ""))
        print("  %s" % name)
        print("      cost : %s" % cost)
        print("      why  : %s" % note)
        if counted is not None:
            cnt, avg = arborescence_stats(counted)
            print("      MEASURED: %d arborescences built, mean size %.1f" % (cnt, avg))
        if st:
            print("      MEASURED: %d curve gain evals, %d realisation gain evals"
                  % (st.get("curve_gain_evals", 0), st.get("realise_gain_evals", 0)))
            if st.get("curve_points_built"):
                print("      MEASURED: %d curve points built"
                      % st.get("curve_points_built", 0))
            if st.get("repaired"):
                print("      MEASURED: repair accepted %d swap(s)" % st.get("repair_swaps", 0))
        if secs == secs:
            print("      wall clock: %s" % fmt(secs))
        print("")

    if probe and args.probe_sample > 0:
        print("  full-graph vs induced-subgraph arborescence size")
        print("  (the ratio below is why CTIM is cheap and GlobalGreedy is not)")
        t0 = time.perf_counter()
        comm = detect_communities(model.pi)                     # Eq (19)
        tf, ti, ns = probe_arborescence_sizes(ds, pp, comm, args.h,
                                              args.probe_sample,
                                              random.Random(args.seed + 5))
        print("      sampled %d nodes in %s" % (ns, fmt(time.perf_counter() - t0)))
        print("      mean |MIIA| on the FULL graph      : %.2f" % tf)
        print("      mean |MIIA| on the INDUCED subgraph: %.2f" % ti)
        if ti > 0:
            print("      ratio  full / induced             : %.1fx" % (tf / ti))
        print("      Both CTIM and GlobalGreedy build ~n arborescences; this ratio")
        print("      is the per-arborescence cost gap between them.")

    print("\n  item %d done in %s   (Eq (12) weights %s, shared by every method)"
          % (item, fmt(time.perf_counter() - t_item), fmt(t_pp)))
    return [(name, f, m, se, secs, len(seeds))
            for (name, seeds, secs, _st, _ct, f, _c, m, se) in scored]


if __name__ == "__main__":
    sys.exit(main())
