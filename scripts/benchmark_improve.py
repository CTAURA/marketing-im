#!/usr/bin/env python3
"""The definitive CTIM / CTIM-G / GlobalGreedy comparison, graded honestly.

Why this script exists
----------------------
Every "GlobalGreedy beats CTIM by +X%" number this project has produced was
measured by scoring the seed set with the *same* truncated MIA that selected it
(``h_eval = h_sel = 0.1``).  That grading is circular, and it has been measured
to INVERT the ordering: at ``h = 0.1`` GlobalGreedy leads CTIM by +0.33%, at a
fixed fine evaluator ``h_eval = 0.001`` CTIM leads GlobalGreedy by +1.27%, and a
threshold-free Monte-Carlo IC referee agrees with CTIM.  So this script:

  * selects with ``h_sel`` in {0.1, 0.05} -- **every** method at **both**, because
    ``h_sel = 0.05`` is worth +2.48% to GlobalGreedy and comparing a tuned method
    against an untuned baseline would be cheating;
  * grades every seed set with ONE ``MIA(..., h=h_eval)`` object, built once,
    never at ``h_sel``;
  * cross-checks with ``ctim.ris_imm.ic_simulate``, a threshold-free referee that
    shares no machinery with MIA, run in batches under common random numbers so
    the batch means give a standard error;
  * ALSO prints the historical ``h = 0.1`` numbers, clearly labelled as circular,
    so the reader can see the inversion happen.

METRIC RULE: an MIA value is comparable to another MIA value only if ``pp``,
``h`` and the item are identical.  One ``pp`` dict object, one item, one
evaluator object, for the whole run.

Methods
-------
  CTIM            ``ctim.ctim.ctim_select_seeds(dp_tiebreak="paper-true")``, as
                  published (Algorithm 2; line 34 scores on the community's
                  node-induced subgraph)
  CTIM+repair     the above, then ``ctim.local_search.local_search`` on the
                  GLOBAL ``MIA(h=h_sel)`` -- fixing line 34's output after the fact
  CTIM-G          ``ctim.ctim_global.select_seeds_global(repair=False)`` --
                  fixing line 34 at the source (global gains, exact DP, quota
                  realisation)
  CTIM-G+repair   the above with ``repair=True``
  GlobalGreedy    ``MIA(h=h_sel).greedy_incremental(K)`` -- the baseline

Standard library only.  Python 3.9 compatible.  All randomness flows through
explicit ``random.Random`` objects.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.ctim import ctim_select_seeds, detect_communities        # noqa: E402
from ctim.ctim_global import select_seeds_global                   # noqa: E402
from ctim.dataset import load_dataset                              # noqa: E402
from ctim.influence import MIA, EdgeWeights                        # noqa: E402
from ctim.local_search import local_search, solo_influence_all     # noqa: E402
from ctim.ris_imm import ic_simulate                               # noqa: E402

METHODS = ["CTIM", "CTIM+repair", "CTIM-G", "CTIM-G+repair", "GlobalGreedy"]


# ---------------------------------------------------------------------------
# small stdlib-only statistics (no numpy, not even for a mean)
# ---------------------------------------------------------------------------


def mean(xs):
    xs = list(xs)
    return sum(xs) / float(len(xs)) if xs else 0.0


def stderr(xs):
    """Standard error of the mean of `xs` (sample sd / sqrt(n))."""
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 0.0
    mu = mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / float(n - 1)
    return math.sqrt(var / n)


def pct(a, b):
    """(a/b - 1) * 100, guarded."""
    return 100.0 * (a / b - 1.0) if b else 0.0


# ---------------------------------------------------------------------------
# MC-IC referee under common random numbers
# ---------------------------------------------------------------------------


def mc_batches(out_adj, pp, S, n_sims, n_batches, seed):
    """`n_batches` independent IC estimates, batch b driven by Random(seed+1000*b).

    The SAME seed sequence is used for every seed set, so the batch index is a
    shared random source and differences between methods are (nominally) paired.
    Caveat stated plainly: `ic_simulate` draws its coin flips in traversal order,
    so two different seed sets consume the shared stream differently -- this is
    seed-level pairing, not true common-random-numbers over a pre-sampled live
    edge set.  It reduces variance but does not eliminate it, which is why the
    batch-difference SE below is reported and used for the 2-sigma test.
    """
    per = max(1, n_sims // n_batches)
    return [ic_simulate(out_adj, pp, S, per, random.Random(seed + 1000 * b))
            for b in range(n_batches)]


# ---------------------------------------------------------------------------
# the methods
# ---------------------------------------------------------------------------


def run_method(name, K, h_sel, ctx):
    """Return (seeds, select_seconds, note).  `ctx` carries every shared object."""
    ds, model, item, ew, pp = ctx["ds"], ctx["model"], ctx["item"], ctx["ew"], ctx["pp"]
    mia_sel = ctx["mia_sel"][h_sel]
    solo_sel = ctx["solo_sel"][h_sel]
    comm = ctx["comm"]
    rng = random.Random(ctx["seed"])

    if name == "CTIM":
        t0 = time.perf_counter()
        seeds = ctim_select_seeds(model, ds, item, K, h=h_sel,
                                  dp_tiebreak="paper-true", edge_weights=ew)
        return list(seeds), time.perf_counter() - t0, ""

    if name == "CTIM+repair":
        base_seeds, base_secs = ctx["cache"][("CTIM", h_sel, K)]
        t0 = time.perf_counter()
        pool = list(mia_sel.nodes) if ctx["repair_pool"] <= 0 else None
        if pool is None:
            from ctim.local_search import top_candidates
            pool = top_candidates(mia_sel, ctx["repair_pool"], solo=solo_sel)
        S2, _v2, ls = local_search(mia_sel, base_seeds, pool, rng,
                                   max_rounds=ctx["repair_rounds"],
                                   solo=solo_sel, time_budget=ctx["repair_budget"])
        secs = time.perf_counter() - t0
        note = "%d swap%s%s" % (ls["swaps_accepted"],
                                "" if ls["swaps_accepted"] == 1 else "s",
                                ", certified" if ls["certified"] else "")
        if ls.get("timed_out"):
            note += ", TIMED OUT"
        return list(S2), base_secs + secs, note

    if name in ("CTIM-G", "CTIM-G+repair"):
        repair = name.endswith("+repair")
        t0 = time.perf_counter()
        seeds, st = select_seeds_global(
            model, ds, pp, K, rng, h=h_sel, comm=comm,
            mia=mia_sel, solo=solo_sel, repair=repair,
            repair_pool=ctx["repair_pool"], repair_rounds=ctx["repair_rounds"],
            repair_budget=ctx["repair_budget"])
        secs = time.perf_counter() - t0
        note = "alloc %d comm, dp %.1f vs realised %.1f (indep gap %.2f%%)" % (
            st["n_communities_used"], st["dp_value"], st["realised_value"],
            st["independence_gap_pct"])
        if st["realisation_fallback"]:
            note += ", FELL BACK to concat"
        if repair:
            note += "; repair %d swap%s%s" % (
                st["repair_swaps"], "" if st["repair_swaps"] == 1 else "s",
                ", certified" if st.get("repair_certified") else "")
        ctx["gstats"][(name, h_sel, K)] = st
        return list(seeds), secs, note

    if name == "GlobalGreedy":
        t0 = time.perf_counter()
        seeds = mia_sel.greedy_incremental(K)
        return list(seeds), time.perf_counter() - t0, ""

    raise ValueError("unknown method %r" % (name,))


# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset",
                   default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache",
                   default=os.path.join(_REPO_ROOT, "data", "processed",
                                        "_calib_model.pkl"))
    p.add_argument("--k-list", default="5,10,20,50")
    p.add_argument("--h-sel", default="0.1,0.05",
                   help="selection thresholds; EVERY method is run at EVERY one")
    p.add_argument("--h-eval", type=float, default=0.001,
                   help="the ONE fixed grading threshold (never equal to h_sel "
                        "in a reported comparison)")
    p.add_argument("--h-hist", type=float, default=0.1,
                   help="the historical/circular threshold, reported separately")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--mc", type=int, default=1000, help="total MC-IC simulations")
    p.add_argument("--mc-batches", type=int, default=5)
    p.add_argument("--skip-mc", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--item-index", type=int, default=0,
                   help="index into blob['test_items']")
    p.add_argument("--repair-pool", type=int, default=0,
                   help="0 = offer every node to local_search (keeps the "
                        "'certified 1-swap local optimum' claim complete)")
    p.add_argument("--repair-rounds", type=int, default=25)
    p.add_argument("--repair-budget", type=float, default=None,
                   help="optional wall-clock cap (s) per repair call")
    p.add_argument("--json", default=None, help="dump raw results here")
    p.add_argument("--load", default=None,
                   help="re-render the tables from a --json dump instead of "
                        "re-selecting and re-grading (costs nothing)")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    ks = [int(x) for x in args.k_list.split(",") if x.strip()]
    hs = [float(x) for x in args.h_sel.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    for m in methods:
        if m not in METHODS:
            print("ERROR: unknown method %r (known: %s)" % (m, ", ".join(METHODS)))
            return 2
    if args.h_eval in hs:
        print("ERROR: h_eval=%g is one of the selection thresholds -- that is "
              "exactly the circular grading this script exists to avoid."
              % args.h_eval)
        return 2

    if args.load:
        with open(args.load) as fh:
            blob = json.load(fh)
        rows = blob["rows"]
        print("[load] re-rendering %d rows from %s (no selection, no grading)"
              % (len(rows), args.load))
        report(rows, args, blob["hs"], blob["ks"], methods)
        print("\nWALL-CLOCK RUNTIME  %.1fs (re-render only; the measured run "
              "is the one that produced %s)"
              % (time.perf_counter() - t_all, args.load))
        return 0

    print("=" * 100)
    print("BENCHMARK  CTIM / CTIM+repair / CTIM-G / CTIM-G+repair / GlobalGreedy")
    print("  dataset   %s" % args.dataset)
    print("  K         %s" % ks)
    print("  h_sel     %s        h_eval %g (fixed, never = h_sel)   h_hist %g "
          "(circular, reported separately)" % (hs, args.h_eval, args.h_hist))
    print("  MC-IC     %d sims in %d batches, seeds %d+1000b (common across "
          "every seed set)" % (args.mc, args.mc_batches, args.seed))
    print("=" * 100)

    # ------------------------------------------------------------ setup
    if not os.path.exists(args.cache):
        print("ERROR: no cached model at %s -- run scripts/calibrate_h.py first."
              % args.cache)
        return 1

    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model = blob["model"]
    item = blob["test_items"][args.item_index]
    ew = EdgeWeights(model, ds)
    print("[setup] dataset + cached model            %7.2fs  (n_users=%d, item=%d)"
          % (time.perf_counter() - t0, ds.n_users, item))

    t0 = time.perf_counter()
    pp = ew.for_item(item)                                    # Eq (12), built ONCE
    print("[setup] Eq (12) pp for item %-8d      %7.2fs  (|pp|=%d)"
          % (item, time.perf_counter() - t0, len(pp)))

    t0 = time.perf_counter()
    comm = detect_communities(model.pi)                       # Eq (19)
    n_comm = max(comm) + 1 if comm else 0
    print("[setup] Eq (19) communities               %7.2fs  (C=%d)"
          % (time.perf_counter() - t0, n_comm))

    mia_sel, solo_sel = {}, {}
    for h in hs:
        t0 = time.perf_counter()
        mia_sel[h] = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)
        t_m = time.perf_counter() - t0
        t0 = time.perf_counter()
        solo_sel[h] = solo_influence_all(mia_sel[h])          # I({u}) for all u
        print("[setup] MIA(h=%-6g) + solo table        %7.2fs  (%.2f + %.2f)"
              % (h, t_m + (time.perf_counter() - t0), t_m,
                 time.perf_counter() - t0))

    ctx = {
        "ds": ds, "model": model, "item": item, "ew": ew, "pp": pp,
        "mia_sel": mia_sel, "solo_sel": solo_sel, "comm": comm,
        "seed": args.seed, "cache": {}, "gstats": {},
        "repair_pool": args.repair_pool, "repair_rounds": args.repair_rounds,
        "repair_budget": args.repair_budget,
    }

    # ------------------------------------------------------------ selection
    print("\n" + "-" * 100)
    print("SELECTION  (wall-clock per method; nothing is graded yet)")
    print("-" * 100)
    rows = []
    for h in hs:
        for K in ks:
            for name in methods:
                t0 = time.perf_counter()
                seeds, secs, note = run_method(name, K, h, ctx)
                ctx["cache"][(name, h, K)] = (list(seeds), secs)
                print("  h_sel=%-5g K=%-3d %-14s %8.2fs  |S|=%-3d %s"
                      % (h, K, name, secs, len(seeds), note), flush=True)
                rows.append({"method": name, "h_sel": h, "K": K,
                             "seeds": list(seeds), "select_seconds": secs,
                             "note": note})

    # ------------------------------------------------------------ grading
    print("\n" + "-" * 100)
    print("GRADING  (one MIA(h=%g) object for every set; deduplicated)" % args.h_eval)
    print("-" * 100)
    distinct = {}
    for r in rows:
        distinct.setdefault(tuple(sorted(r["seeds"])), []).append(r)
    print("  %d (method,h_sel,K) rows -> %d distinct seed sets"
          % (len(rows), len(distinct)))

    t0 = time.perf_counter()
    mia_eval = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h_eval)
    mia_hist = (mia_sel[args.h_hist] if args.h_hist in mia_sel
                else MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h_hist))
    print("  evaluator objects built                 %7.2fs" % (time.perf_counter() - t0))

    scores = {}
    for i, key in enumerate(distinct):
        S = list(key)
        t0 = time.perf_counter()
        v_eval = mia_eval.influence(S)                        # Eq (18) at h_eval
        t_eval = time.perf_counter() - t0
        t0 = time.perf_counter()
        v_hist = mia_hist.influence(S)                        # Eq (18) at h_hist
        t_hist = time.perf_counter() - t0
        if args.skip_mc:
            bs = []
        else:
            t0 = time.perf_counter()
            bs = mc_batches(ds.out_adj, pp, S, args.mc, args.mc_batches, args.seed)
            t_mc = time.perf_counter() - t0
        scores[key] = {"I_eval": v_eval, "I_hist": v_hist, "mc_batches": bs,
                       "mc": mean(bs) if bs else float("nan"),
                       "mc_se": stderr(bs) if bs else float("nan")}
        print("    [%2d/%2d] |S|=%-3d  I_%g=%10.3f (%5.1fs)  I_%g=%9.3f (%4.1fs)"
              "  MC-IC=%s"
              % (i + 1, len(distinct), len(S), args.h_eval, v_eval, t_eval,
                 args.h_hist, v_hist, t_hist,
                 "skipped" if args.skip_mc
                 else "%.2f+-%.2f (%.1fs)" % (scores[key]["mc"],
                                              scores[key]["mc_se"], t_mc)),
              flush=True)

    for r in rows:
        r.update(scores[tuple(sorted(r["seeds"]))])

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "h_eval": args.h_eval, "h_hist": args.h_hist,
                       "item": item, "ks": ks, "hs": hs, "mc": args.mc,
                       "mc_batches": args.mc_batches, "seed": args.seed},
                      fh, indent=1)
        print("\n[json] %s" % args.json)

    report(rows, args, hs, ks, methods)

    print("\n" + "=" * 100)
    print("WALL-CLOCK RUNTIME  %.1fs (%.1f min)"
          % (time.perf_counter() - t_all, (time.perf_counter() - t_all) / 60.0))
    print("=" * 100)
    return 0


def report(rows, args, hs, ks, methods):
    """Print every table and the verdict.  Pure function of `rows`."""

    def find(name, h, K):
        for r in rows:
            if r["method"] == name and r["h_sel"] == h and r["K"] == K:
                return r
        return None

    def paired(a, b):
        """(mean difference, SE) of a's MC-IC batch means minus b's, paired by
        batch index (batch b of every set was driven by Random(seed+1000*b))."""
        if not a.get("mc_batches") or not b.get("mc_batches"):
            return float("nan"), float("nan")
        d = [x - y for x, y in zip(a["mc_batches"], b["mc_batches"])]
        return mean(d), stderr(d)

    def call(dm, se):
        if dm != dm or se != se:
            return "n/a"
        if se == 0.0:
            return "IDENTICAL SET" if dm == 0.0 else "?"
        if dm > 2 * se:
            return "CONFIRMS"
        if dm < -2 * se:
            return "REFUTES"
        return "tie (<2 sigma)"

    print("\n" + "=" * 100)
    print("TABLE 1 -- THE RESULT.  Graded at the FIXED h_eval = %g, never at h_sel."
          % args.h_eval)
    print("=" * 100)
    hdr = ("%-14s %6s %4s %12s %8s %16s %10s %5s %8s"
           % ("method", "h_sel", "K", "I_%g" % args.h_eval, "vs GG",
              "MC-IC +- SE", "select(s)", "|S|", "ovl.CTIM"))
    for h in hs:
        for K in ks:
            print("\n" + hdr)
            print("-" * 100)
            gg = find("GlobalGreedy", h, K)
            ctim_seeds = set(find("CTIM", h, K)["seeds"]) if find("CTIM", h, K) else set()
            best = max((r for r in rows if r["h_sel"] == h and r["K"] == K),
                       key=lambda r: r["I_eval"])
            for name in methods:
                r = find(name, h, K)
                if r is None:
                    continue
                ovl = len(ctim_seeds & set(r["seeds"]))
                mark = "  <-- best" if r is best else ""
                print("%-14s %6g %4d %12.3f %+7.2f%% %16s %10.2f %5d %5d/%-2d%s"
                      % (name, h, K, r["I_eval"],
                         pct(r["I_eval"], gg["I_eval"]) if gg else 0.0,
                         "skipped" if args.skip_mc
                         else "%9.2f+-%.2f" % (r["mc"], r["mc_se"]),
                         r["select_seconds"], len(r["seeds"]),
                         ovl, len(ctim_seeds) if ctim_seeds else 0, mark))

    print("\n" + "=" * 100)
    print("TABLE 2 -- THE HISTORICAL / CIRCULAR NUMBER.  I at h = %g." % args.h_hist)
    print("  For every row whose h_sel = %g this grades a set with the very "
          "evaluator that" % args.h_hist)
    print("  selected it.  It is here ONLY so the inversion against Table 1 is "
          "visible.  Do not")
    print("  quote it as a comparison.")
    print("=" * 100)
    print("%-14s %6s %4s %12s %8s %14s %8s"
          % ("method", "h_sel", "K", "I_%g" % args.h_hist, "vs GG",
             "I_%g vs GG" % args.h_eval, "INVERTS?"))
    print("-" * 100)
    n_invert = 0
    for h in hs:
        for K in ks:
            gg = find("GlobalGreedy", h, K)
            for name in methods:
                r = find(name, h, K)
                if r is None or gg is None:
                    continue
                d_hist = pct(r["I_hist"], gg["I_hist"])
                d_eval = pct(r["I_eval"], gg["I_eval"])
                inv = ""
                if name != "GlobalGreedy" and d_hist * d_eval < 0:
                    inv = "YES"
                    n_invert += 1
                print("%-14s %6g %4d %12.3f %+7.2f%% %+13.2f%% %8s"
                      % (name, h, K, r["I_hist"], d_hist, d_eval, inv))
            print()
    print("  %d of %d non-baseline rows change SIGN between the circular h=%g "
          "grading and the" % (n_invert, len(rows) - len(hs) * len(ks), args.h_hist))
    print("  fixed h=%g grading." % args.h_eval)

    # ------------------------------------------------------------ verdict
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print("Rule applied: a difference smaller than 2 x SE of the paired batch")
    print("differences is NOT a difference.  MC-IC 'confirms' only when it ranks")
    print("the winner above the runner-up by more than 2 sigma.\n")

    print("EVERY method vs GlobalGreedy at the SAME h_sel and K -- the only fair")
    print("comparison.  MC-IC column is the PAIRED batch difference.\n")
    print("%-14s %6s %4s %10s %20s %16s"
          % ("method", "h_sel", "K", "dI_%g" % args.h_eval,
             "MC-IC paired diff", "referee"))
    print("-" * 100)
    for K in ks:
        for h in hs:
            gg = find("GlobalGreedy", h, K)
            for name in methods:
                r = find(name, h, K)
                if r is None or gg is None or name == "GlobalGreedy":
                    continue
                dm, se = paired(r, gg)
                same = tuple(sorted(r["seeds"])) == tuple(sorted(gg["seeds"]))
                print("%-14s %6g %4d %+9.2f%% %13.2f +- %-5.2f %16s"
                      % (name, h, K, pct(r["I_eval"], gg["I_eval"]), dm, se,
                         "IDENTICAL SET" if same else call(dm, se)))
        print()

    print("-" * 100)
    print("Best method at each K (ties on the seed set are named as ties).\n")
    verdicts = []
    for K in ks:
        pool = [r for r in rows if r["K"] == K]
        best = max(pool, key=lambda r: r["I_eval"])
        bkey = tuple(sorted(best["seeds"]))
        tied = sorted(set(r["method"] for r in pool
                          if tuple(sorted(r["seeds"])) == bkey
                          and r["h_sel"] == best["h_sel"]))
        # runner-up must be a DIFFERENT seed set, else the comparison is vacuous
        rest = sorted((r for r in pool if tuple(sorted(r["seeds"])) != bkey),
                      key=lambda r: -r["I_eval"])
        second = rest[0] if rest else None
        print("  K=%-3d  best I_%g = %.3f  at h_sel=%g, achieved by: %s"
              % (K, args.h_eval, best["I_eval"], best["h_sel"], ", ".join(tied)))
        if second is not None:
            dm, se = paired(best, second)
            print("         best DISTINCT runner-up: %-14s (h_sel=%g) I=%.3f "
                  "-> %+.2f%%" % (second["method"], second["h_sel"],
                                  second["I_eval"],
                                  pct(best["I_eval"], second["I_eval"])))
            print("         MC-IC paired %+.2f +- %.2f  ->  %s"
                  % (dm, se, call(dm, se)))
        for h in hs:
            sub = [r for r in pool if r["h_sel"] == h]
            b = max(sub, key=lambda r: r["I_eval"])
            gg = find("GlobalGreedy", h, K)
            dm, se = paired(b, gg)
            print("         at h_sel=%-5g best is %-14s %+6.2f%% vs GlobalGreedy"
                  "  (MC-IC %+.2f +- %.2f, %s)"
                  % (h, b["method"], pct(b["I_eval"], gg["I_eval"]), dm, se,
                     "IDENTICAL SET"
                     if tuple(sorted(b["seeds"])) == tuple(sorted(gg["seeds"]))
                     else call(dm, se)))
        verdicts.append({"K": K, "best": tied, "h_sel": best["h_sel"],
                         "I": best["I_eval"]})
        print()

    # speed-quality frontier
    print("-" * 100)
    print("SPEED-QUALITY FRONTIER  (quality = I at h_eval=%g, relative to "
          "GlobalGreedy at the SAME h_sel)" % args.h_eval)
    print("-" * 100)
    print("%-14s %6s %4s %10s %10s %10s"
          % ("method", "h_sel", "K", "quality", "select(s)", "speedup"))
    for h in hs:
        for K in ks:
            gg = find("GlobalGreedy", h, K)
            for name in methods:
                r = find(name, h, K)
                if r is None or gg is None:
                    continue
                sp = (gg["select_seconds"] / r["select_seconds"]
                      if r["select_seconds"] > 0 else float("inf"))
                print("%-14s %6g %4d %+9.2f%% %10.2f %9.2fx"
                      % (name, h, K, pct(r["I_eval"], gg["I_eval"]),
                         r["select_seconds"], sp))
            print()
    return verdicts


if __name__ == "__main__":
    sys.exit(main())
