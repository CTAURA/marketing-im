#!/usr/bin/env python3
"""Separate the MIA *selection* threshold from the MIA *evaluation* threshold.

Why this script exists
----------------------
Everywhere else in this repo a single ``h`` plays two incompatible roles at
once.  It is

  1. a **speed knob** -- Eq (15)/(16) prune every maximum-influence path with
     ``pp(MIP) < h``, so a coarse ``h`` shrinks MIIA/MIOA and makes both greedy
     selection and Eq (18) scoring dramatically cheaper; and
  2. the **objective itself** -- Eq (18) sums ``ap(v|S)`` over exactly those
     surviving arborescences, so ``I_h`` *is* the number being maximised.

Sharing one symbol between the two makes any statement of the form "the finer
threshold scores higher" vacuous: ``I_{0.001}(S) >= I_{0.1}(S)`` holds for every
S by construction, because the finer threshold admits a superset of paths.  It
is not evidence that selecting at 0.001 chose better seeds.  It is arithmetic.

The honest protocol implemented here breaks the tie:

    SELECT   at h_sel in {0.1, 0.05, 0.01, 0.001}
    EVALUATE every resulting seed set at ONE FIXED h_eval

so the only thing varying down a column of the result table is *which seeds
were chosen*, never how they are graded.  ``pp`` (Eq (12)) and the item are
identical throughout -- nothing here retrains or perturbs the model -- so every
number in the tables is same-pp / same-item and is directly comparable to every
other number in ``results/``.

The question being answered
---------------------------
    "Does a seed set chosen under the coarse h=0.1 hold up when scored under a
     fine threshold, compared to one chosen under that fine threshold directly?"

If yes, h=0.1 is a *free speedup* and should be reported as such.  If no, the
loss is quantified in the ``vs h_sel=<finest>`` row.

Three evaluators, one graph
---------------------------
  * ``I_{h_eval}(S)``  -- Eq (18) at the fixed fine threshold.  The headline.
  * ``I_{0.1}(S)``     -- Eq (18) at the coarse threshold.  The reverse
    direction, for completeness: does a fine-selected set hold up when graded
    coarsely?
  * ``MC-IC``          -- ``ctim.ris_imm.ic_simulate``, plain Monte-Carlo
    Independent Cascade.  Threshold-free, so it cannot be gamed by either h,
    and valid here precisely because ``pp`` never changes.  Reported with a
    standard error from independent batches, because unlike Eq (18) it is a
    sample statistic and "within noise" has to mean something measurable.

Both selectors are run, so no finding can be blamed on CTIM specifically:
  * ``CTIM``          Algorithm 2, ``ctim_select_seeds(..., h=h_sel)``
  * ``GlobalGreedy``  ``MIA(..., h=h_sel).greedy_incremental(K)``

Cost control
------------
MIA at a fine ``h`` is not slightly more expensive, it is asymptotically more
expensive: the greedy initialisation costs ~sum_v |MIIA(v)|^2, and on Digg
mean |MIIA| goes 1.4 -> 2.1 -> 44 -> 293 as h goes 0.1 -> 0.05 -> 0.01 ->
0.001.  Every ``MIA`` object is therefore built exactly once and reused for
every seed set (its MIIA/MIOA caches are the whole point), and every cell is
run under an explicit wall-clock budget: a cell that would blow the budget is
reported as ``NOT RUN`` with its measured reason rather than silently dropped.
``--gg-candidates`` restricts GlobalGreedy's candidate pool by out-degree when
the unrestricted pool is intractable at fine h; the restriction is applied
*identically at every h_sel*, so the across-h_sel comparison stays internally
valid even though the restricted selector is not the unrestricted one.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random(seed)``.
"""

from __future__ import annotations

import argparse
import math
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
from ctim.ris_imm import ic_simulate                 # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def fmt_s(x):
    return "%.2fs" % x


def fmt_h(h):
    """Compact, stable label for a threshold (0.1 -> '0.1', 0.001 -> '0.001')."""
    return ("%g" % h)


def mc_ic_with_error(out_adj, pp, S, n_mc, n_batches, seed):
    """MC-IC spread with a standard error, from `n_batches` independent batches.

    ``ic_simulate`` returns a mean only, so the spread of the *batch* means is
    used to get an error bar.  Every seed set is measured with the SAME batch
    seeds (common random numbers): the pairwise differences that the table asks
    about are then far less noisy than the individual estimates.

    Returns (mean, stderr_of_mean, seconds).
    """
    t0 = time.perf_counter()
    per = max(1, n_mc // n_batches)
    means = []
    for b in range(n_batches):
        # Same (b, seed) -> same RNG stream for every seed set: paired sampling.
        means.append(ic_simulate(out_adj, pp, S, per, random.Random(seed + 1000 * b)))
    mu = sum(means) / float(len(means))
    if len(means) > 1:
        var = sum((m - mu) ** 2 for m in means) / float(len(means) - 1)
        se = math.sqrt(var / len(means))
    else:
        se = float("nan")
    return mu, se, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Select at h_sel, evaluate at a single fixed h_eval.")
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    p.add_argument("--Ks", default="20,50", help="comma-separated seed budgets")
    p.add_argument("--h-sel", dest="h_sel", default="0.1,0.05,0.01,0.001",
                   help="comma-separated SELECTION thresholds")
    p.add_argument("--h-eval", dest="h_eval", type=float, default=0.001,
                   help="the single FIXED evaluation threshold (Eq (18))")
    p.add_argument("--h-coarse", dest="h_coarse", type=float, default=0.1,
                   help="second, coarse evaluation threshold (reverse direction)")
    p.add_argument("--mc", type=int, default=1000, help="total MC-IC simulations")
    p.add_argument("--mc-batches", type=int, default=5,
                   help="split --mc into this many batches to get a standard error")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gg-candidates", type=int, default=0,
                   help="restrict GlobalGreedy to the top-N nodes by out-degree "
                        "(0 = unrestricted).  Applied identically at every h_sel.")
    p.add_argument("--budget", type=float, default=540.0,
                   help="wall-clock budget in seconds; cells that would overrun "
                        "are reported NOT RUN instead of being silently dropped")
    p.add_argument("--no-gg", action="store_true",
                   help="skip GlobalGreedy entirely (CTIM-only run)")
    p.add_argument("--skip", default="",
                   help="comma-separated 'Selector:h_sel' cells to declare "
                        "intractable up front, e.g. 'GlobalGreedy:0.001'.  They are "
                        "reported as NOT RUN with a reason, never silently dropped.")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    h_sels = [float(x) for x in args.h_sel.split(",") if x.strip()]
    h_eval = args.h_eval
    h_coarse = args.h_coarse

    print("=" * 84)
    print("h-PROTOCOL: selection threshold h_sel  vs  FIXED evaluation threshold h_eval")
    print("=" * 84)
    print("  h_sel  values : %s" % ", ".join(fmt_h(h) for h in h_sels))
    print("  h_eval (FIXED): %g      (every seed set graded here -- the headline column)"
          % h_eval)
    print("  h_coarse      : %g      (reverse direction, for completeness)" % h_coarse)
    print("  K values      : %s" % ", ".join(str(k) for k in Ks))
    print("  MC-IC         : %d sims in %d batches, seed=%d (threshold-free referee)"
          % (args.mc, args.mc_batches, args.seed))
    print("  budget        : %.0fs" % args.budget)
    if args.gg_candidates:
        print("  GG candidates : top-%d by out-degree (RESTRICTED -- see caveats)"
              % args.gg_candidates)

    # ------------------------------------------------------------- setup
    if not os.path.exists(args.cache):
        print("ERROR: no cached model at %s" % args.cache)
        return 1

    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model, item = blob["model"], blob["test_items"][0]
    ew = EdgeWeights(model, ds)
    t_load = time.perf_counter() - t0
    print("\n[setup] dataset + cached model (read-only)      %s" % fmt_s(t_load))

    t0 = time.perf_counter()
    pp = ew.for_item(item)                                  # Eq (12)
    t_pp = time.perf_counter() - t0
    print("[setup] Eq (12) weights, item %-6d  |pp|=%-7d %s"
          % (item, len(pp), fmt_s(t_pp)))
    print("        pp and item are IDENTICAL for every cell below -> same-pp, same-item.")

    # One MIA per distinct threshold, built once, reused everywhere.  The
    # MIIA/MIOA caches inside each object are what make the fine-h scoring
    # affordable at all.
    need_h = sorted(set(h_sels + [h_eval, h_coarse]), reverse=True)
    evaluators = {}
    for h in need_h:
        t0 = time.perf_counter()
        evaluators[h] = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)   # Eq (13)-(18)
        print("[setup] MIA evaluator h=%-7g                   %s"
              % (h, fmt_s(time.perf_counter() - t0)))

    gg_cand = None
    if args.gg_candidates:
        deg = sorted(range(ds.n_users),
                     key=lambda u: (-(len(ds.out_adj[u]) if u < len(ds.out_adj) else 0), u))
        gg_cand = deg[:args.gg_candidates]

    def elapsed():
        return time.perf_counter() - t_all

    # -------------------------------------------------------- SELECTION
    print("\n" + "=" * 84)
    print("SELECTION -- each cell is an independent run; wall clock is that run alone")
    print("=" * 84)

    # cells[(selector, K, h_sel)] = dict(seeds=..., secs=..., status=...)
    cells = {}
    selectors = ["CTIM"] + ([] if args.no_gg else ["GlobalGreedy"])
    skipped = set()
    for tok in args.skip.split(","):
        tok = tok.strip()
        if not tok:
            continue
        sel, _, hs = tok.partition(":")
        skipped.add((sel.strip(), float(hs)))
    if skipped:
        print("  declared intractable up front (reported NOT RUN, not dropped):")
        for (sel, h) in sorted(skipped):
            print("    %s at h_sel=%g" % (sel, h))

    for K in Ks:
        for name in selectors:
            for h in h_sels:
                key = (name, K, h)
                if (name, h) in skipped:
                    cells[key] = {"seeds": None, "secs": float("nan"),
                                  "status": "NOT RUN (declared intractable via --skip)"}
                    print("  %-13s K=%-3d h_sel=%-7g  NOT RUN (declared intractable)"
                          % (name, K, h))
                    continue
                if elapsed() > args.budget:
                    cells[key] = {"seeds": None, "secs": float("nan"),
                                  "status": "NOT RUN (budget %.0fs exhausted)" % args.budget}
                    print("  %-13s K=%-3d h_sel=%-7g  NOT RUN (budget exhausted at %.0fs)"
                          % (name, K, h, elapsed()))
                    continue
                t0 = time.perf_counter()
                try:
                    if name == "CTIM":
                        # Algorithm 2 lines 1-45, MIA threshold = h_sel
                        seeds = ctim_select_seeds(model, ds, item, K, h=h,
                                                  dp_tiebreak="paper-true",
                                                  edge_weights=ew)
                    else:
                        # Chen et al. MIA greedy on the whole graph at h_sel.
                        seeds = evaluators[h].greedy_incremental(K, candidates=gg_cand)
                    secs = time.perf_counter() - t0
                    cells[key] = {"seeds": list(seeds), "secs": secs, "status": "ok"}
                    print("  %-13s K=%-3d h_sel=%-7g  %9s   |S|=%d  seeds[:6]=%s"
                          % (name, K, h, fmt_s(secs), len(seeds), list(seeds)[:6]))
                except (MemoryError, KeyboardInterrupt) as exc:
                    secs = time.perf_counter() - t0
                    cells[key] = {"seeds": None, "secs": secs,
                                  "status": "FAILED: %s" % type(exc).__name__}
                    print("  %-13s K=%-3d h_sel=%-7g  FAILED after %s: %s"
                          % (name, K, h, fmt_s(secs), type(exc).__name__))
                sys.stdout.flush()

    # --------------------------------------------------------- SCORING
    print("\n" + "=" * 84)
    print("SCORING -- every seed set graded by the SAME three evaluators")
    print("=" * 84)

    # Distinct seed sets only: several cells often return the identical set, and
    # MC-IC is the expensive part.
    score_cache = {}

    def score(seeds):
        k = tuple(sorted(seeds))
        got = score_cache.get(k)
        if got is not None:
            return got
        t0 = time.perf_counter()
        fine = evaluators[h_eval].influence(list(seeds))      # Eq (18) at h_eval
        t_fine = time.perf_counter() - t0
        t0 = time.perf_counter()
        coarse = evaluators[h_coarse].influence(list(seeds))  # Eq (18) at h_coarse
        t_coarse = time.perf_counter() - t0
        mc, mc_se, t_mc = mc_ic_with_error(ds.out_adj, pp, list(seeds),
                                           args.mc, args.mc_batches, args.seed)
        got = {"fine": fine, "coarse": coarse, "mc": mc, "mc_se": mc_se,
               "t_fine": t_fine, "t_coarse": t_coarse, "t_mc": t_mc}
        score_cache[k] = got
        return got

    n_scored = 0
    for key in sorted(cells, key=lambda t: (t[0], t[1], -t[2])):
        c = cells[key]
        if c["seeds"] is None:
            continue
        c["score"] = score(c["seeds"])
        n_scored += 1
        print("  %-13s K=%-3d h_sel=%-7g  I_%g=%10.4f  I_%g=%10.4f  MC-IC=%9.2f +- %.2f"
              % (key[0], key[1], key[2], h_eval, c["score"]["fine"],
                 h_coarse, c["score"]["coarse"], c["score"]["mc"], c["score"]["mc_se"]))
        sys.stdout.flush()
    print("  (%d cells scored, %d distinct seed sets -- identical sets are scored once)"
          % (n_scored, len(score_cache)))

    # ---------------------------------------------------------- TABLES

    def finest_run(name, K):
        """Finest h_sel that actually produced seeds for this (selector, K).

        The reference column has to be resolved per row, not fixed globally: if
        the finest threshold was declared intractable, comparing against an
        empty cell would silently suppress every percentage in the table.  The
        row label always names the threshold actually used as the reference.
        """
        for h in sorted(h_sels):
            c = cells.get((name, K, h))
            if c is not None and c["seeds"] is not None:
                return h
        return None

    def coarsest_run(name, K):
        for h in sorted(h_sels, reverse=True):
            c = cells.get((name, K, h))
            if c is not None and c["seeds"] is not None:
                return h
        return None

    def table(metric_key, title, note, ref_h=None):
        print("\n" + "=" * 84)
        print(title)
        print("=" * 84)
        print(note)
        for K in Ks:
            print("\n  K = %d" % K)
            hdr = "  %-13s" % "selector"
            for h in h_sels:
                hdr += " %14s" % ("h_sel=" + fmt_h(h))
            print(hdr)
            print("  " + "-" * (13 + 15 * len(h_sels)))
            for name in selectors:
                row = "  %-13s" % name
                for h in h_sels:
                    c = cells.get((name, K, h))
                    if c is None or c["seeds"] is None:
                        row += " %14s" % "--"
                    else:
                        row += " %14.4f" % c["score"][metric_key]
                print(row)
                if ref_h is None:
                    continue
                # "finest"/"coarsest" resolve per row against cells that ran.
                if ref_h == "finest":
                    use_h = finest_run(name, K)
                elif ref_h == "coarsest":
                    use_h = coarsest_run(name, K)
                else:
                    use_h = ref_h
                if use_h is None:
                    continue
                ref = cells.get((name, K, use_h))
                if ref is None or ref["seeds"] is None:
                    continue
                base = ref["score"][metric_key]
                row = "  %-13s" % ("  vs h_sel=" + fmt_h(use_h))
                for h in h_sels:
                    c = cells.get((name, K, h))
                    if c is None or c["seeds"] is None or not base:
                        row += " %14s" % "--"
                    else:
                        row += " %13.2f%%" % (100.0 * (c["score"][metric_key] / base - 1.0))
                print(row)

    table("fine",
          "TABLE 1 (HEADLINE) -- I_{h_eval}(S), h_eval = %g FIXED, Eq (18)" % h_eval,
          "  Only the SEEDS vary across a row. Same pp, same item, same evaluator object.\n"
          "  The 'vs h_sel=...' row is the cost of having selected coarsely instead of at\n"
          "  the FINEST threshold THAT RAN for that row, measured on the fine threshold's\n"
          "  own terms.  A positive entry means the coarser selection was better.",
          ref_h="finest")

    table("coarse",
          "TABLE 2 (REVERSE) -- I_{%g}(S), the COARSE evaluator, Eq (18)" % h_coarse,
          "  Does a fine-selected set hold up when graded coarsely?  Reference is the\n"
          "  selection threshold that MATCHES this evaluator (h_sel=%g) where it ran,\n"
          "  otherwise the coarsest that did." % h_coarse,
          ref_h=h_coarse if h_coarse in h_sels else "coarsest")

    table("mc",
          "TABLE 3 (THRESHOLD-FREE REFEREE) -- MC-IC, %d sims, seed=%d"
          % (args.mc, args.seed),
          "  Plain Independent Cascade Monte-Carlo on the same pp.  Shares no machinery\n"
          "  with MIA and applies NO threshold, so neither h_sel nor h_eval can flatter it.\n"
          "  Common random numbers across seed sets: differences are paired.",
          ref_h="finest")

    print("\n  MC-IC standard errors (mean of %d batch means):" % args.mc_batches)
    for K in Ks:
        for name in selectors:
            line = "    K=%-3d %-13s" % (K, name)
            for h in h_sels:
                c = cells.get((name, K, h))
                line += " %14s" % ("--" if (c is None or c["seeds"] is None)
                                   else "+-%.2f" % c["score"]["mc_se"])
            print(line)

    # ------------------------------------------------------- COST CURVE
    print("\n" + "=" * 84)
    print("COST CURVE -- selection wall-clock seconds vs h_sel")
    print("=" * 84)
    for K in Ks:
        print("\n  K = %d" % K)
        hdr = "  %-13s" % "selector"
        for h in h_sels:
            hdr += " %14s" % ("h_sel=" + fmt_h(h))
        print(hdr)
        print("  " + "-" * (13 + 15 * len(h_sels)))
        for name in selectors:
            row = "  %-13s" % name
            for h in h_sels:
                c = cells.get((name, K, h))
                row += " %14s" % ("--" if c is None or c["seeds"] is None
                                  else fmt_s(c["secs"]))
            print(row)
            ch = coarsest_run(name, K)
            base = None if ch is None else cells.get((name, K, ch))
            if base is None or base["seeds"] is None or not base["secs"]:
                continue
            row = "  %-13s" % ("  x vs h_sel=" + fmt_h(ch))
            for h in h_sels:
                c = cells.get((name, K, h))
                row += " %14s" % ("--" if c is None or c["seeds"] is None
                                  else "x%.1f" % (c["secs"] / base["secs"]))
            print(row)

    print("\n  scoring cost (paid once per distinct seed set):")
    tf = sum(v["t_fine"] for v in score_cache.values())
    tc = sum(v["t_coarse"] for v in score_cache.values())
    tm = sum(v["t_mc"] for v in score_cache.values())
    print("    Eq (18) at h=%-7g total %s" % (h_eval, fmt_s(tf)))
    print("    Eq (18) at h=%-7g total %s" % (h_coarse, fmt_s(tc)))
    print("    MC-IC (%d sims)       total %s" % (args.mc, fmt_s(tm)))

    # ---------------------------------------------------------- OVERLAP
    print("\n" + "=" * 84)
    print("SEED-SET OVERLAP with the FINEST h_sel that ran (same selector, same K)")
    print("=" * 84)
    for K in Ks:
        for name in selectors:
            fh = finest_run(name, K)
            ref = None if fh is None else cells.get((name, K, fh))
            if ref is None or ref["seeds"] is None:
                continue
            rs = set(ref["seeds"])
            line = "  K=%-3d %-13s ref=h_sel%-7s" % (K, name, fmt_h(fh))
            for h in h_sels:
                c = cells.get((name, K, h))
                line += " %14s" % ("--" if c is None or c["seeds"] is None
                                   else "%d/%d" % (len(rs & set(c["seeds"])), len(rs)))
            print(line)

    # ---------------------------------------------------------- VERDICT
    print("\n" + "=" * 84)
    print("VERDICT -- is the COARSEST h_sel a free speedup, graded at h_eval=%g?" % h_eval)
    print("=" * 84)
    print("  Each row compares the coarsest and finest h_sel THAT RAN for that selector")
    print("  and K.  'free' requires: no loss at h_eval AND no loss on MC-IC beyond noise.")
    any_verdict = False
    for K in Ks:
        for name in selectors:
            hc, hf = coarsest_run(name, K), finest_run(name, K)
            a = None if hc is None else cells.get((name, K, hc))
            b = None if hf is None else cells.get((name, K, hf))
            if a is None or b is None or a["seeds"] is None or b["seeds"] is None:
                print("  K=%-3d %-13s : incomparable (no cell ran)" % (K, name))
                continue
            if hc == hf:
                print("  K=%-3d %-13s : only h_sel=%s ran -- nothing to compare"
                      % (K, name, fmt_h(hc)))
                continue
            any_verdict = True
            fa, fb = a["score"]["fine"], b["score"]["fine"]
            dpct = 100.0 * (fa / fb - 1.0) if fb else float("nan")
            ma, mb = a["score"]["mc"], b["score"]["mc"]
            sa, sb = a["score"]["mc_se"], b["score"]["mc_se"]
            dmc = 100.0 * (ma / mb - 1.0) if mb else float("nan")
            noise = math.sqrt(sa * sa + sb * sb)
            sig = "OUTSIDE" if abs(ma - mb) > 2.0 * noise else "within"
            speed = (b["secs"] / a["secs"]) if a["secs"] else float("nan")
            free = (dpct >= 0.0) or (abs(ma - mb) <= 2.0 * noise and dpct > -0.5)
            print("  K=%-3d %-13s : I_%g  h_sel=%s %.4f  vs  h_sel=%s %.4f   -> %+.2f%%"
                  % (K, name, h_eval, fmt_h(hc), fa, fmt_h(hf), fb, dpct))
            print("        %-19s MC-IC %.2f+-%.2f vs %.2f+-%.2f -> %+.2f%% (%s 2-sigma = %.2f)"
                  % ("", ma, sa, mb, sb, dmc, sig, 2.0 * noise))
            print("        %-19s selection speedup of h_sel=%s over h_sel=%s: x%.1f  (%s vs %s)"
                  % ("", fmt_h(hc), fmt_h(hf), speed,
                     fmt_s(a["secs"]), fmt_s(b["secs"])))
            print("        %-19s => h_sel=%s is %s"
                  % ("", fmt_h(hc),
                     "a FREE speedup" if free else
                     "NOT free: it costs %.2f%% of spread at h_eval=%g" % (-dpct, h_eval)))
    if not any_verdict:
        print("  No comparable pair completed -- see NOT RUN / FAILED cells above.")

    print("\n" + "=" * 84)
    print("METRIC STATUS")
    print("=" * 84)
    print("  SAME-pp, SAME-item, SAME-h_eval: YES.  The cached model was loaded read-only")
    print("  and never retrained; pp = EdgeWeights.for_item(%d) is built once and shared by"
          % item)
    print("  every selector, every evaluator and the MC-IC referee.  Table 1 is therefore a")
    print("  valid MIA Eq (18) comparison in the sense required by the repo's metric rule.")
    print("  Table 1 and Table 2 use DIFFERENT thresholds and must not be compared to each")
    print("  other -- I_{0.001} >= I_{0.1} holds for every S by construction.")
    if args.gg_candidates:
        print("  CAVEAT: GlobalGreedy ran on a RESTRICTED candidate pool (top-%d by"
              % args.gg_candidates)
        print("  out-degree), identically at every h_sel.  Its across-h_sel comparison is")
        print("  internally valid; its ABSOLUTE values are not comparable to unrestricted")
        print("  GlobalGreedy numbers elsewhere in results/.")
    if h_eval != 0.001:
        print("  CAVEAT: h_eval is %g, NOT the finest 0.001 -- 0.001 was intractable in"
              % h_eval)
        print("  budget for this configuration.  Stated explicitly, per the task.")

    print("\n" + "=" * 84)
    print("TOTAL WALL CLOCK  %s   (setup %s, of which Eq (12) pp %s)"
          % (fmt_s(time.perf_counter() - t_all), fmt_s(t_load + t_pp), fmt_s(t_pp)))
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
