#!/usr/bin/env python3
"""Is the delivered seed set even a 1-swap local optimum?  (Non-paper probe.)

Both CTIM Algorithm 2 and plain MIA greedy stop at a set carrying only the
``(1 - 1/e)`` submodular-greedy bound.  Nothing in this repository has yet asked
whether that set can be improved by simply *replacing* one seed -- the cheapest
possible post-processing step, and the one greedy structurally cannot perform
because it never revisits a committed seed.

Everything here is scored by the **same** Eq (18) evaluator, built from the
**same** Eq (12) ``pp`` at the **same** ``h``, on the **same** cached model and
the same test item.  Nothing is retrained.  So every MIA number below is a
fully clean same-pp / same-h / same-item comparison, directly comparable with
the rest of ``results/``.  MC-IC (``ic_simulate``) is reported alongside as a
neutral referee, because a single-hop MIA at ``h = 0.1`` is not a cascade.

Expected outcome, stated before the run: with 98 of 100 community subgraphs
edgeless and MIA provably single-hop at ``h = 0.1`` (``pp <= 1/Z = 0.125``, so
any 2-hop path is ``<= 0.0154 < h``), Eq (18) is close to additive, and greedy
is near-optimal on additive objectives.  A 0.00% gain is the likely and
*informative* result -- it upgrades "has a 1-1/e bound" to "is a certified local
optimum".  The script therefore also reports the seed **overlap** between the
start and the end set, because "same value, different set" and "identical set"
are different findings.

Standard library only.  Python 3.9 compatible.  All randomness flows through
explicit `random.Random` instances.
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

from ctim.ctim import ctim_select_seeds                       # noqa: E402
from ctim.dataset import load_dataset                         # noqa: E402
from ctim.influence import MIA, EdgeWeights                    # noqa: E402
from ctim.local_search import (local_search, solo_influence_all,  # noqa: E402
                               top_candidates)
from ctim.ris_imm import ic_simulate                           # noqa: E402


def fmt(s):
    return "%.2fs" % s


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    p.add_argument("--Ks", default="20,50", help="comma separated budgets")
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--candidates", type=int, default=500,
                   help="size of the top-N-by-I({w}) CONTROL pool")
    p.add_argument("--strategy", default="best", choices=("best", "first"))
    p.add_argument("--max-rounds", type=int, default=25)
    p.add_argument("--mc", type=int, default=1000,
                   help="Monte-Carlo IC simulations for the neutral referee")
    p.add_argument("--no-control", action="store_true",
                   help="skip the top-N control run that audits the heuristic prune")
    args = p.parse_args(argv)

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    t_all = time.perf_counter()

    print("=" * 84)
    print("LEVER 4 -- 1-swap local search on the final seed set")
    print("K=%s  h=%g  pool=ALL of V (exact bound-(*) prune)  control=top-%d "
          "by I({w})  strategy=%s  seed=%d  mc=%d"
          % (Ks, args.h, args.candidates, args.strategy, args.seed, args.mc))
    print("=" * 84)

    if not os.path.exists(args.cache):
        print("ERROR: no cached model at %s" % args.cache)
        return 1

    # ------------------------------------------------------------ setup
    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model, item = blob["model"], blob["test_items"][0]
    ew = EdgeWeights(model, ds)
    t_load = time.perf_counter() - t0
    print("[setup] dataset + cached model (READ ONLY)      %s" % fmt(t_load))

    t0 = time.perf_counter()
    pp = ew.for_item(item)                                     # Eq (12)
    t_pp = time.perf_counter() - t0
    print("[setup] Eq (12) weights, item %-6d            %s  (|pp|=%d)"
          % (item, fmt(t_pp), len(pp)))

    t0 = time.perf_counter()
    evaluator = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)  # Eq (13)-(18)
    print("[setup] shared Eq (18) evaluator                %s" % fmt(time.perf_counter() - t0))

    # Candidate ranking: I({w}) in closed form, one forward Dijkstra per node.
    t0 = time.perf_counter()
    solo = solo_influence_all(evaluator)                       # Eq (18), |S|=1
    t_solo = time.perf_counter() - t0
    ranked = sorted(solo, key=lambda w: (-solo[w], w))
    print("[setup] I({w}) for all %d nodes (closed form)  %s"
          % (len(solo), fmt(t_solo)))

    # Audit the closed form against the real evaluator on the nodes that matter.
    rng_audit = random.Random(args.seed)
    worst = 0.0
    for w in rng_audit.sample(ranked[:400], 40):
        worst = max(worst, abs(evaluator.influence({w}) - solo[w]))
    print("        closed-form vs MIA.influence({w}) on 40 top nodes: "
          "max abs err %.2e" % worst)
    print("        I({w}) at rank 1/10/100/%d = %.4f / %.4f / %.4f / %.4f"
          % (args.candidates, solo[ranked[0]], solo[ranked[9]], solo[ranked[99]],
             solo[ranked[min(args.candidates, len(ranked)) - 1]]))

    # PRIMARY pool = every node of V.  That is affordable only because of the
    # exact bound-(*) prune inside `local_search`, which collapses 30358
    # candidates to the few dozen with I({w}) > min_u loss_u; the heuristic
    # "top-N by single-node influence" pool is kept only as a control, to show
    # what the heuristic costs when N is chosen badly.
    ctrl_pool = top_candidates(evaluator, args.candidates, solo=solo)
    cutoff_solo = solo[ranked[min(args.candidates, len(ranked)) - 1]]
    all_nodes = list(evaluator.nodes)

    rows = []
    seed_sets = []

    for K in Ks:
        print("\n" + "-" * 84)
        print("K = %d" % K)
        print("-" * 84)

        starts = []
        t0 = time.perf_counter()
        s_ctim = ctim_select_seeds(model, ds, item, K, h=args.h,
                                   dp_tiebreak="paper-true", edge_weights=ew)
        starts.append(("CTIM(Alg2)", list(s_ctim), time.perf_counter() - t0))

        t0 = time.perf_counter()
        s_gg = evaluator.greedy_incremental(K)
        starts.append(("GlobalGreedy", list(s_gg), time.perf_counter() - t0))

        for (name, S0, t_sel) in starts:
            print("\n  start = %s   (selection %s)" % (name, fmt(t_sel)))

            t0 = time.perf_counter()
            S1, val, st = local_search(evaluator, S0, all_nodes,
                                       random.Random(args.seed),
                                       max_rounds=args.max_rounds,
                                       strategy=args.strategy, solo=solo)
            t_ls = time.perf_counter() - t0

            start_val = st["start_value"]
            gain = val - start_val
            pct = (100.0 * gain / start_val) if start_val > 0 else 0.0
            overlap = len(set(S0) & set(S1))

            # The pool IS V, so a clean pass certifies a 1-swap local optimum
            # over the whole node set with nothing assumed about the pool.
            complete_V = True

            print("    start  I_h(S) = %14.6f" % start_val)
            print("    final  I_h(S) = %14.6f   gain %+0.6f  (%+0.4f%%)"
                  % (val, gain, pct))
            print("    swaps accepted = %d   rounds = %d   seed overlap = %d/%d"
                  % (st["swaps_accepted"], st["rounds"], overlap, len(S0)))
            for sw in st["swaps"]:
                print("        round %d: out %d (I({u})=%.4f) -> in %d "
                      "(I({w})=%.4f)  delta %+0.6f"
                      % (sw["round"], sw["out"], solo.get(sw["out"], float("nan")),
                         sw["in"], solo.get(sw["in"], float("nan")), sw["delta"]))
            print("    influence() calls = %d   ap()/Eq (17) tree evals = %d"
                  % (st["influence_calls"], st["ap_calls"]))
            print("    swap pairs evaluated = %d   pruned by bound (*) = %d"
                  " (naive neighbourhood would be %d)"
                  % (st["pairs_evaluated"], st["pairs_pruned"],
                     st["naive_influence_calls"]))
            print("    live candidates after prune = %d / %d   min_u loss_u = %.6f"
                  % (st["live_candidates"], st["candidates"], st["final_min_loss"]))
            print("    corr in [%.3e, %.3e]  bound (*) violations = %d"
                  % (st["min_corr"], st["max_corr"], st["bound_violations"]))
            print("    CERTIFIED 1-swap local optimum over ALL %d nodes of V: "
                  "%s  (%s)"
                  % (len(all_nodes), st["certified"], st["certificate_reason"]))
            print("    LOCAL SEARCH WALL CLOCK = %s" % fmt(t_ls))

            rows.append((K, name, start_val, val, gain, pct,
                         st["swaps_accepted"], overlap, len(S0),
                         st["influence_calls"], st["ap_calls"], t_ls,
                         st["certified"] and complete_V))
            seed_sets.append((K, name, list(S0), list(S1)))

            # ---- control: the heuristic top-N pool instead of all of V -----
            if not args.no_control:
                t0 = time.perf_counter()
                S2, val2, st2 = local_search(evaluator, S0, ctrl_pool,
                                             random.Random(args.seed),
                                             max_rounds=args.max_rounds,
                                             strategy=args.strategy, solo=solo)
                same = abs(val2 - val) <= 1e-9
                sound = cutoff_solo <= st2["final_min_loss"]
                print("    [control] heuristic pool = top-%d by I({w}): "
                      "I_h = %.6f (%s), %d swap(s), %s"
                      % (args.candidates, val2,
                         "SAME as all-of-V" if same
                         else "LOSES %.6f vs all-of-V" % (val - val2),
                         st2["swaps_accepted"], fmt(time.perf_counter() - t0)))
                print("              pool cutoff I({w})=%.6f %s min_u loss_u="
                      "%.6f, so the top-N restriction is %s here"
                      % (cutoff_solo, "<=" if sound else ">",
                         st2["final_min_loss"],
                         "provably lossless" if sound
                         else "NOT provably lossless"))

    # ------------------------------------------------------------ MC-IC
    print("\n" + "=" * 84)
    print("NEUTRAL REFEREE -- Monte-Carlo IC, %d simulations, seeded" % args.mc)
    print("=" * 84)
    t0 = time.perf_counter()
    mc = {}
    for (K, name, S0, S1) in seed_sets:
        mc[(K, name, "before")] = ic_simulate(ds.out_adj, pp, S0, args.mc,
                                              random.Random(args.seed + 1))
        mc[(K, name, "after")] = ic_simulate(ds.out_adj, pp, S1, args.mc,
                                             random.Random(args.seed + 1))
    t_mc = time.perf_counter() - t0
    print("%-4s %-14s %13s %13s %10s   %13s %13s %10s"
          % ("K", "start", "MIA before", "MIA after", "MIA %",
             "MC-IC before", "MC-IC after", "MC-IC %"))
    for (K, name, s0, s1, g, pct, nsw, ov, ns, nic, nap, tls, cert) in rows:
        b = mc[(K, name, "before")]
        a = mc[(K, name, "after")]
        mpct = (100.0 * (a - b) / b) if b > 0 else 0.0
        print("%-4d %-14s %13.6f %13.6f %+9.4f%%   %13.4f %13.4f %+9.4f%%"
              % (K, name, s0, s1, pct, b, a, mpct))
    print("(MC-IC uses the identical random.Random(%d) stream for before/after,"
          " so the two are paired.)  MC wall clock %s" % (args.seed + 1, fmt(t_mc)))

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 84)
    print("SUMMARY -- same pp (Eq (12), item %d), same h=%g, same Eq (18) "
          "evaluator, model NOT retrained" % (item, args.h))
    print("=" * 84)
    print("%-4s %-14s %13s %13s %10s %7s %9s %10s %9s %8s"
          % ("K", "start", "I_h before", "I_h after", "gain %", "swaps",
             "overlap", "infl calls", "ap evals", "search"))
    for (K, name, s0, s1, g, pct, nsw, ov, ns, nic, nap, tls, cert) in rows:
        print("%-4d %-14s %13.6f %13.6f %+9.4f%% %7d %9s %10d %9d %8s"
              % (K, name, s0, s1, pct, nsw, "%d/%d" % (ov, ns), nic, nap,
                 fmt(tls)))
    print("")
    for (K, name, s0, s1, g, pct, nsw, ov, ns, nic, nap, tls, cert) in rows:
        print("  K=%-3d %-14s certified 1-swap local optimum over ALL of V: %s"
              % (K, name, cert))
    gg = {K: s1 for (K, name, s0, s1, g, pct, nsw, ov, ns, nic, nap, tls, cert)
          in rows if name == "GlobalGreedy"}
    print("")
    for (K, name, s0, s1, g, pct, nsw, ov, ns, nic, nap, tls, cert) in rows:
        if name == "CTIM(Alg2)" and K in gg:
            closed = (100.0 * (s1 - s0) / (gg[K] - s0)) if gg[K] > s0 else float("nan")
            print("  K=%-3d CTIM -> local optimum closes %.2f%% of the "
                  "CTIM-to-GlobalGreedy gap (%.6f -> %.6f, GG = %.6f)"
                  % (K, closed, s0, s1, gg[K]))

    # "same value, different set" and "identical set" are different findings.
    print("")
    final_by = {(K, name): S1 for (K, name, S0, S1) in seed_sets}
    for K in Ks:
        a = final_by.get((K, "CTIM(Alg2)"))
        b = final_by.get((K, "GlobalGreedy"))
        if a is None or b is None:
            continue
        print("  K=%-3d CTIM's local optimum vs GlobalGreedy's local optimum: "
              "%d/%d seeds shared -> the sets are %s"
              % (K, len(set(a) & set(b)), K,
                 "IDENTICAL" if set(a) == set(b)
                 else "DIFFERENT despite scoring equally"))
    for (K, name, S0, S1) in seed_sets:
        print("  K=%-3d %-14s final S = %s" % (K, name, sorted(S1)))

    print("\nTOTAL WALL CLOCK = %s" % fmt(time.perf_counter() - t_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
