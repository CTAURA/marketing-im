#!/usr/bin/env python3
"""Measure how much influence Algorithm 2 line 34 cannot see.

Why this script exists
----------------------
Algorithm 2 line 34 evaluates a candidate's marginal gain as

    dI_m = max_u ( I_m(S u {u}) - I_m(S) ),   u in c_m

where ``I_m`` is Eq (18) on community m's **node-induced subgraph** -- only arcs
with *both* endpoints inside c_m survive (``ctim.influence.MIA(..., nodes=...)``,
and ``ctim/ctim.py``'s own reading note for line 34).  Every unit of influence a
node pushes across a community boundary is therefore invisible to the selector.
A *bridge* node -- mediocre inside its own community, strong globally -- is
structurally undervalued, and no amount of budget re-allocation can rescue it,
because it never wins its own community's argmax.

That is a hypothesis about a mechanism, and so far it has only been *inferred*
(from the Digg-vs-Yelp gap: 2 live communities and +0.33% on Digg, 15 live
communities and +3.82% on Yelp).  This script measures it directly, on one
fixed ``(pp, h, item)``, read-only, from the cached model.

What is and is not compared here
-------------------------------
Everything below is ``I_m`` versus ``I_global`` for the **same node**, under the
**same** ``pp``, the **same** ``h`` and the **same** item.  The two differ only
in the graph they run on, and community m's induced subgraph is a *subgraph* of
the full graph, so ``I_m({u}) <= I_global({u})`` always and the ratio is a
well-defined "fraction visible to line 34".  This is a decomposition, not a
cross-method score, so the metric rule (never grade two *methods* at the
selection ``h``) is not at stake: no seed set is being declared better than
another anywhere in this script.  Section 4 does put CTIM's and GlobalGreedy's
seed sets side by side, but only to look up *ranks*, never to compare spreads.

The four measurements
---------------------
1. For each seed CTIM picks: ``I_m({u})`` vs ``I_global({u})``.  The mean ratio
   is the fraction of real influence line 34 can see for the nodes it actually
   chose.
2. For the top-N nodes by *global* single-node spread: their rank by ``I_m``
   inside their own community vs their global rank.  Globally-top / locally-
   buried nodes are the bridge nodes line 34 will never reach.
3. Arc accounting: what fraction of live arcs (``pp >= h``) cross a community
   boundary, and the influence-weighted version -- how much ``ap(v|{u})`` mass
   lands outside u's own community.
4. The discriminator.  For every GlobalGreedy K-seed that CTIM did not pick,
   two competing explanations are separated:
     * **BLINDNESS** -- the node is top-ranked in its community by
       ``I_global`` but demoted by ``I_m``.  Line 34's induced subgraph is
       hiding it; no budget change can fix that.
     * **ALLOCATION** -- the node is top-ranked by ``I_m`` too, but its
       community was never handed a seed.  Line 34 sees it fine; the DP of
       lines 32-41 simply spent the budget elsewhere.  A different fix.
   The two are distinguished by comparing the node's within-community rank
   under ``I_m`` with its within-community rank under ``I_global``.

Standard library only.  Python 3.9 compatible.  Fully deterministic: no RNG is
used at all (ranks break ties by node id).  Nothing is written; the cached model
is opened read-only.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.ctim import ctim_select_seeds, detect_communities   # noqa: E402
from ctim.dataset import load_dataset                          # noqa: E402
from ctim.influence import MIA, EdgeWeights                    # noqa: E402


def fmt(s):
    return "%.2fs" % s


def mean(xs):
    xs = list(xs)
    return (sum(xs) / float(len(xs))) if xs else 0.0


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return 0.0
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ---------------------------------------------------------------------------
# single-node spread, split by community
# ---------------------------------------------------------------------------


def solo_split(mia, w, comm):
    """``(I({w}), mass landing outside comm[w])``, both from Eq (18).

    With ``S = {w}`` the Eq (17) recursion collapses -- inside every ``MIIA(v,h)``
    only the branch to ``w`` carries probability -- so ``ap(v|{w})`` is the
    product of ``pp`` along ``MIP(w,v)``, i.e. exactly ``pp(MIP(w,v))`` of
    Eq (13)/(14), and

        I({w}) = sum_{v in MIOA(w,h)} pp(MIP(w,v))                    # Eq (18)

    One forward Dijkstra per node, no MIIA construction; this is the same closed
    form as ``ctim.local_search.solo_influence`` (checked against
    ``MIA.influence`` in the setup below), extended to also return the part of
    the sum whose target sits in a different community.
    """
    nodes, parent = mia.mioa(w)  # Eq (16)
    pp = mia.pp
    prob = {w: 1.0}
    for x in nodes:
        if x in prob:
            continue
        chain = []
        y = x
        while y not in prob:
            chain.append(y)
            y = parent[y]
        val = prob[y]
        for z in reversed(chain):
            val *= pp[(parent[z], z)]  # Eq (13): pp(P) is a product along P
            prob[z] = val
    cw = comm[w]
    tot = 0.0
    out = 0.0
    for x, p in prob.items():
        tot += p
        if comm[x] != cw:
            out += p
    return tot, out


def ranks_desc(values, keys):
    """1-based rank of each key by descending value, ties broken by key."""
    order = sorted(keys, key=lambda k: (-values[k], k))
    return {k: i + 1 for i, k in enumerate(order)}


# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",
                   default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache",
                   default=os.path.join(_REPO_ROOT, "data", "processed",
                                        "_calib_model.pkl"))
    p.add_argument("--K", type=int, default=20,
                   help="the K used for the CTIM/GlobalGreedy counterfactual")
    p.add_argument("--K2", type=int, default=50,
                   help="a second K for section 1 only (0 disables)")
    p.add_argument("--h", type=float, default=0.1,
                   help="MIA threshold, Eq (15)/(16); the SAME h everywhere")
    p.add_argument("--item", type=int, default=-1,
                   help="item id for Eq (12); default = cache test_items[0]")
    p.add_argument("--top", type=int, default=50,
                   help="how many globally-strongest nodes section 2 examines")
    p.add_argument("--skip-gg", action="store_true",
                   help="skip GlobalGreedy (section 4 is then unavailable)")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("=" * 78)
    print("BLINDNESS DIAGNOSTIC -- how much influence does Algorithm 2 line 34 see?")
    print("dataset=%s  K=%d  K2=%d  h=%g" % (args.dataset, args.K, args.K2, args.h))
    print("=" * 78)

    if not os.path.exists(args.cache):
        print("ERROR: no cached model at %s -- run scripts/calibrate_h.py first."
              % args.cache)
        return 1

    # ------------------------------------------------------------------ setup
    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model = blob["model"]
    item = blob["test_items"][0] if args.item < 0 else args.item
    ew = EdgeWeights(model, ds)
    print("[setup] dataset + cached model                 %s"
          % fmt(time.perf_counter() - t0))
    print("        U=%d  E=%d  C=%d  item=%d"
          % (ds.n_users, len(ds.edges), len(model.eta), item))

    t0 = time.perf_counter()
    pp = ew.for_item(item)                    # Eq (12) -- built ONCE, reused
    print("[setup] Eq (12) weights                        %s"
          % fmt(time.perf_counter() - t0))

    t0 = time.perf_counter()
    comm = detect_communities(model.pi)       # Eq (19), Algorithm 2 lines 22-24
    members = {}
    for v in range(ds.n_users):
        members.setdefault(comm[v], []).append(v)
    n_nonempty = len(members)
    sizes = sorted((len(m) for m in members.values()), reverse=True)
    print("[setup] Eq (19) partition                      %s"
          % fmt(time.perf_counter() - t0))
    print("        %d non-empty communities of %d; sizes: max=%d median=%d min=%d"
          % (n_nonempty, len(model.eta), sizes[0], median(sizes), sizes[-1]))

    t0 = time.perf_counter()
    mia_g = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)
    print("[setup] full-graph Eq (18) evaluator           %s"
          % fmt(time.perf_counter() - t0))

    # I_global({u}) and the out-of-community part of its ap mass, for every node.
    t0 = time.perf_counter()
    I_glob = {}
    out_mass = {}
    for u in range(ds.n_users):
        tot, out = solo_split(mia_g, u, comm)
        I_glob[u] = tot
        out_mass[u] = out
    t_solo_g = time.perf_counter() - t0
    print("[setup] I_global({u}) for all %d nodes        %s"
          % (ds.n_users, fmt(t_solo_g)))

    # closed form vs MIA.influence, on the 5 strongest nodes -- a real check
    probe = sorted(I_glob, key=lambda u: (-I_glob[u], u))[:5]
    worst = max(abs(I_glob[u] - mia_g.influence({u})) for u in probe)
    print("        closed form vs MIA.influence on top-5: max abs err %.3e %s"
          % (worst, "OK" if worst <= 1e-9 else "** MISMATCH **"))

    # per-community induced subgraphs (exactly what Algorithm 2 line 34 builds)
    t0 = time.perf_counter()
    mia_m = {}
    I_loc = {}
    for c, mem in members.items():
        # I_m = Eq (18) on community c's node-induced subgraph (line 34)
        m = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h, nodes=mem)
        mia_m[c] = m
        for u in mem:
            I_loc[u], _z = solo_split(m, u, comm)
    t_solo_m = time.perf_counter() - t0
    print("[setup] %d induced subgraphs + I_m({u})        %s"
          % (n_nonempty, fmt(t_solo_m)))

    bad = [u for u in range(ds.n_users) if I_loc[u] > I_glob[u] + 1e-9]
    print("        sanity: I_m({u}) <= I_global({u}) for all u: %s"
          % ("OK" if not bad else "** VIOLATED on %d nodes **" % len(bad)))

    # ranks
    t0 = time.perf_counter()
    rank_glob = ranks_desc(I_glob, range(ds.n_users))
    rank_in_c_by_Im = {}
    rank_in_c_by_Ig = {}
    for c, mem in members.items():
        rank_in_c_by_Im.update(ranks_desc(I_loc, mem))
        rank_in_c_by_Ig.update(ranks_desc(I_glob, mem))
    print("[setup] rankings                               %s"
          % fmt(time.perf_counter() - t0))

    # ------------------------------------------------------- CTIM seed sets
    ctim_seeds = {}
    for K in [k for k in (args.K, args.K2) if k and k > 0]:
        t0 = time.perf_counter()
        s = ctim_select_seeds(model, ds, item, K, h=args.h,
                              dp_tiebreak="paper-true", edge_weights=ew)
        ctim_seeds[K] = s
        print("[run]   CTIM K=%-3d                             %s"
              % (K, fmt(time.perf_counter() - t0)))

    # =====================================================================  1
    print("")
    print("=" * 78)
    print("(1) WHAT LINE 34 SEES OF THE SEEDS IT PICKED")
    print("    I_m = Eq (18) on the seed's own community's induced subgraph;")
    print("    I_global = Eq (18) on the full graph.  Same pp, same h, same item.")
    print("=" * 78)
    for K in sorted(ctim_seeds):
        S = ctim_seeds[K]
        print("\n--- CTIM K=%d ---" % K)
        print("  %-4s %-8s %-6s %-7s %11s %11s %8s  %s"
              % ("#", "node", "comm", "|comm|", "I_m", "I_global", "ratio",
                 "rank_in_c(I_m)"))
        ratios = []
        for idx, u in enumerate(S, 1):
            c = comm[u]
            r = I_loc[u] / I_glob[u] if I_glob[u] > 0 else float("nan")
            ratios.append(r)
            print("  %-4d %-8d %-6d %-7d %11.4f %11.4f %7.2f%%  %d/%d"
                  % (idx, u, c, len(members[c]), I_loc[u], I_glob[u],
                     100.0 * r, rank_in_c_by_Im[u], len(members[c])))
        print("  MEAN RATIO  = %.4f%%   (the fraction of influence Algorithm 2 sees)"
              % (100.0 * mean(ratios)))
        print("  MEDIAN      = %.4f%%     MIN = %.4f%%   MAX = %.4f%%"
              % (100.0 * median(ratios), 100.0 * min(ratios), 100.0 * max(ratios)))
        print("  sum I_m = %.3f vs sum I_global = %.3f  (aggregate visible: %.4f%%)"
              % (sum(I_loc[u] for u in S), sum(I_glob[u] for u in S),
                 100.0 * sum(I_loc[u] for u in S) / max(sum(I_glob[u] for u in S), 1e-12)))
        alloc = {}
        for u in S:
            alloc[comm[u]] = alloc.get(comm[u], 0) + 1
        print("  allocation over communities: %s  (%d distinct)"
              % (dict(sorted(alloc.items())), len(alloc)))
        firsts_ok = True
        seen = set()
        for u in S:
            if comm[u] not in seen:
                seen.add(comm[u])
                if rank_in_c_by_Im[u] != 1:
                    firsts_ok = False
        print("  check: each community's FIRST CTIM seed is its I_m-rank-1 node: %s"
              % ("yes" if firsts_ok else "NO (submodular re-ordering)"))

    # =====================================================================  2
    print("")
    print("=" * 78)
    print("(2) THE GLOBALLY STRONGEST NODES, SEEN FROM INSIDE THEIR COMMUNITY")
    print("    top-%d by I_global; is line 34 even aware they are competitive?"
          % args.top)
    print("=" * 78)
    top = sorted(I_glob, key=lambda u: (-I_glob[u], u))[: args.top]
    in_ctim = set()
    for S in ctim_seeds.values():
        in_ctim |= set(S)
    print("  %-6s %-8s %-6s %-7s %11s %11s %8s %10s %10s %s"
          % ("g.rank", "node", "comm", "|comm|", "I_global", "I_m", "ratio",
             "rk_c(I_m)", "rk_c(I_g)", "picked"))
    disp = []
    shift = []
    n_rank1 = n_le3 = n_le10 = n_gt50 = 0
    for u in top:
        c = comm[u]
        rm = rank_in_c_by_Im[u]
        rg = rank_in_c_by_Ig[u]
        r = I_loc[u] / I_glob[u] if I_glob[u] > 0 else float("nan")
        disp.append(rm)
        shift.append(rm - rg)
        if rm == 1:
            n_rank1 += 1
        if rm <= 3:
            n_le3 += 1
        if rm <= 10:
            n_le10 += 1
        if rm > 50:
            n_gt50 += 1
        picked = ",".join("K=%d" % K for K in sorted(ctim_seeds)
                          if u in ctim_seeds[K]) or "-"
        print("  %-6d %-8d %-6d %-7d %11.4f %11.4f %7.2f%% %10d %10d %s"
              % (rank_glob[u], u, c, len(members[c]), I_glob[u], I_loc[u],
                 100.0 * r, rm, rg, picked))
    print("\n  of the top-%d global nodes, their rank INSIDE their own community"
          " by I_m:" % args.top)
    print("    rank 1 : %d/%d (%.1f%%)   <=3 : %d   <=10 : %d   >50 : %d"
          % (n_rank1, len(top), 100.0 * n_rank1 / len(top), n_le3, n_le10, n_gt50))
    print("    mean within-community I_m rank = %.2f, median = %.1f, max = %d"
          % (mean(disp), median(disp), max(disp)))
    print("    (a node that is globally top-%d but ranks deep inside its own"
          % args.top)
    print("     community is a bridge node line 34 can never reach)")
    # Attenuation vs re-ordering.  Line 34 deleting cross-community arcs shrinks
    # every I_m; that alone changes no decision.  It only costs a seed if it
    # also RE-ORDERS nodes inside a community, i.e. if a node's rank under I_m
    # differs from its rank under I_global *within the same community*.
    dem = [d for d in shift if d > 0]
    pro = [d for d in shift if d < 0]
    print("    re-ordering vs attenuation -- rank INSIDE the same community,")
    print("      by I_m vs by I_global: mean shift %+.2f places, max demotion %d,"
          % (mean(shift), max(shift) if shift else 0))
    print("      demoted %d, promoted %d, unchanged %d of %d"
          % (len(dem), len(pro), len(top) - len(dem) - len(pro), len(top)))
    print("      (a large ratio drop with ~zero rank shift = pure attenuation:")
    print("       line 34 sees less, but still picks the same nodes)")
    for K in sorted(ctim_seeds):
        print("    CTIM K=%d picked %d of these %d nodes"
              % (K, len(set(top) & set(ctim_seeds[K])), len(top)))

    # =====================================================================  3
    print("")
    print("=" * 78)
    print("(3) ARC ACCOUNTING -- how much of the graph the induced subgraphs delete")
    print("=" * 78)
    t0 = time.perf_counter()
    n_arcs = len(pp)
    n_live = 0
    n_cross = 0
    n_live_cross = 0
    for (u, v), w in pp.items():
        cross = comm[u] != comm[v]
        if cross:
            n_cross += 1
        if w >= args.h:
            n_live += 1
            if cross:
                n_live_cross += 1
    print("  all arcs        : %d   cross-community %d (%.2f%%)  retained %.2f%%"
          % (n_arcs, n_cross, 100.0 * n_cross / n_arcs,
             100.0 * (n_arcs - n_cross) / n_arcs))
    print("  live arcs pp>=h : %d (%.2f%% of arcs)   cross-community %d (%.2f%%)"
          % (n_live, 100.0 * n_live / n_arcs, n_live_cross,
             100.0 * n_live_cross / max(n_live, 1)))
    print("  -> %.2f%% of LIVE arcs are cut by the induced subgraphs of line 34"
          % (100.0 * n_live_cross / max(n_live, 1)))
    n_intra_live = {}
    for (u, v), w in pp.items():
        if w >= args.h and comm[u] == comm[v]:
            n_intra_live[comm[u]] = n_intra_live.get(comm[u], 0) + 1
    print("  communities with at least one live intra arc: %d of %d"
          % (len(n_intra_live), n_nonempty))
    print("  (arc counting %s)" % fmt(time.perf_counter() - t0))

    tot_all = sum(I_glob.values())
    out_all = sum(out_mass.values())
    print("\n  influence-weighted version -- sum over v of ap(v|{u}), Eq (17):")
    print("    over ALL %d nodes u: %.2f%% of the ap mass lands OUTSIDE comm[u]"
          % (ds.n_users, 100.0 * out_all / max(tot_all, 1e-12)))
    nonself_tot = tot_all - float(ds.n_users)
    nonself_out = out_all
    print("    excluding the seed's own unit of self-mass: %.2f%% of the"
          " remaining %.1f units" % (100.0 * nonself_out / max(nonself_tot, 1e-12),
                                     nonself_tot))
    for K in sorted(ctim_seeds):
        S = ctim_seeds[K]
        t = sum(I_glob[u] for u in S)
        o = sum(out_mass[u] for u in S)
        print("    CTIM K=%d seeds      : %.2f%% of their ap mass lands outside"
              " their own community" % (K, 100.0 * o / max(t, 1e-12)))
    t = sum(I_glob[u] for u in top)
    o = sum(out_mass[u] for u in top)
    print("    top-%d global nodes : %.2f%% of their ap mass lands outside"
          " their own community" % (args.top, 100.0 * o / max(t, 1e-12)))

    # =====================================================================  4
    if args.skip_gg:
        print("\n(4) SKIPPED (--skip-gg)")
        print("\n" + "=" * 78)
        print("TOTAL %s" % fmt(time.perf_counter() - t_all))
        return 0

    print("")
    print("=" * 78)
    print("(4) THE COUNTERFACTUAL -- BLINDNESS or ALLOCATION?")
    print("=" * 78)
    t0 = time.perf_counter()
    gg = mia_g.greedy_incremental(args.K)
    t_gg = time.perf_counter() - t0
    print("  GlobalGreedy K=%d selected in %s" % (args.K, fmt(t_gg)))

    S_ctim = ctim_seeds[args.K]
    set_ctim = set(S_ctim)
    alloc = {}
    for u in S_ctim:
        alloc[comm[u]] = alloc.get(comm[u], 0) + 1
    missed = [u for u in gg if u not in set_ctim]
    print("  seed overlap CTIM n GlobalGreedy: %d/%d;  %d GlobalGreedy seeds"
          " CTIM did NOT pick" % (len(set_ctim & set(gg)), args.K, len(missed)))
    print("  CTIM's allocation at K=%d: %s" % (args.K, dict(sorted(alloc.items()))))

    print("\n  for each GlobalGreedy seed CTIM missed:")
    print("  %-8s %-6s %-7s %-7s %11s %11s %7s %9s %9s %-7s %s"
          % ("node", "comm", "|comm|", "budget", "I_global", "I_m", "ratio",
             "rk_c(I_m)", "rk_c(I_g)", "g.rank", "verdict"))
    n_blind = 0
    n_alloc = 0
    n_both = 0
    n_other = 0
    for u in missed:
        c = comm[u]
        b = alloc.get(c, 0)
        rm = rank_in_c_by_Im[u]
        rg = rank_in_c_by_Ig[u]
        ratio = I_loc[u] / I_glob[u] if I_glob[u] > 0 else float("nan")
        # A node is reachable by line 34 only if its own community ranks it at
        # the top under I_m.  Reading:
        #   rk_c(I_g) == 1 and rk_c(I_m) > 1  -> the induced subgraph DEMOTED it
        #                                        => BLINDNESS (budget cannot help)
        #   rk_c(I_m) == 1 and budget == 0    -> line 34 sees it perfectly well,
        #                                        the DP just never funded c
        #                                        => ALLOCATION
        demoted = rm > rg
        top_local = (rm == 1)
        if top_local and b == 0:
            verdict = "ALLOCATION"
            n_alloc += 1
        elif demoted and not top_local:
            verdict = "BLINDNESS"
            n_blind += 1
        elif top_local and b > 0:
            verdict = "overlap"  # seen, funded, but submodular overlap ruled it out
            n_other += 1
        elif not top_local and not demoted:
            # I_m ranks it no worse than I_global does inside its community, so
            # the induced subgraph did not hide it; it simply lost the greedy
            # race for that community's budget.
            verdict = "NEITHER"
            n_both += 1
        else:
            verdict = "other"
            n_other += 1
        print("  %-8d %-6d %-7d %-7d %11.4f %11.4f %6.2f%% %9d %9d %-7d %s"
              % (u, c, len(members[c]), b, I_glob[u], I_loc[u], 100.0 * ratio,
                 rm, rg, rank_glob[u], verdict))

    print("\n  VERDICT COUNTS over %d missed GlobalGreedy seeds:" % len(missed))
    print("    BLINDNESS  (I_m demotes a node its community ranks 1st globally) : %d"
          % n_blind)
    print("    ALLOCATION (I_m-rank-1 in its community, community got 0 budget) : %d"
          % n_alloc)
    print("    NEITHER    (I_m ranks it no worse than I_global does inside its")
    print("                community, and that community WAS funded -- it just")
    print("                lost the greedy race)                               : %d"
          % n_both)
    print("    other/overlap                                                    : %d"
          % n_other)
    if missed:
        rms = [rank_in_c_by_Im[u] for u in missed]
        rgs = [rank_in_c_by_Ig[u] for u in missed]
        print("    mean within-community rank: by I_m %.2f  vs by I_global %.2f"
              % (mean(rms), mean(rgs)))
        print("    mean visible fraction I_m/I_global of the missed seeds: %.2f%%"
              % (100.0 * mean([I_loc[u] / I_glob[u] for u in missed
                               if I_glob[u] > 0])))
        n_zero_budget = sum(1 for u in missed if alloc.get(comm[u], 0) == 0)
        print("    missed seeds whose community received NO CTIM budget: %d/%d"
              % (n_zero_budget, len(missed)))
        distinct_c = len(set(comm[u] for u in missed))
        print("    they live in %d distinct communities" % distinct_c)

    print("\n  HOW TO READ THIS:")
    print("    If BLINDNESS dominates, line 34's node-induced subgraph is the")
    print("    cause: the nodes GlobalGreedy wants are demoted by I_m and no")
    print("    re-allocation of the K budget can ever surface them.")
    print("    If ALLOCATION dominates, line 34 sees those nodes perfectly (they")
    print("    are their community's I_m argmax) and the DP of lines 32-41 simply")
    print("    never gave their community a seed -- a DIFFERENT fix, in the DP,")
    print("    not in the subgraph.")

    print("\n" + "=" * 78)
    print("TOTAL %s" % fmt(time.perf_counter() - t_all))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
