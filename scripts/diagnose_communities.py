#!/usr/bin/env python3
"""Why are 98 of CTIM's 100 communities edgeless at h = 0.1?

Measured earlier in this repo: under MIA Eq (18) at ``h = 0.1`` on Digg, 98 of
the 100 communities produced by Eq (19) have within-community curve
``I_m(j) = j`` exactly -- a seed placed there activates itself and nothing else.
Every selection-layer experiment (EA, exact allocation DP, brute force) then
reports a 0.00% gain, because there is nothing left in the community subgraphs
for a *selection* rule to be clever about.

Two very different causes produce that same symptom and they demand opposite
fixes:

  (a) **The partition is wrong.**  Eq (19) takes ``argmax_c pi[v][c]`` and pi is
      only 1.7% below uniform, so the labelling is close to a random partition of
      the vertex set.  A random partition into 100 parts cuts ~99% of edges, so
      the induced subgraphs would be edgeless *no matter how strong the
      propagation probabilities are*.  Fix: use a graph-aware detector.

  (b) **The graph is wrong.**  theta never leaves its Dirichlet prior, which
      forces ``pp(u,v) <= 1/Z = 0.125`` (ctim/influence.py line 228), so no
      2-hop path can clear ``h = 0.1``.  Then even a *perfect* partition yields
      subgraphs whose only surviving structure is single hops.  Fix: the model /
      the threshold, not the partition.

This script separates the two, at FIXED ``pp`` and FIXED ``h``.  No seed
selection happens here -- it is pure diagnosis, so every number below is
same-model, same-pp, same-h and directly comparable across the two partitions.

It compares exactly two labellings of the same graph:

  A) Eq (19)  -- ``ctim.ctim.detect_communities(model.pi)``
  B) CNM      -- ``ctim.baselines.air_cga.detect_communities_modularity``,
                 Clauset-Newman-Moore greedy modularity, which sees only the
                 friend graph and never the propagation probabilities.

The headline number is the **intra-community edge fraction**: of all arcs in
``ds.edges``, what fraction survive the partition (both endpoints in the same
community).  If CNM's fraction is high and its live-edge community count is
*still* ~0, cause (b) is proven and no amount of better community detection can
rescue the selection layer.

CNM on Digg costs ~3.5 minutes per value of C, so its labellings can be cached
to disk with ``--cnm-cache DIR``; ``--precompute C`` computes one labelling and
exits, which lets several values of C be filled in in parallel before the
reporting run.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random(seed)``.
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

from ctim.baselines.air_cga import (                      # noqa: E402
    detect_communities_modularity,
    modularity,
)
from ctim.ctim import detect_communities                  # noqa: E402
from ctim.dataset import load_dataset                     # noqa: E402
from ctim.influence import MIA, EdgeWeights               # noqa: E402

__all__ = [
    "community_members",
    "size_stats",
    "edge_stats",
    "top_community_spread",
    "main",
]


def fmt(s):
    return "%.2fs" % s


# ---------------------------------------------------------------------------
# Partition descriptors
# ---------------------------------------------------------------------------


def community_members(comm, n_users):
    """`members[c] -> sorted list of v with comm[v] == c`, empty labels dropped."""
    buckets = {}
    for v in range(n_users):
        buckets.setdefault(comm[v], []).append(v)
    return buckets


def size_stats(members):
    """(n_nonempty, min, median, max, [(c, size)] * 5 largest)."""
    sizes = sorted(len(m) for m in members.values())
    n = len(sizes)
    if n == 0:
        return 0, 0, 0.0, 0, []
    mid = n // 2
    median = float(sizes[mid]) if n % 2 else 0.5 * (sizes[mid - 1] + sizes[mid])
    largest = sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]
    return n, sizes[0], median, sizes[-1], [(c, len(m)) for c, m in largest]


def edge_stats(comm, edges, pp, h):
    """Cut/intra accounting for one labelling, over all arcs and over live arcs.

    An arc (u, v) is *live* when ``pp(u,v) >= h``: a single hop across it
    already clears the Eq (14) threshold, so it is the only kind of edge that
    can carry influence at all once Eq (15)/(16) prune paths below h.

    Returns a dict with the raw counts plus, per community, its intra-arc and
    live-intra-arc counts.
    """
    n_edges = 0
    n_intra = 0
    n_live = 0
    n_live_intra = 0
    intra_by_c = {}
    live_intra_by_c = {}
    for (u, v) in edges:
        n_edges += 1
        cu = comm[u]
        same = (cu == comm[v])
        if same:
            n_intra += 1
            intra_by_c[cu] = intra_by_c.get(cu, 0) + 1
        # Eq (14): a one-hop path survives the MIA threshold iff pp >= h.
        if pp.get((u, v), 0.0) >= h:
            n_live += 1
            if same:
                n_live_intra += 1
                live_intra_by_c[cu] = live_intra_by_c.get(cu, 0) + 1
    return {
        "n_edges": n_edges,
        "n_intra": n_intra,
        "n_live": n_live,
        "n_live_intra": n_live_intra,
        "intra_by_c": intra_by_c,
        "live_intra_by_c": live_intra_by_c,
        "n_comm_with_live": len(live_intra_by_c),
        "n_comm_with_any_intra": len(intra_by_c),
    }


def live_endpoint_concentration(comm, edges, pp, h, k=3):
    """Where do live arcs attach -- at the head or at the tail?

    Eq (12) gives ``pp(u,v) = sum_c a_u[c] * pi_v[c] * thetabar_i[c]`` and on
    this model thetabar collapses to the constant ``1/Z``, so ``pp >= h``
    requires ``sum_c a_u[c] * pi_v[c] >= h * Z``.  Since ``sum_c pi_v[c] = 1``
    and ``a_u[c] <= 1``, that needs a_u and pi_v to be peaked *on the same c*.

    So the question this answers is whether live-ness is a one-sided property
    (concentrated at heads only, or tails only) or a two-sided block: if the
    head histogram and the tail histogram concentrate on the *same* handful of
    communities in roughly equal proportion, the live arcs form a dense block
    among a shared set of peaked nodes, and how a partition splits that block
    is what decides whether any community keeps a working subgraph.
    """
    by_head = {}
    by_tail = {}
    for (u, v) in edges:
        if pp.get((u, v), 0.0) >= h:   # Eq (14) one-hop survival
            by_head[comm[v]] = by_head.get(comm[v], 0) + 1
            by_tail[comm[u]] = by_tail.get(comm[u], 0) + 1
    top_h = sorted(by_head.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    top_t = sorted(by_tail.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return by_head, by_tail, top_h, top_t


def top_by_live(stats, members, k=5):
    """The k communities holding the most live intra arcs (id, size) pairs.

    Ranking by size answers "where are the nodes"; ranking by live arcs answers
    "where can anything actually propagate", and on this graph the two lists are
    almost disjoint.
    """
    live = stats["live_intra_by_c"]
    order = sorted(live.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [(c, len(members[c])) for c, _n in order]


def top_community_spread(ds, pp, h, members, top, stats):
    """For each listed community: does one seed reach past itself?

    Builds Eq (13)-(18) MIA on the node-induced subgraph (``nodes=members``,
    both endpoints inside), takes the best single node under Eq (18), and
    reports ``I_m({best})``.  A value of exactly 1.0 is the ``I_m(j) = j``
    signature: the seed activates itself and no one else.
    """
    rows = []
    for c, size in top:
        t0 = time.perf_counter()
        mem = members[c]
        sub = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h, nodes=mem)
        best = sub.greedy_incremental(1)
        val = sub.influence(best) if best else 0.0   # Eq (18)
        rows.append({
            "c": c,
            "size": size,
            "intra": stats["intra_by_c"].get(c, 0),
            "live_intra": stats["live_intra_by_c"].get(c, 0),
            "best": best[0] if best else None,
            "influence": val,
            "seconds": time.perf_counter() - t0,
        })
    return rows


def report(label, comm, ds, pp, h, elapsed_detect, do_top=True):
    n_users = ds.n_users
    print("")
    print("=" * 76)
    print("PARTITION %s" % label)
    print("=" * 76)
    print("  detection wall clock                 %s" % fmt(elapsed_detect))

    t0 = time.perf_counter()
    members = community_members(comm, n_users)
    n_ne, smin, smed, smax, top = size_stats(members)
    t_size = time.perf_counter() - t0

    t0 = time.perf_counter()
    stats = edge_stats(comm, ds.edges, pp, h)
    t_edge = time.perf_counter() - t0

    t0 = time.perf_counter()
    q = modularity(comm, n_users, ds.edges)
    t_q = time.perf_counter() - t0

    print("")
    print("  (1) non-empty communities            %d" % n_ne)
    print("      sizes  min/median/max            %d / %.1f / %d" % (smin, smed, smax))
    print("      5 largest (id: size)             %s"
          % ", ".join("%d: %d" % (c, s) for c, s in top))
    print("      [size scan %s]" % fmt(t_size))

    ne = stats["n_edges"]
    nl = stats["n_live"]
    print("")
    print("  (2) INTRA-COMMUNITY EDGE FRACTION    %d / %d = %.4f%%   <-- headline"
          % (stats["n_intra"], ne, 100.0 * stats["n_intra"] / ne if ne else 0.0))
    print("  (3) same, live arcs only (pp >= %g)  %d / %d = %.4f%%"
          % (h, stats["n_live_intra"], nl,
             100.0 * stats["n_live_intra"] / nl if nl else 0.0))
    print("  (4) communities with >=1 LIVE intra arc   %d / %d"
          % (stats["n_comm_with_live"], n_ne))
    print("      communities with >=1 intra arc (any pp)  %d / %d"
          % (stats["n_comm_with_any_intra"], n_ne))
    print("  (5) Newman modularity Q              %+.6f   [%s]" % (q, fmt(t_q)))
    print("      [edge scan %s]" % fmt(t_edge))

    nl_tot = stats["n_live"]
    by_head, by_tail, top_h, top_t = live_endpoint_concentration(comm, ds.edges, pp, h)
    print("")
    print("  (7) where live arcs attach:  %d communities contain a live HEAD (v),"
          " %d a live TAIL (u)" % (len(by_head), len(by_tail)))
    print("      top-3 by head comm[v]                %s"
          % ", ".join("c%d: %d (%.1f%%)" % (c, n, 100.0 * n / nl_tot) for c, n in top_h))
    print("      top-3 by tail comm[u]                %s"
          % ", ".join("c%d: %d (%.1f%%)" % (c, n, 100.0 * n / nl_tot) for c, n in top_t))

    rows = []
    rows_live = []
    if do_top:
        print("")
        print("  (6) top-5 communities by size, MIA Eq (18) on the induced subgraph")
        print("      %-8s %-8s %-9s %-11s %-9s %-12s %s"
              % ("comm", "|c|", "intra", "live_intra", "best", "I_m({best})", "time"))
        rows = top_community_spread(ds, pp, h, members, top, stats)
        for r in rows:
            print("      %-8d %-8d %-9d %-11d %-9s %-12.6f %s"
                  % (r["c"], r["size"], r["intra"], r["live_intra"],
                     str(r["best"]), r["influence"], fmt(r["seconds"])))
        spread = sum(1 for r in rows if r["influence"] > 1.0 + 1e-12)
        print("      -> %d of %d top communities spread beyond the seed itself"
              % (spread, len(rows)))

        top_live = top_by_live(stats, members)
        print("")
        print("  (6b) top-5 communities by LIVE INTRA ARCS (where propagation can exist)")
        print("      %-8s %-8s %-9s %-11s %-9s %-12s %s"
              % ("comm", "|c|", "intra", "live_intra", "best", "I_m({best})", "time"))
        rows_live = top_community_spread(ds, pp, h, members, top_live, stats)
        for r in rows_live:
            print("      %-8d %-8d %-9d %-11d %-9s %-12.6f %s"
                  % (r["c"], r["size"], r["intra"], r["live_intra"],
                     str(r["best"]), r["influence"], fmt(r["seconds"])))

    return {"n_nonempty": n_ne, "stats": stats, "q": q, "top": top, "rows": rows,
            "rows_live": rows_live}


# ---------------------------------------------------------------------------
# CNM labelling cache (CNM costs minutes per C; the diagnosis itself is seconds)
# ---------------------------------------------------------------------------


def cnm_labelling(ds, C, seed, cache_dir):
    """Run CNM at target C, or load a previously cached labelling.

    Returns (comm, seconds, from_cache).  `seconds` is the wall clock of the
    run that actually produced the labelling, cached alongside it so the report
    quotes the real cost even on a cache hit.
    """
    path = None
    if cache_dir:
        path = os.path.join(cache_dir, "cnm_%s_C%d_s%d.pkl" % (ds.name, C, seed))
        if os.path.exists(path):
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
            return blob["comm"], blob["seconds"], True
    t0 = time.perf_counter()
    comm = detect_communities_modularity(ds, C=C, rng=random.Random(seed))
    secs = time.perf_counter() - t0
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"comm": comm, "seconds": secs, "C": C, "seed": seed}, fh)
    return comm, secs, False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"),
                   help="pickle of {'model':..., 'test_items':[...]} (read only)")
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=int, default=100, help="CNM target for the main comparison")
    p.add_argument("--sweep", default="10,20,50",
                   help="extra CNM targets for the sweep table (comma separated, '' to skip)")
    p.add_argument("--cnm-cache", default=None,
                   help="directory for cached CNM labellings (recommended: the scratchpad)")
    p.add_argument("--precompute", type=int, default=None,
                   help="only compute+cache the CNM labelling for this C, then exit")
    args = p.parse_args(argv)

    t_all = time.perf_counter()

    # ---- precompute mode: fill one cache slot and get out ------------------
    if args.precompute is not None:
        ds = load_dataset(args.dataset)
        comm, secs, hit = cnm_labelling(ds, args.precompute, args.seed, args.cnm_cache)
        print("[precompute] CNM C=%d -> %d communities in %s%s"
              % (args.precompute, len(set(comm)), fmt(secs), " (cache hit)" if hit else ""))
        print("TOTAL WALL CLOCK %s" % fmt(time.perf_counter() - t_all))
        return 0

    print("=" * 76)
    print("COMMUNITY DIAGNOSIS -- why are the community subgraphs edgeless?")
    print("h=%g  seed=%d  C=%d   (fixed pp, fixed h, fixed item: all numbers comparable)"
          % (args.h, args.seed, args.C))
    print("=" * 76)

    if not os.path.exists(args.cache):
        print("ERROR: no cached model at %s -- run scripts/calibrate_h.py first." % args.cache)
        return 1

    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model, item = blob["model"], blob["test_items"][0]
    ew = EdgeWeights(model, ds)                      # Eq (10)-(12)
    print("[setup] dataset + cached model                 %s" % fmt(time.perf_counter() - t0))
    print("        n_users=%d  n_links=%d  C=%d  Z=%d"
          % (ds.n_users, ds.n_links, len(model.pi[0]), len(model.theta[0])))

    t0 = time.perf_counter()
    pp = ew.for_item(item)                           # Eq (12)
    t_pp = time.perf_counter() - t0
    print("[setup] Eq (12) weights for item %-6d        %s" % (item, fmt(t_pp)))

    pmax = max(pp.values()) if pp else 0.0
    n_live_all = sum(1 for v in pp.values() if v >= args.h)
    print("        arcs with pp>0: %d   max pp = %.6f   arcs with pp >= h: %d (%.2f%% of E)"
          % (len(pp), pmax, n_live_all, 100.0 * n_live_all / ds.n_links))

    # ---------------------------------------------------------------- A: Eq (19)
    t0 = time.perf_counter()
    comm_a = detect_communities(model.pi)            # Eq (19)
    t_a = time.perf_counter() - t0
    res_a = report("A -- Eq (19)  argmax_c pi[v][c]", comm_a, ds, pp, args.h, t_a)

    # ------------------------------------------------------------------- B: CNM
    comm_b, t_b, hit_b = cnm_labelling(ds, args.C, args.seed, args.cnm_cache)
    res_b = report("B -- CNM greedy modularity, target C=%d%s"
                   % (args.C, " (labelling from cache)" if hit_b else ""),
                   comm_b, ds, pp, args.h, t_b)

    # ------------------------------------------------------- CNM sweep over C
    sweep_rows = []
    targets = [int(x) for x in args.sweep.split(",") if x.strip()]
    if targets:
        print("")
        print("=" * 76)
        print("CNM SWEEP -- C is a TARGET, not a guarantee; 'got' is what came out")
        print("=" * 76)
        print("  %-7s %-6s %-13s %-15s %-12s %-22s %s"
              % ("target", "got", "intra edges", "intra live arcs", "live comms",
                 "biggest live comm", "detect"))
        for c_target in targets + [args.C]:
            if c_target == args.C:
                comm_s, t_s, hit_s = comm_b, t_b, hit_b
            else:
                comm_s, t_s, hit_s = cnm_labelling(ds, c_target, args.seed, args.cnm_cache)
            got = len(set(comm_s))
            st = edge_stats(comm_s, ds.edges, pp, args.h)
            mem_s = community_members(comm_s, ds.n_users)
            ne, nl = st["n_edges"], st["n_live"]
            # The single community holding the most live arcs: if even *that* one
            # cannot spread, no partition of this graph can.
            best_live = top_by_live(st, mem_s, k=1)
            if best_live:
                r = top_community_spread(ds, pp, args.h, mem_s, best_live, st)[0]
                blurb = "c%d |c|=%d I=%.3f" % (r["c"], r["size"], r["influence"])
                best_i = r["influence"]
            else:
                blurb = "(none)"
                best_i = 0.0
            print("  %-7d %-6d %-13s %-15s %-12s %-22s %s%s"
                  % (c_target, got,
                     "%.2f%%" % (100.0 * st["n_intra"] / ne if ne else 0.0),
                     "%.2f%%" % (100.0 * st["n_live_intra"] / nl if nl else 0.0),
                     "%d / %d" % (st["n_comm_with_live"], got),
                     blurb, fmt(t_s), "*" if hit_s else ""))
            sweep_rows.append((c_target, got, st, best_i))
        print("  (* = detection time replayed from the cached labelling;")
        print("   'biggest live comm' = the community with the most live intra arcs,")
        print("   I = Eq (18) spread of its single best seed inside it)")

    # -------------------------------------------------------------- verdict
    sa, sb = res_a["stats"], res_b["stats"]
    print("")
    print("=" * 76)
    print("VERDICT")
    print("=" * 76)
    print("  intra-edge fraction   Eq (19) %.4f%%   ->   CNM %.4f%%   (x%.1f)"
          % (100.0 * sa["n_intra"] / sa["n_edges"],
             100.0 * sb["n_intra"] / sb["n_edges"],
             (sb["n_intra"] / sa["n_intra"]) if sa["n_intra"] else float("inf")))
    print("  communities w/ live   Eq (19) %d / %d      ->   CNM %d / %d"
          % (sa["n_comm_with_live"], res_a["n_nonempty"],
             sb["n_comm_with_live"], res_b["n_nonempty"]))
    print("  Newman Q              Eq (19) %+.6f  ->   CNM %+.6f" % (res_a["q"], res_b["q"]))

    best_a = max([r["influence"] for r in res_a["rows_live"]] or [0.0])
    best_b = max([r["influence"] for r in res_b["rows_live"]] or [0.0])
    # Fewest communities in the sweep == closest thing to no decomposition at all.
    ref = None
    if sweep_rows:
        ref = min(sweep_rows, key=lambda r: r[1])
    print("")
    print("  Best single-seed Eq (18) spread INSIDE a community:")
    print("    Eq (19)  %8.3f   (its best live community)" % best_a)
    print("    CNM      %8.3f   (its best live community)" % best_b)
    if ref is not None:
        print("    undecomposed reference: %8.3f  (CNM bottomed out at %d components,"
              % (ref[3], ref[1]))
        print("                                      so its giant component ~= the whole graph)")

    print("")
    print("  Live arcs in the WHOLE graph: %d of %d (%.3f%%)."
          % (n_live_all, ds.n_links, 100.0 * n_live_all / ds.n_links))
    print("  max pp = %.6f  <=  1/Z = %.6f, so a 2-hop path tops out at %.6f < h=%g:"
          % (pmax, 1.0 / len(model.theta[0]), pmax * pmax, args.h))
    print("  MIA here is provably single-hop -- but single-hop is NOT the same as dead.")
    if n_live_all > 0 and ref is not None and ref[3] > 1.0 + 1e-12:
        print("  => The graph carries real spread (best seed reaches %.1f nodes undecomposed)."
              % ref[3])
        print("     The edgeless community subgraphs are therefore an artefact of the")
        print("     PARTITION, not proof of a dead graph: Eq (19) keeps only %.2f%% of arcs"
              % (100.0 * sa["n_intra"] / sa["n_edges"]))
        print("     at Q=%+.4f (indistinguishable from the degree-preserving null model)."
              % res_a["q"])
    print("")
    print("TOTAL WALL CLOCK %s" % fmt(time.perf_counter() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
