#!/usr/bin/env python3
"""A/B the COMMUNITY DETECTOR while holding the trained model, pp and h fixed.

Why this script exists
======================
CTIM's Algorithm 2 restricts every community's seed candidates to that
community's own members (line 34's ``I_m`` lives on the node-induced subgraph).
The partition therefore *is* the search-space restriction, and Eq (19) builds it
as ``argmax_c pi[v][c]``.  On Digg ``model.pi`` sits only ~1.8% below the uniform
ceiling, so that argmax is close to a coin flip: it retains 8.6% of arcs at
Newman ``Q = -0.001``, i.e. it is statistically a random partition.

Swapping the detector is the one lever that is *methodologically clean* under
the repo's chosen metric.  It does not retrain anything: the cached model, the
Eq (12) edge weights ``pp`` and the Eq (18) threshold ``h`` are byte-identical
across every row of the table below.  Only the map ``v -> community`` changes.
So every ``I_h(S)`` printed here is directly comparable to every other one, and
to every MIA number already in ``results/``.

  A) Eq (19)         detect_communities(model.pi)            -- the baseline
  B) CNM C=100       detect_communities_modularity(ds, 100)  -- graph-aware
  C) CNM C=20        (bottoms out at the 33 connected components)
  D) CNM C=10        (identical labelling to C -- merging cannot cross
                      components; the script asserts the two lists are equal
                      and reuses C's result rather than paying for it twice)
  E) single          [0]*n_users -- Algorithm 2 with the decomposition removed,
                      i.e. GlobalGreedy driven through the same code path.  An
                      upper reference, not a competitor.

Two selection paths per partition
=================================
1. ``exact-DP``   -- ``ctim.ea_dp`` curves + ``allocate_exact``.  This removes
   Algorithm 2's line-35 greedy recurrence as a confound: the allocation is the
   exactly optimal ``O(C K^2)`` resource-allocation DP over the same greedy
   curves.  This is the path the task specifies.
2. ``Algorithm 2`` -- the *unmodified* ``ctim.ctim.ctim_select_seeds``, driven
   with a substituted partition and WITHOUT touching ``ctim/ctim.py``.  The
   injection is exact rather than a hack: Eq (19) is ``argmax_c pi[v][c]``, so a
   model shim whose ``pi`` rows are one-hot at ``comm[v]`` makes
   ``detect_communities`` return precisely the desired partition.  ``pp`` is
   protected by passing the real model's ``EdgeWeights`` through ``edge_weights=``
   (wrapped in a memoiser so the 27s Eq (12) build happens once, not once per
   call), so the shim's ``pi`` never reaches the weights.

Both paths are scored with ONE shared ``MIA(..., h=0.1)`` built from that single
``pp`` (Eq (18)), plus ``ic_simulate`` as a neutral Monte-Carlo IC referee on the
same ``pp`` -- so the MC column is a valid comparison here too.

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

from ctim.baselines.air_cga import (                    # noqa: E402
    detect_communities_modularity,
    modularity,
)
from ctim.ctim import ctim_select_seeds, detect_communities  # noqa: E402
from ctim.dataset import load_dataset                   # noqa: E402
from ctim.ea_dp import allocate_exact, build_curves, select_seeds_ea_dp  # noqa: E402
from ctim.influence import MIA, EdgeWeights             # noqa: E402
from ctim.ris_imm import ic_simulate                    # noqa: E402


def fmt(s):
    return "%.2fs" % s


# ---------------------------------------------------------------------------
# injection helpers -- neither touches ctim/
# ---------------------------------------------------------------------------


class _CachedEdgeWeights:
    """Memoising pass-through for ``EdgeWeights``.

    ``ctim_select_seeds`` calls ``ew.for_item(item)`` on every invocation and
    that call rebuilds all |E| Eq (12) weights (~27s on Digg).  Since the item
    is fixed for the whole experiment, the dict is computed once and handed back
    by identity -- which also *guarantees* every row of the table shares one
    ``pp`` object, the precondition for the MIA numbers being comparable.
    """

    __slots__ = ("_ew", "_cache", "approximate", "n_clamped", "n_calls")

    def __init__(self, ew):
        self._ew = ew
        self._cache = {}
        self.approximate = getattr(ew, "approximate", False)
        self.n_clamped = getattr(ew, "n_clamped", 0)
        self.n_calls = 0

    def for_item(self, i):
        self.n_calls += 1
        got = self._cache.get(i)
        if got is None:
            got = self._ew.for_item(i)  # Eq (12)
            self._cache[i] = got
        self.n_clamped = getattr(self._ew, "n_clamped", 0)
        return got

    def prime(self, i, pp):
        """Seed the cache with an already-built ``pp`` for item ``i``."""
        self._cache[i] = pp


class _PartitionModel:
    """A ``ctim.gibbs.Model`` stand-in whose Eq (19) argmax IS a given partition.

    Eq (19) reads ``c^v <- argmax_c pi[v][c]``, so one-hot rows reproduce ``comm``
    exactly (``detect_communities`` breaks ties with a strict ``>``, and a one-hot
    row has a unique maximiser, so there are no ties to break).  Rows are shared
    objects, so the memory cost is ``O(C^2)``, not ``O(U*C)``.

    ``ctim_select_seeds`` reads exactly two attributes off the model:
    ``len(model.eta)`` (the community count ``C``) and ``model.pi``.  ``theta`` and
    ``p_z_given_i`` are carried through only so the object is not silently
    lying about being a model; they are never consulted because ``edge_weights=``
    is always supplied, which short-circuits the ``EdgeWeights(model, ds)``
    construction inside Algorithm 2 lines 1-21.
    """

    __slots__ = ("pi", "eta", "theta", "p_z_given_i")

    def __init__(self, comm, n_users, real_model):
        C = (max(comm) + 1) if comm else 1
        rows = []
        for c in range(C):
            row = [0.0] * C
            row[c] = 1.0
            rows.append(row)
        self.pi = [rows[comm[v]] for v in range(n_users)]
        self.eta = [[0.0] * C for _ in range(C)]  # only len() is ever read
        self.theta = real_model.theta
        self.p_z_given_i = real_model.p_z_given_i


# ---------------------------------------------------------------------------
# partition diagnostics
# ---------------------------------------------------------------------------


def partition_stats(comm, ds, pp, h):
    """Intra-community arc fractions (all arcs and live arcs), sizes, Newman Q."""
    intra = 0
    total = 0
    intra_live = 0
    total_live = 0
    for (u, v) in ds.edges:
        total += 1
        live = pp.get((u, v), 0.0) >= h
        if live:
            total_live += 1
        if u < len(comm) and v < len(comm) and comm[u] == comm[v]:
            intra += 1
            if live:
                intra_live += 1
    sizes = {}
    for v in range(min(len(comm), ds.n_users)):
        sizes[comm[v]] = sizes.get(comm[v], 0) + 1
    sz = sorted(sizes.values(), reverse=True)
    return {
        "intra": intra, "total": total,
        "intra_live": intra_live, "total_live": total_live,
        "n_nonempty": len(sizes),
        "max_size": sz[0] if sz else 0,
        "median_size": sz[len(sz) // 2] if sz else 0,
        "Q": modularity(comm, ds.n_users, ds.edges),
    }


def load_cnm(ds, C, seed, cache_dir):
    """CNM labelling for target ``C``, replayed from the scratchpad cache if present."""
    path = None
    if cache_dir:
        path = os.path.join(cache_dir, "cnm_digg_C%d_s%d.pkl" % (C, seed))
        if os.path.exists(path):
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
            return list(blob["comm"]), float(blob.get("seconds", 0.0)), True
    t0 = time.perf_counter()
    comm = detect_communities_modularity(ds, C=C, rng=random.Random(seed))
    secs = time.perf_counter() - t0
    if path:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump({"comm": comm, "seconds": secs, "C": C, "seed": seed}, fh)
        except OSError:
            pass
    return comm, secs, False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    p.add_argument("--Ks", default="10,20,50")
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mc", type=int, default=1000)
    p.add_argument("--mc-seed", type=int, default=7)
    p.add_argument("--cnm-cache", default="",
                   help="directory holding cnm_digg_C<C>_s<seed>.pkl labellings")
    p.add_argument("--no-alg2", action="store_true",
                   help="skip the true-Algorithm-2 path (exact-DP path only)")
    args = p.parse_args(argv)

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    Kmax = max(Ks)
    t_all = time.perf_counter()

    print("=" * 96)
    print("LEVER 1 -- community DETECTOR A/B.  Same model, same pp, same h, same item.")
    print("Ks=%s  h=%g  seed=%d  mc=%d (rng seed %d)"
          % (Ks, args.h, args.seed, args.mc, args.mc_seed))
    print("=" * 96)

    # ---------------------------------------------------------------- setup
    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model, item = blob["model"], blob["test_items"][0]
    ew = EdgeWeights(model, ds)
    print("[setup] dataset + cached model                  %s  (n_users=%d n_edges=%d)"
          % (fmt(time.perf_counter() - t0), ds.n_users, len(ds.edges)))

    t0 = time.perf_counter()
    pp = ew.for_item(item)                                   # Eq (12)
    t_pp = time.perf_counter() - t0
    print("[setup] Eq (12) weights, item %-6d            %s  (|pp|=%d)"
          % (item, fmt(t_pp), len(pp)))

    cew = _CachedEdgeWeights(ew)
    cew.prime(item, pp)   # identical object -> identical graph for every row

    t0 = time.perf_counter()
    evaluator = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)   # Eq (18)
    print("[setup] shared Eq (18) evaluator                %s" % fmt(time.perf_counter() - t0))

    # ------------------------------------------------------------ partitions
    parts = []          # (label, comm, detect_seconds, note)
    t0 = time.perf_counter()
    comm_a = detect_communities(model.pi)                    # Eq (19)
    parts.append(["A) Eq (19)", comm_a, time.perf_counter() - t0, ""])

    cnm = {}
    for C in (100, 20, 10):
        comm_c, secs, cached = load_cnm(ds, C, args.seed, args.cnm_cache)
        cnm[C] = (comm_c, secs, cached)
    dup_c20_c10 = cnm[20][0] == cnm[10][0]
    parts.append(["B) CNM C=100", cnm[100][0], cnm[100][1],
                  "cached" if cnm[100][2] else ""])
    parts.append(["C) CNM C=20", cnm[20][0], cnm[20][1],
                  "cached" if cnm[20][2] else ""])
    parts.append(["D) CNM C=10", cnm[10][0], cnm[10][1],
                  ("IDENTICAL labelling to C -- results copied, not recomputed"
                   if dup_c20_c10 else ("cached" if cnm[10][2] else ""))])
    parts.append(["E) single comm", [0] * ds.n_users, 0.0,
                  "GlobalGreedy through the Algorithm 2 code path"])
    is_cached = {"A) Eq (19)": False, "B) CNM C=100": cnm[100][2],
                 "C) CNM C=20": cnm[20][2], "D) CNM C=10": cnm[10][2],
                 "E) single comm": False}

    print("\n" + "-" * 96)
    print("PARTITION PROPERTIES  (same pp, h=%g; 'live' arc means pp >= h)" % args.h)
    print("-" * 96)
    print("%-15s %7s %9s %9s %11s %11s %10s %9s"
          % ("partition", "#comm", "maxsize", "medsize", "intra-arc%", "intra-live%",
             "Newman Q", "detect(s)"))
    pstats = {}
    for (label, comm, secs, note) in parts:
        st = partition_stats(comm, ds, pp, args.h)
        pstats[label] = st
        print("%-15s %7d %9d %9d %10.4f%% %10.4f%% %10.6f %9.2f%s"
              % (label, st["n_nonempty"], st["max_size"], st["median_size"],
                 100.0 * st["intra"] / st["total"],
                 100.0 * st["intra_live"] / st["total_live"],
                 st["Q"], secs, "*" if is_cached.get(label) else ""))
    print("  live arcs in the whole graph: %d / %d (%.3f%%) -- the ceiling any"
          " partition could retain"
          % (pstats["A) Eq (19)"]["total_live"], pstats["A) Eq (19)"]["total"],
             100.0 * pstats["A) Eq (19)"]["total_live"] / pstats["A) Eq (19)"]["total"]))
    print("  CNM C=20 labelling == CNM C=10 labelling: %s" % dup_c20_c10)
    for (label, _c, _s, note) in parts:
        if note:
            print("  %-15s %s" % (label, note))

    # ------------------------------------------------------------- selection
    mc_cache = {}

    def mc_of(seeds):
        key = frozenset(seeds)
        got = mc_cache.get(key)
        if got is None:
            got = ic_simulate(ds.out_adj, pp, list(seeds), args.mc,
                              random.Random(args.mc_seed))
            mc_cache[key] = got
        return got

    rows = []   # dicts
    base_seeds = {}   # K -> set(seeds) for partition A, exact-DP path
    curve_secs = {}   # partition -> the ONE shared greedy-curve build cost

    for idx, (label, comm, _secs, note) in enumerate(parts):
        if label.startswith("D)") and dup_c20_c10:
            continue  # handled after the loop by copying C's rows verbatim
        print("\n" + "-" * 96)
        print("PARTITION %s" % label)
        print("-" * 96)
        sys.stdout.flush()

        # ---- path 1: shared greedy curves + exact allocation DP -----------
        t0 = time.perf_counter()
        curves, cstats = build_curves(comm, ds, pp, Kmax, args.h)
        t_curves = time.perf_counter() - t0
        curve_secs[label] = t_curves
        print("  [curves @ Kmax=%d] %d non-empty communities  %s"
              " (MIA %s, greedy %s)"
              % (Kmax, len(curves), fmt(t_curves), fmt(cstats["mia_seconds"]),
                 fmt(cstats["greedy_seconds"])))
        sys.stdout.flush()

        for K in Ks:
            t0 = time.perf_counter()
            seeds, dp_value, alloc = allocate_exact(curves, K)
            t_dp = time.perf_counter() - t0
            mia_val = evaluator.influence(list(seeds))          # Eq (18), shared
            mc_val = mc_of(seeds)
            if label.startswith("A)"):
                base_seeds[K] = set(seeds)
            ov = len(set(seeds) & base_seeds.get(K, set()))
            rows.append({
                "part": label, "path": "exact-DP", "K": K,
                "mia": mia_val, "mc": mc_val,
                "raw_secs": t_dp, "n_used": len(alloc),
                "overlap": ov, "n_seeds": len(seeds),
                "alloc": dict(sorted(alloc.items())),
            })
            print("  [exact-DP K=%2d] I_h=%12.6f  MC-IC=%10.4f  DP %s"
                  "  communities used=%d  |S|=%d"
                  % (K, mia_val, mc_val, fmt(t_dp), len(alloc), len(seeds)))
            sys.stdout.flush()

        # ---- path 2: the UNMODIFIED Algorithm 2, partition injected -------
        if not args.no_alg2:
            shim = _PartitionModel(comm, ds.n_users, model)
            got = detect_communities(shim.pi)
            assert got == list(comm[:ds.n_users]), \
                "one-hot pi injection did not reproduce the partition"
            for K in Ks:
                t0 = time.perf_counter()
                seeds2 = ctim_select_seeds(shim, ds, item, K, h=args.h,
                                           dp_tiebreak="paper-true",
                                           edge_weights=cew)
                t_a2 = time.perf_counter() - t0
                st2 = ctim_select_seeds.last_stats
                assert st2["n_edges"] == len(pp), "pp changed under Algorithm 2"
                mia2 = evaluator.influence(list(seeds2))        # Eq (18), shared
                mc2 = mc_of(seeds2)
                ov2 = len(set(seeds2) & base_seeds.get(K, set()))
                rows.append({
                    "part": label, "path": "Alg 2", "K": K,
                    "mia": mia2, "mc": mc2, "raw_secs": t_a2,
                    "n_used": len(st2["seeds_per_community"]),
                    "overlap": ov2, "n_seeds": len(seeds2),
                    "alloc": {m - 1: v for m, v in
                              sorted(st2["seeds_per_community"].items())},
                })
                print("  [Alg 2   K=%2d] I_h=%12.6f  MC-IC=%10.4f  sel %s"
                      "  communities used=%d  |S|=%d  fallbacks=%d"
                      % (K, mia2, mc2, fmt(t_a2),
                         len(st2["seeds_per_community"]), len(seeds2),
                         st2["n_fallbacks"]))
                sys.stdout.flush()

    # D) is a verbatim copy of C) when the labellings match
    if dup_c20_c10:
        for r in [r for r in rows if r["part"].startswith("C)")]:
            d = dict(r)
            d["part"] = "D) CNM C=10"
            rows.append(d)

    # --------------------------------------------------- validity assertions
    print("\n" + "-" * 96)
    print("VALIDITY CHECKS")
    print("-" * 96)
    print("  Eq (12) weight dict built %d time(s); Algorithm 2 asked for it %d"
          " time(s) and got the SAME object each time" % (1, cew.n_calls))
    # the shared-curve shortcut must equal a real per-K select_seeds_ea_dp call
    K_chk = min(Ks)
    t0 = time.perf_counter()
    ref_seeds, ref_stats = select_seeds_ea_dp(model, ds, pp, K_chk,
                                              random.Random(args.seed),
                                              h=args.h, use_ea=False,
                                              comm=comm_a)
    t_chk = time.perf_counter() - t0
    mine = [r for r in rows if r["part"].startswith("A)") and r["path"] == "exact-DP"
            and r["K"] == K_chk][0]
    same = (set(ref_seeds) == base_seeds[K_chk]
            and abs(evaluator.influence(list(ref_seeds)) - mine["mia"]) < 1e-9)
    print("  shared-Kmax-curve shortcut vs real select_seeds_ea_dp(K=%d, use_ea=False):"
          " %s  (%s, alloc %s)"
          % (K_chk, "IDENTICAL" if same else "*** MISMATCH ***", fmt(t_chk),
             "match" if ref_stats["allocation"] == mine["alloc"] else "DIFFER"))

    # ------------------------------------------------------------ the table
    print("\n" + "=" * 96)
    print("RESULTS -- MIA Eq (18) at h=%g, ONE pp, ONE item (%d), ONE evaluator."
          % (args.h, item))
    print("Every I_h below is directly comparable to every other one.  gap% is"
          " vs the Eq (19) baseline at the same K and path.")
    print("sel(s) for exact-DP is the ALLOCATION DP ALONE; the greedy curves are"
          " built once per partition")
    print("at Kmax=%d and shared across K -- that shared cost is: %s"
          % (Kmax, ", ".join("%s %.2fs" % (k.split(")")[0], v)
                             for k, v in sorted(curve_secs.items()))))
    print("=" * 96)
    header = ("%-15s %-9s %3s %14s %9s %11s %8s %7s %8s"
              % ("partition", "path", "K", "I_h(S) Eq18", "gap%", "MC-IC(n=%d)" % args.mc,
                 "sel(s)", "#comm", "overlapA"))
    order = ["A) Eq (19)", "B) CNM C=100", "C) CNM C=20", "D) CNM C=10", "E) single comm"]
    for path in ("exact-DP", "Alg 2"):
        sub = [r for r in rows if r["path"] == path]
        if not sub:
            continue
        print("\n" + header)
        print("-" * 96)
        for K in Ks:
            base = [r for r in sub if r["K"] == K and r["part"] == "A) Eq (19)"]
            b = base[0]["mia"] if base else float("nan")
            for label in order:
                cand = [r for r in sub if r["K"] == K and r["part"] == label]
                if not cand:
                    continue
                r = cand[0]
                gap = 100.0 * (r["mia"] - b) / b if b else float("nan")
                print("%-15s %-9s %3d %14.6f %+8.3f%% %11.4f %8.2f %7d %8d"
                      % (r["part"], r["path"], r["K"], r["mia"], gap, r["mc"],
                         r["raw_secs"], r["n_used"], r["overlap"]))
            print("")

    # ------------------------------------------------------------- verdict
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    any_win = False
    for path in ("exact-DP", "Alg 2"):
        for K in Ks:
            base = [r for r in rows if r["path"] == path and r["K"] == K
                    and r["part"] == "A) Eq (19)"]
            if not base:
                continue
            b = base[0]["mia"]
            best = None
            for r in rows:
                if r["path"] != path or r["K"] != K or r["part"].startswith("A)"):
                    continue
                if r["part"].startswith("E)"):
                    continue  # upper reference, not a competitor
                if best is None or r["mia"] > best["mia"]:
                    best = r
            if best is None:
                continue
            gap = 100.0 * (best["mia"] - b) / b
            if gap > 0:
                any_win = True
            print("  %-9s K=%-3d best non-baseline detector = %-15s"
                  " I_h %.6f vs baseline %.6f  -> %+.3f%%"
                  % (path, K, best["part"], best["mia"], b, gap))
        for K in Ks:
            base = [r for r in rows if r["path"] == path and r["K"] == K
                    and r["part"] == "A) Eq (19)"]
            ref = [r for r in rows if r["path"] == path and r["K"] == K
                   and r["part"].startswith("E)")]
            if base and ref:
                print("  %-9s K=%-3d upper reference (single community) %+.3f%%"
                      % (path, K, 100.0 * (ref[0]["mia"] - base[0]["mia"]) / base[0]["mia"]))
    print("  LEVER 1 %s"
          % ("SUCCEEDS: a substituted detector beats Eq (19) at some K."
             if any_win else
             "FAILS: no substituted detector beats Eq (19) at any K."))

    print("\nTOTAL WALL CLOCK %.2fs" % (time.perf_counter() - t_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
