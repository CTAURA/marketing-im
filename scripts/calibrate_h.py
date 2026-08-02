#!/usr/bin/env python3
"""Calibrate the MIA threshold ``h`` (Eq (15)) against measured data.

Motivation
----------
``h`` is fixed at 0.1 by SPEC.md Section 8, but ``h`` is only meaningful
*relative to the scale of the Eq (12) edge weights*.  On this Digg substitute
``max pp = 0.124``, so the strongest possible two-edge chain has probability
``0.124^2 = 0.0154 < 0.1``: no 2-hop path survives and MIA degenerates to
single-hop influence.  That is an approximation regime the paper never ran in.

``h`` is a knob of the *inference procedure*, not of the generative model
(it only decides where Eq (13)/(14)'s Dijkstra is truncated), so tuning it does
not tamper with what is being measured -- unlike rescaling Eq (12) would.

What this script measures
-------------------------
1. **Model diagnostics.**  Normalised entropy of ``theta``/``psi``/``pi``.
   Tests the hypothesis that Digg's *derived* item attributes leave ``theta``
   near-uniform, which would cap Eq (12) at ``max_{c,z} theta_cz ~ 1/Z``.
2. **Edge-weight distribution.**  Quantiles of ``pp`` and of ``-ln pp``.
3. **MIP distance distribution.**  The quantity that actually governs MIA:

       d(u -> v) = -ln pp(MIP(u,v))          # Eq (13)/(14)
       |MIIA(v,h)| = #{ u : d(u -> v) <= -ln h }

   So ``h`` is literally a quantile cut on that distribution.  ONE Dijkstra per
   sampled node with a generous cutoff yields the whole ``|MIIA| vs h`` curve by
   re-thresholding -- no need to rebuild MIA per ``h``.
4. **Influence saturation.**  ``I_h(S)`` for a fixed reference seed set as ``h``
   falls.  Pick the largest ``h`` reaching >= a target fraction of the
   saturated value: that is the accuracy side of the trade-off.

Every phase prints its own wall-clock seconds.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random``.
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

from ctim.entropy import mean_entropy                                    # noqa: E402
from ctim.experiments import Experiment, ExperimentConfig, quick_config  # noqa: E402
from ctim.influence import MIA, EdgeWeights                              # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def quantiles(xs, qs):
    """Nearest-rank quantiles of a list (sorted internally)."""
    if not xs:
        return [float("nan")] * len(qs)
    s = sorted(xs)
    n = len(s)
    out = []
    for q in qs:
        idx = min(n - 1, max(0, int(math.ceil(q * n)) - 1))
        out.append(s[idx])
    return out


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def fmt_secs(s):
    return "%8.3fs" % s


# ---------------------------------------------------------------------------
# phase 0 -- prepare (dataset + Gibbs fit), cached
# ---------------------------------------------------------------------------


def prepare(args):
    """Load/fit everything CTIM needs.  Returns (ds, model, ew, test_items)."""
    cache = args.cache
    if cache and os.path.exists(cache):
        t0 = time.perf_counter()
        with open(cache, "rb") as fh:
            blob = pickle.load(fh)
        print("[phase 0] model loaded from cache %s  %s"
              % (cache, fmt_secs(time.perf_counter() - t0)))
        cfg = _build_cfg(args)
        exp = Experiment(cfg)
        t1 = time.perf_counter()
        from ctim.dataset import load_dataset
        ds = load_dataset(cfg.dataset)
        print("[phase 0] dataset reloaded                %s"
              % fmt_secs(time.perf_counter() - t1))
        model = blob["model"]
        test_items = blob["test_items"]
        t2 = time.perf_counter()
        ew = EdgeWeights(model, ds)  # Eq (10)+(11)
        print("[phase 0] EdgeWeights rebuilt             %s"
              % fmt_secs(time.perf_counter() - t2))
        del exp
        return ds, model, ew, test_items

    cfg = _build_cfg(args)
    exp = Experiment(cfg)
    t0 = time.perf_counter()
    exp.prepare()  # load + Definition 1 logs + 60/20/20 split + Gibbs fit
    secs = time.perf_counter() - t0
    print("[phase 0] prepare total (incl. Gibbs fit)  %s" % fmt_secs(secs))
    print("[phase 0]   build_logs                     %s"
          % fmt_secs(exp.timings.get("build_logs", 0.0)))
    print("[phase 0]   Gibbs fit  (Eq (1)-(9))        %s"
          % fmt_secs(exp.model_fit_seconds))

    if cache:
        with open(cache, "wb") as fh:
            pickle.dump({"model": exp.model, "test_items": exp.test_items}, fh)
        print("[phase 0] model cached to %s" % cache)
    return exp.ds, exp.model, exp.ew, exp.test_items


def _build_cfg(args):
    cfg = ExperimentConfig(dataset=args.dataset, C=args.C, Z=args.Z,
                           seed=args.seed, h=args.h_ref)
    if args.quick:
        cfg = quick_config(cfg)
    cfg.gibbs_iters_topic = args.gibbs_iters
    cfg.gibbs_iters_comm = args.gibbs_iters
    cfg.sampler = args.sampler
    cfg.n_test_items = args.n_test_items
    if args.max_logs_per_item > 0:
        cfg.max_logs_per_item = args.max_logs_per_item
    return cfg


# ---------------------------------------------------------------------------
# phase 1 -- model diagnostics
# ---------------------------------------------------------------------------


def phase_model(model, Z, C):
    t0 = time.perf_counter()
    print("\n" + "=" * 74)
    print("PHASE 1 -- model diagnostics (is there any topic signal to find?)")
    print("=" * 74)

    # Hbar = (1/T) sum_t H^(t),  H^(t) = -sum_{i=1..K} p_i log p_i  (ctim.entropy)
    blocks = [
        ("phi[i][z]  ", getattr(model, "phi", None), "items", "topics"),
        ("p_z_given_i", getattr(model, "p_z_given_i", None), "items", "topics"),
        ("theta[c][z]", model.theta, "communities", "topics"),
        ("psi[z][f]  ", getattr(model, "psi", None), "topics", "attributes"),
        ("pi[v][c]   ", model.pi, "users", "communities"),
    ]

    print("  Hbar = (1/T) sum_t H^(t),   H^(t) = -sum_{i=1..K} p_i^(t) log p_i^(t)")
    print("  raw Hbar is in [0, ln K]; normalised is Hbar/ln K in [0, 1],")
    print("  where 1.0 = uniform = the prior was never overcome = nothing learned.\n")
    print("  %-12s %6s %5s %10s %10s %10s   %10s %10s"
          % ("matrix", "T", "K", "Hbar", "ln K", "Hbar/lnK", "mean max_i", "uniform"))

    out = {}
    for name, mat, t_unit, k_unit in blocks:
        if not mat:
            continue
        raw = mean_entropy(mat)                      # the printed formula
        nrm = mean_entropy(mat, normalized=True)     # comparable across K
        mx = [max(r) / sum(r) for r in mat if sum(r) > 0]
        print("  %-12s %6d %5d %10.5f %10.5f %10.5f   %10.5f %10.5f"
              % (name, raw.T, raw.K, raw.mean, raw.max_possible, nrm.mean,
                 mean(mx), 1.0 / raw.K))
        out[name.strip()] = {"Hbar": raw.mean, "Hbar_norm": nrm.mean,
                             "T": raw.T, "K": raw.K, "mean_max": mean(mx)}

    th_max = [max(r) for r in model.theta]
    print("\n  --> Eq (12) is bounded above by max_{c,z} theta = %.4f  (1/Z = %.4f)"
          % (max(th_max), 1.0 / Z))
    print("  --> rows read (T) x components (K); psi is the control: it is the one")
    print("      matrix whose prior is an absolute 0.01 rather than a mass-50 50/X.")

    print("[phase 1] %s" % fmt_secs(time.perf_counter() - t0))
    out["theta_max"] = max(th_max)
    return out


# ---------------------------------------------------------------------------
# phase 2 -- edge weight distribution
# ---------------------------------------------------------------------------


def phase_weights(ds, ew, item, h_grid):
    t0 = time.perf_counter()
    print("\n" + "=" * 74)
    print("PHASE 2 -- Eq (12) edge-weight distribution (item %d)" % item)
    print("=" * 74)

    t = time.perf_counter()
    pp = ew.for_item(item)  # Eq (12)
    t_pp = time.perf_counter() - t

    vals = [p for p in pp.values() if p > 0.0]
    qs = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00]
    qv = quantiles(vals, qs)
    print("  %d edges with pp > 0 (of %d)   [built in %s]"
          % (len(vals), ds.n_links, fmt_secs(t_pp).strip()))
    print("  quantile     " + "".join("%9s" % ("p%g" % (q * 100)) for q in qs))
    print("  pp           " + "".join("%9.5f" % v for v in qv))
    print("  -ln pp       " + "".join("%9.3f" % (-math.log(v) if v > 0 else float('inf'))
                                      for v in qv))
    print("")
    print("  fraction of edges with pp >= h, and the single-edge budget check:")
    print("    %-10s %-12s %-12s %s" % ("h", "-ln h", "frac pp>=h", "max hops of the STRONGEST edge"))
    pmax = max(vals) if vals else 0.0
    for h in h_grid:
        frac = sum(1 for p in vals if p >= h) / float(len(vals)) if vals else 0.0
        hops = (math.log(h) / math.log(pmax)) if 0 < pmax < 1 else float("inf")
        print("    %-10.6g %-12.3f %-12.4f %.2f" % (h, -math.log(h), frac, hops))
    print("  (max pp = %.5f; 'max hops' = ln h / ln max_pp -- an UPPER bound on"
          % pmax)
    print("   the depth any MIP can reach, since no edge is stronger than max pp)")
    print("[phase 2] %s" % fmt_secs(time.perf_counter() - t0))
    return pp


# ---------------------------------------------------------------------------
# phase 3 -- MIP distance distribution  (the thing h actually cuts)
# ---------------------------------------------------------------------------


def phase_mip(ds, pp, h_grid, h_open, n_sample, rng):
    t0 = time.perf_counter()
    print("\n" + "=" * 74)
    print("PHASE 3 -- MIP distance distribution  d(u->v) = -ln pp(MIP(u,v))")
    print("=" * 74)
    print("  one Dijkstra per sampled node at h_open=%g (cutoff -ln h = %.3f);"
          % (h_open, -math.log(h_open)))
    print("  every h in the grid is then just a re-threshold of the same array.")

    t = time.perf_counter()
    mia_open = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h_open)
    t_build = time.perf_counter() - t
    print("  MIA(h_open) constructed                  %s" % fmt_secs(t_build))

    nodes = [v for v in range(ds.n_users) if ds.in_adj[v]]
    if len(nodes) > n_sample:
        nodes = rng.sample(nodes, n_sample)
    nodes.sort()

    t = time.perf_counter()
    # counts_by_h[i] accumulates |MIIA(v, h_grid[i])| over the sample.
    counts_by_h = [0] * len(h_grid)
    depth_by_h = [0] * len(h_grid)
    cut = [-math.log(h) for h in h_grid]
    all_d = []
    for v in nodes:
        dist, parent = mia_open._dijkstra(v, forward=False)  # Eq (15)
        # hop depth of each node on its MIP, memoised along parent chains
        hop = {v: 0}

        def depth_of(x):
            chain = []
            while x not in hop:
                chain.append(x)
                x = parent.get(x)
                if x is None:
                    for y in chain:
                        hop[y] = 0
                    return 0
            d = hop[x]
            while chain:
                d += 1
                hop[chain.pop()] = d
            return d

        for x, dv in dist.items():
            if x == v:
                continue
            all_d.append(dv)
            dep = depth_of(x)
            for i, c in enumerate(cut):
                if dv <= c:
                    counts_by_h[i] += 1
                    if dep > depth_by_h[i]:
                        depth_by_h[i] = dep
    t_dij = time.perf_counter() - t

    n = float(len(nodes))
    print("  %d nodes sampled, %d Dijkstra runs         %s"
          % (len(nodes), len(nodes), fmt_secs(t_dij)))
    qs = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    qv = quantiles(all_d, qs)
    print("  d(u->v) over %d reachable pairs:" % len(all_d))
    print("    quantile   " + "".join("%9s" % ("p%g" % (q * 100)) for q in qs))
    print("    d          " + "".join("%9.3f" % v for v in qv))
    print("    pp(MIP)    " + "".join("%9.2e" % math.exp(-v) for v in qv))
    print("")
    print("  |MIIA| and max MIP depth as a function of h:")
    print("    %-10s %-10s %-14s %s" % ("h", "-ln h", "mean |MIIA|-1", "max depth (hops)"))
    for i, h in enumerate(h_grid):
        print("    %-10.6g %-10.3f %-14.2f %d"
              % (h, cut[i], counts_by_h[i] / n, depth_by_h[i]))
    print("[phase 3] %s" % fmt_secs(time.perf_counter() - t0))
    return {"mean_miia": [counts_by_h[i] / n for i in range(len(h_grid))],
            "max_depth": list(depth_by_h)}


# ---------------------------------------------------------------------------
# phase 4 -- influence saturation I_h(S)
# ---------------------------------------------------------------------------


def phase_saturation(ds, pp, h_grid, K, h_ref, budget_s):
    t0 = time.perf_counter()
    print("\n" + "=" * 74)
    print("PHASE 4 -- influence saturation I_h(S) for a FIXED seed set")
    print("=" * 74)

    t = time.perf_counter()
    mia_ref = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h_ref)
    S = mia_ref.greedy_incremental(K)  # the seed set CTIM would pick today
    t_seeds = time.perf_counter() - t
    print("  reference S = MIA-greedy K=%d at h=%g            %s"
          % (K, h_ref, fmt_secs(t_seeds)))
    print("  S = %s" % S)
    print("")
    print("    %-10s %-12s %-12s %-10s %s"
          % ("h", "I_h(S)", "vs saturated", "|reached|", "seconds"))

    rows = []
    skipped = []
    for h in h_grid:
        if budget_s > 0 and rows and rows[-1][2] > budget_s:
            skipped.append(h)
            continue
        t = time.perf_counter()
        mia = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)
        val = mia.influence(S)  # Eq (18)
        reached = set()
        for u in S:
            reached.update(mia.mioa(u)[0])
        secs = time.perf_counter() - t
        rows.append((h, val, secs, len(reached)))
        print("    %-10.6g %-12.3f %-12s %-10d %.3fs"
              % (h, val, "-", len(reached), secs))

    if rows:
        sat = max(r[1] for r in rows)
        print("")
        print("  saturated value (smallest h reached) = %.3f" % sat)
        print("    %-10s %-12s %s" % ("h", "I_h(S)", "fraction of saturated"))
        for (h, val, _s, _r) in rows:
            print("    %-10.6g %-12.3f %.4f" % (h, val, val / sat if sat else 0.0))
    if skipped:
        print("  SKIPPED (previous h exceeded the %.0fs budget): %s"
              % (budget_s, ", ".join("%g" % h for h in skipped)))
    print("[phase 4] %s" % fmt_secs(time.perf_counter() - t0))
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--C", type=int, default=100)
    p.add_argument("--Z", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", default=True)
    p.add_argument("--gibbs-iters", type=int, default=30)
    p.add_argument("--sampler", default="mh")
    p.add_argument("--n-test-items", type=int, default=3)
    p.add_argument("--max-logs-per-item", type=int, default=20)
    p.add_argument("--h-ref", type=float, default=0.1,
                   help="the SPEC.md threshold under test")
    p.add_argument("--h-open", type=float, default=1e-5,
                   help="generous cutoff for the MIP distance survey")
    p.add_argument("--n-sample", type=int, default=300,
                   help="nodes sampled for the MIP distance survey")
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--budget-seconds", type=float, default=120.0,
                   help="stop the h sweep once one point costs more than this")
    p.add_argument("--cache", default=os.path.join(
        _REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    rng = random.Random(args.seed)

    print("=" * 74)
    print("h CALIBRATION -- dataset=%s C=%d Z=%d seed=%d"
          % (os.path.basename(args.dataset), args.C, args.Z, args.seed))
    print("=" * 74)

    ds, model, ew, test_items = prepare(args)
    item = test_items[0]

    h_grid = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

    phase_model(model, args.Z, args.C)
    pp = phase_weights(ds, ew, item, h_grid)
    phase_mip(ds, pp, h_grid, args.h_open, args.n_sample, rng)
    phase_saturation(ds, pp, h_grid, args.K, args.h_ref, args.budget_seconds)

    print("\n" + "=" * 74)
    print("TOTAL %s" % fmt_secs(time.perf_counter() - t_all))
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
