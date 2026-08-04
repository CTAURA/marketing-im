#!/usr/bin/env python3
"""Empirically test the claim that Digg's propagation weights are capped at 1/Z.

Why this script exists
----------------------
Every selection-layer experiment in this repo has come back flat: EA inside
communities gains 0.00%, the exact O(C K^2) allocation DP gains 0.00%, greedy
matches restricted brute force.  The measured reason is that 98 of the 100
communities have within-community curve ``I_m(j) = j`` -- their induced
subgraphs carry no live edges at ``h = 0.1``.  That is a symptom.  The proposed
*cause* is analytic and lives one layer down, in the trained model:

    ``theta`` never leaves its Dirichlet prior (alpha = 50/Z puts total mass 50
    on rows that Digg feeds ~6.97 attribute tokens each), so
    ``theta[c][z] = 1/Z`` for every c, z.  Then Eq (12) gives
    ``thetabar_i[c] = sum_z P(z|i) * theta[c][z] = 1/Z`` for EVERY item, and
    since Eq (11)+(12) factorise as
    ``pp(u,v) = sum_c a_u[c] * pi_v[c] * thetabar_i[c]``
    with ``a_u[c] <= 1`` and ``sum_c pi_v[c] = 1``, we get the hard bound

        pp(u,v) <= 1/Z = 0.125.

    A 2-hop path therefore has ``pp <= 0.0156 < h = 0.1``, so MIA Eq (15)/(16)
    is *provably* single-hop on this model and there is nothing for a smarter
    selector to find.

An argument of that shape is worth exactly as much as its weakest measured
premise, so this script measures all of them:

  [1] is ``thetabar_i[c]`` really 1/Z, for several items, and is ``theta`` really
      at maximum entropy?
  [2] what does the pp distribution actually look like, and does max(pp) sit at
      or below the predicted ceiling?
  [3] how deep does MIA actually reach, at h in {0.1, 0.01, 0.001}?
  [4] MECHANISM TEST -- if ``theta`` is sharpened synthetically and nothing else
      changes, does the ceiling move?  If it does, ``theta`` is the cause; if it
      does not, the diagnosis is wrong and the blame lies elsewhere.
  [4b] follow-up, run unconditionally because [4] can fail in an informative way:
      Eq (12) mixes ``theta`` through ``P(z|i)``, so a sharp ``theta`` still
      yields a flat ``thetabar`` if ``P(z|i)`` is diffuse.  [4b] sharpens BOTH
      and reports the analytic per-item bound ``max_c thetabar_i[c]`` next to the
      observed ``max(pp)`` in all three worlds, which separates "thetabar is
      flat" from "a_u * pi_v puts no mass on thetabar's argmax".

Metric note (API.md / CLAUDE.md rule)
-------------------------------------
Sections [1]-[3] all use the cached model unchanged, so they share one ``pp``
and are internally comparable.  Section [4] deliberately CHANGES ``pp``: it
builds a different graph.  No MIA influence value is reported for it and none
should be -- it is a mechanism probe, not a comparison.  This is labelled in the
output.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random(seed)``.

Run:
    python -u scripts/diagnose_pp.py
"""

from __future__ import annotations

import argparse
import bisect
import math
import os
import pickle
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.dataset import load_dataset                # noqa: E402
from ctim.entropy import mean_entropy                # noqa: E402
from ctim.influence import MIA, EdgeWeights          # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ThetaSwapModel(object):
    """Minimal stand-in for ``ctim.gibbs.Model`` carrying a substitute ``theta``.

    ``EdgeWeights.__init__`` reads exactly four attributes (``pi``, ``eta``,
    ``theta``, ``p_z_given_i``); this class supplies them so section [4] can
    build a second EdgeWeights without ever mutating -- and therefore without
    any risk of re-pickling -- the cached 28MB model.  Same pattern as the
    ``_M`` shim in ctim/influence.py's self-test block.
    """

    __slots__ = ("pi", "eta", "theta", "p_z_given_i")

    def __init__(self, pi, eta, theta, p_z_given_i):
        self.pi = pi
        self.eta = eta
        self.theta = theta
        self.p_z_given_i = p_z_given_i


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of an already-sorted list.  q in [0,1]."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _sharpen(row, mass=0.9):
    """Put `mass` on the row's argmax and spread `1-mass` uniformly elsewhere.

    Not a paper quantity -- a synthetic counterfactual for the section [4]
    mechanism test.  Ties in the argmax break on the lowest index so the result
    is deterministic without consulting any RNG.
    """
    K = len(row)
    if K == 0:
        return []
    if K == 1:
        return [1.0]
    best = 0
    for k in range(1, K):
        if row[k] > row[best]:
            best = k
    rest = (1.0 - mass) / (K - 1)
    out = [rest] * K
    out[best] = mass
    return out


def _depth_map(parent):
    """Hop-depth of every node in an arborescence given ``x -> next hop``.

    The root is the unique node absent from `parent`'s key set that is reachable
    by following the chain; its depth is 0.  Memoised so the whole tree costs
    O(|tree|).  A malformed (cyclic) parent map would loop, so the walk is
    length-capped and any node hitting the cap is reported at the cap.
    """
    depth = {}
    cap = len(parent) + 2
    for x in parent:
        if x in depth:
            continue
        chain = []
        cur = x
        steps = 0
        while cur in parent and cur not in depth and steps < cap:
            chain.append(cur)
            cur = parent[cur]
            steps += 1
        base = depth.get(cur, 0)
        for node in reversed(chain):
            base += 1
            depth[node] = base
    return depth


def fmt(s):
    return "%.2fs" % s


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--cache", default=os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-items", type=int, default=8,
                   help="how many items to check thetabar on in section [1]")
    p.add_argument("--n-sample", type=int, default=300,
                   help="nodes sampled for the MIIA depth survey in section [3]")
    p.add_argument("--sharpen-mass", type=float, default=0.9,
                   help="mass placed on each theta row's argmax in section [4]")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    rng = random.Random(args.seed)

    print("=" * 78)
    print("diagnose_pp.py -- is pp bounded by 1/Z on the cached Digg model?")
    print("seed=%d  n_items=%d  n_sample=%d" % (args.seed, args.n_items, args.n_sample))
    print("=" * 78)

    if not os.path.exists(args.cache):
        print("ERROR: no cached model at %s -- run scripts/calibrate_h.py first."
              % args.cache)
        return 1

    # -- load ---------------------------------------------------------------
    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    with open(args.cache, "rb") as fh:
        blob = pickle.load(fh)
    model = blob["model"]
    test_items = blob["test_items"]
    item = test_items[0]
    t_load = time.perf_counter() - t0
    print("\n[load] dataset+model in %s" % fmt(t_load))
    print("       n_users=%d  n_links=%d  n_items=%d  test item=%d"
          % (ds.n_users, len(ds.edges), ds.n_items, item))

    t0 = time.perf_counter()
    ew = EdgeWeights(model, ds)            # Eq (10)-(12)
    t_ew = time.perf_counter() - t0
    C, Z = ew.C, ew.Z
    ceiling = 1.0 / Z if Z else float("nan")
    print("       EdgeWeights built in %s   C=%d  Z=%d  predicted ceiling 1/Z=%.6f"
          % (fmt(t_ew), C, Z, ceiling))

    # ================================================================ [1] ===
    print("\n" + "-" * 78)
    print("[1] Is thetabar_i[c] == 1/Z?   (Eq (12), item-dependent half)")
    print("-" * 78)
    t0 = time.perf_counter()

    n_items_avail = len(model.p_z_given_i)
    probe = [item]
    pool = [i for i in test_items[1:] if i != item]
    for i in pool[: max(0, args.n_items - 1)]:
        probe.append(i)
    while len(probe) < args.n_items and n_items_avail > 0:
        cand = rng.randrange(n_items_avail)
        if cand not in probe:
            probe.append(cand)

    print("  %-10s %-14s %-14s %-14s %-14s" % ("item", "min_c", "max_c", "mean_c",
                                               "max|tb-1/Z|"))
    worst_dev = 0.0
    for i in probe:
        tb = ew.thetabar(i)                # Eq (12)
        lo, hi = min(tb), max(tb)
        mean = sum(tb) / float(len(tb))
        dev = max(abs(x - ceiling) for x in tb)
        if dev > worst_dev:
            worst_dev = dev
        tag = "%d*" % i if i == item else str(i)
        print("  %-10s %-14.10f %-14.10f %-14.10f %-14.3e" % (tag, lo, hi, mean, dev))
    print("  (* = test_items[0], the item every other experiment uses)")
    print("  worst deviation from 1/Z over all %d items probed: %.6e  (%.6f%% of 1/Z)"
          % (len(probe), worst_dev, 100.0 * worst_dev / ceiling))

    er_theta = mean_entropy(model.theta)
    print("  mean_entropy(model.theta rows): %.8f nats   ln Z = %.8f   "
          "deficit %.6f%% below uniform"
          % (er_theta.mean, er_theta.max_possible,
             100.0 * (er_theta.max_possible - er_theta.mean) / er_theta.max_possible))
    er_pi = mean_entropy(model.pi)
    print("  mean_entropy(model.pi    rows): %.8f nats   ln C = %.8f   "
          "deficit %.6f%% below uniform  (control)"
          % (er_pi.mean, er_pi.max_possible,
             100.0 * (er_pi.max_possible - er_pi.mean) / er_pi.max_possible))
    er_pz = mean_entropy(model.p_z_given_i)
    print("  mean_entropy(P(z|i)      rows): %.8f nats   ln Z = %.8f   "
          "deficit %.6f%% below uniform  (the OTHER factor in Eq (12))"
          % (er_pz.mean, er_pz.max_possible,
             100.0 * (er_pz.max_possible - er_pz.mean) / er_pz.max_possible))
    print("  [1] took %s" % fmt(time.perf_counter() - t0))

    # ================================================================ [2] ===
    print("\n" + "-" * 78)
    print("[2] The pp distribution over all %d directed edges, item %d"
          % (len(ds.edges), item))
    print("-" * 78)
    t0 = time.perf_counter()
    pp = ew.for_item(item)                 # Eq (12)
    t_pp = time.perf_counter() - t0
    print("  built pp in %s" % fmt(t_pp))

    vals = sorted(pp.values())
    n = len(vals)
    mean_pp = sum(vals) / float(n)
    print("  n=%d   min=%.8e   max=%.8e   mean=%.8e   median=%.8e"
          % (n, vals[0], vals[-1], mean_pp, _quantile(vals, 0.5)))
    print("  deciles:")
    for d in range(1, 10):
        print("    p%02d  %.8e" % (10 * d, _quantile(vals, d / 10.0)))
    print("  max(pp) = %.8e   ceiling 1/Z = %.8e   ratio max/ceiling = %.6f"
          % (vals[-1], ceiling, vals[-1] / ceiling))
    print("  BOUND HOLDS: %s   (EdgeWeights clamps=%d, max raw pp=%.8e)"
          % (vals[-1] <= ceiling + 1e-12, ew.n_clamped, ew.max_raw_pp))
    print("  two-hop best case = max(pp)^2 = %.8e  ->  >= h=0.1 ? %s"
          % (vals[-1] ** 2, vals[-1] ** 2 >= 0.1))
    print("  fraction of edges with pp >= h:")
    for h in (0.1, 0.05, 0.01, 0.001):
        # vals is sorted, so a bisect counts the tail in O(log n).
        k = n - bisect.bisect_left(vals, h)
        print("    h=%-8g  %8d / %d  = %.4f%%" % (h, k, n, 100.0 * k / n))
    print("  [2] took %s" % fmt(time.perf_counter() - t0))

    # ================================================================ [3] ===
    print("\n" + "-" * 78)
    print("[3] MIA reachability: |MIIA(v,h)| and arborescence depth, Eq (15)")
    print("-" * 78)
    t0 = time.perf_counter()
    sample = rng.sample(range(ds.n_users), min(args.n_sample, ds.n_users))
    print("  sampled %d nodes with random.Random(%d)" % (len(sample), args.seed))

    for h in (0.1, 0.01, 0.001):
        th = time.perf_counter()
        mia = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)  # Eq (13)-(18)
        sizes = []
        depths = []
        hist = {}
        for v in sample:
            nodes, parent = mia.miia(v)    # Eq (15)
            sizes.append(len(nodes) - 1)   # exclude the root itself
            dmap = _depth_map(parent)
            dmax = max(dmap.values()) if dmap else 0
            depths.append(dmax)
            hist[dmax] = hist.get(dmax, 0) + 1
        mean_sz = sum(sizes) / float(len(sizes))
        nz = sum(1 for s in sizes if s > 0)
        hist_s = "  ".join("depth %d: %d" % (d, hist[d]) for d in sorted(hist))
        print("  h=%-7g mean |MIIA|-1 = %.4f   max |MIIA|-1 = %d   "
              "nodes with any in-reach: %d/%d"
              % (h, mean_sz, max(sizes), nz, len(sizes)))
        print("           max depth over sample = %d   histogram: %s"
              % (max(depths), hist_s))
        print("           (%s)" % fmt(time.perf_counter() - th))
    print("  [3] took %s" % fmt(time.perf_counter() - t0))

    # ================================================================ [4] ===
    print("\n" + "-" * 78)
    print("[4] MECHANISM TEST -- sharpen theta only, rebuild pp, see if the")
    print("    ceiling moves.  *** THIS CHANGES pp: NOT AN MIA COMPARISON. ***")
    print("    No influence value is reported here and none would be valid;")
    print("    the only claim tested is 'theta causes the 1/Z ceiling'.")
    print("-" * 78)
    t0 = time.perf_counter()

    theta_sharp = [_sharpen(row, args.sharpen_mass) for row in model.theta]
    alt_model = _ThetaSwapModel(model.pi, model.eta, theta_sharp, model.p_z_given_i)
    ew2 = EdgeWeights(alt_model, ds)       # Eq (10)-(12), synthetic theta
    tb2 = ew2.thetabar(item)               # Eq (12)
    print("  synthetic theta: mass %.2f on argmax, %.6f on each of the other %d"
          % (args.sharpen_mass, (1.0 - args.sharpen_mass) / (Z - 1), Z - 1))
    er2 = mean_entropy(theta_sharp)
    print("  mean_entropy(synthetic theta) = %.6f nats (was %.6f; ln Z = %.6f)"
          % (er2.mean, er_theta.mean, er2.max_possible))
    print("  thetabar_i[c] now: min=%.8f  max=%.8f  mean=%.8f   (was flat at %.8f)"
          % (min(tb2), max(tb2), sum(tb2) / float(len(tb2)), ceiling))

    pp2 = ew2.for_item(item)               # Eq (12)
    vals2 = sorted(pp2.values())
    k2 = len(vals2) - bisect.bisect_left(vals2, 0.1)
    k1 = len(vals) - bisect.bisect_left(vals, 0.1)
    print("  max(pp) synthetic = %.8e   vs real %.8e   (x%.3f)"
          % (vals2[-1], vals[-1], vals2[-1] / vals[-1] if vals[-1] else float("inf")))
    print("  new ceiling should be max_z theta_sharp = %.6f;  max(pp)/that = %.6f"
          % (args.sharpen_mass, vals2[-1] / args.sharpen_mass))
    print("  mean(pp) synthetic = %.8e   vs real %.8e"
          % (sum(vals2) / float(len(vals2)), mean_pp))
    print("  edges with pp >= 0.1:  synthetic %d (%.4f%%)   real %d (%.4f%%)"
          % (k2, 100.0 * k2 / len(vals2), k1, 100.0 * k1 / len(vals)))
    print("  two-hop best case synthetic = %.8e  ->  >= h=0.1 ? %s"
          % (vals2[-1] ** 2, vals2[-1] ** 2 >= 0.1))
    print("  [4] took %s" % fmt(time.perf_counter() - t0))

    # ---------------------------------------------------------------- [4b] --
    print("\n" + "-" * 78)
    print("[4b] FOLLOW-UP -- sharpen P(z|i) as well.  Eq (12) is a mixture, so a")
    print("     sharp theta re-flattens if P(z|i) is diffuse.  Still NOT an MIA")
    print("     comparison: pp changes again.")
    print("-" * 78)
    t0 = time.perf_counter()
    pz_sharp = [_sharpen(row, args.sharpen_mass) for row in model.p_z_given_i]
    alt2 = _ThetaSwapModel(model.pi, model.eta, theta_sharp, pz_sharp)
    ew3 = EdgeWeights(alt2, ds)            # Eq (10)-(12), synthetic theta + P(z|i)
    tb3 = ew3.thetabar(item)               # Eq (12)
    pp3 = ew3.for_item(item)               # Eq (12)
    vals3 = sorted(pp3.values())
    k3 = len(vals3) - bisect.bisect_left(vals3, 0.1)
    tb1 = ew.thetabar(item)

    # pp(u,v) = sum_c a_u[c]*pi_v[c]*thetabar_i[c] with a_u[c] <= 1 and
    # sum_c pi_v[c] = 1, so max_c thetabar_i[c] is a hard per-item upper bound.
    print("  %-26s %-14s %-14s %-10s %-12s"
          % ("world", "max_c thetabar", "max(pp) obs", "obs/bound", ">=0.1 edges"))
    for name, tbx, vx in (("real (cached model)", tb1, vals),
                          ("sharp theta", tb2, vals2),
                          ("sharp theta + sharp P(z|i)", tb3, vals3)):
        b = max(tbx)
        kx = len(vx) - bisect.bisect_left(vx, 0.1)
        print("  %-26s %-14.8f %-14.8f %-10.4f %d (%.2f%%)"
              % (name, b, vx[-1], vx[-1] / b, kx, 100.0 * kx / len(vx)))
    print("  thetabar with both sharpened: min=%.8f max=%.8f mean=%.8f"
          % (min(tb3), max(tb3), sum(tb3) / float(len(tb3))))
    print("  mean(pp) = %.8e   two-hop best = %.8e  ->  >= h=0.1 ? %s"
          % (sum(vals3) / float(len(vals3)), vals3[-1] ** 2, vals3[-1] ** 2 >= 0.1))
    print("  edges >= 0.1: %d (%.4f%%)" % (k3, 100.0 * k3 / len(vals3)))
    print("  [4b] took %s" % fmt(time.perf_counter() - t0))

    print("\n" + "=" * 78)
    print("TOTAL WALL-CLOCK RUNTIME: %s" % fmt(time.perf_counter() - t_all))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
