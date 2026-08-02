#!/usr/bin/env python3
"""Track H^(t) of every learned matrix across Gibbs sweeps.

    H^(t) = - sum_{i=1..K} p_i^(t) log p_i^(t),   Hbar = (1/T) sum_t H^(t)

Here ``t`` indexes **Gibbs sweeps**, so ``H^(t)`` is a trajectory rather than a
summary. That distinction is the whole point: a single ``Hbar`` cannot tell
"the sampler has not moved yet" apart from "the sampler converged onto the
uniform prior", and those two have opposite consequences.

  * still descending at the last sweep  -> UNDER-TRAINED; more sweeps help.
  * flat from the first sweep, at ln K  -> PRIOR-DOMINATED; more sweeps cannot
    help, because the posterior mean simply is the prior.

The second case is the one DEVIATIONS.md Section 7 is about: every ``50/X``
prior in the paper places a fixed total Dirichlet mass of 50, while this Digg
preparation supplies ~7 attribute tokens per item and a median of 3
link/log tokens per user. This script measures which case actually holds.

``eta`` is deliberately excluded: ``eta[c'][c]`` is a Bernoulli success
probability per community pair (Eq (8)), not a distribution over ``c``, so a
row-wise entropy would not mean anything.

Nothing in ``ctim/gibbs.py`` is modified -- ``run(1)`` is called repeatedly,
which is exactly one sweep, and the estimator properties are read between calls.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.dataset import (build_potential_influence_logs, load_dataset,  # noqa: E402
                          split_logs)
from ctim.entropy import mean_entropy                                    # noqa: E402
from ctim.gibbs import CommunitySampler, Hyper, TopicSampler             # noqa: E402


def slope(ys):
    """Least-squares slope of ``ys`` against its index (per sweep)."""
    n = len(ys)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(ys) / n
    num = sum((i - mx) * (y - my) for i, y in enumerate(ys))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


def verdict(gaps, lnK):
    """Classify a trajectory of `ln K - H^(t)` gaps.

    The gap is judged **relative to `ln K`**, not in absolute nats: 8e-3 nats is
    negligible against a ceiling of 2.08 but large against one of 0.01, and the
    matrices here span ceilings from 2.08 (K=8) to 4.61 (K=100).
    ``d = (ln K - H) / ln K = 1 - H/ln K`` is the fraction of the maximum
    entropy that the posterior actually gave up.

    Direction is read from the whole trace, never from the tail slope alone:
    at these magnitudes a per-sweep slope of 1e-5 is sampler noise.
    """
    if len(gaps) < 3 or lnK <= 0:
        return "too few sweeps to judge"
    d_final = gaps[-1] / lnK
    d_first = gaps[0] / lnK
    # Below 1% of the ceiling the distribution is uniform for every purpose that
    # matters here: max_i p_i then sits within a fraction of a percent of 1/K,
    # which is exactly the regime the Eq (12) bound cannot escape.
    if d_final < 0.01:
        return ("PRIOR-DOMINATED (%.3f%% below uniform; sweeps cannot help)"
                % (100.0 * d_final))
    rel = (d_final - d_first) / max(d_final, 1e-12)
    if abs(rel) < 0.05:
        return "FLAT (%.1f%% below uniform, and not moving)" % (100.0 * d_final)
    if rel > 0:
        s = slope(gaps[max(0, len(gaps) // 2):])
        if s > 0.02 * gaps[-1] / max(len(gaps), 1):
            return ("STILL SEPARATING (%.1f%% below uniform) -- more sweeps help"
                    % (100.0 * d_final))
        return "SEPARATED and converging (%.1f%% below uniform)" % (100.0 * d_final)
    return "COLLAPSING toward uniform (now %.1f%% below)" % (100.0 * d_final)


def report(name, gaps, hbars, lnK, every):
    print("\n  %s   (ceiling ln K = %.5f)" % (name, lnK))
    step = max(1, len(gaps) // 12)
    idx = list(range(0, len(gaps), step))
    if idx[-1] != len(gaps) - 1:
        idx.append(len(gaps) - 1)
    print("    sweep :" + "".join("%9d" % (i * every) for i in idx))
    print("    H^(t) :" + "".join("%9.5f" % hbars[i] for i in idx))
    print("    lnK-H :" + "".join("%9.2e" % gaps[i] for i in idx))
    print("    Hbar over all sweeps = %.6f   final gap = %.3e   tail slope = %+.2e"
          % (sum(hbars) / len(hbars), gaps[-1], slope(gaps[max(0, len(gaps) // 2):])))
    print("    --> %s" % verdict(gaps, lnK))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--C", type=int, default=100)
    p.add_argument("--Z", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iters", type=int, default=30, help="sweeps per stage")
    p.add_argument("--snap-every", type=int, default=1,
                   help="record H^(t) every N sweeps (pi is a U x C read, so N>1 is faster)")
    p.add_argument("--max-logs-per-item", type=int, default=20, help="0 = uncapped")
    p.add_argument("--delta", type=int, default=30 * 24 * 3600)
    p.add_argument("--sampler", default="mh", choices=("exact", "mh"))
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("=" * 78)
    print("ENTROPY TRACE over Gibbs sweeps   C=%d Z=%d iters=%d sampler=%s seed=%d"
          % (args.C, args.Z, args.iters, args.sampler, args.seed))
    print("=" * 78)

    rng = random.Random(args.seed)
    t0 = time.perf_counter()
    ds = load_dataset(args.dataset)
    logs = build_potential_influence_logs(ds, args.delta,
                                          max_per_item=args.max_logs_per_item, rng=rng)
    train, _valid, _test = split_logs(logs, rng)
    print("[data] U=%d E=%d |D|=%d train=%d attrs/item=%.2f   %.2fs"
          % (ds.n_users, ds.n_links, len(logs), len(train),
             sum(len(a) for a in ds.item_attrs) / float(len(ds.item_attrs)),
             time.perf_counter() - t0))

    hyper = Hyper.make(args.C, args.Z, ds.n_users, ds.n_links, len(train))
    print("[prior] rho=%.4g alpha=%.4g beta=%.4g omega=%.4g  "
          "(total Dirichlet mass: pi=%.1f theta=%.1f phi=%.1f psi=%.3f)"
          % (hyper.rho, hyper.alpha, hyper.beta, hyper.omega,
             hyper.rho * args.C, hyper.alpha * args.Z, hyper.omega * args.Z,
             hyper.beta * ds.n_attrs))

    every = max(1, args.snap_every)

    # ---------------- stage 1: item topics, Eq (1)-(3) --------------------
    print("\n" + "-" * 78)
    print("STAGE 1 -- item topics (Eq (1)-(3))")
    print("-" * 78)
    ts = TopicSampler(ds.item_attrs, ds.n_attrs, hyper, random.Random(args.seed + 1))
    tracks = {"phi[i][z]": ([], []), "psi[z][f]": ([], [])}
    t0 = time.perf_counter()
    n_snap = args.iters // every
    for s in range(n_snap + 1):
        if s:
            ts.run(every)
        for nm, mat in (("phi[i][z]", ts.phi()), ("psi[z][f]", ts.psi())):
            rep = mean_entropy(mat)
            tracks[nm][0].append(rep.mean)
            tracks[nm][1].append(rep.max_possible - rep.mean)
    t_s1 = time.perf_counter() - t0
    for nm in ("phi[i][z]", "psi[z][f]"):
        hb, gp = tracks[nm]
        report(nm, gp, hb, hb[0] + gp[0], every)
    print("\n  stage 1 traced in %.2fs" % t_s1)

    # ---------------- stage 2: communities, Eq (4)-(9) --------------------
    print("\n" + "-" * 78)
    print("STAGE 2 -- communities (Eq (4)-(9))")
    print("-" * 78)
    cs = CommunitySampler(ds.n_users, ds.edges, train, ts.p_z_given_i(), hyper,
                          random.Random(args.seed + 2), sampler=args.sampler)
    tracks2 = {"theta[c][z]": ([], []), "pi[v][c]": ([], [])}
    t0 = time.perf_counter()
    for s in range(n_snap + 1):
        if s:
            cs.run(every)
        for nm, mat in (("theta[c][z]", cs.theta()), ("pi[v][c]", cs.pi())):
            rep = mean_entropy(mat)
            tracks2[nm][0].append(rep.mean)
            tracks2[nm][1].append(rep.max_possible - rep.mean)
    t_s2 = time.perf_counter() - t0
    for nm in ("theta[c][z]", "pi[v][c]"):
        hb, gp = tracks2[nm]
        report(nm, gp, hb, hb[0] + gp[0], every)
    print("\n  stage 2 traced in %.2fs" % t_s2)
    print("  (eta excluded: eta[c'][c] is a Bernoulli per pair, Eq (8), not a"
          " distribution over c)")

    # ---------------- summary --------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("  %-13s %12s %12s   %s"
          % ("matrix", "final gap", "tail slope", "verdict"))
    for nm, (hb, gp) in list(tracks.items()) + list(tracks2.items()):
        lnK = hb[0] + gp[0]
        print("  %-13s %12.3e %+12.2e   %s"
              % (nm, gp[-1], slope(gp[max(0, len(gp) // 2):]), verdict(gp, lnK)))
    print("\n  A gap of ~0 at sweep 0 that never grows means the posterior mean")
    print("  never left the prior: training is not the binding constraint.")
    print("\nTOTAL %.2fs" % (time.perf_counter() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
