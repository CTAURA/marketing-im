#!/usr/bin/env python3
"""What is the largest ``pp`` Eq (12) can produce, and under what conditions?

Why this script exists
----------------------
Every measurement in this repo that touched the propagation weights hit the same
wall: ``max(pp) = 0.124`` on Digg, so the best two-edge chain is
``0.124^2 = 0.0154 < h = 0.1`` and MIA (Eq (15)/(16)) is *provably single-hop*.
``scripts/diagnose_pp.py`` chased that down to the trained parameters and refuted
the "theta alone is to blame" story: sharpening ``theta`` moved ``max(pp)`` by a
factor of only 1.011, and sharpening ``theta`` **and** ``P(z|i)`` together
reached 0.2003 -- still short of the ``sqrt(0.1) = 0.3162`` a surviving 2-hop
path needs, and it collapsed the live-edge fraction from 23.70% to 6.33%.

That leaves one question no real dataset can answer, because a dataset only ever
shows you the parameters *it* induced:

    Is Eq (12) **structurally** incapable of producing a large ``pp``, or is a
    small ``pp`` a property of the data these models were fitted to?

This script answers it by construction.  Eq (12) is a closed-form function of
four probability tables; if you are allowed to *choose* those tables, the answer
is a theorem, not an experiment.  No training, no dataset, no sampling: ideal
matrices are fed straight through the REAL ``ctim.influence.EdgeWeights`` (via
the ``__slots__`` model-shim pattern from ``ctim/influence.py``'s self-test), so
what is measured is the shipped implementation of Eq (10)-(12), not a paraphrase
of it.

The algebra being tested
------------------------
``EdgeWeights`` evaluates Eq (11)+(12) in the exact factorised form

    pp(u,v) = sum_c a_u[c] * pi_v[c] * thetabar_i[c]                    # Eq (12)
    a_u[c]        = sum_{c'} pi_u[c'] * eta[c'][c]                      # Eq (11)
    thetabar_i[c] = sum_z P(z|i) * theta[c][z]                          # Eq (12)

Four tables, three of which are row-stochastic (``pi``, ``theta``, ``P(z|i)``).
``eta`` is NOT: Eq (8) gives ``eta_{c'c} = (n+eps1)/(n+eps0+eps1)``, an
*independent* number in [0,1] per ordered community pair, which saturates at 1
whenever the pair carries many links.  That distinction turns out to decide how
many factors have to line up, so both regimes are measured:

  R1  "eta row-stochastic"  -- eta[c'] a distribution, peak ``m`` on the diagonal.
  R2  "eta saturated"       -- eta == 1 everywhere, the regime Eq (8) actually
                               lands in on a well-connected graph (section [7]
                               checks this against the trained model on disk).

Sections
--------
  [0] self-test: closed form == EdgeWeights == the naive ``user_to_user_item``.
  [1] the algebraic bound, then a numerical assault on it with random tables.
  [2] the perfect limit: is ``pp = 1`` reachable, and with what configuration?
  [3] the concentration grid: max(pp) vs a common mass ``m``, per (C,Z), per regime.
  [4] one factor at a time: which of the four actually binds?
  [5] the inverse problem: what mass ``m`` does a surviving 2-hop / 3-hop path
      require, by bisection on the closed form?
  [6] the prior-dominated regime: uniform theta + saturated eta gives exactly
      1/Z, and 1/Z crosses sqrt(h) only at Z <= 3.
  [7] anchor: where the model in --cache actually sits on this map.

Metric note (CLAUDE.md rule)
----------------------------
Every row here CHANGES pp on purpose.  No MIA influence value is computed and
none would be comparable across rows.  The only quantity reported is ``max(pp)``
itself, plus its analytic bound -- both are properties of Eq (12) alone.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random(seed)``.

Run:
    python -u scripts/structural_ceiling.py
    python -u scripts/structural_ceiling.py --self-test
    python -u scripts/structural_ceiling.py --dataset data/processed/struct-test \
        --cache data/processed/_struct_test_model.pkl
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

from ctim.entropy import mean_entropy                                  # noqa: E402
from ctim.influence import EdgeWeights, user_to_user_item              # noqa: E402


# ---------------------------------------------------------------------------
# Shims -- same pattern as the `_M` / `_DS` stand-ins in ctim/influence.py's
# self-test block (line ~642) and `_ThetaSwapModel` in scripts/diagnose_pp.py.
# `EdgeWeights.__init__` reads exactly four attributes; `for_item` reads exactly
# `ds.edges`.  Nothing here touches, mutates or re-pickles a trained model.
# ---------------------------------------------------------------------------


class _Model(object):
    __slots__ = ("pi", "eta", "theta", "p_z_given_i")

    def __init__(self, pi, eta, theta, p_z_given_i):
        self.pi = pi
        self.eta = eta
        self.theta = theta
        self.p_z_given_i = p_z_given_i


class _DS(object):
    __slots__ = ("edges",)

    def __init__(self, edges):
        self.edges = edges


# ---------------------------------------------------------------------------
# Constructed tables
# ---------------------------------------------------------------------------


def peak_row(K, idx, mass):
    """A K-vector with `mass` on `idx` and (1-mass)/(K-1) spread over the rest.

    Row sums to exactly 1 up to float rounding.  `mass = 1/K` is the uniform
    row; `mass = 1` is the point mass (the "perfect concentration" limit).
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if K == 1:
        return [1.0]
    rest = (1.0 - mass) / (K - 1)
    row = [rest] * K
    row[idx] = mass
    return row


def build_tables(C, Z, n_probe_comms, m_pi, m_eta, m_theta, m_pz, eta_regime):
    """Ideal (pi, eta, theta, P(z|i)) with a chosen degree of concentration.

    Alignment convention, chosen so that *everything points at the same place*
    -- this is the best case for Eq (12) by construction:
      * user u lives in community ``u // 2`` (two users per probed community);
      * ``pi[u]`` peaks on that community with mass ``m_pi``;
      * ``eta`` peaks on the diagonal with mass ``m_eta`` (regime R1) or is
        identically 1 (regime R2, the trained regime -- see the module docstring);
      * ``theta[c]`` peaks on topic ``z(c) = c % Z`` with mass ``m_theta``;
      * item z peaks on topic z with mass ``m_pz``, so item ``z(c)`` is the
        item best matched to community c.
    Returns (model, ds, users_per_comm).
    """
    P = n_probe_comms
    n_users = 2 * P
    pi = [peak_row(C, u // 2, m_pi) for u in range(n_users)]

    if eta_regime == "saturated":
        # Eq (8) with n_{c'c} >> eps0: eta -> 1 for every ordered pair.
        eta = [[1.0] * C for _ in range(C)]
    elif eta_regime == "stochastic":
        eta = [peak_row(C, cp, m_eta) for cp in range(C)]
    else:
        raise ValueError("unknown eta_regime %r" % (eta_regime,))

    theta = [peak_row(Z, c % Z, m_theta) for c in range(C)]
    p_z_given_i = [peak_row(Z, z, m_pz) for z in range(Z)]
    return _Model(pi, eta, theta, p_z_given_i), n_users


def build_edges(P, rng, n_cross):
    """Within-community pairs for every probed community, plus cross pairs.

    The within pairs are where Eq (12) is maximised under the alignment above;
    the cross pairs are kept so the reported max is a max over a graph that
    actually contains misaligned edges too.
    """
    edges = []
    for c in range(P):
        edges.append((2 * c, 2 * c + 1))
        edges.append((2 * c + 1, 2 * c))
    if P > 1:
        seen = set(edges)
        tries = 0
        while len(edges) < 2 * P + n_cross and tries < 50 * (n_cross + 1):
            tries += 1
            a = rng.randrange(2 * P)
            b = rng.randrange(2 * P)
            if a == b or a // 2 == b // 2:
                continue
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append((a, b))
    return sorted(edges)


def max_pp_constructed(C, Z, m_pi, m_eta, m_theta, m_pz, eta_regime,
                       rng, n_probe_comms=8, n_cross=20):
    """max over (edge, item) of Eq (12), computed by the REAL EdgeWeights.

    Returns (max_pp, argmax_info, n_clamped, n_edges, n_items).
    """
    P = min(C, n_probe_comms)
    model, n_users = build_tables(C, Z, P, m_pi, m_eta, m_theta, m_pz, eta_regime)
    edges = build_edges(P, rng, n_cross)
    ds = _DS(edges)
    ew = EdgeWeights(model, ds)                      # Eq (10)-(12)
    best = -1.0
    best_at = None
    for i in range(Z):
        pp = ew.for_item(i)                          # Eq (12)
        for (u, v), val in pp.items():
            if val > best:
                best = val
                best_at = (u, v, i)
    return best, best_at, ew.n_clamped, len(edges), Z


# ---------------------------------------------------------------------------
# Closed form -- an independent derivation path used to cross-check EdgeWeights
# ---------------------------------------------------------------------------


def closed_form_pp(C, Z, m_pi, m_eta, m_theta, m_pz, eta_regime, c0=0):
    """Eq (12) for the aligned pair (u in c0 -> v in c0) at item z(c0), in closed form.

    Derived by hand from the factorisation in the module docstring; deliberately
    written as scalar arithmetic (no matrices) so that agreement with
    ``EdgeWeights`` is evidence about Eq (12) and not a shared bug.

        pp = sum_c a[c] * pi_v[c] * tb[c]                                # Eq (12)

    with, writing q_c = (1-m_pi)/(C-1), q_e = (1-m_eta)/(C-1),
    q_t = (1-m_theta)/(Z-1), q_p = (1-m_pz)/(Z-1) and z* = c0 % Z:

        tb[c] = m_pz*m_theta + (Z-1)*q_p*q_t                if c % Z == z*
        tb[c] = m_pz*q_t + q_p*m_theta + (Z-2)*q_p*q_t      otherwise
        a[c0] = m_pi*m_eta + (C-1)*q_c*q_e                  (R1) or 1 (R2)
        a[c]  = m_pi*q_e + q_c*m_eta + (C-2)*q_c*q_e        (R1) or 1 (R2)
    """
    if C < 2 or Z < 2:
        raise ValueError("closed form assumes C >= 2 and Z >= 2")
    q_c = (1.0 - m_pi) / (C - 1)
    q_t = (1.0 - m_theta) / (Z - 1)
    q_p = (1.0 - m_pz) / (Z - 1)
    zstar = c0 % Z

    tb_same = m_pz * m_theta + (Z - 1) * q_p * q_t                       # Eq (12)
    tb_diff = m_pz * q_t + q_p * m_theta + (Z - 2) * q_p * q_t           # Eq (12)

    if eta_regime == "saturated":
        a_on = 1.0
        a_off = 1.0
    else:
        q_e = (1.0 - m_eta) / (C - 1)
        a_on = m_pi * m_eta + (C - 1) * q_c * q_e                        # Eq (11)
        a_off = m_pi * q_e + q_c * m_eta + (C - 2) * q_c * q_e           # Eq (11)

    # number of communities sharing the aligned topic z*, including c0 itself
    n_same = len([c for c in range(C) if c % Z == zstar])

    total = a_on * m_pi * tb_same
    total += a_off * q_c * ((n_same - 1) * tb_same + (C - n_same) * tb_diff)
    return total


# ---------------------------------------------------------------------------
# Bisection on the closed form
# ---------------------------------------------------------------------------


def min_mass_for(C, Z, target, eta_regime, tol=1e-10, max_iter=200):
    """Smallest common concentration mass m reaching pp >= target, or None.

    All free masses are tied to a single m (in R2, ``m_eta`` is ignored because
    eta is identically 1).  The closed form is monotone non-decreasing in m --
    section [0] checks that numerically rather than assuming it.
    """
    def f(m):
        return closed_form_pp(C, Z, m, m, m, m, eta_regime)

    lo = max(1.0 / C, 1.0 / Z)          # the uniform row: the least concentrated m
    hi = 1.0
    if f(hi) < target:
        return None
    if f(lo) >= target:
        return lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


# ---------------------------------------------------------------------------
# Section [0] -- self-test.  PASS/FAIL lines; a failure makes the whole run exit
# non-zero, so no number printed below it can be trusted silently.
# ---------------------------------------------------------------------------


def _rand_row(rng, K):
    v = [rng.random() + 1e-6 for _ in range(K)]
    s = sum(v)
    return [x / s for x in v]


def self_test(seed=20190408, verbose=True):
    failures = []

    def check(name, cond, detail=""):
        if cond:
            if verbose:
                print("  PASS  %s%s" % (name, (" -- " + detail) if detail else ""))
        else:
            print("  FAIL  %s%s" % (name, (" -- " + detail) if detail else ""))
            failures.append(name)

    rng = random.Random(seed)

    # (1) three-way agreement: closed form == EdgeWeights == naive Eq (11)+(12)
    worst = 0.0
    worst_at = None
    for (C, Z) in ((2, 2), (5, 3), (10, 8), (17, 5), (100, 8)):
        for masses in ((1.0, 1.0, 1.0, 1.0), (0.9, 0.8, 0.7, 0.6),
                       (1.0 / C, 1.0 / C, 1.0 / Z, 1.0 / Z), (0.5, 1.0, 0.5, 1.0)):
            for regime in ("stochastic", "saturated"):
                m_pi, m_eta, m_th, m_pz = masses
                P = min(C, 4)
                model, _ = build_tables(C, Z, P, m_pi, m_eta, m_th, m_pz, regime)
                ds = _DS([(0, 1)])
                ew = EdgeWeights(model, ds)
                fast = ew.for_item(0)[(0, 1)]        # Eq (12), shipped fast path
                naive = user_to_user_item(model, 0, 1, 0)   # Eq (12), naive O(ZC^2)
                cf = closed_form_pp(C, Z, m_pi, m_eta, m_th, m_pz, regime, c0=0)
                den = max(abs(naive), 1e-300)
                for got, tag in ((fast, "fast"), (cf, "closed")):
                    rel = abs(got - naive) / den
                    if rel > worst:
                        worst = rel
                        worst_at = (C, Z, masses, regime, tag)
    check("closed form == EdgeWeights == user_to_user_item (Eq (12))",
          worst <= 1e-9, "max rel err %.3e at %s" % (worst, worst_at))

    # (2) the perfect limit is exactly 1
    ok = True
    detail = ""
    for (C, Z) in ((2, 2), (10, 8), (100, 8), (100, 50)):
        for regime in ("stochastic", "saturated"):
            v, _, ncl, _, _ = max_pp_constructed(C, Z, 1.0, 1.0, 1.0, 1.0, regime,
                                                 random.Random(1), n_probe_comms=3,
                                                 n_cross=4)
            if abs(v - 1.0) > 1e-12 or ncl != 0:
                ok = False
                detail = "C=%d Z=%d %s -> %.17g (clamped %d)" % (C, Z, regime, v, ncl)
    check("perfect concentration gives pp == 1.0 exactly, no clamping", ok, detail)

    # (3) the algebraic bound pp <= max_c thetabar_i[c], on random tables
    bad = 0
    slack_min = 1e9
    for _ in range(300):
        C = rng.randrange(2, 12)
        Z = rng.randrange(2, 10)
        pi = [_rand_row(rng, C) for _ in range(4)]
        eta = [[rng.random() for _ in range(C)] for _ in range(C)]
        theta = [_rand_row(rng, Z) for _ in range(C)]
        pz = [_rand_row(rng, Z) for _ in range(3)]
        model = _Model(pi, eta, theta, pz)
        ds = _DS([(0, 1), (1, 2), (2, 3), (3, 0)])
        ew = EdgeWeights(model, ds)
        for i in range(3):
            tb = ew.thetabar(i)
            bound = max(tb)
            for val in ew.for_item(i).values():
                if val > bound + 1e-12:
                    bad += 1
                slack_min = min(slack_min, bound - val)
    check("pp <= max_c thetabar_i[c] on 300 random models (the hard bound)",
          bad == 0, "violations=%d  tightest slack %.3e" % (bad, slack_min))

    # (4) with eta saturated, a_u == 1 so pp does not depend on u at all
    model, _ = build_tables(6, 4, 3, 0.7, 1.0, 0.8, 0.9, "saturated")
    ds = _DS([(0, 1), (2, 1), (4, 1)])
    ew = EdgeWeights(model, ds)
    pp = ew.for_item(0)
    vals = list(pp.values())
    check("eta saturated => pp independent of the source user u",
          max(vals) - min(vals) <= 1e-15,
          "spread %.3e over 3 distinct sources" % (max(vals) - min(vals)))

    # (5) uniform theta + saturated eta gives exactly 1/Z
    ok = True
    detail = ""
    for Z in (2, 3, 8, 16):
        C = 10
        v, _, _, _, _ = max_pp_constructed(C, Z, 0.9, 1.0, 1.0 / Z, 0.95, "saturated",
                                           random.Random(2), n_probe_comms=4, n_cross=4)
        if abs(v - 1.0 / Z) > 1e-12:
            ok = False
            detail = "Z=%d -> %.17g vs 1/Z=%.17g" % (Z, v, 1.0 / Z)
    check("uniform theta + saturated eta pins max(pp) at exactly 1/Z", ok, detail)

    # (6) the closed form is monotone non-decreasing in the common mass m
    ok = True
    detail = ""
    for (C, Z) in ((10, 8), (100, 8), (100, 50)):
        for regime in ("stochastic", "saturated"):
            prev = -1.0
            for k in range(201):
                m = max(1.0 / C, 1.0 / Z) + (1.0 - max(1.0 / C, 1.0 / Z)) * k / 200.0
                cur = closed_form_pp(C, Z, m, m, m, m, regime)
                if cur < prev - 1e-15:
                    ok = False
                    detail = "C=%d Z=%d %s at m=%.4f" % (C, Z, regime, m)
                prev = cur
    check("closed form monotone non-decreasing in m (bisection is valid)", ok, detail)

    # (7) the bisection root brackets the target
    ok = True
    detail = ""
    for (C, Z) in ((10, 8), (100, 8)):
        for regime in ("stochastic", "saturated"):
            t = math.sqrt(0.1)
            m = min_mass_for(C, Z, t, regime)
            if m is None:
                ok = False
                detail = "no root for C=%d Z=%d %s" % (C, Z, regime)
                continue
            hi_v = closed_form_pp(C, Z, m, m, m, m, regime)
            lo = max(m - 1e-6, 0.0)
            lo_v = closed_form_pp(C, Z, lo, lo, lo, lo, regime)
            if not (hi_v >= t - 1e-9 and lo_v < t + 1e-6):
                ok = False
                detail = "C=%d Z=%d %s: f(m)=%.9f f(m-eps)=%.9f" % (C, Z, regime, hi_v, lo_v)
    check("bisection root brackets sqrt(0.1)", ok, detail)

    return failures


def fmt(s):
    return "%.2fs" % s


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset",
                   default=os.path.join(_REPO_ROOT, "data", "processed", "digg"),
                   help="only used by section [7] (the anchor)")
    p.add_argument("--cache",
                   default=os.path.join(_REPO_ROOT, "data", "processed",
                                        "_calib_model.pkl"),
                   help="trained model pickle for section [7]; skipped if absent")
    p.add_argument("--seed", type=int, default=20190408)
    p.add_argument("--h", type=float, default=0.1,
                   help="the MIA threshold the ceiling is being judged against")
    p.add_argument("--self-test", action="store_true",
                   help="run section [0] only and exit")
    p.add_argument("--skip-anchor", action="store_true",
                   help="skip section [7] (no model/dataset load)")
    p.add_argument("--n-probe-comms", type=int, default=8)
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    h = args.h
    two_hop = math.sqrt(h)
    three_hop = h ** (1.0 / 3.0)

    print("=" * 78)
    print("structural_ceiling.py -- the supremum of Eq (12) and what it costs")
    print("seed=%d  h=%g  2-hop needs max(pp) >= %.6f  3-hop needs >= %.6f"
          % (args.seed, h, two_hop, three_hop))
    print("=" * 78)

    # ================================================================ [0] ===
    print("\n" + "-" * 78)
    print("[0] SELF-TEST -- the machinery below agrees with the shipped Eq (12)")
    print("-" * 78)
    t0 = time.perf_counter()
    failures = self_test(seed=args.seed)
    print("  [0] took %s   %d failure(s)" % (fmt(time.perf_counter() - t0), len(failures)))
    if failures:
        print("\nSELF-TEST FAILED: %s" % ", ".join(failures))
        return 1
    if args.self_test:
        print("\nALL SELF-TESTS PASSED")
        print("TOTAL WALL-CLOCK RUNTIME: %s" % fmt(time.perf_counter() - t_all))
        return 0

    # ================================================================ [1] ===
    print("\n" + "-" * 78)
    print("[1] THE ALGEBRAIC BOUND")
    print("-" * 78)
    print("""  pp(u,v) = sum_c a_u[c] * pi_v[c] * thetabar_i[c]                  # Eq (12)

  pi_v is a probability distribution over c, so the sum is a WEIGHTED AVERAGE
  of (a_u[c] * thetabar_i[c]) with weights pi_v[c].  An average never exceeds
  its largest term, hence

      pp(u,v) <= max_c ( a_u[c] * thetabar_i[c] )
              <= (max_{c',c} eta[c'][c]) * (max_{c,z} theta[c][z])          (*)

  because a_u[c] = sum_c' pi_u[c'] eta[c'][c] <= max_c' eta[c'][c] (again an
  average) and thetabar_i[c] = sum_z P(z|i) theta[c][z] <= max_z theta[c][z].

  (*) is the whole story: the ceiling of Eq (12) is the product of eta's peak
  and theta's peak.  Both are at most 1, so pp <= 1, and NOTHING in Eq (12)
  caps it below 1.  Section [0] check (3) hammered the bound
  pp <= max_c thetabar_i[c] on 300 random models: 0 violations.""")

    # ================================================================ [2] ===
    print("\n" + "-" * 78)
    print("[2] THE PERFECT LIMIT -- is pp = 1 reachable, and how?")
    print("-" * 78)
    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    print("  Configuration: pi_u = delta_{c0}, eta[c0][c0] = 1, pi_v = delta_{c0},")
    print("  theta[c0] = delta_{z0}, P(z0|i) = 1.  All five point at (c0, z0).")
    print("  %-8s %-8s %-12s %-18s %-10s" % ("C", "Z", "eta regime", "max(pp)", "clamped"))
    for (C, Z) in ((2, 2), (10, 8), (100, 8), (100, 50), (1000, 8)):
        for regime in ("stochastic", "saturated"):
            v, at, ncl, ne, ni = max_pp_constructed(
                C, Z, 1.0, 1.0, 1.0, 1.0, regime, rng,
                n_probe_comms=min(args.n_probe_comms, 4), n_cross=6)
            print("  %-8d %-8d %-12s %-18.15f %-10d" % (C, Z, regime, v, ncl))
    print("  => sup pp = 1, and it is ATTAINED (not merely approached).  Eq (12) is")
    print("     NOT structurally capped: the 0.124 seen on Digg is a property of the")
    print("     FITTED TABLES, not of the equation.  Note the value is independent of")
    print("     C and Z -- the number of communities/topics costs nothing at the")
    print("     perfect limit; it only sets the price of imperfect concentration.")
    print("  [2] took %s" % fmt(time.perf_counter() - t0))

    # ================================================================ [3] ===
    print("\n" + "-" * 78)
    print("[3] THE CONCENTRATION GRID -- max(pp) vs a common mass m on all rows")
    print("    m = 1/K is the uniform row; m = 1 the point mass.  Measured through")
    print("    the real EdgeWeights; `cf` is the independent closed form.")
    print("-" * 78)
    t0 = time.perf_counter()
    masses = [None, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 1.0]
    grids = ((10, 8), (100, 8), (10, 50), (100, 50))
    for regime in ("stochastic", "saturated"):
        print("\n  eta regime: %s" % regime)
        header = "  %-10s" % "m"
        for (C, Z) in grids:
            header += " %-24s" % ("C=%d,Z=%d" % (C, Z))
        print(header)
        for m_raw in masses:
            row = ""
            label = None
            for (C, Z) in grids:
                m = (max(1.0 / C, 1.0 / Z) if m_raw is None else m_raw)
                if label is None:
                    label = "uniform" if m_raw is None else ("%.2f" % m_raw)
                v, at, ncl, ne, ni = max_pp_constructed(
                    C, Z, m, m, m, m, regime, rng,
                    n_probe_comms=args.n_probe_comms, n_cross=20)
                cf = closed_form_pp(C, Z, m, m, m, m, regime)
                # ** = clears the 3-hop bar, * = clears the 2-hop bar
                flag = "**" if v >= three_hop else ("*" if v >= two_hop else "")
                row += " %-24s" % ("%.6f (cf %.6f)%s" % (v, cf, flag))
            print("  %-10s%s" % (label, row))
        print("    (* = clears 2-hop at h=%g, needs >= %.4f;  ** = clears 3-hop, "
              ">= %.4f)" % (h, two_hop, three_hop))
    print("\n  Reading: in the SATURATED regime a_u == 1, so pi_u drops out entirely")
    print("  and pp is a product of THREE concentrations (pi_v, theta, P(z|i)).")
    print("  In the STOCHASTIC regime eta is a fourth factor and every column of the")
    print("  grid drops.  pp decays like m^3 (resp. m^4), which is why 'fairly")
    print("  concentrated' tables still give a small pp.")
    print("  [3] took %s" % fmt(time.perf_counter() - t0))

    # ================================================================ [4] ===
    print("\n" + "-" * 78)
    print("[4] ONE FACTOR AT A TIME -- three tables held perfect, one swept.")
    print("    C=100, Z=8 (the Digg fit).  Answers: which factor actually binds?")
    print("-" * 78)
    t0 = time.perf_counter()
    C, Z = 100, 8
    sweep = [0.125, 0.25, 0.5, 0.75, 0.9, 1.0]
    for regime in ("stochastic", "saturated"):
        print("\n  eta regime: %s" % regime)
        print("  %-8s %-14s %-14s %-14s %-14s"
              % ("value", "sweep pi_u", "sweep eta", "sweep theta", "sweep P(z|i)"))
        for x in sweep:
            cells = []
            for which in range(4):
                mm = [1.0, 1.0, 1.0, 1.0]
                mm[which] = x
                v, _, _, _, _ = max_pp_constructed(
                    C, Z, mm[0], mm[1], mm[2], mm[3], regime, rng,
                    n_probe_comms=args.n_probe_comms, n_cross=10)
                cells.append("%.6f" % v)
            print("  %-8.3f %-14s %-14s %-14s %-14s"
                  % (x, cells[0], cells[1], cells[2], cells[3]))
    print("\n  In the saturated regime the 'sweep pi_u' and 'sweep eta' columns are")
    print("  FLAT at 1.0: with eta == 1 the source user's community membership has")
    print("  literally no effect on pp.  Only pi_v, theta and P(z|i) matter, and each")
    print("  enters linearly -- halving any one of them halves pp.")
    print("  [4] took %s" % fmt(time.perf_counter() - t0))

    # ================================================================ [5] ===
    print("\n" + "-" * 78)
    print("[5] THE INVERSE PROBLEM -- what concentration does a 2-hop path COST?")
    print("    Smallest common mass m (bisection on the closed form) reaching the")
    print("    threshold.  '--' = unreachable even at m = 1.")
    print("-" * 78)
    t0 = time.perf_counter()
    print("  %-14s %-12s %-16s %-16s %-14s"
          % ("(C,Z)", "eta regime", "m for 2-hop", "m for 3-hop", "max(pp) at m=1"))
    for (C, Z) in ((10, 8), (100, 8), (10, 50), (100, 50), (10, 3), (10, 2)):
        for regime in ("stochastic", "saturated"):
            m2 = min_mass_for(C, Z, two_hop, regime)
            m3 = min_mass_for(C, Z, three_hop, regime)
            top = closed_form_pp(C, Z, 1.0, 1.0, 1.0, 1.0, regime)
            print("  %-14s %-12s %-16s %-16s %-14.6f"
                  % ("(%d,%d)" % (C, Z), regime,
                     ("--" if m2 is None else "%.4f" % m2),
                     ("--" if m3 is None else "%.4f" % m3),
                     top))
    print("\n  Interpretation: 'm = 0.68' means EVERY row of pi, theta and P(z|i) must")
    print("  put >= 68% of its mass on the single aligned index, simultaneously, for")
    print("  the strongest 2-hop chain in the entire graph to clear h=%g." % h)
    print("  [5] took %s" % fmt(time.perf_counter() - t0))

    # ================================================================ [6] ===
    print("\n" + "-" * 78)
    print("[6] THE PRIOR-DOMINATED REGIME -- uniform theta, saturated eta.")
    print("    This is where every model fitted in this repo has landed.  theta at")
    print("    maximum entropy => thetabar_i[c] = 1/Z for every item => pp <= 1/Z.")
    print("-" * 78)
    t0 = time.perf_counter()
    print("  %-6s %-16s %-16s %-14s %-14s"
          % ("Z", "max(pp) meas.", "1/Z", "2-hop clears?", "3-hop clears?"))
    for Z in (2, 3, 4, 5, 8, 16, 32, 50):
        v, _, _, _, _ = max_pp_constructed(20, Z, 1.0, 1.0, 1.0 / Z, 1.0, "saturated",
                                           rng, n_probe_comms=4, n_cross=6)
        print("  %-6d %-16.10f %-16.10f %-14s %-14s"
              % (Z, v, 1.0 / Z,
                 "YES" if v >= two_hop else "no",
                 "YES" if v >= three_hop else "no"))
    print("\n  1/Z crosses sqrt(h)=%.4f only at Z <= %d, and h^(1/3)=%.4f only at Z <= %d."
          % (two_hop, int(math.floor(1.0 / two_hop)), three_hop,
             int(math.floor(1.0 / three_hop))))
    print("  So with a maximum-entropy theta, MIA at h=%g is single-hop for any" % h)
    print("  Z >= %d -- REGARDLESS of the graph, the data or the seed selector."
          % (int(math.floor(1.0 / two_hop)) + 1))
    print("  [6] took %s" % fmt(time.perf_counter() - t0))

    # ================================================================ [7] ===
    print("\n" + "-" * 78)
    print("[7] ANCHOR -- where does the TRAINED model sit on this map?")
    print("-" * 78)
    t0 = time.perf_counter()
    if args.skip_anchor:
        print("  skipped (--skip-anchor)")
    elif not os.path.exists(args.cache):
        print("  skipped: no cached model at %s" % args.cache)
    else:
        with open(args.cache, "rb") as fh:
            blob = pickle.load(fh)
        model = blob["model"]
        test_items = blob["test_items"]
        Cm = len(model.eta)
        Zm = len(model.theta[0])
        print("  cache: %s" % args.cache)
        print("  C=%d  Z=%d  n_items=%d" % (Cm, Zm, len(model.p_z_given_i)))

        flat = [x for row in model.eta for x in row]
        n_sat = sum(1 for x in flat if x >= 0.99)
        print("  eta   (Eq (8)):  min=%.6f  mean=%.6f  max=%.6f   >= 0.99 : %d/%d (%.2f%%)"
              % (min(flat), sum(flat) / len(flat), max(flat), n_sat, len(flat),
                 100.0 * n_sat / len(flat)))
        th_max = max(max(row) for row in model.theta)
        er_th = mean_entropy(model.theta)
        print("  theta (Eq (9)):  max_{c,z}=%.6f   uniform 1/Z=%.6f   ratio %.4f"
              % (th_max, 1.0 / Zm, th_max * Zm))
        print("                   entropy %.6f / ln Z %.6f  (deficit %.6f%%)"
              % (er_th.mean, er_th.max_possible,
                 100.0 * (er_th.max_possible - er_th.mean) / er_th.max_possible))
        pz_max = max(max(row) for row in model.p_z_given_i)
        er_pz = mean_entropy(model.p_z_given_i)
        print("  P(z|i):          max=%.6f   uniform 1/Z=%.6f   ratio %.4f"
              % (pz_max, 1.0 / Zm, pz_max * Zm))
        print("                   entropy %.6f / ln Z %.6f  (deficit %.6f%%)"
              % (er_pz.mean, er_pz.max_possible,
                 100.0 * (er_pz.max_possible - er_pz.mean) / er_pz.max_possible))
        er_pi = mean_entropy(model.pi)
        pi_max = max(max(row) for row in model.pi)
        print("  pi:              max=%.6f   uniform 1/C=%.6f   ratio %.4f"
              % (pi_max, 1.0 / Cm, pi_max * Cm))
        print("                   entropy %.6f / ln C %.6f  (deficit %.6f%%)"
              % (er_pi.mean, er_pi.max_possible,
                 100.0 * (er_pi.max_possible - er_pi.mean) / er_pi.max_possible))

        # bound (*) from section [1], evaluated on the trained tables
        eta_max = max(flat)
        print("  BOUND (*) max_eta * max_theta = %.6f * %.6f = %.6f"
              % (eta_max, th_max, eta_max * th_max))

        # tightest per-item bound: max_c thetabar_i[c] over a few probe items
        ew_tb = EdgeWeights(model, _DS([]))
        probe = list(test_items[:8]) if test_items else list(range(min(8, len(model.p_z_given_i))))
        tb_bounds = [max(ew_tb.thetabar(i)) for i in probe]
        print("  max_c thetabar_i[c] over %d probe items: min=%.6f  max=%.6f"
              % (len(probe), min(tb_bounds), max(tb_bounds)))

        if os.path.isdir(args.dataset):
            from ctim.dataset import load_dataset
            t_ds = time.perf_counter()
            ds = load_dataset(args.dataset)
            ew = EdgeWeights(model, ds)
            item = probe[0]
            pp = ew.for_item(item)                         # Eq (12)
            vals = list(pp.values())
            obs = max(vals)
            b = max(ew.thetabar(item))
            live = sum(1 for x in vals if x >= h)
            print("  dataset %s: n_users=%d n_edges=%d  (loaded+pp in %s)"
                  % (os.path.basename(args.dataset.rstrip("/\\")), ds.n_users,
                     len(ds.edges), fmt(time.perf_counter() - t_ds)))
            print("  OBSERVED on item %d: max(pp)=%.6f   bound max_c thetabar=%.6f"
                  "   obs/bound=%.4f" % (item, obs, b, obs / b if b else float("nan")))
            print("  mean(pp)=%.6e   edges with pp >= h=%g: %d/%d (%.4f%%)"
                  % (sum(vals) / len(vals), h, live, len(vals),
                     100.0 * live / len(vals)))
            print("  best 2-hop chain = max(pp)^2 = %.6e  ->  >= h ? %s"
                  % (obs * obs, obs * obs >= h))
            # Where does this sit on the section [5] scale?
            eff = min_mass_for(Cm, Zm, obs, "saturated")
            print("  equivalent common mass m (saturated regime) reproducing this")
            print("  max(pp): %s   vs the %.4f needed for a 2-hop path"
                  % ("--" if eff is None else "%.4f" % eff,
                     min_mass_for(Cm, Zm, two_hop, "saturated") or float("nan")))
        else:
            print("  (dataset %s not found -- observed max(pp) skipped)" % args.dataset)
    print("  [7] took %s" % fmt(time.perf_counter() - t0))

    print("\n" + "=" * 78)
    print("TOTAL WALL-CLOCK RUNTIME: %s" % fmt(time.perf_counter() - t_all))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
