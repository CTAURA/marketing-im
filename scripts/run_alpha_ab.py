#!/usr/bin/env python3
"""A/B the Dirichlet prior on ``theta`` (and ``phi``) -- LEVER 2.

WHY THIS EXISTS
---------------
Eq (12) factorises to ``pp(u,v) = sum_c a_u[c] * pi[v][c] * thetabar_i[c]`` with
``a_u[c] <= 1`` and ``sum_c pi[v][c] = 1``, so

    pp(u,v) <= max_c thetabar_i[c],   thetabar_i[c] = sum_z P(z|i) theta[c][z].

If ``theta[c][z] == 1/Z`` for every ``(c,z)`` then ``thetabar_i[c] == 1/Z`` for
every item and the bound collapses to ``pp <= 1/Z = 0.125``.  A two-hop path
then tops out at ``0.0156 < h = 0.1``, so the MIA model of Eq (15)-(18) is
provably single-hop and community decomposition has almost nothing to
decompose.  Prior sessions measured exactly that: ``theta`` sits 0.0017% below
``ln Z``, i.e. it never leaves its prior.

The suspected cause is the prior itself.  The paper sets ``alpha = 50/Z`` and
``omega = 50/Z`` -- *per-cell* values engineered so the **total** Dirichlet mass
of a row is 50 whatever ``Z`` is -- while ``beta = 0.01`` is an **absolute**
per-cell constant.  This Digg preparation supplies ~6.97 attribute tokens per
item, so a row mass of 50 outweighs the likelihood by roughly 7:1 and the
posterior mean of Eq (9) is the prior.  ``psi`` (the one matrix with an absolute
prior) is the control: it learns.

So this script retrains small models with ``alpha`` (and optionally ``omega``)
made absolute, and answers four separate yes/no questions:

    Q1  does a smaller alpha make theta informative?      (entropy gap grows)
    Q2  does it raise the pp ceiling?                     (max pp, frac >= h)
    Q3  does it produce multi-hop MIA?                    (MIIA depth)
    Q4  does it improve held-out likelihood?              (the `valid` split)

THE METRIC TRAP -- READ BEFORE QUOTING ANY NUMBER
-------------------------------------------------
Changing the prior retrains the model, which changes ``pp``, which changes the
*graph* that Eq (18) is evaluated on.  An ``I_h(S)`` of 900 on one graph and 761
on another proves nothing at all: they are spreads on two different graphs.
Every ``I_h(S)`` printed below is therefore labelled NOT-COMPARABLE, and the row
comparison is carried by a metric that *is* well defined across models:

    HELD-OUT PREDICTIVE LOG-LIKELIHOOD on the `valid` split (which nothing else
    in the repo uses).  For a held-out potential-influence triple ``(u, v, i)``
    the candidate set is fixed independently of the model, Eq (12) supplies an
    unnormalised score for each candidate, and the scores are normalised over
    that fixed candidate set into a categorical distribution:

        P(v | u, i) = pp(u,v,i) / sum_{v' in Cand(u)} pp(u,v',i)          Eq (12)

    reported as the mean of ``ln P(v | u, i)``.  Two candidate sets are used:
      * FRIEND: Cand(u) = out_adj[u], every friend of u.  Fully deterministic,
        no sampling, and it is the question that actually matters -- *which* of
        u's friends did u influence on item i.  Uniform baseline = -ln |out_adj[u]|.
      * RAND-20: the true v plus 19 users drawn uniformly by a fixed seeded RNG,
        identical for every variant.  Uniform baseline = -ln 20.
    Higher is better, both are proper normalised likelihoods over identical
    candidate sets, and both ARE comparable across models.

WHAT THIS SCRIPT WRITES
-----------------------
Nothing in the repo.  Trained models are pickled under ``--cache-dir`` (the
scratchpad by default).  ``data/processed/_calib_model.pkl`` is never opened.

Standard library only.  Python 3.9 compatible.  All randomness flows through
explicit ``random.Random`` instances.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import tempfile
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.ctim import ctim_select_seeds                                   # noqa: E402
from ctim.dataset import (build_potential_influence_logs, load_dataset,   # noqa: E402
                          split_logs)
from ctim.entropy import mean_entropy                                     # noqa: E402
from ctim.gibbs import CommunitySampler, Hyper, Model, TopicSampler       # noqa: E402
from ctim.influence import EdgeWeights, MIA                               # noqa: E402


# (label, alpha_abs, omega_abs).  alpha_abs=None / omega_abs=None keeps the
# paper's 50/Z, so row "A" is the control and is bit-for-bit the paper prior.
VARIANTS = [
    ("A_paper",       None, None),
    ("B_alpha0.1",    0.1,  None),
    ("C_alpha0.01",   0.01, None),
    ("D_a0.1_w0.1",   0.1,  0.1),
]


def fmt(s):
    return "%7.2fs" % s


class _FixedWeights:
    """Adapter handed to ``ctim_select_seeds(edge_weights=...)``.

    ``ctim_select_seeds`` calls ``edge_weights.for_item(item)`` itself, which
    would rebuild the same O(E*C) Eq (12) dictionary a second time (~27s per
    variant).  This returns the *already built* dict, so CTIM scores against
    byte-identical ``pp`` -- no approximation, purely a cache.
    """

    __slots__ = ("_pp", "_item")

    def __init__(self, pp, item):
        self._pp = pp
        self._item = item

    def for_item(self, i):
        if i != self._item:
            raise ValueError("_FixedWeights holds item %r, asked for %r"
                             % (self._item, i))
        return self._pp


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_variant(ds, train_logs, C, Z, n_topic, n_comm, seed, sampler,
                  alpha_abs, omega_abs, verbose=True):
    """Both stages of Section 4.1.2, with the optional absolute alpha/omega.

    Deliberately does NOT call ``ctim.gibbs.train_model``: that helper has no
    way to pass the new kwargs through, and adding one would mean editing a
    shared file beyond the single backward-compatible ``Hyper.make`` kwarg this
    task owns.  The sampler calls below are exactly what ``train_model`` does.
    """
    hyper = Hyper.make(C, Z, ds.n_users, ds.n_links, len(train_logs),
                       alpha_abs=alpha_abs, omega_abs=omega_abs)
    if verbose:
        print("    prior: rho=%.4g alpha=%.6g beta=%.4g omega=%.6g   "
              "(row mass: pi=%.1f theta=%.3f phi=%.3f psi=%.3f)"
              % (hyper.rho, hyper.alpha, hyper.beta, hyper.omega,
                 hyper.rho * C, hyper.alpha * Z, hyper.omega * Z,
                 hyper.beta * ds.n_attrs))

    t0 = time.perf_counter()
    ts = TopicSampler(ds.item_attrs, ds.n_attrs, hyper, random.Random(seed + 1))
    ts.run(n_topic)                                   # Eq (1)
    p_z_i = ts.p_z_given_i()                          # Eq (2)
    t_topic = time.perf_counter() - t0

    t0 = time.perf_counter()
    cs = CommunitySampler(ds.n_users, ds.edges, train_logs, p_z_i, hyper,
                          random.Random(seed + 2), sampler=sampler)
    cs.run(n_comm)                                    # Eq (4)-(6)
    t_comm = time.perf_counter() - t0

    model = Model(hyper=hyper,
                  pi=cs.pi(),            # Eq (7)
                  eta=cs.eta(),          # Eq (8)
                  theta=cs.theta(),      # Eq (9)
                  p_z_given_i=p_z_i,     # Eq (2)
                  phi=ts.phi(),          # Eq (2)
                  psi=ts.psi())          # Eq (3)
    return model, t_topic, t_comm


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

def tree_depths(nodes, parent, root):
    """Hop count from each node of an MIIA/MIOA arborescence to its root."""
    memo = {root: 0}
    for x in nodes:
        if x in memo:
            continue
        chain = []
        y = x
        while y not in memo:
            chain.append(y)
            y = parent[y]
        d = memo[y]
        while chain:
            d += 1
            memo[chain.pop()] = d
    return memo


def miia_survey(evaluator, probe_nodes):
    """Mean |MIIA(v,h)| - 1 and the maximum hop depth over a fixed node sample.

    Eq (15).  ``probe_nodes`` is a pre-drawn list so every variant is surveyed
    on exactly the same nodes.
    """
    tot = 0
    mx = 0
    max_depth = 0
    hist = {}
    for v in probe_nodes:
        nodes, parent = evaluator.miia(v)   # Eq (15)
        k = len(nodes) - 1
        tot += k
        if k > mx:
            mx = k
        d = max(tree_depths(nodes, parent, v).values())
        if d > max_depth:
            max_depth = d
        hist[d] = hist.get(d, 0) + 1
    return (tot / float(len(probe_nodes)), mx, max_depth, hist)


def pp_stats(pp, h):
    vals = list(pp.values())
    n = len(vals)
    ge = sum(1 for x in vals if x >= h)
    return {
        "n": n,
        "max": max(vals) if vals else 0.0,
        "mean": (sum(vals) / n) if n else 0.0,
        "frac_ge_h": (ge / float(n)) if n else 0.0,
        "n_ge_h": ge,
    }


def heldout_ll(ew, ds, triples, item_ok, rng_seed, n_rand=19, floor=1e-300):
    """Held-out predictive log-likelihood on the `valid` split.

    For each triple ``(u, v, i)`` Eq (12) scores every candidate and the scores
    are normalised over a model-independent candidate set:

        P(v | u, i) = pp(u,v,i) / sum_{v' in Cand(u)} pp(u,v',i)        # Eq (12)

    Returns a dict with the FRIEND variant (Cand = out_adj[u]) and the RAND-20
    variant (Cand = {v} + 19 uniform users), each against its uniform baseline.
    The RAND-20 candidate sets come from a freshly seeded RNG so they are
    identical for every model compared.
    """
    pi = ew.pi
    C = ew.C
    n_users = ds.n_users
    rng = random.Random(rng_seed)

    ll_f_unif = 0.0
    per_f = []      # per-triple ln P, kept so variants can be compared PAIRED
    per_r = []
    for (u, v, i) in triples:
        if not item_ok(i):
            continue
        a = ew.a_u(u)                 # Eq (11), item-independent half
        tb = ew.thetabar(i)           # Eq (12), item-dependent half
        w = [a[c] * tb[c] for c in range(C)]

        def score(x):
            pi_x = pi[x]
            # Eq (12) = sum_c a_u[c] * pi[x][c] * thetabar_i[c]
            return sum(wc * px for wc, px in zip(w, pi_x))

        s_v = score(v)

        # --- FRIEND candidate set (deterministic) ---
        cand = ds.out_adj[u]
        if len(cand) > 1 and v in cand:
            tot = 0.0
            for x in cand:
                tot += score(x)
            if tot > 0.0:
                per_f.append(math.log(max(s_v / tot, floor)))
                ll_f_unif += -math.log(len(cand))

        # --- RAND-20 candidate set (fixed seeded negatives) ---
        negs = []
        while len(negs) < n_rand:
            x = rng.randrange(n_users)
            if x != v and x != u:
                negs.append(x)
        tot = s_v
        for x in negs:
            tot += score(x)
        if tot > 0.0:
            per_r.append(math.log(max(s_v / tot, floor)))

    n_f = len(per_f)
    n_r = len(per_r)
    return {
        "n_friend": n_f,
        "ll_friend": (sum(per_f) / n_f) if n_f else float("nan"),
        "ll_friend_unif": (ll_f_unif / n_f) if n_f else float("nan"),
        "n_rand": n_r,
        "ll_rand": (sum(per_r) / n_r) if n_r else float("nan"),
        "ll_rand_unif": -math.log(n_rand + 1),
        "per_friend": per_f,
        "per_rand": per_r,
    }


def paired_delta(a, b):
    """Mean of ``b_i - a_i`` with its standard error and t statistic.

    The two vectors are per-triple log-likelihoods on the SAME held-out triples
    in the same order, so the comparison is paired: it removes the (large)
    triple-to-triple variance that would otherwise swamp a ~0.003-nat effect.
    Not a paper equation -- plain textbook paired-sample statistics.
    """
    n = len(a)
    if n != len(b) or n < 2:
        return (float("nan"), float("nan"), float("nan"), n)
    d = [bi - ai for ai, bi in zip(a, b)]
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    return (m, se, (m / se) if se > 0.0 else float("inf"), n)


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    p.add_argument("--C", type=int, default=100)
    p.add_argument("--Z", type=int, default=8)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--h", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topic-iters", type=int, default=20)
    p.add_argument("--comm-iters", type=int, default=10,
                   help="community Gibbs sweeps; 10 (not 20) to fit the 9-minute cap. "
                        "trace_entropy already showed theta/pi are equally flat at sweep "
                        "0 and sweep 30 under the paper prior, so the control's verdict "
                        "does not depend on this")
    p.add_argument("--sampler", default="mh", choices=("exact", "mh"))
    p.add_argument("--max-logs-per-item", type=int, default=20)
    p.add_argument("--delta", type=int, default=30 * 24 * 3600)
    p.add_argument("--item", type=int, default=2369,
                   help="probe item; 2369 is test_items[0] of the cached calibration model")
    p.add_argument("--probe-nodes", type=int, default=250,
                   help="nodes in the MIIA depth survey (task asks for >= 200)")
    p.add_argument("--valid-triples", type=int, default=2000,
                   help="held-out triples subsampled from the `valid` split")
    p.add_argument("--variants", default="",
                   help="comma-separated subset of variant labels; default = all")
    p.add_argument("--cache-dir", default="",
                   help="directory for trained-model pickles (NEVER _calib_model.pkl)")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    print("=" * 78)
    print("LEVER 2 -- absolute alpha/omega A/B   C=%d Z=%d K=%d h=%g item=%d"
          % (args.C, args.Z, args.K, args.h, args.item))
    print("  sweeps: %d topic + %d community, sampler=%s, seed=%d"
          % (args.topic_iters, args.comm_iters, args.sampler, args.seed))
    print("=" * 78)

    wanted = [w.strip() for w in args.variants.split(",") if w.strip()]
    variants = [v for v in VARIANTS if (not wanted or v[0] in wanted)]
    if not variants:
        print("ERROR: --variants matched nothing; known: %s"
              % ", ".join(v[0] for v in VARIANTS))
        return 1

    # ---- data: built ONCE, so every variant sees identical splits ----------
    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    ds = load_dataset(args.dataset)
    logs = build_potential_influence_logs(ds, args.delta,
                                          max_per_item=args.max_logs_per_item, rng=rng)
    train, valid, test = split_logs(logs, rng)
    t_data = time.perf_counter() - t0
    print("[data] U=%d E=%d |D|=%d train=%d valid=%d test=%d attrs/item=%.2f  %s"
          % (ds.n_users, ds.n_links, len(logs), len(train), len(valid), len(test),
             sum(len(a) for a in ds.item_attrs) / float(len(ds.item_attrs)), fmt(t_data)))

    # fixed probe sets, identical across every variant
    probe_nodes = random.Random(args.seed + 77).sample(range(ds.n_users),
                                                       min(args.probe_nodes, ds.n_users))
    vr = random.Random(args.seed + 88)
    vsub = list(valid)
    vr.shuffle(vsub)
    vsub = vsub[:args.valid_triples]
    n_items = len(ds.item_attrs)

    def item_ok(i):
        return 0 <= i < n_items

    print("[probe] %d MIIA nodes, %d held-out valid triples (both fixed across variants)"
          % (len(probe_nodes), len(vsub)))

    # Retrained models go to a scratch directory OUTSIDE the repo.  In
    # particular they never go anywhere near data/processed/_calib_model.pkl,
    # which this script does not open at all.
    cache_dir = os.path.abspath(
        args.cache_dir
        or os.environ.get("CTIM_SCRATCH", "")
        or os.path.join(tempfile.gettempdir(), "ctim_alpha_ab_cache"))

    lnZ = math.log(args.Z)
    rows = []

    for (label, alpha_abs, omega_abs) in variants:
        print("\n" + "-" * 78)
        print("VARIANT %s   alpha_abs=%s  omega_abs=%s" % (label, alpha_abs, omega_abs))
        print("-" * 78)
        t_var = time.perf_counter()

        # ---- train (or reload a cached retrain -- NEVER _calib_model.pkl) ---
        tag = "alphaab_%s_C%d_Z%d_t%d_c%d_%s_s%d.pkl" % (
            label, args.C, args.Z, args.topic_iters, args.comm_iters,
            args.sampler, args.seed)
        cpath = os.path.join(cache_dir, tag)
        model = None
        t_topic = t_comm = 0.0
        if not args.no_cache and os.path.exists(cpath):
            t0 = time.perf_counter()
            with open(cpath, "rb") as fh:
                model = pickle.load(fh)
            print("    [train] RELOADED from %s  %s" % (cpath, fmt(time.perf_counter() - t0)))
        if model is None:
            model, t_topic, t_comm = train_variant(
                ds, train, args.C, args.Z, args.topic_iters, args.comm_iters,
                args.seed, args.sampler, alpha_abs, omega_abs)
            print("    [train] stage1 %s  stage2 %s" % (fmt(t_topic), fmt(t_comm)))
            if not args.no_cache:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(cpath, "wb") as fh:
                        pickle.dump(model, fh, protocol=4)
                except OSError as exc:
                    print("    [train] cache write skipped: %s" % exc)

        # ---- (3a) entropies -------------------------------------------------
        t0 = time.perf_counter()
        e_theta = mean_entropy(model.theta)
        e_phi = mean_entropy(model.phi)
        e_psi = mean_entropy(model.psi)
        e_pi = mean_entropy(model.pi)
        # P(z|i) is the OTHER factor of thetabar_i (Eq (12)); Eq (9) builds
        # theta from ncz = sum_i n_{c,i} P(z|i), so a flat P(z|i) forces a flat
        # theta no matter how small alpha is.  Measuring it is what separates
        # "alpha is the constraint" from "omega/phi is the constraint".
        e_pzi = mean_entropy(model.p_z_given_i)
        t_ent = time.perf_counter() - t0
        print("    [entropy] theta %.7f / lnZ %.7f  gap %.3e  (%.4f%% below uniform)"
              % (e_theta.mean, e_theta.max_possible,
                 e_theta.max_possible - e_theta.mean,
                 100.0 * (1.0 - e_theta.mean / e_theta.max_possible)))
        print("    [entropy] phi   %.7f / lnZ %.7f  gap %.3e  (%.4f%% below uniform)"
              % (e_phi.mean, e_phi.max_possible,
                 e_phi.max_possible - e_phi.mean,
                 100.0 * (1.0 - e_phi.mean / e_phi.max_possible)))
        print("    [entropy] P(z|i)%.7f / lnZ %.7f  gap %.3e  (%.4f%% below uniform)"
              % (e_pzi.mean, e_pzi.max_possible,
                 e_pzi.max_possible - e_pzi.mean,
                 100.0 * (1.0 - e_pzi.mean / e_pzi.max_possible)))
        print("    [entropy] psi   %.7f / lnF %.7f  (%.4f%% below)  [control: absolute beta]"
              % (e_psi.mean, e_psi.max_possible,
                 100.0 * (1.0 - e_psi.mean / e_psi.max_possible)))
        print("    [entropy] pi    %.7f / lnC %.7f  (%.4f%% below)"
              % (e_pi.mean, e_pi.max_possible,
                 100.0 * (1.0 - e_pi.mean / e_pi.max_possible)))
        print("              %s" % fmt(t_ent))

        # ---- (3b) thetabar ceiling and pp ----------------------------------
        t0 = time.perf_counter()
        ew = EdgeWeights(model, ds)
        tb = ew.thetabar(args.item)                    # Eq (12)
        max_tb = max(tb)
        pp = ew.for_item(args.item)                    # Eq (12), every edge
        t_pp = time.perf_counter() - t0
        st = pp_stats(pp, args.h)
        print("    [Eq 12] max_c thetabar_i[c] = %.9f   (paper ceiling 1/Z = %.6f, x%.3f)"
              % (max_tb, 1.0 / args.Z, max_tb * args.Z))
        print("    [Eq 12] max(pp)=%.9f  mean=%.9f  frac(pp>=%g)=%.4f%% (%d/%d)  clamps=%d  %s"
              % (st["max"], st["mean"], args.h, 100.0 * st["frac_ge_h"],
                 st["n_ge_h"], st["n"], ew.n_clamped, fmt(t_pp)))
        print("    [Eq 12] best 2-hop path = max(pp)^2 = %.9f   (%s h=%g)"
              % (st["max"] ** 2, ">=" if st["max"] ** 2 >= args.h else "<", args.h))

        # ---- (3c) MIA depth survey -----------------------------------------
        t0 = time.perf_counter()
        evaluator = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=args.h)  # Eq (13)-(18)
        t_mia = time.perf_counter() - t0
        t0 = time.perf_counter()
        mean_miia, max_miia, max_depth, hist = miia_survey(evaluator, probe_nodes)
        t_surv = time.perf_counter() - t0
        print("    [Eq 15] mean |MIIA|-1 = %.4f   max = %d   MAX DEPTH = %d   depths %s   "
              "(MIA ctor %s, survey %s)"
              % (mean_miia, max_miia, max_depth,
                 " ".join("%d:%d" % (d, hist[d]) for d in sorted(hist)),
                 fmt(t_mia), fmt(t_surv)))

        # ---- (3d) CTIM I_h(S) -- NOT comparable across rows -----------------
        t0 = time.perf_counter()
        seeds = ctim_select_seeds(model, ds, args.item, args.K, h=args.h,
                                  dp_tiebreak="paper-true",
                                  edge_weights=_FixedWeights(pp, args.item))
        t_sel = time.perf_counter() - t0
        t0 = time.perf_counter()
        spread = evaluator.influence(seeds)            # Eq (18)
        t_inf = time.perf_counter() - t0
        print("    [Eq 18] CTIM K=%d  I_h(S) = %.6f   <<< NOT COMPARABLE ACROSS ROWS "
              "(pp differs) >>>   (select %s, eval %s)"
              % (args.K, spread, fmt(t_sel), fmt(t_inf)))

        # ---- (3e) held-out likelihood -- IS comparable ----------------------
        t0 = time.perf_counter()
        hl = heldout_ll(ew, ds, vsub, item_ok, rng_seed=args.seed + 99)
        t_hl = time.perf_counter() - t0
        print("    [valid] FRIEND  mean ln P(v|u,i) = %+.6f  (uniform %+.6f, lift %+.6f nats"
              ", n=%d)" % (hl["ll_friend"], hl["ll_friend_unif"],
                           hl["ll_friend"] - hl["ll_friend_unif"], hl["n_friend"]))
        print("    [valid] RAND20  mean ln P(v|u,i) = %+.6f  (uniform %+.6f, lift %+.6f nats"
              ", n=%d)   %s" % (hl["ll_rand"], hl["ll_rand_unif"],
                                hl["ll_rand"] - hl["ll_rand_unif"], hl["n_rand"], fmt(t_hl)))

        rows.append({
            "label": label, "alpha": model.hyper.alpha, "omega": model.hyper.omega,
            "H_theta": e_theta.mean, "gap_theta": e_theta.max_possible - e_theta.mean,
            "pct_theta": 100.0 * (1.0 - e_theta.mean / e_theta.max_possible),
            "H_phi": e_phi.mean, "pct_phi": 100.0 * (1.0 - e_phi.mean / e_phi.max_possible),
            "pct_pzi": 100.0 * (1.0 - e_pzi.mean / e_pzi.max_possible),
            "pct_pi": 100.0 * (1.0 - e_pi.mean / e_pi.max_possible),
            "max_tb": max_tb, "max_pp": st["max"], "frac": 100.0 * st["frac_ge_h"],
            "mean_miia": mean_miia, "max_depth": max_depth,
            "spread": spread, "ll_f": hl["ll_friend"], "ll_f_u": hl["ll_friend_unif"],
            "ll_r": hl["ll_rand"], "ll_r_u": hl["ll_rand_unif"],
            "per_f": hl["per_friend"], "per_r": hl["per_rand"],
            "seconds": time.perf_counter() - t_var,
        })
        print("    VARIANT TOTAL %s" % fmt(rows[-1]["seconds"]))

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY   (lnZ = %.6f, 1/Z = %.6f)" % (lnZ, 1.0 / args.Z))
    print("=" * 78)
    hdr = ("%-14s %7s %7s %8s %8s %8s %8s %10s %10s %7s %7s %6s %10s %10s %7s"
           % ("variant", "alpha", "omega", "%th_bel", "%ph_bel", "%pzi_bel",
              "%pi_bel", "max_tbar", "max_pp", "%pp>=h", "|MIIA|", "depth",
              "LL_friend", "LL_rand20", "secs"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-14s %7.4f %7.4f %8.4f %8.4f %8.4f %8.4f %10.6f %10.6f %7.3f %7.3f %6d "
              "%+10.5f %+10.5f %7.1f"
              % (r["label"], r["alpha"], r["omega"], r["pct_theta"], r["pct_phi"],
                 r["pct_pzi"], r["pct_pi"], r["max_tb"], r["max_pp"], r["frac"],
                 r["mean_miia"], r["max_depth"], r["ll_f"], r["ll_r"], r["seconds"]))
    if rows:
        print("%-14s %7s %7s %8s %8s %8s %8s %10s %10s %7s %7s %6s %+10.5f %+10.5f"
              % ("(uniform base)", "", "", "", "", "", "", "", "", "", "", "",
                 rows[0]["ll_f_u"], rows[0]["ll_r_u"]))
    print("\n  I_h(S) at K=%d (NOT COMPARABLE ACROSS ROWS -- each row is a different graph):"
          % args.K)
    for r in rows:
        print("      %-14s %12.6f" % (r["label"], r["spread"]))

    if len(rows) >= 2:
        print("\n  PAIRED held-out delta vs %s, same triples, same order "
              "(THIS is the cross-model-valid comparison):" % rows[0]["label"])
        print("      %-14s %12s %10s %8s   %12s %10s %8s"
              % ("variant", "d_FRIEND", "SE", "t", "d_RAND20", "SE", "t"))
        for r in rows[1:]:
            mf, sef, tf, nf = paired_delta(rows[0]["per_f"], r["per_f"])
            mr, ser, tr, nr = paired_delta(rows[0]["per_r"], r["per_r"])
            print("      %-14s %+12.6f %10.6f %+8.2f   %+12.6f %10.6f %+8.2f"
                  % (r["label"], mf, sef, tf, mr, ser, tr))
        print("      (n=%d paired triples; |t| > 2 is the usual 'not noise' line)"
              % len(rows[0]["per_f"]))

    # ---- the four yes/no answers ------------------------------------------
    if len(rows) >= 2:
        base = rows[0]
        print("\n" + "=" * 78)
        print("THE FOUR ANSWERS   (control = %s)" % base["label"])
        print("=" * 78)
        for r in rows[1:]:
            q1 = r["pct_theta"] > 10.0 * max(base["pct_theta"], 1e-9)
            q2 = r["max_pp"] > base["max_pp"] * 1.05
            q3 = r["max_depth"] > 1
            mf, sef, tf, _nf = paired_delta(base["per_f"], r["per_f"])
            q4 = r["ll_f"] > base["ll_f"] and tf > 2.0
            print("  %-14s Q1 theta informative? %-3s (%.4f%% -> %.4f%% below uniform)"
                  % (r["label"], "YES" if q1 else "NO", base["pct_theta"], r["pct_theta"]))
            print("  %-14s Q2 pp ceiling up?     %-3s (max pp %.6f -> %.6f, x%.3f)"
                  % ("", "YES" if q2 else "NO", base["max_pp"], r["max_pp"],
                     r["max_pp"] / base["max_pp"] if base["max_pp"] else float("nan")))
            print("  %-14s Q3 multi-hop MIA?     %-3s (max MIIA depth %d -> %d)"
                  % ("", "YES" if q3 else "NO", base["max_depth"], r["max_depth"]))
            print("  %-14s Q4 held-out better?   %-3s (LL_friend %+.5f -> %+.5f, "
                  "paired delta %+.6f +- %.6f nats, t=%+.2f)"
                  % ("", "YES" if q4 else "NO", base["ll_f"], r["ll_f"], mf, sef, tf))
            print("")

    print("TOTAL WALL CLOCK %.2fs" % (time.perf_counter() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
