"""AIR+CGA baseline: topic-aware AIR diffusion model + CGA community detection & selection.

SPEC.md Section 7, Table 2 -- the ``AIR+CGA`` row:

    "the AIR topic diffusion model of Barbieri et al. [35] (learned by
     Expectation-Maximization) combined with CGA's [22] community detection
     and seed selection."

[35] Nicola Barbieri, Francesco Bonchi, Giuseppe Manco.
     "Topic-aware social influence propagation models", ICDM 2012.
[22] Yu Wang, Gao Cong, Guojie Song, Kunqing Xie.
     "Community-based greedy algorithm for mining top-K influential nodes in
     mobile social networks", KDD 2010.


The AIR variant of the model
============================

Barbieri et al. propose two topic-aware Independent-Cascade families.  The
first (TIC) keeps a free per-edge, per-topic parameter ``p_z(u,v)``, which
costs ``|E| * Z`` parameters and overfits badly on sparse propagation data.
The second -- **AIR**, for **A**uthoritativeness / **I**nterest /
**R**elevance -- factorises that per-edge parameter into two *per-user* vectors,
which is the variant implemented here (and the one named in the task):

    A[u][z]   authoritativeness of u on topic z   ("how persuasive is u about z")
    I[v][z]   interest / susceptibility of v      ("how receptive is v to z")
    gamma[i]  topic membership (relevance) of item i, a distribution over z

giving ``|U| * 2Z + |M| * Z`` parameters instead of ``|E| * Z``.

    AIR Eq (A1)   p_z(u,v) = A[u][z] * I[v][z]
    AIR Eq (A2)   p_i(u,v) = sum_z gamma[i][z] * p_z(u,v)

Eq (A1) is the exact AIR form of [35]: the activation of ``v`` by ``u`` on
topic ``z`` is the *conjunction of two independent Bernoulli events* -- ``u``
speaks authoritatively about ``z`` (probability ``A[u][z]``) **and** ``v`` is
interested in ``z`` (probability ``I[v][z]``).  That conjunction reading is not
cosmetic: it is what makes the EM below have a genuinely closed-form M step
(see the derivation), and it is how [35] motivates the factorisation.

Note that ``sum_z gamma[i][z] = 1``, so Eq (A2) is a *mixture* over topics and
therefore so is its complement:

    AIR Eq (A3)   1 - p_i(u,v) = sum_z gamma[i][z] * (1 - A[u][z] * I[v][z])

which is used below to give the negative examples the same latent-topic
treatment as the positive ones, with no approximation.


Fitting by Expectation-Maximization
===================================

Exactly the setting of [35]: the training potential-influence logs are the
**positive** examples and the graph edges that carried no log are the
**negative** examples.

    positives  D+ = the TRAIN split of the potential-influence logs (u,v,i):
               u adopted i, then v adopted i within Delta, and (u,v) in E.
               -> the trial on edge (u,v) for item i SUCCEEDED.
    negatives  D- = (u,v,i) with u an adopter of i and (u,v) in E, but no
               log (u,v,i): u had the opportunity to influence v and did not.
               -> the trial on edge (u,v) for item i FAILED.

Objective (a MAP log-posterior; the priors are conjugate and keep every
parameter strictly interior, which is what makes the fixed point well defined
for users that appear in very few trials):

    L = sum_{D+} log p_i(u,v)
      + sum_{D-} log (1 - p_i(u,v))
      + sum_{u,z} [ pa*log A[u][z] + pb*log(1-A[u][z]) ]      Beta(1+pa, 1+pb)
      + sum_{v,z} [ pa*log I[v][z] + pb*log(1-I[v][z]) ]      Beta(1+pa, 1+pb)
      + sum_{i,z}   pg*log gamma[i][z]                        Dirichlet(1+pg)

*Complete data.*  Each trial carries three latent quantities: the topic ``z``
it happened under, and the two Bernoulli indicators ``X_a`` (u's authority
fired) and ``X_b`` (v's interest fired).  The trial succeeds iff
``X_a = X_b = 1``.  Under that augmentation the complete-data log-likelihood
of one trial is

    log gamma[i][z] + X_a log A[u][z] + (1-X_a) log(1-A[u][z])
                    + X_b log I[v][z] + (1-X_b) log(1-I[v][z])

which is separable in A, I and gamma -- hence the closed-form M step.

**E step** -- responsibility of topic z for each observed propagation:

    AIR Eq (A4)  positive trial:
                 r_d[z] = gamma[i][z] * A[u][z] * I[v][z] / p_i(u,v)
                 and given z the success forces X_a = X_b = 1, so
                 E[X_a . 1(z)] = E[X_b . 1(z)] = r_d[z]

    AIR Eq (A5)  negative trial (uses Eq (A3)):
                 s_d[z] = gamma[i][z] * (1 - A[u][z]*I[v][z]) / (1 - p_i(u,v))
                 and given z the failure is one of three cases, so
                 E[X_a . 1(z)] = s_d[z] * A[u][z](1-I[v][z]) / (1-A[u][z]I[v][z])
                 E[X_b . 1(z)] = s_d[z] * (1-A[u][z])I[v][z] / (1-A[u][z]I[v][z])

    (the three failure cases (X_a,X_b) = (1,0),(0,1),(0,0) have probabilities
     A(1-I), (1-A)I, (1-A)(1-I), which sum to exactly 1-AI, so Eq (A5) is a
     normalised posterior -- no bound, no approximation.)

**M step** -- closed form, plain weighted Bernoulli / multinomial MLEs with the
prior pseudo-counts added:

    AIR Eq (A6)  A[u][z] = ( sum_{trials with source u} E[X_a . 1(z)] + pa )
                         / ( sum_{trials with source u} E[1(z)]       + pa + pb )
    AIR Eq (A7)  I[v][z] = ( sum_{trials with target v} E[X_b . 1(z)] + pa )
                         / ( sum_{trials with target v} E[1(z)]       + pa + pb )
    AIR Eq (A8)  gamma[i][z] = ( sum_{trials on item i} E[1(z)] + pg )
                             / ( sum_z ... )

where ``E[1(z)]`` is ``r_d[z]`` on a positive trial and ``s_d[z]`` on a
negative one.  Because this is a bona fide EM (not a bound-and-hope scheme),
``L`` is non-decreasing at every iteration -- the self-test asserts exactly
that.  Iteration stops at ``n_iter`` or when the relative gain in ``L`` falls
below ``tol``.


Community detection and seed selection
======================================

Per SPEC.md Section 7, AIR+CGA uses CGA's community detection and seed
selection.  That community detection is *independent of the diffusion model*
(plain modularity on the friend graph) -- which is precisely the paper's
criticism of the community-based baselines, and the structural reason AIR+CGA
loses to CTIM/CTIM_CGA even though it is topic-aware: its communities know
nothing about how influence actually flows.

The modularity routine is shared with the other community baselines: it is
imported from ``ctim.baselines.cinema`` or ``ctim.baselines.cga`` when either
provides one (see ``_resolve_community_detector``), so that AIR+CGA and CINEMA
genuinely run the SAME method, and only falls back on the local CNM
implementation below when neither sibling exposes one.  Which one was used is
always recorded in ``RunResult.extra["community_source"]``.  Seed selection is
likewise delegated to ``ctim.baselines.cga.cga_select_seeds`` when available.


Cost
====

SPEC.md Section 9 expects AIR+CGA to be the **slowest** method, and it is,
structurally: it pays for (a) an EM fit over ``|D+| + |D-|`` trials times ``Z``
topics for up to ``n_iter`` sweeps, which no other baseline pays at all --
CTIM_CGA reuses one shared Gibbs model -- on top of (b) CGA's MixedGreedy,
whose Monte-Carlo spread estimates it shares with CTIM_CGA.  The reported
``seconds`` therefore covers fit **and** selection; the split is broken out in
``extra["fit_seconds"]`` / ``extra["select_seconds"]``.  The fit is not cached
across calls by default, because the fit is part of the method's cost.

Standard library only.  Python 3.9 compatible.  All randomness flows through
an explicit ``random.Random`` instance.
"""

from __future__ import annotations

import heapq
import math
import os
import random as _random  # instances only (random.Random(seed)); never _random.<fn>()
import sys
import time
from array import array
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Import bootstrap: work both as `ctim.baselines.air_cga` and as a script run
# directly from anywhere (`python3 ctim/baselines/air_cga.py`).
# --------------------------------------------------------------------------
try:
    from ctim.influence import MIA
except ImportError:  # pragma: no cover - direct script execution
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from ctim.influence import MIA

# Share the package-level RunResult so every baseline returns the *same* class.
try:
    from ctim.baselines import RunResult
except Exception:  # pragma: no cover - package init unavailable
    @dataclass
    class RunResult:  # type: ignore[no-redef]
        seeds: list
        spread: float
        seconds: float
        extra: dict = field(default_factory=dict)


__all__ = [
    "AIRModel",
    "build_air_trials",
    "fit_air",
    "air_edge_weights",
    "detect_communities_modularity",
    "modularity",
    "cga_select_seeds_local",
    "air_cga_select_seeds",
    "air_cga_select",
    "select_seeds",
    "RunResult",
]


# Parameters are kept strictly inside (LO, HI); with the Beta priors below the
# MAP fixed point is interior anyway, so this only guards against denormals.
_LO = 1e-9
_HI = 1.0 - 1e-9


def _clamp(x):
    if x < _LO:
        return _LO
    if x > _HI:
        return _HI
    return x


# ==========================================================================
# The fitted AIR model
# ==========================================================================


@dataclass
class AIRModel:
    """Fitted AIR parameters (Barbieri et al. [35], the A/I/R factorisation).

    ``A``, ``I`` and ``gamma`` are *flat* ``array('d')`` buffers rather than
    lists of lists: on Yelp (|U| = 366,715, Z = 8) the nested-list form costs
    ~100 MB of boxed floats where the flat form costs ~23 MB.  Index them as
    ``A[u * Z + z]``, or use the row accessors.

    Attributes
    ----------
    Z          number of topics
    n_users    |U|
    n_items    |M|
    A          A[u*Z+z] -- authoritativeness of u on topic z
    I          I[v*Z+z] -- interest / susceptibility of v in topic z
    gamma      gamma[i*Z+z] -- topic membership (relevance) of item i
    loglik     the MAP log-posterior after each EM sweep (index 0 = at init)
    n_sweeps   number of EM sweeps actually run
    converged  True if the run stopped on `tol` rather than on `n_iter`
    n_pos      number of positive trials used
    n_neg      number of negative trials used
    """

    Z: int
    n_users: int
    n_items: int
    A: array
    I: array
    gamma: array
    loglik: list = field(default_factory=list)
    n_sweeps: int = 0
    converged: bool = False
    n_pos: int = 0
    n_neg: int = 0

    # -- row accessors ------------------------------------------------------

    def A_row(self, u):
        z = self.Z
        return list(self.A[u * z:(u + 1) * z])

    def I_row(self, v):
        z = self.Z
        return list(self.I[v * z:(v + 1) * z])

    def gamma_row(self, i):
        z = self.Z
        return list(self.gamma[i * z:(i + 1) * z])

    # -- the model itself ---------------------------------------------------

    def p_topic(self, u, v, z):
        """AIR Eq (A1): p_z(u,v) = A[u][z] * I[v][z]."""
        Z = self.Z
        return self.A[u * Z + z] * self.I[v * Z + z]  # AIR Eq (A1)

    def pp(self, u, v, i):
        """AIR Eq (A2): p_i(u,v) = sum_z gamma[i][z] * A[u][z] * I[v][z]."""
        Z = self.Z
        A = self.A
        I = self.I
        g = self.gamma
        ub = u * Z
        vb = v * Z
        ib = i * Z
        s = 0.0
        for z in range(Z):
            # AIR Eq (A2), with Eq (A1) inlined
            s += g[ib + z] * A[ub + z] * I[vb + z]
        if s < 0.0:
            return 0.0
        if s > 1.0:
            return 1.0
        return s

    def for_item(self, i, edges):
        """{(u,v): p_i(u,v)} over `edges` -- the AIR analogue of
        ``ctim.influence.EdgeWeights.for_item``."""
        Z = self.Z
        A = self.A
        I = self.I
        g = self.gamma
        ib = i * Z
        gi = [g[ib + z] for z in range(Z)]
        out = {}
        for (u, v) in edges:
            ub = u * Z
            vb = v * Z
            s = 0.0
            for z in range(Z):
                s += gi[z] * A[ub + z] * I[vb + z]  # AIR Eq (A2)
            out[(u, v)] = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        return out


def air_edge_weights(air, ds, item):
    """{(u,v): p_item(u,v)} for every directed edge of G (AIR Eq (A2))."""
    return air.for_item(item, ds.edges)


# ==========================================================================
# Training data: positive and negative trials
# ==========================================================================


def build_air_trials(ds, logs_train, rng, n_neg_per_pos=3, max_neg_per_item=0):
    """Split the training data into AIR's positive and negative trials.

    Exactly the setting of [35] (see the module docstring):

    * **positive** trials are the training potential-influence logs themselves:
      ``(u,v,i)`` means the trial on edge ``(u,v)`` for item ``i`` succeeded;
    * **negative** trials are the graph edges *without* a log: ``u`` adopted
      ``i`` (so a trial on every out-edge of ``u`` took place) but the log
      ``(u,v,i)`` is absent, so that trial failed.

    Enumerating every negative costs ``sum_i sum_{u in adopters(i)} outdeg(u)``,
    which is the same budget as building the logs in the first place, but the
    *number* of negatives is far larger than the number of positives.  We keep
    a uniform reservoir sample of ``n_neg_per_pos`` negatives per positive
    (per item), which is the standard treatment and keeps the EM sweeps linear
    in ``|D+|``.  ``max_neg_per_item > 0`` caps it further.  Set
    ``n_neg_per_pos = 0`` to disable negatives entirely (not recommended: the
    likelihood then has no downward pressure and every probability saturates).

    Returns ``(positives, negatives)``, both lists of ``(u, v, i)``.
    """
    if n_neg_per_pos < 0:
        raise ValueError("n_neg_per_pos must be >= 0, got %r" % (n_neg_per_pos,))
    positives = [tuple(d) for d in logs_train]
    if n_neg_per_pos == 0:
        return positives, []
    if rng is None:
        raise ValueError("build_air_trials requires an explicit random.Random instance")

    pos_set = set(positives)
    pos_by_item = {}
    for (u, v, i) in positives:
        pos_by_item[i] = pos_by_item.get(i, 0) + 1

    out_adj = ds.out_adj

    # Adopters of each item, from the raw adoption logs when we have them.  A
    # user is "at risk" of influencing all of their out-neighbours on an item
    # only once they have themselves adopted it.
    adopters_by_item = {}
    for (u, i, _t) in getattr(ds, "logs", ()) or ():
        if i in pos_by_item:
            bucket = adopters_by_item.get(i)
            if bucket is None:
                bucket = set()
                adopters_by_item[i] = bucket
            bucket.add(u)
    # Fall back on the sources of the positive logs when raw adoptions are not
    # available (synthetic fixtures, or a Dataset carrying only D).
    for (u, _v, i) in positives:
        bucket = adopters_by_item.get(i)
        if bucket is None:
            bucket = set()
            adopters_by_item[i] = bucket
        bucket.add(u)

    negatives = []
    for i in sorted(pos_by_item.keys()):
        cap = n_neg_per_pos * pos_by_item[i]
        if max_neg_per_item > 0:
            cap = min(cap, max_neg_per_item)
        if cap <= 0:
            continue
        kept = []
        seen = 0
        for u in sorted(adopters_by_item.get(i, ())):
            for v in out_adj[u]:
                if (u, v, i) in pos_set:
                    continue  # this trial succeeded; it is a positive
                seen += 1
                # Reservoir sampling (algorithm R): a uniform sample of `cap`
                # of the negatives, without materialising all of them.
                if len(kept) < cap:
                    kept.append((u, v, i))
                else:
                    j = rng.randrange(seen)
                    if j < cap:
                        kept[j] = (u, v, i)
        negatives.extend(kept)

    return positives, negatives


# ==========================================================================
# Expectation-Maximization
# ==========================================================================


def _air_log_posterior(pos, neg, A, I, gamma, Z, n_users, n_items, pa, pb, pg):
    """The MAP objective L of the module docstring.  Must be non-decreasing."""
    total = 0.0
    for (u, v, i) in pos:
        ub, vb, ib = u * Z, v * Z, i * Z
        p = 0.0
        for z in range(Z):
            p += gamma[ib + z] * A[ub + z] * I[vb + z]  # AIR Eq (A2)
        total += math.log(p) if p > _LO else math.log(_LO)
    for (u, v, i) in neg:
        ub, vb, ib = u * Z, v * Z, i * Z
        q = 0.0
        for z in range(Z):
            # AIR Eq (A3): 1 - p_i(u,v) = sum_z gamma[z] (1 - A*I)
            q += gamma[ib + z] * (1.0 - A[ub + z] * I[vb + z])
        total += math.log(q) if q > _LO else math.log(_LO)
    if pa or pb:
        for k in range(n_users * Z):
            a = A[k]
            total += pa * math.log(a) + pb * math.log(1.0 - a)
            b = I[k]
            total += pa * math.log(b) + pb * math.log(1.0 - b)
    if pg:
        for k in range(n_items * Z):
            total += pg * math.log(gamma[k])
    return total


def fit_air(ds, logs_train, Z, rng, n_iter=50, tol=1e-6,
            n_neg_per_pos=3, max_neg_per_item=0,
            prior_a=0.5, prior_b=1.5, prior_gamma=0.5,
            trials=None, verbose=False):
    """Fit AIR (A, I, gamma) by Expectation-Maximization.  See module docstring.

    ``prior_a``/``prior_b`` are the Beta pseudo-counts on every ``A[u][z]`` and
    ``I[v][z]``; the default ``(0.5, 1.5)`` has mean 0.25, i.e. an unseen user
    gets ``p_z = 0.25 * 0.25 ~ 0.06``, a sane Independent-Cascade prior.
    ``prior_gamma`` is the Dirichlet pseudo-count on every ``gamma[i][z]``.

    ``trials`` optionally supplies a pre-built ``(positives, negatives)`` pair
    from :func:`build_air_trials` (so a caller can build them once and refit).

    Returns an :class:`AIRModel` whose ``loglik`` trace records the objective
    after every sweep.
    """
    if rng is None:
        raise ValueError("fit_air requires an explicit random.Random instance")
    Z = int(Z)
    if Z < 1:
        raise ValueError("Z must be >= 1, got %r" % (Z,))

    if trials is None:
        pos, neg = build_air_trials(ds, logs_train, rng,
                                    n_neg_per_pos=n_neg_per_pos,
                                    max_neg_per_item=max_neg_per_item)
    else:
        pos, neg = trials
        pos = [tuple(d) for d in pos]
        neg = [tuple(d) for d in neg]

    n_users = ds.n_users
    n_items = ds.n_items
    pa = float(prior_a)
    pb = float(prior_b)
    pg = float(prior_gamma)

    # ---- initialisation ---------------------------------------------------
    # A and I are drawn around the prior mean; gamma is a random point on the
    # simplex.  Symmetry between topics has to be broken by the draw, otherwise
    # EM sits at the (unstable) uniform fixed point forever.
    mean = pa / (pa + pb) if (pa + pb) > 0 else 0.25
    lo = max(_LO, 0.5 * mean)
    hi = min(_HI, 1.5 * mean)
    A = array('d', [lo + (hi - lo) * rng.random() for _ in range(n_users * Z)])
    I = array('d', [lo + (hi - lo) * rng.random() for _ in range(n_users * Z)])
    gamma = array('d', [0.0] * (n_items * Z))
    for i in range(n_items):
        row = [rng.random() + 1e-3 for _ in range(Z)]
        s = sum(row)
        ib = i * Z
        for z in range(Z):
            gamma[ib + z] = row[z] / s

    loglik = [_air_log_posterior(pos, neg, A, I, gamma, Z,
                                 n_users, n_items, pa, pb, pg)]
    if verbose:
        print("[fit_air] |D+|=%d |D-|=%d Z=%d  L0=%.6f"
              % (len(pos), len(neg), Z, loglik[0]))

    converged = False
    sweeps = 0
    prior_den = pa + pb

    for it in range(int(n_iter)):
        # ---------------- E step: accumulate expected sufficient statistics --
        # num_a[u][z] = sum of E[X_a . 1(z)] over trials with source u
        # den_a[u][z] = sum of E[1(z)]       over trials with source u
        # (and symmetrically for the target side), gsum[i][z] = sum of E[1(z)]
        num_a = array('d', [0.0] * (n_users * Z))
        den_a = array('d', [0.0] * (n_users * Z))
        num_b = array('d', [0.0] * (n_users * Z))
        den_b = array('d', [0.0] * (n_users * Z))
        gsum = array('d', [0.0] * (n_items * Z))

        resp = [0.0] * Z

        for (u, v, i) in pos:
            ub, vb, ib = u * Z, v * Z, i * Z
            tot = 0.0
            for z in range(Z):
                w = gamma[ib + z] * A[ub + z] * I[vb + z]  # AIR Eq (A2) term
                resp[z] = w
                tot += w
            if tot <= 0.0:
                continue
            for z in range(Z):
                # AIR Eq (A4): responsibility of topic z for this propagation.
                r = resp[z] / tot
                # A success forces X_a = X_b = 1, so both numerators get r.
                num_a[ub + z] += r
                den_a[ub + z] += r
                num_b[vb + z] += r
                den_b[vb + z] += r
                gsum[ib + z] += r

        for (u, v, i) in neg:
            ub, vb, ib = u * Z, v * Z, i * Z
            tot = 0.0
            for z in range(Z):
                # AIR Eq (A3): the failure probability is a mixture too.
                w = gamma[ib + z] * (1.0 - A[ub + z] * I[vb + z])
                resp[z] = w
                tot += w
            if tot <= 0.0:
                continue
            for z in range(Z):
                s = resp[z] / tot  # AIR Eq (A5): topic responsibility, failure
                au = A[ub + z]
                iv = I[vb + z]
                fail = 1.0 - au * iv
                if fail <= _LO:
                    continue
                # AIR Eq (A5): split the failure over its three causes.
                num_a[ub + z] += s * (au * (1.0 - iv)) / fail
                num_b[vb + z] += s * ((1.0 - au) * iv) / fail
                den_a[ub + z] += s
                den_b[vb + z] += s
                gsum[ib + z] += s

        # ---------------- M step: closed form ------------------------------
        for k in range(n_users * Z):
            # AIR Eq (A6): weighted Bernoulli MLE + Beta pseudo-counts
            A[k] = _clamp((num_a[k] + pa) / (den_a[k] + prior_den))
            # AIR Eq (A7)
            I[k] = _clamp((num_b[k] + pa) / (den_b[k] + prior_den))
        for i in range(n_items):
            ib = i * Z
            tot = 0.0
            for z in range(Z):
                tot += gsum[ib + z] + pg
            if tot <= 0.0:
                for z in range(Z):
                    gamma[ib + z] = 1.0 / Z
            else:
                for z in range(Z):
                    # AIR Eq (A8): multinomial MLE + Dirichlet pseudo-counts
                    gamma[ib + z] = _clamp((gsum[ib + z] + pg) / tot)

        sweeps = it + 1
        cur = _air_log_posterior(pos, neg, A, I, gamma, Z,
                                 n_users, n_items, pa, pb, pg)
        prev = loglik[-1]
        loglik.append(cur)
        if verbose:
            print("[fit_air] sweep %3d  L=%.6f  (dL=%+.3e)" % (sweeps, cur, cur - prev))
        denom = abs(prev) if abs(prev) > 1.0 else 1.0
        if (cur - prev) / denom < tol:
            converged = True
            break

    return AIRModel(
        Z=Z, n_users=n_users, n_items=n_items,
        A=A, I=I, gamma=gamma,
        loglik=loglik, n_sweeps=sweeps, converged=converged,
        n_pos=len(pos), n_neg=len(neg),
    )


# ==========================================================================
# Community detection -- modularity, independent of the diffusion model
# ==========================================================================


def _undirected_projection(n_users, edges):
    """Symmetrise G into a weighted undirected graph; self-loops dropped.

    A reciprocated pair (u,v)+(v,u) contributes weight 2, which is the usual
    convention when projecting a directed graph for modularity.
    """
    adj = [dict() for _ in range(n_users)]
    for (u, v) in edges:
        if u == v:
            continue
        adj[u][v] = adj[u].get(v, 0.0) + 1.0
        adj[v][u] = adj[v].get(u, 0.0) + 1.0
    return adj


def modularity(comm, n_users, edges):
    """Newman modularity Q of the labelling `comm` on the undirected projection.

        Q = sum_c [ w_in_c / (2m) - (deg_c / (2m))^2 ]
    """
    adj = _undirected_projection(n_users, edges)
    two_m = 0.0
    for u in range(n_users):
        for w in adj[u].values():
            two_m += w
    if two_m <= 0.0:
        return 0.0
    deg = {}
    w_in = {}
    for u in range(n_users):
        cu = comm[u]
        d = 0.0
        for v, w in adj[u].items():
            d += w
            if comm[v] == cu:
                w_in[cu] = w_in.get(cu, 0.0) + w
        deg[cu] = deg.get(cu, 0.0) + d
    q = 0.0
    for c, d in deg.items():
        q += w_in.get(c, 0.0) / two_m - (d / two_m) ** 2
    return q


def detect_communities_modularity(ds, C=100, rng=None):
    """Greedy agglomerative modularity maximisation (Clauset-Newman-Moore).

    This is the *diffusion-model-independent* community detection that CGA (and
    CINEMA) use: it looks only at the friend graph, never at the propagation
    probabilities.  SPEC.md Section 7 makes that independence the defining
    weakness of these baselines relative to CTIM, whose communities come out of
    the latent model itself (Eq (19)).

    Communities start as singletons and the pair with the largest

        dQ(i,j) = 2 * ( w_ij / (2m) - a_i * a_j ),   a_c = deg_c / (2m)

    is merged repeatedly.  With ``C = None`` merging stops when no merge
    improves Q (the natural CNM stopping point).  With an integer ``C`` merging
    continues to exactly ``C`` communities when the graph is connected enough
    to allow it -- the paper fixes ``C = 100`` for AIR+CGA and CINEMA, so the
    target has to be reachable rather than discovered.  Disconnected graphs may
    bottom out above the target; the achieved count is returned to the caller
    via :func:`air_cga_select_seeds`'s ``extra``.

    ``rng`` is accepted for signature compatibility with sibling detectors and
    is unused: the merge order is fully deterministic (ties break on the lower
    community id, then the lower partner id).

    Returns ``comm`` with ``comm[v]`` in ``[0, C_achieved)``.
    """
    n = ds.n_users
    edges = ds.edges
    adj = _undirected_projection(n, edges)

    two_m = 0.0
    for u in range(n):
        for w in adj[u].values():
            two_m += w
    if two_m <= 0.0:
        # No edges at all: every node is its own community.
        return list(range(n))

    # Community state: members, weighted degree, and weights to neighbours.
    members = [[u] for u in range(n)]
    deg = [0.0] * n
    link = [dict() for _ in range(n)]
    for u in range(n):
        d = 0.0
        for v, w in adj[u].items():
            d += w
            link[u][v] = w
        deg[u] = d
    alive = set(range(n))

    def dq(i, j):
        w = link[i].get(j, 0.0)
        # dQ of merging communities i and j
        return 2.0 * (w / two_m - (deg[i] / two_m) * (deg[j] / two_m))

    # ---- CNM's actual data structure: ONE entry per community ------------
    #
    # Merging into `i` changes deg[i] and every w_{i,x}, so it invalidates every
    # heap entry touching `i`.  Because w can rise while a_i*a_j also rises, a
    # stale price is neither a reliable upper nor lower bound, so it cannot be
    # trusted lazily -- it has to be recomputed.
    #
    # Keeping one heap entry per *pair* and refreshing all of them after each
    # merge is what Clauset-Newman-Moore explicitly avoid, and why: on Digg
    # (30,358 nodes) a merged community reaches ~7,000 neighbours, so the heap
    # grows to 27M entries after 12k of the 30k merges and the run never
    # finishes.  Measured push counts: 29.4M and climbing superlinearly.
    #
    # CNM instead keep, for each community, only its BEST partner, and a global
    # heap over those maxima.  The global max over communities is exactly the
    # global max over pairs, so the merge order is unchanged -- only the
    # bookkeeping is.  Entries carry generation stamps for both endpoints; a
    # stale pop recomputes that community's best and re-publishes it rather
    # than being dropped, which keeps the invariant "every live community with
    # a neighbour has at least one entry in the heap".
    gen = [0] * n

    def best_of(i):
        """Community i's highest-dQ partner, ties broken on the lower id."""
        bq = None
        bj = -1
        for j in link[i]:
            q = dq(i, j)
            if bq is None or q > bq or (q == bq and j < bj):
                bq, bj = q, j
        return (bq, bj) if bj >= 0 else (None, -1)

    def entry(i, j, q):
        # Canonical (lower id, higher id) ordering so ties break exactly as a
        # pairwise heap keyed on (-dq, i, j) would, keeping the merge order
        # identical to the naive implementation.
        a, b = (i, j) if i < j else (j, i)
        return (-q, a, b, i, gen[a], gen[b])

    heap = []
    for i in range(n):
        q, j = best_of(i)
        if j >= 0:
            heap.append(entry(i, j, q))
    heapq.heapify(heap)

    target = None if C is None else max(1, int(C))
    n_alive = n

    def republish(owner):
        if owner in alive:
            q, j = best_of(owner)
            if j >= 0:
                heapq.heappush(heap, entry(owner, j, q))

    while heap and (target is None or n_alive > target):
        negq, a, b, owner, ga, gb = heapq.heappop(heap)
        if owner not in alive:
            continue
        if a not in alive or b not in alive or ga != gen[a] or gb != gen[b]:
            republish(owner)  # outdated: re-price this community and retry
            continue
        i, j = (a, b) if a == owner else (b, a)
        if j not in link[i]:
            republish(owner)
            continue
        cur = -negq
        if target is None and cur <= 0.0:
            break  # natural CNM stopping point

        # ---- merge j into i ----
        if len(members[j]) > len(members[i]):
            i, j = j, i  # merge the smaller side, keeps the relabelling cheap
        members[i].extend(members[j])
        members[j] = []
        deg[i] += deg[j]
        alive.discard(j)
        n_alive -= 1

        for x, w in link[j].items():
            if x == i:
                continue
            link[i][x] = link[i].get(x, 0.0) + w
            link[x][i] = link[x].get(i, 0.0) + w
            link[x].pop(j, None)
        link[i].pop(j, None)
        link[j] = {}

        # `i` absorbed `j`, so every entry touching `i` is now outdated.  Bump
        # its stamp and re-publish only i's own best partner -- O(deg) work and
        # ONE push, instead of one push per neighbour.  Neighbours x whose best
        # pointed at i now hold outdated stamps; they are re-priced lazily by
        # `republish` when they surface, which is what bounds the heap.
        gen[i] += 1
        republish(i)

    # Dense, deterministic relabelling: communities ordered by their lowest
    # member id, so the labelling does not depend on the merge order.
    order = sorted(alive, key=lambda c: min(members[c]) if members[c] else c)
    comm = [0] * n
    for new_id, c in enumerate(order):
        for u in members[c]:
            comm[u] = new_id
    return comm


# ==========================================================================
# CGA seed selection (local fallback) -- DP allocation + MixedGreedy
# ==========================================================================


class _CommunityMixedGreedy:
    """MixedGreedy (Chen, Wang, Yang KDD 2009) inside one community subgraph.

    MixedGreedy = **NewGreedy** for the first seed of a community + **CELF**
    (lazy forward) for every subsequent one, which is exactly what CGA [22]
    runs inside the community it picks.

    NewGreedy exploits the fact that one live-edge sample of the subgraph
    yields the spread of *every* singleton at once: sample each edge (u,v)
    independently with probability pp(u,v), then |R(u)| in the sampled graph is
    an unbiased draw of I({u}).  CELF then reuses submodularity to avoid
    re-estimating every candidate's marginal gain at every round.

    Everything is Monte-Carlo -- that is CGA's influence computation model, and
    the reason it is orders of magnitude slower than CTIM's exact MIA.
    """

    def __init__(self, nodes, out_adj, pp, n_mc, rng, n_mc_first=None):
        self.nodes = sorted(nodes)
        nodeset = set(self.nodes)
        self.sub = {}
        for u in self.nodes:
            row = []
            for v in out_adj[u]:
                if v in nodeset:
                    p = pp.get((u, v), 0.0)
                    if p > 0.0:
                        row.append((v, p))
            row.sort()
            self.sub[u] = row
        self.pp = pp
        self.n_mc = int(n_mc)
        self.n_mc_first = int(n_mc_first) if n_mc_first else max(1, int(n_mc) // 4)
        self.rng = rng
        self.S = []
        self._S_set = set()
        self._base = 0.0
        self._heap = None          # CELF heap of (-gain, node, stamp)
        self._pending = None       # cached (node, gain) for the current S
        self._exhausted = False

    # -- Monte-Carlo Independent Cascade ------------------------------------

    def _spread(self, S, n_sims):
        """Mean activated-node count under IC on the community subgraph."""
        if not S:
            return 0.0
        sub = self.sub
        rnd = self.rng.random
        total = 0
        seeds = list(S)
        for _ in range(n_sims):
            active = set(seeds)
            frontier = list(seeds)
            while frontier:
                nxt = []
                for u in frontier:
                    for (v, p) in sub.get(u, ()):
                        if v not in active and rnd() < p:
                            active.add(v)
                            nxt.append(v)
                frontier = nxt
            total += len(active)
        return total / float(n_sims)

    def _newgreedy_singletons(self):
        """One NewGreedy pass: I({u}) for every u, from shared live-edge samples."""
        totals = dict.fromkeys(self.nodes, 0)
        rnd = self.rng.random
        sub = self.sub
        R = self.n_mc_first
        for _ in range(R):
            live = {}
            for u in self.nodes:
                row = [v for (v, p) in sub[u] if rnd() < p]
                if row:
                    live[u] = row
            for u in self.nodes:
                # forward-reachable set of u in the sampled graph
                seen = {u}
                stack = [u]
                while stack:
                    x = stack.pop()
                    for y in live.get(x, ()):
                        if y not in seen:
                            seen.add(y)
                            stack.append(y)
                totals[u] += len(seen)
        return {u: totals[u] / float(R) for u in self.nodes}

    # -- CGA / Algorithm 2 interface ----------------------------------------

    def best_gain(self):
        """max over u in this community of I_m(S_m u {u}) - I_m(S_m).

        Algorithm 2, line 34 (`dI_m`).  The result is cached until
        :meth:`commit` actually grows S_m, because the DP re-reads dI_m for
        every k while S_m only changes when this community is chosen.
        """
        if self._pending is not None:
            return self._pending
        if self._exhausted or not self.nodes:
            self._pending = (None, 0.0)
            return self._pending

        if self._heap is None:
            # ---- NewGreedy round: all singleton spreads in one sweep ----
            singles = self._newgreedy_singletons()
            self._heap = [(-singles[u], u, 0) for u in self.nodes]
            heapq.heapify(self._heap)
            self._base = 0.0

        stamp = len(self.S)
        heap = self._heap
        while heap:
            neg, u, st = heapq.heappop(heap)
            if u in self._S_set:
                continue
            if st == stamp:
                heapq.heappush(heap, (neg, u, st))  # put it back for commit()
                self._pending = (u, -neg)
                return self._pending
            # CELF: stale estimate, re-evaluate against the current S
            g = self._spread(self.S + [u], self.n_mc) - self._base
            if g < 0.0:
                g = 0.0
            heapq.heappush(heap, (-g, u, stamp))
        self._exhausted = True
        self._pending = (None, 0.0)
        return self._pending

    def commit(self, u, gain):
        """Algorithm 2, line 44: S_m <- S_m u {u_k}."""
        heap = self._heap
        if heap:
            # drop u's entry
            self._heap = [e for e in heap if e[1] != u]
            heapq.heapify(self._heap)
        self.S.append(u)
        self._S_set.add(u)
        self._base = self._spread(self.S, self.n_mc)
        self._pending = None


def cga_select_seeds_local(comm, ds, pp, K, rng, n_mc=1000,
                           dp_tiebreak="paper-true", n_mc_first=None):
    """CGA [22] seed selection: DP allocation across communities + MixedGreedy.

    Local fallback used only when ``ctim.baselines.cga.cga_select_seeds`` is not
    importable.  Signature matches the one API.md specifies for that function
    (``cga_select_seeds(comm, ds, pp, K, rng, n_mc)``) plus two keyword extras.

    The dynamic program is Algorithm 2 lines 25-45 of SPEC.md, which is CGA's
    DP verbatim: ``I[m][k] = max(I[m-1][k], I[m][k-1] + dI_m)``, allocating one
    seed per round ``k`` to whichever community the DP's argmax chain points at.

    ``dp_tiebreak`` -- which reading of Algorithm 2 lines 35/36 to run
    (DEVIATIONS.md 1.1):
      * ``"paper-true"``    -- both lines read ``I[C][k-1]``, as printed;
      * ``"consistent"``    -- both lines read ``I[m][k-1]``;
      * ``"paper-literal"`` -- line 35 reads ``I[m][k-1]``, line 36 ``I[C][k-1]``.
    """
    if dp_tiebreak not in ("paper-true", "consistent", "paper-literal"):
        raise ValueError("dp_tiebreak must be 'paper-true', 'consistent' or "
                         "'paper-literal', got %r" % (dp_tiebreak,))
    if K <= 0:
        return []

    C = (max(comm) + 1) if comm else 0
    if C <= 0:
        return []
    groups = [[] for _ in range(C)]
    for v, c in enumerate(comm):
        groups[c].append(v)

    # One deterministic child rng per community, drawn once from the master.
    child_seeds = [rng.randrange(1 << 30) for _ in range(C)]
    states = [
        _CommunityMixedGreedy(groups[c], ds.out_adj, pp, n_mc,
                              _random.Random(child_seeds[c]),
                              n_mc_first=n_mc_first)
        for c in range(C)
    ]

    # Algorithm 2, lines 25-31: initialise S, S_1..S_C and the DP table.
    S = []
    Itab = [[0.0] * (K + 1) for _ in range(C + 1)]
    stab = [[0] * (K + 1) for _ in range(C + 1)]

    for k in range(1, K + 1):  # Algorithm 2, line 32
        for m in range(1, C + 1):  # Algorithm 2, line 33
            # Algorithm 2, line 34: dI_m = max_{u in c_m} I_m(S u u) - I_m(S)
            _u, dI = states[m - 1].best_gain()
            # Algorithm 2, line 35.  As printed the reference is I[C,k-1].
            ref35 = Itab[C][k - 1] if dp_tiebreak == "paper-true" else Itab[m][k - 1]
            cand = ref35 + dI
            Itab[m][k] = cand if cand > Itab[m - 1][k] else Itab[m - 1][k]
            # Algorithm 2, line 36
            if dp_tiebreak == "consistent":
                lhs = Itab[m][k - 1] + dI
            else:
                lhs = Itab[C][k - 1] + dI  # the printed form
            if lhs >= Itab[m - 1][k]:
                stab[m][k] = m  # Algorithm 2, line 37
            else:
                stab[m][k] = stab[m - 1][k]  # Algorithm 2, line 39

        j = stab[C][k]  # Algorithm 2, line 42
        if j <= 0:
            break
        # Algorithm 2, line 43: u_k = argmax_{u in c_j} I(S_j u u) - I(S_j)
        u_k, gain = states[j - 1].best_gain()
        if u_k is None:
            # This community is exhausted; fall back on the best gain anywhere
            # so the DP cannot stall before K seeds are chosen.
            best_m, best_u, best_g = None, None, -1.0
            for m in range(1, C + 1):
                cu, cg = states[m - 1].best_gain()
                if cu is not None and cg > best_g:
                    best_m, best_u, best_g = m, cu, cg
            if best_u is None:
                break
            j, u_k, gain = best_m, best_u, best_g
        states[j - 1].commit(u_k, gain)  # Algorithm 2, line 44
        S.append(u_k)

    return S


# ==========================================================================
# Sibling-module resolution
# ==========================================================================


def _param_names(fn):
    """Positional parameter names of `fn` without importing `inspect`."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return ()
    return tuple(code.co_varnames[:code.co_argcount])


_COMMUNITY_CANDIDATES = (
    # SPEC.md/task: prefer the shared modularity routine wherever it lives, so
    # AIR+CGA and CINEMA genuinely run the SAME community detection.
    ("ctim.baselines.cinema", "detect_communities_modularity"),
    ("ctim.baselines.cinema", "modularity_communities"),
    ("ctim.baselines.cinema", "detect_communities"),
    ("ctim.baselines.cga", "detect_communities_modularity"),
    ("ctim.baselines.cga", "modularity_communities"),
    ("ctim.baselines.cga", "detect_communities"),
)


def _resolve_community_detector():
    """(fn, source) for the shared modularity detector, or the local fallback."""
    import importlib
    for modname, attr in _COMMUNITY_CANDIDATES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        fn = getattr(mod, attr, None)
        if fn is not None and callable(fn):
            return fn, "%s.%s" % (modname, attr)
    return detect_communities_modularity, "ctim.baselines.air_cga.detect_communities_modularity"


def _call_community_detector(fn, ds, C, rng):
    """Call a sibling detector whose exact signature we do not control."""
    if fn is detect_communities_modularity:
        return fn(ds, C=C, rng=rng)
    names = _param_names(fn)
    supply = {
        "ds": ds, "dataset": ds, "graph": ds,
        "n_users": ds.n_users, "edges": ds.edges,
        "C": C, "n_comm": C, "n_communities": C, "target_c": C, "k": C, "K": C,
        "rng": rng,
    }
    args = []
    for nm in names:
        if nm in supply:
            args.append(supply[nm])
        else:
            break  # unknown parameter: stop and rely on its default
    return fn(*args)


def _resolve_cga_selector():
    """(fn, source) for CGA seed selection, or the local fallback."""
    import importlib
    try:
        mod = importlib.import_module("ctim.baselines.cga")
    except Exception:
        return cga_select_seeds_local, "ctim.baselines.air_cga.cga_select_seeds_local"
    fn = getattr(mod, "cga_select_seeds", None)
    if fn is not None and callable(fn):
        return fn, "ctim.baselines.cga.cga_select_seeds"
    return cga_select_seeds_local, "ctim.baselines.air_cga.cga_select_seeds_local"


# ==========================================================================
# The baseline entry point
# ==========================================================================


def air_cga_select_seeds(ds, logs_train, item, K, rng,
                         Z=8, C=100, n_mc=1000,
                         n_em_iter=50, em_tol=1e-6,
                         n_neg_per_pos=3, max_neg_per_item=0,
                         h=0.1, dp_tiebreak="paper-true",
                         air=None, comm=None, evaluator=None,
                         community_detector=None, cga_selector=None,
                         n_mc_first=None, strict=False, verbose=False):
    """AIR+CGA (SPEC.md Section 7): AIR fitted by EM, then CGA detection+selection.

    Parameters
    ----------
    ds            :class:`ctim.dataset.Dataset`
    logs_train    TRAIN split of the potential-influence logs, list of (u,v,i)
    item          the test item whose topic mixture drives the edge weights
    K             number of seeds
    rng           explicit ``random.Random`` (drives EM init, negative sampling
                  and every Monte-Carlo simulation)
    Z             number of AIR topics (paper: 8)
    C             number of communities for the modularity detection (paper: 100)
    n_mc          Monte-Carlo simulations per MixedGreedy spread estimate
    n_em_iter     maximum EM sweeps
    em_tol        relative log-posterior gain at which EM stops
    n_neg_per_pos negatives sampled per positive trial
    h             MIA threshold used by the evaluator (SPEC.md: 0.1)
    air           a pre-fitted :class:`AIRModel` to reuse (skips the fit; the
                  reported ``fit_seconds`` is then 0)
    comm          a precomputed community labelling to reuse
    evaluator     ``f(seeds) -> spread``; defaults to exact MIA (Eq (17)/(18))
                  under AIR's own edge weights.  ``ctim/experiments.py`` passes
                  the shared CTIM evaluator here so all five methods are scored
                  identically (API.md).
    strict        re-raise instead of falling back when a sibling module is
                  present but its call fails

    Returns a :class:`RunResult`.  ``seconds`` covers the EM fit *and* the
    selection, because both are part of what this method costs.
    """
    t_start = time.time()
    if rng is None:
        raise ValueError("air_cga_select_seeds requires an explicit random.Random instance")

    extra = {
        "method": "AIR+CGA",
        "topic_aware": True,
        "community_based": True,
        "item": item,
        "Z": Z,
        "C_requested": C,
        "n_mc": n_mc,
        "h": h,
        "dp_tiebreak": dp_tiebreak,
        "monte_carlo": True,
    }

    # ---- 1. fit AIR by EM (AIR Eq (A4)-(A8)) ------------------------------
    if air is None:
        t0 = time.time()
        air = fit_air(ds, logs_train, Z, rng,
                      n_iter=n_em_iter, tol=em_tol,
                      n_neg_per_pos=n_neg_per_pos,
                      max_neg_per_item=max_neg_per_item,
                      verbose=verbose)
        fit_seconds = time.time() - t0
        extra["air_refit"] = True
    else:
        # A caller-supplied model costs this run nothing, so the fit time is
        # exactly zero rather than a few microseconds of branch overhead.
        fit_seconds = 0.0
        extra["air_refit"] = False
    extra["fit_seconds"] = fit_seconds
    extra["em_sweeps"] = air.n_sweeps
    extra["em_converged"] = air.converged
    extra["em_loglik_final"] = air.loglik[-1] if air.loglik else None
    extra["n_pos_trials"] = air.n_pos
    extra["n_neg_trials"] = air.n_neg

    # ---- 2. edge weights for this item (AIR Eq (A2)) ----------------------
    pp = air_edge_weights(air, ds, item)

    # ---- 3. community detection (modularity, diffusion-model independent) --
    t0 = time.time()
    if comm is None:
        if community_detector is not None:
            fn, src = community_detector, "caller-supplied"
        else:
            fn, src = _resolve_community_detector()
        try:
            comm = _call_community_detector(fn, ds, C, rng)
        except Exception as exc:
            if strict:
                raise
            extra["community_fallback_reason"] = "%s: %s" % (type(exc).__name__, exc)
            fn, src = (detect_communities_modularity,
                       "ctim.baselines.air_cga.detect_communities_modularity")
            comm = detect_communities_modularity(ds, C=C, rng=rng)
        extra["community_source"] = src
    else:
        extra["community_source"] = "caller-supplied"
    detect_seconds = time.time() - t0
    extra["detect_seconds"] = detect_seconds
    extra["C_achieved"] = (max(comm) + 1) if comm else 0
    extra["modularity"] = modularity(comm, ds.n_users, ds.edges)

    # ---- 4. CGA seed selection (DP + MixedGreedy) -------------------------
    t0 = time.time()
    if cga_selector is not None:
        sel, sel_src = cga_selector, "caller-supplied"
    else:
        sel, sel_src = _resolve_cga_selector()
    try:
        if sel is cga_select_seeds_local:
            seeds = cga_select_seeds_local(comm, ds, pp, K, rng, n_mc=n_mc,
                                           dp_tiebreak=dp_tiebreak,
                                           n_mc_first=n_mc_first)
        else:
            # API.md fixes the positional signature
            # cga_select_seeds(comm, ds, pp, K, rng, n_mc) but says nothing about
            # keyword-only extras, so `dp_tiebreak` is offered and withdrawn if
            # the sibling does not accept it.  Passing it matters: without this
            # the flag was recorded in `extra["dp_tiebreak"]` while the sibling
            # silently ran its own default, so a `--dp-tiebreak` run mislabelled
            # AIR+CGA (DEVIATIONS.md 4.5).
            try:
                seeds = sel(comm, ds, pp, K, rng, n_mc, dp_tiebreak=dp_tiebreak)
            except TypeError:
                seeds = sel(comm, ds, pp, K, rng, n_mc)
            if hasattr(seeds, "seeds"):  # sibling may return a RunResult
                seeds = list(seeds.seeds)
    except Exception as exc:
        if strict:
            raise
        extra["cga_fallback_reason"] = "%s: %s" % (type(exc).__name__, exc)
        sel_src = "ctim.baselines.air_cga.cga_select_seeds_local"
        seeds = cga_select_seeds_local(comm, ds, pp, K, rng, n_mc=n_mc,
                                       dp_tiebreak=dp_tiebreak,
                                       n_mc_first=n_mc_first)
    select_seconds = time.time() - t0
    extra["select_seconds"] = select_seconds
    extra["cga_source"] = sel_src

    seeds = list(seeds)[:K]

    # ---- 5. spread, Eq (18) ------------------------------------------------
    if evaluator is None:
        mia = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)
        spread = mia.influence(seeds)  # Eq (18), exact per Eq (17)
        extra["evaluator"] = "MIA(air_pp)"
    else:
        spread = float(evaluator(seeds))
        extra["evaluator"] = "caller-supplied"

    seconds = time.time() - t_start
    return RunResult(seeds=seeds, spread=float(spread), seconds=seconds, extra=extra)


# Aliases: API.md refers to a generic `select_seeds`, ctim/baselines/__init__.py
# exports `air_cga_select_seeds`, and the task text calls it `air_cga_select`.
air_cga_select = air_cga_select_seeds
select_seeds = air_cga_select_seeds


# ==========================================================================
# Self-test
# ==========================================================================

if __name__ == "__main__":
    import random

    from ctim.dataset import Dataset, build_adjacency

    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("  PASS  %s%s" % (name, (" -- " + detail) if detail else ""))
        else:
            print("  FAIL  %s%s" % (name, (" -- " + detail) if detail else ""))
            failures.append(name)

    # ------------------------------------------------------------------
    # A synthetic world with planted communities AND planted topics, so we can
    # check that EM recovers the structure it is supposed to recover.
    #   - n_comm dense blocks of n_per users, sparse links between blocks
    #   - block b is authoritative/interested on topic (b % Z)
    #   - each item is drawn from one dominant topic
    # ------------------------------------------------------------------
    def make_world(rng, n_comm=4, n_per=18, Z=3, n_items=24,
                   p_in=0.30, p_out=0.01, n_adopt=14, delta=100):
        n_users = n_comm * n_per
        block = [u // n_per for u in range(n_users)]
        edges = []
        for u in range(n_users):
            for v in range(n_users):
                if u == v:
                    continue
                p = p_in if block[u] == block[v] else p_out
                if rng.random() < p:
                    edges.append((u, v))
        out_adj, in_adj = build_adjacency(n_users, edges)

        item_topic = [rng.randrange(Z) for _ in range(n_items)]
        logs = []
        t = 1000
        for i in range(n_items):
            zt = item_topic[i]
            # users of the blocks aligned with this topic adopt it, in waves
            pool = [u for u in range(n_users) if (block[u] % Z) == zt]
            if len(pool) < 2:
                pool = list(range(n_users))
            rng.shuffle(pool)
            for u in pool[:n_adopt]:
                logs.append((u, i, t))
                t += 7
            t += 500
        logs.sort(key=lambda r: (r[2], r[0], r[1]))
        ds = Dataset(name="synthetic", n_users=n_users, n_items=n_items,
                     n_attrs=Z, edges=edges, out_adj=out_adj, in_adj=in_adj,
                     logs=logs, item_attrs=[[item_topic[i]] for i in range(n_items)])
        return ds, block, item_topic, delta

    rng = random.Random(20190408)
    ds, block, item_topic, delta = make_world(rng)
    print("world: U=%d E=%d M=%d logs=%d"
          % (ds.n_users, ds.n_links, ds.n_items, ds.n_logs))

    from ctim.dataset import build_potential_influence_logs, split_logs
    D = build_potential_influence_logs(ds, delta=delta)
    train, valid, test = split_logs(D, random.Random(7))
    print("potential-influence logs: |D|=%d  train=%d" % (len(D), len(train)))
    check("Definition 1 produced a non-trivial training set", len(train) >= 50,
          "|train|=%d" % len(train))

    # ---------------------------------------------------------------- test 1
    print("[1] trials: positives are the logs, negatives are log-free edges")
    pos, neg = build_air_trials(ds, train, random.Random(1), n_neg_per_pos=3)
    check("positives == the training logs", pos == [tuple(d) for d in train])
    check("negatives are disjoint from positives", not (set(pos) & set(neg)))
    check("negatives are real graph edges",
          all((u, v) in ds.edge_set() for (u, v, _i) in neg))
    check("negatives respect the per-positive budget",
          0 < len(neg) <= 3 * len(pos), "|D-|=%d for |D+|=%d" % (len(neg), len(pos)))
    p2, n2 = build_air_trials(ds, train, random.Random(1), n_neg_per_pos=3)
    check("negative sampling is deterministic for a fixed seed", (pos, neg) == (p2, n2))
    n3 = build_air_trials(ds, train, random.Random(2), n_neg_per_pos=3)[1]
    check("negative sampling actually depends on the seed", n3 != neg)
    check("n_neg_per_pos=0 disables negatives",
          build_air_trials(ds, train, random.Random(1), n_neg_per_pos=0)[1] == [])

    # ---------------------------------------------------------------- test 2
    print("[2] EM: the log-posterior is monotone non-decreasing (AIR Eq (A4)-(A8))")
    Z = 3
    air = fit_air(ds, train, Z, random.Random(11), n_iter=40, tol=1e-10,
                  n_neg_per_pos=3)
    ll = air.loglik
    worst = 0.0
    for a, b in zip(ll, ll[1:]):
        worst = min(worst, b - a)
    check("EM ran several sweeps", air.n_sweeps >= 5, "sweeps=%d" % air.n_sweeps)
    check("log-posterior never decreases (this is a true EM, not a bound)",
          worst >= -1e-9, "worst step = %+.3e" % worst)
    check("log-posterior strictly improved overall", ll[-1] > ll[0] + 1e-6,
          "L0=%.4f -> L=%.4f" % (ll[0], ll[-1]))

    # ---------------------------------------------------------------- test 3
    print("[3] AIR parameter sanity (AIR Eq (A1)/(A2))")
    check("A in (0,1)", all(0.0 < x < 1.0 for x in air.A))
    check("I in (0,1)", all(0.0 < x < 1.0 for x in air.I))
    ok_g = True
    for i in range(ds.n_items):
        row = air.gamma_row(i)
        if abs(sum(row) - 1.0) > 1e-9 or any(x <= 0.0 for x in row):
            ok_g = False
    check("gamma rows are probability distributions (AIR Eq (A8))", ok_g)
    u0, v0, i0 = ds.edges[0][0], ds.edges[0][1], 0
    naive = sum(air.gamma_row(i0)[z] * air.p_topic(u0, v0, z) for z in range(Z))
    check("pp() == sum_z gamma[i][z] * A[u][z] * I[v][z] (Eq (A1)+(A2))",
          abs(air.pp(u0, v0, i0) - naive) <= 1e-12,
          "%.12f vs %.12f" % (air.pp(u0, v0, i0), naive))
    ew = air_edge_weights(air, ds, i0)
    worst_ew = max(abs(ew[(u, v)] - air.pp(u, v, i0)) for (u, v) in ds.edges)
    check("for_item agrees with pp() on every edge", worst_ew <= 1e-12,
          "max abs err = %.3e over %d edges" % (worst_ew, len(ds.edges)))
    check("all edge weights are valid probabilities",
          all(0.0 <= p <= 1.0 for p in ew.values()))

    # ---------------------------------------------------------------- test 4
    print("[4] EM recovers the planted topic structure")
    # An item on topic zt should score higher on edges inside a block aligned
    # with zt than on edges inside a misaligned block.
    aligned, misaligned = [], []
    for i in range(ds.n_items):
        zt = item_topic[i]
        w = air.for_item(i, ds.edges)
        for (u, v) in ds.edges:
            if block[u] != block[v]:
                continue
            if (block[u] % Z) == zt:
                aligned.append(w[(u, v)])
            else:
                misaligned.append(w[(u, v)])
    ma = sum(aligned) / len(aligned)
    mm = sum(misaligned) / len(misaligned)
    check("topic-aligned edges get higher p_i(u,v) than misaligned ones",
          ma > mm, "aligned=%.5f vs misaligned=%.5f (ratio %.2fx)" % (ma, mm, ma / mm))

    # ---------------------------------------------------------------- test 5
    print("[5] modularity community detection (independent of the diffusion model)")
    comm = detect_communities_modularity(ds, C=4)
    q = modularity(comm, ds.n_users, ds.edges)
    check("labels are dense and in range",
          sorted(set(comm)) == list(range(max(comm) + 1)),
          "C_achieved=%d" % (max(comm) + 1))
    check("modularity Q is substantially positive on a planted-block graph",
          q > 0.2, "Q=%.4f" % q)
    # planted blocks recovered: every block should map to one dominant label
    good = 0
    for b in range(4):
        labs = [comm[u] for u in range(ds.n_users) if block[u] == b]
        top = max(set(labs), key=labs.count)
        if labs.count(top) >= 0.8 * len(labs):
            good += 1
    check("greedy modularity recovers the planted blocks", good == 4,
          "%d/4 blocks recovered" % good)
    check("detection is deterministic",
          detect_communities_modularity(ds, C=4) == comm)
    check("detection ignores the diffusion model (no pp argument exists)",
          "pp" not in _param_names(detect_communities_modularity))
    q_free = modularity(detect_communities_modularity(ds, C=None), ds.n_users, ds.edges)
    check("C=None stops at the natural CNM optimum with Q >= Q(C=4)",
          q_free >= q - 1e-9, "Q_free=%.4f vs Q_4=%.4f" % (q_free, q))

    # ---------------------------------------------------------------- test 6
    print("[6] CGA selection: DP allocation + MixedGreedy")
    K = 6
    seeds = cga_select_seeds_local(comm, ds, ew, K, random.Random(3), n_mc=60)
    check("returns K seeds", len(seeds) == K, "%s" % (seeds,))
    check("seeds are distinct", len(set(seeds)) == len(seeds))
    check("seeds are valid nodes", all(0 <= s < ds.n_users for s in seeds))
    s2 = cga_select_seeds_local(comm, ds, ew, K, random.Random(3), n_mc=60)
    check("CGA selection is deterministic for a fixed seed", seeds == s2)
    lit = cga_select_seeds_local(comm, ds, ew, K, random.Random(3), n_mc=60,
                                 dp_tiebreak="paper-literal")
    check("paper-literal tiebreak also returns K seeds", len(lit) == K, "%s" % (lit,))

    # ---------------------------------------------------------------- test 7
    print("[7] air_cga_select_seeds end-to-end -> RunResult")
    res = air_cga_select_seeds(ds, train, item=0, K=K, rng=random.Random(5),
                               Z=Z, C=4, n_mc=60, n_em_iter=25)
    for fld in ("seeds", "spread", "seconds", "extra"):
        check("RunResult has .%s" % fld, hasattr(res, fld))
    check("RunResult.seeds has K distinct entries",
          len(res.seeds) == K and len(set(res.seeds)) == K, "%s" % (res.seeds,))
    check("RunResult.spread > 0", res.spread > 0.0, "spread=%.4f" % res.spread)
    check("RunResult.seconds > 0", res.seconds > 0.0, "%.3fs" % res.seconds)
    check("seconds covers fit + select (AIR+CGA is the slowest method)",
          res.seconds >= res.extra["fit_seconds"] + res.extra["select_seconds"] - 1e-6,
          "total=%.3fs = fit %.3fs + detect %.3fs + select %.3fs"
          % (res.seconds, res.extra["fit_seconds"],
             res.extra["detect_seconds"], res.extra["select_seconds"]))
    check("the EM fit is a real part of the cost",
          res.extra["fit_seconds"] > 0.0, "%.3fs" % res.extra["fit_seconds"])
    check("extra records the community source",
          isinstance(res.extra.get("community_source"), str),
          res.extra["community_source"])
    check("extra records the CGA source",
          isinstance(res.extra.get("cga_source"), str), res.extra["cga_source"])
    res2 = air_cga_select_seeds(ds, train, item=0, K=K, rng=random.Random(5),
                                Z=Z, C=4, n_mc=60, n_em_iter=25)
    check("end-to-end run is deterministic for a fixed seed",
          res.seeds == res2.seeds and abs(res.spread - res2.spread) <= 1e-12)

    # ---------------------------------------------------------------- test 8
    print("[8] seed quality: beats random, and topic-awareness pays")
    # MixedGreedy is Monte-Carlo, so a *quality* claim needs enough simulations
    # for the estimate to mean anything; n_mc=60 (used above, where only the
    # structure of the result is under test) is far too noisy to rank seed sets.
    N_MC_Q = 200
    N_DRAWS = 200
    res_q = air_cga_select_seeds(ds, train, item=0, K=K, rng=random.Random(5),
                                 Z=Z, C=4, n_mc=N_MC_Q, air=air, comm=comm)
    mia = MIA(ds.n_users, ds.out_adj, ds.in_adj, ew, h=0.1)
    air_spread = mia.influence(res_q.seeds)

    rnd_rng = random.Random(99)
    draws = sorted(mia.influence(rnd_rng.sample(range(ds.n_users), K))
                   for _ in range(N_DRAWS))
    mean_rnd = sum(draws) / len(draws)
    beaten = sum(1 for d in draws if d < air_spread)
    check("AIR+CGA beats the mean random seed set",
          air_spread > mean_rnd,
          "AIR+CGA=%.3f vs random mean=%.3f (%.2fx)"
          % (air_spread, mean_rnd, air_spread / max(mean_rnd, 1e-9)))
    check("AIR+CGA beats at least 95%% of %d random seed sets" % N_DRAWS,
          beaten >= 0.95 * N_DRAWS,
          "beats %d/%d (p95=%.3f, max=%.3f)"
          % (beaten, N_DRAWS, draws[int(0.95 * N_DRAWS)], draws[-1]))

    # Topic-awareness is the whole point of AIR, and the structural reason
    # SPEC.md Section 9 puts AIR+CGA above the topic-blind baselines.  Hold the
    # selector and the communities fixed and vary ONLY the edge weights: the
    # per-item weights of Eq (A2) against weights averaged over all items (which
    # is the best a topic-blind model can do).  Both are scored by the SAME
    # per-item MIA evaluator, so the comparison isolates topic-awareness.
    blind = dict.fromkeys(ds.edges, 0.0)
    for i in range(ds.n_items):
        wi = air.for_item(i, ds.edges)
        for e in ds.edges:
            blind[e] += wi[e] / ds.n_items
    wins = 0
    trials_n = 3
    detail = []
    for it in range(trials_n):
        w_item = air.for_item(it, ds.edges)
        mia_it = MIA(ds.n_users, ds.out_adj, ds.in_adj, w_item, h=0.1)
        s_aware = mia_it.influence(
            cga_select_seeds_local(comm, ds, w_item, K, random.Random(3), n_mc=N_MC_Q))
        s_blind = mia_it.influence(
            cga_select_seeds_local(comm, ds, blind, K, random.Random(3), n_mc=N_MC_Q))
        detail.append("item %d: %.3f vs %.3f" % (it, s_aware, s_blind))
        if s_aware > s_blind:
            wins += 1
    check("topic-aware AIR weights beat topic-blind averaged weights on every item",
          wins == trials_n, "; ".join(detail))

    w_a = air.for_item(0, ds.edges)
    other = next(i for i in range(ds.n_items) if item_topic[i] != item_topic[0])
    w_b = air.for_item(other, ds.edges)
    diff = max(abs(w_a[e] - w_b[e]) for e in ds.edges)
    check("per-item edge weights genuinely differ across topics",
          diff > 1e-3, "max |p_i - p_j| = %.5f (items 0 and %d)" % (diff, other))

    # ---------------------------------------------------------------- test 9
    print("[9] injection points and evaluator override")
    res3 = air_cga_select_seeds(ds, train, item=0, K=3, rng=random.Random(5),
                                Z=Z, C=4, n_mc=40, air=air, comm=comm)
    check("a pre-fitted AIR model is reused (fit_seconds == 0)",
          res3.extra["fit_seconds"] == 0.0 and res3.extra["air_refit"] is False)
    check("a supplied community labelling is reused",
          res3.extra["community_source"] == "caller-supplied")
    res4 = air_cga_select_seeds(ds, train, item=0, K=3, rng=random.Random(5),
                                Z=Z, C=4, n_mc=40, air=air, comm=comm,
                                evaluator=lambda S: 42.0)
    check("a supplied evaluator overrides the built-in MIA scorer",
          res4.spread == 42.0 and res4.extra["evaluator"] == "caller-supplied")

    # A sibling detector with a different signature must still be callable.
    def fake_sibling_detector(ds, n_comm, rng=None):
        return detect_communities_modularity(ds, C=n_comm)

    res5 = air_cga_select_seeds(ds, train, item=0, K=3, rng=random.Random(5),
                                Z=Z, C=4, n_mc=40, air=air,
                                community_detector=fake_sibling_detector)
    check("a sibling detector with a different signature is adapted",
          res5.extra["community_source"] == "caller-supplied"
          and res5.extra["C_achieved"] == 4)

    # A broken sibling must fall back rather than crash the baseline.
    def broken_detector(ds, C, rng=None):
        raise RuntimeError("sibling module is half-written")

    res6 = air_cga_select_seeds(ds, train, item=0, K=3, rng=random.Random(5),
                                Z=Z, C=4, n_mc=40, air=air,
                                community_detector=broken_detector)
    check("a broken sibling detector falls back and records why",
          "community_fallback_reason" in res6.extra
          and len(res6.seeds) == 3,
          res6.extra.get("community_fallback_reason", ""))
    try:
        air_cga_select_seeds(ds, train, item=0, K=3, rng=random.Random(5),
                             Z=Z, C=4, n_mc=40, air=air,
                             community_detector=broken_detector, strict=True)
        raised = False
    except RuntimeError:
        raised = True
    check("strict=True re-raises instead of falling back", raised)

    # ---------------------------------------------------------------- done
    print("")
    if failures:
        print("FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("ALL CHECKS PASSED")
