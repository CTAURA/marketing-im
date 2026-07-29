"""Collapsed Gibbs sampling for the CTIM comprehensive latent variable model.

Reference implementation of Section 4.1 of

    Huimin Huang, Hong Shen, Zaiqiao Meng, Huajian Chang, Huaiwen He.
    "Community-based influence maximization for viral marketing",
    Applied Intelligence (2019). DOI 10.1007/s10489-018-1387-8

Everything here is standard library only and Python 3.9 compatible.  All
randomness flows through an explicit ``random.Random`` instance so a run is
reproducible from its seed alone.

The generative story (Algorithm 1 of the paper) factorises into two blocks that
are statistically independent given the data, so inference runs in two stages
(Section 4.1.2):

  * **Stage 1** (:class:`TopicSampler`) — latent item topics ``z`` from the
    item/attribute matrix ``A``.  This is plain collapsed LDA with
    *item = document* and *attribute value = word*: Eq (1) for the sampler,
    Eq (2) and Eq (3) for the estimators.

  * **Stage 2** (:class:`CommunitySampler`) — the community indicators of the
    friend links (``s'_e, s_e``, Eq (4)) and of the potential-influence logs
    (``c'_d, c_d``, Eq (6)), sampled in a single interleaved sweep because they
    share the count structures ``n_uc`` and ``n_cc``.  Estimators are Eq (7),
    Eq (8) and Eq (9).

Both stages are *collapsed*: the current assignment is removed from the counts
before its full conditional is evaluated and re-added afterwards.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import random
import time
from dataclasses import dataclass

__all__ = [
    "Hyper",
    "TopicSampler",
    "CommunitySampler",
    "Model",
    "train_model",
]


# Smallest eps0 we will ever hand to the Beta(eps0, eps1) prior on eta.  eps0
# must stay strictly positive: eta_{c'c} = (n+eps1)/(n+eps0+eps1) degenerates to
# the constant 1 at eps0 = 0, which would make every community pair equally
# attractive and destroy the block structure entirely.
EPS0_FLOOR = 1e-3


# --------------------------------------------------------------------------
# Hyperparameters
# --------------------------------------------------------------------------
@dataclass
class Hyper:
    """Fixed hyperparameters of the model (SPEC Section 8).

    ``warning`` is non-empty when :meth:`make` had to fall back on a guarded
    value (degenerate ``N_neg``); callers may surface it to the user.
    """

    C: int
    Z: int
    rho: float
    alpha: float
    beta: float
    omega: float
    eps0: float
    eps1: float
    zeta: float = 1.0
    warning: str = ""

    @staticmethod
    def make(C, Z, n_users, n_links, n_logs, zeta=1.0):
        # Section 4.1.1 + SPEC Section 8:
        #   rho = 50/C, beta = 0.01, alpha = 50/Z, omega = 50/Z, eps1 = 0.1
        #   N_neg = U*(U-1)*(1 + D/E) - D - E
        #   eps0  = zeta * ln(N_neg / C^2)
        C = int(C)
        Z = int(Z)
        if C < 1:
            raise ValueError("C must be >= 1, got %r" % (C,))
        if Z < 1:
            raise ValueError("Z must be >= 1, got %r" % (Z,))

        U = int(n_users)
        E = int(n_links)
        D = int(n_logs)
        warnings = []

        # N_neg: the implicitly modelled number of negative (non-)links.  The
        # U*(U-1) term is the number of ordered user pairs; (1 + D/E) inflates it
        # by the observed log-to-link ratio; the observed positives D and E are
        # then removed.
        if E > 0:
            n_neg = float(U) * float(U - 1) * (1.0 + float(D) / float(E)) - float(D) - float(E)
        else:
            # No links at all: the D/E ratio is undefined.  Fall back on the bare
            # pair count so the formula still produces something finite.
            n_neg = float(U) * float(U - 1) - float(D)
            warnings.append(
                "n_links == 0, so the D/E inflation factor of N_neg is undefined; "
                "used N_neg = U*(U-1) - D instead"
            )

        c_sq = float(C) * float(C)
        eps0 = zeta * math.log(n_neg / c_sq) if (n_neg > 0.0 and c_sq < n_neg) else None
        if eps0 is None:
            # N_neg <= 0, or C^2 >= N_neg so ln(N_neg/C^2) <= 0.  Either way the
            # paper's formula cannot produce a usable (strictly positive) Beta
            # pseudo-count, so clamp to the floor and say so.
            warnings.append(
                "degenerate Beta prior: N_neg=%.6g, C^2=%.6g, so ln(N_neg/C^2) is "
                "not strictly positive; eps0 clamped to %g" % (n_neg, c_sq, EPS0_FLOOR)
            )
            eps0 = EPS0_FLOOR
        elif eps0 < EPS0_FLOOR:
            # Reachable through a very small (or non-positive) zeta.
            warnings.append(
                "zeta=%.6g gives eps0=%.6g <= 0; clamped to %g" % (zeta, eps0, EPS0_FLOOR)
            )
            eps0 = EPS0_FLOOR

        return Hyper(
            C=C,
            Z=Z,
            rho=50.0 / C,
            alpha=50.0 / Z,
            beta=0.01,
            omega=50.0 / Z,
            eps0=eps0,
            eps1=0.1,
            zeta=float(zeta),
            warning="; ".join(warnings),
        )

    def to_dict(self):
        return {
            "C": self.C,
            "Z": self.Z,
            "rho": self.rho,
            "alpha": self.alpha,
            "beta": self.beta,
            "omega": self.omega,
            "eps0": self.eps0,
            "eps1": self.eps1,
            "zeta": self.zeta,
            "warning": self.warning,
        }

    @staticmethod
    def from_dict(d):
        return Hyper(
            C=int(d["C"]),
            Z=int(d["Z"]),
            rho=float(d["rho"]),
            alpha=float(d["alpha"]),
            beta=float(d["beta"]),
            omega=float(d["omega"]),
            eps0=float(d["eps0"]),
            eps1=float(d["eps1"]),
            zeta=float(d.get("zeta", 1.0)),
            warning=str(d.get("warning", "")),
        )


# --------------------------------------------------------------------------
# Stage 1 — item topics
# --------------------------------------------------------------------------
class TopicSampler:
    """Stage 1: collapsed LDA over (item = document, attribute value = word).

    Implements Eq (1) for the sweep and Eq (2)/Eq (3) for the estimators.  The
    vocabulary size used by the ``beta`` normaliser is ``n_attrs`` (= F); the
    paper writes ``M x F`` only because ``A`` is an ``M x F`` matrix.
    """

    def __init__(self, item_attrs, n_attrs, hyper, rng):
        self.hyper = hyper
        self.rng = rng
        self.verbose = False
        self.progress_every = 10

        F = int(n_attrs)
        Z = int(hyper.Z)
        if F < 1:
            raise ValueError("n_attrs must be >= 1, got %r" % (n_attrs,))
        self.n_attrs = F
        self.Z = Z

        self.item_attrs = [list(bag) for bag in item_attrs]
        self.M = len(self.item_attrs)
        for i, bag in enumerate(self.item_attrs):
            for w in bag:
                if not (0 <= w < F):
                    raise ValueError(
                        "item %d has attribute id %r outside [0, %d)" % (i, w, F)
                    )

        # n_iz[i*Z + z]  = #times topic z is used by an attribute token of item i
        # n_zw[z*F + w]  = #times attribute value w is generated by topic z
        # n_z[z]         = sum_w n_zw[z][w]
        self.n_iz = [0] * (self.M * Z)
        self.n_zw = [0] * (Z * F)
        self.n_z = [0] * Z
        self.n_i = [len(bag) for bag in self.item_attrs]
        self.z_assign = [[0] * len(bag) for bag in self.item_attrs]

        # Uniform random initialisation, one draw per attribute token.
        randrange = rng.randrange
        n_iz = self.n_iz
        n_zw = self.n_zw
        n_z = self.n_z
        for i, bag in enumerate(self.item_attrs):
            row = self.z_assign[i]
            ibase = i * Z
            for j, w in enumerate(bag):
                z = randrange(Z)
                row[j] = z
                n_iz[ibase + z] += 1
                n_zw[z * F + w] += 1
                n_z[z] += 1

        self.n_iter_done = 0
        self._n_tokens = sum(self.n_i)
        self._cum = [0.0] * Z

    # -- sampling -----------------------------------------------------------
    def run(self, n_iter):
        """Run ``n_iter`` collapsed Gibbs sweeps over every attribute token."""
        n_iter = int(n_iter)
        if n_iter <= 0:
            return
        t0 = time.time()
        every = max(1, int(self.progress_every))
        for it in range(1, n_iter + 1):
            self._sweep()
            self.n_iter_done += 1
            if self.verbose and (it % every == 0 or it == n_iter or it == 1):
                print(
                    "  [stage 1: topics] iter %d/%d  tokens=%d  elapsed=%.2fs"
                    % (it, n_iter, self._n_tokens, time.time() - t0)
                )

    def _sweep(self):
        Z = self.Z
        F = self.n_attrs
        beta = self.hyper.beta
        omega = self.hyper.omega
        fbeta = F * beta
        zomega = Z * omega

        n_iz = self.n_iz
        n_zw = self.n_zw
        n_z = self.n_z
        cum = self._cum
        rnd = self.rng.random
        br = bisect.bisect_right

        for i, bag in enumerate(self.item_attrs):
            if not bag:
                continue
            row = self.z_assign[i]
            ibase = i * Z
            # The first denominator of Eq (1), sum_Z (n_{i,z} + omega).  Exactly
            # one token of item i is excluded at any point in the loop below, so
            # this is the same constant for every token of the item.  It is
            # constant in z as well and therefore cancels in the normalisation;
            # SPEC keeps it for numerical clarity, so we keep it too.
            den_i = (self.n_i[i] - 1) + zomega
            for j, w in enumerate(bag):
                z = row[j]
                # --- collapsed: remove the current assignment ---
                n_iz[ibase + z] -= 1
                n_zw[z * F + w] -= 1
                n_z[z] -= 1

                # Eq (1): P(z_ij = z | z_-ij, w)
                #   ∝ (n_{i,z} + omega) / sum_Z (n_{i,z} + omega)
                #   * (n_{z,w_ij} + beta) / sum_F (n_{z,w} + beta)
                tot = 0.0
                for zz in range(Z):
                    tot += (
                        (n_iz[ibase + zz] + omega)
                        / den_i
                        * (n_zw[zz * F + w] + beta)
                        / (n_z[zz] + fbeta)
                    )
                    cum[zz] = tot

                k = br(cum, rnd() * tot)
                if k >= Z:
                    k = Z - 1

                # --- collapsed: re-add the new assignment ---
                row[j] = k
                n_iz[ibase + k] += 1
                n_zw[k * F + w] += 1
                n_z[k] += 1

    # -- estimators ---------------------------------------------------------
    def phi(self):
        # Eq (2): phi_iz = (n_{i,z} + omega) / sum_Z (n_{i,z} + omega)
        Z = self.Z
        omega = self.hyper.omega
        zomega = Z * omega
        n_iz = self.n_iz
        out = []
        for i in range(self.M):
            base = i * Z
            den = self.n_i[i] + zomega
            out.append([(n_iz[base + z] + omega) / den for z in range(Z)])
        return out

    def psi(self):
        # Eq (3): psi_zw = (n_{z,w} + beta) / sum_F (n_{z,w} + beta)
        Z = self.Z
        F = self.n_attrs
        beta = self.hyper.beta
        fbeta = F * beta
        n_zw = self.n_zw
        out = []
        for z in range(Z):
            base = z * F
            den = self.n_z[z] + fbeta
            out.append([(n_zw[base + w] + beta) / den for w in range(F)])
        return out

    def p_z_given_i(self):
        # SPEC Section 2: item-topic relevance P(z|i) is "inferred by counts",
        # i.e. it is exactly Eq (2).  Returns a fresh list, not an alias of phi().
        return self.phi()


# --------------------------------------------------------------------------
# Stage 2 — communities
# --------------------------------------------------------------------------
class CommunitySampler:
    """Stage 2: Eq (4) for friend links and Eq (6) for potential-influence logs.

    Links and logs are sampled in **one interleaved sweep** per iteration (the
    paper samples ``s, s', c, c'`` together in stage 2) because they share every
    count structure:

    ``n_uc[u*C + c]``
        #times user ``u`` acts as a member of community ``c``, counted across
        **both** links and logs and **both** as source and as target.

    ``n_cc[c'*C + c]``
        #links from ``c'`` to ``c`` **plus** #logs with source community ``c'``
        and target community ``c``.

    ``ncz[c*Z + z]``
        the soft count ``sum_i n_{c,i} * P(z|i)`` of Eq (6), maintained
        *incrementally* in O(Z) whenever a log gains or loses a target
        community.  It is never recomputed from scratch inside the sampler.

    ``et[c'*C + c]``
        the memoised Eq (8) factor ``(n_cc + eps1) / (n_cc + eps0 + eps1)``.
        Exactly one entry changes per add/remove, so it is patched in O(1).

    The topic ``z_i`` of a log is redrawn from ``Mul(phi_i)`` at every visit and
    is *not* stored: none of the count structures depend on it (``ncz`` is a soft
    count over the whole topic distribution), it only selects which column of
    ``ncz`` the Eq (6) theta-factor reads.
    """

    def __init__(self, n_users, edges, logs_d, p_z_given_i, hyper, rng,
                 sampler="exact", n_mh=2):
        if sampler not in ("exact", "mh"):
            raise ValueError("sampler must be 'exact' or 'mh', got %r" % (sampler,))
        self.hyper = hyper
        self.rng = rng
        self.sampler = sampler
        self.n_mh = max(1, int(n_mh))
        self.verbose = False
        self.progress_every = 10

        C = int(hyper.C)
        Z = int(hyper.Z)
        U = int(n_users)
        if U < 1:
            raise ValueError("n_users must be >= 1, got %r" % (n_users,))
        self.C = C
        self.Z = Z
        self.U = U

        self.edges = [(int(u), int(v)) for (u, v) in edges]
        self.logs = [(int(u), int(v), int(i)) for (u, v, i) in logs_d]
        self.n_links = len(self.edges)
        self.n_logs = len(self.logs)
        for (u, v) in self.edges:
            if not (0 <= u < U and 0 <= v < U):
                raise ValueError("link (%d,%d) outside [0,%d)" % (u, v, U))

        # Normalised copy of P(z|i).  Rows are forced to sum to 1 so that
        # sum_z ncz[c][z] == n_ctgt[c] holds (up to float rounding) and the Eq (6)
        # / Eq (9) denominator can use the exact integer log count.
        M = len(p_z_given_i)
        self.M = M
        pz = []
        cum_pz = []
        for row in p_z_given_i:
            if len(row) != Z:
                raise ValueError(
                    "p_z_given_i rows must have length Z=%d, got %d" % (Z, len(row))
                )
            s = 0.0
            for p in row:
                if p < 0.0:
                    raise ValueError("p_z_given_i has a negative entry")
                s += p
            if s <= 0.0:
                r = [1.0 / Z] * Z
            else:
                r = [p / s for p in row]
            pz.append(r)
            acc = 0.0
            crow = [0.0] * Z
            for z in range(Z):
                acc += r[z]
                crow[z] = acc
            crow[Z - 1] = 1.0
            cum_pz.append(crow)
        self.pz = pz
        self.cum_pz = cum_pz
        for (u, v, i) in self.logs:
            if not (0 <= u < U and 0 <= v < U):
                raise ValueError("log (%d,%d,%d) has a user outside [0,%d)" % (u, v, i, U))
            if not (0 <= i < M):
                raise ValueError("log (%d,%d,%d) has an item outside [0,%d)" % (u, v, i, M))

        eps0 = hyper.eps0
        eps1 = hyper.eps1

        self.n_uc = [0] * (U * C)
        self.n_cc = [0] * (C * C)
        self.et = [eps1 / (eps0 + eps1)] * (C * C)
        self.ncz = [0.0] * (C * Z)
        self.n_ctgt = [0] * C

        self.link_src = [0] * self.n_links
        self.link_tgt = [0] * self.n_links
        self.log_src = [0] * self.n_logs
        self.log_tgt = [0] * self.n_logs

        # Uniform random initialisation of every community indicator.
        randrange = rng.randrange
        n_uc = self.n_uc
        n_cc = self.n_cc
        for e, (u, v) in enumerate(self.edges):
            cp = randrange(C)
            c = randrange(C)
            self.link_src[e] = cp
            self.link_tgt[e] = c
            n_uc[u * C + cp] += 1
            n_uc[v * C + c] += 1
            n_cc[cp * C + c] += 1
        ncz = self.ncz
        n_ctgt = self.n_ctgt
        for d, (u, v, i) in enumerate(self.logs):
            cp = randrange(C)
            c = randrange(C)
            self.log_src[d] = cp
            self.log_tgt[d] = c
            n_uc[u * C + cp] += 1
            n_uc[v * C + c] += 1
            n_cc[cp * C + c] += 1
            zb = c * Z
            for z, p in enumerate(pz[i]):
                ncz[zb + z] += p
            n_ctgt[c] += 1
        et = self.et
        for k in range(C * C):
            n = n_cc[k]
            et[k] = (n + eps1) / (n + eps0 + eps1)

        # One interleaved event stream: ev < n_links is link ev, otherwise it is
        # log (ev - n_links).  Shuffled once so links and logs really do
        # interleave, then reused in the same order every sweep (deterministic).
        self._events = list(range(self.n_links + self.n_logs))
        rng.shuffle(self._events)

        # Scratch buffers, allocated once.
        self._A = [0.0] * C
        self._B = [0.0] * C
        self._cum = [0.0] * (C * C)
        self._cumA = [0.0] * C
        self._cumB = [0.0] * C

        self.n_iter_done = 0

    # -- sampling -----------------------------------------------------------
    def run(self, n_iter):
        """Run ``n_iter`` interleaved sweeps over all links and all logs."""
        n_iter = int(n_iter)
        if n_iter <= 0:
            return
        sweep = self._sweep_exact if self.sampler == "exact" else self._sweep_mh
        t0 = time.time()
        every = max(1, int(self.progress_every))
        n_ev = len(self._events)
        for it in range(1, n_iter + 1):
            sweep()
            self.n_iter_done += 1
            if self.verbose and (it % every == 0 or it == n_iter or it == 1):
                print(
                    "  [stage 2: communities/%s] iter %d/%d  events=%d  elapsed=%.2fs"
                    % (self.sampler, it, n_iter, n_ev, time.time() - t0)
                )

    def _sweep_exact(self):
        """One interleaved sweep, enumerating all C^2 pairs per event.

        Both Eq (4) and Eq (6) share the shape ``A[c'] * B[c] * etatilde[c'][c]``
        with

            A[c'] = n_{u,c'} + rho
            B[c]  = (n_{v,c} + rho)                          for a link,  Eq (4)
            B[c]  = (n_{v,c} + rho) * thetafactor[c]         for a log,   Eq (6)

        The per-user denominators ``sum_C (n_{u,c'} + rho)`` and
        ``sum_C (n_{v,c} + rho)`` of Eq (4)/(6) are constant with respect to the
        sampled indices c' and c (they sum over *all* communities), so they are
        the same factor on every one of the C^2 candidates and cancel in the
        normalisation.  We therefore drop them.
        """
        C = self.C
        Z = self.Z
        CC = C * C
        rho = self.hyper.rho
        alpha = self.hyper.alpha
        eps0 = self.hyper.eps0
        eps1 = self.hyper.eps1
        zalpha = Z * alpha

        n_uc = self.n_uc
        n_cc = self.n_cc
        et = self.et
        ncz = self.ncz
        n_ctgt = self.n_ctgt
        edges = self.edges
        logs = self.logs
        lsrc = self.link_src
        ltgt = self.link_tgt
        dsrc = self.log_src
        dtgt = self.log_tgt
        pz = self.pz
        cum_pz = self.cum_pz
        A = self._A
        B = self._B
        cum = self._cum
        rnd = self.rng.random
        br = bisect.bisect_right
        n_links = self.n_links
        rng_C = range(C)

        for ev in self._events:
            if ev < n_links:
                # ---------------- friend link e = (u,v), Eq (4) ----------------
                u, v = edges[ev]
                cp = lsrc[ev]
                c = ltgt[ev]
                ub = u * C
                vb = v * C
                # collapsed: decrement the current assignment
                n_uc[ub + cp] -= 1
                n_uc[vb + c] -= 1
                k = cp * C + c
                n = n_cc[k] - 1
                n_cc[k] = n
                et[k] = (n + eps1) / (n + eps0 + eps1)

                for cc in rng_C:
                    A[cc] = n_uc[ub + cc] + rho
                    B[cc] = n_uc[vb + cc] + rho
            else:
                # ------- potential-influence log d = (u,v,i), Eq (6) -------
                d = ev - n_links
                u, v, i = logs[d]
                cp = dsrc[d]
                c = dtgt[d]
                ub = u * C
                vb = v * C
                # collapsed: decrement the current assignment
                n_uc[ub + cp] -= 1
                n_uc[vb + c] -= 1
                k = cp * C + c
                n = n_cc[k] - 1
                n_cc[k] = n
                et[k] = (n + eps1) / (n + eps0 + eps1)
                # incremental O(Z) removal of this log's soft topic counts from
                # its *target* community: ncz[c][z] -= P(z|i)   (Eq (6))
                pzi = pz[i]
                zb = c * Z
                for z, p in enumerate(pzi):
                    ncz[zb + z] -= p
                n_ctgt[c] -= 1

                # Algorithm 1, line 24: draw item i's topic z_i ~ Mul(phi_i)
                # using P(z|i), then sample the (c', c) pair given z_i.
                zi = br(cum_pz[i], rnd())
                if zi >= Z:
                    zi = Z - 1

                # Eq (6) theta factor of the *target* community:
                #   ( sum_M n_{c,i} P(z|i) + alpha ) / sum_Z ( ... + alpha )
                # The denominator sums to n_ctgt[c] + Z*alpha because
                # sum_z P(z|i) = 1 for every log counted into community c.
                for cc in rng_C:
                    A[cc] = n_uc[ub + cc] + rho
                    B[cc] = (n_uc[vb + cc] + rho) * (
                        (ncz[cc * Z + zi] + alpha) / (n_ctgt[cc] + zalpha)
                    )

            # ---- shared C^2 enumeration: A[c'] * B[c] * etatilde[c'][c] ----
            tot = 0.0
            k = 0
            base = 0
            for a in A:
                for b, e in zip(B, et[base:base + C]):
                    tot += a * b * e
                    cum[k] = tot
                    k += 1
                base += C

            k = br(cum, rnd() * tot)
            if k >= CC:
                k = CC - 1
            cp = k // C
            c = k - cp * C

            # ---- collapsed: re-add the new assignment ----
            n_uc[ub + cp] += 1
            n_uc[vb + c] += 1
            k = cp * C + c
            n = n_cc[k] + 1
            n_cc[k] = n
            et[k] = (n + eps1) / (n + eps0 + eps1)
            if ev < n_links:
                lsrc[ev] = cp
                ltgt[ev] = c
            else:
                dsrc[d] = cp
                dtgt[d] = c
                zb = c * Z
                for z, p in enumerate(pzi):
                    ncz[zb + z] += p
                n_ctgt[c] += 1

    def _sweep_mh(self):
        """One interleaved sweep using Metropolis-Hastings instead of the full
        C^2 enumeration.

        The target is the exact Eq (4) / Eq (6) full conditional

            p(c', c) ∝ A[c'] * B[c] * etatilde[c'][c]

        and the proposal is the *independent* (state-free) distribution

            q(c', c) ∝ A[c'] * B[c]

        which factorises, so c' and c are drawn independently by cumulative
        search over A and B (O(C) to build the two CDFs, O(log C) per draw).
        An alias table would not help: the weights change at every event, so the
        O(C) build dominates either way.

        Because the proposal is independent of the current state, the Hastings
        ratio collapses to the ratio of the un-proposed factor:

            alpha_acc = min(1,  [p(new)/q(new)] / [p(old)/q(old)])
                      = min(1,  etatilde[c'_new][c_new] / etatilde[c'_old][c_old])

        etatilde is strictly positive (eps1 > 0), so the ratio is always well
        defined.  Each such move satisfies detailed balance with respect to the
        exact full conditional, hence a fixed number ``n_mh`` of them per event
        is a valid Metropolis-within-Gibbs kernel with **the same stationary
        distribution as sampler="exact"** — the two differ only in mixing speed,
        not in the distribution they target.  The counts (and therefore A, B and
        etatilde) are held at their "current assignment removed" values for the
        whole MH inner loop, exactly as the exact sampler holds them for its
        enumeration.
        """
        C = self.C
        Z = self.Z
        rho = self.hyper.rho
        alpha = self.hyper.alpha
        eps0 = self.hyper.eps0
        eps1 = self.hyper.eps1
        zalpha = Z * alpha

        n_uc = self.n_uc
        n_cc = self.n_cc
        et = self.et
        ncz = self.ncz
        n_ctgt = self.n_ctgt
        edges = self.edges
        logs = self.logs
        lsrc = self.link_src
        ltgt = self.link_tgt
        dsrc = self.log_src
        dtgt = self.log_tgt
        pz = self.pz
        cum_pz = self.cum_pz
        A = self._A
        B = self._B
        cumA = self._cumA
        cumB = self._cumB
        rnd = self.rng.random
        br = bisect.bisect_right
        n_links = self.n_links
        n_mh = self.n_mh
        rng_C = range(C)

        for ev in self._events:
            if ev < n_links:
                u, v = edges[ev]                      # Eq (4)
                cp = lsrc[ev]
                c = ltgt[ev]
                ub = u * C
                vb = v * C
                n_uc[ub + cp] -= 1
                n_uc[vb + c] -= 1
                k = cp * C + c
                n = n_cc[k] - 1
                n_cc[k] = n
                et[k] = (n + eps1) / (n + eps0 + eps1)
                for cc in rng_C:
                    A[cc] = n_uc[ub + cc] + rho
                    B[cc] = n_uc[vb + cc] + rho
            else:
                d = ev - n_links                      # Eq (6)
                u, v, i = logs[d]
                cp = dsrc[d]
                c = dtgt[d]
                ub = u * C
                vb = v * C
                n_uc[ub + cp] -= 1
                n_uc[vb + c] -= 1
                k = cp * C + c
                n = n_cc[k] - 1
                n_cc[k] = n
                et[k] = (n + eps1) / (n + eps0 + eps1)
                pzi = pz[i]
                zb = c * Z
                for z, p in enumerate(pzi):
                    ncz[zb + z] -= p
                n_ctgt[c] -= 1

                # Algorithm 1, line 24: z_i ~ Mul(phi_i)
                zi = br(cum_pz[i], rnd())
                if zi >= Z:
                    zi = Z - 1
                for cc in rng_C:
                    A[cc] = n_uc[ub + cc] + rho
                    B[cc] = (n_uc[vb + cc] + rho) * (
                        (ncz[cc * Z + zi] + alpha) / (n_ctgt[cc] + zalpha)
                    )

            # ---- proposal CDFs, q(c',c) ∝ A[c'] * B[c] ----
            ta = 0.0
            for cc in rng_C:
                ta += A[cc]
                cumA[cc] = ta
            tb = 0.0
            for cc in rng_C:
                tb += B[cc]
                cumB[cc] = tb

            cur = et[cp * C + c]
            for _ in range(n_mh):
                k1 = br(cumA, rnd() * ta)
                if k1 >= C:
                    k1 = C - 1
                k2 = br(cumB, rnd() * tb)
                if k2 >= C:
                    k2 = C - 1
                new = et[k1 * C + k2]
                # Hastings acceptance ratio = etatilde_new / etatilde_old
                if new >= cur or rnd() * cur < new:
                    cp = k1
                    c = k2
                    cur = new

            # ---- collapsed: re-add the new assignment ----
            n_uc[ub + cp] += 1
            n_uc[vb + c] += 1
            k = cp * C + c
            n = n_cc[k] + 1
            n_cc[k] = n
            et[k] = (n + eps1) / (n + eps0 + eps1)
            if ev < n_links:
                lsrc[ev] = cp
                ltgt[ev] = c
            else:
                dsrc[d] = cp
                dtgt[d] = c
                zb = c * Z
                for z, p in enumerate(pzi):
                    ncz[zb + z] += p
                n_ctgt[c] += 1

    # -- estimators ---------------------------------------------------------
    def pi(self):
        # Eq (7): pi_vc = (n_{v,c} + rho) / sum_C (n_{v,c} + rho)
        C = self.C
        rho = self.hyper.rho
        crho = C * rho
        n_uc = self.n_uc
        out = []
        for v in range(self.U):
            base = v * C
            s = 0
            for c in range(C):
                s += n_uc[base + c]
            den = s + crho
            out.append([(n_uc[base + c] + rho) / den for c in range(C)])
        return out

    def eta(self):
        # Eq (8): eta_{c'c} = (n_{c'c} + eps1) / (n_{c'c} + eps0 + eps1)
        C = self.C
        eps0 = self.hyper.eps0
        eps1 = self.hyper.eps1
        n_cc = self.n_cc
        out = []
        for cp in range(C):
            base = cp * C
            out.append(
                [
                    (n_cc[base + c] + eps1) / (n_cc[base + c] + eps0 + eps1)
                    for c in range(C)
                ]
            )
        return out

    def theta(self):
        # Eq (9): theta_cz = ( sum_M n_{c,i} P(z|i) + alpha )
        #                    / sum_Z ( sum_M n_{c,i} P(z|i) + alpha )
        C = self.C
        Z = self.Z
        alpha = self.hyper.alpha
        ncz = self.ncz
        out = []
        for c in range(C):
            base = c * Z
            row = [ncz[base + z] for z in range(Z)]
            # Guard against the O(1e-12) negative drift that incremental float
            # add/subtract can leave behind on a genuinely-zero soft count.
            row = [x if x > 0.0 else 0.0 for x in row]
            den = sum(row) + Z * alpha
            out.append([(x + alpha) / den for x in row])
        return out

    # -- diagnostics --------------------------------------------------------
    def validate_counts(self, tol=1e-6):
        """Recompute every count from the stored assignments and compare.

        Only used by tests/self-checks: it is the O(E + D) audit that the
        incremental O(Z) maintenance of ``ncz`` is actually correct.  Returns a
        list of human-readable discrepancies (empty when all invariants hold).
        """
        C = self.C
        Z = self.Z
        problems = []

        n_uc = [0] * (self.U * C)
        n_cc = [0] * (C * C)
        ncz = [0.0] * (C * Z)
        n_ctgt = [0] * C
        for e, (u, v) in enumerate(self.edges):
            cp = self.link_src[e]
            c = self.link_tgt[e]
            n_uc[u * C + cp] += 1
            n_uc[v * C + c] += 1
            n_cc[cp * C + c] += 1
        for d, (u, v, i) in enumerate(self.logs):
            cp = self.log_src[d]
            c = self.log_tgt[d]
            n_uc[u * C + cp] += 1
            n_uc[v * C + c] += 1
            n_cc[cp * C + c] += 1
            for z, p in enumerate(self.pz[i]):
                ncz[c * Z + z] += p
            n_ctgt[c] += 1

        if n_uc != self.n_uc:
            problems.append("n_uc mismatch")
        if n_cc != self.n_cc:
            problems.append("n_cc mismatch")
        if n_ctgt != self.n_ctgt:
            problems.append("n_ctgt mismatch")
        worst = 0.0
        for k in range(C * Z):
            worst = max(worst, abs(ncz[k] - self.ncz[k]))
        if worst > tol:
            problems.append("ncz drift %.3g > tol %.3g" % (worst, tol))
        eps0 = self.hyper.eps0
        eps1 = self.hyper.eps1
        for k in range(C * C):
            want = (self.n_cc[k] + eps1) / (self.n_cc[k] + eps0 + eps1)
            if abs(want - self.et[k]) > 1e-12:
                problems.append("etatilde stale at %d" % k)
                break
        return problems


# --------------------------------------------------------------------------
# The fitted model
# --------------------------------------------------------------------------
@dataclass
class Model:
    """Everything downstream code (influence.py, ctim.py) needs from inference."""

    hyper: Hyper
    pi: list
    eta: list
    theta: list
    p_z_given_i: list
    phi: list
    psi: list

    FORMAT = "ctim-model-v1"

    def save(self, path):
        """Serialise to JSON (deliberately not pickle: results stay inspectable)."""
        d = {
            "format": Model.FORMAT,
            "hyper": self.hyper.to_dict(),
            "pi": self.pi,
            "eta": self.eta,
            "theta": self.theta,
            "p_z_given_i": self.p_z_given_i,
            "phi": self.phi,
            "psi": self.psi,
        }
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        fmt = d.get("format")
        if fmt is not None and fmt != Model.FORMAT:
            raise ValueError("unknown model format %r" % (fmt,))
        return Model(
            hyper=Hyper.from_dict(d["hyper"]),
            pi=d["pi"],
            eta=d["eta"],
            theta=d["theta"],
            p_z_given_i=d["p_z_given_i"],
            phi=d["phi"],
            psi=d["psi"],
        )


def train_model(ds, logs_train, C, Z, n_iter_topic, n_iter_comm, rng,
                zeta=1.0, sampler="exact", verbose=False):
    """Run both inference stages of Section 4.1.2 and return the fitted Model.

    ``ds`` only has to expose the ``Dataset`` attributes ``n_users``,
    ``n_attrs``, ``edges`` and ``item_attrs`` (duck-typed on purpose, so this
    module has no import dependency on ctim.dataset).
    ``logs_train`` is the TRAIN split of the potential-influence logs, a list of
    ``(u, v, i)`` triples.
    """
    t0 = time.time()
    edges = ds.edges
    hyper = Hyper.make(C, Z, ds.n_users, len(edges), len(logs_train), zeta=zeta)
    if verbose:
        print(
            "[train_model] C=%d Z=%d U=%d E=%d D=%d M=%d F=%d sampler=%s"
            % (
                hyper.C,
                hyper.Z,
                ds.n_users,
                len(edges),
                len(logs_train),
                len(ds.item_attrs),
                ds.n_attrs,
                sampler,
            )
        )
        print(
            "[train_model] rho=%.4g alpha=%.4g beta=%.4g omega=%.4g eps0=%.4g eps1=%.4g"
            % (hyper.rho, hyper.alpha, hyper.beta, hyper.omega, hyper.eps0, hyper.eps1)
        )
        if hyper.warning:
            print("[train_model] WARNING: %s" % hyper.warning)

    # ---- Stage 1: item topics, Eq (1)-(3) ----
    if verbose:
        print("[train_model] stage 1: sampling item topics (%d iters)" % n_iter_topic)
    ts = TopicSampler(ds.item_attrs, ds.n_attrs, hyper, rng)
    ts.verbose = verbose
    ts.run(n_iter_topic)
    p_z_i = ts.p_z_given_i()          # Eq (2), "inferred by counts"
    t1 = time.time()
    if verbose:
        print("[train_model] stage 1 done in %.2fs" % (t1 - t0))

    # ---- Stage 2: communities, Eq (4)-(9) ----
    if verbose:
        print("[train_model] stage 2: sampling communities (%d iters)" % n_iter_comm)
    cs = CommunitySampler(
        ds.n_users, edges, logs_train, p_z_i, hyper, rng, sampler=sampler
    )
    cs.verbose = verbose
    cs.run(n_iter_comm)
    t2 = time.time()
    if verbose:
        print("[train_model] stage 2 done in %.2fs" % (t2 - t1))
        print("[train_model] total %.2fs" % (t2 - t0))

    return Model(
        hyper=hyper,
        pi=cs.pi(),          # Eq (7)
        eta=cs.eta(),        # Eq (8)
        theta=cs.theta(),    # Eq (9)
        p_z_given_i=p_z_i,   # Eq (2)
        phi=ts.phi(),        # Eq (2)
        psi=ts.psi(),        # Eq (3)
    )


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    @dataclass
    class _FakeDataset:
        """Minimal duck-type of ctim.dataset.Dataset for the self-test."""

        n_users: int
        n_items: int
        n_attrs: int
        edges: list
        item_attrs: list

    def _adjusted_rand(a, b):
        """Adjusted Rand index between two labellings of the same n points."""
        n = len(a)
        if n != len(b):
            raise ValueError("labellings of different length")
        table = {}
        ca = {}
        cb = {}
        for x, y in zip(a, b):
            table[(x, y)] = table.get((x, y), 0) + 1
            ca[x] = ca.get(x, 0) + 1
            cb[y] = cb.get(y, 0) + 1

        def c2(m):
            return m * (m - 1) / 2.0

        sum_ij = sum(c2(m) for m in table.values())
        sum_a = sum(c2(m) for m in ca.values())
        sum_b = sum(c2(m) for m in cb.values())
        total = c2(n)
        if total == 0:
            return 1.0
        expected = sum_a * sum_b / total
        maximum = 0.5 * (sum_a + sum_b)
        if maximum == expected:
            return 1.0
        return (sum_ij - expected) / (maximum - expected)

    def _make_synthetic(rng, n_per_comm=30, n_items=20, n_attrs=20, p_in=0.20,
                        p_out=0.005, logs_per_edge=30, attr_tokens=200, noise=0.05):
        """A planted-partition graph with 2 communities and 2 planted topics.

        Community 0 adopts topic-0 items, community 1 adopts topic-1 items, so
        both the link structure (Eq 4) and the topic structure (Eq 6) point at
        the same 2-way partition.

        The instance is deliberately **log-rich** (D/E ~ 25).  At C = 2 the
        paper's own hyperparameter rule makes the link term of Eq (4) almost
        uninformative: rho = 50/C = 25 swamps the per-user counts, and with only
        C^2 = 4 blocks every n_{c'c} is far above eps0, so every etatilde is
        ~0.99 whatever the partition.  Essentially all of the community signal at
        C = 2 therefore comes from the Eq (6) theta factor, which needs enough
        potential-influence logs per community for the soft counts to dominate
        alpha = 50/Z = 25.  Empirically this instance sits well past the
        ordering transition (which is near logs_per_edge ~ 12 here), so both
        samplers recover the planted partition with a comfortable margin.
        """
        U = 2 * n_per_comm
        labels = [0] * n_per_comm + [1] * n_per_comm

        edges = []
        for u in range(U):
            for v in range(U):
                if u == v:
                    continue
                p = p_in if labels[u] == labels[v] else p_out
                if rng.random() < p:
                    edges.append((u, v))

        half = n_attrs // 2
        item_topic = [i % 2 for i in range(n_items)]
        item_attrs = []
        for i in range(n_items):
            t = item_topic[i]
            bag = []
            for _ in range(attr_tokens):
                tt = (1 - t) if rng.random() < noise else t
                bag.append(tt * half + rng.randrange(half))
            item_attrs.append(bag)
        by_topic = {
            0: [i for i in range(n_items) if item_topic[i] == 0],
            1: [i for i in range(n_items) if item_topic[i] == 1],
        }

        logs = []
        for (u, v) in edges:
            if labels[u] != labels[v]:
                continue
            for _ in range(logs_per_edge):
                t = labels[u] if rng.random() >= noise else 1 - labels[u]
                logs.append((u, v, rng.choice(by_topic[t])))

        ds = _FakeDataset(
            n_users=U,
            n_items=n_items,
            n_attrs=n_attrs,
            edges=edges,
            item_attrs=item_attrs,
        )
        return ds, labels, item_topic, logs

    def _argmax_labels(mat):
        out = []
        for row in mat:
            best = 0
            bv = row[0]
            for k in range(1, len(row)):
                if row[k] > bv:
                    bv = row[k]
                    best = k
            out.append(best)
        return out

    def _close(x, y, tol=1e-9):
        return abs(x - y) <= tol * max(1.0, abs(x), abs(y))

    t_start = time.time()
    failures = []

    def check(cond, msg):
        if cond:
            print("  ok   %s" % msg)
        else:
            print("  FAIL %s" % msg)
            failures.append(msg)

    print("=" * 72)
    print("ctim/gibbs.py self-test")
    print("=" * 72)

    # ---------------- 1. Hyper.make ----------------
    print("\n[1] Hyper.make")
    h = Hyper.make(C=100, Z=8, n_users=30358, n_links=99846, n_logs=50000)
    check(_close(h.rho, 0.5), "rho == 50/C")
    check(_close(h.alpha, 6.25), "alpha == 50/Z")
    check(_close(h.omega, 6.25), "omega == 50/Z")
    check(_close(h.beta, 0.01), "beta == 0.01")
    check(_close(h.eps1, 0.1), "eps1 == 0.1")
    _U, _E, _D = 30358, 99846, 50000
    _nneg = _U * (_U - 1) * (1.0 + _D / _E) - _D - _E
    check(_close(h.eps0, math.log(_nneg / 10000.0)), "eps0 == zeta*ln(N_neg/C^2)")
    check(h.warning == "", "no warning on a healthy configuration")
    h2 = Hyper.make(C=100, Z=8, n_users=30358, n_links=99846, n_logs=50000, zeta=2.0)
    check(_close(h2.eps0, 2.0 * h.eps0), "zeta scales eps0 linearly")
    # degenerate: C^2 >= N_neg  ->  clamped, warning recorded
    hd = Hyper.make(C=50, Z=4, n_users=5, n_links=4, n_logs=2)
    check(hd.eps0 == EPS0_FLOOR, "eps0 clamped to the floor when C^2 >= N_neg")
    check("clamped" in hd.warning, "clamp recorded in Hyper.warning")
    hz = Hyper.make(C=2, Z=2, n_users=100, n_links=200, n_logs=200, zeta=0.0)
    check(hz.eps0 == EPS0_FLOOR and hz.warning != "", "zeta=0 clamped and warned")
    he = Hyper.make(C=2, Z=2, n_users=100, n_links=0, n_logs=0)
    check("n_links == 0" in he.warning, "E == 0 handled and warned")

    # ---------------- 2. synthetic data ----------------
    print("\n[2] synthetic planted-partition data")
    rng = random.Random(12345)
    ds, planted_comm, planted_topic, logs = _make_synthetic(rng)
    print(
        "  U=%d  E=%d  D=%d  M=%d  F=%d"
        % (ds.n_users, len(ds.edges), len(logs), ds.n_items, ds.n_attrs)
    )
    check(len(ds.edges) > 200 and len(logs) > 5000, "graph and logs are non-trivial")

    # ---------------- 3. stage 1: topics ----------------
    print("\n[3] TopicSampler (Eq 1-3)")
    hyp = Hyper.make(C=2, Z=2, n_users=ds.n_users, n_links=len(ds.edges), n_logs=len(logs))
    ts = TopicSampler(ds.item_attrs, ds.n_attrs, hyp, random.Random(7))
    ts.run(150)
    phi = ts.phi()
    psi = ts.psi()
    check(all(_close(sum(r), 1.0) for r in phi), "Eq (2) phi rows are normalised")
    check(all(_close(sum(r), 1.0) for r in psi), "Eq (3) psi rows are normalised")
    ari_topic = _adjusted_rand(planted_topic, _argmax_labels(phi))
    print("  topic ARI = %.3f" % ari_topic)
    check(ari_topic > 0.7, "recovered item topics agree with the planted topics")
    check(ts.p_z_given_i() is not phi, "p_z_given_i() returns a fresh list")

    p_z_i = ts.p_z_given_i()

    # ------- 4. the samplers reproduce the analytic full conditionals -------
    # The sharpest correctness check available: freeze the count state, resample
    # ONE event many times, and compare the empirical histogram over the C^2
    # candidate pairs against Eq (4) / Eq (6) evaluated by brute force.
    print("\n[4] full conditionals vs brute-force Eq (4) / Eq (6)")
    rng_s = random.Random(4)
    ds_s, _, _, logs_s = _make_synthetic(
        rng_s, n_per_comm=6, n_items=6, p_in=0.5, logs_per_edge=2, attr_tokens=40
    )
    Cs, Zs = 3, 2
    hyp_s = Hyper.make(C=Cs, Z=Zs, n_users=ds_s.n_users, n_links=len(ds_s.edges),
                       n_logs=len(logs_s))
    ts_s = TopicSampler(ds_s.item_attrs, ds_s.n_attrs, hyp_s, random.Random(7))
    ts_s.run(50)
    cs_s = CommunitySampler(ds_s.n_users, ds_s.edges, logs_s, ts_s.p_z_given_i(),
                            hyp_s, random.Random(3))
    cs_s.run(5)

    # Force P(z|i) of the probed log's item to a delta, so its z_i draw is
    # deterministic and the histogram isolates the (c', c) conditional given z_i.
    probe_log = 0
    probe_item = cs_s.logs[probe_log][2]
    cs_s.pz[probe_item] = [1.0] + [0.0] * (Zs - 1)
    cs_s.cum_pz[probe_item] = [1.0] * Zs
    cs_s.ncz = [0.0] * (Cs * Zs)
    cs_s.n_ctgt = [0] * Cs
    for _d, (_u, _v, _i) in enumerate(cs_s.logs):
        _c = cs_s.log_tgt[_d]
        for _z, _p in enumerate(cs_s.pz[_i]):
            cs_s.ncz[_c * Zs + _z] += _p
        cs_s.n_ctgt[_c] += 1
    check(not cs_s.validate_counts(), "probe instance is internally consistent")

    def _snapshot(cs):
        return (list(cs.n_uc), list(cs.n_cc), list(cs.ncz), list(cs.n_ctgt),
                list(cs.et), list(cs.link_src), list(cs.link_tgt),
                list(cs.log_src), list(cs.log_tgt))

    def _restore(cs, s):
        cs.n_uc[:] = s[0]
        cs.n_cc[:] = s[1]
        cs.ncz[:] = s[2]
        cs.n_ctgt[:] = s[3]
        cs.et[:] = s[4]
        cs.link_src[:] = s[5]
        cs.link_tgt[:] = s[6]
        cs.log_src[:] = s[7]
        cs.log_tgt[:] = s[8]

    def _analytic(cs, ev, zi):
        """Eq (4) (link) or Eq (6) (log), written out directly, no shortcuts."""
        C, Z = cs.C, cs.Z
        h = cs.hyper
        n_uc = list(cs.n_uc)
        n_cc = list(cs.n_cc)
        ncz = list(cs.ncz)
        n_ctgt = list(cs.n_ctgt)
        if ev < cs.n_links:
            u, v = cs.edges[ev]
            cp0, c0 = cs.link_src[ev], cs.link_tgt[ev]
        else:
            u, v, i = cs.logs[ev - cs.n_links]
            cp0, c0 = cs.log_src[ev - cs.n_links], cs.log_tgt[ev - cs.n_links]
            for z, p in enumerate(cs.pz[i]):
                ncz[c0 * Z + z] -= p
            n_ctgt[c0] -= 1
        n_uc[u * C + cp0] -= 1
        n_uc[v * C + c0] -= 1
        n_cc[cp0 * C + c0] -= 1
        w = []
        for cp in range(C):
            for c in range(C):
                n = n_cc[cp * C + c]
                x = (
                    (n_uc[u * C + cp] + h.rho)
                    / sum(n_uc[u * C + k] + h.rho for k in range(C))
                    * (n_uc[v * C + c] + h.rho)
                    / sum(n_uc[v * C + k] + h.rho for k in range(C))
                    * (n + h.eps1) / (n + h.eps0 + h.eps1)
                )
                if ev >= cs.n_links:
                    x *= (ncz[c * Z + zi] + h.alpha) / (n_ctgt[c] + Z * h.alpha)
                w.append(x)
        s = sum(w)
        return [x / s for x in w]

    def _histogram(cs, ev, n_draws, sweeps, seed0):
        snap = _snapshot(cs)
        saved_events = cs._events
        saved_rng = cs.rng
        cnt = [0] * (cs.C * cs.C)
        cs._events = [ev]
        sweep = cs._sweep_exact if cs.sampler == "exact" else cs._sweep_mh
        for t in range(n_draws):
            _restore(cs, snap)
            cs.rng = random.Random(seed0 + t)
            for _ in range(sweeps):
                sweep()
            if ev < cs.n_links:
                cnt[cs.link_src[ev] * cs.C + cs.link_tgt[ev]] += 1
            else:
                d = ev - cs.n_links
                cnt[cs.log_src[d] * cs.C + cs.log_tgt[d]] += 1
        _restore(cs, snap)
        cs._events = saved_events
        cs.rng = saved_rng
        return [c / float(n_draws) for c in cnt]

    N_DRAWS = 15000
    TOL = 0.015  # ~5 Monte-Carlo sd at N=15000; the run is seeded, so it is stable

    want = _analytic(cs_s, 0, 0)
    got = _histogram(cs_s, 0, N_DRAWS, 1, 1000)
    err = max(abs(a - b) for a, b in zip(got, want))
    print("  link  Eq (4), exact: max|empirical - analytic| = %.4f" % err)
    check(err < TOL, "exact sampler reproduces Eq (4) for links")

    ev_log = cs_s.n_links + probe_log
    want = _analytic(cs_s, ev_log, 0)
    got = _histogram(cs_s, ev_log, N_DRAWS, 1, 5000)
    err = max(abs(a - b) for a, b in zip(got, want))
    print("  log   Eq (6), exact: max|empirical - analytic| = %.4f" % err)
    check(err < TOL, "exact sampler reproduces Eq (6) for logs")

    # Same target for MH.  One MH visit is not a draw from the conditional, so
    # the inner chain is run long enough to converge to it: agreement here is
    # the empirical evidence for the "same stationary distribution" claim.
    cs_mh_probe = CommunitySampler(ds_s.n_users, ds_s.edges, logs_s,
                                   ts_s.p_z_given_i(), hyp_s, random.Random(3),
                                   sampler="mh")
    cs_mh_probe.pz = cs_s.pz
    cs_mh_probe.cum_pz = cs_s.cum_pz
    _restore(cs_mh_probe, _snapshot(cs_s))
    got = _histogram(cs_mh_probe, ev_log, N_DRAWS, 40, 9000)
    err = max(abs(a - b) for a, b in zip(got, want))
    print("  log   Eq (6), mh   : max|empirical - analytic| = %.4f" % err)
    check(err < TOL, "MH targets the same conditional as the exact sampler")

    # ---------------- 5. stage 2: communities, exact ----------------
    print("\n[5] CommunitySampler, sampler='exact' (Eq 4, 6-9)")
    cs = CommunitySampler(ds.n_users, ds.edges, logs, p_z_i, hyp, random.Random(99),
                          sampler="exact")
    t0 = time.time()
    cs.run(80)
    dt_exact = time.time() - t0
    print("  80 sweeps in %.2fs" % dt_exact)
    probs = cs.validate_counts()
    check(not probs, "incremental counts (n_uc, n_cc, ncz, etatilde) stay exact: %s"
          % (probs or "clean"))
    pi_ = cs.pi()
    eta_ = cs.eta()
    theta_ = cs.theta()
    check(all(_close(sum(r), 1.0) for r in pi_), "Eq (7) pi rows are normalised")
    check(all(_close(sum(r), 1.0) for r in theta_), "Eq (9) theta rows are normalised")
    check(all(0.0 < x < 1.0 for r in eta_ for x in r), "Eq (8) eta in (0,1)")
    ari_exact = _adjusted_rand(planted_comm, _argmax_labels(pi_))
    print("  community ARI (exact) = %.3f" % ari_exact)
    check(ari_exact > 0.7, "Eq (19) argmax pi recovers the planted communities [exact]")

    # ---------------- 5. stage 2: communities, MH ----------------
    print("\n[6] CommunitySampler, sampler='mh'")
    cs_mh = CommunitySampler(ds.n_users, ds.edges, logs, p_z_i, hyp, random.Random(99),
                             sampler="mh")
    t0 = time.time()
    cs_mh.run(80)
    dt_mh = time.time() - t0
    print("  80 sweeps in %.2fs" % dt_mh)
    check(not cs_mh.validate_counts(), "MH sweep keeps the same count invariants")
    ari_mh = _adjusted_rand(planted_comm, _argmax_labels(cs_mh.pi()))
    print("  community ARI (mh)    = %.3f" % ari_mh)
    check(ari_mh > 0.7, "Eq (19) argmax pi recovers the planted communities [mh]")

    # ---------------- 6. determinism ----------------
    print("\n[7] determinism")
    cs_a = CommunitySampler(ds.n_users, ds.edges, logs, p_z_i, hyp, random.Random(5))
    cs_a.run(5)
    cs_b = CommunitySampler(ds.n_users, ds.edges, logs, p_z_i, hyp, random.Random(5))
    cs_b.run(5)
    check(cs_a.pi() == cs_b.pi(), "same seed -> bit-identical pi")

    # ---------------- 7. train_model + Model round-trip ----------------
    print("\n[8] train_model + Model.save/load")
    model = train_model(ds, logs, C=2, Z=2, n_iter_topic=80, n_iter_comm=60,
                        rng=random.Random(2024), verbose=True)
    ari_train = _adjusted_rand(planted_comm, _argmax_labels(model.pi))
    print("  community ARI (train_model) = %.3f" % ari_train)
    check(ari_train > 0.7, "train_model end-to-end recovers the planted communities")
    check(len(model.pi) == ds.n_users and len(model.pi[0]) == 2, "pi has shape U x C")
    check(len(model.eta) == 2 and len(model.eta[0]) == 2, "eta has shape C x C")
    check(len(model.theta) == 2 and len(model.theta[0]) == 2, "theta has shape C x Z")
    check(len(model.psi) == 2 and len(model.psi[0]) == ds.n_attrs, "psi has shape Z x F")
    check(len(model.p_z_given_i) == ds.n_items, "p_z_given_i has shape M x Z")

    tmp = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "ctim_gibbs_selftest_model.json"
    )
    model.save(tmp)
    reloaded = Model.load(tmp)
    check(reloaded.pi == model.pi and reloaded.theta == model.theta,
          "JSON round-trip preserves the parameters exactly")
    check(reloaded.hyper.eps0 == model.hyper.eps0 and reloaded.hyper.C == model.hyper.C,
          "JSON round-trip preserves the hyperparameters")
    with open(tmp, "r", encoding="utf-8") as fh:
        check("eps0" in fh.read(), "saved file is plain inspectable JSON")
    os.remove(tmp)

    print("\n" + "=" * 72)
    print("total self-test time: %.1fs" % (time.time() - t_start))
    if failures:
        print("SELF-TEST FAILED (%d):" % len(failures))
        for m in failures:
            print("  - %s" % m)
        sys.exit(1)
    print("SELF-TEST PASSED")
    print("=" * 72)
