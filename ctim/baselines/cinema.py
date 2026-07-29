"""CINEMA (Li et al. [24]) — conformity-aware community-based influence maximization.

Baseline for

    Huimin Huang, Hong Shen, Zaiqiao Meng, Huajian Chang, Huaiwen He.
    "Community-based influence maximization for viral marketing",
    Applied Intelligence (2019). DOI 10.1007/s10489-018-1387-8

referring to

    [24] Hui Li, Sourav S. Bhowmick, Aixin Sun.  "CINEMA: conformity-aware
    greedy algorithm for influence maximization in online social networks",
    EDBT 2013.

SPEC.md Table 2 classifies CINEMA as **community-based but NOT topic-aware**.
This module reproduces that classification faithfully, including the two
structural properties the CTIM paper criticises it for:

1.  **Community detection is independent of the diffusion model.**  CINEMA
    partitions ``G`` with an off-the-shelf structural method and only *then*
    runs influence maximization inside the resulting partition.  CTIM instead
    infers communities *jointly* with the diffusion parameters (Algorithm 1),
    so a single Gibbs run yields both.  CINEMA therefore pays for a full
    modularity agglomeration over every edge of ``G`` before it can select its
    first seed — one of the two sources of its extra running time.  We use a
    CNM/Louvain-style greedy modularity agglomeration (Clauset, Newman & Moore
    2004) written from scratch here: a max-heap over candidate merges with
    lazy re-validation, merging the smaller neighbour map into the larger, for
    roughly ``O(E log U)`` behaviour rather than the ``O(U^2)`` of the naive
    "score every pair" formulation.

2.  **Spread is estimated by Monte Carlo.**  CINEMA has no closed-form
    influence-computation model, so every candidate evaluation is an average
    over ``n_mc`` Independent-Cascade simulations.  CTIM's MIA model (Eq
    (13)-(18)) is exact and closed form.  This is the second — and larger —
    source of CINEMA's extra running time.

Both effects are *emergent*: nothing in this file slows the method down or
weakens its spread on purpose.  The expected outcome (SPEC.md Section 9:
CINEMA slower than CTIM by orders of magnitude and below the topic-aware
methods in spread) arises because (a) structural communities ignore who
actually influences whom, (b) the edge probabilities carry no topic
information, and (c) Monte-Carlo marginal gains are noisy.

Pipeline
--------
``cinema_select_seeds`` performs, in order:

    (i)   structural community detection over G          `detect_communities_modularity`
    (ii)  conformity / influence-index estimation        `estimate_conformity`
    (iii) conformity-aware edge reweighting              `conformity_aware_pp`
    (iv)  per-community CELF Monte-Carlo greedy          `celf_greedy_community`
    (v)   dynamic-programming allocation of K seeds      `dp_allocate`
          across communities (same DP family as CGA [22] / Algorithm 2 lines
          25-45: an optimal budget split given each community's spread curve)

Standard library only.  Python 3.9 compatible.  Every stochastic step is
driven by an explicit ``random.Random`` instance; the module-level ``random.*``
functions are never used.
"""

from __future__ import annotations

import heapq
import math
import random
import time
from collections import defaultdict

# Allow `python3 ctim/baselines/cinema.py` as well as `python3 -m
# ctim.baselines.cinema` by putting the repository root on sys.path when this
# file is executed as a top-level script.
if __package__ in (None, ""):  # pragma: no cover - script-execution convenience
    import os as _os
    import sys as _sys

    _sys.path.insert(
        0,
        _os.path.dirname(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        ),
    )

from ctim.baselines import RunResult
from ctim.influence import MIA

__all__ = [
    "DEFAULT_CONFORMITY_DELTA",
    "DEFAULT_UNIFORM_P",
    "ConformityProfile",
    "detect_communities_modularity",
    "modularity",
    "n_communities",
    "estimate_conformity",
    "conformity_aware_pp",
    "mc_spread",
    "celf_greedy_community",
    "dp_allocate",
    "cinema_select_seeds",
    "select_seeds",
]


#: Default time window (seconds) within which a follower's adoption counts as
#: "after" an in-neighbour's.  30 days, the same order as the ``Delta`` of
#: SPEC.md Definition 1.
DEFAULT_CONFORMITY_DELTA = 30 * 24 * 3600

#: Fallback per-edge propagation probability when the caller supplies no base
#: ``pp`` map.  CINEMA is topic-blind, so a single scalar is the honest default.
DEFAULT_UNIFORM_P = 0.05


# ===========================================================================
# (i)  Structural community detection — independent of the diffusion model
# ===========================================================================


def _undirected_projection(n_users, edges):
    """Undirected, weighted projection of the directed graph ``G``.

    Each directed arc ``(u, v)`` contributes weight 1 to the undirected pair
    ``{u, v}``; a reciprocated pair therefore has weight 2, which is the usual
    way of saying "these two are more strongly tied".  Self-loops are dropped
    (they contribute nothing to modularity differences).

    Returns ``(nbr, deg, m)`` where

        nbr[c]  dict  neighbour community -> total weight between them
        deg[c]  float weighted degree of community c
        m       float total undirected edge weight (so 2m = sum of degrees)

    Nodes with no incident arc are absent from ``nbr``; they are handled as
    permanent singletons by the caller.
    """
    nbr = {}
    deg = defaultdict(float)
    m = 0.0
    for (u, v) in edges:
        if u == v:
            continue
        au = nbr.get(u)
        if au is None:
            au = nbr[u] = {}
        av = nbr.get(v)
        if av is None:
            av = nbr[v] = {}
        au[v] = au.get(v, 0.0) + 1.0
        av[u] = av.get(u, 0.0) + 1.0
        deg[u] += 1.0
        deg[v] += 1.0
        m += 1.0
    return nbr, deg, m


def _find(par, x):
    """Union-find root with path compression (iterative, no recursion limit)."""
    root = x
    while root in par:
        root = par[root]
    while x in par and par[x] != root:
        nxt = par[x]
        par[x] = root
        x = nxt
    return root


def _relabel(n_users, par):
    """Turn the union-find forest into a dense ``comm[v] in [0, n_comm)`` list.

    Labels are assigned in increasing order of each community's smallest member
    id, so the output is fully deterministic and independent of merge order.
    """
    roots = [_find(par, v) for v in range(n_users)]
    order = {}
    for v in range(n_users):
        r = roots[v]
        if r not in order:
            order[r] = len(order)
    return [order[roots[v]] for v in range(n_users)]


def detect_communities_modularity(n_users, edges, target_c=0,
                                  force_exact=False, verbose=False):
    """CNM/Louvain-style greedy modularity agglomeration.  Returns ``comm[v]``.

    This is CINEMA's community detection step, and it deliberately looks at
    **nothing but the graph structure** — no adoption logs, no diffusion
    parameters, no topics.  That independence is precisely the property the
    CTIM paper criticises (SPEC.md Section 7).

    Method.  Every node starts in its own community.  For the undirected
    projection with total edge weight ``m``, merging communities ``a`` and
    ``b`` changes Newman's modularity

        Q = sum_c [ L_c / m  -  (D_c / 2m)^2 ]

    (``L_c`` = internal weight of ``c``, ``D_c`` = weighted degree of ``c``) by
    exactly

        dQ(a, b) = W_ab / m  -  D_a * D_b / (2 m^2)

    where ``W_ab`` is the weight of edges between them.  We repeatedly apply
    the merge with the largest ``dQ``.

    Complexity.  Candidate merges live in a binary max-heap whose entries are
    allowed to go stale and are re-validated (and re-pushed) on pop, and each
    merge folds the smaller neighbour map into the larger.  That gives the
    standard CNM bound of roughly ``O(E log U)`` on sparse graphs, not the
    ``O(U^2)`` of rescoring all pairs.

    Parameters
    ----------
    n_users, edges
        The directed graph; ``edges`` is a list of ``(u, v)``.
    target_c
        If > 0, stop the agglomeration as soon as this many communities remain
        (SPEC.md Section 8 runs CINEMA at ``C = 100``).  If 0, run to the
        modularity maximum, i.e. stop when no merge has ``dQ > 0``.
    force_exact
        Only meaningful with ``target_c > 0``.  A graph with more connected
        components than ``target_c`` cannot reach the target by modularity
        merges alone (there is no edge to merge across).  With
        ``force_exact=True`` the leftover components are then coalesced
        smallest-first until exactly ``target_c`` communities remain.  This is
        a documented deviation, needed only to honour a hard ``C``; it is off
        by default, in which case the returned partition may be coarser than
        requested and the true count is reported by ``n_communities()``.

    Returns
    -------
    list
        ``comm[v] in [0, n_comm)`` for every ``v in [0, n_users)``.
    """
    if n_users <= 0:
        return []

    nbr, deg, m = _undirected_projection(n_users, edges)
    par = {}
    n_comm = n_users

    if m <= 0.0:
        # No edges at all: every node is its own community.
        if target_c > 0 and force_exact:
            _coalesce_to_target(n_users, par, target_c)
        return _relabel(n_users, par)

    inv_m = 1.0 / m
    inv_2m2 = 1.0 / (2.0 * m * m)

    def dq_of(a, b, w):
        # dQ(a,b) = W_ab/m - D_a*D_b/(2 m^2)
        return w * inv_m - deg[a] * deg[b] * inv_2m2

    alive = set(nbr.keys())

    # ---- one heap entry per community, not per pair ----------------------
    # Pushing a refreshed entry for every neighbour after each merge makes the
    # heap grow without bound on a real graph: on Digg (30,358 nodes) merged
    # communities reach ~7,000 neighbours and the heap passes 27M entries
    # before the run is half done, so it never finishes.  Clauset-Newman-Moore
    # keep only each community's BEST partner in the global heap; the global
    # max over communities equals the global max over pairs, so the merge order
    # is identical.  Generation stamps mark entries stale, and a stale pop
    # re-prices that community and re-publishes it (rather than dropping it),
    # preserving "every live community with a neighbour has an entry".
    # This mirrors the same fix in ctim/baselines/air_cga.py.
    gen = dict((a, 0) for a in nbr)

    def best_of(a):
        """Community a's highest-dQ partner, ties broken on the lower id."""
        bq = None
        bb = None
        for b, w in nbr[a].items():
            q = dq_of(a, b, w)
            if bq is None or q > bq or (q == bq and b < bb):
                bq, bb = q, b
        return bq, bb

    def entry(owner, other, q):
        # Canonical (lower, higher) ordering so ties break exactly as the
        # pairwise heap keyed on (-dq, a, b) did.
        x, y = (owner, other) if owner < other else (other, owner)
        return (-q, x, y, owner, gen[x], gen[y])

    heap = []
    for a in sorted(nbr.keys()):
        q, b = best_of(a)
        if b is not None:
            heap.append(entry(a, b, q))
    heapq.heapify(heap)

    def republish(owner):
        if owner in alive:
            q, b = best_of(owner)
            if b is not None:
                heapq.heappush(heap, entry(owner, b, q))

    n_merges = 0
    stop_at = target_c if target_c > 0 else 0

    while heap and (stop_at <= 0 or n_comm > stop_at):
        negdq, x, y, owner, gx, gy = heapq.heappop(heap)
        if owner not in alive:
            continue
        if (x not in alive or y not in alive
                or gen.get(x) != gx or gen.get(y) != gy):
            republish(owner)  # outdated: re-price this community and retry
            continue
        a, b = (x, y) if x == owner else (y, x)
        w = nbr[a].get(b)
        if w is None:
            republish(owner)  # no longer adjacent
            continue
        dq = -negdq
        if stop_at <= 0 and dq <= 0.0:
            break  # modularity maximum reached

        # -- merge b into a (or a into b): keep the larger neighbour map so
        #    that the fold below is "smaller into larger"; ties by smaller id.
        la, lb = len(nbr[a]), len(nbr[b])
        if la > lb or (la == lb and a < b):
            keep, gone = a, b
        else:
            keep, gone = b, a

        # Only dQ is ever needed, and dQ depends on W and the degrees alone --
        # the internal weights L_c never have to be maintained here.
        nbr[keep].pop(gone, None)
        nbr[gone].pop(keep, None)
        deg[keep] += deg[gone]

        nk = nbr[keep]
        for x, wx in nbr[gone].items():
            nbr[x].pop(gone, None)
            if x in nk:
                nk[x] += wx
                nbr[x][keep] += wx
            else:
                nk[x] = wx
                nbr[x][keep] = wx
        del nbr[gone]
        deg.pop(gone, None)
        alive.discard(gone)
        gen.pop(gone, None)
        par[gone] = keep
        n_comm -= 1
        n_merges += 1

        # `keep` absorbed `gone`, so every entry touching `keep` is outdated.
        # Bump its stamp and re-publish only its own best partner: O(deg) work
        # and ONE push instead of one push per neighbour.  Neighbours pointing
        # at `keep` are re-priced lazily by `republish` when they surface.
        gen[keep] = gen.get(keep, 0) + 1
        republish(keep)

        if verbose and n_merges % 10000 == 0:
            print("  [cinema/cnm] %d merges, %d communities" % (n_merges, n_comm))

    if target_c > 0 and force_exact and n_comm > target_c:
        _coalesce_to_target(n_users, par, target_c)

    return _relabel(n_users, par)


def _coalesce_to_target(n_users, par, target_c):
    """Merge the smallest remaining communities until exactly ``target_c`` remain.

    Only used with ``force_exact=True``; see ``detect_communities_modularity``.
    Deterministic: always merges the two smallest communities, ties by smallest
    representative id.
    """
    sizes = defaultdict(int)
    for v in range(n_users):
        sizes[_find(par, v)] += 1
    if len(sizes) <= target_c or target_c <= 0:
        return
    heap = [(sz, r) for r, sz in sizes.items()]
    heapq.heapify(heap)
    n_comm = len(sizes)
    while n_comm > target_c and len(heap) >= 2:
        sa, a = heapq.heappop(heap)
        sb, b = heapq.heappop(heap)
        keep, gone = (a, b) if a < b else (b, a)
        par[gone] = keep
        heapq.heappush(heap, (sa + sb, keep))
        n_comm -= 1


def modularity(n_users, edges, comm):
    """Newman modularity Q of partition ``comm`` on the undirected projection.

    Q = sum_c [ L_c / m - (D_c / 2m)^2 ].  Used by the self-test to verify that
    the agglomeration actually recovers structure; not part of the pipeline.
    """
    m = 0.0
    L = defaultdict(float)
    D = defaultdict(float)
    for (u, v) in edges:
        if u == v:
            continue
        cu, cv = comm[u], comm[v]
        D[cu] += 1.0
        D[cv] += 1.0
        m += 1.0
        if cu == cv:
            L[cu] += 1.0
    if m <= 0.0:
        return 0.0
    q = 0.0
    for c in set(list(L.keys()) + list(D.keys())):
        q += L[c] / m - (D[c] / (2.0 * m)) ** 2
    return q


def n_communities(comm):
    """Number of distinct communities in a ``comm[v]`` assignment."""
    return len(set(comm)) if comm else 0


# ===========================================================================
# (ii)  Conformity and influence indices, estimated from the adoption logs
# ===========================================================================


class ConformityProfile:
    """Per-user conformity and influence indices estimated from adoption logs.

    Intuition (Li et al. [24]).  CINEMA's central claim is that a user's
    behaviour splits into two distinct roles that classical IM conflates:

    * **conformity** — the user's *susceptibility*, how readily they follow
      others.  A node that repeatedly adopts items **after** its in-neighbours
      have already adopted them is a conformist; a node that adopts **first**,
      before any in-neighbour, is a non-conformist (an independent taste-maker).
      In [24] conformity is read off a signed network via a modified PageRank;
      here the signed network is unavailable, so we read the same quantity off
      the adoption logs, which is what the CTIM paper's datasets provide.
    * **influence index** — the user's *ability to persuade*, how often their
      own adoption is followed by an out-neighbour.

    The two are genuinely different: a highly conformist user may still be
    influential (they adopt late but their followers copy them), and a
    taste-maker with no followers has low conformity and low influence.  Keeping
    them separate is exactly what CINEMA buys over plain greedy.

    Estimators.  For every user ``v`` and item ``i`` that ``v`` adopted (only
    ``v``'s FIRST adoption of ``i`` counts, per SPEC.md Section 8):

        follow event   some in-neighbour ``u`` of ``v`` adopted ``i`` at ``t_u``
                       with ``0 < t_v - t_u <= delta``
        lead event     otherwise

        conformity[v]      = (n_follow[v] + s*g_f) / (n_adopt[v] + s)
        influence_index[u] = (n_succ[u]   + s*g_i) / (n_opp[u]   + s)

    where ``n_succ[u]`` counts the (item, out-neighbour) pairs that ``u``
    demonstrably preceded within ``delta`` — i.e. exactly the potential-influence
    logs of SPEC.md Definition 1 sourced at ``u`` — and ``n_opp[u] = n_adopt[u] *
    outdeg(u)`` is the number of chances ``u`` had.  ``s`` is the smoothing
    strength and ``g_f``, ``g_i`` are the corresponding global rates, so the
    estimate is the posterior mean of a Beta prior centred on the population
    average (empirical-Bayes shrinkage).  Users with no logs fall back to the
    global rate rather than to an arbitrary constant.  Both indices lie in
    ``[0, 1]``.

    Attributes
    ----------
    conformity        list, per user
    influence_index   list, per user
    n_follow, n_adopt, n_succ, n_opp   raw counts, per user
    global_follow_rate, global_influence_rate   the shrinkage targets
    delta, prior      the settings used
    """

    __slots__ = (
        "conformity", "influence_index", "n_follow", "n_adopt", "n_succ",
        "n_opp", "global_follow_rate", "global_influence_rate", "delta",
        "prior", "seconds",
    )

    def __init__(self, conformity, influence_index, n_follow, n_adopt, n_succ,
                 n_opp, global_follow_rate, global_influence_rate, delta,
                 prior, seconds=0.0):
        self.conformity = conformity
        self.influence_index = influence_index
        self.n_follow = n_follow
        self.n_adopt = n_adopt
        self.n_succ = n_succ
        self.n_opp = n_opp
        self.global_follow_rate = global_follow_rate
        self.global_influence_rate = global_influence_rate
        self.delta = delta
        self.prior = prior
        self.seconds = seconds

    def summary(self):
        """Small dict of diagnostics, suitable for ``RunResult.extra``."""
        conf = self.conformity
        infl = self.influence_index
        n = len(conf) if conf else 1
        return {
            "conformity_mean": sum(conf) / n,
            "conformity_max": max(conf) if conf else 0.0,
            "conformity_min": min(conf) if conf else 0.0,
            "influence_index_mean": sum(infl) / n,
            "influence_index_max": max(infl) if infl else 0.0,
            "global_follow_rate": self.global_follow_rate,
            "global_influence_rate": self.global_influence_rate,
            "n_users_with_adoptions": sum(1 for x in self.n_adopt if x > 0),
            "delta": self.delta,
            "prior": self.prior,
        }

    def __repr__(self):
        return ("ConformityProfile(users={}, mean_conformity={:.4f}, "
                "mean_influence={:.4f})".format(
                    len(self.conformity),
                    sum(self.conformity) / max(1, len(self.conformity)),
                    sum(self.influence_index) / max(1, len(self.influence_index)),
                ))


# Flip the inner loop when walking an item's adopters is this many times
# cheaper than walking a user's out-neighbours (mirrors ctim/dataset.py).
_FLIP_FACTOR = 4


def estimate_conformity(ds, delta=DEFAULT_CONFORMITY_DELTA, prior=1.0,
                        logs=None):
    """Estimate every user's conformity and influence index.  See
    :class:`ConformityProfile` for the definitions and the intuition from [24].

    ``logs`` defaults to ``ds.logs``; pass the training split to keep the test
    adoptions unseen.  Complexity is the same as SPEC.md Definition 1's log
    construction: ``O(sum_i sum_{u in adopters(i)} min(outdeg(u), |adopters(i)|))``.
    """
    t0 = time.time()
    n_users = ds.n_users
    out_adj = ds.out_adj
    if logs is None:
        logs = ds.logs

    # First adoption time per (item, user); repeated adoptions are dropped.
    by_item = {}
    for (u, i, t) in logs:
        bucket = by_item.get(i)
        if bucket is None:
            bucket = by_item[i] = {}
        prev = bucket.get(u)
        if prev is None or t < prev:
            bucket[u] = t

    n_follow = [0] * n_users
    n_adopt = [0] * n_users
    n_succ = [0] * n_users
    out_sorted = [None] * n_users

    for i in sorted(by_item.keys()):
        times = by_item[i]
        n_adopters = len(times)
        adopters = sorted(times.keys())
        for v in adopters:
            n_adopt[v] += 1
        if n_adopters < 2:
            continue

        followed = set()
        for u in adopters:
            tu = times[u]
            nbrs = out_adj[u] if u < len(out_adj) else ()
            degu = len(nbrs)
            if degu == 0:
                continue
            if n_adopters * _FLIP_FACTOR < degu:
                # Cheaper to walk the adopters and test arc membership.
                srt = out_sorted[u]
                if srt is None:
                    srt = out_sorted[u] = sorted(set(nbrs))
                for v in adopters:
                    dt = times[v] - tu
                    if dt <= 0 or dt > delta:
                        continue
                    lo, hi = 0, len(srt)
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if srt[mid] < v:
                            lo = mid + 1
                        else:
                            hi = mid
                    if lo < len(srt) and srt[lo] == v:
                        followed.add(v)
                        n_succ[u] += 1
            else:
                for v in nbrs:
                    tv = times.get(v)
                    if tv is None:
                        continue
                    dt = tv - tu
                    if dt <= 0 or dt > delta:
                        continue
                    followed.add(v)
                    n_succ[u] += 1
        for v in followed:
            n_follow[v] += 1

    # Opportunities to influence: one per (adopted item, out-neighbour).
    n_opp = [n_adopt[u] * (len(out_adj[u]) if u < len(out_adj) else 0)
             for u in range(n_users)]

    tot_follow = sum(n_follow)
    tot_adopt = sum(n_adopt)
    tot_succ = sum(n_succ)
    tot_opp = sum(n_opp)
    g_f = (tot_follow / tot_adopt) if tot_adopt > 0 else 0.0
    g_i = (tot_succ / tot_opp) if tot_opp > 0 else 0.0

    s = float(prior)
    conformity = [
        (n_follow[v] + s * g_f) / (n_adopt[v] + s) if (n_adopt[v] + s) > 0 else g_f
        for v in range(n_users)
    ]
    influence_index = [
        (n_succ[u] + s * g_i) / (n_opp[u] + s) if (n_opp[u] + s) > 0 else g_i
        for u in range(n_users)
    ]

    return ConformityProfile(
        conformity=conformity,
        influence_index=influence_index,
        n_follow=n_follow,
        n_adopt=n_adopt,
        n_succ=n_succ,
        n_opp=n_opp,
        global_follow_rate=g_f,
        global_influence_rate=g_i,
        delta=delta,
        prior=prior,
        seconds=time.time() - t0,
    )


# ===========================================================================
# (iii)  Conformity-aware edge reweighting  (CINEMA's C^2 model)
# ===========================================================================


def conformity_aware_pp(ds, pp_base, profile, preserve_mean=True):
    """Reweight every arc by the *target's conformity* and the *source's
    influence index*, as in CINEMA's conformity-aware cascade model [24].

        pp(u, v)  =  scale * pp_base(u, v) * infl(u)/mean_infl * conf(v)/mean_conf

    The two ratios are the whole point of [24]: an arc into a highly conformist
    target, out of a demonstrably persuasive source, carries more probability
    than the topic-blind base rate; an arc into an independent taste-maker
    carries less.

    ``preserve_mean=True`` chooses the single global ``scale`` that makes the
    mean reweighted probability equal the mean base probability.  Without it,
    CINEMA's spread would differ from the other baselines' partly because its
    edges are systematically weaker or stronger, which would confound the
    comparison; with it, CINEMA differs only in *where* the probability mass
    sits, which is the property being tested.  Results are clamped into
    ``[0, 1]``.

    Returns ``(pp, stats)`` — the new ``{(u, v): p}`` map and a diagnostics dict
    (scale applied, clamp count, mean before/after).
    """
    conf = profile.conformity
    infl = profile.influence_index
    n = len(conf) if conf else 1
    mean_conf = sum(conf) / n
    mean_infl = sum(infl) / n

    edges = ds.edges
    if not edges:
        return {}, {"scale": 1.0, "n_clamped": 0, "mean_base": 0.0, "mean_new": 0.0,
                    "degenerate": True}

    def base_of(u, v):
        if pp_base is None:
            return DEFAULT_UNIFORM_P
        return pp_base.get((u, v), 0.0)

    degenerate = mean_conf <= 0.0 or mean_infl <= 0.0
    if degenerate:
        # No usable log signal (e.g. no adoptions at all).  Fall back to the
        # base probabilities untouched rather than zeroing the whole graph.
        pp = {(u, v): base_of(u, v) for (u, v) in edges}
        mb = sum(pp.values()) / len(pp)
        return pp, {"scale": 1.0, "n_clamped": 0, "mean_base": mb,
                    "mean_new": mb, "degenerate": True}

    inv_conf = 1.0 / mean_conf
    inv_infl = 1.0 / mean_infl

    raw = {}
    sum_base = 0.0
    sum_raw = 0.0
    for (u, v) in edges:
        b = base_of(u, v)
        # CINEMA C^2: source's influence index x target's conformity.
        r = b * (infl[u] * inv_infl) * (conf[v] * inv_conf)
        raw[(u, v)] = r
        sum_base += b
        sum_raw += r

    scale = 1.0
    if preserve_mean and sum_raw > 0.0:
        scale = sum_base / sum_raw

    pp = {}
    n_clamped = 0
    total = 0.0
    for key, r in raw.items():
        p = r * scale
        if p > 1.0:
            p = 1.0
            n_clamped += 1
        elif p < 0.0:
            p = 0.0
            n_clamped += 1
        pp[key] = p
        total += p

    return pp, {
        "scale": scale,
        "n_clamped": n_clamped,
        "mean_base": sum_base / len(edges),
        "mean_new": total / len(edges),
        "mean_conformity": mean_conf,
        "mean_influence_index": mean_infl,
        "degenerate": False,
    }


# ===========================================================================
# (iv)  Monte-Carlo Independent Cascade + CELF greedy inside a community
# ===========================================================================


def mc_spread(sub_out, S, n_mc, seed):
    """Mean activated-node count over ``n_mc`` Independent-Cascade simulations.

    ``sub_out`` maps ``u -> [(v, p), ...]``; restricting it to a community's
    induced subgraph is what makes this the *within-community* spread that the
    DP allocation needs.

    ``seed`` fixes an explicit ``random.Random`` — never the module-level
    ``random.*``.  Reusing the same ``seed`` for every evaluation inside one
    community implements *common random numbers*: candidate comparisons then
    share their simulation noise, which both removes most of the variance from
    the greedy's argmax and makes the whole run reproducible.

    This is CINEMA's influence-computation model and the dominant cost of the
    method — contrast MIA (Eq (13)-(18)), which is exact and closed form.
    """
    S = list(S)
    if not S:
        return 0.0
    if n_mc <= 0:
        return float(len(S))
    rnd = random.Random(seed).random
    total = 0
    for _ in range(n_mc):
        active = set(S)
        frontier = S
        while frontier:
            nxt = []
            for u in frontier:
                for (v, p) in sub_out.get(u, ()):
                    if v not in active and rnd() < p:
                        active.add(v)
                        nxt.append(v)
            frontier = nxt
        total += len(active)
    return total / float(n_mc)


def celf_greedy_community(nodes, sub_out, K, n_mc, seed, use_celf=True):
    """Greedy seed selection inside one community, with CELF lazy-forward.

    Returns ``(seeds, curve)`` where ``curve[j]`` is the Monte-Carlo estimate of
    the spread of ``seeds[:j]`` within the community (``curve[0] == 0.0``).  The
    curve is the per-community input to the DP allocation.

    CELF (Leskovec et al.) exploits submodularity: a candidate's marginal gain
    can only shrink as the seed set grows, so a candidate whose *stale* gain
    already tops the heap is guaranteed optimal and needs no re-evaluation.
    With ``use_celf=False`` the plain Kempe et al. greedy re-scores every
    candidate every round — same answer under common random numbers, far more
    simulations.  The flag is reported in ``RunResult.extra``.

    The curve is forced non-decreasing (running max).  Monte-Carlo noise can
    otherwise make a longer prefix score marginally below a shorter one, which
    would let the DP "buy" a seed for negative gain; the true spread is
    monotone by construction, so this only removes estimator artefacts.
    """
    cand = sorted(nodes)
    K = min(K, len(cand))
    if K <= 0 or not cand:
        return [], [0.0]

    S = []
    cur = 0.0
    curve = [0.0]

    if not use_celf:
        remaining = set(cand)
        while len(S) < K and remaining:
            best_u, best_val = None, None
            for u in sorted(remaining):
                val = mc_spread(sub_out, S + [u], n_mc, seed)
                if best_val is None or val > best_val:
                    best_u, best_val = u, val
            if best_u is None or best_val <= cur:
                break
            S.append(best_u)
            remaining.discard(best_u)
            cur = best_val
            curve.append(cur)
        return S, _running_max(curve)

    # CELF.  Heap entries are (-marginal_gain, node, |S| when it was computed).
    heap = []
    for u in cand:
        # Marginal gain at S = {} is simply sigma({u}).
        heap.append((-mc_spread(sub_out, [u], n_mc, seed), u, 0))
    heapq.heapify(heap)

    in_S = set()
    while len(S) < K and heap:
        neg, u, last = heapq.heappop(heap)
        if u in in_S:
            continue
        if last == len(S):
            gain = -neg
            if gain <= 0.0 and S:
                break  # no candidate can add anything
            cur = cur + gain  # == sigma(S u {u}) by construction of `gain`
            S.append(u)
            in_S.add(u)
            curve.append(cur)
        else:
            gain = mc_spread(sub_out, S + [u], n_mc, seed) - cur
            heapq.heappush(heap, (-gain, u, len(S)))

    return S, _running_max(curve)


def _running_max(curve):
    out = []
    best = 0.0
    for x in curve:
        if x > best:
            best = x
        out.append(best)
    return out


# ===========================================================================
# (v)  Dynamic-programming allocation of the seed budget across communities
# ===========================================================================


def dp_allocate(curves, K):
    """Optimally split a budget of ``K`` seeds across communities.

    Same DP family as CGA [22] (and Algorithm 2 lines 25-45, which walks the
    same table one seed at a time): given each community ``m``'s spread curve
    ``g_m[j]`` — the estimated spread of its best ``j`` seeds — choose
    ``j_1 + ... + j_C <= K`` maximising ``sum_m g_m[j_m]``.

        I[m][k] = max_{0 <= j <= k} ( I[m-1][k-j] + g_m[j] )

    Because every ``g_m`` is non-decreasing with ``g_m[0] = 0``, "at most K" and
    "exactly K" coincide at the optimum.  Cost ``O(C * K^2)``.

    Returns ``(alloc, best_value)`` with ``alloc[m]`` the number of seeds given
    to community ``m``.  Ties go to the smaller ``j`` and then the earlier
    community, so the result is deterministic.
    """
    n_comm = len(curves)
    if K <= 0 or n_comm == 0:
        return [0] * n_comm, 0.0

    prev = [0.0] * (K + 1)
    choice = []
    for m in range(n_comm):
        g = curves[m]
        gmax = len(g) - 1
        cur = [0.0] * (K + 1)
        ch = [0] * (K + 1)
        for k in range(K + 1):
            best_val = prev[k] + g[0]
            best_j = 0
            hi = k if k < gmax else gmax
            for j in range(1, hi + 1):
                val = prev[k - j] + g[j]
                if val > best_val:
                    best_val = val
                    best_j = j
            cur[k] = best_val
            ch[k] = best_j
        prev = cur
        choice.append(ch)

    alloc = [0] * n_comm
    k = K
    for m in range(n_comm - 1, -1, -1):
        j = choice[m][k]
        alloc[m] = j
        k -= j
    return alloc, prev[K]


# ===========================================================================
# CINEMA — top level
# ===========================================================================


def cinema_select_seeds(ds, pp, K, rng, n_mc=200, C=100, use_celf=True,
                        logs=None, delta=DEFAULT_CONFORMITY_DELTA, prior=1.0,
                        comm=None, force_exact=False, cand_top=0,
                        preserve_mean=True, h=0.1, evaluate=True,
                        verbose=False):
    """CINEMA (Li et al. [24]).  Returns a :class:`RunResult`.

    Parameters
    ----------
    ds
        A ``ctim.dataset.Dataset``.
    pp
        Base ``{(u, v): p}`` propagation probabilities, or ``None`` for a
        uniform ``DEFAULT_UNIFORM_P``.  CINEMA is **not topic-aware** (SPEC.md
        Table 2), so this must be a single topic-independent map — the same
        contract as ``greedy.greedy_select``.  It is then reweighted by
        conformity, which is the only user-level signal CINEMA adds.
    K
        Seed-set size.
    rng
        Explicit ``random.Random``; seeds every Monte-Carlo stream.
    n_mc
        Simulations per spread evaluation.  Li et al. use 10,000; 200 is the
        default here so a smoke run finishes, and the value is reported.
    C
        Target community count (SPEC.md Section 8 runs CINEMA at ``C = 100``).
        ``0`` runs the agglomeration to its own modularity maximum.
    use_celf
        CELF lazy-forward inside each community.  Reported in ``extra``.
    logs
        Adoption logs used for conformity estimation; defaults to ``ds.logs``.
        Pass the training split to keep test adoptions unseen.
    comm
        Pre-computed ``comm[v]``; skips step (i).  Supplying it means the
        reported ``seconds`` no longer includes community detection.
    cand_top
        If > 0, only the ``cand_top`` highest weighted-out-degree nodes of each
        community are considered as candidates.  An **approximation**, OFF by
        default, needed only to make very large graphs tractable; flagged in
        ``extra["approximate"]``.
    h
        MIA threshold used by the final evaluation only (Eq (15)).
    evaluate
        If True, ``RunResult.spread`` is ``MIA.influence(S)`` on the full graph
        under CINEMA's own edge weights (exact, Eq (17)/(18)), so it is
        comparable with the other methods.  The method's *internal* Monte-Carlo
        DP estimate is always reported as ``extra["dp_estimate"]``.

    Notes
    -----
    Running time is dominated by (i) the structural agglomeration over all of
    ``E`` and (iv) the Monte-Carlo evaluations — the two costs the CTIM paper
    identifies.  Neither is inflated artificially; ``extra`` breaks the wall
    clock down per stage so the claim can be checked.
    """
    if rng is None:
        raise ValueError("cinema_select_seeds requires an explicit random.Random")
    t_start = time.time()
    timings = {}

    # -- (i) structural community detection, independent of the diffusion model
    t0 = time.time()
    if comm is None:
        comm = detect_communities_modularity(
            ds.n_users, ds.edges, target_c=C, force_exact=force_exact,
            verbose=verbose,
        )
        detected = True
    else:
        comm = list(comm)
        detected = False
    timings["community_detection"] = time.time() - t0
    n_comm = n_communities(comm)
    if verbose:
        print("[cinema] %d communities (target %s)" % (n_comm, C or "auto"))

    # -- (ii) conformity / influence indices from the adoption logs
    t0 = time.time()
    profile = estimate_conformity(ds, delta=delta, prior=prior, logs=logs)
    timings["conformity"] = time.time() - t0

    # -- (iii) conformity-aware edge weights
    t0 = time.time()
    pp_c, pp_stats = conformity_aware_pp(ds, pp, profile,
                                         preserve_mean=preserve_mean)
    timings["edge_weights"] = time.time() - t0

    # -- induced subgraphs, one per community
    t0 = time.time()
    by_comm = defaultdict(list)
    for v in range(ds.n_users):
        by_comm[comm[v]].append(v)
    sub_out = {}
    w_out = defaultdict(float)
    for (u, v) in ds.edges:
        if comm[u] != comm[v]:
            continue
        p = pp_c.get((u, v), 0.0)
        if p <= 0.0:
            continue
        lst = sub_out.get(u)
        if lst is None:
            lst = sub_out[u] = []
        lst.append((v, p))
        w_out[u] += p
    timings["subgraphs"] = time.time() - t0

    # -- (iv) per-community CELF Monte-Carlo greedy
    t0 = time.time()
    order = sorted(by_comm.keys())
    curves = []
    seq = []
    n_evals_communities = 0
    for c in order:
        nodes = by_comm[c]
        if cand_top > 0 and len(nodes) > cand_top:
            nodes = sorted(nodes, key=lambda x: (-w_out.get(x, 0.0), x))[:cand_top]
        # One fixed stream per community => common random numbers within it.
        c_seed = rng.randrange(1 << 30)
        seeds_c, curve_c = celf_greedy_community(
            nodes, sub_out, K, n_mc, c_seed, use_celf=use_celf,
        )
        curves.append(curve_c)
        seq.append(seeds_c)
        n_evals_communities += 1
    timings["community_greedy"] = time.time() - t0

    # -- (v) DP allocation of the K seeds across communities
    t0 = time.time()
    alloc, dp_value = dp_allocate(curves, K)
    timings["dp_allocation"] = time.time() - t0

    seeds = []
    for idx, c in enumerate(order):
        take = alloc[idx]
        if take > 0:
            seeds.extend(seq[idx][:take])
    # Selection order: strongest communities first, then within-community
    # greedy order.  Ties broken by community index for determinism.
    seeds = seeds[:K]

    seconds = time.time() - t_start

    spread = float(dp_value)
    if evaluate and seeds:
        mia = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp_c, h=h)
        spread = mia.influence(seeds)  # Eq (17)/(18), same evaluator as CTIM

    extra = {
        "method": "CINEMA",
        "community_based": True,
        "topic_aware": False,
        "n_communities": n_comm,
        "community_detection": "greedy modularity agglomeration (CNM), "
                               "independent of the diffusion model",
        "communities_detected_here": detected,
        "target_C": C,
        "force_exact": force_exact,
        "modularity": modularity(ds.n_users, ds.edges, comm),
        "n_mc": n_mc,
        "use_celf": use_celf,
        "influence_model": "Monte-Carlo Independent Cascade",
        "dp_estimate": float(dp_value),
        "allocation": {order[i]: alloc[i] for i in range(len(order)) if alloc[i]},
        "approximate": cand_top > 0,
        "cand_top": cand_top,
        "timings": timings,
        "pp_stats": pp_stats,
        "conformity": profile.summary(),
        "h": h,
    }
    return RunResult(seeds=seeds, spread=float(spread), seconds=seconds,
                     extra=extra)


def select_seeds(ds, pp, K, rng, **kwargs):
    """Generic baseline entry point (API.md, ``ctim/baselines/``).

    Thin alias for :func:`cinema_select_seeds`.
    """
    return cinema_select_seeds(ds, pp, K, rng, **kwargs)


# ===========================================================================
# Self-test
# ===========================================================================


def _self_test():  # pragma: no cover - executable documentation
    import itertools
    import sys

    from ctim.dataset import Dataset, build_adjacency

    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("  PASS  %s%s" % (name, (" -- " + detail) if detail else ""))
        else:
            print("  FAIL  %s%s" % (name, (" -- " + detail) if detail else ""))
            failures.append(name)

    rng = random.Random(20190408)

    # ------------------------------------------------------------------ [1]
    print("[1] CNM greedy modularity agglomeration recovers planted blocks")
    # 4 blocks of 25 nodes: dense inside, sparse across.
    n_block, n_blocks = 25, 4
    n_users = n_block * n_blocks
    block_of = [v // n_block for v in range(n_users)]
    eset = set()
    grng = random.Random(7)
    for a in range(n_users):
        for b in range(n_users):
            if a == b:
                continue
            p = 0.35 if block_of[a] == block_of[b] else 0.008
            if grng.random() < p:
                eset.add((a, b))
    edges = sorted(eset)

    comm = detect_communities_modularity(n_users, edges)
    q_found = modularity(n_users, edges, comm)
    q_true = modularity(n_users, edges, block_of)
    q_rand = modularity(n_users, edges,
                        [random.Random(1).randrange(4) for _ in range(n_users)])
    check("modularity of detected partition beats a random one",
          q_found > q_rand + 0.1, "Q=%.4f vs random %.4f" % (q_found, q_rand))
    check("modularity of detected partition is near the planted one",
          q_found >= q_true - 0.02,
          "Q_found=%.4f, Q_planted=%.4f" % (q_found, q_true))

    # Agreement with the planted blocks: for each detected community, the
    # fraction of its members in its majority block.
    grp = defaultdict(list)
    for v in range(n_users):
        grp[comm[v]].append(v)
    purity = 0
    for c, members in grp.items():
        counts = defaultdict(int)
        for v in members:
            counts[block_of[v]] += 1
        purity += max(counts.values())
    purity /= float(n_users)
    check("detected communities are pure w.r.t. the planted blocks",
          purity >= 0.90, "purity=%.3f, n_comm=%d" % (purity, n_communities(comm)))

    # Determinism.
    comm_b = detect_communities_modularity(n_users, edges)
    check("community detection is deterministic", comm == comm_b)

    # ------------------------------------------------------------------ [2]
    print("[2] target_c stops the agglomeration at exactly C communities")
    for target in (2, 3, 4, 8):
        ct = detect_communities_modularity(n_users, edges, target_c=target)
        check("target_c=%d yields %d communities" % (target, n_communities(ct)),
              n_communities(ct) == target)
    # Disconnected graph: more components than the target, force_exact needed.
    iso_edges = [(0, 1), (1, 0), (2, 3), (3, 2)]
    ci = detect_communities_modularity(8, iso_edges, target_c=2)
    check("without force_exact a component-limited graph stays coarser",
          n_communities(ci) > 2, "n_comm=%d" % n_communities(ci))
    cf = detect_communities_modularity(8, iso_edges, target_c=2, force_exact=True)
    check("force_exact reaches the target on a disconnected graph",
          n_communities(cf) == 2, "n_comm=%d" % n_communities(cf))
    check("comm labels are dense and in range",
          sorted(set(cf)) == list(range(n_communities(cf))))
    check("empty graph -> every node its own community",
          n_communities(detect_communities_modularity(5, [])) == 5)

    # ------------------------------------------------------------------ [3]
    print("[3] conformity separates planted followers from planted leaders")
    # Leaders 0..3 adopt first; followers 4..11 adopt right after an
    # in-neighbour; independents 12..15 adopt at random times with no arc in.
    leaders = [0, 1, 2, 3]
    followers = [4, 5, 6, 7, 8, 9, 10, 11]
    independents = [12, 13, 14, 15]
    n_u2 = 16
    e2 = []
    for f in followers:
        for L in leaders:
            e2.append((L, f))
    for L in leaders:            # leaders also follow each other structurally,
        for L2 in leaders:       # but never adopt after one another in time
            if L != L2:
                e2.append((L, L2))
    for f in followers:          # followers point back at the leaders: plenty of
        for L in leaders:        # opportunity to influence, never any success,
            e2.append((f, L))    # since leaders always adopt first
    e2 = sorted(set(e2))
    out2, in2 = build_adjacency(n_u2, e2)

    logs2 = []
    day = 86400
    for i in range(40):
        base = i * 100 * day
        for L in leaders:
            logs2.append((L, i, base))
        for j, f in enumerate(followers):
            logs2.append((f, i, base + (j + 1) * day))  # within delta, after
        for k, x in enumerate(independents):
            logs2.append((x, i, base + 3 * day))
    logs2.sort(key=lambda r: (r[2], r[0], r[1]))
    ds2 = Dataset(name="conf", n_users=n_u2, n_items=40, n_attrs=1, edges=e2,
                  out_adj=out2, in_adj=in2, logs=logs2,
                  item_attrs=[[0] for _ in range(40)])

    prof = estimate_conformity(ds2, delta=30 * day)
    conf_followers = sum(prof.conformity[v] for v in followers) / len(followers)
    conf_leaders = sum(prof.conformity[v] for v in leaders) / len(leaders)
    conf_indep = sum(prof.conformity[v] for v in independents) / len(independents)
    check("followers are more conformist than leaders",
          conf_followers > conf_leaders + 0.5,
          "followers=%.3f, leaders=%.3f" % (conf_followers, conf_leaders))
    check("independents (no in-arcs) have low conformity",
          conf_indep < 0.2, "independents=%.3f" % conf_indep)
    infl_leaders = sum(prof.influence_index[v] for v in leaders) / len(leaders)
    infl_followers = sum(prof.influence_index[v] for v in followers) / len(followers)
    # Followers here DO have out-arcs (back to the leaders), so they have ample
    # opportunity to influence and demonstrably never take it -- their low
    # influence index is real evidence, not a fallback to the population mean.
    check("leaders have a much higher influence index than followers",
          infl_leaders > infl_followers + 0.5,
          "leaders=%.4f, followers=%.4f" % (infl_leaders, infl_followers))
    check("followers had real opportunities to influence and took none",
          all(prof.n_opp[f] > 0 and prof.n_succ[f] == 0 for f in followers),
          "n_opp=%d n_succ=%d" % (prof.n_opp[followers[0]],
                                  prof.n_succ[followers[0]]))
    check("both indices lie in [0,1]",
          all(0.0 <= x <= 1.0 for x in prof.conformity)
          and all(0.0 <= x <= 1.0 for x in prof.influence_index))
    check("conformity summary is JSON-shaped",
          set(["conformity_mean", "global_follow_rate"]) <= set(prof.summary()))

    # Delta must bite: a window shorter than the follower lag sees no following.
    prof_tight = estimate_conformity(ds2, delta=1)
    check("a delta below every lag drives conformity to ~0",
          max(prof_tight.conformity) < 0.05,
          "max=%.4f" % max(prof_tight.conformity))

    # Cross-check the counts against a naive O(adopters^2) reference, on a
    # random dataset AND on one with a hub whose out-degree forces the flipped
    # inner loop (n_adopters * _FLIP_FACTOR < outdeg) -- both branches must
    # produce identical counts.
    def _naive_counts(dsx, delta_x, logs_x):
        eset = set(dsx.edges)
        byi = {}
        for (u, i, t) in logs_x:
            b = byi.setdefault(i, {})
            if u not in b or t < b[u]:
                b[u] = t
        nf = [0] * dsx.n_users
        na = [0] * dsx.n_users
        ns = [0] * dsx.n_users
        for i, times in byi.items():
            for v in times:
                na[v] += 1
            fol = set()
            for u in times:
                for v in times:
                    dt = times[v] - times[u]
                    if 0 < dt <= delta_x and (u, v) in eset:
                        ns[u] += 1
                        fol.add(v)
            for v in fol:
                nf[v] += 1
        return nf, na, ns

    for label, hub_deg in (("normal out-degrees", 0), ("hub forcing the flip", 400)):
        nr = random.Random(88)
        n_u3 = 30 + hub_deg
        e3 = set()
        while len(e3) < 120:
            a, b = nr.randrange(30), nr.randrange(30)
            if a != b:
                e3.add((a, b))
        for x in range(30, 30 + hub_deg):  # user 0 becomes a huge hub
            e3.add((0, x))
        e3 = sorted(e3)
        o3, i3 = build_adjacency(n_u3, e3)
        lg3 = []
        for i in range(12):
            for v in range(30):
                if nr.random() < 0.5:
                    lg3.append((v, i, i * 1000 * day + nr.randrange(0, 20) * day))
        lg3.sort(key=lambda r: (r[2], r[0], r[1]))
        ds3 = Dataset(name="x", n_users=n_u3, n_items=12, n_attrs=1, edges=e3,
                      out_adj=o3, in_adj=i3, logs=lg3,
                      item_attrs=[[0] for _ in range(12)])
        p3 = estimate_conformity(ds3, delta=10 * day)
        nf, na, ns = _naive_counts(ds3, 10 * day, lg3)
        check("counts match the naive reference (%s)" % label,
              p3.n_follow == nf and p3.n_adopt == na and p3.n_succ == ns)

    # ------------------------------------------------------------------ [4]
    print("[4] conformity-aware reweighting")
    base_pp = {(u, v): 0.1 for (u, v) in e2}
    pp_c, stats = conformity_aware_pp(ds2, base_pp, prof)
    check("all reweighted probabilities are in [0,1]",
          all(0.0 <= p <= 1.0 for p in pp_c.values()))
    check("mean edge probability is preserved",
          abs(stats["mean_new"] - stats["mean_base"]) < 1e-9,
          "%.6f vs %.6f" % (stats["mean_new"], stats["mean_base"]))
    check("every arc of G is present", len(pp_c) == len(set(e2)))
    # Arcs into conformists must beat arcs into non-conformists from the same source.
    into_follower = pp_c[(0, followers[0])]
    into_leader = pp_c[(0, 1)]
    check("an arc into a conformist outweighs one into a leader",
          into_follower > into_leader,
          "%.5f vs %.5f" % (into_follower, into_leader))
    # Degenerate profile (no logs) must fall back rather than zero the graph.
    ds_nolog = Dataset(name="nl", n_users=n_u2, n_items=1, n_attrs=1, edges=e2,
                       out_adj=out2, in_adj=in2, logs=[], item_attrs=[[]])
    prof0 = estimate_conformity(ds_nolog)
    pp0, st0 = conformity_aware_pp(ds_nolog, base_pp, prof0)
    check("a log-free dataset falls back to the base probabilities",
          st0["degenerate"] and all(abs(p - 0.1) < 1e-12 for p in pp0.values()))

    # ------------------------------------------------------------------ [5]
    print("[5] Monte-Carlo IC spread")
    # Chain 0->1->2 with p=0.5: E|active| = 1 + 0.5 + 0.25 = 1.75
    chain = {0: [(1, 0.5)], 1: [(2, 0.5)]}
    est = mc_spread(chain, [0], 40000, 11)
    check("MC spread on a p=0.5 chain ~= 1.75",
          abs(est - 1.75) < 0.02, "got %.4f" % est)
    check("MC spread is deterministic for a fixed seed",
          mc_spread(chain, [0], 500, 3) == mc_spread(chain, [0], 500, 3))
    check("different seeds give different draws",
          mc_spread(chain, [0], 200, 3) != mc_spread(chain, [0], 200, 4))
    check("an isolated seed has spread 1", mc_spread({}, [7], 100, 1) == 1.0)
    check("an empty seed set has spread 0", mc_spread(chain, [], 100, 1) == 0.0)

    # ------------------------------------------------------------------ [6]
    print("[6] CELF greedy == plain greedy under common random numbers")
    n6 = 24
    e6 = set()
    r6 = random.Random(5)
    while len(e6) < 70:
        a, b = r6.randrange(n6), r6.randrange(n6)
        if a != b:
            e6.add((a, b))
    sub6 = defaultdict(list)
    for (a, b) in sorted(e6):
        sub6[a].append((b, 0.15 + 0.45 * random.Random(a * 100 + b).random()))
    sub6 = dict(sub6)
    nodes6 = list(range(n6))
    s_celf, c_celf = celf_greedy_community(nodes6, sub6, 5, 400, 99, use_celf=True)
    s_plain, c_plain = celf_greedy_community(nodes6, sub6, 5, 400, 99, use_celf=False)
    check("CELF and plain greedy pick the same seeds",
          s_celf == s_plain, "%s vs %s" % (s_celf, s_plain))
    check("CELF and plain greedy report the same curve",
          all(abs(a - b) < 1e-9 for a, b in zip(c_celf, c_plain)))
    check("curve is non-decreasing and starts at 0",
          c_celf[0] == 0.0 and all(c_celf[i] <= c_celf[i + 1] + 1e-12
                                   for i in range(len(c_celf) - 1)),
          "%s" % ["%.3f" % x for x in c_celf])
    check("curve length matches the seed count", len(c_celf) == len(s_celf) + 1)
    check("K larger than the community is capped",
          len(celf_greedy_community([1, 2], {}, 9, 20, 1)[0]) == 2)

    # ------------------------------------------------------------------ [7]
    print("[7] DP allocation is optimal (vs brute force)")
    alloc, val = dp_allocate([[0.0, 3.0, 4.0], [0.0, 2.0, 5.0]], 2)
    check("hand-checked DP: curves [0,3,4] & [0,2,5], K=2 -> 5.0",
          abs(val - 5.0) < 1e-12 and sum(alloc) == 2,
          "alloc=%s val=%.3f" % (alloc, val))

    dr = random.Random(31)
    worst = 0.0
    for trial in range(40):
        n_c = dr.randrange(1, 5)
        Kt = dr.randrange(1, 7)
        cs = []
        for _ in range(n_c):
            L = dr.randrange(0, Kt + 1)
            cur, row = 0.0, [0.0]
            for _ in range(L):
                cur += dr.random() * 3.0
                row.append(cur)
            cs.append(row)
        got_alloc, got = dp_allocate(cs, Kt)
        # brute force over all allocations
        ranges = [range(0, len(c)) for c in cs]
        best = 0.0
        for combo in itertools.product(*ranges):
            if sum(combo) > Kt:
                continue
            v = sum(cs[m][combo[m]] for m in range(n_c))
            if v > best:
                best = v
        worst = max(worst, abs(best - got))
        if sum(got_alloc) > Kt:
            worst = 1e9
    check("DP matches brute force on 40 random instances",
          worst < 1e-9, "max abs diff = %.3e" % worst)
    check("DP with K=0 allocates nothing", dp_allocate([[0.0, 1.0]], 0) == ([0], 0.0))
    check("DP with no communities is a no-op", dp_allocate([], 5) == ([], 0.0))

    # ------------------------------------------------------------------ [8]
    print("[8] end-to-end cinema_select_seeds")
    # Reuse the 4-block graph, add adoption logs with a planted follow pattern.
    out1, in1 = build_adjacency(n_users, edges)
    logs1 = []
    lr = random.Random(17)
    n_items1 = 30
    for i in range(n_items1):
        base = i * 60 * day
        starters = [b * n_block for b in range(n_blocks)]  # one seed per block
        for s in starters:
            logs1.append((s, i, base))
        for v in range(n_users):
            if v in starters:
                continue
            if lr.random() < 0.35:
                logs1.append((v, i, base + lr.randrange(1, 10) * day))
    logs1.sort(key=lambda r: (r[2], r[0], r[1]))
    ds1 = Dataset(name="blocks", n_users=n_users, n_items=n_items1, n_attrs=2,
                  edges=edges, out_adj=out1, in_adj=in1, logs=logs1,
                  item_attrs=[[0] for _ in range(n_items1)])
    base_pp1 = {(u, v): 0.08 for (u, v) in edges}

    K = 6
    res = cinema_select_seeds(ds1, base_pp1, K, random.Random(3),
                              n_mc=60, C=4, use_celf=True)
    check("returns a RunResult with the four API.md fields",
          all(hasattr(res, f) for f in ("seeds", "spread", "seconds", "extra")))
    check("selects exactly K seeds", len(res.seeds) == K, "%s" % (res.seeds,))
    check("seeds are distinct valid node ids",
          len(set(res.seeds)) == K
          and all(0 <= s < n_users for s in res.seeds))
    check("spread is positive and at least K", res.spread >= K - 1e-9,
          "spread=%.3f" % res.spread)
    check("seconds is recorded", res.seconds > 0.0, "%.3fs" % res.seconds)
    check("extra reports CINEMA as community-based and NOT topic-aware",
          res.extra["community_based"] is True
          and res.extra["topic_aware"] is False)
    check("extra exposes the number of communities found",
          res.extra["n_communities"] == 4, "%d" % res.extra["n_communities"])
    check("extra reports the Monte-Carlo influence model",
          res.extra["influence_model"] == "Monte-Carlo Independent Cascade")
    check("extra breaks the wall clock down per stage",
          set(["community_detection", "conformity", "community_greedy",
               "dp_allocation"]) <= set(res.extra["timings"]))
    check("the DP allocation sums to K",
          sum(res.extra["allocation"].values()) == K,
          "%s" % res.extra["allocation"])
    check("approximation flag is off by default",
          res.extra["approximate"] is False)

    # Determinism.
    res_b = cinema_select_seeds(ds1, base_pp1, K, random.Random(3),
                                n_mc=60, C=4, use_celf=True)
    check("identical seeds for an identical rng seed",
          res.seeds == res_b.seeds and abs(res.spread - res_b.spread) < 1e-12)
    res_c = cinema_select_seeds(ds1, base_pp1, K, random.Random(4),
                                n_mc=60, C=4, use_celf=True)
    check("the rng actually drives the Monte Carlo",
          res_c.extra["dp_estimate"] != res.extra["dp_estimate"])

    # Monotonicity in K.
    prev_spread = 0.0
    ok_mono = True
    for k in (1, 3, 6):
        rk = cinema_select_seeds(ds1, base_pp1, k, random.Random(3),
                                 n_mc=40, C=4)
        if rk.spread < prev_spread - 1e-9:
            ok_mono = False
        prev_spread = rk.spread
    check("spread is non-decreasing in K", ok_mono)

    # A supplied partition must be honoured and must skip detection.
    fixed = [v // n_block for v in range(n_users)]
    res_f = cinema_select_seeds(ds1, base_pp1, K, random.Random(3),
                                n_mc=40, C=4, comm=fixed)
    check("a supplied partition is used as-is",
          res_f.extra["communities_detected_here"] is False
          and res_f.extra["n_communities"] == 4)
    check("all seeds live in the supplied communities",
          all(0 <= fixed[s] < 4 for s in res_f.seeds))

    # cand_top must flag itself as an approximation.
    res_a = cinema_select_seeds(ds1, base_pp1, K, random.Random(3),
                                n_mc=40, C=4, cand_top=8)
    check("cand_top flags the run as approximate",
          res_a.extra["approximate"] is True and len(res_a.seeds) == K)

    # select_seeds alias.
    res_s = select_seeds(ds1, base_pp1, 2, random.Random(3), n_mc=40, C=4)
    check("select_seeds alias works", len(res_s.seeds) == 2)

    # pp=None -> uniform fallback.
    res_u = cinema_select_seeds(ds1, None, 2, random.Random(3), n_mc=40, C=4)
    check("pp=None falls back to a uniform probability",
          len(res_u.seeds) == 2
          and abs(res_u.extra["pp_stats"]["mean_base"] - DEFAULT_UNIFORM_P) < 1e-12)

    # ------------------------------------------------------------------ [9]
    print("[9] CINEMA beats a degree-blind random choice (sanity, not a claim)")
    mia = MIA(ds1.n_users, ds1.out_adj, ds1.in_adj,
              conformity_aware_pp(ds1, base_pp1,
                                  estimate_conformity(ds1))[0], h=0.1)
    rr = random.Random(21)
    rand_spreads = []
    for _ in range(8):
        S = rr.sample(range(n_users), K)
        rand_spreads.append(mia.influence(S))
    mean_rand = sum(rand_spreads) / len(rand_spreads)
    check("CINEMA's seeds beat the mean of 8 random seed sets",
          res.spread > mean_rand,
          "cinema=%.3f vs random=%.3f" % (res.spread, mean_rand))

    # ------------------------------------------------------------------ done
    print("")
    if failures:
        print("FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("ALL CHECKS PASSED (ctim/baselines/cinema.py)")


if __name__ == "__main__":
    _self_test()
