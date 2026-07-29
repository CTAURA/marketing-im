"""Influence strength inference and the MIA influence computation model.

Implements Section 4.2 of

    Huimin Huang, Hong Shen, Zaiqiao Meng, Huajian Chang, Huaiwen He.
    "Community-based influence maximization for viral marketing",
    Applied Intelligence (2019). DOI 10.1007/s10489-018-1387-8

Covered equations:

    Eq (10)  P(c | z, c')  = eta[c'][c] * theta[c][z]
    Eq (11)  P(v | z, u)   = sum_{c,c'} pi[v][c] * pi[u][c'] * P(c | z, c')
    Eq (12)  P(v | i, u)   = sum_z P(z|i) * P(v | z, u)
    Eq (13)  pp(P)         = prod_k pp(w_k, w_{k+1})
    Eq (14)  MIP(u,v)      = argmax_P pp(P)                (Dijkstra on -ln pp)
    Eq (15)  MIIA(v,h)     = union of MIP(u,v) with pp >= h
    Eq (16)  MIOA(v,h)     = union of MIP(v,u) with pp >= h
    Eq (17)  ap(v|S)       activation probability on MIIA(v,h)
    Eq (18)  I(S)          = sum_v ap(v|S)

Standard library only.  Python 3.9 compatible.  No module-level randomness:
the only stochastic code here is the test-only Monte-Carlo IC simulator, which
takes an explicit `random.Random` instance.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

__all__ = [
    "community_to_community",
    "user_to_user_topic",
    "user_to_user_item",
    "EdgeWeights",
    "MIA",
]


# ---------------------------------------------------------------------------
# Section 4.2.1 -- influence strength inference, Eq (10)-(12)
# ---------------------------------------------------------------------------


def community_to_community(model):
    """Eq (10). Returns P[c'][c][z] = eta[c'][c] * theta[c][z].

    `model` is a `ctim.gibbs.Model` (or anything exposing `.eta` and `.theta`).
    Indexing follows the paper's conditional P(c | z, c'): the *source*
    community c' is the outermost index, then the target community c, then the
    topic z.
    """
    eta = model.eta
    theta = model.theta
    C = len(eta)
    Z = len(theta[0]) if theta else 0
    out = []
    for cp in range(C):
        eta_cp = eta[cp]
        rows = []
        for c in range(C):
            e = eta_cp[c]
            theta_c = theta[c]
            # Eq (10)
            rows.append([e * theta_c[z] for z in range(Z)])
        out.append(rows)
    return out


def user_to_user_topic(model, u, v, z):
    """Eq (11), naive reference implementation. O(C^2). Used by tests.

    P(v | z, u) = sum_{c,c'} pi[v][c] * pi[u][c'] * P(c | z, c')
    with P(c | z, c') = eta[c'][c] * theta[c][z]  (Eq (10)) inlined.
    """
    pi = model.pi
    eta = model.eta
    theta = model.theta
    pi_u = pi[u]
    pi_v = pi[v]
    C = len(eta)
    total = 0.0
    for cp in range(C):
        p_u = pi_u[cp]
        if p_u == 0.0:
            continue
        eta_cp = eta[cp]
        for c in range(C):
            # Eq (11), with Eq (10) inlined as eta[cp][c] * theta[c][z]
            total += pi_v[c] * p_u * (eta_cp[c] * theta[c][z])
    return total


def user_to_user_item(model, u, v, i):
    """Eq (12), naive reference implementation. Used by tests.

    P(v | i, u) = sum_{z=1..Z} P(z|i) * P(v | z, u)
    """
    p_z_i = model.p_z_given_i[i]
    total = 0.0
    for z in range(len(p_z_i)):
        pz = p_z_i[z]
        if pz == 0.0:
            continue
        # Eq (12)
        total += pz * user_to_user_topic(model, u, v, z)
    return total


class EdgeWeights:
    """Fast *exact* evaluation of Eq (11)+(12) for all edges of G.

    The naive form of Eq (11)+(12) costs O(Z * C^2) per edge per item.  The
    following regrouping is algebraically identical (SPEC.md Section 3):

        P(v|i,u) = sum_z P(z|i) sum_{c,c'} pi[u][c'] pi[v][c] eta[c'][c] theta[c][z]
                 = sum_c ( sum_{c'} pi[u][c'] eta[c'][c] ) * pi[v][c]
                       * ( sum_z P(z|i) theta[c][z] )
                 = sum_c a_u[c] * pi[v][c] * thetabar_i[c]

    with

        a_u[c]        = sum_{c'} pi[u][c'] * eta[c'][c]      (item independent)
        thetabar_i[c] = sum_z P(z|i) * theta[c][z]           (user independent)

    `a_u` is cached across items (computed at most once per user, O(C^2)) and
    `thetabar_i` is cached per item (O(C*Z)), leaving O(C) per edge per item.
    This must reproduce `user_to_user_item()` up to floating point; the
    self-test asserts agreement to 1e-9 relative on every edge.
    """

    def __init__(self, model, ds, top_c: int = 0):
        self.model = model
        self.ds = ds
        self.top_c = int(top_c)
        self.approximate = self.top_c > 0
        self.eta = model.eta
        self.theta = model.theta
        self.p_z_given_i = model.p_z_given_i
        self.C = len(self.eta)
        self.Z = len(self.theta[0]) if self.theta else 0

        # Number of times a computed pp had to be clamped into [0, 1], and the
        # largest raw (pre-clamp) value ever seen.  Both are public so that a
        # caller can see whether the learned parameters violated the model's
        # invariants -- see the comment in `for_item`.  Cumulative across calls.
        self.n_clamped = 0
        self.max_raw_pp = 0.0
        self.min_raw_pp = 0.0

        pi = model.pi
        if self.approximate:
            # Approximation (OFF by default): keep only the `top_c` largest
            # entries of each pi row, zeroing the rest.  Rows are NOT
            # renormalised -- this is pure truncation, so every pp can only
            # shrink, never grow.  `self.approximate` flags the result.
            trunc = []
            for row in pi:
                if self.top_c >= len(row):
                    trunc.append(list(row))
                    continue
                keep = sorted(range(len(row)), key=lambda c: (-row[c], c))[: self.top_c]
                keep_set = set(keep)
                trunc.append([row[c] if c in keep_set else 0.0 for c in range(len(row))])
            self.pi = trunc
        else:
            self.pi = pi

        # Sparse view of pi used to skip structural zeros when forming a_u.
        self._pi_nz = [
            [(c, val) for c, val in enumerate(row) if val != 0.0] for row in self.pi
        ]

        self._a_cache = {}
        self._thetabar_cache = {}

    # -- cached factors -----------------------------------------------------

    def a_u(self, u):
        """a_u[c] = sum_{c'} pi[u][c'] * eta[c'][c]  (the item-independent half
        of Eq (11)).  Computed once per user and cached across items."""
        a = self._a_cache.get(u)
        if a is not None:
            return a
        C = self.C
        eta = self.eta
        a = [0.0] * C
        for cp, p_u in self._pi_nz[u]:  # Eq (11), source-community factor
            eta_cp = eta[cp]
            for c in range(C):
                a[c] += p_u * eta_cp[c]
        self._a_cache[u] = a
        return a

    def thetabar(self, i):
        """thetabar_i[c] = sum_z P(z|i) * theta[c][z]  (the item-dependent half
        of Eq (12)).  Cached per item."""
        tb = self._thetabar_cache.get(i)
        if tb is not None:
            return tb
        theta = self.theta
        p_z_i = self.p_z_given_i[i]
        nz = [(z, p) for z, p in enumerate(p_z_i) if p != 0.0]
        tb = []
        for c in range(self.C):
            theta_c = theta[c]
            s = 0.0
            for z, p in nz:  # Eq (12)
                s += p * theta_c[z]
            tb.append(s)
        self._thetabar_cache[i] = tb
        return tb

    # -- public API ---------------------------------------------------------

    def pp(self, u, v, i):
        """P(v|i,u) for a single directed pair, clamped into [0,1]."""
        return self._pp(u, v, self.thetabar(i))

    def _pp(self, u, v, tb):
        a = self.a_u(u)
        pi_v = self.pi[v]
        s = 0.0
        for c in range(self.C):
            pv = pi_v[c]
            if pv != 0.0:
                # Eq (11)+(12) factorised: sum_c a_u[c] * pi_v[c] * thetabar_i[c]
                s += a[c] * pv * tb[c]
        # Clamp into [0,1].  With a well-formed model this is a no-op: rows of
        # pi and theta are probability distributions and eta[c'][c] in [0,1],
        # hence a_u[c] <= 1 and thetabar_i[c] <= 1, so the sum is bounded by
        # sum_c pi_v[c] = 1.  But the value is a *learned* estimate that may be
        # read from disk, hand-supplied, or produced by a variant sampler whose
        # rows do not sum to exactly 1 (or that emits eta > 1); a pp outside
        # [0,1] would silently corrupt -ln(pp) in Eq (13)/(14) and make Eq (17)
        # non-probabilistic.  So we clamp defensively and count every clamp in
        # the public `n_clamped` / `max_raw_pp` / `min_raw_pp` attributes rather
        # than hiding the violation.
        if s > 1.0:
            if s > self.max_raw_pp:
                self.max_raw_pp = s
            self.n_clamped += 1
            return 1.0
        if s < 0.0:
            if s < self.min_raw_pp:
                self.min_raw_pp = s
            self.n_clamped += 1
            return 0.0
        if s > self.max_raw_pp:
            self.max_raw_pp = s
        return s

    def for_item(self, i: int) -> dict:
        """Returns {(u,v): pp} for every directed edge of G, for item i.

        Eq (12), evaluated through the exact factorisation above.
        """
        tb = self.thetabar(i)
        out = {}
        for (u, v) in self.ds.edges:
            out[(u, v)] = self._pp(u, v, tb)
        return out


# ---------------------------------------------------------------------------
# Section 4.2.2 -- MIA influence computation model, Eq (13)-(18)
# ---------------------------------------------------------------------------


class MIA:
    """Eq (13)-(18), Chen et al. [19] maximum influence arborescence model.

    Construction on a node-induced subgraph is supported through the optional
    `nodes` argument: only nodes in that subset and only edges with *both*
    endpoints inside it are considered.  Algorithm 2 line 34 uses this to
    compute the within-community spread I_m on community m's induced subgraph.
    """

    def __init__(self, n_users, out_adj, in_adj, pp: dict, h: float = 0.1,
                 nodes=None):
        self.n_users = n_users
        self.h = h
        self.pp = pp

        if nodes is None:
            self.nodes = list(range(n_users))
            self._allowed = None  # None == "everything", avoids a set lookup
        else:
            self.nodes = sorted(set(nodes))
            self._allowed = set(self.nodes)

        allowed = self._allowed

        def _ok(x):
            return allowed is None or x in allowed

        # Restrict the adjacency to the induced subgraph, drop edges with no
        # (or zero) propagation probability -- they can never carry influence.
        self._out = {}
        self._in = {}
        for u in self.nodes:
            outs = []
            for v in (out_adj[u] if u < len(out_adj) else ()):
                if _ok(v):
                    w = pp.get((u, v), 0.0)
                    if w > 0.0:
                        outs.append((v, w))
            outs.sort()
            self._out[u] = outs
        for v in self.nodes:
            ins = []
            for u in (in_adj[v] if v < len(in_adj) else ()):
                if _ok(u):
                    w = pp.get((u, v), 0.0)
                    if w > 0.0:
                        ins.append((u, w))
            ins.sort()
            self._in[v] = ins

        # Eq (14)/(15)/(16): a path survives iff pp(P) >= h, i.e. iff its
        # additive -ln pp length is <= -ln h.
        if h is None or h <= 0.0:
            self._cutoff = float("inf")
        else:
            self._cutoff = -math.log(h)

        self._miia_cache = {}
        self._mioa_cache = {}

    # -- Eq (13)/(14): maximum influence paths via Dijkstra on -ln pp -------

    def _dijkstra(self, root, forward):
        """Eq (13)+(14). Single-source shortest paths under the additive weight
        -ln pp(u,v); since Eq (13) makes pp(P) a *product* of edge
        probabilities, minimising sum -ln pp maximises pp(P), so the shortest
        path tree is exactly the tree of maximum influence paths MIP.

        forward=True  -> follows out-edges  (paths root -> x), used by MIOA.
        forward=False -> follows in-edges   (paths x -> root), used by MIIA.

        Returns (dist, parent) where parent[x] is x's neighbour one hop closer
        to `root` along MIP, in both directions.
        """
        dist = {root: 0.0}
        parent = {}
        cutoff = self._cutoff
        adj = self._out if forward else self._in
        # (distance, node) -- node in the tuple makes ties deterministic.
        heap = [(0.0, root)]
        done = set()
        while heap:
            d, x = heapq.heappop(heap)
            if x in done:
                continue
            done.add(x)
            if d > cutoff:
                continue
            for (y, w) in adj[x]:
                if y in done:
                    continue
                # Eq (13): multiplying probabilities == adding -ln probabilities
                nd = d - math.log(w) if w < 1.0 else d
                if nd > cutoff:
                    continue
                old = dist.get(y)
                if old is None or nd < old or (nd == old and x < parent.get(y, x)):
                    dist[y] = nd
                    parent[y] = x
                    heapq.heappush(heap, (nd, y))
        # Nodes popped beyond the cutoff were never expanded; drop them so the
        # returned arborescence contains exactly the surviving MIPs.
        keep = {x: dv for x, dv in dist.items() if dv <= cutoff}
        parent = {x: p for x, p in parent.items() if x in keep and p in keep}
        return keep, parent

    def _miia_full(self, v):
        """(nodes, parent, children) of MIIA(v,h); cached."""
        got = self._miia_cache.get(v)
        if got is not None:
            return got
        dist, parent = self._dijkstra(v, forward=False)  # Eq (15)
        children = defaultdict(list)
        for x, p in parent.items():
            children[p].append(x)
        for p in children:
            children[p].sort()  # deterministic product order in Eq (17)
        got = (sorted(dist.keys()), parent, dict(children))
        self._miia_cache[v] = got
        return got

    def miia(self, v) -> tuple:
        """Eq (15). Returns (nodes, parent_edges) of MIIA(v,h).

        `nodes` is the sorted list of u with pp(MIP(u,v)) >= h (v included).
        `parent_edges` maps x -> the next hop on MIP(x,v); the arborescence
        edge is therefore the directed graph edge (x, parent_edges[x]) and all
        paths lead into the root v.
        """
        nodes, parent, _children = self._miia_full(v)
        return nodes, parent

    def mioa(self, v) -> tuple:
        """Eq (16). Returns (nodes, parent_edges) of MIOA(v,h).

        `nodes` is the sorted list of u with pp(MIP(v,u)) >= h (v included).
        `parent_edges` maps x -> its predecessor on MIP(v,x); the arborescence
        edge is the directed graph edge (parent_edges[x], x) and all paths lead
        out of the root v.
        """
        got = self._mioa_cache.get(v)
        if got is not None:
            return got
        dist, parent = self._dijkstra(v, forward=True)  # Eq (16)
        got = (sorted(dist.keys()), parent)
        self._mioa_cache[v] = got
        return got

    # -- Eq (17): activation probability ------------------------------------

    def _ap_on_tree(self, v, S):
        """Eq (17), evaluated bottom-up over MIIA(v,h).

        ap(x|S) = 1                                          if x in S
                = 0                                          if N_in(x) = empty
                = 1 - prod_{w in N_in(x)} (1 - ap(w|S) pp(w,x))  otherwise

        N_in(x) is x's in-neighbourhood *within MIIA(v,h)*, i.e. the children of
        x in the arborescence rooted at v.  The slice is a tree, so a post-order
        traversal evaluates every node after its children; we use an explicit
        stack instead of recursion to stay safe on deep arborescences.
        """
        if v in S:
            return 1.0  # Eq (17), case v in S
        nodes, _parent, children = self._miia_full(v)
        pp = self.pp
        apval = {}
        stack = [(v, False)]
        while stack:
            x, expanded = stack.pop()
            if not expanded:
                if x in S:
                    apval[x] = 1.0  # Eq (17), case x in S -- no need to descend
                    continue
                ch = children.get(x)
                if not ch:
                    apval[x] = 0.0  # Eq (17), case N_in(x) = empty
                    continue
                stack.append((x, True))
                for w in ch:
                    stack.append((w, False))
            else:
                prod = 1.0
                for w in children[x]:
                    # Eq (17): edge w -> x inside MIIA(v,h)
                    prod *= 1.0 - apval[w] * pp[(w, x)]
                apval[x] = 1.0 - prod
        return apval[v]

    def ap(self, v, S) -> float:
        """Eq (17). Activation probability of v given seed set S."""
        if self._allowed is not None and v not in self._allowed:
            return 0.0
        if not isinstance(S, (set, frozenset)):
            S = set(S)
        return self._ap_on_tree(v, S)

    # -- Eq (18): influence spread ------------------------------------------

    def influence(self, S) -> float:
        """Eq (18). I(S) = sum_{v in V} ap(v|S).

        ap(v|S) can only be non-zero if some u in S lies in MIIA(v,h), which
        (both being the condition pp(MIP(u,v)) >= h) holds iff v lies in
        MIOA(u,h).  Summing over that union is therefore exactly the sum over
        all of V, just without visiting the nodes S cannot reach.
        """
        if not isinstance(S, (set, frozenset)):
            S = set(S)
        if not S:
            return 0.0
        affected = set()
        for u in S:
            if self._allowed is not None and u not in self._allowed:
                continue
            affected.update(self.mioa(u)[0])
        total = 0.0
        for v in sorted(affected):
            total += self._ap_on_tree(v, S)  # Eq (18)
        return total

    def marginal_gain(self, S, u) -> float:
        """I(S u {u}) - I(S), exact via Eq (17)/(18).

        Only nodes in MIOA(u,h) can have their ap changed by adding u, so the
        recomputation is restricted to those MIIAs.
        """
        if not isinstance(S, (set, frozenset)):
            S = set(S)
        if u in S:
            return 0.0
        if self._allowed is not None and u not in self._allowed:
            return 0.0
        S2 = set(S)
        S2.add(u)
        gain = 0.0
        for v in self.mioa(u)[0]:  # Eq (18) restricted to the affected nodes
            gain += self._ap_on_tree(v, S2) - self._ap_on_tree(v, S)
        return gain

    # -- greedy seed selection ----------------------------------------------

    def greedy_incremental(self, K, candidates=None) -> list:
        """Standard MIA greedy (Chen et al. [19]) with incremental IncInf
        updates via MIOA, and a max-heap with lazy re-evaluation.

        IncInf[u] = I(S u {u}) - I(S) is maintained exactly for every candidate
        u.  Adding a seed s can only change ap(v|S) for v in MIOA(s,h), so only
        the IncInf contributions coming from those v are subtracted (at the old
        S) and re-added (at the new S); every other candidate's IncInf is
        untouched.  The heap holds (-IncInf[u], u) entries that may go stale; a
        popped entry is re-pushed with its current value instead of being
        trusted, which is the usual lazy-evaluation trick.

        `candidates` restricts which nodes may be selected (Algorithm 2 line 43
        selects only within community c_j) but never restricts which nodes are
        counted in the spread.
        """
        if candidates is None:
            cand = list(self.nodes)
        else:
            if self._allowed is None:
                cand = sorted(set(candidates))
            else:
                cand = sorted(set(candidates) & self._allowed)
        if K <= 0 or not cand:
            return []
        cand_set = set(cand)

        # IncInf[u] for S = empty: ap(v|{u}) - ap(v|empty) = ap(v|{u}).
        inc = dict.fromkeys(cand, 0.0)
        for v in self.nodes:
            nodes_v, _parent, _children = self._miia_full(v)
            touched = [w for w in nodes_v if w in cand_set]
            if not touched:
                continue
            for w in touched:
                inc[w] += self._ap_on_tree(v, {w})

        heap = [(-inc[u], u) for u in cand]
        heapq.heapify(heap)

        S = []
        S_set = set()
        while len(S) < K and heap:
            neg, u = heapq.heappop(heap)
            if u in S_set:
                continue
            cur = inc[u]
            if -neg < cur - 1e-12 or -neg > cur + 1e-12:
                heapq.heappush(heap, (-cur, u))  # lazy re-evaluation
                continue
            if cur <= 0.0 and S:
                break  # no candidate can add anything

            affected = self.mioa(u)[0]

            # Remove the contributions of the affected MIIAs at the current S.
            for v in affected:
                nodes_v, _p, _c = self._miia_full(v)
                base = self._ap_on_tree(v, S_set)
                for w in nodes_v:
                    if w in cand_set and w not in S_set and w != u:
                        S_set.add(w)
                        inc[w] -= self._ap_on_tree(v, S_set) - base
                        S_set.discard(w)

            S.append(u)
            S_set.add(u)

            # Re-add them at the new S.
            dirty = set()
            for v in affected:
                nodes_v, _p, _c = self._miia_full(v)
                base = self._ap_on_tree(v, S_set)
                for w in nodes_v:
                    if w in cand_set and w not in S_set:
                        S_set.add(w)
                        inc[w] += self._ap_on_tree(v, S_set) - base
                        S_set.discard(w)
                        dirty.add(w)

            for w in sorted(dirty):
                heapq.heappush(heap, (-inc[w], w))

        return S


# ---------------------------------------------------------------------------
# Test-only helper: Monte-Carlo Independent Cascade simulation.
# Not part of the public API -- MIA.influence() is exact per Eq (17)/(18) and
# never uses Monte Carlo.  This exists purely so the self-test can check the
# MIA arborescence approximation against the diffusion process it models.
# ---------------------------------------------------------------------------


def _monte_carlo_ic(out_adj, pp, S, n_sims, rng):
    """Mean number of activated nodes under the Independent Cascade model.

    `rng` must be an explicit random.Random instance (determinism rule).
    """
    seeds = list(S)
    total = 0
    random_fn = rng.random
    for _ in range(n_sims):
        active = set(seeds)
        frontier = list(seeds)
        while frontier:
            nxt = []
            for u in frontier:
                for v in out_adj[u]:
                    if v in active:
                        continue
                    p = pp.get((u, v), 0.0)
                    if p > 0.0 and random_fn() < p:
                        active.add(v)
                        nxt.append(v)
            frontier = nxt
        total += len(active)
    return total / float(n_sims)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import sys
    import time

    class _M(object):
        """Minimal stand-in for ctim.gibbs.Model (same attribute names)."""

        def __init__(self, pi, eta, theta, p_z_given_i):
            self.pi = pi
            self.eta = eta
            self.theta = theta
            self.p_z_given_i = p_z_given_i
            self.phi = p_z_given_i
            self.psi = None
            self.hyper = None

    class _DS(object):
        def __init__(self, edges):
            self.edges = edges

    def _dirichlet_row(rng, n):
        vals = [rng.random() + 1e-3 for _ in range(n)]
        s = sum(vals)
        return [v / s for v in vals]

    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("  PASS  %s%s" % (name, (" -- " + detail) if detail else ""))
        else:
            print("  FAIL  %s%s" % (name, (" -- " + detail) if detail else ""))
            failures.append(name)

    rng = random.Random(20190408)

    # ---------------------------------------------------------------- test 1
    print("[1] Eq (10)-(12): EdgeWeights.for_item == naive user_to_user_item")
    U, C, Z, M = 40, 7, 5, 6
    pi = [_dirichlet_row(rng, C) for _ in range(U)]
    eta = [[rng.random() for _ in range(C)] for _ in range(C)]
    theta = [_dirichlet_row(rng, Z) for _ in range(C)]
    p_z_given_i = [_dirichlet_row(rng, Z) for _ in range(M)]
    model = _M(pi, eta, theta, p_z_given_i)

    edges = set()
    while len(edges) < 220:
        a = rng.randrange(U)
        b = rng.randrange(U)
        if a != b:
            edges.add((a, b))
    edges = sorted(edges)
    ds = _DS(edges)

    # Eq (10) shape/value check.
    P = community_to_community(model)
    ok10 = True
    for cp in range(C):
        for c in range(C):
            for z in range(Z):
                if abs(P[cp][c][z] - eta[cp][c] * theta[c][z]) > 0.0:
                    ok10 = False
    check("Eq (10) P[c'][c][z] == eta[c'][c]*theta[c][z]", ok10)

    ew = EdgeWeights(model, ds)
    worst = 0.0
    worst_at = None
    for i in range(M):
        fast = ew.for_item(i)
        check_count = 0
        for (u, v) in edges:
            ref = user_to_user_item(model, u, v, i)  # Eq (11)+(12), naive
            got = fast[(u, v)]
            denom = abs(ref) if abs(ref) > 0.0 else 1.0
            rel = abs(got - ref) / denom
            if rel > worst:
                worst = rel
                worst_at = (u, v, i)
            check_count += 1
    check("for_item agrees with user_to_user_item to 1e-9 relative "
          "(%d edges x %d items)" % (len(edges), M),
          worst <= 1e-9,
          "max rel err = %.3e at %s" % (worst, worst_at))
    check("no clamping needed for a well-formed model",
          ew.n_clamped == 0,
          "n_clamped=%d, max_raw_pp=%.6f" % (ew.n_clamped, ew.max_raw_pp))

    # Eq (11) reference vs. a direct triple sum through Eq (10).
    u0, v0, z0 = edges[0][0], edges[0][1], 2
    ref11 = 0.0
    for cp in range(C):
        for c in range(C):
            ref11 += pi[v0][c] * pi[u0][cp] * P[cp][c][z0]
    check("Eq (11) matches sum over Eq (10) table",
          abs(user_to_user_topic(model, u0, v0, z0) - ref11) <= 1e-12)

    # Clamping must be visible when the model violates its invariants.
    # eta > 1 (illegal: it is a Bernoulli parameter) together with degenerate
    # theta rows drives sum_c a_u[c]*pi_v[c]*thetabar_i[c] up to ~1.7.
    bad = _M(pi, [[1.7] * C for _ in range(C)],
             [[1.0] * Z for _ in range(C)], p_z_given_i)
    ew_bad = EdgeWeights(bad, ds)
    ew_bad.for_item(0)
    check("clamping is counted in the public n_clamped attribute",
          ew_bad.n_clamped > 0 and ew_bad.max_raw_pp > 1.0,
          "n_clamped=%d, max_raw_pp=%.4f" % (ew_bad.n_clamped, ew_bad.max_raw_pp))

    # ---------------------------------------------------------------- test 2
    print("[2] Eq (17): hand-computed ap on a 4-node chain")
    # Chain 0 -> 1 -> 2 -> 3 with pp = 0.5, 0.4, 0.6 and h = 0.1.
    #   pp(MIP(0,1)) = 0.5,  pp(MIP(0,2)) = 0.20,  pp(MIP(0,3)) = 0.12  (all >= h)
    # Eq (17) with S = {0}:
    #   ap(0) = 1                          (0 in S)
    #   ap(1) = 1 - (1 - ap(0)*0.5) = 0.5
    #   ap(2) = 1 - (1 - ap(1)*0.4) = 0.20
    #   ap(3) = 1 - (1 - ap(2)*0.6) = 0.12
    # Eq (18): I({0}) = 1 + 0.5 + 0.20 + 0.12 = 1.82
    chain_out = [[1], [2], [3], []]
    chain_in = [[], [0], [1], [2]]
    chain_pp = {(0, 1): 0.5, (1, 2): 0.4, (2, 3): 0.6}
    mia_chain = MIA(4, chain_out, chain_in, chain_pp, h=0.1)
    for node, expect in ((0, 1.0), (1, 0.5), (2, 0.20), (3, 0.12)):
        check("ap(%d|{0}) == %.2f" % (node, expect),
              abs(mia_chain.ap(node, {0}) - expect) <= 1e-12,
              "got %.12f" % mia_chain.ap(node, {0}))
    got = mia_chain.influence({0})
    check("Eq (18) I({0}) on the chain == 1.82 (hand computed)",
          abs(got - 1.82) <= 1e-12, "got %.12f" % got)

    # Uniform variant: p = 0.5 => I({0}) = 1 + .5 + .25 + .125 = 1.875
    mia_chain2 = MIA(4, chain_out, chain_in,
                     {(0, 1): 0.5, (1, 2): 0.5, (2, 3): 0.5}, h=0.1)
    check("Eq (18) I({0}) with p=0.5 chain == 1.875",
          abs(mia_chain2.influence({0}) - 1.875) <= 1e-12,
          "got %.12f" % mia_chain2.influence({0}))

    # The threshold h must actually cut paths.  With h = 0.3:
    #   MIIA(3) keeps 2 (pp 0.6) but drops 1 (0.24) and 0 (0.12)  -> {2,3}
    #   MIIA(2) keeps 1 (pp 0.4) but drops 0 (0.20)               -> {1,2}
    # so ap(2|{0}) = ap(3|{0}) = 0 and Eq (18) gives 1 + 0.5 = 1.5.
    mia_chain3 = MIA(4, chain_out, chain_in, chain_pp, h=0.3)
    check("Eq (15) h=0.3 truncates MIIA(3) to {2,3}",
          mia_chain3.miia(3)[0] == [2, 3],
          "got %s" % (mia_chain3.miia(3)[0],))
    check("Eq (15) h=0.3 truncates MIIA(2) to {1,2}",
          mia_chain3.miia(2)[0] == [1, 2],
          "got %s" % (mia_chain3.miia(2)[0],))
    check("Eq (18) I({0}) with h=0.3 == 1.5",
          abs(mia_chain3.influence({0}) - 1.5) <= 1e-12,
          "got %.12f" % mia_chain3.influence({0}))

    # marginal_gain consistency with a full recomputation.
    mg = mia_chain.marginal_gain({0}, 2)
    full = mia_chain.influence({0, 2}) - mia_chain.influence({0})
    check("marginal_gain == I(S u u) - I(S) on the chain",
          abs(mg - full) <= 1e-12, "%.12f vs %.12f" % (mg, full))

    # ---------------------------------------------------------------- test 3
    print("[3] Eq (17)/(18) vs 200k-run Monte-Carlo IC")
    # MIA is a tree approximation: it keeps only the maximum influence path into
    # each node, so it under-counts nodes fed by several comparable paths.  It
    # is *exact* on an arborescence (a unique path to every node), so we check
    # both regimes: exact agreement on a random out-tree, and agreement to a few
    # percent on a sparse random digraph with small probabilities, where
    # multi-path corrections are second order.
    N_MC = 200000

    # (a) random out-tree rooted at 0 -- MIA is exact here.
    n_a = 18
    out_a = [[] for _ in range(n_a)]
    in_a = [[] for _ in range(n_a)]
    pp_a = {}
    for v in range(1, n_a):
        u = rng.randrange(0, v)
        out_a[u].append(v)
        in_a[v].append(u)
        pp_a[(u, v)] = 0.25 + 0.5 * rng.random()
    mia_a = MIA(n_a, out_a, in_a, pp_a, h=1e-12)
    exact_a = mia_a.influence({0})
    t0 = time.time()
    mc_a = _monte_carlo_ic(out_a, pp_a, {0}, N_MC, random.Random(12345))
    ta = time.time() - t0
    rel_a = abs(exact_a - mc_a) / mc_a
    check("tree: MIA %.4f vs MC(%d) %.4f within 1%%" % (exact_a, N_MC, mc_a),
          rel_a <= 0.01, "rel diff = %.4f%% (%.1fs)" % (100 * rel_a, ta))

    # (b) sparse random digraph, small probabilities.
    n_b = 22
    eset = set()
    while len(eset) < 30:
        a = rng.randrange(n_b)
        b = rng.randrange(n_b)
        if a != b:
            eset.add((a, b))
    out_b = [[] for _ in range(n_b)]
    in_b = [[] for _ in range(n_b)]
    pp_b = {}
    for (a, b) in sorted(eset):
        out_b[a].append(b)
        in_b[b].append(a)
        pp_b[(a, b)] = 0.02 + 0.10 * rng.random()
    mia_b = MIA(n_b, out_b, in_b, pp_b, h=1e-12)
    S_b = {0, 1}
    exact_b = mia_b.influence(S_b)
    t0 = time.time()
    mc_b = _monte_carlo_ic(out_b, pp_b, S_b, N_MC, random.Random(999))
    tb = time.time() - t0
    rel_b = abs(exact_b - mc_b) / mc_b
    check("graph: MIA %.4f vs MC(%d) %.4f within 3%%" % (exact_b, N_MC, mc_b),
          rel_b <= 0.03, "rel diff = %.4f%% (%.1fs)" % (100 * rel_b, tb))

    # ---------------------------------------------------------------- test 4
    print("[4] greedy_incremental == brute-force greedy")
    n_c = 30
    eset = set()
    while len(eset) < 90:
        a = rng.randrange(n_c)
        b = rng.randrange(n_c)
        if a != b:
            eset.add((a, b))
    out_c = [[] for _ in range(n_c)]
    in_c = [[] for _ in range(n_c)]
    pp_c = {}
    for (a, b) in sorted(eset):
        out_c[a].append(b)
        in_c[b].append(a)
        pp_c[(a, b)] = 0.05 + 0.5 * rng.random()
    mia_c = MIA(n_c, out_c, in_c, pp_c, h=0.1)

    def brute_greedy(mia, K, cands):
        S = []
        for _ in range(K):
            best, best_g = None, -1.0
            for u in sorted(cands):
                if u in S:
                    continue
                g = mia.influence(S + [u]) - mia.influence(S)
                if g > best_g + 1e-12:
                    best, best_g = u, g
            if best is None or best_g <= 0.0:
                break
            S.append(best)
        return S

    K = 5
    fast_S = mia_c.greedy_incremental(K)
    slow_S = brute_greedy(mia_c, K, range(n_c))
    check("greedy_incremental matches brute force (all candidates)",
          fast_S == slow_S, "%s vs %s" % (fast_S, slow_S))
    check("greedy spread matches",
          abs(mia_c.influence(fast_S) - mia_c.influence(slow_S)) <= 1e-9,
          "I=%.6f" % mia_c.influence(fast_S))

    cands = sorted(rng.sample(range(n_c), 12))
    fast_R = mia_c.greedy_incremental(4, candidates=cands)
    slow_R = brute_greedy(mia_c, 4, cands)
    check("greedy_incremental honours the candidate restriction",
          fast_R == slow_R and set(fast_R) <= set(cands),
          "%s vs %s" % (fast_R, slow_R))

    # marginal_gain agreement on the random graph.
    worst_mg = 0.0
    base = [fast_S[0], fast_S[1]]
    for u in range(n_c):
        a = mia_c.marginal_gain(base, u)
        b = mia_c.influence(base + [u]) - mia_c.influence(base)
        worst_mg = max(worst_mg, abs(a - b))
    check("marginal_gain exact on random graph",
          worst_mg <= 1e-9, "max abs err = %.3e" % worst_mg)

    # ---------------------------------------------------------------- test 5
    print("[5] node-induced subgraph construction (Algorithm 2, line 34)")
    sub = sorted(rng.sample(range(n_c), 14))
    mia_sub = MIA(n_c, out_c, in_c, pp_c, h=0.1, nodes=sub)
    check("subgraph nodes are exactly the requested subset",
          mia_sub.nodes == sub)
    inside = [u for u in mia_sub.miia(sub[0])[0]]
    check("MIIA on the subgraph stays inside the subset",
          all(x in set(sub) for x in inside))
    check("ap of an excluded node is 0",
          mia_sub.ap([x for x in range(n_c) if x not in set(sub)][0], {sub[0]}) == 0.0)
    # An induced subgraph can only lose paths, so I_m <= I on the full graph.
    check("I_m(S) <= I(S) on the full graph",
          mia_sub.influence({sub[0]}) <= mia_c.influence({sub[0]}) + 1e-12,
          "%.6f <= %.6f" % (mia_sub.influence({sub[0]}), mia_c.influence({sub[0]})))
    sub_S = mia_sub.greedy_incremental(3)
    check("greedy on the subgraph selects only subgraph nodes",
          set(sub_S) <= set(sub), "%s" % sub_S)

    # ---------------------------------------------------------------- done
    print("")
    if failures:
        print("FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("ALL CHECKS PASSED")
