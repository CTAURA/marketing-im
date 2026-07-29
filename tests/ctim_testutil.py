"""Shared fixtures and helpers for the CTIM test suite.

Standard library only.  Python 3.9 compatible.  Every random draw goes through
an explicit ``random.Random`` instance handed in by the caller, so every fixture
built here is bit-reproducible from its seed.

This module is imported by the ``test_*.py`` files through the sys.path
bootstrap at the top of each of them, so it works under both

    python3 -m unittest discover -s tests -v
    python3 -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import os
import sys

# --- repo bootstrap --------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
for _p in (REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ctim.dataset import Dataset, build_adjacency  # noqa: E402

__all__ = [
    "REPO_ROOT",
    "adjusted_rand",
    "best_label_agreement",
    "argmax_labels",
    "dirichlet_row",
    "FakeModel",
    "FakeDataset",
    "make_planted_topic_corpus",
    "make_planted_block_graph",
    "make_planted_dataset",
    "make_random_digraph",
    "make_random_model",
    "monte_carlo_ic",
]


# ---------------------------------------------------------------------------
# Clustering agreement metrics
# ---------------------------------------------------------------------------


def adjusted_rand(a, b):
    """Adjusted Rand index between two labellings of the same n points.

    Permutation invariant, so it is the right way to check "the sampler
    recovered the planted partition" when the label *names* are arbitrary (a
    Gibbs sampler has no reason to call the planted community 0 "community 0").
    """
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


def best_label_agreement(planted, found, n_labels=2):
    """Fraction of points agreeing under the best relabelling (small n_labels).

    Only used for the 2-label planted fixtures, where enumerating both
    permutations is trivial and the resulting number ("48 of 50 users landed in
    the right block") is far more legible in a failure message than an ARI.
    """
    import itertools

    n = len(planted)
    if n == 0:
        return 1.0
    best = 0
    for perm in itertools.permutations(range(n_labels)):
        agree = sum(1 for p, f in zip(planted, found) if perm[f] == p)
        if agree > best:
            best = agree
    return best / float(n)


def argmax_labels(mat):
    """Row-wise argmax with ties broken by the lowest index (matches Eq (19))."""
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


# ---------------------------------------------------------------------------
# Duck-typed stand-ins for ctim.gibbs.Model / ctim.dataset.Dataset
# ---------------------------------------------------------------------------


class FakeModel(object):
    """Minimal stand-in for ``ctim.gibbs.Model`` (same attribute names).

    ``ctim.influence`` and ``ctim.ctim`` only ever read ``.pi``, ``.eta``,
    ``.theta`` and ``.p_z_given_i``, so a hand-built model lets the influence
    tests use exact, known parameter values instead of whatever a sampler
    happened to converge to.
    """

    def __init__(self, pi, eta, theta, p_z_given_i):
        self.pi = pi
        self.eta = eta
        self.theta = theta
        self.p_z_given_i = p_z_given_i
        self.phi = p_z_given_i
        self.psi = None
        self.hyper = None


class FakeDataset(object):
    """Minimal stand-in exposing only what ``EdgeWeights`` needs."""

    def __init__(self, edges, n_users=None):
        self.edges = list(edges)
        if n_users is None:
            n_users = 1 + max((max(u, v) for u, v in self.edges), default=-1)
        self.n_users = n_users
        self.out_adj, self.in_adj = build_adjacency(self.n_users, self.edges)


def dirichlet_row(rng, n):
    """A random point in the (n-1)-simplex (rows of pi / theta / P(z|i))."""
    vals = [rng.random() + 1e-3 for _ in range(n)]
    s = sum(vals)
    return [v / s for v in vals]


def make_random_model(rng, n_users, C, Z, n_items):
    """A well-formed random model: pi/theta/P(z|i) are distributions, eta in (0,1)."""
    pi = [dirichlet_row(rng, C) for _ in range(n_users)]
    eta = [[rng.random() for _ in range(C)] for _ in range(C)]
    theta = [dirichlet_row(rng, Z) for _ in range(C)]
    p_z_given_i = [dirichlet_row(rng, Z) for _ in range(n_items)]
    return FakeModel(pi, eta, theta, p_z_given_i)


# ---------------------------------------------------------------------------
# Random / planted graph fixtures
# ---------------------------------------------------------------------------


def make_random_digraph(rng, n_users, n_edges, p_lo=0.05, p_hi=0.55):
    """A simple random digraph with random edge probabilities.

    Returns ``(out_adj, in_adj, pp)`` where ``pp`` maps ``(u, v) -> p``.
    """
    eset = set()
    while len(eset) < n_edges:
        a = rng.randrange(n_users)
        b = rng.randrange(n_users)
        if a != b:
            eset.add((a, b))
    out_adj = [[] for _ in range(n_users)]
    in_adj = [[] for _ in range(n_users)]
    pp = {}
    for (a, b) in sorted(eset):
        out_adj[a].append(b)
        in_adj[b].append(a)
        pp[(a, b)] = p_lo + (p_hi - p_lo) * rng.random()
    return out_adj, in_adj, pp


def make_planted_topic_corpus(rng, n_items=16, n_attrs=20, attr_tokens=120,
                              noise=0.05):
    """A 2-topic planted corpus for stage 1 (Eq (1)-(3)).

    Item ``i`` has planted topic ``i % 2``.  Topic 0 draws its attribute values
    uniformly from the lower half of the vocabulary, topic 1 from the upper
    half, with probability ``noise`` of drawing from the wrong half.  The two
    topics are therefore perfectly separable in principle and the sampler has to
    find that separation.

    Returns ``(item_attrs, item_topic)``.
    """
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
    return item_attrs, item_topic


def make_planted_block_graph(rng, n_per_comm=25, n_items=16, n_attrs=20,
                             p_in=0.20, p_out=0.005, logs_per_edge=25,
                             attr_tokens=120, noise=0.05):
    """A planted-partition instance with 2 communities AND 2 aligned topics.

    Community 0 adopts topic-0 items and community 1 adopts topic-1 items, so
    the link structure (Eq (4)) and the topic structure (Eq (6)) both point at
    the same 2-way partition.

    The instance is deliberately log-rich (D/E ~ 25).  At C = 2 the paper's own
    hyperparameter rule makes the link term of Eq (4) nearly uninformative:
    rho = 50/C = 25 swamps the per-user counts and, with only C^2 = 4 blocks,
    every n_{c'c} sits far above eps0 so every eta-tilde is ~0.99 whatever the
    partition.  Essentially all of the community signal at C = 2 therefore comes
    from the Eq (6) theta factor, which needs enough potential-influence logs
    per community for the soft counts to dominate alpha = 50/Z.

    Returns ``(n_users, edges, item_attrs, logs_d, comm_labels, item_topic)``
    where ``logs_d`` is a list of ``(u, v, i)`` potential-influence logs.
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

    item_attrs, item_topic = make_planted_topic_corpus(
        rng, n_items=n_items, n_attrs=n_attrs, attr_tokens=attr_tokens,
        noise=noise,
    )
    by_topic = {
        0: [i for i in range(n_items) if item_topic[i] == 0],
        1: [i for i in range(n_items) if item_topic[i] == 1],
    }

    logs_d = []
    for (u, v) in edges:
        if labels[u] != labels[v]:
            continue
        for _ in range(logs_per_edge):
            t = labels[u] if rng.random() >= noise else 1 - labels[u]
            logs_d.append((u, v, rng.choice(by_topic[t])))

    return U, edges, item_attrs, logs_d, labels, item_topic


def make_planted_dataset(rng, n_comm=3, per_comm=12, n_items=15, n_attrs=9,
                         z_true=3, p_in=0.30, p_out=0.02):
    """A full ``Dataset`` with planted communities, for the CTIM/baseline tests.

    Dense intra-community arcs, sparse inter-community arcs, and item adoptions
    that cascade inside a single community so the potential-influence logs of
    Definition 1 land on intra-community arcs.

    Returns ``(ds, true_comm)``.
    """
    n_users = n_comm * per_comm
    true_comm = [v // per_comm for v in range(n_users)]

    edges = []
    for u in range(n_users):
        for v in range(n_users):
            if u == v:
                continue
            p = p_in if true_comm[u] == true_comm[v] else p_out
            if rng.random() < p:
                edges.append((u, v))

    per_topic = max(1, n_attrs // z_true)
    item_attrs = []
    for i in range(n_items):
        z = i % z_true
        block = list(range(z * per_topic, min(n_attrs, (z + 1) * per_topic)))
        if not block:
            block = [0]
        attrs = [rng.choice(block) for _ in range(4)]
        if rng.random() < 0.2:
            attrs.append(rng.randrange(n_attrs))  # a little cross-topic noise
        item_attrs.append(sorted(attrs))

    logs = []
    t = 1000
    for i in range(n_items):
        c = i % n_comm
        pool = [v for v in range(n_users) if true_comm[v] == c]
        rng.shuffle(pool)
        adopters = pool[: max(4, len(pool) // 2)]
        for j, u in enumerate(adopters):
            logs.append((u, i, t + 10 * j))
        t += 10000
    logs.sort(key=lambda r: (r[2], r[0], r[1]))

    out_adj, in_adj = build_adjacency(n_users, edges)
    ds = Dataset(name="planted", n_users=n_users, n_items=n_items,
                 n_attrs=n_attrs, edges=edges, out_adj=out_adj, in_adj=in_adj,
                 logs=logs, item_attrs=item_attrs, source="ctim test suite")
    return ds, true_comm


# ---------------------------------------------------------------------------
# Monte-Carlo Independent Cascade reference simulator
# ---------------------------------------------------------------------------


def monte_carlo_ic(out_adj, pp, S, n_sims, rng):
    """Mean number of activated nodes under the Independent Cascade model.

    This is the *ground truth diffusion process* that Section 4.2.2's MIA model
    approximates; ``MIA.influence`` itself never uses Monte Carlo.  ``rng`` must
    be an explicit ``random.Random``.
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
