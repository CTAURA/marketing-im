"""Algorithm 2 -- community-based topic-aware influence maximization (CTIM).

Reference implementation of Algorithm 2 (Section 4.2.3) of

    Huimin Huang, Hong Shen, Zaiqiao Meng, Huajian Chang, Huaiwen He.
    "Community-based influence maximization for viral marketing",
    Applied Intelligence (2019). DOI 10.1007/s10489-018-1387-8

The algorithm as printed in the paper::

     1: for c  = 1..C do
     2:   for c' = 1..C do
     3:     for z = 1..Z do
     4:       P(c | z, c') = eta_{c'c} * theta_{c,z}                    # Eq (10)
     5:     end for
     6:   end for
     7: end for
     8: for d = (u,v,i) in D do
     9:   for c  = 1..C do
    10:     for c' = 1..C do
    11:       P(v | z, u) = sum_{c,c'} pi_vc * pi_uc' * P(c | z, c')    # Eq (11)
    12:     end for
    13:   end for
    14: end for
    15: for i = 1..M do
    16:   for d = (u,v,i) in D do
    17:     for z = 1..Z do
    18:       P(v | i, u) = sum_z P(z|i) * P(v | z, u)                  # Eq (12)
    19:     end for
    20:   end for
    21: end for
    22: for v = 1..U do
    23:   c^v_m <- argmax_c pi_{v,c}                                    # Eq (19)
    24: end for
    25: S = S_1 = S_2 = ... = S_C = empty
    26: for k = 1..K do
    27:   I[0,k] = 0;  s[0,k] = 0
    28: end for
    29: for m = 1..C do
    30:   I[m,0] = 0
    31: end for
    32: for k = 1..K do
    33:   for m = 1..C do
    34:     dI_m = max( I_m(S union u) - I_m(S) ),  u in c_m
    35:     I[m,k] = max( I[m-1,k], I[m,k-1] + dI_m )
    36:     if I[C,k-1] + dI_m >= I[m-1,k] then
    37:       s[m,k] = m
    38:     else
    39:       s[m,k] = s[m-1,k]
    40:     end if
    41:   end for
    42:   j = s[C,k]
    43:   u_k = argmax_{u in c_j} ( I(S_j union u) - I(S_j) )
    44:   S_j = S_j union u_k ;  S = S union u_k
    45: end for

Documented deviations (SPEC.md Section 6, "Reading notes"):

 * **Lines 8-21.**  Written literally these loops cost O(C^2 |D| + M |D| Z) and,
   worse, only cover pairs that appear in the potential-influence logs `D`.  The
   diffusion graph of Eq (13)-(18) is `G`, so we evaluate Eq (11)+(12) for
   *every* directed edge of `G` through the exact algebraic factorisation of
   SPEC.md Section 3 (`ctim.influence.EdgeWeights`), which is algebraically
   identical, bit-for-bit up to floating point, and costs O(U C^2 + E C) per
   item.  `D` is used for *learning* the model, not for defining the graph.

 * **Line 36 vs line 35.**  Line 35 takes `max(I[m-1,k], I[m,k-1] + dI_m)` but
   line 36 tests `I[C,k-1] + dI_m >= I[m-1,k]`.  The back-pointer therefore does
   not track the max that was actually taken.  `dp_tiebreak="consistent"` (the
   default) tests `I[m,k-1] + dI_m >= I[m-1,k]`, so `s` always records which
   branch of line 35 won; `dp_tiebreak="paper-literal"` reproduces the printed
   `I[C,k-1]` form.  Both are supported so the two can be compared.

 * **Line 34, `I_m`.**  `I_m` is the influence spread evaluated on community
   `m`'s *node-induced subgraph*, hence `S` intersected with `c_m` is exactly
   `S_m` and the term is `I_m(S_m u {u}) - I_m(S_m)`.

 * **Line 43, `I` vs `I_j`.**  Line 43 writes `I(S_j u u) - I(S_j)`.  Read as the
   global `I` the dynamic program would be optimising a quantity it never
   selects on, which makes the DP vacuous; read as `I_j` it is precisely the
   maximiser already found at line 34 for `m = j`.  We use `I_j` -- so line 43
   is answered from the cached per-community incremental gains at no extra cost.

 * **Line 42, empty or exhausted `c_j`.**  For an empty community `dI_m = 0`
   (max over the empty set), which makes it possible for `s[C,k]` to point at a
   community with no selectable node.  Rather than emit fewer than `K` seeds we
   fall back to the community with the largest available gain, and count the
   event in `last_stats["n_fallbacks"]`.

Efficiency.  The paper's claim is that CTIM is faster than the baselines by
orders of magnitude, so the `K x C` double loop of lines 32-41 must not
recompute anything.  Two caches carry that:

  * one `MIA` object per non-empty community, built once on that community's
    induced subgraph (total cost O(|E|) across all communities);
  * one exact incremental `IncInf[u] = I_m(S_m u {u}) - I_m(S_m)` table per
    community, repaired only for the single community that received a seed
    (adding a seed can only change `ap` inside `MIOA(u_k, h)`).

Line 34 is then an O(log |c_m|) heap peek instead of an O(|c_m|) rescan, and the
whole `K x C` loop costs O(K C log U) plus the repair work for `K` seeds.

Standard library only.  Python 3.9 compatible.  No module-level randomness: this
module is fully deterministic and takes no rng at all (the only `random.Random`
in the self-test drives the synthetic dataset).
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

try:  # normal package import
    from .influence import EdgeWeights, MIA
except ImportError:  # running this file directly as a script
    import os as _os
    import sys as _sys

    _sys.path.insert(
        0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    )
    from ctim.influence import EdgeWeights, MIA

__all__ = [
    "detect_communities",
    "ctim_select_seeds",
    "ctim_run",
    "RunResult",
]

# Tolerance used when deciding whether a lazily-cached heap key is stale.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """One end-to-end seed-selection run.

    seeds    the selected seed set, in selection order
    spread   I(S) per Eq (18), measured on the FULL graph with this item's
             edge weights (the common evaluator all methods are scored with)
    seconds  wall-clock seconds of the seed-selection procedure itself
    extra    free-form diagnostics; for CTIM this is `ctim_select_seeds`'s
             `last_stats` plus the evaluation timing
    """

    seeds: list
    spread: float
    seconds: float
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Eq (19) -- community detection
# ---------------------------------------------------------------------------


def detect_communities(pi) -> list:
    """Eq (19).  `c^v_m <- argmax_c pi_{v,c}`.  Returns comm[v] in [0, C).

    Algorithm 2, lines 22-24.  Ties are broken deterministically by the lowest
    community index (the comparison below is strict `>`, so the first maximiser
    encountered wins).
    """
    comm = []
    for v in range(len(pi)):
        row = pi[v]
        if not row:
            comm.append(0)
            continue
        best_c = 0
        best_p = row[0]
        for c in range(1, len(row)):
            p = row[c]
            if p > best_p:  # Eq (19): argmax_c pi[v][c], lowest index wins ties
                best_p = p
                best_c = c
        comm.append(best_c)
    return comm


# ---------------------------------------------------------------------------
# Per-community incremental state (the cache behind Algorithm 2, line 34)
# ---------------------------------------------------------------------------


class _CommunitySeedState:
    """Community `m`'s induced-subgraph MIA plus its exact incremental gains.

    `inc[u]` is maintained equal to

        IncInf_m(u) = I_m(S_m u {u}) - I_m(S_m)              # Algorithm 2, line 34

    at all times, where `I_m` is Eq (18) evaluated on the node-induced subgraph
    of community `m` and `S_m` is the community's current seed set.  A max-heap
    keyed by `(-inc[u], u)` answers line 34 in O(log |c_m|) amortised, with ties
    broken by the lowest node index because the node id is the heap's secondary
    key.

    The heap uses the usual lazy-evaluation invariant: every unseeded candidate
    has at least one heap entry whose key is an *upper* bound on its current
    gain (safe because `I_m` is monotone and submodular, so `inc[u]` can only
    decrease as `S_m` grows).  A stale entry is corrected in place on peek.
    """

    __slots__ = ("m", "mia", "members", "cand_set", "S", "S_set", "inc", "_heap")

    def __init__(self, m, mia, members):
        self.m = m
        self.mia = mia
        self.members = sorted(members)
        self.cand_set = set(self.members)
        self.S = []  # S_m, in selection order (Algorithm 2, line 25 / 44)
        self.S_set = set()
        self.inc = dict.fromkeys(self.members, 0.0)
        self._init_inc()
        self._heap = [(-self.inc[u], u) for u in self.members]
        heapq.heapify(self._heap)

    # -- construction -------------------------------------------------------

    def _init_inc(self):
        """IncInf_m(u) for S_m = empty, i.e. I_m({u}) = sum_v ap(v|{u}).

        Algorithm 2, line 34 at k = 1.  `ap(v|{u})` is non-zero only for v whose
        MIIA contains u (Eq (15)/(17)), so scanning the arborescences visits
        exactly the non-zero terms of Eq (18).
        """
        mia = self.mia
        inc = self.inc
        cand = self.cand_set
        for v in mia.nodes:
            nodes_v = mia.miia(v)[0]  # Eq (15)
            for w in nodes_v:
                if w in cand:
                    inc[w] += mia.ap(v, {w})  # Eq (17), summed per Eq (18)

    # -- line 34 ------------------------------------------------------------

    def best(self):
        """Algorithm 2, line 34: `max_{u in c_m} ( I_m(S_m u u) - I_m(S_m) )`.

        Returns `(u, dI_m)`, or `(None, 0.0)` when the community has no
        selectable node left -- the max over an empty set, taken as 0.
        """
        heap = self._heap
        inc = self.inc
        S_set = self.S_set
        while heap:
            neg, u = heap[0]
            if u in S_set:
                heapq.heappop(heap)  # already a seed: its gain is 0 by definition
                continue
            cur = inc[u]
            if -neg > cur + _EPS:
                heapq.heapreplace(heap, (-cur, u))  # stale upper bound, correct it
                continue
            return u, cur
        return None, 0.0

    # -- line 44 ------------------------------------------------------------

    def add(self, u):
        """Algorithm 2, line 44 for one community: `S_m <- S_m u {u}`.

        Adding `u` can only change `ap(v|S_m)` for `v in MIOA(u,h)` (Eq (16)),
        because every other node's MIIA excludes `u`.  So the exact IncInf table
        is repaired by subtracting those nodes' contributions at the old `S_m`
        and re-adding them at the new one; every other candidate's entry is
        already correct and is left untouched.  This is what keeps the `K x C`
        loop of lines 32-41 out of quadratic territory.
        """
        mia = self.mia
        inc = self.inc
        cand = self.cand_set
        S_set = self.S_set

        affected = mia.mioa(u)[0]  # Eq (16)
        trees = [(v, mia.miia(v)[0]) for v in affected]  # Eq (15)

        # Remove the affected nodes' contributions, evaluated at the OLD S_m.
        for v, nodes_v in trees:
            base = mia.ap(v, S_set)  # Eq (17)
            for w in nodes_v:
                if w in cand and w not in S_set and w != u:
                    S_set.add(w)
                    inc[w] -= mia.ap(v, S_set) - base  # Eq (17)/(18)
                    S_set.discard(w)

        self.S.append(u)
        S_set.add(u)

        # Re-add them, evaluated at the NEW S_m.
        dirty = set()
        for v, nodes_v in trees:
            base = mia.ap(v, S_set)  # Eq (17)
            for w in nodes_v:
                if w in cand and w not in S_set:
                    S_set.add(w)
                    inc[w] += mia.ap(v, S_set) - base  # Eq (17)/(18)
                    S_set.discard(w)
                    dirty.add(w)

        for w in sorted(dirty):
            heapq.heappush(self._heap, (-inc[w], w))

    def spread(self):
        """I_m(S_m), Eq (18) on the induced subgraph."""
        return self.mia.influence(self.S_set)


# ---------------------------------------------------------------------------
# Algorithm 2
# ---------------------------------------------------------------------------


def ctim_select_seeds(model, ds, item, K, h=0.1,
                      dp_tiebreak="consistent",  # or "paper-literal"
                      edge_weights=None) -> list:
    """Algorithm 2 in full, lines 1-45.  Returns the seed list, in selection order.

    Parameters
    ----------
    model         a `ctim.gibbs.Model` (needs `.pi`, `.eta`, `.theta`,
                  `.p_z_given_i`)
    ds            a `ctim.dataset.Dataset` (needs `.n_users`, `.edges`,
                  `.out_adj`, `.in_adj`)
    item          the item id whose topic mix defines the edge weights, Eq (12)
    K             seed-set size
    h             MIA threshold of Eq (15)/(16); the paper uses 0.1
    dp_tiebreak   "consistent" (default, line 36 reads `I[m,k-1]`, matching the
                  max actually taken on line 35) or "paper-literal" (line 36
                  reads `I[C,k-1]` exactly as printed).  See the module
                  docstring and SPEC.md Section 6.
    edge_weights  a pre-built `ctim.influence.EdgeWeights`; supply it to share
                  the O(U*C^2) `a_u` cache across several items

    Per-phase wall-clock seconds are recorded in `ctim_select_seeds.last_stats`
    under the keys `weights` (lines 1-21), `detect` (lines 22-24), `subgraphs`
    (the per-community induced-subgraph MIA plus the initial IncInf table, i.e.
    the first evaluation of line 34, hoisted out of the loop), `dp` (lines
    32-41) and `select` (lines 42-45).  Those five are disjoint and sum to
    `total`.
    """
    if dp_tiebreak not in ("consistent", "paper-literal"):
        raise ValueError(
            "dp_tiebreak must be 'consistent' or 'paper-literal', got %r"
            % (dp_tiebreak,)
        )
    K = int(K)
    t_start = time.time()

    if K <= 0:
        ctim_select_seeds.last_stats = {
            "weights": 0.0, "detect": 0.0, "subgraphs": 0.0, "dp": 0.0,
            "select": 0.0, "total": 0.0, "K": K, "item": item, "h": h,
            "dp_tiebreak": dp_tiebreak, "n_seeds": 0, "n_fallbacks": 0,
        }
        return []

    C = len(model.eta)
    if C <= 0:
        raise ValueError("model has no communities (len(model.eta) == 0)")
    if model.pi and len(model.pi[0]) != C:
        raise ValueError(
            "model.pi has width %d but model.eta has %d communities"
            % (len(model.pi[0]), C)
        )
    n_users = ds.n_users
    if len(model.pi) < n_users:
        raise ValueError(
            "model.pi has %d rows but the dataset has %d users"
            % (len(model.pi), n_users)
        )

    # ---------------------------------------------------------------- 1-21
    # Algorithm 2, lines 1-7:   P(c|z,c') = eta[c'][c] * theta[c][z]   # Eq (10)
    # Algorithm 2, lines 8-14:  P(v|z,u)  = sum_{c,c'} pi[v][c] pi[u][c'] P(c|z,c')
    #                                                                   # Eq (11)
    # Algorithm 2, lines 15-21: P(v|i,u)  = sum_z P(z|i) P(v|z,u)       # Eq (12)
    #
    # `EdgeWeights` fuses all three loop nests into the exact factorisation of
    # SPEC.md Section 3,
    #     P(v|i,u) = sum_c a_u[c] * pi[v][c] * thetabar_i[c],
    #     a_u[c] = sum_c' pi[u][c'] eta[c'][c],  thetabar_i[c] = sum_z P(z|i) theta[c][z],
    # which is algebraically identical to Eq (10)+(11)+(12) but costs
    # O(U C^2 + E C) per item instead of O(C^2 |D| + M |D| Z).  Unlike the
    # printed lines 8-21 it covers every edge of G, not only pairs occurring in
    # D -- the diffusion graph of Eq (13)-(18) is G (SPEC.md Section 6, note 3).
    t0 = time.time()
    ew = EdgeWeights(model, ds) if edge_weights is None else edge_weights
    pp = ew.for_item(item)  # Eq (12) for every directed edge of G
    t_weights = time.time() - t0

    # ---------------------------------------------------------------- 22-24
    t0 = time.time()
    comm = detect_communities(model.pi)  # Algorithm 2, lines 22-24 -- Eq (19)
    # members_by_m[m] holds community m-1's nodes; the DP indexes m = 1..C.
    members_by_m = [[] for _ in range(C + 1)]
    for v in range(n_users):
        cv = comm[v]
        if 0 <= cv < C:
            members_by_m[cv + 1].append(v)
    t_detect = time.time() - t0

    # ---------------------------------------------------------------- 25
    # Algorithm 2, line 25: S = S_1 = S_2 = ... = S_C = empty.
    S = []       # the global seed set, in selection order
    S_set = set()

    # Per-community induced-subgraph MIA + incremental IncInf table.  This is
    # the first evaluation of line 34 for every m, hoisted out of the k-loop:
    # line 33 sweeps every community at k = 1 regardless, so nothing is built
    # here that the loop would not have built anyway.  Empty communities are
    # skipped entirely -- their dI_m is the max over an empty set, i.e. 0.
    t0 = time.time()
    states = {}
    for m in range(1, C + 1):
        members = members_by_m[m]
        if not members:
            continue  # skip empty communities
        # I_m = Eq (18) on community m's node-induced subgraph (line 34).
        mia_m = MIA(n_users, ds.out_adj, ds.in_adj, pp, h=h, nodes=members)
        states[m] = _CommunitySeedState(m, mia_m, members)
    t_subgraphs = time.time() - t0

    # ---------------------------------------------------------------- 26-31
    # I[m][k] and s[m][k], (C+1) x (K+1), both zero-initialised.
    Iv = [[0.0] * (K + 1) for _ in range(C + 1)]
    sp = [[0] * (K + 1) for _ in range(C + 1)]
    for k in range(1, K + 1):
        Iv[0][k] = 0.0  # Algorithm 2, line 27
        sp[0][k] = 0     # Algorithm 2, line 27
    for m in range(1, C + 1):
        Iv[m][0] = 0.0  # Algorithm 2, line 30

    t_dp = 0.0
    t_select = 0.0
    n_fallbacks = 0

    # ---------------------------------------------------------------- 32-45
    for k in range(1, K + 1):  # Algorithm 2, line 32
        t0 = time.time()
        for m in range(1, C + 1):  # Algorithm 2, line 33
            st = states.get(m)
            if st is None:
                dI_m = 0.0  # empty community: max over the empty set
            else:
                # Algorithm 2, line 34:
                #   dI_m = max_{u in c_m} ( I_m(S u u) - I_m(S) )
                # with I_m taken on c_m's induced subgraph, so S n c_m = S_m.
                _u_m, dI_m = st.best()

            # Algorithm 2, line 35
            cand = Iv[m][k - 1] + dI_m
            prev = Iv[m - 1][k]
            Iv[m][k] = prev if prev > cand else cand

            # Algorithm 2, lines 36-40.  "consistent" uses the same I[m,k-1]
            # that line 35 maximised over; "paper-literal" uses the printed
            # I[C,k-1] (column k-1 is complete, so this read is well defined).
            ref = Iv[C][k - 1] if dp_tiebreak == "paper-literal" else Iv[m][k - 1]
            if ref + dI_m >= prev:
                sp[m][k] = m       # Algorithm 2, line 37
            else:
                sp[m][k] = sp[m - 1][k]  # Algorithm 2, line 39
        t_dp += time.time() - t0

        t0 = time.time()
        j = sp[C][k]  # Algorithm 2, line 42
        st = states.get(j)
        u_k = None
        if st is not None:
            u_k, _gain = st.best()
        if u_k is None:
            # Documented deviation: the DP pointed at an empty or exhausted
            # community.  Fall back to the community with the largest available
            # gain (lowest index on ties) rather than return fewer than K seeds.
            best_g = None
            for m2 in sorted(states):
                u2, g2 = states[m2].best()
                if u2 is None:
                    continue
                if best_g is None or g2 > best_g:
                    best_g, u_k, j = g2, u2, m2
            if u_k is None:
                t_select += time.time() - t0
                break  # no selectable node anywhere: stop early
            st = states[j]
            n_fallbacks += 1

        # Algorithm 2, line 43: u_k = argmax_{u in c_j} ( I_j(S_j u u) - I_j(S_j) ).
        # That argmax is exactly the maximiser line 34 already found for m = j,
        # so it is read straight off the cached incremental table.
        # Algorithm 2, line 44: S_j = S_j u {u_k};  S = S u {u_k}.
        st.add(u_k)
        S.append(u_k)
        S_set.add(u_k)
        t_select += time.time() - t0

    total = time.time() - t_start
    ctim_select_seeds.last_stats = {
        # per-phase wall-clock seconds (disjoint, summing to `total`)
        "weights": t_weights,      # lines 1-21,  Eq (10)/(11)/(12)
        "detect": t_detect,        # lines 22-24, Eq (19)
        "subgraphs": t_subgraphs,  # per-community MIA + initial IncInf (line 34)
        "dp": t_dp,                # lines 32-41
        "select": t_select,        # lines 42-45
        "total": total,
        # diagnostics
        "K": K,
        "C": C,
        "h": h,
        "item": item,
        "dp_tiebreak": dp_tiebreak,
        "n_users": n_users,
        "n_edges": len(pp),
        "n_communities_nonempty": len(states),
        "largest_community": max((len(v) for v in members_by_m), default=0),
        "n_seeds": len(S),
        "n_fallbacks": n_fallbacks,
        "dp_value": Iv[C][len(S)] if len(S) <= K else Iv[C][K],
        "seeds_per_community": {m: len(st.S) for m, st in states.items() if st.S},
        "edge_weights_approximate": getattr(ew, "approximate", False),
        "edge_weights_clamped": getattr(ew, "n_clamped", 0),
    }
    return S


# `ctim_select_seeds.last_stats` is part of the API contract (API.md): the
# experiment harness reads it after each call to report where time goes.
ctim_select_seeds.last_stats = {}


def ctim_run(model, ds, item, K, h=0.1, dp_tiebreak="consistent",
             edge_weights=None) -> RunResult:
    """Convenience wrapper: run Algorithm 2 and score it with the common evaluator.

    The spread is Eq (18) on the FULL graph with this item's Eq (12) weights --
    the same evaluator every method is scored with (API.md, experiments), not
    the per-community `I_m` the DP optimises.
    """
    t0 = time.time()
    ew = EdgeWeights(model, ds) if edge_weights is None else edge_weights
    seeds = ctim_select_seeds(
        model, ds, item, K, h=h, dp_tiebreak=dp_tiebreak, edge_weights=ew
    )
    stats = dict(ctim_select_seeds.last_stats)
    seconds = stats.get("total", time.time() - t0)

    t1 = time.time()
    pp = ew.for_item(item)  # Eq (12)
    mia = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)
    spread = mia.influence(seeds)  # Eq (18)
    stats["eval_seconds"] = time.time() - t1
    stats["run_seconds"] = time.time() - t0

    return RunResult(seeds=seeds, spread=spread, seconds=seconds, extra=stats)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import sys

    from ctim.dataset import (Dataset, build_adjacency,
                              build_potential_influence_logs, split_logs)
    from ctim.gibbs import train_model

    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("  PASS  %s%s" % (name, (" -- " + detail) if detail else ""))
        else:
            print("  FAIL  %s%s" % (name, (" -- " + detail) if detail else ""))
            failures.append(name)

    # ----------------------------------------------------------- synthetic
    def make_synthetic(rng, n_comm=4, per_comm=15, n_items=24, n_attrs=12,
                       z_true=3, p_in=0.30, p_out=0.02):
        """A planted-community dataset: dense intra-community arcs, sparse
        inter-community arcs, and item adoptions that cascade inside one
        community so that potential-influence logs land on intra-community
        arcs."""
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

        per_topic = n_attrs // z_true
        item_attrs = []
        for i in range(n_items):
            z = i % z_true
            block = list(range(z * per_topic, (z + 1) * per_topic))
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
        ds = Dataset(name="synthetic", n_users=n_users, n_items=n_items,
                     n_attrs=n_attrs, edges=edges, out_adj=out_adj,
                     in_adj=in_adj, logs=logs, item_attrs=item_attrs,
                     source="ctim.ctim self-test")
        return ds, true_comm

    # ------------------------------------------------------------- test 1
    print("[1] Eq (19): detect_communities")
    pi_hand = [
        [0.1, 0.7, 0.2],
        [0.5, 0.5, 0.0],   # tie -> lowest index
        [0.0, 0.0, 1.0],
        [0.34, 0.33, 0.33],
    ]
    got = detect_communities(pi_hand)
    check("argmax_c pi[v][c] with ties -> lowest index",
          got == [1, 0, 2, 0], "got %s" % (got,))

    # ------------------------------------------------------------- test 2
    print("[2] training a tiny model end to end")
    rng = random.Random(20190408)
    ds, true_comm = make_synthetic(rng)
    logs_d = build_potential_influence_logs(ds, delta=100)
    train, valid, test = split_logs(logs_d, rng)
    print("      U=%d  E=%d  M=%d  F=%d  |D|=%d  (train %d / valid %d / test %d)"
          % (ds.n_users, ds.n_links, ds.n_items, ds.n_attrs, len(logs_d),
             len(train), len(valid), len(test)))
    check("potential-influence logs were built", len(train) > 50,
          "|train| = %d" % len(train))

    C, Z = 4, 3
    model = train_model(ds, train, C=C, Z=Z, n_iter_topic=60, n_iter_comm=60,
                        rng=random.Random(7))
    check("model shapes", (len(model.pi) == ds.n_users and len(model.eta) == C
                           and len(model.theta) == C
                           and len(model.p_z_given_i) == ds.n_items))

    comm = detect_communities(model.pi)  # Eq (19)
    sizes = {}
    for v in range(ds.n_users):
        sizes[comm[v]] = sizes.get(comm[v], 0) + 1
    print("      detected community sizes: %s" % (sorted(sizes.items()),))
    check("more than one community is populated", len(sizes) > 1)

    # ------------------------------------------------------------- test 3
    print("[3] end-to-end: select K=5 seeds")
    K = 5
    item = 0
    ew = EdgeWeights(model, ds)
    res = ctim_run(model, ds, item, K, h=0.1, edge_weights=ew)
    st = res.extra
    print("      seeds  = %s" % (res.seeds,))
    print("      I(S)   = %.6f   (on the full graph, item %d)" % (res.spread, item))
    print("      per-community seeds: %s" % (st["seeds_per_community"],))
    print("      timings: weights=%.4fs detect=%.4fs subgraphs=%.4fs "
          "dp=%.4fs select=%.4fs total=%.4fs"
          % (st["weights"], st["detect"], st["subgraphs"], st["dp"],
             st["select"], st["total"]))
    check("K seeds returned", len(res.seeds) == K, "got %d" % len(res.seeds))
    check("seeds are distinct", len(set(res.seeds)) == len(res.seeds))
    check("seeds are valid node ids",
          all(0 <= u < ds.n_users for u in res.seeds))
    check("spread exceeds the trivial |S| (the seeds actually propagate)",
          res.spread > K, "I(S)=%.4f vs K=%d" % (res.spread, K))
    check("last_stats carries every required phase key",
          all(key in st for key in ("weights", "detect", "dp", "select")),
          "keys=%s" % sorted(k for k in st if isinstance(st[k], float)))
    phase_sum = st["weights"] + st["detect"] + st["subgraphs"] + st["dp"] + st["select"]
    check("phase timings are disjoint and sum to total",
          abs(phase_sum - st["total"]) <= 0.05 * max(st["total"], 1e-6) + 1e-3,
          "sum=%.6f total=%.6f" % (phase_sum, st["total"]))
    check("seeds_per_community sums to K",
          sum(st["seeds_per_community"].values()) == K)

    # Every seed must lie in the community whose S_m it joined: the histogram of
    # comm[u]+1 over the returned seeds has to equal seeds_per_community exactly.
    hist = {}
    for u in res.seeds:
        key = comm[u] + 1
        hist[key] = hist.get(key, 0) + 1
    check("every seed lies in the community it was charged to",
          hist == st["seeds_per_community"],
          "hist=%s vs stats=%s" % (sorted(hist.items()),
                                   sorted(st["seeds_per_community"].items())))

    pp_full = ew.for_item(item)
    seeds2 = ctim_select_seeds(model, ds, item, K, h=0.1, edge_weights=ew)
    check("selection is deterministic across identical calls",
          seeds2 == res.seeds, "%s vs %s" % (seeds2, res.seeds))

    # ------------------------------------------------------------- test 4
    print("[4] incremental IncInf == exact MIA.marginal_gain")
    biggest = max(sizes, key=lambda c: (sizes[c], -c))
    members = [v for v in range(ds.n_users) if comm[v] == biggest]
    mia_m = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp_full, h=0.1, nodes=members)
    state = _CommunitySeedState(biggest + 1, mia_m, members)
    worst = 0.0
    for step in range(4):
        for u in members:
            if u in state.S_set:
                continue
            ref = mia_m.marginal_gain(state.S_set, u)  # Eq (17)/(18)
            worst = max(worst, abs(state.inc[u] - ref))
        u_best, g = state.best()
        if u_best is None:
            break
        # the heap's argmax must be the true argmax of line 34
        true_best = None
        true_g = -1.0
        for u in members:
            if u in state.S_set:
                continue
            gu = mia_m.marginal_gain(state.S_set, u)
            if gu > true_g + 1e-12:
                true_g, true_best = gu, u
        check("step %d: heap argmax == brute-force argmax (gain %.6f)" % (step, g),
              abs(g - true_g) <= 1e-9, "%s(%.9f) vs %s(%.9f)"
              % (u_best, g, true_best, true_g))
        state.add(u_best)
    check("IncInf table stays exact under incremental updates",
          worst <= 1e-9, "max abs err = %.3e" % worst)

    # ------------------------------------------------------------- test 5
    print("[5] cached CTIM == literal recompute-everything CTIM")

    def reference_ctim(model, ds, item, K, h, dp_tiebreak, ew):
        """Algorithm 2, lines 25-45, transcribed with NO caching at all: every
        dI_m is recomputed from scratch with MIA.marginal_gain."""
        pp = ew.for_item(item)
        comm = detect_communities(model.pi)             # lines 22-24, Eq (19)
        Cn = len(model.eta)
        mem = [[] for _ in range(Cn + 1)]
        for v in range(ds.n_users):
            mem[comm[v] + 1].append(v)
        mias = {}
        for m in range(1, Cn + 1):
            if mem[m]:
                mias[m] = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h,
                              nodes=mem[m])
        S, Sm = [], {m: set() for m in range(1, Cn + 1)}     # line 25
        Iv = [[0.0] * (K + 1) for _ in range(Cn + 1)]        # lines 26-31
        sp = [[0] * (K + 1) for _ in range(Cn + 1)]
        for k in range(1, K + 1):                            # line 32
            argmax_u = {}
            for m in range(1, Cn + 1):                       # line 33
                dI = 0.0
                bu = None
                if m in mias:
                    for u in mem[m]:                         # line 34
                        if u in Sm[m]:
                            continue
                        g = mias[m].marginal_gain(Sm[m], u)
                        if bu is None or g > dI + 1e-12:
                            dI, bu = g, u
                    if bu is None:
                        dI = 0.0
                argmax_u[m] = bu
                cand = Iv[m][k - 1] + dI
                prev = Iv[m - 1][k]
                Iv[m][k] = prev if prev > cand else cand     # line 35
                ref = Iv[Cn][k - 1] if dp_tiebreak == "paper-literal" else Iv[m][k - 1]
                sp[m][k] = m if ref + dI >= prev else sp[m - 1][k]  # lines 36-40
            j = sp[Cn][k]                                    # line 42
            u_k = argmax_u.get(j)
            if u_k is None:
                best_g = None
                for m2 in sorted(mias):
                    if argmax_u.get(m2) is None:
                        continue
                    g2 = mias[m2].marginal_gain(Sm[m2], argmax_u[m2])
                    if best_g is None or g2 > best_g:
                        best_g, u_k, j = g2, argmax_u[m2], m2
                if u_k is None:
                    break
            Sm[j].add(u_k)                                   # line 44
            S.append(u_k)
        return S

    ref_seeds = reference_ctim(model, ds, item, K, 0.1, "consistent", ew)
    same = ref_seeds == res.seeds
    if not same:
        mia_full = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp_full, h=0.1)
        same = abs(mia_full.influence(ref_seeds) - res.spread) <= 1e-9
    check("cached selection matches the literal transcription",
          same, "cached=%s reference=%s" % (res.seeds, ref_seeds))

    # ------------------------------------------------------------- test 6
    print("[6] dp_tiebreak variants and DP-table invariants")
    lit = ctim_run(model, ds, item, K, h=0.1, dp_tiebreak="paper-literal",
                   edge_weights=ew)
    print("      consistent    seeds=%s  I(S)=%.6f" % (res.seeds, res.spread))
    print("      paper-literal seeds=%s  I(S)=%.6f" % (lit.seeds, lit.spread))
    check("paper-literal also returns K seeds", len(lit.seeds) == K)
    check("paper-literal seeds are distinct", len(set(lit.seeds)) == len(lit.seeds))
    try:
        ctim_select_seeds(model, ds, item, K, dp_tiebreak="nonsense")
        check("unknown dp_tiebreak raises ValueError", False)
    except ValueError:
        check("unknown dp_tiebreak raises ValueError", True)
    check("K=0 returns an empty seed set",
          ctim_select_seeds(model, ds, item, 0, edge_weights=ew) == [])

    # ------------------------------------------------------------- test 7
    print("[7] CTIM beats random and monotone seed-set growth")
    mia_full = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp_full, h=0.1)
    rr = random.Random(11)
    rand_best = 0.0
    for _ in range(20):
        rand_best = max(rand_best,
                        mia_full.influence(rr.sample(range(ds.n_users), K)))
    check("CTIM spread beats the best of 20 random seed sets",
          res.spread > rand_best, "CTIM %.4f vs random %.4f"
          % (res.spread, rand_best))

    prev = -1.0
    mono = True
    for kk in range(1, K + 1):
        r = ctim_run(model, ds, item, kk, h=0.1, edge_weights=ew)
        if r.spread < prev - 1e-9:
            mono = False
        prev = r.spread
    check("I(S) is non-decreasing in K", mono, "I(S_%d)=%.4f" % (K, prev))

    # ------------------------------------------------------------- done
    print("")
    if failures:
        print("FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("ALL CHECKS PASSED")
