"""Tests for ``ctim/influence.py`` -- Eq (10)-(18).

Covers:
  * ``EdgeWeights`` reproduces the naive Eq (11)+(12) reference to 1e-9 relative
    (SPEC.md Section 3 states the factorisation is exact, not an approximation)
  * Eq (17) ``ap(v|S)`` on hand-computed 4-node examples: a chain and a
    two-parent join, both worked out in the docstrings
  * ``MIA.influence`` (Eq (17)/(18)) against a large Monte-Carlo Independent
    Cascade run -- exact on an arborescence, close on a sparse digraph
  * the MIA threshold ``h`` (Eq (15)/(16)) really truncates: a path whose
    ``pp(MIP(u,v)) < h`` contributes nothing
  * ``MIA.influence`` is monotone and submodular (spot-checked on triples of a
    random instance)
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctim_testutil import (FakeDataset, dirichlet_row,  # noqa: E402
                           make_random_digraph, make_random_model,
                           monte_carlo_ic)

from ctim.influence import (MIA, EdgeWeights,  # noqa: E402
                            community_to_community, user_to_user_item,
                            user_to_user_topic)


# ---------------------------------------------------------------------------
# Eq (10)-(12)
# ---------------------------------------------------------------------------


class TestInfluenceStrength(unittest.TestCase):
    """Section 4.2.1: Eq (10), Eq (11), Eq (12) and their fast factorisation."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(20190408)
        cls.U, cls.C, cls.Z, cls.M = 40, 7, 5, 6
        cls.model = make_random_model(rng, cls.U, cls.C, cls.Z, cls.M)

        eset = set()
        while len(eset) < 220:
            a = rng.randrange(cls.U)
            b = rng.randrange(cls.U)
            if a != b:
                eset.add((a, b))
        cls.edges = sorted(eset)
        cls.ds = FakeDataset(cls.edges, n_users=cls.U)

    def test_eq10_is_eta_times_theta(self):
        P = community_to_community(self.model)
        self.assertEqual(len(P), self.C)
        for cp in range(self.C):
            self.assertEqual(len(P[cp]), self.C)
            for c in range(self.C):
                self.assertEqual(len(P[cp][c]), self.Z)
                for z in range(self.Z):
                    # Eq (10): P(c | z, c') = eta[c'][c] * theta[c][z]
                    self.assertEqual(
                        P[cp][c][z],
                        self.model.eta[cp][c] * self.model.theta[c][z],
                        "P[%d][%d][%d]" % (cp, c, z))

    def test_eq11_matches_a_direct_sum_over_the_eq10_table(self):
        P = community_to_community(self.model)
        pi = self.model.pi
        for (u, v) in self.edges[:20]:
            for z in range(self.Z):
                # Eq (11): P(v|z,u) = sum_{c,c'} pi[v][c] pi[u][c'] P(c|z,c')
                ref = 0.0
                for cp in range(self.C):
                    for c in range(self.C):
                        ref += pi[v][c] * pi[u][cp] * P[cp][c][z]
                self.assertAlmostEqual(
                    user_to_user_topic(self.model, u, v, z), ref, places=12,
                    msg="(u,v,z)=(%d,%d,%d)" % (u, v, z))

    def test_eq12_is_the_topic_mixture_of_eq11(self):
        for (u, v) in self.edges[:20]:
            for i in range(self.M):
                # Eq (12): P(v|i,u) = sum_z P(z|i) * P(v|z,u)
                ref = sum(self.model.p_z_given_i[i][z]
                          * user_to_user_topic(self.model, u, v, z)
                          for z in range(self.Z))
                self.assertAlmostEqual(
                    user_to_user_item(self.model, u, v, i), ref, places=12)

    def test_edge_weights_equal_the_naive_reference_to_1e_9(self):
        """SPEC.md Section 3: the factorisation is exact, not an approximation."""
        ew = EdgeWeights(self.model, self.ds)
        worst = 0.0
        worst_at = None
        for i in range(self.M):
            fast = ew.for_item(i)
            self.assertEqual(len(fast), len(self.edges))
            for (u, v) in self.edges:
                ref = user_to_user_item(self.model, u, v, i)  # Eq (11)+(12)
                got = fast[(u, v)]
                denom = abs(ref) if abs(ref) > 0.0 else 1.0
                rel = abs(got - ref) / denom
                if rel > worst:
                    worst, worst_at = rel, (u, v, i)
        self.assertLessEqual(worst, 1e-9,
                             "max relative error %.3e at (u,v,i)=%s"
                             % (worst, worst_at))

    def test_edge_weights_pp_helper_agrees_with_for_item(self):
        ew = EdgeWeights(self.model, self.ds)
        table = ew.for_item(3)
        for (u, v) in self.edges[:50]:
            self.assertAlmostEqual(ew.pp(u, v, 3), table[(u, v)], places=15)

    def test_no_clamping_for_a_well_formed_model(self):
        ew = EdgeWeights(self.model, self.ds)
        for i in range(self.M):
            ew.for_item(i)
        self.assertEqual(ew.n_clamped, 0,
                         "max raw pp = %.9f" % ew.max_raw_pp)
        self.assertLessEqual(ew.max_raw_pp, 1.0)
        self.assertFalse(ew.approximate)

    def test_pp_is_bounded_by_the_largest_theta_entry(self):
        """P(v|i,u) <= max_{c,z} theta[c][z].

        Proof, straight from the factorisation of SPEC.md Section 3:

            P(v|i,u) = sum_c a_u[c] * pi_v[c] * thetabar_i[c]

        with a_u[c] = sum_c' pi_u[c'] eta[c'][c] <= 1 (pi_u is a distribution
        and eta in [0,1]) and thetabar_i[c] = sum_z P(z|i) theta[c][z] <=
        max_z theta[c][z].  Since pi_v is a distribution the whole sum is a
        convex combination bounded by max_c thetabar_i[c].

        CONSEQUENCE, and the reason this test exists: when theta is diffuse
        (theta[c][z] ~ 1/Z, which is what alpha = 50/Z produces unless the
        community-topic counts are large), EVERY edge weight is <= ~1/Z.  At
        the paper's Z = 8 that caps pp at ~0.125 and at Z >= 10 it falls below
        the paper's own MIA threshold h = 0.1 -- at which point Eq (15) admits
        no path at all, every MIIA is a singleton and Eq (18) degenerates to
        I(S) = |S| for every method.  See DEVIATIONS.md section "Eq (12) scale
        vs h = 0.1".
        """
        ew = EdgeWeights(self.model, self.ds)
        bound = max(max(row) for row in self.model.theta)
        for i in range(self.M):
            for p in ew.for_item(i).values():
                self.assertLessEqual(p, bound + 1e-12)

        # ... and the bound really does collapse as Z grows with a diffuse
        # theta: a uniform theta over Z topics caps every pp at exactly 1/Z.
        rng = random.Random(5)
        for Z in (2, 8, 16):
            model = make_random_model(rng, self.U, self.C, Z, 2)
            model.theta = [[1.0 / Z] * Z for _ in range(self.C)]
            model.eta = [[1.0] * self.C for _ in range(self.C)]  # the max case
            ew_z = EdgeWeights(model, self.ds)
            top = max(ew_z.for_item(0).values())
            self.assertLessEqual(top, 1.0 / Z + 1e-12, "Z=%d" % Z)
            self.assertAlmostEqual(top, 1.0 / Z, places=9,
                                   msg="Z=%d: eta==1 should saturate the bound" % Z)

    def test_top_c_truncation_is_flagged_and_only_shrinks_weights(self):
        exact = EdgeWeights(self.model, self.ds)
        approx = EdgeWeights(self.model, self.ds, top_c=2)
        self.assertFalse(exact.approximate)
        self.assertTrue(approx.approximate)
        a = exact.for_item(0)
        b = approx.for_item(0)
        for e in a:
            self.assertLessEqual(b[e], a[e] + 1e-15, e)


# ---------------------------------------------------------------------------
# Eq (13)-(18)
# ---------------------------------------------------------------------------


class TestEq17HandComputed(unittest.TestCase):
    """Eq (17) on 4-node examples whose answers are computed by hand below."""

    def test_chain_of_four(self):
        """Chain 0 -> 1 -> 2 -> 3 with pp = 0.5, 0.4, 0.6 and h = 0.1.

        pp(MIP(0,1)) = 0.5, pp(MIP(0,2)) = 0.20, pp(MIP(0,3)) = 0.12, all >= h,
        so MIIA(3,h) is the whole chain.  Eq (17) with S = {0}:

            ap(0|S) = 1                              (0 in S)
            ap(1|S) = 1 - (1 - 1.00*0.5) = 0.50
            ap(2|S) = 1 - (1 - 0.50*0.4) = 0.20
            ap(3|S) = 1 - (1 - 0.20*0.6) = 0.12

        Eq (18): I({0}) = 1 + 0.50 + 0.20 + 0.12 = 1.82
        """
        out_adj = [[1], [2], [3], []]
        in_adj = [[], [0], [1], [2]]
        pp = {(0, 1): 0.5, (1, 2): 0.4, (2, 3): 0.6}
        mia = MIA(4, out_adj, in_adj, pp, h=0.1)

        for node, expect in ((0, 1.0), (1, 0.5), (2, 0.20), (3, 0.12)):
            self.assertAlmostEqual(mia.ap(node, {0}), expect, places=12,
                                   msg="ap(%d|{0})" % node)
        self.assertAlmostEqual(mia.influence({0}), 1.82, places=12)

        # a node with no in-neighbours inside its MIIA has ap == 0 (Eq (17))
        self.assertEqual(mia.ap(0, {3}), 0.0)
        self.assertAlmostEqual(mia.influence({3}), 1.0, places=12)

    def test_uniform_chain(self):
        """Same chain with p = 0.5 throughout: I({0}) = 1 + .5 + .25 + .125."""
        out_adj = [[1], [2], [3], []]
        in_adj = [[], [0], [1], [2]]
        pp = {(0, 1): 0.5, (1, 2): 0.5, (2, 3): 0.5}
        mia = MIA(4, out_adj, in_adj, pp, h=0.1)
        self.assertAlmostEqual(mia.influence({0}), 1.875, places=12)

    def test_two_parent_join(self):
        """4 nodes: 0 -> 2 (0.5), 1 -> 2 (0.4), 2 -> 3 (0.5); h = 0.1.

        MIIA(2,h) = {0,1,2} (both in-paths have pp >= h), so node 2 has TWO
        in-neighbours and Eq (17) must take the product over both:

            ap(2|{0,1}) = 1 - (1 - 1*0.5)(1 - 1*0.4) = 1 - 0.5*0.6 = 0.70
            ap(3|{0,1}) = 1 - (1 - 0.70*0.5)          = 0.35
            I({0,1})    = 1 + 1 + 0.70 + 0.35 = 3.05

        and with only S = {0}:

            ap(1) = 0,  ap(2) = 1 - (1 - 1*0.5)(1 - 0*0.4) = 0.50
            ap(3) = 1 - (1 - 0.50*0.5) = 0.25
            I({0}) = 1 + 0 + 0.50 + 0.25 = 1.75
        """
        out_adj = [[2], [2], [3], []]
        in_adj = [[], [], [0, 1], [2]]
        pp = {(0, 2): 0.5, (1, 2): 0.4, (2, 3): 0.5}
        mia = MIA(4, out_adj, in_adj, pp, h=0.1)

        self.assertEqual(mia.miia(2)[0], [0, 1, 2])
        self.assertAlmostEqual(mia.ap(2, {0, 1}), 0.70, places=12)
        self.assertAlmostEqual(mia.ap(3, {0, 1}), 0.35, places=12)
        self.assertAlmostEqual(mia.influence({0, 1}), 3.05, places=12)

        self.assertAlmostEqual(mia.ap(2, {0}), 0.50, places=12)
        self.assertAlmostEqual(mia.ap(3, {0}), 0.25, places=12)
        self.assertAlmostEqual(mia.influence({0}), 1.75, places=12)

    def test_marginal_gain_matches_a_full_recomputation(self):
        out_adj = [[1], [2], [3], []]
        in_adj = [[], [0], [1], [2]]
        pp = {(0, 1): 0.5, (1, 2): 0.4, (2, 3): 0.6}
        mia = MIA(4, out_adj, in_adj, pp, h=0.1)
        for u in range(4):
            got = mia.marginal_gain({0}, u)
            want = mia.influence({0, u}) - mia.influence({0})
            self.assertAlmostEqual(got, want, places=12, msg="u=%d" % u)


class TestMiaThreshold(unittest.TestCase):
    """Eq (15)/(16): a path with pp(MIP(u,v)) < h contributes nothing."""

    def setUp(self):
        # 0 -> 1 (0.5) -> 2 (0.15).  pp(MIP(0,2)) = 0.075 < h = 0.1.
        self.out_adj = [[1], [2], []]
        self.in_adj = [[], [0], [1]]
        self.pp = {(0, 1): 0.5, (1, 2): 0.15}

    def test_below_threshold_path_is_dropped(self):
        mia = MIA(3, self.out_adj, self.in_adj, self.pp, h=0.1)
        # MIIA(2, 0.1) keeps 1 (pp 0.15 >= 0.1) but must drop 0 (pp 0.075).
        self.assertEqual(mia.miia(2)[0], [1, 2])
        self.assertNotIn(0, mia.miia(2)[0])
        # ... so seeding 0 cannot activate 2 at all.
        self.assertEqual(mia.ap(2, {0}), 0.0)
        # Eq (18): I({0}) = ap(0) + ap(1) + ap(2) = 1 + 0.5 + 0 = 1.5
        self.assertAlmostEqual(mia.influence({0}), 1.5, places=12)
        # symmetric statement on the out-arborescence, Eq (16)
        self.assertEqual(mia.mioa(0)[0], [0, 1])

    def test_lowering_h_restores_the_path(self):
        mia = MIA(3, self.out_adj, self.in_adj, self.pp, h=0.01)
        self.assertEqual(mia.miia(2)[0], [0, 1, 2])
        self.assertAlmostEqual(mia.ap(2, {0}), 0.075, places=12)
        # I({0}) = 1 + 0.5 + 0.075
        self.assertAlmostEqual(mia.influence({0}), 1.575, places=12)
        self.assertEqual(mia.mioa(0)[0], [0, 1, 2])

    def test_paper_default_h_is_used_when_unspecified(self):
        mia = MIA(3, self.out_adj, self.in_adj, self.pp)
        self.assertEqual(mia.h, 0.1)  # SPEC.md Section 4: "The paper sets h = 0.1"
        self.assertEqual(mia.ap(2, {0}), 0.0)

    def test_threshold_truncation_on_the_four_node_chain(self):
        # Chain 0->1 (0.5), 1->2 (0.4), 2->3 (0.6) with h = 0.3:
        #   MIIA(3): keeps 2 (0.6); drops 1 (0.24) and 0 (0.144)   -> {2,3}
        #   MIIA(2): keeps 1 (0.4); drops 0 (0.20)                 -> {1,2}
        # so I({0}) = ap(0) + ap(1) = 1 + 0.5 = 1.5
        out_adj = [[1], [2], [3], []]
        in_adj = [[], [0], [1], [2]]
        pp = {(0, 1): 0.5, (1, 2): 0.4, (2, 3): 0.6}
        mia = MIA(4, out_adj, in_adj, pp, h=0.3)
        self.assertEqual(mia.miia(3)[0], [2, 3])
        self.assertEqual(mia.miia(2)[0], [1, 2])
        self.assertAlmostEqual(mia.influence({0}), 1.5, places=12)


class TestMiaVsMonteCarlo(unittest.TestCase):
    """Eq (17)/(18) against the Independent Cascade process MIA approximates."""

    N_MC = 60000

    def test_exact_on_an_arborescence(self):
        """On an out-tree every node has a unique in-path, so MIA is EXACT."""
        rng = random.Random(4242)
        n = 18
        out_adj = [[] for _ in range(n)]
        in_adj = [[] for _ in range(n)]
        pp = {}
        for v in range(1, n):
            u = rng.randrange(0, v)
            out_adj[u].append(v)
            in_adj[v].append(u)
            pp[(u, v)] = 0.25 + 0.5 * rng.random()
        mia = MIA(n, out_adj, in_adj, pp, h=1e-12)
        exact = mia.influence({0})
        mc = monte_carlo_ic(out_adj, pp, {0}, self.N_MC, random.Random(12345))
        rel = abs(exact - mc) / mc
        self.assertLessEqual(rel, 0.015,
                             "MIA %.4f vs MC(%d) %.4f -- %.3f%%"
                             % (exact, self.N_MC, mc, 100 * rel))

    def test_close_on_a_sparse_digraph(self):
        """MIA keeps only the maximum influence path, so it under-counts nodes
        fed by several comparable paths.  With small edge probabilities those
        multi-path corrections are second order, so the two agree to a few
        percent -- and MIA must be the LOWER of the two."""
        rng = random.Random(999)
        n = 25
        out_adj, in_adj, pp = make_random_digraph(rng, n, 60, p_lo=0.05,
                                                  p_hi=0.30)
        mia = MIA(n, out_adj, in_adj, pp, h=1e-12)
        S = {0, 1}
        exact = mia.influence(S)
        mc = monte_carlo_ic(out_adj, pp, S, self.N_MC, random.Random(777))
        rel = abs(exact - mc) / mc
        self.assertLessEqual(rel, 0.03,
                             "MIA %.4f vs MC(%d) %.4f -- %.3f%%"
                             % (exact, self.N_MC, mc, 100 * rel))
        self.assertLessEqual(exact, mc * 1.01,
                             "MIA should not exceed the true IC spread")

    def test_seed_only_graph_gives_exactly_k(self):
        out_adj = [[] for _ in range(5)]
        in_adj = [[] for _ in range(5)]
        mia = MIA(5, out_adj, in_adj, {}, h=0.1)
        self.assertAlmostEqual(mia.influence({0, 2, 4}), 3.0, places=12)
        self.assertEqual(mia.influence(set()), 0.0)


class TestMiaMonotoneSubmodular(unittest.TestCase):
    """``I(S)`` from Eq (17)/(18) is monotone and submodular (Chen et al. [19])."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(31337)
        cls.n = 30
        cls.out_adj, cls.in_adj, cls.pp = make_random_digraph(
            rng, cls.n, 90, p_lo=0.05, p_hi=0.55)
        cls.mia = MIA(cls.n, cls.out_adj, cls.in_adj, cls.pp, h=0.1)
        cls.rng = random.Random(2718)

    def test_monotone_under_adding_any_node(self):
        rng = random.Random(11)
        for _ in range(40):
            S = set(rng.sample(range(self.n), rng.randrange(0, 5)))
            base = self.mia.influence(S)
            for u in rng.sample(range(self.n), 6):
                self.assertGreaterEqual(
                    self.mia.influence(S | {u}), base - 1e-12,
                    "I(S u {%d}) < I(S) for S=%s" % (u, sorted(S)))

    def test_influence_is_at_least_the_seed_count(self):
        rng = random.Random(12)
        for _ in range(20):
            S = set(rng.sample(range(self.n), 4))
            self.assertGreaterEqual(self.mia.influence(S), len(S) - 1e-12)

    def test_submodular_on_triples(self):
        """For S subset of T and u not in T:

            I(S u {u}) - I(S)  >=  I(T u {u}) - I(T)

        Spot-checked on random (S, T, u) triples with S = T minus one element,
        which is the tight case: if diminishing returns hold for every
        single-element extension it holds for every nested pair by telescoping.
        """
        rng = random.Random(13)
        worst = 0.0
        worst_at = None
        for _ in range(120):
            T = set(rng.sample(range(self.n), rng.randrange(2, 6)))
            drop = rng.choice(sorted(T))
            S = T - {drop}
            choices = [x for x in range(self.n) if x not in T]
            u = rng.choice(choices)
            gain_small = self.mia.influence(S | {u}) - self.mia.influence(S)
            gain_big = self.mia.influence(T | {u}) - self.mia.influence(T)
            violation = gain_big - gain_small
            if violation > worst:
                worst, worst_at = violation, (sorted(S), sorted(T), u)
        self.assertLessEqual(worst, 1e-9,
                             "submodularity violated by %.3e at %s"
                             % (worst, worst_at))

    def test_submodular_on_all_small_subsets(self):
        """Exhaustive check on every S subset T subset of a fixed 5-node pool."""
        pool = [0, 1, 2, 3, 4]
        cache = {}

        def inf(fs):
            if fs not in cache:
                cache[fs] = self.mia.influence(set(fs))
            return cache[fs]

        for r in range(0, 4):
            for T in itertools.combinations(pool, r + 1):
                Ts = frozenset(T)
                for drop in T:
                    Ss = Ts - {drop}
                    for u in range(self.n):
                        if u in Ts:
                            continue
                        gs = inf(Ss | {u}) - inf(Ss)
                        gt = inf(Ts | {u}) - inf(Ts)
                        self.assertLessEqual(
                            gt, gs + 1e-9,
                            "S=%s T=%s u=%d: %.12f > %.12f"
                            % (sorted(Ss), sorted(Ts), u, gt, gs))

    def test_marginal_gain_is_exact_everywhere(self):
        rng = random.Random(14)
        S = sorted(rng.sample(range(self.n), 3))
        base = self.mia.influence(S)
        for u in range(self.n):
            got = self.mia.marginal_gain(S, u)
            want = self.mia.influence(set(S) | {u}) - base
            self.assertAlmostEqual(got, want, places=9, msg="u=%d" % u)

    def test_greedy_incremental_matches_brute_force(self):
        def brute(K, cands):
            S = []
            for _ in range(K):
                best, best_g = None, -1.0
                for u in sorted(cands):
                    if u in S:
                        continue
                    g = self.mia.influence(S + [u]) - self.mia.influence(S)
                    if g > best_g + 1e-12:
                        best, best_g = u, g
                if best is None or best_g <= 0.0:
                    break
                S.append(best)
            return S

        self.assertEqual(self.mia.greedy_incremental(5), brute(5, range(self.n)))
        cands = sorted(random.Random(15).sample(range(self.n), 12))
        got = self.mia.greedy_incremental(4, candidates=cands)
        self.assertEqual(got, brute(4, cands))
        self.assertLessEqual(set(got), set(cands))


class TestMiaInducedSubgraph(unittest.TestCase):
    """``MIA`` must be constructible on a node-induced subgraph (Alg. 2 line 34)."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(5150)
        cls.n = 30
        cls.out_adj, cls.in_adj, cls.pp = make_random_digraph(
            rng, cls.n, 90, p_lo=0.05, p_hi=0.55)
        cls.full = MIA(cls.n, cls.out_adj, cls.in_adj, cls.pp, h=0.1)
        cls.sub_nodes = sorted(rng.sample(range(cls.n), 14))
        cls.sub = MIA(cls.n, cls.out_adj, cls.in_adj, cls.pp, h=0.1,
                      nodes=cls.sub_nodes)

    def test_nodes_are_exactly_the_requested_subset(self):
        self.assertEqual(self.sub.nodes, self.sub_nodes)

    def test_arborescences_stay_inside_the_subset(self):
        allowed = set(self.sub_nodes)
        for v in self.sub_nodes:
            self.assertLessEqual(set(self.sub.miia(v)[0]), allowed)
            self.assertLessEqual(set(self.sub.mioa(v)[0]), allowed)

    def test_excluded_nodes_have_zero_ap(self):
        outside = [x for x in range(self.n) if x not in set(self.sub_nodes)]
        self.assertTrue(outside)
        for v in outside:
            self.assertEqual(self.sub.ap(v, {self.sub_nodes[0]}), 0.0)

    def test_subgraph_spread_never_exceeds_the_full_graph(self):
        # An induced subgraph can only lose paths, so I_m(S) <= I(S).
        for v in self.sub_nodes[:6]:
            self.assertLessEqual(self.sub.influence({v}),
                                 self.full.influence({v}) + 1e-12)

    def test_greedy_on_the_subgraph_only_selects_subgraph_nodes(self):
        S = self.sub.greedy_incremental(3)
        self.assertLessEqual(set(S), set(self.sub_nodes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
