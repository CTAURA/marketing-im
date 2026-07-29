"""Tests for ``ctim/gibbs.py`` -- Algorithm 1 and the collapsed Gibbs sampler.

Covers:
  * ``Hyper.make``: rho = 50/C, beta = 0.01, alpha = 50/Z, omega = 50/Z,
    eps1 = 0.1, N_neg = U*(U-1)*(1 + D/E) - D - E and eps0 = zeta*ln(N_neg/C^2)
    (SPEC.md Section 4.1.1 / Section 8)
  * stage 1 (:class:`TopicSampler`, Eq (1)-(3)) recovers a planted 2-topic corpus
  * stage 2 (:class:`CommunitySampler`, Eq (4)-(9)) recovers a planted 2-block
    graph, with both the "exact" and the "mh" sampler
  * the estimator outputs are valid: pi (Eq 7) and theta (Eq 9) rows sum to 1,
    eta (Eq 8) lies in [0, 1]

Every fixture is seeded, so these are deterministic pass/fail tests, not flaky
statistical ones.
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctim_testutil import (adjusted_rand, argmax_labels,  # noqa: E402
                           best_label_agreement, make_planted_block_graph,
                           make_planted_topic_corpus)

from ctim.gibbs import (EPS0_FLOOR, CommunitySampler, Hyper,  # noqa: E402
                        Model, TopicSampler, train_model)


def expected_n_neg(U, E, D):
    """SPEC.md Section 4.1.1: N_neg = U*(U-1)*(1 + D/E) - D - E."""
    return float(U) * float(U - 1) * (1.0 + float(D) / float(E)) - float(D) - float(E)


class TestHyper(unittest.TestCase):
    """SPEC.md Section 8 'Hyperparameters (fixed)' + Section 4.1.1."""

    # A configuration comfortably inside the regime where the paper's formula
    # produces a strictly positive eps0 (Digg-sized: U=30358, E=99846).
    U, E, D = 30358, 99846, 50000

    def test_fixed_hyperparameters(self):
        for C, Z in ((100, 8), (25, 2), (150, 16), (1, 1)):
            h = Hyper.make(C, Z, self.U, self.E, self.D)
            self.assertEqual(h.C, C)
            self.assertEqual(h.Z, Z)
            self.assertAlmostEqual(h.rho, 50.0 / C, places=12)      # rho = 50/C
            self.assertAlmostEqual(h.beta, 0.01, places=12)         # beta = 0.01
            self.assertAlmostEqual(h.alpha, 50.0 / Z, places=12)    # alpha = 50/Z
            self.assertAlmostEqual(h.omega, 50.0 / Z, places=12)    # omega = 50/Z
            self.assertAlmostEqual(h.eps1, 0.1, places=12)          # eps1 = 0.1

    def test_eps0_matches_the_n_neg_formula(self):
        C, Z = 100, 8
        h = Hyper.make(C, Z, self.U, self.E, self.D)
        n_neg = expected_n_neg(self.U, self.E, self.D)
        want = math.log(n_neg / (float(C) * float(C)))
        self.assertAlmostEqual(h.eps0, want, places=12)
        self.assertGreater(h.eps0, 0.0)
        self.assertEqual(h.warning, "", h.warning)

    def test_zeta_scales_eps0_linearly(self):
        C, Z = 100, 8
        base = Hyper.make(C, Z, self.U, self.E, self.D, zeta=1.0)
        for zeta in (0.5, 2.0, 3.25):
            h = Hyper.make(C, Z, self.U, self.E, self.D, zeta=zeta)
            self.assertAlmostEqual(h.eps0, zeta * base.eps0, places=10)
            self.assertAlmostEqual(h.zeta, zeta, places=12)

    def test_eps0_depends_on_c_through_c_squared(self):
        n_neg = expected_n_neg(self.U, self.E, self.D)
        for C in (25, 50, 100, 150):
            h = Hyper.make(C, 8, self.U, self.E, self.D)
            self.assertAlmostEqual(h.eps0, math.log(n_neg / (float(C) ** 2)),
                                   places=12)

    def test_degenerate_configurations_are_clamped_and_warned(self):
        # C^2 >= N_neg  =>  ln(N_neg/C^2) <= 0, which is not a usable Beta
        # pseudo-count.  The implementation clamps to a floor and says so.
        h = Hyper.make(C=50, Z=4, n_users=5, n_links=4, n_logs=2)
        self.assertEqual(h.eps0, EPS0_FLOOR)
        self.assertNotEqual(h.warning, "")

        # zeta = 0 kills eps0 the same way.
        hz = Hyper.make(C=2, Z=2, n_users=100, n_links=200, n_logs=200, zeta=0.0)
        self.assertEqual(hz.eps0, EPS0_FLOOR)
        self.assertNotEqual(hz.warning, "")

        # E == 0 makes the D/E inflation factor undefined.
        he = Hyper.make(C=2, Z=2, n_users=100, n_links=0, n_logs=0)
        self.assertNotEqual(he.warning, "")
        self.assertGreater(he.eps0, 0.0)

    def test_invalid_c_or_z_rejected(self):
        with self.assertRaises(ValueError):
            Hyper.make(0, 8, self.U, self.E, self.D)
        with self.assertRaises(ValueError):
            Hyper.make(100, 0, self.U, self.E, self.D)

    def test_dict_round_trip(self):
        h = Hyper.make(100, 8, self.U, self.E, self.D, zeta=1.5)
        h2 = Hyper.from_dict(h.to_dict())
        self.assertEqual(h2.C, h.C)
        self.assertEqual(h2.Z, h.Z)
        for field in ("rho", "alpha", "beta", "omega", "eps0", "eps1", "zeta"):
            self.assertEqual(getattr(h2, field), getattr(h, field), field)


class TestTopicSampler(unittest.TestCase):
    """Stage 1: Eq (1) sampler, Eq (2)/(3) estimators."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(12345)
        cls.n_attrs = 20
        cls.item_attrs, cls.item_topic = make_planted_topic_corpus(
            rng, n_items=16, n_attrs=cls.n_attrs, attr_tokens=120, noise=0.05)
        cls.hyper = Hyper.make(C=2, Z=2, n_users=50, n_links=255, n_logs=6150)
        cls.ts = TopicSampler(cls.item_attrs, cls.n_attrs, cls.hyper,
                              random.Random(7))
        cls.ts.run(80)
        cls.phi = cls.ts.phi()
        cls.psi = cls.ts.psi()

    def test_recovers_the_planted_two_topic_structure(self):
        found = argmax_labels(self.phi)          # Eq (2), then argmax
        ari = adjusted_rand(self.item_topic, found)
        agree = best_label_agreement(self.item_topic, found, n_labels=2)
        self.assertGreater(ari, 0.7, "topic ARI = %.3f, labels %s" % (ari, found))
        self.assertGreaterEqual(agree, 0.9,
                                "agreement %.2f, labels %s" % (agree, found))

    def test_phi_rows_are_distributions(self):
        # Eq (2): phi_iz = (n_iz + omega) / sum_Z (n_iz + omega)
        for i, row in enumerate(self.phi):
            self.assertEqual(len(row), self.hyper.Z)
            self.assertAlmostEqual(sum(row), 1.0, places=12, msg="item %d" % i)
            for p in row:
                self.assertGreater(p, 0.0)   # omega > 0 makes every entry positive
                self.assertLess(p, 1.0)

    def test_psi_rows_are_distributions_over_the_attribute_vocabulary(self):
        # Eq (3): psi_zw = (n_zw + beta) / sum_F (n_zw + beta); the beta
        # normaliser runs over F, the number of distinct attribute values.
        self.assertEqual(len(self.psi), self.hyper.Z)
        for z, row in enumerate(self.psi):
            self.assertEqual(len(row), self.n_attrs)
            self.assertAlmostEqual(sum(row), 1.0, places=12, msg="topic %d" % z)
            for p in row:
                self.assertGreater(p, 0.0)

    def test_psi_separates_the_two_attribute_halves(self):
        # Each planted topic draws from one half of the vocabulary, so each
        # recovered psi row must put most of its mass on one half.
        half = self.n_attrs // 2
        masses = [sum(row[:half]) for row in self.psi]
        self.assertGreater(max(masses), 0.85, masses)
        self.assertLess(min(masses), 0.15, masses)

    def test_p_z_given_i_is_eq_2_but_a_fresh_list(self):
        pz = self.ts.p_z_given_i()
        self.assertEqual(pz, self.phi)
        self.assertIsNot(pz, self.phi)

    def test_deterministic_for_a_fixed_seed(self):
        a = TopicSampler(self.item_attrs, self.n_attrs, self.hyper,
                         random.Random(31))
        a.run(10)
        b = TopicSampler(self.item_attrs, self.n_attrs, self.hyper,
                         random.Random(31))
        b.run(10)
        self.assertEqual(a.phi(), b.phi())

    def test_rejects_out_of_range_attribute_ids(self):
        with self.assertRaises(ValueError):
            TopicSampler([[0, 99]], 5, self.hyper, random.Random(0))


class _PlantedBlockFixture(unittest.TestCase):
    """Shared planted 2-block instance for the CommunitySampler tests."""

    SAMPLER = "exact"
    N_ITER = 60
    MIN_AGREEMENT = 0.85

    @classmethod
    def setUpClass(cls):
        if cls is _PlantedBlockFixture:
            raise unittest.SkipTest("abstract fixture")
        rng = random.Random(12345)
        (cls.U, cls.edges, cls.item_attrs, cls.logs_d,
         cls.labels, cls.item_topic) = make_planted_block_graph(
            rng, n_per_comm=25, n_items=16, n_attrs=20, logs_per_edge=25,
            attr_tokens=120)
        cls.hyper = Hyper.make(C=2, Z=2, n_users=cls.U, n_links=len(cls.edges),
                               n_logs=len(cls.logs_d))
        ts = TopicSampler(cls.item_attrs, 20, cls.hyper, random.Random(7))
        ts.run(80)
        cls.p_z_given_i = ts.p_z_given_i()
        cls.cs = CommunitySampler(cls.U, cls.edges, cls.logs_d, cls.p_z_given_i,
                                  cls.hyper, random.Random(99),
                                  sampler=cls.SAMPLER)
        cls.cs.run(cls.N_ITER)
        cls.pi = cls.cs.pi()
        cls.eta = cls.cs.eta()
        cls.theta = cls.cs.theta()


class TestCommunitySamplerExact(_PlantedBlockFixture):
    """Stage 2 with the exact O(C^2) enumeration of Eq (4)/(6)."""

    SAMPLER = "exact"

    def test_recovers_the_planted_two_block_partition(self):
        found = argmax_labels(self.pi)  # Eq (19) applied to Eq (7)
        ari = adjusted_rand(self.labels, found)
        agree = best_label_agreement(self.labels, found, n_labels=2)
        self.assertGreater(ari, 0.5, "community ARI = %.3f" % ari)
        self.assertGreaterEqual(
            agree, self.MIN_AGREEMENT,
            "%s: %.0f%% of %d users in the right block"
            % (self.SAMPLER, 100 * agree, self.U))

    def test_pi_rows_are_distributions(self):
        # Eq (7): pi_vc = (n_vc + rho) / sum_C (n_vc + rho)
        self.assertEqual(len(self.pi), self.U)
        for v, row in enumerate(self.pi):
            self.assertEqual(len(row), self.hyper.C)
            self.assertAlmostEqual(sum(row), 1.0, places=12, msg="user %d" % v)
            for p in row:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_theta_rows_are_distributions(self):
        # Eq (9): theta_cz = (sum_M n_ci P(z|i) + alpha) / sum_Z (... + alpha)
        self.assertEqual(len(self.theta), self.hyper.C)
        for c, row in enumerate(self.theta):
            self.assertEqual(len(row), self.hyper.Z)
            self.assertAlmostEqual(sum(row), 1.0, places=12,
                                   msg="community %d" % c)
            for p in row:
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_eta_entries_are_probabilities(self):
        # Eq (8): eta_{c'c} = (n_cc + eps1) / (n_cc + eps0 + eps1) in (0,1)
        self.assertEqual(len(self.eta), self.hyper.C)
        for cp, row in enumerate(self.eta):
            self.assertEqual(len(row), self.hyper.C)
            for c, x in enumerate(row):
                self.assertGreaterEqual(x, 0.0, "eta[%d][%d]" % (cp, c))
                self.assertLessEqual(x, 1.0, "eta[%d][%d]" % (cp, c))

    def test_theta_aligns_with_the_planted_topic_of_each_block(self):
        # Community 0 adopts topic-0 items and community 1 topic-1 items, so the
        # two theta rows must prefer *different* topics.
        prefer = argmax_labels(self.theta)
        self.assertNotEqual(prefer[0], prefer[1], self.theta)

    def test_incremental_counts_stay_exact(self):
        # ncz is maintained incrementally in O(Z) per update (SPEC.md Eq (6)
        # implementation note); validate_counts() recomputes it from scratch.
        self.assertEqual(self.cs.validate_counts(), [])

    def test_deterministic_for_a_fixed_seed(self):
        a = CommunitySampler(self.U, self.edges, self.logs_d, self.p_z_given_i,
                             self.hyper, random.Random(5), sampler=self.SAMPLER)
        a.run(5)
        b = CommunitySampler(self.U, self.edges, self.logs_d, self.p_z_given_i,
                             self.hyper, random.Random(5), sampler=self.SAMPLER)
        b.run(5)
        self.assertEqual(a.pi(), b.pi())
        self.assertEqual(a.eta(), b.eta())
        self.assertEqual(a.theta(), b.theta())


class TestCommunitySamplerMH(TestCommunitySamplerExact):
    """Stage 2 with the Metropolis-Hastings kernel (same stationary law)."""

    SAMPLER = "mh"
    MIN_AGREEMENT = 0.80


class TestCommunitySamplerValidation(unittest.TestCase):

    def setUp(self):
        self.hyper = Hyper.make(C=2, Z=2, n_users=10, n_links=8, n_logs=4)
        self.pz = [[0.5, 0.5], [0.5, 0.5]]

    def test_unknown_sampler_rejected(self):
        with self.assertRaises(ValueError):
            CommunitySampler(10, [(0, 1)], [], self.pz, self.hyper,
                             random.Random(0), sampler="nope")

    def test_out_of_range_user_rejected(self):
        with self.assertRaises(ValueError):
            CommunitySampler(3, [(0, 9)], [], self.pz, self.hyper,
                             random.Random(0))

    def test_out_of_range_item_rejected(self):
        with self.assertRaises(ValueError):
            CommunitySampler(3, [(0, 1)], [(0, 1, 7)], self.pz, self.hyper,
                             random.Random(0))

    def test_wrong_p_z_given_i_width_rejected(self):
        with self.assertRaises(ValueError):
            CommunitySampler(3, [(0, 1)], [], [[1.0, 0.0, 0.0]], self.hyper,
                             random.Random(0))


class TestTrainModel(unittest.TestCase):
    """``train_model`` wires both stages together and returns a ``Model``."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(2024)
        (cls.U, cls.edges, cls.item_attrs, cls.logs_d,
         cls.labels, _topic) = make_planted_block_graph(
            rng, n_per_comm=25, n_items=16, n_attrs=20, logs_per_edge=30,
            attr_tokens=100)

        class _DS(object):
            pass

        ds = _DS()
        ds.n_users = cls.U
        ds.n_items = 16
        ds.n_attrs = 20
        ds.edges = cls.edges
        ds.item_attrs = cls.item_attrs
        cls.ds = ds
        cls.model = train_model(ds, cls.logs_d, C=2, Z=2, n_iter_topic=80,
                                n_iter_comm=80, rng=random.Random(2024))

    def test_model_shapes(self):
        m = self.model
        self.assertEqual(len(m.pi), self.U)
        self.assertEqual(len(m.pi[0]), 2)
        self.assertEqual(len(m.eta), 2)
        self.assertEqual(len(m.eta[0]), 2)
        self.assertEqual(len(m.theta), 2)
        self.assertEqual(len(m.theta[0]), 2)
        self.assertEqual(len(m.p_z_given_i), 16)
        self.assertEqual(len(m.psi), 2)
        self.assertEqual(len(m.psi[0]), 20)

    def test_end_to_end_recovers_the_planted_communities(self):
        found = argmax_labels(self.model.pi)
        agree = best_label_agreement(self.labels, found, n_labels=2)
        self.assertGreaterEqual(agree, 0.80,
                                "%.0f%% of %d users" % (100 * agree, self.U))

    def test_estimators_are_valid_distributions(self):
        for row in self.model.pi:
            self.assertAlmostEqual(sum(row), 1.0, places=12)
        for row in self.model.theta:
            self.assertAlmostEqual(sum(row), 1.0, places=12)
        for row in self.model.p_z_given_i:
            self.assertAlmostEqual(sum(row), 1.0, places=12)
        for row in self.model.eta:
            for x in row:
                self.assertGreaterEqual(x, 0.0)
                self.assertLessEqual(x, 1.0)

    def test_json_round_trip(self):
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json", prefix="ctim-model-")
        os.close(fd)
        self.addCleanup(os.remove, path)
        self.model.save(path)
        again = Model.load(path)
        self.assertEqual(again.pi, self.model.pi)
        self.assertEqual(again.eta, self.model.eta)
        self.assertEqual(again.theta, self.model.theta)
        self.assertEqual(again.p_z_given_i, self.model.p_z_given_i)
        self.assertEqual(again.hyper.C, self.model.hyper.C)
        self.assertEqual(again.hyper.eps0, self.model.hyper.eps0)

    def test_deterministic_for_a_fixed_seed(self):
        m2 = train_model(self.ds, self.logs_d, C=2, Z=2, n_iter_topic=80,
                         n_iter_comm=80, rng=random.Random(2024))
        self.assertEqual(m2.pi, self.model.pi)
        self.assertEqual(m2.theta, self.model.theta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
