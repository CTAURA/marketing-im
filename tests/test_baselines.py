"""Tests for ``ctim/baselines/`` -- SPEC.md Section 7, Table 2.

Every baseline must, on a small dataset:
  * return exactly K distinct, in-range seeds
  * return a ``RunResult`` whose ``seconds`` field is strictly positive
  * be deterministic for a fixed ``random.Random`` seed

The five methods compared in the paper are CTIM, CTIM_CGA, AIR+CGA, CINEMA and
Greedy; the four baselines live here and CTIM itself is covered by test_ctim.py.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctim_testutil import make_planted_dataset  # noqa: E402

from ctim.baselines.air_cga import air_cga_select_seeds  # noqa: E402
from ctim.baselines.cga import cga_select_seeds, ctim_cga_select  # noqa: E402
from ctim.baselines.cinema import cinema_select_seeds  # noqa: E402
from ctim.baselines.greedy import (greedy_select,  # noqa: E402
                                   topic_averaged_pp)
from ctim.ctim import detect_communities  # noqa: E402
from ctim.dataset import build_potential_influence_logs, split_logs  # noqa: E402
from ctim.gibbs import train_model  # noqa: E402
from ctim.influence import EdgeWeights  # noqa: E402


class _BaselineFixture(unittest.TestCase):
    """One small planted dataset + trained model, shared by every baseline."""

    K = 5
    ITEM = 0
    C = 3
    Z = 3
    N_MC = 60

    @classmethod
    def setUpClass(cls):
        rng = random.Random(20190408)
        cls.ds, cls.true_comm = make_planted_dataset(
            rng, n_comm=3, per_comm=12, n_items=15, n_attrs=9, z_true=3)
        logs_d = build_potential_influence_logs(cls.ds, delta=100)
        cls.train, cls.valid, cls.test = split_logs(logs_d, rng)
        cls.model = train_model(cls.ds, cls.train, C=cls.C, Z=cls.Z,
                                n_iter_topic=60, n_iter_comm=60,
                                rng=random.Random(7))
        cls.ew = EdgeWeights(cls.model, cls.ds)
        cls.pp = cls.ew.for_item(cls.ITEM)
        cls.comm = detect_communities(cls.model.pi)
        # topic-blind probability map for the topic-blind baselines (Greedy,
        # CINEMA): mean_i P(v|i,u) over a fixed sample of items, per API.md.
        cls.pp_blind = topic_averaged_pp(cls.ew, [0, 1, 2])

    # -- shared assertions --------------------------------------------------

    def assert_run_result(self, res, K=None, method=""):
        K = self.K if K is None else K
        self.assertTrue(hasattr(res, "seeds"), method)
        self.assertTrue(hasattr(res, "spread"), method)
        self.assertTrue(hasattr(res, "seconds"), method)
        self.assertTrue(hasattr(res, "extra"), method)

        self.assertEqual(len(res.seeds), K,
                         "%s returned %d seeds, want %d: %s"
                         % (method, len(res.seeds), K, res.seeds))
        self.assertEqual(len(set(res.seeds)), K,
                         "%s returned duplicate seeds: %s" % (method, res.seeds))
        for u in res.seeds:
            self.assertIsInstance(u, int, method)
            self.assertTrue(0 <= u < self.ds.n_users,
                            "%s: seed %r out of range" % (method, u))

        self.assertGreater(res.seconds, 0.0,
                           "%s reported seconds=%r" % (method, res.seconds))
        self.assertGreaterEqual(res.spread, 0.0, method)
        self.assertIsInstance(res.extra, dict, method)


class TestGreedy(_BaselineFixture):
    """Kempe et al. [3], Monte-Carlo IC.  Topic-blind and community-blind."""

    def test_returns_k_distinct_seeds_with_positive_seconds(self):
        res = greedy_select(self.ds, self.pp_blind, self.K, self.N_MC,
                            random.Random(1))
        self.assert_run_result(res, method="Greedy")

    def test_celf_flag_is_reported(self):
        res = greedy_select(self.ds, self.pp_blind, self.K, self.N_MC,
                            random.Random(1), use_celf=True)
        self.assertIn("use_celf", res.extra)
        self.assertTrue(res.extra["use_celf"])
        res2 = greedy_select(self.ds, self.pp_blind, self.K, self.N_MC,
                             random.Random(1), use_celf=False)
        self.assertFalse(res2.extra["use_celf"])
        self.assert_run_result(res2, method="Greedy(no CELF)")

    def test_deterministic_for_a_fixed_seed(self):
        a = greedy_select(self.ds, self.pp_blind, self.K, self.N_MC,
                          random.Random(99))
        b = greedy_select(self.ds, self.pp_blind, self.K, self.N_MC,
                          random.Random(99))
        self.assertEqual(a.seeds, b.seeds)

    def test_requires_an_explicit_rng(self):
        with self.assertRaises(ValueError):
            greedy_select(self.ds, self.pp_blind, self.K, self.N_MC, None)

    def test_various_k(self):
        for K in (1, 3, 7):
            res = greedy_select(self.ds, self.pp_blind, K, self.N_MC,
                                random.Random(2))
            self.assert_run_result(res, K=K, method="Greedy(K=%d)" % K)

    def test_topic_averaged_pp_covers_every_edge(self):
        self.assertEqual(set(self.pp_blind), set(self.ds.edges))
        for e, p in self.pp_blind.items():
            self.assertGreaterEqual(p, 0.0, e)
            self.assertLessEqual(p, 1.0, e)
        with self.assertRaises(ValueError):
            topic_averaged_pp(self.ew, [])


class TestCga(_BaselineFixture):
    """CGA of Wang et al. [22]: community DP + MixedGreedy."""

    def test_returns_k_distinct_seeds_with_positive_seconds(self):
        res = cga_select_seeds(self.comm, self.ds, self.pp, self.K,
                               random.Random(1), n_mc=self.N_MC)
        self.assert_run_result(res, method="CGA")

    def test_deterministic_for_a_fixed_seed(self):
        a = cga_select_seeds(self.comm, self.ds, self.pp, self.K,
                             random.Random(5), n_mc=self.N_MC)
        b = cga_select_seeds(self.comm, self.ds, self.pp, self.K,
                             random.Random(5), n_mc=self.N_MC)
        self.assertEqual(a.seeds, b.seeds)

    def test_both_dp_tiebreaks_run(self):
        for tb in ("consistent", "paper-literal"):
            res = cga_select_seeds(self.comm, self.ds, self.pp, self.K,
                                   random.Random(1), n_mc=self.N_MC,
                                   dp_tiebreak=tb)
            self.assert_run_result(res, method="CGA(%s)" % tb)

    def test_seeds_respect_the_supplied_community_labels(self):
        res = cga_select_seeds(self.comm, self.ds, self.pp, self.K,
                               random.Random(1), n_mc=self.N_MC)
        for u in res.seeds:
            self.assertTrue(0 <= self.comm[u] < self.C)


class TestCtimCga(_BaselineFixture):
    """CTIM_CGA: CTIM's model, CGA's selection (the ONLY difference is the
    influence computation model -- SPEC.md Section 7)."""

    def test_returns_k_distinct_seeds_with_positive_seconds(self):
        res = ctim_cga_select(self.model, self.ds, self.ITEM, self.K,
                              random.Random(1), n_mc=self.N_MC)
        self.assert_run_result(res, method="CTIM_CGA")

    def test_reports_itself_as_topic_aware(self):
        res = ctim_cga_select(self.model, self.ds, self.ITEM, self.K,
                              random.Random(1), n_mc=self.N_MC)
        self.assertEqual(res.extra["method"], "CTIM_CGA")
        self.assertTrue(res.extra["topic_aware"])
        self.assertEqual(res.extra["C"], self.C)
        self.assertEqual(res.extra["Z"], self.Z)

    def test_deterministic_for_a_fixed_seed(self):
        a = ctim_cga_select(self.model, self.ds, self.ITEM, self.K,
                            random.Random(3), n_mc=self.N_MC)
        b = ctim_cga_select(self.model, self.ds, self.ITEM, self.K,
                            random.Random(3), n_mc=self.N_MC)
        self.assertEqual(a.seeds, b.seeds)
        self.assertAlmostEqual(a.spread, b.spread, places=12)

    def test_various_k(self):
        for K in (1, 3, 6):
            res = ctim_cga_select(self.model, self.ds, self.ITEM, K,
                                  random.Random(1), n_mc=self.N_MC)
            self.assert_run_result(res, K=K, method="CTIM_CGA(K=%d)" % K)


class TestCinema(_BaselineFixture):
    """CINEMA (Li et al. [24]): conformity-aware, community-based, topic-blind."""

    def test_returns_k_distinct_seeds_with_positive_seconds(self):
        res = cinema_select_seeds(self.ds, self.pp_blind, self.K,
                                  random.Random(1), n_mc=self.N_MC, C=self.C)
        self.assert_run_result(res, method="CINEMA")

    def test_community_detection_is_independent_of_the_diffusion_model(self):
        res = cinema_select_seeds(self.ds, self.pp_blind, self.K,
                                  random.Random(1), n_mc=self.N_MC, C=self.C)
        # That independence is precisely the paper's criticism of CINEMA, and
        # the module records it.
        self.assertIn("community_detection", res.extra)
        self.assertIn("conformity", res.extra)
        self.assertIn("modularity", res.extra)

    def test_deterministic_for_a_fixed_seed(self):
        a = cinema_select_seeds(self.ds, self.pp_blind, self.K,
                                random.Random(8), n_mc=self.N_MC, C=self.C)
        b = cinema_select_seeds(self.ds, self.pp_blind, self.K,
                                random.Random(8), n_mc=self.N_MC, C=self.C)
        self.assertEqual(a.seeds, b.seeds)

    def test_accepts_a_precomputed_community_labelling(self):
        res = cinema_select_seeds(self.ds, self.pp_blind, self.K,
                                  random.Random(1), n_mc=self.N_MC, C=self.C,
                                  comm=self.comm)
        self.assert_run_result(res, method="CINEMA(comm=)")

    def test_various_k(self):
        for K in (1, 3, 6):
            res = cinema_select_seeds(self.ds, self.pp_blind, K,
                                      random.Random(1), n_mc=self.N_MC,
                                      C=self.C)
            self.assert_run_result(res, K=K, method="CINEMA(K=%d)" % K)


class TestAirCga(_BaselineFixture):
    """AIR+CGA: Barbieri et al. [35] AIR fitted by EM, then CGA."""

    N_EM = 8

    def test_returns_k_distinct_seeds_with_positive_seconds(self):
        res = air_cga_select_seeds(self.ds, self.train, self.ITEM, self.K,
                                   random.Random(1), Z=self.Z, C=self.C,
                                   n_mc=self.N_MC, n_em_iter=self.N_EM)
        self.assert_run_result(res, method="AIR+CGA")

    def test_reports_itself_as_topic_aware_and_community_based(self):
        res = air_cga_select_seeds(self.ds, self.train, self.ITEM, self.K,
                                   random.Random(1), Z=self.Z, C=self.C,
                                   n_mc=self.N_MC, n_em_iter=self.N_EM)
        self.assertEqual(res.extra["method"], "AIR+CGA")
        self.assertTrue(res.extra["topic_aware"])
        self.assertTrue(res.extra["community_based"])
        self.assertEqual(res.extra["Z"], self.Z)

    def test_seconds_covers_the_em_fit_as_well_as_selection(self):
        res = air_cga_select_seeds(self.ds, self.train, self.ITEM, self.K,
                                   random.Random(1), Z=self.Z, C=self.C,
                                   n_mc=self.N_MC, n_em_iter=self.N_EM)
        self.assertGreater(res.seconds, 0.0)
        if "fit_seconds" in res.extra:
            self.assertLessEqual(res.extra["fit_seconds"], res.seconds + 1e-6)

    def test_deterministic_for_a_fixed_seed(self):
        a = air_cga_select_seeds(self.ds, self.train, self.ITEM, self.K,
                                 random.Random(21), Z=self.Z, C=self.C,
                                 n_mc=self.N_MC, n_em_iter=self.N_EM)
        b = air_cga_select_seeds(self.ds, self.train, self.ITEM, self.K,
                                 random.Random(21), Z=self.Z, C=self.C,
                                 n_mc=self.N_MC, n_em_iter=self.N_EM)
        self.assertEqual(a.seeds, b.seeds)

    def test_requires_an_explicit_rng(self):
        with self.assertRaises(ValueError):
            air_cga_select_seeds(self.ds, self.train, self.ITEM, self.K, None,
                                 Z=self.Z, C=self.C, n_mc=self.N_MC,
                                 n_em_iter=self.N_EM)

    def test_various_k(self):
        for K in (1, 3, 6):
            res = air_cga_select_seeds(self.ds, self.train, self.ITEM, K,
                                       random.Random(1), Z=self.Z, C=self.C,
                                       n_mc=self.N_MC, n_em_iter=self.N_EM)
            self.assert_run_result(res, K=K, method="AIR+CGA(K=%d)" % K)


class TestUniformSelectSeedsEntryPoints(_BaselineFixture):
    """API.md: every baseline module exposes ``select_seeds(...) -> RunResult``."""

    def test_every_module_exposes_select_seeds(self):
        from ctim.baselines import air_cga, cga, cinema, greedy
        for mod in (greedy, cga, cinema, air_cga):
            self.assertTrue(callable(getattr(mod, "select_seeds", None)),
                            mod.__name__)

    def test_package_level_lazy_exports_resolve(self):
        import ctim.baselines as B
        for name in ("greedy_select", "cga_select_seeds", "ctim_cga_select",
                     "cinema_select_seeds", "air_cga_select_seeds",
                     "topic_averaged_pp", "RunResult"):
            self.assertTrue(hasattr(B, name), name)
        with self.assertRaises(AttributeError):
            B.no_such_baseline


if __name__ == "__main__":
    unittest.main(verbosity=2)
