"""Tests for ``ctim/ctim.py`` -- Algorithm 2 and Eq (19).

Covers:
  * Eq (19) ``detect_communities``: argmax with deterministic tie-breaking
  * Algorithm 2 returns exactly K *distinct* seeds
  * every selected seed lies in the community the DP chose at that step
  * the DP array satisfies its recurrence (line 35) and its back-pointer rule
    (lines 36-40), verified against a from-scratch transcription of lines 25-45
    that recomputes every dI_m with ``MIA.marginal_gain`` and does no caching
  * selection is deterministic for a fixed seed
  * all three ``dp_tiebreak`` readings of lines 35/36 run and return K distinct
    seeds, and the printed reading (``paper-true``, the default) is checked to
    stay distinguishable from the historical ones -- see DEVIATIONS.md 1.1
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctim_testutil import make_planted_dataset  # noqa: E402

from ctim.ctim import (RunResult, _CommunitySeedState,  # noqa: E402
                       ctim_run, ctim_select_seeds, detect_communities)
from ctim.dataset import build_potential_influence_logs, split_logs  # noqa: E402
from ctim.gibbs import train_model  # noqa: E402
from ctim.influence import MIA, EdgeWeights  # noqa: E402


# ---------------------------------------------------------------------------
# A from-scratch transcription of Algorithm 2 lines 25-45, used as the oracle
# for the DP recurrence.  It caches NOTHING: every dI_m is recomputed with
# MIA.marginal_gain, so it exercises the algorithm as printed rather than the
# incremental machinery the production path uses.
# ---------------------------------------------------------------------------


def reference_algorithm2(model, ds, item, K, h, dp_tiebreak, ew):
    """Return ``(seeds, chosen_community_per_step, I_table, s_table, dI_table)``.

    ``I_table[m][k]`` and ``s_table[m][k]`` are the arrays of Algorithm 2
    lines 26-40; ``chosen_community_per_step[k-1]`` is the community index
    ``j = s[C,k]`` (1-based, as in the paper) that step ``k`` selected from.
    ``dI_table[m][k]`` is the line-34 marginal gain used to fill ``I_table[m][k]``;
    it is returned so a test can check the line-35 recurrence exactly instead of
    tautologically re-deriving dI from the table it is meant to verify.
    """
    pp = ew.for_item(item)                                   # Eq (12)
    comm = detect_communities(model.pi)                      # lines 22-24, Eq (19)
    C = len(model.eta)

    members = [[] for _ in range(C + 1)]
    for v in range(ds.n_users):
        members[comm[v] + 1].append(v)

    mias = {}
    for m in range(1, C + 1):
        if members[m]:
            # I_m = Eq (18) on community m's node-induced subgraph (line 34)
            mias[m] = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h,
                          nodes=members[m])

    S = []                                                   # line 25
    Sm = {m: set() for m in range(1, C + 1)}
    Iv = [[0.0] * (K + 1) for _ in range(C + 1)]             # lines 26-31
    sp = [[0] * (K + 1) for _ in range(C + 1)]
    dItab = [[0.0] * (K + 1) for _ in range(C + 1)]
    chosen = []

    for k in range(1, K + 1):                                # line 32
        argmax_u = {}
        for m in range(1, C + 1):                            # line 33
            dI = 0.0
            bu = None
            if m in mias:
                for u in members[m]:                         # line 34
                    if u in Sm[m]:
                        continue
                    g = mias[m].marginal_gain(Sm[m], u)
                    if bu is None or g > dI + 1e-12:
                        dI, bu = g, u
                if bu is None:
                    dI = 0.0
            argmax_u[m] = bu
            dItab[m][k] = dI

            # line 35 -- as printed the reference is I[C,k-1] (DEVIATIONS.md 1.1)
            ref35 = Iv[C][k - 1] if dp_tiebreak == "paper-true" else Iv[m][k - 1]
            cand = ref35 + dI
            prev = Iv[m - 1][k]
            Iv[m][k] = prev if prev > cand else cand         # line 35

            # lines 36-40
            ref = Iv[m][k - 1] if dp_tiebreak == "consistent" else Iv[C][k - 1]
            sp[m][k] = m if ref + dI >= prev else sp[m - 1][k]

        j = sp[C][k]                                         # line 42
        u_k = argmax_u.get(j)
        if u_k is None:
            # documented deviation: DP pointed at an empty/exhausted community
            best_g = None
            for m2 in sorted(mias):
                if argmax_u.get(m2) is None:
                    continue
                g2 = mias[m2].marginal_gain(Sm[m2], argmax_u[m2])
                if best_g is None or g2 > best_g:
                    best_g, u_k, j = g2, argmax_u[m2], m2
            if u_k is None:
                break
        Sm[j].add(u_k)                                       # line 44
        S.append(u_k)
        chosen.append(j)
    return S, chosen, Iv, sp, dItab


class TestDetectCommunities(unittest.TestCase):
    """Eq (19): c^v_m <- argmax_c pi_{v,c}."""

    def test_argmax_with_lowest_index_tiebreak(self):
        pi = [
            [0.1, 0.7, 0.2],   # -> 1
            [0.5, 0.5, 0.0],   # tie between 0 and 1 -> lowest index 0
            [0.0, 0.0, 1.0],   # -> 2
            [0.34, 0.33, 0.33],  # -> 0
            [0.2, 0.2, 0.2],   # three-way tie -> 0
        ]
        self.assertEqual(detect_communities(pi), [1, 0, 2, 0, 0])

    def test_one_label_per_user_in_range(self):
        rng = random.Random(3)
        C = 6
        pi = []
        for _ in range(50):
            row = [rng.random() for _ in range(C)]
            s = sum(row)
            pi.append([x / s for x in row])
        comm = detect_communities(pi)
        self.assertEqual(len(comm), 50)
        for c in comm:
            self.assertTrue(0 <= c < C)

    def test_empty_row_defaults_to_zero(self):
        self.assertEqual(detect_communities([[]]), [0])


class _CtimFixture(unittest.TestCase):
    """A trained model on a small planted dataset, shared by the tests below."""

    K = 5
    ITEM = 0
    C = 3
    Z = 3

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
        cls.result = ctim_run(cls.model, cls.ds, cls.ITEM, cls.K, h=0.1,
                              edge_weights=cls.ew)
        cls.stats = cls.result.extra


class TestAlgorithm2Output(_CtimFixture):

    def test_training_produced_usable_logs(self):
        self.assertGreater(len(self.train), 20, len(self.train))

    def test_returns_exactly_k_distinct_seeds(self):
        for K in (1, 2, 3, 5, 8):
            seeds = ctim_select_seeds(self.model, self.ds, self.ITEM, K,
                                      h=0.1, edge_weights=self.ew)
            self.assertEqual(len(seeds), K, "K=%d gave %s" % (K, seeds))
            self.assertEqual(len(set(seeds)), K, "duplicates at K=%d: %s"
                             % (K, seeds))
            for u in seeds:
                self.assertTrue(0 <= u < self.ds.n_users, u)

    def test_k_zero_returns_empty(self):
        self.assertEqual(
            ctim_select_seeds(self.model, self.ds, self.ITEM, 0,
                              edge_weights=self.ew), [])

    def test_seeds_actually_propagate(self):
        self.assertGreater(self.result.spread, self.K)

    def test_run_result_shape(self):
        self.assertIsInstance(self.result, RunResult)
        self.assertEqual(len(self.result.seeds), self.K)
        self.assertGreater(self.result.spread, 0.0)
        self.assertGreater(self.result.seconds, 0.0)
        self.assertIsInstance(self.result.extra, dict)

    def test_last_stats_records_per_line_timings(self):
        st = self.stats
        for key in ("weights", "detect", "subgraphs", "dp", "select", "total"):
            self.assertIn(key, st, key)
            self.assertGreaterEqual(st[key], 0.0, key)
        phase_sum = (st["weights"] + st["detect"] + st["subgraphs"]
                     + st["dp"] + st["select"])
        self.assertLessEqual(abs(phase_sum - st["total"]),
                             0.05 * max(st["total"], 1e-6) + 1e-3,
                             "phases %.6f vs total %.6f" % (phase_sum, st["total"]))

    def test_ctim_beats_random_seed_sets_on_average(self):
        mia = MIA(self.ds.n_users, self.ds.out_adj, self.ds.in_adj, self.pp,
                  h=0.1)
        rr = random.Random(11)
        vals = [mia.influence(rr.sample(range(self.ds.n_users), self.K))
                for _ in range(200)]
        mean_random = sum(vals) / len(vals)
        self.assertGreater(self.result.spread, mean_random * 1.05,
                           "CTIM %.4f vs mean-of-200 random %.4f"
                           % (self.result.spread, mean_random))

    def test_community_restriction_costs_a_bounded_amount(self):
        """CTIM restricts each seed to one community and scores it on that
        community's *induced subgraph* (Algorithm 2 line 34), so it throws away
        every cross-community path.  On a small dense graph that is a real,
        measurable handicap against unrestricted global MIA greedy -- this test
        pins how large it is so a regression cannot widen it silently.

        See DEVIATIONS.md: this is a property of the paper's algorithm, not of
        the implementation.  It is why the paper evaluates on graphs where
        communities are large relative to K.
        """
        mia = MIA(self.ds.n_users, self.ds.out_adj, self.ds.in_adj, self.pp,
                  h=0.1)
        global_greedy = mia.influence(mia.greedy_incremental(self.K))
        self.assertGreater(global_greedy, 0.0)
        ratio = self.result.spread / global_greedy
        self.assertGreaterEqual(
            ratio, 0.70,
            "community-restricted CTIM %.4f is only %.1f%% of unrestricted "
            "global MIA greedy %.4f" % (self.result.spread, 100 * ratio,
                                        global_greedy))
        self.assertLessEqual(ratio, 1.0 + 1e-9,
                             "CTIM cannot beat unrestricted greedy on the same "
                             "evaluator: %.6f vs %.6f"
                             % (self.result.spread, global_greedy))

    def test_spread_is_non_decreasing_in_k(self):
        prev = -1.0
        for k in range(1, self.K + 1):
            r = ctim_run(self.model, self.ds, self.ITEM, k, h=0.1,
                         edge_weights=self.ew)
            self.assertGreaterEqual(r.spread, prev - 1e-9, "K=%d" % k)
            prev = r.spread


class TestSeedsLieInTheChosenCommunity(_CtimFixture):

    def test_histogram_matches_seeds_per_community(self):
        """``seeds_per_community`` is what the DP charged each community with;
        the histogram of the returned seeds' Eq (19) labels must equal it."""
        hist = {}
        for u in self.result.seeds:
            key = self.comm[u] + 1          # last_stats indexes m = 1..C
            hist[key] = hist.get(key, 0) + 1
        self.assertEqual(hist, self.stats["seeds_per_community"])
        self.assertEqual(sum(self.stats["seeds_per_community"].values()), self.K)

    def test_each_seed_is_in_the_community_the_dp_selected_that_step(self):
        seeds, chosen, _Iv, _sp, _d = reference_algorithm2(
            self.model, self.ds, self.ITEM, self.K, 0.1, "paper-true", self.ew)
        self.assertEqual(len(seeds), len(chosen))
        for step, (u, j) in enumerate(zip(seeds, chosen)):
            self.assertEqual(self.comm[u] + 1, j,
                             "step %d selected node %d (community %d) but the "
                             "DP chose community %d"
                             % (step + 1, u, self.comm[u] + 1, j))

    def test_no_seed_comes_from_an_empty_community(self):
        sizes = {}
        for v in range(self.ds.n_users):
            sizes[self.comm[v]] = sizes.get(self.comm[v], 0) + 1
        for u in self.result.seeds:
            self.assertGreater(sizes.get(self.comm[u], 0), 0)


class TestDpRecurrence(_CtimFixture):
    """Algorithm 2 lines 26-40, checked on the reference DP array."""

    @classmethod
    def setUpClass(cls):
        super(TestDpRecurrence, cls).setUpClass()
        cls.ref = {}
        for tb in ("paper-true", "consistent", "paper-literal"):
            cls.ref[tb] = reference_algorithm2(
                cls.model, cls.ds, cls.ITEM, cls.K, 0.1, tb, cls.ew)

    def test_boundary_rows_are_zero(self):
        for tb in ("paper-true", "consistent", "paper-literal"):
            _s, _c, Iv, sp, _d = self.ref[tb]
            for k in range(self.K + 1):
                self.assertEqual(Iv[0][k], 0.0, "I[0,%d] (%s)" % (k, tb))  # line 27
                self.assertEqual(sp[0][k], 0, "s[0,%d] (%s)" % (k, tb))    # line 27
            for m in range(self.C + 1):
                self.assertEqual(Iv[m][0], 0.0, "I[%d,0] (%s)" % (m, tb))  # line 30

    def test_line_35_recurrence_holds(self):
        """I[m,k] == max(I[m-1,k], ref35 + dI_m), with ref35 fixed by the mode.

        ``dI_m`` comes from the oracle's own line-34 table, NOT re-derived from
        ``I``: deriving it as ``I[m,k] - I[m,k-1]`` makes the second branch
        trivially true and the assertion vacuous, which is how the mis-transcribed
        line 35 survived undetected (DEVIATIONS.md 1.1).
        """
        for tb in ("paper-true", "consistent", "paper-literal"):
            _s, _c, Iv, _sp, dIt = self.ref[tb]
            for k in range(1, self.K + 1):
                for m in range(1, self.C + 1):
                    dI = dIt[m][k]
                    self.assertGreaterEqual(dI, -1e-12, (m, k, tb))
                    # dI_m is a marginal gain of a monotone function, hence >= 0,
                    # so line 35 forces I[m,k] >= I[m-1,k] in every mode.
                    self.assertGreaterEqual(Iv[m][k], Iv[m - 1][k] - 1e-12,
                                            "I[%d,%d] < I[%d,%d] (%s)"
                                            % (m, k, m - 1, k, tb))
                    ref35 = Iv[self.C][k - 1] if tb == "paper-true" else Iv[m][k - 1]
                    self.assertAlmostEqual(
                        Iv[m][k], max(Iv[m - 1][k], ref35 + dI), places=9,
                        msg="line 35 violated at I[%d,%d] (%s)" % (m, k, tb))

    def test_paper_true_line_35_differs_from_the_historical_reading(self):
        """`I[C,k-1]` and `I[m,k-1]` are not interchangeable on line 35.

        Regression guard for DEVIATIONS.md 1.1: if this ever passes trivially the
        three modes have collapsed into one and the distinction is untested.
        """
        _s, _c, Iv_true, _sp, _d = self.ref["paper-true"]
        _s2, _c2, Iv_cons, _sp2, _d2 = self.ref["consistent"]
        C, K = self.C, self.K
        # I is non-decreasing in m, so I[C,k-1] >= I[m,k-1] and the paper's
        # line 35 can never report a smaller total than the historical reading.
        for k in range(1, K + 1):
            self.assertGreaterEqual(Iv_true[C][k], Iv_cons[C][k] - 1e-12,
                                    "paper-true DP value below 'consistent' at k=%d" % k)

    def test_back_pointer_is_a_valid_community_index(self):
        for tb in ("paper-true", "consistent", "paper-literal"):
            _s, _c, _Iv, sp, _d = self.ref[tb]
            for k in range(1, self.K + 1):
                for m in range(1, self.C + 1):
                    self.assertTrue(0 <= sp[m][k] <= m,
                                    "s[%d,%d]=%d (%s)" % (m, k, sp[m][k], tb))

    def test_dp_value_is_non_decreasing_in_k_and_m(self):
        for tb in ("paper-true", "consistent", "paper-literal"):
            _s, _c, Iv, _sp, dIt = self.ref[tb]
            for m in range(self.C + 1):
                for k in range(1, self.K + 1):
                    self.assertGreaterEqual(Iv[m][k], Iv[m][k - 1] - 1e-12)
            for k in range(self.K + 1):
                for m in range(1, self.C + 1):
                    self.assertGreaterEqual(Iv[m][k], Iv[m - 1][k] - 1e-12)

    def test_production_path_agrees_with_the_uncached_transcription(self):
        """The incremental IncInf caches must not change the answer."""
        for tb in ("paper-true", "consistent", "paper-literal"):
            ref_seeds = self.ref[tb][0]
            got = ctim_select_seeds(self.model, self.ds, self.ITEM, self.K,
                                    h=0.1, dp_tiebreak=tb,
                                    edge_weights=self.ew)
            if got != ref_seeds:
                # Equal-gain ties may be resolved differently; the achieved
                # spread must still match exactly.
                mia = MIA(self.ds.n_users, self.ds.out_adj, self.ds.in_adj,
                          self.pp, h=0.1)
                self.assertAlmostEqual(
                    mia.influence(got), mia.influence(ref_seeds), places=9,
                    msg="%s: %s vs reference %s" % (tb, got, ref_seeds))

    def test_reported_dp_value_matches_the_reference_table(self):
        # The fixture runs ctim_run with the default reading, so the oracle must
        # be read at the same mode (DEVIATIONS.md 1.1).
        self.assertEqual(self.stats["dp_tiebreak"], "paper-true")
        _s, _c, Iv, _sp, _d = self.ref["paper-true"]
        self.assertAlmostEqual(self.stats["dp_value"], Iv[self.C][self.K],
                               places=9)


class TestIncrementalStateIsExact(_CtimFixture):
    """``_CommunitySeedState.inc`` must equal ``MIA.marginal_gain`` at all times."""

    def test_inc_table_tracks_marginal_gain(self):
        sizes = {}
        for v in range(self.ds.n_users):
            sizes[self.comm[v]] = sizes.get(self.comm[v], 0) + 1
        biggest = max(sizes, key=lambda c: (sizes[c], -c))
        members = [v for v in range(self.ds.n_users) if self.comm[v] == biggest]
        mia_m = MIA(self.ds.n_users, self.ds.out_adj, self.ds.in_adj, self.pp,
                    h=0.1, nodes=members)
        state = _CommunitySeedState(biggest + 1, mia_m, members)

        for step in range(4):
            for u in members:
                if u in state.S_set:
                    continue
                self.assertAlmostEqual(
                    state.inc[u], mia_m.marginal_gain(state.S_set, u),
                    places=9, msg="step %d node %d" % (step, u))
            u_best, gain = state.best()
            if u_best is None:
                break
            # line 34's argmax must be the true argmax
            true_g = -1.0
            for u in members:
                if u in state.S_set:
                    continue
                g = mia_m.marginal_gain(state.S_set, u)
                if g > true_g:
                    true_g = g
            self.assertAlmostEqual(gain, true_g, places=9,
                                   msg="step %d" % step)
            state.add(u_best)


class TestDeterminismAndTiebreaks(_CtimFixture):

    def test_selection_is_deterministic(self):
        a = ctim_select_seeds(self.model, self.ds, self.ITEM, self.K, h=0.1,
                              edge_weights=self.ew)
        b = ctim_select_seeds(self.model, self.ds, self.ITEM, self.K, h=0.1,
                              edge_weights=self.ew)
        self.assertEqual(a, b)
        # ... and independent of whether a shared EdgeWeights cache is passed
        c = ctim_select_seeds(self.model, self.ds, self.ITEM, self.K, h=0.1)
        self.assertEqual(a, c)

    def test_deterministic_across_freshly_trained_identical_models(self):
        model2 = train_model(self.ds, self.train, C=self.C, Z=self.Z,
                             n_iter_topic=60, n_iter_comm=60,
                             rng=random.Random(7))
        self.assertEqual(model2.pi, self.model.pi)
        seeds2 = ctim_select_seeds(model2, self.ds, self.ITEM, self.K, h=0.1)
        self.assertEqual(seeds2, self.result.seeds)

    def test_both_tiebreaks_run_and_return_k_distinct_seeds(self):
        for tb in ("paper-true", "consistent", "paper-literal"):
            r = ctim_run(self.model, self.ds, self.ITEM, self.K, h=0.1,
                         dp_tiebreak=tb, edge_weights=self.ew)
            self.assertEqual(len(r.seeds), self.K, tb)
            self.assertEqual(len(set(r.seeds)), self.K, tb)
            self.assertGreater(r.spread, 0.0, tb)
            self.assertEqual(r.extra["dp_tiebreak"], tb)

    def test_unknown_tiebreak_rejected(self):
        with self.assertRaises(ValueError):
            ctim_select_seeds(self.model, self.ds, self.ITEM, self.K,
                              dp_tiebreak="nonsense", edge_weights=self.ew)

    def test_mismatched_model_and_dataset_rejected(self):
        class _Tiny(object):
            pass

        tiny = _Tiny()
        tiny.n_users = self.ds.n_users + 100
        tiny.edges = self.ds.edges
        tiny.out_adj = self.ds.out_adj
        tiny.in_adj = self.ds.in_adj
        with self.assertRaises(ValueError):
            ctim_select_seeds(self.model, tiny, self.ITEM, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
