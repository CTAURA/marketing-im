"""Tests for ``ctim/dataset.py``.

Covers:
  * the canonical on-disk format of API.md (save -> load round-trip)
  * **Definition 1** (SPEC.md Section 0), the potential-influence log set ``D``,
    on a hand-built example small enough that the correct answer is enumerated
    by hand in the docstring -- including every boundary case:
        t_q - t_p == delta   -> INCLUDED
        t_q - t_p == 0       -> EXCLUDED (strict 0 < dt)
        t_q - t_p >  delta   -> EXCLUDED
        (u,v) not in E       -> EXCLUDED
        arc present only in the wrong direction -> EXCLUDED
  * the 60/20/20 split of SPEC.md Section 5.2 (sizes, disjointness, coverage)
"""

from __future__ import annotations

import gzip
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctim_testutil  # noqa: F401,E402  (performs the repo-root sys.path bootstrap)

from ctim.dataset import (Dataset, build_adjacency,  # noqa: E402
                          build_potential_influence_logs, load_dataset,
                          save_dataset, split_logs, summarize)


# ---------------------------------------------------------------------------
# The hand-built fixture
# ---------------------------------------------------------------------------
#
# 6 users.  Directed arcs (note (4,5) is DELIBERATELY absent; only (5,4) exists):
#
#     E = { (0,1), (0,2), (0,3), (3,0), (5,4), (1,2) }
#
# Adoption logs (u, i, t):
#
#     item 0:  (0,0,1000) (3,0,1000) (4,0,1050) (5,0,1080)
#              (1,0,1100) (2,0,1101) (1,0,1200)   <- repeat, must be dropped
#     item 1:  (1,1,2000) (2,1,2100)
#
# First-adoption times for item 0:  0:1000  3:1000  4:1050  5:1080  1:1100  2:1101
# First-adoption times for item 1:  1:2000  2:2100
#
# Definition 1 with delta = 100, enumerated over every arc of E:
#
#   item 0
#     (0,1)  1100-1000 =  100  == delta          -> INCLUDE (0,1,0)
#     (0,2)  1101-1000 =  101  >  delta          -> exclude
#     (0,3)  1000-1000 =    0  boundary dt == 0  -> exclude
#     (3,0)  1000-1000 =    0  boundary dt == 0  -> exclude
#     (5,4)  1050-1080 =  -30  dt <= 0           -> exclude
#     (1,2)  1101-1100 =    1  ok                -> INCLUDE (1,2,0)
#   non-arcs that WOULD have qualified on time, so they prove E is consulted:
#     (4,5)    30   time-respecting but arc absent (only (5,4) exists) -> exclude
#     (3,1)   100   exactly delta but (3,1) not in E                   -> exclude
#     (0,4) 50, (0,5) 80, (5,1) 20, (5,2) 21, (4,1) 50, (4,2) 51       -> exclude
#
#   item 1
#     (1,2)  2100-2000 =  100  == delta          -> INCLUDE (1,2,1)
#
# Therefore  D = { (0,1,0), (1,2,0), (1,2,1) }.
# ---------------------------------------------------------------------------

HAND_EDGES = [(0, 1), (0, 2), (0, 3), (3, 0), (5, 4), (1, 2)]
HAND_LOGS = [
    (0, 0, 1000),
    (3, 0, 1000),
    (4, 0, 1050),
    (5, 0, 1080),
    (1, 0, 1100),
    (2, 0, 1101),
    (1, 0, 1200),  # repeated adoption of item 0 by user 1 -- collapses to 1100
    (1, 1, 2000),
    (2, 1, 2100),
]
HAND_ITEM_ATTRS = [[0, 1, 1], []]  # item 1 deliberately has an empty bag

EXPECTED_D_DELTA_100 = [(0, 1, 0), (1, 2, 0), (1, 2, 1)]


def build_hand_dataset(extra_edges=()):
    edges = list(HAND_EDGES) + list(extra_edges)
    n_users = 6
    out_adj, in_adj = build_adjacency(n_users, edges)
    logs = sorted(HAND_LOGS, key=lambda r: (r[2], r[0], r[1]))
    return Dataset(
        name="hand",
        n_users=n_users,
        n_items=2,
        n_attrs=3,
        edges=edges,
        out_adj=out_adj,
        in_adj=in_adj,
        logs=logs,
        item_attrs=[list(b) for b in HAND_ITEM_ATTRS],
        source="unit test",
        notes="hand-built Definition 1 fixture",
    )


class TestCanonicalFormatRoundTrip(unittest.TestCase):
    """API.md 'Canonical on-disk dataset format'."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ctim-test-dataset-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ds = build_hand_dataset()
        self.dir = os.path.join(self.tmp, "hand")

    def test_round_trip_preserves_every_field(self):
        save_dataset(self.ds, self.dir)
        for fname in ("graph.tsv", "logs.tsv", "items.tsv", "meta.json"):
            self.assertTrue(os.path.exists(os.path.join(self.dir, fname)), fname)

        ds2 = load_dataset(self.dir)
        self.assertEqual(ds2.name, "hand")
        self.assertEqual(ds2.n_users, self.ds.n_users)
        self.assertEqual(ds2.n_items, self.ds.n_items)
        self.assertEqual(ds2.n_attrs, self.ds.n_attrs)
        self.assertEqual(sorted(ds2.edges), sorted(self.ds.edges))
        self.assertEqual(ds2.logs, self.ds.logs)
        self.assertEqual(ds2.item_attrs, self.ds.item_attrs)
        self.assertEqual(ds2.out_adj, self.ds.out_adj)
        self.assertEqual(ds2.in_adj, self.ds.in_adj)
        self.assertEqual(ds2.source, "unit test")
        self.assertEqual(ds2.notes, "hand-built Definition 1 fixture")

    def test_meta_json_has_the_declared_schema(self):
        save_dataset(self.ds, self.dir)
        with open(os.path.join(self.dir, "meta.json"), "rt", encoding="utf-8") as fh:
            meta = json.load(fh)
        for key in ("name", "n_users", "n_links", "n_items", "n_attrs",
                    "n_logs", "source", "notes"):
            self.assertIn(key, meta, key)
        self.assertEqual(meta["n_users"], 6)
        self.assertEqual(meta["n_links"], len(HAND_EDGES))
        self.assertEqual(meta["n_logs"], len(HAND_LOGS))
        self.assertEqual(meta["n_items"], 2)

    def test_logs_are_written_sorted_by_time(self):
        unsorted_logs = list(reversed(sorted(HAND_LOGS, key=lambda r: r[2])))
        out_adj, in_adj = build_adjacency(6, HAND_EDGES)
        ds = Dataset("hand", 6, 2, 3, list(HAND_EDGES), out_adj, in_adj,
                     unsorted_logs, [list(b) for b in HAND_ITEM_ATTRS])
        save_dataset(ds, self.dir)
        times = []
        with open(os.path.join(self.dir, "logs.tsv"), "rt", encoding="utf-8") as fh:
            for line in fh:
                times.append(int(line.split("\t")[2]))
        self.assertEqual(times, sorted(times))

    def test_empty_attribute_bag_round_trips(self):
        save_dataset(self.ds, self.dir)
        ds2 = load_dataset(self.dir)
        self.assertEqual(ds2.item_attrs[1], [])
        self.assertEqual(ds2.item_attrs[0], [0, 1, 1])

    def test_gzip_fallback(self):
        save_dataset(self.ds, self.dir)
        graph = os.path.join(self.dir, "graph.tsv")
        with open(graph, "rt", encoding="utf-8") as fh:
            raw = fh.read()
        with gzip.open(graph + ".gz", "wt", encoding="utf-8") as fh:
            fh.write(raw)
        os.remove(graph)
        ds3 = load_dataset(self.dir)
        self.assertEqual(sorted(ds3.edges), sorted(self.ds.edges))

    def test_edge_set_is_directed_and_cached(self):
        self.assertEqual(self.ds.edge_set(), set(HAND_EDGES))
        self.assertIn((5, 4), self.ds.edge_set())
        self.assertNotIn((4, 5), self.ds.edge_set())
        self.assertIs(self.ds.edge_set(), self.ds.edge_set())

    def test_summarize_reports_the_spec_section_8_columns(self):
        s = summarize(self.ds)
        for key in ("n_users", "n_links", "n_items", "n_attrs", "n_logs"):
            self.assertIn(key, s, key)
        self.assertEqual(s["n_users"], 6)
        self.assertEqual(s["n_links"], 6)
        self.assertEqual(s["n_logs"], 9)
        self.assertEqual(s["n_items_no_attrs"], 1)


class TestDefinition1(unittest.TestCase):
    """Definition 1: d=(u,v,i) iff 0 < t_q - t_p <= Delta and (u,v) in E."""

    def setUp(self):
        self.ds = build_hand_dataset()

    def test_hand_enumerated_answer(self):
        got = build_potential_influence_logs(self.ds, delta=100)
        self.assertEqual(sorted(got), EXPECTED_D_DELTA_100)

    def test_no_duplicate_logs_from_repeated_adoption(self):
        got = build_potential_influence_logs(self.ds, delta=100)
        self.assertEqual(len(got), len(set(got)))
        # user 1's SECOND adoption of item 0 is at t=1200, which would make
        # (0,1) a dt=200 pair -- dropped, so (0,1,0) survives via t=1100.
        self.assertIn((0, 1, 0), got)

    def test_boundary_dt_equals_delta_is_included(self):
        # (0,1) on item 0 has exactly dt == 100.
        self.assertIn((0, 1, 0), build_potential_influence_logs(self.ds, delta=100))
        # ... and disappears the moment delta drops below it.
        self.assertNotIn((0, 1, 0), build_potential_influence_logs(self.ds, delta=99))
        # (1,2) on item 1 also has exactly dt == 100.
        self.assertIn((1, 2, 1), build_potential_influence_logs(self.ds, delta=100))
        self.assertNotIn((1, 2, 1), build_potential_influence_logs(self.ds, delta=99))

    def test_boundary_dt_equals_zero_is_excluded(self):
        # users 0 and 3 both adopt item 0 at t=1000 and BOTH arcs (0,3),(3,0)
        # exist, so only the strict `0 < dt` rule can exclude them.
        got = build_potential_influence_logs(self.ds, delta=10 ** 9)
        self.assertNotIn((0, 3, 0), got)
        self.assertNotIn((3, 0, 0), got)

    def test_delta_zero_admits_nothing(self):
        self.assertEqual(build_potential_influence_logs(self.ds, delta=0), [])

    def test_wrong_edge_direction_is_excluded(self):
        # 4 adopts item 0 at 1050 and 5 at 1080, so (4,5) is time-respecting
        # with dt=30.  The graph only has the arc (5,4), so no log may appear.
        got = build_potential_influence_logs(self.ds, delta=100)
        self.assertNotIn((4, 5, 0), got)
        self.assertNotIn((5, 4, 0), got)

        # Adding the missing arc (4,5) -- and nothing else -- must create it.
        ds_with_arc = build_hand_dataset(extra_edges=[(4, 5)])
        got2 = build_potential_influence_logs(ds_with_arc, delta=100)
        self.assertEqual(sorted(got2),
                         sorted(EXPECTED_D_DELTA_100 + [(4, 5, 0)]))

    def test_non_edge_pairs_are_excluded_even_at_exactly_delta(self):
        # (3,1) on item 0 has dt = 1100-1000 = 100 == delta but is not an arc.
        got = build_potential_influence_logs(self.ds, delta=100)
        self.assertNotIn((3, 1, 0), got)
        # widening delta must never conjure a log across a non-existent arc
        wide = build_potential_influence_logs(self.ds, delta=10 ** 9)
        edge_set = self.ds.edge_set()
        for (u, v, _i) in wide:
            self.assertIn((u, v), edge_set)

    def test_wide_delta_admits_every_time_respecting_arc(self):
        got = build_potential_influence_logs(self.ds, delta=10 ** 9)
        # item 0: (0,1) 100, (0,2) 101, (1,2) 1  |  item 1: (1,2) 100
        # (0,3)/(3,0) still excluded (dt == 0); (5,4) still excluded (dt < 0).
        self.assertEqual(sorted(got),
                         [(0, 1, 0), (0, 2, 0), (1, 2, 0), (1, 2, 1)])

    def test_flipped_inner_loop_branch_gives_identical_answer(self):
        # build_potential_influence_logs switches enumeration strategy when a
        # user's out-degree dwarfs the item's adopter count.  Force that branch
        # by giving user 0 a huge out-degree and check the answer is unchanged.
        big_edges = list(HAND_EDGES) + [(0, x) for x in range(6, 400)]
        out_adj, in_adj = build_adjacency(400, big_edges)
        ds_big = Dataset("big", 400, 2, 3, big_edges, out_adj, in_adj,
                         sorted(HAND_LOGS, key=lambda r: (r[2], r[0], r[1])),
                         [list(b) for b in HAND_ITEM_ATTRS])
        self.assertEqual(
            sorted(build_potential_influence_logs(ds_big, delta=100)),
            EXPECTED_D_DELTA_100,
        )
        self.assertEqual(
            sorted(build_potential_influence_logs(ds_big, delta=10 ** 9)),
            [(0, 1, 0), (0, 2, 0), (1, 2, 0), (1, 2, 1)],
        )

    def test_negative_delta_rejected(self):
        with self.assertRaises(ValueError):
            build_potential_influence_logs(self.ds, delta=-1)

    def test_max_per_item_cap_subsamples_and_is_deterministic(self):
        ds = build_hand_dataset()
        uncapped = build_potential_influence_logs(ds, delta=10 ** 9)
        self.assertEqual(len(uncapped), 4)

        capped = build_potential_influence_logs(
            ds, delta=10 ** 9, max_per_item=1, rng=random.Random(7))
        self.assertEqual(len(capped), 2)                      # 1 per item, 2 items
        self.assertEqual(len({i for _u, _v, i in capped}), 2)
        self.assertLessEqual(set(capped), set(uncapped))

        again = build_potential_influence_logs(
            ds, delta=10 ** 9, max_per_item=1, rng=random.Random(7))
        self.assertEqual(capped, again)

        # a cap above the true per-item count is a no-op
        self.assertEqual(
            sorted(build_potential_influence_logs(
                ds, delta=10 ** 9, max_per_item=99, rng=random.Random(1))),
            sorted(uncapped),
        )

    def test_subsampling_is_uniform(self):
        # Item 0 has 3 candidate logs at delta=inf: (0,1,0), (0,2,0), (1,2,0).
        # Reservoir sampling 1 of them must pick each ~1/3 of the time.
        ds = build_hand_dataset()
        counts = {}
        trials = 1800
        for s in range(trials):
            for rec in build_potential_influence_logs(
                    ds, delta=10 ** 9, max_per_item=1, rng=random.Random(s)):
                if rec[2] == 0:
                    counts[rec] = counts.get(rec, 0) + 1
        self.assertEqual(len(counts), 3, counts)
        for rec, c in counts.items():
            self.assertGreater(c, trials * 0.28, (rec, counts))
            self.assertLess(c, trials * 0.39, (rec, counts))


class TestSplitLogs(unittest.TestCase):
    """SPEC.md Section 5.2: 60% train / 20% validation / 20% test."""

    def setUp(self):
        self.pool = [(u, v, i) for i in range(10) for u in range(10)
                     for v in range(10)]
        self.assertEqual(len(self.pool), 1000)

    def test_sizes_are_60_20_20(self):
        tr, va, te = split_logs(self.pool, random.Random(42))
        self.assertEqual(len(tr), 600)
        self.assertEqual(len(va), 200)
        self.assertEqual(len(te), 200)

    def test_parts_are_disjoint_and_cover_the_input(self):
        tr, va, te = split_logs(self.pool, random.Random(42))
        str_, sva, ste = set(tr), set(va), set(te)
        self.assertEqual(str_ & sva, set())
        self.assertEqual(sva & ste, set())
        self.assertEqual(str_ & ste, set())
        self.assertEqual(str_ | sva | ste, set(self.pool))
        self.assertEqual(len(tr) + len(va) + len(te), len(self.pool))

    def test_split_is_deterministic_and_seed_sensitive(self):
        a = split_logs(self.pool, random.Random(42))
        b = split_logs(self.pool, random.Random(42))
        self.assertEqual(a, b)
        c = split_logs(self.pool, random.Random(43))
        self.assertNotEqual(a[0], c[0])

    def test_split_does_not_mutate_the_input(self):
        before = list(self.pool)
        split_logs(self.pool, random.Random(1))
        self.assertEqual(self.pool, before)

    def test_ragged_sizes_still_partition_exactly(self):
        for n in (0, 1, 2, 3, 7, 13, 99, 101):
            pool = list(range(n))
            tr, va, te = split_logs(pool, random.Random(n))
            self.assertEqual(len(tr), int(0.6 * n), n)
            self.assertEqual(len(va), int(0.2 * n), n)
            self.assertEqual(len(tr) + len(va) + len(te), n, n)
            self.assertEqual(set(tr) | set(va) | set(te), set(pool), n)

    def test_requires_an_explicit_rng(self):
        with self.assertRaises(ValueError):
            split_logs(self.pool, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
