"""Dataset container, on-disk (de)serialisation and potential-influence logs.

Implements the canonical on-disk format of API.md and **Definition 1** of
SPEC.md (construction of the potential-influence log set ``D``) plus the
60/20/20 split of SPEC.md Section 5.2.

Standard library only.  Python 3.9 compatible.  All randomness flows through
an explicit ``random.Random`` instance supplied by the caller.

On-disk layout of ``data/processed/<name>/``::

    graph.tsv   u<TAB>v                      one directed edge per line
    logs.tsv    u<TAB>i<TAB>t                one adoption per line, sorted by t
    items.tsv   i<TAB>f1,f2,f3               attribute bag w_i (2nd field may be empty)
    meta.json   {"name","n_users","n_links","n_items","n_attrs","n_logs",
                 "source","notes"}

Each ``*.tsv`` may also be present gzip-compressed as ``*.tsv.gz``; loading
tries the plain file first and transparently falls back to the ``.gz``.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import statistics
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Dataset",
    "load_dataset",
    "save_dataset",
    "build_adjacency",
    "build_potential_influence_logs",
    "split_logs",
    "summarize",
]

# Deterministic fallback seed, used only when a caller asks for subsampling
# but forgets to hand us an rng.  Never module-level ``random.*``.
_FALLBACK_SEED = 20190101

# When scanning an item's adopters is this many times cheaper than scanning a
# user's out-neighbours, we flip the inner loop (see build_potential_influence_logs).
_FLIP_FACTOR = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """A processed dataset: directed friend graph G=(U,E), adoption logs, item attributes.

    Attributes
    ----------
    name        dataset name (matches the directory name)
    n_users     |U|; user ids are dense integers in [0, n_users)
    n_items     |M|; item ids are dense integers in [0, n_items)
    n_attrs     |F|; attribute-value ids are dense integers in [0, n_attrs)
    edges       list of (u, v) directed arcs "u influences v"
    out_adj     out_adj[u] -> list of v such that (u,v) in E
    in_adj      in_adj[v]  -> list of u such that (u,v) in E
    logs        list of (u, i, t) adoptions, t = integer unix seconds, sorted by t
    item_attrs  item_attrs[i] -> list of attribute ids, the bag w_i of Algorithm 1 line 6
    source      provenance string, round-tripped through meta.json
    notes       free-form notes, round-tripped through meta.json
    """

    name: str
    n_users: int
    n_items: int
    n_attrs: int
    edges: list  # List[Tuple[int, int]]
    out_adj: list  # List[List[int]]
    in_adj: list  # List[List[int]]
    logs: list  # List[Tuple[int, int, int]]
    item_attrs: list  # List[List[int]]
    source: str = ""
    notes: str = ""

    # lazily-built cache for edge_set(); excluded from equality/repr so that two
    # datasets compare equal regardless of whether the cache has been warmed.
    _edge_set_cache: Optional[set] = field(
        default=None, repr=False, compare=False
    )

    # -- derived sizes ------------------------------------------------------

    @property
    def n_links(self) -> int:
        """|E| — number of directed arcs."""
        return len(self.edges)

    @property
    def n_logs(self) -> int:
        """Number of adoption records (NOT potential-influence logs)."""
        return len(self.logs)

    def edge_set(self) -> set:
        """Set of directed arcs ``(u, v)``.  Built once and cached."""
        if self._edge_set_cache is None:
            self._edge_set_cache = set(self.edges)
        return self._edge_set_cache

    def out_degree(self, u: int) -> int:
        return len(self.out_adj[u])

    def in_degree(self, v: int) -> int:
        return len(self.in_adj[v])

    def __repr__(self) -> str:  # keep repr cheap for big graphs
        return (
            "Dataset(name={!r}, n_users={}, n_links={}, n_items={}, "
            "n_attrs={}, n_logs={})".format(
                self.name, self.n_users, self.n_links, self.n_items,
                self.n_attrs, self.n_logs,
            )
        )


def build_adjacency(n_users: int, edges: Sequence[Tuple[int, int]]) -> Tuple[list, list]:
    """Build ``(out_adj, in_adj)`` from a directed edge list.  O(|U| + |E|)."""
    out_adj: List[List[int]] = [[] for _ in range(n_users)]
    in_adj: List[List[int]] = [[] for _ in range(n_users)]
    for u, v in edges:
        out_adj[u].append(v)
        in_adj[v].append(u)
    return out_adj, in_adj


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _open_read(path: str):
    """Open ``path`` for text reading, transparently falling back to ``path + '.gz'``."""
    if os.path.exists(path):
        return open(path, "rt", encoding="utf-8")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", encoding="utf-8")
    raise FileNotFoundError("neither {} nor {}.gz exists".format(path, path))


def _exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(path + ".gz")


def load_dataset(path: str) -> Dataset:
    """Load a processed dataset from directory ``path`` (canonical format, API.md).

    ``meta.json`` supplies the declared cardinalities; if the data files
    reference larger ids the cardinalities are widened so that indexing by id
    is always safe.
    """
    meta_path = os.path.join(path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "rt", encoding="utf-8") as fh:
            meta = json.load(fh)
    else:
        meta = {}

    name = meta.get("name") or os.path.basename(os.path.normpath(path))

    # -- graph.tsv ----------------------------------------------------------
    edges: List[Tuple[int, int]] = []
    max_user = -1
    with _open_read(os.path.join(path, "graph.tsv")) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            u = int(parts[0])
            v = int(parts[1])
            edges.append((u, v))
            if u > max_user:
                max_user = u
            if v > max_user:
                max_user = v

    # -- logs.tsv -----------------------------------------------------------
    logs: List[Tuple[int, int, int]] = []
    max_item = -1
    with _open_read(os.path.join(path, "logs.tsv")) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            u = int(parts[0])
            i = int(parts[1])
            t = int(parts[2])
            logs.append((u, i, t))
            if u > max_user:
                max_user = u
            if i > max_item:
                max_item = i

    # -- items.tsv ----------------------------------------------------------
    raw_attrs: Dict[int, List[int]] = {}
    max_attr = -1
    if _exists(os.path.join(path, "items.tsv")):
        with _open_read(os.path.join(path, "items.tsv")) as fh:
            for line in fh:
                line = line.rstrip("\n").rstrip("\r")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                i = int(parts[0].strip())
                field_str = parts[1].strip() if len(parts) > 1 else ""
                if field_str:
                    attrs = [int(tok) for tok in field_str.split(",") if tok.strip() != ""]
                else:
                    attrs = []
                raw_attrs[i] = attrs
                if i > max_item:
                    max_item = i
                for a in attrs:
                    if a > max_attr:
                        max_attr = a

    n_users = max(int(meta.get("n_users", 0)), max_user + 1)
    n_items = max(int(meta.get("n_items", 0)), max_item + 1)
    n_attrs = max(int(meta.get("n_attrs", 0)), max_attr + 1)

    item_attrs: List[List[int]] = [[] for _ in range(n_items)]
    for i, attrs in raw_attrs.items():
        item_attrs[i] = attrs

    # canonical format promises logs sorted by t; enforce it defensively so that
    # downstream code may rely on it regardless of how the file was produced.
    logs.sort(key=lambda r: (r[2], r[0], r[1]))

    out_adj, in_adj = build_adjacency(n_users, edges)

    return Dataset(
        name=name,
        n_users=n_users,
        n_items=n_items,
        n_attrs=n_attrs,
        edges=edges,
        out_adj=out_adj,
        in_adj=in_adj,
        logs=logs,
        item_attrs=item_attrs,
        source=str(meta.get("source", "")),
        notes=str(meta.get("notes", "")),
    )


def save_dataset(ds: Dataset, path: str) -> None:
    """Write ``ds`` to directory ``path`` in the canonical format of API.md."""
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "graph.tsv"), "wt", encoding="utf-8") as fh:
        for u, v in ds.edges:
            fh.write("{}\t{}\n".format(u, v))

    # logs.tsv is defined to be sorted by t ascending
    with open(os.path.join(path, "logs.tsv"), "wt", encoding="utf-8") as fh:
        for u, i, t in sorted(ds.logs, key=lambda r: (r[2], r[0], r[1])):
            fh.write("{}\t{}\t{}\n".format(u, i, t))

    with open(os.path.join(path, "items.tsv"), "wt", encoding="utf-8") as fh:
        for i in range(ds.n_items):
            attrs = ds.item_attrs[i] if i < len(ds.item_attrs) else []
            fh.write("{}\t{}\n".format(i, ",".join(str(a) for a in attrs)))

    meta = {
        "name": ds.name,
        "n_users": ds.n_users,
        "n_links": ds.n_links,
        "n_items": ds.n_items,
        "n_attrs": ds.n_attrs,
        "n_logs": ds.n_logs,
        "source": ds.source,
        "notes": ds.notes,
    }
    with open(os.path.join(path, "meta.json"), "wt", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Definition 1 — potential-influence logs
# ---------------------------------------------------------------------------


def build_potential_influence_logs(ds: Dataset, delta: int,
                                   max_per_item: int = 0,
                                   rng=None) -> list:
    """Definition 1 (SPEC.md Section 0).  Returns the list of logs ``(u, v, i)``.

    A potential-influence log ``d = (u, v, i)`` exists iff there are adoptions
    ``(u, i, t_p)`` and ``(v, i, t_q)`` with ``0 < t_q - t_p <= delta`` and
    ``(u, v)`` a DIRECTED arc of G (u influences v along u->v).

    Repeated adoptions of the same item by the same user are collapsed to that
    user's FIRST adoption time (SPEC.md Section 8: "repeated reviews of the
    same item are dropped"), so no pair ``(u, v, i)`` is ever emitted twice.

    ``max_per_item > 0`` caps the number of logs kept per item by uniform
    subsampling (reservoir sampling, algorithm R) driven by ``rng``; ``0``
    means uncapped.

    Complexity: O(|logs| + sum_i sum_{u in adopters(i)} min(outdeg(u), |adopters(i)|)),
    i.e. within the O(sum_i sum_u outdeg(u)) budget required by API.md — never
    a quadratic scan over an item's adopters.
    """
    if delta < 0:
        raise ValueError("delta must be non-negative, got {!r}".format(delta))
    if max_per_item < 0:
        raise ValueError("max_per_item must be >= 0, got {!r}".format(max_per_item))
    if max_per_item > 0 and rng is None:
        rng = random.Random(_FALLBACK_SEED)

    out_adj = ds.out_adj

    # Sorted copies of the out-neighbour lists, so that "is v an out-neighbour
    # of u?" costs O(log outdeg(u)) in the flipped branch below.  Built lazily:
    # only paid for if the flipped branch is actually taken.
    out_sorted: List[Optional[List[int]]] = [None] * ds.n_users

    # -- group adoptions by item -------------------------------------------
    # by_item[i] : {user -> first adoption time}
    by_item: Dict[int, Dict[int, int]] = {}
    for u, i, t in ds.logs:
        bucket = by_item.get(i)
        if bucket is None:
            bucket = {}
            by_item[i] = bucket
        prev = bucket.get(u)
        if prev is None or t < prev:
            bucket[u] = t

    logs_d: List[Tuple[int, int, int]] = []

    for i in sorted(by_item.keys()):
        times = by_item[i]
        n_adopters = len(times)
        if n_adopters < 2:
            continue

        # adopters ordered by adoption time (ties broken by id) — keeps the
        # emitted order deterministic and reproducible run to run.
        adopters = sorted(times.keys(), key=lambda x: (times[x], x))

        item_logs: List[Tuple[int, int, int]] = []
        seen = 0  # number of candidate logs seen for this item (reservoir counter)
        cap = max_per_item

        for u in adopters:
            tu = times[u]
            nbrs = out_adj[u]
            deg = len(nbrs)
            if deg == 0:
                continue

            # Pick the cheaper of the two equivalent enumerations:
            #   (a) walk u's out-neighbours and test adoption   -> O(outdeg(u))
            #   (b) walk the item's adopters and test the arc   -> O(|adopters| log outdeg(u))
            # Both yield exactly the same set of (u, v, i).
            if n_adopters * _FLIP_FACTOR < deg:
                srt = out_sorted[u]
                if srt is None:
                    srt = sorted(nbrs)
                    out_sorted[u] = srt
                for v in adopters:
                    dt = times[v] - tu
                    if dt <= 0 or dt > delta:
                        continue
                    j = bisect_left(srt, v)
                    if j < len(srt) and srt[j] == v:  # Definition 1: (u,v) in E
                        seen += 1
                        if cap == 0:
                            item_logs.append((u, v, i))
                        else:
                            _reservoir_add(item_logs, (u, v, i), cap, seen, rng)
            else:
                for v in nbrs:
                    tv = times.get(v)
                    if tv is None:
                        continue
                    dt = tv - tu  # Definition 1: 0 < t_q - t_p <= Delta
                    if dt <= 0 or dt > delta:
                        continue
                    seen += 1
                    if cap == 0:
                        item_logs.append((u, v, i))
                    else:
                        _reservoir_add(item_logs, (u, v, i), cap, seen, rng)

        logs_d.extend(item_logs)

    return logs_d


def _reservoir_add(kept: List[Tuple[int, int, int]], x: Tuple[int, int, int],
                   cap: int, seen: int, rng) -> None:
    """Reservoir sampling, algorithm R: keep a uniform sample of ``cap`` items.

    ``seen`` is the 1-based index of ``x`` in the candidate stream.  After the
    stream is exhausted every candidate is in ``kept`` with probability
    ``cap / seen_total``.
    """
    if len(kept) < cap:
        kept.append(x)
        return
    j = rng.randrange(seen)
    if j < cap:
        kept[j] = x


# ---------------------------------------------------------------------------
# Section 5.2 — train / validation / test split
# ---------------------------------------------------------------------------


def split_logs(logs: list, rng) -> tuple:
    """SPEC.md Section 5.2: 60% train / 20% validation / 20% test.

    The split is a uniform random shuffle driven by ``rng`` (a
    ``random.Random`` instance), so it is reproducible for a fixed seed.  The
    friendship links are *not* split — per the paper all links belong to the
    test set as well as being used for training.

    Returns ``(train, valid, test)`` — three disjoint lists whose union is a
    permutation of ``logs``.
    """
    if rng is None:
        raise ValueError("split_logs requires an explicit random.Random instance")
    shuffled = list(logs)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(0.6 * n)
    n_valid = int(0.2 * n)
    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test = shuffled[n_train + n_valid:]
    return train, valid, test


# ---------------------------------------------------------------------------
# Summary statistics (for the dataset tables printed by the experiments)
# ---------------------------------------------------------------------------


def _stats(values: Iterable[int], prefix: str) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {
            prefix + "_mean": 0.0,
            prefix + "_median": 0.0,
            prefix + "_max": 0,
            prefix + "_min": 0,
            prefix + "_zero": 0,
        }
    return {
        prefix + "_mean": sum(vals) / len(vals),
        prefix + "_median": float(statistics.median(vals)),
        prefix + "_max": max(vals),
        prefix + "_min": min(vals),
        prefix + "_zero": sum(1 for x in vals if x == 0),
    }


def summarize(ds: Dataset) -> dict:
    """Return a flat dict of dataset statistics, for printing dataset tables.

    Always contains ``n_users``, ``n_links``, ``n_items``, ``n_attrs``,
    ``n_logs`` (the SPEC.md Section 8 dataset table columns) plus in/out degree
    statistics and a few adoption/attribute aggregates.
    """
    out: Dict[str, object] = {
        "name": ds.name,
        "n_users": ds.n_users,
        "n_links": ds.n_links,
        "n_items": ds.n_items,
        "n_attrs": ds.n_attrs,
        "n_logs": ds.n_logs,
    }

    out.update(_stats((len(a) for a in ds.out_adj), "out_deg"))
    out.update(_stats((len(a) for a in ds.in_adj), "in_deg"))

    denom = ds.n_users * (ds.n_users - 1)
    out["density"] = (ds.n_links / denom) if denom > 0 else 0.0

    # attribute bags
    attr_sizes = [len(a) for a in ds.item_attrs]
    out["n_attr_tokens"] = sum(attr_sizes)
    out["attrs_per_item_mean"] = (sum(attr_sizes) / len(attr_sizes)) if attr_sizes else 0.0
    out["n_items_no_attrs"] = sum(1 for s in attr_sizes if s == 0)

    # adoption aggregates
    adopters: Set[int] = set()
    adopted_items: Set[int] = set()
    tmin = None
    tmax = None
    for u, i, t in ds.logs:
        adopters.add(u)
        adopted_items.add(i)
        if tmin is None or t < tmin:
            tmin = t
        if tmax is None or t > tmax:
            tmax = t
    out["n_adopters"] = len(adopters)
    out["n_adopted_items"] = len(adopted_items)
    out["logs_per_item_mean"] = (ds.n_logs / len(adopted_items)) if adopted_items else 0.0
    out["logs_per_user_mean"] = (ds.n_logs / len(adopters)) if adopters else 0.0
    out["t_min"] = tmin if tmin is not None else 0
    out["t_max"] = tmax if tmax is not None else 0
    out["t_span_days"] = ((tmax - tmin) / 86400.0) if (tmin is not None and tmax is not None) else 0.0

    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import shutil
    import tempfile

    # ---- toy dataset --------------------------------------------------
    # users 0..4, directed arcs, 2 items, 3 attribute values
    edges = [(0, 1), (0, 2), (1, 2), (2, 1), (3, 0)]
    n_users = 5
    out_adj, in_adj = build_adjacency(n_users, edges)
    logs = [
        (3, 0, 90),     # item 0
        (0, 0, 100),
        (1, 0, 150),
        (2, 0, 400),
        (0, 0, 5000),   # duplicate adoption of item 0 by user 0 -> must be dropped
        (1, 1, 1000),   # item 1
        (2, 1, 1050),
        (0, 1, 1200),
    ]
    logs.sort(key=lambda r: (r[2], r[0], r[1]))
    ds = Dataset(
        name="toy",
        n_users=n_users,
        n_items=2,
        n_attrs=3,
        edges=edges,
        out_adj=out_adj,
        in_adj=in_adj,
        logs=logs,
        item_attrs=[[0, 1], []],   # item 1 deliberately has an empty bag
        source="synthetic",
        notes="self-test",
    )

    assert ds.n_links == 5 and ds.n_logs == 8
    assert ds.edge_set() == set(edges)
    assert ds.edge_set() is ds.edge_set()          # cached
    assert (0, 3) not in ds.edge_set()             # direction matters

    # ---- save / reload round-trip ------------------------------------
    tmp = tempfile.mkdtemp(prefix="ctim-dataset-selftest-")
    try:
        out_dir = os.path.join(tmp, "toy")
        save_dataset(ds, out_dir)
        for fname in ("graph.tsv", "logs.tsv", "items.tsv", "meta.json"):
            assert os.path.exists(os.path.join(out_dir, fname)), fname

        ds2 = load_dataset(out_dir)
        assert ds2.name == "toy"
        assert ds2.n_users == 5 and ds2.n_items == 2 and ds2.n_attrs == 3
        assert ds2.n_links == 5 and ds2.n_logs == 8
        assert sorted(ds2.edges) == sorted(ds.edges)
        assert ds2.logs == ds.logs
        assert ds2.item_attrs == [[0, 1], []]
        assert ds2.out_adj == ds.out_adj and ds2.in_adj == ds.in_adj
        assert ds2.source == "synthetic" and ds2.notes == "self-test"
        with open(os.path.join(out_dir, "meta.json"), "rt", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["n_links"] == 5 and meta["n_logs"] == 8

        # ---- Definition 1 ---------------------------------------------
        # item 0 times: {3:90, 0:100, 1:150, 2:400}
        #   3->0 : 100-90  =  10  <= 100  -> (3,0,0)
        #   0->1 : 150-100 =  50  <= 100  -> (0,1,0)
        #   0->2 : 400-100 = 300  >  100  x
        #   1->2 : 400-150 = 250  >  100  x
        #   2->1 : 150-400 = -250 <= 0    x
        # item 1 times: {1:1000, 2:1050, 0:1200}
        #   1->2 : 1050-1000 = 50 <= 100  -> (1,2,1)
        #   2->1 : negative               x
        #   0->1, 0->2 : negative         x
        d = build_potential_influence_logs(ds2, delta=100)
        assert sorted(d) == [(0, 1, 0), (1, 2, 1), (3, 0, 0)], d
        assert len(d) == len(set(d)), "duplicate adoption must not duplicate a log"

        # delta = 0 admits nothing (strict 0 < dt)
        assert build_potential_influence_logs(ds2, delta=0) == []

        # a wide delta admits every time-respecting arc pair
        d_wide = build_potential_influence_logs(ds2, delta=10 ** 9)
        # item 0: (0,1) (0,2) (1,2) (3,0) all time-respecting; item 1: only (1,2)
        assert sorted(d_wide) == [
            (0, 1, 0), (0, 2, 0), (1, 2, 0), (1, 2, 1), (3, 0, 0),
        ], d_wide

        # the flipped-inner-loop branch must give an identical answer: force it
        # by making user 0 very high out-degree relative to the adopter count.
        big_edges = list(edges) + [(0, x) for x in range(5, 200)]
        big_out, big_in = build_adjacency(200, big_edges)
        ds_big = Dataset("big", 200, 2, 3, big_edges, big_out, big_in,
                         ds.logs, ds.item_attrs)
        d_big = build_potential_influence_logs(ds_big, delta=10 ** 9)
        assert sorted(d_big) == sorted(d_wide), (d_big, d_wide)

        # ---- max_per_item cap -----------------------------------------
        rng = random.Random(7)
        capped = build_potential_influence_logs(ds2, delta=10 ** 9,
                                                max_per_item=1, rng=rng)
        assert len(capped) == 2, capped                     # 1 per item, 2 items
        assert len(set(i for _, _, i in capped)) == 2
        assert set(capped) <= set(d_wide)
        # deterministic for a fixed seed
        again = build_potential_influence_logs(ds2, delta=10 ** 9,
                                               max_per_item=1,
                                               rng=random.Random(7))
        first = build_potential_influence_logs(ds2, delta=10 ** 9,
                                               max_per_item=1,
                                               rng=random.Random(7))
        assert again == first
        # a cap above the true count is a no-op
        assert sorted(build_potential_influence_logs(
            ds2, delta=10 ** 9, max_per_item=99, rng=random.Random(1))) == sorted(d_wide)

        # reservoir sampling must be (approximately) uniform: item 0 has 4
        # candidate logs at delta=inf; sample 1 of them many times.
        counts: Dict[Tuple[int, int, int], int] = {}
        for s in range(4000):
            sample = build_potential_influence_logs(
                ds2, delta=10 ** 9, max_per_item=1, rng=random.Random(s))
            for rec in sample:
                if rec[2] == 0:
                    counts[rec] = counts.get(rec, 0) + 1
        assert len(counts) == 4, counts
        for rec, c in counts.items():
            assert 800 < c < 1200, (rec, c, counts)   # expected 1000 each

        # ---- split_logs -----------------------------------------------
        pool = [(u, v, i) for i in range(10) for u in range(10) for v in range(10)]
        tr, va, te = split_logs(pool, random.Random(42))
        assert len(tr) == 600 and len(va) == 200 and len(te) == 200
        assert len(set(tr) | set(va) | set(te)) == len(pool)
        assert not (set(tr) & set(va)) and not (set(va) & set(te))
        assert not (set(tr) & set(te))
        tr2, va2, te2 = split_logs(pool, random.Random(42))
        assert (tr, va, te) == (tr2, va2, te2)             # deterministic
        tr3, _, _ = split_logs(pool, random.Random(43))
        assert tr3 != tr                                    # seed actually matters
        assert split_logs([], random.Random(0)) == ([], [], [])

        # ---- summarize -------------------------------------------------
        s = summarize(ds2)
        for key in ("n_users", "n_links", "n_items", "n_attrs", "n_logs"):
            assert key in s, key
        assert s["n_users"] == 5 and s["n_links"] == 5
        assert s["n_items"] == 2 and s["n_attrs"] == 3 and s["n_logs"] == 8
        assert s["out_deg_max"] == 2 and s["out_deg_min"] == 0
        assert abs(s["out_deg_mean"] - 1.0) < 1e-12         # 5 arcs / 5 users
        assert abs(s["in_deg_mean"] - 1.0) < 1e-12
        assert s["out_deg_zero"] == 1                        # user 4 has no arcs
        assert s["n_adopters"] == 4 and s["n_adopted_items"] == 2
        assert s["n_items_no_attrs"] == 1 and s["n_attr_tokens"] == 2
        assert abs(s["density"] - 5.0 / 20.0) < 1e-12
        assert s["t_min"] == 90 and s["t_max"] == 5000

        # ---- gzip fallback --------------------------------------------
        with open(os.path.join(out_dir, "graph.tsv"), "rt", encoding="utf-8") as fh:
            raw = fh.read()
        with gzip.open(os.path.join(out_dir, "graph.tsv.gz"), "wt", encoding="utf-8") as fh:
            fh.write(raw)
        os.remove(os.path.join(out_dir, "graph.tsv"))
        ds3 = load_dataset(out_dir)
        assert sorted(ds3.edges) == sorted(ds.edges)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("ctim/dataset.py self-test: OK")


if __name__ == "__main__":
    _self_test()
