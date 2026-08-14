"""Prepare the HetRec 2011 Last.fm 2k dataset into the canonical layout.

Source
------
    https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

    Cantador, Brusilovsky & Kuflik.  "2nd Workshop on Information Heterogeneity
    and Fusion in Recommender Systems (HetRec 2011)", RecSys 2011.

**This dataset is not in the paper.**  It is an addition by this project, and
the reason is specific: on both Digg and Yelp the item-dependent term of
Eq (12) degenerates.  ``theta_bar_i[c] = sum_z P(z|i) * theta[c][z]`` collapses
to ``1/Z`` for every community and every item (median spread across items
0.052%), so every item induces the same diffusion graph and a multi-item run
measures nothing.  Digg is the extreme case: its public release carries no
story category at all, so :func:`ctim.datasets.digg.derive_item_attributes`
*synthesises* attributes from vote-time profiles over a vocabulary of 32.

Last.fm has real, user-assigned tags: 11,946 of them over 12,523 tagged
artists, at a density (mean 8.76 tags/artist) comparable to Digg's 6.97
tokens/item.  Same density, 373x the vocabulary -- so items are actually
distinguishable.  If ``theta_bar_i`` still collapses here, the collapse belongs
to the model family Eq (10)-(12) rather than to the data, which is the point of
running it.

Mapping onto the canonical layout
---------------------------------
``graph.tsv``
    ``user_friends.dat``.  Friendship is undirected; the file already lists
    both directions (25,434 rows = 12,717 undirected pairs), and duplicates are
    collapsed so each direction appears exactly once.

``logs.tsv``
    ``user_taggedartists-timestamps.dat``, deduplicated per ``(user, artist)``
    keeping the EARLIEST event -- the same rule SPEC.md Section 8 gives for
    repeated Yelp reviews.  186,479 tag events collapse to 71,064 adoptions.

    ``user_artists.dat`` (the listening counts) is deliberately NOT used: it
    carries no timestamp, and the model needs adoption times.  Tagging is the
    only timestamped user-item interaction in this release.

``items.tsv``
    The set of distinct tag ids applied to that artist by anyone.  One
    attribute column per distinct tag.

Consequences worth stating
--------------------------
* An artist with no tags cannot appear in the log at all (the log IS the tag
  stream), so the 5,109 untagged artists of ``artists.dat`` drop out by
  construction rather than by a filter.  12,523 of 17,632 artists survive.
* Because attributes and adoptions come from the same stream, an item's
  attribute bag is the union of the tags its adopters chose.  The bag does not
  encode *who* adopted, so it does not leak the log, but the two are not
  independent either and any conclusion drawn here must say so.
* 5 of 186,479 rows carry a corrupt timestamp (4 predate the epoch, landing in
  1956; 1 predates 2005).  They are dropped and counted in ``meta.json``.
* Scale: 1,892 users, ~16x smaller than Digg.  This dataset can settle the
  ``theta`` question; it is not a performance benchmark.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import io
import os
import random
import time
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from ctim.dataset import Dataset, build_adjacency, save_dataset
from ctim.datasets import paper_comparison_table
from ctim.datasets.subsample import select_dense_core, thin_arcs

__all__ = [
    "LASTFM_DOWNLOAD_URL",
    "find_lastfm_files",
    "missing_files_message",
    "prepare_lastfm",
]

LASTFM_DOWNLOAD_URL = (
    "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip")

FRIENDS_DAT = "user_friends.dat"
TAGGED_DAT = "user_taggedartists-timestamps.dat"
TAGS_DAT = "tags.dat"
ARTISTS_DAT = "artists.dat"

# Sanity window for the tag timestamps.  Last.fm launched in 2002 and this
# release was crawled in May 2011; anything outside is corruption, not data.
_T_MIN = 1104537600      # 2005-01-01
_T_MAX = 1325289600      # 2011-12-31


# ---------------------------------------------------------------------------
# File discovery / IO
# ---------------------------------------------------------------------------


def find_lastfm_files(raw_dir: str) -> Dict[str, Optional[str]]:
    """Locate the .dat files, either directly in ``raw_dir`` or one level down.

    The archive extracts flat, but people often keep the zip's own folder, so
    a single nested directory is searched too.
    """
    out: Dict[str, Optional[str]] = {
        "friends": None, "tagged": None, "tags": None, "artists": None}
    names = {"friends": FRIENDS_DAT, "tagged": TAGGED_DAT,
             "tags": TAGS_DAT, "artists": ARTISTS_DAT}

    roots = [raw_dir]
    if os.path.isdir(raw_dir):
        roots += sorted(os.path.join(raw_dir, d) for d in os.listdir(raw_dir)
                        if os.path.isdir(os.path.join(raw_dir, d)))
    for root in roots:
        for key, fname in names.items():
            if out[key] is None:
                cand = os.path.join(root, fname)
                if os.path.exists(cand):
                    out[key] = cand
    return out


def missing_files_message(raw_dir: str) -> str:
    return (
        "HetRec 2011 Last.fm files not found under {!r}.\n"
        "Download and extract:\n"
        "    curl -L -o lastfm.zip {}\n"
        "    unzip lastfm.zip -d {}\n"
        "Required: {}, {}, {} (and optionally {}).".format(
            raw_dir, LASTFM_DOWNLOAD_URL, raw_dir,
            FRIENDS_DAT, TAGGED_DAT, TAGS_DAT, ARTISTS_DAT))


def _iter_dat(path: str) -> Iterator[List[str]]:
    """Yield tab-split rows, header skipped.

    ``artists.dat`` is not valid UTF-8 in every row (multilingual artist
    names), so decoding replaces rather than raises.  Only ids are consumed
    from these files, never the names, so replacement is harmless.
    """
    with io.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        first = True
        for line in fh:
            if first:                       # header: userID<TAB>friendID etc.
                first = False
                continue
            line = line.rstrip("\n").rstrip("\r")
            if line:
                yield line.split("\t")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _read_tag_stream(path: str) -> Tuple[Dict[int, Set[int]],
                                         Dict[Tuple[int, int], int],
                                         int, int]:
    """(artist -> tag ids, (user, artist) -> earliest t, n_rows, n_bad_time).

    One pass builds both the attribute bags and the adoption log, because both
    live in this one file.  Deduplication keeps the earliest event per
    ``(user, artist)``, matching the repeated-review rule of SPEC.md Section 8.
    """
    bags: Dict[int, Set[int]] = {}
    first_seen: Dict[Tuple[int, int], int] = {}
    n_rows = 0
    n_bad = 0

    for row in _iter_dat(path):
        if len(row) < 4:
            continue
        n_rows += 1
        try:
            u = int(row[0])
            a = int(row[1])
            g = int(row[2])
            t = int(row[3]) // 1000          # the file is in milliseconds
        except ValueError:
            n_bad += 1
            continue
        if t < _T_MIN or t > _T_MAX:
            n_bad += 1
            continue
        bags.setdefault(a, set()).add(g)
        key = (u, a)
        prev = first_seen.get(key)
        if prev is None or t < prev:
            first_seen[key] = t

    return bags, first_seen, n_rows, n_bad


def _read_friend_arcs(path: str) -> Set[Tuple[int, int]]:
    """Directed arcs.  The file lists both directions already; dedupe them."""
    arcs: Set[Tuple[int, int]] = set()
    for row in _iter_dat(path):
        if len(row) < 2:
            continue
        try:
            u, v = int(row[0]), int(row[1])
        except ValueError:
            continue
        if u != v:
            arcs.add((u, v))
            arcs.add((v, u))     # explicit: friendship is symmetric
    return arcs


def _count_tags(path: Optional[str]) -> int:
    if not path or not os.path.exists(path):
        return 0
    return sum(1 for _ in _iter_dat(path))


# ---------------------------------------------------------------------------
# Preparer
# ---------------------------------------------------------------------------


def prepare_lastfm(raw_dir: str, out_dir: str,
                   target_users: int = 0,
                   target_links: int = 0,
                   target_items: int = 0,
                   seed: int = 42,
                   write: bool = True,
                   verbose: bool = True) -> Dataset:
    """Build the processed Last.fm benchmark and (by default) write it out.

    Same shape as :func:`ctim.datasets.yelp.prepare_yelp`: read, restrict to
    users that both adopt and have friends, optionally take the densest active
    core / thin arcs / keep the most-adopted items, then renumber densely.

    All three ``target_*`` default to ``0`` (no truncation), because unlike
    Digg and Yelp this dataset has no SPEC.md Section 8 row to hit -- the paper
    never used it.  The knobs exist so it can be cut down for quick runs.

    Raises ``FileNotFoundError`` with download instructions when the raw files
    are absent.
    """
    _rng = random.Random(seed)   # reserved: the pipeline itself is rng-free
    t0 = time.time()

    files = find_lastfm_files(raw_dir)
    if not (files["friends"] and files["tagged"]):
        raise FileNotFoundError(missing_files_message(raw_dir))
    if verbose:
        for key in ("friends", "tagged", "tags", "artists"):
            print("[lastfm] {:<8}: {}".format(key, files[key]))

    bags, first_seen, n_rows, n_bad_time = _read_tag_stream(files["tagged"])
    n_tags_declared = _count_tags(files["tags"])
    if verbose:
        print("[lastfm] {:,} tag events -> {:,} adoptions over {:,} artists "
              "({:,} rows dropped for a corrupt timestamp)".format(
                  n_rows, len(first_seen), len(bags), n_bad_time))

    arcs_all = _read_friend_arcs(files["friends"])
    if verbose:
        print("[lastfm] {:,} directed friendship arcs".format(len(arcs_all)))

    # -- users must both adopt and have friends -----------------------------
    adopters = {u for (u, _a) in first_seen}
    linked = {u for arc in arcs_all for u in arc}
    users = adopters & linked
    if verbose:
        print("[lastfm] users: {:,} adopt, {:,} have friends, {:,} do both"
              .format(len(adopters), len(linked), len(users)))

    activity: Dict[int, int] = {}
    for (u, _a) in first_seen:
        if u in users:
            activity[u] = activity.get(u, 0) + 1

    arcs = sorted(a for a in arcs_all if a[0] in users and a[1] in users)

    # -- optional truncation (all off by default) ---------------------------
    core_info: dict = {"procedure": "no user subsampling", "k_star": 0, "n_wcc": 0}
    if target_users > 0:
        keep, core_info = select_dense_core(arcs, target_users,
                                            activity=activity, verbose=verbose)
        users = users & set(keep)
        arcs = [a for a in arcs if a[0] in users and a[1] in users]
    if target_links > 0 and len(arcs) > target_links:
        arcs = thin_arcs(arcs, target_links, weight=activity,
                         required_nodes=set(users))
        users = {u for arc in arcs for u in arc} & users

    logs_raw = [(u, a, t) for (u, a), t in first_seen.items() if u in users]

    item_pop: Dict[int, int] = {}
    for (_u, a, _t) in logs_raw:
        item_pop[a] = item_pop.get(a, 0) + 1
    items = {a for a in item_pop if a in bags}
    if target_items > 0 and len(items) > target_items:
        ranked = sorted(items, key=lambda a: (-item_pop[a], a))
        items = set(ranked[:target_items])

    # -- dense renumbering --------------------------------------------------
    user_ids = sorted(users)
    item_ids = sorted(items)
    umap = {u: j for j, u in enumerate(user_ids)}
    imap = {a: j for j, a in enumerate(item_ids)}

    edges = sorted((umap[u], umap[v]) for (u, v) in arcs
                   if u in umap and v in umap)
    logs = [(umap[u], imap[a], t) for (u, a, t) in logs_raw if a in imap]
    logs.sort(key=lambda r: (r[2], r[0], r[1]))

    item_attrs_raw = [sorted(bags[a]) for a in item_ids]
    used = sorted({g for bag in item_attrs_raw for g in bag})
    amap = {g: j for j, g in enumerate(used)}
    item_attrs = [[amap[g] for g in bag] for bag in item_attrs_raw]

    out_adj, in_adj = build_adjacency(len(user_ids), edges)

    n_tokens = sum(len(bag) for bag in item_attrs)
    notes = (
        "REAL data from HetRec 2011 Last.fm 2k.  *** NOT a dataset the paper "
        "used: added by this project to test whether the collapse of "
        "theta_bar_i to 1/Z observed on Digg and Yelp is a property of the data "
        "or of the model family Eq (10)-(12). ***  Friendship is undirected and "
        "is materialised as two directed arcs.  Adoption logs are the tag events "
        "of user_taggedartists-timestamps.dat, deduplicated per (user, artist) "
        "keeping the earliest, matching the repeated-review rule of SPEC.md "
        "Section 8; user_artists.dat is unused because it carries no timestamp.  "
        "Item attributes are the REAL user-assigned tags, one attribute per "
        "distinct tag ({:,} of the {:,} in tags.dat are used).  Because the log "
        "and the attributes come from the same stream, an item's bag is the "
        "union of the tags its adopters chose: the bag does not encode who "
        "adopted, but the two are not independent.  Artists with no tag cannot "
        "appear in the log at all, so untagged artists drop out by construction, "
        "not by a filter.  {:,} of {:,} rows were dropped for a corrupt "
        "timestamp (4 predate the epoch).  Attribute density: {:,} tokens over "
        "{:,} items = {:.2f} per item.  Subsampling: {}."
    ).format(len(used), n_tags_declared, n_bad_time, n_rows, n_tokens,
             len(item_ids), (n_tokens / len(item_ids)) if item_ids else 0.0,
             core_info["procedure"])

    ds = Dataset(
        name="lastfm",
        n_users=len(user_ids),
        n_items=len(item_ids),
        n_attrs=len(used),
        edges=edges,
        out_adj=out_adj,
        in_adj=in_adj,
        logs=logs,
        item_attrs=item_attrs,
        source="HetRec 2011 Last.fm 2k ({})".format(LASTFM_DOWNLOAD_URL),
        notes=notes,
    )

    if write:
        save_dataset(ds, out_dir)
        if verbose:
            print("[lastfm] written to {}".format(out_dir))

    if verbose:
        print()
        print(paper_comparison_table(
            {"n_users": ds.n_users, "n_links": ds.n_links, "n_items": ds.n_items,
             "n_attrs": ds.n_attrs, "n_logs": ds.n_logs},
            benchmark=None,
            title="Last.fm -- produced (the paper never used this dataset)"))
        print()
        print("[lastfm] done in {:.1f}s".format(time.time() - t0))

    return ds


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Round-trip a tiny synthetic copy of the .dat layout through the preparer."""
    import shutil
    import tempfile

    from ctim.dataset import load_dataset

    tmp = tempfile.mkdtemp(prefix="lastfm_selftest_")
    try:
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        t = 1200000000000  # 2008, in ms

        with io.open(os.path.join(raw, FRIENDS_DAT), "wt", encoding="utf-8") as fh:
            fh.write("userID\tfriendID\n")
            fh.write("1\t2\n2\t1\n2\t3\n3\t2\n")          # both directions given
        with io.open(os.path.join(raw, TAGS_DAT), "wt", encoding="utf-8") as fh:
            fh.write("tagID\ttagValue\n10\trock\n11\tjazz\n12\tunused\n")
        with io.open(os.path.join(raw, TAGGED_DAT), "wt", encoding="utf-8") as fh:
            fh.write("userID\tartistID\ttagID\ttimestamp\n")
            fh.write("1\t100\t10\t{}\n".format(t + 5000))   # later event ...
            fh.write("1\t100\t11\t{}\n".format(t))          # ... earlier wins
            fh.write("2\t100\t10\t{}\n".format(t))
            fh.write("2\t101\t11\t{}\n".format(t))
            fh.write("3\t101\t11\t{}\n".format(t))
            fh.write("9\t102\t10\t{}\n".format(t))          # user 9 has no friends
            fh.write("1\t103\t10\t-428720400000\n")         # corrupt: 1956

        out = os.path.join(tmp, "processed")
        ds = prepare_lastfm(raw, out, verbose=False)

        assert ds.n_users == 3, ds.n_users            # 9 excluded: no friends
        assert ds.n_items == 2, ds.n_items            # 102 gone with user 9, 103 corrupt
        assert ds.n_attrs == 2, ds.n_attrs            # tag 12 never used
        assert ds.n_links == 4, ds.n_links            # 2 undirected pairs -> 4 arcs
        assert ds.n_logs == 4, ds.n_logs              # (1,100) (2,100) (2,101) (3,101)

        earliest = [r for r in ds.logs if r[1] == 0 and r[0] == 0]
        assert earliest and earliest[0][2] == t // 1000, earliest

        back = load_dataset(out)
        assert back.n_users == ds.n_users and back.n_logs == ds.n_logs
        assert back.item_attrs == ds.item_attrs
        assert ds.logs == sorted(ds.logs, key=lambda r: (r[2], r[0], r[1]))
        print("lastfm self-test: ALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _self_test()
