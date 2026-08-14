"""Prepare the HetRec 2011 Delicious 2k dataset into the canonical layout.

Source
------
    https://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip

    Cantador, Brusilovsky & Kuflik.  "2nd Workshop on Information Heterogeneity
    and Fusion in Recommender Systems (HetRec 2011)", RecSys 2011.

**Not a dataset the paper used.**  Added by this project alongside
:mod:`ctim.datasets.lastfm`, and it fixes that one's main methodological
weakness.

Why it is worth having on top of Last.fm
----------------------------------------
In ``lastfm.py`` the item attributes and the adoption log are read from the
same file, so an item's tag bag is the union of the tags its own adopters
chose.  Delicious ships a *separate* item-tag table, ``bookmark_tags.dat``, and
it is measurably NOT a re-aggregation of the sampled users' tagging:

    (bookmark, tag) pairs in user_taggedbookmarks-timestamps.dat   410,663
    (bookmark, tag) pairs in bookmark_tags.dat                     487,131
        in both                                                    189,784
        only in the tagging stream                                 220,879
        only in bookmark_tags                                      297,347

61% of ``bookmark_tags`` never appears in the sampled stream, which is what a
crawl of the *whole* Delicious population over the same URLs looks like.  So
the attributes here are genuinely independent of who adopted what, unlike
Last.fm.  It is also the denser of the two sources: 7.09 tags/item against 5.93
for the stream (the file is capped at 10 tags per URL, by weight).

Mapping onto the canonical layout
---------------------------------
``graph.tsv``
    ``user_contacts-timestamps.dat``.  The readme defines a contact as a
    *mutual fan* relation, so it is symmetric by construction; measured, all
    15,328 arcs have their reverse present (7,668 undirected pairs).

``logs.tsv``
    ``user_taggedbookmarks-timestamps.dat``, deduplicated per ``(user,
    bookmark)`` keeping the EARLIEST event -- the same rule SPEC.md Section 8
    gives for repeated Yelp reviews.  437,593 tag events collapse to 104,799
    adoptions, matching the readme's own "104799 bookmarks" count exactly.

``items.tsv``
    ``bookmark_tags.dat``: the tags assigned to each URL, one attribute column
    per distinct tag.

Consequences worth stating
--------------------------
* An item needs BOTH an adoption and a row in ``bookmark_tags.dat`` to survive.
  Unlike Last.fm, where untagged items dropped out for free because the log was
  the tag stream, here the two sources disagree and the intersection is taken.
* ``bookmark_tags.dat`` is capped at 10 tags per URL, so attribute density
  cannot exceed 10 no matter how heavily a URL was tagged.  Against the paper's
  Dirichlet row mass of 50 (``alpha = 50/Z``) that is still far too few to move
  ``phi[i][z]`` off its prior; see the ``theta`` discussion in ``lastfm.py``.
* Timestamps are clean: 0 of 437,593 fall outside 2003-2011.  The sanity window
  is kept anyway, and anything it drops is counted in ``meta.json``.
* Scale: 1,861 users, comparable to Last.fm and ~16x smaller than Digg.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import io
import os
import random
import time
from typing import Dict, Iterator, List, Optional, Set, Tuple

from ctim.dataset import Dataset, build_adjacency, save_dataset
from ctim.datasets import paper_comparison_table
from ctim.datasets.subsample import select_dense_core, thin_arcs

__all__ = [
    "DELICIOUS_DOWNLOAD_URL",
    "find_delicious_files",
    "missing_files_message",
    "prepare_delicious",
]

DELICIOUS_DOWNLOAD_URL = (
    "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-delicious-2k.zip")

CONTACTS_DAT = "user_contacts-timestamps.dat"
TAGGED_DAT = "user_taggedbookmarks-timestamps.dat"
BOOKMARK_TAGS_DAT = "bookmark_tags.dat"
TAGS_DAT = "tags.dat"

# Delicious launched in 2003 and this release was crawled in 2011.
_T_MIN = 1041379200      # 2003-01-01
_T_MAX = 1325289600      # 2011-12-31


# ---------------------------------------------------------------------------
# File discovery / IO
# ---------------------------------------------------------------------------


def find_delicious_files(raw_dir: str) -> Dict[str, Optional[str]]:
    """Locate the .dat files, either directly in ``raw_dir`` or one level down."""
    names = {"contacts": CONTACTS_DAT, "tagged": TAGGED_DAT,
             "bookmark_tags": BOOKMARK_TAGS_DAT, "tags": TAGS_DAT}
    out: Dict[str, Optional[str]] = {k: None for k in names}

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
        "HetRec 2011 Delicious files not found under {!r}.\n"
        "Download and extract:\n"
        "    curl -L -o delicious.zip {}\n"
        "    unzip delicious.zip -d {}\n"
        "Required: {}, {}, {} (and optionally {}).".format(
            raw_dir, DELICIOUS_DOWNLOAD_URL, raw_dir,
            CONTACTS_DAT, TAGGED_DAT, BOOKMARK_TAGS_DAT, TAGS_DAT))


def _iter_dat(path: str) -> Iterator[List[str]]:
    """Yield tab-split rows, header skipped.

    ``bookmarks.dat`` carries multilingual page titles and is not valid UTF-8
    throughout; decoding replaces rather than raises.  Only ids are consumed.
    """
    with io.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        first = True
        for line in fh:
            if first:
                first = False
                continue
            line = line.rstrip("\n").rstrip("\r")
            if line:
                yield line.split("\t")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _read_adoptions(path: str) -> Tuple[Dict[Tuple[int, int], int], int, int]:
    """((user, bookmark) -> earliest t, n_rows, n_bad_time)."""
    first_seen: Dict[Tuple[int, int], int] = {}
    n_rows = 0
    n_bad = 0
    for row in _iter_dat(path):
        if len(row) < 4:
            continue
        n_rows += 1
        try:
            u = int(row[0])
            b = int(row[1])
            t = int(row[3]) // 1000          # the file is in milliseconds
        except ValueError:
            n_bad += 1
            continue
        if t < _T_MIN or t > _T_MAX:
            n_bad += 1
            continue
        key = (u, b)
        prev = first_seen.get(key)
        if prev is None or t < prev:
            first_seen[key] = t
    return first_seen, n_rows, n_bad


def _read_item_tags(path: str) -> Dict[int, Set[int]]:
    """bookmark -> set of tag ids, from the population-wide item-tag table."""
    bags: Dict[int, Set[int]] = {}
    for row in _iter_dat(path):
        if len(row) < 2:
            continue
        try:
            b, g = int(row[0]), int(row[1])
        except ValueError:
            continue
        bags.setdefault(b, set()).add(g)
    return bags


def _read_contact_arcs(path: str) -> Set[Tuple[int, int]]:
    """Directed arcs.  Contacts are mutual-fan relations, hence symmetric."""
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
            arcs.add((v, u))     # explicit: the relation is symmetric
    return arcs


def _count_rows(path: Optional[str]) -> int:
    if not path or not os.path.exists(path):
        return 0
    return sum(1 for _ in _iter_dat(path))


# ---------------------------------------------------------------------------
# Preparer
# ---------------------------------------------------------------------------


def prepare_delicious(raw_dir: str, out_dir: str,
                      target_users: int = 0,
                      target_links: int = 0,
                      target_items: int = 0,
                      seed: int = 42,
                      write: bool = True,
                      verbose: bool = True) -> Dataset:
    """Build the processed Delicious benchmark and (by default) write it out.

    Same shape as :func:`ctim.datasets.lastfm.prepare_lastfm`.  All three
    ``target_*`` default to ``0`` (no truncation): there is no SPEC.md Section 8
    row to hit, because the paper never used this dataset.

    Raises ``FileNotFoundError`` with download instructions when the raw files
    are absent.
    """
    _rng = random.Random(seed)   # reserved: the pipeline itself is rng-free
    t0 = time.time()

    files = find_delicious_files(raw_dir)
    if not (files["contacts"] and files["tagged"] and files["bookmark_tags"]):
        raise FileNotFoundError(missing_files_message(raw_dir))
    if verbose:
        for key in ("contacts", "tagged", "bookmark_tags", "tags"):
            print("[delicious] {:<14}: {}".format(key, files[key]))

    first_seen, n_rows, n_bad_time = _read_adoptions(files["tagged"])
    bags = _read_item_tags(files["bookmark_tags"])
    n_tags_declared = _count_rows(files["tags"])
    if verbose:
        print("[delicious] {:,} tag events -> {:,} adoptions "
              "({:,} rows dropped for a corrupt timestamp)".format(
                  n_rows, len(first_seen), n_bad_time))
        print("[delicious] {:,} bookmarks carry tags in {}".format(
            len(bags), BOOKMARK_TAGS_DAT))

    arcs_all = _read_contact_arcs(files["contacts"])
    if verbose:
        print("[delicious] {:,} directed contact arcs".format(len(arcs_all)))

    # -- users must both adopt and have contacts ----------------------------
    adopters = {u for (u, _b) in first_seen}
    linked = {u for arc in arcs_all for u in arc}
    users = adopters & linked
    if verbose:
        print("[delicious] users: {:,} adopt, {:,} have contacts, {:,} do both"
              .format(len(adopters), len(linked), len(users)))

    activity: Dict[int, int] = {}
    for (u, _b) in first_seen:
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

    logs_raw = [(u, b, t) for (u, b), t in first_seen.items() if u in users]

    item_pop: Dict[int, int] = {}
    for (_u, b, _t) in logs_raw:
        item_pop[b] = item_pop.get(b, 0) + 1
    # An item needs BOTH an adoption and a tag row -- the two sources disagree.
    adopted = set(item_pop)
    items = adopted & set(bags)
    n_untagged = len(adopted - set(bags))
    if verbose:
        print("[delicious] items: {:,} adopted, {:,} of those tagged "
              "({:,} dropped for having no tag row)".format(
                  len(adopted), len(items), n_untagged))
    if target_items > 0 and len(items) > target_items:
        ranked = sorted(items, key=lambda b: (-item_pop[b], b))
        items = set(ranked[:target_items])

    # -- dense renumbering --------------------------------------------------
    user_ids = sorted(users)
    item_ids = sorted(items)
    umap = {u: j for j, u in enumerate(user_ids)}
    imap = {b: j for j, b in enumerate(item_ids)}

    edges = sorted((umap[u], umap[v]) for (u, v) in arcs
                   if u in umap and v in umap)
    logs = [(umap[u], imap[b], t) for (u, b, t) in logs_raw if b in imap]
    logs.sort(key=lambda r: (r[2], r[0], r[1]))

    item_attrs_raw = [sorted(bags[b]) for b in item_ids]
    used = sorted({g for bag in item_attrs_raw for g in bag})
    amap = {g: j for j, g in enumerate(used)}
    item_attrs = [[amap[g] for g in bag] for bag in item_attrs_raw]

    out_adj, in_adj = build_adjacency(len(user_ids), edges)

    n_tokens = sum(len(bag) for bag in item_attrs)
    notes = (
        "REAL data from HetRec 2011 Delicious 2k.  *** NOT a dataset the paper "
        "used: added by this project. ***  Contacts are mutual-fan relations, "
        "symmetric by construction (measured: all arcs have their reverse), "
        "materialised as two directed arcs.  Adoption logs are the tag events of "
        "user_taggedbookmarks-timestamps.dat, deduplicated per (user, bookmark) "
        "keeping the earliest, matching the repeated-review rule of SPEC.md "
        "Section 8.  Item attributes come from bookmark_tags.dat, which is NOT a "
        "re-aggregation of that stream: of its 487,131 (bookmark, tag) pairs only "
        "189,784 appear in the sampled users' tagging, so it reflects the wider "
        "Delicious population and the attributes are independent of who adopted "
        "what -- the opposite of ctim.datasets.lastfm, where both come from one "
        "file.  bookmark_tags.dat is capped at 10 tags per URL.  {:,} of the {:,} "
        "tags in tags.dat are used.  An item needs both an adoption and a tag row; "
        "{:,} adopted items were dropped for having no tag row.  {:,} of {:,} log "
        "rows were dropped for a timestamp outside 2003-2011.  Attribute density: "
        "{:,} tokens over {:,} items = {:.2f} per item.  Subsampling: {}."
    ).format(len(used), n_tags_declared, n_untagged, n_bad_time, n_rows,
             n_tokens, len(item_ids),
             (n_tokens / len(item_ids)) if item_ids else 0.0,
             core_info["procedure"])

    ds = Dataset(
        name="delicious",
        n_users=len(user_ids),
        n_items=len(item_ids),
        n_attrs=len(used),
        edges=edges,
        out_adj=out_adj,
        in_adj=in_adj,
        logs=logs,
        item_attrs=item_attrs,
        source="HetRec 2011 Delicious 2k ({})".format(DELICIOUS_DOWNLOAD_URL),
        notes=notes,
    )

    if write:
        save_dataset(ds, out_dir)
        if verbose:
            print("[delicious] written to {}".format(out_dir))

    if verbose:
        print()
        print(paper_comparison_table(
            {"n_users": ds.n_users, "n_links": ds.n_links, "n_items": ds.n_items,
             "n_attrs": ds.n_attrs, "n_logs": ds.n_logs},
            benchmark=None,
            title="Delicious -- produced (the paper never used this dataset)"))
        print()
        print("[delicious] done in {:.1f}s".format(time.time() - t0))

    return ds


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Round-trip a tiny synthetic copy of the .dat layout through the preparer."""
    import shutil
    import tempfile

    from ctim.dataset import load_dataset

    tmp = tempfile.mkdtemp(prefix="delicious_selftest_")
    try:
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        t = 1200000000000  # 2008, in ms

        with io.open(os.path.join(raw, CONTACTS_DAT), "wt", encoding="utf-8") as fh:
            fh.write("userID\tcontactID\ttimestamp\n")
            fh.write("1\t2\t{}\n2\t1\t{}\n2\t3\t{}\n3\t2\t{}\n".format(t, t, t, t))
        with io.open(os.path.join(raw, TAGS_DAT), "wt", encoding="utf-8") as fh:
            fh.write("id\tvalue\n10\tnews\n11\tblog\n12\tunused\n")
        with io.open(os.path.join(raw, BOOKMARK_TAGS_DAT), "wt", encoding="utf-8") as fh:
            fh.write("bookmarkID\ttagID\ttagWeight\n")
            fh.write("100\t10\t5\n100\t11\t2\n101\t11\t9\n")
            # 102 deliberately absent: adopted but untagged
        with io.open(os.path.join(raw, TAGGED_DAT), "wt", encoding="utf-8") as fh:
            fh.write("userID\tbookmarkID\ttagID\ttimestamp\n")
            fh.write("1\t100\t10\t{}\n".format(t + 5000))   # later event ...
            fh.write("1\t100\t11\t{}\n".format(t))          # ... earlier wins
            fh.write("2\t100\t10\t{}\n".format(t))
            fh.write("2\t101\t11\t{}\n".format(t))
            fh.write("3\t101\t11\t{}\n".format(t))
            fh.write("1\t102\t10\t{}\n".format(t))          # item with no tag row
            fh.write("9\t100\t10\t{}\n".format(t))          # user 9 has no contacts
            fh.write("1\t100\t10\t1\n")                     # corrupt: 1970

        out = os.path.join(tmp, "processed")
        ds = prepare_delicious(raw, out, verbose=False)

        assert ds.n_users == 3, ds.n_users        # 9 excluded: no contacts
        assert ds.n_items == 2, ds.n_items        # 102 dropped: no tag row
        assert ds.n_attrs == 2, ds.n_attrs        # tag 12 never assigned
        assert ds.n_links == 4, ds.n_links        # 2 undirected pairs -> 4 arcs
        assert ds.n_logs == 4, ds.n_logs          # (1,100) (2,100) (2,101) (3,101)

        # attributes come from bookmark_tags, NOT from what the adopters tagged:
        # user 2 tagged item 100 only with tag 10, yet the bag holds both.
        assert ds.item_attrs[0] == [0, 1], ds.item_attrs

        earliest = [r for r in ds.logs if r[0] == 0 and r[1] == 0]
        assert earliest and earliest[0][2] == t // 1000, earliest

        back = load_dataset(out)
        assert back.n_users == ds.n_users and back.n_logs == ds.n_logs
        assert back.item_attrs == ds.item_attrs
        assert ds.logs == sorted(ds.logs, key=lambda r: (r[2], r[0], r[1]))
        print("delicious self-test: ALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _self_test()
