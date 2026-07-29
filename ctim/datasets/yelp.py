"""Yelp benchmark preparation -- SPEC.md Section 8, benchmark #1.

The paper reports Yelp as **366,715 users / 2,949,285 links / 61,184 items**,
taken from the *Yelp Dataset Challenge 2014* snapshot.

    ============================ IMPORTANT ============================
    That snapshot is RETIRED.  Yelp no longer distributes the 2014
    release and it cannot be downloaded automatically.  This module
    reads the CURRENT official Yelp open dataset, whose dimensions are
    roughly 2-5x larger, so absolute counts CANNOT match the paper.
    The comparison table printed by the preparer always shows produced
    counts beside the paper's; ``meta.json["notes"]`` records the
    discrepancy.  Download page:  https://www.yelp.com/dataset
    ===================================================================

Inputs (any of them may be plain or ``.gz``; all are JSON-**lines**, one JSON
object per line, and are parsed strictly line by line -- ``review.json`` alone
is ~5 GB, so nothing is ever slurped whole):

``yelp_academic_dataset_business.json``
    ``business_id``, ``categories`` (a comma-separated string, or null).  Each
    distinct category string becomes one column of the binary attribute matrix
    ``A in {0,1}^(M x F)``.
``yelp_academic_dataset_review.json``
    ``user_id``, ``business_id``, ``date`` (``"YYYY-MM-DD HH:MM:SS"``).
    SPEC.md Section 8: "we assume the time at which the user reviews the item
    is the time when the item is purchased", and repeated reviews of the same
    item by the same user are dropped (earliest kept).
``yelp_academic_dataset_user.json``
    ``user_id``, ``friends`` (comma-separated user ids).  Yelp friendship is
    **undirected**, so each pair is materialised as two directed arcs, matching
    the directed ``G = (U, E)`` the model expects.

As a convenience this module also accepts the compact intermediates produced by
``scripts/stream_yelp.py`` (``businesses.tsv`` / ``reviews.tsv`` /
``friends.tsv`` / ``categories.tsv``), which exist because the official archive
ships as a 4.3 GB zip-of-tar that does not fit on disk twice.  Both paths yield
identical processed output.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    # Direct execution (`python3 ctim/datasets/<mod>.py`) has no package
    # context, so put the repo root on sys.path before importing ctim.*.
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))

import calendar
import gzip
import json
import os
import random
import time
from array import array
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from ctim.dataset import Dataset, build_adjacency, save_dataset
from ctim.datasets import PAPER_TARGETS, paper_comparison_table
from ctim.datasets.subsample import select_dense_core, thin_arcs

__all__ = [
    "YELP_PAPER_USERS",
    "YELP_PAPER_LINKS",
    "YELP_PAPER_ITEMS",
    "YELP_DOWNLOAD_URL",
    "find_yelp_files",
    "iter_json_lines",
    "parse_yelp_date",
    "missing_files_message",
    "prepare_yelp",
]

YELP_PAPER_USERS, YELP_PAPER_LINKS, YELP_PAPER_ITEMS = PAPER_TARGETS["yelp"]

YELP_DOWNLOAD_URL = "https://www.yelp.com/dataset"

BUSINESS_JSON = "yelp_academic_dataset_business.json"
REVIEW_JSON = "yelp_academic_dataset_review.json"
USER_JSON = "yelp_academic_dataset_user.json"

_STREAM_FILES = ("businesses.tsv", "reviews.tsv", "friends.tsv")


# ---------------------------------------------------------------------------
# File discovery / IO
# ---------------------------------------------------------------------------


def _find(raw_dir: str, *names: str) -> Optional[str]:
    """First existing path among ``names`` under ``raw_dir``, plain or gzipped."""
    for name in names:
        for cand in (name, name + ".gz"):
            p = os.path.join(raw_dir, cand)
            if os.path.isfile(p):
                return p
    return None


def find_yelp_files(raw_dir: str) -> Dict[str, Optional[str]]:
    """Locate the Yelp inputs under ``raw_dir``.

    Returns ``{"mode": "json"|"stream"|None, "business":..., "review":...,
    "user":..., "categories":...}``.  ``mode`` is ``None`` when neither a
    complete JSON-lines set nor a complete streamed-TSV set is present.
    """
    out: Dict[str, Optional[str]] = {
        "business": _find(raw_dir, BUSINESS_JSON),
        "review": _find(raw_dir, REVIEW_JSON),
        "user": _find(raw_dir, USER_JSON),
        "categories": None,
        "mode": None,
    }
    if out["business"] and out["review"] and out["user"]:
        out["mode"] = "json"
        return out

    # Fall back to the streamed intermediates (possibly in a ``stream/`` subdir).
    for sub in ("", "stream"):
        base = os.path.join(raw_dir, sub) if sub else raw_dir
        paths = {n: _find(base, n) for n in _STREAM_FILES}
        if all(paths.values()):
            out["business"] = paths["businesses.tsv"]
            out["review"] = paths["reviews.tsv"]
            out["user"] = paths["friends.tsv"]
            out["categories"] = _find(base, "categories.tsv")
            out["mode"] = "stream"
            return out
    return out


def missing_files_message(raw_dir: str) -> str:
    """Actionable instructions for a user who has not downloaded Yelp yet."""
    return (
        "\n"
        "Yelp raw data not found under {raw!r}.\n"
        "\n"
        "The paper used the Yelp Dataset Challenge 2014 snapshot (366,715 users /\n"
        "2,949,285 links / 61,184 items).  Yelp RETIRED that release: it is no longer\n"
        "distributed and cannot be downloaded automatically.  Use the current official\n"
        "open dataset instead -- its dimensions differ, which is expected and reported.\n"
        "\n"
        "  1. Open {url}\n"
        "  2. Accept the Dataset User Agreement and download the JSON archive.\n"
        "  3. Extract these three files into {raw}/ (gzipped copies are also fine):\n"
        "         {b}\n"
        "         {r}\n"
        "         {u}\n"
        "  4. Re-run:  python3 scripts/prepare_data.py yelp --raw-dir {raw}\n"
        "\n"
        "Alternatively, run scripts/stream_yelp.py first (it streams the 4.3 GB archive\n"
        "into businesses.tsv / reviews.tsv / friends.tsv without extracting it), or, if\n"
        "you only need a dataset of the right SHAPE for smoke tests, generate the\n"
        "clearly-labelled synthetic stand-in:\n"
        "\n"
        "  python3 scripts/prepare_data.py synthetic --name yelp-scale \\\n"
        "      --users {pu} --links {pl} --items {pi}\n"
    ).format(raw=raw_dir, url=YELP_DOWNLOAD_URL, b=BUSINESS_JSON, r=REVIEW_JSON,
             u=USER_JSON, pu=YELP_PAPER_USERS, pl=YELP_PAPER_LINKS,
             pi=YELP_PAPER_ITEMS)


def iter_json_lines(path: str) -> Iterator[dict]:
    """Yield one parsed object per line of a JSON-lines file (``.gz`` supported).

    Streaming is mandatory here: ``yelp_academic_dataset_review.json`` is about
    5 GB and ``..._user.json`` about 3 GB.  Blank lines and lines that fail to
    parse are skipped rather than aborting a multi-hour pass.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _iter_tsv(path: str) -> Iterator[List[str]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if line:
                yield line.split("\t")


def parse_yelp_date(s: str) -> int:
    """``"2016-03-09 20:56:34"`` (or just the date) -> unix seconds (UTC)."""
    if not s:
        return 0
    date_part, _, time_part = s.partition(" ")
    try:
        y, mo, d = (int(x) for x in date_part.split("-"))
    except ValueError:
        return 0
    hh = mm = ss = 0
    if time_part:
        bits = time_part.split(":")
        try:
            hh = int(bits[0])
            mm = int(bits[1]) if len(bits) > 1 else 0
            ss = int(float(bits[2])) if len(bits) > 2 else 0
        except ValueError:
            hh = mm = ss = 0
    return calendar.timegm((y, mo, d, hh, mm, ss, 0, 1, -1))


# ---------------------------------------------------------------------------
# Raw readers -- each returns compact int-keyed structures
# ---------------------------------------------------------------------------


def _read_businesses(path: str, mode: str,
                     categories_path: Optional[str]) -> Tuple[Dict[str, int],
                                                              List[List[int]],
                                                              List[str]]:
    """business_id -> dense int; per-business category-id bag; category names.

    A category string is one column of the attribute matrix ``A`` (SPEC.md
    Section 0: ``A in {0,1}^(M x F)``, ``a_i`` = item ``i``'s attribute vector).
    """
    bmap: Dict[str, int] = {}
    bags: List[List[int]] = []
    cat_ids: Dict[str, int] = {}
    cat_names: List[str] = []

    if mode == "stream":
        if categories_path:
            for row in _iter_tsv(categories_path):
                if len(row) >= 2:
                    cid = int(row[0])
                    while len(cat_names) <= cid:
                        cat_names.append("")
                    cat_names[cid] = row[1]
                    cat_ids[row[1]] = cid
        for row in _iter_tsv(path):
            bid = row[0]
            if bid in bmap:
                continue
            field = row[1] if len(row) > 1 else ""
            bag = sorted({int(x) for x in field.split(",") if x}) if field else []
            bmap[bid] = len(bags)
            bags.append(bag)
        if not cat_names:
            top = max((max(b) for b in bags if b), default=-1)
            cat_names = ["category_{}".format(j) for j in range(top + 1)]
        return bmap, bags, cat_names

    for obj in iter_json_lines(path):
        bid = obj.get("business_id")
        if not bid or bid in bmap:
            continue
        raw = obj.get("categories") or ""
        if isinstance(raw, list):
            names = [c.strip() for c in raw if c and c.strip()]
        else:
            names = [c.strip() for c in raw.split(",") if c.strip()]
        bag: Set[int] = set()
        for nm in names:
            cid = cat_ids.get(nm)
            if cid is None:
                cid = len(cat_names)
                cat_ids[nm] = cid
                cat_names.append(nm)
            bag.add(cid)
        bmap[bid] = len(bags)
        bags.append(sorted(bag))
    return bmap, bags, cat_names


def _read_reviews(path: str, mode: str, bmap: Dict[str, int],
                  verbose: bool) -> Tuple[Dict[str, int], array, array, array]:
    """Stream reviews into three parallel arrays, de-duplicated by (user, item).

    Memory discipline: 6.7M reviews as Python tuples would cost ~1 GB, so ids
    are interned to ints on the fly and stored in ``array`` buffers (4/8 bytes
    per value).  De-duplication keeps the EARLIEST review of an item by a user
    (SPEC.md Section 8) and is done by sorting an index permutation rather than
    by a 6.7M-entry dict.
    """
    umap: Dict[str, int] = {}
    ru = array("i")
    ri = array("i")
    rt = array("q")

    n_seen = 0
    if mode == "stream":
        rows: Iterator[Sequence[str]] = _iter_tsv(path)
        for row in rows:
            if len(row) < 3:
                continue
            n_seen += 1
            bid = bmap.get(row[1])
            if bid is None:
                continue
            uid = umap.get(row[0])
            if uid is None:
                uid = len(umap)
                umap[row[0]] = uid
            ru.append(uid)
            ri.append(bid)
            rt.append(int(row[2]))
    else:
        for obj in iter_json_lines(path):
            n_seen += 1
            bid = bmap.get(obj.get("business_id"))
            if bid is None:
                continue
            key = obj.get("user_id")
            if not key:
                continue
            uid = umap.get(key)
            if uid is None:
                uid = len(umap)
                umap[key] = uid
            ru.append(uid)
            ri.append(bid)
            # "we assume the review time is the purchase time" (SPEC.md Sec. 8)
            rt.append(parse_yelp_date(obj.get("date") or ""))
    if verbose:
        print("[yelp] {:,} review lines read, {:,} usable, {:,} reviewers"
              .format(n_seen, len(ru), len(umap)))

    # de-duplicate (user, item) keeping the earliest timestamp
    order = sorted(range(len(ru)), key=lambda k: (ru[k], ri[k], rt[k]))
    du = array("i")
    di = array("i")
    dt = array("q")
    prev_u = -1
    prev_i = -1
    for k in order:
        u = ru[k]
        i = ri[k]
        if u == prev_u and i == prev_i:
            continue  # a later review of the same item by the same user
        prev_u, prev_i = u, i
        du.append(u)
        di.append(i)
        dt.append(rt[k])
    if verbose:
        print("[yelp] {:,} adoptions after dropping repeated reviews"
              .format(len(du)))
    return umap, du, di, dt


def _read_friend_arcs(path: str, mode: str, umap: Dict[str, int],
                      allowed: Optional[Set[int]], n_ids: int,
                      verbose: bool) -> List[Tuple[int, int]]:
    """Undirected Yelp friendships -> two directed arcs each, reviewers only.

    Users who never reviewed anything cannot appear in any potential-influence
    log (Definition 1), so arcs touching them are dropped immediately -- that
    is what keeps this pass linear in memory over a 3 GB file.  ``allowed``, if
    given, restricts further to that set of user ids (the activity prefilter of
    :func:`prepare_yelp`).

    Pairs are accumulated as *packed* canonical ints (``min * n_ids + max``)
    rather than tuples.  On the current Yelp release that is 7.3M ints instead
    of 14.6M tuples -- roughly 450 MB instead of 1.6 GB.
    """
    pairs: Set[int] = set()
    n_lines = 0

    def add(a: int, b: int) -> None:
        if a == b:
            return
        if allowed is not None and (a not in allowed or b not in allowed):
            return
        x, y = (a, b) if a < b else (b, a)
        pairs.add(x * n_ids + y)

    if mode == "stream":
        for row in _iter_tsv(path):
            n_lines += 1
            if len(row) < 2:
                continue
            a = umap.get(row[0])
            if a is None or (allowed is not None and a not in allowed):
                continue
            for f in row[1].split(","):
                b = umap.get(f)
                if b is not None:
                    add(a, b)
    else:
        for obj in iter_json_lines(path):
            n_lines += 1
            a = umap.get(obj.get("user_id"))
            if a is None or (allowed is not None and a not in allowed):
                continue
            raw = obj.get("friends") or ""
            names = raw if isinstance(raw, list) else raw.split(",")
            for f in names:
                f = f.strip()
                if not f:
                    continue
                b = umap.get(f)
                if b is not None:
                    add(a, b)

    # Yelp friendship is undirected: materialise each pair as two directed arcs.
    arcs: List[Tuple[int, int]] = []
    for key in pairs:
        x, y = divmod(key, n_ids)
        arcs.append((x, y))
        arcs.append((y, x))
    del pairs
    arcs.sort()
    if verbose:
        print("[yelp] {:,} user records scanned, {:,} directed arcs between reviewers"
              .format(n_lines, len(arcs)))
    return arcs


# ---------------------------------------------------------------------------
# The preparer
# ---------------------------------------------------------------------------


def prepare_yelp(raw_dir: str, out_dir: str,
                 target_users: int = YELP_PAPER_USERS,
                 target_links: int = YELP_PAPER_LINKS,
                 target_items: int = YELP_PAPER_ITEMS,
                 seed: int = 42,
                 max_graph_users: int = 0,
                 write: bool = True,
                 verbose: bool = True) -> Dataset:
    """Build the processed Yelp benchmark and (by default) write it to ``out_dir``.

    Same shape of pipeline as :func:`ctim.datasets.digg.prepare_digg`: read,
    restrict to active users, take the densest active core down to
    ``target_users``, thin arcs to ``target_links``, keep the ``target_items``
    most-reviewed businesses, then renumber to dense ids.  Any ``target_* <= 0``
    disables that truncation step.

    ``max_graph_users`` caps how many users enter the graph stage, keeping the
    most-active reviewers (ties by ascending id).  ``0`` means "auto": twice
    ``target_users`` when a user target is set, uncapped otherwise.  The current
    Yelp release has ~906k users with friends and ~14.6M arcs between them;
    building a dict-of-sets adjacency over all of that costs several GB in pure
    Python, and since only ``target_users`` of them survive anyway, the cap is
    both a memory guard and the same "most-active core" criterion the rest of
    the pipeline applies.  At 2x headroom it drops only users with a single
    review, so the k-core still has plenty of choice.

    Raises ``FileNotFoundError`` with actionable download instructions when the
    raw files are absent -- see :func:`missing_files_message`.
    """
    _rng = random.Random(seed)  # reserved: the pipeline itself is rng-free
    t0 = time.time()

    files = find_yelp_files(raw_dir)
    if not files["mode"]:
        raise FileNotFoundError(missing_files_message(raw_dir))
    mode = files["mode"]
    if verbose:
        print("[yelp] mode     : {}".format(
            "official JSON-lines" if mode == "json" else "streamed TSV intermediates"))
        for key in ("business", "review", "user"):
            print("[yelp] {:<9}: {}".format(key, files[key]))
        if mode == "json":
            print("[yelp] NOTE: the paper used the RETIRED Yelp Challenge 2014 snapshot; "
                  "the current release differs in size.")

    bmap, biz_bags, cat_names = _read_businesses(files["business"], mode,
                                                 files["categories"])
    if verbose:
        print("[yelp] {:,} businesses, {:,} distinct categories ({:.1f}s)"
              .format(len(bmap), len(cat_names), time.time() - t0))

    umap, ru, ri, rt = _read_reviews(files["review"], mode, bmap, verbose)

    activity: Dict[int, int] = {}
    for k in range(len(ru)):
        u = ru[k]
        activity[u] = activity.get(u, 0) + 1

    # -- activity prefilter (memory guard; see the docstring) ---------------
    cap = max_graph_users
    if cap <= 0:
        cap = 2 * target_users if target_users > 0 else 0
    allowed: Optional[Set[int]] = None
    n_prefiltered = 0
    if cap > 0 and len(activity) > cap:
        ranked = sorted(activity, key=lambda u: (-activity[u], u))
        allowed = set(ranked[:cap])
        n_prefiltered = len(activity) - len(allowed)
        if verbose:
            print("[yelp] activity prefilter: keeping the {:,} most-active reviewers "
                  "of {:,} (>= {} reviews each)"
                  .format(cap, len(activity), activity[ranked[cap - 1]]))

    arcs = _read_friend_arcs(files["user"], mode, umap, allowed, len(umap), verbose)
    if not arcs:
        raise ValueError("no friendship arcs between reviewing users -- "
                         "check that the user file matches the review file")

    # -- active users -------------------------------------------------------
    graph_users = {x for arc in arcs for x in arc}
    active = graph_users & set(activity)
    active_arcs = [(u, v) for (u, v) in arcs if u in active and v in active]
    if verbose:
        print("[yelp] active users (reviewing AND with friends): {:,}; arcs {:,}"
              .format(len(active), len(active_arcs)))

    selected, core_info = select_dense_core(active_arcs, target_users,
                                            activity=activity, verbose=verbose)
    kept_arcs = [(u, v) for (u, v) in active_arcs if u in selected and v in selected]
    kept_arcs = thin_arcs(kept_arcs, target_links, weight=activity,
                          required_nodes=selected)
    users_with_arcs = {x for arc in kept_arcs for x in arc}
    if users_with_arcs:
        selected = selected & users_with_arcs
        kept_arcs = [(u, v) for (u, v) in kept_arcs
                     if u in selected and v in selected]
    if verbose:
        print("[yelp] core: k*={} WCC={:,} -> {:,} users, {:,} arcs"
              .format(core_info["k_star"], core_info["n_wcc"],
                      len(selected), len(kept_arcs)))

    # -- items --------------------------------------------------------------
    item_count: Dict[int, int] = {}
    for k in range(len(ru)):
        if ru[k] in selected:
            item_count[ri[k]] = item_count.get(ri[k], 0) + 1
    if target_items > 0 and len(item_count) > target_items:
        ranked = sorted(item_count, key=lambda i: (-item_count[i], i))
        keep_items = set(ranked[:target_items])
    else:
        keep_items = set(item_count)
    if verbose:
        print("[yelp] items kept: {:,}".format(len(keep_items)))

    # -- renumber -----------------------------------------------------------
    user_ids = sorted(selected)
    umap2 = {u: j for j, u in enumerate(user_ids)}
    item_ids = sorted(keep_items)
    imap2 = {i: j for j, i in enumerate(item_ids)}

    edges = sorted((umap2[u], umap2[v]) for u, v in kept_arcs)
    logs: List[Tuple[int, int, int]] = []
    for k in range(len(ru)):
        u = umap2.get(ru[k])
        if u is None:
            continue
        i = imap2.get(ri[k])
        if i is None:
            continue
        logs.append((u, i, int(rt[k])))
    logs.sort(key=lambda r: (r[2], r[0], r[1]))

    item_attrs = [biz_bags[i] for i in item_ids]
    used = sorted({a for bag in item_attrs for a in bag})
    amap = {a: j for j, a in enumerate(used)}
    item_attrs = [[amap[a] for a in bag] for bag in item_attrs]

    out_adj, in_adj = build_adjacency(len(user_ids), edges)

    notes = (
        "REAL data from the official Yelp open dataset.  *** NOT the Yelp Dataset "
        "Challenge 2014 snapshot the paper used: Yelp retired that release and no "
        "longer distributes it, so absolute counts cannot match SPEC.md Section 8 "
        "(366,715 users / 2,949,285 links / 61,184 items). ***  Friendship is "
        "undirected in Yelp and is materialised as two directed arcs.  Review time is "
        "taken as purchase time and repeated reviews of the same item by the same user "
        "are dropped, keeping the earliest (SPEC.md Section 8).  Item attributes are "
        "the REAL business 'categories' strings, one attribute per distinct category.  "
        "Subsampled to the paper's dimensions by: restrict to users that both review "
        "and have friends; {}; per-source-quota arc thinning; keep the most-reviewed "
        "businesses.  k*={}, largest WCC={}.  Source mode: {}.{}"
    ).format(core_info["procedure"], core_info["k_star"], core_info["n_wcc"], mode,
             ("  Activity prefilter: {:,} least-active reviewers were excluded before "
              "the graph stage (cap {:,} = 2x the user target), a memory guard that "
              "also applies the same most-active-core criterion.".format(
                  n_prefiltered, cap) if n_prefiltered else ""))

    ds = Dataset(
        name="yelp",
        n_users=len(user_ids),
        n_items=len(item_ids),
        n_attrs=len(used),
        edges=edges,
        out_adj=out_adj,
        in_adj=in_adj,
        logs=logs,
        item_attrs=item_attrs,
        source="Yelp open dataset ({} files, {})".format(mode, YELP_DOWNLOAD_URL),
        notes=notes,
    )

    if write:
        save_dataset(ds, out_dir)
        if verbose:
            print("[yelp] written to {}".format(out_dir))

    if verbose:
        print()
        print(paper_comparison_table(
            {"n_users": ds.n_users, "n_links": ds.n_links, "n_items": ds.n_items,
             "n_attrs": ds.n_attrs, "n_logs": ds.n_logs},
            benchmark="yelp",
            title="Yelp -- produced vs. paper (SPEC.md Section 8)"))
        print()
        print("!! The paper's Yelp Challenge 2014 snapshot is RETIRED; this is the "
              "current release. Differences are expected.")
        print("[yelp] done in {:.1f}s".format(time.time() - t0))

    return ds


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import shutil
    import tempfile

    assert parse_yelp_date("1970-01-01 00:00:00") == 0
    assert parse_yelp_date("2016-03-09 20:56:34") == 1457556994
    assert parse_yelp_date("2016-03-09") == 1457481600
    assert parse_yelp_date("") == 0
    assert parse_yelp_date("not-a-date") == 0

    tmp = tempfile.mkdtemp(prefix="ctim-yelp-selftest-")
    try:
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)

        businesses = [
            {"business_id": "b0", "categories": "Bars, Nightlife"},
            {"business_id": "b1", "categories": "Bars, Pizza"},
            {"business_id": "b2", "categories": None},
            {"business_id": "b3", "categories": "Pizza"},
        ]
        reviews = [
            {"user_id": "u0", "business_id": "b0", "date": "2015-01-01 00:00:00"},
            {"user_id": "u1", "business_id": "b0", "date": "2015-01-01 01:00:00"},
            {"user_id": "u2", "business_id": "b0", "date": "2015-01-01 02:00:00"},
            {"user_id": "u0", "business_id": "b1", "date": "2015-01-02 00:00:00"},
            {"user_id": "u1", "business_id": "b1", "date": "2015-01-02 01:00:00"},
            {"user_id": "u3", "business_id": "b3", "date": "2015-01-03 00:00:00"},
            {"user_id": "u2", "business_id": "b3", "date": "2015-01-03 05:00:00"},
            # repeated review of b0 by u0, later -> must be dropped
            {"user_id": "u0", "business_id": "b0", "date": "2016-01-01 00:00:00"},
            # review of an unknown business -> ignored
            {"user_id": "u0", "business_id": "nope", "date": "2016-01-01 00:00:00"},
        ]
        users = [
            {"user_id": "u0", "friends": "u1, u2"},
            {"user_id": "u1", "friends": "u0, u3"},
            {"user_id": "u2", "friends": "u0"},
            {"user_id": "u3", "friends": "u1"},
            {"user_id": "u9", "friends": "u0"},   # never reviewed -> dropped
        ]

        def dump(name: str, rows: List[dict]) -> None:
            with open(os.path.join(raw, name), "wt", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")

        dump(BUSINESS_JSON, businesses)
        dump(REVIEW_JSON, reviews)
        dump(USER_JSON, users)

        files = find_yelp_files(raw)
        assert files["mode"] == "json", files

        out = os.path.join(tmp, "processed", "yelp")
        ds = prepare_yelp(raw, out, target_users=0, target_links=0, target_items=0,
                          seed=1, verbose=False)

        assert ds.n_users == 4, ds.n_users            # u9 never reviewed
        assert ds.n_items == 3, ds.n_items            # b2 never reviewed
        # 8 raw usable reviews - 1 repeat = 7
        assert ds.n_logs == 7, ds.n_logs
        assert ds.logs == sorted(ds.logs, key=lambda r: (r[2], r[0], r[1]))
        # undirected friendship -> symmetric arc set
        es = ds.edge_set()
        assert all((v, u) in es for u, v in ds.edges), "arcs must be symmetric"
        # distinct undirected pairs among reviewers: u0-u1, u0-u2, u1-u3 (u9
        # never reviewed, so u9-u0 is dropped) -> 3 pairs x 2 directed arcs
        assert ds.n_links == 6, ds.n_links
        # attributes are real categories: Bars/Nightlife/Pizza, b2 excluded
        assert ds.n_attrs == 3, ds.n_attrs
        assert all(0 <= a < ds.n_attrs for bag in ds.item_attrs for a in bag)
        assert sorted(len(b) for b in ds.item_attrs) == [1, 2, 2]

        from ctim.dataset import load_dataset
        back = load_dataset(out)
        assert back.n_users == ds.n_users and back.n_links == ds.n_links
        assert back.logs == ds.logs and back.item_attrs == ds.item_attrs
        assert "RETIRED" in back.notes or "retired" in back.notes

        # gzipped inputs must work identically
        gz_raw = os.path.join(tmp, "raw_gz")
        os.makedirs(gz_raw)
        for name in (BUSINESS_JSON, REVIEW_JSON, USER_JSON):
            with open(os.path.join(raw, name), "rb") as src:
                with gzip.open(os.path.join(gz_raw, name + ".gz"), "wb") as dst:
                    dst.write(src.read())
        assert find_yelp_files(gz_raw)["mode"] == "json"
        ds_gz = prepare_yelp(gz_raw, out, target_users=0, target_links=0,
                             target_items=0, seed=1, write=False, verbose=False)
        assert ds_gz.edges == ds.edges and ds_gz.logs == ds.logs
        assert ds_gz.item_attrs == ds.item_attrs

        # streamed-TSV mode must agree with JSON mode
        st = os.path.join(tmp, "raw_stream", "stream")
        os.makedirs(st)
        cat_ids = {"Bars": 0, "Nightlife": 1, "Pizza": 2}
        with open(os.path.join(st, "categories.tsv"), "wt", encoding="utf-8") as fh:
            for nm, cid in sorted(cat_ids.items(), key=lambda kv: kv[1]):
                fh.write("{}\t{}\n".format(cid, nm))
        with open(os.path.join(st, "businesses.tsv"), "wt", encoding="utf-8") as fh:
            for b in businesses:
                raw_c = b["categories"] or ""
                ids = [str(cat_ids[c.strip()]) for c in raw_c.split(",") if c.strip()]
                fh.write("{}\t{}\n".format(b["business_id"], ",".join(ids)))
        with open(os.path.join(st, "reviews.tsv"), "wt", encoding="utf-8") as fh:
            for r in reviews:
                fh.write("{}\t{}\t{}\n".format(r["user_id"], r["business_id"],
                                               parse_yelp_date(r["date"])))
        with open(os.path.join(st, "friends.tsv"), "wt", encoding="utf-8") as fh:
            for u in users:
                fh.write("{}\t{}\n".format(
                    u["user_id"], ",".join(f.strip() for f in u["friends"].split(","))))

        sroot = os.path.join(tmp, "raw_stream")
        assert find_yelp_files(sroot)["mode"] == "stream"
        ds_s = prepare_yelp(sroot, out, target_users=0, target_links=0,
                            target_items=0, seed=1, write=False, verbose=False)
        assert ds_s.n_users == ds.n_users and ds_s.n_items == ds.n_items
        assert ds_s.n_logs == ds.n_logs and ds_s.edges == ds.edges
        assert ds_s.item_attrs == ds.item_attrs, (ds_s.item_attrs, ds.item_attrs)

        # truncation binds
        ds_t = prepare_yelp(raw, out, target_users=3, target_links=4, target_items=2,
                            seed=1, write=False, verbose=False)
        assert ds_t.n_users <= 3 and ds_t.n_links <= 4 and ds_t.n_items <= 2

        # absent files -> actionable error
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        try:
            prepare_yelp(empty, out, verbose=False)
        except FileNotFoundError as exc:
            msg = str(exc)
            assert YELP_DOWNLOAD_URL in msg and "retired" in msg.lower()
            assert BUSINESS_JSON in msg and "synthetic" in msg
        else:
            raise AssertionError("expected FileNotFoundError for an empty raw dir")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("ctim/datasets/yelp.py self-test: OK")


if __name__ == "__main__":
    _self_test()
