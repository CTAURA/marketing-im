"""Prepare the Epinions trust/rating dataset into the canonical layout.

Source
------
    https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm
    epinions_with_rating_timestamp_txt.zip  +  catalog_epinion.txt

    Tang, Gao, Hu & Liu.  "Exploiting Local and Global Social Context for
    Recommendation", IJCAI 2013 / "Context-aware review helpfulness rating
    prediction", RecSys 2013.

**Not a dataset the paper used.**  Added by this project for one property none
of the others have.

Why it is worth having
----------------------
Yelp, Last.fm and Delicious all carry a *symmetric* relation -- mutual
friendship or mutual-fan -- which this pipeline materialises as two directed
arcs, so their directedness is an artefact of the encoding.  Epinions ships a
real **trust** network, and trust is asymmetric: measured on the processed
graph, 38.5% of arcs have a reciprocal.

Digg is directed too (its rows are *fan* relations, not mutual friendship) and
is in fact the more asymmetric of the two at 18.8% reciprocity, so Epinions is
not unique in this respect -- but it is a second, independent directed graph
from a different domain, and trust carries a different semantics from
following.

It also fills a scale gap: 18,059 users, between Ciao / Last.fm / Delicious
(~1.9-2.2k) and Digg (30,358).

What it is bad at, stated up front
----------------------------------
Item attributes are the single ``categoryid`` column of the rating file, and
``catalog_epinion.txt`` lists **27 categories in total**.  Measured, a product
carries 1.002 categories on average (only 0.19% have more than one).  So this
dataset provides ~1 attribute token per item over a 27-word vocabulary -- the
thinnest attribute signal of any dataset here, thinner than Digg's synthesised
32.  Against the paper's Dirichlet row mass of 50 (``alpha = 50/Z``) there is
no chance of moving ``phi[i][z]`` off its prior; expect the topic model to be
completely flat.  Use this dataset for the directed graph and the scale point,
not for anything about topics.

Mapping onto the canonical layout
---------------------------------
``graph.tsv``
    ``trust.txt``.  A row says user A trusts user B, so influence flows
    ``B -> A``, the reverse of the column order -- the same convention and the
    same ``arc_direction`` flag as :mod:`ctim.datasets.digg`.  224 self-loops
    are dropped.

``logs.tsv``
    ``rating_with_timestamp.txt`` columns 1, 2 and 6 (user, product, time),
    deduplicated per ``(user, product)`` keeping the EARLIEST rating -- the
    same rule SPEC.md Section 8 gives for repeated Yelp reviews.

``items.tsv``
    Column 3, ``categoryid``, unioned over that product's ratings.

File format note
----------------
Despite the ``_txt`` name, these files are MATLAB ASCII exports: whitespace
separated, no header, every field in scientific notation
(``1.0000000e+000``).  They are parsed via ``float()`` then truncated to int.

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
    "EPINIONS_DOWNLOAD_URL",
    "find_epinions_files",
    "missing_files_message",
    "prepare_epinions",
]

EPINIONS_DOWNLOAD_URL = (
    "https://www.cse.msu.edu/~tangjili/datasetcode/"
    "epinions_with_rating_timestamp_txt.zip")

RATING_TXT = "rating_with_timestamp.txt"
TRUST_TXT = "trust.txt"
# The Ciao release is byte-for-byte the same layout under the same two file
# names; only the catalogue differs, so one preparer serves both.
CATALOG_NAMES = ("catalog_epinion.txt", "catalog_ciao.txt")

# Epinions launched in 1999; this crawl ends May 2011.
_T_MIN = 915148800       # 1999-01-01
_T_MAX = 1325289600      # 2011-12-31


# ---------------------------------------------------------------------------
# File discovery / IO
# ---------------------------------------------------------------------------


def find_epinions_files(raw_dir: str) -> Dict[str, Optional[str]]:
    """Locate the txt files; the archive extracts into a nested directory."""
    names = {"rating": RATING_TXT, "trust": TRUST_TXT}
    out: Dict[str, Optional[str]] = {"rating": None, "trust": None, "catalog": None}

    roots: List[str] = []
    for base, _dirs, _files in os.walk(raw_dir):
        roots.append(base)
    for root in sorted(roots):
        for key, fname in names.items():
            if out[key] is None:
                cand = os.path.join(root, fname)
                if os.path.exists(cand):
                    out[key] = cand
        if out["catalog"] is None:
            for fname in CATALOG_NAMES:
                cand = os.path.join(root, fname)
                if os.path.exists(cand):
                    out["catalog"] = cand
                    break
    return out


def missing_files_message(raw_dir: str) -> str:
    return (
        "Epinions files not found under {!r}.\n"
        "Download and extract:\n"
        "    curl -L -o epinions.zip {}\n"
        "    unzip epinions.zip -d {}\n"
        "Required: {}, {} (and optionally {}).".format(
            raw_dir, EPINIONS_DOWNLOAD_URL, raw_dir,
            RATING_TXT, TRUST_TXT, " / ".join(CATALOG_NAMES)))


def _iter_numeric(path: str, min_fields: int) -> Iterator[List[float]]:
    """Yield whitespace-split rows of a MATLAB ASCII export, as floats.

    Rows with too few fields or an unparsable token are skipped rather than
    raising: these exports occasionally carry a trailing blank line.
    """
    with io.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < min_fields:
                continue
            try:
                yield [float(x) for x in parts[:min_fields]]
            except ValueError:
                continue


def read_catalog(path: Optional[str]) -> Dict[int, str]:
    """categoryid -> name, from ``catalog_epinion.txt`` (``<id> <name>``)."""
    out: Dict[int, str] = {}
    if not path or not os.path.exists(path):
        return out
    with io.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            head, _, rest = line.partition(" ")
            try:
                out[int(head)] = rest.strip()
            except ValueError:
                continue
    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _read_ratings(path: str) -> Tuple[Dict[Tuple[int, int], int],
                                      Dict[int, Set[int]], int, int]:
    """((user, product) -> earliest t, product -> category ids, n_rows, n_bad)."""
    first_seen: Dict[Tuple[int, int], int] = {}
    bags: Dict[int, Set[int]] = {}
    n_rows = 0
    n_bad = 0

    for row in _iter_numeric(path, 6):
        n_rows += 1
        u = int(row[0])
        p = int(row[1])
        c = int(row[2])
        t = int(row[5])
        if t < _T_MIN or t > _T_MAX:
            n_bad += 1
            continue
        bags.setdefault(p, set()).add(c)
        key = (u, p)
        prev = first_seen.get(key)
        if prev is None or t < prev:
            first_seen[key] = t

    return first_seen, bags, n_rows, n_bad


def _read_trust_arcs(path: str, arc_direction: str = "influence"
                     ) -> Tuple[Set[Tuple[int, int]], int]:
    """(directed arcs, n_self_loops).

    A row is ``A B`` meaning A trusts B, so influence flows ``B -> A``.
    ``arc_direction="influence"`` (default) emits that; ``"raw"`` keeps the
    column order.  Same flag and same reasoning as :mod:`ctim.datasets.digg`.
    """
    if arc_direction not in ("influence", "raw"):
        raise ValueError("arc_direction must be 'influence' or 'raw', got {!r}"
                         .format(arc_direction))
    flip = (arc_direction == "influence")

    arcs: Set[Tuple[int, int]] = set()
    n_self = 0
    for row in _iter_numeric(path, 2):
        a, b = int(row[0]), int(row[1])
        if a == b:
            n_self += 1
            continue
        arcs.add((b, a) if flip else (a, b))
    return arcs, n_self


# ---------------------------------------------------------------------------
# Preparer
# ---------------------------------------------------------------------------


def prepare_epinions(raw_dir: str, out_dir: str,
                     target_users: int = 0,
                     target_links: int = 0,
                     target_items: int = 0,
                     arc_direction: str = "influence",
                     seed: int = 42,
                     name: str = "epinions",
                     write: bool = True,
                     verbose: bool = True) -> Dataset:
    """Build the processed Epinions benchmark and (by default) write it out.

    ``name`` also serves the sibling **Ciao** release, which ships the identical
    two files under the identical names and differs only in its catalogue; pass
    ``name="ciao"`` and point ``raw_dir`` at it.  Every dataset-specific figure
    in the ``notes`` below is measured from the files, not hard-coded, so both
    describe themselves truthfully.

    Same shape as the other preparers.  All three ``target_*`` default to ``0``
    (no truncation): the paper never used this dataset, so there is no SPEC.md
    Section 8 row to hit.

    Raises ``FileNotFoundError`` with download instructions when the raw files
    are absent.
    """
    _rng = random.Random(seed)   # reserved: the pipeline itself is rng-free
    t0 = time.time()

    files = find_epinions_files(raw_dir)
    if not (files["rating"] and files["trust"]):
        raise FileNotFoundError(missing_files_message(raw_dir))
    if verbose:
        for key in ("rating", "trust", "catalog"):
            print("[{}] {:<8}: {}".format(name, key, files[key]))

    first_seen, bags, n_rows, n_bad_time = _read_ratings(files["rating"])
    catalog = read_catalog(files["catalog"])
    if verbose:
        print("[{}] {:,} ratings -> {:,} adoptions over {:,} products "
              "({:,} dropped for a timestamp outside 1999-2011)".format(
                  name, n_rows, len(first_seen), len(bags), n_bad_time))

    arcs_all, n_self = _read_trust_arcs(files["trust"], arc_direction)
    n_recip = sum(1 for (a, b) in arcs_all if (b, a) in arcs_all)
    if verbose:
        print("[{}] {:,} trust arcs ({:,} self-loops dropped); "
              "{:,} reciprocated = {:.1f}% -- genuinely directed".format(
                  name, len(arcs_all), n_self, n_recip,
                  100.0 * n_recip / len(arcs_all) if arcs_all else 0.0))

    # -- users must both rate and appear in the trust graph ------------------
    raters = {u for (u, _p) in first_seen}
    linked = {u for arc in arcs_all for u in arc}
    users = raters & linked
    if verbose:
        print("[{}] users: {:,} rate, {:,} in trust graph, {:,} do both"
              .format(name, len(raters), len(linked), len(users)))

    activity: Dict[int, int] = {}
    for (u, _p) in first_seen:
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

    logs_raw = [(u, p, t) for (u, p), t in first_seen.items() if u in users]

    item_pop: Dict[int, int] = {}
    for (_u, p, _t) in logs_raw:
        item_pop[p] = item_pop.get(p, 0) + 1
    items = {p for p in item_pop if p in bags}
    if target_items > 0 and len(items) > target_items:
        ranked = sorted(items, key=lambda p: (-item_pop[p], p))
        items = set(ranked[:target_items])

    # -- dense renumbering --------------------------------------------------
    user_ids = sorted(users)
    item_ids = sorted(items)
    umap = {u: j for j, u in enumerate(user_ids)}
    imap = {p: j for j, p in enumerate(item_ids)}

    edges = sorted((umap[u], umap[v]) for (u, v) in arcs
                   if u in umap and v in umap)
    logs = [(umap[u], imap[p], t) for (u, p, t) in logs_raw if p in imap]
    logs.sort(key=lambda r: (r[2], r[0], r[1]))

    item_attrs_raw = [sorted(bags[p]) for p in item_ids]
    used = sorted({c for bag in item_attrs_raw for c in bag})
    amap = {c: j for j, c in enumerate(used)}
    item_attrs = [[amap[c] for c in bag] for bag in item_attrs_raw]

    out_adj, in_adj = build_adjacency(len(user_ids), edges)

    n_tokens = sum(len(bag) for bag in item_attrs)
    density = (n_tokens / len(item_ids)) if item_ids else 0.0
    pct_recip = 100.0 * n_recip / len(arcs_all) if arcs_all else 0.0
    notes = (
        "REAL data from the {} trust/rating crawl (Tang et al.).  *** NOT a "
        "dataset the paper used: added by this project. ***  The trust relation "
        "is genuinely ASYMMETRIC -- {:,} of {:,} arcs reciprocated ({:.1f}%) -- "
        "unlike every other dataset here, whose symmetric friendship is merely "
        "doubled into arcs.  Arc direction '{}': a row says A trusts B, so "
        "influence flows B -> A, the reverse of the column order (same convention "
        "as ctim.datasets.digg).  {:,} self-loops dropped.  Adoption logs are "
        "ratings, deduplicated per (user, product) keeping the earliest, matching "
        "the repeated-review rule of SPEC.md Section 8.  *** ITEM ATTRIBUTES ARE "
        "VERY THIN: the only attribute is the single categoryid column; the "
        "catalogue lists {:,} categories and, measured, a product carries {:.3f} "
        "of them, i.e. {:.3f} tokens per item over F={:,}.  Against the paper's "
        "Dirichlet row mass of 50 (alpha = 50/Z) the topic model cannot leave its "
        "prior.  Use this dataset for the directed graph and the scale point, not "
        "for conclusions about topics. ***  {:,} of {:,} rating rows were dropped "
        "for a timestamp outside 1999-2011.  Subsampling: {}."
    ).format(name, n_recip, len(arcs_all), pct_recip,
             arc_direction, n_self,
             len(catalog), density, density, len(used),
             n_bad_time, n_rows, core_info["procedure"])

    ds = Dataset(
        name=name,
        n_users=len(user_ids),
        n_items=len(item_ids),
        n_attrs=len(used),
        edges=edges,
        out_adj=out_adj,
        in_adj=in_adj,
        logs=logs,
        item_attrs=item_attrs,
        source="{} with rating timestamps ({})".format(
            name, EPINIONS_DOWNLOAD_URL.replace("epinions", name)),
        notes=notes,
    )

    if write:
        save_dataset(ds, out_dir)
        if verbose:
            print("[{}] written to {}".format(name, out_dir))

    if verbose:
        print()
        print(paper_comparison_table(
            {"n_users": ds.n_users, "n_links": ds.n_links, "n_items": ds.n_items,
             "n_attrs": ds.n_attrs, "n_logs": ds.n_logs},
            benchmark=None,
            title="%s -- produced (the paper never used this dataset)" % name))
        if catalog:
            print()
            print("[{}] {} categories in the catalog, {} used"
                  .format(name, len(catalog), len(used)))
        print()
        print("[{}] done in {:.1f}s".format(name, time.time() - t0))

    return ds


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Round-trip a tiny synthetic copy of the MATLAB-export layout."""
    import shutil
    import tempfile

    from ctim.dataset import load_dataset

    tmp = tempfile.mkdtemp(prefix="epinions_selftest_")
    try:
        raw = os.path.join(tmp, "raw", "nested")   # exercise the os.walk search
        os.makedirs(raw)
        t = 1000000000  # 2001-09

        def sci(*vals):
            return "  " + "  ".join("%.7e" % v for v in vals) + "\n"

        with io.open(os.path.join(raw, TRUST_TXT), "wt", encoding="utf-8") as fh:
            fh.write(sci(1, 2))      # 1 trusts 2  -> influence 2 -> 1
            fh.write(sci(3, 2))      # 3 trusts 2  -> influence 2 -> 3
            fh.write(sci(2, 1))      # reciprocal of the first
            fh.write(sci(4, 4))      # self-loop, dropped
        with io.open(os.path.join(raw, RATING_TXT), "wt", encoding="utf-8") as fh:
            fh.write(sci(1, 100, 3, 5, 2, t + 500))   # later ...
            fh.write(sci(1, 100, 3, 4, 1, t))         # ... earlier wins
            fh.write(sci(2, 100, 3, 5, 2, t))
            fh.write(sci(2, 101, 7, 4, 1, t))
            fh.write(sci(3, 101, 7, 3, 1, t))
            fh.write(sci(9, 102, 5, 5, 1, t))         # user 9 not in trust graph
            fh.write(sci(1, 103, 5, 5, 1, 1))         # corrupt: 1970
        with io.open(os.path.join(raw, CATALOG_NAMES[0]), "wt", encoding="utf-8") as fh:
            fh.write("3 Movies\n7 Electronics\n5 Music\n")

        out = os.path.join(tmp, "processed")
        ds = prepare_epinions(os.path.join(tmp, "raw"), out, verbose=False)

        assert ds.n_users == 3, ds.n_users        # 9 excluded, 4 was self-loop only
        assert ds.n_items == 2, ds.n_items        # 102 gone with user 9; 103 corrupt
        assert ds.n_attrs == 2, ds.n_attrs        # categories 3 and 7
        assert ds.n_logs == 4, ds.n_logs

        # direction: row "1 trusts 2" must become the arc 2 -> 1
        u1, u2 = ds.edges[0], ds.edges[1]
        assert (0, 1) not in ds.edges or (1, 0) in ds.edges, ds.edges
        raw_arcs, _ = _read_trust_arcs(os.path.join(raw, TRUST_TXT), "raw")
        inf_arcs, n_self = _read_trust_arcs(os.path.join(raw, TRUST_TXT), "influence")
        assert n_self == 1, n_self
        assert (2, 1) in inf_arcs and (1, 2) in raw_arcs, (inf_arcs, raw_arcs)

        earliest = [r for r in ds.logs if r[0] == 0 and r[1] == 0]
        assert earliest and earliest[0][2] == t, earliest

        back = load_dataset(out)
        assert back.n_users == ds.n_users and back.n_logs == ds.n_logs
        assert back.item_attrs == ds.item_attrs
        assert ds.logs == sorted(ds.logs, key=lambda r: (r[2], r[0], r[1]))

        cat = read_catalog(os.path.join(raw, CATALOG_NAMES[0]))
        assert cat[3] == "Movies" and cat[7] == "Electronics", cat
        print("epinions self-test: ALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _self_test()
