#!/usr/bin/env python3
"""Build Digg variants that differ ONLY in attribute-token density per item.

Why this script exists (NON-PAPER ADDITION, cf. DEVIATIONS.md 4.7)
------------------------------------------------------------------
Fact (2) of this project says ``theta`` never leaves its Dirichlet prior, which
caps Eq (12) at ``pp <= max_c thetabar_i[c]``.  The chain of blame runs

    Eq (2)  phi_iz  = (n_iz + omega) / (n_i + omega*Z),  omega = 50/Z
    Eq (9)  theta_cz ~ sum_i n_ci * P(z|i) + alpha

so if ``n_i`` -- the number of *attribute tokens* item i owns -- is ~7 while the
row prior mass is 50, ``phi_i`` is pinned at ``1/Z``, ``P(z|i)`` is flat, and
Eq (9) collapses to ``theta_cz = N_c/Z + alpha`` which is flat too.  The obvious
data-side test is "give the model more attribute tokens".

The obvious KNOB IS THE WRONG KNOB.  ``prepare_data.py digg --n-hash-attrs N``
sets the *width* of the hashed-story-id block, not how many buckets each story
lands in: ``ctim.datasets.digg.derive_item_attributes`` hard-codes
``n_hash_draws = 2``, so every story gets ``5 + 2 = 7`` tokens for ANY N.
Raising N from 32 to 128 only removes the handful of self-collisions, moving
tokens/item from 6.9694 to 6.9918 (+0.32%).  It cannot move ``n_i`` and so it
cannot move ``phi``.

This script therefore varies the quantity that actually appears in Eq (2): the
number of draws.  It reads an already-prepared dataset directory, keeps
``graph.tsv`` and ``logs.tsv`` BYTE-IDENTICAL (so the graph, the communities'
link evidence and the adoption counts are untouched and every difference in the
fitted model is attributable to attributes alone), keeps the five real derived
tokens of every item (popularity / lifetime / hour-band / weekday / burstiness,
dense ids 0..29), and rewrites only the hashed-identity block with
``--draws`` md5 draws into ``--buckets`` buckets -- the same construction
``derive_item_attributes`` already uses, just not stopped at two.

HONEST CAVEAT, state it wherever these datasets are used: this raises token
DENSITY, not token INFORMATION.  A hashed story id is a per-item identity
signal with no shared semantics, so a positive result here means "the prior was
the binding constraint", and a null result means "not even unlimited token mass
moves theta", but neither means Digg has richer real attributes available.  It
does not, which is the whole reason the block is hashed in the first place.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random(seed)`` -- in fact the token construction is md5-based
and uses no RNG at all, so it is reproducible across processes.

Run:
    python -u scripts/attr_density_ab.py --dataset data/processed/digg \\
        --out data/processed/digg_d45 --draws 45 --buckets 4096
    python -u scripts/attr_density_ab.py --summarize data/processed/digg \\
        data/processed/digg32 data/processed/digg_d45
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.dataset import load_dataset, summarize          # noqa: E402

# Width of the real derived (non-hashed) attribute block after prepare_digg's
# dense renumbering: popularity decile, log2 lifetime, first-vote 3h band,
# first-vote weekday, burstiness decile.  Verified on data/processed/digg:
# every one of the 3553 items owns exactly 5 tokens with id < 30.
_N_DERIVED_DENSE = 30


def _stable_hash(*parts):
    """Deterministic 64-bit hash -- same construction as ctim.datasets.digg.

    ``hash()`` is salted per process; md5 is not, so this reproduces across
    runs and machines.
    """
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.md5(payload).digest()[:8], "big")


def read_items(path):
    """items.tsv -> list of (item_id, sorted attribute id list)."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            iid = int(parts[0])
            attrs = []
            if len(parts) > 1 and parts[1]:
                attrs = [int(x) for x in parts[1].split(",") if x != ""]
            rows.append((iid, attrs))
    return rows


def token_stats(rows):
    """(mean tokens/item, min, max, n_distinct_ids, max_id)."""
    if not rows:
        return (float("nan"), 0, 0, 0, -1)
    counts = [len(a) for _i, a in rows]
    seen = set()
    for _i, a in rows:
        seen.update(a)
    return (sum(counts) / float(len(counts)), min(counts), max(counts),
            len(seen), max(seen) if seen else -1)


def densify(rows, draws, buckets):
    """Keep the 5 real derived tokens, replace the hash block with `draws` draws.

    Mirrors ``derive_item_attributes``' hash block exactly:
        ``bag.add(base_hash + _stable_hash("digg-story", s, draw) % n_hash)``
    with ``s`` the dense item id (the original story id is not recoverable from
    a prepared directory; both are equally arbitrary identity signals).
    Duplicate draws collapse, so tokens/item is ``5 + |distinct draws|``.
    """
    out = []
    for iid, attrs in rows:
        bag = set(a for a in attrs if a < _N_DERIVED_DENSE)
        for d in range(draws):
            bag.add(_N_DERIVED_DENSE
                    + _stable_hash("digg-story", iid, d) % buckets)
        out.append((iid, sorted(bag)))
    return out


def write_variant(src, dst, rows, draws, buckets, n_attrs):
    """Write dst as a copy of src with items.tsv replaced and meta.json updated."""
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("refusing to overwrite the source dataset %r" % src)
    if not os.path.isdir(dst):
        os.makedirs(dst)
    # graph.tsv / logs.tsv copied verbatim: the ONLY difference is attributes.
    for name in ("graph.tsv", "logs.tsv"):
        shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))

    with open(os.path.join(dst, "items.tsv"), "w", encoding="utf-8",
              newline="\n") as fh:
        for iid, attrs in rows:
            fh.write("%d\t%s\n" % (iid, ",".join(str(a) for a in attrs)))

    with open(os.path.join(src, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["n_attrs"] = n_attrs
    meta["notes"] = (
        meta.get("notes", "")
        + "  ATTRIBUTE-DENSITY VARIANT (non-paper, scripts/attr_density_ab.py): "
          "graph.tsv and logs.tsv are byte-identical to the source; the five real "
          "derived tokens per item are kept; the hashed-story-id block was "
          "rebuilt with {} md5 draws into {} buckets so that the Eq (2) token "
          "count n_i rises while nothing else about the data changes.  The added "
          "tokens carry item IDENTITY, not semantics.".format(draws, buckets))
    with open(os.path.join(dst, "meta.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")


def cmd_summarize(paths):
    print("%-34s %8s %8s %10s %8s %8s"
          % ("dataset", "n_items", "F", "tok/item", "min", "max"))
    for p in paths:
        rows = read_items(os.path.join(p, "items.tsv"))
        mean_t, lo, hi, ndist, maxid = token_stats(rows)
        with open(os.path.join(p, "meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        print("%-34s %8d %8d %10.4f %8d %8d"
              % (os.path.basename(os.path.normpath(p)), len(rows),
                 meta.get("n_attrs", maxid + 1), mean_t, lo, hi))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",
                   default=os.path.join(_REPO_ROOT, "data", "processed", "digg"),
                   help="source prepared dataset directory")
    p.add_argument("--cache", default=None,
                   help="unused by this script; accepted so every driver in this "
                        "project takes the same two flags and can be re-pointed "
                        "at another dataset (e.g. Yelp) without edits")
    p.add_argument("--out", default=None,
                   help="destination directory for the density variant")
    p.add_argument("--draws", type=int, default=45,
                   help="md5 draws per item into the hash block; tokens/item "
                        "becomes 5 + |distinct draws| (default: %(default)s, "
                        "which puts n_i ~ 50 = the omega row prior mass)")
    p.add_argument("--buckets", type=int, default=4096,
                   help="width of the hash block (default: %(default)s; keep it "
                        ">> draws so self-collisions stay rare)")
    p.add_argument("--seed", type=int, default=42,
                   help="accepted for interface uniformity; the md5 construction "
                        "consumes no randomness")
    p.add_argument("--summarize", nargs="*", default=None,
                   help="just tabulate tokens/item for these dataset dirs and exit")
    args = p.parse_args(argv)

    t_all = time.perf_counter()
    if args.summarize is not None:
        rc = cmd_summarize(args.summarize or [args.dataset])
        print("\nWALL-CLOCK RUNTIME: %.2fs" % (time.perf_counter() - t_all))
        return rc

    if not args.out:
        p.error("--out is required unless --summarize is used")
    if args.draws < 1:
        p.error("--draws must be >= 1")
    if args.buckets < args.draws:
        p.error("--buckets (%d) must be >= --draws (%d)"
                % (args.buckets, args.draws))

    print("=" * 74)
    print("attr_density_ab.py -- attribute-density variant")
    print("  source : %s" % args.dataset)
    print("  dest   : %s" % args.out)
    print("  draws  : %d into %d buckets" % (args.draws, args.buckets))
    print("=" * 74)

    src_rows = read_items(os.path.join(args.dataset, "items.tsv"))
    s_mean, s_lo, s_hi, _sd, _sm = token_stats(src_rows)
    print("  source tokens/item: mean %.4f  min %d  max %d  (%d items)"
          % (s_mean, s_lo, s_hi, len(src_rows)))

    new_rows = densify(src_rows, args.draws, args.buckets)
    # Determinism check: md5, not hash(), so a second call must agree exactly.
    if densify(src_rows, args.draws, args.buckets) != new_rows:
        print("FAIL: densify() is not deterministic")
        return 1
    d_mean, d_lo, d_hi, _dd, d_max = token_stats(new_rows)
    n_attrs = max(_N_DERIVED_DENSE + args.buckets, d_max + 1)
    print("  variant tokens/item: mean %.4f  min %d  max %d   (x%.3f the source)"
          % (d_mean, d_lo, d_hi, d_mean / s_mean if s_mean else float("nan")))
    print("  declared F = %d" % n_attrs)

    write_variant(args.dataset, args.out, new_rows, args.draws, args.buckets,
                  n_attrs)
    print("  written.")

    ds = load_dataset(args.out)
    st = summarize(ds)
    print("\n  verification -- load_dataset(%r)" % args.out)
    print("    %s" % ds)
    print("    attrs/item %.4f   adopters %d   items adopted %d"
          % (st["attrs_per_item_mean"], st["n_adopters"], st["n_adopted_items"]))
    if abs(st["attrs_per_item_mean"] - d_mean) > 1e-6:
        print("FAIL: round-tripped attrs/item %.6f != written %.6f"
              % (st["attrs_per_item_mean"], d_mean))
        return 1
    for name in ("graph.tsv", "logs.tsv"):
        a = os.path.getsize(os.path.join(args.dataset, name))
        b = os.path.getsize(os.path.join(args.out, name))
        if a != b:
            print("FAIL: %s size changed (%d -> %d); the graph must be identical"
                  % (name, a, b))
            return 1
    print("    graph.tsv / logs.tsv unchanged: OK")

    print("\nWALL-CLOCK RUNTIME: %.2fs" % (time.perf_counter() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
