#!/usr/bin/env python3
"""Prepare the CTIM benchmark datasets into ``data/processed/<name>/``.

Usage
-----
::

    python3 scripts/prepare_data.py digg [--target-users N --target-links N --target-items N]
    python3 scripts/prepare_data.py yelp --raw-dir DIR
    python3 scripts/prepare_data.py synthetic --name yelp-scale \\
        --users 366715 --links 2949285 --items 61184

Every subcommand writes the canonical format of API.md (``graph.tsv`` /
``logs.tsv`` / ``items.tsv`` / ``meta.json``), loads the result back through
``ctim.dataset.load_dataset`` to prove it is readable, and prints a summary
table putting the produced counts NEXT TO the paper's reported counts
(SPEC.md Section 8):

    ===========================  =======  =========  ======
    Dataset                      Users    Links      Items
    ===========================  =======  =========  ======
    Yelp Dataset Challenge 2014  366,715  2,949,285  61,184
    Digg                          30,358     99,846   7,100
    ===========================  =======  =========  ======

The heavy lifting lives in :mod:`ctim.datasets`; this file is only the CLI.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random(seed)``.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running as a plain script from anywhere: put the repo root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.dataset import load_dataset, summarize                    # noqa: E402
from ctim.datasets import PAPER_TARGETS, paper_comparison_table     # noqa: E402
from ctim.datasets.digg import prepare_digg                         # noqa: E402
from ctim.datasets.synthetic import prepare_synthetic               # noqa: E402
from ctim.datasets.yelp import prepare_yelp                         # noqa: E402

DEFAULT_RAW = os.path.join(_REPO_ROOT, "data", "raw")
DEFAULT_PROCESSED = os.path.join(_REPO_ROOT, "data", "processed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verify(out_dir: str, verbose: bool = True) -> dict:
    """Load the written dataset back through the public API and report on it.

    This is the acceptance check: every downstream module reaches the data
    through ``ctim.dataset.load_dataset``, so if this round-trip works the
    dataset composes with the rest of the pipeline.
    """
    ds = load_dataset(out_dir)
    stats = summarize(ds)
    if verbose:
        print()
        print("verification -- load_dataset({!r})".format(out_dir))
        print("  {}".format(ds))
        print("  out-degree  mean {:.2f}  median {:.0f}  max {}  users with none {}"
              .format(stats["out_deg_mean"], stats["out_deg_median"],
                      stats["out_deg_max"], stats["out_deg_zero"]))
        print("  in-degree   mean {:.2f}  median {:.0f}  max {}"
              .format(stats["in_deg_mean"], stats["in_deg_median"],
                      stats["in_deg_max"]))
        print("  density     {:.3e}".format(stats["density"]))
        print("  adopters {:,}  items adopted {:,}  attrs/item {:.2f}"
              .format(stats["n_adopters"], stats["n_adopted_items"],
                      stats["attrs_per_item_mean"]))
        print("  adoptions per item {:.1f}  per user {:.1f}  time span {:.1f} days"
              .format(stats["logs_per_item_mean"], stats["logs_per_user_mean"],
                      stats["t_span_days"]))
    return stats


def _resolve_out(args: argparse.Namespace, default_name: str) -> str:
    out = getattr(args, "out", None)
    if out:
        return out
    name = getattr(args, "name", None) or default_name
    return os.path.join(args.processed_dir, name)


def _add_common(sp: argparse.ArgumentParser) -> None:
    """Re-expose the global options on a subparser.

    ``default=SUPPRESS`` means the option lands in the namespace only when the
    user actually types it, so the global default is not silently clobbered by
    the subparser's own default.
    """
    sp.add_argument("--out", default=argparse.SUPPRESS,
                    help="explicit output directory (overrides --processed-dir/<name>)")
    sp.add_argument("--seed", type=int, default=argparse.SUPPRESS,
                    help="seed for every random.Random used")
    sp.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS,
                    help="suppress progress output")
    sp.add_argument("--no-verify", action="store_true", default=argparse.SUPPRESS,
                    help="skip loading the result back through load_dataset()")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_digg(args: argparse.Namespace) -> int:
    out_dir = _resolve_out(args, "digg")
    prepare_digg(
        raw_dir=args.raw_dir,
        out_dir=out_dir,
        target_users=args.target_users,
        target_links=args.target_links,
        target_items=args.target_items,
        seed=args.seed,
        arc_direction=args.arc_direction,
        n_hash_attrs=args.n_hash_attrs,
        force_simulate=args.simulate_logs,
        verbose=not args.quiet,
    )
    if not args.no_verify:
        _verify(out_dir, verbose=not args.quiet)
    return 0


def cmd_yelp(args: argparse.Namespace) -> int:
    out_dir = _resolve_out(args, "yelp")
    try:
        prepare_yelp(
            raw_dir=args.raw_dir,
            out_dir=out_dir,
            target_users=args.target_users,
            target_links=args.target_links,
            target_items=args.target_items,
            seed=args.seed,
            max_graph_users=args.max_graph_users,
            verbose=not args.quiet,
        )
    except FileNotFoundError as exc:
        # Absent raw data is a user-actionable condition, not a crash: print the
        # download instructions and exit non-zero without a traceback.
        print(str(exc), file=sys.stderr)
        return 2
    if not args.no_verify:
        _verify(out_dir, verbose=not args.quiet)
    return 0


def cmd_synthetic(args: argparse.Namespace) -> int:
    n_users = args.users
    n_links = args.links
    n_comms = args.communities

    # Back-compat with the earlier `--n-comm/--per-comm` spelling of this CLI.
    per_comm = getattr(args, "per_comm", None)
    n_comm = getattr(args, "n_comm", None)
    if n_comm is not None:
        n_comms = n_comm
    if per_comm is not None:
        n_users = n_comms * per_comm
        if args.links == _SYNTH_DEFAULT_LINKS:  # user did not ask for a link budget
            # ~14% intra-community density, matching the old generator
            n_links = max(n_users, int(0.14 * per_comm * (per_comm - 1)) * n_comms)

    out_dir = _resolve_out(args, "synthetic")
    # If --out was given but --name was not, name the dataset after the directory.
    name = args.name
    if name == "synthetic" and getattr(args, "out", None):
        name = os.path.basename(os.path.normpath(args.out))

    # Compare against the paper row this stand-in is impersonating, if any.
    benchmark = args.benchmark
    if benchmark is None:
        for key, (pu, pl, pi) in PAPER_TARGETS.items():
            if (n_users, n_links, args.items) == (pu, pl, pi):
                benchmark = key
                break

    prepare_synthetic(
        out_dir=out_dir,
        name=name,
        n_users=n_users,
        n_links=n_links,
        n_items=args.items,
        n_attrs=args.attrs,
        n_comms=n_comms,
        n_topics=args.topics,
        seed=args.seed,
        benchmark=benchmark,
        verbose=not args.quiet,
    )
    if not args.no_verify:
        _verify(out_dir, verbose=not args.quiet)
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Re-print the produced-vs-paper table for an already-prepared dataset."""
    ds = load_dataset(args.dataset)
    benchmark = args.benchmark
    if benchmark is None and ds.name in PAPER_TARGETS:
        benchmark = ds.name
    print(paper_comparison_table(
        {"n_users": ds.n_users, "n_links": ds.n_links, "n_items": ds.n_items,
         "n_attrs": ds.n_attrs, "n_logs": ds.n_logs},
        benchmark=benchmark,
        title="{} -- produced vs. paper (SPEC.md Section 8)".format(ds.name)))
    if ds.source:
        print()
        print("source: " + ds.source)
    if ds.notes:
        print()
        print("notes : " + ds.notes)
    _verify(args.dataset, verbose=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SYNTH_DEFAULT_LINKS = 4000


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare_data.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", default=DEFAULT_PROCESSED,
                   help="root for outputs (default: %(default)s)")
    p.add_argument("--out", default=None,
                   help="explicit output directory, overrides --processed-dir/<name>")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for every random.Random used (default: %(default)s)")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    p.add_argument("--no-verify", action="store_true",
                   help="skip loading the result back through load_dataset()")

    sub = p.add_subparsers(dest="command", required=True)

    # -- digg ---------------------------------------------------------------
    du, dl, di = PAPER_TARGETS["digg"]
    d = sub.add_parser(
        "digg", help="prepare the Digg benchmark (SPEC.md Sec. 8, benchmark #2)",
        description="Real Digg 2009 friendship graph + story votes, subsampled to "
                    "the paper's reported {:,} users / {:,} arcs / {:,} items."
                    .format(du, dl, di))
    d.add_argument("--raw-dir", default=DEFAULT_RAW,
                   help="directory holding the raw Digg files (default: %(default)s)")
    d.add_argument("--target-users", type=int, default=du,
                   help="user budget; 0 disables subsampling "
                        "(default: %(default)s, the paper's count)")
    d.add_argument("--target-links", type=int, default=dl,
                   help="directed-arc budget; 0 disables truncation (default: %(default)s)")
    d.add_argument("--target-items", type=int, default=di,
                   help="item budget; 0 keeps every item (default: %(default)s)")
    d.add_argument("--arc-direction", choices=("influence", "raw"), default="influence",
                   help="'influence' (default) reads a raw row as 'user became a fan "
                        "of friend' and emits friend->user; 'raw' keeps column order")
    d.add_argument("--n-hash-attrs", type=int, default=32,
                   help="width of the hashed-story-id attribute block "
                        "(default: %(default)s)")
    d.add_argument("--simulate-logs", action="store_true",
                   help="force SIMULATED adoption logs even when a votes file exists "
                        "(recorded loudly in meta.json['notes'])")
    _add_common(d)
    d.set_defaults(func=cmd_digg)

    # -- yelp ---------------------------------------------------------------
    yu, yl, yi = PAPER_TARGETS["yelp"]
    y = sub.add_parser(
        "yelp", help="prepare the Yelp benchmark (SPEC.md Sec. 8, benchmark #1)",
        description="Official Yelp open dataset JSON files.  The paper's Yelp "
                    "Dataset Challenge 2014 snapshot is RETIRED and cannot be "
                    "downloaded; absolute counts will differ from the paper's "
                    "{:,} users / {:,} links / {:,} items.".format(yu, yl, yi))
    y.add_argument("--raw-dir", default=os.path.join(DEFAULT_RAW, "yelp"),
                   help="directory holding the Yelp JSON files (default: %(default)s)")
    y.add_argument("--target-users", type=int, default=yu,
                   help="user budget; 0 disables subsampling (default: %(default)s)")
    y.add_argument("--target-links", type=int, default=yl,
                   help="directed-arc budget; 0 disables truncation (default: %(default)s)")
    y.add_argument("--target-items", type=int, default=yi,
                   help="item budget; 0 keeps every item (default: %(default)s)")
    y.add_argument("--max-graph-users", type=int, default=0,
                   help="cap on how many users enter the graph stage, keeping the "
                        "most-active reviewers; 0 = auto (2x --target-users). "
                        "A memory guard for the ~906k-user / 14.6M-arc current "
                        "release (default: %(default)s)")
    _add_common(y)
    y.set_defaults(func=cmd_yelp)

    # -- synthetic ----------------------------------------------------------
    s = sub.add_parser(
        "synthetic", help="generate a SYNTHETIC dataset with planted structure",
        description="Planted community + topic generator.  NOT real data: use it "
                    "for unit tests, the --quick profile, or as an explicitly "
                    "labelled Yelp-scaled stand-in.")
    s.add_argument("--name", default="synthetic",
                   help="dataset name, also the output directory (default: %(default)s)")
    s.add_argument("--users", type=int, default=500, help="|U| (default: %(default)s)")
    s.add_argument("--links", type=int, default=_SYNTH_DEFAULT_LINKS,
                   help="|E|, distinct directed arcs (default: %(default)s)")
    s.add_argument("--items", type=int, default=200, help="|M| (default: %(default)s)")
    s.add_argument("--attrs", type=int, default=40,
                   help="|F|, attribute vocabulary size (default: %(default)s)")
    s.add_argument("--communities", type=int, default=10,
                   help="planted |C| (default: %(default)s)")
    s.add_argument("--topics", type=int, default=8,
                   help="planted |Z| (default: %(default)s)")
    s.add_argument("--benchmark", choices=sorted(PAPER_TARGETS), default=None,
                   help="paper row to compare against; inferred automatically when "
                        "the requested dimensions match one exactly")
    # deprecated spellings kept so older invocations keep working
    s.add_argument("--n-comm", type=int, default=None,
                   help=argparse.SUPPRESS)
    s.add_argument("--per-comm", type=int, default=None,
                   help=argparse.SUPPRESS)
    s.add_argument("--n-items", type=int, dest="items", default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    _add_common(s)
    s.set_defaults(func=cmd_synthetic)

    # -- summary ------------------------------------------------------------
    m = sub.add_parser(
        "summary",
        help="re-print the produced-vs-paper table for an already-prepared dataset")
    m.add_argument("dataset", help="path to data/processed/<name>")
    m.add_argument("--benchmark", choices=sorted(PAPER_TARGETS), default=None)
    m.set_defaults(func=cmd_summary)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
