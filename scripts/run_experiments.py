#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line driver for the CTIM figure reproduction (API.md).

    python3 scripts/run_experiments.py --dataset data/processed/digg \\
        --figures 2,4,5 --C 100 --Z 8 --seed 42 --out results/digg

    python3 scripts/run_experiments.py --dataset data/processed/synthetic_small \\
        --quick

Produces, in ``--out`` (default ``results/<dataset name>/``):

    fig{2,3}a_spread_vs_K.svg        influence spread vs K, 5 methods
    fig{2,3}b_time_vs_K.svg          running time vs K, log scale, 5 methods
    fig{2,3}_spread_and_time_vs_K.csv
    fig4_Z_spread.svg / fig4_Z_time.svg / fig4_Z_sweep.csv
    fig5_C_spread.svg / fig5_C_time.svg / fig5_C_sweep.csv
    table2_feature_comparison.csv
    report.md                        our numbers beside the paper's, plus the
                                     REPRODUCED / NOT REPRODUCED verdicts

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make `import ctim` work when the script is run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json

from ctim.experiments import (ALL_METHODS, PAPER_C_GRID, PAPER_K_LIST,
                              PAPER_Z_GRID, ExperimentConfig, check_claims,
                              quick_config, run_experiments, scale_C_to_dataset)


def _meta_n_users(dataset_dir):
    """Read n_users out of a processed dataset's meta.json (0 if unreadable)."""
    try:
        with open(os.path.join(dataset_dir, "meta.json")) as fh:
            return int(json.load(fh).get("n_users", 0))
    except Exception:
        return 0


def _int_list(text: str, what: str):
    """Parse '1,11,21' into (1, 11, 21)."""
    out = []
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s: %r is not an integer (expected a comma-separated list)"
                % (what, chunk))
    if not out:
        raise argparse.ArgumentTypeError("%s: empty list" % what)
    return tuple(out)


def _parse_figures(text: str):
    """'2,4,5' | 'all' -> tuple of figure numbers.

    2 and 3 name the same panel pair (the paper uses Fig 2 for Yelp and Fig 3
    for Digg); either spelling selects the K sweep, and the report picks the
    matching paper table from the dataset name.
    """
    if str(text).strip().lower() in ("all", "*"):
        return (2, 4, 5)
    figs = _int_list(text, "--figures")
    bad = [f for f in figs if f not in (2, 3, 4, 5)]
    if bad:
        raise argparse.ArgumentTypeError(
            "--figures: %s is not one of 2, 3, 4, 5 (or 'all')"
            % ", ".join(str(b) for b in bad))
    return figs


def _parse_methods(text: str):
    if str(text).strip().lower() in ("all", "*"):
        return ALL_METHODS
    raw = [c.strip() for c in str(text).split(",") if c.strip()]
    canon = {m.lower().replace("-", "").replace("_", "").replace("+", ""): m
             for m in ALL_METHODS}
    out = []
    for name in raw:
        key = name.lower().replace("-", "").replace("_", "").replace("+", "")
        if key not in canon:
            raise argparse.ArgumentTypeError(
                "--methods: unknown method %r; choose from %s"
                % (name, ", ".join(ALL_METHODS)))
        if canon[key] not in out:
            out.append(canon[key])
    if not out:
        raise argparse.ArgumentTypeError("--methods: empty list")
    # keep the canonical order so tables read consistently
    return tuple(m for m in ALL_METHODS if m in out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiments.py",
        description="Reproduce Figs 2/3, 4, 5 and Table 2 of the CTIM paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # -- required by API.md ------------------------------------------------
    p.add_argument("--dataset", required=True,
                   help="path to a processed dataset directory "
                        "(graph.tsv, logs.tsv, items.tsv, meta.json)")
    p.add_argument("--figures", default="2,4,5", type=_parse_figures,
                   help="which figures to produce: any of 2,3,4,5 or 'all' "
                        "(2 and 3 both mean the spread/time vs K panels). "
                        "Default: 2,4,5")
    p.add_argument("--C", type=int, default=100,
                   help="number of communities (SPEC.md Section 8: 100)")
    p.add_argument("--Z", type=int, default=8,
                   help="number of topics (SPEC.md Section 8: 8)")
    p.add_argument("--K-list", dest="K_list", default=None, type=str,
                   help="comma-separated seed-set sizes. Default: %s"
                        % ",".join(str(k) for k in PAPER_K_LIST))
    p.add_argument("--seed", type=int, default=42,
                   help="master seed; every random.Random is derived from it")
    p.add_argument("--out", default="",
                   help="output directory. Default: results/<dataset name>")
    p.add_argument("--n-test-items", dest="n_test_items", type=int, default=10,
                   help="how many of the most-adopted test-split items to "
                        "evaluate on (default 10)")
    p.add_argument("--n-mc", dest="n_mc", type=int, default=200,
                   help="Monte-Carlo simulations for the sampling baselines "
                        "(Greedy, CINEMA, MixedGreedy inside CGA)")
    p.add_argument("--gibbs-iters-topic", dest="gibbs_iters_topic", type=int,
                   default=200, help="Gibbs sweeps for stage 1, Eq (1)")
    p.add_argument("--gibbs-iters-comm", dest="gibbs_iters_comm", type=int,
                   default=200, help="Gibbs sweeps for stage 2, Eq (4)/(6)")
    p.add_argument("--sampler", choices=("exact", "mh"), default="exact",
                   help="stage-2 sampler: 'exact' enumerates all C^2 community "
                        "pairs per token; 'mh' is Metropolis-Hastings with the "
                        "same stationary distribution at O(C) per token")
    p.add_argument("--quick", action="store_true",
                   help="smoke profile: fewer Gibbs/EM iterations, fewer "
                        "Monte-Carlo sims, fewer test items. Exercises every "
                        "code path in a few minutes on a small dataset.")
    p.add_argument("--methods", default="all", type=_parse_methods,
                   help="comma-separated subset of %s, or 'all'"
                        % ", ".join(ALL_METHODS))

    # -- secondary knobs ---------------------------------------------------
    g = p.add_argument_group("secondary (defaults follow SPEC.md Section 8)")
    g.add_argument("--delta", type=int, default=30 * 24 * 3600,
                   help="Definition 1 time window in seconds (default 30 days)")
    g.add_argument("--h", type=float, default=0.1,
                   help="MIA threshold of Eq (15)/(16); the paper uses 0.1")
    g.add_argument("--dp-tiebreak", dest="dp_tiebreak",
                   choices=("consistent", "paper-literal"), default="consistent",
                   help="Algorithm 2 line 36; see SPEC.md Section 6 note 1")
    g.add_argument("--zeta", type=float, default=1.0,
                   help="weight in eps0 = zeta * ln(N_neg / C^2)")
    g.add_argument("--air-em-iters", dest="air_em_iters", type=int, default=50,
                   help="maximum EM sweeps for the AIR baseline")
    g.add_argument("--max-logs-per-item", dest="max_logs_per_item", type=int,
                   default=0,
                   help="cap potential-influence logs per item by uniform "
                        "subsampling (0 = uncapped)")
    g.add_argument("--z-grid", dest="z_grid", default=None, type=str,
                   help="Fig 4 grid. Default: %s"
                        % ",".join(str(z) for z in PAPER_Z_GRID))
    g.add_argument("--c-grid", dest="c_grid", default=None, type=str,
                   help="Fig 5 grid. Default: %s"
                        % ",".join(str(c) for c in PAPER_C_GRID))
    g.add_argument("--fig45-K", dest="fig45_K", type=int, default=20,
                   help="seed-set size used by Figs 4 and 5 (paper: 20)")
    g.add_argument("--verbose", action="store_true",
                   help="verbose model fitting")
    return p


def config_from_args(args) -> ExperimentConfig:
    cfg = ExperimentConfig(
        dataset=args.dataset,
        out=args.out,
        figures=tuple(args.figures),
        C=args.C,
        Z=args.Z,
        K_list=_int_list(args.K_list, "--K-list") if args.K_list else PAPER_K_LIST,
        seed=args.seed,
        n_test_items=args.n_test_items,
        n_mc=args.n_mc,
        gibbs_iters_topic=args.gibbs_iters_topic,
        gibbs_iters_comm=args.gibbs_iters_comm,
        sampler=args.sampler,
        methods=tuple(args.methods),
        h=args.h,
        dp_tiebreak=args.dp_tiebreak,
        delta=args.delta,
        max_logs_per_item=args.max_logs_per_item,
        zeta=args.zeta,
        air_em_iters=args.air_em_iters,
        z_grid=_int_list(args.z_grid, "--z-grid") if args.z_grid else PAPER_Z_GRID,
        c_grid=_int_list(args.c_grid, "--c-grid") if args.c_grid else PAPER_C_GRID,
        fig45_K=args.fig45_K,
        verbose=args.verbose,
    )
    if args.quick:
        cfg = quick_config(cfg)
        # The exact stage-2 sampler costs O(C^2) per token, which a smoke run
        # cannot afford at C=150 (the top of the Fig 5 grid).  MH has the same
        # stationary distribution at O(C), so switching keeps the sweep honest
        # while finishing in minutes.  An explicit --sampler on the command
        # line always wins.
        if "--sampler" not in sys.argv:
            cfg.sampler = "mh"
        if not args.K_list:
            # K=1 and K=51 are the two points SPEC.md Section 9 digitises for
            # the timing figure, so both survive the shrink.
            cfg.K_list = (1, 11, 21, 51)
        # A smoke run uses a small graph, and C=100 on a small graph makes
        # every Eq (12) weight fall under the Eq (15) threshold h, so every
        # method scores exactly |S| and the whole report is vacuous.  Cap C to
        # the graph -- unless the user pinned --C explicitly, which always wins.
        if "--C" not in sys.argv:
            n_users = _meta_n_users(args.dataset)
            if n_users:
                notes = scale_C_to_dataset(
                    cfg, n_users, scale_c_grid=not args.c_grid)
                for n in notes:
                    print("  note      : %s" % n)
                cfg.cfg_notes = tuple(notes)
    if not cfg.out:
        cfg.out = os.path.join("results", cfg.dataset_name())
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.isdir(args.dataset):
        print("error: --dataset %r is not a directory" % args.dataset,
              file=sys.stderr)
        return 2
    meta = os.path.join(args.dataset, "meta.json")
    if not os.path.exists(meta):
        print("error: %r has no meta.json; is it a processed dataset "
              "directory?" % args.dataset, file=sys.stderr)
        return 2

    cfg = config_from_args(args)

    print("=" * 72)
    print("CTIM reproduction - %s" % cfg.dataset)
    print("  profile   : %s" % ("quick" if cfg.quick else "full"))
    print("  figures   : %s" % ",".join(str(f) for f in cfg.figures))
    print("  methods   : %s" % ", ".join(cfg.methods))
    print("  C=%d Z=%d K=%s seed=%d sampler=%s"
          % (cfg.C, cfg.Z, ",".join(str(k) for k in cfg.K_list), cfg.seed,
             cfg.sampler))
    print("  out       : %s" % cfg.out)
    print("=" * 72)

    try:
        res = run_experiments(cfg)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - top-level guard
        import traceback
        traceback.print_exc()
        print("error: the experiment could not run: %s: %s"
              % (type(exc).__name__, exc), file=sys.stderr)
        return 1

    # Exit non-zero only if the harness itself failed to produce results; a
    # NOT-REPRODUCED claim is a legitimate scientific outcome, not an error.
    if not (res.k_points or res.z_points or res.c_points):
        print("error: no results were produced", file=sys.stderr)
        return 1
    n_repro = sum(1 for c in check_claims(res) if c.reproduced)
    print("claims reproduced: %d/4" % n_repro)
    return 0


if __name__ == "__main__":
    sys.exit(main())
