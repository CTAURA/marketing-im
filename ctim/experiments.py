# -*- coding: utf-8 -*-
"""Reproduction harness for the five figures / one table of the CTIM paper.

    Huimin Huang, Hong Shen, Zaiqiao Meng, Huajian Chang, Huaiwen He.
    "Community-based influence maximization for viral marketing",
    Applied Intelligence (2019).  DOI 10.1007/s10489-018-1387-8

What this module produces (SPEC.md Sections 8 and 9, API.md):

    Fig 2a / 3a   influence spread   vs K in {1,11,21,31,41,51}, 5 methods
    Fig 2b / 3b   running time (s)   vs the same K, same 5 methods, log scale
    Fig 4         impact of Z in {2,4,6,8,10,12,14,16} on CTIM (K=20, C=100)
    Fig 5         impact of C in {25,50,75,100,125,150} on CTIM (K=20, Z=8)
    Table 2       feature comparison of the five methods

into ``results/<dataset>/``: one ``.svg`` per figure panel, one ``.csv`` per
figure, and a ``report.md`` that places our numbers next to the paper's
digitised numbers and states, for each qualitative claim, whether it
REPRODUCED or NOT.

-------------------------------------------------------------------------------
Methodology, stated up front because it is what makes the numbers meaningful
-------------------------------------------------------------------------------

**Train / test discipline.**  Potential-influence logs (SPEC.md Definition 1)
are split 60/20/20 by :func:`ctim.dataset.split_logs`.  Every model -- CTIM's
Gibbs sampler and AIR's EM -- sees the TRAIN split only.  The validation split
is held out and untouched.  Evaluation happens on items drawn from the TEST
split.

**One shared evaluator for all five methods.**  Every method's seed set is
scored with the *same* function: exact MIA influence, Eq (17)/(18), on the FULL
graph, using the reference CTIM model's Eq (12) edge weights for the test item
being evaluated.  No Monte Carlo, no per-method scoring rule.  A method's own
internal objective (MixedGreedy's live-edge estimate, Greedy's IC simulation,
CINEMA's conformity-reweighted DP value) is *never* what gets reported as its
spread -- it is recorded in the CSV as a diagnostic only.  See the CAVEAT in
the generated report: the shared evaluator's weights come from CTIM's model,
which is a bias in CTIM's favour that we state rather than hide.

**Topic-aware vs topic-blind methods.**  CTIM, CTIM_CGA and AIR+CGA consume the
test item's topic mixture, so they are re-run per (item, K) and the reported
spread is the mean over ``n_test_items``.  Greedy and CINEMA are topic-blind
(SPEC.md Table 2): they get one topic-independent probability map
(``topic_averaged_pp``, the mean of Eq (12) over the evaluated items) and are
therefore run once per K and evaluated against every test item's weights.  That
is exactly the asymmetry the paper's Table 2 describes, and running them once
rather than n_test_items times is a *credit* to them in the timing figures.

**Timing.**  Wall clock via ``time.perf_counter``.  For every method the
reported seconds are

    seconds(method, K) = mean_over_runs(selection seconds)
                       + fit_seconds(method) / len(K_list)

where "selection seconds" excludes the shared evaluator (which is not part of
any method) but includes everything the method itself must do per run, and
"fit_seconds" is the method's one-off, K-independent, item-independent
preparation, amortised across the K sweep so that summing the plotted curve
reproduces the true total cost of the sweep.  What lands in ``fit_seconds`` per
method:

    CTIM       collapsed Gibbs sampling, Eq (1)-(9)          (shared with CTIM_CGA)
    CTIM_CGA   the same Gibbs sampling                        (shared with CTIM)
    AIR+CGA    AIR's EM fit  +  CNM modularity detection
    CINEMA     CNM modularity detection
    Greedy     building the topic-averaged probability map

and what stays in the per-run selection time:

    CTIM       Eq (10)-(12) edge weights, Eq (19) detection, Algorithm 2
    CTIM_CGA   Eq (10)-(12) edge weights, Eq (19) detection, CGA DP+MixedGreedy
    AIR+CGA    AIR Eq (A2) edge weights, CGA DP + MixedGreedy
    CINEMA     conformity estimation, conformity reweighting, DP + CELF greedy
    Greedy     CELF greedy with Monte-Carlo IC

Both components are written to the CSV separately, so nothing is hidden behind
the amortisation.

For Fig 4 and Fig 5 the paper attributes the time trend to Algorithm 1 *and*
Algorithm 2 ("running time is more sensitive to C than to Z ... both Algorithm 1
and Algorithm 2 depend on C"), so those two figures report the UN-amortised
    seconds(Z or C) = fit_seconds + mean_over_items(selection seconds)
which is the quantity whose C-sensitivity the paper is describing.

Standard library only.  Python 3.9 compatible.  All randomness flows through
explicit ``random.Random`` instances.
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ctim.dataset import (Dataset, build_potential_influence_logs, load_dataset,
                          split_logs)
from ctim.gibbs import train_model
from ctim.influence import MIA, EdgeWeights
from ctim.ctim import ctim_select_seeds
from ctim.plotting import ascii_table, line_chart, write_csv

# --------------------------------------------------------------------------
# Method registry
# --------------------------------------------------------------------------

ALL_METHODS = ("CTIM", "CTIM_CGA", "AIR+CGA", "CINEMA", "Greedy")

#: Whether a method consumes the evaluated item's topic mixture.  Topic-blind
#: methods are run once per K and scored against every test item's weights.
TOPIC_AWARE = {
    "CTIM": True,
    "CTIM_CGA": True,
    "AIR+CGA": True,
    "CINEMA": False,
    "Greedy": False,
}

#: SPEC.md Section 7 / Table 2 -- the paper's feature comparison.
TABLE2_ROWS = [
    # method, community-based, topic-aware, influence computation model, reference
    ("Greedy", "no", "no", "Monte-Carlo Independent Cascade", "Kempe et al. [3]"),
    ("CINEMA", "yes", "no", "Monte-Carlo IC (conformity-aware)", "Li et al. [24]"),
    ("AIR+CGA", "yes", "yes", "MixedGreedy (Monte-Carlo IC)", "Barbieri et al. [35] + CGA [22]"),
    ("CTIM_CGA", "yes", "yes", "MixedGreedy (Monte-Carlo IC)", "this paper + CGA [22]"),
    ("CTIM", "yes", "yes", "MIA, exact Eq (17)/(18)", "this paper"),
]

# --------------------------------------------------------------------------
# The paper's digitised target numbers (SPEC.md Section 9)
# --------------------------------------------------------------------------

#: Fig 2a -- Yelp, influence spread vs K (Z=8, C=100).
PAPER_FIG2A = {
    1:  {"CTIM": 900,  "CTIM_CGA": 900,  "AIR+CGA": 700,  "CINEMA": 500,  "Greedy": 300},
    11: {"CTIM": 3000, "CTIM_CGA": 3000, "AIR+CGA": 2700, "CINEMA": 2400, "Greedy": 2000},
    21: {"CTIM": 4350, "CTIM_CGA": 4300, "AIR+CGA": 3900, "CINEMA": 3400, "Greedy": 2900},
    31: {"CTIM": 4700, "CTIM_CGA": 4700, "AIR+CGA": 4400, "CINEMA": 3900, "Greedy": 3300},
    41: {"CTIM": 4950, "CTIM_CGA": 4900, "AIR+CGA": 4700, "CINEMA": 4300, "Greedy": 3700},
    51: {"CTIM": 5050, "CTIM_CGA": 5000, "AIR+CGA": 4700, "CINEMA": 4600, "Greedy": 3950},
}

#: Fig 2b -- Yelp, running time in seconds (log scale).  Only K=1 and K=51 are
#: digitised in SPEC.md Section 9.
PAPER_FIG2B = {
    1:  {"CTIM": 2.0e2, "CTIM_CGA": 3.0e3, "CINEMA": 1.0e4, "AIR+CGA": 2.0e4, "Greedy": 3.0e3},
    51: {"CTIM": 7.0e2, "CTIM_CGA": 3.0e4, "CINEMA": 1.4e5, "AIR+CGA": 2.0e5, "Greedy": 2.5e4},
}

#: Fig 3a -- Digg, influence spread vs K.
PAPER_FIG3A = {
    1:  {"CTIM": 30,   "CTIM_CGA": 30,   "AIR+CGA": 30,  "CINEMA": 30,  "Greedy": 30},
    11: {"CTIM": 490,  "CTIM_CGA": 480,  "AIR+CGA": 430, "CINEMA": 420, "Greedy": 190},
    21: {"CTIM": 730,  "CTIM_CGA": 720,  "AIR+CGA": 610, "CINEMA": 570, "Greedy": 280},
    31: {"CTIM": 1000, "CTIM_CGA": 990,  "AIR+CGA": 730, "CINEMA": 700, "Greedy": 520},
    41: {"CTIM": 1210, "CTIM_CGA": 1190, "AIR+CGA": 830, "CINEMA": 790, "Greedy": 610},
    51: {"CTIM": 1420, "CTIM_CGA": 1400, "AIR+CGA": 950, "CINEMA": 890, "Greedy": 800},
}

#: Fig 3b -- Digg, running time in seconds (log scale).
PAPER_FIG3B = {
    1:  {"CTIM": 1.0e1, "CTIM_CGA": 1.7e1, "CINEMA": 1.6e1, "AIR+CGA": 1.8e1, "Greedy": 1.3e1},
    51: {"CTIM": 4.0e1, "CTIM_CGA": 1.5e2, "CINEMA": 1.1e3, "AIR+CGA": 9.0e3, "Greedy": 1.2e3},
}

#: Fig 4 -- impact of Z on Yelp (K=20, C=100): spread and seconds.
PAPER_FIG4 = {
    2:  (2500, 230), 4:  (3500, 380), 6:  (4180, 430), 8:  (4320, 450),
    10: (4320, 465), 12: (4320, 475), 14: (4320, 490), 16: (4320, 495),
}

#: Fig 5 -- impact of C on Yelp (K=20, Z=8): spread and seconds.
PAPER_FIG5 = {
    25: (2650, 60), 50: (3800, 330), 75: (4200, 430),
    100: (4320, 480), 125: (3800, 490), 150: (3600, 500),
}

PAPER_K_LIST = (1, 11, 21, 31, 41, 51)
PAPER_Z_GRID = (2, 4, 6, 8, 10, 12, 14, 16)
PAPER_C_GRID = (25, 50, 75, 100, 125, 150)


def paper_tables_for(dataset_name: str) -> Tuple[dict, dict, str, str]:
    """Pick the paper figure pair that matches this dataset.

    The paper reports Yelp in Fig 2 and Digg in Fig 3; SPEC.md Section 9
    digitises both.  Any other dataset is compared against the Yelp numbers
    (Fig 2) since that is the paper's headline experiment, and the report says
    so explicitly.
    """
    low = (dataset_name or "").lower()
    if "digg" in low:
        return PAPER_FIG3A, PAPER_FIG3B, "3a", "3b"
    return PAPER_FIG2A, PAPER_FIG2B, "2a", "2b"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Everything the harness needs; every field is settable from the CLI."""

    dataset: str
    out: str = ""
    figures: Tuple[int, ...] = (2, 4, 5)
    C: int = 100
    Z: int = 8
    K_list: Tuple[int, ...] = PAPER_K_LIST
    seed: int = 42
    n_test_items: int = 10
    n_mc: int = 200
    gibbs_iters_topic: int = 200
    gibbs_iters_comm: int = 200
    sampler: str = "exact"
    quick: bool = False
    methods: Tuple[str, ...] = ALL_METHODS

    # secondary knobs (defaults follow SPEC.md Section 8)
    h: float = 0.1
    dp_tiebreak: str = "paper-true"
    delta: int = 30 * 24 * 3600
    max_logs_per_item: int = 0
    zeta: float = 1.0
    air_em_iters: int = 50
    z_grid: Tuple[int, ...] = PAPER_Z_GRID
    c_grid: Tuple[int, ...] = PAPER_C_GRID
    fig45_K: int = 20
    verbose: bool = False
    #: notes accumulated while building the config (e.g. quick-profile C cap)
    cfg_notes: Tuple[str, ...] = ()

    def dataset_name(self) -> str:
        return os.path.basename(os.path.normpath(self.dataset))

    def as_rows(self) -> List[List[Any]]:
        """Configuration rendered as (key, value) rows for the report."""
        return [
            ["dataset", self.dataset],
            ["profile", "quick" if self.quick else "full"],
            ["figures", ",".join(str(f) for f in self.figures)],
            ["methods", ", ".join(self.methods)],
            ["C (communities)", self.C],
            ["Z (topics)", self.Z],
            ["K list", ",".join(str(k) for k in self.K_list)],
            ["K for Fig 4/5", self.fig45_K],
            ["Z grid (Fig 4)", ",".join(str(z) for z in self.z_grid)],
            ["C grid (Fig 5)", ",".join(str(c) for c in self.c_grid)],
            ["seed", self.seed],
            ["n test items", self.n_test_items],
            ["n Monte-Carlo sims", self.n_mc],
            ["Gibbs iters (topic, Eq 1)", self.gibbs_iters_topic],
            ["Gibbs iters (community, Eq 4/6)", self.gibbs_iters_comm],
            ["Gibbs sampler", self.sampler],
            ["AIR EM iters", self.air_em_iters],
            ["MIA threshold h (Eq 15)", self.h],
            ["Algorithm 2 line-35/36 reading", self.dp_tiebreak],
            ["Delta for Definition 1 (s)", self.delta],
            ["max logs per item (0 = uncapped)", self.max_logs_per_item],
        ]


def quick_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Shrink a config so a smoke run finishes in a few minutes.

    Every code path stays live: all five methods, all three figure groups, both
    Gibbs stages, the EM fit, both community detectors, MIA, MixedGreedy, CELF,
    the SVG/CSV writers and the report generator.  Only iteration counts and
    sample sizes shrink -- never the set of things exercised.
    """
    cfg.quick = True
    # 30, not 8.  Eq (12) is bounded above by max_{c,z} theta_cz (there is a
    # unit test asserting exactly that), and alpha=50/Z smooths theta towards
    # its uniform value 1/Z = 0.125 at Z=8 -- barely above the Eq (15)
    # threshold h=0.1.  So theta has to actually concentrate before ANY edge
    # clears h.  Measured on Digg (C=100, Z=8): 8 sweeps give pp_max=0.095 and
    # a vacuous run where every method scores |S|; 30 sweeps give pp_max=0.125
    # with 24% of edges above h; 80 sweeps give 0.125 -- i.e. 30 is converged.
    cfg.gibbs_iters_topic = min(cfg.gibbs_iters_topic, 30)
    cfg.gibbs_iters_comm = min(cfg.gibbs_iters_comm, 30)
    cfg.n_mc = min(cfg.n_mc, 20)
    cfg.n_test_items = min(cfg.n_test_items, 3)
    cfg.air_em_iters = min(cfg.air_em_iters, 8)
    # Definition 1 is quadratic in each item's adopter count, so a real trace
    # yields far more potential-influence logs than a smoke run can sample:
    # Digg at the default Delta=30d produces 3.8M logs (2.3M in the train
    # split), and one O(C)-per-token Gibbs sweep over that does not finish in
    # minutes.  API.md gives max_per_item as THE tractability knob for exactly
    # this; cap it so |D| stays proportional to the item count.  0 means the
    # user did not ask for a cap, so the quick profile picks one; an explicit
    # --max-logs-per-item always wins.
    if cfg.max_logs_per_item <= 0:
        cfg.max_logs_per_item = QUICK_MAX_LOGS_PER_ITEM
    return cfg


#: Per-item cap on potential-influence logs used by the quick profile.
QUICK_MAX_LOGS_PER_ITEM = 20


#: Users per community that keeps Eq (7) concentrated enough for Eq (12) to
#: clear the Eq (15) threshold.  The paper's own settings sit far above this
#: floor (Yelp 366,715/100 = 3,667 users per community; Digg 30,358/100 = 304),
#: so this only ever binds on the small graphs a smoke run uses.
MIN_USERS_PER_COMMUNITY = 40


def scale_C_to_dataset(cfg: ExperimentConfig, n_users: int,
                       scale_c_grid: bool = True) -> List[str]:
    """Cap ``C`` so a small graph does not produce a vacuous run.

    ``rho = 50/C`` (SPEC.md Section 8) is a smoothing prior on ``pi_v``.  When
    ``C`` approaches the number of users, every user has only a handful of
    Eq (4)/(6) tokens spread over ``C`` communities, ``rho`` dominates the
    Eq (7) counts, and ``pi_v`` stays essentially uniform at ``1/C``.  The
    Eq (12) product then collapses below the Eq (15) threshold ``h``, no path
    survives, and Eq (18) returns ``I(S) = |S|`` for every method -- the run
    completes but every spread figure is vacuous.

    Keeping at least ``MIN_USERS_PER_COMMUNITY`` users per community leaves the
    Eq (7) counts able to outvote ``rho``.  ``C`` is only ever lowered, never
    raised, so the paper's C=100 survives untouched on any graph with >= 4,000
    users (both paper benchmarks qualify).  Returns a list of human-readable
    notes describing what was changed, for the report.
    """
    notes: List[str] = []
    capped = max(2, min(cfg.C, n_users // MIN_USERS_PER_COMMUNITY))
    if capped >= cfg.C:
        return notes
    old_C = cfg.C
    cfg.C = capped
    notes.append(
        "quick profile: C lowered from %d to %d because the graph has only "
        "%d users; at C=%d that is %.1f users per community, below the %d "
        "needed for Eq (7) to concentrate pi_v above its 1/C floor (rho=50/C "
        "would otherwise dominate and drive every Eq (12) weight under h)."
        % (old_C, capped, n_users, old_C, n_users / float(old_C),
           MIN_USERS_PER_COMMUNITY))
    if scale_c_grid and tuple(cfg.c_grid) == PAPER_C_GRID:
        # Keep the SHAPE of the paper's Fig 5 grid (0.25x .. 1.5x of the
        # reference C) so the "spread peaks at the reference C" claim stays
        # evaluable, just centred on the capped C instead of on 100.
        grid = []
        for c in PAPER_C_GRID:
            v = int(round(capped * (c / 100.0)))
            v = max(2, v)
            if v not in grid:
                grid.append(v)
        if capped not in grid:
            grid.append(capped)
        cfg.c_grid = tuple(sorted(grid))
        notes.append(
            "quick profile: Fig 5 C grid rescaled from %s to %s (the paper's "
            "0.25x..1.5x shape recentred on C=%d)."
            % (",".join(str(c) for c in PAPER_C_GRID),
               ",".join(str(c) for c in cfg.c_grid), capped))
    return notes


# --------------------------------------------------------------------------
# Result records
# --------------------------------------------------------------------------


@dataclass
class MethodPoint:
    """One (method, K) cell of Fig 2/3."""

    method: str
    K: int
    spread: float = float("nan")
    spread_sd: float = 0.0
    select_seconds: float = 0.0     # mean per-run, evaluator excluded
    fit_seconds: float = 0.0        # one-off, un-amortised
    fit_amortised: float = 0.0      # fit_seconds / len(K_list)
    seconds: float = float("nan")   # what gets plotted
    n_runs: int = 0
    n_seeds: int = 0
    own_estimate: float = float("nan")   # the method's internal objective value
    failed: bool = False
    error: str = ""
    notes: str = ""


@dataclass
class SweepPoint:
    """One point of Fig 4 (varying Z) or Fig 5 (varying C)."""

    value: int
    spread: float = float("nan")
    select_seconds: float = 0.0
    fit_seconds: float = 0.0
    seconds: float = float("nan")
    n_runs: int = 0
    n_communities_used: int = 0
    failed: bool = False
    error: str = ""
    #: True when this point's OWN model produces no Eq (12) weight >= h, so
    #: Eq (15) admits no path, Algorithm 2 sees zero marginal gain everywhere
    #: and its seed choice carries no information.  The resulting spread is an
    #: artefact (roughly |S|), not a measurement -- see DEVIATIONS.md Section 7.
    degenerate: bool = False
    pp_max: float = float("nan")


@dataclass
class ExperimentResults:
    """Everything the report generator needs."""

    config: ExperimentConfig
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    model_diag: Dict[str, Any] = field(default_factory=dict)
    k_points: List[MethodPoint] = field(default_factory=list)
    z_points: List[SweepPoint] = field(default_factory=list)
    c_points: List[SweepPoint] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def k_cell(self, method: str, K: int) -> Optional[MethodPoint]:
        for p in self.k_points:
            if p.method == method and p.K == K:
                return p
        return None


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _fmt(value: Any, nd: int = 1) -> str:
    """Format a number for an ASCII table; '-' for missing, 'FAIL' handled by caller."""
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "-"
    if math.isinf(f):
        return "inf"
    return ("%." + str(nd) + "f") % f


def _fmt_sci(value: Any) -> str:
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "-"
    if f == 0.0:
        return "0"
    return "%.2e" % f


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _sd(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return 0.0
    return statistics.pstdev(xs)


def _zero_evaluator(seeds) -> float:
    """A free stand-in evaluator.

    Handed to baselines that would otherwise score their own seed set inside
    the timed region.  Scoring is not part of any method's cost -- the harness
    scores every method afterwards with the one shared evaluator -- so this
    keeps the running-time figures comparing like with like.
    """
    return 0.0


def _progress(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


class Experiment:
    """Runs the figures for one processed dataset.

    Usage::

        exp = Experiment(cfg)
        exp.prepare()
        results = exp.run()
        exp.write_outputs(results)
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.ds: Optional[Dataset] = None
        self.logs_d: List[Tuple[int, int, int]] = []
        self.train: List[Tuple[int, int, int]] = []
        self.valid: List[Tuple[int, int, int]] = []
        self.test: List[Tuple[int, int, int]] = []
        self.test_items: List[int] = []

        # reference model (C, Z from the config) -- shared by CTIM, CTIM_CGA and
        # the shared evaluator
        self.model = None
        self.ew: Optional[EdgeWeights] = None
        self.model_fit_seconds = 0.0

        self._evaluators: Dict[int, MIA] = {}
        self.failures: List[str] = []
        # Notes recorded while the config was built (e.g. the quick profile
        # capping C to the graph size) travel into the report with the rest.
        self.notes: List[str] = list(getattr(cfg, "cfg_notes", ()) or ())
        self.timings: Dict[str, float] = {}

        # lazily-built, cached per-method preparation
        self._pp_blind: Optional[dict] = None
        self._pp_blind_seconds = 0.0
        self._air = None
        self._air_comm = None
        self._air_fit_seconds = 0.0
        self._cinema_comm = None
        self._cinema_fit_seconds = 0.0

    # -- setup --------------------------------------------------------------

    def prepare(self) -> None:
        """Load the dataset, build and split the logs, choose the test items,
        and fit the reference CTIM model on the TRAIN split."""
        cfg = self.cfg
        t0 = time.perf_counter()
        _progress("[prepare] loading dataset from %s" % cfg.dataset)
        self.ds = load_dataset(cfg.dataset)
        _progress("[prepare] %s: U=%d E=%d M=%d F=%d adoptions=%d"
                  % (self.ds.name, self.ds.n_users, self.ds.n_links,
                     self.ds.n_items, self.ds.n_attrs, self.ds.n_logs))

        # Definition 1: potential-influence logs d = (u, v, i)
        t = time.perf_counter()
        self.logs_d = build_potential_influence_logs(
            self.ds, cfg.delta, max_per_item=cfg.max_logs_per_item, rng=self.rng
        )
        self.timings["build_logs"] = time.perf_counter() - t
        _progress("[prepare] %d potential-influence logs (Delta=%ds) in %.2fs"
                  % (len(self.logs_d), cfg.delta, self.timings["build_logs"]))
        if not self.logs_d:
            raise RuntimeError(
                "no potential-influence logs were built -- check --delta and the "
                "dataset's timestamps; every method needs D to be non-empty"
            )

        # SPEC.md Section 8: 60 / 20 / 20.  The model trains on `train` only.
        self.train, self.valid, self.test = split_logs(self.logs_d, self.rng)
        _progress("[prepare] split 60/20/20 -> train=%d valid=%d test=%d"
                  % (len(self.train), len(self.valid), len(self.test)))

        self.test_items = self._pick_test_items()
        _progress("[prepare] evaluating on %d test items: %s"
                  % (len(self.test_items), self.test_items))

        # Reference model: fitted ONCE on the train split, shared by CTIM and
        # CTIM_CGA (SPEC.md Section 7: their only difference is the influence
        # computation model) and used by the shared evaluator.
        self.model, self.model_fit_seconds = self._fit_ctim(cfg.C, cfg.Z, tag="reference")
        self.ew = EdgeWeights(self.model, self.ds)  # Eq (10)+(11) factorised
        self.timings["prepare_total"] = time.perf_counter() - t0

    def _pick_test_items(self) -> List[int]:
        """The `n_test_items` most-adopted items of the TEST split.

        "Most adopted" is counted over the test split of the potential-influence
        logs, i.e. the items that actually carry observed diffusion in held-out
        data.  Falls back to the raw adoption logs when the test split is too
        thin, and finally to item 0, so a degenerate dataset still runs.
        """
        counts: Dict[int, int] = {}
        for (_u, _v, i) in self.test:
            counts[i] = counts.get(i, 0) + 1
        if not counts:
            self.notes.append(
                "test split of D had no logs; test items fall back to the most "
                "adopted items of the raw adoption log"
            )
            for (_u, i, _t) in self.ds.logs:
                counts[i] = counts.get(i, 0) + 1
        if not counts:
            self.notes.append("no adoptions at all; falling back to item 0")
            return [0]
        ranked = sorted(counts.keys(), key=lambda i: (-counts[i], i))
        items = [i for i in ranked if 0 <= i < self.ds.n_items][: self.cfg.n_test_items]
        return items or [0]

    def _fit_ctim(self, C: int, Z: int, tag: str = ""):
        """Collapsed Gibbs sampling, Eq (1)-(9), on the TRAIN split only."""
        cfg = self.cfg
        _progress("[fit] CTIM model C=%d Z=%d (%s): %d+%d Gibbs iters, sampler=%s"
                  % (C, Z, tag, cfg.gibbs_iters_topic, cfg.gibbs_iters_comm, cfg.sampler))
        rng = random.Random(cfg.seed * 1000003 + C * 101 + Z)
        t = time.perf_counter()
        model = train_model(                       # Eq (1)-(9)
            self.ds, self.train, C, Z,
            cfg.gibbs_iters_topic, cfg.gibbs_iters_comm, rng,
            zeta=cfg.zeta, sampler=cfg.sampler, verbose=cfg.verbose,
        )
        secs = time.perf_counter() - t
        _progress("[fit] done in %.2fs" % secs)
        return model, secs

    # -- the one shared evaluator ------------------------------------------

    def evaluator_for(self, item: int) -> MIA:
        """Exact MIA on the FULL graph with item `item`'s Eq (12) weights.

        Cached per item: the first ``influence`` call pays for the MIIA
        arborescences of every node (Eq (15)), and every later call for the same
        item -- any method, any K -- reuses them.  That is the whole point of
        having one evaluator: the comparison is not contaminated by differing
        evaluation cost or differing estimator variance.
        """
        mia = self._evaluators.get(item)
        if mia is None:
            pp = self.ew.for_item(item)  # Eq (12)
            mia = MIA(self.ds.n_users, self.ds.out_adj, self.ds.in_adj,
                      pp, h=self.cfg.h)
            self._evaluators[item] = mia
        return mia

    def evaluate(self, seeds: Sequence[int], item: int) -> float:
        """Eq (18): I(S) = sum_v ap(v|S), exact, no Monte Carlo."""
        if not seeds:
            return 0.0
        return self.evaluator_for(item).influence(list(seeds))  # Eq (18)

    # -- per-method one-off preparation ------------------------------------

    def _blind_pp(self) -> Tuple[dict, float]:
        """The topic-independent probability map handed to Greedy and CINEMA.

        ``mean_i P(v|i,u)`` over the evaluated test items -- the convention
        documented by ``ctim.baselines.greedy.greedy_select``.  Note this is
        generous to the two topic-blind baselines: they get probabilities
        averaged over exactly the items they will be scored on.
        """
        if self._pp_blind is None:
            from ctim.baselines.greedy import topic_averaged_pp
            t = time.perf_counter()
            self._pp_blind = topic_averaged_pp(self.ew, self.test_items)  # Eq (12)
            self._pp_blind_seconds = time.perf_counter() - t
        return self._pp_blind, self._pp_blind_seconds

    def _air_prepared(self):
        """Fit AIR by EM on the TRAIN split and detect CNM communities, once."""
        if self._air is None:
            from ctim.baselines.air_cga import (detect_communities_modularity,
                                                fit_air)
            cfg = self.cfg
            rng = random.Random(cfg.seed * 7919 + 11)
            t = time.perf_counter()
            _progress("[fit] AIR EM (Z=%d, <=%d sweeps) on %d train logs"
                      % (cfg.Z, cfg.air_em_iters, len(self.train)))
            self._air = fit_air(self.ds, self.train, cfg.Z, rng,
                                n_iter=cfg.air_em_iters, verbose=cfg.verbose)
            _progress("[fit] AIR modularity community detection (target C=%d)" % cfg.C)
            self._air_comm = detect_communities_modularity(self.ds, C=cfg.C, rng=rng)
            self._air_fit_seconds = time.perf_counter() - t
            _progress("[fit] AIR+CGA preparation done in %.2fs" % self._air_fit_seconds)
        return self._air, self._air_comm, self._air_fit_seconds

    def _cinema_prepared(self):
        """CNM modularity communities for CINEMA, once.

        CINEMA's community detection looks only at the friend graph -- it is
        independent of the diffusion model, which is precisely the paper's
        criticism of it (SPEC.md Section 7).
        """
        if self._cinema_comm is None:
            from ctim.baselines.cinema import detect_communities_modularity
            t = time.perf_counter()
            _progress("[fit] CINEMA modularity community detection (target C=%d)"
                      % self.cfg.C)
            self._cinema_comm = detect_communities_modularity(
                self.ds.n_users, self.ds.edges, target_c=self.cfg.C
            )
            self._cinema_fit_seconds = time.perf_counter() - t
            _progress("[fit] CINEMA preparation done in %.2fs" % self._cinema_fit_seconds)
        return self._cinema_comm, self._cinema_fit_seconds

    def _cinema_adoption_logs(self) -> List[Tuple[int, int, int]]:
        """Adoption logs CINEMA may use for conformity estimation.

        Adoptions of the evaluated test items are removed so the conformity
        profile is estimated without seeing the held-out items it is scored on.
        """
        blocked = set(self.test_items)
        return [r for r in self.ds.logs if r[1] not in blocked]

    def method_fit_seconds(self, method: str) -> float:
        """The method's one-off, K- and item-independent preparation cost."""
        if method in ("CTIM", "CTIM_CGA"):
            return self.model_fit_seconds
        if method == "AIR+CGA":
            return self._air_prepared()[2]
        if method == "CINEMA":
            return self._cinema_prepared()[1]
        if method == "Greedy":
            return self._blind_pp()[1]
        return 0.0

    # -- one selection run --------------------------------------------------

    def run_method(self, method: str, K: int, item: Optional[int]):
        """Run one method for one (item, K).  Returns (seeds, seconds, extra).

        ``seconds`` is the method's own cost with the shared evaluator removed:
        anything a method spends scoring its own output is subtracted, because
        scoring is the harness's job and is done identically for everybody.
        """
        cfg = self.cfg
        rng = random.Random((cfg.seed * 31 + K) * 1000003 + (item or 0))
        t0 = time.perf_counter()

        if method == "CTIM":
            # Algorithm 2, lines 1-45 (Eq (10)-(12), Eq (19), Eq (15)-(18))
            seeds = ctim_select_seeds(
                self.model, self.ds, item, K, h=cfg.h,
                dp_tiebreak=cfg.dp_tiebreak, edge_weights=self.ew,
            )
            extra = dict(ctim_select_seeds.last_stats)
            own = float(extra.get("dp_value", float("nan")))

        elif method == "CTIM_CGA":
            # SPEC.md Section 7: CTIM's model, CGA's influence computation model.
            from ctim.baselines.cga import ctim_cga_select
            res = ctim_cga_select(self.model, self.ds, item, K, rng,
                                  n_mc=cfg.n_mc, h=cfg.h,
                                  dp_tiebreak=cfg.dp_tiebreak)
            seeds, extra = res.seeds, dict(res.extra)
            own = float(res.spread)

        elif method == "AIR+CGA":
            from ctim.baselines.air_cga import air_cga_select_seeds
            air, comm, _ = self._air_prepared()
            res = air_cga_select_seeds(
                self.ds, self.train, item, K, rng,
                Z=cfg.Z, C=cfg.C, n_mc=cfg.n_mc, h=cfg.h,
                dp_tiebreak=cfg.dp_tiebreak,
                air=air, comm=comm,
                # scoring is the harness's job -- see _zero_evaluator
                evaluator=_zero_evaluator,
            )
            seeds, extra = res.seeds, dict(res.extra)
            own = float("nan")

        elif method == "CINEMA":
            from ctim.baselines.cinema import cinema_select_seeds
            comm, _ = self._cinema_prepared()
            pp_blind, _ = self._blind_pp()
            res = cinema_select_seeds(
                self.ds, pp_blind, K, rng, n_mc=cfg.n_mc, C=cfg.C,
                logs=self._cinema_adoption_logs(), comm=comm,
                h=cfg.h, evaluate=False,
            )
            seeds, extra = res.seeds, dict(res.extra)
            own = float(extra.get("dp_estimate", float("nan")))

        elif method == "Greedy":
            from ctim.baselines.greedy import greedy_select
            pp_blind, _ = self._blind_pp()
            res = greedy_select(self.ds, pp_blind, K, cfg.n_mc, rng,
                                use_celf=True)
            seeds, extra = res.seeds, dict(res.extra)
            own = float(res.spread)

        else:
            raise ValueError("unknown method %r" % (method,))

        elapsed = time.perf_counter() - t0
        seconds = max(0.0, elapsed - float(extra.get("eval_seconds", 0.0) or 0.0))
        extra["own_estimate"] = own
        extra["wall_seconds_raw"] = elapsed
        return list(seeds), seconds, extra

    # -- Fig 2a/2b (or 3a/3b): spread and time vs K ------------------------

    def run_k_sweep(self) -> List[MethodPoint]:
        """Fig 2a/3a and Fig 2b/3b: all methods over the K grid."""
        cfg = self.cfg
        points: List[MethodPoint] = []
        n_k = max(1, len(cfg.K_list))

        for method in cfg.methods:
            try:
                fit_secs = self.method_fit_seconds(method)
            except Exception as exc:  # a broken preparation must not kill the run
                msg = "%s: preparation failed: %s: %s" % (method, type(exc).__name__, exc)
                self.failures.append(msg)
                _progress("[FAIL] " + msg)
                for K in cfg.K_list:
                    points.append(MethodPoint(method=method, K=K, failed=True,
                                              error=str(exc)))
                continue

            fit_amort = fit_secs / n_k
            topic_aware = TOPIC_AWARE[method]
            # Topic-blind methods produce ONE seed set per K, scored against
            # every test item's weights.  Topic-aware ones are re-run per item.
            run_items = self.test_items if topic_aware else [None]

            for K in cfg.K_list:
                t_start = time.perf_counter()
                spreads: List[float] = []
                secs: List[float] = []
                owns: List[float] = []
                n_seeds = 0
                err = ""
                for item in run_items:
                    try:
                        seeds, sec, extra = self.run_method(method, K, item)
                    except Exception as exc:
                        err = "%s: %s" % (type(exc).__name__, exc)
                        msg = ("%s K=%d item=%s: %s"
                               % (method, K, item, err))
                        self.failures.append(msg)
                        _progress("[FAIL] " + msg)
                        continue
                    n_seeds = max(n_seeds, len(seeds))
                    secs.append(sec)
                    owns.append(extra.get("own_estimate", float("nan")))
                    if topic_aware:
                        spreads.append(self.evaluate(seeds, item))  # Eq (18)
                    else:
                        # one seed set, scored under every test item's weights
                        for it in self.test_items:
                            spreads.append(self.evaluate(seeds, it))  # Eq (18)

                if not secs:
                    points.append(MethodPoint(method=method, K=K, failed=True,
                                              error=err or "no successful run",
                                              fit_seconds=fit_secs,
                                              fit_amortised=fit_amort))
                    continue

                sel = _mean(secs)
                mp = MethodPoint(
                    method=method, K=K,
                    spread=_mean(spreads), spread_sd=_sd(spreads),
                    select_seconds=sel, fit_seconds=fit_secs,
                    fit_amortised=fit_amort, seconds=sel + fit_amort,
                    n_runs=len(secs), n_seeds=n_seeds,
                    own_estimate=_mean(owns),
                    notes="one seed set, scored on all test items"
                          if not topic_aware else "",
                )
                points.append(mp)
                _progress(
                    "[run] %-9s K=%-3d spread=%9.2f  select=%7.3fs "
                    "(+fit/%d=%.3fs)  total=%7.3fs  [%.1fs elapsed]"
                    % (method, K, mp.spread, sel, n_k, fit_amort, mp.seconds,
                       time.perf_counter() - t_start)
                )
        return points

    # -- Fig 4: impact of Z -------------------------------------------------

    def run_z_sweep(self) -> List[SweepPoint]:
        """Fig 4: CTIM at K=20, C=100, for each Z in the grid.

        Every Z gets its own Gibbs fit (Z changes Eq (1) and Eq (6)), but the
        spread is always measured with the SHARED evaluator -- the reference
        model's weights -- so the numbers across the sweep are commensurable.
        Scoring each Z under its own weights would compare different rulers.
        """
        return self._param_sweep("Z", self.cfg.z_grid)

    def run_c_sweep(self) -> List[SweepPoint]:
        """Fig 5: CTIM at K=20, Z=8, for each C in the grid."""
        return self._param_sweep("C", self.cfg.c_grid)

    def _param_sweep(self, param: str, grid: Sequence[int]) -> List[SweepPoint]:
        cfg = self.cfg
        K = cfg.fig45_K
        out: List[SweepPoint] = []
        for val in grid:
            C = val if param == "C" else cfg.C
            Z = val if param == "Z" else cfg.Z
            t_start = time.perf_counter()
            try:
                if C == cfg.C and Z == cfg.Z:
                    # the reference fit is exactly this configuration; reuse it
                    model, fit_secs = self.model, self.model_fit_seconds
                else:
                    model, fit_secs = self._fit_ctim(C, Z, tag="%s=%d" % (param, val))
                ew = EdgeWeights(model, self.ds)  # Eq (10)+(11)

                # Can this point's own model drive Algorithm 2 at all?  Eq (12)
                # is bounded by max_{c,z} theta_cz ~ 1/Z, so for large Z every
                # weight falls under h, Eq (15) admits no path, and every
                # candidate ties at zero marginal gain.  Record it instead of
                # letting the resulting |S|-shaped artefact pass as a spread.
                _ppv = ew.for_item(self.test_items[0])  # Eq (12)
                _pp_max = max(_ppv.values()) if _ppv else 0.0
                _degenerate = _pp_max < cfg.h

                spreads: List[float] = []
                secs: List[float] = []
                for item in self.test_items:
                    t = time.perf_counter()
                    seeds = ctim_select_seeds(   # Algorithm 2, lines 1-45
                        model, self.ds, item, K, h=cfg.h,
                        dp_tiebreak=cfg.dp_tiebreak, edge_weights=ew,
                    )
                    secs.append(time.perf_counter() - t)
                    spreads.append(self.evaluate(seeds, item))  # Eq (18)

                from ctim.ctim import detect_communities
                comm = detect_communities(model.pi)  # Eq (19)
                n_used = len(set(comm))

                sel = _mean(secs)
                sp = SweepPoint(
                    value=val, spread=_mean(spreads), select_seconds=sel,
                    fit_seconds=fit_secs,
                    # Fig 4/5 report the un-amortised cost: the paper attributes
                    # the trend to Algorithm 1 *and* Algorithm 2 together.
                    seconds=fit_secs + sel,
                    n_runs=len(secs), n_communities_used=n_used,
                    degenerate=_degenerate, pp_max=_pp_max,
                )
                out.append(sp)
                _progress(
                    "[run] CTIM %s=%-4d spread=%9.2f  fit=%7.3fs select=%7.3fs "
                    "total=%7.3fs  [%.1fs elapsed]%s"
                    % (param, val, sp.spread, fit_secs, sel, sp.seconds,
                       time.perf_counter() - t_start,
                       ("  [DEGENERATE: max pp=%.4g < h=%.3g, spread is an "
                        "artefact]" % (_pp_max, cfg.h)) if _degenerate else "")
                )
            except Exception as exc:
                msg = "%s sweep %s=%d: %s: %s" % (param, param, val,
                                                  type(exc).__name__, exc)
                self.failures.append(msg)
                _progress("[FAIL] " + msg)
                out.append(SweepPoint(value=val, failed=True, error=str(exc)))
        return out

    def model_diagnostics(self) -> Dict[str, Any]:
        """Did the fitted model actually learn anything?

        Every spread in this report is produced by MIA on Eq (12) edge weights,
        and Eq (15) only keeps a path while its probability product stays above
        `h`.  So if the learned `pp` all sit below `h`, no influence propagates,
        every method scores exactly |S|, and the figures are vacuous.  These
        four numbers make that visible instead of leaving a reader to infer it
        from suspiciously round spreads:

        `mean max(pi_v)` against its uniform floor `1/C` says whether Eq (7)
        concentrated users onto communities at all; `frac(pp >= h)` and
        `max(pp)` say whether Eq (12) cleared the Eq (15) threshold.
        """
        from ctim.ctim import detect_communities
        diag: Dict[str, Any] = {}
        try:
            C = len(self.model.eta)
            pi = self.model.pi
            diag["C"] = C
            diag["mean_max_pi"] = (sum(max(r) for r in pi) / len(pi)) if pi else 0.0
            diag["uniform_max_pi"] = 1.0 / C if C else 0.0
            diag["n_communities_eq19"] = len(set(detect_communities(pi)))  # Eq (19)
            item = self.test_items[0]
            pp = self.ew.for_item(item)  # Eq (12)
            vals = sorted(pp.values())
            n = len(vals)
            diag["item"] = item
            diag["pp_max"] = vals[-1] if n else 0.0
            diag["pp_median"] = vals[n // 2] if n else 0.0
            diag["pp_frac_ge_h"] = (sum(1 for v in vals if v >= self.cfg.h) / n) if n else 0.0
            diag["h"] = self.cfg.h
            diag["degenerate"] = diag["pp_frac_ge_h"] == 0.0
        except Exception as exc:  # diagnostics must never break the run
            diag["error"] = "%s: %s" % (type(exc).__name__, exc)
        return diag

    # -- driver -------------------------------------------------------------

    def run(self) -> ExperimentResults:
        cfg = self.cfg
        figs = set(cfg.figures)
        res = ExperimentResults(config=cfg)

        res.dataset_summary = {
            "name": self.ds.name,
            "n_users": self.ds.n_users,
            "n_links": self.ds.n_links,
            "n_items": self.ds.n_items,
            "n_attrs": self.ds.n_attrs,
            "n_adoptions": self.ds.n_logs,
            "n_potential_influence_logs": len(self.logs_d),
            "n_train": len(self.train),
            "n_valid": len(self.valid),
            "n_test": len(self.test),
            "test_items": list(self.test_items),
        }
        res.model_diag = self.model_diagnostics()
        if res.model_diag.get("degenerate"):
            msg = ("no learned edge probability reaches the MIA threshold "
                   "h=%.3g (max pp = %.4g), so Eq (15) admits no path and every "
                   "method scores exactly |S|; the spread figures are vacuous "
                   "for this configuration"
                   % (res.model_diag.get("h", 0.0), res.model_diag.get("pp_max", 0.0)))
            self.notes.append(msg)
            _progress("[WARN] " + msg)

        # 2 and 3 are the same panel pair -- Yelp is Fig 2, Digg is Fig 3.
        if figs & {2, 3}:
            _a, _b, name_a, name_b = paper_tables_for(self.ds.name)
            _progress("\n=== Fig %s/%s: spread and running time vs K ==="
                      % (name_a, name_b))
            res.k_points = self.run_k_sweep()
        if 4 in figs:
            _progress("\n=== Fig 4: impact of Z (K=%d, C=%d) ==="
                      % (cfg.fig45_K, cfg.C))
            res.z_points = self.run_z_sweep()
        if 5 in figs:
            _progress("\n=== Fig 5: impact of C (K=%d, Z=%d) ==="
                      % (cfg.fig45_K, cfg.Z))
            res.c_points = self.run_c_sweep()

        res.failures = list(self.failures)
        res.notes = list(self.notes)
        res.timings = dict(self.timings)
        return res

    # -- outputs ------------------------------------------------------------

    def write_outputs(self, res: ExperimentResults) -> List[str]:
        return write_outputs(res)


# --------------------------------------------------------------------------
# Qualitative claim checking (SPEC.md Section 10.5)
# --------------------------------------------------------------------------


@dataclass
class Claim:
    key: str
    text: str
    reproduced: bool
    detail: str


def _chain_ok(values: Dict[str, float]) -> Tuple[bool, List[str]]:
    """CTIM >= CTIM_CGA > AIR+CGA > CINEMA > Greedy."""
    pairs = [("CTIM", "CTIM_CGA", ">="), ("CTIM_CGA", "AIR+CGA", ">"),
             ("AIR+CGA", "CINEMA", ">"), ("CINEMA", "Greedy", ">")]
    broken = []
    for a, b, op in pairs:
        va, vb = values.get(a), values.get(b)
        if va is None or vb is None or math.isnan(va) or math.isnan(vb):
            broken.append("%s%s%s (missing)" % (a, op, b))
            continue
        ok = (va >= vb - 1e-9) if op == ">=" else (va > vb + 1e-9)
        if not ok:
            broken.append("%s(%.1f) %s %s(%.1f) violated" % (a, va, op, b, vb))
    return (not broken), broken


def check_claims(res: ExperimentResults) -> List[Claim]:
    """Evaluate the four qualitative claims of SPEC.md Section 10.5."""
    cfg = res.config
    claims: List[Claim] = []

    # ---- (a) spread ordering ---------------------------------------------
    have_all = all(m in cfg.methods for m in ALL_METHODS)
    if not res.k_points or not have_all:
        claims.append(Claim(
            "a", "CTIM >= CTIM_CGA > AIR+CGA > CINEMA > Greedy on influence spread",
            False,
            "not evaluable: " + ("the K sweep did not run"
                                 if not res.k_points else
                                 "only %s were run" % ", ".join(cfg.methods)),
        ))
    else:
        ok_k, bad = [], []
        for K in cfg.K_list:
            vals = {}
            for m in ALL_METHODS:
                p = res.k_cell(m, K)
                vals[m] = None if (p is None or p.failed) else p.spread
            good, broken = _chain_ok(vals)
            (ok_k if good else bad).append(
                K if good else "K=%d: %s" % (K, "; ".join(broken)))
        claims.append(Claim(
            "a", "CTIM >= CTIM_CGA > AIR+CGA > CINEMA > Greedy on influence spread",
            not bad,
            ("full chain holds at every K in %s" % (list(cfg.K_list),)) if not bad
            else ("holds at %d/%d K values; violations: %s"
                  % (len(ok_k), len(cfg.K_list), " | ".join(bad))),
        ))

    # ---- (b) CTIM fastest by orders of magnitude -------------------------
    others = [m for m in cfg.methods if m != "CTIM"]
    if "CTIM" not in cfg.methods or not others or not res.k_points:
        claims.append(Claim("b", "CTIM is fastest by orders of magnitude", False,
                            "not evaluable: CTIM and at least one baseline "
                            "must both run in the K sweep"))
    else:
        # Judged on SELECTION seconds, not on the fit-amortised total.
        # CTIM and CTIM_CGA share one Gibbs fit by construction (SPEC.md
        # Section 7: their only difference is the influence computation model),
        # so any fit cost enters both sides of that ratio identically and
        # drives it towards 1 whenever fitting dominates -- which it does on
        # small graphs.  Including it would test the model, not the thing the
        # paper's Fig 2b/3b is actually about.  The fit-amortised totals are
        # still what the figure plots and what the CSV records.
        worst = worst_total = None
        for K in cfg.K_list:
            pc = res.k_cell("CTIM", K)
            if pc is None or pc.failed or pc.select_seconds <= 0:
                continue
            for m in others:
                pm = res.k_cell(m, K)
                if pm is None or pm.failed or math.isnan(pm.select_seconds):
                    continue
                r = pm.select_seconds / pc.select_seconds
                if worst is None or r < worst[0]:
                    worst = (r, m, K)
                if pc.seconds > 0 and not math.isnan(pm.seconds):
                    rt = pm.seconds / pc.seconds
                    if worst_total is None or rt < worst_total[0]:
                        worst_total = (rt, m, K)
        # The speed-up against the SLOWEST baseline at the LARGEST K -- this is
        # where the paper's "orders of magnitude" actually lives (see the
        # criterion note below).
        best_at_max_K = None
        for K in sorted(cfg.K_list, reverse=True):
            pc = res.k_cell("CTIM", K)
            if pc is None or pc.failed or pc.select_seconds <= 0:
                continue
            ratios = []
            for m in others:
                pm = res.k_cell(m, K)
                if pm is None or pm.failed or math.isnan(pm.select_seconds):
                    continue
                ratios.append((pm.select_seconds / pc.select_seconds, m))
            if ratios:
                r, m = max(ratios)
                best_at_max_K = (r, m, K)
                break
        if worst is None or best_at_max_K is None:
            claims.append(Claim("b", "CTIM is fastest by orders of magnitude",
                                False, "no comparable timings were produced"))
        else:
            # CRITERION (corrected -- see DEVIATIONS.md).  Requiring >=10x
            # against EVERY baseline at EVERY K is stricter than the paper's
            # own data: SPEC.md Section 9 Fig 3b digitises Digg K=1 as
            # CTIM 1.0e1 vs Greedy 1.3e1, a 1.3x gap, and K=51 as CTIM 4.0e1
            # vs CTIM_CGA 1.5e2, a 3.75x gap.  The paper's claim is that CTIM
            # is fastest everywhere and pulls away by orders of magnitude as K
            # grows -- against the slowest baseline, at the largest K
            # (Digg K=51: 9.0e3/4.0e1 = 225x).  So we test exactly that.
            fastest_everywhere = worst[0] >= 1.0
            orders_at_max_K = best_at_max_K[0] >= 10.0
            ok = fastest_everywhere and orders_at_max_K
            claims.append(Claim(
                "b", "CTIM is fastest by orders of magnitude", ok,
                "judged on selection time (model fitting excluded, see below). "
                "Fastest everywhere: smallest speed-up over any baseline at any "
                "K is %.2fx (vs %s at K=%d), needs >=1x -- %s. Orders of "
                "magnitude at the largest K: speed-up vs the slowest baseline "
                "at K=%d is %.1fx (vs %s), needs >=10x -- %s. On the "
                "fit-amortised total the smallest speed-up is %s"
                % (worst[0], worst[1], worst[2],
                   "PASS" if fastest_everywhere else "FAIL",
                   best_at_max_K[2], best_at_max_K[0], best_at_max_K[1],
                   "PASS" if orders_at_max_K else "FAIL",
                   ("%.1fx (vs %s at K=%d)" % worst_total) if worst_total else "n/a"),
            ))

    # ---- (c) spread plateaus in Z ----------------------------------------
    all_zs = [p for p in res.z_points if not p.failed and not math.isnan(p.spread)]
    # A degenerate point's spread is |S| by construction (no Eq (12) weight
    # clears h, so Algorithm 2 has no signal).  Including it would test the
    # Eq (12)-vs-h scale problem of DEVIATIONS.md Section 7, not the plateau.
    zs = [p for p in all_zs if not p.degenerate]
    n_degen = len(all_zs) - len(zs)
    degen_note = ("; %d of %d Z points excluded as degenerate (max pp < h, "
                  "spread = |S| artefact: Z=%s)"
                  % (n_degen, len(all_zs),
                     ",".join(str(p.value) for p in all_zs if p.degenerate))
                  ) if n_degen else ""
    if len(zs) < 3:
        claims.append(Claim("c", "influence spread plateaus in Z", False,
                            "not evaluable: Fig 4 needs at least 3 usable "
                            "Z points, got %d%s" % (len(zs), degen_note)))
    else:
        zs.sort(key=lambda p: p.value)
        knee = 8 if any(p.value == 8 for p in zs) else zs[len(zs) // 2].value
        head = [p for p in zs if p.value <= knee]
        tail = [p for p in zs if p.value >= knee]
        rise = head[-1].spread - head[0].spread
        if len(tail) < 2:
            claims.append(Claim(
                "c", "influence spread plateaus in Z", False,
                "not evaluable: only %d usable Z point(s) at or above the knee "
                "Z=%d, need 2 to measure a plateau%s"
                % (len(tail), knee, degen_note)))
        else:
            tail_vals = [p.spread for p in tail]
            wobble = max(tail_vals) - min(tail_vals)
            scale = max(abs(rise), 1e-9)
            ok = rise > 0 and wobble <= 0.25 * scale
            claims.append(Claim(
                "c", "influence spread plateaus in Z", ok,
                "rise over Z=%d..%d is %+.1f; spread over Z=%d..%d varies by "
                "%.1f (%.0f%% of the rise); plateau accepted at <=25%%%s"
                % (head[0].value, knee, rise, knee, tail[-1].value, wobble,
                   100.0 * wobble / scale, degen_note),
            ))

    # ---- (d) spread peaks at the reference C -----------------------------
    # The paper's reference C is 100 (SPEC.md Section 9, Fig 5).  A quick run
    # on a small graph recentres the grid on a smaller C (see
    # scale_C_to_dataset), so the claim is tested against cfg.C -- the centre
    # of whatever grid was actually swept -- and the label says which.
    ref_C = cfg.C
    label_d = "influence spread peaks at C=%d" % ref_C
    cs = [p for p in res.c_points if not p.failed and not math.isnan(p.spread)]
    if len(cs) < 2:
        claims.append(Claim("d", label_d, False,
                            "not evaluable: Fig 5 needs at least 2 successful "
                            "C points, got %d" % len(cs)))
    elif not any(p.value == ref_C for p in cs):
        claims.append(Claim("d", label_d, False,
                            "not evaluable: C=%d is not in the C grid %s"
                            % (ref_C, list(cfg.c_grid))))
    else:
        best = max(cs, key=lambda p: p.spread)
        claims.append(Claim(
            "d", label_d, best.value == ref_C,
            "argmax over C in %s is C=%d (spread %.1f); C=%d gives %.1f"
            % ([p.value for p in cs], best.value, best.spread, ref_C,
               [p.spread for p in cs if p.value == ref_C][0]),
        ))

    return claims


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------


def write_outputs(res: ExperimentResults) -> List[str]:
    """Write every .svg, .csv and report.md for `res`.  Returns the paths."""
    cfg = res.config
    outdir = cfg.out or os.path.join("results", cfg.dataset_name())
    os.makedirs(outdir, exist_ok=True)
    written: List[str] = []
    fig_a, fig_b, name_a, name_b = paper_tables_for(res.dataset_summary.get("name", ""))
    fignum = name_a[0]  # '2' or '3'

    methods = [m for m in cfg.methods]

    # ---- Fig Na / Nb ------------------------------------------------------
    if res.k_points:
        spread_series = []
        time_series = []
        for m in methods:
            sp, tm = [], []
            for K in cfg.K_list:
                p = res.k_cell(m, K)
                if p is None or p.failed:
                    continue
                if not math.isnan(p.spread):
                    sp.append((K, p.spread))
                if not math.isnan(p.seconds):
                    tm.append((K, p.seconds))
            if sp:
                spread_series.append((m, sp))
            if tm:
                time_series.append((m, tm))

        p1 = os.path.join(outdir, "fig%sa_spread_vs_K.svg" % fignum)
        line_chart(p1, spread_series, "seed set size K", "influence spread I(S)",
                   "Fig %sa - influence spread vs K (%s, C=%d, Z=%d)"
                   % (fignum, res.dataset_summary.get("name", "?"), cfg.C, cfg.Z),
                   xticks=list(cfg.K_list))
        written.append(p1)

        p2 = os.path.join(outdir, "fig%sb_time_vs_K.svg" % fignum)
        line_chart(p2, time_series, "seed set size K", "running time (s)",
                   "Fig %sb - running time vs K (log scale)" % fignum,
                   xticks=list(cfg.K_list), logy=True)
        written.append(p2)

        header = ["method", "K", "spread_ours", "spread_sd", "spread_paper",
                  "seconds_ours", "seconds_paper", "select_seconds",
                  "fit_seconds_total", "fit_seconds_amortised",
                  "own_internal_estimate", "n_runs", "n_seeds", "status"]
        rows = []
        for m in methods:
            for K in cfg.K_list:
                p = res.k_cell(m, K)
                if p is None:
                    continue
                rows.append([
                    m, K,
                    "" if p.failed else _fmt(p.spread, 3),
                    "" if p.failed else _fmt(p.spread_sd, 3),
                    fig_a.get(K, {}).get(m, ""),
                    "" if p.failed else _fmt(p.seconds, 4),
                    fig_b.get(K, {}).get(m, ""),
                    _fmt(p.select_seconds, 4),
                    _fmt(p.fit_seconds, 4),
                    _fmt(p.fit_amortised, 4),
                    "" if p.failed else _fmt(p.own_estimate, 3),
                    p.n_runs, p.n_seeds,
                    ("FAILED: " + p.error) if p.failed else "ok",
                ])
        p3 = os.path.join(outdir, "fig%s_spread_and_time_vs_K.csv" % fignum)
        write_csv(p3, header, rows)
        written.append(p3)

    # ---- Fig 4 ------------------------------------------------------------
    if res.z_points:
        written += _write_sweep(outdir, "fig4", "Z", res.z_points, PAPER_FIG4,
                                "Fig 4 - impact of Z on CTIM (K=%d, C=%d)"
                                % (cfg.fig45_K, cfg.C))
    # ---- Fig 5 ------------------------------------------------------------
    if res.c_points:
        written += _write_sweep(outdir, "fig5", "C", res.c_points, PAPER_FIG5,
                                "Fig 5 - impact of C on CTIM (K=%d, Z=%d)"
                                % (cfg.fig45_K, cfg.Z))

    # ---- Table 2 ----------------------------------------------------------
    p_t2 = os.path.join(outdir, "table2_feature_comparison.csv")
    write_csv(p_t2, ["method", "community_based", "topic_aware",
                     "influence_computation_model", "reference"], TABLE2_ROWS)
    written.append(p_t2)

    # ---- report.md --------------------------------------------------------
    p_rep = os.path.join(outdir, "report.md")
    with open(p_rep, "w", encoding="utf-8") as fh:
        fh.write(build_report(res, written))
    written.append(p_rep)
    return written


def _write_sweep(outdir: str, stem: str, param: str,
                 points: List[SweepPoint], paper: dict, title: str) -> List[str]:
    """Two SVG panels (spread, time) plus one CSV for Fig 4 or Fig 5."""
    good = [p for p in points if not p.failed and not math.isnan(p.spread)]
    written = []

    ours_sp = [(p.value, p.spread) for p in good]
    ours_tm = [(p.value, p.seconds) for p in good if not math.isnan(p.seconds)]

    # Plot the paper's curve only where its grid overlaps ours.  A scaled-down
    # grid (e.g. C in 4..14 on a 400-user graph) shares no x values with the
    # paper's C in 25..150, and drawing both would stretch the axis to 150 and
    # crush our six points against the left edge -- a misleading chart. The
    # paper's full numbers are always in the CSV and in report.md regardless.
    ours_x = set(p.value for p in points)
    shared = sorted(v for v in paper if v in ours_x)
    paper_sp = [(v, paper[v][0]) for v in shared]
    paper_tm = [(v, paper[v][1]) for v in shared]
    series_sp = [("ours", ours_sp)] + ([("paper (digitised)", paper_sp)] if paper_sp else [])
    series_tm = [("ours", ours_tm)] + ([("paper (digitised)", paper_tm)] if paper_tm else [])

    p1 = os.path.join(outdir, "%s_%s_spread.svg" % (stem, param))
    line_chart(p1, series_sp,
               "number of %s" % ("topics Z" if param == "Z" else "communities C"),
               "influence spread I(S)", title + " - spread",
               xticks=[p.value for p in points])
    written.append(p1)

    p2 = os.path.join(outdir, "%s_%s_time.svg" % (stem, param))
    line_chart(p2, series_tm,
               "number of %s" % ("topics Z" if param == "Z" else "communities C"),
               "running time (s)", title + " - running time",
               xticks=[p.value for p in points], logy=True)
    written.append(p2)

    header = [param, "spread_ours", "spread_paper", "seconds_ours",
              "seconds_paper", "fit_seconds", "select_seconds",
              "n_runs", "n_nonempty_communities", "pp_max", "status"]
    rows = []
    for p in points:
        pv = paper.get(p.value, (None, None))
        if p.failed:
            status = "FAILED: " + p.error
        elif p.degenerate:
            # Say so in the data file, not only in the prose: this row's
            # spread is |S| by construction and is not a measurement.
            status = ("DEGENERATE (max pp < h; spread is an |S| artefact, "
                      "not a measurement)")
        else:
            status = "ok"
        rows.append([
            p.value,
            "" if p.failed else _fmt(p.spread, 3),
            pv[0] if pv[0] is not None else "",
            "" if p.failed else _fmt(p.seconds, 4),
            pv[1] if pv[1] is not None else "",
            _fmt(p.fit_seconds, 4), _fmt(p.select_seconds, 4),
            p.n_runs, p.n_communities_used,
            "" if (p.failed or math.isnan(p.pp_max)) else _fmt(p.pp_max, 5),
            status,
        ])
    p3 = os.path.join(outdir, "%s_%s_sweep.csv" % (stem, param))
    write_csv(p3, header, rows)
    written.append(p3)
    return written


# --------------------------------------------------------------------------
# report.md
# --------------------------------------------------------------------------

_TIMING_POLICY = """\
Wall clock is measured with `time.perf_counter`.  For the K sweep (Fig %(fig)sa/%(fig)sb):

    seconds(method, K) = mean over runs of (selection seconds)
                       + fit_seconds(method) / %(nk)d

`selection seconds` covers everything the method itself does per run and
EXCLUDES the shared evaluator, which is the harness's cost and is identical for
every method.  Where a baseline scores its own seed set inside its timed region
(CTIM_CGA) that cost is subtracted via its reported `eval_seconds`; where the
baseline accepts an injected evaluator (AIR+CGA) a free stand-in is injected and
the harness scores afterwards; CINEMA is called with `evaluate=False`.

`fit_seconds` is the method's one-off, K-independent and item-independent
preparation, amortised over the %(nk)d K values so that summing the plotted
curve reproduces the true cost of the whole sweep.  Both components appear
un-collapsed in the CSV.

| method | fit_seconds contains | selection seconds contains |
|---|---|---|
| CTIM | collapsed Gibbs sampling, Eq (1)-(9) | Eq (10)-(12) edge weights, Eq (19) detection, Algorithm 2 lines 1-45 |
| CTIM_CGA | the SAME Gibbs fit (shared with CTIM) | Eq (10)-(12) edge weights, Eq (19) detection, CGA DP + MixedGreedy |
| AIR+CGA | AIR EM fit + CNM modularity detection | AIR Eq (A2) edge weights, CGA DP + MixedGreedy |
| CINEMA | CNM modularity detection | conformity estimation + reweighting, DP + CELF greedy |
| Greedy | building the topic-averaged probability map | CELF greedy with Monte-Carlo IC |

CTIM and CTIM_CGA share one Gibbs fit, exactly as in the paper: SPEC.md
Section 7 states their ONLY difference is the influence computation model, so
charging the same model cost to both is what isolates that difference.

Fig 4 and Fig 5 report the UN-amortised `fit_seconds + mean selection seconds`,
because the paper attributes those two trends to Algorithm 1 and Algorithm 2
together ("both Algorithm 1 and Algorithm 2 depend on C").
"""


def build_report(res: ExperimentResults, written: Sequence[str]) -> str:
    """The full report.md: our numbers beside the paper's, then the verdicts."""
    cfg = res.config
    ds_name = res.dataset_summary.get("name", cfg.dataset_name())
    fig_a, fig_b, name_a, name_b = paper_tables_for(ds_name)
    fignum = name_a[0]
    methods = list(cfg.methods)
    out: List[str] = []
    W = out.append

    W("# CTIM reproduction report - `%s`" % ds_name)
    W("")
    W("Reference implementation of Huang, Shen, Meng, Chang & He, "
      "*Community-based influence maximization for viral marketing*, "
      "Applied Intelligence (2019).")
    W("")
    W("Generated %s. Standard library only, Python %d.%d."
      % (time.strftime("%Y-%m-%d %H:%M:%S"), sys.version_info[0], sys.version_info[1]))
    W("")

    # ---- 1. configuration -------------------------------------------------
    W("## 1. Configuration")
    W("")
    W(ascii_table(["setting", "value"], cfg.as_rows()))
    W("")

    # ---- 2. dataset -------------------------------------------------------
    W("## 2. Dataset and split")
    W("")
    ds_sum = res.dataset_summary
    W(ascii_table(["quantity", "value"], [
        ["users U", ds_sum.get("n_users")],
        ["directed links E", ds_sum.get("n_links")],
        ["items M", ds_sum.get("n_items")],
        ["attribute values F", ds_sum.get("n_attrs")],
        ["adoption logs", ds_sum.get("n_adoptions")],
        ["potential-influence logs D (Definition 1)",
         ds_sum.get("n_potential_influence_logs")],
        ["train (60%)", ds_sum.get("n_train")],
        ["validation (20%, held out)", ds_sum.get("n_valid")],
        ["test (20%)", ds_sum.get("n_test")],
        ["evaluated test items",
         ", ".join(str(i) for i in ds_sum.get("test_items", []))],
    ]))
    W("")
    W("Models are fitted on the TRAIN split only. The validation split is never "
      "touched by any method in this run. Evaluation items are the "
      "most-adopted items of the TEST split.")
    W("")

    # ---- 3. timing policy -------------------------------------------------
    W("## 3. What each method's running time includes")
    W("")
    W(_TIMING_POLICY % {"fig": fignum, "nk": max(1, len(cfg.K_list))})
    W("")

    # ---- 4. evaluator -----------------------------------------------------
    W("## 4. The shared evaluator")
    W("")
    W("Every method's seed set is scored with the same function: exact MIA "
      "influence, Eq (17)/(18), on the full graph, using the reference CTIM "
      "model's Eq (12) edge weights for the test item in question. No Monte "
      "Carlo. A method's own internal objective is reported separately in the "
      "CSV (`own_internal_estimate`) and never used as its spread.")
    W("")
    W("Topic-aware methods (CTIM, CTIM_CGA, AIR+CGA) are re-run for each test "
      "item and the reported spread is the mean over items. Topic-blind methods "
      "(CINEMA, Greedy - SPEC.md Table 2) produce one seed set per K which is "
      "then scored under every test item's weights.")
    W("")

    d = res.model_diag
    if d:
        W("### Did the model learn anything?")
        W("")
        if d.get("error"):
            W("Diagnostics unavailable: `%s`" % d["error"])
        else:
            W(ascii_table(["diagnostic", "value", "reference"], [
                ["communities C", d.get("C"), ""],
                ["distinct Eq (19) communities", d.get("n_communities_eq19"),
                 "out of C=%s" % d.get("C")],
                ["mean max_c pi[v][c] (Eq 7)", _fmt(d.get("mean_max_pi"), 4),
                 "uniform floor = 1/C = %s" % _fmt(d.get("uniform_max_pi"), 4)],
                ["max pp over edges (Eq 12)", _fmt(d.get("pp_max"), 4),
                 "MIA threshold h = %s" % _fmt(d.get("h"), 3)],
                ["median pp over edges", _fmt(d.get("pp_median"), 4), ""],
                ["fraction of edges with pp >= h", _fmt(d.get("pp_frac_ge_h"), 4),
                 "0 means nothing can propagate"],
            ]))
            W("")
            if d.get("degenerate"):
                W("**This configuration is DEGENERATE.** No learned edge "
                  "probability reaches `h`, so the Eq (15) arborescence of "
                  "every node is just that node, Eq (17) gives `ap(v|S)=1` "
                  "exactly on the seeds, and Eq (18) returns `I(S) = |S|` for "
                  "every method. The spread panels below carry no information "
                  "and no ordering claim can be tested against them. This is "
                  "a property of the configuration, not a failure of any "
                  "method: it happens when C is large relative to the graph, "
                  "because `pi` then spreads across many communities and the "
                  "Eq (12) product falls under `h`. Re-run with a smaller `--C` "
                  "or a larger graph.")
            elif (d.get("mean_max_pi") or 0.0) < 2.0 * (d.get("uniform_max_pi") or 0.0):
                W("**Weak community structure.** `mean max_c pi[v][c]` is less "
                  "than twice its uniform floor, so Eq (7) barely concentrates "
                  "users onto communities and the Eq (19) labelling that "
                  "CTIM and CTIM_CGA depend on is close to arbitrary. Any "
                  "ordering result below should be read with that in mind - "
                  "CTIM's advantage in the paper comes precisely from communities "
                  "that carry signal.")
            W("")

    # ---- 5. Fig Na --------------------------------------------------------
    if res.k_points:
        W("## 5. Fig %s - influence spread vs K (ours vs paper)" % name_a)
        W("")
        header = ["K"] + [c for m in methods for c in (m + " ours", m + " paper")]
        rows = []
        for K in cfg.K_list:
            row: List[Any] = [K]
            for m in methods:
                p = res.k_cell(m, K)
                if p is None:
                    row += ["-", fig_a.get(K, {}).get(m, "-")]
                elif p.failed:
                    row += ["FAIL", fig_a.get(K, {}).get(m, "-")]
                else:
                    row += [_fmt(p.spread, 1), fig_a.get(K, {}).get(m, "-")]
            rows.append(row)
        W(ascii_table(header, rows))
        W("")
        W("The paper's column is the digitised Fig %s of SPEC.md Section 9 "
          "(Yelp 2014 / Digg). Absolute values are not expected to match - the "
          "graph is different - so the acceptance criterion is the ORDERING and "
          "the curve SHAPE, checked in Section 9 below." % name_a)
        W("")

        # ---- 6. Fig Nb ----------------------------------------------------
        W("## 6. Fig %s - running time in seconds (ours vs paper)" % name_b)
        W("")
        header = ["K"] + [c for m in methods for c in (m + " ours", m + " paper")]
        rows = []
        for K in cfg.K_list:
            row = [K]
            for m in methods:
                p = res.k_cell(m, K)
                pv = fig_b.get(K, {}).get(m)
                if p is None:
                    row += ["-", _fmt_sci(pv) if pv else "-"]
                elif p.failed:
                    row += ["FAIL", _fmt_sci(pv) if pv else "-"]
                else:
                    row += [_fmt(p.seconds, 3), _fmt_sci(pv) if pv else "-"]
            rows.append(row)
        W(ascii_table(header, rows))
        W("")
        W("SPEC.md Section 9 digitises the paper's timing curve only at K=1 and "
          "K=51; the other paper cells are legitimately blank. Our absolute "
          "seconds are far smaller than the paper's because this graph is far "
          "smaller - the claim under test is the RATIO between methods, not the "
          "magnitude.")
        W("")
        if "CTIM" in methods:
            for label, attr in (("fit-amortised total (what Fig %s plots)" % name_b,
                                 "seconds"),
                                ("selection only (model fitting excluded)",
                                 "select_seconds")):
                W("Speed-up of CTIM over each baseline - %s:" % label)
                W("")
                hdr = ["K"] + ["%s / CTIM" % m for m in methods if m != "CTIM"]
                rows = []
                for K in cfg.K_list:
                    pc = res.k_cell("CTIM", K)
                    row = [K]
                    for m in methods:
                        if m == "CTIM":
                            continue
                        pm = res.k_cell(m, K)
                        base = getattr(pc, attr, 0.0) if pc else 0.0
                        val = getattr(pm, attr, float("nan")) if pm else float("nan")
                        if (pc is None or pm is None or pc.failed or pm.failed
                                or base <= 0 or math.isnan(val)):
                            row.append("-")
                        else:
                            row.append("%.1fx" % (val / base))
                    rows.append(row)
                W(ascii_table(hdr, rows))
                W("")
            W("Claim (b) is judged on the SELECTION-only table. CTIM and "
              "CTIM_CGA share one Gibbs fit by construction (SPEC.md Section 7: "
              "their only difference is the influence computation model), so "
              "fit cost enters both sides of that ratio identically and pulls "
              "it towards 1x whenever fitting dominates - which it does on "
              "small graphs, and does not on the paper's. Judging claim (b) on "
              "the total would therefore measure the shared model, not the "
              "influence computation model the paper's Fig %s is about." % name_b)
            W("")

    # ---- 7. Fig 4 ---------------------------------------------------------
    if res.z_points:
        W("## 7. Fig 4 - impact of Z on CTIM (K=%d, C=%d)" % (cfg.fig45_K, cfg.C))
        W("")
        rows = []
        for p in res.z_points:
            pv = PAPER_FIG4.get(p.value, (None, None))
            rows.append([
                p.value,
                "FAIL" if p.failed else _fmt(p.spread, 1),
                pv[0] if pv[0] is not None else "-",
                "FAIL" if p.failed else _fmt(p.seconds, 3),
                pv[1] if pv[1] is not None else "-",
                "-" if (p.failed or math.isnan(p.pp_max)) else _fmt(p.pp_max, 4),
                "DEGENERATE" if p.degenerate else ("FAIL" if p.failed else "ok"),
            ])
        W(ascii_table(["Z", "spread ours", "spread paper",
                       "seconds ours", "seconds paper", "max pp", "status"],
                      rows))
        W("")
        if any(p.degenerate for p in res.z_points):
            W("Rows marked DEGENERATE have no Eq (12) edge weight reaching "
              "`h = %.3g`, so Eq (15) admits no path, Algorithm 2 sees zero "
              "marginal gain for every candidate, and the reported spread is "
              "an `|S|` artefact rather than a measurement of influence. This "
              "is the `P(v|i,u) <= max_{c,z} theta_cz ~ 1/Z` bound of "
              "DEVIATIONS.md Section 7 biting at large `Z`, not a failure of "
              "any method. Those points are excluded from the plateau claim "
              "below and should be read as 'not measurable at this Z', not as "
              "'spread collapsed'." % cfg.h)
            W("")

    # ---- 8. Fig 5 ---------------------------------------------------------
    if res.c_points:
        W("## 8. Fig 5 - impact of C on CTIM (K=%d, Z=%d)" % (cfg.fig45_K, cfg.Z))
        W("")
        rows = []
        for p in res.c_points:
            pv = PAPER_FIG5.get(p.value, (None, None))
            rows.append([
                p.value,
                "FAIL" if p.failed else _fmt(p.spread, 1),
                pv[0] if pv[0] is not None else "-",
                "FAIL" if p.failed else _fmt(p.seconds, 3),
                pv[1] if pv[1] is not None else "-",
                "-" if (p.failed or math.isnan(p.pp_max)) else _fmt(p.pp_max, 4),
                "DEGENERATE" if p.degenerate else ("FAIL" if p.failed else "ok"),
            ])
        W(ascii_table(["C", "spread ours", "spread paper",
                       "seconds ours", "seconds paper", "max pp", "status"],
                      rows))
        W("")

    # ---- 9. claims --------------------------------------------------------
    W("## 9. Qualitative claims")
    W("")
    claims = check_claims(res)
    rows = [[c.key, c.text, "REPRODUCED" if c.reproduced else "NOT REPRODUCED"]
            for c in claims]
    W(ascii_table(["#", "claim", "verdict"], rows))
    W("")
    for c in claims:
        W("* **(%s) %s** - %s. %s"
          % (c.key, "REPRODUCED" if c.reproduced else "NOT REPRODUCED",
             c.text, c.detail))
    W("")

    # ---- 10. Table 2 ------------------------------------------------------
    W("## 10. Table 2 - feature comparison")
    W("")
    W(ascii_table(["method", "community-based", "topic-aware",
                   "influence computation model", "reference"], TABLE2_ROWS))
    W("")
    W("Reproduced from SPEC.md Section 7 / the paper's Table 2. The two "
      "columns the paper tabulates are the first two; the third is what "
      "actually separates CTIM from CTIM_CGA and is the source of the running "
      "time gap.")
    W("")

    # ---- 11. failures -----------------------------------------------------
    W("## 11. Failures and caveats")
    W("")
    if res.failures:
        W("The following runs raised and were recorded rather than aborting "
          "the experiment:")
        W("")
        for f in res.failures:
            W("* `%s`" % f)
    else:
        W("No method raised during this run.")
    W("")
    if res.notes:
        for n in res.notes:
            W("* note: %s" % n)
        W("")
    W("Caveats that materially affect how these numbers should be read:")
    W("")
    W("1. **The shared evaluator uses CTIM's learned edge weights.** API.md "
      "requires one evaluator for all methods and CTIM's Eq (12) weights are "
      "the only per-item topic-aware weights available. This favours CTIM and "
      "CTIM_CGA, which optimise (an approximation of) the same objective the "
      "evaluator computes. AIR+CGA optimises its own EM-fitted weights and is "
      "scored under CTIM's - a real handicap, and the honest reading of any "
      "CTIM > AIR+CGA gap must account for it.")
    W("2. **Greedy and CINEMA receive probabilities averaged over exactly the "
      "items they are scored on** (`topic_averaged_pp`), which is generous to "
      "them relative to a strict train/test separation.")
    W("3. **Absolute values are not comparable to the paper.** Yelp 2014 is "
      "retired and the paper's Digg snapshot is not the one here; SPEC.md "
      "Section 9 accordingly sets the acceptance criterion to ordering and "
      "shape, not magnitude.")
    if cfg.quick:
        W("4. **This is a `--quick` run.** Gibbs iterations, EM sweeps, "
          "Monte-Carlo sample counts and the number of test items are all "
          "reduced. Every code path is exercised, but the numbers are noisy "
          "and the Monte-Carlo baselines (CTIM_CGA, AIR+CGA, CINEMA, Greedy) "
          "are hit hardest by the reduction. Do not read a `--quick` verdict "
          "as the implementation's verdict.")
    W("")

    # ---- 12. files --------------------------------------------------------
    W("## 12. Files written")
    W("")
    for p in written:
        W("* `%s`" % p)
    W("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Top-level entry point used by scripts/run_experiments.py
# --------------------------------------------------------------------------


def run_experiments(cfg: ExperimentConfig) -> ExperimentResults:
    """Prepare, run and write everything for one dataset."""
    t0 = time.perf_counter()
    exp = Experiment(cfg)
    exp.prepare()
    res = exp.run()
    res.timings["total_seconds"] = time.perf_counter() - t0
    written = write_outputs(res)
    _progress("")
    _progress("=== wrote %d files ===" % len(written))
    for p in written:
        _progress("  " + p)
    _progress("")
    for c in check_claims(res):
        _progress("  (%s) %-9s %s"
                  % (c.key, "REPRODUCED" if c.reproduced else "NOT REPROD.", c.text))
    _progress("")
    _progress("total %.1fs" % res.timings["total_seconds"])
    return res


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    failures = []

    def check(name, cond, detail=""):
        if cond:
            print("  PASS  %s%s" % (name, (" -- " + detail) if detail else ""))
        else:
            failures.append(name)
            print("  FAIL  %s%s" % (name, (" -- " + detail) if detail else ""))

    print("ctim/experiments.py self-test")

    # -- paper target tables are well formed --------------------------------
    for nm, tbl in (("2a", PAPER_FIG2A), ("3a", PAPER_FIG3A)):
        check("paper Fig %s has all K" % nm,
              sorted(tbl.keys()) == sorted(PAPER_K_LIST))
        check("paper Fig %s has all methods" % nm,
              all(sorted(v.keys()) == sorted(ALL_METHODS) for v in tbl.values()))
    check("paper Fig 4 grid", sorted(PAPER_FIG4.keys()) == sorted(PAPER_Z_GRID))
    check("paper Fig 5 grid", sorted(PAPER_FIG5.keys()) == sorted(PAPER_C_GRID))
    check("paper Fig 5 peaks at C=100",
          max(PAPER_FIG5, key=lambda c: PAPER_FIG5[c][0]) == 100)

    # -- dataset -> paper table routing -------------------------------------
    check("digg routes to Fig 3", paper_tables_for("digg")[2] == "3a")
    check("yelp routes to Fig 2", paper_tables_for("yelp")[2] == "2a")

    # -- the claim (b) criterion must accept the PAPER'S OWN timings --------
    # A reproduction criterion that the paper itself fails is a broken
    # criterion.  Both digitised timing tables must satisfy "CTIM fastest at
    # every K" and ">=10x vs the slowest baseline at the largest K".
    for tbl, nm in ((PAPER_FIG2B, "2b"), (PAPER_FIG3B, "3b")):
        Ks = sorted(tbl)
        worst_ratio = min(tbl[K][m] / tbl[K]["CTIM"]
                          for K in Ks for m in tbl[K] if m != "CTIM")
        maxK = Ks[-1]
        best_ratio = max(tbl[maxK][m] / tbl[maxK]["CTIM"]
                         for m in tbl[maxK] if m != "CTIM")
        check("paper Fig %s: CTIM fastest at every K (%.2fx)" % (nm, worst_ratio),
              worst_ratio >= 1.0)
        check("paper Fig %s: >=10x vs slowest baseline at K=%d (%.1fx)"
              % (nm, maxK, best_ratio), best_ratio >= 10.0)
    # ...and the OLD criterion (>=10x vs every baseline at every K) must be
    # shown to reject the paper's own Digg data -- that is why it was replaced.
    _old = min(PAPER_FIG3B[K][m] / PAPER_FIG3B[K]["CTIM"]
               for K in PAPER_FIG3B for m in PAPER_FIG3B[K] if m != "CTIM")
    check("old claim-(b) criterion provably too strict (Digg min %.2fx < 10x)"
          % _old, _old < 10.0)

    # -- the ordering predicate ---------------------------------------------
    good = {"CTIM": 100.0, "CTIM_CGA": 100.0, "AIR+CGA": 90.0,
            "CINEMA": 80.0, "Greedy": 70.0}
    check("chain accepts the paper ordering", _chain_ok(good)[0])
    bad = dict(good)
    bad["Greedy"] = 95.0
    check("chain rejects a violation", not _chain_ok(bad)[0])

    # -- claim checking on synthetic results --------------------------------
    cfg = ExperimentConfig(dataset="x", K_list=(1, 11), methods=ALL_METHODS)
    r = ExperimentResults(config=cfg)
    r.dataset_summary = {"name": "toy"}
    for K, base in ((1, 100.0), (11, 300.0)):
        for j, m in enumerate(ALL_METHODS):
            # CTIM 100x faster than every baseline, on BOTH the selection time
            # (which claim (b) is judged on) and the fit-amortised total.
            secs = 0.01 if m == "CTIM" else 1.0
            r.k_points.append(MethodPoint(
                method=m, K=K, spread=base - 10.0 * j,
                seconds=secs, select_seconds=secs, n_runs=1, n_seeds=K))
    # CTIM_CGA must be <= CTIM, so give them the same spread
    for p in r.k_points:
        if p.method == "CTIM_CGA":
            p.spread = [q.spread for q in r.k_points
                        if q.method == "CTIM" and q.K == p.K][0]
    for z in PAPER_Z_GRID:
        r.z_points.append(SweepPoint(value=z, spread=(400.0 if z >= 8 else 100.0 * z),
                                     seconds=1.0 + 0.1 * z, n_runs=1))
    for c in PAPER_C_GRID:
        r.c_points.append(SweepPoint(value=c, spread=PAPER_FIG5[c][0] * 1.0,
                                     seconds=1.0, n_runs=1))
    claims = {c.key: c for c in check_claims(r)}
    check("claim (a) detected on clean synthetic results", claims["a"].reproduced,
          claims["a"].detail)
    check("claim (b) detected", claims["b"].reproduced, claims["b"].detail)
    check("claim (c) plateau detected", claims["c"].reproduced, claims["c"].detail)
    check("claim (d) peak at C=100 detected", claims["d"].reproduced,
          claims["d"].detail)

    # -- degenerate sweep points must not be read as measurements ----------
    # A point whose own model clears no edge over h returns |S| by
    # construction.  Left in, it looks like a spread collapse and drags the
    # plateau claim down; it must be excluded and reported instead.
    r_deg = ExperimentResults(config=cfg)
    r_deg.dataset_summary = {"name": "toy"}
    r_deg.k_points = list(r.k_points)
    for z in PAPER_Z_GRID:
        if z >= 10:  # the 1/Z bound bites: no weight clears h
            r_deg.z_points.append(SweepPoint(value=z, spread=20.0, seconds=1.0,
                                             n_runs=1, degenerate=True,
                                             pp_max=0.05))
        else:
            r_deg.z_points.append(SweepPoint(value=z, spread=(400.0 if z >= 8
                                                              else 100.0 * z),
                                             seconds=1.0, n_runs=1,
                                             pp_max=0.2))
    r_deg.c_points = list(r.c_points)
    cd = {c.key: c for c in check_claims(r_deg)}
    check("degenerate Z points are excluded from the plateau claim",
          "excluded as degenerate" in cd["c"].detail
          and "Z=10,12,14,16" in cd["c"].detail, cd["c"].detail)
    check("a degenerate |S| artefact is not read as a spread collapse",
          "20.0" not in cd["c"].detail, cd["c"].detail)
    text_deg = build_report(r_deg, ["a.svg"])
    check("report marks degenerate sweep rows", "DEGENERATE" in text_deg)

    # negative controls
    r2 = ExperimentResults(config=cfg)
    r2.dataset_summary = {"name": "toy"}
    for K in (1, 11):
        for j, m in enumerate(ALL_METHODS):
            # every method equally fast -> claim (b) must be rejected because
            # the ratio is 1.0x, not because the timings are missing
            r2.k_points.append(MethodPoint(method=m, K=K, spread=100.0 + 10.0 * j,
                                           seconds=1.0, select_seconds=1.0,
                                           n_runs=1))
    for z in PAPER_Z_GRID:  # no plateau: keeps climbing
        r2.z_points.append(SweepPoint(value=z, spread=100.0 * z, seconds=1.0, n_runs=1))
    for c in PAPER_C_GRID:  # peak at the wrong C
        r2.c_points.append(SweepPoint(value=c, spread=float(200 - c), seconds=1.0,
                                      n_runs=1))
    claims2 = {c.key: c for c in check_claims(r2)}
    check("claim (a) rejected when ordering is inverted", not claims2["a"].reproduced)
    check("claim (b) rejected when CTIM is not faster", not claims2["b"].reproduced)
    check("claim (c) rejected when spread keeps climbing", not claims2["c"].reproduced)
    check("claim (d) rejected when the peak is elsewhere", not claims2["d"].reproduced)

    # -- a failed cell must not crash the report ----------------------------
    r3 = ExperimentResults(config=cfg)
    r3.dataset_summary = {"name": "toy", "test_items": [1, 2]}
    r3.k_points.append(MethodPoint(method="CTIM", K=1, failed=True, error="boom"))
    r3.failures.append("CTIM K=1 item=0: RuntimeError: boom")
    text = build_report(r3, ["a.svg"])
    check("report survives a failed cell", "boom" in text and "FAIL" in text)
    check("report states verdicts", "NOT REPRODUCED" in text)

    # -- quick_config only shrinks ------------------------------------------
    _full = ExperimentConfig(dataset="x")
    q = quick_config(ExperimentConfig(dataset="x"))
    # The invariant is "quick only ever shrinks", not any particular constant.
    check("quick shrinks Gibbs iters",
          0 < q.gibbs_iters_topic <= _full.gibbs_iters_topic
          and 0 < q.gibbs_iters_comm <= _full.gibbs_iters_comm)
    # ...and it must leave enough sweeps for theta to clear the Eq (15)
    # threshold; below ~30 the model is vacuous on a real trace (see
    # quick_config).
    check("quick keeps enough Gibbs sweeps to be non-vacuous",
          q.gibbs_iters_topic >= 30 and q.gibbs_iters_comm >= 30)
    check("quick shrinks MC and items", q.n_mc <= 20 and q.n_test_items <= 3)
    check("quick caps logs per item for tractability",
          q.max_logs_per_item == QUICK_MAX_LOGS_PER_ITEM)
    check("quick respects an explicit log cap",
          quick_config(ExperimentConfig(dataset="x",
                                        max_logs_per_item=5)).max_logs_per_item == 5)
    check("quick keeps all five methods", tuple(q.methods) == ALL_METHODS)
    check("quick keeps the paper's Z and C grids",
          tuple(q.z_grid) == PAPER_Z_GRID and tuple(q.c_grid) == PAPER_C_GRID)

    print("")
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        sys.exit(1)
    print("all self-tests passed")
