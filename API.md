# Module API contract — CTIM reference implementation

**Hard constraints for every module:**
* Python **3.9** compatible (no `match`, no `X | Y` runtime unions, no `dict[str,int]`
  in evaluated annotations unless `from __future__ import annotations` is present).
* **Standard library only.** Allowed: `math`, `random`, `time`, `json`, `os`, `sys`,
  `array`, `heapq`, `collections`, `itertools`, `bisect`, `csv`, `gzip`, `bz2`,
  `argparse`, `dataclasses`, `typing`, `pickle`, `multiprocessing`, `hashlib`, `statistics`.
  **Forbidden:** numpy, scipy, matplotlib, networkx, pandas, sklearn, tqdm — anything pip-installed.
* Every function implementing a paper equation carries a comment `# Eq (N)` or
  `# Algorithm 2, line NN`.
* Deterministic given a seed: all randomness goes through an explicit
  `random.Random(seed)` instance passed in or stored on the object. Never use the
  module-level `random.*` functions.
* No global mutable state.

Repository root: `/Users/maby/marketing-im`

---

## Canonical on-disk dataset format

A processed dataset lives in `data/processed/<name>/` and contains:

* `graph.tsv` — one directed edge per line: `u<TAB>v`, integer ids in `[0, n_users)`.
* `logs.tsv` — one adoption per line: `u<TAB>i<TAB>t`, `t` = integer unix seconds,
  `i` in `[0, n_items)`. Sorted by `t` ascending.
* `items.tsv` — one item per line: `i<TAB>f1,f2,f3` = comma-separated attribute ids
  in `[0, n_attrs)`. An item with no attributes has an empty second field.
* `meta.json` — `{"name":str,"n_users":int,"n_links":int,"n_items":int,"n_attrs":int,
  "n_logs":int,"source":str,"notes":str}`

---

## `ctim/dataset.py`

```python
@dataclass
class Dataset:
    name: str
    n_users: int
    n_items: int
    n_attrs: int
    edges: list            # list of (u, v) directed
    out_adj: list          # out_adj[u] -> list of v
    in_adj: list           # in_adj[v]  -> list of u
    logs: list             # list of (u, i, t), sorted by t
    item_attrs: list       # item_attrs[i] -> list of attribute ids (the bag w_i)

    def edge_set(self) -> set: ...

def load_dataset(path: str) -> Dataset: ...
def save_dataset(ds: Dataset, path: str) -> None: ...

def build_potential_influence_logs(ds: Dataset, delta: int,
                                   max_per_item: int = 0,
                                   rng=None) -> list:
    """Definition 1. Returns list of (u, v, i).
    A log exists iff (u,i,tp) and (v,i,tq) in logs, 0 < tq - tp <= delta,
    and (u,v) is a DIRECTED edge of G.
    max_per_item > 0 caps logs per item by uniform subsampling (for tractability);
    0 = no cap. Must be O(sum over items of adopters * out-degree) not O(n^2)."""

def split_logs(logs: list, rng) -> tuple:
    """Section 5.2: returns (train, valid, test) = 60/20/20 random split."""
```

---

## `ctim/gibbs.py`

```python
@dataclass
class Hyper:
    C: int; Z: int
    rho: float; alpha: float; beta: float; omega: float
    eps0: float; eps1: float
    zeta: float = 1.0

    @staticmethod
    def make(C, Z, n_users, n_links, n_logs, zeta=1.0) -> "Hyper":
        """rho=50/C, beta=0.01, alpha=50/Z, omega=50/Z,
        N_neg = U*(U-1)*(1+D/E) - D - E, eps0 = zeta*ln(N_neg/C^2), eps1 = 0.1"""

class TopicSampler:
    """Stage 1: Eq (1)-(3). Item = document, attribute value = word."""
    def __init__(self, item_attrs, n_attrs, hyper, rng): ...
    def run(self, n_iter: int) -> None: ...
    def phi(self) -> list:       # phi[i][z], Eq (2)
    def psi(self) -> list:       # psi[z][w], Eq (3)
    def p_z_given_i(self) -> list:  # == phi, "inferred by counts"

class CommunitySampler:
    """Stage 2: Eq (4)-(9)."""
    def __init__(self, n_users, edges, logs_d, p_z_given_i, hyper, rng,
                 sampler: str = "exact"): ...
        # logs_d: list of (u, v, i) potential-influence logs (TRAIN split)
        # sampler: "exact" -> full O(C^2) enumeration of Eq (4)/(6)
        #          "mh"    -> Metropolis-Hastings with proposal
        #                     q(c',c) ∝ (n_uc'+rho)(n_vc+rho)*thetafactor,
        #                     accept ratio on the eta term. Same stationary dist.
    def run(self, n_iter: int) -> None: ...
    def pi(self) -> list:        # pi[v][c], Eq (7)
    def eta(self) -> list:       # eta[c'][c], Eq (8)
    def theta(self) -> list:     # theta[c][z], Eq (9)

@dataclass
class Model:
    hyper: Hyper
    pi: list; eta: list; theta: list; p_z_given_i: list; phi: list; psi: list
    def save(self, path): ...
    @staticmethod
    def load(path) -> "Model": ...

def train_model(ds, logs_train, C, Z, n_iter_topic, n_iter_comm, rng,
                zeta=1.0, sampler="exact", verbose=False) -> Model: ...
```

---

## `ctim/influence.py`

```python
def community_to_community(model) -> list:
    """Eq (10). Returns P[c'][c][z] = eta[c'][c] * theta[c][z]."""

def user_to_user_topic(model, u, v, z) -> float:
    """Eq (11), naive reference implementation. O(C^2). Used by tests."""

def user_to_user_item(model, u, v, i) -> float:
    """Eq (12), naive reference implementation. Used by tests."""

class EdgeWeights:
    """Fast exact evaluation of Eq (11)+(12) for all edges of G.
    Uses a_u = pi_u^T @ eta precomputed once (O(U*C^2)), then
    P(v|i,u) = sum_c a_u[c] * pi_v[c] * thetabar_i[c]   with
    thetabar_i[c] = sum_z P(z|i) * theta[c][z].
    Must agree with user_to_user_item() to 1e-9 relative."""
    def __init__(self, model, ds, top_c: int = 0): ...
        # top_c > 0 truncates pi to its top_c entries for speed (approximation,
        # must be OFF by default and reported when used)
    def for_item(self, i: int) -> dict:
        """Returns {(u,v): pp} for every directed edge of G, for item i."""

class MIA:
    """Eq (13)-(18), Chen et al. [19] maximum influence arborescence model."""
    def __init__(self, n_users, out_adj, in_adj, pp: dict, h: float = 0.1): ...
    def miia(self, v) -> tuple:
        """Eq (15). Returns (nodes, parent_edges) of MIIA(v,h)."""
    def mioa(self, v) -> tuple:  # Eq (16)
    def ap(self, v, S) -> float:  # Eq (17)
    def influence(self, S) -> float:  # Eq (18) I(S) = sum_v ap(v|S)
    def greedy_incremental(self, K, candidates=None) -> list:
        """Standard MIA greedy with incremental IncInf updates via MIOA."""
    def marginal_gain(self, S, u) -> float:  # I(S ∪ u) - I(S)
```

`MIA.influence(S)` must be exact per Eq (17)/(18) — no Monte Carlo.
`MIA` must be constructible on a node-induced subgraph (for `I_m`, the
within-community spread of Algorithm 2 line 34).

---

## `ctim/ctim.py`

```python
def detect_communities(pi) -> list:
    """Eq (19). Returns comm[v] in [0, C)."""

def ctim_select_seeds(model, ds, item, K, h=0.1,
                      dp_tiebreak="paper-true",   # or "consistent"/"paper-literal"
                      edge_weights=None) -> list:
    """Algorithm 2 in full, lines 1-45. Returns the seed list of length K,
    in selection order. Must record per-line timings in `.last_stats`."""

@dataclass
class RunResult:
    seeds: list; spread: float; seconds: float; extra: dict
```

---

## `ctim/baselines/` (one file each, all exposing `select_seeds(...) -> RunResult`)

* `greedy.py` — `greedy_select(ds, pp, K, n_mc, rng, use_celf=True)`.
  Kempe et al. [3] with Monte-Carlo IC estimation (`n_mc` simulations).
  Topic-blind: `pp` must be a single topic-independent probability map
  (uniform `p` or the topic-averaged learned probabilities — document which).
  CELF lazy-forward is allowed and must be flagged in the result.
* `cga.py` — CGA of Wang et al. [22]: community detection + dynamic-programming
  seed allocation + **MixedGreedy** within the chosen community.
  Exposes `cga_select_seeds(comm, ds, pp, K, rng, n_mc)`.
  `ctim_cga_select(model, ds, item, K, rng, n_mc)` = CTIM's model + CGA's selection
  (this is the CTIM_CGA baseline; the ONLY difference from CTIM is the influence
  computation model).
* `cinema.py` — CINEMA (Li et al. [24]): conformity-aware community-based IM.
  Community detection is independent of the diffusion model here (that is the
  paper's criticism of it). Conformity is estimated from the adoption logs.
* `air_cga.py` — AIR topic-aware diffusion model of Barbieri et al. [35], parameters
  fit by **Expectation-Maximization** on the training logs, then CGA community
  detection + CGA seed selection.

---

## `ctim/plotting.py`

Pure-Python SVG. No matplotlib.

```python
def line_chart(path, series, xlabel, ylabel, title,
               xticks=None, logy=False, width=560, height=380,
               colors=None, ymin=None, ymax=None) -> None:
    """series: list of (label, [(x,y), ...]). Writes a standalone .svg.
    Must support log-scale y (Figs 2b, 3b) with decade gridlines."""

def write_csv(path, header, rows) -> None: ...
def ascii_table(header, rows) -> str: ...
```

---

## `ctim/experiments.py` + `scripts/run_experiments.py`

Reproduce, for a given processed dataset:
* **Fig 2/3a** influence spread vs `K in {1,11,21,31,41,51}` for the 5 methods
* **Fig 2/3b** running time vs the same `K`
* **Fig 4** impact of `Z in {2,4,6,8,10,12,14,16}` (K=20, C=100)
* **Fig 5** impact of `C in {25,50,75,100,125,150}` (K=20, Z=8)
* **Table 2** feature comparison

Outputs to `results/<dataset>/`: `*.svg`, `*.csv`, and `report.md` containing an ASCII
table that puts **our numbers next to the paper's digitised numbers** (Section 9 of
SPEC.md) for every figure.

Influence spread for a topic-aware method is measured for a *test item*; report the
mean over a fixed sample of test items (`--n-test-items`, default 10, chosen as the
most-adopted items in the test split) so the comparison against topic-blind Greedy is fair.
All methods are evaluated with the SAME evaluator: `MIA.influence(S)` on the full graph
with that item's edge weights. Record wall-clock seconds per method per K.

CLI:
```
python3 scripts/run_experiments.py --dataset data/processed/digg \
    --figures 2,4,5 --C 100 --Z 8 --seed 42 --out results/digg
```
Must have a `--quick` profile that finishes in a few minutes for smoke-testing.
</content>
