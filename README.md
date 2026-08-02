# CTIM — Community-based Influence Maximization for Viral Marketing

A reference implementation, in **pure Python 3.9 standard library**, of

> Huimin Huang, Hong Shen, Zaiqiao Meng, Huajian Chang, Huaiwen He.
> *"Community-based influence maximization for viral marketing."*
> **Applied Intelligence** (2019). DOI [10.1007/s10489-018-1387-8](https://doi.org/10.1007/s10489-018-1387-8)

No numpy, no scipy, no matplotlib, no networkx, no pandas — nothing pip-installed.
Charts are hand-written SVG. Every random draw goes through an explicit
`random.Random(seed)` instance, so every run is bit-reproducible from its seed.

`SPEC.md` is the ground truth extracted from the paper (every equation, both
algorithms, the hyperparameters and the digitised target numbers). `API.md` is
the module contract. `DEVIATIONS.md` lists every place this implementation
departs from the paper and why — **read it before trusting a number.**

---

## 1. What the paper is about

Influence maximization asks: given a social graph and a diffusion model, which
`K` users should you seed so that the expected number of eventually-influenced
users, `I(S)`, is largest? It is NP-hard, and the classical greedy algorithm of
Kempe et al. needs tens of thousands of Monte-Carlo simulations per candidate,
which does not scale.

CTIM attacks this with three ideas at once:

1. **A latent-variable model that learns communities and topics jointly** from
   the friendship graph *and* from the adoption logs (Algorithm 1). Most prior
   community-based methods detect communities from graph structure alone, which
   is unrelated to how influence actually flows — the paper's main criticism of
   CINEMA and CGA. Here, community membership `pi_v`, community-to-community
   influence strength `eta_{c'c}` and community topic interest `theta_{c,z}` are
   all inferred by one collapsed Gibbs sampler, so communities are defined by
   *diffusion*, not by edge density.
2. **Topic-aware, per-item edge weights.** The propagation probability along an
   arc depends on the item being marketed: `P(v|i,u)` (Eq 12) is a topic mixture
   of community-level influence, so a seed set is chosen *for a product*.
3. **A community-decomposed dynamic program (Algorithm 2)** that allocates the
   `K` seeds across communities, combined with the MIA arborescence model
   (Chen et al.) which evaluates `I(S)` **exactly and analytically** — no Monte
   Carlo. This is where the claimed orders-of-magnitude speed-up comes from.

The paper's headline claims: CTIM ≥ CTIM_CGA > AIR+CGA > CINEMA > Greedy on
influence spread; CTIM fastest by orders of magnitude; spread plateaus in `Z`
and peaks at `C = 100`.

---

## 2. What is implemented — equation → `file:line`

### Model and inference (Section 4.1, Algorithm 1)

| Paper | What it is | Location |
|---|---|---|
| Definition 1 | potential-influence log `d=(u,v,i)`, `0 < t_q−t_p ≤ Δ`, `(u,v) ∈ E` | `ctim/dataset.py:297` `build_potential_influence_logs` |
| Section 4.1.1 | `rho=50/C`, `beta=0.01`, `alpha=omega=50/Z`, `eps1=0.1`, `N_neg=U(U−1)(1+D/E)−D−E`, `eps0=zeta·ln(N_neg/C²)` | `ctim/gibbs.py:81` `Hyper.make` |
| **Eq (1)** | collapsed sampler for the item topic `z_ij` | `ctim/gibbs.py:285` (inside `TopicSampler._sweep`) |
| **Eq (2)** | `phi_iz` estimator (= `P(z\|i)`, "inferred by counts") | `ctim/gibbs.py:309` `TopicSampler.phi` / `:336` `p_z_given_i` |
| **Eq (3)** | `psi_zw` estimator | `ctim/gibbs.py:322` `TopicSampler.psi` |
| **Eq (4)** | sampler for link community indicators `s'_e, s_e` | `ctim/gibbs.py:567` (link branch of `_sweep_exact`) |
| **Eq (5)/(6)** | sampler for log community indicators `c'_d, c_d`, with the soft count `n_{c,z} = Σ_i n_{c,i} P(z\|i)` | `ctim/gibbs.py:585` (log branch of `_sweep_exact`) |
| Algorithm 1 line 24 | `z_i ~ Mul(phi_i)` drawn per log visit | `ctim/gibbs.py:607` |
| **Eq (7)** | `pi_vc` estimator | `ctim/gibbs.py:811` `CommunitySampler.pi` |
| **Eq (8)** | `eta_{c'c}` estimator | `ctim/gibbs.py:827` `CommunitySampler.eta` |
| **Eq (9)** | `theta_cz` estimator | `ctim/gibbs.py:844` `CommunitySampler.theta` |
| — | Metropolis-Hastings variant of Eq (4)/(6), same stationary law | `ctim/gibbs.py:658` `_sweep_mh` |
| — | two-stage driver | `ctim/gibbs.py:968` `train_model` |

### Influence strength and computation (Section 4.2)

| Paper | What it is | Location |
|---|---|---|
| **Eq (10)** | `P(c\|z,c') = eta_{c'c}·theta_{c,z}` | `ctim/influence.py:46` `community_to_community` |
| **Eq (11)** | `P(v\|z,u)`, naive `O(C²)` reference | `ctim/influence.py:71` `user_to_user_topic` |
| **Eq (12)** | `P(v\|i,u)`, naive reference | `ctim/influence.py:95` `user_to_user_item` |
| Eq (11)+(12) | exact `O(UC² + EC)` factorisation used in production | `ctim/influence.py:111` `EdgeWeights`, `:254` `for_item` |
| **Eq (13)/(14)** | `pp(P)` and `MIP(u,v)` via Dijkstra on `−ln pp` | `ctim/influence.py:333` `MIA._dijkstra` |
| **Eq (15)** | `MIIA(v,h)` | `ctim/influence.py:392` `MIA.miia` |
| **Eq (16)** | `MIOA(v,h)` | `ctim/influence.py:403` `MIA.mioa` |
| **Eq (17)** | `ap(v\|S)` on the arborescence | `ctim/influence.py:421` `_ap_on_tree` / `:460` `ap` |
| **Eq (18)** | `I(S) = Σ_v ap(v\|S)` | `ctim/influence.py:470` `MIA.influence` |
| — | exact marginal gain, incremental greedy | `ctim/influence.py:492`, `:513` |

### Seed selection (Section 4.2.3, Algorithm 2)

| Paper | What it is | Location |
|---|---|---|
| **Eq (19)** | `c^v_m ← argmax_c pi_{v,c}` (lines 22–24) | `ctim/ctim.py:164` `detect_communities` |
| Algorithm 2 lines 1–45 | the whole algorithm, with per-line comments | `ctim/ctim.py:323` `ctim_select_seeds` |
| lines 1–21 | Eq (10)+(11)+(12) fused into `EdgeWeights` | `ctim/ctim.py:396` |
| line 25 | `S = S_1 = … = S_C = ∅` | `ctim/ctim.py:413` |
| lines 26–31 | `I[·,·]`, `s[·,·]` initialisation | `ctim/ctim.py:435–441` |
| line 34 | `dI_m = max_u (I_m(S∪u) − I_m(S))`, `I_m` on the induced subgraph | `ctim/ctim.py:455`, backed by `_CommunitySeedState` at `:193` |
| line 35 | `I[m,k] = max(I[m−1,k], I[m,k−1] + dI_m)` | `ctim/ctim.py:460` |
| lines 36–40 | back-pointer, both tie-break readings | `ctim/ctim.py:465` |
| lines 42–44 | `j = s[C,k]`, pick `u_k`, update `S_j` and `S` | `ctim/ctim.py:476–504` |
| — | `RunResult(seeds, spread, seconds, extra)` | `ctim/ctim.py:142` |

### Baselines (Section 5.2, Table 2)

| Method | Community-based | Topic-aware | Location |
|---|---|---|---|
| Greedy (Kempe et al. [3], Monte-Carlo IC + CELF) | | | `ctim/baselines/greedy.py:277` `greedy_select` |
| CINEMA (Li et al. [24], conformity-aware) | ✓ | | `ctim/baselines/cinema.py:886` `cinema_select_seeds` |
| AIR+CGA (Barbieri et al. [35], EM-fitted, + CGA) | ✓ | ✓ | `ctim/baselines/air_cga.py:1075` `air_cga_select_seeds` |
| CGA (Wang et al. [22], DP + MixedGreedy) | ✓ | | `ctim/baselines/cga.py:481` `cga_select_seeds` |
| CTIM_CGA (CTIM's model, CGA's selection) | ✓ | ✓ | `ctim/baselines/cga.py:632` `ctim_cga_select` |

Every baseline module also exposes the uniform `select_seeds(...) -> RunResult`
entry point required by `API.md`.

### Supporting modules

* `ctim/dataset.py` — canonical on-disk format, Definition 1, 60/20/20 split.
* `ctim/datasets/` — preparers turning raw downloads into the canonical layout
  (`digg.py`, `yelp.py`, `synthetic.py`, `subsample.py`).
* `ctim/plotting.py` — hand-written SVG line charts with log-scale support,
  CSV writer, ASCII tables. No matplotlib.
* `ctim/experiments.py` + `scripts/run_experiments.py` — Figures 2–5, Table 2,
  claim checking, `report.md` generation.

---

## 3. Getting the data

The paper uses two benchmarks (SPEC.md Section 8):

| Dataset | Users | Links | Items | Availability |
|---|---|---|---|---|
| Yelp Dataset Challenge **2014** | 366,715 | 2,949,285 | 61,184 | **retired — no longer distributed by Yelp** |
| Digg 2009 | 30,358 | 99,846 | 7,100 | available (KONECT `digg-friends` + `digg-votes`) |

### Digg (recommended — real graph *and* real adoption logs)

Download the KONECT `digg-friends` and `digg-votes` archives into
`data/raw/`, then:

```bash
python3 scripts/prepare_data.py digg --raw-dir data/raw --out data/processed/digg
```

### Yelp

Yelp's *current* open dataset is a different, much larger snapshot than the 2014
Challenge release the paper used, so **absolute numbers cannot match the paper's
Yelp figures**. `scripts/stream_yelp.py` streams the 4.35 GB `Yelp-JSON.zip`
without ever materialising the ~10 GB of JSON, emitting compact intermediates
into `data/raw/yelp/stream/`; `scripts/prepare_data.py yelp` then subsamples
them to the canonical format.

```bash
python3 scripts/stream_yelp.py --out-dir data/raw/yelp/stream
python3 scripts/prepare_data.py yelp --raw-dir data/raw/yelp --out data/processed/yelp
```

### Synthetic (no download, always works)

```bash
python3 scripts/prepare_data.py synthetic --name synthetic_small \
    --users 400 --links 4000 --items 200 --attrs 40 --communities 10 --topics 8
```

Any processed dataset is a directory containing `graph.tsv`, `logs.tsv`,
`items.tsv` and `meta.json` exactly as specified in `API.md`. `meta.json`'s
`notes` field always records how the data was derived — in particular **whether
the adoption logs are real or simulated**.

---

## 4. Running the experiments

```bash
# full run
python3 scripts/run_experiments.py --dataset data/processed/digg \
    --figures 2,4,5 --C 100 --Z 8 --seed 42 --out results/digg

# smoke run, finishes in a few minutes
python3 scripts/run_experiments.py --dataset data/processed/synthetic_small \
    --quick --out results/synthetic_small
```

Useful flags: `--methods`, `--K-list`, `--n-test-items`, `--n-mc`,
`--gibbs-iters-topic`, `--gibbs-iters-comm`, `--sampler {exact,mh}`,
`--dp-tiebreak {paper-true,consistent,paper-literal}`, `--h`, `--delta`,
`--max-logs-per-item`, `--z-grid`, `--c-grid`, `--verbose`.
`python3 scripts/run_experiments.py --help` lists them all.

### Expected outputs

Written to `results/<dataset>/`:

```
fig2a_spread_vs_K.svg          Fig 2a/3a — influence spread vs K, 5 methods
fig2b_time_vs_K.svg            Fig 2b/3b — running time vs K, log-scale y
fig4_Z_spread.svg  fig4_Z_time.svg     Fig 4 — impact of Z (K=20, C=100)
fig5_C_spread.svg  fig5_C_time.svg     Fig 5 — impact of C (K=20, Z=8)
fig2_spread_and_time_vs_K.csv
fig4_Z_sweep.csv   fig5_C_sweep.csv
table2_feature_comparison.csv
report.md
```

`report.md` puts **our numbers next to the paper's digitised numbers**
(SPEC.md Section 9) for every figure, states exactly what each method's timing
includes, and ends with a pass/fail table for the paper's five qualitative
claims. Absolute values are not expected to match — the graph is different — so
the acceptance criterion is **ordering and curve shape**.

### Running the tests

```bash
python3 -m unittest discover -s tests -v      # 146 tests, ~9 s
```

Plain stdlib `unittest`; **pytest is not installed and is not required**. Each
module also carries an executable self-test:

```bash
python3 ctim/dataset.py     python3 ctim/gibbs.py     python3 ctim/influence.py
python3 ctim/ctim.py        python3 ctim/plotting.py
```

---

## 5. Runtime and scale caveats

**Read these before pointing this at a large graph.**

* **Everything is pure Python.** Expect a 30–100× slowdown against a
  numpy/C implementation. The Gibbs sampler is the bottleneck: the exact
  sampler for Eq (4)/(6) costs `O((E + D)·C²)` per sweep, so at the paper's
  `C = 100` that is 10,000 candidate pairs *per link and per log per sweep*.
  Use `--sampler mh` (Metropolis-Hastings, `O((E+D)·C)` per sweep, same
  stationary distribution) for anything large.
* **The full Yelp graph is out of reach here.** 366k users × 2.9M links at
  `C = 100` is several CPU-days of pure-Python Gibbs sampling. The dataset
  preparers subsample to a target size, and the subsampling ratio is recorded in
  `meta.json`. Treat Yelp results as *scaled-down stand-ins*, not reproductions.
* **`build_potential_influence_logs` can explode.** `|D|` grows with
  `Σ_i (adopters of i) × (out-degree)`. A popular item adopted by 10⁵ users on a
  graph with mean out-degree 8 contributes ~10⁶ candidate logs on its own. Use
  `--max-logs-per-item` (uniform reservoir subsampling) to cap it; the cap is
  reported.
* **`MIA` caches an arborescence per node.** Memory is `O(Σ_v |MIIA(v,h)|)`,
  which is fine at `h = 0.1` on a sparse graph but grows quickly as `h → 0`.
* **`I(S) = |S|` is the degenerate-run signature.** If every method's reported
  spread equals `K` exactly, the learned edge weights are all below the MIA
  threshold `h` and the run is vacuous. This is a real and easily-triggered
  regime — see the `Eq (12) scale vs h = 0.1` section of `DEVIATIONS.md`, which
  explains why it happens at the paper's own `Z = 8` and what to do about it.
* **Baseline Monte-Carlo counts are the tuning knob.** The paper's baselines use
  10,000 IC simulations per evaluation; the default here is 200 and `--quick`
  uses far fewer. Lowering `--n-mc` makes the baselines faster *and worse*, which
  flatters CTIM on both axes. The value used is always recorded in `report.md`
  and in `RunResult.extra`.

---

## 6. Repository layout

```
SPEC.md          ground truth extracted from the paper
API.md           module API contract
DEVIATIONS.md    every departure from the paper, with reasons
ctim/
  dataset.py     Dataset, canonical I/O, Definition 1, 60/20/20 split
  gibbs.py       Algorithm 1 + collapsed Gibbs, Eq (1)-(9)
  influence.py   Eq (10)-(18): edge weights + MIA
  ctim.py        Algorithm 2 + Eq (19)
  plotting.py    pure-Python SVG charts, CSV, ASCII tables
  experiments.py Figures 2-5, Table 2, claim checks, report.md
  baselines/     greedy.py, cga.py, cinema.py, air_cga.py
  datasets/      digg.py, yelp.py, synthetic.py, subsample.py
scripts/
  prepare_data.py    raw -> data/processed/<name>/
  stream_yelp.py     stream Yelp-JSON.zip without extracting it
  run_experiments.py CLI for the figures
tests/             plain-stdlib unittest suite
data/raw/          third-party downloads (not in version control)
data/processed/    canonical datasets
results/           figures, CSVs, report.md
```
