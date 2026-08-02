# Deviations from the paper

Every place this implementation departs from Huang et al. (2019), and why.
Organised by cause. Items marked **[SPEC]** are decisions already recorded in
`SPEC.md` Section 6 ("Reading notes"); the rest were found while implementing
and testing.

Severity key:

| | meaning |
|---|---|
| **A** | changes what is computed; can move the reported numbers |
| **B** | changes how it is computed; provably identical results |
| **C** | fills a gap the paper leaves undefined |
| **D** | environment / data availability, outside our control |

---

## 1. Ambiguities and errors in the printed algorithm

### 1.1 **[SPEC]** Algorithm 2 line 35 was mis-transcribed — severity A (CORRECTED)

**Earlier revisions of this file and of SPEC.md were wrong.** They claimed the
paper's lines 35 and 36 contradict each other. They do not.

**Provenance of this correction.** Pages 6 and 9 of `viral_marketing.pdf` carry
**no text layer at all** — `pdftotext -bbox` returns zero words for both, because
Algorithm 1 and Algorithm 2 are images. No extraction mode can recover them, so
the transcription below was read off the *rendered* page. Anyone re-checking
Algorithm 1 or 2 must do the same; text-dump-based audits structurally cannot.

The paper (page 9 of `viral_marketing.pdf`) prints:

```
35:  I[m,k] = max( I[m-1,k],  I[C,k-1] + dI_m );
36:  if     I[C,k-1] + dI_m >= I[m-1,k]  then
```

Both lines read `I[C,k-1]`. SPEC.md Section 6 — under the heading "Verbatim from
the paper" — transcribed line 35 as `I[m,k-1] + dI_m`. That single character
error produced three downstream consequences, all now fixed:

1. This entry previously asserted an internal inconsistency in the paper. There
   is none; the recurrence is coherent as printed.
2. `ctim/ctim.py` implemented line 35 as `Iv[m][k-1] + dI_m` in **both**
   `dp_tiebreak` modes, so neither mode reproduced the printed recurrence, and
   the mode named `"paper-literal"` was not literal — it made only line 36 match.
3. The from-scratch oracle in `tests/test_ctim.py` repeated the same reading, so
   it was not independent of SPEC.md and the test suite could not detect any of
   this.

**What the printed recurrence means.** `I[C,k-1]` does not depend on `m`, so
unrolling line 35 across `m = 1..C` collapses the table:

```
I[C,k] = max_m ( I[C,k-1] + dI_m ) = I[C,k-1] + max_m dI_m
s[C,k] = argmax_m dI_m
```

The "dynamic program" therefore reduces to plain greedy over communities: each
round, take a seed from whichever community currently offers the largest
marginal gain. That matches the paper's own prose ("we propose to use dynamic
programming to choose which community the k-th seed node should come from", "As
CGA Algorithm in [22]") and matches CGA's formulation.

**What we do.** Three readings are selectable; the paper's is the default:

| `dp_tiebreak` | line 35 reference | line 36 reference |
|---|---|---|
| `"paper-true"` (**default**) | `I[C,k-1]` | `I[C,k-1]` |
| `"consistent"` | `I[m,k-1]` | `I[m,k-1]` |
| `"paper-literal"` | `I[m,k-1]` | `I[C,k-1]` |

`ctim/ctim.py:465`; mirrored in `ctim/baselines/cga.py` and
`ctim/baselines/air_cga.py`. All three are exercised by
`tests/test_ctim.py::TestDeterminismAndTiebreaks` and recorded in
`RunResult.extra["dp_tiebreak"]`, so the choice is always visible in results.

**Measured impact of the correction** (`scripts/check_line35.py`, real Digg,
`C=100 Z=8 K=20 h=0.1`):

| line-35 reading | `I(S)` Eq (18) | reported DP value |
|---|---|---|
| `paper-true` | 761.4040 | 181.1065 |
| `consistent` | 761.4040 | 169.9614 |
| `paper-literal` | 761.4040 | 169.9614 |

The final seed **set** is identical across all three (20/20 shared) and the
influence spread is unchanged, so no published figure moves. Only the selection
*order* and the reported `dp_value` differ. The `paper-true` DP value is the one
that is actually meaningful: 181.1065 equals the achieved sum of within-community
spreads (`I_39(10) + I_52(10)`), whereas the `consistent` table no longer tracks
the quantity it is maximising.

The seed set is insensitive here only because 98 of 100 communities carry no
internal propagation on this benchmark (see Section 7.1), leaving the allocation
with just two real options. On a graph with many live communities the readings
do diverge — the planted-community fixture in `ctim/ctim.py`'s self-test
(3 communities, all live) gives:

| reading | seeds | `I(S)` Eq (18) | DP value |
|---|---|---|---|
| `paper-true` | `[5, 39, 26, 7, 53]` | 12.3879 | 8.1352 |
| `consistent` | `[5, 39, 6, 4, 26]` | 12.9962 | 7.6714 |
| `paper-literal` | `[5, 39, 6, 4, 26]` | 12.9962 | 7.6714 |

**The faithful reading scores 4.7% *lower* global spread on that fixture.** That
is expected and is not an argument against it: the paper's line 35 collapses to
greedy over communities, which is not optimal, and this repository's purpose is
to reproduce the paper rather than to improve on it. The `consistent` mode
remains available for anyone who wants the better-performing variant, and the
mode is always recorded in `RunResult.extra["dp_tiebreak"]`.

Note the two columns move in opposite directions: `paper-true` has the higher DP
value but the lower Eq (18) spread. That is the objective mismatch of DEVIATIONS
1.2/1.3 showing through — the DP maximises the sum of *within-community* spreads
`I_m`, while the reported metric is the *global* `I` on the full graph.

### 1.2 **[SPEC]** Line 34: `I_m` is not defined in the paper — severity C

The paper writes `dI_m = max(I_m(S ∪ u) − I_m(S)), u ∈ c_m` without saying what
`I_m` is. We read it as **Eq (18) evaluated on community `m`'s node-induced
subgraph**, so `S ∩ c_m = S_m` and the term is `I_m(S_m ∪ {u}) − I_m(S_m)`.
This is the only reading under which the DP's additive decomposition across
communities is sound. `ctim/ctim.py:429` builds one `MIA` per community with
`nodes=members`.

**Consequence, measured.** Because `I_m` discards every cross-community path,
CTIM's *global* spread is strictly below unrestricted global MIA greedy on the
same weights. On our small planted fixtures the ratio is 0.80–0.90. This is a
property of the paper's algorithm, not of the implementation, and it is pinned
by `tests/test_ctim.py::test_community_restriction_costs_a_bounded_amount`.

### 1.3 **[SPEC]** Line 43 says `I`, but must mean `I_j` — severity A

Line 43 writes `u_k = argmax_{u ∈ c_j} (I(S_j ∪ u) − I(S_j))` with the *global*
`I`. Taken literally the dynamic program would optimise a quantity (`I_m`) it
never selects on, making lines 32–41 vacuous. We use `I_j`, which makes line 43
exactly the maximiser line 34 already computed for `m = j` — so it is answered
from the cached per-community incremental table at no extra cost.
`ctim/ctim.py:498`.

### 1.4 Line 42 can select an empty or exhausted community — severity C

`dI_m` for an empty community is a max over the empty set, i.e. `0`, so
`s[C,k]` may point at a community with no selectable node. The paper does not
say what to do. Returning fewer than `K` seeds would break every comparison, so
we fall back to the community with the largest available gain (lowest index on
ties) and count the event in `last_stats["n_fallbacks"]`. `ctim/ctim.py:481`.

### 1.5 Eq (19) tie-breaking is unspecified — severity C

`c^v_m ← argmax_c pi_{v,c}` with ties "broken arbitrarily". We break by **lowest
community index**, deterministically, so a run is reproducible.
`ctim/ctim.py:181`.

---

## 2. Normalisers and priors

### 2.1 **[SPEC]** Eq (3): the `beta` normaliser is `F`, not `M × F` — severity A

The paper writes the denominator of Eq (3) as a sum over `M × F`, because `A` is
an `M × F` matrix. But `psi_z` is a multinomial over **attribute values**
(Table 1: "multinomial over attribute values, specific to topic `z`, len F"), so
its Dirichlet normaliser must run over the `F` distinct attribute values, not
over all `M·F` matrix cells. Using `M·F` would inflate the smoothing mass by a
factor of `M` and flatten `psi` toward uniform. We use `F`.
`ctim/gibbs.py:322`, and the same `F·beta` in the Eq (1) sampler at
`ctim/gibbs.py:293`.

### 2.2 Eq (1)'s first denominator is kept although it cancels — severity B

`sum_Z (n_{i,z} + omega)` is constant in `z` for a fixed item, so it cancels in
the normalisation. SPEC keeps it for numerical clarity and so does the code
(`ctim/gibbs.py:277`). No effect on the sampled distribution.

### 2.3 The per-user denominators of Eq (4)/(6) are dropped — severity B

`sum_C (n_{u,c'} + rho)` and `sum_C (n_{v,c} + rho)` sum over *all* communities
and so are identical for every one of the `C²` candidate pairs. They cancel
exactly in the normalisation and are not computed. `ctim/gibbs.py:526`.
Verified against a brute-force evaluation of Eq (4)/(6) in the module self-test.

### 2.4 `eps0` is clamped to a positive floor — severity C

`eps0 = zeta·ln(N_neg/C²)` is non-positive whenever `C² ≥ N_neg` (small graphs,
large `C`) or `zeta ≤ 0`. At `eps0 = 0`, Eq (8) degenerates to
`eta_{c'c} = 1` for every pair, destroying all block structure. We clamp to
`EPS0_FLOOR = 1e-3` and record the reason in `Hyper.warning`, which
`train_model` prints. `ctim/gibbs.py:114`. The paper does not discuss the
degenerate regime; `zeta` itself is described only as "a tunable weight" with no
value given, so we default to `zeta = 1.0`.

---

## 3. Algebraic and structural reformulations (results provably identical)

### 3.1 **[SPEC]** Eq (11) is evaluated on all edges of `G`, not only pairs in `D` — severity A

Algorithm 2 lines 8–21 compute `P(v|z,u)` and `P(v|i,u)` by looping over
`d = (u,v,i) ∈ D`. But the diffusion graph of Eq (13)–(18) is `G`, not `D`: an
arc that carries no observed co-adoption still carries influence, and that is
the entire point of a *latent* model. Restricting the weights to `D` would leave
most of `G` with no propagation probability at all and make `MIIA(v,h)` depend
on which pairs happened to co-adopt.

**What we do.** `EdgeWeights.for_item(i)` evaluates Eq (12) for **every directed
edge of `G`**. `D` is used for *learning* the model (Eq (4)/(6)) and nothing
else. `ctim/influence.py:254`, `ctim/ctim.py:396`.

This is a genuine semantic change from the printed pseudocode and can move the
numbers; it is the reading that makes the algorithm coherent.

### 3.2 **[SPEC]** Lines 8–21 are replaced by an exact factorisation — severity B

As printed the loops cost `O(C²|D| + M|D|Z)`. We use

```
P(v|i,u) = Σ_c a_u[c] · pi_v[c] · thetabar_i[c]
a_u[c]        = Σ_c' pi_u[c'] · eta[c'][c]      (item independent, cached per user)
thetabar_i[c] = Σ_z P(z|i) · theta[c][z]        (user independent, cached per item)
```

which is algebraically identical and costs `O(UC² + EC)` per item.
`ctim/influence.py:111`. `tests/test_influence.py::test_edge_weights_equal_the_naive_reference_to_1e_9`
asserts agreement with the naive Eq (11)+(12) to 1e-9 relative on every edge of
a random instance.

### 3.3 Links and logs are sampled in one interleaved sweep — severity B

Eq (4) and Eq (6) share the count structures `n_{u,c}` and `n_{c'c}`, so they
are sampled in a single interleaved pass per iteration rather than in two
separate passes. The event order is shuffled once (seeded) and reused, keeping
runs deterministic. `ctim/gibbs.py:488`.

### 3.4 `n_{c,z}` is maintained incrementally — severity B

Eq (6)'s soft count `n_{c,z} = Σ_i n_{c,i} P(z|i)` is patched in `O(Z)` per
add/remove instead of being recomputed. `CommunitySampler.validate_counts()`
recomputes everything from the stored assignments and is asserted clean in
`tests/test_gibbs.py::test_incremental_counts_stay_exact`.

### 3.5 The topic `z_i` of a log is redrawn, not stored — severity B

Algorithm 1 line 24 draws `z_i ~ Mul(phi_i)` per log. Since `ncz` is a soft
count over the *whole* topic distribution, no count structure depends on the
drawn `z_i` — it only selects which column of `ncz` the Eq (6) theta-factor
reads. So it is redrawn at each visit and never stored. `ctim/gibbs.py:607`.

---

## 4. Additions the paper does not have

### 4.1 A Metropolis-Hastings sampler alternative — severity B (opt-in)

The exact Eq (4)/(6) sampler enumerates all `C²` pairs per event: at the paper's
`C = 100` that is 10,000 candidates *per link and per log per sweep*, which is
not tractable in pure Python on a large graph. `sampler="mh"`
(`ctim/gibbs.py:658`) proposes from the factorised `q(c',c) ∝ A[c']·B[c]` and
accepts on the `eta` ratio alone. Because the proposal is state-independent the
Hastings ratio collapses to `etatilde_new / etatilde_old`, so each move
satisfies detailed balance with respect to the exact full conditional: **the two
samplers have the same stationary distribution** and differ only in mixing
speed. The module self-test verifies this empirically against a brute-force
evaluation of Eq (6); `tests/test_gibbs.py::TestCommunitySamplerMH` runs the
whole estimator suite under it. The paper describes only the exact sampler.
Default is `"exact"`; `--sampler` selects.

### 4.2 CELF lazy-forward in Greedy — severity B

`greedy.py` uses Leskovec et al.'s CELF by default. Submodularity makes every
stale gain an upper bound, so CELF and plain greedy select **identical** seeds;
only the cost differs. Flagged in `RunResult.extra["use_celf"]` and disableable
with `use_celf=False`. Kempe et al.'s original algorithm has no CELF.

### 4.3 `pp` is clamped into `[0,1]` and clamps are counted — severity C

A learned `P(v|i,u)` read from disk, or produced by a variant sampler whose rows
do not sum to exactly 1, could fall outside `[0,1]`, which would corrupt
`−ln pp` in Eq (13)/(14) and make Eq (17) non-probabilistic. We clamp and expose
`EdgeWeights.n_clamped` / `.max_raw_pp` / `.min_raw_pp` rather than hide the
violation. With a well-formed model no clamp ever fires
(`tests/test_influence.py::test_no_clamping_for_a_well_formed_model`).

### 4.4 Approximations that are OFF by default — severity A when enabled

All are disabled by default and flagged in `RunResult.extra` when used:

| knob | what it approximates | flag |
|---|---|---|
| `EdgeWeights(top_c=n)` | truncates `pi` rows to their top `n` entries | `extra["edge_weights_approximate"]` |
| `build_potential_influence_logs(max_per_item=n)` | uniform reservoir subsample of `D` per item | reported by the harness |
| `cinema_select_seeds(cand_top=n)` | restricts candidates to the top-degree nodes per community | `extra["approximate"]` |
| dataset preparer `--target-users/-links/-items` | subsamples the raw graph | `meta.json["notes"]` |

### 4.5 AIR+CGA used to drop `dp_tiebreak` on the executing path — severity C (FIXED)

`air_cga_select_seeds` recorded the requested reading in
`extra["dp_tiebreak"]` and forwarded it to `cga_select_seeds_local`, but
`_resolve_cga_selector()` prefers the sibling `ctim.baselines.cga.cga_select_seeds`,
which always imports in this repository — so the local branch was dead and the
call site passed six positional arguments only. `dp_tiebreak` being keyword-only
on the sibling, it was silently dropped: a `--dp-tiebreak` run had CTIM and
CTIM_CGA honouring the flag while AIR+CGA ran its own default and *reported the
flag it had ignored*.

Fixed by offering the keyword and withdrawing it on `TypeError`, since API.md
fixes only the six-positional signature and says nothing about keyword extras.
Verified by spying on the sibling: all three readings now arrive.

No published number changes — every shipped run used the default, where both
sides agreed.

### 4.6 AIR+CGA detects communities with CNM, not with CGA's own detector — severity B

The paper (p.10) defines the baseline as "topic diffusion model AIR [35] with
community detection **and** seed-set selection of CGA". This repository supplies
CGA's *selection* (the Algorithm 2 DP plus MixedGreedy) but substitutes
**greedy modularity agglomeration (Clauset–Newman–Moore)** for CGA's detection:
`ctim/experiments.py:684` calls `detect_communities_modularity`, and
`cga_select_seeds` only ever receives a partition from its caller — CGA's own
detector is implemented nowhere in the repository.

The paper distinguishes the two explicitly. On p.8 it calls CGA's detector
"a two-step community detection algorithm ... the time complexity of which is
non-trivial", and on p.3 it attributes CNM [21] to OASNET [20], a *different*
method. So the substitution is not a reading of the paper; it is a stand-in.

This matters for the efficiency claim, not the spread claim: the paper's Fig 2b/3b
story rests on CGA-family baselines paying a detection cost CTIM avoids by folding
detection into Algorithm 1. A cheaper detector understates that cost and therefore
*flatters the baseline*, not CTIM. Wang et al. [22] is not available in this
environment, so implementing CGA's two-step detector faithfully was not possible.

### 4.7 Two non-paper modules ship inside the package — severity C

* `ctim/ea_dp.py` — an evolutionary-algorithm variant of the within-community
  search plus an exact `O(C K^2)` resource-allocation DP,
  `I[m,k] = max_{0<=j<=min(k,cap_m)} ( I[m-1,k-j] + I_m(j) )`. This is **not**
  Algorithm 2 line 35 and is not claimed to be. Reachable only from
  `scripts/run_ea_dp.py`; no results path imports it. Measured on real Digg it
  changes nothing (identical seed set and spread to Algorithm 2), which is why
  it stays a side experiment.
* `scripts/calibrate_h.py` — diagnostics for the Eq (15) threshold and for the
  learned parameters (normalised entropy of `theta`/`psi`/`pi`, MIP distance
  distribution, `I_h(S)` saturation). Entropy is not a paper quantity; it is an
  instrument for testing whether the model learned anything. Measured
  conclusion: on Digg at `C=100, Z=8`, `h=0.1` already captures 96.0 % of the
  saturated influence, so moving `h` buys ~4 % spread for 15-97x the evaluation
  cost — the threshold is **not** the bottleneck. See Section 7.1.

Neither is on any figure's code path, so no reported number depends on them.

---

## 5. Definition 1, the split, and other data decisions

### 5.1 **[SPEC]** Definition 1 requires `0 < t_q − t_p ≤ Δ` — severity C

The paper writes only `t_q − t_p ≤ Δ`, which admits `t_q = t_p` (simultaneous
adoption, no influence) and negative differences (`v` adopted *first*). We
require **strictly later** adoption by `v`. `ctim/dataset.py:392`.
Boundary behaviour is pinned by
`tests/test_dataset.py::test_boundary_dt_equals_delta_is_included` and
`::test_boundary_dt_equals_zero_is_excluded`.

### 5.2 **[SPEC]** `(u,v)` is required as a *directed* arc — severity C

Influence flows along `u → v`. `tests/test_dataset.py::test_wrong_edge_direction_is_excluded`
checks that a time-respecting pair with only the reverse arc present yields no
log, and that adding the forward arc creates exactly one.

### 5.3 `Δ` is never given a value — severity C

The paper says only that `Δ` "is set manually". We default to **30 days**
(`--delta`, `2592000` seconds). This directly controls `|D|` and is therefore
one of the most consequential undocumented choices in the paper.

### 5.4 Repeated adoptions collapse to the first — severity C

SPEC Section 8 says repeated reviews of the same item are dropped, but not which
one is kept. We keep the **earliest** adoption time per `(user, item)`, which is
the only choice consistent with "`v` was influenced by `u`".
`ctim/dataset.py:340`.

### 5.5 The 60/20/20 split is uniform at random over `D` — severity C

Section 5.2 gives the proportions but not the mechanism (random? temporal?). We
shuffle `D` uniformly with the supplied `random.Random` and cut 60/20/20.
Friend links are **not** split — per the paper all links belong to the test set
as well as being used for training. `ctim/dataset.py:427`.

### 5.6 Digg item attributes are *derived*, not observed — severity A

The model needs an item-attribute matrix `A ∈ {0,1}^{M×F}` (Algorithm 1 line 6),
but the public Digg 2009 release contains **no story topic, container or
category field** — only `(user, story, timestamp)`. The paper does not say where
its 7,100 Digg items' attributes came from. We derive them from each story's
observed diffusion profile (popularity, lifetime, time-of-day, weekday,
burstiness) plus a hashed story-id block. `ctim/datasets/digg.py`;
`meta.json["notes"]` states this explicitly. **Any Digg topic result is
therefore about derived features, not real content topics.**

### 5.7 Digg arc direction is reversed relative to the raw columns — severity A

A raw Digg row says "`user_id` became a fan of `friend_id`", i.e. the fan
subscribes to the friend's feed, so influence flows `friend_id → user_id` —
the reverse of the column order. That is our default (`--arc-direction
influence`); `raw` keeps the columns as printed. Getting this backwards silently
inverts the whole diffusion graph.

---

## 6. Data availability

### 6.1 The Yelp Dataset Challenge 2014 snapshot no longer exists — severity D

The paper's headline benchmark is the **Yelp Dataset Challenge 2014** release
(366,715 users / 2,949,285 links / 61,184 items). Yelp has retired that
snapshot and does not distribute it. The *current* official Yelp open dataset is
a different, substantially larger corpus (our stream of it found 150,346
businesses, 6,990,280 reviews, 1,987,897 users, 14,611,748 friend links).

**Consequence: the paper's Yelp numbers cannot be reproduced, only paralleled.**
`ctim/datasets/yelp.py` prints produced counts beside the paper's and records
the discrepancy in `meta.json["notes"]`. Fig 2/4/5 on Yelp should be read as
*shape and ordering* comparisons only. The Digg benchmark is still genuinely
available and is the more meaningful reproduction target.

### 6.2 Yelp is subsampled to fit the machine — severity A

366k users × 2.9M links at `C = 100` is several CPU-days of pure-Python Gibbs
sampling. The preparer subsamples to a target size; the ratio is recorded in
`meta.json`. Yelp results here are **scaled-down stand-ins**, labelled as such.

### 6.3 Baseline Monte-Carlo budgets are far below the paper's — severity A

Li et al. use 10,000 IC simulations per spread evaluation; our default `--n-mc`
is 200, and `--quick` uses far fewer. **Lowering `--n-mc` makes the baselines
both faster and worse, which flatters CTIM on both axes of Figs 2 and 3.** The
value used is always recorded in `report.md` and in `RunResult.extra["n_mc"]`.
Any timing comparison against the paper must account for this.

---

## 7. A structural problem: Eq (12) scale vs `h = 0.1`

**Severity A. This is the most important entry in this file.**

The paper fixes the MIA threshold at `h = 0.1` (Section 4.2.2) and uses `Z = 8`
topics (Section 5). Those two choices are in tension, and at scale they are
incompatible.

From the exact factorisation (Section 3.2 above),

```
P(v|i,u) = Σ_c a_u[c] · pi_v[c] · thetabar_i[c]
```

with `a_u[c] = Σ_c' pi_u[c'] eta[c'][c] ≤ 1` (a convex combination of `eta`
entries, each in `[0,1]`) and `thetabar_i[c] = Σ_z P(z|i) theta[c][z] ≤
max_z theta[c][z]`. Since `pi_v` is a distribution, the whole expression is a
convex combination, so

> **`P(v|i,u) ≤ max_{c,z} theta[c][z]`**

Whenever `theta` is diffuse — and `alpha = 50/Z` actively pushes it toward
uniform unless the community-topic soft counts are large — that bound is
approximately `1/Z`. So:

| `Z` | bound on any `pp` | vs `h = 0.1` |
|---|---|---|
| 2 | ~0.50 | fine |
| 8 | ~0.125 | marginal |
| 10+ | ~0.10 or less | **every edge is below threshold** |

Measured on `data/processed/synthetic_small` (400 users, 4,000 links, `C = 100`):

| `Z` | max `theta` | `1/Z` | max `pp` over all 4,000 edges | edges with `pp ≥ h` |
|---|---|---|---|---|
| 2 | 0.5017 | 0.5000 | 0.26296 | 3976 / 4000 |
| 8 | 0.1281 | 0.1250 | 0.06213 | **0 / 4000** |
| 16 | 0.0645 | 0.0625 | 0.03715 | **0 / 4000** |

At the paper's own `Z = 8`, **no edge clears `h = 0.1`**. Eq (15) then admits no
path, every `MIIA(v,h)` is the singleton `{v}`, and Eq (18) collapses to
`I(S) = |S|` — for *every* method, including all baselines. That is exactly what
the current `--quick` run on `synthetic_small` reports (spread = 1.0, 11.0,
21.0, 51.0 at K = 1, 11, 21, 51). The figures are structurally vacuous, not
merely inaccurate.

**Diagnostic.** If every method's spread equals `K` exactly, this is why.

**What this is not.** It is not an arithmetic error in any module. `EdgeWeights`
is asserted equal to the naive Eq (11)+(12) to 1e-9
(`tests/test_influence.py`), and Eq (17)/(18) are asserted against hand-computed
examples and against 60k-run Monte-Carlo IC. Each piece is individually
correct; the incompatibility is between the paper's `h` and the scale that
Eq (12) can produce under the paper's own priors.

### 7.1 RESOLVED on real Digg — it was under-training plus an over-large `C`

The open question above ("worth checking on real Digg before anything else")
has now been checked, and **resolution 1 is confirmed**. On the real Digg
benchmark at the paper's own `C = 100, Z = 8, h = 0.1`:

| Gibbs sweeps (each stage) | max `theta` | max `pp` | edges with `pp ≥ h` |
|---|---|---|---|
| 8  | 0.1278 | 0.0946 | **0 %** (vacuous) |
| 30 | 0.1281 | 0.1245 | 24.1 % |
| 80 | 0.1285 | 0.1247 | 24.1 % |

So the collapse was **not** structural on real data. Two separate causes were
being conflated, and both are now fixed in the harness:

1. **Under-training.** `theta` needs ~30 sweeps to concentrate enough for
   Eq (12) to clear `h`. The old `--quick` profile used 8, which lands at
   `pp_max = 0.095` — just below `h = 0.1`, hence a fully vacuous report.
   `quick_config` now floors both stages at 30 (converged by 80).
2. **`C` too large for the graph.** `rho = 50/C` dominates the Eq (7) counts
   when a user has only a handful of Eq (4)/(6) tokens. The
   `synthetic_small` measurements in the table above were taken at `C = 100`
   on a **400-user** graph — 4 users per community, where `pi_v` cannot leave
   its `1/C` floor. That is a property of the configuration, not of the model:
   the same generator at `C = 10` gives `pp ≈ 0.15` and propagates normally.
   `scale_C_to_dataset` now caps `C` for the quick profile at
   `n_users / 40`, which never binds on either paper benchmark
   (Yelp 3,667 and Digg 304 users per community at `C = 100`).

The `P(v|i,u) ≤ max_{c,z} theta[c][z]` bound stands and is still a regression
test — it is genuinely tight, and `Z ≥ 10` with `h = 0.1` remains structurally
impossible. Fig 4's `Z = 10..16` points are expected to be vacuous for that
reason, and the report labels them as such rather than silently plotting `|S|`.

**Remaining resolutions**, if the `Z ≥ 10` range matters:

1. ~~The paper's real datasets may yield more concentrated `theta`.~~
   **Confirmed above for `Z = 8`; still insufficient for `Z ≥ 10`.**
2. `h` may need to scale with `Z` (e.g. `h = 0.1/Z`), which the paper does not
   say but which is what keeps `MIIA` non-trivial.
3. `eta` and `theta` may be intended to combine multiplicatively *without* the
   `pi` convex combination shrinking them — but Eq (11) as printed is
   unambiguous, and the naive reference implementation agrees with the fast one.

The bound itself is pinned as a regression test:
`tests/test_influence.py::test_pp_is_bounded_by_the_largest_theta_entry`.

---

## 7.2 The claim-(b) reproduction criterion was too strict

The harness checks the paper's five qualitative claims (SPEC.md Section 10.5).
Claim (b), "CTIM is fastest by orders of magnitude", was originally tested as
*≥ 10× faster than every baseline at every K*.

**That criterion is wrong: the paper's own digitised numbers fail it.** From
SPEC.md Section 9, Fig 3b (Digg):

| | K=1 | K=51 |
|---|---|---|
| CTIM | 1.0e1 | 4.0e1 |
| Greedy | 1.3e1 (**1.30×**) | 1.2e3 (30×) |
| CTIM_CGA | 1.7e1 (1.70×) | 1.5e2 (**3.75×**) |
| AIR+CGA | 1.8e1 (1.80×) | 9.0e3 (225×) |

The smallest gap in the paper's own data is **1.30×**, not ≥ 10×. What the
paper actually shows is that CTIM is fastest *everywhere* and pulls away by
orders of magnitude *as K grows*, against the slowest baseline. The criterion
is now:

* CTIM is fastest vs every baseline at every K (ratio ≥ 1×), **and**
* at the largest K, the speed-up vs the **slowest** baseline is ≥ 10×.

Both digitised tables (Fig 2b Yelp and Fig 3b Digg) satisfy this, and that is
asserted in `python3 -m ctim.experiments` — a reproduction criterion the
original paper fails is a broken criterion, so the self-test now pins it
against the paper's own numbers in both directions (it also asserts the old
criterion provably rejects Digg, which is why it was replaced).

Claim (b) is judged on **selection** seconds, excluding model fitting: CTIM and
CTIM_CGA share one Gibbs fit by construction (SPEC.md Section 7 — their only
difference is the influence computation model), so fit cost enters both sides
of that ratio identically and drives it to 1× whenever fitting dominates.

## 7.3 `--quick` caps potential-influence logs per item

Definition 1 is quadratic in each item's adopter count, so real traces produce
far more logs than a smoke run can sample: Digg at the default `Delta = 30 d`
yields **3,827,719** potential-influence logs (2.3 M in the train split), and a
single O(C)-per-token Gibbs sweep over that does not finish in minutes.

`--quick` therefore sets `max_logs_per_item = 20` (API.md's documented
tractability knob for exactly this), taking Digg to 71,058 logs. An explicit
`--max-logs-per-item` always wins. This is a smoke-profile shortcut only — the
full profile remains uncapped, as the paper is.

## 7.4 CNM community detection rebuilt around a per-community heap

`detect_communities_modularity` (greedy modularity / Clauset-Newman-Moore) is
used by AIR+CGA and, in a second copy, by CINEMA. Both copies kept **one heap
entry per adjacent community pair** and, after each merge, re-priced and
re-pushed an entry for *every* neighbour of the merged community.

On the synthetic graphs that is fine. On real Digg it does not terminate:

| after N merges (of 30,258) | live communities | heap entries | pushes |
|---|---|---|---|
| 2,000 | 28,358 | 95,260 | 7,109 |
| 6,000 | 24,358 | 8,402,519 | 8,917,490 |
| 12,000 | 18,358 | 27,339,382 | 29,393,868 |

A merged community reaches ~7,000 neighbours, so each merge costs ~7,000
pushes and the heap grows superlinearly. Both copies ran >20 min without
finishing, which is what made `--quick` unusable on the real benchmark.

This is exactly the cost CNM's published data structure exists to avoid: keep
only each community's **best** partner in the global heap. The global maximum
over communities is identical to the global maximum over pairs, so **the merge
order and the returned partition are unchanged** — only the bookkeeping is.
Entries carry generation stamps for both endpoints; a stale pop re-prices that
community and re-publishes it (rather than dropping it), which preserves the
invariant that every live community with a neighbour has an entry in the heap.
Tie-breaking is kept byte-identical by canonicalising each entry's key to
`(-dQ, lower_id, higher_id)`, matching the original ordering.

Verified equal to the original implementation on every partition both can
compute (`synthetic_small` and `synthetic-small`, at the natural stopping point
and at several explicit `C` targets):

| detector | Digg, C=100 | result |
|---|---|---|
| AIR+CGA (`air_cga.py`) | >20 min, never finished → **168 s** | 100 communities, no singletons |
| CINEMA (`cinema.py`) | >10 min, never finished → **186 s** | 100 communities |

Note this is a pure performance fix, not an algorithmic one: no merge decision
changes, and the two detectors remain independent of the diffusion model, which
SPEC.md Section 7 makes the defining weakness of these baselines vs CTIM.

## 8. Test-suite scope

`tests/` covers `dataset`, `gibbs`, `influence`, `ctim` and the four baselines
(146 tests). Not covered by the unittest suite:

* `ctim/plotting.py` — has its own executable self-test (`python3 ctim/plotting.py`).
* `ctim/experiments.py`, `scripts/run_experiments.py`, `scripts/prepare_data.py`,
  `ctim/datasets/*` — exercised end-to-end by the `--quick` profile rather than
  by unit tests.

Statistical tests (planted-community and planted-topic recovery, MIA vs Monte
Carlo) use fixed seeds, so they are deterministic pass/fail, not flaky.
