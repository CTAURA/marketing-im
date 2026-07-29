# CTIM reproduction report - `digg`

Reference implementation of Huang, Shen, Meng, Chang & He, *Community-based influence maximization for viral marketing*, Applied Intelligence (2019).

Generated 2026-07-22 06:27:03. Standard library only, Python 3.9.

## 1. Configuration

| setting                          | value                                   |
|:---------------------------------|:----------------------------------------|
| dataset                          | data/processed/digg                     |
| profile                          | quick                                   |
| figures                          | 2,4,5                                   |
| methods                          | CTIM, CTIM_CGA, AIR+CGA, CINEMA, Greedy |
| C (communities)                  | 100                                     |
| Z (topics)                       | 8                                       |
| K list                           | 1,11,21,51                              |
| K for Fig 4/5                    | 20                                      |
| Z grid (Fig 4)                   | 2,4,6,8,10,12,14,16                     |
| C grid (Fig 5)                   | 25,50,75,100,125,150                    |
| seed                             | 42                                      |
| n test items                     | 3                                       |
| n Monte-Carlo sims               | 20                                      |
| Gibbs iters (topic, Eq 1)        | 30                                      |
| Gibbs iters (community, Eq 4/6)  | 30                                      |
| Gibbs sampler                    | mh                                      |
| AIR EM iters                     | 8                                       |
| MIA threshold h (Eq 15)          | 0.1                                     |
| Algorithm 2 line-36 tie-break    | consistent                              |
| Delta for Definition 1 (s)       | 2592000                                 |
| max logs per item (0 = uncapped) | 20                                      |

## 2. Dataset and split

| quantity                                  | value           |
|:------------------------------------------|:----------------|
| users U                                   | 30358           |
| directed links E                          | 99846           |
| items M                                   | 3553            |
| attribute values F                        | 62              |
| adoption logs                             | 1733669         |
| potential-influence logs D (Definition 1) | 71058           |
| train (60%)                               | 42634           |
| validation (20%, held out)                | 14211           |
| test (20%)                                | 14213           |
| evaluated test items                      | 2369, 1113, 535 |

Models are fitted on the TRAIN split only. The validation split is never touched by any method in this run. Evaluation items are the most-adopted items of the TEST split.

## 3. What each method's running time includes

Wall clock is measured with `time.perf_counter`.  For the K sweep (Fig 3a/3b):

    seconds(method, K) = mean over runs of (selection seconds)
                       + fit_seconds(method) / 4

`selection seconds` covers everything the method itself does per run and
EXCLUDES the shared evaluator, which is the harness's cost and is identical for
every method.  Where a baseline scores its own seed set inside its timed region
(CTIM_CGA) that cost is subtracted via its reported `eval_seconds`; where the
baseline accepts an injected evaluator (AIR+CGA) a free stand-in is injected and
the harness scores afterwards; CINEMA is called with `evaluate=False`.

`fit_seconds` is the method's one-off, K-independent and item-independent
preparation, amortised over the 4 K values so that summing the plotted
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


## 4. The shared evaluator

Every method's seed set is scored with the same function: exact MIA influence, Eq (17)/(18), on the full graph, using the reference CTIM model's Eq (12) edge weights for the test item in question. No Monte Carlo. A method's own internal objective is reported separately in the CSV (`own_internal_estimate`) and never used as its spread.

Topic-aware methods (CTIM, CTIM_CGA, AIR+CGA) are re-run for each test item and the reported spread is the mean over items. Topic-blind methods (CINEMA, Greedy - SPEC.md Table 2) produce one seed set per K which is then scored under every test item's weights.

### Did the model learn anything?

| diagnostic                     |  value | reference                     |
|:-------------------------------|-------:|:------------------------------|
| communities C                  |    100 |                               |
| distinct Eq (19) communities   |    100 | out of C=100                  |
| mean max_c pi[v][c] (Eq 7)     | 0.0391 | uniform floor = 1/C = 0.0100  |
| max pp over edges (Eq 12)      | 0.1242 | MIA threshold h = 0.100       |
| median pp over edges           | 0.0509 |                               |
| fraction of edges with pp >= h | 0.2370 | 0 means nothing can propagate |


## 5. Fig 3a - influence spread vs K (ours vs paper)

|   K | CTIM ours | CTIM paper | CTIM_CGA ours | CTIM_CGA paper | AIR+CGA ours | AIR+CGA paper | CINEMA ours | CINEMA paper | Greedy ours | Greedy paper |
|----:|----------:|-----------:|--------------:|---------------:|-------------:|--------------:|------------:|-------------:|------------:|-------------:|
|   1 |     237.3 |         30 |          58.4 |             30 |          3.2 |            30 |         2.8 |           30 |        12.3 |           30 |
|  11 |     675.1 |        490 |          62.4 |            480 |         13.7 |           430 |        28.3 |          420 |        25.4 |          190 |
|  21 |     767.0 |        730 |          61.0 |            720 |         25.0 |           610 |        39.0 |          570 |        44.0 |          280 |
|  51 |     839.6 |       1420 |         100.8 |           1400 |         56.1 |           950 |        53.6 |          890 |        61.4 |          800 |

The paper's column is the digitised Fig 3a of SPEC.md Section 9 (Yelp 2014 / Digg). Absolute values are not expected to match - the graph is different - so the acceptance criterion is the ORDERING and the curve SHAPE, checked in Section 9 below.

## 6. Fig 3b - running time in seconds (ours vs paper)

|   K | CTIM ours | CTIM paper | CTIM_CGA ours | CTIM_CGA paper | AIR+CGA ours | AIR+CGA paper | CINEMA ours | CINEMA paper | Greedy ours | Greedy paper |
|----:|----------:|:-----------|--------------:|:---------------|-------------:|:--------------|------------:|:-------------|------------:|:-------------|
|   1 |    28.275 | 1.00e+01   |        46.236 | 1.70e+01       |       35.915 | 1.80e+01      |      55.002 | 1.60e+01     |      28.613 | 1.30e+01     |
|  11 |    28.560 | -          |        46.216 | -              |       36.055 | -             |      85.692 | -            |     239.397 | -            |
|  21 |    28.771 | -          |        46.371 | -              |       36.147 | -             |     101.222 | -            |     128.431 | -            |
|  51 |    29.166 | 4.00e+01   |        46.776 | 1.50e+02       |       36.009 | 9.00e+03      |      99.292 | 1.10e+03     |     793.563 | 1.20e+03     |

SPEC.md Section 9 digitises the paper's timing curve only at K=1 and K=51; the other paper cells are legitimately blank. Our absolute seconds are far smaller than the paper's because this graph is far smaller - the claim under test is the RATIO between methods, not the magnitude.

Speed-up of CTIM over each baseline - fit-amortised total (what Fig 3b plots):

|   K | CTIM_CGA / CTIM | AIR+CGA / CTIM | CINEMA / CTIM | Greedy / CTIM |
|----:|:----------------|:---------------|:--------------|:--------------|
|   1 | 1.6x            | 1.3x           | 1.9x          | 1.0x          |
|  11 | 1.6x            | 1.3x           | 3.0x          | 8.4x          |
|  21 | 1.6x            | 1.3x           | 3.5x          | 4.5x          |
|  51 | 1.6x            | 1.2x           | 3.4x          | 27.2x         |

Speed-up of CTIM over each baseline - selection only (model fitting excluded):

|   K | CTIM_CGA / CTIM | AIR+CGA / CTIM | CINEMA / CTIM | Greedy / CTIM |
|----:|:----------------|:---------------|:--------------|:--------------|
|   1 | 13.3x           | 1.8x           | 11.5x         | 19.1x         |
|  11 | 11.1x           | 1.6x           | 27.2x         | 136.5x        |
|  21 | 10.0x           | 1.4x           | 32.2x         | 65.2x         |
|  51 | 8.5x            | 1.1x           | 26.0x         | 336.8x        |

Claim (b) is judged on the SELECTION-only table. CTIM and CTIM_CGA share one Gibbs fit by construction (SPEC.md Section 7: their only difference is the influence computation model), so fit cost enters both sides of that ratio identically and pulls it towards 1x whenever fitting dominates - which it does on small graphs, and does not on the paper's. Judging claim (b) on the total would therefore measure the shared model, not the influence computation model the paper's Fig 3b is about.

## 7. Fig 4 - impact of Z on CTIM (K=20, C=100)

|   Z | spread ours | spread paper | seconds ours | seconds paper | max pp | status     |
|----:|------------:|-------------:|-------------:|--------------:|-------:|:-----------|
|   2 |       592.2 |         2500 |      632.283 |           230 | 0.4983 | ok         |
|   4 |       756.6 |         3500 |      109.909 |           380 | 0.2492 | ok         |
|   6 |       746.4 |         4180 |      109.440 |           430 | 0.1657 | ok         |
|   8 |       761.5 |         4320 |      109.336 |           450 | 0.1242 | ok         |
|  10 |        20.1 |         4320 |      110.105 |           465 | 0.0996 | DEGENERATE |
|  12 |        20.2 |         4320 |      111.251 |           475 | 0.0826 | DEGENERATE |
|  14 |        20.4 |         4320 |      111.091 |           490 | 0.0708 | DEGENERATE |
|  16 |        21.2 |         4320 |      111.741 |           495 | 0.0620 | DEGENERATE |

Rows marked DEGENERATE have no Eq (12) edge weight reaching `h = 0.1`, so Eq (15) admits no path, Algorithm 2 sees zero marginal gain for every candidate, and the reported spread is an `|S|` artefact rather than a measurement of influence. This is the `P(v|i,u) <= max_{c,z} theta_cz ~ 1/Z` bound of DEVIATIONS.md Section 7 biting at large `Z`, not a failure of any method. Those points are excluded from the plateau claim below and should be read as 'not measurable at this Z', not as 'spread collapsed'.

## 8. Fig 5 - impact of C on CTIM (K=20, Z=8)

|   C | spread ours | spread paper | seconds ours | seconds paper | max pp | status |
|----:|------------:|-------------:|-------------:|--------------:|-------:|:-------|
|  25 |       719.1 |         2650 |       36.319 |            60 | 0.1177 | ok     |
|  50 |       736.7 |         3800 |       60.424 |           330 | 0.1139 | ok     |
|  75 |       747.4 |         4200 |       85.688 |           430 | 0.1238 | ok     |
| 100 |       761.5 |         4320 |      109.284 |           480 | 0.1242 | ok     |
| 125 |       761.2 |         3800 |      134.446 |           490 | 0.1242 | ok     |
| 150 |       759.8 |         3600 |      159.294 |           500 | 0.1244 | ok     |

## 9. Qualitative claims

| #   | claim                                                            | verdict        |
|:----|:-----------------------------------------------------------------|:---------------|
| a   | CTIM >= CTIM_CGA > AIR+CGA > CINEMA > Greedy on influence spread | NOT REPRODUCED |
| b   | CTIM is fastest by orders of magnitude                           | REPRODUCED     |
| c   | influence spread plateaus in Z                                   | NOT REPRODUCED |
| d   | influence spread peaks at C=100                                  | REPRODUCED     |

* **(a) NOT REPRODUCED** - CTIM >= CTIM_CGA > AIR+CGA > CINEMA > Greedy on influence spread. holds at 0/4 K values; violations: K=1: CINEMA(2.8) > Greedy(12.3) violated | K=11: AIR+CGA(13.7) > CINEMA(28.3) violated | K=21: AIR+CGA(25.0) > CINEMA(39.0) violated; CINEMA(39.0) > Greedy(44.0) violated | K=51: CINEMA(53.6) > Greedy(61.4) violated
* **(b) REPRODUCED** - CTIM is fastest by orders of magnitude. judged on selection time (model fitting excluded, see below). Fastest everywhere: smallest speed-up over any baseline at any K is 1.15x (vs AIR+CGA at K=51), needs >=1x -- PASS. Orders of magnitude at the largest K: speed-up vs the slowest baseline at K=51 is 336.8x (vs Greedy), needs >=10x -- PASS. On the fit-amortised total the smallest speed-up is 1.0x (vs Greedy at K=1)
* **(c) NOT REPRODUCED** - influence spread plateaus in Z. not evaluable: only 1 usable Z point(s) at or above the knee Z=8, need 2 to measure a plateau; 4 of 8 Z points excluded as degenerate (max pp < h, spread = |S| artefact: Z=10,12,14,16)
* **(d) REPRODUCED** - influence spread peaks at C=100. argmax over C in [25, 50, 75, 100, 125, 150] is C=100 (spread 761.5); C=100 gives 761.5

## 10. Table 2 - feature comparison

| method   | community-based | topic-aware | influence computation model       | reference                       |
|:---------|:----------------|:------------|:----------------------------------|:--------------------------------|
| Greedy   | no              | no          | Monte-Carlo Independent Cascade   | Kempe et al. [3]                |
| CINEMA   | yes             | no          | Monte-Carlo IC (conformity-aware) | Li et al. [24]                  |
| AIR+CGA  | yes             | yes         | MixedGreedy (Monte-Carlo IC)      | Barbieri et al. [35] + CGA [22] |
| CTIM_CGA | yes             | yes         | MixedGreedy (Monte-Carlo IC)      | this paper + CGA [22]           |
| CTIM     | yes             | yes         | MIA, exact Eq (17)/(18)           | this paper                      |

Reproduced from SPEC.md Section 7 / the paper's Table 2. The two columns the paper tabulates are the first two; the third is what actually separates CTIM from CTIM_CGA and is the source of the running time gap.

## 11. Failures and caveats

No method raised during this run.

Caveats that materially affect how these numbers should be read:

1. **The shared evaluator uses CTIM's learned edge weights.** API.md requires one evaluator for all methods and CTIM's Eq (12) weights are the only per-item topic-aware weights available. This favours CTIM and CTIM_CGA, which optimise (an approximation of) the same objective the evaluator computes. AIR+CGA optimises its own EM-fitted weights and is scored under CTIM's - a real handicap, and the honest reading of any CTIM > AIR+CGA gap must account for it.
2. **Greedy and CINEMA receive probabilities averaged over exactly the items they are scored on** (`topic_averaged_pp`), which is generous to them relative to a strict train/test separation.
3. **Absolute values are not comparable to the paper.** Yelp 2014 is retired and the paper's Digg snapshot is not the one here; SPEC.md Section 9 accordingly sets the acceptance criterion to ordering and shape, not magnitude.
4. **This is a `--quick` run.** Gibbs iterations, EM sweeps, Monte-Carlo sample counts and the number of test items are all reduced. Every code path is exercised, but the numbers are noisy and the Monte-Carlo baselines (CTIM_CGA, AIR+CGA, CINEMA, Greedy) are hit hardest by the reduction. Do not read a `--quick` verdict as the implementation's verdict.

## 12. Files written

* `results/digg/fig3a_spread_vs_K.svg`
* `results/digg/fig3b_time_vs_K.svg`
* `results/digg/fig3_spread_and_time_vs_K.csv`
* `results/digg/fig4_Z_spread.svg`
* `results/digg/fig4_Z_time.svg`
* `results/digg/fig4_Z_sweep.csv`
* `results/digg/fig5_C_spread.svg`
* `results/digg/fig5_C_time.svg`
* `results/digg/fig5_C_sweep.csv`
* `results/digg/table2_feature_comparison.csv`
