# CTIM-G vs CTIM -- Yelp, K=20, paper protocol

Source: `yelp2.txt`, produced by

    python3 -u scripts/run_ctim_global.py \
      --dataset data/processed/yelp --cache data/processed/_yelp_model.pkl \
      --K 20 --h 0.1 --paper-eval --items all --methods "CTIM,CTIM-G" \
      --mc 0 --probe-sample 0

Graph: `U=366427  E=2949285  C=100`.  `test_items=[57914, 14239, 29323]`.
Seed 42.  Setup (dataset + model + EdgeWeights) 15.17s.  TOTAL 3099.48s.

## Grading protocol

`--paper-eval` sets `h_eval = h_sel = 0.1`, which is what the paper does: one
threshold for both selection and Eq (18).  That is valid for the paper's own
baselines, none of which maximise Eq (18) directly.  CTIM-G *does* maximise it,
on the full graph, so this comparison is circular in CTIM-G's favour and the
gap below is an upper bound on the true one.

On Yelp the circularity cannot be removed cheaply: `h_eval = 0.05` returns
byte-identical spreads (measured, `y1.log`), because all edges are live at
h=0.1 while 2-hop paths sit at `pp <= 1/Z^2 ~ 0.0156` -- the band [0.05, 0.1)
is empty.  Only MC-IC, or `h_eval <= 0.01`, can referee this independently.
Neither has been run.

## Result

| # | item | method | I @ h=0.1 | gap | seed overlap |
|---|-------|--------|-----------|--------|--------------|
| 1 | 57914 | CTIM   | 6841.9875 | --     | 20/20 |
| 1 | 57914 | CTIM-G | 7414.6047 | +8.369% | 14/20 |
| 2 | 14239 | CTIM   | 6843.0937 | --     | 20/20 |
| 2 | 14239 | CTIM-G | 7415.7900 | +8.369% | 14/20 |
| 3 | 29323 | CTIM   | 6846.2319 | --     | 20/20 |
| 3 | 29323 | CTIM-G | 7419.1532 | +8.368% | 14/20 |

Spread of the gap across items: 0.0008 percentage points.

## Time per run

| # | item | Eq (12) | CTIM select | CTIM-G select | scoring | item total |
|---|-------|---------|-------------|---------------|---------|------------|
| 1 | 57914 | 284.16s | 62.58s | 719.63s | 169.64s | 1251.97s |
| 2 | 14239 |  32.56s | 60.24s | 665.17s | 184.06s |  947.00s |
| 3 | 29323 |  30.97s | 59.28s | 622.75s | 153.20s |  882.20s |
|   | mean  | 115.90s | 60.70s | 669.18s | 168.97s | 1027.06s |

CTIM-G / CTIM wall clock = **11.02x**.

Eq (12) costs 284.16s on item 1 vs ~31.8s afterwards (8.9x).  That is one-time
warm-up, not a per-item cost; exclude it from any marginal-cost figure.

## CTIM-G phase breakdown

| # | solo | curves | DP | realise | unattributed |
|---|------|--------|----|---------|--------------|
| 1 | 144.70s (20.1%) | 504.81s (70.1%) | 0.01s | 22.50s (3.1%) | 48.61s (6.8%) |
| 2 | 138.75s (20.9%) | 479.21s (72.0%) | 0.00s | 26.82s (4.0%) | 20.39s (3.1%) |
| 3 | 148.46s (23.8%) | 442.91s (71.1%) | 0.00s | 21.70s (3.5%) |  9.68s (1.6%) |

The DP -- the actual Algorithm 2 contribution -- costs 0.00-0.01s.  ~71% of the
time is building the per-community gain curves.  That is the only phase worth
optimising.

## Complexity

CTIM

    O(nC) Eq (19)  +  sum_c O(n_c * t_c log t_c) init  +  O(CK) DP cells

    n arborescences, each on an INDUCED subgraph, so t_c is small.
    NOT MEASURED: ctim_select_seeds builds its MIA objects internally, so the
    driver cannot read the memo dicts back as counters.  The 11.02x above is a
    wall-clock ratio with no primitive count behind it.

CTIM-G

    O(n * t log t) global init  +  sum_c O(n_c * t log t) curves
      +  O(C K^2) DP  +  O(K * t^2) realisation

    n arborescences, each on the FULL graph, so t is the global mean.
    MEASURED: 476264 arborescences (= 1.30 n), mean size 11.9
    MEASURED: 3221 / 3221 / 3222 curve gain evals, 41 realisation gain evals
    Derived : 1.51 / 1.40 / 1.31 ms per arborescence

Caveat on the counts: arborescence count is NOT a reliable cost proxy.
Measured earlier, CTIM-G builds 17% more arborescences than GlobalGreedy, of
larger mean size, yet runs 4.8x faster.  The `[C]` section of
`scripts/run_ctim_global.py` claims "(count, mean size) IS the empirical cost";
that claim is refuted and still unfixed.

## Stability -- what the 3 items do and do not show

Identical across all three runs: 476264 arborescences, 18/100 communities used,
independence gap 10.64-10.65%, and the allocation

    {7:1, 8:1, 14:1, 15:1, 24:1, 34:1, 40:1, 44:1, 54:1, 55:1,
     58:2, 59:1, 60:1, 69:1, 75:1, 80:1, 87:1, 97:2}

`dp_value` differs only in the 4th significant digit (8203.70 / 8205.14 /
8209.22, a 0.07% spread).

Cause, measured on the Digg model (`data/processed/_calib_model.pkl`):
`theta_bar_i[c] = sum_z P(z|i) * theta[c][z]` is the ONLY item-dependent term
in Eq (12), and theta is near-uniform, so `theta_bar_i[c] ~ 1/Z = 0.125` for
every community and every item -- median spread across items 0.052%, max 0.268%.

So `sd = 0.00` establishes reproducibility, NOT robustness across items: the
three items are effectively the same item.  Any results table that averages
over items to raise confidence is empty for the same reason.
