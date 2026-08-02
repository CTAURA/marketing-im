# CTIM Implementation Spec — ground truth extracted from the paper

Paper: Huimin Huang, Hong Shen, Zaiqiao Meng, Huajian Chang, Huaiwen He.
"Community-based influence maximization for viral marketing", *Applied Intelligence* (2019).
DOI 10.1007/s10489-018-1387-8.

This file is the single source of truth for the implementation. Every equation number
below matches the paper. Implementations MUST cite the equation number in a comment.

---

## 0. Notation (Table 1)

| Symbol | Meaning |
|---|---|
| `S`, `K` | seed set, size of seed set |
| `C`, `Z`, `M`, `F` | #communities, #topics, #items, #attributes |
| `I`, `U`, `E`, `D` | set of items / users / links / potential-influence logs |
| `U`, `E`, `D` | sizes of the above |
| `psi_z` | multinomial over attribute values, specific to topic `z` (len F) |
| `phi_i` | multinomial over topics of item `i` (len Z) |
| `z_i` | topic of item `i` |
| `w_i` | bag of attribute values of item `i` |
| `pi_v` | multinomial over communities, specific to user `v` (len C) |
| `c'_d, c_d` | communities of `u`, `v` for potential-influence log `d=(u,v,i)` |
| `s'_e, s_e` | communities of `u`, `v` for friend link `e=(u,v)` |
| `theta_c` | topic interest of community `c` (len Z) |
| `eta_{c'c}` | community-level (topic-irrelevant) influence strength |
| `eps = (eps0, eps1)` | Beta priors on `eta` |
| `rho, alpha, beta, omega` | Dirichlet priors on `pi_v`, `theta_c`, `psi`, `phi` |

`G = (U, E)` is a **directed** friend-relationship network.
`A in {0,1}^{M x F}` is the item-attribute matrix; `a_i` is item `i`'s attribute vector.

### Definition 1 — potential-influence log
`d = (u, v, i)` is constructed from two purchase logs `(u, i, t_p)` and `(v, i, t_q)`
where `t_q - t_p <= Delta` and `(u, v) in E`.
Meaning: `v` purchasing `i` was potentially influenced by `u`.
`Delta` is set manually.
**Implementation detail:** we require `0 < t_q - t_p <= Delta` (strictly later adoption by v),
and `(u,v) in E` as a *directed* edge (u influences v along the arc u->v).

### Definition 4 — community-topic relevance
`theta_cz` = as *propagation target*, community `c` influences the purchase of an item on topic `z`.

---

## 1. Generative process (Algorithm 1)

```
 1: for each topic z = 1..Z do
 2:     draw psi_z ~ Dir(beta)
 3: end for
 4: for each item i = 1..M do
 5:     draw z_i  ~ Mul(phi_i)
 6:     draw w_i  ~ Mul(psi_{z_i})
 7: end for
 8: for each community c = 1..C do
 9:     draw topic interest theta_c ~ Dir(alpha)
10:     for each community c' = 1..C do
11:         draw community-level diffusion prob eta_{c'c} ~ Beta(eps0, eps1)
12:     end for
13: end for
14: for each user v = 1..U do
15:     draw pi_v ~ Dir(rho)
16:     for each link e = (u,v) in E_v do
17:         draw u's community s'_e ~ Mul(pi_u)
18:         draw v's community s_e  ~ Mul(pi_v)
19:         draw existence indicator I_e ~ Ber(eta_{s'_e s_e})
20:     end for
21:     for each potential-influence log d = (u,v,i) in D_v do
22:         draw u's community c'_d ~ Mul(pi_u)
23:         draw v's community c_d  ~ Mul(pi_v)
24:         draw item i's topic indicator z_i ~ Mul(phi_i)
25:         draw existence indicator I_d ~ Ber(eta_{c'_d c_d} * theta_{c_d z_i})
26:     end for
27: end for
```

### Beta prior on eta (Section 4.1.1)
```
eps0 = zeta * ln(N_neg / C^2)
eps1 = 0.1
N_neg = U*(U-1)*(1 + D/E) - D - E
```
`zeta` is a tunable weight. (Negative samples are modelled implicitly, as in ref [37].)

---

## 2. Model inference — collapsed Gibbs sampling (Section 4.1.2)

Two independent sampling stages, because item-topic generation is independent of
influence diffusion:

* **Stage 1** — sample `z`; on convergence estimate `phi`, `psi`, and `P(z|i)`.
* **Stage 2** — sample `s, s', c, c'`; on convergence estimate `pi`, `theta`, `eta`.

### Eq (1) — sample latent topic `z_ij` for each attribute `w_ij` in matrix A
```
P(z_ij = z | z_-ij, w)  ∝  (n_{i,z} + omega) / sum_Z (n_{i,z} + omega)
                          * (n_{z,w_ij} + beta) / sum_{F} (n_{z,w} + beta)
```
* `n_{i,z}` = #times topic `z` observed with an attribute in item `i`
* `n_{z,w}` = #times attribute value `w` observed with topic `z`
* The first denominator is constant over `z` for a fixed `i` **only** if we exclude the
  current token consistently; keep it for numerical clarity, it cancels in normalisation.
* Vocabulary size for the `beta` normaliser is `F` (the number of distinct attribute
  values). The paper writes `M x F` because `A` is an `M x F` matrix.

### Eq (2), (3) — parameter estimates after stage 1
```
phi_iz = (n_{i,z} + omega) / sum_Z (n_{i,z} + omega)
psi_zw = (n_{z,w} + beta)  / sum_F (n_{z,w} + beta)
```
Item-topic relevance `P(z|i)` is inferred "by counts", i.e. Eq (2): `P(z|i) = phi_iz`.

### Eq (4) — sample community indicators `s'_e, s_e` for each link e=(u,v) in E
```
P(s'_e = c', s_e = c | s_-e, s'_-e, .)
  ∝ (n_{u,c'} + rho) / sum_C (n_{u,c'} + rho)
  * (n_{v,c}  + rho) / sum_C (n_{v,c}  + rho)
  * (n_{c'c}  + eps1) / (n_{c'c} + eps0 + eps1)
```
* `n_{u,c}` = #times user `u` acts as member of community `c` across **all links AND all
  potential-influence logs** (both as source and as target).
* `n_{c'c}` = #links from community `c'` to `c` **plus** #potential-influence logs with
  source community `c'` and target community `c`.
* Collapsed sampling ⇒ decrement the current assignment's counts before sampling,
  increment after.

### Eq (5) — sample `c'_d, c_d` for each potential-influence log d=(u,v,i) in D
```
P(c'_d = c', c_d = c | ., z_i = z)
  ∝ (n_{u,c'} + rho)/sum_C(...) * (n_{v,c} + rho)/sum_C(...)
  * (n_{c'c} + eps1)/(n_{c'c} + eps0 + eps1)
  * (n_{c,z} + alpha) / sum_Z (n_{c,z} + alpha)
```
`n_{c,z}` = #potential-influence logs relevant to topic `z` with community `c` as
propagation target.

### Eq (6) — the form actually used
Every log `d` is associated with topic `z_i` drawn from item `i`'s topic distribution.
Therefore `n_{c,z}` is replaced by its soft/expected count:
```
n_{c,z}  :=  sum_{i in M} n_{c,i} * P(z|i)
```
where `n_{c,i}` = #potential-influence logs about item `i` with community `c` as
propagation target. Substituting into Eq (5) gives Eq (6):
```
P(c'_d = c', c_d = c | ., z_i = z)
  ∝ (n_{u,c'} + rho)/sum_C(...) * (n_{v,c} + rho)/sum_C(...)
  * (n_{c'c} + eps1)/(n_{c'c} + eps0 + eps1)
  * ( sum_M n_{c,i} P(z|i) + alpha ) / sum_Z ( sum_M n_{c,i} P(z|i) + alpha )
```
**Procedure per log `d`:** first draw `z_i ~ Mul(phi_i)`, then sample the pair `(c', c)`
with the formula above. (Section 4.1.2 / Algorithm 1 line 24.)

**Implementation:** maintain `ncz_soft[c][z] = sum_i n_{c,i} * P(z|i)` incrementally.
When a log about item `i` gains/loses target community `c`, do
`ncz_soft[c][z] += / -= P(z|i)` for all `z` — O(Z) per update.

### Eq (7), (8), (9) — parameter estimates after stage 2
```
pi_vc     = (n_{v,c} + rho) / sum_C (n_{v,c} + rho)
eta_{c'c} = (n_{c'c} + eps1) / (n_{c'c} + eps0 + eps1)
theta_cz  = ( sum_M n_{c,i} P(z|i) + alpha ) / sum_Z ( sum_M n_{c,i} P(z|i) + alpha )
```

---

## 3. Influence strength inference (Section 4.2.1)

### Eq (10) — community-to-community, per topic
```
P(c | z, c') = eta_{c'c} * theta_{c,z}
```

### Eq (11) — user-to-user, per topic
```
P(v | z, u) = sum_{c,c'} pi_{v,c} * pi_{u,c'} * P(c | z, c')
```

### Eq (12) — user-to-user, per item
```
P(v | i, u) = sum_{z=1}^{Z} P(z|i) * P(v | z, u)
```

**Algebraic identity used for efficiency (exact, no approximation):**
```
P(v|i,u) = sum_z P(z|i) sum_{c,c'} pi_uc' pi_vc eta_{c'c} theta_cz
         = sum_{c,c'} pi_uc' eta_{c'c} pi_vc * thetabar_i[c]
   where   thetabar_i[c] = sum_z P(z|i) * theta_cz
```
So with `a_u = pi_u^T @ eta` (item-independent, computed once per user, O(C^2)):
```
P(v|i,u) = sum_c a_u[c] * pi_vc * thetabar_i[c]
```
which is O(C) per edge per item. This MUST reproduce the naive Eq (11)+(12)
bit-for-bit up to floating point; a unit test asserts it.

---

## 4. Influence computation model — MIA (Section 4.2.2)

### Eq (13) — propagation probability along a path P = <u=w1, w2, ..., wr=v>
```
pp(P) = prod_{k=1}^{r-1} pp(w_k, w_{k+1})
```

### Eq (14) — maximum influence path
```
MIP(u,v) = argmax_P { pp(P) | P in Path(G,u,v) }
```
Computed as a shortest path under edge weight `-ln pp(u,v)` (Dijkstra).

### Eq (15), (16) — arborescences, threshold `h`
```
MIIA(v,h) = union over u in U, pp(MIP(u,v)) >= h  of  MIP(u,v)
MIOA(v,h) = union over u in U, pp(MIP(v,u)) >= h  of  MIP(v,u)
```
**The paper sets `h = 0.1`.** If `pp(MIP(u,v)) < h`, `v` cannot be activated by `u`.

### Eq (17) — activation probability on MIIA(v,h)
```
ap(v|S) = 1                                            if v in S
        = 0                                            if N_in(v) = empty
        = 1 - prod_{w in N_in(v)} (1 - ap(w|S)*pp(w,v))  otherwise
```
`N_in(v)` = in-neighbours of `v` **within MIIA(v,h)**; computed recursively
(the arborescence is a tree/DAG rooted at v, so recursion terminates).

### Eq (18) — influence spread
```
I(S) = sum_{v in V} ap(v|S)
```

---

## 5. Community detection (Eq 19)

```
c^v_m  <-  argmax_c pi_{v,c}
```
Ties broken arbitrarily (we break by lowest community index, deterministically).

---

## 6. Algorithm 2 — community-based influence maximization

Verbatim from the paper:

```
Input:  directed graph G=(U,E), set of potential-influence logs D,
        pi, theta, eta, item-topic relevance P(z|i), seed size K
Output: seed node set S

 1: for c  = 1..C do
 2:   for c' = 1..C do
 3:     for z = 1..Z do
 4:       P(c | z, c') = eta_{c'c} * theta_{c,z}          # Eq (10)
 5:     end for
 6:   end for
 7: end for
 8: for d = (u,v,i) in D do
 9:   for c  = 1..C do
10:     for c' = 1..C do
11:       P(v | z, u) = sum_{c,c'} pi_vc * pi_uc' * P(c | z, c')   # Eq (11)
12:     end for
13:   end for
14: end for
15: for i = 1..M do
16:   for d = (u,v,i) in D do
17:     for z = 1..Z do
18:       P(v | i, u) = sum_{z=1..Z} P(z|i) * P(v | z, u)          # Eq (12)
19:     end for
20:   end for
21: end for
22: for v = 1..U do
23:   c^v_m <- argmax_c pi_{v,c}                                    # Eq (19)
24: end for
25: S = S_1 = S_2 = ... = S_C = empty
26: for k = 1..K do
27:   I[0,k] = 0;  s[0,k] = 0
28: end for
29: for m = 1..C do
30:   I[m,0] = 0
31: end for
32: for k = 1..K do
33:   for m = 1..C do
34:     dI_m = max( I_m(S union u) - I_m(S) ),  u in c_m
35:     I[m,k] = max( I[m-1,k], I[C,k-1] + dI_m )
36:     if I[C,k-1] + dI_m >= I[m-1,k] then
37:       s[m,k] = m
38:     else
39:       s[m,k] = s[m-1,k]
40:     end if
41:   end for
42:   j = s[C,k]
43:   u_k = argmax_{u in c_j} ( I(S_j union u) - I(S_j) )
44:   S_j = S_j union u_k ;  S = S union u_k
45: end for
```

### Reading notes / decisions (documented deviations)
1. **Lines 35 and 36 both read `I[C,k-1]`** — they agree, and the recurrence is
   internally consistent. Earlier revisions of this file mis-transcribed line 35 as
   `I[m,k-1] + dI_m`, which manufactured a contradiction that is not in the paper;
   see DEVIATIONS.md 1.1 for the correction and its consequences.
   Because `I[C,k-1]` does not depend on `m`, unrolling line 35 over `m` gives
   `I[C,k] = I[C,k-1] + max_m dI_m` and `s[C,k] = argmax_m dI_m`: the DP degenerates
   to picking the community with the largest marginal gain each round, which is what
   CGA [22] does and what the surrounding prose describes.
   `dp_tiebreak="paper-true"` (the default) implements this. The two historical
   readings `consistent` and `paper-literal` remain selectable for comparison, and
   the choice is recorded in the results.
2. `I_m(.)` is the influence spread computed **within community m's induced subgraph**,
   so `S ∩ c_m = S_m`; we pass `S_m`.
3. Line 11 computes `P(v|z,u)` only for pairs in `D`; but Eq (11) is well-defined for any
   pair. The diffusion graph is `G`, so we compute edge weights for **all edges of `G`**
   via Eq (12) and use `D` only for learning. Pairs never co-adopting still get a
   (small) learned probability, which is the intended semantics of a latent model.
4. Lines 8-21 as written are O(C^2 D + M D Z); we use the exact factorisation from
   Section 3 above, which is algebraically identical but O(U C^2 + E C) per item.

---

## 7. Baselines (Section 5.2, Table 2)

| Method | Community-based | Topic-aware |
|---|---|---|
| Greedy | | |
| CINEMA | yes | |
| AIR+CGA | yes | yes |
| CTIM_CGA | yes | yes |
| CTIM | yes | yes |

* **Greedy** — the original greedy algorithm of Kempe et al. [3], Monte-Carlo IC.
  Topic-blind and community-blind: no `Z`, no `C`.
* **CINEMA** — Li et al. [24], conformity-aware community-based IM.
* **AIR+CGA** — the AIR topic diffusion model of Barbieri et al. [35] (learned by
  Expectation-Maximization) combined with CGA's [22] community detection and seed selection.
* **CTIM_CGA** — CTIM's latent variable model, but CGA's seed-set selection
  (MixedGreedy as influence computation model instead of MIA).
  "The only difference between CTIM_CGA and CTIM is influence computation model."

---

## 8. Experimental setup (Section 5)

### Datasets
| Dataset | Users | Links | Items |
|---|---|---|---|
| Yelp Dataset Challenge 2014 | 366,715 | 2,949,285 | 61,184 |
| Digg | 30,358 | 99,846 (directed arcs) | 7,100 |

For Yelp: review time is used as purchase time; repeated reviews of the same item are dropped.

### Split
60% of potential-influence logs = train, 20% = validation,
20% of potential-influence logs **and all links of the friendship graph** = test.

### Hyperparameters (fixed)
```
rho   = 50 / C
beta  = 0.01
alpha = 50 / Z
omega = 50 / Z
eps0  = zeta * ln(N_neg / C^2)      eps1 = 0.1
h (MIA threshold) = 0.1
```

Verified against the rendered page 10 (the symbol font has no ToUnicode map, so
`pdftotext` yields `{, , , , }` in every mode and this pairing cannot be
recovered from the text layer): *"As regards to the hyperparameters
{alpha, rho, beta, omega, epsilon}, we adopt a fixed value, i.e., rho = 50/C,
beta = 0.01, alpha = 50/Z, omega = 50/Z, and eps0, eps1 are set as Section 4."*

Note `alpha = 50/Z` puts a **total** Dirichlet mass of 50 on `theta_c` whatever
`Z` is — a strong pull toward the uniform `1/Z`. Combined with the
`P(v|i,u) <= max_{c,z} theta_cz` bound this is the mechanism behind
DEVIATIONS.md Section 7, and it is the paper's own choice, not a deviation.

### Parameter grids
* `K` from 1 to 50 (figures plot K in {1, 11, 21, 31, 41, 51})
* `Z` from 2 to 16 (Fig. 4: Z in {2,4,6,8,10,12,14,16}, with K=20, C=100)
* `C` from 25 to 150 (Fig. 5: C in {25,50,75,100,125,150}, with K=20, Z=8)
* Performance comparison fixes `Z=8, C=100` for CTIM and CTIM_CGA, `C=100` for
  AIR+CGA and CINEMA.

### Metrics
Influence spread `I(S)` (Eq 18) and running time (seconds).

---

## 9. Target numbers digitised from the paper's figures

These are the values the implementation is expected to reproduce **in shape and
ordering**. Absolute values depend on the exact dataset snapshot (Yelp 2014 is retired),
so the acceptance criterion is *qualitative ordering + curve shape*, with absolute
values reported side by side.

### Fig. 2a — Yelp, influence spread vs K (Z=8, C=100)
| K | CTIM | CTIM_CGA | AIR+CGA | CINEMA | Greedy |
|---|---|---|---|---|---|
| 1  | 900  | 900  | 700  | 500  | 300  |
| 11 | 3000 | 3000 | 2700 | 2400 | 2000 |
| 21 | 4350 | 4300 | 3900 | 3400 | 2900 |
| 31 | 4700 | 4700 | 4400 | 3900 | 3300 |
| 41 | 4950 | 4900 | 4700 | 4300 | 3700 |
| 51 | 5050 | 5000 | 4700 | 4600 | 3950 |

### Fig. 2b — Yelp, running time (sec, log scale) vs K
| K | CTIM | CTIM_CGA | CINEMA | AIR+CGA | Greedy |
|---|---|---|---|---|---|
| 1  | 2.0e2 | 3.0e3 | 1.0e4 | 2.0e4 | 3.0e3 |
| 51 | 7.0e2 | 3.0e4 | 1.4e5 | 2.0e5 | 2.5e4 |

Ordering (fastest → slowest): **CTIM << CTIM_CGA < Greedy < CINEMA < AIR+CGA**.
CTIM is "faster than all the baselines by orders of magnitude".

### Fig. 3a — Digg, influence spread vs K
| K | CTIM | CTIM_CGA | AIR+CGA | CINEMA | Greedy |
|---|---|---|---|---|---|
| 1  | 30   | 30   | 30  | 30  | 30  |
| 11 | 490  | 480  | 430 | 420 | 190 |
| 21 | 730  | 720  | 610 | 570 | 280 |
| 31 | 1000 | 990  | 730 | 700 | 520 |
| 41 | 1210 | 1190 | 830 | 790 | 610 |
| 51 | 1420 | 1400 | 950 | 890 | 800 |

### Fig. 3b — Digg, running time (sec, log scale) vs K
| K | CTIM | CTIM_CGA | CINEMA | AIR+CGA | Greedy |
|---|---|---|---|---|---|
| 1  | 1.0e1 | 1.7e1 | 1.6e1 | 1.8e1 | 1.3e1 |
| 51 | 4.0e1 | 1.5e2 | 1.1e3 | 9.0e3 | 1.2e3 |

### Fig. 4 — impact of Z on Yelp (K=20, C=100)
| Z | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| spread | 2500 | 3500 | 4180 | 4320 | 4320 | 4320 | 4320 | 4320 |
| time (s) | 230 | 380 | 430 | 450 | 465 | 475 | 490 | 495 |

Expected shape: spread rises sharply for Z=2..8 then **plateaus** (robustness to Z);
time rises **relatively stably** (only Algorithm 2 depends on Z).

### Fig. 5 — impact of C on Yelp (K=20, Z=8)
| C | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|
| spread | 2650 | 3800 | 4200 | **4320 (peak)** | 3800 | 3600 |
| time (s) | 60 | 330 | 430 | 480 | 490 | 500 |

Expected shape: spread **peaks at C=100** then declines; running time is **more
sensitive to C than to Z** (both Algorithm 1 and Algorithm 2 depend on C).

---

## 10. Acceptance criteria for this implementation

1. Every equation (1)-(19) implemented and unit-tested where testable.
2. Algorithm 1 and Algorithm 2 implemented line-by-line, with line-number comments.
3. Pure Python standard library only (no numpy / scipy / matplotlib / networkx).
   Python 3.9 compatible.
4. Runs end-to-end on a real benchmark dataset and reproduces the five figures
   as SVG + CSV + printed tables.
5. Qualitative claims reproduced:
   - CTIM >= CTIM_CGA > AIR+CGA > CINEMA > Greedy on influence spread
   - CTIM fastest by orders of magnitude
   - spread plateaus in Z, peaks at C=100
6. Any deviation from the paper is documented in `DEVIATIONS.md` with a reason.
</content>
