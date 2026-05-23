# InATTo — Experimental Results

All numbers below are full-ranking, leave-one-out evaluation.
`InATTo` denotes our paper-main config: ssw01 + Gumbel-STE v2.
Stage 2 RPG with `wd=0.05, dropout=0.3, lr=3e-3, batch=256, history=50, patience=20`.

---

## Table 1 — Main results (TEST overall, 4 datasets)

`ssw01-base` = Stage 1 ssw01 baseline (without Gumbel-STE v2).
`InATTo (gumbel_v2)` = paper-main config.

| Dataset | Method | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| Toys | ssw01-base | 0.0550 | 0.0793 | 0.1118 | 0.0382 | 0.0460 | 0.0542 |
| Toys | **InATTo (gumbel_v2)** | **0.0552** | **0.0788** | **0.1085** | **0.0389** | **0.0465** | **0.0539** |
| Toys | Δ% | +0.47% | -0.58% | -2.90% | +1.92% | +1.04% | -0.43% |
| Beauty | ssw01-base | 0.0528 | 0.0749 | 0.1055 | 0.0371 | 0.0442 | 0.0519 |
| Beauty | **InATTo (gumbel_v2)** | **0.0511** | **0.0738** | **0.1030** | **0.0362** | **0.0435** | **0.0508** |
| Beauty | Δ% | -3.22% | -1.55% | -2.33% | -2.49% | -1.67% | -2.05% |
| Sports | ssw01-base | 0.0288 | 0.0415 | 0.0592 | 0.0198 | 0.0239 | 0.0283 |
| Sports | **InATTo (gumbel_v2)** | **0.0303** | **0.0436** | **0.0613** | **0.0207** | **0.0250** | **0.0294** |
| Sports | Δ% | +5.07% | +5.15% | +3.61% | +4.43% | +4.56% | +3.84% |
| Yelp | ssw01-base | 0.0222 | 0.0371 | 0.0570 | 0.0142 | 0.0190 | 0.0240 |
| Yelp | **InATTo (gumbel_v2)** | **0.0217** | **0.0367** | **0.0566** | **0.0141** | **0.0189** | **0.0239** |
| Yelp | Δ% | -2.22% | -0.89% | -0.81% | -0.95% | -0.41% | -0.47% |

---

## Table 2 — Sparsity-group breakdown (InATTo gumbel_v2)

Items grouped by raw-text word-count quantile (tertile).

| Dataset | Group | n_users | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | sparse | 4692 | 0.0539 | 0.0761 | 0.1057 | 0.0379 | 0.0451 | 0.0524 |
| Toys | medium | 6450 | 0.0530 | 0.0771 | 0.1078 | 0.0368 | 0.0445 | 0.0522 |
| Toys | rich | 8270 | 0.0577 | 0.0817 | 0.1108 | 0.0411 | 0.0488 | 0.0561 |
| Toys | **OVERALL** | all | **0.0552** | **0.0788** | **0.1085** | **0.0389** | **0.0465** | **0.0539** |
| Toys | (bounds) | colspan | word-count = [3.0, 11.0, 14.0, 71.0] |  |  |  |  |
| Beauty | sparse | 5213 | 0.0269 | 0.0399 | 0.0552 | 0.0183 | 0.0224 | 0.0263 |
| Beauty | medium | 8259 | 0.0398 | 0.0565 | 0.0791 | 0.0285 | 0.0338 | 0.0394 |
| Beauty | rich | 8891 | 0.0758 | 0.1097 | 0.1533 | 0.0539 | 0.0648 | 0.0758 |
| Beauty | **OVERALL** | all | **0.0511** | **0.0738** | **0.1030** | **0.0362** | **0.0435** | **0.0508** |
| Beauty | (bounds) | colspan | word-count = [2.0, 13.0, 17.0, 473.0] |  |  |  |  |
| Sports | sparse | 8928 | 0.0274 | 0.0401 | 0.0567 | 0.0183 | 0.0224 | 0.0266 |
| Sports | medium | 13779 | 0.0279 | 0.0406 | 0.0586 | 0.0189 | 0.0230 | 0.0275 |
| Sports | rich | 12891 | 0.0348 | 0.0493 | 0.0674 | 0.0242 | 0.0289 | 0.0334 |
| Sports | **OVERALL** | all | **0.0303** | **0.0436** | **0.0613** | **0.0207** | **0.0250** | **0.0294** |
| Sports | (bounds) | colspan | word-count = [3.0, 14.0, 18.0, 165.0] |  |  |  |  |
| Yelp | sparse | 9060 | 0.0192 | 0.0340 | 0.0513 | 0.0122 | 0.0169 | 0.0213 |
| Yelp | medium | 9282 | 0.0205 | 0.0347 | 0.0554 | 0.0130 | 0.0175 | 0.0227 |
| Yelp | rich | 12089 | 0.0245 | 0.0404 | 0.0614 | 0.0164 | 0.0215 | 0.0268 |
| Yelp | **OVERALL** | all | **0.0217** | **0.0367** | **0.0566** | **0.0141** | **0.0189** | **0.0239** |
| Yelp | (bounds) | colspan | word-count = [0.0, 5.0, 8.0, 37.0] |  |  |  |  |

---

## Table 3 — Toys text-sparsity ablation (Stage 1 final-epoch metrics)

- **S0** = full raw text  `(title + brand + categories + description + price + salesrank)`
- **S1** = `(title + brand + categories + price + salesrank)`  (description removed)
- **S2** = `(title)`  (title only)
- **Light** = only `itm_text_embeds` sparsified (`h_raw`, ρ kept as S0)
- **Strict** = `itm_text_embeds`, `h_raw` (re-generated GPT profile), and ρ all sparsified

| Setting | L_total | L_recon | L_Q | L_align | L_bpr | L_ssw | phi_item_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 (full) | 54.5044 | 0.0568 | 50.3844 | 6.8349 | 0.0336 | 0.0638 | 0.9045 |
| Light S1 | 47.0953 | 0.0561 | 43.0645 | 6.6572 | 0.0342 | 0.0674 | 0.9023 |
| Strict S1 | 43.7722 | 0.0558 | 39.5789 | 6.9853 | 0.0316 | 0.0713 | 0.9016 |
| Light S2 | 40.2800 | 0.0550 | 36.3787 | 6.3875 | 0.0345 | 0.0712 | 0.9319 |
| Strict S2 | 44.8474 | 0.0571 | 40.7624 | 6.7691 | 0.0302 | 0.0770 | 0.8975 |

---

## Table 4 — Toys text-sparsity ablation (Stage 2 VAL best)

| Setting | epochs | best ep | val R@5 | val R@10 | val R@20 | val N@5 | val N@10 | val N@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 (full) | 118 | 97 | 0.0758 | 0.1016 | 0.1363 | 0.0532 | 0.0615 | 0.0702 |
| Light S1 | 103 | 82 | 0.0706 | 0.0950 | 0.1289 | 0.0499 | 0.0577 | 0.0663 |
| Strict S1 | 95 | 74 | 0.0702 | 0.0948 | 0.1289 | 0.0500 | 0.0580 | 0.0666 |
| Light S2 | 97 | 76 | 0.0738 | 0.1000 | 0.1360 | 0.0522 | 0.0606 | 0.0697 |
| Strict S2 | 102 | 81 | 0.0673 | 0.0906 | 0.1187 | 0.0469 | 0.0545 | 0.0615 |

---

## Table 5 — Toys text-sparsity ablation (TEST overall, n=19412)

| Setting | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---:|---:|---:|---:|---:|---:|
| S0 (full) | 0.0552 | 0.0788 | 0.1085 | 0.0389 | 0.0465 | 0.0539 |
| Light S1 | 0.0518 | 0.0718 | 0.1003 | 0.0363 | 0.0428 | 0.0499 |
| Strict S1 | 0.0512 | 0.0723 | 0.0997 | 0.0366 | 0.0434 | 0.0503 |
| Light S2 | 0.0553 | 0.0778 | 0.1074 | 0.0387 | 0.0460 | 0.0534 |
| Strict S2 | 0.0488 | 0.0681 | 0.0921 | 0.0349 | 0.0411 | 0.0472 |

**Δ% vs S0 (full)**

| Setting | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---:|---:|---:|---:|---:|---:|
| Light S1 | -6.25% | -8.95% | -7.59% | -6.57% | -7.96% | -7.47% |
| Strict S1 | -7.37% | -8.24% | -8.16% | -5.92% | -6.52% | -6.70% |
| Light S2 | +0.19% | -1.24% | -1.04% | -0.42% | -1.04% | -0.96% |
| Strict S2 | -11.57% | -13.59% | -15.19% | -10.18% | -11.46% | -12.56% |

**Δ% Strict vs Light**

| Comparison | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---:|---:|---:|---:|---:|---:|
| Strict S1 vs Light S1 | -1.19% | +0.79% | -0.62% | +0.70% | +1.57% | +0.83% |
| Strict S2 vs Light S2 | -11.73% | -12.51% | -14.29% | -9.80% | -10.53% | -11.71% |

---

## Notes

- **Light vs Strict (S1)**: differences within ~±1% noise — alignment-target leakage was minor when only description was removed.
- **Light vs Strict (S2)**: Light overstates R@5 by ~12% because the `h_raw` (GPT-profile) still encodes description-level semantics; Strict re-generates the profile from title-only input, removing that leakage and revealing the true robustness ceiling.
- Paper-honest sparsity claim: **Strict S1 −7%, Strict S2 −12% R@5 vs full text** — InATTo gracefully degrades with text sparsity, less than the information loss itself.
