# PAIRED_MOLECULE_STATS.md — skill items A6 + A7

Statistical unit: MOLECULE (ligand row of the 200x200 matrix).
Bootstrap: 5000 replicates over molecules, seed 12345.
Wilcoxon signed-rank where n>=5 and non-zero differences exist.
Per-seed uncertainty uses the honest true-split re-evaluation
(`~/rankbind_revision/honest_reeval_matrix_metrics.csv`).

## A6 paired per-molecule comparisons

| comparison | n | mean A | mean B | dMRR | 95% CI | Wilcoxon p | rank-biserial |
|---|---:|---:|---:|---:|---|---|---:|
| RankBind default s42(model) vs BCE control s42(model) | 30 | 0.2474 | 0.0126 | +0.2348 | [+0.1592, +0.3161] | 1.30e-08 | 0.983 |
| RankBind default s7(model) vs BCE control s7(model) | 30 | 0.2178 | 0.0171 | +0.2008 | [+0.1114, +0.3122] | 4.73e-06 | 0.957 |
| RankBind default s1337(model) vs BCE control s1337(model) | 30 | 0.1816 | 0.0138 | +0.1678 | [+0.1127, +0.2258] | 2.55e-07 | 0.935 |
| RankBind default s42(model) vs RankBind default s42(prior) | 30 | 0.2474 | 0.0208 | +0.2266 | [+0.1512, +0.3085] | 5.51e-06 | 0.948 |
| RankBind default s7(model) vs RankBind default s7(prior) | 30 | 0.2178 | 0.0208 | +0.1970 | [+0.1074, +0.3072] | 7.61e-06 | 0.935 |
| RankBind default s1337(model) vs RankBind default s1337(prior) | 30 | 0.1816 | 0.0208 | +0.1607 | [+0.1052, +0.2185] | 1.94e-05 | 0.892 |

## A7 per-seed uncertainty (matrix metrics, honest true-split eval)

| family | seeds | MRR per-seed | H@10 per-seed | MRR mean±SD | H@10 mean±SD |
|---|---|---|---|---|---|
| brenda200_attn_pool | 42;7;1337 | 0.3162;0.3740;0.2423 | 0.7059;0.7073;0.5283 | 0.311 ± 0.066 | 0.647 ± 0.103 |
| brenda200_bce_only | 7;1337 | 0.0214;0.0284 | 0.0244;0.0189 | 0.025 ± 0.005 | 0.022 ± 0.004 |
| brenda200_default | 7;1337;42 | 0.1675;0.1884;0.2465 | 0.4634;0.5283;0.6471 | 0.201 ± 0.041 | 0.546 ± 0.093 |
| brenda200_no_bilinear | 7;1337;42 | 0.2342;0.0507;0.1728 | 0.5854;0.1321;0.4706 | 0.153 ± 0.093 | 0.396 ± 0.236 |
| brenda200_no_margin | 7;1337 | 0.0473;0.0251 | 0.0732;0.0377 | 0.036 ± 0.016 | 0.055 ± 0.025 |
| brenda200_no_sampler | 7;1337;42 | 0.0412;0.1354;0.1676 | 0.0732;0.3585;0.5294 | 0.115 ± 0.066 | 0.320 ± 0.231 |
| kcat_km_hp1400 | 42;7;1337 | 0.2280;0.3307;0.3213 | 0.4079;0.6061;0.7429 | 0.293 ± 0.057 | 0.586 ± 0.168 |
| km_hp3400 | 42;7;1337 | 0.2190;0.4077;0.2607 | 0.3898;0.8158;0.6923 | 0.296 ± 0.099 | 0.633 ± 0.219 |
| turnover_hp2000 | 42;1337;7 | 0.3442;0.2748;0.3294 | 0.7143;0.6061;0.7532 | 0.316 ± 0.037 | 0.691 ± 0.076 |

## Notes

- Runs are seed-PAIRED canonical pinned-split runs (RankBind default
  vs BCE control, same training seed per row; seed-42 anchors are the
  clean April runs, s7/s1337 the August v5 sweep). The candidate pool
  and axes are identical, so molecule-level pairing is well defined;
  each side's positives come from its own true split.
- All runs are evaluated on the PINNED canonical split
  (data.split_seed=42), not on a split drawn from the training seed.
- Prior baselines are deterministic given the split; their molecule-
  level MRR is the chance-adjusted reference (expected MRR of random
  ranking with m_pos positives among 200 candidates is ~H_200/m/…;
  empirically reported above instead of analytically).
- No pair-level averaging anywhere: every test aggregates molecules.

**Verdict:** see table — RankBind's molecule-level advantage over the
BCE control and over the prior baseline holds in every seed-paired
comparison and survives molecule-level bootstrapping.
