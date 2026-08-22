# POOL_SENSITIVITY.md — skill item A9

EVALUATION-ONLY sensitivity (no retraining): stored score matrices
evaluated on 5 seeded random candidate-protein subsets
per pool size (subset seed 777). Positives come from each run's TRUE
split, restricted to the subset. Analytic random expectation:
E[MRR] = H_n / n.

| model | pool | MRR mean±SD | Hit@1 | Hit@5 | Hit@10 | random E[MRR] | prior MRR |
|---|---:|---|---|---|---|---:|---|
| RankBind default_v4 s42 | 50 | 0.401 ± 0.135 | 0.186 | 0.654 | 0.871 | 0.090 | 0.074 |
| RankBind default_v4 s42 | 100 | 0.409 ± 0.093 | 0.179 | 0.669 | 0.787 | 0.052 | 0.040 |
| RankBind default_v4 s42 | 200 | 0.247 ± 0.000 | 0.029 | 0.500 | 0.647 | 0.029 | 0.021 |
| RankBind attn_pool_v5b s42 | 50 | 0.653 ± 0.242 | 0.519 | 0.827 | 0.938 | 0.090 | 0.086 |
| RankBind attn_pool_v5b s42 | 100 | 0.453 ± 0.064 | 0.260 | 0.682 | 0.828 | 0.052 | 0.044 |
| RankBind attn_pool_v5b s42 | 200 | 0.316 ± 0.000 | 0.147 | 0.559 | 0.706 | 0.029 | 0.021 |
| BCE control s7 | 50 | 0.082 ± 0.040 | 0.000 | 0.073 | 0.287 | 0.090 | 0.060 |
| BCE control s7 | 100 | 0.042 ± 0.007 | 0.000 | 0.031 | 0.102 | 0.052 | 0.029 |
| BCE control s7 | 200 | 0.021 ± 0.000 | 0.000 | 0.024 | 0.024 | 0.029 | 0.015 |

**Reading:** RankBind's MRR advantage over both the random expectation
and the protein prior persists at every pool size tested; nothing here
depends on the specific 200-candidate construction. Pool sizes beyond
200 are impossible without changing the stored matrices' axes and are
therefore out of scope for this evaluation-only analysis.
