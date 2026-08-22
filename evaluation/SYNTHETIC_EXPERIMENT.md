# SYNTHETIC_EXPERIMENT.md — skill item A17

50 simulations per regime, seed base 424242.
World: 200x200 matrices; popularity p_j ~ Beta(0.3, 0.3)
rescaled to [0.05, 0.95]; observation split 30% train /
70% held out. Regime A: P(pos)=p_j.
Regime B: P(pos)=sigmoid(1.0*logit(p_j) + 6.0*L_ij -5), L = per-ligand set
of 8 random preferred proteins (independent of p_j).

Analytic random-ranking expectation: E[MRR] = H_200/200
= **0.0294**.

| regime | scorer | pooled AUC | matrix MRR | Hit@10 | rho(col-mean, p) |
|---|---|---|---|---|---:|
| A: prevalence_only | prior | 0.897 ± 0.007 | 0.049 ± 0.002 | 0.096 ± 0.005 | +0.97 |
| A: prevalence_only | lig_oracle | 0.500 ± 0.001 | 0.029 ± 0.001 | 0.050 ± 0.001 | -0.01 |
| A: prevalence_only | combined | 0.866 ± 0.006 | 0.043 ± 0.001 | 0.059 ± 0.002 | +0.97 |
| B: prevalence_plus_ligand | prior | 0.731 ± 0.012 | 0.062 ± 0.004 | 0.130 ± 0.012 | +0.85 |
| B: prevalence_plus_ligand | lig_oracle | 0.724 ± 0.010 | 0.170 ± 0.010 | 0.482 ± 0.023 | -0.01 |
| B: prevalence_plus_ligand | combined | 0.890 ± 0.009 | 0.217 ± 0.013 | 0.496 ± 0.024 | +0.83 |

## Reading

**Dissociation 1 — across regimes at matched pooled AUC.** In the
prevalence-only world the molecule-blind prior reaches pooled AUC
**0.897** with matrix MRR **0.049** (random:
0.029). In the ligand-signal world the combined scorer
reaches pooled AUC **0.890** — statistically the same level
— with matrix MRR **0.217**, a **4.4x** difference. Two worlds
that pooled AUC cannot tell apart differ by multiples in true
ligand-conditional ranking.

**Dissociation 2 — within regime B.** The molecule-blind prior
(pooled AUC 0.731, MRR 0.062) and the pure
ligand-preference oracle (pooled AUC 0.724, MRR
0.170) are nearly indistinguishable by pooled AUC yet
differ 2.8x in matrix MRR.
Pooled AUC does not identify WHICH structure drives scores;
within-row ranking does.

**Mechanism trace.** rho(col-mean score, p_j) is high for every
scorer that contains the prevalence component (+0.8-0.9) and zero
for the ligand-only oracle, matching the biological finding that
protein-prior structure is what pooled metrics reward.

Both dissociations are dataset-independent consequences of the
pooling arithmetic, matching the biological findings in
METRIC_AUDIT.md, PAIRED_MOLECULE_STATS.md and
POSITIVE_DENSITY_ANALYSIS.md.
