# POSITIVE_DENSITY_ANALYSIS.md — skill item A16

Mechanism check for: *stronger protein-prevalence imbalance -> more
prior-explainable pooled AUC*. Everything below uses the canonical
seed-42 protein splits.

## P1 Prevalence imbalance per dataset (descriptive)

| dataset | n train prot | rate median | rate CV | rate Gini | test pairs | test pos rate |
|---|---:|---:|---:|---:|---:|---:|
| BRENDA-200 | 618 | 0.39 | 0.66 | 0.36 | 1404 | 0.33 |
| km_with_decoys | 6650 | 1.00 | 0.39 | 0.19 | 8638 | 0.50 |
| kcat_km_with_decoys | 2671 | 0.33 | 0.64 | 0.33 | 6797 | 0.29 |
| turnover_with_decoys | 3972 | 1.00 | 0.51 | 0.27 | 6440 | 0.41 |
| davis | 267 | 0.04 | 1.02 | 0.47 | 3808 | 0.06 |
| kiba | 161 | 0.21 | 0.66 | 0.36 | 19432 | 0.17 |
| bindingdb_kd | 991 | 0.08 | 1.26 | 0.65 | 6312 | 0.16 |
| esp | 8006 | 0.50 | 0.62 | 0.31 | 4320 | 0.50 |

## P2 Why pair-level prior AUC on unseen proteins is 0.500 by
### construction

A molecule-blind prior scores a pair by its protein's TRAIN positive
rate. Under a protein-disjoint split no test protein has training
rows, so every test pair receives the same global-rate fallback:
the prior's pooled AUC on strictly unseen proteins is **exactly 0.500**
(verified empirically below). Any model exceeding that on this split
must generalise beyond the train-rate statistic — the shortcut
analysed in this paper operates through mixed candidate pools, not
through raw pair scoring.

## P3/P4 BRENDA-200: pool composition, prior AUC by pair origin,
### and the BCE-vs-RankBind gradient

**RankBind default_v4 s42** — pool composition: 148 train / 33 val / 19 test of 200 candidates.
- observed train-split pairs inside the pool: n=312, positive rate 0.958, prior pooled AUC **0.657**
- observed test-split pairs inside the pool: n=36, positive rate 0.944, prior pooled AUC **0.500**
- Spearman(train rate, mean column score) over the 148 seen proteins: **-0.166**
**BCE control (abl_bce_only s7)** — pool composition: 151 train / 25 val / 24 test of 200 candidates.
- observed train-split pairs inside the pool: n=302, positive rate 0.967, prior pooled AUC **0.658**
- observed test-split pairs inside the pool: n=43, positive rate 0.953, prior pooled AUC **0.500**
- Spearman(train rate, mean column score) over the 151 seen proteins: **+0.206**
- Figure: `fig_positive_density.png` — z-scored mean column
  scores against training prevalence deciles. The BCE control
  climbs with prevalence (rho +0.21); RankBind
  after margin-based ranking optimisation is decorrelated or
  inverted (rho -0.17) while improving
  ligand-conditional ranking (see PAIRED_MOLECULE_STATS.md).

**Interpretation.** The mechanism is confirmed where it is defined:
the protein-prior reproduces elevated pooled AUC only on pairs whose
proteins were seen in training (P3), models inherit exactly that
structure under BCE (positive rate-score correlation), and replacing
the objective with within-ligand margins removes the inheritance
while improving true ligand-conditional ranking (P4). Absolute
prevalence imbalance varies by dataset (P1); the paper reports it
as a descriptive covariate, not as an independent success criterion.
