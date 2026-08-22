# DECOY_LEAKAGE_AUDIT.md — skill item A18

Frozen-feature linear probe on the BRENDA decoy dataset (canonical seed-42 protein split):

- ChemBERTa mean-pool [384] + ESM2 mean-pool [1280], both frozen
- StandardScaler + LogisticRegression (C=1, max_iter=2000)
- train rows 6593 / test rows 1404 (test pos-rate 0.330)

| variant | train AUC | test pooled AUC | matrix MRR | H@5 | H@10 | n matched |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.974 | 0.833 | 0.029 | 0.000 | 0.118 | 34 |
| molecule_only | 0.933 | 0.887 | — | — | — | — |
| protein_only | 0.801 | 0.603 | — | — | — | — |

## Reading

**1. The skill's anticipated finding.** The frozen linear probe
(full) reaches pooled test AUC **0.833** with no
deep learning at all — trained deep DTI baselines on this dataset
span global AUC 0.63–0.95 (DrugBAN 0.954, MolTrans 0.937, GraphDTA
0.869, GEMS 0.633; RankBind v4 0.634 ± 0.010). A representation-
free linear readout of frozen features recovers a large share of
the published pooled-AUC level.

**2. The stronger finding: molecule-side role assignment.** The
molecule-only probe (**0.887**) BEATS the full
probe. Mechanism: of the unique ligands, **1417 are positive-only** and **3157 are
decoy-only** (99.3% pure overall) — the decoy protocol
assigns each molecule a fixed role rather than sampling negatives
per protein. Because the canonical split is protein-based, it does
NOT hold out molecules (53.6% of test-pair ligands also
occur in training rows), so ligand identity alone transfers to the
test split and any model with molecular memory scores high without
learning protein-ligand interaction.

**3. Protein side is consistent with the null-prior analysis.** The
protein-only probe reaches 0.604 and its score
correlates at rho=+0.96
with the train-rate prior — i.e. at null-baseline level. The full
probe's shortcut diagnostics (rho +0.10,
top-10 Jaccard vs prior 0.05) show
its pair-level signal is NOT primarily the protein-prevalence
shortcut analysed elsewhere; it is the molecule-side structure.

**4. A fourth dissociation for free.** The full probe scores
pooled AUC 0.833 while its matrix MRR is 0.029 (chance = H_200/200 = 0.0294; Hit@5 0.00). High pooled AUC with chance-level
ligand-conditional ranking — the paper's central metric critique,
reproduced by a linear probe in isolation.

**Interpretation (skill-mandated framing).** The decoy construction
itself contains learnable pair-level structure that can be
exploited by representation-based models — dominated by molecule-
role memorisation, not protein prevalence — motivating cautious
interpretation of ABSOLUTE pooled-AUC values on BRENDA-with-decoys,
and supporting the paper's choice of ligand-conditional matrix
metrics as primary. This is a limitation, not evidence that BRENDA
is invalid.
