# DIAGNOSTIC_TABLE.md — information source vs task-aligned metrics

Canonical BRENDA-200 benchmark (seed-42 protein-stratified split).
Pooled AUC = held-out TEST pairs, full split surface (same surface
as every model's global AUC); matrix MRR = within-ligand target
ranking over the canonical 200x200 pool (chance ~ H_200/200 ~ 0.029;
tie-aware convention for degenerate scorers). Top-10 overlap vs
prot_prior: 1.0 = identical shortcut geometry.

| Information source | Sees ligand | Sees protein | Pooled AUC | Matrix MRR | Jac@10 vs prot_prior | n | note |
|---|---|---|---:|---:|---:|---:|---|
| Random baseline | no | no | 0.500 | 0.072 | 0.023 |  | uniform noise |
| Protein-only prior (molecule-blind) | no | yes | 0.500 | 0.013 | 1.000 |  | per-protein train positive rate |
| Ligand-only prior (train-rate) | yes | no | 0.915 | 0.019 | 0.000 |  | per-ligand train rate; constant along each row |
| Ligand-only probe (frozen ChemBERTa, linear) | yes | no | 0.887 |  |  |  | cannot rank proteins within a ligand by construction (ligand-only score) |
| Ligand+protein probe (both blocks, linear) | yes | yes | 0.833 | 0.029 | 0.053 |  | chance MRR = 1/200 ~ 0.029 |
| BCE control (pairwise objective) | yes | yes | 0.918 | 0.014 | 0.674 | 3 | +/-SD pooled 0.003, MRR 0.0020 |
| RankBind (within-ligand margin + hard negs) | yes | yes | 0.620 | 0.182 | 0.027 | 10 | +/-SD pooled 0.025, MRR 0.0751 |

## Reading

- **The molecule axis alone carries the pooled signal**: the
ligand-only train-rate prior reaches pooled AUC 0.915 while carrying ZERO
within-ligand ranking information (tie-aware MRR
0.019); the ligand-only frozen-encoder probe
confirms it (0.887).
- **The protein axis is closed on this split** by construction:
protein-disjointness caps the molecule-blind prior at pooled AUC
0.500.
- **Pairwise BCE training reproduces the shortcut, not the task:**
pooled AUC 0.918 at chance-level matrix MRR
0.014 and prot-prior-like top-10 overlap
0.67.
- **Training toward the ranking property moves the task metric,
not the pooled one:** RankBind trades pooled AUC down to
0.620 while lifting matrix MRR to
0.182 (~13x BCE) and dropping the prior overlap to
0.03.

Sources (regenerable, do not hand-edit numbers):
`null_baseline_firstclass.csv`, `decoy_leakage_probe.csv`,
`attractor_results/phase2_rankbind_multiseed.csv`. Script:
`evaluation/diagnostic_table.py`.
