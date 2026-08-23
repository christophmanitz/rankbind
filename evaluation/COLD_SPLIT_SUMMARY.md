# COLD_SPLIT_SUMMARY.md — cold-ligand / double-cold stress tests

Matched controls: same encoders, capacity class, training budget,
hard-negative logic, seeds {42, 7, 1337}, canonical 200x200 pool;
ONLY the split definition changes (skill §7). Pooled AUC = global
AUC over each split's FULL test set. Matrix MRR/H@10 = within-
ligand ranking on canonical-pool test positives — small-n under
double_cold (~6 pairs), read with the full-split columns. Nulls
per split (skill §8); a null's lack of signal is itself a result.

## Split: cold-ligand (ligand-disjoint; proteins recur) (`ligand`)

Split structure pinned per run manifest: test ligands seen in train 0.0%, test proteins seen in train 94.5%; pairs tr/va/te 6523/1413/1495.

| source | pooled AUC | matrix MRR | H@10 | n |
|---|---:|---:|---:|---:|
| null_random (null) | 0.496 | 0.028 | 0.069 | - |
| null_prot_prior (null) | 0.655 | 0.023 | 0.042 | - |
| null_lig_prior (null) | 0.500 | 0.024 | 0.056 | - |
| BCE control (pairwise) `cold_lig_bce` | 0.850 ± 0.018 | 0.028 ± 0.013 | 0.037 ± 0.042 | 3 |
| RankBind (margin + hard negs) `cold_lig_rankbind` | 0.591 ± 0.041 | 0.295 ± 0.064 | 0.565 ± 0.008 | 3 |

## Split: double-cold (neither axis recurs) (`double_cold`)

Split structure pinned per run manifest: test ligands seen in train 0.0%, test proteins seen in train 0.0%; pairs tr/va/te 4575/249/213.

| source | pooled AUC | matrix MRR | H@10 | n |
|---|---:|---:|---:|---:|
| null_random (null) | 0.535 | 0.072 | 0.167 | - |
| null_prot_prior (null) | 0.500 | 0.009 | 0.000 | - |
| null_lig_prior (null) | 0.500 | 0.012 | 0.000 | - |
| BCE control (pairwise) `cold_both_bce` | 0.813 ± 0.010 | 0.025 ± 0.006 | 0.000 ± 0.000 | 5 |
| RankBind (margin + hard negs) `cold_both_rankbind` | 0.573 ± 0.051 | 0.092 ± 0.078 | 0.233 ± 0.253 | 5 |

## Reference: canonical protein-stratified split

| source | pooled AUC | matrix MRR | H@10 | n |
|---|---:|---:|---:|---:|
| null_random (null) | 0.500 | 0.072 | 0.147 | - |
| null_prot_prior (null) | 0.500 | 0.013 | 0.000 | - |
| null_lig_prior (null) | 0.915 | 0.019 | 0.029 | - |
| BCE control (pairwise) | 0.918 ± 0.003 | 0.014 ± 0.002 | 0.000 ± 0.000 | 3 |
| RankBind (margin + hard negs) | 0.620 ± 0.025 | 0.182 ± 0.075 | 0.447 ± 0.206 | 10 |

Per-run provenance: `attractor_results/cold_split_runs.csv`; aggregates: `attractor_results/cold_split_multiseed.csv`. Regenerate: `python scripts/aggregate_cold_splits.py`.
