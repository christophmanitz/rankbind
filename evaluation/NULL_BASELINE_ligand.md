# NULL_BASELINE_LIGAND.md — first-class null baselines

Split: **ligand** (seed 42). BRENDA-200 canonical pool (first 200 proteins x first 200 unique
SMILES), held-out test split, identical axes to every model run.

Test pairs in pool: **73** (pos-rate 0.986, only 1 negatives);
full test split: **1495** pairs (pos-rate 0.357). Unique positive pairs matched for ranking:
72.

| null | AUC pool | AUC full split | MRR raw | MRR tie-aware | H@5 tie | H@10 tie | Gini | Jac-top10 vs prot_prior |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| null_random | 0.944 | 0.496 | 0.028 | 0.028 | 0.028 | 0.069 | 0.509 | 0.03 |
| null_prot_prior | 0.701 | 0.655 | 0.120 | 0.023 | 0.014 | 0.042 | 0.995 | 1.00 |
| null_lig_prior | 0.500 | 0.500 | 1.000 | 0.024 | 0.028 | 0.056 | 0.995 | 0.00 |

## Reading

Split structure: 94.5% of test PROTEINS and 0.0% of test LIGANDS (canonical identity) also occur in train.

**Chance reference is the random row itself**, not the analytic
single-positive constant: empirical random performance here is
MRR 0.028, H@10 0.069, pooled AUC 0.944/0.496.

**Cold-ligand mirror image (skill §6).** With ligands
disjoint, null_lig_prior falls back to the global training
rate for EVERY test pair: pooled AUC
0.500 — the molecule-side
shortcut is structurally unavailable. Conversely
null_prot_prior now carries signal:
pooled AUC **0.655**
(94% of test proteins recur, so their train
prevalence transfers). Under cold-ligand evaluation the
dominant residual shortcut is the protein marginal.

**Tie artefacts matter for degenerate scorers.** lig_prior is
constant along each row; its raw matrix MRR (1.000)
is a strict-greater-counting artefact (every column 'rank 0',
H@K = 1.0). Tie-aware MRR is 0.024: a per-ligand
constant carries zero within-row ranking information.

prot_prior matrix structure: tie-aware MRR 0.023; its Gini 0.995
matches every trained Phase-1 model: Gini reflects data geometry,
not learned pathology (Phase-1 pivot).

**Headline answer:** the informative nulls differ BY SPLIT —
report them next to every model number (skill §8). A null's lack
of signal on a given split is itself a result.
