# NULL_BASELINE_DOUBLE_COLD.md — first-class null baselines

Split: **double_cold** (seed 42). BRENDA-200 canonical pool (first 200 proteins x first 200 unique
SMILES), held-out test split, identical axes to every model run.

Test pairs in pool: **6** (pos-rate 1.000, only 0 negatives);
full test split: **213** pairs (pos-rate 0.347). Unique positive pairs matched for ranking:
6.
NOTE: the 200x200 pool subset contains 6 test pairs, ALL positive (0 negatives) — pooled AUC is undefined (nan) on the pool surface for every scorer; the full-split column is the informative surface under this split.

| null | AUC pool | AUC full split | MRR raw | MRR tie-aware | H@5 tie | H@10 tie | Gini | Jac-top10 vs prot_prior |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| null_random | nan | 0.535 | 0.072 | 0.072 | 0.167 | 0.167 | 0.509 | 0.03 |
| null_prot_prior | nan | 0.500 | 0.014 | 0.009 | 0.000 | 0.000 | 0.995 | 1.00 |
| null_lig_prior | nan | 0.500 | 1.000 | 0.012 | 0.000 | 0.000 | 0.995 | 0.00 |

## Reading

Split structure: 0.0% of test PROTEINS and 0.0% of test LIGANDS (canonical identity) also occur in train.

**Chance reference is the random row itself**, not the analytic
single-positive constant: empirical random performance here is
MRR 0.072, H@10 0.167, pooled AUC nan/0.535.

**Double-cold (skill §5/§7).** Neither axis recurs across the
split, so BOTH priors collapse to the global-rate fallback:
prot_prior pooled AUC 0.500,
lig_prior 0.500. Any pooled
AUC above chance under this split must come from genuine
ligand-protein generalisation.

**Tie artefacts matter for degenerate scorers.** lig_prior is
constant along each row; its raw matrix MRR (1.000)
is a strict-greater-counting artefact (every column 'rank 0',
H@K = 1.0). Tie-aware MRR is 0.012: a per-ligand
constant carries zero within-row ranking information.

prot_prior matrix structure: tie-aware MRR 0.009; its Gini 0.995
matches every trained Phase-1 model: Gini reflects data geometry,
not learned pathology (Phase-1 pivot).

**Headline answer:** the informative nulls differ BY SPLIT —
report them next to every model number (skill §8). A null's lack
of signal on a given split is itself a result.
