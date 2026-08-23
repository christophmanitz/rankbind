# NULL_BASELINE.md — first-class null baselines

Split: **protein** (seed 42). BRENDA-200 canonical pool (first 200 proteins x first 200 unique
SMILES), held-out test split, identical axes to every model run.

Test pairs in pool: **36** (pos-rate 0.944, only 2 negatives);
full test split: **1404** pairs (pos-rate 0.330). Unique positive pairs matched for ranking:
34.

| null | AUC pool | AUC full split | MRR raw | MRR tie-aware | H@5 tie | H@10 tie | Gini | Jac-top10 vs prot_prior |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| null_random | 0.529 | 0.500 | 0.072 | 0.072 | 0.088 | 0.147 | 0.509 | 0.02 |
| null_prot_prior | 0.500 | 0.500 | 0.021 | 0.013 | 0.000 | 0.000 | 0.995 | 1.00 |
| null_lig_prior | 0.632 | 0.915 | 1.000 | 0.019 | 0.029 | 0.029 | 0.995 | 0.00 |

## Reading

Split structure: 0.0% of test PROTEINS and 53.6% of test LIGANDS (canonical identity) also occur in train.

**Chance reference is the random row itself**, not the analytic
single-positive constant: empirical random performance here is
MRR 0.072, H@10 0.147, pooled AUC 0.529/0.500.

**Protein prior cannot reproduce pooled performance on the
protein-disjoint split.** null_prot_prior reaches pooled AUC
**0.500 exactly — by CONSTRUCTION**: no test protein has
training rows, so every test pair receives the same
global-rate fallback (verified on both surfaces). Any model
beating this must generalise beyond train prevalence.

**Molecule-side prior transfer.** On the FULL test split,
null_lig_prior reaches pooled AUC
**0.915**: per-ligand train
rates transfer across the protein split because molecules are
shared and ~99% of ligands have a fixed role under the decoy
protocol (DECOY_LEAKAGE_AUDIT.md). Molecular memory alone
reproduces a large share of the trained models' global-AUC
range without any interaction learning.

**Tie artefacts matter for degenerate scorers.** lig_prior is
constant along each row; its raw matrix MRR (1.000)
is a strict-greater-counting artefact (every column 'rank 0',
H@K = 1.0). Tie-aware MRR is 0.019: a per-ligand
constant carries zero within-row ranking information.

prot_prior matrix structure: tie-aware MRR 0.013; its Gini 0.995
matches every trained Phase-1 model: Gini reflects data geometry,
not learned pathology (Phase-1 pivot).

**Headline answer:** the informative nulls differ BY SPLIT —
report them next to every model number (skill §8). A null's lack
of signal on a given split is itself a result.
