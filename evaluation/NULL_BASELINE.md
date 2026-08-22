# NULL_BASELINE.md — skill item A10: first-class null baselines

BRENDA-200 canonical pool (first 200 proteins x first 200 unique
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

**Chance reference is the random row itself**, not the analytic
single-positive constant: with ~1-2 positives per row, empirical
random performance is MRR 0.072, H@10 
0.147, pooled AUC 0.529/0.500.

**Protein prior cannot reproduce pooled performance here.**
null_prot_prior reaches pooled AUC **0.500 exactly — by
CONSTRUCTION**: the split is protein-disjoint, no test protein has
training rows, so every test pair receives the same global-rate
fallback (verified on both the pool-restricted and full-split
surface). Any model beating this must generalise beyond train
prevalence; the models' global AUC range 0.63-0.95 does.

**Molecule-side prior transfer (quantifying the A18 finding).** On
the FULL test split, null_lig_prior reaches pooled AUC
**0.915**: per-ligand train rates
transfer across the protein split because molecules are shared and
~99% of ligands have a fixed role under the decoy protocol
(DECOY_LEAKAGE_AUDIT.md). Molecular memory alone reproduces a
large share of the trained models' global-AUC range without any
interaction learning. Context: Phase-1 baselines score global AUC
DrugBAN 0.954 / MolTrans 0.937 / GraphDTA 0.869 / GEMS 0.633, and
RankBind v4 0.634 +/- 0.010 - i.e. two of four deep baselines sit
at or below the molecule-prior line, and RankBind deliberately
trades global AUC away. (On the 36-pair pool subset the estimate
is noisy: only 2 negatives -> 0.632.)

**Tie artefacts matter for degenerate scorers.** lig_prior is
constant along each row; its raw matrix MRR (1.000)
is a strict-greater-counting artefact (every column 'rank 0',
H@K = 1.0). Tie-aware MRR is 0.019: a per-ligand
constant carries zero within-row ranking information. Raw numbers
are kept only for comparability with earlier tables that share the
convention.

prot_prior matrix structure: tie-aware MRR 0.013, below its own random reference - its
informative columns are the train-seen proteins while all test
positives sit on fallback-scored unseen columns. Its Gini 0.995
matches every trained Phase-1 model: Gini reflects data geometry,
not learned pathology (Phase-1 pivot).

Reference (models, same pool/test positives): RankBind v4 default
matrix MRR 0.247 (seed 42) / 0.326 +/- 0.072 (3 seeds); BCE control
gAUC 0.92 at matrix MRR ~0.01. See phase2_rankbind_multiseed.csv.

**Headline answer:** on BRENDA-200, protein-prevalence information
alone reproduces NONE of the models' pooled-AUC advantage (hard cap
at 0.500 on this split), while molecule-side role memorisation
reproduces much of it - the shortcut critique applies to both
axes, with the molecule axis dominant under the current decoy
protocol. Ligand-conditional matrix metrics are the appropriate
primary instruments, as adopted.
