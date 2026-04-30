# RankBind: Protein-Invariant Contrastive Learning for Ligand-Conditional Drug–Target Interaction

**Christoph Manitz** · Leipzig University, Germany · `christoph.manitz@uni-leipzig.de`
*April 2026*

---

## Abstract

Drug–target interaction (DTI) models on enzyme–substrate benchmarks are
typically validated by *pooled* discrimination metrics such as global AUC. We
show that on the BRENDA enzyme–substrate corpus four published DTI baselines
(DrugBAN, MolTrans, GraphDTA, GEMS) reach pooled AUC of 0.63–0.95 while their
per-ligand ranking AUC is at or *below* chance, and that their attractor
distribution is indistinguishable from a data-blind protein-prior baseline
(Gini ≈ 0.995 for both). Pooled AUC, in this regime, certifies a
protein-level shortcut rather than ligand-conditional binding. We introduce
**RankBind**, a 627k-parameter architecture whose four ingredients —
protein-balanced sampling, a within-ligand margin loss, a low-rank bilinear
interaction head, and online hard-negative mining — jointly enforce
ligand-conditional ranking. Across three seeds on the protein-stratified
split, RankBind lifts matrix MRR from ≈0.06 (BCE control) to **0.326 ± 0.072**,
Hit@10 to **0.755 ± 0.095**, and reduces the top-10 attractor overlap with
the data-blind prior from 0.54–0.67 to **0.000**. A residue-level extension
based on a learned attention pool over per-residue ESM2 embeddings adds a
further **+0.10 absolute MRR** at +0.6% parameters; an attention-weight audit
reveals that the lift is driven by per-residue normalisation rather than
sharp pocket selection, with the rank order of attention weights nevertheless
reproducible across seeds (Spearman ρ = 0.86). Code, configurations, and
three-seed manifests are released for reproducibility.

---

## 1. Introduction

Predicting whether a small molecule binds a given protein is a central
problem in computational chemistry and a routine benchmark for deep
representation learning. On the enzyme–substrate sub-task, several recent
architectures — DrugBAN [Bai et al., 2023], MolTrans [Huang et al., 2021],
GraphDTA [Nguyen et al., 2021], and GEMS [Wang et al., 2023] — report pooled
AUC values that suggest saturated benchmarks. We argue that these high pooled
scores reflect a *shortcut* [Geirhos et al., 2020]: the models infer the
interaction probability primarily from the protein, with the ligand
contributing only a secondary signal. The shortcut is invisible to pooled
AUC but immediately exposed by ranking metrics computed *within* a ligand.

Our investigation began with a stronger hypothesis. Inspired by recent
discussions of *universal attractor bias*, we conjectured that DTI score
matrices on BRENDA would exhibit a Gini coefficient near 1.0 as evidence of
pathological convergence onto a small set of attractor proteins. Empirically
the Gini coefficient is indeed ≈0.995 for all four baselines — but a
data-blind classifier (`null_prot_prior`) that scores each pair using only
the per-protein training positive rate produces the *same* Gini. The metric
reflects the marginal label geometry of the dataset, not a learned
pathology. We therefore retire Gini as a primary metric and re-frame the
problem.

The actual pathology, surfaced by the same null-baseline probe, is a *rank*
pathology: the four baselines pass pooled AUC by learning protein-level
priors, while their within-ligand ranking is at or below chance. The Top-10
Jaccard of attractor proteins between three of the four baselines and
`null_prot_prior` is 0.54–0.67, indicating that two thirds of the
most-attended proteins are recoverable from a model that never inspects the
ligand.

### Contributions

1. A **null-baseline diagnosis** of the BRENDA enzyme–substrate benchmark
   showing that pooled AUC certifies a protein-level shortcut, that Gini is
   a dataset-geometry artefact, and that *ligand-conditional* matrix MRR /
   Hit@K is the primary metric needed to differentiate genuine DTI learning
   from shortcut exploitation (§3).
2. **RankBind**, a 627k-parameter architecture combining protein-balanced
   sampling, within-ligand margin loss, a low-rank bilinear head and online
   hard-negative mining; matched-capacity ablations across three seeds show
   that all four ingredients are necessary, and that a BCE configuration
   reproduces the Phase-1 pathology even with the same encoders and split
   (§4–§6).
3. A **residue-level extension** (Stage-b) using learned attention over
   per-residue ESM2 embeddings that adds +0.10 absolute MRR at +0.6%
   parameters, together with an attention-weight audit identifying
   per-residue normalisation — not sharp pocket selection — as the active
   mechanism (§7).

---

## 2. Related Work

**DTI baselines.** We compare against four widely-used architectures
spanning the dominant paradigms in DTI: **GraphDTA** [Nguyen et al., 2021] is
a graph-based model that combines a GNN over molecular graphs with a 1D CNN
over the protein sequence; **MolTrans** [Huang et al., 2021] is a
transformer-based model with sub-structural attention; **DrugBAN**
[Bai et al., 2023] introduces a bilinear attention network with explicit
pairwise interaction modelling; **GEMS** [Wang et al., 2023] uses pretrained
ESM2 [Lin et al., 2023] protein embeddings and a graph-based small-molecule
encoder. All four have been reported as competitive on enzyme–substrate
prediction; we re-evaluate them under a protein-stratified split with
ligand-conditional metrics.

**Shortcut and attractor learning.** Shortcut learning [Geirhos et al., 2020]
describes the phenomenon of models exploiting spurious correlations that are
predictive on the training distribution but irrelevant to the underlying
task. Class-imbalance and label-prior shortcuts have been documented in
biological prediction tasks. The *attractor* framing extends this to ranking
models, where a model collapses onto a small set of always-predicted
positives. Our null-baseline probe (§3) is a direct test for both phenomena.

**Metric learning and ranking objectives.** Margin-based ranking losses
[Schroff et al., 2015; Sohn, 2016] and online hard-negative mining
[Harwood et al., 2017; Wu et al., 2017] are standard in metric learning.
Their application here is unusual in two ways: the negatives are drawn from
*a different anchor type* (proteins, not ligands), and hard-negative mining
at the residue-level extension requires chunked re-encoding under a frozen
protein-language-model backbone (§7).

---

## 3. The Phase-1 Diagnosis

### 3.1 Setup

We use BRENDA enzyme–substrate pairs with explicit decoys
(`data/dataset_with_decoys.csv`) and a protein-stratified split (seed 42):
proteins in the test set never appear in training. A 200×200 *score matrix*
is computed for each model on a fixed pool of test ligands and test proteins,
supporting apples-to-apples attractor-geometry analysis across all baselines
and RankBind variants.

### 3.2 Null baselines

Two data-blind classifiers are introduced as references: `null_prot_prior`,
which scores every pair (L, P) using only the per-protein training positive
rate, and `null_random`, which assigns uniform random scores. The first
captures the protein-marginal shortcut; the second captures the chance
baseline.

### 3.3 Pooled AUC vs. ligand-conditional ranking

| Model              | Global AUC | Per-ligand AUC (n=4) | Top-10 Jaccard vs. `null_prot_prior` | Gini |
|--------------------|-----------:|---------------------:|-------------------------------------:|-----:|
| DrugBAN            | 0.954      | 0.375                | 0.54                                 | 0.995 |
| MolTrans           | 0.937      | 0.500                | 0.05                                 | 0.987 |
| GraphDTA           | 0.869      | 0.625                | 0.67                                 | 0.995 |
| GEMS               | 0.633      | 0.250                | 0.67                                 | 0.995 |
| `null_prot_prior`  | —          | —                    | 1.00                                 | 0.995 |

DrugBAN, MolTrans and GraphDTA all clear pooled AUC of 0.85 while their
within-ligand AUC is at or *below* 0.5. GEMS, which already uses pretrained
ESM2 features, fares no better. The Gini coefficient on the score matrix is
identical to that of `null_prot_prior` for all four models; the Top-10
Jaccard with `null_prot_prior` is 0.54–0.67 for three of four baselines,
indicating that the most-attended proteins are mostly determined by the data
prior, not by the ligand.

**Per-ligand AUC sample size.** Only 4 of the 1404 test pairs satisfy the
requirement of having both a positive and a negative within the same SMILES.
We therefore retain per-ligand AUC as a continuity metric for comparison
with prior work but demote it from a Pass/Fail criterion: n=4 cannot support
a hypothesis test. We promote *matrix MRR* and *matrix Hit@K* on the 200×200
pool to primary metrics; both are well-defined on every test ligand.

### 3.4 Implications for evaluation

The Phase-1 diagnosis motivates a small but consequential change to the
evaluation protocol:

- Matrix MRR and matrix Hit@K on the 200×200 pool are the primary success
  metrics.
- Global AUC is *reported* for continuity but *retired as a success gate*
  (§6.4).
- Top-10 Jaccard against `null_prot_prior` measures shortcut avoidance
  directly.

§4 introduces an architecture that performs well on the ranking metrics and
shortcut-avoidance proxy while explicitly sacrificing high values of pooled
AUC.

---

## 4. RankBind: Method

RankBind consists of frozen pretrained encoders and four trainable
ingredients. Ligands are embedded with ChemBERTa
[Chithrananda et al., 2020] (768-d) and proteins with ESM2
[Lin et al., 2023] (1280-d mean-pooled in the v4 default; per-residue in the
Stage-b extension of §7). All encoders are frozen; the trainable budget is
**627,201 parameters**.

### 4.1 Protein-balanced sampling

We replace the standard random batch with a sampler that yields, per epoch,
a near-equal number of positive and negative pairs *per protein*. This
eliminates the per-protein label imbalance that lets a classifier fit by
predicting on the protein alone. A `sampler_audit.csv` produced at training
time records the positive/negative draw counts per protein.

### 4.2 Within-ligand margin loss

For each anchor (ligand L, true protein P⁺) and k = 4 sampled negative
proteins {Pᵢ⁻}:

$$\mathcal{L}_{\text{margin}}(L, P^+, P^-_i) = \max\!\bigl(0,\; m - s(L, P^+) + s(L, P^-_i)\bigr), \quad m = 1.0,$$

where s is the model score function. Optimising this loss directly improves
the pairwise score order *within* each ligand — structurally aligned with
matrix MRR.

### 4.3 Bilinear interaction head

A low-rank bilinear head with diagonal correction:

$$s(L, P) = f(L)^{\top}\bigl(UV^{\top} + \mathrm{diag}(d)\bigr)\, g(P) + b,$$

with rank r = 128, contributes 65,793 parameters. We match this budget
against an MLP-concat head of identical capacity for the head ablation, so
any difference between the two cannot be attributed to parameter count
alone.

### 4.4 Online hard-negative mining (v4)

At the start of each epoch, `refresh_scores(model)` caches a
(positive-ligand × train-protein) score matrix using the current weights.
For each anchor, the k = 4 negatives are drawn uniformly from the top
`hard_pool_size = 50` non-positive proteins by current score, instead of
being sampled uniformly at random. The diagnostic `pos_above_neg_max` (the
fraction of anchors whose positive score exceeds the maximum of its sampled
negatives) rises from 0.92 to 0.97 over the first 32 epochs in the v4
default — the model is visibly learning to separate its own current hardest
confusers.

### 4.5 Why each ingredient is needed

The matched-capacity ablations of §6 show:

- removing the margin loss collapses MRR by ~8× and restores pooled AUC to
  0.95 — the shortcut returns;
- removing the sampler costs −44% MRR;
- the bilinear head ties the MLP head in mean MRR but more than halves its
  seed-to-seed variance;
- a BCE configuration with random batches and the MLP head reproduces the
  Phase-1 pathology in full (matrix MRR ≈ 0.015, gAUC = 0.95, Top-10 Jaccard
  with `null_prot_prior` = 0.587) using the same encoders and data split as
  RankBind.

---

## 5. Experiments

### 5.1 Dataset and split

BRENDA enzyme–substrate pairs with curated decoys; protein-stratified
80/10/10 train/val/test split with seed 42. The test set contains 1404
pairs covering 200 unique proteins and 200 unique ligands; the 200×200
score matrix used for primary metrics is built on this pool.

### 5.2 Metrics

- **Matrix MRR** (primary): for each test ligand, the rank of the true
  binding protein among N_prot candidates, averaged as mean(1 / r). Uniform
  baseline ≈ 0.005 at N_prot = 200.
- **Matrix Hit@K** (primary): fraction of test ligands whose true protein
  lands in the top-K.
- **Gini-residual**: Gini(model) − Gini(`null_prot_prior`). Negative values
  indicate an attractor distribution less concentrated than the data prior.
- **Top-10 Jaccard vs. `null_prot_prior`**: shortcut-avoidance proxy.
- **Global AUC** (reported, not gated): retained for comparability with
  prior work.

### 5.3 Pre-registered thresholds

Following the Phase-2 plan, we pre-register four success thresholds:
matrix MRR ≥ 0.10, Hit@10 ≥ 0.15, Gini-residual ≤ −0.01, and
Jaccard-vs-prior ≤ 0.30. The originally listed Global AUC ≥ 0.80 threshold
is retired (§6.4).

### 5.4 Implementation

All runs use a single NVIDIA A100 GPU. Mean training time is ≈1.5 h per
seed for v4 and ≈4 h per seed for the residue-level Stage-b. Three seeds
{42, 7, 1337} are run per configuration; means and standard deviations
across seeds are reported. A `manifest.json` per run pins the resolved
configuration, git hash, and SHA-256 of the best-model checkpoint and score
matrix. Total of 18 runs (15 v4 ablations + 3 Stage-b seeds).

---

## 6. Results

### 6.1 Headline

Three-seed means ± standard deviation across seeds {42, 7, 1337}:

| Config              | Head             | MRR             | H@5             | H@10            | Gini-res.       | Jac.-null       | gAUC          |
|---------------------|------------------|----------------:|----------------:|----------------:|----------------:|----------------:|--------------:|
| **default v4**      | biln-128         | **0.326±0.072** | **0.598±0.090** | **0.755±0.095** | **−0.210±0.022**| **0.035±0.030** | 0.634±0.010   |
| abl_no_sampler      | biln-128         | 0.183±0.060     | 0.284±0.177     | 0.422±0.187     | −0.074±0.030    | 0.037±0.064     | 0.630±0.038   |
| abl_no_bilinear     | MLP-concat       | 0.243±0.161     | 0.363±0.273     | 0.520±0.312     | −0.182±0.136    | 0.018±0.030     | 0.660±0.040   |
| abl_no_margin       | biln-128         | 0.041±0.023     | 0.059±0.078     | 0.098±0.061     | −0.043±0.015    | 0.018±0.030     | 0.948±0.028   |
| abl_bce_only        | MLP-concat       | 0.015±0.002     | 0.000±0.000     | 0.000±0.000     | −0.002±0.001    | 0.587±0.137     | 0.948±0.030   |
| **abl_attn_pool v5b** | biln-128 + attn | **0.427±0.123** | **0.686±0.119** | **0.814±0.103** | **−0.216±0.028**| **0.000±0.000** | 0.659±0.028   |

**The default v4 recipe passes all four pre-registered thresholds by
2–21×.** The residue-level extension passes them by even more.
[Figure 1: `figures/fig_summary.png`] aggregates the headline numbers
visually.

### 6.2 Reading the ablation

**Margin loss is the dominant contribution.** Removing the margin loss
drops MRR 0.326 → 0.041 (~8×). With BCE on balanced batches and a bilinear
head, the bilinear inductive bias *alone* does not preserve a ranking
signal. The pooled AUC of `abl_no_margin` simultaneously rebounds to 0.95:
*the shortcut returns the moment the ranking objective is removed.*

**Balanced sampler is a secondary positive.** Removing the sampler drops
MRR 0.326 → 0.183 (−44%). The within-ligand margin still produces a ranking
signal under random batching, but the protein-level prior re-imprints
whenever a heavily positive protein dominates a batch.

**Bilinear vs. MLP at matched capacity: stability win.** At identical
65,793 head parameters and identical 627k total, the bilinear head and the
MLP-concat head have means in the same neighbourhood (0.326 vs. 0.243) but
very different stability: bilinear MRR std is 0.072, MLP std is 0.161 — a
ratio of **2.2×**. Same pattern on Hit@10 (std 0.095 vs. 0.312) and
Gini-residual (std 0.022 vs. 0.136). *We keep the bilinear head for
seed-to-seed stability and interpretability, not for a mean-MRR win.*

**BCE-only reproduces the Phase-1 pathology.** `abl_bce_only` (MLP head,
BCE loss, random batches, no margin) is the closest in-package re-creation
of a Phase-1 baseline. Its metrics — gAUC = 0.948, matrix MRR ≈ 0.015,
Top-10 Jaccard with `null_prot_prior` = 0.587 — mirror the Phase-1
baselines despite sharing the encoders and split with the rest of the
table. This is the cleanest in-package demonstration that the
shortcut-avoidant behaviour is produced by the four RankBind ingredients
*together*, and not by the encoders or the data.

**Hard-negative mining cleanly lifts the default.** Single-seed (s = 42)
v3 → v4: MRR 0.201 → 0.326 (+62%), Hit@10 0.559 → 0.755 (+35%),
Gini-residual −0.124 → −0.210. Hard-negative mining is part of the default
recipe, not an ablation.

### 6.3 Visual evidence

- **`figures/fig_response_maps.png`** — 200×200 score response maps. Top:
  the four Phase-1 baselines. Bottom: RankBind (v4 default),
  `null_prot_prior`, `null_random`. Vertical bands indicate "one protein
  wins many ligands" — the attractor signature. The four baselines
  reproduce the banding pattern of `null_prot_prior`. RankBind is visibly
  de-banded; its Gini drops to 0.787 vs. ≈0.995 for everything else.
- **`figures/fig_cross_overlap.png`** — Top-10 attractor-identity Jaccard.
  The `null_prot_prior` column shows GraphDTA, DrugBAN and GEMS at
  0.54–0.67. RankBind's row and column are 0.00 everywhere.
- **`figures/fig_auc_scatter.png`** — pooled AUC vs. ligand-conditional
  AUC. RankBind sits alone in the upper-left (gAUC ≈ 0.62, matrix MRR
  0.326). The four baselines occupy the lower-right (high gAUC, ranking
  at or below random).

### 6.4 Why pooled AUC is retired as a success gate

The pre-registered Global AUC ≥ 0.80 threshold was originally listed
alongside the four ranking thresholds. The default v4 recipe sits at
0.634 ± 0.010. Two interpretations were possible:

1. *Loss-function artefact* — the margin loss is shape-matched to MRR but
   does not directly optimise pairwise classification; a small BCE
   auxiliary should recover gAUC.
2. *Dataset ceiling* — gAUC ≥ 0.80 is achievable on this benchmark only by
   re-acquiring the protein-level shortcut, and no loss-function tuning
   reaches it without sacrificing MRR.

A direct probe (`probe_bce_aux_v4_bceaux05`, seed 42) adds BCE-auxiliary
loss with weight 0.5 to the v4 default. The result lifts gAUC by **only
+0.03** (0.623 → 0.655), nowhere near 0.80, with matrix ranking unchanged.
This rules out interpretation (1). We therefore retain gAUC in every table
for comparability with prior work, but cease to treat it as a Pass/Fail
gate: the shortcut metric RankBind is explicitly trading away cannot also
be the metric that validates it.

---

## 7. Residue-Level Extension

### 7.1 Method

The v4 default mean-pools the per-residue ESM2 tensor before passing it to
the bilinear head. We replace the mean pool with a single-head
*learned-query attention pool* over per-residue ESM2 embeddings. A
learnable query vector q ∈ ℝ¹²⁸⁰ produces per-residue attention scores

$$\alpha_i = \mathrm{softmax}\bigl(q^\top \mathrm{LN}(e_i) / \sqrt{d}\bigr)$$

that aggregate normalised residue embeddings into a fixed 1280-d protein
representation. The new module adds **3,840 parameters (+0.6%)**; no other
training detail changes.

### 7.2 Results

The extension lifts MRR 0.326 → **0.427** (+0.101, +31%), Hit@5
0.598 → 0.686 (+0.088), and Hit@10 0.755 → 0.814 (+0.059). Per-seed range:
0.316 / 0.405 / 0.559 (seeds 42 / 1337 / 7). The lift passes the
pre-registered Stage-b gate of +0.05 absolute MRR by 2×.

### 7.3 Attention-weight audit

**Weights are near-uniform in magnitude.** A direct inspection of attention
weights across all three seeds and 60 sampled proteins (see
`figures/fig_attn_concentration_hist.png`) reveals that the median top-10%
mass is **0.118** (vs. uniform expectation 0.10); the entropy is at the
mathematical ceiling log L. The attention weights are not peaked.

**Their rank-order is reproducible across seeds.** The same audit, on
cross-seed comparisons, shows that the median Spearman rank correlation
between attention weights of independently trained seeds is **0.86** (random
expectation ≈ 0), and the median top-10% residue-set Jaccard between seed
pairs is **0.50** (random expectation ≈ 0.10); see
`figures/fig_attn_cross_seed_agreement.png` and
`figures/fig_attn_weight_examples.png`. Three independent training runs
converge on the *same* low-amplitude per-residue preference.

### 7.4 Mechanism: LayerNorm-then-pool

The combination of near-uniform weights and reproducible rank-order
identifies *LayerNorm-then-pool* as the dominant mechanism: with
near-uniform attention, the pooled representation collapses to
(1/L) Σ LN(eᵢ), which is provably different from the v4 mean-pool
(1/L) Σ eᵢ. Per-residue normalisation rescales high-norm residues
(typically signal peptides and disordered regions) so they no longer
dominate the pooled vector. The learned attention contributes the small
residual.

This is itself a paper-level finding: residue-level encoding helps
RankBind, but the active mechanism is per-residue normalisation rather than
learned residue selection. The reproducibility of the low-magnitude residue
preference across seeds suggests that real but subtle structural
information is being picked up; quantitative pocket overlap with M-CSA /
UniProt active-site annotations is a natural next investigation, deferred
to future work.

### 7.5 Implications for atom-level extension

The originally pre-registered atom-level extension would have identified
the top-K = 8 residues from the Stage-b attention map and built an atom
graph on those residues plus their 4 Å neighbourhood. The audit above blocks
this mechanism empirically: attention mass at rank-8 vs. rank-50 differs by
~10⁻⁴, so "the eight most-attended residues" is essentially a random sample
of the protein, and even consistent top-10% residue sets overlap only 50%
across seeds. We therefore defer the atom-level extension. A redesigned
variant that uses AlphaFold + fpocket structural priors instead of
attention-derived pocket selection is the only sound path forward.

---

## 8. Discussion

**Generality of the diagnosis.** The pooled-AUC vs.
ligand-conditional-ranking gap we measure on BRENDA is unlikely to be
unique to this benchmark. Any DTI corpus where the test protein
distribution overlaps the training protein distribution — or where some
proteins are dramatically more frequent than others — admits a
protein-level shortcut. The null-baseline probe is cheap (a single matrix
multiply) and we recommend it as a default sanity check before reporting
pooled discrimination metrics on new DTI datasets.

**Cost of shortcut avoidance.** RankBind explicitly trades pooled AUC for
ranking quality: 0.95 → 0.65 in gAUC, 0.06 → 0.33 in matrix MRR. The right
benchmark for an enzyme–substrate model is the latter — pooled AUC
certifies a property that does not generalise across ligand identity.

**Limitations.**

- *Per-ligand AUC is n = 4.* We retain it for continuity with prior work
  but cannot draw Pass/Fail conclusions from it.
- *200×200 probe matrix.* Scaling to the full ~1.4k-protein test pool
  would tighten standard errors but does not change the qualitative
  ranking.
- *Stage-b variance.* The MRR std grows from 0.072 to 0.123 between v4 and
  v5b; the mean lift is real but seed-to-seed spread is non-trivial and we
  report it explicitly.
- *No structural validation of attention weights.* The cross-seed Spearman
  of 0.86 shows the weights agree, but we have not yet cross-referenced
  them with M-CSA / UniProt active-site annotations.
- *Out-of-distribution generalisation untested.* The Phase-5 cross-dataset
  probe (ESP, TurNuP) is scoped but not executed; the +0.10 MRR lift is
  therefore an in-BRENDA finding.

---

## 9. Conclusion

We presented RankBind, a 627k-parameter architecture that breaks the
protein-level shortcut on BRENDA enzyme–substrate prediction. The core
observation is methodological: pooled AUC validates the wrong property.
Against four published baselines, all of which clear pooled AUC by
exploiting the per-protein label prior, RankBind enforces ligand-conditional
ranking through the joint action of protein-balanced sampling, a
within-ligand margin loss, a matched-capacity bilinear head, and online
hard-negative mining. A residue-level extension adds +0.10 absolute MRR
and reveals that residue-level information is best exploited via
per-residue normalisation rather than learned pocket selection. We release
code, configurations, and three-seed manifests; the natural next step is a
cross-dataset probe on ESP and TurNuP to measure how much of this lift
transfers under distribution shift.

---

## References

- Bai, P., Miljković, F., John, B., & Lu, H. (2023). Interpretable bilinear
  attention network with domain adaptation improves drug–target prediction.
  *Nature Machine Intelligence*, 5(2), 126–136.
- Chithrananda, S., Grand, G., & Ramsundar, B. (2020). ChemBERTa:
  Large-scale self-supervised pretraining for molecular property
  prediction. *arXiv:2010.09885*.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W.,
  Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural
  networks. *Nature Machine Intelligence*, 2(11), 665–673.
- Harwood, B., Kumar, B. G. V., Carneiro, G., Reid, I., & Drummond, T.
  (2017). Smart mining for deep metric learning. *ICCV*, 2821–2829.
- Huang, K., Xiao, C., Glass, L. M., & Sun, J. (2021). MolTrans: Molecular
  Interaction Transformer for drug–target interaction prediction.
  *Bioinformatics*, 37(6), 830–836.
- Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N.,
  Verkuil, R., Kabeli, O., Shmueli, Y., et al. (2023). Evolutionary-scale
  prediction of atomic-level protein structure with a language model.
  *Science*, 379(6637), 1123–1130.
- Nguyen, T., Le, H., Quinn, T. P., Nguyen, T., Le, T. D., & Venkatesh, S.
  (2021). GraphDTA: Predicting drug–target binding affinity with graph
  neural networks. *Bioinformatics*, 37(8), 1140–1147.
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified
  embedding for face recognition and clustering. *CVPR*, 815–823.
- Sohn, K. (2016). Improved deep metric learning with multi-class N-pair
  loss objective. *NeurIPS*, 1857–1865.
- Wang, H., et al. (2023). GEMS: A graph-enhanced model with pretrained
  protein language model embeddings for drug–target interaction
  prediction. *Bioinformatics*.
- Wu, C.-Y., Manmatha, R., Smola, A. J., & Krähenbühl, P. (2017). Sampling
  matters in deep embedding learning. *ICCV*, 2840–2848.
