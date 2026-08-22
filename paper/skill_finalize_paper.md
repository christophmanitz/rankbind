# RankBind Paper Revision Plan
## Target journals: JCIM + Scientific Reports
## Priority: Compute first, writing second

---

# 0. Mission

Revise the current RankBind manuscript into a submission-ready paper suitable for:

1. Journal of Chemical Information and Modeling (JCIM)
2. Scientific Reports

The paper's primary contribution must be framed correctly:

> The main contribution is not that RankBind achieves state-of-the-art DTI prediction.
> The main contribution is the quantitative demonstration that pooled pairwise AUC can be dominated by
> protein-level label priors in enzyme–substrate prediction, together with a simple diagnostic null model,
> ligand-conditional ranking metrics, and a training recipe that substantially reduces this shortcut.

The paper must explicitly acknowledge that shortcut learning, benchmark bias,
memorization, leakage, and unrealistic DTI evaluation are NOT new concepts.

The novelty must instead be positioned as:

> a specific, directly measurable protein-prior failure mode of pooled AUC,
> demonstrated with a molecule-blind baseline, controlled experiments,
> ligand-conditional ranking evaluation, and a simple mitigation strategy.

Do NOT artificially inflate the novelty of the architecture.

Do NOT claim that pooled AUC is universally invalid.

Do NOT claim that shortcut learning or cold-start problems were discovered here.

Do NOT claim that RankBind solves DTI prediction.

The scientific story is:

    known general problem
            ↓
    specific underappreciated failure mode
            ↓
    direct null-model diagnosis
            ↓
    quantitative dissociation of AUC and ranking
            ↓
    simple mitigation
            ↓
    cross-dataset validation
            ↓
    explicit limitations

---

# 1. CRITICAL WORKFLOW RULE

## COMPUTE FIRST. WRITING SECOND.

Do NOT spend substantial time rewriting the manuscript before the computational revision is complete.

The correct order is:

    Phase A — computational experiments
    Phase B — metric/statistical audits
    Phase C — leakage/reproducibility audit
    Phase D — literature/novelty audit
    Phase E — hostile reviewer analysis
    Phase F — manuscript rewriting
    Phase G — final fact check

The reason is simple:

Changing experiments after rewriting the manuscript creates
inconsistent claims, figures, tables, and conclusions.

Therefore all expensive experiments must be performed BEFORE
the final scientific narrative is rewritten.

---

# 2. PHASE A — ALL COMPUTATION FIRST

Run the following in this order.

---

## A0 — Freeze the current version

Before changing anything:

Create:

    revision_v0/

Store:

- current code
- current configs
- current checkpoints
- current results
- current manuscript
- current figures
- current tables

Record:

- git commit
- Python version
- CUDA version
- package versions
- GPU/CPU information

Create:

    BASELINE_REPRODUCTION.md

Goal:

The original paper must remain reproducible.

---

# 3. A1 — Metric implementation audit

Before running new experiments, independently verify:

- pooled AUC
- per-molecule AUC
- matrix MRR
- Hit@1
- Hit@5
- Hit@10
- Gini
- top-K overlap
- row-wise Spearman correlation with protein prior

The most important metrics must be independently reimplemented.

At minimum independently verify:

1. pooled AUC
2. matrix MRR
3. Hit@10
4. protein-prior overlap

Create:

    METRIC_AUDIT.md

For every metric record:

| Metric | Original implementation | Independent implementation | Difference | Status |
|---|---|---|---:|---|

No manuscript revision proceeds until the main metrics pass this audit.

---

# 4. A2 — Leakage and split audit

Audit:

- molecule overlap
- protein overlap
- duplicate SMILES
- duplicate proteins/sequences
- duplicate pairs
- scaffold overlap if available
- sequence similarity between train/test proteins
- decoy construction leakage
- preprocessing leakage
- validation/test contamination
- hard-negative mining leakage

Verify that:

- test labels never influence training
- test proteins never enter hard-negative pools
- test information never influences hyperparameters
- test information never influences checkpoint selection
- test information never influences candidate-pool construction

Create:

    LEAKAGE_AUDIT.md

This is mandatory.

---

# 5. A3 — Fix model selection before rerunning experiments

Current Methods state:

    early stopping on validation pooled AUC

This is potentially inconsistent with the central thesis.

The paper argues that pooled AUC can reward the protein shortcut.

Therefore change default checkpoint selection to:

    validation matrix MRR

Preferred:

    early stopping on validation matrix MRR

with the same patience/minimum epoch logic unless there is a strong
technical reason to change it.

IMPORTANT:

Do not merely change the manuscript.

RERUN the experiments.

---

# 6. A4 — Model-selection sensitivity experiment

Run both:

### Condition A

Checkpoint selected by:

    validation pooled AUC

### Condition B

Checkpoint selected by:

    validation matrix MRR

For both report:

- pooled AUC
- matrix MRR
- Hit@1
- Hit@5
- Hit@10
- protein-prior correlation
- top-K overlap

Question:

> Does the central conclusion depend on the model-selection criterion?

If YES:

- report it explicitly.
- explain the mechanism.
- avoid pretending the difference does not exist.

If NO:

- use matrix MRR selection as the primary protocol.
- report robustness in Supplementary Information.

This is a high-priority reviewer defense.

---

# 7. A5 — Multi-seed transfer experiments

Current weakness:

- BRENDA-200 core = 3 seeds
- transfer experiments = mostly single seed

Run at least:

    3 seeds

for:

1. BRENDA+SABIO kcat/KM
2. BRENDA+SABIO KM
3. BRENDA+SABIO turnover
4. ESP

If compute allows, also:

5. BindingDBKd
6. Davis
7. KIBA

Priority:

    enzyme datasets > kinase datasets

For each seed report:

- matrix MRR
- Hit@1
- Hit@5
- Hit@10
- pooled AUC
- row-wise prior correlation
- top-K prior overlap

Do not report only aggregate means.

Store:

    transfer_per_seed.csv

and:

    transfer_summary.csv

---

# 8. A6 — Paired per-molecule analysis

For each test molecule compute:

    RR_RankBind
    RR_BCE
    RR_prior

Then calculate:

    RankBind - BCE
    RankBind - prior

Run:

- paired distribution analysis
- median difference
- mean difference
- Wilcoxon signed-rank test if appropriate
- effect size
- 95% CI where appropriate

The molecule is the statistical unit.

Do NOT treat individual protein–molecule pairs as independent observations
when comparing molecule-level ranking performance.

This should become a central statistical validation.

---

# 9. A7 — Uncertainty estimation

For multi-seed experiments report:

    mean ± SD

and individual seed values.

For molecule-level metrics, if appropriate:

- bootstrap over molecules
- 95% CI
- fixed bootstrap seed
- number of replicates

Recommended:

    5,000 bootstrap replicates

Do not use bootstrap over individual pairs if the ranking unit is the molecule.

For n=4 analyses:

- do not use them as primary evidence.
- report descriptively only.

---

# 10. A8 — n=4 analysis demotion

Current issue:

Some per-molecule AUC calculations effectively have:

    n = 4

This is too small for a primary scientific conclusion.

Required:

- remove from headline results.
- move to Supplementary Information or methodological note.
- explicitly state n.
- avoid inferential claims.

Primary analysis should be:

    complete candidate matrix
    → per-molecule ranking
    → MRR / Hit@K

If n=30 molecules are available for the BRENDA-200 matrix:

    use n=30 as the primary molecule-level sample.

---

# 11. A9 — Candidate-pool sensitivity

The main BRENDA setup uses:

    200 candidate proteins

Test:

- 50
- 100
- 200
- 500 if computationally feasible

Compare:

- random
- protein prior
- BCE
- RankBind

Metrics:

- MRR
- Hit@1
- Hit@5
- Hit@10

If retraining at every pool size is too expensive:

- evaluate trained models on controlled candidate subsets.
- clearly label this as evaluation-only sensitivity.

Goal:

Show that the observed ranking advantage is not an artifact of exactly 200 candidates.

---

# 12. A10 — Protein-prior null baseline

Make this a first-class experiment.

Define:

    null_prot_prior(L,P) = positive_rate_train(P)

The baseline:

- does not use the molecule
- does not learn molecular embeddings
- does not train
- uses only training-set protein prevalence

Evaluate:

- pooled AUC
- matrix MRR
- Hit@K
- top-K overlap
- Gini

Main question:

> How much of the pooled-AUC performance can be reproduced without molecular information?

This is one of the strongest experiments in the paper.

---

# 13. A11 — Strong baseline control suite

Run:

1. Random
2. null_prot_prior
3. BCE
4. RankBind
5. RankBind without margin loss
6. RankBind without protein-balanced sampling
7. RankBind without hard-negative mining
8. RankBind with MLP head instead of bilinear head

Optional:

9. molecule-only
10. protein-only
11. simple bilinear model
12. constant prior + molecular residual

Purpose:

Determine whether RankBind:

- learns ligand-specific information,
- merely suppresses protein popularity,
- or benefits primarily from architectural complexity.

---

# 14. A12 — Simple architecture control experiment

The paper MUST explicitly test and explain the role of the simple architecture.

The architecture is NOT the novelty.

The purpose is:

    controlled diagnostic instrument

The architecture should answer:

> Can a deliberately simple ligand–protein interaction model
> learn ligand-conditional ranking when the training objective is
> changed from pooled BCE to ranking-oriented optimization?

Therefore compare:

### Simple model

    ligand encoder
        ↓
    ligand embedding

    protein encoder
        ↓
    protein embedding

    bilinear interaction
        ↓
    score matrix

against:

### Same model + BCE

and:

### Same model + RankBind objective

The encoders should remain identical.

Only the training objective / sampling / negative-mining
components should change where possible.

This creates a clean causal comparison.

---

# 15. A13 — Architecture ablation philosophy

The manuscript should state:

> We deliberately use a compact interaction architecture rather than proposing a new encoder architecture. This isolates the effect of ligand-conditional ranking objectives from architectural novelty and makes the model suitable as a diagnostic instrument.

This is important.

The model should NOT be marketed as:

- novel transformer
- novel graph architecture
- novel protein encoder
- SOTA architecture

Instead:

> RankBind is a controlled experimental vehicle for testing whether
> ligand-conditional ranking reduces protein-prior shortcut behavior.

---

# 16. A14 — Simple architecture section for the paper

Add a dedicated subsection:

## "A deliberately simple diagnostic architecture"

Content:

### 1. Motivation

The goal is not to win through architectural complexity.

The goal is to isolate the evaluation/training issue.

### 2. Architecture

Describe:

- molecular encoder
- protein encoder
- embedding dimensions
- bilinear interaction
- scalar score

Conceptually:

    z_L = f_ligand(L)
    z_P = f_protein(P)

    s(L,P) = z_L^T W z_P

### 3. Why bilinear interaction?

Because it provides:

- explicit cross-modal interaction
- low parameter count
- easy interpretation
- easy ablation
- direct score matrix construction

### 4. What is NOT claimed

Explicitly state:

> Neither the encoder components nor the bilinear interaction function are presented as architectural innovations.

### 5. What is being tested

The experiment tests:

    same representations
    +
    different objective
    =
    different shortcut behavior

This is much more informative than comparing unrelated architectures.

---

# 17. A15 — Margin-loss ablation

The current evidence suggests margin loss is the dominant component.

Verify systematically:

    BCE
    BCE + balanced sampling
    BCE + hard negatives
    BCE + bilinear head
    margin only
    full RankBind

Report:

- MRR
- Hit@K
- pooled AUC
- prior correlation
- prior overlap

Goal:

Determine whether the main effect comes from:

- ranking objective
- sampling
- negative mining
- architecture

The likely desired conclusion is:

> The ranking objective is the principal driver, while balanced sampling and hard-negative mining provide additional stabilization.

Only state this if the experiments support it.

---

# 18. A16 — Positive-density / protein-degree analysis

The protein-prior hypothesis predicts:

    stronger protein prevalence imbalance
        →
    stronger prior-based pooled-AUC advantage

Test this.

Analyze:

- protein positive-rate distribution
- degree distribution
- null-prior pooled AUC
- null-prior MRR
- model pooled AUC
- model MRR

Possible analysis:

Bin proteins by training positive rate.

Plot:

    positive rate
        vs
    model score / prior score

Also quantify:

    correlation(score, protein positive rate)

This directly tests the proposed mechanism.

---

# 19. A17 — Controlled synthetic experiment

Highly recommended.

Construct synthetic interaction matrices.

## Synthetic A — No ligand signal

Generate:

- strong protein prevalence skew
- no molecule-specific interaction information

Expected:

    pooled AUC = potentially high
    MRR = near random

## Synthetic B — Ligand-specific signal

Add:

- molecule-specific interaction structure

Expected:

    ranking model > prior

Purpose:

Demonstrate the statistical mechanism independently from any biological dataset.

This makes the argument much harder to dismiss as a BRENDA artifact.

---

# 20. A18 — BRENDA decoy-leakage audit

Audit the current decoy construction.

Run:

    frozen ligand embedding
    +
    protein embedding
    +
    simple linear classifier

No deep fine-tuning.

Measure:

- pooled AUC
- matrix ranking if meaningful

If AUC ≈ 0.86 remains:

interpret as evidence that the decoy construction itself contains
learnable pair-level structure.

Do NOT say:

    "BRENDA is invalid."

Say:

> The decoy construction introduces pair-level structure that can be exploited by representation-based models, motivating cautious interpretation of absolute pooled-AUC values.

This should be a limitation, not a reason to discard the dataset.

---

# 21. A19 — Attention analysis

Keep attention analysis secondary.

Current finding:

- attention improves MRR
- attention does not reliably identify catalytic residues
- hydrophobicity may correlate with attention

Do NOT claim:

    attention = binding-site discovery

Instead:

> Attention improves predictive representation but should not be interpreted as direct mechanistic evidence of catalytic-site localization.

If space is limited:

- keep one main figure
- move detailed residue analysis to Supplementary Information.

---

# 22. A20 — KIBA / negative transfer must remain

Do NOT remove poor results.

Keep:

- Davis
- KIBA

If RankBind fails to improve them:

say so.

This is scientifically useful because it prevents the paper from becoming:

    "RankBind always wins."

Instead:

> The proposed mitigation is most effective in datasets where protein-level label imbalance produces a measurable prior signal. Performance does not uniformly improve across all DTI benchmarks.

This strengthens the causal interpretation.

---

# 23. PHASE B — STATISTICAL AUDIT

Only after all major experiments are complete.

Create:

    STATISTICS_AUDIT.md

Check:

- n
- seeds
- unit of analysis
- confidence intervals
- SD vs SEM
- bootstrap methodology
- paired tests
- multiple comparisons
- effect sizes
- exact P values if used

Do not use statistical significance to rescue weak experiments.

Effect size and reproducibility are more important.

---

# 24. PHASE C — REPRODUCIBILITY AUDIT

Create:

    REPRODUCIBILITY_AUDIT.md

Verify:

- exact git commit
- data hashes
- split hashes
- configs
- seeds
- preprocessing
- checkpoints
- metrics
- figure generation
- table generation

Every main figure/table should be reproducible from code.

---

# 25. PHASE D — LITERATURE AND NOVELTY AUDIT

## Central positioning

The paper must explicitly say:

> Shortcut learning and benchmark bias are established problems.

Relevant literature includes:

### Pahikkala et al. (2015)

Pahikkala et al. argued for more realistic DTI prediction settings and highlighted that prediction for new drugs or new targets is substantially different from ordinary pair prediction.

Reference:

Pahikkala, T. et al.
"Toward more realistic drug-target interaction predictions."
Briefings in Bioinformatics (2015).

DOI:

10.1093/bib/bbu010

This establishes that evaluation setting and generalization regime are fundamental DTI issues.

### Wallach & Heifets (2018)

Wallach & Heifets demonstrated that benchmark redundancy can cause ligand-based classification benchmarks to reward memorization rather than generalization.

Reference:

Wallach, I.; Heifets, A.
"Most Ligand-Based Classification Benchmarks Reward Memorization Rather than Generalization."
Journal of Chemical Information and Modeling 58, 916–932 (2018).

DOI:

10.1021/acs.jcim.7b00403

This is particularly important for JCIM because it directly establishes that
benchmark construction can inflate apparent ML performance.

### More recent benchmark-bias literature

Search and include recent work on:

- DTI leakage
- degree bias
- entity-balanced evaluation
- dataset imbalance
- graph benchmark leakage
- ligand/protein memorization
- protein-ligand affinity benchmark leakage

The literature audit must extend through 2026.

---

# 26. IMPORTANT NOVELTY POSITIONING

The paper must NOT say:

> Shortcut learning in DTI is a new problem.

Instead:

> Shortcut learning, benchmark bias, memorization, and unrealistic generalization regimes have been recognized in DTI and related chemical ML tasks for years.

Then identify the gap:

> However, a particularly simple protein-marginal shortcut can remain hidden when evaluation relies on pooled pairwise AUC: a model can reproduce substantial apparent performance by assigning higher scores to proteins that are more frequently positive in the training data, without conditioning those predictions on the ligand.

Then:

> Our contribution is to make this failure mode explicit and measurable using a molecule-blind protein-prior baseline and to show its consequences for ligand-conditional ranking across enzyme–substrate datasets.

This is the correct novelty claim.

---

# 27. Distinguish GENERAL shortcut learning from THIS shortcut

The paper should distinguish:

## Previously known

- benchmark leakage
- chemical similarity leakage
- target similarity
- memorization
- cold-start failures
- degree imbalance
- network shortcuts
- dataset bias

## Specific focus of this paper

    protein positive-rate prior
            ↓
    pooled pairwise AUC
            ↓
    apparent performance
            ↓
    poor ligand-conditional ranking

This distinction is essential.

Do not claim:

> "Shortcut learning is ignored by the field."

Better:

> "Although benchmark shortcuts and leakage are well documented, the extent to which a simple protein-level label prior can dominate pooled pairwise AUC is not adequately exposed by conventional reporting."

Only retain "not adequately exposed" if the literature search confirms it.

---

# 28. PHASE E — HOSTILE REVIEWER STRESS TEST

Run three independent LLM reviewers.

## Reviewer A — JCIM computational chemistry

Focus:

- chemical relevance
- novelty
- benchmark construction
- DTI literature
- baseline quality
- interpretation

## Reviewer B — ML reviewer

Focus:

- metric validity
- leakage
- statistical independence
- seeds
- model selection
- ranking formulation

## Reviewer C — statistician/biologist

Focus:

- n
- biological interpretation
- substrate vs binding terminology
- generalization
- causal claims

Each reviewer must answer:

1. What is the strongest claim?
2. What evidence supports it?
3. What evidence is insufficient?
4. What experiment could falsify it?
5. Why might the paper be rejected?
6. What must be changed before submission?

Create:

    JCIM_REVIEW.md
    SCIREP_REVIEW.md
    ML_REVIEW.md

---

# 29. Claim-evidence matrix

Create:

| Claim | Evidence | Dataset | n | Seeds | Limitation | Keep? |
|---|---|---|---:|---:|---|---|

Every strong scientific statement in:

- Abstract
- Introduction
- Results
- Discussion

must be represented here.

Rules:

If evidence is weak:

    weaken claim

If evidence is missing:

    run experiment or remove claim

If evidence is BRENDA-only:

    say BRENDA

If evidence is enzyme-wide:

    say enzyme datasets

If kinase datasets are not convincing:

    do not generalize to all DTI.

---

# 30. Architecture positioning in the manuscript

Add a subsection:

## A deliberately simple architecture for diagnosis

Suggested scientific framing:

> We intentionally avoid introducing a novel encoder architecture. The purpose of RankBind is to isolate the effect of ligand-conditional ranking from architectural complexity. The model therefore uses standard molecular and protein encoders followed by a compact bilinear interaction function. This design produces an explicit ligand–protein score matrix while keeping the number of additional modeling assumptions small.

Then:

> This simplicity is deliberate: if the same representations can obtain high pooled AUC under BCE but substantially different ligand-conditional ranking behavior under a ranking-oriented objective, the resulting difference provides evidence about the evaluation/training objective rather than about architectural capacity.

Then explicitly:

> We therefore do not interpret the RankBind architecture itself as a novel contribution.

This should become an important methodological strength.

---

# 31. Primary scientific claim

The final manuscript should communicate:

> A model may obtain a high pooled AUC without learning ligand-conditional target preference because protein-level label prevalence can provide a strong predictive shortcut.

Then:

> A molecule-blind protein-prior baseline makes this shortcut directly measurable.

Then:

> Matrix MRR and Hit@K expose whether predictions actually rank the correct protein for a given ligand.

Then:

> RankBind demonstrates one simple way to reduce this shortcut through ligand-conditional ranking optimization.

Then:

> The phenomenon is strongest in enzyme–substrate settings studied here and should not automatically be generalized to every DTI benchmark.

---

# 32. Terminology

Use:

    enzyme–substrate prediction

for:

- BRENDA
- SABIO
- ESP
- catalytic enzyme datasets

Use:

    drug–target interaction (DTI)

only for the broader field.

Do not use "binding" when the label actually represents:

    documented substrate relationship

unless explicitly defined.

Suggested definition:

> Throughout the enzyme–substrate experiments, "interaction" denotes a documented enzyme–substrate relationship rather than generic physical binding.

---

# 33. Metrics

Primary:

    matrix MRR

Secondary:

    Hit@1
    Hit@5
    Hit@10

Diagnostic:

    pooled AUC
    protein-prior correlation
    top-K overlap

Descriptive only:

    Gini

Gini must NOT be presented as proof of shortcut learning because
the protein-prior baseline itself can produce highly concentrated scores.

Correct interpretation:

> Gini measures score concentration but does not distinguish learned molecular signal from concentration already present in the label distribution.

---

# 34. Small sample policy

For every result report:

    exact n

For n=4:

- descriptive only
- no strong inferential claim

For n≈30:

- matrix ranking is primary
- bootstrap if appropriate
- report uncertainty

For multi-seed:

    mean ± SD
    + individual seeds

Never hide sample size in supplementary material.

---

# 35. RankBind performance positioning

Current approximate result:

    matrix MRR ≈ 0.326 ± 0.072

Interpret carefully.

Do NOT say:

    near-perfect

    solved

    state-of-the-art

Instead:

> RankBind substantially exceeds random-ranking performance while remaining far from perfect target retrieval, indicating that the proposed objective mitigates the shortcut but does not solve enzyme–substrate prediction.

Explicitly state:

    MRR = 1.0

would correspond to perfect first-rank retrieval.

The contribution is:

    diagnosis + mitigation

not:

    complete prediction solution.

---

# 36. Literature novelty statement

The Introduction should contain a paragraph with the following logical structure:

1. Benchmark shortcuts are known.
2. DTI cold-start/generalization problems are known.
3. Ligand benchmark memorization is known.
4. Dataset imbalance and degree effects are known.
5. Therefore the conceptual existence of shortcuts is not novel.
6. What is less directly exposed is the specific effect of protein-level positive prevalence on pooled pairwise AUC.
7. This paper quantifies that effect using a molecule-blind baseline.
8. The paper then tests whether ligand-conditional ranking reduces it.

This is a much more defensible novelty claim.

---

# 37. Recommended contribution list

Use exactly 3–4 contributions:

1. **A direct diagnostic**
   A molecule-blind protein-prior baseline that quantifies how much pooled AUC can be explained without molecular information.

2. **A ligand-conditional evaluation framework**
   Matrix MRR and Hit@K evaluate the actual per-ligand target-ranking task.

3. **A controlled empirical demonstration**
   Published models, matched BCE controls, and multiple enzyme–substrate datasets demonstrate the dissociation between pooled AUC and ligand-conditional ranking.

4. **A simple mitigation strategy**
   RankBind uses standard representations with a compact bilinear interaction model and ranking-oriented optimization to reduce protein-prior dependence.

Do NOT list:

    "novel architecture"

as a contribution.

---

# 38. Results structure

Recommended:

## 3.1 Pooled AUC can be reproduced by a molecule-blind protein prior

Main result.

## 3.2 Pooled AUC and ligand-conditional ranking can strongly disagree

Show:

- AUC
- MRR
- Hit@K
- score matrices

## 3.3 A simple ranking-oriented model reduces the shortcut

Explain simple architecture.

## 3.4 Ablation identifies the dominant mechanism

Margin loss / balancing / hard negatives.

## 3.5 Cross-dataset validation

3 seeds.

## 3.6 Robustness and failure cases

- candidate pool
- positive density
- Davis
- KIBA
- decoy construction

---

# 39. Figures

## Figure 1

Conceptual:

    protein prevalence
          ↓
    model score
          ↓
    high pooled AUC

versus:

    ligand + protein
          ↓
    conditional score
          ↓
    correct target ranking

## Figure 2

Score matrices:

- DrugBAN
- BCE
- protein prior
- RankBind

## Figure 3

Scatter:

    pooled AUC
        vs
    matrix MRR

This is likely the central figure.

## Figure 4

Ablation:

- BCE
- no margin
- no balanced sampling
- no hard negatives
- RankBind

## Figure 5

Cross-dataset multi-seed results.

## Figure 6

Optional:

candidate-pool / positive-density robustness.

Attention should move to Supplementary if necessary.

---

# 40. Tables

## Table 1

Main BRENDA results:

| Model | pooled AUC | Matrix MRR | Hit@1 | Hit@5 | Hit@10 | Prior rho | Prior overlap |
|---|---:|---:|---:|---:|---:|---:|---:|

## Table 2

Ablation.

## Table 3

Cross-dataset results.

Include:

- n molecules
- n proteins
- seeds

## Supplementary

- per-seed results
- dataset statistics
- hyperparameters
- leakage audit
- decoy analysis
- metric validation
- sensitivity analyses

---

# 41. JCIM positioning

JCIM version should emphasize:

- chemical informatics
- molecular representation
- protein–ligand prediction
- benchmark evaluation
- ranking
- reproducibility

The paper should look like:

    chemical ML methodology + benchmark diagnosis

not:

    generic deep learning paper.

The fact that the architecture is deliberately simple should be presented
as a methodological advantage.

---

# 42. Scientific Reports positioning

Scientific Reports version should emphasize:

- methodological robustness
- transparent statistics
- reproducibility
- cross-dataset evidence
- limitations
- broad scientific interpretation

Do not depend on novelty alone.

Make the paper understandable to readers outside ML/chemoinformatics.

---

# 43. PHASE F — MANUSCRIPT REWRITE

Only begin after:

- all compute is finished
- all tables are frozen
- all figures are generated
- metric audit is complete
- literature audit is complete
- reviewer stress test is complete

Then rewrite:

1. Title
2. Abstract
3. Introduction
4. Results
5. Discussion
6. Methods
7. Supplementary Information

---

# 44. Abstract strategy

Abstract must emphasize:

    known general problem
        →
    specific protein-prior failure
        →
    diagnostic baseline
        →
    ligand-conditional metric
        →
    RankBind mitigation
        →
    multi-dataset evidence
        →
    limitations

Do NOT lead with architecture.

Do NOT claim SOTA.

Do NOT claim universal failure of AUC.

---

# 45. Suggested JCIM title

Preferred:

> Ligand-Conditional Ranking Reveals Protein-Prior Shortcuts in Enzyme–Substrate Prediction

Alternative:

> Protein-Prior Shortcuts in Enzyme–Substrate Prediction: Diagnosis with Ligand-Conditional Ranking

---

# 46. Suggested Scientific Reports title

Preferred:

> Protein-Prior Shortcuts Can Inflate Pooled AUC in Enzyme–Substrate Prediction

Alternative:

> Ligand-Conditional Ranking for Detecting Protein-Prior Shortcuts in Enzyme–Substrate Prediction

Avoid:

    "When AUC Lies"

in the final scientific title.

---

# 47. Discussion structure

## 47.1 What was already known

Shortcut learning, benchmark bias, memorization, and unrealistic
generalization regimes are established concerns.

## 47.2 What this study adds

A direct protein-prior null model exposes a specific mechanism
by which pooled AUC can be inflated.

## 47.3 Why the simple architecture matters

It isolates objective/evaluation effects from architectural novelty.

## 47.4 What RankBind achieves

It reduces prior-driven behavior and improves ligand-conditional ranking.

## 47.5 What RankBind does NOT achieve

It does not solve DTI prediction.

## 47.6 Generalization limits

Strongest evidence:

    enzyme–substrate datasets

Weaker:

    kinase datasets

Therefore do not generalize universally.

## 47.7 Future work

- structure-aware models
- better negative labels
- protein-disjoint evaluation
- ligand-disjoint evaluation
- calibrated candidate generation
- prospective experimental validation

---

# 48. Final reviewer defense

The paper must be able to answer:

### Reviewer:
"Is shortcut learning new?"

Answer:

> No. Shortcut learning, benchmark bias, leakage, and memorization are established concerns in chemical ML and DTI. Prior work has shown that evaluation design can reward similarity and memorization rather than generalization. Our contribution is narrower: we isolate a protein-level label-prior shortcut that can produce apparently strong pooled AUC without ligand-conditional target ranking, quantify it using a molecule-blind baseline, and test a simple ranking-based mitigation.

### Reviewer:
"Is RankBind architecture novel?"

Answer:

> No. The architecture is deliberately simple and is not presented as a novel architectural contribution. Its purpose is to isolate the effect of ligand-conditional ranking from architectural complexity.

### Reviewer:
"Is MRR = 0.326 enough?"

Answer:

> The goal is not to claim complete prediction accuracy. The result demonstrates substantial improvement over random ranking and reduction of the identified shortcut, while the remaining gap to perfect ranking is explicitly acknowledged.

### Reviewer:
"Why not just use pooled AUC?"

Answer:

> Pooled AUC remains useful for comparability, but it does not guarantee that a model ranks the correct target for a given ligand. We therefore recommend reporting pooled AUC together with ligand-conditional ranking metrics.

### Reviewer:
"Does this apply to all DTI?"

Answer:

> The strongest evidence concerns enzyme–substrate prediction. We do not claim universal applicability to every DTI benchmark.

---

# 49. Final acceptance criteria

The paper is submission-ready only when:

## Computational

- [ ] All expensive experiments complete before final rewrite.
- [ ] Core transfer experiments use ≥3 seeds.
- [ ] Model selection uses ligand-conditional validation metric.
- [ ] Metric implementations independently verified.
- [ ] Leakage audit passed.
- [ ] Hard-negative mining is leakage-free.
- [ ] n reported everywhere.
- [ ] n=4 analysis is not primary evidence.
- [ ] Candidate-pool sensitivity completed if feasible.
- [ ] Protein-prior null model verified.
- [ ] Architecture ablation completed.
- [ ] Margin-loss ablation completed.
- [ ] Decoy-leakage audit completed.

## Scientific

- [ ] Shortcut learning acknowledged as established.
- [ ] Pahikkala et al. cited.
- [ ] Wallach & Heifets cited.
- [ ] Recent benchmark-bias literature checked through 2026.
- [ ] Novelty claim narrowed appropriately.
- [ ] Simple architecture explicitly framed as diagnostic.
- [ ] RankBind not presented as architectural novelty.
- [ ] RankBind not presented as SOTA.
- [ ] Negative transfer results retained.
- [ ] Enzyme–substrate vs generic DTI terminology consistent.

## Statistical

- [ ] Seed-level uncertainty reported.
- [ ] Molecule-level uncertainty reported where appropriate.
- [ ] Correct statistical unit used.
- [ ] No pseudoreplication.
- [ ] Effect sizes reported where relevant.
- [ ] Statistical tests fully specified.

## Writing

- [ ] Abstract emphasizes diagnosis.
- [ ] Introduction distinguishes known problem from new contribution.
- [ ] Results follow evidence hierarchy.
- [ ] Discussion clearly separates:
      known
      demonstrated
      inferred
      not demonstrated
- [ ] Limitations are explicit.
- [ ] All numerical claims fact-checked.

---

# 50. Final scientific positioning

The final paper should communicate exactly this:

> Shortcut learning in DTI and chemical ML is not new.
> Benchmark bias, memorization, leakage, and unrealistic generalization have been documented for years.
> The contribution of this work is to expose a particularly simple but consequential protein-prior shortcut:
> pooled pairwise AUC can reward predictions based largely on protein-level positive prevalence,
> even when the model does not correctly rank targets conditional on a ligand.
>
> The molecule-blind protein-prior baseline makes this shortcut directly measurable.
> Matrix MRR and Hit@K reveal whether a model actually performs ligand-conditional target ranking.
> A deliberately simple interaction architecture allows the effect of the ranking objective to be studied
> without confounding it with architectural novelty.
> RankBind therefore serves primarily as a controlled mitigation and diagnostic framework,
> not as a claim of a fundamentally new neural architecture or a solved DTI problem.
>
> The strongest evidence concerns enzyme–substrate prediction, and the study deliberately retains
> negative and limited-transfer results to avoid overgeneralization.

---

# 51. Priority order — FINAL

If compute/time is limited:

## P0 — mandatory and computationally expensive first

1. Freeze current version.
2. Metric audit.
3. Leakage/split audit.
4. Change checkpoint selection to validation matrix MRR.
5. Rerun BRENDA core with 3 seeds.
6. Rerun four enzyme transfer datasets with 3 seeds.
7. Run full baseline/control suite.
8. Run paired per-molecule analysis.
9. Run uncertainty analysis.
10. Run decoy-leakage diagnostic.

## P1 — high-value

11. Candidate-pool sensitivity.
12. Protein-degree / positive-density analysis.
13. Architecture/objective ablation.
14. Synthetic experiment.
15. Additional kinase seeds if compute allows.

## P2 — literature and reviewer work

16. Literature search through 2026.
17. Novelty audit.
18. Hostile JCIM review.
19. Hostile Scientific Reports review.
20. Claim-evidence matrix.

## P3 — only now rewrite

21. Figures.
22. Tables.
23. Abstract.
24. Introduction.
25. Results.
26. Discussion.
27. Methods.
28. Supplementary Information.

## P4 — final verification

29. Fact check every number.
30. Re-run all figures from clean environment.
31. Verify repository reproducibility.
32. Generate JCIM manuscript version.
33. Generate Scientific Reports manuscript version.

---

# END
