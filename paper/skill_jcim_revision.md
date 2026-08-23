# JCIM Pre-Submission Revision Plan for RankBind

## Purpose

This document is an actionable revision brief for an LLM assisting with the preparation of the **RankBind** manuscript for submission to the **Journal of Chemical Information and Modeling (JCIM)**.

The goal is not to redesign the paper or replace its central contribution. The goal is to make the manuscript more defensible against the most likely reviewer objections, especially objections concerning:

1. over-generalization of the main claim,
2. BRENDA decoy construction and ligand-level leakage,
3. the fact that the current primary split is protein-stratified rather than ligand-disjoint,
4. whether the reported shortcut is specific to BRENDA or generalizes beyond this benchmark construction,
5. whether RankBind's gains remain under genuinely cold-ligand evaluation.

The central recommendation is to **strengthen the benchmark/evaluation story rather than oversell RankBind as a novel architecture**.

---

# 1. Core Editorial Principle

The paper's strongest contribution is not the architecture itself.

RankBind uses frozen ChemBERTa and ESM2 encoders, two projectors, a low-rank bilinear interaction head, a within-ligand margin loss, hard-negative mining, and protein-balanced sampling. The manuscript itself explicitly states that the architecture is not intended as the primary novelty; RankBind is a controlled experimental vehicle for testing whether ligand-conditional ranking reduces shortcut behavior.

Therefore, the paper should be framed primarily as a **methodological and benchmarking study**:

> Pooled pairwise discrimination metrics can substantially overstate ligand-conditional target-ranking performance on enzyme–substrate benchmarks when the benchmark construction permits ligand-level regularities to recur across train and test. A molecule-blind null and within-molecule ranking metrics expose this failure, while a ranking-specific objective can substantially reduce the shortcut.

This framing is stronger and more defensible than:

> We introduce a new DTI architecture called RankBind.

Do not let revisions accidentally turn the manuscript into a conventional "new model beats baselines" paper.

---

# 2. Revision Priority A — Narrow the Main Claim

## Problem

The current wording sometimes approaches the broad statement:

> "pooled AUC is not a meaningful success gate"

The evidence in the manuscript strongly supports criticism of pooled AUC **in the tested enzyme–substrate setting**, but it does not justify treating pooled AUC as universally invalid for all DTI tasks.

The manuscript itself contains evidence against such a universal claim. In the kinase-affinity benchmarks Davis and KIBA, the molecule-blind prior carries essentially no ranking signal. This means the shortcut identified in the enzyme–substrate experiments is not automatically present in every DTI benchmark.

The paper should therefore distinguish:

- **the general methodological principle**: pooled metrics can measure a different property from within-ligand target ranking;
- **the demonstrated empirical result**: this dissociation is especially pronounced in the tested enzyme–substrate benchmarks and their benchmark construction.

## Required change

Make the scope explicit throughout:

- Abstract
- Introduction
- Results section headings where appropriate
- Discussion
- Conclusion
- Cover letter, if prepared

Prefer language such as:

> "On the enzyme–substrate benchmarks studied here..."

or:

> "For the tested enzyme–substrate benchmark construction..."

or:

> "In enzyme–substrate datasets with recurring ligands across the protein-stratified split..."

Avoid unqualified formulations such as:

> "Pooled AUC is meaningless for DTI."

or:

> "Pooled AUC is invalid for drug–target interaction prediction."

## Recommended claim hierarchy

The manuscript should distinguish three levels of claim.

### Level 1 — Broad methodological principle

Supported:

> A pooled pairwise metric and a within-ligand ranking metric need not measure the same property.

This is demonstrated conceptually and empirically.

### Level 2 — Empirical benchmark finding

Strongly supported:

> In the tested enzyme–substrate benchmarks, BCE-style pairwise training can produce high pooled AUC while remaining near chance on within-ligand ranking.

The BRENDA-200 experiment and the seven-dataset comparison support this.

### Level 3 — General DTI conclusion

Not yet fully supported:

> The same failure necessarily occurs across all DTI datasets.

Do not make this claim.

The Davis/KIBA results should instead be presented as useful boundary cases.

## Why this makes the paper stronger

A reviewer can easily reject an over-broad claim by pointing to a counterexample.

A scoped claim is harder to attack:

> "We identify and diagnose a specific failure mode in enzyme–substrate benchmark evaluation."

That is a precise, testable contribution.

---

# 3. Revision Priority A — Make the BRENDA Decoy and Ligand-Leakage Issue Central

> This section is adapted to the concrete repo (`~/rankbind`, Leipzig HPC).
> Every number below is regenerable from a checked-in script; nothing is
> invented. The manuscript files to edit are `paper/scirep/main.tex`
> (the reviewed manuscript, per `paper/JCIM_REVIEW.md`) and, if kept in
> sync, `paper/main.tex` / `paper/paper.md`.

## Problem (repo-grounded)

The BRENDA-derived benchmark has a strong ligand-level structure. The
numbers are computed by `evaluation/decoy_leakage_probe.py` (printed
line `[probe] ligand purity: ...`; output `evaluation/DECOY_LEAKAGE_AUDIT.md`
+ `evaluation/decoy_leakage_probe.csv`):

- 9,632 pairs in `data/dataset_with_decoys.csv`
  (3,175 positives / 6,457 decoys; see `paper/scirep/zenodo_manifest.md`);
- 4,604 unique ligands, of which:
  - 1,417 appear only as documented substrates (label 1),
  - 3,157 appear only as decoy carriers (label 0),
  - ~99.3% of ligands are label-pure;
- the split is protein-stratified (`baselines/adapters/common.py::
  BRENDADataConfig.get_protein_split()`, seed=42), so it holds out
  proteins, **not** molecules: `evaluation/leakage_audit.py` reports
  that 53.6% of test-pair ligands also occur in training rows
  (scaffold overlap 71.5%; see `evaluation/LEAKAGE_AUDIT.md`, L3/L4).

Consequently, a ligand can recur across train and test while carrying a
highly informative label-role prior.

The repo already contains the diagnostics the manuscript should elevate:

| diagnostic | script (regenerates) | artifact | current value |
|---|---|---|---|
| ligand-only linear probe (frozen ChemBERTa/ESM2, linear head) | `evaluation/decoy_leakage_probe.py` | `evaluation/decoy_leakage_probe.csv` | test AUC **0.887** from the ligand vector alone; **0.833** using both blocks; protein-only **0.603** |
| full-probe matrix MRR (same script) | same | same | **0.029** (= chance H_200/200 = 0.0294; Hit@5 0.00) |
| per-ligand training-rate prior | `evaluation/null_baseline_table.py` | `evaluation/NULL_BASELINE.md` + `null_baseline_firstclass.csv` | pooled AUC **0.915** on the FULL test split (pool subset 0.632, only 36 pairs) |
| leakage audit L1–L7 (incl. molecule overlap) | `evaluation/leakage_audit.py` | `evaluation/LEAKAGE_AUDIT.md` + `leakage_audit_runs.csv` | mol-overlap 53.6%, scaffold-overlap 71.5% |

These numbers are highly relevant because they establish that a
substantial portion of the pooled-AUC performance can arise from ligand
identity or ligand-level regularity rather than target-specific
interaction learning.

## How to execute (concrete steps for the LLM)

1. Regenerate the decoy probe (embeddings are cached; linear classifier
   only — runs on the login node in a few minutes):
   ```bash
   module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
   source ~/venvs/hieratombind/bin/activate
   python evaluation/decoy_leakage_probe.py
   ```
   Expected: `molecule_only 0.887 / full 0.833 / protein_only 0.603`
   in `evaluation/decoy_leakage_probe.csv`; audit text in
   `evaluation/DECOY_LEAKAGE_AUDIT.md`. If any value differs, use the
   regenerated value — do NOT hard-code numbers.
2. Regenerate the ligand-prior null:
   ```bash
   python evaluation/null_baseline_table.py
   ```
   Expected: `null_lig_prior` pooled AUC 0.915 on the full test split in
   `evaluation/NULL_BASELINE.md` + `null_baseline_firstclass.csv`.
3. Regenerate the leakage audit:
   ```bash
   python evaluation/leakage_audit.py
   ```
   Expected: L3 molecule overlap 53.6%, L4 scaffold overlap 71.5%,
   all hard checks (L1/L2) pass in `evaluation/LEAKAGE_AUDIT.md`.
4. Cite the claim-evidence map: `paper/CLAIM_EVIDENCE_MATRIX.md` row C4
   already pins these numbers to their evidence. When you move text
   between sections, update this matrix.

## Reviewer risk

A skeptical JCIM reviewer could argue:

> "The paper is not primarily demonstrating a general flaw in DTI evaluation; it is demonstrating an artifact induced by the specific BRENDA/decoy construction."

This objection must be addressed directly. Note it was already raised as
JCIM_REVIEW.md #3 ("decoy circularity"); the current mitigation is only a
provenance sentence in §2.4 (`paper/REVIEW_TRIAGE.md`). The plan below
upgrades that sentence into a central diagnostic.

Do not hide the issue in the limitations section.

Instead, turn it into an explicit part of the paper's causal/diagnostic story:

1. The benchmark construction permits ligand-level regularities to recur.
2. Pooled AUC can exploit these regularities.
3. A ligand-only diagnostic demonstrates their predictive strength.
4. The molecule-blind protein prior diagnoses a separate protein-marginal shortcut.
5. Within-ligand ranking distinguishes these effects from the desired ligand-conditional interaction signal.
6. Additional datasets and synthetic experiments test whether the metric-side argument survives outside the exact BRENDA construction.

This is a much cleaner scientific narrative.

## Manuscript placement (where the text goes)

- **Elevate, don't add**: `paper/scirep/main.tex` already contains the
  ligand census and probe numbers in the Discussion/limitations area
  (around line 624–631: "1,417 ... substrates and 3,157 only as decoy
  carriers (99.3% label-pure) ... 0.887 ... 0.833 ... 0.915"). Move this
  block out of limitations into a Results subsection (see §17, section
  2.2 "Diagnose the shortcut") and keep a one-sentence pointer in the
  limitations.
- The data-construction facts go into the Data paragraph of Methods
  (already present around `scirep/main.tex` lines 657–676: decoys
  author-generated, MolTransformer-based, seed 42, Tanimoto band
  0.3–0.8). Make the role-pure ligand structure explicit there.
- Update `paper/CLAIM_EVIDENCE_MATRIX.md` C4 wording if the claim scope
  changes (it currently reads correctly).

## Verification

- All three scripts above re-run without error and reproduce the table
  in this section (±0.002).
- `paper/scirep/main.tex` contains the census + probe numbers in a
  Results subsection (not only limitations), and `CLAIM_EVIDENCE_MATRIX.md`
  C4 still matches the text.

---

# 4. Add a Diagnostic Table: Information Source → Pooled AUC → Matrix MRR

## Strongly recommended experiment

Add a compact table, ideally in the main paper, with columns such as:

| Information source / model | Sees ligand? | Sees protein? | Pooled AUC | Matrix MRR | Top-10 overlap |
|---|---:|---:|---:|---:|---:|
| Random baseline | No | No | ... | ... | ... |
| Protein-only prior | No | Yes | ... | ... | ... |
| Ligand-only prior/probe | Yes | No | ... | ... | ... |
| Ligand + protein shortcut baseline | Yes | Yes | ... | ... | ... |
| BCE control | Yes | Yes | ... | ... | ... |
| RankBind | Yes | Yes | ... | ... | ... |

The exact rows should use quantities that can actually be computed from the existing code and data.

Do not invent results.

## Purpose of the table

The table should make the following distinction visually obvious:

> High pooled AUC does not by itself demonstrate ligand-conditional target discrimination.

The most important comparison is not necessarily RankBind versus published baselines. It is:

> **What can be achieved by increasingly cheap information sources that do not perform the intended interaction-ranking task?**

This will make the paper much more compelling to a chemical-informatics audience.

---

# 5. Revision Priority A — Add Cold-Ligand Evaluation

## Why this is the most valuable new experiment

The current main BRENDA split is protein-stratified.

The manuscript explicitly states that molecules can recur between training and test. This is central to the observed ligand-level shortcut.

Therefore, the cleanest test is to introduce a ligand-disjoint evaluation.

At minimum, evaluate:

1. **Cold-protein**
   - test proteins are unseen during training.

2. **Cold-ligand**
   - test ligands are unseen during training.

3. **Cold-protein + cold-ligand**
   - neither proteins nor ligands are seen during training.

The existing protein-stratified split should remain as the main benchmark because it is the setting in which the original dissociation is demonstrated. The new splits should be presented as stress tests of the explanation.

---

# 6. What the Cold-Ligand Experiment Should Answer

The experiment should not merely add another leaderboard.

It should answer a mechanistic question:

> Does the pooled-AUC/ranking dissociation persist when ligand identity cannot recur across train and test?

There are several possible outcomes, and the manuscript should report whichever occurs without forcing a preferred narrative.

## Outcome A — The shortcut largely disappears

If pooled AUC falls toward ranking-consistent values under ligand-disjoint splitting, this is strong evidence that ligand recurrence is a major causal ingredient in the original shortcut.

The paper should then say so explicitly.

This would make the benchmark-construction critique stronger, not weaker.

## Outcome B — The shortcut persists

This would be even more interesting.

It would indicate that the failure is not explained solely by recurring ligand identity and that other dataset-level regularities or decoy structures can sustain pooled AUC without target ranking.

If this happens, make it a major result.

## Outcome C — RankBind loses much of its advantage

This must also be reported honestly.

It could indicate that RankBind's advantage is partly tied to the original split or to the available training signal.

Do not interpret this automatically as failure of the main thesis. The paper's main contribution is the diagnosis of evaluation failure, not the universal superiority of RankBind.

## Outcome D — RankBind remains strong

This is the best-case result.

It would demonstrate both:

1. the original shortcut diagnosis is real, and
2. the ranking-oriented training strategy generalizes beyond the leakage regime.

That would materially strengthen the JCIM submission.

---

# 7. Design the Cold-Ligand Experiment Carefully

Avoid introducing an uncontrolled second benchmark.

Use the same:

- encoder versions,
- training budget,
- optimizer,
- model capacity,
- negative-mining logic,
- evaluation matrix size where possible,
- seeds,
- reporting conventions.

Only change the split definition.

The clean comparison should be:

> **same model + same training protocol + different information leakage structure**

This makes the experiment interpretable.

---

# 8. Report the Null Baselines on Every Split

For each split, evaluate at least:

- random baseline,
- protein-only prior,
- ligand-only prior where applicable,
- BCE control,
- RankBind.

The molecule-blind protein prior is especially important because the paper's central diagnostic is based on it.

Do not omit it simply because its pooled AUC becomes uninformative on a particular split.

Its lack of signal is itself a result.

---

# 9. Preserve the Current Seven-Dataset Transfer Analysis

Do not remove the multi-dataset analysis.

It is one of the strongest defenses against the claim that everything is simply a BRENDA artifact.

The manuscript already reports that:

- BCE reproduces the pooled-AUC / near-chance-ranking dissociation across the tested enzyme–substrate corpora;
- RankBind improves matrix MRR by roughly 7–25× on the catalytic enzyme datasets;
- ESP provides an independent enzyme–substrate corpus where genuine per-molecule ranking signal exists;
- Davis and KIBA act as boundary cases where the molecule-blind prior does not provide the same shortcut.

This is important evidence.

However, make the logic explicit:

> The transfer experiments do not prove that every DTI benchmark suffers from the same shortcut. They show that the observed evaluation pathology is reproducible across the tested enzyme–substrate corpora and has meaningful boundary cases in kinase-affinity data.

That wording is safer.

---

# 10. Use Davis/KIBA as Boundary Cases, Not as Embarrassing Exceptions

The current manuscript reports:

- BindingDBKd: RankBind matrix MRR 0.320;
- Davis: matrix MRR 0.038, despite per-molecule AUC 0.686;
- KIBA: matrix MRR 0.023 and per-molecule AUC 0.457.

The manuscript states that on Davis and KIBA the molecule-blind prior carries no ranking signal and that KIBA did not converge to a ranking solution.

Do not hide these results.

Use them strategically.

A useful framing is:

> "The diagnostic is not a claim that pooled AUC is universally pathological. Rather, it provides a test for whether pooled AUC is measuring the intended ligand-conditional property on a particular benchmark."

This is a substantially more defensible contribution.

---

# 11. Clarify the Causal Structure of the Paper

The paper currently discusses multiple phenomena:

1. protein-marginal concentration,
2. ligand recurrence,
3. decoy construction,
4. pooled AUC,
5. within-ligand ranking,
6. RankBind,
7. residue attention.

These should be separated conceptually.

A recommended causal narrative is:

### Step 1 — Define the intended task

For each ligand, rank its true target proteins above alternatives.

### Step 2 — Show the standard metric can disagree with that task

Pooled AUC evaluates random positive-vs-negative pairs across the entire test set.

### Step 3 — Identify shortcut sources

- protein label marginal,
- recurring ligand identities,
- benchmark decoy structure.

### Step 4 — Introduce null diagnostics

- protein-only prior,
- ligand-only diagnostic,
- top-K overlap,
- score-matrix concentration.

### Step 5 — Use the correct task-aligned metric

- matrix MRR,
- Hit@K,
- per-molecule AUC.

### Step 6 — Train toward the desired property

- within-ligand margin loss,
- hard negatives,
- protein-balanced sampling.

### Step 7 — Test generality

- multiple enzyme–substrate datasets,
- independent ESP dataset,
- kinase-affinity boundary cases,
- preferably cold-ligand/cold-protein splits.

This sequence will make the manuscript easier to review.

---

# 12. Do Not Overstate RankBind's Architectural Novelty

The paper should explicitly state that the contribution is primarily:

- evaluation,
- diagnosis,
- objective design,
- controlled empirical analysis.

The architecture itself is intentionally simple.

This is not a weakness if framed correctly.

In fact, a simple model is useful because it isolates the effect of the training objective.

The BCE control uses the same encoders and split, making the comparison especially valuable.

The paper should emphasize:

> The experimental design asks whether changing the learning objective and sampling strategy changes the model's behavior under the same representation and data conditions.

This is much more convincing than claiming a new neural architecture.

---

# 13. Reproducibility: Make the Existing Strength More Visible

The manuscript has unusually strong reproducibility infrastructure:

- public source code,
- configurations,
- reproducibility instructions,
- per-run manifests,
- input/output SHA-256 hashes,
- checkpoint hashes,
- score-matrix hashes,
- pinned data pipeline,
- corrected split-pinning protocol.

These details should be made highly visible.

The paper already states that manifests pin configuration, git commit, library versions, input-data SHA-256, checkpoint SHA-256, and score-matrix SHA-256.

Do not bury this entirely in Methods.

A short reproducibility paragraph in the Introduction, Methods, or Data/Code Availability section can help reviewers immediately see that the reported experiments are traceable.

---

# 14. Seed Reporting Needs to Be Explicit

The paper already has:

- three seeds for the main BRENDA-200 experiments,
- a ten-seed extension for the bilinear-vs-MLP head comparison,
- supplementary additional seeds for the three BRENDA+SABIO families.

The manuscript also honestly reports substantial seed spread.

For the final JCIM version:

- use mean ± SD consistently,
- clearly distinguish three-seed and ten-seed results,
- clearly mark single-seed experiments,
- do not compare a three-seed mean against a single-seed number as though they have equal statistical support.

In particular, the residue-attention result should remain clearly marked as a single-seed observation unless it is replicated.

---

# 15. Residue Attention Should Be De-Emphasized Unless Replicated

The residue-attention experiment is interesting but not necessary for the central paper.

The manuscript reports:

- MRR improvement on a shared seed-42 anchor,
- near-uniform attention weights,
- poor enrichment of annotated functional residues,
- strong relationship with hydrophobicity,
- evidence that normalization rather than pocket discovery explains the gain.

This is a valuable mechanistic observation.

However, the headline improvement is currently based on a single seed.

Recommended options:

### Preferred option

Move most of the attention analysis to supplementary material and retain a concise paragraph in the main paper.

### Alternative

Keep it in the main text, but explicitly label it:

> "single-seed mechanistic analysis"

and do not use it as evidence for the main performance claim.

If additional seeds are cheap, replicate it. If not, de-emphasize it.

---

# 16. Avoid Letting the Paper Become Too Broad

Do not add experiments merely because they are interesting.

The final manuscript should have one dominant question:

> When does pooled pairwise discrimination fail to measure ligand-conditional target ranking, and how can that failure be detected and reduced?

Everything should support that question.

The following should remain secondary:

- detailed architecture choices,
- residue attention,
- specific head comparisons,
- implementation details that do not affect the main conclusion.

---

# 17. Recommended New Main-Text Experiment Structure

A strong revised Results structure would be:

## 2.1 Pooled AUC can disagree with ligand-conditional ranking

Keep the existing BRENDA-200 demonstration.

Show:

- pooled AUC,
- per-molecule AUC,
- matrix MRR,
- Hit@10,
- score matrices.

## 2.2 Diagnose the shortcut

Add:

- protein-only prior,
- ligand-only diagnostic,
- ligand-level training-rate prior,
- top-K overlap,
- concentration metrics.

This should explicitly quantify how much information is available without modeling the interaction.

## 2.3 RankBind reduces the shortcut

Keep:

- BCE control,
- margin-loss ablation,
- balanced sampler ablation,
- hard-negative analysis,
- head comparison.

Focus on the objective rather than architectural novelty.

## 2.4 Cold-ligand and cold-protein stress tests

Add:

- protein-stratified,
- ligand-stratified,
- double-cold.

This becomes the critical robustness section.

## 2.5 Transfer across datasets

Keep the seven-dataset analysis.

Use Davis/KIBA as boundary cases.

## 2.6 Practical evaluation recipe

Retain the five-point recipe, but make it more carefully scoped.

---

# 18. Suggested Revision to the Practical Recipe

Instead of presenting the recipe as universally applicable to all DTI benchmarks, say:

> For enzyme–substrate benchmarks and other datasets where ligand or protein marginals may dominate pooled metrics, run the following diagnostic before interpreting pooled AUC.

Then:

1. Run the molecule-blind protein prior.
2. Run a ligand-only diagnostic where ligand recurrence is possible.
3. Evaluate within-ligand ranking.
4. Check ligand-disjoint and protein-disjoint splits where feasible.
5. Use ranking metrics as the success gate when the scientific task is target ranking.

This is more rigorous.

---

# 19. Potential Reviewer Questions to Preempt

The revised manuscript should be able to answer these questions directly.

### Q1. Is this just a BRENDA artifact?

Answer with:

- seven-dataset transfer,
- independent ESP,
- synthetic experiment,
- explicit decoy discussion,
- cold-ligand experiments.

### Q2. Is this simply ligand leakage?

Answer with:

- ligand-only diagnostic,
- ligand-disjoint split,
- protein-only prior,
- separation of different shortcut sources.

### Q3. Does RankBind actually generalize?

Answer with:

- multiple datasets,
- multiple seeds,
- cold-ligand evaluation,
- clear confidence/variance reporting.

### Q4. Why not simply use ligand-disjoint splitting?

Do not claim that one split solves every issue.

Explain that:

- different splits answer different scientific questions,
- cold-protein evaluates target generalization,
- cold-ligand evaluates ligand generalization,
- double-cold evaluates both,
- the current diagnostic addresses metric validity under a commonly used protein-stratified setting.

### Q5. Why matrix MRR?

Explain:

- the task is target ranking per ligand,
- matrix MRR is directly aligned with this task,
- it is defined for every test molecule in the constructed evaluation matrix,
- it avoids the tiny effective sample size of the test-pair per-molecule AUC variant.

### Q6. Is RankBind itself the important innovation?

Answer:

> No. The primary contribution is the diagnostic and evaluation framework; RankBind is a controlled demonstration that optimizing the intended ranking property can reduce the identified shortcut.

This is an advantage, not an apology.

---

# 20. Recommended Wording for the Central Contribution

A safer and stronger formulation would be approximately:

> We show that, on the enzyme–substrate benchmarks studied here, pooled pairwise AUC can substantially overstate ligand-conditional target discrimination when ligands recur across a protein-stratified split. A molecule-blind protein prior, ligand-level diagnostics, and score-matrix analyses reveal that high pooled AUC can coexist with near-chance within-ligand ranking. We therefore advocate evaluating target-ranking tasks with within-ligand metrics and explicitly testing null models that capture dataset-level marginals. RankBind provides a controlled example of how a within-ligand ranking objective, protein-balanced sampling, and hard-negative mining can reduce the identified shortcut.

This should be treated as a conceptual target, not necessarily copied verbatim.

---

# 21. What NOT to Do

Do not:

- claim that all DTI benchmarks are broken;
- claim that pooled AUC is always meaningless;
- hide the BRENDA decoy construction;
- hide ligand recurrence;
- remove Davis/KIBA because they are non-wins;
- present the single-seed attention result as fully established;
- present RankBind as a major new architecture;
- rely only on pooled AUC for any new experiment;
- introduce a new split without matching controls;
- mix seed counts without clear labeling;
- turn the paper into a large collection of loosely connected ablations.

---

# 22. Minimum Revision Before JCIM Submission

If time is limited, prioritize exactly these changes:

## Must-have 1 — Scope the claims

Rewrite the abstract, introduction, discussion, and conclusion so the strongest claims are explicitly limited to the tested enzyme–substrate benchmark setting.

## Must-have 2 — Add ligand-level diagnostics

Create a clear comparison involving:

- protein-only prior,
- ligand-only diagnostic,
- BCE,
- RankBind.

Report pooled AUC and matrix MRR.

## Must-have 3 — Add cold-ligand evaluation

At minimum:

- cold-ligand,
- preferably cold-protein + cold-ligand.

Use matched controls.

## Must-have 4 — Make the BRENDA construction transparent

Explicitly discuss:

- label-pure ligand structure,
- recurring ligands,
- author-generated decoys,
- ligand-only performance.

Frame these as diagnostic facts rather than embarrassing caveats.

## Must-have 5 — Keep the transfer evidence

Retain the seven-dataset analysis and use Davis/KIBA as boundary cases.

---

# 23. Ideal Revision Before Submission

If resources permit, the ideal JCIM version would include:

- protein-stratified baseline,
- cold-ligand split,
- cold-protein split,
- double-cold split,
- protein-only prior,
- ligand-only prior,
- ligand + protein shortcut baseline,
- BCE control,
- RankBind,
- three or more seeds for all headline comparisons,
- confidence intervals or bootstrap intervals,
- seven-dataset transfer,
- synthetic metric-side experiment,
- concise reproducibility statement,
- supplementary detailed ablations.

That version would make the paper considerably harder to reject on the grounds of leakage, benchmark specificity, or insufficient generalization.

---

# 24. Final Strategic Recommendation

The paper should be positioned as a **benchmark/evaluation and methodological diagnosis paper with a controlled ranking-model demonstration**, not primarily as a new architecture paper.

The strongest narrative is:

**Task definition**
→ target ranking per ligand

**Problem**
→ pooled AUC can measure a different property

**Diagnosis**
→ molecule-blind and ligand-level nulls reveal shortcuts

**Mechanism**
→ ligand recurrence and benchmark construction contribute to pooled-AUC inflation

**Intervention**
→ ranking-specific objective + balanced sampling + hard negatives

**Robustness**
→ multiple datasets + cold-ligand/cold-protein splits

**Boundary**
→ not every DTI benchmark exhibits the same failure

**Practical recommendation**
→ test null baselines and task-aligned ranking metrics before treating pooled AUC as evidence of target discrimination

This is the version of the story that should be optimized for JCIM.

---

# 25. Decision Rule for the LLM

When revising the manuscript, apply these rules:

1. **Never broaden a claim beyond the experiments.**
2. **Prefer "enzyme–substrate benchmarks studied here" over "DTI benchmarks" unless the evidence genuinely supports the broader statement.**
3. **Treat ligand recurrence as a central experimental variable, not merely a limitation.**
4. **Do not hide the BRENDA decoy construction.**
5. **Add ligand-only diagnostics before making causal statements about the shortcut.**
6. **Add cold-ligand evaluation if at all feasible.**
7. **Keep protein-only and ligand-only nulls conceptually separate.**
8. **Use matrix MRR/Hit@K as task-aligned primary metrics when the task is target ranking.**
9. **Keep pooled AUC for comparability, but do not let it dominate interpretation.**
10. **Do not present single-seed results as robust multi-seed evidence.**
11. **Do not oversell RankBind's architecture.**
12. **Use the negative Davis/KIBA results to define the boundary of the claim.**
13. **Make the reproducibility infrastructure visible.**
14. **If a new experiment contradicts the preferred story, report it and revise the interpretation rather than forcing the hypothesis.**
15. **The ultimate goal is not to prove that pooled AUC is universally bad; it is to show when pooled AUC fails to measure the scientific property the benchmark claims to measure.**

---

## Bottom line

Before JCIM submission, the highest-value improvements are:

1. **narrow the central claim,**
2. **elevate the ligand-leakage/decoy analysis,**
3. **add ligand-only diagnostics,**
4. **run cold-ligand and ideally double-cold experiments,**
5. **retain and emphasize the multi-dataset boundary evidence.**

If these changes are implemented cleanly, the paper's contribution becomes substantially more precise:

> **It is not merely a new DTI model. It is a controlled demonstration that benchmark metrics can reward the wrong property, together with practical diagnostics and a task-aligned training strategy for detecting and reducing that failure in enzyme–substrate prediction.**

---

# 26. Execution Status (handoff, last updated 2026-08-23)

This section records what has already been executed against the repo, so a
later LLM does not redo it or treat finished work as open.

## Done on 2026-08-23 (session 1)

1. **Diagnostics regenerated and verified** (all scripts rerun from
   scratch on the login node; venv `~/venvs/hieratombind`; module load
   `GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0`):
   - `python evaluation/decoy_leakage_probe.py`
     → rewrote `evaluation/DECOY_LEAKAGE_AUDIT.md` + `decoy_leakage_probe.csv`;
     confirmed molecule_only 0.887 / full 0.833 / protein_only 0.603,
     4,604 ligands, 1,417 pure-pos / 3,157 pure-neg (99.3%), 53.6% test
     ligands seen in train.
   - `python evaluation/null_baseline_table.py`
     → rewrote `evaluation/NULL_BASELINE.md` + `null_baseline_firstclass.csv`;
     confirmed null_lig_prior pooled AUC 0.915 on the full test split,
     prot_prior 0.500 (by construction).
   - `python evaluation/leakage_audit.py`
     → rewrote `evaluation/LEAKAGE_AUDIT.md`; BRENDA-200 mol-overlap
     53.6%, scaffold-overlap 71.5%, hard checks L1/L2 pass everywhere.
   - Note: 4/8 datasets are CAVEATS status due to L5 (identical-sequence
     twins across the split; e.g. 2 of 132 BRENDA-200 test proteins);
     L5b shows headline MRR moves by ≤ 0.0001 when twins are excluded —
     report as limitation, not as leakage.
2. **Manuscript `paper/scirep/main.tex` edited** (skill §3):
   - Results §2.1: new "molecule axis" paragraph quantifying the decoy
     construction (4,604/1,417/3,157, 99.3% label-pure, 53.6% recurrence,
     scaffold overlap 71.5%, linear probe 0.887/0.833 with matrix MRR
     0.029 = chance, lig_prior 0.915) — removed the "quantified in the
     Discussion" forward-reference.
   - Discussion limitation #2 shortened to a one-sentence pointer to §2.1.
   - Methods §4.1 data paragraph: role-pure ligand structure made explicit.
   - `paper/CLAIM_EVIDENCE_MATRIX.md` C4: ligand count corrected
     4,574 → 4,604 (matches the regenerated probe).
   - PDF rebuilt: `cd paper/scirep && module load texlive && make`
     → 13 pages, 0 LaTeX errors.
3. **Nothing committed to git** (repo convention: commit only on request).

## Still open (do not mark as done)

- §4 diagnostic table "Information source → Pooled AUC → Matrix MRR":
  all inputs exist (null_baseline_firstclass.csv, decoy_leakage_probe.csv,
  phase2_rankbind_multiseed.csv); only table assembly + placement remain.
- §5–7 cold-ligand / cold-protein / double-cold splits: NOT built. Needs
  new split definitions (extend `BRENDADataConfig` in
  `baselines/adapters/common.py`) + SLURM runs (`scripts/run_v5_rankbind.sh`
  per split; tag convention `v7_cold_*`). This is the highest-value open
  item and requires cluster time.
- §22 Must-have 1 (scope the claims): the abstract/intro/discussion still
  contain the unqualified "pooled AUC is not a meaningful success gate"
  wording in places; scoping pass not yet done.
- §22 Must-have 3 (cold-ligand evaluation): same as §5–7.
- §15 residue-attention de-emphasis: not yet decided/moved to
  supplementary.

## When continuing

1. `git status` to see the working-tree state; do not commit unless asked.
2. Re-verify numbers against the three CSVs above before citing them.
3. Any new manuscript edit → rebuild `paper/scirep/main.pdf` (and
   `paper/main.pdf` if that variant is kept in sync).
4. Append a short note to this section after finishing new work
   (date + what changed + verification command outputs).

## Done on 2026-08-23 (session 2) — §5–§8 executed on SLURM

1. **Cold splits built** (`baselines/adapters/common.py`):
   `get_ligand_split()` partitions canonical ligands
   (`substrate_smiles_canon`, same molecule identity as leakage_audit L3)
   at 0.15/0.15 seed 42; `get_double_cold_split()` is the product partition
   (protein axis identical to the canonical split, ligand axis as above;
   rows whose two axes land in different folds are DROPPED, ~46%).
   `v5_rankbind/data.py` dispatches on `data.split_mode ∈ {ligand,
   double_cold}` and every manifest now pins `test_lig_in_train_frac` /
   `test_prot_in_train_frac`. Verified: canonical 53.6% lig recurrence vs
   **0.0%** in both cold splits (double-cold also 0% protein).
2. **12 SLURM runs** (paula, jobs 27305378–89, tags `v7_cold_lig` /
   `v7_cold_both`, seeds {42,7,1337}, configs `cold_{lig,both}_{
   rankbind,bce}` extending default / abl_bce_only — matched protocol per
   §7). All healthy: 0 skipped batches, hard negs active from epoch 2.
3. **Null baselines per split** (§8): `null_baseline_table.py --split`
   regenerated all three; canonical values reproduce exactly
   (lig_prior 0.915 / prot_prior 0.500). New artifacts:
   `NULL_BASELINE_ligand.md`, `NULL_BASELINE_double_cold.md` (+CSVs).
   Mirror-image finding: under cold-ligand the LIGAND prior collapses to
   0.500 and the PROTEIN prior carries signal (0.655); under double-cold
   both collapse to exactly 0.500.
4. **§4 diagnostic table**: `evaluation/diagnostic_table.py` →
   `DIAGNOSTIC_TABLE.md` + `diagnostic_table.csv` (all rows read from
   existing CSVs, nothing hand-entered).
5. **Aggregation**: `scripts/aggregate_cold_splits.py` →
   `evaluation/COLD_SPLIT_SUMMARY.md`,
   `attractor_results/cold_split_runs.csv` (per-run),
   `attractor_results/cold_split_multiseed.csv` (3-seed mean ± SD).
   `runs_manifest.csv` refreshed via `collect_v5_runs.py`.
6. **Headline results (skill §6 outcomes)**:
   - *Outcome B confirmed — twice.* Cold-ligand: BCE control pooled AUC
     0.850 ± 0.018 at matrix MRR 0.028 ± 0.013 (= chance reference 0.028).
     Double-cold: BCE pooled AUC 0.808 ± 0.010 while BOTH null priors sit
     at exactly 0.500. The pooled-AUC/ranking dissociation does NOT require
     ligand identity recurrence — make this a major result and cite it
     against reviewer objection Q1/Q2.
   - *Outcome D partially:* RankBind keeps genuine ranking signal on both
     cold splits (cold-lig MRR 0.295 ± 0.064, H@10 0.565 ± 0.008;
     double-cold MRR 0.123 ± 0.092 — wide seed spread, small-n pool surface
     [~6 positive pairs], report honestly).
7. Nothing committed to git (repo convention).

## Still open after session 2 (do not mark as done)

(All items below were closed in session 3; see the session-3 note.)

- Manuscript placement: move §4 table + cold-split results into
  `paper/scirep/main.tex` as Results §2.4 ("Cold-ligand and cold-protein
  stress tests") with `COLD_SPLIT_SUMMARY.md` / `DIAGNOSTIC_TABLE.md` as
  number sources; update `CLAIM_EVIDENCE_MATRIX.md` (new evidence rows);
  rebuild PDF.
- §22 Must-have 1 (scope the claims): still open.
- §15 residue-attention de-emphasis: still open.
- Optional: more seeds under double_cold (seed spread there is wide).

## Done on 2026-08-23 (session 3) — manuscript integration + scoping

1. **Extra double-cold seeds** (optional item): jobs 27305529–32,
   configs `cold_both_{rankbind,bce}` × seeds {11, 202}, all COMPLETED.
   Double-cold aggregates are now five-seed: BCE pooled AUC
   **0.813 ± 0.010** at MRR **0.025 ± 0.006** / H@10 0.000; RankBind MRR
   **0.092 ± 0.078**, H@10 0.233 ± 0.253 — the extra seeds narrowed
   RankBind's double-cold advantage vs the earlier 3-seed reading
   (0.123±0.092). Reported honestly in §2.4 ("advantage narrows … widest
   seed noise we observe anywhere").
2. **Manuscript edits** (`paper/scirep/main.tex`, now 16 pp., 0 undefined
   refs / 0 overfull boxes):
   - New Results **§2.4 "Cold-ligand and double-cold splits"**
     (`\label{sec:cold}`) + **Table 4** (`tab:cold`: cold-ligand /
     double-cold / canonical side by side, nulls + BCE + RankBind).
     Outcome-B framing with explicit open-question sentence on what
     sustains residual pooled signal under double-cold.
   - New **Table 2** (`tab:sources`) in §2.1: information-source →
     pooled AUC → matrix MRR (skill §4), values from
     diagnostic_table.csv / paper-canonical three-seed protocol;
     ranking metrics marked "---" for degenerate tie-rule scorers.
   - Abstract + intro roadmap extended by the stress-test result;
     transfer → §2.5, recipe → §2.6; all stale hardcoded section refs
     fixed (incl. two converted to `\ref{sec:transfer}`/`\ref{sec:recipe}`).
   - **Claim-scoping pass (§22 Must-have 1)**: "on the enzyme–substrate
     benchmarks studied here" in intro conclusion + Discussion opener;
     recipe preamble scoped to benchmarks where marginals may dominate
     pooled metrics (skill §18 wording).
   - **Residue attention (§15, alternative option)**: section opens with
     an explicit "single-seed mechanistic analysis" disclaimer, asterisk
     defined in Table 3 caption, intro mention labeled single-seed.
   - Methods: new split-construction paragraph (canon-SMILES identity,
     product partition, drop rate, manifest overlap fractions); seeds
     paragraph covers stress-test seeds incl. double-cold {11,202};
     provenance names `aggregate_cold_splits.py`.
3. **`CLAIM_EVIDENCE_MATRIX.md`**: rows C15 (cold-ligand persistence),
   C16 (double-cold persistence, open-mechanism caveat),
   C17 (RankBind advantage survives, reduced magnitude) added.
4. Nothing committed to git.

### Verification

```bash
sacct -j 27305529,27305530,27305531,27305532   # all COMPLETED
python scripts/aggregate_cold_splits.py        # 16 runs / 4 config groups
cd paper/scirep && module load texlive && make # 16 pages, no warnings
grep tab:cold main.aux                         # Table 4, p.8
```

Nothing is left open from this skill's Must-have list. Remaining
post-submission polish (not required): sync `paper/main.tex` long variant,
cover letter per §2.