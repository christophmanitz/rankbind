# Phase-5 Cross-Dataset Probe, dataset scoping

**Date:** 2026-04-27
**Goal:** Find a realistic external enzyme-substrate dataset on which to
evaluate `RankBind` (trained on BRENDA) under distribution shift, a real
test of the Phase-2 + Stage-(b) thesis rather than another ablation.

The §10 risk-row in PLAN.md already pre-registered this as a Phase-5
deliverable. The decision rule below makes that concrete.

## 1. What we need

For the probe to be informative, the external dataset must satisfy:

1. Pair structure: (protein, small-molecule) tuples with a binary or
   thresholdable binding label. Continuous kcat is acceptable if we
   threshold (e.g. treat log10 ≥ -2 as a binder).
2. UniProt IDs: so we can reuse the existing `data/esm2_embeddings/`
   cache without recomputing PLM embeddings on thousands of new
   sequences.
3. SMILES strings: reusable through our ChemBERTa cache (one-shot
   `ensure_chemberta_cache` call on the deduped substrate set).
4. Non-overlap with BRENDA: at minimum a protein-level non-overlap
   check, ideally also a substrate-level one. Overlap is the
   Phase-1-equivalent of leakage and would silently inflate metrics.
5. License: redistributable or non-commercial use is fine for an
   academic publication; closed-source datasets are out.
6. Size: ≥ 500 protein-substrate pairs in the post-overlap-filter set,
   ≥ 100 unique proteins. Smaller and the n-too-thin problem from
   Phase 1 (per-ligand AUC, n=4) repeats.

## 2. Candidate datasets

### 2.1 ESP, Enzyme-Substrate Prediction (Kroll et al. 2023)

- **Source:** https://github.com/AlexanderKroll/ESP, paper in
  *Nature Communications* 2023.
- **Size:** ~280,000 protein-substrate pairs, binary label
  (substrate / non-substrate). Curated from KEGG, BRENDA, MetaCyc,
  Sabio-RK.
- **IDs:** UniProt + SMILES.
- **License:** open (MIT or similar; permissive academic use).
- **Pros:** largest, binary-label, cleanly aligned with our task
  framing. The authors explicitly designed protein-stratified train /
  test splits.
- **Cons:** high BRENDA overlap risk, KEGG and BRENDA are both source
  corpora. Without a per-pair UniProt-overlap filter, this is not a
  true cross-dataset probe. Required: filter out any pair where
  `uniprot ∈ BRENDA-train-proteins`.
- **Net:** first choice if the overlap filter still leaves ≥ 500 clean
  pairs.

### 2.2 TurNuP, Turnover Number Prediction (Kroll et al. 2023)

- **Source:** https://github.com/AlexanderKroll/TurNuP, paper in
  *Nature Communications* 2023.
- **Size:** ~15,000 (enzyme, substrate, kcat) tuples, cleaned from
  Sabio-RK + manual literature curation. Continuous kcat label.
- **IDs:** UniProt + SMILES.
- **License:** open.
- **Pros:** cleaner curation than DLKcat (the authors explicitly
  rebuilt to fix DLKcat's BRENDA-leakage issues). Continuous label
  threshold-able.
- **Cons:** much smaller than ESP. Overlap with BRENDA still present
  (Sabio-RK shares sources with BRENDA). Also, continuous kcat is not
  the same as a binding label, many high-kcat enzymes ARE BRENDA
  substrates already; the labels are correlated but not identical.
- **Net:** second choice. Use as a complementary probe alongside ESP
  if both yield clean post-filter sets.

### 2.3 DLKcat (Li et al. 2022)

- **Size:** ~17,000 (enzyme, substrate, kcat) tuples.
- **License:** open.
- **Cons:** heavy BRENDA overlap, DLKcat is largely a BRENDA re-export
  plus literature additions. Not a clean external probe by
  construction. The TurNuP paper specifically critiqued DLKcat's
  leakage issues.
- **Net:** reject. Not a meaningful generalisation test.

### 2.4 BindingDB

- **Source:** http://bindingdb.org
- **Size:** ~2.8M binding measurements across ~1.1M unique molecules
  and ~9k targets.
- **IDs:** UniProt + SMILES + InChI.
- **License:** free for academic use; redistributable with attribution.
- **Pros:** massive, broad; very different from BRENDA's enzyme-only
  scope (includes drug targets / receptors / channels too).
- **Cons:** heterogeneous label quality (Ki / Kd / IC50 from many
  source publications). Not enzyme-substrate-specific; using it would
  test "does RankBind generalise to all protein-ligand binding"
  rather than "does the enzyme-substrate ranking story transfer". The
  former is a different and broader claim that the architecture was
  not designed for.
- **Net:** out of scope for the primary probe. Could appear in a
  paper appendix as "we also tested on a non-enzyme corpus and found
  X" but should not be the headline external test.

### 2.5 DAVIS / KIBA (kinase-inhibitor)

- **Size:** DAVIS ~30k pairs (442 kinases × 68 inhibitors); KIBA
  similar.
- **License:** open, public.
- **Cons:** kinases are over-represented in BRENDA training data
  (kinase ECs 2.7.x are the biggest BRENDA EC class). Heavy overlap
  expected. Also, kinase-inhibitor is not enzyme-substrate
  semantically, inhibitors compete with substrates rather than bind
  as substrates.
- **Net:** reject for the primary probe; potentially useful for a
  small "did the paper-claim hold on a different bioactivity type"
  appendix.

### 2.6 ChEMBL bioactivity export

- **Size:** ~20M activity records.
- **License:** CC-BY-SA, permissive.
- **Cons:** heterogeneous (similar to BindingDB). Not enzyme-substrate
  framed.
- **Net:** reject as primary; same logic as BindingDB.

## 3. Comparison matrix

| Dataset | n_pairs | UniProt | SMILES | License | Enzyme-substrate? | BRENDA-overlap | Verdict |
|---|---:|---:|---:|---|---|---|---|
| ESP | ~280k | yes | yes | open | yes | high, filter required | Primary |
| TurNuP | ~15k | yes | yes | open | yes (kcat-thresh) | medium | Secondary |
| DLKcat | ~17k | yes | yes | open | yes | very high (BRENDA-derived) | Reject |
| BindingDB | ~2.8M | yes | yes | open | partial | unknown | Appendix only |
| DAVIS / KIBA | ~30k | yes | yes | open | partial (kinases) | high | Reject |
| ChEMBL | ~20M | yes | yes | open | partial | unknown | Appendix only |

## 4. Recommendation

Primary probe: ESP, with explicit BRENDA-overlap filtering.

Secondary probe: TurNuP, used as a smaller robustness check if ESP
turns out clean.

Implementation outline (to be ratified before any code):

1. Pull ESP pairs CSV from upstream repo. Verify UniProt + SMILES
   columns are present.
2. Build the non-overlap test set:
   `ESP_filtered = {(uni, smi) ∈ ESP : uni ∉ BRENDA_train_proteins}`.
   Drop also any (uni, smi) duplicate of BRENDA test pairs, even
   though the protein side is non-overlapping, a substrate-overlap
   could still inflate.
3. Verify ≥ 500 clean pairs and ≥ 100 unique proteins in the filtered
   set; if not, fall back to using TurNuP as the primary.
4. Build ESM2 + ChemBERTa caches for the new proteins / SMILES (this
   is the only potentially expensive step; ESM2 over a few thousand
   new sequences is ~30 min on a V100).
5. Run `eval.py` with the trained `default_v4` and `abl_attn_pool_v5b`
   checkpoints (3 seeds each, 6 total inference passes), score the
   new pairs.
6. Report:
   - Global AUC on the filtered ESP set (binary task, directly
     applicable).
   - Matrix MRR / H@K on a constructed `n × n` probe matrix from a
     stratified subset (mirror Phase-1 / Phase-2 evaluation
     geometry).
   - Per-class MRR on the polyhydroxy class identified in Stage-(c):
     does the lift specifically hold there, or is it uniform?

Cost estimate:
- Dataset acquisition + filter: ~half a day.
- ESM2 / ChemBERTa cache build: ~1 hour cluster time.
- Inference + report: ~half a day.
- Total: ~1.5 days, no model training needed. Stage (b) checkpoints
  are reused.

## 5. Decision-gate (probe to paper-claim)

The Phase-5 probe is a generalisation test, so the question is not
"does the lift exist?" (we already know it does on BRENDA) but
"does it transfer?". Concretely:

- Strong pass: matrix MRR or AUC lift of `abl_attn_pool` over
  `default_v4` is at least half of the BRENDA-internal lift (i.e.
  ≥ +0.05 absolute MRR or equivalent AUC). The Phase-2 + Stage-(b)
  story holds out-of-distribution.
- Partial pass: lift is in the right direction but small
  (+0.01 to +0.05). Reported honestly; paper claims robustness with a
  qualifier.
- Fail: no lift or reverse. The Phase-2 + Stage-(b) story is
  BRENDA-specific. Paper still publishable but with a
  "limitations: no out-of-distribution generalisation" section.

Each outcome is actionable and none invalidates the existing in-
distribution story.

## 6. Out of scope for this memo

- Implementation: this is a scoping document only. Concrete
  filter-script + eval-driver are the next task.
- AlphaFold/PDB requirements: not needed for the residue-level probe;
  ESM2 + ChemBERTa caches are sufficient.
- Hyper-parameter retraining on the external data: explicitly NOT
  done. The probe is zero-shot transfer of the BRENDA-trained model.

## 7. Pointers

- Memo: `evaluation/attractor_results/phase5_dataset_scoping.md` (this
  file).
- Upstream candidates:
  - ESP, github.com/AlexanderKroll/ESP
  - TurNuP, github.com/AlexanderKroll/TurNuP
- Relevant local context:
  - Trained checkpoints: `results/v5_rankbind/*_v4*/best_model.pt`,
    `results/v5_rankbind/*_v5b_*/best_model.pt`
  - BRENDA training proteins (used for the overlap filter): derived
    from `BRENDADataConfig.get_protein_split()` (seed=42), specifically
    the train-split UniProt set, n_train_proteins = 618 per
    `manifest.json`.
