# ResidueOnlyBind to RankBind: Development Plan

Goal: we evolve the current ResidueOnlyBind model into a publishable architecture that addresses the protein-attractor bias problem with a ranking-normalised, granularity-adaptive framework.

Paper thesis: naive dual-encoder models for protein-ligand binding prediction suffer from protein-attractor bias, a failure mode where structurally promiscuous proteins dominate top-k predictions regardless of the query ligand. We propose RankBind, a ranking-normalised model that combines pretrained embeddings with structure-aware GNNs and adaptive atom-level re-integration, and we show that this failure mode is widespread across existing architectures.

Core design principle: pretrained encoders (ESM2, Uni-Mol) provide rich node and edge features for the GNN inputs. The GNNs remain the core architecture and propagate these features over 3D structural graphs (Cα contacts for proteins, atom bonds for ligands). This is the key distinction from pure fine-tuning approaches: ESM2 alone has no 3D structure awareness, and Uni-Mol alone has no protein-ligand interaction modelling. The GNNs add structural reasoning on top of pretrained chemical and biological knowledge.

---

## Phase Overview

| Phase | Title | Duration | Key Output |
|-------|-------|----------|------------|
| 1 | Attractor Bias Diagnosis | 3-4 wk | Proof that bias is universal across architectures |
| 2 | Pretrained GNN Features | 2-3 wk | ESM2/Uni-Mol as node features; GNN stays |
| 3 | Ranking-Based Objective | 3-4 wk | Anti-attractor loss + protein-grouped batching |
| 4 | Adaptive Atom Gating | 3-4 wk | Confidence-gated atom module for hard cases |
| 5 | Validation & Paper | 4-6 wk | External test sets + manuscript |

---

## Phase 1: Demonstrate Attractor Bias in Existing Models

Why first: we start here because showing that the attractor problem is not specific to our model, but a widespread failure mode across architectures, gives the paper its central scientific contribution. It turns a limitation into a research question. It also provides the reference metrics (ROC-AUC, Hit@k) on our BRENDA dataset before any architectural changes, and it informs the design of the ranking loss in Phase 3.

### 1.1 Select Baseline Models

Two tiers of baselines: established references (widely cited, expected in any comparison) and current SOTA (recent, strong performers that represent the field's frontier).

#### Tier 1 - Established References

| Model | Year | Type | Why include |
|-------|------|------|-------------|
| DeepDTA | 2018 | CNN+CNN, sequence-only | Most cited DTI baseline; no structural info; known to overfit on PDBbind due to data leakage |
| GraphDTA | 2020 | GNN+CNN | GNN on ligand, CNN on protein sequence; simple dual-encoder reference |
| MolTrans | 2021 | Transformer, interaction fingerprints | Interaction-focused architecture with attention; different inductive bias |

#### Tier 2 - Current SOTA (2023-2026)

| Model | Year | Type | Why include |
|-------|------|------|-------------|
| DrugBAN | 2023 | Bilinear attention network (GCN + 1D-CNN + BAN) | Nat Mach Intell; explicit pairwise local interactions via bilinear attention; domain adaptation for OOD; closest to our bilinear fusion design, key comparison for attractor analysis |
| GIGN | 2023 | Geometric interaction GNN | J Phys Chem Lett; models protein-ligand interactions as a geometric graph with distance-aware edges; structure-based, strong on PDBbind |
| DualBind (NVIDIA) | 2024 | SE(3)-invariant FANN + dual loss (MSE + DSM) | Combines supervised affinity loss with unsupervised denoising score matching; model-agnostic framework; code on GitHub (`NVIDIA-Digital-Bio/dualbind`) |
| DualBind (Lin et al.) | 2026 | Adaptive GNN + structure-aware transformer | Expert Syst Appl; dual-module design with adaptive GNN, very close to our architectural space; most recent direct competitor |
| GEMS | 2024 | GNN + ESM2 + CleanSplit | ETH; explicitly addresses data leakage in PDBbind with leak-proof splits; uses protein language model integration; methodologically rigorous comparison |

#### 1.1.1 Implementation Priority
1. DrugBAN: code available on GitHub (`peizhenbai/DrugBAN`), well-documented
2. GIGN: code available, standard PyG-based implementation
3. GEMS: code available on GitHub (`camlab-ethz/GEMS`), uses ESM2
4. DualBind (NVIDIA): code on GitHub (`NVIDIA-Digital-Bio/dualbind`), may require adaptation
5. DualBind (Lin et al.): check if code is available; if not, cite results on comparable benchmarks
6. DeepDTA / GraphDTA / MolTrans: well-established codebases, quick to set up

### 1.2 Attractor Diagnosis Protocol

We design a standardised evaluation protocol that exposes attractor bias:

1. Response map construction: for each model, score every (ligand, protein) pair in the validation set and turn the scores into a heatmap matrix
2. Attractor score: for each protein p, compute `attractor(p) = fraction of ligands for which p is top-1 predicted partner`. A uniform model would give `attractor(p) = 1/N_proteins` for all p
3. Gini coefficient of attractor scores: measures concentration; Gini = 0 means uniform, Gini approaches 1 means a few proteins dominate
4. Rank displacement: for each ligand, measure `rank(true_protein) - rank(true_protein | excluding attractors)`; this quantifies how much the attractor shifts the true partner's rank
5. Score variance per ligand: low variance means the model predicts similar scores for all proteins (no discrimination), high variance means it is selective

### 1.3 Expected Findings

- Hypothesis: all naive dual-encoder models exhibit attractor bias to varying degrees
- Prediction by architecture type:
  - Pure dual-encoders (DeepDTA, GraphDTA): strongest attractor bias, no explicit interaction modelling
  - Bilinear attention (DrugBAN): moderate bias; BAN captures local interactions but still uses separate encoders
  - Geometric interaction (GIGN): potentially less bias, direct protein-ligand contact modelling
  - Dual-loss (DualBind NVIDIA): unclear; the DSM loss may implicitly regularise against attractors
- The response map heatmaps should show vertical bright bands for all models, not just ours
- Models with explicit cross-attention (DrugBAN, GIGN) may show less attractor bias, which supports our argument for ranking-based correction

### 1.4 Presentation

- Key figure: side-by-side response maps for 6+ models on the same dataset (our current v4 included)
- Key table: attractor Gini coefficient, Hit@k, rank displacement, score variance for each model
- Narrative: this section forms the problem statement of the paper and justifies why RankBind is needed

### 1.5 Dataset Suitability Analysis

Before running all baselines, we verify that the BRENDA hydrolase dataset is well-suited for exposing attractor bias. A dataset where attractor bias cannot manifest (for example all proteins equally represented and structurally distinct) would not be informative. We run the following checks:

- Protein representation imbalance: compute the distribution of ligands per protein. The current dataset has extreme skew (Protein 22: 38 ligands vs. several with 1-2). This imbalance is a prerequisite for attractors, since proteins with many training examples have more opportunity to develop inflated baselines. Quantify with the Gini coefficient over ligand counts.
- Structural similarity between proteins: compute pairwise sequence identity (for example via MMseqs2) and structural similarity (TM-score via TM-align or Foldseek) across all ~400 proteins. High similarity clusters mark regions where attractor bias is most likely, because models will struggle to distinguish proteins within the same structural cluster. Visualise as a dendrogram or similarity heatmap.
- EC sub-class overlap: check how many proteins share the same EC sub-class (for example multiple EC 3.1.1.* esterases). Proteins within the same EC group are functionally similar and more likely to become attractors for each other's ligands.
- Ligand chemical diversity: compute the Tanimoto similarity matrix across all ligands. If many ligands are chemically similar, the model has less discriminative signal to assign them to different proteins, which amplifies attractor effects.
- Minimum viable dataset properties: define thresholds, for example at least 20 proteins with at least 3 ligands each for meaningful attractor analysis. Check whether the validation split alone meets these criteria, or whether cross-validation is needed.
- Comparison to standard benchmarks: where possible, run the same suitability checks on PDBbind or BindingDB subsets to contextualise whether our dataset is more or less prone to attractor bias than typical benchmarks.

If the dataset has insufficient protein diversity for a convincing attractor analysis, we consider expanding to additional EC classes (beyond EC 3.*) or supplementing with BindingDB entries before training the baselines.

### Deliverables
- [ ] Training scripts for 5-7 baselines on BRENDA hydrolase data (same splits, same decoys)
- [ ] `attractor_diagnosis.py`: standardised evaluation script producing all metrics and heatmaps
- [ ] `dataset_suitability.py`: protein similarity, ligand diversity, and representation balance analysis
- [ ] Response map visualisations for each model
- [ ] Comparison table with attractor metrics and standard binding prediction metrics

---

## Phase 2: Pretrained Embeddings as GNN Input Features

Why second: Phase 1 provides reference metrics and confirms the attractor bias is real and widespread. Now we upgrade the input features while keeping the GNN architecture. ESM2 (1280-dim per residue) replaces the 33-dim handcrafted protein node features; the GNN propagates them over the 3D Cα graph. This is the foundation for the ranking loss in Phase 3.

Key principle: GNNs stay, only input features change.

### 2.1 Protein Side: ESM2 to ProteinGraphTransformer Node Features

ESM2 is a protein language model pretrained on ~250M sequences. It produces 1280-dim per-residue embeddings that capture evolutionary and biochemical context, but it has no 3D structural awareness. The ProteinGraphTransformer adds exactly this: it propagates ESM2 features over the Cα contact graph (8 Å cutoff), so the model learns structure-dependent residue interactions that ESM2 alone cannot capture.

#### 2.1.1 Precompute ESM2 Residue Embeddings
- Use `esm2_t33_650M_UR50D` (650M params, 1280-dim output per residue)
- For each UniProt ID in the dataset, extract per-residue embeddings
- Store as `{uniprot_id}_esm2.pt` tensors (shape: `[N_residues, 1280]`)
- Script: `precompute_esm2.py`, batch processing on GPU, roughly 2-3 hours for ~400 proteins on A30
- Fallback: if the sequence exceeds 1024 residues, use a sliding window with overlap and average

#### 2.1.2 Modify ProteinGraphTransformer Input
- Node features: replace 33-dim handcrafted with 1280-dim ESM2 embeddings
  - Input projection: `Linear(1280, 128) + LayerNorm` (replaces `Linear(33, 128)`)
  - The 4-layer TransformerConv architecture is unchanged
- Edge features: keep the existing 7-dim structural edge features (Cα distance, sequence separation, contact flags, 3D direction); these encode 3D geometry that ESM2 cannot provide
- Ablation variant: concatenate ESM2 (1280) + handcrafted (33) to get 1313-dim and let the model learn which signals are complementary
- ESM2 backbone is frozen, only the projection layer trains

#### 2.1.3 Why GNN on Top of ESM2 Matters
This is a testable claim and a key argument for the paper:
- ESM2-only baseline: pool ESM2 residue embeddings (mean/attention) into a protein vector, no GNN. This captures sequence-level information but misses 3D contacts.
- ESM2 + GNN (our approach): ESM2 residue embeddings propagated over the Cα contact graph. This captures both sequence context and 3D spatial relationships.
- Expected: ESM2+GNN beats ESM2-only, because proteins with similar sequences but different folds (or the reverse) will be better discriminated. This justifies keeping the GNN as the core architecture.

### 2.2 Ligand Side: Enriched Atom Features to LigandGraphTransformer

The ligand GNN stays as-is architecturally. The question is whether to enrich its input features.

#### 2.2.1 Current State Analysis
- The current 25-dim atom features (element, hybridisation, charge, aromaticity, ring) are standard RDKit descriptors
- The v4 failures (organophosphates, phenylpropanoids) are primarily protein-side discrimination problems, not ligand encoding failures
- The ligand GNN converges well and gradient norms are balanced, so it is not the bottleneck

#### 2.2.2 Options (in order of priority)

Option A, keep 25-dim features (recommended start):
- Least effort, strongest ablation baseline
- If the Phase 3 ranking loss solves the attractor bias, ligand features may not need improvement
- Defensible: at this data scale (~9.6k pairs), the ligand GNN learns effective representations from standard features

Option B, Uni-Mol atom embeddings as node features:
- Use Uni-Mol's pretrained molecular encoder (SE(3)-equivariant transformer, pretrained on 209M conformations) to extract per-atom embeddings (512-dim)
- Replace the 25-dim node features with 512-dim Uni-Mol embeddings in the LigandGraphTransformer
- Uni-Mol backbone is frozen, only `Linear(512, 128)` projection trains
- The ligand GNN still propagates these features over the molecular bond graph
- Uni-Mol provides atom-level features, but the GNN adds bond-topology reasoning and interaction with the protein encoder via the fusion module

Option C, hybrid enrichment:
- Concatenate original 25-dim + Morgan fingerprint bits (2048 to 64 via learned projection) to get 89-dim
- Lightweight, no additional pretrained model needed
- Morgan FP captures substructure patterns that atom-level features miss

#### 2.2.3 Recommended Strategy
We start with Option A (unchanged). After Phase 3, if ligand-side encoding is still a bottleneck (visible in case studies), we add Option B as an ablation. This keeps the experimental matrix manageable.

### 2.3 Validation Protocol

| Experiment | Protein features | Ligand features | Purpose |
|-----------|-----------------|----------------|---------|
| Baseline (v4) | 33-dim handcrafted | 25-dim handcrafted | Reference |
| ESM2-only (no GNN) | ESM2 1280-dim, mean-pooled | 25-dim handcrafted | Justify GNN |
| ESM2 + GNN | ESM2 1280-dim into GNN | 25-dim handcrafted | Main model |
| ESM2 concat | ESM2 + handcrafted (1313) into GNN | 25-dim handcrafted | Feature complementarity |
| ESM2 + GNN + Uni-Mol | ESM2 into GNN | Uni-Mol 512-dim into GNN | Full pretrained |

For each run we track:
- ROC-AUC, discrimination accuracy, average precision
- Per-protein Hit@k (especially for previously failed proteins like Protein 22)
- Attractor metrics (does ESM2 alone fix the bias, or is Phase 3 still needed?)

### Deliverables
- [ ] `precompute_esm2.py`: ESM2 embedding extraction
- [ ] `precompute_unimol.py`: Uni-Mol embedding extraction (for later use)
- [ ] Modified `model.py` with configurable input feature source (handcrafted / ESM2 / concat)
- [ ] Training runs for all 5 experiments in the validation table
- [ ] Comparison table with metrics

---

## Phase 3: Ranking-Based Objective (Core Novelty)

Why: the attractor bias is fundamentally a ranking problem. The model produces correct absolute scores but wrong relative rankings because some proteins have inflated baselines. A ranking-based loss directly optimises for the correct ordering.

### 3.1 Replace Global BCE with Protein-Normalised Ranking

The key architectural change: instead of predicting `p(bind | protein, ligand)` with a global sigmoid, we predict the relative rank of each ligand within a protein's ligand pool.

Option A, ListMLE loss:
- For each protein in the batch, collect all ligands paired with it (true binders + decoys)
- Compute binding scores for all pairs
- Apply ListMLE to maximise the likelihood of the correct ranking (true binders ranked above decoys)
- This requires protein-grouped batching (all ligands for protein p in the same batch)

Option B, protein-normalised contrastive loss:
- For each ligand, score it against all proteins in the batch
- Normalise scores per protein: `s_norm(l, p) = (s(l, p) - μ_p) / σ_p` where μ_p and σ_p are the mean and std of protein p's scores across all ligands in the batch
- Apply InfoNCE on the normalised scores
- This removes the protein-level baseline that creates attractors

Option C, hybrid (recommended):
- Keep BCE for absolute binding classification (the model still needs to distinguish binders from non-binders)
- Add a ranking loss as an additional objective that specifically penalises attractor behaviour
- Ramp in its weight during Phase 2/3 of the curriculum, similar to the regression ramp

### 3.2 Protein-Grouped Batching

- Current batching: random sampling of (ligand, protein) pairs with batch size 4
- Required: mini-batches that contain multiple ligands per protein AND multiple proteins per ligand
- Implementation: `ProteinGroupedSampler` that constructs batches of 4-8 proteins × 4-8 ligands each, forming a 2D scoring matrix per batch
- This is necessary so the ranking loss has enough comparisons

### 3.3 Anti-Attractor Regularisation

An explicit regularisation term that penalises attractor behaviour:

```
L_anti_attractor = Σ_p max(0, attractor(p) - 1/N_proteins)²
```

where `attractor(p)` is the fraction of ligands in the batch for which protein p has the highest score. This directly penalises any protein that captures more than its fair share of top-1 predictions.

### 3.4 Validation

- Primary metric: Hit@k improvement, especially for previously attractor-dominated ligands
- Track the attractor Gini coefficient over training epochs, it should decrease
- Track per-protein score distributions, attractors should flatten
- Compare with Phase 1 baselines to show that the ranking loss specifically addresses the attractor problem

### Deliverables
- [ ] `losses_v2.py` with ListMLE / normalised contrastive / anti-attractor terms
- [ ] `sampler.py` with ProteinGroupedSampler
- [ ] Modified training loop with ranking-aware batching
- [ ] Ablation: BCE-only vs. BCE+ranking vs. ranking-only
- [ ] Attractor Gini coefficient curves over training

---

## Phase 4: Adaptive Atom-Level Gating (Architectural Novelty)

Why: the case studies show that residue-level fails for specific substrate classes (organophosphates, phenylpropanoids) where binding chemistry requires atom-level resolution. Instead of always using atom-level (expensive, hurts generalisation at small data scale) or never using it (misses fine-grained chemistry), we let the model decide per sample.

### 4.1 Confidence-Gated Atom Module

Architecture:

```
                    ┌─────────────────────────────────┐
                    │   ResidueOnlyBind (Phase 1–3)   │
                    │   → binding score s_res         │
                    │   → confidence c = |s_res - 0.5|│
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  c > threshold τ ?  │
                    │  (learned, per-sample)│
                    └───┬─────────────┬───┘
                     YES│             │NO
                        │             │
              ┌─────────▼──┐   ┌──────▼──────────────┐
              │ Use s_res  │   │ Activate AtomModule  │
              │ (skip atom)│   │ → s_atom             │
              └────────────┘   │ → s_final = gate·s_atom│
                               │   + (1-gate)·s_res   │
                               └──────────────────────┘
```

- The AtomModule is a lightweight 2-layer GNN on the atom graph of the top-K residues (from attention pool weights, not a separate selector)
- The gate is a learned scalar per sample, trained end-to-end
- During training the gate is applied softly (sigmoid); during inference it can be hard-thresholded for efficiency
- Key difference from v3: the atom module is only activated when the residue-level model is uncertain, not for every sample

### 4.2 Training Strategy

- Phases 1-3 of the curriculum: train ResidueOnlyBind as before (atom module frozen/inactive)
- Phase 4 (new): unfreeze the atom module and add the atom-gating loss
- The atom module receives gradients only from samples where the gate is open
- Regularisation: L1 penalty on gate openness to encourage sparsity (use atom-level only when truly needed)

### 4.3 Validation

- Track the gate activation rate across substrate classes; we expect high activation for organophosphates, low for well-resolved substrates
- Compare: ResidueOnly vs. Always-Atom vs. Adaptive-Atom
- Compute efficiency metrics: FLOPs per sample with and without atom activation

### Deliverables
- [ ] `atom_module.py`: lightweight atom GNN
- [ ] Modified `model.py` with confidence-gated atom pathway
- [ ] Extended curriculum (5 phases)
- [ ] Analysis: gate activation rate by EC sub-class and substrate type
- [ ] Ablation: adaptive vs. always-on vs. off

---

## Phase 5: External Validation & Paper Writing

### 5.1 External Test Sets

- BindingDB hydrolases: extract EC 3.* entries from BindingDB that do not overlap with the BRENDA training set (match by UniProt ID)
- PDBbind refined set: use protein-ligand complexes with experimental binding affinity, filtered to hydrolases
- LP-PDBBind (Leak-Proof): use the cleaned, leak-proof PDBBind splits from Li et al. (2026, J Phys Chem B) for a rigorous generalisation assessment
- Time-split validation: hold out BRENDA entries added after a certain date (if timestamps are available)

### 5.2 Paper Structure

```
1. Introduction
   - Protein–ligand binding prediction problem
   - Dual-encoder architectures and their limitations
   - Contribution: identification of attractor bias + RankBind solution

2. The Protein-Attractor Bias Problem
   - Formal definition of attractor bias
   - Demonstration across 6+ existing architectures (Phase 1 results)
   - Analysis: why dual-encoders are structurally prone to this failure

3. RankBind: Method
   3.1 Pretrained embeddings as GNN input features (ESM2 for protein, Uni-Mol for ligand)
       - Why GNN on top of pretrained embeddings: structure-aware propagation
       - ESM2-only vs. ESM2+GNN ablation
   3.2 Ranking-normalised training objective
   3.3 Adaptive atom-level gating
   3.4 Curriculum training

4. Experiments
   4.1 Dataset (BRENDA hydrolases + external test sets)
   4.2 Baselines: DeepDTA, GraphDTA, MolTrans, DrugBAN, GIGN,
       DualBind (NVIDIA), DualBind (Lin et al.), GEMS
   4.3 Evaluation protocol: standard metrics + attractor diagnosis
   4.4 Main results (ROC-AUC, Hit@k, attractor Gini)
   4.5 Ablation studies (feature source, ranking loss, atom gating)
   4.6 Case studies (attractor resolution, atom gating activation)

5. Discussion
   - When does atom-level matter?
   - Pretrained features vs. learned features: the role of the GNN
   - Scaling behaviour and data efficiency
   - Limitations

6. Conclusion
```

### 5.3 Target Venues

| Venue | Deadline (typical) | Fit |
|-------|-------------------|-----|
| *Bioinformatics* (Oxford) | Rolling | Strong fit, methods + bio validation |
| *Briefings in Bioinformatics* | Rolling | Good for the broader narrative |
| *NeurIPS ML4Bio Workshop* | ~Sep 2026 | Good for early visibility |
| *ICML CompBio Workshop* | ~May 2026 | Tight but possible if Phases 1-3 done |
| *ICLR 2027* | ~Oct 2026 | Ambitious; needs strong baselines |

---

## Timeline Estimate

| Phase | Duration | Dependencies | Key risk |
|-------|----------|-------------|----------|
| Phase 1 (Attractor diagnosis) | 3-4 weeks | Baseline codebases available | Some baselines may not reproduce; DualBind (Lin) code may not be public |
| Phase 2 (Pretrained features) | 2-3 weeks | ESM2 + Uni-Mol model download, GPU access | Large embedding files (~5GB); Uni-Mol env setup |
| Phase 3 (Ranking loss) | 3-4 weeks | Phase 2 complete | Protein-grouped batching may need larger GPU memory |
| Phase 4 (Adaptive atom) | 3-4 weeks | Phase 3 complete | Gating may collapse to always-on or always-off |
| Phase 5 (Validation + paper) | 4-6 weeks | Phases 1-4 complete | External datasets may have limited hydrolase coverage |
| Total | ~15-21 weeks | | |

Phases 1 and 2 can run in parallel: while baselines train, we precompute ESM2 embeddings. Phase 3 depends on Phase 2 (pretrained features). Phase 4 depends on Phase 3 (ranking objective). Phase 5 writing starts early; experiments complete last.

---

## File Structure (Planned)

```
project/
├── data/
│   ├── esm2_embeddings/          # Phase 2 — per-residue [N_res, 1280]
│   ├── unimol_embeddings/        # Phase 2 — per-atom [N_atoms, 512]
│   ├── processed_hieratom/       # existing
│   └── external_testsets/        # Phase 5
├── models/
│   ├── model_v4_residue_only.py  # existing (baseline)
│   ├── model_v5_pretrained.py    # Phase 2 — ESM2/Uni-Mol as GNN input features
│   ├── model_v6_rankbind.py      # Phase 3 — + ranking loss
│   └── model_v7_adaptive.py      # Phase 4 — + adaptive atom gating
├── baselines/                    # Phase 1
│   ├── deepdta/
│   ├── graphdta/
│   ├── moltrans/
│   ├── drugban/
│   ├── gign/
│   ├── dualbind_nvidia/
│   ├── dualbind_lin/
│   └── gems/
├── evaluation/
│   ├── attractor_diagnosis.py    # Phase 1 — Gini, rank displacement, response maps
│   ├── response_map.py           # existing
│   └── external_eval.py          # Phase 5
├── training/
│   ├── losses_v2.py              # Phase 3 — ListMLE, normalised contrastive, anti-attractor
│   ├── sampler.py                # Phase 3 — ProteinGroupedSampler
│   └── train_rankbind.py         # Phase 3
├── scripts/
│   ├── precompute_esm2.py        # Phase 2
│   ├── precompute_unimol.py      # Phase 2
│   └── run_baselines.sh          # Phase 1
└── paper/
    ├── figures/
    └── rankbind_draft.tex         # Phase 5
```
