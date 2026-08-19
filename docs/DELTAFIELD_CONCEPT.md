# DeltaField-RankBind concept note

A new model architecture for ligand-conditional binding-interaction
prediction, derived by transferring the difference-encoding idea of GMSF
(Zhao et al., *JCIM* 2026, "GMSF: Dual-Path Multimodal Framework … via
Difference Graph Encoding") to the RankBind setting. Designed and
novelty-checked by a 4-proposal design panel (2026-06-09).

---

## 1. The source idea (GMSF) and the transferable kernel

GMSF predicts enzyme EC numbers from reactions. Its key device: a chemical
reaction is a reactant-to-product transition; Atom-Atom Mapping (AAM) aligns
the same atom across the two states; the model encodes the per-atom feature
difference `d(v) = c_product(v) − c_reactant(v)`. The difference localises
the reaction center (where bonds break/form) and cancels the conserved
scaffold (whose difference is ≈ 0). A second GNN then propagates the
difference so it becomes a synergistic, pocket-scale signal.

**Transferable kernel:** represent a process by the per-element difference
between two aligned states; the difference localises the active region and
cancels the inert background; then propagate it.

GMSF ablation (what carries the weight): the difference module is the main
structural win (GIN 0.912 to 0.921); naive fusion of two unaligned modalities
causes negative transfer (GIN+BERT 0.906 < GIN alone). The alignment anchor
is what makes fusion help.

---

## 2. The transfer to binding

| GMSF (reaction)                       | DeltaField (binding)                                   |
|---------------------------------------|--------------------------------------------------------|
| reactant to product                   | unbound (free) to bound (coupled) state                |
| AAM (atom identity across states)     | identity of the same residue / ligand atom (free vs coupled) |
| difference `c_p − c_r`                | coupling-induced perturbation `H_coupled − H_free`     |
| reaction center                       | binding interface / pocket                             |
| conserved scaffold cancels (Δ≈0)      | ligand-agnostic protein identity cancels (Δ≈0)         |

**The creative leap.** GMSF computes the difference because both states
exist. Binding's two states are apo (unbound) to holo (bound), but no holo
structure exists at inference in RankBind. Rather than predict or hallucinate
a virtual holo (the discarded "CounterBind"/"DeltaDock" route, unsupervisable
without PDBbind), DeltaField realises both states inside one forward pass of
a single weight-shared network, toggling only the cross-modal attention
mask:

- **Free pass**, block-diagonal mask: residues attend to residues, atoms to
  atoms, no cross-modal edges. The ligand and protein never see each other,
  so this representation can encode at most ligand-agnostic protein identity,
  i.e. exactly the protein-prior shortcut.
- **Coupled pass**, same weights, cross edges on: ligand atoms and protein
  residues attend to each other.
- **Difference field** `Δ = H_coupled − H_free` per residue / per atom.

Any component invariant to the ligand's presence lives in the free pass and
cancels in the subtraction. What survives flows only through the
cross-attention keys/values built from the ligand, so the surviving signal is
ligand-conditional by the chain rule, not by a loss penalty.

---

## 3. Architecture (drops into the existing v5_rankbind stack)

Tensor shapes use the repo's dims: ESM2 per-residue `1280`, ChemBERTa
per-token `384`, projection `d = 256`.

1. **Inputs.** Protein (apo): frozen ESM2 per-residue `E_P ∈ [B, L≤1024, 1280]`
   + mask, already produced by `data.py:load_protein` in `attn_pool` mode.
   Ligand: frozen ChemBERTa per-token `E_L ∈ [B, A≤128, 384]` + mask.
   One-line data change: in `ensure_chemberta_cache`, also save
   `last_hidden_state` (minus CLS) into a `chemberta_token_cache/`, plus a
   token→RDKit-atom offset map (BPE subword → atom; subword-mean fallback).
2. **Projectors** (reuse `v5_rankbind.model` verbatim, applied along the
   token/residue axis): `u = ProteinProjector(E_P) ∈ [B,L,256]`,
   `v = LigandProjector(E_L) ∈ [B,A,256]`; add modality-type embeddings; stack
   `X0 = [u ; v] ∈ [B, L+A, 256]` with concatenated node mask.
3. **Weight-shared bipartite encoder** (the engine): `T=2` transformer blocks
   (4 heads, FFN 4d), run twice with the same weights, differing only in the
   attention mask (`A_free` block-diagonal vs `A_coup` cross-open), producing
   `H_free`, `H_coup ∈ [B, L+A, 256]`. Cache the coupled cross-attention
   weights `W_cross ∈ [B, heads, L, A]` for the contact map.
4. **Difference field** (load-bearing): `D = H_coup − H_free`, split to
   `d_res ∈ [B,L,256]`, `d_atom ∈ [B,A,256]`; scalar energies
   `e_res = ‖d_res‖₂ ∈ [B,L]`, `e_atom = ‖d_atom‖₂ ∈ [B,A]`.
5. **Difference-graph propagation** (GMSF "second GNN" graft, ablatable
   `prop_layers ∈ {0,1}`): `x'_node = Linear([H_free ; D])`; one more
   weight-shared layer under the coupled mask to get propagated `D̃`. Only
   `D̃` routes into the score; `H_free` is a context channel only.
6. **Additive perturbation-energy score head** (graft from "PField", mass
   conserving): difference-gated pocket weights `w_res = softmax_i(φ·D̃_res)`
   gated by `e_res` (analogous `w_atom`); signed contributions
   `c_res[i] = aᵀD̃_res[i]`, `c_atom[a] = bᵀD̃_atom[a]`;
   `score = b0 + Σ_i w_res[i]·c_res[i] + Σ_a w_atom[a]·c_atom[a]`.
   Every term is linear in `D̃`. No `aᵀH_free` / `g(P)` term ever reaches the
   scalar. If the ligand induces no field, `D̃→0` and `score→b0`. A single
   global constant identical for every protein, hence unable to rank.
7. **Outputs** (matching the `head(fL,gP)→scalar` contract): (a) scalar score
   for matrix-MRR ranking; (b) residue interaction map `e_res` / signed
   `c_res`; (c) atom map `e_atom` back-projected to SMILES atoms; (d) residue×atom
   contact map `C[i,a] = mean_heads W_cross[i,a] · e_res[i] · e_atom[a]`.
8. **Integration:** `head_type='deltafield'` in `RankBind.__init__`, exposing
   the same `score_pairs` / `score_triplet` signatures (train/eval/loss/metrics
   untouched). `refresh_scores` is refactored to pre-project `u(P)`, `v(L)`,
   and cache the ligand-independent free pass `H_free(P)` per protein (it
   never changes with the ligand); only the coupled pass + field run per
   (lig,prot) chunk, amortising the 200×200 matrix and the hard-neg refresh.
   Ship as `configs/abl_deltafield.json` extending `default.json`.

---

## 4. Why it is anti-shortcut by design (not by loss)

Three structural locks, tied to measured failure modes in this repo:

1. **The free pass is the shortcut, and it is subtracted out.** `H_free`
   encodes only ligand-agnostic protein identity, the exact signal that gives
   `null_prot_prior` its Gini ≈ 0.995 and the Phase-1 baselines their Top-10
   Jaccard 0.54-0.67. `Δ = H_coup − H_free` cancels it.
2. **No additive protein term reaches the scalar.** Remove the ligand, then
   `D̃→0` and `score→b0` (one global constant). A per-protein prior is
   literally unrepresentable in the score, the formal analog of GMSF's
   conserved-scaffold cancellation.
3. **Sparsity + TV locality + within-ligand margin** cap the protein's degrees
   of freedom (it cannot smear a global identity across all residues to fake
   conditionality), and hard negatives (same hot protein, competing ligands)
   force the field to differ across ligands or the margin is violated.

**CI-testable invariant (the sharpest novelty hook):** mask the cross-edges,
then `Δ == 0` exactly and `score == b0` for every protein. A unit test
asserts this; the audit metric is Top-10 Jaccard vs `null_prot_prior` staying
≈ 0. The guarantee is verifiable, not asserted.

---

## 5. Novelty & honest differentiation

Novel as a composition (the panel's prior-art checks could not find this
device in any DTI/PLI paper); the constituent pieces are borrowed and cited
as such. The paper must contrast head-on with:

- **TAPB** (Wang et al., *Nat. Commun.* 2025): same "target-prior bias", but
  removed via causal backdoor adjustment over a confounder dictionary.
  DeltaField removes it via representation-level subtraction (free pass).
- **FlowDock** (Morehead & Cheng): same apo-to-holo and affinity framing, but
  a separate head on a generated 3D complex. DeltaField's coupled pass is a
  counterfactual virtual-holo computed by masking, not hallucinated; the score
  is difference-only in embedding space, no 3D generation.
- **MONN / LaMPSite / ICAN / ArkDTA**: residue×atom cross-attention
  interaction maps. DeltaField borrows the map but its score is read only
  from the difference field, with the zero-field invariant; those models
  score from the coupled representation directly.

The single defensible novelty: a single weight-shared bipartite residue×atom
transformer run twice (free vs coupled mask), with the per-node coupled−free
difference as the sole scoring substrate and the interaction map, giving a
by-construction, CI-testable anti-shortcut guarantee for ligand-conditional
DTI ranking.

---

## 6. Dual output & the falsifiable interpretability claim

The interaction map is produced, not explained post-hoc:
- residue map `e_res` (plus signed `c_res`, the % of predicted affinity,
  mass-conserving), atom map `e_atom`, contact map `C[i,a]`.
- Validated by feeding `e_res` into the existing `evaluation/attn_annotation_scan.py`
  / pocket-overlap audit (within-protein ROC-AUC of a per-residue score vs the
  UniProt binding/active-site mask).
- Pre-registered go/no-go: flip the project's own v5b finding that attention
  avoids active sites (AUC 0.08-0.21) to > 0.5 (ideally > 0.7), i.e. show the
  difference field concentrates on the pocket where free attention avoided it.
  Reported honestly as a secondary metric; the ranking claim does not depend
  on it.

---

## 7. Training recipe (slots into v4)

- Primary loss unchanged: within-ligand `margin_loss` (k=4, m=1.0) on
  `TripletCollator` triplets with hard-neg mining (`hard_pool_size=50`),
  refreshed every 2-3 epochs (the field forward pass is heavier than a
  bilinear dot). Metric of record stays matrix-MRR / H@5 / H@10; 3 seeds
  {42,7,1337}; AdamW/cosine/bf16; frozen PLMs; early-stop on `val_global_auc`.
- Added field regularisers (small, ablatable):
  - `L_sparse = λ_s·(mean e_res + mean e_atom)`, localising the interface
    (needs λ annealing and a floor on total field energy to avoid collapse).
  - `L_tv = λ_tv·TV(e_res)` along the backbone, for a contiguous pocket, not
    speckle.
  - `L_neg = λ_neg·mean‖D̃(L, P⁻)‖²` over hard negatives; a non-binder must
    induce a near-zero field (trains the score-collapse property directly).
  - Coupling-edge dropout (p≈0.15), the GMSF AAM-noise transplant.
- Optional, pretraining-only (off the critical path): if PDBbind 4.5 Å
  contacts become available, add focal-BCE on `C[i,a]` and distil
  `ESM2(holo)−ESM2(apo)` into `D̃`. The model must pass margin-only on BRENDA
  with none of this. No apo/holo pairs exist locally.

---

## 8. Minimal first experiment (`tag v6_deltafield`)

Single-seed (42) on the existing BRENDA protein split, `abl_deltafield.json`,
one GPU via `scripts/run_v5_rankbind.sh`. Steps: (1) per-token ChemBERTa
cache + token-to-atom map; (2) implement `head_type='deltafield'`; (3)
refactor `refresh_scores` (cache free pass per protein); (4) train margin +
`L_sparse` + `L_neg` (λ_tv=0, contact OFF); (5) eval on the 200×200 matrix
with the unchanged `matrix_ranking_metrics` + `null_baselines`.

Three pre-registered pass/fail checks (priority order):
- (A) Ranking: matrix-MRR ≥ v4 bilinear default (0.326 ± 0.072) at matched
  capacity. The difference-only score must not cost ranking.
- (B) Anti-shortcut: Top-10 Jaccard vs `null_prot_prior` ≤ 0.11 (ideally ≈ 0)
  and the zero-field unit test passes exactly.
- (C) Interpretability (headline secondary): `e_res` binding/active-site
  within-protein ROC-AUC > 0.5, flipping the v5b 0.08-0.21.

(A)+(B) validate the core claim with zero new data; (C) is the
interaction-map dividend. Only then run 3 seeds + the matched-capacity head
ablation (deltafield vs bilinear).

---

## 9. Open risks

- Sparsity collapse (most likely failure): too-strong `L_sparse`/`L_neg`
  drives `Δ→0` everywhere and MRR to chance. Mitigate with λ annealing + a
  field-energy floor; monitor field energy in `train_log.jsonl`.
- Free-pass leakage via the `[H_free; D]` re-injection in the propagation
  layer. If `H_free` reaches the scalar, the prior re-enters. Route only `D̃`
  to the score; gate on the Jaccard audit; ablate the channel if it creeps up.
- Inherited pocket anti-correlation. ESM2 is sequence-trained; the field may
  track biophysics (hydrophobicity) rather than the interface even after
  subtraction (cf. the v5b finding). Pocket-overlap is a gating experiment.
- Compute. The coupled pass + L×A cross-attention over B·k negatives is far
  heavier than a bilinear dot. Free-pass caching per protein + entmax/top-k
  truncation are load-bearing; may need single-head + gradient checkpointing.
- ChemBERTa BPE ≠ heavy atoms. The atom-level map is approximate; frame as
  coarse, validate against RDKit substructure.
- Novelty defense. Without the head-on TAPB / MONN / LaMPSite contrast a
  reviewer will call it derivative. Lead with the concrete null-baseline
  failure mode and the CI-tested zero-field theorem, not the abstract
  "shortcut" label.

---

*Provenance:* design panel `wf_b9969c42-7e2` (4 proposals: CounterBind,
CoupliNet, DeltaDock, PField, novelty-checked and synthesised). Backbone =
CoupliNet (free-vs-coupled difference); grafts from PField (additive score),
CounterBind (sparsity/TV + virtual-holo narrative), GMSF (difference
propagation + noise injection), DeltaDock (optional PDBbind distillation).
