# RankBind v7: Structure-aware GNN (3D protein and 3D ligand)

Implementation plan. Synthesised by a 5-proposal design panel with adversarial
critique (2026-06-12), then extended per the explicit user requirement that
both the protein and the ligand be processed in 3D. The winning backbone is
GearBind-v7: a relational residue contact-graph GNN over AlphaFold structures,
plus an RDKit 3D-conformer atom-graph GNN for the ligand, both feeding the v6
DeltaField bipartite difference-field head verbatim, so the anti-shortcut
zero-field theorem is inherited unchanged.

This supersedes the 2D-ligand choice of the raw panel synthesis; the ligand now
carries genuine 3D geometry, symmetric to the protein branch. How the two 3D
structures are coupled is a deliberate, staged decision (§1a). Stage 1
(pharmacophore/shape complementarity via dense cross-attention) ships now;
Stage 2 (3D pocket/pharmacophore-gated coupling, no pose) is a pre-registered
P6 ablation.

## Why v7 (where v4 and v6 stand)

| Model | matrix-MRR | Hit@10 | global AUC | anti-shortcut (Top-10 Jaccard / gini-res) | pocket-ROC |
|---|---|---:|---:|---:|---|---:|
| v4 bilinear (best ranker) | 0.326 ± 0.072 | 0.755 | 0.634 | ~0 / neg (pass) | n/a |
| v6 DeltaField | 0.206 (fail) | 0.559 | 0.608 | 0.053 / -0.139 (pass) | 0.50 (fail) |
| v7 GearBind (target) | > 0.326 | ≥ 0.755 | > 0.634 | ~0.05 / neg + zero-field theorem | > 0.5 (target 0.7) |

v6 proved the anti-shortcut idea (a CI-testable theorem: mask cross edges, then
the difference field `D≡0` and `score≡b0`, so a per-protein prior is literally
unrepresentable in the score) but lost ranking and produced a chance-level
interaction map. v7 keeps that theorem byte-for-byte and bets that 3D geometry
plus explicit graph message passing restores ranking and lifts interpretability.

## Verified ground truth (checked against the live repo, 2026-06-12)

- Protein 3D is already local, no download for the headline run. 9,912
  AlphaFold v6 PDBs at
  `reactionDataFiltering/data/raw/brenda_sabio_2026-04-29/structures/AF-<uniprot>-F1-model_v6.pdb`,
  exactly 1:1 with the 9,912 ESM2 embeddings, so default-set coverage is 100%.
- Residue alignment is trivial 1:1. For sampled proteins
  `AF-residues == ESM2-length == sequences.csv-length` exactly, and the AF
  sequence is byte-identical to the ESM2 sequence. The structure graph indexes
  the same residues as the embedding; only the same ≤1024 N-terminus crop must
  be applied.
- Ligand 3D is cheap and reliable. 29,786 unique substrate SMILES; RDKit
  ETKDGv3 (with random-coords fallback) embeds 99% (1/120 sample failed),
  heavy-atom count median 20, max 115 (below the 128 cap). ~118 min single-core
  for all of them, a one-time, parallelisable precompute.
- Stack: torch 2.8.0+cu128, `torch_geometric` 2.7.0, RDKit 2026.03.1, Bio
  1.87 import; `torch_scatter`/`torch_sparse`/`torch_cluster` are absent, so
  ops that hard-require them must be avoided; single Tesla V100-PCIE-32GB.
- FIELD contract intact: `forward_field` returns a dict, `score_pairs_field`,
  `score_triplet_field`, bias-free readout `b0 + Σ w·c` (`model.py:288`),
  `_force_no_coupling` (`model.py:258`), zero-field test
  (`tests/test_deltafield.py:44`), capacity assert `550_000 < n < 720_000`
  (`tests/test_deltafield.py:126`).
- v6 numbers reproduced: MRR 0.2059, Jaccard 0.0526, gini_residual -0.1394.
  v6 used `cross_protein_implicit` negatives, not hard mining, so v7 is the
  first run to stress the hard-neg + deltafield path.

---

## 1. Recommended architecture

A relational residue contact-graph GNN over frozen ESM2 per-residue node
features produces geometry-aware residue states `U[B,L,d]`. An RDKit 3D
conformer atom-graph GNN over heavy atoms produces geometry-aware atom states
`V[B,A,d]`. Both feed the v6 DeltaFieldHead verbatim (free pass = no cross
edges; coupled pass = cross edges on; `D = H_coupled - H_free`; score is read
only from `D` via the existing bias-free `b0 + Σ w_res·c_res + Σ w_atom·c_atom`).

Where 3D actually helps (the honest, pre-registered claim). The two GNNs
reshape `U_res` / `V_atom` before the bipartite block. Because the same weights
process the same node states in both the free and coupled passes
(`model.py:261-262`), any purely intra-molecular geometric signal (protein
fold, pLDDT, ligand conformation) cancels in `D` for the self-channel. The only
geometric signal that survives the subtraction is the interaction term: in the
coupled pass, ligand-atom queries attend to geometry-shaped residue
keys/values across the cross-edges, and vice versa, a cross term with no
free-pass counterpart, so it does not cancel. The bet: make a buried catalytic
residue (far in sequence, close in space) and a 3D-defined pharmacophore atom
better mutual cross-attention partners than v6's flat sequence/2D attention
could. The zero-field theorem is inherited unchanged: mask cross edges, no
cross term, `D≡0`, `score≡b0`.

### 1a. How the two 3D structures couple, and what that buys (decision)

The strongest 3D signal in binding is the joint interface geometry (which
ligand atom contacts which residue in the bound complex). That requires a
shared coordinate frame, a bound pose, which we do not have: AlphaFold gives
the apo monomer, RDKit ETKDG gives a ligand conformer in an unrelated frame,
and there is no correspondence between them. Generating a pose per pair
(docking) is both infeasible here (no docking tool installed; the 200×N eval
matrix and the `B·k` triplets per step need millions of placements; DiffDock
was deliberately removed from the project) and circular for ranking
(non-binders, which dominate the pool, have no meaningful pose). So v7 does
not score from a pose.

What both-3D buys without a pose is pharmacophore/shape complementarity: each
molecule's intramolecular 3D shapes its node states, and the cross-attention
learns which 3D-defined ligand features (a spatially-arranged H-bond donor, an
aromatic centroid) are compatible with which 3D-defined pocket features, plus a
spatial-locality inductive bias on both sides and an interpretable map on both
molecules. The only thing given up vs a pose is the hard constraint "atom *a*
is 3.5 Å from residue *i*".

**Decision (user, 2026-06-12): Stage 1 now, Stage 2 as a P6 ablation.**

- Stage 1 (default, this plan): separate 3D graphs, dense feature
  cross-attention = complementarity matching. Fast ranking signal; gated by
  the P2 kill-switch.
- Stage 2 (ablation rung P6(f)): a learned per-residue pocket score
  `p_res ∈ [0,1]` (from the protein GNN states) and per-atom pharmacophore
  score `p_atom ∈ [0,1]` (from the ligand GNN states) gate the coupled
  cross-edges, biasing the cross-attention logits by
  `log p_res[i] + log p_atom[a]`, spatially restricting the interaction to both
  molecules' 3D-defined functional regions: joint-3D coupling without a pose.
  The zero-field theorem is untouched (gating only modulates existing
  cross-edges; masking them still yields `D≡0`). P6(f) measures whether this
  lifts MRR / pocket-ROC over Stage 1; if not, dense Stage-1 coupling stands.
  Stage 3 (real docking with inter-molecular 3D edges) is out of scope on this
  stack; revisit only with co-crystal / PDBbind poses.

Data flow:
```
SMILES ─RDKit ETKDGv3─> 3D conformer  ─> atom graph
                                          nodes: element/charge/hybrid/aromatic/ring
                                                 + 3D env + pooled ChemBERTa ctx
                                          edges: bonds(4 rel) ∪ spatial-kNN8 ∪ <4.5Å
                                                 RBF(3D dist) edge feats
                                          │  3-layer relational GNN (PyG MessagePassing)
                                          ▼
                                    V_atom [B,A,d=128]
                                                ┌──────────────────────────────┐
UniProt ─AF PDB cache─> CB (CA for GLY)         │  FREE pass: block-diag mask   │
ESM2 [L,1280] ─ProteinProjector─> h0[L,d]       │   (no residue↔atom edges)     │─> H_free
        │  4-layer relational contact-GNN       │  COUPLED pass: cross edges on │─> H_coup
        │  edges: backbone ∪ 8Å CB ∪ kNN16,     │  (SAME weights, run twice)    │
        │  RBF dist edge feats, pLDDT-gated      └──────────────────────────────┘
        ▼                                           D = H_coup − H_free
  U_res [B,L,d=128] ───────────────────────────────────┘   (only the atom↔res
                                                             CROSS term survives)
                                          e_res=‖D_res‖, e_atom=‖D_atom‖   (maps)
                                          score = b0 + Σ w_res·c_res + Σ w_atom·c_atom
```

Backbone choice = panel #1 (GearBind-v7, composite 7.32, top of 5). Grafts:
clean-RDKit-atom ligand graph from #2/#3; difference-field head + zero-field
theorem from v6/StructGeoField; dual falsification controls from #3.

---

## 2. 3D data acquisition & caching

### 2a. Protein structures (already local)

No bulk download for the headline run, the 9,912 AF v6 PDBs cover the ESM2
pool 1:1. Acquisition is a one-time parse+cache pass. The download path is only
for km/turnover/benchmark accessions not yet on disk, via the existing
idempotent `reactionDataFiltering/reaction_data/structures.py::download_alphafold_structures`
(query the API `https://alphafold.ebi.ac.uk/api/prediction/{ACC}` for the
version-robust `pdbUrl`/`cifUrl`; the naive `_v4` path 404s, the current
version is v6). Run gap-fill on a login node (compute/sandbox has no outbound
network).

Cache build, `scripts/build_structure_cache.py` (Bio.PDB, parallelised ~8
workers). Per residue store a β-carbon, GLY-safe (glycine has no Cβ in any AF
PDB, ~9% of residues, so this is the default code path, not an edge case):
```python
cb = res['CB'].coord if 'CB' in res else res['CA'].coord   # GLY -> CA
```
Write `structure_dir/<uniprot>.pt`:
```
{ 'cb':[L,3] f16, 'plddt':[L] u8, 'edge_index':[2,E] i32,
  'edge_attr':[E,edim] f16, 'is_gly_ca':[L] bool }
```
Edges are built once at cache time (coords are static). Store under
`reactionDataFiltering/data/interim/<dataset>/structure_graphs/` (or a shared
store + symlinks mirroring the ESM2 dedup pattern).

Alignment assert (cap-crop aware; do NOT discard long proteins). Most proteins
match exactly; a few percent are legitimate N-terminus cap-crops
(`L_esm2==1024`, PDB longer, prefix aligns); a tiny fraction are genuine
non-cap sequence mismatches (ESM2 from an older sequence than AF):
```python
assert L_struct == L_esm2 or (L_esm2 == max_residues and L_struct >= max_residues)
```
For cap-crop, truncate the PDB in lockstep to `[:1024]` (`cb`, `plddt`,
`is_gly_ca`, and the ESM2 tensor), and drop edges with either endpoint ≥1024.
Reserve the sequence-only fallback (below) for the true non-cap mismatches
only, since routing cap-crops to fallback would bias the structure ablation
against the largest, most multi-domain enzymes.

pLDDT (3 uses): (1) node feature `pLDDT/100`; (2) drop contact edges where
`min(pLDDT_i,pLDDT_j) < 50` but always keep backbone edges (chain stays
connected); (3) edge-message confidence gate `sigmoid((min(plddt)-50)/10)`.

Coverage fallback (true-mismatch + cross-dataset only): missing/obsolete/
non-cap-mismatch accession gets a sequence-only graph (backbone `|i-j|≤2` +
positional kNN, `structure_present=False`, geometric edge feats zeroed). Never
drop the protein; split preserved; becomes ablation rung P6(d). Emit a
`structure_coverage.csv` data card listing exact / cap-crop / fallback buckets.

### 2b. Ligand 3D conformers (the added requirement)

Cache build, `scripts/build_ligand_conformer_cache.py`, keyed by SMILES hash
(sibling of the existing `chemberta_token_cache` on `/work2`). Per ligand:
```python
m  = Chem.MolFromSmiles(smiles)
mh = Chem.AddHs(m)
p  = AllChem.ETKDGv3(); p.randomSeed = 42
cid = AllChem.EmbedMolecule(mh, p)
if cid < 0: cid = AllChem.EmbedMolecule(mh, useRandomCoords=True, randomSeed=42)  # 99% overall
AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
m3d = Chem.RemoveHs(mh)        # keep heavy-atom coords, A<=128
```
Store `ligand_dir/<sha1>.pt = { 'pos':[A,3] f16, 'z':[A] (atomic number),
'atom_feat':[A,~30] f16, 'bond_index':[2,Eb], 'bond_type':[Eb], 'conf_ok':bool }`.
For the ~1% ETKDG failures set `conf_ok=False` and fall back to a 2D-distance
(graph-shortest-path) geometry so the spatial-edge code never NaNs. One-time
parallelised precompute (~2 h single-core, far less parallelised).

Caveat to pre-register: an ETKDG conformer is a single low-energy guess, not
the bioactive pose. It is a soft geometric prior on the ligand, not ground
truth, hence the P6(e) ablation (3D-conformer vs 2D-topology ligand) measures
whether it actually helps. Optionally a small (k=3) conformer ensemble averaged
at the atom-feature level if single-conformer noise hurts.

---

## 3. Graph construction

### Protein graph (3 relation types)

- R0 backbone: `(i, i±1)`, always present (survives fallback).
- R1 contact: `‖CB_i - CB_j‖ < 8.0 Å`, the load-bearing 3D signal (8 Å is the
  canonical contact cutoff; keeps avg degree ~10 vs ~18 at 10 Å).
- R2 kNN-16: 16 nearest CB neighbours, guarantees connectivity and caps degree
  at 16 (`Kmax=16`, important for the gather-based aggregation).

Edge features (SE(3)-invariant, distances/separations only, no raw coords):
`[16-Gaussian RBF of ‖CB_i-CB_j‖ over 0-20 Å; signed seq-separation
clip(i-j,-32,32) to an 8-d embed; pLDDT confidence-gate scalar]`. Self-loops +
symmetric edges per relation.

### Ligand graph (3D conformer, symmetric to the protein)

Heavy atoms only (`A ≤ 128`). Three relation types, mirroring the protein:

- B0 bond: RDKit bonds, 4 bond-type sub-relations + in-ring flag, always
  present (survives the ETKDG-failure fallback).
- B1 spatial-contact: atom pairs with `‖pos_i - pos_j‖ < 4.5 Å`
  (through-space proximity that bonds miss, the 3D signal).
- B2 spatial-kNN8: 8 nearest atoms by 3D distance, connectivity + degree cap.

Atom features (~30-d): one-hot element, degree, formal charge, hybridisation,
aromaticity, num-H, in-ring, plus 3D-env scalars (count of spatial neighbours
within 4.5 Å). Edge features mirror the protein: `[RBF of 3D distance;
bond-type/relation embed]`. ChemBERTa to atoms: do not force per-atom BPE
alignment (a known v6 footgun); use the mean-pooled ChemBERTa[384] (already
cached) broadcast to every atom as global context.

This makes the ligand branch genuinely 3D and structurally symmetric to the
protein branch, the user's requirement, while the through-space edges are the
concrete mechanism by which conformer geometry enters the representation.

---

## 4. GNN backbone + interaction module

Backbone: relational message passing, non-equivariant, GearNet-style. A custom
`RelationalEdgeConv(MessagePassing)`:
`message(x_j, edge_attr, edge_type) = W[edge_type] @ [x_j ; edge_mlp(edge_attr)] * plddt_gate`,
aggregated by masked-MEAN over a fixed `Kmax` neighbour axis via plain
`gather`+`mean`, pure batched torch, never `scatter_softmax` or any
`torch_scatter` op. (`torch_geometric.utils.scatter(reduce='mean')` is a
verified fallback, but gather-over-Kmax is the safe default given the absent
`torch_scatter`.)

- Protein GNN: 4 layers, `d=128`, 3 relations, pre-LayerNorm + residual +
  GELU, JumpingKnowledge concat (anti-over-smoothing). Shallow by design.
- Ligand GNN: 3 layers, `d=128`, 3 relations (bond + 2 spatial), same block.
- No global mean-pool feeds the score. A mean-pool is computed only for the
  hard-neg hook (§8), strictly outside the score path.

Interaction = the v6 DeltaFieldHead, run verbatim on GNN outputs. The GNNs
replace v6's projection step; `U_res[B,L,128]` / `V_atom[B,A,128]` go straight
into the free/coupled bipartite machinery. Keep `df_prop_layers=0` (exact
zero-field invariant; `prop>0` holds it only approximately, per
`model.py:266`). An optional top-k=8 atoms/residue cross-sparsification is held
as a memory lever, applied inside the coupled mask, never before `D`. The
coupled cross-attention is dense by default (Stage 1); the Stage-2 pocket/
pharmacophore gate (§1a, ablation P6(f)) only biases the cross-attention logits
by `log p_res[i] + log p_atom[a]`, leaving the zero-field theorem intact.

Interpretability weights. `model.py:167` calls `self.attn(...,
need_weights=False)` (fused, fast), so the contact map is not recoverable from
attention by default. Default training/eval keeps `need_weights=False`; the
`C[i,a]` map is built two ways (§6): (i) always-available `outer(e_res,e_atom)`
masked by the structural contact graph; (ii) optional eval-only forward with
`need_weights=True, average_attn_weights=False` for the headline figure on a
few cases. The deliverable contract is form (i).

V100 32 GB feasibility: the OOM risk is the TRIPLET path, not eval. v6 "fit"
only because it used `cross_protein_implicit` negatives (k negs are
batch-resident proteins, reused, ~32 graphs/batch). v7 mandates `hard` mining:
`score_triplet_field` (`model.py:430-438`) expands to `B + B·k` independent
`forward_field` passes. At `B=16, k=4` that is 80 simultaneous protein graphs;
worst-case `N=L+A≈1152` gives MHA scores `[80,4,1152,1152]` ≈ 0.8
GB/block/pass, ×2 passes ×2 blocks ≈ 3.4 GB before gradients + GNN, which fits
at B=16 with checkpointing but does NOT fit at v6's B=32. Ligand graphs (≤128
atoms, low degree) add negligible memory. Pinned defaults (not fallbacks):
`batch_size_ligands=16`, `k≤4`, `df_n_blocks=2`, `bf16`, gradient checkpointing
on GNN + cross layers, per-protein free-pass caching (§8). Estimated ~5-15
min/epoch on the default set; comfortably inside the 12 h walltime arg.

---

## 5. Anti-shortcut mechanism (the zero-field theorem survives)

The readout is byte-identical to v6 (`model.py:285-288`): `c_res=a_res(D_res)`,
`c_atom=b_atom(D_atom)` are bias-free linear in `D`; `b0` is the sole additive
scalar. Mask cross edges (`_force_no_coupling=True`), then coupled ≡ free,
`D≡0`, every `D`-linear term is 0, `score≡b0`, one global constant for every
protein, so the protein-prior is literally unrepresentable in the score.

This holds regardless of how rich the protein or ligand node states are: both
pass through the same weights in both passes, so all intra-molecular geometry
(protein fold and ligand conformation) cancels in the self-channel; only the
cross-interaction survives. Adding ligand 3D therefore does not weaken the
guarantee.

Audited every run (3 gates, existing tooling):
1. CI zero-field unit test: extend
   `tests/test_deltafield.py::test_zero_field_invariant` to instantiate the v7
   head with full 3D protein+ligand graphs and assert `score==b0` to
   `atol=1e-6` across distinct proteins. Add a `torch_scatter` import-guard
   that fails the build.
2. Top-10 Jaccard vs `null_prot_prior` (`evaluation/null_baselines.py`): ≤
   0.11, expected ~0.05.
3. gini_residual = `gini(model) - gini(null)` < 0, expected ~-0.13.

A leak the theorem does NOT cover (mining, not score): a rich 3D protein graph
can encode protein identity hard in the free-pass mean-pool used for hard-neg
centroids (`sampler.py:354`). That can make "hard" negatives = structurally
similar proteins (same fold/substrate) that are trivially easy, degrading the
mining v7 relies on. Mitigation + gate in §8/§9 (P2): assert a non-degenerate
`pos_above_neg_max`; if the pool collapses, fall back to score-based mining
over the cached matrix.

---

## 6. Interpretability readout + validation

Artifacts on the dict contract `{score, e_res, e_atom, c_res, c_atom}` + `C`:
- `e_res[B,L] = ‖D_res‖₂`, per-residue interaction energy, indexed to 3D
  coords.
- `e_atom[B,A]` back-projects cleanly to RDKit heavy atoms (clean 1:1, unlike
  v6's BPE tokens) and to their 3D conformer positions, giving a 3D
  pharmacophore view of which ligand substructure drives the interaction.
- `C[i,a]`: default `outer(e_res,e_atom)` masked by the structural contact
  graph (no attention weights); optional eval-only attention-weighted form for
  the headline figure.

Validation (reuse existing harness): feed `e_res` into
`evaluation/deltafield_pocket_roc.py` (within-protein ROC-AUC of `e_res` vs
UniProt active/binding-site mask; 52 annotated BRENDA proteins; Wilcoxon vs
0.5; UniProt cache at `evaluation/attractor_results/_uniprot_cache/`). Because
v7's `e_res` indexes 3D coords, also run
`evaluation/attn_trough_pocket_proximity.py` (CA-CA spatial proximity of
high-`e_res` residues to binding sites).

Pre-registered go/no-go: pocket ROC-AUC PASS > 0.5, target > 0.7, must beat
v6's 0.501 (chance) and flip v5b's active-site avoidance (0.08-0.21). Three
falsification controls:
- (a) Protein coord-shuffle (random contact graph, real ESM2 features): pocket
  ROC should collapse toward ~0.5. Isolates protein graph topology.
- (b) Protein node-feature-shuffle (permute ESM2 rows, real graph): isolates
  the ESM2-feature contribution (v5b: ESM2 attention tracks hydrophobicity
  ρ+0.24, so `e_res` could survive (a) via residual hydrophobicity; (a) is not
  clean alone).
- (c) Ligand conformer-shuffle (random/2D coords, real bonds): tests whether
  ligand 3D contributes to the interaction map.

Report all controls. Pocket-ROC is headline-secondary; the ranking claim does
not depend on it.

Stage-2 only (§1a): the learned pocket score `p_res` and pharmacophore score
`p_atom` are themselves directly validatable, `p_res` against the UniProt
active/binding-site mask (same pocket-ROC harness) and `p_atom` against
RDKit-flagged pharmacophore features (`Chem.ChemicalFeatures`), so the gated
variant adds two checkable 3D artifacts, not just a ranking number.

---

## 7. Score head + loss + training recipe

- Score: `score = b0 + Σ w_res·c_res + Σ w_atom·c_atom`, unchanged, so
  `eval.py`/`metrics.py` stay untouched; `train.py`/`loss.py` change only to
  surface neg-side field dicts (see below).
- Primary loss: within-ligand `margin_loss(pos[B], neg[B,k], m=1.0)`, `k=4`
  (hard cap), via `score_triplet_field` (`loss.py:19`).
  `ProteinBalancedSampler` + hard-negative mining (`hard_pool_size=50`), the
  first run to exercise hard mining with the field head.
- Auxiliary losses (default OFF for run 1): `L_neg = λ_neg·mean‖D(L,P⁻)‖²`
  over hard negs (drives the non-binder field to 0); `L_sparse` with a
  field-energy floor (prevents collapse); `L_tv` along the protein backbone.
  Start `λ_neg=0.1`, `λ_sparse=0.01` only after ranking passes (P4). Monitor
  mean `e_res`/`e_atom` in `train_log.jsonl` for collapse. Plumbing
  prerequisite: `score_triplet_field` (`train.py:166`) returns only `(pos,
  neg)` scores today; `L_neg` needs neg-side `e_res`/`e_atom`, so the
  field-dict return-signature widening must land and be tested in P1, not P4.
- Early-stop on `val_global_auc` (not per-ligand AUC, n=2 noise). 3 seeds
  {42,7,1337}, AdamW/cosine/bf16, frozen ESM2 + ChemBERTa.
- Capacity (compute, do not assert). The zero-field test asserts
  `550_000 < n < 720_000` for the current head; adding two GNNs (JK-concat
  multiplies the post-GNN projection input by `n_layers`) will likely push a
  d=128 head past 720k. The "matched ~627k" framing is forbidden until
  `count_parameters()` (`model.py:472`) is actually run on both a d=128
  variant and a trimmed (d=96 / last-layer-JK) variant; print exact counts to
  the manifest, then pick the capacity-matched comparator and update
  `test_deltafield.py:126`. v4 reference = 627,201 trainable.

---

## 8. Integration points (exact files/functions)

Use the FIELD/dict path. Generalise the 5 hardcoded `head_type=="deltafield"`
checks to a `FIELD_HEADS = {"deltafield","gearbind"}` set (or
`model.is_field_head()`):

- `model.py`: add `elif self.head_type == "gearbind"` in `RankBind.__init__`
  registering `GearBindHead(d=128, protein_graph_cfg, ligand_graph_cfg,
  cross=DeltaFieldHead(...))`; require `protein_encoder='attn_pool'`. Implement
  `forward_field(lig_*, prot_res, prot_mask, *, prot_edge_index, prot_edge_attr,
  prot_neighbor_idx, plddt, lig_pos, lig_atom_feat, lig_bond_index,
  lig_spatial_index)` returning the same dict. `score_pairs_field` /
  `score_triplet_field` signatures MUST widen to thread the graph tensors, more
  than a 3-line edit; budget it. At `model.py:167` keep `need_weights=False`
  default; add an eval-only flag for the optional `C`-map figure.
- `data.py`: add `structure_dir` + `ligand_conformer_dir` config keys; add
  `load_structure(uniprot)` and `load_ligand_graph(smiles)` readers analogous
  to `load_protein` (`data.py:246-253`), applying the same 1024 crop +
  cap-crop-aware assert. `RankBindDataset.__getitem__` carries protein edge
  tensors + pLDDT + the ligand 3D graph. Extend `_pad_residues` +
  `TripletCollator.__call__` / `collate_pointwise` to carry padded
  `[B,L,Kmax]` neighbour-index tensors + offset-concatenated edge indices for
  both molecules (keep the padded `[B,L,*]` + mask layout the field path
  expects; avoid `to_dense_batch` / any torch_cluster op).
- `eval.py:61,303`: dispatch to a field-matrix builder; add a per-protein
  free-pass cache (`build_score_matrix_deltafield` at `eval.py:222-233` loops
  per pair with no cache today, this is new code).
- `sampler.py:302` (`refresh_scores` field branch), `:354` (centroid
  mean-pool, the M5 leak point). Protein embedding for mining = out-of-score-
  path global mean-pool of the GNN free-pass residue states, which are
  ligand-independent and static across an epoch, so cache once per protein per
  refresh and reuse for the eval matrix too. Refresh every 2-3 epochs (~1
  epoch-equivalent wall-clock). Guard: if structural similarity collapses the
  pool, fall back to score-based mining over the cached score matrix.
- `train.py:113,164,166`: `score_triplet_field` branch; widen its return to
  surface pos/neg field dicts so P4's `L_neg` is computable.
- Config: `configs/abl_gearbind.json` extends `default.json`:
  `head='gearbind'`, `protein_encoder='attn_pool'`, `ligand_encoder='per_token'`
  (graph built from SMILES + conformer cache), `structure_dir=...`,
  `ligand_conformer_dir=...`, `contact_threshold=8.0`, `knn=16`,
  `lig_spatial_threshold=4.5`, `lig_knn=8`, `gnn_layers_prot=4`,
  `gnn_layers_lig=3`, `df_prop_layers=0`, `df_n_blocks=2`,
  `batch_size_ligands=16`, `triplet.k=4`, `triplet.negative_sampling='hard'`,
  `hard_pool_size=50`, `lambda_*=0`, `grad_checkpointing=true`, `bf16=true`.
  Tag runs `v7 / v7_s7 / v7_s1337` via
  `scripts/run_v5_rankbind.sh <cfg> paula v7 <seed> 12:00:00`.

---

## 9. Phased build & ablation ladder (pre-registered gates vs v4)

| Phase | Build | Pre-registered gate |
|---|---|---|
| P0 Cache | `build_structure_cache.py` (GLY to CA default; parallel; cap-crop assert; `structure_coverage.csv`) + `build_ligand_conformer_cache.py` (ETKDGv3, conf_ok flag) | protein coverage 100% (verified); ligand ETKDG ≥ 98% (verified 99%); alignment asserts pass |
| P1 Plumbing | structure + ligand-graph loaders, collate, both GNNs; widen `score_triplet_field` return (neg-side field dicts); CI zero-field test green on v7 head with real 3D graphs | `test_zero_field_invariant` 7/7; no `torch_scatter` import; first-epoch `n_batches_skipped==0`; `train_keep_ratio_mean ≥ 0.65` (matches v6's 0.681); fallback proteins/ligands don't reduce keep ratio; neg-side dicts confirmed plumbed |
| P2 Margin-only run (λ=0, seed 42, B=16, k=4) | full train+eval, 200×200 matrix; assert hard-neg pool non-degenerate | CHECK A (kill switch): matrix-MRR ≥ 0.28 AND Top-10 Jaccard ≤ 0.11 AND `pos_above_neg_max` rises. MRR below 0.25 is a v6-parity failure; halt and reassess. A degenerate pool means switching to score-based mining and re-running P2 |
| P3 Interpretability | pocket-ROC + 3 falsification controls | pocket ROC-AUC > 0.5 (target > 0.7); protein coord-shuffle (a) collapses to ~0.5; node-feature-shuffle (b) and ligand conformer-shuffle (c) reported |
| P4 Aux losses + tuning | enable `L_neg=0.1`, `L_sparse=0.01` w/ floor | matrix-MRR > 0.326 (beat v4); Hit@10 ≥ 0.755; field energy not collapsing |
| P5 Multi-seed | seeds {42,7,1337}; capacity-matched variant with `count_parameters()` printed first | 3-seed mean MRR > 0.326±0.072; gini_residual < 0; capacity bound updated |
| P6 Ablations | (a) protein contact-graph vs sequence-kNN; (b) 4 vs 2 protein GNN layers; (c) d=96 vs 128; (d) sequence-only fallback rung; (e) ligand 3D-conformer-graph vs 2D-topology-graph; (f) Stage-2 pocket/pharmacophore-gated coupling vs Stage-1 dense cross-attention | contact-graph beats sequence-kNN on MRR and pocket-ROC (else protein-3D is cosmetic); ligand-3D beats ligand-2D on MRR/pocket (else ligand-3D is cosmetic); Stage-2 gating beats Stage-1 on MRR and/or pocket-ROC (else spatial coupling is cosmetic, keep Stage 1) |

Headline success = P4 + P3 + P5: beat v4 MRR while keeping Jaccard ~0.05 and
lifting pocket-ROC > 0.5. Defensible fallback (MRR ≈ v4 parity ~0.32):
"recovered v6's lost ranking + closed the anti-shortcut gap + delivered the 3D
interpretable contact map (protein residue and ligand pharmacophore) v6 could
not."

---

## 10. Top risks + mitigations

1. Ranking doesn't beat v4 (highest, the honest gamble). v6 hit MRR 0.206
   with the same difference-only bottleneck v7 keeps; the surviving geometric
   channel is narrow (only atom-residue cross-interaction). Mitigation:
   P6(a)/(e) quantify the 3D contribution; CHECK A is a hard kill switch at
   MRR < 0.25; fallback framing pre-registered.
2. 3D redundant with ESM2 (ESM2 implicitly predicts contacts). Mitigation:
   P6(a) decisive; P3 node-feature-shuffle separates geometry from features.
3. Field/sparsity collapse (D collapsing to 0, MRR to chance, v6's #1
   failure). Mitigation: aux losses OFF for run 1; field-energy floor +
   λ-annealing; monitor `train_log.jsonl`.
4. Memory OOM on the TRIPLET path (v6's "it fit" is invalid; it used
   `cross_protein_implicit`, not hard mining). Mitigation: pinned `B=16`,
   `k≤4`, `df_n_blocks=2`, bf16, grad-checkpoint, top-k=8, Kmax, free-pass
   cache.
5. Capacity mismatch breaks the head-ablation framing. Mitigation:
   "matched ~627k" forbidden until `count_parameters()` is run; trim JK/width;
   update `test_deltafield.py:126`.
6. Hard-neg mining degraded by 3D structural identity in the centroid.
   Mitigation: P2 asserts non-degenerate `pos_above_neg_max`; fall back to
   score-based mining if the pool collapses.
7. Interpretability `C[i,a]` not extractable under `need_weights=False`.
   Mitigation: default `C=outer(e_res,e_atom)` masked by the contact graph;
   optional eval-only attention forward for the figure.
8. pLDDT-noisy contacts in disordered regions. Mitigation: drop edges with
   min-pLDDT < 50, confidence-gate messages, pLDDT node feature, keep backbone.
9. GLY-has-no-CB crash. Mitigation: unconditional `CB if present else CA`
   (~9% of residues); `is_gly_ca` flag.
10. Cap-crop proteins wrongly discarded. Mitigation: cap-crop-aware assert;
    lockstep `[:1024]` truncation; fallback only for the ~0.2% true mismatch.
11. Ligand conformer ≠ bioactive pose (added with the ligand-3D requirement).
    A single ETKDG conformer is a low-energy guess, noisier than AlphaFold
    protein structure. Mitigation: 3D enters only as a soft geometric prior
    (through-space edges); P6(e) measures whether it helps at all; optional
    3-conformer ensemble if single-conformer noise hurts; `conf_ok=False`
    fallback to 2D-distance geometry for the ~1% ETKDG failures.
12. Anti-shortcut leakage via cross-attention. Mitigation: `c_res/c_atom`
    strictly bias-free-linear in `D`; CI zero-field gate every build; Jaccard
    audit every run.

---

## Key file references

`v5_rankbind/model.py:167` (`need_weights=False`), `:174` (DeltaFieldHead to
wrap), `:258` (`_force_no_coupling`), `:261-262` (same weights both passes),
`:285-288` (bias-free readout), `:296`/`:326` (head dispatch), `:405`
(`forward_field`), `:430-438` (`B+B·k` triplet expansion), `:472`
(`count_parameters`); `v5_rankbind/eval.py:61,303` +
`build_score_matrix_deltafield:222-233` (no cache today);
`v5_rankbind/train.py:113,164,166` (widen return for `L_neg`);
`v5_rankbind/sampler.py:294` (untrained-attn_pool note), `:302`
(`refresh_scores`), `:354` (centroid mean-pool); `v5_rankbind/loss.py:19`
(margin_loss); `v5_rankbind/data.py:246-253` (`load_protein` analog);
`v5_rankbind/tests/test_deltafield.py:44` (zero-field test to extend), `:126`
(capacity assert); `evaluation/deltafield_pocket_roc.py`,
`evaluation/attn_trough_pocket_proximity.py` (pocket validation);
`evaluation/attractor_results/v6_deltafield_null_summary.csv` (v6 MRR 0.2059 /
Jaccard 0.0526 / gini_residual -0.1394);
`reactionDataFiltering/data/raw/brenda_sabio_2026-04-29/structures/` (9,912
PDBs); `reactionDataFiltering/reaction_data/structures.py::download_alphafold_structures`
(gap-fill); `v5_rankbind/configs/abl_deltafield.json` (config to extend).

---

*Provenance: design panel `wf_8943111e-76c` (5 proposals, GearBind-v7 [winner,
7.32], GeoDeltaField, v7-InteractGraph, StructGeoField, EGNN-GeoDeltaField,
each scored by 3 lens-diverse judges, synthesised, adversarially critiqued
[4 blockers + 5 majors + 5 minors all resolved], then extended for the
3D-ligand requirement). Data facts independently verified against the repo
2026-06-12.*
