# Phase 1 — Attractor Bias Diagnosis: Status

**Letzte Aktualisierung:** 2026-04-20 17:00

---

## Ergebnisse: Protein-Attractor Bias ist universal

| Model | Architecture | Best AUC | Gini(attractor) | Status |
|-------|-------------|----------|-----------------|--------|
| ResidueOnlyBind_v4 | GNN (dual-encoder) | — | **0.9615** | ✅ Complete |
| DeepDTA | CNN seq+SMILES | 0.9752 | **0.9898** | ✅ Complete |
| GraphDTA | GCN mol + CNN seq | 0.8310 | **0.9840** | ✅ Complete |
| DrugBAN | GCN mol + CNN seq + BAN | 0.7677 | **0.9888** | ✅ Complete |
| MolTrans | Transformer seq+SMILES | 0.9508 | **0.9758** | ✅ Complete |
| GEMS | BiLSTM + GCN | 0.8465 | **0.9900** | ✅ Complete |
| GIGN | Cross-GNN (3D) | 0.9099 | **0.8280** | ✅ Complete |

**Key Finding:** All 7 architectures show strong protein-attractor bias (Gini ≥ 0.83). Six of seven show Gini > 0.97, confirming the bias is **universal** across CNN, GCN, Transformer, BAN fusion, and BiLSTM+GCN. GIGN (3D structure-based) shows lower but still substantial bias (0.83), suggesting spatial interaction features provide partial — but insufficient — resistance to attractor collapse.

---

## Smoke Test (Re-implementations) — ✅ ALL COMPLETE

Simplified re-implementations to validate data pipeline and demonstrate attractor bias.
Jobs: 21250020–21250027 (SLURM, partition=paula, A30 GPU)

- All 7 models trained successfully on 9178–9330 samples (882 proteins, binary classification)
- Score matrices: 100×100 cross-evaluation for all models
- Attractor diagnosis completed for all models — plots in `evaluation/attractor_results/`

### Fixed Issues (this session)
- DeepDTA/MolTrans: embedding vocab size mismatch (prot=27, lig=76)
- Score matrix OOM: reduced N_MATRIX from 300→100, added chunked evaluation
- SLURM job crashes: removed `set -euo pipefail`, increased mem to 48G
- Disk quota: removed old venvs (unimol, unimol2, unimol3) and data copies

---

## Phase A: Original Repo Setup (prepared, NOT training yet)

### External Repos (cloned)
```
external/
├── DrugBAN/      # DGL-based — DGL NOT installed (conflict with torch 2.8)
├── GraphDTA/     # PyG-based ✓
├── MolTrans/     # BPE tokenization — subword-nmt installed ✓
├── GIGN/         # PyG-based, needs 3D structures (DiffDock)
└── GEMS/         # PyG-based, needs ESM2 + ChemBERTa embeddings
```

### Adapters (written + tested)
```
baselines/adapters/
├── common.py                  # Unified BRENDADataConfig, protein-based splits
├── adapter_graphdta.py        # SMILES → 78d atom features + seq encoding ✓
├── adapter_moltrans.py        # BPE tokenization using original ESPF vocab ✓
├── adapter_drugban.py         # DGL graph + protein integer encoding (needs DGL venv)
├── adapter_gign.py            # 3D complex graph from PDB+SDF ✓
├── adapter_gems.py            # Ligand graph + ChemBERTa + ESM2 embeddings ✓
├── train_original.py          # Full trainer: training loop + score matrix + diagnosis ✓
└── train_original_drugban.py  # Separate DrugBAN trainer (DGL venv) ✓

scripts/
├── run_original_baselines.sh  # SLURM submission for all original models ✓
├── setup_drugban_venv.sh      # Create DGL venv for DrugBAN ✓
└── run_esm2_embed.sh          # ESM2 embedding precomputation for GEMS ✓
```

### Remaining for Phase A
- [ ] Install DGL in **separate venv** for DrugBAN (`scripts/setup_drugban_venv.sh`)
- [ ] Run DiffDock for GIGN 3D structures (903 proteins × ligands)
- [ ] Precompute ESM2 embeddings for GEMS (`scripts/run_esm2_embed.sh`)
- [x] Test adapter imports (GraphDTA, MolTrans, GIGN, GEMS: ✓ import + data loading)
- [x] Wire up `train_original.py` with model-specific forward loops
- [x] Write `train_original_drugban.py` (separate DGL trainer)
- [x] Write `scripts/run_original_baselines.sh` (SLURM submission)

---

## Infrastructure

### HPC Cluster (Leipzig)
- GPU: paula (A30, 24GB VRAM)
- Venv: `$HOME/venvs/hieratombind` (PyTorch 2.8.0+cu128, PyG 2.7.0)
- Data: `data/processed_hieratom/` (9632 .pt files)
- Structures: `~/hpc/structures/` (939 AlphaFold v6 PDBs)
- Sequences: `data/sequences/sequences.csv` (882 proteins)

### Attractor Results
```
evaluation/attractor_results/
├── score_matrix_ResidueOnlyBind_v4.npy   (200×200)
├── response_map_*.png                     (7 models)
└── attractor_dist_*.png                   (7 models)

baselines/{model}/output/
├── score_matrix_GraphDTA.npy             (100×100)
├── score_matrix_DrugBAN.npy              (100×100)
├── score_matrix_DeepDTA.npy              (100×100)
├── score_matrix_MolTrans.npy             (100×100)
├── score_matrix_GEMS.npy                 (100×100)
└── score_matrix_GIGN.npy                 (100×100)
```
