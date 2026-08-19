# RankBind: session handoff (Phase 2 in progress)

## Update 2026-04-30

- Workspace is now a git repo. `~/rankbind/` itself is tracked
  (`main` branch). `.gitignore` excludes `data/`, `results/`, `logs/`,
  `external/`, `v4_residue_only/checkpoints/`, `*.tar.gz`, `*.pt`.
  `reactionDataFiltering/` is a git submodule pinned to a specific
  commit of https://github.com/christophmanitz/reactionDataFiltering.
  Clone with `git clone --recurse-submodules`, or
  `git submodule update --init` after a plain clone.
  See [`README.md`](README.md) for the submodule workflow.
  Tracked footprint (excluding submodule contents): ~170 files /
  ~10 MB. No remote yet.
- BRENDA+SABIO training run in flight (tag `bs_v1`). Three
  `*_no_decoys` configs submitted on `paula` (jobs 21484693 kcat_km,
  21484695 turnover, 21484823 km). km uses 12 h walltime because the
  dataset is ~9.5 k proteins / ~19 min/epoch. The decoys job
  (21480544_0/1/2 on `paul`) is still running for the `*_with_decoys`
  variants, which train next.
- ESM2 embeddings deduplicated. All three datasets shared ~3.4 k
  uniprots; the per-dataset `.pt` copies were 18 987 files / 39 GB.
  Migrated to a single shared store at
  `reactionDataFiltering/data/interim/esm2_embeddings_shared/` (9 912
  unique files / 21 GB) with relative symlinks in each dataset
  folder. The loader (`v5_rankbind/data.py`) follows symlinks
  transparently, so no code change was needed. Migration script:
  `reactionDataFiltering/scripts/dedup_embeddings.py`, idempotent.
- `scripts/run_v5_rankbind.sh` accepts a 5th arg `WALLTIME`
  (default `06:00:00`). Use it for big datasets:
  `bash scripts/run_v5_rankbind.sh datasets/km_no_decoys paula bs_v1_km "" 12:00:00`.

## Status at a glance (2026-04-23)

- Phase 1: done. Four baselines trained and evaluated; the
  null-baseline pivot refuted the Gini-attractor thesis. Narrative and
  figures: `evaluation/attractor_results/phase1_report.html`.
- Phase 2: the `v5_rankbind/` package is built, the matched-capacity
  ablation is done, and hard-negative mining (Priority A), the ablation
  re-runs (A1), and the multi-seed sweep (B) all landed 2026-04-23 as
  tag v4. What remains is documentation-only (Priority C, the PLAN.md
  addendum; D, figure regeneration; E, the Phase-2 HTML report). The
  authoritative delta is `v5_rankbind/PHASE2_LOG.md`: the top "Resume
  state" section dated 2026-04-23 is the current state, the older
  2026-04-22 section is historical.
- gAUC retirement decision confirmed by probe (2026-04-23 ~16:00):
  a BCE-auxiliary probe (`bce_aux_weight=0.5`, seed=42) lifted test
  gAUC by only +0.03 (0.623 to 0.655), nowhere near 0.80. Matrix
  ranking unchanged. This confirms the 0.80 threshold is a dataset
  ceiling under ligand-conditional optimisation, not a loss-function
  artefact. `Global AUC ≥ 0.80` stays retired as a success gate; gAUC
  is still reported in every table. Evidence: run_dir
  `results/v5_rankbind/20260423-151536_9ee7fdbfbc_probe_bce_aux_v4_bceaux05/`,
  summary CSV row tagged `probe_bce_aux,v4_bceaux05`.
- v4 default headline numbers, paper-ready, 3-seed mean ± std
  (biln rank=128, hard negs pool=50, 627,201 params, seeds {42, 7,
  1337}): MRR 0.326 ± 0.072, H@5 0.598 ± 0.090, H@10
  0.755 ± 0.095, Gini-residual −0.210 ± 0.022, Top-10 Jaccard
  vs null_prot_prior 0.000, test global AUC 0.634 ± 0.010. All PLAN.md
  thresholds pass except `Global AUC ≥ 0.80`, retired as miscalibrated
  (it is the shortcut metric we are trading away). Multi-seed CSV:
  `evaluation/attractor_results/phase2_rankbind_multiseed.csv`.
  Aggregator: `scripts/aggregate_multiseed.py`. v3 single-seed numbers
  (no hard negs) for reference: MRR 0.201, H@5 0.412, H@10 0.559,
  Gini-residual −0.124.

## Read this first

- `docs/development_plan.md`, the pre-Phase-1 plan. Do not edit.
  (Moved from project root 2026-05-04.)
- `v5_rankbind/PLAN.md`, the pre-Phase-2 plan. Do not edit §1-§9.
  Append addenda only.
- `CLAUDE.md` (this file) and `v5_rankbind/PHASE2_LOG.md` are the
  authoritative deltas. When they conflict, PHASE2_LOG wins for
  Phase-2-specific guidance.

Phase 1 was executed but the original framing (Gini-based universal
attractor bias) did not survive a null-baseline probe. The empirical
findings reframe Phases 2-5:

- `evaluation/attractor_results/phase1_report.html`: Phase-1 diagnostic
  report with glossary, figures, and plan pivot. Open in a browser.
- `v5_rankbind/PHASE2_LOG.md`: Phase-2 session log, what is built, the
  ablation table, key findings, Priority A-E for the next session.

## Project orientation

- Leipzig HPC cluster. Primary venv: `~/venvs/hieratombind`
  (torch 2.8.0+cu128). DrugBAN needs `~/venvs/drugban` (torch 2.4.0+cu124, DGL).
  - Rebuilt 2026-04-29 after the original venv was deleted to free disk.
    Installed packages: `torch==2.8.0+cu128`, `numpy`, `pandas`,
    `scikit-learn`, `scipy`, `transformers`, `pyyaml`, `tqdm`, `matplotlib`,
    `rdkit`. `torch_geometric` is NOT installed, so only the v5_rankbind /
    evaluation stack works. If retraining a Phase-1 baseline (GraphDTA, GIGN,
    GEMS), reinstall PyG against torch 2.8 first; DrugBAN keeps its own venv
    (`~/venvs/drugban`).
- Always `module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0`
  before activating a venv; the system libs are required.
- Data: `data/dataset_with_decoys.csv` (BRENDA + decoys),
  `data/sequences/sequences.csv`, `~/hpc/structures/` (AlphaFold PDBs).
- Unified split (protein-based, seed=42): see `baselines/adapters/common.py`
  (`BRENDADataConfig.get_protein_split()`). Every model and every diagnostic
  uses this same split. Do not re-invent it.

## Phase 1 outcome (one paragraph)

All four baselines (GraphDTA, MolTrans, DrugBAN, GEMS) were trained on the
BRENDA protein-based split. They all produce `Gini ≈ 0.995` on the 200×200
score matrix, but a null baseline that uses only per-protein training
positive rates (`null_prot_prior`) produces the same `Gini ≈ 0.995`. Gini is
therefore a data-geometry artifact, not evidence of learned pathology. The
real pathology is visible on a different axis: all four models achieve high
global AUC (0.63-0.95) but per-ligand AUC ≤ 0.625, three of the four score
below random (0.5). Models learn protein-level shortcuts, not ligand-protein
interaction.

### Numbers to cite

| Model    | Global AUC | Per-ligand AUC | Top-10 Jaccard vs null_prot_prior |
|----------|-----------:|---------------:|----------------------------------:|
| DrugBAN  | 0.954      | 0.375          | 0.54                              |
| MolTrans | 0.937      | 0.500          | 0.05                              |
| GraphDTA | 0.869      | 0.625          | 0.67                              |
| GEMS     | 0.633      | 0.250          | 0.67                              |

Source: `evaluation/attractor_results/test_summary_all.csv` and
`cross_model_overlap.csv`. Gini for all four ≈ 0.995, same as
`null_prot_prior`.

## What changed vs docs/development_plan.md

- Phase 1.2 Gini-based diagnosis: demoted. Gini stays as a secondary
  descriptor but is no longer a success criterion. The new primary metric is
  per-ligand AUC.
- Phase 1 scope: added (now complete) null baselines, cross-model
  attractor overlap, test-set evaluation. Removed: DeepDTA, DualBind (NVIDIA
  and Lin et al.), GIGN. Baseline set is locked to the four trained models.
- Phase 2 (pretrained features): deprioritized. GEMS already uses ESM2
  and has the worst per-ligand AUC of the four; richer features do not
  address the shortcut. Feature upgrades are now an ablation inside Phase 3,
  not a standalone phase.
- Phase 3 (ranking objective): promoted to the new Phase 2 and expanded.
  This is where the method contribution lives.
- Phase 4 (adaptive atom gating): kept as old Phase 4 but lower priority;
  only pursue if Phase 2-new lifts per-ligand AUC meaningfully.
- Paper thesis: shifts from "universal attractor bias exists" (refuted in
  the trivial sense) to "DTI models pass global AUC by learning
  protein-level shortcuts; RankBind is an architecture that enforces
  ligand-conditional ranking, measured by per-ligand AUC."

Working title: *RankBind: Protein-Invariant Contrastive Learning for
Ligand-Conditional DTI.*

## Continue here: Phase 2 (in progress)

All four Phase-2 components shipped:

1. Protein-balanced sampling: `v5_rankbind/sampler.py::ProteinBalancedSampler`
2. Within-ligand margin loss: `v5_rankbind/loss.py::margin_loss`, k=4, m=1.0
3. Bilinear head: `v5_rankbind/model.py::BilinearHead`, low-rank + diag,
   `bilinear_rank=128` in `default.json` (matches MLPConcatHead's 65,793
   params for fair head ablation)
4. Hard-negative mining (v4): `v5_rankbind/sampler.py::TripletCollator`;
   `refresh_scores(model, device)` caches the (positive-ligand × train-protein)
   score matrix at each epoch start; for each anchor, sample k negatives from
   the top `hard_pool_size=50` non-positive proteins by current score. Config
   flag: `triplet.negative_sampling = "hard"` in `default.json`. The
   `pos_above_neg_max` diagnostic in `train_log.jsonl` confirms the model
   learns to separate its own hardest confusers (0.92 to 0.97 over 32 epochs
   in v4 default).

Ablation table and paper reading live in `v5_rankbind/PHASE2_LOG.md`.

### What the ablation table actually says (v3 matched capacity)

- Margin loss is the dominant contribution. Removing it drops MRR ~9×
  (0.20 to 0.02) and H@10 ~15× (0.56 to 0.03).
- Balanced sampler is a secondary positive: +24% MRR, +75% H@5, +72%
  H@10 vs random batches.
- Bilinear vs MLP at matched capacity is a wash: bilinear wins H@5,
  ties H@10, trails MRR. Paper framing: keep bilinear for interpretability
  and inductive bias, not for a raw-metric win.
- BCE configs reproduce Phase-1 pathology regardless of architecture:
  gAUC 0.92 with matrix MRR ≈ 0.01 and Jac-null 0.43.
- Hard-negative mining (v4) lifts default cleanly: 3-seed mean MRR
  0.326 ± 0.072 (v3 was 0.201 single-seed), H@10 0.755 ± 0.095 (v3
  0.559), Gini-residual −0.210 ± 0.022 (v3 −0.124). Same capacity.
- Bilinear head is strictly more stable than MLP under hard negs.
  Multi-seed: bilinear MRR std 0.072, MLP std 0.161, so the MLP head
  has 2.2× wider seed-to-seed variance. Means are comparable (0.326
  vs 0.243) but MLP's range is 0.130-0.428 while bilinear's is
  0.247-0.386. This is the paper's argument for keeping bilinear.

### Pending for next session (see PHASE2_LOG.md for the full list)

- A. Hard-negative mining: DONE 2026-04-23 as tag v4.
- A1. Margin-ablation re-runs with hard negatives: DONE 2026-04-23.
- B. Multi-seed (3 seeds × 5 configs, 15 runs): DONE 2026-04-23.
  Aggregator: `scripts/aggregate_multiseed.py`. Output:
  `evaluation/attractor_results/phase2_rankbind_multiseed.csv`.
- C. `v5_rankbind/PLAN.md` addendum. Start here. No cluster time
  needed, just retire `Global AUC ≥ 0.80`, promote matrix MRR / H@10
  to primary, and cite the multiseed CSV as canonical numbers.
- D. Figure regeneration: add v4 default to
  `evaluation/phase_d_figures.py::MATRIX_FILES`.
- E. Phase-2 HTML report: numbers are now final, ready to write.

### Evaluation conventions (what every v5 run must report)

- Primary: matrix-level MRR, Hit@5, Hit@10 on the 200×200 score matrix
  (same pool as `evaluation/null_baselines.py`). `per_ligand_auc` is n=4 on
  the test split and demoted to supplementary.
- Global AUC / AUPR for baseline compatibility (not a success gate).
- Gini-residual = `gini(model) − gini(null_prot_prior)`; negative = good.
- Top-10 Jaccard vs `null_prot_prior`; low = shortcut-avoidant.
- Pipeline reuses `evaluation/test_set_eval.py`, `null_baselines.py`,
  `cross_model_overlap.py`, `phase_d_figures.py`.

## Key files (already written, do not rewrite)

### Baselines: trained, evaluated, do not retrain

- `baselines/adapters/common.py`: unified split + data config
- `baselines/adapters/adapter_{graphdta,moltrans,drugban,gems}.py`
- `baselines/adapters/train_original.py`: unified trainer (GraphDTA, MolTrans, GEMS)
- `baselines/adapters/train_original_drugban.py`: DrugBAN (separate venv)
- `results/original_{graphdta,moltrans,drugban,gems}/best_model.pt` + score matrices

### Phase-2 model: built, ablated, do not rewrite from scratch

- `v5_rankbind/data.py`: RankBindDataset + protein-split pipeline (reuses
  `BRENDADataConfig`)
- `v5_rankbind/sampler.py`: `ProteinBalancedSampler`, `TripletCollator`
  (currently `cross_protein_implicit` negatives, the hard-negative variant is
  the Priority-A work)
- `v5_rankbind/model.py`: `LigandProjector`, `ProteinProjector`,
  `BilinearHead` (rank configurable), `MLPConcatHead`
- `v5_rankbind/loss.py`: `margin_loss`, BCE, diagnostic parts
- `v5_rankbind/metrics.py`: `matrix_ranking_metrics` (primary),
  `per_ligand_auc`, `hit_at_k`, `global_metrics`
- `v5_rankbind/train.py` / `eval.py`: full training loop with provenance
  (manifest.json), 200×200 score matrix at eval
- `v5_rankbind/run_manifest.py`: JSON config loader with
  `"extends": "parent.json"`, provenance helpers
- `v5_rankbind/configs/{default,abl_*}.json`: 5 configs, all extend default
- `scripts/run_v5_rankbind.sh`: single SLURM job (train + eval),
  race-safe run-dir detection via parsed train.py stdout
- `scripts/run_v5_ablations.sh`: submits all 5
- `scripts/collect_v5_runs.py`: walks `manifest.json`, emits
  `results/v5_rankbind/runs_manifest.csv`

### Diagnostics: stable, ready to reuse

- `evaluation/attractor_diagnosis.py`: Gini, attractor scores, rank displacement
- `evaluation/null_baselines.py`: the null-prior probe (THE Phase-1 pivot tool)
- `evaluation/cross_model_overlap.py`: Jaccard + Spearman between models
- `evaluation/test_set_eval.py`: test-set AUC / per-ligand AUC / Hit@K
- `evaluation/phase_d_figures.py`: publication figures

### Reports & paper-ready tables

- `evaluation/attractor_results/phase1_report.html`: Phase-1 narrative
- `evaluation/attractor_results/fig_{summary,response_maps,cross_overlap,auc_scatter}.png`
- `evaluation/attractor_results/test_summary_all.csv`
- `evaluation/attractor_results/gini_comparison.csv`
- `evaluation/attractor_results/cross_model_overlap.csv`
- `evaluation/attractor_results/phase2_rankbind_summary.csv`: Phase-2
  ablation + Gini + Jaccard in one table (paper-ready)
- `v5_rankbind/PHASE2_LOG.md`: Phase-2 session log / handoff doc
- `results/v5_rankbind/runs_manifest.csv`: flat table of all v5 runs

## Do / don't

**Do**
- Reuse `BRENDADataConfig.get_protein_split()`. Seed=42. Do not re-split.
- Report matrix MRR / H@10 (primary) alongside global AUC every time.
  `per_ligand_auc` stays as supplementary (n=4 on the test split).
- Tag every SLURM run: `bash scripts/run_v5_rankbind.sh <cfg> paula <tag>`.
  Current tag progression: v1 (broken collator) to v2 (first valid) to v3
  (matched capacity) to v4 (hard-negative mining, the reported numbers,
  seed=42 canonical) to v4_s7 / v4_s1337 (multi-seed error bars).
  For a seed sweep, the 4th arg is the seed, e.g.
  `bash scripts/run_v5_rankbind.sh default paula v4 7` (auto-appends
  `_s7` to the tag and threads `--seed 7` to train.py). Bulk:
  `bash scripts/run_v5_multiseed.sh`.
- Extend configs via `"extends": "default.json"` + minimal overrides.
- After any training change, spot-check `train_keep_ratio_mean` and
  `n_batches_skipped` in the first epoch log; zero skipped / high keep
  ratio means the collator is healthy.
- When adding a new model, add its score matrix to `MATRIX_FILES` in
  `phase_d_figures.py` so the comparison figures regenerate.
- Open `phase1_report.html` / `PHASE2_LOG.md` if the plan context has decayed.

**Don't**
- Don't reintroduce GIGN, SKiD, DiffDock, DeepDTA, DualBind. These were
  removed after Phase 1. Adding a fifth baseline does not change the story.
- Don't edit `docs/development_plan.md` or `v5_rankbind/PLAN.md` §1-§9.
  Both are historical references; append addenda sections instead.
- Don't treat Gini as a success metric. It is bounded above by the data
  prior; any model that fits the training distribution will match.
- Don't treat `Global AUC ≥ 0.80` as a success gate. Retired, it is the
  shortcut metric RankBind is trading away.
- Don't rerun the four Phase-1 baselines. Checkpoints and score matrices
  are pinned.
- Don't use `ls -td` to find run_dirs in SLURM scripts; race condition
  between concurrent eval steps. The fixed script parses run_dir from
  train.py stdout via `tee`.
- Don't early-stop on `val_per_lig_auc`; it's n=2 on the val split, pure
  noise. Use `val_global_auc` (current default).

## First commands for next session

```bash
module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
source ~/venvs/hieratombind/bin/activate

# Orient: re-read the Phase-2 log (2026-04-23 section at the top is current state)
cat v5_rankbind/PHASE2_LOG.md

# Sanity-check artifacts still exist
ls results/v5_rankbind/*_v4/manifest.json results/v5_rankbind/*_v3/manifest.json
cat evaluation/attractor_results/phase2_rankbind_summary.csv
ls results/original_*/best_model.pt

# Priority C (next): append PLAN.md addendum retiring Global AUC gate,
#   promoting matrix MRR / H@10, citing
#   evaluation/attractor_results/phase2_rankbind_multiseed.csv.
# No cluster time needed.
cat evaluation/attractor_results/phase2_rankbind_multiseed.csv

# If you ever need another seed:
bash scripts/run_v5_rankbind.sh default paula v4 <seed>
# Or a full sweep:
bash scripts/run_v5_multiseed.sh
```