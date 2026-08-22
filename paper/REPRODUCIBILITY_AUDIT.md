# REPRODUCIBILITY_AUDIT.md — skill Phase C

Audit date: 2026-08-22. Repository: `~/rankbind` (git `main`).

## 1. Code state

- Manuscript numbers as of commit `63de30e`
  (paper: incorporate A10/A16/A17/A18 audit findings).
- Every audit artefact referenced by the manuscript is committed:
  METRIC_AUDIT.md (f9a5080), LEAKAGE_AUDIT.md (414fddc),
  PAIRED_MOLECULE_STATS.md + per-seed uncertainty (16a0846),
  pool sensitivity (cc16e34), positive-density (00969ed), synthetic
  experiment (ae93fb1), decoy probe (a45e079), null table (3613495).
- Model code: `v5_rankbind/` package; checkpoint selection fix for
  matrix-MRR model selection committed at 969c67b (reruns pending on the
  cluster — see STATISTICS_AUDIT.md pending items).

## 2. Data hashes

| file | md5 |
|---|---|
| data/dataset_with_decoys.csv | `13ca3c99aa970a5f6d1ea0645ce3abd5` |
| data/sequences/sequences.csv | `b73c8ea28670c938b4c83ba2df5ed25c` |

Split construction: `baselines/adapters/common.py::BRENDADataConfig.
get_protein_split()`, seed 42, protein-stratified; identical call path in
every consumer (v5_rankbind/data.py, evaluation/*, baselines/adapters/*).
The split is a pure function of (csv hash, sequences hash, seed,
val_frac/test_frac) — no stored split file to drift.

## 3. Run inventory (headline protocol)

- Canonical seed-42 anchors: `results/v5_rankbind/20260423-112928_012a2695c2_default_v4`
  (+ abl_no_sampler / abl_no_margin / abl_no_bilinear v4, abl_bce_only v2/v3 lineage)
- Multi-seed: `*_v4_s7`, `*_v4_s1337` run dirs from 20260423;
  aggregated by `scripts/aggregate_multiseed.py` into
  `evaluation/attractor_results/phase2_rankbind_multiseed.csv`.
- Each run dir carries `manifest.json` (resolved config incl. seed,
  git-sha of the code at submission time, output list) +
  `train_log.jsonl` + `score_matrix_rankbind.npy` + axes JSON.

## 4. Figure / table provenance

| artefact | generator |
|---|---|
| Table 1 (phase-1 dissociation) | `evaluation/test_set_eval.py`, `null_baselines.py`, `cross_model_overlap.py` |
| ablation table | `evaluation/attractor_results/phase2_rankbind_summary.csv` <- `scripts/collect_v5_runs.py` |
| multiseed error bars | `scripts/aggregate_multiseed.py` |
| response maps / jaccard figs | `evaluation/phase_d_figures.py::MATRIX_FILES` |
| seven-dataset table | benchmark runs + `evaluation/benchmark_null_eval.py` |
| synthetic figure/numbers | `evaluation/synthetic_experiment.py` (seed base 424242) |
| decoy-probe numbers | `evaluation/decoy_leakage_probe.py` (deterministic given caches) |
| null-baseline table | `evaluation/null_baseline_table.py` (pos_pairs sorted -> deterministic) |

## 5. Known gaps

1. **Model-selection rerun in flight**: A3/A4 condition-B runs
   (`abl_mrrsel`, SLURM 27295847-49) and the Protocol-A multiseed sweep
   (split pinned via `data.split_seed=42`) are queued; Tables that will
   cite them are marked PENDING in STATISTICS_AUDIT.md.
2. Transfer runs remain single-seed except the bs peak-seed additions
   landing incrementally (`bs_v2/v3_hp*` dirs); A5 per-seed CSVs are
   produced after those jobs finish.
3. The manuscript PDF is built with texlive/20230313 module
   (`cd paper/scirep && make`); build artifacts are gitignored.
