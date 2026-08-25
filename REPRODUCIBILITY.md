# Reproducibility

Every number, table, and figure in the paper maps to a command in this
file. Each entry below lists the command to run from the project root,
the input artefacts it expects to exist, and the output paths it writes,
so a single pass through §3 regenerates the paper's evidence from a fresh
clone. The companion file `docs/HP_SWEEP_INTEGRATION_PLAN.md` documents
how a sweep result that arrives after this commit gets folded back into
the paper.

---

## Artefacts on Zenodo

Two provisional Zenodo records (created, not yet public; DOIs will be
assigned on publication) hold everything that is not in the git repo:

- **Record 1 — code, data and per-run artefacts** (`rankbind-paper-v1`):
  repository snapshot tar (`rankbind-paper-snapshot.tar.gz`, incl.
  the `reactionDataFiltering` submodule),
  BRENDA-200 pairs+sequences
  (`brenda200.tar.gz`), the BRENDA+SABIO raw snapshot
  (`brenda_sabio_raw_2026-04-29.tar`) and interim tables
  (`brenda_sabio_interim.tar.gz`), the re-downloaded external benchmarks
  (`benchmarks_csvs.tar.gz`), the BRENDA/SABIO ESM2 embeddings
  (`brenda200_esm2.tar`, `brenda_sabio_esm2.tar`), all Phase-1 checkpoints
  (`phase1.tar`), all RankBind run manifests/checkpoints/score matrices
  incl. the cold-split runs (`v5_rankbind.tar`), and the committed result
  CSVs behind every paper table (`attractor_results_csvs.tar.gz`).
- **Record 2 — benchmark ESM2 embeddings** (`rankbind-benchmark-embeddings-v1`):
  regenerated per-residue ESM2 embeddings for Davis, KIBA, BindingDB_Kd
  and ESP (`davis_esm2.tar`, `kiba_esm2.tar`, `bindingdb_kd_esm2.tar`,
  `esp_esm2.tar`).

Both records ship a `README.md` (layout + reproduction commands) and a
`SHA256SUMS` file listing the SHA-256 of every archived file. Staging
lives on the cluster at `/work2/zw93onug-rankbind_zenodo/zenodo_staging/`;
uploads go to the pre-created (unsubmitted) deposits via
`zenodo_upload3.sh`-style `curl` PUTs. Do not publish until submission.

---

## 1. One-time setup

```bash
# Fresh clone with submodule (the dataset pipeline lives there):
git clone --recurse-submodules https://github.com/christophmanitz/rankbind
cd rankbind

# Or if already cloned plain:
git submodule update --init --recursive

# Leipzig HPC environment used for every run in the paper:
module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
python -m venv ~/venvs/hieratombind && source ~/venvs/hieratombind/bin/activate
pip install -r requirements.txt
```

Venvs used: `~/venvs/hieratombind` covers the `v5_rankbind`
training/eval stack and every `evaluation/` diagnostic. DrugBAN
training requires `~/venvs/drugban` (separate torch + DGL). ESM2
embedding generation uses `~/venvs/esm2` and lives inside the
submodule.

GPU: all paper runs use one NVIDIA A30 on the Leipzig cluster,
SLURM partition `paula` (with `clara` as fallback). Mean per-seed
walltime is ≈1.5 h for `v4_default` and ≈4 h for the Stage-b residue
extension.

---

## 2. Pin discipline

Every training run produces a `manifest.json` under
`results/v5_rankbind/<run_id>/` that captures:

- Resolved configuration JSON (after `extends`-merging).
- Git commit + dirty flag at submission time.
- Library versions (torch, transformers, scikit-learn, etc.).
- SHA-256 of the input CSV and sequence file.
- SHA-256 of the resulting `best_model.pt` and
  `score_matrix_rankbind.npy`.
- Train/val/test split statistics.

So any number in the paper can be traced from its CSV row in
`evaluation/attractor_results/...` back to the manifest, the config,
the input data hash, and the code commit. The reverse direction also
works: `scripts/collect_v5_runs.py` walks the manifests and emits
`results/v5_rankbind/runs_manifest.csv`, one row per run.

---

## 3. Per-result regeneration

Each entry is a single piece of evidence in the paper. "Cmd" is the
command to run from the project root; "Reads" lists the input files
the command expects to exist; "Writes" lists the artefacts it emits.

### 3.1 §3.3 Phase-1 baseline numbers (Table 1: pooled AUC, per-ligand AUC, Top-10 Jaccard, Gini)

- Cmd:
  ```bash
  python evaluation/test_set_eval.py
  python evaluation/null_baselines.py
  python evaluation/cross_model_overlap.py
  ```
- Reads:
  `results/original_{graphdta,moltrans,drugban,gems}/best_model.pt`
  and the matching `score_matrix_*.npy`.
- Writes:
  - `evaluation/attractor_results/test_summary_all.csv`
  - `evaluation/attractor_results/gini_comparison.csv`
  - `evaluation/attractor_results/cross_model_overlap.csv`
  - `evaluation/attractor_results/score_matrix_null_*.npy`
- The baseline checkpoints under `results/original_*` are pinned;
  rebuilding them from scratch is not required to reproduce the paper.

### 3.2 §6.1 Headline three-seed table (RankBind v4 default + ablations)

- Cmd: submit fifteen jobs (5 configs × 3 seeds), then aggregate.
  ```bash
  bash scripts/run_v5_multiseed.sh           # submits all 15
  # …wait for jobs to finish…
  python scripts/aggregate_multiseed.py
  ```
- Reads (after jobs complete):
  `results/v5_rankbind/*_v4_s{42,7,1337}/manifest.json` for each of
  the 5 configs (`default`, `abl_no_sampler`, `abl_no_bilinear`,
  `abl_no_margin`, `abl_bce_only`).
- Writes:
  `evaluation/attractor_results/phase2_rankbind_multiseed.csv`
  (the canonical paper-grade table).

### 3.3 §6.1 Stage-b headline (`abl_attn_pool` row)

- Cmd: three single-seed runs.
  ```bash
  bash scripts/run_v5_rankbind.sh abl_attn_pool paula v5b 42
  bash scripts/run_v5_rankbind.sh abl_attn_pool paula v5b 7
  bash scripts/run_v5_rankbind.sh abl_attn_pool paula v5b 1337
  python scripts/aggregate_multiseed.py
  ```
- Reads/writes: same path family as §3.2.

### 3.4 §6.4 BCE-auxiliary probe (gAUC retirement justification)

- Cmd:
  ```bash
  bash scripts/run_v5_rankbind.sh probe_bce_aux paula v4_bceaux05 42
  ```
- Reads: `v5_rankbind/configs/probe_bce_aux.json`.
- Writes:
  `results/v5_rankbind/<run_id>/test_summary.json`, the gAUC = 0.655
  number cited at the top of §6.4 lives here.

### 3.5 §7.3 Attention-weight audit (cross-seed Spearman = 0.86)

- Cmd:
  ```bash
  python evaluation/attn_weight_inspection.py
  ```
- Reads:
  `results/v5_rankbind/*_v5b_s*/best_model.pt` (the three Stage-b
  checkpoints from §3.3).
- Writes:
  - `paper/figures/fig_attn_concentration_hist.png`
  - `paper/figures/fig_attn_cross_seed_agreement.png`
  - `paper/figures/fig_attn_weight_examples.png`
  - `evaluation/attractor_results/attn_audit_summary.csv`

### 3.6 §8.1 Transferability probe on BRENDA+SABIO with-decoys (this commit)

- Cmd:
  ```bash
  python evaluation/null_prior_probe_brenda_sabio.py
  ```
- Reads:
  - `results/v5_rankbind/20260503-112034_..._kcat_km_with_decoys_bs_v1/`
    (manifest + `score_matrix_rankbind.npy` + `score_matrix_axes.json`)
  - same for the `km` and `turnover` runs
  - `reactionDataFiltering/data/interim/{kcat_km,km,turnover}_brenda_sabio/with_decoys.csv`
- Writes:
  - `evaluation/attractor_results/null_prior_probe_brenda_sabio.csv`
  - `evaluation/attractor_results/null_prior_probe_brenda_sabio.txt`
  - `evaluation/attractor_results/score_matrix_<dataset>_null_*.npy`
- This is the source of every number in the §8.1 transferability
  table.

### 3.7 §8.1 v4 BRENDA+SABIO training runs themselves

The three `*_with_decoys_bs_v1` runs in §8.1 are produced by:

```bash
bash scripts/run_v5_rankbind.sh datasets/kcat_km_with_decoys  paula bs_v1 ""  06:00:00
bash scripts/run_v5_rankbind.sh datasets/km_with_decoys       paula bs_v1 ""  12:00:00
bash scripts/run_v5_rankbind.sh datasets/turnover_with_decoys paula bs_v1 ""  08:00:00
```

These runs depend on the BRENDA+SABIO dataset pipeline (submodule),
which is rebuilt with:

```bash
cd reactionDataFiltering
# see its own README for the full pipeline; pinned commit lives in
# the parent's .gitmodules. Briefly:
bash hpc/run_decoys.sh kcat_km
bash hpc/run_decoys.sh km
bash hpc/run_decoys.sh turnover
```

### 3.8 Publication figures

- Cmd:
  ```bash
  python evaluation/phase_d_figures.py
  ```
- Reads:
  Every `score_matrix_*.npy` listed in
  `evaluation/phase_d_figures.py::MATRIX_FILES` (Phase-1 baselines +
  RankBind v4 default + the three nulls). Adding a new model only
  requires extending that dict.
- Writes:
  - `paper/figures/fig_summary.png`
  - `paper/figures/fig_response_maps.png`
  - `paper/figures/fig_cross_overlap.png`
  - `paper/figures/fig_auc_scatter.png`

### 3.9 Hyperparameter sweep (in flight at submission)

Twelve jobs, configured for follow-up integration (see
`docs/HP_SWEEP_INTEGRATION_PLAN.md` for the integration rules):

```bash
bash scripts/run_v5_brenda_sabio_hp_sweep.sh           # submit all 12
bash scripts/run_v5_brenda_sabio_hp_sweep.sh dryrun    # print only
bash scripts/run_v5_brenda_sabio_hp_sweep.sh kcat_km   # only kcat_km
```

Reads: `v5_rankbind/configs/sweeps/hp_brenda_sabio/*.json`.
Writes: 12 new run dirs under `results/v5_rankbind/*bs_v2_hp*`,
each with the standard `manifest.json` + `score_matrix_rankbind.npy`
+ `test_summary.json`.

---

## 4. Provenance: what is *not* re-runnable from this repo alone

- AlphaFold structures under `~/hpc/structures/` are too large to
  redistribute in the repo. These are AlphaFold v6 PDBs keyed by
  UniProt accession, pulled by `reactionDataFiltering/hpc/...`.
- ESM2 per-residue embeddings under
  `reactionDataFiltering/data/interim/.../esm2_embeddings/` are also
  too large to redistribute. Regenerable from the sequences via the
  submodule's ESM2 batch script in the `~/venvs/esm2` venv.
- The BRENDA+SABIO 2026-04-29 raw snapshot is the dataset cut
  pinned by the submodule commit; the full snapshot tarball
  (`brenda_sabio_datasets_2026-04-29.tar.gz`, ~650 MB) lives next to
  the parent repo on disk and is not tracked.

The paper does not require any of these to be regenerated: every
result is downstream of pinned `score_matrix_*.npy` artefacts, and
the regeneration commands above only need the trained checkpoints
and the dataset CSVs.

---

## 5. Sanity checklist after a clean clone

```bash
# Code + submodule on disk and on the right branch:
git status                 # working tree clean
cd reactionDataFiltering && git rev-parse HEAD && cd ..

# Environment loads:
python - <<'PY'
import torch, transformers, sklearn, numpy, pandas
print(torch.__version__, torch.cuda.is_available(), transformers.__version__,
      sklearn.__version__, numpy.__version__, pandas.__version__)
PY

# Phase-1 score matrices present (paper §3 numbers depend on these):
ls results/original_*/best_model.pt
ls results/original_*/score_matrix_*.npy

# Phase-2 manifests for the three-seed v4 default present:
ls results/v5_rankbind/*_v4_s{42,7,1337}/manifest.json

# §8.1 transferability inputs present:
ls results/v5_rankbind/20260503-112034_*_kcat_km_with_decoys_bs_v1/manifest.json
ls results/v5_rankbind/20260503-112035_*_km_with_decoys_bs_v1/manifest.json
ls results/v5_rankbind/20260503-112035_*_turnover_with_decoys_bs_v1/manifest.json
```

If all of the above succeed, every command in §3 of this file is
runnable end-to-end and the paper's numbers will reproduce within
their reported standard deviations.