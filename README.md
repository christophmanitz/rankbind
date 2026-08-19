# RankBind

Research repository for *RankBind: Protein-Invariant Contrastive Learning
for Ligand-Conditional DTI* (working title).

The paper draft (`paper/main.tex`, `paper/paper.md`) is the canonical
description of the work; this README is just a navigational index. Every
numerical result in the paper is reproducible end-to-end from this
repository at the current commit, see `REPRODUCIBILITY.md` for the
exact commands.

---

## 1. Paper at a glance

The paper makes four contributions, located as follows in the source
tree:

| Contribution                                                       | Code lives in                              | Pinned artefacts under                                            | Paper section |
|--------------------------------------------------------------------|--------------------------------------------|-------------------------------------------------------------------|---------------|
| **(C1) Null-baseline diagnosis** of pooled-AUC shortcut on BRENDA  | `evaluation/null_baselines.py`, `evaluation/attractor_diagnosis.py`, `evaluation/cross_model_overlap.py`, `evaluation/test_set_eval.py` | `results/original_*/score_matrix_*.npy` + `evaluation/attractor_results/{test_summary_all,gini_comparison,cross_model_overlap}.csv` | §3 |
| **(C2) RankBind architecture** (sampler + margin + bilinear + hard-negs) | `v5_rankbind/{data,sampler,model,loss,metrics,train,eval}.py` + `v5_rankbind/configs/` | `results/v5_rankbind/*_v4_s{42,7,1337}/` (3 seeds × 5 ablations)  | §4-§6 |
| **(C3) Residue-level extension** (attention pool over per-residue ESM2) | `v5_rankbind/configs/abl_attn_pool.json` (model already in `v5_rankbind/model.py`); audit in `evaluation/attn_weight_inspection.py` | `results/v5_rankbind/*_v5b_s*/` + `paper/figures/fig_attn_*.png`  | §7 |
| **(C4) Transferability probe + practitioner recipe** on enzyme-wide BRENDA+SABIO | `evaluation/null_prior_probe_brenda_sabio.py`, `v4_residue_only/{train_brenda_sabio.py,dataset_split.py,run_brenda_sabio.sh}` | `evaluation/attractor_results/null_prior_probe_brenda_sabio.{csv,txt}` + the three `*_with_decoys_bs_v1` v5 run dirs | §8.1 + §8.2 |

The Stage-1 hyperparameter sweep submitted alongside this commit
(`scripts/run_v5_brenda_sabio_hp_sweep.sh`, twelve configs under
`v5_rankbind/configs/sweeps/hp_brenda_sabio/`) is not in the paper's
headline results; how its outcomes are folded back into the paper is
documented in `docs/HP_SWEEP_INTEGRATION_PLAN.md`.

---

## 2. Layout

```
v5_rankbind/        Phase-2 ranking model — the paper's headline
                    architecture. Data, sampler, model, loss, metrics,
                    train/eval, configs (datasets/, sweeps/), run-manifest
                    helpers.
baselines/          Phase-1 adapters + unified trainer (GraphDTA,
                    MolTrans, DrugBAN, GEMS). Used to produce the four
                    score matrices §3.3 cites.
evaluation/         Diagnostic stack — null baselines (BRENDA-200 + a
                    BRENDA+SABIO-aware variant), attractor diagnosis,
                    cross-model overlap, test-set eval, attention-weight
                    audit, publication figures.
v4_residue_only/    The Phase-0 residue-only training stack, retained
                    because §8.1 reuses it for the BRENDA+SABIO v4 runs
                    (drop-in runner: train_brenda_sabio.py +
                    dataset_split.py + run_brenda_sabio.sh).
scripts/            SLURM submission scripts and aggregators
                    (run_v5_rankbind.sh, run_v5_multiseed.sh,
                    run_v5_ablations.sh, run_v5_brenda_sabio_hp_sweep.sh,
                    aggregate_multiseed.py, collect_v5_runs.py).
paper/              LaTeX + Markdown draft, figures, references.bib,
                    Makefile. `poster_scads/` holds the ScaDS.AI poster
                    (sources + `build.sh`), `poster_figure_data/` the
                    exported CSV evidence behind its figures.
docs/               Historical plans + the HP-sweep integration plan.

reactionDataFiltering/   git submodule — the dataset pipeline
                         (BRENDA + SABIO-RK -> filter -> decoys ->
                         augment -> structures -> sequences ->
                         embeddings -> graphs). Has its own GitHub
                         remote and CI; see its README.

CLAUDE.md           Live session-by-session status + do/don't list.
                    Read first if you are continuing a paused session.
README.md           This file. Paper-result-to-code index.
REPRODUCIBILITY.md  Exact commands per paper finding.

data/, results/, logs/, external/, *.tar.gz, *.pt, *.npy
                    gitignored. Regenerable from code + the dataset
                    pipeline. See REPRODUCIBILITY.md §4 for what is
                    regenerable from this repo alone vs. external.
```

---

## 3. Setup

The repository depends on the `reactionDataFiltering/` submodule;
clone with `--recurse-submodules`, or initialise after a plain clone:

```bash
# Fresh clone (recommended):
git clone --recurse-submodules https://github.com/christophmanitz/rankbind

# Already cloned without --recurse-submodules:
git submodule update --init --recursive
```

The dataset pipeline lives entirely in the submodule. Without it,
training/eval will fail when looking up sequence CSVs and ESM2
embeddings under `reactionDataFiltering/data/interim/`.

### Environment (Leipzig HPC)

```bash
module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
python -m venv ~/venvs/hieratombind && source ~/venvs/hieratombind/bin/activate
pip install -r requirements.txt
```

The `hieratombind` venv covers the v5_rankbind training/eval stack
and the evaluation diagnostics. DrugBAN needs its own venv
(`~/venvs/drugban`); ESM2 embedding generation in the submodule
uses `~/venvs/esm2`.

### Smoke check after clone

See `REPRODUCIBILITY.md` §5.

---

## 4. Working with the submodule

Day-to-day editing of dataset-pipeline code happens inside the
submodule, not from the parent repo:

```bash
cd reactionDataFiltering
git checkout main
git pull
# ... edit, test, commit, push as usual
```

After you've pushed a change in the submodule, the parent rankbind
repo will show the submodule as "modified" because the pinned
commit is no longer the submodule's HEAD. To bump the pin:

```bash
git status                          # shows: modified: reactionDataFiltering
git diff reactionDataFiltering      # shows old hash -> new hash
git add reactionDataFiltering
git commit -m "bump reactionDataFiltering to <new-hash>"
```

That commit in rankbind records "this version of the research code
goes with this version of the dataset pipeline", which is
the whole point of using a submodule instead of two unrelated clones.

### Pulling someone else's submodule update

```bash
git pull
git submodule update --init --recursive  # check out the new pinned hash
```

If you forget the second step, you'll be running new parent code
against an old submodule and may see weird import or path errors.

### Common gotchas

- Detached HEAD inside the submodule. After `git submodule
  update`, the submodule is at a specific commit, not on a branch.
  Before editing files in there, do `cd reactionDataFiltering &&
  git checkout main` so your future commits land on a branch.
- Forgot `--recurse-submodules` on clone. `reactionDataFiltering/`
  is empty. Fix with `git submodule update --init --recursive`.
- Pushed in submodule, forgot to bump in parent. Collaborators
  will check out the *old* pinned hash and not see your changes.
  Always pair the two pushes.

---

## 5. Documentation surface

| File | What it is |
|------|------------|
| [`paper/main.tex`](paper/main.tex), [`paper/paper.md`](paper/paper.md) | The paper draft. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Per-result regeneration commands. |
| [`CLAUDE.md`](CLAUDE.md) | Live status + do/don't list. Read first if continuing a paused session. |
| [`docs/HP_SWEEP_INTEGRATION_PLAN.md`](docs/HP_SWEEP_INTEGRATION_PLAN.md) | Pre-registered decision rules for the in-flight HP sweep. |
| [`docs/development_plan.md`](docs/development_plan.md) | Pre-Phase-1 plan. Historical, do not edit. |
| [`docs/phase1_status.md`](docs/phase1_status.md) | Phase-1 outcome snapshot. |
| [`v5_rankbind/PHASE2_LOG.md`](v5_rankbind/PHASE2_LOG.md) | Phase-2 session log (live; appendable). |
| [`v5_rankbind/PLAN.md`](v5_rankbind/PLAN.md) | Pre-Phase-2 plan. Do not edit §1-§9, append addenda only. |
| [`reactionDataFiltering/README.md`](reactionDataFiltering/README.md) | Dataset-pipeline overview + CLI reference. |