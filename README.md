# RankBind

Research workspace for *RankBind: Protein-Invariant Contrastive Learning
for Ligand-Conditional DTI* (working title). This repository holds the
human-authored Python source, SLURM submission scripts, configs,
diagnostics and paper draft. Bulk artefacts (datasets, model
checkpoints, run outputs, logs) are produced by the code in here but
are not tracked.

For session-by-session deltas and the live status, read
[`CLAUDE.md`](CLAUDE.md). For the pre-Phase-1 plan,
[`development_plan_rankbind.md`](development_plan_rankbind.md). For
the Phase-2 log,
[`v5_rankbind/PHASE2_LOG.md`](v5_rankbind/PHASE2_LOG.md).

## Layout

```
v5_rankbind/        Phase-2 ranking model (data, sampler, model, loss,
                    metrics, train/eval). Configs under configs/.
baselines/          Phase-1 adapters + unified trainer (GraphDTA,
                    MolTrans, DrugBAN, GEMS).
evaluation/         Diagnostic stack — null baselines, attractor
                    diagnosis, cross-model overlap, figures.
scripts/            SLURM submission scripts and aggregators.
paper/              LaTeX + Markdown draft and figures.
v4_residue_only/    Deprecated Phase-0 code, kept for reference.

reactionDataFiltering/   git submodule — the dataset pipeline
                         (BRENDA + SABIO-RK -> filter -> decoys ->
                         augment -> structures -> sequences ->
                         embeddings -> graphs). Has its own GitHub
                         remote and CI; see its README.

data/, results/, logs/, external/, *.tar.gz   gitignored. Regenerable
                                              from code + the dataset
                                              pipeline.
```

## Setup

This repository depends on the `reactionDataFiltering/` submodule —
clone with `--recurse-submodules`, or initialise after a plain clone:

```bash
# Fresh clone (recommended):
git clone --recurse-submodules <rankbind-url>

# Already cloned without --recurse-submodules:
cd rankbind
git submodule update --init --recursive
```

The dataset pipeline lives entirely in the submodule. Without it,
training/eval will fail when looking up sequence CSVs and ESM2
embeddings under `reactionDataFiltering/data/interim/`.

### Environment (Leipzig HPC)

```bash
module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
source ~/venvs/hieratombind/bin/activate
```

The `hieratombind` venv covers the v5_rankbind training/eval stack
and the evaluation diagnostics. DrugBAN needs its own venv
(`~/venvs/drugban`); ESM2 embedding generation in the submodule
uses `~/venvs/esm2`.

## Working with the submodule

Day-to-day editing of dataset-pipeline code happens **inside** the
submodule, not from the parent repo:

```bash
cd reactionDataFiltering
# normal git workflow — commits and pushes go to the
# reactionDataFiltering remote on GitHub
git checkout main
git pull
# ... edit, test, commit, push as usual
```

After you've pushed a change in the submodule, the parent rankbind
repo will show the submodule as "modified" because the pinned
commit is no longer the submodule's HEAD. To bump the pin:

```bash
cd ~/rankbind
git status                          # shows: modified: reactionDataFiltering
git diff reactionDataFiltering      # shows old hash -> new hash
git add reactionDataFiltering
git commit -m "bump reactionDataFiltering to <new-hash>"
```

That commit in rankbind records "this version of the research code
goes with this version of the dataset pipeline" — which is the
whole point of using a submodule instead of two unrelated clones.

### Pulling someone else's submodule update

```bash
git pull
git submodule update --init --recursive  # check out the new pinned hash
```

If you forget the second step, you'll be running new parent code
against an old submodule and may see weird import or path errors.

### Common gotchas

- **Detached HEAD inside the submodule.** After `git submodule
  update`, the submodule is at a specific commit, not on a branch.
  Before editing files in there, do `cd reactionDataFiltering &&
  git checkout main` so your future commits land on a branch.
- **Forgot `--recurse-submodules` on clone.** `reactionDataFiltering/`
  is empty. Fix with `git submodule update --init --recursive`.
- **Pushed in submodule, forgot to bump in parent.** Collaborators
  will check out the *old* pinned hash and not see your changes.
  Always pair the two pushes.

## Documentation surface

- [`CLAUDE.md`](CLAUDE.md) — live status + do/don't list (read first).
- [`development_plan_rankbind.md`](development_plan_rankbind.md) —
  pre-Phase-1 plan. Historical, do not edit.
- [`PHASE1_STATUS.md`](PHASE1_STATUS.md) — Phase-1 outcome snapshot.
- [`v5_rankbind/PHASE2_LOG.md`](v5_rankbind/PHASE2_LOG.md) — Phase-2
  session log.
- [`v5_rankbind/PLAN.md`](v5_rankbind/PLAN.md) — pre-Phase-2 plan
  (do not edit §1-§9, append addenda only).
- [`reactionDataFiltering/README.md`](reactionDataFiltering/README.md)
  — dataset pipeline overview + CLI reference.
