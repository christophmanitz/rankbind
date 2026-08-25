# RankBind: Ligand-Conditional DTI Ranking

A ranking-based approach to drug–target interaction that overcomes the
pooled-AUC shortcut on enzyme–substrate benchmarks. RankBind replaces
pairwise classification with within-ligand margin ranking, hard-negative
mining, and protein-balanced sampling.

```bash
git clone --recurse-submodules https://github.com/christophmanitz/rankbind
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for how to reproduce
every result from a fresh clone.

---

## Key results

| Model | Pooled AUC | Per-ligand AUC | Matrix MRR | Hit@10 |
|-------|-----------:|---------------:|-----------:|-------:|
| BCE control | 0.918 | 0.500 | 0.014 | 0.000 |
| **RankBind** (3 seeds) | 0.618 | 0.878 | **0.220** | **0.598** |
| + residue attention | 0.646 | — | 0.316 | 0.706 |

Pooled AUC drops because RankBind stops exploiting the protein-popularity
shortcut. Per-ligand ranking and MRR improve dramatically.

---

## Setup

Requires the `reactionDataFiltering/` submodule:

```bash
git clone --recurse-submodules https://github.com/christophmanitz/rankbind
# or, after a plain clone:
git submodule update --init --recursive
```

### Environment (Leipzig HPC)

```bash
module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
python -m venv ~/venvs/hieratombind && source ~/venvs/hieratombind/bin/activate
pip install -r requirements.txt
```

DrugBAN needs its own venv (`~/venvs/drugban`).

---

## Repository structure

```
v5_rankbind/        Ranking model — data, sampler, model, loss, metrics,
                    train/eval, configs.
baselines/          Phase-1 adapters + unified trainer (GraphDTA,
                    MolTrans, DrugBAN, GEMS).
evaluation/         Diagnostic stack — null baselines, attractor diagnosis,
                    cross-model overlap, test-set eval, attention audit,
                    publication figures.
v4_residue_only/    Residue-only training stack (BRENDA+SABIO runs).
scripts/            SLURM submission scripts and aggregators.
docs/               Plans and integration documentation.

reactionDataFiltering/   git submodule — dataset pipeline
                         (BRENDA + SABIO-RK → filter → decoys →
                         augment → structures → sequences → embeddings).
```

---

## Training

Single run:

```bash
bash scripts/run_v5_rankbind.sh default paula v4
```

Multi-seed sweep:

```bash
bash scripts/run_v5_multiseed.sh
```

All configs live in `v5_rankbind/configs/`. The default config
(`default.json`) uses bilinear rank 128, margin loss with k=4, m=1.0,
and hard-negative pool size 50.

---

## Documentation

| File | Description |
|------|-------------|
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Per-result regeneration commands. |
| [`docs/development_plan.md`](docs/development_plan.md) | Pre-Phase-1 plan (historical). |
| [`docs/phase1_status.md`](docs/phase1_status.md) | Phase-1 outcome snapshot. |
| [`v5_rankbind/PHASE2_LOG.md`](v5_rankbind/PHASE2_LOG.md) | Phase-2 session log. |
| [`v5_rankbind/PLAN.md`](v5_rankbind/PLAN.md) | Pre-Phase-2 plan (historical). |
| [`reactionDataFiltering/README.md`](reactionDataFiltering/README.md) | Dataset pipeline reference. |
