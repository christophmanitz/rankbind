# Baseline Reproduction Record (skill item A0)

Goal: the Phase-1 baseline numbers in the paper must remain reproducible
from pinned artifacts, without retraining.

## What is frozen

- **Checkpoints + score matrices**: `results/original_{graphdta,moltrans,drugban,gems}/best_model.pt`
  and `score_matrix*.npy` per model. SHA256-pinned in
  [`revision_v0/FREEZE_RECORD.md`](revision_v0/FREEZE_RECORD.md).
- **Code state**: git commit `4ff68dc` (public GitHub snapshot) contains all
  training code for the four baselines. Nothing under `baselines/` has been
  modified since; policy is *do not retrain* (see CLAUDE.md "Don't" list).
- **Split**: every model and diagnostic uses the unified protein-based split,
  `baselines/adapters/common.py::BRENDADataConfig.get_protein_split()`,
  seed=42, val_frac=test_frac=0.15 (i.e. 70/15/15). Do not re-split.

## How to reproduce each baseline's reported numbers

| Model | Trainer | Env |
|---|---|---|
| GraphDTA | `baselines/adapters/train_original.py --model graphdta` | `~/venvs/hieratombind` (+ PyG) |
| MolTrans | `baselines/adapters/train_original.py --model moltrans` | `~/venvs/hieratombind` |
| GEMS | `baselines/adapters/train_original.py --model gems` | `~/venvs/hieratombind` (+ ESM2 weights) |
| DrugBAN | `baselines/adapters/train_original_drugban.py` | `~/venvs/drugban` (torch 2.4.0+cu124, DGL) |

Environment: `module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0`,
then the venv above. Hardware: single NVIDIA A30 (`--gres=gpu:1`, partition
paula). Note: after the 2026-04-29 venv rebuild, PyG is NOT installed in
`~/venvs/hieratombind`; reproducing GraphDTA/GIGN-style graph models requires
reinstalling PyG against torch 2.8 first.

Evaluation of a trained checkpoint → score matrix → paper tables:

```bash
python evaluation/test_set_eval.py        # global AUC / per-ligand AUC / Hit@K
python evaluation/attractor_diagnosis.py  # Gini / attractor scores
python evaluation/null_baselines.py       # null_prot_prior etc., same 200x200 pool
python evaluation/cross_model_overlap.py  # top-K Jaccard vs null_prot_prior
```

## Reported Phase-1 numbers (source CSVs)

- `evaluation/attractor_results/test_summary_all.csv`
- `evaluation/attractor_results/cross_model_overlap.csv`
- `evaluation/attractor_results/gini_comparison.csv`

| Model    | Global AUC | Per-ligand AUC | Top-10 Jaccard vs null_prot_prior |
|----------|-----------:|---------------:|----------------------------------:|
| DrugBAN  | 0.954      | 0.375          | 0.54                              |
| MolTrans | 0.937      | 0.500          | 0.05                              |
| GraphDTA | 0.869      | 0.625          | 0.67                              |
| GEMS     | 0.633      | 0.250          | 0.67                              |

Gini for all four ≈ 0.995, identical to `null_prot_prior` — the Phase-1 pivot.
