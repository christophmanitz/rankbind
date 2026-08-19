# Dataset configs (BRENDA + SABIO-RK, 2026-04-29 bundle)

Six v5_rankbind configs for the cross-dataset comparison: three kinetic
parameters × {with decoys, without decoys}. All configs extend
`../default.json`; only the `data` block and a `dataset_meta` annotation
differ between them.

| Config                       | Parameter         | Decoys | Pairs   | UniProts |
|------------------------------|-------------------|:------:|--------:|---------:|
| `kcat_km_with_decoys.json`   | `kcat_km`         |   ✓    | 12 527  | 3 815    |
| `kcat_km_no_decoys.json`     | `kcat_km`         |   —    | 12 527  | 3 815    |
| `km_with_decoys.json`        | `km_value`        |   ✓    | 29 797  | 9 500    |
| `km_no_decoys.json`          | `km_value`        |   —    | 29 797  | 9 500    |
| `turnover_with_decoys.json`  | `turnover_number` |   ✓    | 16 984  | 5 672    |
| `turnover_no_decoys.json`    | `turnover_number` |   —    | 16 984  | 5 672    |

The "with decoys" pair counts above are the underlying positive count;
each row gets ~5-15 decoy rows after augmentation (Pareto frontier
filtered to TS in 0.3..0.8). The exact augmented row counts will be
written to each `with_decoys.csv.manifest.json` once the decoy SLURM job
finishes.

## What the no-decoys variant does (and doesn't)

Without explicit decoy rows, every CSV row has `label=1`. The training
pipeline still works because the margin loss + cross-protein-implicit
sampler generates negatives at runtime: for each anchor `(L_real, P+)`
the sampler picks `k` other proteins from the batch as negatives for
the same ligand `L_real`. The implicit assumption is "any protein not
known to bind `L_real` is a negative example."

What does not work for no-decoys:
- `global_auc`: needs both classes in the test set, with positives only
  it is undefined. Report as N/A.
- `per_ligand_auc`: same reason. Report as N/A.

What still works (and is the paper-relevant story):
- Matrix-MRR, Hit@5, Hit@10 (these only need positive pairs to compute).
- Gini-residual vs `null_prot_prior`.
- Top-10 Jaccard vs `null_prot_prior`.

This makes the comparison "do explicit decoys help, or are implicit
cross-protein negatives enough?" a clean experiment along identical
training infrastructure, with the only difference being the row set.

## Prerequisites before running

1. Decoys + augmentation (only for `*_with_decoys.json`):
   `data/interim/<dataset>/with_decoys.csv` must exist. SLURM job
   `hpc/run_decoys.sh` (in `reactionDataFiltering/`) produces it.
2. Sequences CSV (both variants):
   `data/interim/<dataset>/sequences.csv`, produced by
   `reaction-data sequences` (already done; see manifests).
3. ESM2 embeddings (both variants):
   `data/interim/<dataset>/esm2_embeddings/{uniprot}.pt`, produced by
   `hpc/run_embeddings.sh` (in `reactionDataFiltering/`); 1× GPU job per
   dataset.

## Running a single dataset

```bash
module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
source ~/venvs/hieratombind/bin/activate
cd ~/rankbind

bash scripts/run_v5_rankbind.sh \
    configs/datasets/kcat_km_with_decoys.json \
    paula \
    bs_v1
```

Tag convention for these runs: `bs_v1` (BRENDA-Sabio first run),
`bs_v1_s7` etc. for seed sweeps.

## What `dataset_meta` records

Each config carries a `dataset_meta` block that is *not* consumed by the
training pipeline; it is a documentation hook so a stranger reading the
config knows:

- `source_dataset_dir`: where the raw CSVs came from (provenance).
- `parameter`: which BRENDA/SABIO kinetic parameter this is.
- `decoys`: boolean.
- `decoy_ts_min` / `decoy_ts_max`: the Tanimoto window applied during
  decoy augmentation (omit if `decoys=false`).
- `label_source`: exactly which column drove the label. This matters
  because `BRENDADataConfig.load_pairs` falls through several columns.
- `valid_metrics`: metrics that are well-defined for this config.
- `invalid_metrics_note`: for no-decoy variants, an explicit reminder
  that AUC-style metrics are N/A.
