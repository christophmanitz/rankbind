#!/bin/bash
# scripts/run_graphdta_recipe.sh — GraphDTA on the anti-shortcut recipe.
#
# Progressive ablation: {orig,a,b,c} x {brenda,turnover}, multi-seed on BRENDA.
# Each job trains GraphDTA under the RankBind v4 data regime
# (train_graphdta_recipe.py) and then runs the §8.3 null-baseline instruments
# (benchmark_null_eval.py) on the produced run dir. Reuses the submit_pyg_model
# sbatch header from run_original_baselines.sh (paula, gpu:1, 48G, hieratombind).
#
#   orig = random-shuffle BCE, NO balanced sampler  (within-GraphDTA floor)
#   a    = ProteinBalancedSampler + BCE             (+ balanced sampling)
#   b    = + within-ligand margin loss (random cross-protein negatives)
#   c    = + hard-negative mining
#
# Selection/early-stop runs on val matrix MRR (the ranking objective), not the
# val global AUC shortcut — see train_graphdta_recipe.py. The orig→a→b→c trend
# isolates each recipe ingredient on a fixed, mediocre architecture.
#
# Usage:
#   bash scripts/run_graphdta_recipe.sh [orig|a|b|c|all] [brenda|turnover|both] [seeds]
# Defaults: all variants, both datasets, seeds per-dataset (BRENDA 42 7 1337,
# turnover 42). `seeds` overrides both datasets if given, e.g. "42".

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

VENV_MAIN="$HOME/venvs/hieratombind"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

WHICH_VARIANT="${1:-all}"
WHICH_DATA="${2:-both}"
SEEDS_OVERRIDE="${3:-}"

EPOCHS=100
BATCH_SIZE=32
LR=1e-4
N_MATRIX=200
PAIRS_PER_PROTEIN=16
N_NEGATIVES=4
HARD_POOL=50
MIN_EPOCHS=20

# Canonical seed: its run dirs carry NO seed suffix (a_brenda, …) so the eval
# wiring (matrix_per_ligand_auc_all.py RECIPE_RUNS, graphdta_recipe_table.py)
# reads them unchanged. Extra seeds get a _s<seed> suffix for error bars.
CANON_SEED=42
BRENDA_SEEDS="42 7 1337"
TURNOVER_SEEDS="42"

# Dataset paths
BRENDA_CSV="data/dataset_with_decoys.csv"
BRENDA_SEQ="data/sequences/sequences.csv"
TURN_CSV="reactionDataFiltering/data/interim/turnover_brenda_sabio/with_decoys.csv"
TURN_SEQ="reactionDataFiltering/data/interim/turnover_brenda_sabio/sequences.csv"

VARIANTS="orig a b c"

submit_recipe() {
    local VARIANT=$1     # orig | a | b | c
    local DATASET=$2     # brenda | turnover
    local CSV=$3
    local SEQ=$4
    local WALLTIME=$5
    local SEED=$6
    local EXTRA=$7       # hard-neg knobs

    local SUFFIX=""
    [ "$SEED" != "$CANON_SEED" ] && SUFFIX="_s${SEED}"
    local TAG="${VARIANT}_${DATASET}${SUFFIX}"
    local JOBNAME="gdta_recipe_${TAG}"
    local OUT_DIR="$PROJECT_ROOT/results/graphdta_recipe/${TAG}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${LOG_DIR}/${JOBNAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOBNAME}_%j.err
#SBATCH --time=${WALLTIME}
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=paula

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0
source "${VENV_MAIN}/bin/activate"

echo "Host: \$(hostname) | Date: \$(date)"
echo "Recipe variant=${VARIANT} dataset=${DATASET} seed=${SEED}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT/baselines/adapters"
python train_graphdta_recipe.py \\
    --variant ${VARIANT} \\
    --csv_path "${CSV}" \\
    --seq_csv "${SEQ}" \\
    --out_dir "${OUT_DIR}" \\
    --tag ${TAG} \\
    --seed ${SEED} \\
    --epochs ${EPOCHS} \\
    --batch_size ${BATCH_SIZE} \\
    --lr ${LR} \\
    --n_matrix ${N_MATRIX} \\
    --pairs_per_protein ${PAIRS_PER_PROTEIN} \\
    --n_negatives ${N_NEGATIVES} \\
    --hard_pool_size ${HARD_POOL} \\
    --early_stop_min_epochs ${MIN_EPOCHS} \\
    ${EXTRA}

echo "Running §8.3 null-baseline instruments…"
cd "$PROJECT_ROOT"
python evaluation/benchmark_null_eval.py "${OUT_DIR}" \\
    --out "$PROJECT_ROOT/evaluation/attractor_results/graphdta_recipe_null_${TAG}.csv"

echo "Done: ${JOBNAME}"
EOF
    echo "Submitted: ${JOBNAME} -> ${OUT_DIR}"
}

run_dataset() {
    local DATASET=$1 CSV=$2 SEQ=$3 WALLTIME=$4 SEEDS=$5 EXTRA=$6
    for V in $VARIANTS; do
        if [ "$WHICH_VARIANT" = "all" ] || [ "$WHICH_VARIANT" = "$V" ]; then
            for SEED in $SEEDS; do
                # hard-neg knobs only matter for variant c
                local EXTRA_V=""
                [ "$V" = "c" ] && EXTRA_V="$EXTRA"
                submit_recipe "$V" "$DATASET" "$CSV" "$SEQ" "$WALLTIME" "$SEED" "$EXTRA_V"
            done
        fi
    done
}

# BRENDA (~618 train proteins): full hard-neg refresh every epoch (the separable
# GCN refresh makes full coverage cheap — no caps). Multi-seed for error bars.
if [ "$WHICH_DATA" = "both" ] || [ "$WHICH_DATA" = "brenda" ]; then
    SEEDS="${SEEDS_OVERRIDE:-$BRENDA_SEEDS}"
    run_dataset brenda "$BRENDA_CSV" "$BRENDA_SEQ" "06:00:00" "$SEEDS" \
        "--hard_refresh_every 1"
fi

# turnover (~5.7k train proteins, thousands of positive ligands): the separable
# refresh encodes each ligand/protein once, so FULL-coverage refresh every epoch
# costs seconds — no caps (was prot_cap/lig_cap; dropped). Single canonical seed.
if [ "$WHICH_DATA" = "both" ] || [ "$WHICH_DATA" = "turnover" ]; then
    SEEDS="${SEEDS_OVERRIDE:-$TURNOVER_SEEDS}"
    run_dataset turnover "$TURN_CSV" "$TURN_SEQ" "10:00:00" "$SEEDS" \
        "--hard_refresh_every 1"
fi

echo "All requested jobs submitted."
