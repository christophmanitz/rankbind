#!/bin/bash
# scripts/run_original_baselines.sh — Submit original baseline training jobs
#
# Prerequisites:
#   - GraphDTA, MolTrans, GIGN, GEMS: hieratombind venv (torch 2.8 + PyG)
#   - DrugBAN: drugban venv (torch 2.4 + DGL + DGLLife)
#   - GIGN: docked structures in data/docked_complexes/
#   - GEMS: ESM2 embeddings in data/esm2_embeddings/ (optional, uses zeros otherwise)
#
# Usage:
#   bash scripts/run_original_baselines.sh [graphdta|moltrans|gign|gems|drugban|all]

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

VENV_MAIN="$HOME/venvs/hieratombind"
VENV_DRUGBAN="$HOME/venvs/drugban"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

EPOCHS=100
BATCH_SIZE=32
N_MATRIX=200
LR=1e-4

submit_pyg_model() {
    local MODEL=$1
    local JOBNAME="orig_${MODEL}"
    local OUT_DIR="$PROJECT_ROOT/results/original_${MODEL}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${LOG_DIR}/${JOBNAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOBNAME}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=paula

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

source "${VENV_MAIN}/bin/activate"

echo "Host:  \$(hostname)"
echo "Date:  \$(date)"
echo "Model: ${MODEL} (original repo)"
echo "GPUs:  \$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT/baselines/adapters"
python train_original.py \\
    --model ${MODEL} \\
    --out_dir "${OUT_DIR}" \\
    --epochs ${EPOCHS} \\
    --batch_size ${BATCH_SIZE} \\
    --lr ${LR} \\
    --n_matrix ${N_MATRIX}

echo "Done: ${MODEL}"
EOF
    echo "Submitted: ${JOBNAME}"
}

submit_drugban() {
    local JOBNAME="orig_drugban"
    local OUT_DIR="$PROJECT_ROOT/results/original_drugban"

    if [ ! -d "$VENV_DRUGBAN" ]; then
        echo "ERROR: DrugBAN venv not found at $VENV_DRUGBAN"
        echo "  Run: bash scripts/setup_drugban_venv.sh"
        return 1
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${LOG_DIR}/${JOBNAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOBNAME}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=paula

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

source "${VENV_DRUGBAN}/bin/activate"

echo "Host:  \$(hostname)"
echo "Date:  \$(date)"
echo "Model: DrugBAN (original repo, DGL venv)"
echo "GPUs:  \$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT/baselines/adapters"
python train_original_drugban.py \\
    --out_dir "${OUT_DIR}" \\
    --epochs ${EPOCHS} \\
    --batch_size ${BATCH_SIZE} \\
    --lr ${LR} \\
    --n_matrix ${N_MATRIX}

echo "Done: DrugBAN"
EOF
    echo "Submitted: ${JOBNAME}"
}

# Parse argument
TARGET="${1:-all}"

case "$TARGET" in
    graphdta)  submit_pyg_model graphdta ;;
    moltrans)  submit_pyg_model moltrans ;;
    gign)      submit_pyg_model gign ;;
    gems)      submit_pyg_model gems ;;
    drugban)   submit_drugban ;;
    all)
        submit_pyg_model graphdta
        submit_pyg_model moltrans
        submit_pyg_model gign
        submit_pyg_model gems
        submit_drugban
        ;;
    *)
        echo "Usage: $0 [graphdta|moltrans|gign|gems|drugban|all]"
        exit 1
        ;;
esac
