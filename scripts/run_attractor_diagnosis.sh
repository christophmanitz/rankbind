#!/bin/bash
#SBATCH --job-name=attractor_v4
#SBATCH --output=logs/attractor_v4_%j.out
#SBATCH --error=logs/attractor_v4_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=paula

# Compute attractor bias metrics for ResidueOnlyBind v4.
# Usage: sbatch scripts/run_attractor_diagnosis.sh  (from rankbind/ root)
# Override: MODEL_PATH=... N_LIGANDS=300 sbatch scripts/run_attractor_diagnosis.sh

set -euo pipefail

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/evaluation/attractor_results"

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

source "$HOME/venvs/hieratombind/bin/activate"

echo "Host:  $(hostname)"
echo "Date:  $(date)"
echo "GPUs:  $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

MODEL_PATH="${MODEL_PATH:-v4_residue_only/checkpoints/best_model.pt}"
MODEL_NAME="${MODEL_NAME:-ResidueOnlyBind_v4}"
N_LIGANDS="${N_LIGANDS:-200}"
N_PROTEINS="${N_PROTEINS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"

export HIERATOMBIND_ROOT="${HIERATOMBIND_ROOT:-$HOME/newclaudemodel}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Model:      $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Matrix:     $N_LIGANDS ligands × $N_PROTEINS proteins"
echo "HierRoot:   $HIERATOMBIND_ROOT"

python evaluation/attractor_diagnosis.py \
    --model_path   "$MODEL_PATH"  \
    --model_name   "$MODEL_NAME"  \
    --n_ligands    "$N_LIGANDS"   \
    --n_proteins   "$N_PROTEINS"  \
    --batch_size   "$BATCH_SIZE"  \
    --device       "$DEVICE"      \
    --out_dir      evaluation/attractor_results

echo "Done. Results in evaluation/attractor_results/"
