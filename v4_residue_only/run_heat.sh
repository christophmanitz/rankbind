#!/bin/bash
#SBATCH --job-name=heat_residue
#SBATCH --output=logs_v2/heat_%j.out
#SBATCH --error=logs_v2/heat_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --mail-user=christoph.manitz@uni-leipzig.de
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$SCRIPT_DIR/logs_v2"
cd "$SCRIPT_DIR"

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

source "$HOME/venvs/hieratombind/bin/activate"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo ""

python heat.py

echo ""
echo "Done at $(date)"
