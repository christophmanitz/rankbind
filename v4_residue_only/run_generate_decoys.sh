#!/bin/bash
#SBATCH --job-name=gen_decoys
#SBATCH --output=logs/gen_decoys_%j.out
#SBATCH --error=logs/gen_decoys_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=paula
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --mail-user=christoph.manitz@uni-leipzig.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR"

module purge
module load GCC/11.3.0
module load Python/3.10.4-GCCcore-11.3.0

source "$HOME/venvs/hieratombind/bin/activate"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

python generate_decoys.py

echo ""
echo "Done at $(date)"
