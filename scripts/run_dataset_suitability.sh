#!/bin/bash
#SBATCH --job-name=dataset_suitability
#SBATCH --output=logs/dataset_suitability_%j.out
#SBATCH --error=logs/dataset_suitability_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=paul

# Run dataset suitability analysis — no GPU needed
# Usage: sbatch scripts/run_dataset_suitability.sh  (from rankbind/ root)

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/evaluation/suitability_results"

module purge
module load GCC/11.3.0
module load Python/3.10.4-GCCcore-11.3.0

source "$HOME/venvs/hieratombind/bin/activate"

echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "Python: $(which python)"

python evaluation/dataset_suitability.py

echo "Done. Results in evaluation/suitability_results/"
