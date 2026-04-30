#!/bin/bash
#SBATCH --job-name=fetch_sequences
#SBATCH --output=logs/fetch_sequences_%j.out
#SBATCH --error=logs/fetch_sequences_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=paul

# Fetch protein sequences from UniProt REST API for all 903 proteins.
# Usage: sbatch scripts/run_fetch_sequences.sh  (from rankbind/ root)

set -euo pipefail

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/logs" "$SCRIPT_DIR/data/sequences"

module purge
module load GCC/11.3.0
module load Python/3.10.4-GCCcore-11.3.0

source "$HOME/venvs/hieratombind/bin/activate"

echo "Host:  $(hostname)"
echo "Date:  $(date)"
echo "Dir:   $(pwd)"

python scripts/fetch_sequences.py \
    --batch_size 50 \
    --delay 0.3 \
    --resume

echo "Done. Sequences in data/sequences/"
