#!/bin/bash
#SBATCH --job-name=residue_v2_decoys
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs_v2/train_%j.out
#SBATCH --error=logs_v2/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=christoph.manitz@uni-leipzig.de

set -euo pipefail

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$SCRIPT_DIR/logs_v2" "$SCRIPT_DIR/checkpoints_v2" "$SCRIPT_DIR/plots_v2"
cd "$SCRIPT_DIR"

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

source "$HOME/venvs/hieratombind/bin/activate"

echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"
echo "Model:   ResidueOnlyBind v2 — with 4000 shuffled decoys"
echo ""

PROTEIN_DIR="/home/sc.uni-leipzig.de/zw93onug/hpc/structures"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CSV_PATH="$PROJECT_ROOT/data/dataset_with_decoys.csv"
DATASET_ROOT="$PROJECT_ROOT/data"

python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
import train_v2_decoys as t
# Override config-level paths that train.py imports its settings from
import train as _t
_t.CSV_PATH     = '$CSV_PATH'
_t.PROTEIN_DIR  = '$PROTEIN_DIR'
_t.DATASET_ROOT = '$DATASET_ROOT'
t.CSV_PATH      = '$CSV_PATH'
t.PROTEIN_DIR   = '$PROTEIN_DIR'
t.DATASET_ROOT  = '$DATASET_ROOT'
t.main()
"

echo ""
echo "Training finished at $(date)"
