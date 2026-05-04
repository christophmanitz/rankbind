#!/bin/bash
#SBATCH --job-name=v4res_bs
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=runs_brenda_sabio/slurm_%x_%j.out
#SBATCH --error=runs_brenda_sabio/slurm_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=christoph.manitz@uni-leipzig.de

set -euo pipefail

DATASET="${1:?usage: sbatch -J v4res_bs_<dataset> run_brenda_sabio.sh <dataset> [<tag>]}"
TAG="${2:-bs_v1}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
cd "$SCRIPT_DIR"
mkdir -p runs_brenda_sabio

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0
source "$HOME/venvs/hieratombind/bin/activate"

echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "Dataset: $DATASET"
echo "Tag:     $TAG"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo

python -u train_brenda_sabio.py --dataset "$DATASET" --tag "$TAG"

echo
echo "Finished at $(date)"
