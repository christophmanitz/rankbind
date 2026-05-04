#!/bin/bash
# scripts/run_v5_rankbind.sh — Submit a RankBind training run as SLURM job.
#
# Usage:
#   bash scripts/run_v5_rankbind.sh                             # default, paula, no tag, cfg seed
#   bash scripts/run_v5_rankbind.sh abl_no_sampler              # any JSON under v5_rankbind/configs/
#   bash scripts/run_v5_rankbind.sh default paula v4            # tagged run, cfg seed
#   bash scripts/run_v5_rankbind.sh default paula v4 7          # tagged run, seed=7 (overrides cfg)
#
# Positional args:
#   1. CFG_NAME   — config file stem under v5_rankbind/configs/ (default: default)
#   2. PARTITION  — SLURM partition (default: paula)
#   3. TAG        — run tag appended to run_id; if SEED is set and tag does not
#                   already contain the seed, "_s<SEED>" is appended automatically
#   4. SEED       — integer; passed to train.py as --seed (overrides cfg seed).
#                   Used by the multi-seed sweep; leave empty to use cfg['seed'].
#
# Notes
# - Cluster: sc.uni-leipzig.de (paula preferred, clara fallback).
# - The script writes log files to rankbind/logs/v5_<cfg>_<jobid>.out/.err.
# - All outputs of the actual run land in results/v5_rankbind/<run_id>/
#   with a manifest.json. run_id is minted inside train.py.

set -euo pipefail

CFG_NAME="${1:-default}"
PARTITION="${2:-paula}"
TAG="${3:-}"
SEED="${4:-}"
WALLTIME="${5:-06:00:00}"  # SLURM HH:MM:SS; bump for big datasets (e.g. km ~10h)

# When a seed override is supplied, fold it into the tag so run_ids remain
# unique per (config, tag, seed). If the caller already included "_s7" etc.
# in the tag we do not double-append.
if [ -n "$SEED" ]; then
    case "$TAG" in
        *"_s${SEED}"|*"_s${SEED}_"*) ;;
        *) TAG="${TAG:+${TAG}}_s${SEED}" ;;
    esac
fi

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
CFG_PATH="$PROJECT_ROOT/v5_rankbind/configs/${CFG_NAME}.json"
if [ ! -f "$CFG_PATH" ]; then
    echo "ERROR: config not found: $CFG_PATH"
    exit 1
fi

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Slashes in CFG_NAME (e.g. sweeps/hp_brenda_sabio/foo) are invalid in SLURM
# job names and produce nonexistent log paths. Sanitize for these two uses,
# CFG_PATH is already resolved above so the actual config still loads fine.
JOBNAME="v5_${CFG_NAME//\//_}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${LOG_DIR}/${JOBNAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOBNAME}_%j.err
#SBATCH --time=${WALLTIME}
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}

PROJECT_ROOT="\$SLURM_SUBMIT_DIR"
cd "\$PROJECT_ROOT"

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0
source "\$HOME/venvs/hieratombind/bin/activate"

echo "Host:      \$(hostname)"
echo "Date:      \$(date)"
echo "Job:       \$SLURM_JOB_ID (\$SLURM_JOB_PARTITION)"
echo "Config:    ${CFG_PATH}"
echo "GPU:       \$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
echo "Python:    \$(python --version)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_LOG=\$(mktemp)
python -m v5_rankbind.train \\
    --config "${CFG_PATH}" \\
    --tag "${TAG}" \\
    ${SEED:+--seed "${SEED}"} 2>&1 | tee "\$TRAIN_LOG"

echo "Training done at \$(date)"

# Parse run_dir from train.py's own stdout — concurrent jobs each need their
# OWN run dir. The \`ls -td\` approach races with sibling jobs and clobbers.
RUN_DIR=\$(grep -m1 '^\[manifest\] run_dir = ' "\$TRAIN_LOG" | sed 's/^\[manifest\] run_dir = //')
rm -f "\$TRAIN_LOG"
if [ -n "\$RUN_DIR" ] && [ -f "\$RUN_DIR/best_model.pt" ]; then
    echo "Eval on run: \$RUN_DIR"
    python -m v5_rankbind.eval --run_dir "\$RUN_DIR"
else
    echo "ERROR: could not locate RUN_DIR or best_model.pt (RUN_DIR=\$RUN_DIR)"
fi

echo "All done at \$(date)"
EOF

echo "Submitted: ${JOBNAME} on partition ${PARTITION}"
