#!/bin/bash
# scripts/run_v5_ablations.sh — Submit the 5-run ablation sweep for Phase 2.
#
# Produces five SLURM jobs, one per config under v5_rankbind/configs/.
# Monitor with `squeue -u $USER` and collect results via:
#   python scripts/collect_v5_runs.py
#
# Usage:
#   bash scripts/run_v5_ablations.sh           # all five
#   bash scripts/run_v5_ablations.sh default   # single named config

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

CONFIGS=(default abl_no_sampler abl_no_margin abl_no_bilinear abl_bce_only)

TARGET="${1:-all}"
if [ "$TARGET" != "all" ]; then
    CONFIGS=("$TARGET")
fi

for cfg in "${CONFIGS[@]}"; do
    echo "Submitting: ${cfg}"
    bash "$PROJECT_ROOT/scripts/run_v5_rankbind.sh" "${cfg}"
done

echo "Submitted ${#CONFIGS[@]} job(s). Watch with:  squeue -u \$USER"
