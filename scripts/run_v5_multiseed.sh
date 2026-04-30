#!/bin/bash
# scripts/run_v5_multiseed.sh — Multi-seed sweep for Phase-2 Priority B.
#
# Submits (configs × seeds) SLURM jobs for error-bar estimation. Seed 42 is
# assumed to already exist on disk as v4 (default, abl_no_sampler,
# abl_no_bilinear) or v3 (abl_no_margin) / v2 (abl_bce_only) — so this
# script only submits the missing (config, seed) pairs.
#
# Every job is tagged "<TAG>_s<SEED>" so run_ids look like:
#   20260423-140000_<sha>_default_v4_s7
#   20260423-140010_<sha>_default_v4_s1337
#   ...
#
# Aggregation is handled by scripts/aggregate_multiseed.py after all jobs
# finish.
#
# Usage:
#   bash scripts/run_v5_multiseed.sh            # all configs × seeds {7,1337}
#   bash scripts/run_v5_multiseed.sh default    # single config, both seeds
#
# Positional args:
#   1. CONFIG   — optional single config name (default: submit all five)
#
# Each sub-submission calls scripts/run_v5_rankbind.sh; cf. that script for
# the tag-and-seed threading.

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

CONFIGS=(default abl_no_sampler abl_no_margin abl_no_bilinear abl_bce_only)
SEEDS=(7 1337)
TAG_BASE="v4"

TARGET="${1:-all}"
if [ "$TARGET" != "all" ]; then
    CONFIGS=("$TARGET")
fi

echo "Multi-seed sweep:"
echo "  configs: ${CONFIGS[*]}"
echo "  seeds:   ${SEEDS[*]}"
echo "  tag base: ${TAG_BASE}"
echo

for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Submitting: cfg=${cfg} seed=${seed}"
        bash "$PROJECT_ROOT/scripts/run_v5_rankbind.sh" "${cfg}" paula "${TAG_BASE}" "${seed}"
    done
done

TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
echo
echo "Submitted ${TOTAL} job(s). Watch with:  squeue -u \$USER"
echo "Aggregate when done:  python scripts/aggregate_multiseed.py"
