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
TAG_BASE="v5"

# Protocol-A note (2026-08-22): data.split_seed is pinned to 42 in
# default.json, so cfg["seed"] only varies init/shuffling — every seed now
# trains AND evaluates on the canonical protein split. The existing clean
# seed-42 anchors stay valid:
#   default / abl_no_sampler / abl_no_bilinear -> v4 tag (seed 42 on disk)
#   abl_no_margin / abl_bce_only               -> NO seed-42 run exists;
#                                                 submit it here under v5.
NEED_S42=(abl_no_margin abl_bce_only)

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

for cfg in "${NEED_S42[@]}"; do
    echo "Submitting missing anchor: cfg=${cfg} seed=42"
    bash "$PROJECT_ROOT/scripts/run_v5_rankbind.sh" "${cfg}" paula "${TAG_BASE}" 42
done

TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} + ${#NEED_S42[@]} ))
echo
echo "Submitted ${TOTAL} job(s). Watch with:  squeue -u \$USER"
echo "Aggregate when done:  ~/rankbind_revision/reeval_true_split.py (honest matrix metrics)"
