#!/bin/bash
# scripts/run_v5_bs_v2_peak_seeds.sh — multi-seed rerun of the bs_v2 sweep
# PEAK hard_pool_size per dataset, to turn the single-seed §8.1 transferability
# numbers into 3-seed (mean ± std) figures matching the BRENDA-200 headline
# protocol (seeds {42, 7, 1337}).
#
# Background: the bs_v2 hard_pool_size sweep (run_v5_brenda_sabio_hp_sweep.sh)
# was single-seed (seed 42). Its per-dataset peaks were:
#   kcat_km  hp=1400  MRR 0.228   (52% coverage of n_train_proteins)
#   km       hp=3400  MRR 0.219   (51%)
#   turnover hp=2000  MRR 0.344   (50%)  <- triggers the pre-registered Major win
# Seed 42 already exists for all three (tag bs_v2_hp<value>); this script only
# submits the two MISSING seeds {7, 1337} -> 6 jobs total. The seed is folded
# into the tag automatically by run_v5_rankbind.sh (e.g. bs_v2_hp2000_s7).
#
# Walltimes mirror the proven sweep budget (all 12 sweep runs completed within
# these on the same configs).
#
# Usage:
#   bash scripts/run_v5_bs_v2_peak_seeds.sh dryrun      # print, do not sbatch
#   bash scripts/run_v5_bs_v2_peak_seeds.sh             # submit all 6
#   bash scripts/run_v5_bs_v2_peak_seeds.sh turnover    # only turnover's 2 seeds
#
# After the runs land, aggregate with scripts/aggregate_multiseed.py (same tool
# used for the BRENDA-200 3-seed table) pointed at results/v5_rankbind/*bs_v2_hp*_s*.

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

WHICH="${1:-all}"
PARTITION="${PARTITION:-paula}"
SEEDS=(7 1337)

# (dataset, peak_hp, walltime) — walltime from run_v5_brenda_sabio_hp_sweep.sh
PEAKS=(
    "kcat_km  1400 05:00:00"
    "km       3400 08:00:00"
    "turnover 2000 08:00:00"
)

n=0
for entry in "${PEAKS[@]}"; do
    read -r dataset hp walltime <<< "$entry"
    if [ "$WHICH" != "all" ] && [ "$WHICH" != "dryrun" ] && [ "$WHICH" != "$dataset" ]; then
        continue
    fi
    cfg_rel="sweeps/hp_brenda_sabio/${dataset}_with_decoys_hp${hp}"
    # bs_v3 = same sweep configs, re-trained under Protocol A
    # (data.split_seed pinned to 42; --seed varies init only). The old
    # bs_v2_hp*_s7/_s1337 runs were evaluated on the wrong split — do not
    # reuse their test metrics (see ~/rankbind_revision/PLAN.md, C2).
    tag="bs_v3_hp${hp}"
    for seed in "${SEEDS[@]}"; do
        if [ "$WHICH" = "dryrun" ]; then
            echo "[dryrun] cfg=$cfg_rel tag=${tag}_s${seed} seed=$seed walltime=$walltime partition=$PARTITION"
        else
            bash "$PROJECT_ROOT/scripts/run_v5_rankbind.sh" \
                "$cfg_rel" "$PARTITION" "$tag" "$seed" "$walltime"
        fi
        n=$((n + 1))
    done
done

echo "Submitted $n jobs (selector: $WHICH, seeds: ${SEEDS[*]}, partition: $PARTITION)"
