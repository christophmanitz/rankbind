#!/bin/bash
# scripts/run_v5_brenda_sabio_hp_sweep.sh — Stage-1 hard_pool_size sweep on
# BRENDA+SABIO with_decoys datasets.
#
# Submits 12 SLURM jobs total: 4 hard_pool values × 3 datasets (kcat_km, km,
# turnover). Tag scheme: bs_v2_hp<value>. All runs land under
# results/v5_rankbind/ with a manifest.json carrying the sweep config name
# (e.g. km_with_decoys_hp1700) so they roll up cleanly with collect_v5_runs.py.
#
# Per-dataset walltime allowance is sized from the bs_v1 baseline:
#   kcat_km: 184 min wall  → 5 h budget
#   km:      340 min wall  → 8 h budget
#   turnover:347 min wall  → 8 h budget
#
# Usage:
#   bash scripts/run_v5_brenda_sabio_hp_sweep.sh           # submit all 12
#   bash scripts/run_v5_brenda_sabio_hp_sweep.sh kcat_km   # only kcat_km set
#   bash scripts/run_v5_brenda_sabio_hp_sweep.sh dryrun    # print, do not sbatch

set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

WHICH="${1:-all}"
PARTITION="${PARTITION:-paula}"

# (dataset, hp_value, walltime) tuples
JOBS=(
    "kcat_km  100  05:00:00"
    "kcat_km  300  05:00:00"
    "kcat_km  700  05:00:00"
    "kcat_km  1400 05:00:00"
    "km       250  08:00:00"
    "km       700  08:00:00"
    "km       1700 08:00:00"
    "km       3400 08:00:00"
    "turnover 150  08:00:00"
    "turnover 400  08:00:00"
    "turnover 1000 08:00:00"
    "turnover 2000 08:00:00"
)

submit_one() {
    local dataset="$1" hp="$2" walltime="$3"
    local cfg_rel="sweeps/hp_brenda_sabio/${dataset}_with_decoys_hp${hp}"
    local tag="bs_v2_hp${hp}"

    if [ "$WHICH" = "dryrun" ]; then
        echo "[dryrun] cfg=$cfg_rel tag=$tag walltime=$walltime partition=$PARTITION"
        return
    fi

    bash "$PROJECT_ROOT/scripts/run_v5_rankbind.sh" \
        "$cfg_rel" "$PARTITION" "$tag" "" "$walltime"
}

n=0
for entry in "${JOBS[@]}"; do
    read -r dataset hp walltime <<< "$entry"
    if [ "$WHICH" != "all" ] && [ "$WHICH" != "dryrun" ] && [ "$WHICH" != "$dataset" ]; then
        continue
    fi
    submit_one "$dataset" "$hp" "$walltime"
    n=$((n + 1))
done

echo "Submitted $n jobs (selector: $WHICH, partition: $PARTITION)"
