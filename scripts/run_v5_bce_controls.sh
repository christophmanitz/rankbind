#!/bin/bash
# scripts/run_v5_bce_controls.sh — BCE shortcut-CONTROL runs across all datasets.
#
# Purpose: the §8.3 "shortcut can be circumvented (in general)" demonstration
# needs a CONTRAST per dataset. RankBind (v4/v5b) is the anti-shortcut arm and
# is already run everywhere; this script supplies the matching shortcut-TAKER
# arm — the Phase-1-style recipe (random sampler + BCE + MLP-concat head, no
# triplets, config stem "*_bce") — on every dataset that still lacks one.
# A dataset where BCE takes the shortcut (high pooled AUC + high Top-10 Jaccard
# vs null_prot_prior + low matrix MRR) while RankBind does not (low Jaccard +
# ligand-conditional ranking) is one where the shortcut is demonstrably
# circumvented. BRENDA-200 already has its control (abl_bce_only_v4).
#
# After the runs land, score every BCE run with
#   python evaluation/benchmark_null_eval.py <run_dir> ...
# and assemble the unified BCE-vs-RankBind table.
#
# Usage:
#   bash scripts/run_v5_bce_controls.sh                 # all, no dependency
#   bash scripts/run_v5_bce_controls.sh davis_bce       # just one
#   DEPENDENCY=afterany:123 bash scripts/run_v5_bce_controls.sh   # chain

set -euo pipefail
cd "$(dirname "$0")/.."

WHICH="${1:-all}"
PARTITION="${PARTITION:-paula}"
export DEPENDENCY="${DEPENDENCY:-}"   # default: no dependency, run as GPUs free

# (key, config stem under v5_rankbind/configs/, tag, walltime)
JOBS=(
  "davis_bce        datasets/benchmarks/davis_bce         bce_davis      02:00:00"
  "kiba_bce         datasets/benchmarks/kiba_bce          bce_kiba       06:00:00"
  "bindingdb_kd_bce datasets/benchmarks/bindingdb_kd_bce  bce_bindingdb  06:00:00"
  "esp_bce          datasets/benchmarks/esp_bce           bce_esp        12:00:00"
  "kcat_km_bce      datasets/kcat_km_with_decoys_bce      bce_kcat_km    08:00:00"
  "km_bce           datasets/km_with_decoys_bce           bce_km         12:00:00"
  "turnover_bce     datasets/turnover_with_decoys_bce     bce_turnover   08:00:00"
)

n=0
for entry in "${JOBS[@]}"; do
  read -r key cfg tag wt <<< "$entry"
  if [ "$WHICH" != "all" ] && [ "$WHICH" != "$key" ]; then continue; fi
  echo "→ $key  ($cfg, tag=$tag, walltime=$wt)"
  bash scripts/run_v5_rankbind.sh "$cfg" "$PARTITION" "$tag" "" "$wt"
  n=$((n + 1))
done
echo "Submitted $n BCE shortcut-control job(s) on $PARTITION."
