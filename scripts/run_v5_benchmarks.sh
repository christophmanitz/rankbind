#!/bin/bash
# scripts/run_v5_benchmarks.sh — §8.3 external-benchmark runs.
#
# Trains + evaluates RankBind on the four prepared benchmarks, choosing the
# model per the pre-registered plan (docs/BENCHMARK_INTEGRATION_PLAN.md):
#   ESP            -> esp_attn  (v5b / attention-pool: ESP is a second
#                                enzyme-substrate corpus, RankBind competes)
#   Davis/KIBA/    -> v4 default (kinase affinity: diagnosis only, does the
#   BindingDB         protein-shortcut appear? not a competitive claim)
#
# By default the jobs are queued to start only AFTER all of the user's
# currently-queued jobs finish (afterany), so they don't fight the running
# §8.1 peak jobs for GPUs. Override with DEPENDENCY=... or DEPENDENCY=none.
#
# After the runs land, compute the null_prot_prior Jaccard per benchmark
# (re-point evaluation/null_prior_probe_brenda_sabio.py at the run dirs) and
# assemble the §8.3 table.
#
# Usage:
#   bash scripts/run_v5_benchmarks.sh            # all 4, waiting on current queue
#   bash scripts/run_v5_benchmarks.sh esp        # just ESP
#   DEPENDENCY=none bash scripts/run_v5_benchmarks.sh   # no wait

set -euo pipefail
cd "$(dirname "$0")/.."

WHICH="${1:-all}"
PARTITION="${PARTITION:-paula}"

# Dependency: wait for everything currently in my queue unless told otherwise.
if [ -z "${DEPENDENCY:-}" ]; then
    cur=$(squeue -u "$USER" -h -o "%i" 2>/dev/null | paste -sd: -)
    DEPENDENCY="${cur:+afterany:$cur}"
elif [ "$DEPENDENCY" = "none" ]; then
    DEPENDENCY=""
fi
export DEPENDENCY
echo "Dependency for §8.3 jobs: ${DEPENDENCY:-<none>}"

# (benchmark, config (under v5_rankbind/configs/), tag, walltime)
JOBS=(
    "esp          datasets/benchmarks/esp_attn      bench_esp_v5b        12:00:00"
    "davis        datasets/benchmarks/davis         bench_davis_v4       02:00:00"
    "kiba         datasets/benchmarks/kiba          bench_kiba_v4        06:00:00"
    "bindingdb_kd datasets/benchmarks/bindingdb_kd  bench_bindingdb_v4   06:00:00"
)

n=0
for entry in "${JOBS[@]}"; do
    read -r bench cfg tag wt <<< "$entry"
    if [ "$WHICH" != "all" ] && [ "$WHICH" != "$bench" ]; then continue; fi
    echo "→ $bench  ($cfg, tag=$tag, walltime=$wt)"
    bash scripts/run_v5_rankbind.sh "$cfg" "$PARTITION" "$tag" "" "$wt"
    n=$((n + 1))
done
echo "Submitted $n §8.3 benchmark job(s) on $PARTITION."
