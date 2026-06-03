#!/bin/bash
# scripts/embed_benchmark.sh — per-residue ESM2 embeddings for a prepared
# benchmark dataset (ESP / Davis / KIBA / BindingDB).
#
# Reuses reaction_data.embeddings.compute_esm2_embeddings
# (facebook/esm2_t33_650M_UR50D, per-residue, 1280-dim = v5_rankbind default).
# Reads <benchmark>/sequences.csv, writes {uniprot}.pt into
# <benchmark>/esm2_embeddings/. Idempotent (skips existing .pt).
#
# Submit AFTER the turnover multi-seed jobs free the GPU queue.
#
# Usage:
#   bash scripts/embed_benchmark.sh esp                 # default 06:00:00
#   bash scripts/embed_benchmark.sh davis paula 01:00:00
#   bash scripts/embed_benchmark.sh all                 # submit one job per benchmark
#
# Rough sizing (per-residue, A30, batch 1): ESP ~11.4k prot is the long one
# (~28 GB / several hours); BindingDB ~1.4k; Davis/KIBA few hundred (minutes).

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

BENCH="${1:-esp}"
PARTITION="${2:-paula}"
WALLTIME="${3:-}"

BENCH_ROOT="reactionDataFiltering/data/interim/benchmarks"

# per-benchmark default walltime (override with arg 3)
declare -A WT=( [esp]=08:00:00 [davis]=01:00:00 [kiba]=01:00:00 [bindingdb_kd]=03:00:00 )

submit_one() {
    local bench="$1"
    local seqs="$BENCH_ROOT/$bench/sequences.csv"
    local outdir="$BENCH_ROOT/$bench/esm2_embeddings"
    local wt="${WALLTIME:-${WT[$bench]:-06:00:00}}"
    if [ ! -f "$PROJECT_ROOT/$seqs" ]; then
        echo "ERROR: $seqs not found — run prep_benchmark_datasets.py $bench first"; return 1
    fi
    mkdir -p "$PROJECT_ROOT/$outdir"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=esm2_${bench}
#SBATCH --output=${PROJECT_ROOT}/logs/esm2_${bench}_%j.out
#SBATCH --error=${PROJECT_ROOT}/logs/esm2_${bench}_%j.err
#SBATCH --time=${wt}
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=${PARTITION}

cd "\$SLURM_SUBMIT_DIR"
module purge
module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
source "\$HOME/venvs/hieratombind/bin/activate"
echo "Host: \$(hostname)  Job: \$SLURM_JOB_ID  Bench: ${bench}  Walltime: ${wt}"

python - <<'PY'
import sys
sys.path.insert(0, "reactionDataFiltering")
from reaction_data.embeddings import compute_esm2_embeddings, DEFAULT_MODEL
c = compute_esm2_embeddings(
    sequences="${BENCH_ROOT}/${bench}/sequences.csv",
    output_dir="${BENCH_ROOT}/${bench}/esm2_embeddings",
    model_name=DEFAULT_MODEL,
    device="cuda",
    batch_size=1,
    max_residues=1024,
)
print("embeddings done:", c)
PY
echo "All done at \$(date)"
EOF
    echo "Submitted: esm2_${bench} (walltime ${wt}, partition ${PARTITION})"
}

if [ "$BENCH" = "all" ]; then
    for b in davis kiba bindingdb_kd esp; do submit_one "$b"; done
else
    submit_one "$BENCH"
fi
