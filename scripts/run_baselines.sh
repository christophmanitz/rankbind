#!/bin/bash
# run_baselines.sh — Submit all Phase 1 baseline training jobs to SLURM.
#
# Prerequisites:
#   1. fetch_sequences has completed: data/sequences/sequences.csv exists
#   2. processed_hieratom .pt files exist
#
# Usage:
#   bash scripts/run_baselines.sh   (from rankbind/ root)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs

# ── Helper to write + submit a SLURM job ─────────────────────────────────────
submit_baseline() {
    local name="$1"
    local train_script="$2"
    local out_dir="$3"
    local extra_env="${4:-}"  # optional "KEY=val KEY2=val2" env overrides

    local job_script="logs/slurm_${name}.sh"
    cat > "$job_script" << SLURM
#!/bin/bash
#SBATCH --job-name=baseline_${name}
#SBATCH --output=$PROJECT_ROOT/logs/baseline_${name}_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/baseline_${name}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=paula

set -euo pipefail
cd "$PROJECT_ROOT"

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

source "\$HOME/venvs/hieratombind/bin/activate"

echo "Host:  \$(hostname)"
echo "Date:  \$(date)"
echo "GPUs:  \$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

export PT_DIR="$PROJECT_ROOT/data/processed_hieratom"
export SEQ_CSV="$PROJECT_ROOT/data/sequences/sequences.csv"
export STRUCT_DIR="/home/sc.uni-leipzig.de/zw93onug/hpc/structures"
export OUT_DIR="$out_dir"
export EPOCHS=100
export BATCH_SIZE=32
export N_MATRIX=300
export DEVICE=cuda
${extra_env}

mkdir -p "\$OUT_DIR"
echo "Training $name..."
python "$train_script"
echo "$name training done."

# Run attractor diagnosis on the produced score matrix
MATRIX="\$OUT_DIR/score_matrix_${name}.npy"
IDX="\$OUT_DIR/true_prot_idx_${name}.npy"
if [ -f "\$MATRIX" ]; then
    echo "Running attractor diagnosis for $name..."
    python evaluation/attractor_diagnosis.py \\
        --score_matrix "\$MATRIX" \\
        --model_name   "$name"    \\
        --out_dir      evaluation/attractor_results
fi
SLURM

    local job_id
    job_id=$(sbatch "$job_script" | awk '{print $NF}')
    echo "Submitted $name — job $job_id"
}

# ── Submit sequence-based baselines (require sequences.csv) ──────────────────
if [ ! -f "$PROJECT_ROOT/data/sequences/sequences.csv" ]; then
    echo "WARNING: data/sequences/sequences.csv not found."
    echo "Run fetch_sequences first: sbatch scripts/run_fetch_sequences.sh"
    echo "Skipping DeepDTA, GraphDTA, DrugBAN, MolTrans, GEMS."
else
    submit_baseline "DeepDTA" \
        "$PROJECT_ROOT/baselines/deepdta/train.py" \
        "$PROJECT_ROOT/baselines/deepdta/output"

    submit_baseline "GraphDTA" \
        "$PROJECT_ROOT/baselines/graphdta/train.py" \
        "$PROJECT_ROOT/baselines/graphdta/output"

    submit_baseline "DrugBAN" \
        "$PROJECT_ROOT/baselines/drugban/train.py" \
        "$PROJECT_ROOT/baselines/drugban/output"

    submit_baseline "MolTrans" \
        "$PROJECT_ROOT/baselines/moltrans/train.py" \
        "$PROJECT_ROOT/baselines/moltrans/output"

    submit_baseline "GEMS" \
        "$PROJECT_ROOT/baselines/gems/train.py" \
        "$PROJECT_ROOT/baselines/gems/output" \
        "export ESM_CACHE=$HOME/.cache/torch/hub"
fi

# ── GIGN: structure-based (requires PDB files, no sequences needed) ───────────
submit_baseline "GIGN" \
    "$PROJECT_ROOT/baselines/gign/train.py" \
    "$PROJECT_ROOT/baselines/gign/output"

# ── Submit v4 attractor diagnosis (no training needed) ───────────────────────
cat > logs/slurm_attractor_v4.sh << SLURM
#!/bin/bash
#SBATCH --job-name=attractor_v4
#SBATCH --output=$PROJECT_ROOT/logs/attractor_v4_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/attractor_v4_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=paula

set -euo pipefail
cd "$PROJECT_ROOT"

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

source "\$HOME/venvs/hieratombind/bin/activate"

export HIERATOMBIND_ROOT="\$HOME/newclaudemodel"

echo "Host:  \$(hostname)"
echo "Date:  \$(date)"

python evaluation/attractor_diagnosis.py \\
    --model_path  v4_residue_only/checkpoints/best_model.pt \\
    --model_name  ResidueOnlyBind_v4 \\
    --n_ligands   300 \\
    --n_proteins  300 \\
    --batch_size  32  \\
    --device      cuda \\
    --out_dir     evaluation/attractor_results
SLURM

job_id=$(sbatch logs/slurm_attractor_v4.sh | awk '{print $NF}')
echo "Submitted ResidueOnlyBind_v4 attractor diagnosis — job $job_id"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Results will be in evaluation/attractor_results/"