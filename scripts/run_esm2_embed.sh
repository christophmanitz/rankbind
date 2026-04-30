#!/bin/bash
# scripts/run_esm2_embed.sh — Precompute ESM2 embeddings for all proteins
#
# Uses HuggingFace transformers + facebook/esm2_t33_650M_UR50D (~2.5GB model)
# Generates per-residue embeddings (1280-dim, matching GEMS's t33 checkpoint).
#
# Output: data/esm2_embeddings/{uniprot}.pt (each file: [seq_len, 1280])
# Model must be pre-cached in ~/.cache/huggingface/ (compute nodes lack internet).
#
# Estimated time: ~1-2h for 882 proteins on 1 A30 GPU

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=esm2_embed
#SBATCH --output=logs/esm2_embed_%j.out
#SBATCH --error=logs/esm2_embed_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=paula

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

source "$HOME/venvs/hieratombind/bin/activate"

# Use cached model (pre-downloaded on login node)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

cd /home/sc.uni-leipzig.de/zw93onug/rankbind

python -c "
import os, torch, pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

SEQ_CSV = 'data/sequences/sequences.csv'
OUT_DIR = 'data/esm2_embeddings'
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'

# Load sequences
df = pd.read_csv(SEQ_CSV)
seqs = dict(zip(df['uniprot'], df['sequence']))
print(f'Loaded {len(seqs)} protein sequences')

# Load ESM2 from HuggingFace cache
print(f'Loading {MODEL_NAME} from cache...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval().cuda()
print(f'ESM2 loaded: {sum(p.numel() for p in model.parameters()):,} params')

# Process each protein
done = 0
skipped = 0
errors = 0
for uniprot, seq in tqdm(seqs.items(), desc='ESM2 embeddings'):
    out_path = os.path.join(OUT_DIR, f'{uniprot}.pt')
    if os.path.exists(out_path):
        skipped += 1
        continue

    try:
        # Truncate to 1022 (ESM2 max with special tokens, same as GEMS)
        seq_trunc = seq[:1022]
        token_ids = tokenizer(seq_trunc, return_tensors='pt')['input_ids'].cuda()

        with torch.no_grad():
            outputs = model(token_ids).last_hidden_state
            # Crop BOS/EOS tokens (GEMS convention)
            embedding = outputs[0, 1:-1].cpu().float()

        torch.save(embedding, out_path)
        done += 1
    except Exception as e:
        print(f'Error for {uniprot}: {e}')
        errors += 1

print(f'Done: {done} computed, {skipped} already existed, {errors} errors')
"

echo "ESM2 embedding job finished: $(date)"
EOF

echo "Submitted ESM2 embedding job"
