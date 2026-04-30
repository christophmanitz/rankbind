"""
generate_decoys.py — Create ~4000 shuffled protein-ligand decoy pairs.

Reads all processed .pt files from data/processed_hieratom/, keeps only
true binders (y > 0), shuffles their protein-ligand assignments
(mol_i paired with prot_j, i != j), sets y=0 (non-binder / binding unknown),
and saves to:
  data/processed_hieratom_shuffled/decoy_NNNNN.pt

Each saved file has the same format as the original processed files:
  (mol_graph, protein_graph, y_tensor)
where y_tensor = tensor([0.0]).

Run via:  sbatch run_generate_decoys.sh
"""

import os
import sys
import glob
import random
import torch
from tqdm import tqdm

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))

SRC_DIR  = os.path.join(PROJECT_ROOT, 'data', 'processed_hieratom')
DST_DIR  = os.path.join(PROJECT_ROOT, 'data', 'processed_hieratom_shuffled')
N_DECOYS = 4000
SEED     = 123

os.makedirs(DST_DIR, exist_ok=True)
print(f'Source : {SRC_DIR}')
print(f'Dest   : {DST_DIR}')
print(f'Target : {N_DECOYS} decoys')

# ── Collect all source files ──────────────────────────────────────────────────
all_files = sorted(
    glob.glob(os.path.join(SRC_DIR, 'data_*.pt')),
    key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
)
n_src = len(all_files)
print(f'Source files found: {n_src}')

if n_src < 2:
    raise RuntimeError('Not enough source files to shuffle.')

# ── Filter: keep only true binders (y > 0) ───────────────────────────────────
print('Scanning for true binders (y > 0) ...')
pos_files = []
for path in tqdm(all_files, desc='Filtering positives'):
    try:
        data = torch.load(path, weights_only=False)
        y_val = data[2]
        v = y_val.item() if y_val.numel() == 1 else float(y_val[0])
        if v > 0:
            pos_files.append(path)
    except Exception:
        pass

n_pos = len(pos_files)
print(f'True binders found: {n_pos} / {n_src}')
if n_pos < 2:
    raise RuntimeError('Not enough positive entries to shuffle.')

# ── Build shuffled pairs from positives only ──────────────────────────────────
# Strategy: shuffle the positive list once, then pair index i with index
# (i + n//2) % n — this guarantees no self-pairing after the shuffle.
rng = random.Random(SEED)
shuffled = pos_files[:]
rng.shuffle(shuffled)

half = n_pos // 2
lig_paths  = shuffled[:N_DECOYS]
prot_paths = [shuffled[(i + half) % n_pos] for i in range(N_DECOYS)]

# Sanity: no self-pairs
n_self = sum(1 for l, p in zip(lig_paths, prot_paths) if l == p)
print(f'Self-pairs (should be 0): {n_self}')

# ── Generate and save ─────────────────────────────────────────────────────────
y_zero = torch.tensor([0.0])
n_saved = n_error = 0

for k, (lig_path, prot_path) in enumerate(tqdm(
        zip(lig_paths, prot_paths), total=N_DECOYS, desc='Generating decoys')):
    out_path = os.path.join(DST_DIR, f'decoy_{k:05d}.pt')
    if os.path.exists(out_path):
        n_saved += 1
        continue
    try:
        mol,  _, _  = torch.load(lig_path,  weights_only=False)
        _,  prot, _ = torch.load(prot_path, weights_only=False)

        if mol.x is None or prot.x is None:
            n_error += 1
            continue
        if not torch.isfinite(mol.x).all() or not torch.isfinite(prot.x).all():
            n_error += 1
            continue

        torch.save((mol, prot, y_zero), out_path)
        n_saved += 1
    except Exception as e:
        n_error += 1
        if n_error <= 5:
            print(f'  Error at {k}: {e}')

print(f'\nDone — saved: {n_saved}, errors: {n_error}')
print(f'Decoys written to: {DST_DIR}')
