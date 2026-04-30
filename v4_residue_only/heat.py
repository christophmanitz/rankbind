"""
heat.py — Ligand×Protein response matrix for ResidueOnlyBind.

Loads the best checkpoint, precomputes ligand and protein embeddings
separately via the model's internal GNNs, then builds an N_LIGANDS×N_PROTEINS
binding-probability matrix and saves:
  logs/response_matrix.npy
  logs/ligand_partner_analysis.csv
  logs/ligand_protein_response_map.png
"""

import os
import sys
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from torch_geometric.data import Batch

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, _HERE)

from ResidueOnlyBind import ResidueOnlyBind

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(_HERE, 'checkpoints_v2', 'best_model.pt')
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed_hieratom')
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_LIGANDS  = 128
N_PROTEINS = 128
BATCH_SIZE = 16

os.makedirs(os.path.join(_HERE, 'logs_v2'), exist_ok=True)
print('Device:', DEVICE)
print('Model: ', MODEL_PATH)

# ── Load pre-processed .pt files ──────────────────────────────────────────────
pt_files = sorted(
    glob.glob(os.path.join(DATASET_ROOT, 'data_*.pt')),
    key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
)
print(f'Found {len(pt_files)} processed samples')


def load_item(path):
    data = torch.load(path, weights_only=False)
    return data[0], data[1], data[2]   # mol, prot, y


def filter_bad(files, n):
    valid = []
    for path in files:
        if len(valid) >= n:
            break
        try:
            mol, prot, y = load_item(path)
            if mol.x is None or prot.x is None:
                continue
            if not torch.isfinite(mol.x).all():
                continue
            if not torch.isfinite(prot.x).all():
                continue
            valid.append(path)
        except Exception as e:
            print(f'  skip {os.path.basename(path)}: {e}')
    return valid


needed = max(N_LIGANDS, N_PROTEINS)
print(f'Filtering up to {needed} valid samples...')
valid_paths = filter_bad(pt_files, needed)
print(f'Valid: {len(valid_paths)}')

ligand_paths  = valid_paths[:N_LIGANDS]
protein_paths = valid_paths[:N_PROTEINS]

mols  = [load_item(p)[0] for p in ligand_paths]
prots = [load_item(p)[1] for p in protein_paths]
ys    = [load_item(p)[2] for p in ligand_paths]

ligand_y = [y.item() if y.numel() == 1 else float(y[0]) for y in ys]
print(f'Using {len(mols)} ligands × {len(prots)} proteins')
print(f'True Positives (y>0): {sum(v > 0 for v in ligand_y)} / {len(ligand_y)}')

# ── Load model ────────────────────────────────────────────────────────────────
model = ResidueOnlyBind().to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
# Support raw state_dict or wrapped checkpoint
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    ckpt = ckpt['model_state_dict']
model.load_state_dict(ckpt)
model.eval()
print('Model loaded')

# ── Precompute embeddings ─────────────────────────────────────────────────────

def embed_proteins(prot_list, batch_size):
    """Run residue_gnn + res_pool → [N, H]."""
    out = []
    for i in range(0, len(prot_list), batch_size):
        batch = Batch.from_data_list(prot_list[i:i+batch_size]).to(DEVICE)
        with torch.no_grad():
            embs, pad = model.residue_gnn(batch)   # [B, N_res, H]
            pooled    = model.res_pool(embs, pad)   # [B, H]
        out.append(pooled.cpu())
    return torch.cat(out, dim=0)   # [N, H]


def embed_ligands(mol_list, batch_size):
    """Run ligand_gnn + lig_pool → [N, H]."""
    out = []
    for i in range(0, len(mol_list), batch_size):
        batch = Batch.from_data_list(mol_list[i:i+batch_size]).to(DEVICE)
        with torch.no_grad():
            embs, pad = model.ligand_gnn(batch)    # [B, N_lig, H]
            pooled    = model.lig_pool(embs, pad)   # [B, H]
        out.append(pooled.cpu())
    return torch.cat(out, dim=0)   # [N, H]


print('Embedding proteins...')
prot_emb = embed_proteins(prots, BATCH_SIZE)   # [N_PROTEINS, H]

print('Embedding ligands...')
lig_emb  = embed_ligands(mols,  BATCH_SIZE)    # [N_LIGANDS, H]

# ── Build N×M score matrix ────────────────────────────────────────────────────
scores = torch.zeros(len(mols), len(prots))
print('Computing cross-prediction matrix...')

prot_emb_dev = prot_emb.to(DEVICE)

with torch.no_grad():
    for i in range(len(mols)):
        lig = lig_emb[i].unsqueeze(0).to(DEVICE).expand(len(prots), -1)  # [P, H]
        interaction = lig * prot_emb_dev                                   # [P, H]
        fused  = model.fusion_mlp(
            torch.cat([interaction, prot_emb_dev, lig], dim=-1)
        )                                                                   # [P, H]
        logits = model.bind_head(fused).squeeze(-1)                        # [P]
        probs  = torch.sigmoid(logits).cpu()
        scores[i] = probs

print('Matrix computed')
np.save(os.path.join(_HERE, 'logs_v2', 'response_matrix.npy'), scores.numpy())

# ── Find true partner index for each ligand ───────────────────────────────────
# Each pair in the dataset has one true partner protein.
# We match by comparing the x tensor of the true partner against prots[].

true_prots = [load_item(p)[1] for p in ligand_paths]

ligand_true_protein_idx = []
for li, true_prot in enumerate(true_prots):
    best = -1
    for pi, p in enumerate(prots):
        if (p.x.shape == true_prot.x.shape and
                torch.allclose(p.x, true_prot.x, atol=1e-5)):
            best = pi
            break
    ligand_true_protein_idx.append(best)

print('True-partner mapping:', Counter(ligand_true_protein_idx))

# ── Retrieval analysis ────────────────────────────────────────────────────────
results = []
for li in range(len(mols)):
    true_pi = ligand_true_protein_idx[li]
    y_val   = ligand_y[li]
    is_pos  = y_val > 0

    row          = scores[li]
    pred_max_pi  = row.argmax().item()
    score_max    = row[pred_max_pi].item()
    row_mean     = row.mean().item()
    row_var      = row.var().item()

    if true_pi == -1 or not is_pos:
        results.append(dict(
            ligand_idx=li, y_value=y_val, is_positive=False,
            true_protein_idx=true_pi, pred_max_protein=pred_max_pi,
            score_at_true=None, score_max=round(score_max, 4),
            mean_score=round(row_mean, 4), variance=round(row_var, 6),
            rank_of_true=None, hit_at_1=None, hit_at_5=None, hit_at_10=None,
        ))
    else:
        score_at_true = row[true_pi].item()
        rank_of_true  = (row > score_at_true).sum().item() + 1
        results.append(dict(
            ligand_idx=li, y_value=y_val, is_positive=True,
            true_protein_idx=true_pi, pred_max_protein=pred_max_pi,
            score_at_true=round(score_at_true, 4), score_max=round(score_max, 4),
            mean_score=round(row_mean, 4), variance=round(row_var, 6),
            rank_of_true=rank_of_true,
            hit_at_1=pred_max_pi == true_pi,
            hit_at_5=rank_of_true <= 5,
            hit_at_10=rank_of_true <= 10,
        ))

df_all = pd.DataFrame(results)
df_all.to_csv(os.path.join(_HERE, 'logs_v2', 'ligand_partner_analysis.csv'), index=False)

df_pos = df_all[df_all['is_positive']]
df_neg = df_all[~df_all['is_positive']]

print(f'\n=== Dataset ===')
print(f'True Positives : {len(df_pos)}')
print(f'True Negatives : {len(df_neg)}')

if len(df_pos) > 0:
    print(f'\n=== Retrieval Metrics (True Positives only) ===')
    print(f'Hit@1  : {df_pos["hit_at_1"].mean():.1%}')
    print(f'Hit@5  : {df_pos["hit_at_5"].mean():.1%}')
    print(f'Hit@10 : {df_pos["hit_at_10"].mean():.1%}')
    print(f'Mean Rank (true partner) : {df_pos["rank_of_true"].mean():.1f} / {N_PROTEINS}')
    print(f'Mean Score @ true partner: {df_pos["score_at_true"].mean():.4f}')
    print(f'Mean Max Score           : {df_pos["score_max"].mean():.4f}')

print(f'\n=== Score Comparison Pos vs Neg ===')
print(f'Mean Score Positives : {df_pos["mean_score"].mean():.4f}' if len(df_pos) else '')
print(f'Mean Score Negatives : {df_neg["mean_score"].mean():.4f}' if len(df_neg) else '')

print('\nSaved: logs/ligand_partner_analysis.csv')

# ── Heatmap ───────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.imshow(scores.numpy(), aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='Binding probability')
plt.xlabel('Protein index')
plt.ylabel('Ligand index')
plt.title('Ligand–Protein Response Map (ResidueOnlyBind)')
plt.tight_layout()
plt.savefig(os.path.join(_HERE, 'logs_v2', 'ligand_protein_response_map.png'), dpi=300)
print('Saved: logs_v2/ligand_protein_response_map.png')

# ── Interaction diagnostics ───────────────────────────────────────────────────
row_var_mean = scores.var(dim=1).mean().item()
col_var_mean = scores.var(dim=0).mean().item()

print('\n=== Interaction Diagnostics ===')
print(f'Mean row variance (ligand sensitivity)   : {row_var_mean:.6f}')
print(f'Mean col variance (protein sensitivity)  : {col_var_mean:.6f}')
print(f'Ligand/Protein influence ratio           : {row_var_mean / (col_var_mean + 1e-8):.3f}')
