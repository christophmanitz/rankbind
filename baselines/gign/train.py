"""
baselines/gign/train.py — Train GIGN on BRENDA hydrolase data.

GIGN (Gao et al., 2023): Geometric Interaction GNN on protein residue + ligand atom graphs.
Simplified version with virtual cross-edges (no PDB coordinate parsing).
Adapted for binary binding classification.

Usage:
  python baselines/gign/train.py

Environment variables (override defaults):
  PT_DIR, OUT_DIR, EPOCHS, BATCH_SIZE, LR, VAL_FRAC, SEED, N_MATRIX, DEVICE
"""

import os
import sys
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..'))

from model import GIGN
from baseline_dataset import BaselineDataset

# ── Hyperparameters ───────────────────────────────────────────────────────────
PT_DIR     = os.environ.get('PT_DIR',    os.path.join(PROJECT_ROOT, 'data', 'processed_hieratom'))
OUT_DIR    = os.environ.get('OUT_DIR',   os.path.join(_HERE, 'output'))
EPOCHS     = int(os.environ.get('EPOCHS',     '100'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
LR         = float(os.environ.get('LR',       '1e-4'))
VAL_FRAC   = float(os.environ.get('VAL_FRAC', '0.15'))
SEED       = int(os.environ.get('SEED',       '42'))
N_MATRIX   = int(os.environ.get('N_MATRIX',   '300'))
DEVICE     = os.environ.get('DEVICE', 'cuda')

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
log.info(f"Device: {device}")


class GIGNDataset(BaselineDataset):
    """Extends BaselineDataset to also return the protein graph."""

    def __getitem__(self, idx):
        mol, prot, y = torch.load(self.pt_files[idx], weights_only=False)
        y_bin = torch.tensor(float(y.item() > 0), dtype=torch.float32)
        return mol, prot, y_bin


def collate_gign(batch):
    """Collate for GIGN — returns lists of individual graphs + stacked labels."""
    mols, prots, ys = zip(*batch)
    return list(mols), list(prots), torch.stack(ys)


def train():
    dataset = GIGNDataset(PT_DIR)
    n_val   = max(1, int(len(dataset) * VAL_FRAC))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    log.info(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_gign, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_gign, num_workers=0)

    model = GIGN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_auc  = 0.0
    best_path = os.path.join(OUT_DIR, 'best_model.pt')

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for mols, prots, ys in train_loader:
            mols_dev  = [m.to(device) for m in mols]
            prots_dev = [p.to(device) for p in prots]
            ys = ys.to(device)

            prob = model(mols_dev, prots_dev)
            loss = criterion(prob, ys)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(ys)
        train_loss = total_loss / n_train

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        all_probs, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for mols, prots, ys in val_loader:
                mols_dev  = [m.to(device) for m in mols]
                prots_dev = [p.to(device) for p in prots]
                prob = model(mols_dev, prots_dev).cpu()
                val_loss += criterion(prob, ys).item() * len(ys)
                all_probs.extend(prob.numpy())
                all_labels.extend(ys.numpy())
        val_loss /= n_val

        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_probs) if all_labels.sum() > 0 else 0.0
        ap  = average_precision_score(all_labels, all_probs) if all_labels.sum() > 0 else 0.0

        log.info(f"Ep {epoch:3d}/{EPOCHS} | train={train_loss:.4f} val={val_loss:.4f} "
                 f"AUC={auc:.4f} AP={ap:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)
            log.info(f"  -> Best AUC={best_auc:.4f}")

    log.info(f"Training done. Best AUC: {best_auc:.4f}")

    # ── Score matrix (cross-evaluation) ───────────────────────────────────────
    log.info(f"Building score matrix ({N_MATRIX}x{N_MATRIX})...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # Load mol and prot graphs separately from first N_MATRIX items
    n_act = min(N_MATRIX, len(dataset))
    mol_list = []
    prot_list = []
    for i in range(n_act):
        mol, prot, _ = dataset[i]
        mol_list.append(mol)
        prot_list.append(prot)

    score_matrix = np.zeros((n_act, n_act), dtype=np.float32)
    with torch.no_grad():
        for i in range(n_act):
            mol_i = mol_list[i].to(device)
            # Score mol_i against all proteins
            row_scores = []
            for j in range(n_act):
                prot_j = prot_list[j].to(device)
                s = model.forward_single(mol_i, prot_j).cpu().item()
                row_scores.append(s)
            score_matrix[i] = np.array(row_scores, dtype=np.float32)
            if (i + 1) % 50 == 0:
                log.info(f"  Score matrix row {i+1}/{n_act}")

    mat_path = os.path.join(OUT_DIR, 'score_matrix_GIGN.npy')
    np.save(mat_path, score_matrix)
    np.save(os.path.join(OUT_DIR, 'true_prot_idx_GIGN.npy'), np.arange(n_act))
    log.info(f"Score matrix saved: {mat_path}")


if __name__ == '__main__':
    train()
