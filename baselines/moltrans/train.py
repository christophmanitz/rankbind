"""
baselines/moltrans/train.py — Train MolTrans on BRENDA hydrolase data.

MolTrans (Huang et al., 2021): Transformer on protein sequence + Transformer on SMILES.
Adapted for binary binding classification.

Usage:
  python baselines/moltrans/train.py

Environment variables (override defaults):
  PT_DIR, SEQ_CSV, OUT_DIR, EPOCHS, BATCH_SIZE, LR, VAL_FRAC, SEED, N_MATRIX, DEVICE
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

from model import MolTrans
from baseline_dataset import SeqBaselineDataset, collate_seq, ProteinTokenizer, LigandTokenizer

# ── Hyperparameters ───────────────────────────────────────────────────────────
PT_DIR     = os.environ.get('PT_DIR',    os.path.join(PROJECT_ROOT, 'data', 'processed_hieratom'))
SEQ_CSV    = os.environ.get('SEQ_CSV',   os.path.join(PROJECT_ROOT, 'data', 'sequences', 'sequences.csv'))
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


def train():
    prot_tok = ProteinTokenizer(max_len=1000)
    lig_tok  = LigandTokenizer(max_len=100)

    dataset = SeqBaselineDataset(PT_DIR, SEQ_CSV, prot_tok, lig_tok)
    n_val   = max(1, int(len(dataset) * VAL_FRAC))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    log.info(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_seq, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_seq, num_workers=2, pin_memory=True)

    model = MolTrans().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_auc  = 0.0
    best_path = os.path.join(OUT_DIR, 'best_model.pt')

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for prot_ids, lig_ids, _, ys in train_loader:
            prot_ids = prot_ids.to(device)
            lig_ids  = lig_ids.to(device)
            ys       = ys.to(device)

            prob = model(prot_ids, lig_ids)
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
            for prot_ids, lig_ids, _, ys in val_loader:
                prot_ids = prot_ids.to(device)
                lig_ids  = lig_ids.to(device)
                prob = model(prot_ids, lig_ids).cpu()
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

    # ── Score matrix ──────────────────────────────────────────────────────────
    log.info(f"Building score matrix ({N_MATRIX}x{N_MATRIX})...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    n_act = min(N_MATRIX, len(dataset))
    items = [dataset[i] for i in range(n_act)]
    prot_tensor = torch.stack([it[0] for it in items])  # [N, L_prot]
    lig_tensor  = torch.stack([it[1] for it in items])  # [N, L_lig]

    score_matrix = np.zeros((n_act, n_act), dtype=np.float32)
    with torch.no_grad():
        for i in range(n_act):
            p = prot_tensor[i:i+1].expand(n_act, -1).to(device)
            l = lig_tensor.to(device)
            score_matrix[i] = model(p, l).cpu().numpy()

    mat_path = os.path.join(OUT_DIR, 'score_matrix_MolTrans.npy')
    np.save(mat_path, score_matrix)
    np.save(os.path.join(OUT_DIR, 'true_prot_idx_MolTrans.npy'), np.arange(n_act))
    log.info(f"Score matrix saved: {mat_path}")


if __name__ == '__main__':
    train()
