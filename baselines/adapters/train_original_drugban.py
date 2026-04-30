"""
baselines/adapters/train_original_drugban.py — Trainer for original DrugBAN.

DrugBAN requires DGL which conflicts with torch 2.8.
This script must be run from a separate venv with:
  - torch 2.4.x
  - dgl 2.4.x
  - dgllife

Usage:
  python baselines/adapters/train_original_drugban.py --out_dir results/original_drugban

See scripts/setup_drugban_venv.sh for venv creation.
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _HERE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_matrix', type=int, default=200)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training, load existing best_model.pt')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_drugban(batch):
    """Collate function for DrugBAN: DGL graphs + protein ids + labels."""
    import dgl
    graphs, prot_ids, labels = zip(*batch)
    return dgl.batch(graphs), torch.stack(prot_ids), torch.stack(labels)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_samples = 0
    for graphs, prot_ids, labels in loader:
        graphs = graphs.to(device)
        prot_ids = prot_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(graphs, prot_ids)
        # DrugBAN returns (v_d, v_p, f, score) — take score
        if isinstance(out, tuple):
            out = out[-1]
        loss = criterion(out.squeeze(-1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n_samples += len(labels)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_samples = 0
    all_preds, all_labels = [], []
    for graphs, prot_ids, labels in loader:
        graphs = graphs.to(device)
        prot_ids = prot_ids.to(device)
        labels = labels.to(device)
        out = model(graphs, prot_ids)
        # DrugBAN returns (v_d, v_p, f, score) — take score
        if isinstance(out, tuple):
            out = out[-1]
        loss = criterion(out.squeeze(-1), labels)
        total_loss += loss.item() * len(labels)
        n_samples += len(labels)
        all_preds.append(out.squeeze(-1).cpu())
        all_labels.append(labels.cpu())
    avg_loss = total_loss / max(n_samples, 1)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    return avg_loss, auc


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log.info(f"DrugBAN Original, Device: {device}")

    if args.out_dir is None:
        args.out_dir = os.path.join(PROJECT_ROOT, 'results', 'original_drugban')
    os.makedirs(args.out_dir, exist_ok=True)

    from common import BRENDADataConfig
    from adapter_drugban import DrugBANDataset, get_model

    config = BRENDADataConfig(seed=args.seed)
    train_idx, val_idx, test_idx = config.get_protein_split()
    log.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_ds = DrugBANDataset(config, train_idx)
    val_ds = DrugBANDataset(config, val_idx)
    # drop_last=True: DrugBAN has BatchNorm layers that fail on batch_size=1
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_drugban,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_drugban,
                            drop_last=True)

    ModelClass = get_model()
    # DrugBAN requires a config dict — use their defaults
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'DrugBAN'))
    from configs import get_cfg_defaults
    drugban_cfg = get_cfg_defaults()
    model = ModelClass(**drugban_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_path = os.path.join(args.out_dir, 'best_model.pt')
    patience_counter = 0

    if args.skip_train and os.path.exists(best_path):
        log.info(f"--skip_train set, loading {best_path}")
        model.load_state_dict(torch.load(best_path, weights_only=True))
        args.epochs = 0
        best_auc = float('nan')

    log.info(f"Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            log.info(f"Epoch {epoch:3d} | TrainLoss={train_loss:.4f} "
                     f"| ValLoss={val_loss:.4f} | ValAUC={val_auc:.4f} "
                     f"| Best={best_auc:.4f}")

        if patience_counter >= args.patience:
            log.info(f"Early stopping at epoch {epoch}")
            break

    # Score matrix
    log.info(f"Building {args.n_matrix}×{args.n_matrix} score matrix...")
    from adapter_drugban import smiles_to_dgl_graph, integer_label_protein
    import dgl

    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    pairs = config.load_pairs()
    seqs = config.load_sequences()
    proteins = list(seqs.keys())[:args.n_matrix]
    smiles_list = pairs['substrate_smiles'].unique()[:args.n_matrix]
    n_prot = len(proteins)
    n_lig = len(smiles_list)

    prot_ids = [integer_label_protein(seqs[p]) for p in proteins]
    lig_graphs = [smiles_to_dgl_graph(s) for s in smiles_list]

    score_matrix = np.zeros((n_prot, n_lig), dtype=np.float32)
    CHUNK = 32
    with torch.no_grad():
        for i in range(n_prot):
            for j in range(0, n_lig, CHUNK):
                chunk_end = min(j + CHUNK, n_lig)
                graphs = [lig_graphs[k][0] for k in range(j, chunk_end) if lig_graphs[k][0] is not None]
                if not graphs:
                    continue
                batch_g = dgl.batch(graphs).to(device)
                p_batch = prot_ids[i].unsqueeze(0).expand(len(graphs), -1).to(device)
                out = model(batch_g, p_batch)
                if isinstance(out, tuple):
                    out = out[-1]
                out = out.detach().cpu().numpy().flatten()
                score_matrix[i, j:j + len(graphs)] = out

    matrix_path = os.path.join(args.out_dir, 'score_matrix_DrugBAN.npy')
    np.save(matrix_path, score_matrix)
    log.info(f"Score matrix saved: {matrix_path}")

    # Attractor diagnosis
    diag_script = os.path.join(PROJECT_ROOT, 'evaluation', 'attractor_diagnosis.py')
    if os.path.exists(diag_script):
        import subprocess
        subprocess.run([
            sys.executable, diag_script,
            '--matrix', matrix_path,
            '--name', 'DrugBAN',
            '--out_dir', os.path.join(PROJECT_ROOT, 'evaluation', 'attractor_results'),
        ], check=False)

    log.info(f"Done. Best AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
