"""
baselines/adapters/train_original.py — Unified trainer for original baseline models.

Usage:
  python baselines/adapters/train_original.py --model graphdta --out_dir results/graphdta

This script:
  1. Loads data via the appropriate adapter
  2. Imports the original model from external/
  3. Trains with unified hyperparameters
  4. Generates N×N score matrix
  5. Runs attractor_diagnosis.py

Supported models: graphdta, moltrans, gign, gems
(DrugBAN requires DGL in a separate venv — use train_original_drugban.py)
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
    parser.add_argument('--model', required=True,
                        choices=['graphdta', 'moltrans', 'gign', 'gems'])
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_matrix', type=int, default=200)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--patience', type=int, default=15)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model-specific forward calls ─────────────────────────────────────────────

def forward_graphdta(model, batch, device):
    """GraphDTA: PyG batch with .x, .edge_index, .target, .batch → scalar output."""
    batch = batch.to(device)
    out = model(batch)
    # view(-1) stays 1D even when batch_size=1 (squeeze would collapse to scalar)
    return out.view(-1), batch.y.view(-1)


def forward_moltrans(model, batch, device):
    """MolTrans: (drug_ids, drug_mask, prot_ids, prot_mask, label) → score."""
    d_ids, d_mask, p_ids, p_mask, labels = batch
    d_ids = d_ids.to(device)
    d_mask = d_mask.to(device)
    p_ids = p_ids.to(device)
    p_mask = p_mask.to(device)
    labels = labels.to(device)
    score = model(d_ids.long(), p_ids.long(), d_mask, p_mask)
    return score.squeeze(-1), labels.float()


def forward_gign(model, batch, device):
    """GIGN: PyG batch with .x, .edge_index_intra, .edge_index_inter, .pos → scalar."""
    batch = batch.to(device)
    out = model(batch)
    return out, batch.y.squeeze(-1)


def forward_gems(model, batch, device):
    """GEMS: PyG batch with ligand graph + prot_emb + lig_emb → scalar."""
    batch = batch.to(device)
    out = model(batch)
    return out.view(-1), batch.y.view(-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, forward_fn, device):
    model.train()
    total_loss = 0
    n_samples = 0
    for batch in loader:
        optimizer.zero_grad()
        preds, labels = forward_fn(model, batch, device)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n_samples += len(labels)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, forward_fn, device):
    model.eval()
    total_loss = 0
    n_samples = 0
    all_preds, all_labels = [], []
    for batch in loader:
        preds, labels = forward_fn(model, batch, device)
        loss = criterion(preds, labels)
        total_loss += loss.item() * len(labels)
        n_samples += len(labels)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    avg_loss = total_loss / max(n_samples, 1)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    return avg_loss, auc


# ── Score matrix ──────────────────────────────────────────────────────────────

@torch.no_grad()
def build_score_matrix_graphdta(model, config, n_matrix, device):
    """Build N×N protein-ligand score matrix for GraphDTA."""
    from adapter_graphdta import GraphDTADataset, smile_to_graph, seq_to_ids
    from torch_geometric.data import Batch

    pairs = config.load_pairs()
    seqs = config.load_sequences()

    # Get unique proteins and ligands
    proteins = list(seqs.keys())[:n_matrix]
    smiles_list = pairs['substrate_smiles'].unique()[:n_matrix]

    n_prot = len(proteins)
    n_lig = len(smiles_list)

    # Precompute
    prot_ids = [seq_to_ids(seqs[p]) for p in proteins]
    lig_graphs = [smile_to_graph(s) for s in smiles_list]

    # Filter valid
    valid_lig = [(i, g) for i, g in enumerate(lig_graphs) if g is not None]
    n_lig = min(n_matrix, len(valid_lig))
    valid_lig = valid_lig[:n_lig]
    n_prot = min(n_matrix, n_prot)

    score_matrix = np.zeros((n_prot, n_lig), dtype=np.float32)
    model.eval()

    CHUNK = 32
    for i in range(n_prot):
        for j in range(0, n_lig, CHUNK):
            chunk_end = min(j + CHUNK, n_lig)
            batch_graphs = []
            for k in range(j, chunk_end):
                g = valid_lig[k][1].clone()
                # Match dataset: 2D [1, max_len] so PyG concats to [B, max_len]
                g.target = prot_ids[i].unsqueeze(0)
                batch_graphs.append(g)
            batch = Batch.from_data_list(batch_graphs).to(device)
            out = model(batch).cpu().numpy()
            score_matrix[i, j:chunk_end] = out.flatten()[:chunk_end - j]

    return score_matrix


@torch.no_grad()
def build_score_matrix_moltrans(model, config, n_matrix, device, batch_size=32):
    """Build N×N score matrix for MolTrans."""
    from adapter_moltrans import drug2emb, protein2emb

    pairs = config.load_pairs()
    seqs = config.load_sequences()

    proteins = list(seqs.keys())[:n_matrix]
    smiles_list = pairs['substrate_smiles'].unique()[:n_matrix]

    n_prot = len(proteins)
    n_lig = len(smiles_list)

    # Precompute encodings
    prot_encs = [protein2emb(seqs[p]) for p in proteins]
    lig_encs = [drug2emb(s) for s in smiles_list]

    score_matrix = np.zeros((n_prot, n_lig), dtype=np.float32)
    model.eval()

    # drug2emb / protein2emb return numpy arrays — convert to tensors once
    lig_encs = [(torch.as_tensor(ids), torch.as_tensor(mask))
                for ids, mask in lig_encs]
    prot_encs = [(torch.as_tensor(ids), torch.as_tensor(mask))
                 for ids, mask in prot_encs]

    for i in range(n_prot):
        p_ids, p_mask = prot_encs[i]
        for j in range(0, n_lig, batch_size):
            chunk_end = min(j + batch_size, n_lig)
            chunk_size = chunk_end - j

            d_ids_batch = torch.stack([lig_encs[k][0] for k in range(j, chunk_end)])
            d_mask_batch = torch.stack([lig_encs[k][1] for k in range(j, chunk_end)])
            p_ids_batch = p_ids.unsqueeze(0).expand(chunk_size, -1)
            p_mask_batch = p_mask.unsqueeze(0).expand(chunk_size, -1)

            # MolTrans hardcodes int(batch_size/gpus) in .view(); keep gpus=1.
            old_bs = model.batch_size
            model.batch_size = chunk_size
            model.gpus = 1

            out = model(
                d_ids_batch.long().to(device),
                p_ids_batch.long().to(device),
                d_mask_batch.to(device),
                p_mask_batch.to(device),
            )
            model.batch_size = old_bs
            score_matrix[i, j:chunk_end] = out.cpu().numpy().flatten()[:chunk_size]

    return score_matrix


@torch.no_grad()
def build_score_matrix_gign(model, config, n_matrix, device):
    """Build N×N score matrix for GIGN (requires docked structures)."""
    from adapter_gign import load_docked_complex
    from torch_geometric.data import Batch

    pairs = config.load_pairs()
    seqs = config.load_sequences()
    docked_dir = os.path.join(config.project_root, 'data', 'docked_complexes')

    proteins = list(seqs.keys())[:n_matrix]
    smiles_list = pairs['substrate_smiles'].unique()[:n_matrix]

    n_prot = min(n_matrix, len(proteins))
    n_lig = min(n_matrix, len(smiles_list))

    score_matrix = np.zeros((n_prot, n_lig), dtype=np.float32)
    model.eval()

    for i in range(n_prot):
        pdb_path = os.path.join(config.struct_dir, f'{proteins[i]}.pdb')
        if not os.path.exists(pdb_path):
            continue
        for j in range(n_lig):
            # Try to find docked structure
            sdf_path = os.path.join(docked_dir, f'{proteins[i]}_{j}.sdf')
            if not os.path.exists(sdf_path):
                continue
            data = load_docked_complex(pdb_path, sdf_path)
            if data is None:
                continue
            batch = Batch.from_data_list([data]).to(device)
            out = model(batch)
            score_matrix[i, j] = out.cpu().item()

    return score_matrix


@torch.no_grad()
def build_score_matrix_gems(model, config, n_matrix, device):
    """Build N×N score matrix for GEMS."""
    from adapter_gems import smiles_to_ligand_graph, get_chemberta_embedding
    from torch_geometric.data import Data, Batch

    pairs = config.load_pairs()
    seqs = config.load_sequences()
    esm2_dir = os.path.join(config.project_root, 'data', 'esm2_embeddings')

    proteins = list(seqs.keys())[:n_matrix]
    smiles_list = pairs['substrate_smiles'].unique()[:n_matrix]

    n_prot = min(n_matrix, len(proteins))
    n_lig = min(n_matrix, len(smiles_list))

    # Precompute protein embeddings
    prot_embs = []
    for p in proteins[:n_prot]:
        esm_path = os.path.join(esm2_dir, f'{p}.pt')
        if os.path.exists(esm_path):
            emb = torch.load(esm_path, weights_only=True).mean(dim=0)
        else:
            emb = torch.zeros(1280)
        prot_embs.append(emb)

    # Precompute ligand graphs — lig_emb/prot_emb stored as [1, D] so PyG
    # batches to [B, D] (matching the adapter's __getitem__ convention).
    lig_graphs = []
    for s in smiles_list[:n_lig]:
        g = smiles_to_ligand_graph(s)
        if g is None:
            g = Data(x=torch.zeros(1, 8), edge_index=torch.zeros(2, 0, dtype=torch.long),
                     edge_attr=torch.zeros(0, 4))
        g.lig_emb = get_chemberta_embedding(s).unsqueeze(0)
        lig_graphs.append(g)
    prot_embs = [p.unsqueeze(0) for p in prot_embs]

    score_matrix = np.zeros((n_prot, n_lig), dtype=np.float32)
    model.eval()

    CHUNK = 32
    for i in range(n_prot):
        for j in range(0, n_lig, CHUNK):
            chunk_end = min(j + CHUNK, n_lig)
            batch_graphs = []
            for k in range(j, chunk_end):
                g = lig_graphs[k].clone()
                g.prot_emb = prot_embs[i]
                batch_graphs.append(g)
            batch = Batch.from_data_list(batch_graphs).to(device)
            out = model(batch)
            score_matrix[i, j:chunk_end] = out.cpu().numpy().flatten()[:chunk_end - j]

    return score_matrix


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log.info(f"Model: {args.model}, Device: {device}")

    if args.out_dir is None:
        args.out_dir = os.path.join(PROJECT_ROOT, 'results', f'original_{args.model}')
    os.makedirs(args.out_dir, exist_ok=True)

    # Import adapter
    from common import BRENDADataConfig
    config = BRENDADataConfig(seed=args.seed)

    # Get splits
    train_idx, val_idx, test_idx = config.get_protein_split()
    log.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── Model-specific setup ──────────────────────────────────────────────────
    if args.model == 'graphdta':
        from adapter_graphdta import GraphDTADataset, get_model
        from torch_geometric.loader import DataLoader as PyGLoader
        train_ds = GraphDTADataset(config, train_idx)
        val_ds = GraphDTADataset(config, val_idx)
        train_loader = PyGLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = PyGLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        ModelClass = get_model()
        model = ModelClass().to(device)
        forward_fn = forward_graphdta
        build_score_matrix = build_score_matrix_graphdta

    elif args.model == 'moltrans':
        from adapter_moltrans import MolTransDataset, get_model_config
        train_ds = MolTransDataset(config, train_idx)
        val_ds = MolTransDataset(config, val_idx)
        # MolTrans hardcodes batch_size in .view() → drop partial batches
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                drop_last=True)
        # MolTrans needs config dict
        model_config = get_model_config(batch_size=args.batch_size)
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'MolTrans'))
        from models import BIN_Interaction_Flat
        model = BIN_Interaction_Flat(**model_config).to(device)
        # MolTrans uses `self.batch_size / self.gpus` in its forward view().
        # Force gpus=1 so the view always matches our actual batch size on
        # single-GPU compute nodes (login node may report 2 GPUs).
        model.gpus = 1
        forward_fn = forward_moltrans
        build_score_matrix = lambda m, c, n, d: build_score_matrix_moltrans(
            m, c, n, d, batch_size=args.batch_size)

    elif args.model == 'gign':
        from adapter_gign import GIGNDataset, get_model
        from torch_geometric.loader import DataLoader as PyGLoader
        train_ds = GIGNDataset(config, train_idx)
        val_ds = GIGNDataset(config, val_idx)
        train_loader = PyGLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = PyGLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        ModelClass = get_model()
        # GIGN expects node_dim (from atom features) and hidden_dim
        model = ModelClass(node_dim=10, hidden_dim=128).to(device)
        forward_fn = forward_gign
        build_score_matrix = build_score_matrix_gign

    elif args.model == 'gems':
        from adapter_gems import GEMSDataset, get_model
        from torch_geometric.loader import DataLoader as PyGLoader
        train_ds = GEMSDataset(config, train_idx)
        val_ds = GEMSDataset(config, val_idx)
        # GEMS has BatchNorm layers — drop_last to avoid batch_size=1 crash
        train_loader = PyGLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                 drop_last=True)
        val_loader = PyGLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               drop_last=True)
        ModelClass = get_model()
        # GEMS18d: dropout_prob, in_channels, edge_dim, conv_dropout_prob
        model = ModelClass(
            dropout_prob=0.1, in_channels=8, edge_dim=4, conv_dropout_prob=0.1
        ).to(device)
        forward_fn = forward_gems
        build_score_matrix = build_score_matrix_gems

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")
    log.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # ── Training loop ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_path = os.path.join(args.out_dir, 'best_model.pt')
    patience_counter = 0

    log.info(f"Training for {args.epochs} epochs (patience={args.patience})...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 forward_fn, device)
        val_loss, val_auc = eval_epoch(model, val_loader, criterion,
                                       forward_fn, device)
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
            log.info(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Score matrix ──────────────────────────────────────────────────────────
    log.info(f"Loading best model (AUC={best_auc:.4f}) for score matrix...")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    log.info(f"Building {args.n_matrix}×{args.n_matrix} score matrix...")
    score_matrix = build_score_matrix(model, config, args.n_matrix, device)

    matrix_path = os.path.join(args.out_dir, f'score_matrix_{args.model}.npy')
    np.save(matrix_path, score_matrix)
    log.info(f"Score matrix saved: {matrix_path} (shape={score_matrix.shape})")

    # ── Attractor diagnosis ───────────────────────────────────────────────────
    log.info("Running attractor diagnosis...")
    diag_script = os.path.join(PROJECT_ROOT, 'evaluation', 'attractor_diagnosis.py')
    if os.path.exists(diag_script):
        import subprocess
        subprocess.run([
            sys.executable, diag_script,
            '--score_matrix', matrix_path,
            '--model_name', args.model,
            '--out_dir', os.path.join(PROJECT_ROOT, 'evaluation', 'attractor_results'),
        ], check=False)

    log.info("Training complete.")
    log.info(f"Results saved to: {args.out_dir}")
    log.info(f"Best validation AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()
