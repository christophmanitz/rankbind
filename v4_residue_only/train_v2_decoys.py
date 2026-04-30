"""
train_v2_decoys.py — ResidueOnlyBind re-training on original data + 4000
                     shuffled decoys (non-binders).

All model architecture and training logic is shared with train.py.
New additions:
  - ShuffledDecoyDataset: wraps data/processed_hieratom_shuffled/ .pt files
  - Combined train/val loaders: HierAtomBindAugmentedDataset + ShuffledDecoyDataset
  - Separate output paths (checkpoints_v2/, logs_v2/, plots_v2/) so no
    existing results are overwritten.

Usage:
  sbatch run_train_v2.sh
  (or: python train_v2_decoys.py directly for quick testing)
"""

import os
import sys
import glob
import random
import math
import csv
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, _HERE)

# ── Import shared utilities from train.py ─────────────────────────────────────
# train.py sets up logging and device at module level — that's fine.
import train as _t

# Re-export everything we need from train.py under local names
get_phase              = _t.get_phase
get_phase_config       = _t.get_phase_config
build_optimizer        = _t.build_optimizer
set_phase_freeze       = _t.set_phase_freeze
update_lrs_for_phase3  = _t.update_lrs_for_phase3
compute_target_stats   = _t.compute_target_stats
cap_per_protein        = _t.cap_per_protein
clip_gradients         = _t.clip_gradients
train_epoch            = _t.train_epoch
validate_epoch         = _t.validate_epoch
init_metrics_csv       = _t.init_metrics_csv
log_metrics_csv        = _t.log_metrics_csv
plot_training_curves   = _t.plot_training_curves
plot_predictions       = _t.plot_predictions

# Shared training hyper-parameters
CSV_PATH         = _t.CSV_PATH
PROTEIN_DIR      = _t.PROTEIN_DIR
SMILES_COL       = _t.SMILES_COL
PROTEIN_COL      = _t.PROTEIN_COL
TARGET_COLS      = _t.TARGET_COLS
EC_COL           = _t.EC_COL
TANIMOTO_COL     = _t.TANIMOTO_COL
DATASET_ROOT     = _t.DATASET_ROOT
BATCH_SIZE       = _t.BATCH_SIZE
NUM_EPOCHS       = _t.NUM_EPOCHS
WEIGHT_DECAY     = _t.WEIGHT_DECAY
EARLY_STOP_PATIENCE = _t.EARLY_STOP_PATIENCE
VAL_SPLIT        = _t.VAL_SPLIT
RANDOM_SEED      = _t.RANDOM_SEED
MAX_PER_PROTEIN  = _t.MAX_PER_PROTEIN
HARD_NEG_PER_POS = _t.HARD_NEG_PER_POS
PLOT_EVERY       = _t.PLOT_EVERY
PHASE0_END       = _t.PHASE0_END
PHASE1_END       = _t.PHASE1_END
WARMUP_EPOCHS    = _t.WARMUP_EPOCHS
WARMUP_GROUPS    = _t.WARMUP_GROUPS
TARGET_LRS       = _t.TARGET_LRS
device           = _t.device

# ── v2-specific imports ───────────────────────────────────────────────────────
from dataset import HierAtomBindDataset, HierAtomBindAugmentedDataset, collate_fn
from ResidueOnlyBind import ResidueOnlyBind

# ── v2 output paths (never overwrite existing results) ───────────────────────
DECOY_DIR           = os.path.join(PROJECT_ROOT, 'data', 'processed_hieratom_shuffled')
BEST_MODEL_PATH_V2  = os.path.join(_HERE, 'checkpoints_v2', 'best_model.pt')
METRICS_CSV_PATH_V2 = os.path.join(_HERE, 'logs_v2', 'metrics.csv')
PLOT_DIR_V2         = os.path.join(_HERE, 'plots_v2')

os.makedirs(os.path.join(_HERE, 'checkpoints_v2'), exist_ok=True)
os.makedirs(os.path.join(_HERE, 'logs_v2'),        exist_ok=True)
os.makedirs(PLOT_DIR_V2,                           exist_ok=True)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ShuffledDecoyDataset
# ─────────────────────────────────────────────────────────────────────────────

class ShuffledDecoyDataset(Dataset):
    """
    Wraps the pre-generated shuffled decoy .pt files.

    Each file contains (mol_graph, protein_graph, y=tensor([0.0])).
    Returns the same 8-tuple as HierAtomBindAugmentedDataset so that
    collate_fn and compute_total_loss work unchanged:
      (mol_graph, protein_graph, y, prot_id_str, tanimoto, ec_class,
       is_hard_neg, lig_idx)
    """

    # Large offset so lig_idx values don't collide with original dataset indices
    LIG_ID_OFFSET = 200_000

    def __init__(self, decoy_dir):
        self.files = sorted(
            glob.glob(os.path.join(decoy_dir, 'decoy_*.pt')),
            key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
        )
        if not self.files:
            raise RuntimeError(
                f'No decoy files found in {decoy_dir}. '
                'Run generate_decoys.py first.'
            )
        log.info(f'ShuffledDecoyDataset: {len(self.files)} decoys from {decoy_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mol, prot, y = torch.load(self.files[idx], weights_only=False)

        # Unique string protein ID per file slot — collate_fn uses np.unique
        # to convert these to batch-local integers; decoys are excluded from
        # contrastive losses (is_hard_neg=True & y=0), so the exact value
        # only matters for uniqueness within the batch.
        prot_id = f'__decoy_prot_{idx}__'

        tanimoto    = 0.0
        ec_class    = (prot.ec_class
                       if hasattr(prot, 'ec_class')
                       else torch.tensor(-1, dtype=torch.long))
        is_hard_neg = True
        lig_idx     = self.LIG_ID_OFFSET + idx

        return mol, prot, y, prot_id, tanimoto, ec_class, is_hard_neg, lig_idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Original dataset ──────────────────────────────────────────────────────
    dataset = HierAtomBindDataset(
        root=DATASET_ROOT, csv_path=CSV_PATH,
        smiles_col=SMILES_COL, protein_col=PROTEIN_COL,
        protein_dir=PROTEIN_DIR, target_cols=TARGET_COLS,
        ec_col=EC_COL, tanimoto_col=TANIMOTO_COL, resume=True,
    )

    if len(dataset) == 0:
        raise RuntimeError('Dataset is empty — check CSV and protein directory.')
    log.info(f'Original dataset size: {len(dataset)}')

    df = pd.read_csv(CSV_PATH)
    if EC_COL and EC_COL in df.columns:
        df = df[df[EC_COL].astype(str).str.split('.').str[0] == '3'].reset_index(drop=True)
    file_indices = [
        int(os.path.basename(f).split('_')[1].split('.')[0])
        for f in dataset.files
    ]
    protein_ids_arr = df.iloc[file_indices][PROTEIN_COL].values

    all_indices = list(range(len(dataset)))
    all_indices = cap_per_protein(all_indices, protein_ids_arr, MAX_PER_PROTEIN, rng)

    train_idx, val_idx = train_test_split(
        all_indices, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    y_mean, y_std = compute_target_stats(dataset, train_idx)
    log.info(f'Target stats: mean={y_mean:.3f}, std={y_std:.3f}')

    # ── Original augmented datasets (with in-memory hard negatives) ───────────
    aug_train_ds = HierAtomBindAugmentedDataset(
        dataset, train_idx, protein_ids_arr,
        hard_neg_per_pos=HARD_NEG_PER_POS, seed=RANDOM_SEED,
    )
    aug_val_ds = HierAtomBindAugmentedDataset(
        dataset, val_idx, protein_ids_arr,
        hard_neg_per_pos=HARD_NEG_PER_POS, seed=RANDOM_SEED + 1,
    )

    # ── Shuffled decoy dataset split 80/20 ────────────────────────────────────
    decoy_ds = ShuffledDecoyDataset(DECOY_DIR)
    n_decoy  = len(decoy_ds)
    n_decoy_train = int(n_decoy * (1.0 - VAL_SPLIT))

    # Shuffle decoy indices before splitting so val set is representative
    decoy_order = list(range(n_decoy))
    random.Random(RANDOM_SEED + 99).shuffle(decoy_order)

    decoy_train_ds = Subset(decoy_ds, decoy_order[:n_decoy_train])
    decoy_val_ds   = Subset(decoy_ds, decoy_order[n_decoy_train:])

    log.info(
        f'Decoys — train: {len(decoy_train_ds)}, val: {len(decoy_val_ds)}'
    )

    # ── Combined datasets ─────────────────────────────────────────────────────
    train_ds = ConcatDataset([aug_train_ds, decoy_train_ds])
    val_ds   = ConcatDataset([aug_val_ds,   decoy_val_ds])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    log.info(
        f'Train: {len(train_ds)} '
        f'(orig_aug={len(aug_train_ds)}, decoys={len(decoy_train_ds)}) | '
        f'Val: {len(val_ds)} '
        f'(orig_aug={len(aug_val_ds)}, decoys={len(decoy_val_ds)})'
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ResidueOnlyBind().to(device)
    optimizer = build_optimizer(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'ResidueOnlyBind parameters: {n_params:,}')

    # ── Training loop ─────────────────────────────────────────────────────────
    init_metrics_csv(METRICS_CSV_PATH_V2)

    best_val_roc_auc = 0.0
    patience_counter = 0
    prev_phase       = -1
    last_val_metrics = {}

    for epoch in range(1, NUM_EPOCHS + 1):
        phase = get_phase(epoch)

        if phase != prev_phase:
            log.info(f'=== Entering Phase {phase} (epoch {epoch}) ===')
            torch.cuda.empty_cache()
            set_phase_freeze(model, phase)
            if phase == 3:
                update_lrs_for_phase3(optimizer)
            prev_phase = phase
            patience_counter = 0

        # LR warmup for freshly unfrozen modules at Phase 1
        if phase == 1:
            warmup_progress = min((epoch - PHASE0_END) / WARMUP_EPOCHS, 1.0)
            warmup_factor   = 0.1 + 0.9 * warmup_progress
            for g in optimizer.param_groups:
                name = g.get('name', '')
                if name in WARMUP_GROUPS:
                    g['lr'] = TARGET_LRS.get(name, 1e-4) * warmup_factor

        phase_config = get_phase_config(epoch)

        train_m, gnorm_res, gnorm_lig, n_nan = train_epoch(
            model, train_loader, optimizer, phase_config, y_mean, y_std
        )
        val_m = validate_epoch(model, val_loader, phase_config, y_mean, y_std)

        dom_ratio = gnorm_res / (gnorm_lig + 1e-8)

        log.info(
            f'Epoch {epoch:3d} | Phase {phase} | '
            f'train={train_m.get("loss", 0):.4f} | '
            f'val={val_m.get("loss", 0):.4f} | '
            f'roc_auc={val_m.get("roc_auc", 0):.4f} | '
            f'disc_acc={val_m.get("disc_acc", 0):.4f} | '
            f'dom={dom_ratio:.2f} | nan_skip={n_nan}'
        )

        log_metrics_csv(METRICS_CSV_PATH_V2, epoch, phase, train_m, val_m, n_nan)
        last_val_metrics = val_m

        if epoch % PLOT_EVERY == 0 or epoch == 1:
            plot_training_curves(METRICS_CSV_PATH_V2, PLOT_DIR_V2)

        # Checkpoint — saved to checkpoints_v2/
        torch.save({
            'epoch':       epoch,
            'phase':       phase,
            'model_state': model.state_dict(),
            'opt_state':   optimizer.state_dict(),
            'val_metrics': {k: v for k, v in val_m.items()
                            if not k.startswith('_')},
            'y_mean':      y_mean,
            'y_std':       y_std,
        }, os.path.join(_HERE, 'checkpoints_v2', f'epoch_{epoch:03d}.pt'))

        val_roc_auc = val_m.get('roc_auc', 0.0)
        if np.isfinite(val_roc_auc) and val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH_V2)
            log.info(f'  → New best model (roc_auc={best_val_roc_auc:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log.info(
                    f'Early stopping at epoch {epoch} '
                    f'(no improvement for {EARLY_STOP_PATIENCE} epochs).'
                )
                break

    # ── Final plots ───────────────────────────────────────────────────────────
    plot_training_curves(METRICS_CSV_PATH_V2, PLOT_DIR_V2)

    probs  = last_val_metrics.get('_probs',         np.array([]))
    labels = last_val_metrics.get('_labels',        np.array([]))
    a_pred = last_val_metrics.get('_affinity_pred', np.array([]))
    a_true = last_val_metrics.get('_affinity_true', np.array([]))

    if len(probs) > 0:
        plot_predictions(probs, labels, a_pred, a_true, PLOT_DIR_V2, epoch)
        log.info(f'Prediction plot saved to {PLOT_DIR_V2}/predictions_final.png')


if __name__ == '__main__':
    main()
