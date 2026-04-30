"""
train.py — ResidueOnlyBind training, 100 epochs, 3-phase curriculum.

Phases:
  Phase 0 (1–5)     Protein-only warm-up: residue_gnn + ec_head only.
                    Loss: EC auxiliary only.
  Phase 1 (6–20)    All unfrozen. BCE + pair_nce + triplet + EC.
                    Regression off. LR warmup for freshly unfrozen modules.
  Phase 2 (21–50)   + regression ramp (0→1) + tanimoto contrastive.
  Phase 3 (51–100)  Full training, all LRs equalised to 2e-4.

Optimizer parameter groups (AdamW):
  residue_gnn   LR=3e-4
  ligand_gnn    LR=1e-4
  fusion_heads  LR=1e-4
  ec_head       LR=3e-4

Gradient clipping:
  affinity_head  max_norm=1.0
  ec_head        max_norm=2.0
  rest           max_norm=5.0

Plots generated (in plots/):
  training_curves.png       — loss, ROC-AUC, disc_acc, loss components
  training_diagnostics.png  — gradient norms, train/val overfitting
  training_analytics.png    — per-phase distributions, metric improvement rate
  predictions_final.png     — ROC curve, PR curve, score distributions, affinity scatter
"""

import os
import sys
import logging
import csv
import random
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)

# ── Project root on sys.path ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from dataset import HierAtomBindDataset, HierAtomBindAugmentedDataset, collate_fn
from losses import compute_total_loss
from ResidueOnlyBind import ResidueOnlyBind

# ─────────────────────────────────────────────────────────────────────────────
# Config  (overridable from run_train.sh via module-level assignment)
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH     = os.path.join(PROJECT_ROOT, 'data', 'dataset_with_decoys.csv')
PROTEIN_DIR  = '/home/sc.uni-leipzig.de/zw93onug/hpc/structures'
SMILES_COL   = 'substrate_smiles'
PROTEIN_COL  = 'uniprot'
TARGET_COLS  = ['value']
EC_COL       = 'ec'
TANIMOTO_COL = 'TanimotoSimilarity'
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data')

BEST_MODEL_PATH  = os.path.join(_HERE, 'checkpoints', 'best_model.pt')
METRICS_CSV_PATH = os.path.join(_HERE, 'logs', 'metrics.csv')
PLOT_DIR         = os.path.join(_HERE, 'plots')
PLOT_EVERY       = 5

BATCH_SIZE          = 4
NUM_EPOCHS          = 100
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 30
VAL_SPLIT           = 0.2
RANDOM_SEED         = 42
MAX_PER_PROTEIN     = 20
HARD_NEG_PER_POS    = 1

# ── Phase boundaries ──────────────────────────────────────────────────────────
PHASE0_END = 5
PHASE1_END = 20
PHASE2_END = 50
# Phase 3: epoch > 50

WARMUP_EPOCHS = 3
WARMUP_GROUPS = {'ligand_gnn', 'fusion_heads'}
TARGET_LRS = {
    'residue_gnn': 3e-4,
    'ligand_gnn':  1e-4,
    'fusion_heads': 1e-4,
    'ec_head':     3e-4,
}

REG_WARMUP_START = PHASE1_END + 1
REG_WARMUP_END   = PHASE2_END

os.makedirs(os.path.join(_HERE, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(_HERE, 'logs'),        exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Device: {device}')


# ─────────────────────────────────────────────────────────────────────────────
# Phase configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_phase(epoch):
    if epoch <= PHASE0_END:
        return 0
    elif epoch <= PHASE1_END:
        return 1
    elif epoch <= PHASE2_END:
        return 2
    else:
        return 3


def get_phase_config(epoch):
    phase = get_phase(epoch)
    # attn_entropy and site_entropy are always 0 (no cross-attn, no site-selector)
    # distill and cross are 0 because z_prot and z_prot_res would be degenerate
    if phase == 0:
        return dict(
            w_cls=0.0, w_reg=0.0, reg_weight=0.0,
            w_pair_nce=0.0, w_distill=0.0, w_cross=0.0,
            w_triplet=0.0, w_ec=1.0, w_tanimoto=0.0,
            w_attn_entropy=0.0, w_site_entropy=0.0,
        )
    elif phase == 1:
        return dict(
            w_cls=1.0, w_reg=0.0, reg_weight=0.0,
            w_pair_nce=0.20, w_distill=0.0, w_cross=0.0,
            w_triplet=0.30, w_ec=0.50, w_tanimoto=0.0,
            w_attn_entropy=0.0, w_site_entropy=0.0,
        )
    elif phase == 2:
        progress   = (epoch - PHASE1_END) / max(PHASE2_END - PHASE1_END, 1)
        reg_weight = float(np.clip(progress, 0, 1))
        return dict(
            w_cls=1.0, w_reg=1.0, reg_weight=reg_weight,
            w_pair_nce=0.20, w_distill=0.0, w_cross=0.0,
            w_triplet=0.30, w_ec=0.50, w_tanimoto=0.20,
            w_attn_entropy=0.0, w_site_entropy=0.0,
        )
    else:  # Phase 3
        return dict(
            w_cls=1.0, w_reg=1.0, reg_weight=1.0,
            w_pair_nce=0.20, w_distill=0.0, w_cross=0.0,
            w_triplet=0.30, w_ec=0.50, w_tanimoto=0.20,
            w_attn_entropy=0.0, w_site_entropy=0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer & freeze management
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model):
    def _p(module):
        return list(module.parameters())

    groups = [
        {'params': _p(model.residue_gnn), 'lr': 3e-4, 'name': 'residue_gnn'},
        {'params': _p(model.ligand_gnn),  'lr': 1e-4, 'name': 'ligand_gnn'},
        {'params': (
            _p(model.fusion_mlp) +
            _p(model.bind_head) +
            _p(model.affinity_head) +
            _p(model.res_pool) +
            _p(model.lig_pool) +
            _p(model.prot_proj) +
            _p(model.lig_proj) +
            _p(model.prot_res_proj)
        ), 'lr': 1e-4, 'name': 'fusion_heads'},
        {'params': _p(model.ec_head), 'lr': 3e-4, 'name': 'ec_head'},
    ]
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def set_phase_freeze(model, phase):
    if phase == 0:
        frozen = [
            model.ligand_gnn,
            model.fusion_mlp, model.bind_head, model.affinity_head,
            model.lig_pool, model.prot_proj, model.lig_proj,
        ]
        for m in frozen:
            for p in m.parameters():
                p.requires_grad_(False)
        trainable = [
            model.residue_gnn, model.res_pool,
            model.prot_res_proj, model.ec_head,
        ]
        for m in trainable:
            for p in m.parameters():
                p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(True)


def update_lrs_for_phase3(optimizer):
    for g in optimizer.param_groups:
        g['lr'] = 2e-4


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_target_stats(dataset, indices):
    vals = []
    for i in indices:
        _, _, y = dataset[i]
        v = y[0].item()
        if v > 0:
            vals.append(math.log10(v))
    if len(vals) < 2:
        return 0.0, 1.0
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def cap_per_protein(indices, protein_ids_arr, max_per_protein, rng):
    prot_to_indices = defaultdict(list)
    for i in indices:
        prot_to_indices[protein_ids_arr[i]].append(i)
    result = []
    for pid, idx_list in prot_to_indices.items():
        if len(idx_list) > max_per_protein:
            result.extend(rng.sample(idx_list, max_per_protein))
        else:
            result.extend(idx_list)
    return result


def clip_gradients(model):
    nn.utils.clip_grad_norm_(model.affinity_head.parameters(), max_norm=1.0)
    nn.utils.clip_grad_norm_(model.ec_head.parameters(),       max_norm=2.0)
    other = [p for n, p in model.named_parameters()
             if not any(s in n for s in ('affinity_head', 'ec_head'))
             and p.grad is not None]
    if other:
        nn.utils.clip_grad_norm_(other, max_norm=5.0)


def _module_grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Train / Validate
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, phase_config, y_mean, y_std):
    model.train()
    accum    = defaultdict(float)
    gnorm_res = gnorm_lig = 0.0
    n_batches = n_nan = 0

    for batch in loader:
        (lig_b, prot_b, y, prot_ids, tanimoto, ec_class, is_hn, lig_ids) = batch
        lig_b    = lig_b.to(device)
        prot_b   = prot_b.to(device)
        y        = y.float().to(device)
        prot_ids = prot_ids.to(device)
        tanimoto = tanimoto.to(device)
        ec_class = ec_class.to(device)
        is_hn    = is_hn.to(device)

        optimizer.zero_grad()
        outputs = model(lig_b, prot_b)
        batch_dev = (lig_b, prot_b, y, prot_ids, tanimoto, ec_class, is_hn, lig_ids)
        total, details = compute_total_loss(outputs, batch_dev, phase_config,
                                            y_mean=y_mean, y_std=y_std)

        if not torch.isfinite(total):
            n_nan += 1
            continue
        if not total.requires_grad:
            n_nan += 1
            continue

        total.backward()

        has_nan_grad = any(
            torch.isnan(p.grad).any()
            for p in model.parameters() if p.grad is not None
        )
        if has_nan_grad:
            optimizer.zero_grad()
            n_nan += 1
            continue

        clip_gradients(model)
        gnorm_res += _module_grad_norm(model.residue_gnn)
        gnorm_lig += _module_grad_norm(model.ligand_gnn)
        optimizer.step()

        for k, v in details.items():
            if v == v:
                accum[k] += v
        n_batches += 1

    if n_batches == 0:
        return {}, 0.0, 0.0, n_nan

    metrics = {k: v / n_batches for k, v in accum.items()}
    metrics['grad_norm_residue_gnn'] = gnorm_res / n_batches
    metrics['grad_norm_ligand_gnn']  = gnorm_lig / n_batches

    return metrics, gnorm_res / n_batches, gnorm_lig / n_batches, n_nan


@torch.no_grad()
def validate_epoch(model, loader, phase_config, y_mean, y_std):
    model.eval()
    accum = defaultdict(float)
    n_batches = 0
    all_probs, all_labels, all_affinity_pred, all_affinity_true = [], [], [], []
    lig_pos_scores = {}
    lig_neg_scores = defaultdict(list)

    for batch in loader:
        (lig_b, prot_b, y, prot_ids, tanimoto, ec_class, is_hn, lig_ids) = batch
        lig_b    = lig_b.to(device)
        prot_b   = prot_b.to(device)
        y        = y.float().to(device)
        prot_ids = prot_ids.to(device)
        tanimoto = tanimoto.to(device)
        ec_class = ec_class.to(device)
        is_hn    = is_hn.to(device)

        outputs = model(lig_b, prot_b)
        bind_logit, affinity = outputs[0], outputs[1]

        batch_dev = (lig_b, prot_b, y, prot_ids, tanimoto, ec_class, is_hn, lig_ids)
        _, details = compute_total_loss(outputs, batch_dev, phase_config,
                                        y_mean=y_mean, y_std=y_std)
        for k, v in details.items():
            accum[k] += v
        n_batches += 1

        probs      = torch.sigmoid(bind_logit).cpu().numpy()
        y_np       = y.view(-1).cpu().numpy()
        is_hn_np   = is_hn.cpu().numpy()
        lig_ids_np = lig_ids.cpu().numpy()
        aff_np     = affinity.cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend((y_np > 0).astype(float))

        for bi in range(len(y_np)):
            li = int(lig_ids_np[bi])
            if not is_hn_np[bi] and y_np[bi] > 0:
                lig_pos_scores[li] = float(probs[bi])
                all_affinity_pred.append(float(aff_np[bi]))
                all_affinity_true.append(float(math.log10(max(y_np[bi], 1e-10))))
            elif is_hn_np[bi]:
                lig_neg_scores[li].append(float(probs[bi]))

    metrics = {k: v / max(n_batches, 1) for k, v in accum.items()}

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)

    try:
        metrics['roc_auc'] = roc_auc_score(labels_arr, probs_arr)
    except ValueError:
        metrics['roc_auc'] = float('nan')
    try:
        metrics['avg_precision'] = average_precision_score(labels_arr, probs_arr)
    except ValueError:
        metrics['avg_precision'] = float('nan')

    n_correct = n_total = 0
    for li, pos_s in lig_pos_scores.items():
        neg_ss = lig_neg_scores.get(li, [])
        if neg_ss:
            n_total  += 1
            n_correct += int(pos_s > max(neg_ss))
    metrics['disc_acc'] = n_correct / max(n_total, 1)

    # Store raw arrays for prediction plots
    metrics['_probs']          = probs_arr
    metrics['_labels']         = labels_arr
    metrics['_affinity_pred']  = np.array(all_affinity_pred)
    metrics['_affinity_true']  = np.array(all_affinity_true)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

def init_metrics_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'phase',
            'train_loss', 'train_cls', 'train_reg', 'train_pair_nce',
            'train_ec', 'train_triplet', 'train_tanimoto',
            'train_grad_norm_res', 'train_grad_norm_lig', 'nan_skipped',
            'val_loss', 'val_cls', 'val_reg', 'val_ec',
            'val_roc_auc', 'val_avg_precision', 'val_disc_acc',
        ])


def log_metrics_csv(path, epoch, phase, train_m, val_m, nan_skipped):
    with open(path, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch, phase,
            train_m.get('loss', 0),      train_m.get('cls', 0),
            train_m.get('reg', 0),       train_m.get('pair_nce', 0),
            train_m.get('ec', 0),        train_m.get('triplet', 0),
            train_m.get('tanimoto', 0),
            train_m.get('grad_norm_residue_gnn', 0),
            train_m.get('grad_norm_ligand_gnn', 0),
            nan_skipped,
            val_m.get('loss', 0),  val_m.get('cls', 0),
            val_m.get('reg', 0),   val_m.get('ec', 0),
            val_m.get('roc_auc', 0), val_m.get('avg_precision', 0),
            val_m.get('disc_acc', 0),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Training plots
# ─────────────────────────────────────────────────────────────────────────────

PHASE_COLORS = {0: '#2196F3', 1: '#FF9800', 2: '#4CAF50', 3: '#F44336'}
PHASE_LABELS = {
    0: 'P0: EC only',
    1: 'P1: +BCE/NCE',
    2: 'P2: +Regression',
    3: 'P3: Full',
}


def _add_phase_bg(ax, ep, phases):
    for p in sorted(set(phases)):
        mask = phases == p
        if mask.any():
            idxs = np.where(mask)[0]
            ax.axvspan(ep[idxs[0]] - 0.5, ep[idxs[-1]] + 0.5,
                       alpha=0.08, color=PHASE_COLORS.get(int(p), 'gray'))


def plot_training_curves(csv_path, plot_dir):
    """Generate 3-figure training plot suite from metrics CSV."""
    os.makedirs(plot_dir, exist_ok=True)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    if len(df) < 2:
        return

    ep     = df['epoch'].values
    phases = df['phase'].values

    # ── Figure 1: Main overview ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ResidueOnlyBind — Training Progress', fontsize=14, fontweight='bold')

    # 1a: Loss
    ax = axes[0, 0]
    _add_phase_bg(ax, ep, phases)
    ax.plot(ep, df['train_loss'], 'o-', ms=3, lw=1.2, color='#1565C0', label='Train')
    ax.plot(ep, df['val_loss'],   's-', ms=3, lw=1.2, color='#C62828', label='Val')
    ax.set_ylabel('Loss'); ax.set_title('Total Loss')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    tl = df['train_loss'].values
    nonzero = tl[(tl > 0) & np.isfinite(tl)]
    if len(nonzero) > 0 and nonzero.max() / (nonzero.min() + 1e-10) > 50:
        ax.set_yscale('log')

    # 1b: ROC-AUC & Avg Precision
    ax = axes[0, 1]
    _add_phase_bg(ax, ep, phases)
    roc = df['val_roc_auc'].values.astype(float)
    ap  = df['val_avg_precision'].values.astype(float)
    vr  = np.isfinite(roc) & (roc > 0)
    va  = np.isfinite(ap)  & (ap  > 0)
    if vr.any():
        ax.plot(ep[vr], roc[vr], 'o-', ms=3, lw=1.2, color='#6A1B9A', label='ROC-AUC')
    if va.any():
        ax.plot(ep[va], ap[va],  's-', ms=3, lw=1.2, color='#00838F', label='Avg Prec')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.4, label='Random')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
    ax.set_title('Validation Classification')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 1c: Discrimination Accuracy
    ax = axes[1, 0]
    _add_phase_bg(ax, ep, phases)
    da  = df['val_disc_acc'].values.astype(float)
    vd  = np.isfinite(da) & (da > 0)
    if vd.any():
        ax.plot(ep[vd], da[vd], 'o-', ms=3, lw=1.2, color='#00695C')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.4, label='Random')
    ax.set_ylim(0, 1.05); ax.set_ylabel('Disc Acc')
    ax.set_title('Discrimination Accuracy')
    ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 1d: Loss components
    ax = axes[1, 1]
    _add_phase_bg(ax, ep, phases)
    for col, label, color in [
        ('train_cls',      'BCE',     '#E65100'),
        ('train_ec',       'EC',      '#6A1B9A'),
        ('train_pair_nce', 'NCE',     '#00838F'),
        ('train_triplet',  'Triplet', '#AD1457'),
        ('train_reg',      'Reg',     '#2E7D32'),
        ('train_tanimoto', 'Tanimoto','#FF8F00'),
    ]:
        vals  = df[col].values.astype(float)
        valid = np.isfinite(vals) & (vals > 0)
        if valid.any():
            ax.plot(ep[valid], vals[valid], '.-', ms=2, lw=1,
                    label=label, color=color)
    ax.set_ylabel('Loss'); ax.set_title('Loss Components')
    ax.set_xlabel('Epoch'); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Figure 2: Diagnostics ─────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))
    fig2.suptitle('ResidueOnlyBind — Training Diagnostics',
                  fontsize=13, fontweight='bold')

    # 2a: Gradient norms
    ax = axes2[0]
    _add_phase_bg(ax, ep, phases)
    gnr = df['train_grad_norm_res'].values.astype(float)
    gnl = df['train_grad_norm_lig'].values.astype(float)
    vr  = np.isfinite(gnr) & (gnr > 0)
    vl  = np.isfinite(gnl) & (gnl > 0)
    if vr.any():
        ax.plot(ep[vr], gnr[vr], '.-', ms=3, lw=1, color='#E65100', label='Residue GNN')
    if vl.any():
        ax.plot(ep[vl], gnl[vl], '.-', ms=3, lw=1, color='#1565C0', label='Ligand GNN')
    ax.set_ylabel('Grad Norm'); ax.set_title('Gradient Norms')
    ax.set_xlabel('Epoch'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    if vr.any() or vl.any():
        ax.set_yscale('log')

    # 2b: Train vs Val loss scatter
    ax = axes2[1]
    for p in sorted(df['phase'].unique()):
        mask = df['phase'] == p
        ax.scatter(df.loc[mask, 'train_loss'], df.loc[mask, 'val_loss'],
                   c=PHASE_COLORS.get(int(p), 'gray'),
                   label=PHASE_LABELS.get(int(p), f'P{int(p)}'),
                   s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
    lo = min(df['train_loss'].min(), df['val_loss'].min()) * 0.9
    hi = max(df['train_loss'].max(), df['val_loss'].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Train Loss'); ax.set_ylabel('Val Loss')
    ax.set_title('Overfit Check (train vs val loss)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 2c: Dominance ratio (prot/lig grad norms)
    ax = axes2[2]
    _add_phase_bg(ax, ep, phases)
    if vr.any() and vl.any():
        dom = gnr / (gnl + 1e-8)
        vdom = np.isfinite(dom) & (dom > 0)
        if vdom.any():
            ax.plot(ep[vdom], dom[vdom], 'o-', ms=3, lw=1.2, color='#BF360C')
            ax.set_yscale('log')
    ax.set_ylabel('Ratio (log)'); ax.set_title('Prot/Lig Grad Dominance')
    ax.set_xlabel('Epoch'); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig2.savefig(os.path.join(plot_dir, 'training_diagnostics.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # ── Figure 3: Advanced analytics ─────────────────────────────────────────
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 15))
    fig3.suptitle('ResidueOnlyBind — Advanced Analytics',
                  fontsize=14, fontweight='bold')

    # 3a: Per-phase loss distribution
    ax = axes3[0, 0]
    phase_losses  = [df.loc[df['phase'] == p, 'train_loss'].dropna().values
                     for p in sorted(df['phase'].unique())]
    phase_lbls    = [PHASE_LABELS.get(int(p), f'P{int(p)}')
                     for p in sorted(df['phase'].unique())]
    bp = ax.boxplot(phase_losses, labels=phase_lbls, patch_artist=True)
    for i, p in enumerate(sorted(df['phase'].unique())):
        bp['boxes'][i].set_facecolor(PHASE_COLORS.get(int(p), 'gray'))
        bp['boxes'][i].set_alpha(0.5)
    ax.set_ylabel('Train Loss'); ax.set_title('Loss Distribution per Phase')
    ax.grid(alpha=0.3, axis='y')

    # 3b: ROC-AUC vs Disc Accuracy
    ax = axes3[0, 1]
    roc_vals = df['val_roc_auc'].values.astype(float)
    da_vals  = df['val_disc_acc'].values.astype(float)
    valid_b  = np.isfinite(roc_vals) & np.isfinite(da_vals) & (roc_vals > 0)
    if valid_b.any():
        sc = ax.scatter(roc_vals[valid_b], da_vals[valid_b],
                        c=ep[valid_b], cmap='viridis', s=35,
                        edgecolors='k', linewidths=0.3)
        plt.colorbar(sc, ax=ax, label='Epoch')
        z     = np.polyfit(roc_vals[valid_b], da_vals[valid_b], 1)
        x_fit = np.linspace(roc_vals[valid_b].min(), roc_vals[valid_b].max(), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), 'r--', alpha=0.5, lw=1.5)
    ax.set_xlabel('ROC-AUC'); ax.set_ylabel('Disc Accuracy')
    ax.set_title('ROC-AUC vs Disc Accuracy'); ax.grid(alpha=0.3)

    # 3c: Loss delta distribution
    ax = axes3[1, 0]
    td = np.diff(df['train_loss'].values)
    vd = np.diff(df['val_loss'].values)
    ftd = td[np.isfinite(td)]
    fvd = vd[np.isfinite(vd)]
    if len(ftd) > 0:
        lo_b = min(ftd.min(), fvd.min())
        hi_b = max(ftd.max(), fvd.max())
        bins = np.linspace(lo_b, hi_b, 25)
        ax.hist(ftd, bins=bins, alpha=0.6, color='#1565C0', label='Train Δloss')
        ax.hist(fvd, bins=bins, alpha=0.6, color='#C62828', label='Val Δloss')
    ax.axvline(0, color='k', ls='--', alpha=0.4)
    ax.set_xlabel('Loss Change'); ax.set_ylabel('Count')
    ax.set_title('Loss Delta Distribution')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 3d: Contrastive losses stacked
    ax = axes3[1, 1]
    _add_phase_bg(ax, ep, phases)
    nce = np.where(np.isfinite(df['train_pair_nce'].values.astype(float)),
                   df['train_pair_nce'].values.astype(float), 0)
    tri = np.where(np.isfinite(df['train_triplet'].values.astype(float)),
                   df['train_triplet'].values.astype(float), 0)
    tan = np.where(np.isfinite(df['train_tanimoto'].values.astype(float)),
                   df['train_tanimoto'].values.astype(float), 0)
    ax.stackplot(ep, nce, tri, tan,
                 labels=['NCE', 'Triplet', 'Tanimoto'],
                 colors=['#00838F', '#AD1457', '#FF8F00'], alpha=0.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Contrastive Losses (stacked)')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.3)

    # 3e: Metric improvement rate
    ax = axes3[2, 0]
    window = max(3, len(df) // 15)
    for col, label, color in [
        ('val_roc_auc',       'ROC-AUC', '#6A1B9A'),
        ('val_avg_precision', 'Avg Prec', '#00838F'),
        ('val_disc_acc',      'Disc Acc', '#00695C'),
    ]:
        vals = df[col].values.astype(float)
        if len(vals) > window:
            rolling = pd.Series(vals).rolling(window, min_periods=1).mean().values
            improvement = np.gradient(rolling)
            valid = np.isfinite(improvement)
            if valid.any():
                ax.plot(ep[valid], improvement[valid], '.-', ms=2, lw=1,
                        label=label, color=color)
    ax.axhline(0, color='k', ls='--', alpha=0.4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Δ (rolling w={window})')
    ax.set_title('Validation Metric Improvement Rate')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # 3f: NaN skipped
    ax = axes3[2, 1]
    _add_phase_bg(ax, ep, phases)
    if 'nan_skipped' in df.columns:
        ns = df['nan_skipped'].values.astype(float)
        ax.bar(ep, ns, color='#B71C1C', alpha=0.7, width=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Batches skipped')
    ax.set_title('NaN Skipped Batches'); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fig3.savefig(os.path.join(plot_dir, 'training_analytics.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)


def plot_predictions(probs, labels, affinity_pred, affinity_true, plot_dir, epoch):
    """
    Detailed prediction analysis plots.
    Called at end of training and periodically.

      probs          : [N] predicted binding probability (sigmoid)
      labels         : [N] binary ground truth (1=binder, 0=non-binder)
      affinity_pred  : [M] predicted log10(affinity) for true binders
      affinity_true  : [M] ground-truth log10(affinity) for true binders
    """
    os.makedirs(plot_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle(f'ResidueOnlyBind — Prediction Analysis (epoch {epoch})',
                 fontsize=14, fontweight='bold')

    # ── Panel A: Score distribution ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    bins = np.linspace(0, 1, 31)
    ax.hist(neg_probs, bins=bins, alpha=0.6, color='#C62828', label=f'Non-binder (n={len(neg_probs)})')
    ax.hist(pos_probs, bins=bins, alpha=0.6, color='#1565C0', label=f'Binder (n={len(pos_probs)})')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Count')
    ax.set_title('Binding Score Distribution')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel B: ROC curve ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val     = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, lw=2, color='#6A1B9A',
                label=f'ROC (AUC = {auc_val:.3f})')
    except ValueError:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.fill_between(fpr if 'fpr' in dir() else [], tpr if 'tpr' in dir() else [],
                    alpha=0.1, color='#6A1B9A')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    # ── Panel C: Precision-Recall curve ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    try:
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap_val = average_precision_score(labels, probs)
        ax.plot(recall, precision, lw=2, color='#00838F',
                label=f'PR (AP = {ap_val:.3f})')
        baseline = labels.mean()
        ax.axhline(baseline, color='gray', ls='--', alpha=0.5,
                   label=f'Baseline ({baseline:.2f})')
    except ValueError:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    # ── Panel D: Affinity scatter ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if len(affinity_true) > 1:
        ax.scatter(affinity_true, affinity_pred,
                   alpha=0.5, s=20, color='#1565C0', edgecolors='none')
        lo = min(affinity_true.min(), affinity_pred.min())
        hi = max(affinity_true.max(), affinity_pred.max())
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5, lw=1.5, label='y=x')
        # Pearson r
        r = np.corrcoef(affinity_true, affinity_pred)[0, 1]
        ax.set_title(f'Affinity Regression (r={r:.3f})')
        ax.legend(fontsize=8)
    else:
        ax.set_title('Affinity Regression (no positive data)')
    ax.set_xlabel('True log10(affinity)')
    ax.set_ylabel('Predicted (z-normalized)')
    ax.grid(alpha=0.3)

    # ── Panel E: Calibration (reliability diagram) ───────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    n_bins = 10
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bin_centers, frac_pos, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            frac_pos.append(labels[mask].mean())
            bin_counts.append(mask.sum())
    if bin_centers:
        bc = np.array(bin_centers)
        fp = np.array(frac_pos)
        bct = np.array(bin_counts)
        sc = ax.scatter(bc, fp, s=bct / max(bct) * 200 + 20,
                        c=bct, cmap='YlOrRd', edgecolors='k',
                        linewidths=0.5, zorder=3)
        plt.colorbar(sc, ax=ax, label='Bin count')
        ax.plot(bc, fp, 'o-', color='#1565C0', lw=1.5, ms=0, alpha=0.6)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration (Reliability Diagram)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel F: Score CDF ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    for arr, label, color in [
        (pos_probs, 'Binders',     '#1565C0'),
        (neg_probs, 'Non-binders', '#C62828'),
    ]:
        if len(arr) > 0:
            sorted_arr = np.sort(arr)
            cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
            ax.plot(sorted_arr, cdf, lw=2, color=color, label=label)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Score CDF by Class')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    fig.savefig(os.path.join(plot_dir, 'predictions_final.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = HierAtomBindDataset(
        root=DATASET_ROOT, csv_path=CSV_PATH,
        smiles_col=SMILES_COL, protein_col=PROTEIN_COL,
        protein_dir=PROTEIN_DIR, target_cols=TARGET_COLS,
        ec_col=EC_COL, tanimoto_col=TANIMOTO_COL, resume=True,
    )

    if len(dataset) == 0:
        raise RuntimeError('Dataset is empty — check CSV and protein directory.')
    log.info(f'Dataset size: {len(dataset)}')

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

    train_ds = HierAtomBindAugmentedDataset(
        dataset, train_idx, protein_ids_arr,
        hard_neg_per_pos=HARD_NEG_PER_POS, seed=RANDOM_SEED,
    )
    val_ds = HierAtomBindAugmentedDataset(
        dataset, val_idx, protein_ids_arr,
        hard_neg_per_pos=HARD_NEG_PER_POS, seed=RANDOM_SEED + 1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    log.info(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ResidueOnlyBind().to(device)
    optimizer = build_optimizer(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'ResidueOnlyBind parameters: {n_params:,}')

    # ── Training loop ─────────────────────────────────────────────────────────
    init_metrics_csv(METRICS_CSV_PATH)

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

        log_metrics_csv(METRICS_CSV_PATH, epoch, phase, train_m, val_m, n_nan)
        last_val_metrics = val_m

        # Periodic plots
        if epoch % PLOT_EVERY == 0 or epoch == 1:
            plot_training_curves(METRICS_CSV_PATH, PLOT_DIR)

        # Checkpoint every epoch
        torch.save({
            'epoch':       epoch,
            'phase':       phase,
            'model_state': model.state_dict(),
            'opt_state':   optimizer.state_dict(),
            'val_metrics': {k: v for k, v in val_m.items()
                            if not k.startswith('_')},
            'y_mean':      y_mean,
            'y_std':       y_std,
        }, os.path.join(_HERE, 'checkpoints', f'epoch_{epoch:03d}.pt'))

        # Best model
        val_roc_auc = val_m.get('roc_auc', 0.0)
        if np.isfinite(val_roc_auc) and val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
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
    plot_training_curves(METRICS_CSV_PATH, PLOT_DIR)

    probs  = last_val_metrics.get('_probs',         np.array([]))
    labels = last_val_metrics.get('_labels',        np.array([]))
    a_pred = last_val_metrics.get('_affinity_pred', np.array([]))
    a_true = last_val_metrics.get('_affinity_true', np.array([]))

    if len(probs) > 0:
        final_epoch = epoch
        plot_predictions(probs, labels, a_pred, a_true, PLOT_DIR, final_epoch)
        log.info(f'Prediction plot saved to {PLOT_DIR}/predictions_final.png')

    log.info(f'All plots saved to {PLOT_DIR}/')
    log.info(f'Best val ROC-AUC: {best_val_roc_auc:.4f}')
    log.info('Training complete.')


if __name__ == '__main__':
    main()
