"""
scatter.py — Regression scatter: predicted log10(kcat/km) vs true log10(kcat/km).

Loads best_model.pt (state dict) and epoch_100.pt (for y_mean / y_std),
runs the affinity_head on all processed samples with y > 0,
denormalises predictions, and saves a publication-quality scatter plot to
  plots/affinity_scatter.png
"""

import os
import sys
import glob
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from torch_geometric.data import Batch

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, _HERE)

from ResidueOnlyBind import ResidueOnlyBind

# ── Paths ─────────────────────────────────────────────────────────────────────
BEST_MODEL   = os.path.join(_HERE, 'checkpoints_v2', 'best_model.pt')
EPOCH_CKPT   = os.path.join(_HERE, 'checkpoints_v2', 'epoch_088.pt')  # for y_mean/y_std
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data', 'processed_hieratom')
OUT_PATH     = os.path.join(_HERE, 'plots_v2', 'affinity_scatter.png')
DEVICE       = torch.device('cpu')   # GPU OOM on large protein graphs; CPU sufficient for inference
BATCH_SIZE   = 8

os.makedirs(os.path.join(_HERE, 'plots_v2'), exist_ok=True)
print('Device:', DEVICE)

# ── Load normalisation stats ──────────────────────────────────────────────────
epoch_ckpt = torch.load(EPOCH_CKPT, map_location='cpu')
# v2 checkpoints: keys are epoch/phase/model_state/opt_state/val_metrics/y_mean/y_std
y_mean = float(epoch_ckpt['y_mean'])
y_std  = float(epoch_ckpt['y_std'])
print(f'y_mean={y_mean:.3f}, y_std={y_std:.3f}')

# ── Load model ────────────────────────────────────────────────────────────────
model = ResidueOnlyBind().to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.eval()
print('Model loaded')

# ── Load all .pt files ────────────────────────────────────────────────────────
pt_files = sorted(
    glob.glob(os.path.join(DATASET_ROOT, 'data_*.pt')),
    key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
)
print(f'Found {len(pt_files)} files')

# Collect valid samples (mol, prot, y) where y > 0 and data is finite
samples = []
for path in pt_files:
    try:
        data = torch.load(path, weights_only=False)
        mol, prot, y = data[0], data[1], data[2]
        y_val = y.item() if y.numel() == 1 else float(y[0])
        if y_val <= 0:
            continue
        if mol.x is None or prot.x is None:
            continue
        if not torch.isfinite(mol.x).all() or not torch.isfinite(prot.x).all():
            continue
        samples.append((mol, prot, y_val))
    except Exception as e:
        pass

print(f'Valid positives: {len(samples)}')

# ── Inference in batches ──────────────────────────────────────────────────────
aff_pred_log10 = []
aff_true_log10 = []

with torch.no_grad():
    for i in range(0, len(samples), BATCH_SIZE):
        batch_items = samples[i:i+BATCH_SIZE]
        mols  = [s[0] for s in batch_items]
        prots = [s[1] for s in batch_items]
        ys    = [s[2] for s in batch_items]

        lig_b  = Batch.from_data_list(mols).to(DEVICE)
        prot_b = Batch.from_data_list(prots).to(DEVICE)

        outputs = model(lig_b, prot_b)
        affinity = outputs[1].cpu().numpy()   # z-normalised prediction, shape [B]

        for bi, y_val in enumerate(ys):
            pred = float(affinity[bi]) * y_std + y_mean   # → log10(kcat/km)
            true = math.log10(max(y_val, 1e-10))
            aff_pred_log10.append(pred)
            aff_true_log10.append(true)

aff_pred = np.array(aff_pred_log10)
aff_true = np.array(aff_true_log10)

# ── Statistics ────────────────────────────────────────────────────────────────
r, p_val   = stats.pearsonr(aff_true, aff_pred)
rho, _     = stats.spearmanr(aff_true, aff_pred)
rmse       = float(np.sqrt(np.mean((aff_pred - aff_true) ** 2)))
mae        = float(np.mean(np.abs(aff_pred - aff_true)))

print(f'\nPearson  r  = {r:.3f}  (p={p_val:.2e})')
print(f'Spearman rho= {rho:.3f}')
print(f'RMSE        = {rmse:.3f} log10 units')
print(f'MAE         = {mae:.3f} log10 units')
print(f'N           = {len(aff_true)}')

# ── Plot ──────────────────────────────────────────────────────────────────────
lo = min(aff_true.min(), aff_pred.min()) - 0.3
hi = max(aff_true.max(), aff_pred.max()) + 0.3

fig, ax = plt.subplots(figsize=(6, 5.5))

# Density-coloured scatter
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
xy  = np.vstack([aff_true, aff_pred])
kde = gaussian_kde(xy)
z   = kde(xy)
idx = z.argsort()
sc  = ax.scatter(aff_true[idx], aff_pred[idx],
                 c=z[idx], cmap='viridis', s=18, alpha=0.75, linewidths=0)
plt.colorbar(sc, ax=ax, label='Density')

# Identity line
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2, label='y = x')

# Linear regression line
slope, intercept, *_ = stats.linregress(aff_true, aff_pred)
xs = np.array([lo, hi])
ax.plot(xs, slope * xs + intercept, 'k-', lw=1.0, alpha=0.6, label='OLS fit')

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect('equal')
ax.set_xlabel('True log₁₀(kcat/Km)  [s⁻¹M⁻¹]', fontsize=12)
ax.set_ylabel('Predicted log₁₀(kcat/Km)  [s⁻¹M⁻¹]', fontsize=12)
ax.set_title('ResidueOnlyBind — Affinity Regression', fontsize=13, fontweight='bold')

stats_txt = (
    f'Pearson r = {r:.3f}\n'
    f'Spearman ρ = {rho:.3f}\n'
    f'RMSE = {rmse:.2f} log₁₀\n'
    f'MAE  = {mae:.2f} log₁₀\n'
    f'N = {len(aff_true)}'
)
ax.text(0.04, 0.97, stats_txt, transform=ax.transAxes,
        va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

ax.legend(fontsize=9, loc='lower right')
ax.grid(True, lw=0.4, alpha=0.5)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f'\nSaved: {OUT_PATH}')
