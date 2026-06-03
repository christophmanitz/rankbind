"""
evaluation/phase_d_figures.py — Final figures for Phase 1 diagnosis.

Produces three publication-ready figures in evaluation/attractor_results/:
  - fig_response_maps.png    2×3 grid: 4 trained models + 2 null baselines
  - fig_cross_overlap.png    Jaccard heatmap over top-10 attractor proteins
  - fig_auc_scatter.png      Global-AUC vs Per-Ligand-AUC (+ Gini bubbles)
  - fig_summary.png          One-glance dashboard combining all three

These are the inputs to the HTML report (phase_d_report.html).
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
OUT_DIR = os.path.join(_HERE, 'attractor_results')
sys.path.insert(0, _HERE)

from attractor_diagnosis import compute_attractor_scores, gini

MATRIX_FILES = {
    'GraphDTA':        os.path.join(PROJECT_ROOT, 'results', 'original_graphdta', 'score_matrix_graphdta.npy'),
    'MolTrans':        os.path.join(PROJECT_ROOT, 'results', 'original_moltrans', 'score_matrix_moltrans.npy'),
    'DrugBAN':         os.path.join(PROJECT_ROOT, 'results', 'original_drugban',  'score_matrix_DrugBAN.npy'),
    'GEMS':            os.path.join(PROJECT_ROOT, 'results', 'original_gems',     'score_matrix_gems.npy'),
    'RankBind':        os.path.join(PROJECT_ROOT, 'results', 'v5_rankbind', '20260423-112928_012a2695c2_default_v4', 'score_matrix_rankbind.npy'),
    'Null: prot_prior': os.path.join(OUT_DIR, 'score_matrix_null_prot_prior.npy'),
    'Null: random':    os.path.join(OUT_DIR, 'score_matrix_null_random.npy'),
}


def load_matrices(ref_shape=None):
    out = {}
    for name, path in MATRIX_FILES.items():
        if not os.path.exists(path):
            continue
        M = np.load(path)
        if ref_shape and M.shape != ref_shape and M.T.shape == ref_shape:
            M = M.T
        out[name] = M
    return out


def fig_response_maps(matrices):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    axes = axes.flatten()
    order = ['GraphDTA', 'MolTrans', 'DrugBAN', 'GEMS',
             'RankBind', 'Null: prot_prior', 'Null: random']
    for ax, name in zip(axes, order):
        if name not in matrices:
            ax.set_visible(False); continue
        M = matrices[name]
        attr = compute_attractor_scores(M)
        g = gini(attr)
        # Per-panel robust colour scale: 1st-99th percentile clipping so a few
        # extreme outliers (notably MolTrans, range ~ -1000..200) don't flatten
        # the rest of the matrix to one colour.
        lo, hi = np.percentile(M, [1.0, 99.0])
        if lo == hi:
            lo, hi = float(M.min()), float(M.max())
        im = ax.imshow(M, aspect='auto', cmap='viridis',
                       interpolation='nearest', vmin=lo, vmax=hi)
        ax.set_title(f"{name}\nGini = {g:.3f}", fontsize=11)
        ax.set_xlabel("Protein idx", fontsize=9)
        ax.set_ylabel("Ligand idx", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    if len(order) < len(axes):
        for ax in axes[len(order):]:
            ax.set_visible(False)
    fig.suptitle("Score Response Maps (200×200) — trained models vs null baselines",
                 fontsize=13, y=1.02)
    path = os.path.join(OUT_DIR, 'fig_response_maps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def fig_cross_overlap(matrices, k=10):
    names = list(matrices.keys())
    attr = {n: compute_attractor_scores(M) for n, M in matrices.items()}
    top = {n: set(np.argsort(-a)[:k].tolist()) for n, a in attr.items()}
    J = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            u = len(top[a] | top[b]); J[i, j] = len(top[a] & top[b]) / u if u else 0.0

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(J, vmin=0, vmax=1, cmap='Purples')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            txt_color = 'white' if J[i, j] > 0.55 else 'black'
            ax.text(j, i, f"{J[i, j]:.2f}", ha='center', va='center',
                    color=txt_color, fontsize=9)
    plt.colorbar(im, ax=ax, label=f'Jaccard (top-{k} attractors)')
    ax.set_title(f"Cross-model attractor-identity overlap (top-{k})", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_cross_overlap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def load_test_summaries():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, 'test_summary_*.json'))):
        with open(p) as f:
            rows.append(json.load(f))
    return pd.DataFrame(rows)


def fig_auc_scatter(test_df, matrices):
    """Scatter of global AUC vs per-ligand AUC. Bubble size = Gini."""
    gini_map = {n: gini(compute_attractor_scores(M)) for n, M in matrices.items()}
    # Align names (lowercase in summaries, TitleCase in matrices)
    name_map = {'graphdta': 'GraphDTA', 'moltrans': 'MolTrans',
                'drugban': 'DrugBAN', 'gems': 'GEMS',
                'rankbind': 'RankBind'}
    test_df = test_df.copy()
    test_df['name'] = test_df['model'].map(name_map)
    test_df['gini'] = test_df['name'].map(gini_map)

    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = 400 + 2000 * (test_df['gini'] - test_df['gini'].min())
    colors = plt.cm.tab10(np.arange(len(test_df)))
    ax.scatter(test_df['global_auc'], test_df['per_ligand_auc'],
               s=sizes, c=colors, alpha=0.7, edgecolors='black', linewidths=1.2)
    for _, r in test_df.iterrows():
        ax.annotate(f" {r['name']}\n Gini={r['gini']:.3f}",
                    (r['global_auc'], r['per_ligand_auc']),
                    fontsize=10, va='center')

    ax.axhline(0.5, color='red', linestyle='--', alpha=0.6,
               label='Per-lig AUC = 0.5 (random within ligand)')
    ax.set_xlabel("Global AUC  (distinguishing positives from negatives, all pairs)", fontsize=11)
    ax.set_ylabel("Per-ligand AUC  (ranking proteins for a given ligand)", fontsize=11)
    ax.set_title("Shortcut AUC vs. Ligand-Conditional Ranking AUC", fontsize=12)
    ax.set_xlim(0.55, 1.0); ax.set_ylim(0.1, 0.8)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_auc_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def fig_summary(matrices, test_df):
    """Single publication-figure dashboard combining the three panels."""
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.1])

    # Top row: 3 response maps (RankBind vs strongest baseline vs prior)
    keep = ['RankBind', 'GraphDTA', 'Null: prot_prior']
    for i, name in enumerate(keep):
        if name not in matrices: continue
        ax = fig.add_subplot(gs[0, i])
        M = matrices[name]; g = gini(compute_attractor_scores(M))
        lo, hi = np.percentile(M, [1.0, 99.0])
        if lo == hi:
            lo, hi = float(M.min()), float(M.max())
        im = ax.imshow(M, aspect='auto', cmap='viridis', vmin=lo, vmax=hi)
        ax.set_title(f"{name} | Gini={g:.3f}", fontsize=10)
        ax.set_xlabel("Protein", fontsize=8); ax.set_ylabel("Ligand", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.75)

    # Bottom-left: overlap heatmap
    ax1 = fig.add_subplot(gs[1, 0])
    names = list(matrices.keys())
    attr = {n: compute_attractor_scores(M) for n, M in matrices.items()}
    top = {n: set(np.argsort(-a)[:10].tolist()) for n, a in attr.items()}
    J = np.array([[len(top[a] & top[b]) / max(len(top[a] | top[b]), 1)
                   for b in names] for a in names])
    im = ax1.imshow(J, vmin=0, vmax=1, cmap='Purples')
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax1.text(j, i, f"{J[i, j]:.2f}", ha='center', va='center',
                     color='white' if J[i, j] > 0.55 else 'black', fontsize=7)
    ax1.set_title("Top-10 attractor Jaccard", fontsize=11)

    # Bottom-middle: AUC scatter
    ax2 = fig.add_subplot(gs[1, 1])
    gini_map = {n: gini(compute_attractor_scores(M)) for n, M in matrices.items()}
    nm = {'graphdta': 'GraphDTA', 'moltrans': 'MolTrans',
          'drugban': 'DrugBAN', 'gems': 'GEMS',
          'rankbind': 'RankBind'}
    tdf = test_df.copy(); tdf['name'] = tdf['model'].map(nm)
    tdf['gini'] = tdf['name'].map(gini_map)
    colors = plt.cm.tab10(np.arange(len(tdf)))
    ax2.scatter(tdf['global_auc'], tdf['per_ligand_auc'],
                s=400 + 1500 * (tdf['gini'] - tdf['gini'].min()),
                c=colors, alpha=0.7, edgecolors='black')
    for _, r in tdf.iterrows():
        ax2.annotate(f" {r['name']}", (r['global_auc'], r['per_ligand_auc']), fontsize=9)
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6)
    ax2.set_xlabel("Global AUC", fontsize=10); ax2.set_ylabel("Per-ligand AUC", fontsize=10)
    ax2.set_title("Global vs ligand-conditional AUC", fontsize=11)
    ax2.set_xlim(0.55, 1.0); ax2.set_ylim(0.1, 0.8); ax2.grid(alpha=0.3)

    # Bottom-right: Gini bar
    ax3 = fig.add_subplot(gs[1, 2])
    all_names = list(gini_map.keys())
    ginis = [gini_map[n] for n in all_names]
    colors = ['#1f77b4' if 'Null' not in n else '#d62728' for n in all_names]
    ax3.barh(all_names, ginis, color=colors)
    ax3.set_xlim(0, 1.0)
    ax3.axvline(gini_map.get('Null: prot_prior', 0), color='red', linestyle='--',
                label='Prior baseline')
    ax3.set_xlabel("Gini(attractor)", fontsize=10)
    ax3.set_title("Gini: models vs null baselines", fontsize=11)
    ax3.legend(loc='lower right')

    fig.suptitle("Phase 1 diagnosis: DTI models pass global AUC but fail ligand-conditional ranking",
                 fontsize=13)
    path = os.path.join(OUT_DIR, 'fig_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def main():
    matrices = load_matrices()
    ref_shape = next(iter(matrices.values())).shape
    matrices = load_matrices(ref_shape=ref_shape)

    test_df = load_test_summaries()
    if test_df.empty:
        print("No test_summary_*.json files. Run test_set_eval.py first.")
        return

    p1 = fig_response_maps(matrices)
    p2 = fig_cross_overlap(matrices)
    p3 = fig_auc_scatter(test_df, matrices)
    p4 = fig_summary(matrices, test_df)
    for p in (p1, p2, p3, p4):
        print(f"Saved: {p}")


if __name__ == '__main__':
    main()
