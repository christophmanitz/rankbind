"""
evaluation/cross_model_overlap.py — Attractor identity overlap across models.

Framing-3 diagnostic: the null-baseline analysis showed all models share
Gini ≈ 0.995, so Gini alone does not distinguish models. This script tests
whether models nonetheless select DIFFERENT proteins as attractors.

Outputs:
  - cross_model_overlap.csv   (top-K Jaccard for every model pair)
  - spearman_attractor.csv    (Spearman rho of attractor_scores)
  - gini_residual.csv         (Gini model − Gini null_prot_prior)
  - cross_model_overlap.png   (heatmap)

Verdict printed:
  - If mean pairwise Jaccard is low (<0.3) → Framing 3 holds: models
    converge on DIFFERENT attractors despite identical statistics.
  - If high (>0.7)                         → Framing 3 fails: attractor
    identity is also data-forced, Framing 2 is all that remains.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _HERE)

from attractor_diagnosis import compute_attractor_scores, gini

OUT_DIR = os.path.join(_HERE, 'attractor_results')
os.makedirs(OUT_DIR, exist_ok=True)

MATRICES = {
    'graphdta':        os.path.join(PROJECT_ROOT, 'results', 'original_graphdta', 'score_matrix_graphdta.npy'),
    'moltrans':        os.path.join(PROJECT_ROOT, 'results', 'original_moltrans', 'score_matrix_moltrans.npy'),
    'drugban':         os.path.join(PROJECT_ROOT, 'results', 'original_drugban',  'score_matrix_DrugBAN.npy'),
    'gems':            os.path.join(PROJECT_ROOT, 'results', 'original_gems',     'score_matrix_gems.npy'),
    'null_prot_prior': os.path.join(OUT_DIR, 'score_matrix_null_prot_prior.npy'),
    'null_random':     os.path.join(OUT_DIR, 'score_matrix_null_random.npy'),
}

TOP_K = 10


def align_prot_axis(M: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Ensure score matrix has shape [N_lig, N_prot] matching target.

    Some trainers saved [N_prot, N_lig]; detect by matching target_shape.
    """
    if M.shape == target_shape:
        return M
    if M.T.shape == target_shape:
        return M.T
    raise ValueError(f"Shape {M.shape} incompatible with target {target_shape}")


def main():
    # Load all matrices, normalise orientation
    raw = {}
    for name, path in MATRICES.items():
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} not found")
            continue
        raw[name] = np.load(path)

    if len(raw) < 2:
        print("Need at least 2 matrices. Aborting.")
        return

    # Use the first matrix's shape as reference
    ref_shape = next(iter(raw.values())).shape
    matrices = {}
    for name, M in raw.items():
        try:
            matrices[name] = align_prot_axis(M, ref_shape)
        except ValueError as e:
            print(f"  [skip] {name}: {e}")

    names = list(matrices.keys())
    print(f"Loaded {len(names)} matrices at shape {ref_shape}: {names}")

    # Compute attractor scores and top-K protein indices for each
    attr_scores = {n: compute_attractor_scores(M) for n, M in matrices.items()}
    top_sets = {n: set(np.argsort(-s)[:TOP_K].tolist()) for n, s in attr_scores.items()}

    # Pairwise Jaccard of top-K attractor sets
    jac = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            inter = len(top_sets[a] & top_sets[b])
            union = len(top_sets[a] | top_sets[b])
            jac.loc[a, b] = inter / union if union else 0.0

    # Pairwise Spearman of attractor_scores vectors
    spear = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            rho, _ = spearmanr(attr_scores[a], attr_scores[b])
            spear.loc[a, b] = rho if np.isfinite(rho) else 0.0

    # Prior-residual Gini
    gini_vals = {n: gini(attr_scores[n]) for n in names}
    base = gini_vals.get('null_prot_prior', 0.0)
    residual = pd.DataFrame(
        [(n, gini_vals[n], gini_vals[n] - base) for n in names],
        columns=['model', 'gini', 'gini_minus_prot_prior'],
    )

    jac.to_csv(os.path.join(OUT_DIR, 'cross_model_overlap.csv'))
    spear.to_csv(os.path.join(OUT_DIR, 'spearman_attractor.csv'))
    residual.to_csv(os.path.join(OUT_DIR, 'gini_residual.csv'), index=False)

    # Heatmap (Jaccard)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(jac.values.astype(float), vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{jac.iloc[i,j]:.2f}", ha='center', va='center',
                    color='white' if jac.iloc[i,j] < 0.5 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, label=f'Top-{TOP_K} Jaccard')
    ax.set_title(f'Cross-model attractor overlap (top-{TOP_K} proteins)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cross_model_overlap.png'), dpi=150)
    plt.close()

    # Verdict
    print(f"\n=== Top-{TOP_K} Jaccard overlap ===")
    print(jac.round(3).to_string())
    print(f"\n=== Spearman rho of attractor_scores ===")
    print(spear.round(3).to_string())
    print(f"\n=== Gini residual (model − null_prot_prior) ===")
    print(residual.to_string(index=False))

    model_names = [n for n in names if not n.startswith('null_')]
    if len(model_names) >= 2:
        pairs = [(a, b) for i, a in enumerate(model_names) for b in model_names[i+1:]]
        model_jac = np.mean([jac.loc[a, b] for a, b in pairs])
        model_sp  = np.mean([spear.loc[a, b] for a, b in pairs])
        prior_jac = np.mean([jac.loc[m, 'null_prot_prior'] for m in model_names]) \
                    if 'null_prot_prior' in names else float('nan')

        print(f"\n--- Summary across {len(model_names)} trained models ---")
        print(f"  mean pairwise Jaccard:           {model_jac:.3f}")
        print(f"  mean pairwise Spearman:          {model_sp:.3f}")
        print(f"  mean Jaccard vs null_prot_prior: {prior_jac:.3f}")

        if model_jac < 0.3:
            verdict = ("FRAMING 3 HOLDS: models converge on DIFFERENT attractors "
                       "despite matching Gini statistics.")
        elif model_jac > 0.7:
            verdict = ("FRAMING 3 FAILS: models converge on the SAME attractors; "
                       "identity is also data-forced. Fall back to Framing 2.")
        else:
            verdict = ("AMBIGUOUS: partial overlap. Framing 3 viable but needs "
                       "a more careful hypothesis (e.g. graph-based vs seq-based clusters).")
        print(f"\n>>> {verdict}")


if __name__ == '__main__':
    main()
