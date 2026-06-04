"""
attractor_diagnosis.py — Phase 1.2: Standardised attractor bias evaluation.

Given a pre-computed score matrix (N_ligands × N_proteins), this module
computes all attractor metrics defined in development_plan_rankbind.md §1.2:

  1. Response map heatmap
  2. Attractor score per protein
      attractor(p) = fraction of ligands for which p is top-1 predicted partner
  3. Gini coefficient of attractor scores
  4. Rank displacement per ligand
      rank(true_protein) - rank(true_protein | excluding attractors)
  5. Score variance per ligand

Can also be called as a script to evaluate a saved ResidueOnlyBind checkpoint
and generate all figures.

HPC usage (see run_attractor_diagnosis.sh):
  python evaluation/attractor_diagnosis.py \
      --model_path _archive/v4_residue_only/checkpoints/best_model.pt \
      --model_name ResidueOnlyBind_v4 \
      --n_ligands 200 --n_proteins 200

  # For a pre-computed matrix (other baselines):
  python evaluation/attractor_diagnosis.py \
      --score_matrix evaluation/attractor_results/score_matrix_DrugBAN.npy \
      --labels_csv   evaluation/attractor_results/true_prot_idx_DrugBAN.csv \
      --model_name   DrugBAN

HierAtomBind path:
  ResidueOnlyBind imports from HierAtomBind.py. On the HPC, adjust
  HIER_ROOT below to point to the directory containing HierAtomBind.py.
  Default assumes it is one level up: ../newclaudemodel/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
# HierAtomBind lives one level above rankbind/ on HPC — adjust if needed
HIER_ROOT    = os.environ.get(
    'HIERATOMBIND_ROOT',
    os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'newclaudemodel'))
)
OUT_DIR      = os.path.join(_HERE, 'attractor_results')
os.makedirs(OUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Core metric functions (model-agnostic)
# ═════════════════════════════════════════════════════════════════════════════

def gini(x: np.ndarray) -> float:
    """Gini coefficient of array x (0 = uniform, 1 = maximally concentrated)."""
    x = np.sort(np.abs(x.flatten()))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * x.sum())) - (n + 1) / n)


def compute_attractor_scores(score_matrix: np.ndarray) -> np.ndarray:
    """
    Compute attractor score for each protein.

    Parameters
    ----------
    score_matrix : [N_lig, N_prot]  binding scores (higher = more likely to bind)

    Returns
    -------
    attractor : [N_prot]  fraction of ligands for which this protein is top-1
    """
    top1   = np.argmax(score_matrix, axis=1)   # [N_lig]
    N_prot = score_matrix.shape[1]
    counts = np.bincount(top1, minlength=N_prot).astype(float)
    return counts / len(top1)


def compute_rank_displacement(
    score_matrix: np.ndarray,
    true_protein_idx: np.ndarray,
    top_k_attractors: int = 5,
) -> np.ndarray:
    """
    For each ligand, compute rank(true_protein) - rank(true_protein | excluding top-k attractors).

    Positive displacement means attractor proteins push the true partner down.

    Parameters
    ----------
    score_matrix       : [N_lig, N_prot]
    true_protein_idx   : [N_lig]  index of the true binding protein for each ligand
    top_k_attractors   : how many top-attractor proteins to mask out

    Returns
    -------
    displacements : [N_lig]  (NaN for ligands whose true protein is itself an attractor)
    """
    attractor_scores = compute_attractor_scores(score_matrix)
    attractor_mask   = np.argsort(attractor_scores)[::-1][:top_k_attractors]

    N_lig, N_prot = score_matrix.shape
    displacements  = np.zeros(N_lig)

    for i in range(N_lig):
        scores   = score_matrix[i]
        true_idx = int(true_protein_idx[i])

        # Rank in full matrix (0 = best)
        full_rank = int(np.sum(scores > scores[true_idx]))

        # Mask out top attractors
        mask = np.ones(N_prot, dtype=bool)
        mask[attractor_mask] = False
        if not mask[true_idx]:
            displacements[i] = np.nan
            continue
        masked_rank = int(np.sum(scores[mask] > scores[true_idx]))
        displacements[i] = full_rank - masked_rank  # positive = attractors hurt ranking

    return displacements


def compute_attractor_metrics(
    score_matrix: np.ndarray,
    true_protein_idx: Optional[np.ndarray] = None,
    top_k_attractors: int = 5,
) -> dict:
    """
    Compute all attractor metrics for a given score matrix.

    Returns dict with:
        attractor_scores, gini_attractor, score_variance_per_lig,
        mean_score_variance, (optionally) hit_at_1/5/10,
        mean_rank_true, rank_displacement, mean_rank_displacement
    """
    N_lig, N_prot    = score_matrix.shape
    attractor_scores = compute_attractor_scores(score_matrix)
    gini_attr        = gini(attractor_scores)
    score_var        = score_matrix.var(axis=1)

    metrics = {
        'N_lig':                  N_lig,
        'N_prot':                 N_prot,
        'attractor_scores':       attractor_scores,
        'gini_attractor':         gini_attr,
        'score_variance_per_lig': score_var,
        'mean_score_variance':    float(score_var.mean()),
    }

    if true_protein_idx is not None:
        sorted_idx = np.argsort(-score_matrix, axis=1)
        for k in [1, 5, 10]:
            hits = sum(
                int(true_protein_idx[i]) in sorted_idx[i, :k]
                for i in range(N_lig)
            )
            metrics[f'hit_at_{k}'] = hits / N_lig

        ranks = np.array([
            int(np.sum(score_matrix[i] > score_matrix[i, int(true_protein_idx[i])]))
            for i in range(N_lig)
        ])
        metrics['mean_rank_true']   = float(ranks.mean())
        metrics['median_rank_true'] = float(np.median(ranks))

        displacements = compute_rank_displacement(
            score_matrix, true_protein_idx, top_k_attractors
        )
        metrics['rank_displacement']      = displacements
        valid = displacements[~np.isnan(displacements)]
        metrics['mean_rank_displacement'] = float(valid.mean()) if len(valid) > 0 else np.nan

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═════════════════════════════════════════════════════════════════════════════

def plot_response_map(
    score_matrix: np.ndarray,
    model_name: str,
    true_protein_idx: Optional[np.ndarray] = None,
    out_dir: str = OUT_DIR,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(score_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Binding score')
    if true_protein_idx is not None:
        ax.scatter(true_protein_idx, np.arange(len(true_protein_idx)),
                   c='red', s=3, alpha=0.6, label='True partner', zorder=5)
        ax.legend(loc='upper right', markerscale=3)
    ax.set_xlabel("Protein index")
    ax.set_ylabel("Ligand index")
    ax.set_title(f"Response Map — {model_name}\n"
                 f"({score_matrix.shape[0]} ligands × {score_matrix.shape[1]} proteins)")
    fname = os.path.join(out_dir, f'response_map_{model_name}.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_attractor_distribution(
    metrics: dict,
    model_name: str,
    out_dir: str = OUT_DIR,
) -> str:
    attractor_scores = metrics['attractor_scores']
    N_prot           = len(attractor_scores)
    uniform_line     = 1.0 / N_prot

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sorted_scores = np.sort(attractor_scores)[::-1]
    axes[0].bar(np.arange(N_prot), sorted_scores, width=1.0,
                color='steelblue', edgecolor='none')
    axes[0].axhline(uniform_line, color='red', linestyle='--',
                    label=f'Uniform (1/{N_prot})')
    axes[0].set_xlabel("Protein rank (by attractor score)")
    axes[0].set_ylabel("Fraction of ligands where protein is top-1")
    axes[0].set_title(f"Attractor Scores — {model_name}\nGini = {metrics['gini_attractor']:.4f}")
    axes[0].legend()

    axes[1].hist(metrics['score_variance_per_lig'], bins=40, edgecolor='k', linewidth=0.3)
    axes[1].set_xlabel("Score variance per ligand (across all proteins)")
    axes[1].set_ylabel("# ligands")
    axes[1].set_title(f"Per-ligand score variance\nmean = {metrics['mean_score_variance']:.4f}")

    plt.tight_layout()
    fname = os.path.join(out_dir, f'attractor_dist_{model_name}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_comparison_table(all_metrics: dict, out_dir: str = OUT_DIR) -> pd.DataFrame:
    """Print and save comparison table for all evaluated models."""
    rows = []
    for name, m in all_metrics.items():
        row = {
            'Model':            name,
            'Gini(attractor)':  f"{m.get('gini_attractor', float('nan')):.4f}",
            'Mean score var':   f"{m.get('mean_score_variance', float('nan')):.4f}",
        }
        for k in [1, 5, 10]:
            key = f'hit_at_{k}'
            row[f'Hit@{k}'] = f"{m[key]:.3f}" if key in m else '-'
        if 'mean_rank_displacement' in m:
            row['Mean rank disp.'] = f"{m['mean_rank_displacement']:.2f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("ATTRACTOR BIAS COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))

    csv_path = os.path.join(out_dir, 'attractor_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# ResidueOnlyBind score-matrix builder
# ═════════════════════════════════════════════════════════════════════════════

def build_score_matrix_residueonlybind(
    model_path: str,
    dataset_root: str,
    n_ligands: int = 200,
    n_proteins: int = 200,
    batch_size: int = 16,
    device_str: str = 'cuda',
) -> tuple:
    """
    Load ResidueOnlyBind checkpoint and compute [N_lig, N_prot] score matrix.

    Returns
    -------
    score_matrix  : [N_lig, N_prot]  binding probabilities
    true_prot_idx : [N_lig]          diagonal indices (same-index = true partner)
    """
    import glob
    import torch
    from torch_geometric.data import Batch

    sys.path.insert(0, HIER_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, '_archive', 'v4_residue_only'))
    from ResidueOnlyBind import ResidueOnlyBind

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    pt_files = sorted(
        glob.glob(os.path.join(dataset_root, 'data_*.pt')),
        key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
    )
    print(f"Found {len(pt_files)} .pt files")

    def load_item(path):
        data = torch.load(path, weights_only=False)
        return data[0], data[1], data[2]

    def is_valid(path):
        try:
            mol, prot, _ = load_item(path)
            return (mol.x is not None and prot.x is not None
                    and torch.isfinite(mol.x).all()
                    and torch.isfinite(prot.x).all())
        except Exception:
            return False

    needed = max(n_ligands, n_proteins)
    print(f"Filtering valid samples (need {needed})...")
    valid_paths = []
    for p in pt_files:
        if len(valid_paths) >= needed:
            break
        if is_valid(p):
            valid_paths.append(p)

    n_lig  = min(n_ligands,  len(valid_paths))
    n_prot = min(n_proteins, len(valid_paths))
    print(f"Using {n_lig} ligands × {n_prot} proteins")

    mols  = [load_item(p)[0] for p in valid_paths[:n_lig]]
    prots = [load_item(p)[1] for p in valid_paths[:n_prot]]

    model = ResidueOnlyBind().to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print("Model loaded.")

    score_matrix = np.zeros((n_lig, n_prot), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_lig, batch_size):
            li_end = min(i + batch_size, n_lig)
            n_l    = li_end - i
            row_scores = np.zeros((n_l, n_prot), dtype=np.float32)

            for j in range(0, n_prot, batch_size):
                pj_end = min(j + batch_size, n_prot)
                n_p    = pj_end - j

                lig_repeat  = Batch.from_data_list(
                    [mols[li] for li in range(i, li_end) for _ in range(n_p)]
                ).to(device)
                prot_repeat = Batch.from_data_list(
                    [prots[pj] for _ in range(n_l) for pj in range(j, pj_end)]
                ).to(device)

                bind_logit = model(lig_repeat, prot_repeat)[0].cpu().numpy()
                probs      = 1.0 / (1.0 + np.exp(-bind_logit))
                row_scores[:, j:pj_end] = probs.reshape(n_l, n_p)

            score_matrix[i:li_end] = row_scores
            print(f"  Ligand rows {i}–{li_end-1} done")

    true_prot_idx = np.arange(n_lig)
    return score_matrix, true_prot_idx


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Attractor bias diagnosis")
    parser.add_argument('--model_path',    type=str, default=None)
    parser.add_argument('--score_matrix',  type=str, default=None,
                        help='Pre-computed score matrix .npy [N_lig, N_prot]')
    parser.add_argument('--labels_csv',    type=str, default=None,
                        help='CSV with true_protein_idx column')
    parser.add_argument('--model_name',    type=str, default='model')
    parser.add_argument('--n_ligands',     type=int, default=200)
    parser.add_argument('--n_proteins',    type=int, default=200)
    parser.add_argument('--batch_size',    type=int, default=16)
    parser.add_argument('--dataset_root',  type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'processed_hieratom'))
    parser.add_argument('--out_dir',       type=str, default=OUT_DIR)
    parser.add_argument('--device',        type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.score_matrix is not None:
        print(f"Loading score matrix: {args.score_matrix}")
        score_matrix = np.load(args.score_matrix)
        true_prot_idx = None
        if args.labels_csv:
            true_prot_idx = pd.read_csv(args.labels_csv)['true_protein_idx'].values
    elif args.model_path is not None:
        print(f"Computing score matrix for: {args.model_name}")
        score_matrix, true_prot_idx = build_score_matrix_residueonlybind(
            model_path=args.model_path,
            dataset_root=args.dataset_root,
            n_ligands=args.n_ligands,
            n_proteins=args.n_proteins,
            batch_size=args.batch_size,
            device_str=args.device,
        )
        np.save(os.path.join(args.out_dir, f'score_matrix_{args.model_name}.npy'),
                score_matrix)
        np.save(os.path.join(args.out_dir, f'true_prot_idx_{args.model_name}.npy'),
                true_prot_idx)
    else:
        parser.error("Provide either --model_path or --score_matrix")

    print(f"\nComputing attractor metrics for {args.model_name}...")
    metrics = compute_attractor_metrics(score_matrix, true_prot_idx)

    print(f"\n{'='*50}")
    print(f"Model:                  {args.model_name}")
    print(f"Matrix shape:           {score_matrix.shape}")
    print(f"Gini(attractor):        {metrics['gini_attractor']:.4f}")
    print(f"Mean score variance:    {metrics['mean_score_variance']:.4f}")
    if 'hit_at_1' in metrics:
        print(f"Hit@1:                  {metrics['hit_at_1']:.3f}")
        print(f"Hit@5:                  {metrics['hit_at_5']:.3f}")
        print(f"Hit@10:                 {metrics['hit_at_10']:.3f}")
        print(f"Mean rank (true):       {metrics['mean_rank_true']:.1f} / {metrics['N_prot']}")
    if 'mean_rank_displacement' in metrics:
        print(f"Mean rank displacement: {metrics['mean_rank_displacement']:.2f}")

    f1 = plot_response_map(score_matrix, args.model_name, true_prot_idx, args.out_dir)
    f2 = plot_attractor_distribution(metrics, args.model_name, args.out_dir)
    print(f"\nSaved: {f1}")
    print(f"Saved: {f2}")


if __name__ == '__main__':
    main()
