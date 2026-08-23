"""
evaluation/null_baselines.py — Data-driven null baselines for attractor diagnosis.

Builds three null score matrices over the SAME 200×200 geometry used by the
trained models, so Gini values are directly comparable:

  1. random      — uniform noise, no structure
  2. prot_prior  — score[i, j] = per-protein positive rate in training set
                   (i.e. model that ignores ligand entirely, only knows protein)
  3. lig_prior   — score[i, j] = per-ligand positive rate in training set

If `prot_prior` yields Gini ≈ the model Gini AND picks the same top attractors,
the high model-Gini is a *data geometry* artifact, not a model pathology.

Usage:
  python evaluation/null_baselines.py --out_dir evaluation/attractor_results
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baselines', 'adapters'))

from common import BRENDADataConfig
from attractor_diagnosis import (
    compute_attractor_metrics,
    plot_response_map,
    plot_attractor_distribution,
)


def build_null_matrices(config: BRENDADataConfig, n_matrix: int = 200, seed: int = 42,
                        split_mode: str = 'protein'):
    """Build the three null score matrices over the same geometry as trained models.

    split_mode selects which fold counts as "training" for the priors:
      'protein'     canonical protein-stratified split (default, backward compatible)
      'ligand'      ligand-disjoint cold-ligand split (skill §5)
      'double_cold' product partition, neither axis recurs (skill §5/§7)
    """
    pairs = config.load_pairs()
    seqs = config.load_sequences()
    if split_mode == 'ligand':
        train_idx, _, _ = config.get_ligand_split()
    elif split_mode == 'double_cold':
        train_idx, _, _ = config.get_double_cold_split()
    else:
        train_idx, _, _ = config.get_protein_split()

    # Same sampling as train_original.py:
    proteins = list(seqs.keys())[:n_matrix]
    smiles_list = pairs['substrate_smiles'].unique()[:n_matrix]
    n_prot = len(proteins)
    n_lig = len(smiles_list)

    # Training-set label distribution, indexed by protein and by ligand
    train_pairs = pairs[pairs['idx'].isin(train_idx)]
    prot_pos_rate = train_pairs.groupby('uniprot')['label'].mean()
    lig_pos_rate = train_pairs.groupby('substrate_smiles')['label'].mean()

    # Fallback: global training positive rate
    global_rate = float(train_pairs['label'].mean())

    prot_vec = np.array([prot_pos_rate.get(p, global_rate) for p in proteins],
                        dtype=np.float32)
    lig_vec = np.array([lig_pos_rate.get(s, global_rate) for s in smiles_list],
                       dtype=np.float32)

    rng = np.random.default_rng(seed)
    matrices = {
        'null_random':     rng.uniform(0, 1, size=(n_lig, n_prot)).astype(np.float32),
        'null_prot_prior': np.broadcast_to(prot_vec, (n_lig, n_prot)).copy(),
        'null_lig_prior':  np.broadcast_to(lig_vec[:, None], (n_lig, n_prot)).copy(),
    }
    return matrices, proteins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_matrix', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default=os.path.join(_HERE, 'attractor_results'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = BRENDADataConfig(seed=args.seed)
    matrices, proteins = build_null_matrices(config, args.n_matrix, args.seed)

    rows = []
    top_attractors = {}
    for name, M in matrices.items():
        np.save(os.path.join(args.out_dir, f'score_matrix_{name}.npy'), M)
        metrics = compute_attractor_metrics(M)
        rows.append({
            'model':              name,
            'gini_attractor':     metrics['gini_attractor'],
            'mean_score_var':     metrics['mean_score_variance'],
            'N_prot':             metrics['N_prot'],
            'N_lig':              metrics['N_lig'],
        })
        top5 = np.argsort(-metrics['attractor_scores'])[:5]
        top_attractors[name] = [(proteins[i], float(metrics['attractor_scores'][i]))
                                for i in top5]
        plot_response_map(M, name, out_dir=args.out_dir)
        plot_attractor_distribution(metrics, name, out_dir=args.out_dir)

    # Also evaluate model matrices with the same top-attractor comparison
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    model_matrices = {
        'graphdta': os.path.join(results_dir, 'original_graphdta', 'score_matrix_graphdta.npy'),
        'moltrans': os.path.join(results_dir, 'original_moltrans', 'score_matrix_moltrans.npy'),
        'drugban':  os.path.join(results_dir, 'original_drugban', 'score_matrix_DrugBAN.npy'),
        'gems':     os.path.join(results_dir, 'original_gems', 'score_matrix_gems.npy'),
    }
    for name, path in model_matrices.items():
        if not os.path.exists(path):
            continue
        M = np.load(path)
        metrics = compute_attractor_metrics(M)
        rows.append({
            'model':              name,
            'gini_attractor':     metrics['gini_attractor'],
            'mean_score_var':     metrics['mean_score_variance'],
            'N_prot':             metrics['N_prot'],
            'N_lig':              metrics['N_lig'],
        })
        top5 = np.argsort(-metrics['attractor_scores'])[:5]
        top_attractors[name] = [(proteins[i], float(metrics['attractor_scores'][i]))
                                for i in top5]

    df = pd.DataFrame(rows).sort_values('gini_attractor', ascending=False)
    csv_path = os.path.join(args.out_dir, 'gini_comparison.csv')
    df.to_csv(csv_path, index=False)

    print("\n=== Gini comparison (null baselines vs models) ===")
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}")

    # Top-attractor overlap: how many of model's top-5 attractors also appear in
    # the prot_prior top-5? If overlap is high, the model learned the prior.
    print("\n=== Top-5 attractor proteins (shared = learned the prior) ===")
    prior_top = set(p for p, _ in top_attractors.get('null_prot_prior', []))
    for name, top in top_attractors.items():
        prots = [p for p, _ in top]
        shared = sum(1 for p in prots if p in prior_top)
        print(f"  {name:20s}: {prots}  (overlap w/ prot_prior: {shared}/5)")


if __name__ == '__main__':
    main()
