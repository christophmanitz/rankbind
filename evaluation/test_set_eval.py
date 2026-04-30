"""
evaluation/test_set_eval.py — Test-set evaluation for original baselines.

Runs each trained model on its protein-based test split and reports:
  - Global AUC / AUPRC (binary classification over all test pairs)
  - Per-ligand AUC (mean over ligands with ≥1 positive and ≥1 negative)
  - Hit@K per ligand (fraction of ligands whose positive test protein is
    among its top-K scored proteins in the test set)

Purpose: decide between Framing 3 interpretations. If the "prior-inheriting"
models (GraphDTA/DrugBAN/GEMS) have higher Hit@K than the "prior-deviating"
MolTrans, then the prior is an effective shortcut — not pathology.

Usage:
  # Main venv (hieratombind): graphdta, moltrans, gems
  python evaluation/test_set_eval.py --model graphdta
  python evaluation/test_set_eval.py --model moltrans
  python evaluation/test_set_eval.py --model gems

  # DrugBAN venv (torch 2.4 + DGL):
  python evaluation/test_set_eval.py --model drugban
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'baselines', 'adapters'))

from common import BRENDADataConfig


def hit_at_k(pair_df: pd.DataFrame, ks=(1, 5, 10)) -> dict:
    """
    For each ligand with ≥1 positive test pair, rank all its test proteins
    by predicted score and check whether the positive protein is in top-K.

    pair_df: columns ['smiles', 'uniprot', 'score', 'label'].
    """
    out = {f'hit_at_{k}': np.nan for k in ks}
    out['n_ligands_evaluated'] = 0

    rows = []
    for smiles, grp in pair_df.groupby('smiles'):
        if grp['label'].sum() < 1:
            continue
        if len(grp) < 2:
            continue
        grp_sorted = grp.sort_values('score', ascending=False).reset_index(drop=True)
        pos_ranks = grp_sorted.index[grp_sorted['label'] == 1].tolist()
        if not pos_ranks:
            continue
        rows.append((smiles, pos_ranks[0], len(grp)))

    if not rows:
        return out
    out['n_ligands_evaluated'] = len(rows)
    for k in ks:
        out[f'hit_at_{k}'] = float(np.mean([1.0 if r < k else 0.0 for _, r, _ in rows]))
    return out


def per_ligand_auc(pair_df: pd.DataFrame) -> float:
    aucs = []
    for smiles, grp in pair_df.groupby('smiles'):
        if grp['label'].nunique() < 2:
            continue
        try:
            aucs.append(roc_auc_score(grp['label'].values, grp['score'].values))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float('nan')


def run_graphdta(config, test_idx, device, ckpt_path, batch_size=32):
    from adapter_graphdta import GraphDTADataset, get_model
    from torch_geometric.loader import DataLoader as PyGLoader

    test_ds = GraphDTADataset(config, test_idx)
    loader = PyGLoader(test_ds, batch_size=batch_size, shuffle=False)

    ModelClass = get_model()
    model = ModelClass().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1).cpu().numpy()
            preds.append(out)
            labels.append(batch.y.view(-1).cpu().numpy())
    smiles = [it[3] for it in test_ds.items]
    uniprots = [it[4] for it in test_ds.items]
    return np.concatenate(preds), np.concatenate(labels), smiles, uniprots


def run_moltrans(config, test_idx, device, ckpt_path, batch_size=32):
    from adapter_moltrans import MolTransDataset, get_model_config
    from torch.utils.data import DataLoader

    test_ds = MolTransDataset(config, test_idx)
    # drop_last=False here: we can tolerate a final partial batch for eval
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'MolTrans'))
    from models import BIN_Interaction_Flat
    model_config = get_model_config(batch_size=batch_size)
    model = BIN_Interaction_Flat(**model_config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.gpus = 1
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for d_ids, d_mask, p_ids, p_mask, lbl in loader:
            # MolTrans's forward uses view(self.batch_size // self.gpus, ...) —
            # patch model.batch_size to match the actual batch we're feeding
            actual = d_ids.size(0)
            model.batch_size = actual
            d_ids = d_ids.to(device).long()
            d_mask = d_mask.to(device)
            p_ids = p_ids.to(device).long()
            p_mask = p_mask.to(device)
            score = model(d_ids, p_ids, d_mask, p_mask).squeeze(-1).cpu().numpy()
            preds.append(score)
            labels.append(lbl.numpy())
    # MolTrans items: (seq, smiles, label) — need to map seq back to uniprot
    seq_to_uni = {v: k for k, v in config.load_sequences().items()}
    smiles = [it[1] for it in test_ds.items]
    uniprots = [seq_to_uni.get(it[0]) for it in test_ds.items]
    return np.concatenate(preds), np.concatenate(labels), smiles, uniprots


def run_gems(config, test_idx, device, ckpt_path, batch_size=32):
    from adapter_gems import GEMSDataset, get_model
    from torch_geometric.loader import DataLoader as PyGLoader

    test_ds = GEMSDataset(config, test_idx)
    # GEMS has BatchNorm → drop_last for safety
    loader = PyGLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    ModelClass = get_model()
    model = ModelClass(dropout_prob=0.1, in_channels=8,
                       edge_dim=4, conv_dropout_prob=0.1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    preds, labels = [], []
    kept_items = []
    with torch.no_grad():
        n_seen = 0
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1).cpu().numpy()
            preds.append(out)
            labels.append(batch.y.view(-1).cpu().numpy())
            # The drop_last=True means we only keep full batches' items
            bs = out.shape[0]
            kept_items.extend(test_ds.items[n_seen:n_seen + bs])
            n_seen += bs
    # GEMS items: (smiles, uniprot, label)
    smiles = [it[0] for it in kept_items]
    uniprots = [it[1] for it in kept_items]
    return np.concatenate(preds), np.concatenate(labels), smiles, uniprots


def run_drugban(config, test_idx, device, ckpt_path, batch_size=32):
    import dgl
    from torch.utils.data import DataLoader
    from adapter_drugban import DrugBANDataset, get_model
    from train_original_drugban import collate_drugban  # noqa

    test_ds = DrugBANDataset(config, test_idx)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_drugban, drop_last=True)

    ModelClass = get_model()
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'external', 'DrugBAN'))
    from configs import get_cfg_defaults
    model = ModelClass(**get_cfg_defaults()).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    preds, labels = [], []
    kept_items = []
    with torch.no_grad():
        n_seen = 0
        for graphs, prot_ids, lbls in loader:
            graphs = graphs.to(device)
            prot_ids = prot_ids.to(device)
            out = model(graphs, prot_ids)
            if isinstance(out, tuple):
                out = out[-1]
            out = out.squeeze(-1).cpu().numpy()
            preds.append(out)
            labels.append(lbls.cpu().numpy())
            bs = out.shape[0]
            kept_items.extend(test_ds.items[n_seen:n_seen + bs])
            n_seen += bs
    # DrugBAN items: (smiles, uniprot, label)
    smiles = [it[0] for it in kept_items]
    uniprots = [it[1] for it in kept_items]
    return np.concatenate(preds), np.concatenate(labels), smiles, uniprots


def run_rankbind(config, test_idx, device, ckpt_path, batch_size=64):
    """Run v5_rankbind on its test split.

    ckpt_path here is the *run_dir* (not a checkpoint file): v5_rankbind
    evaluation is run-id-scoped. A canonical path is results/v5_rankbind/current/
    which should symlink to the chosen run.
    """
    run_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else ckpt_path
    sys.path.insert(0, PROJECT_ROOT)
    from v5_rankbind.run_manifest import load_config
    from v5_rankbind.data import build_datasets
    from v5_rankbind.model import RankBind
    from v5_rankbind.eval import run_test_set

    manifest = json.load(open(os.path.join(run_dir, 'manifest.json')))
    cfg = load_config(manifest['config_path'])

    chemberta_cache = os.path.join(PROJECT_ROOT, 'data', 'chemberta_cache')
    _, _, test_ds, _ = build_datasets(cfg, chemberta_cache)

    model = RankBind(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pt'),
                                     map_location=device, weights_only=True))
    df = run_test_set(model, test_ds, device, batch_size=batch_size)
    return (df['score'].to_numpy(), df['label'].to_numpy(),
            df['smiles'].tolist(), df['uniprot'].tolist())


RUNNERS = {
    'graphdta': run_graphdta,
    'moltrans': run_moltrans,
    'gems':     run_gems,
    'drugban':  run_drugban,
    'rankbind': run_rankbind,
}


def make_df(scores, labels, smiles, uniprots):
    n = min(len(scores), len(labels), len(smiles), len(uniprots))
    return pd.DataFrame({
        'smiles':  smiles[:n],
        'uniprot': uniprots[:n],
        'score':   scores[:n],
        'label':   labels[:n].astype(int),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(RUNNERS.keys()))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_dir', default=os.path.join(_HERE, 'attractor_results'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.model == 'rankbind':
        # For rankbind, ckpt_path is the run directory, not a single .pt file.
        run_dir = os.path.join(PROJECT_ROOT, 'results', 'v5_rankbind', 'current')
        if not os.path.isdir(run_dir):
            print(f"ERROR: no rankbind current run at {run_dir}. "
                  "Symlink results/v5_rankbind/current to a run_dir first.")
            sys.exit(1)
        ckpt_path = run_dir
    else:
        results_dir = os.path.join(PROJECT_ROOT, 'results', f'original_{args.model}')
        ckpt_path = os.path.join(results_dir, 'best_model.pt')
        if not os.path.exists(ckpt_path):
            print(f"ERROR: checkpoint not found at {ckpt_path}")
            sys.exit(1)

    config = BRENDADataConfig(seed=args.seed)
    _, _, test_idx = config.get_protein_split()
    print(f"Model: {args.model} | Test indices: {len(test_idx)}")

    scores, labels, smiles, uniprots = RUNNERS[args.model](
        config, test_idx, device, ckpt_path, batch_size=args.batch_size
    )
    print(f"Test predictions: {len(scores)} | Positives: {int(labels.sum())}")

    df = make_df(scores, labels, smiles, uniprots)
    df_path = os.path.join(args.out_dir, f'test_preds_{args.model}.csv')
    df.to_csv(df_path, index=False)

    try:
        auc = roc_auc_score(df['label'], df['score'])
    except ValueError:
        auc = float('nan')
    try:
        aupr = average_precision_score(df['label'], df['score'])
    except ValueError:
        aupr = float('nan')

    lig_auc = per_ligand_auc(df)
    hits = hit_at_k(df, ks=(1, 5, 10))

    summary = {
        'model':               args.model,
        'n_test_pairs':        int(len(df)),
        'n_positives':         int(df['label'].sum()),
        'global_auc':          float(auc),
        'global_aupr':         float(aupr),
        'per_ligand_auc':      lig_auc,
        **hits,
    }
    print(json.dumps(summary, indent=2))

    summary_path = os.path.join(args.out_dir, f'test_summary_{args.model}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {df_path}\nSaved: {summary_path}")


if __name__ == '__main__':
    main()
