"""
baselines/adapters/train_graphdta_recipe.py — GraphDTA on the anti-shortcut
recipe (progressive ablation).

Trains the off-the-shelf GraphDTA GCN under the RankBind v4 data regime to test
whether the anti-shortcut advantage is the *recipe* or the RankBind architecture.
Three cumulative variants:

  a  ProteinBalancedSampler + BCE            (sampling axis only)
  b  + within-ligand margin loss             (random cross-protein negatives)
  c  + hard-negative mining                  (top-scoring confuser proteins)

All variants share the protein-disjoint seed-42 split and emit a canonical
[n_lig, n_prot] score matrix + the artifact set evaluation/benchmark_null_eval.py
consumes (score_matrix_rankbind.npy, score_matrix_axes.json,
test_preds_rankbind.csv, test_matrix_ranking.json, manifest.json) so the §8.3
null-baseline instruments and the matrix per-ligand-AUC harness run unchanged.

Usage:
  python baselines/adapters/train_graphdta_recipe.py --variant c \
      --csv_path data/dataset_with_decoys.csv \
      --seq_csv  data/sequences/sequences.csv \
      --out_dir  results/graphdta_recipe/c_brenda --tag c_brenda
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, PROJECT_ROOT)

from common import BRENDADataConfig                                   # noqa: E402
from adapter_graphdta import (                                        # noqa: E402
    GraphDTADataset, get_model, smile_to_graph, seq_to_ids,
)
from graphdta_recipe_collator import (                               # noqa: E402
    GraphDTARecipeDataset, GraphDTATripletCollator, refresh_scores_graphdta,
    _gcn_encode_ligands, _gcn_encode_proteins, _gcn_head, _has_separable_gcn,
)
from v5_rankbind.negative_selection import NegativeSelector          # noqa: E402
from v5_rankbind.sampler import ProteinBalancedSampler               # noqa: E402
from v5_rankbind.loss import RankBindLoss                            # noqa: E402
from v5_rankbind.metrics import (                                     # noqa: E402
    matrix_ranking_metrics, matrix_per_ligand_auc,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["orig", "a", "b", "c"])
    p.add_argument("--csv_path", required=True)
    p.add_argument("--seq_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_matrix", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--early_stop_min_epochs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    # recipe knobs
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--n_negatives", type=int, default=4)
    p.add_argument("--pairs_per_protein", type=int, default=16)
    p.add_argument("--hard_pool_size", type=int, default=50)
    p.add_argument("--hard_refresh_every", type=int, default=1)
    p.add_argument("--hard_refresh_prot_cap", type=int, default=0)  # 0 = no cap
    p.add_argument("--hard_refresh_lig_cap", type=int, default=0)   # 0 = no cap
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── pointwise (variant a) ─────────────────────────────────────────────────────

def _train_epoch_bce(model, loader, optimizer, criterion, device):
    model.train()
    tot, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch).view(-1)
        loss = criterion(out, batch.y.view(-1))
        loss.backward(); optimizer.step()
        tot += loss.item() * batch.y.numel(); n += batch.y.numel()
    return tot / max(n, 1)


# ── margin (variants b / c) ───────────────────────────────────────────────────

def _train_epoch_margin(model, loader, optimizer, lossfn, device):
    model.train()
    tot, n_batches, n_skipped = 0.0, 0, 0
    pos_above = []
    for out in loader:
        if out is None:               # batch with no positive anchor
            n_skipped += 1; continue
        B, k = out["B"], out["k"]
        batch = out["batch"].to(device)
        optimizer.zero_grad()
        s = model(batch).view(-1)
        pos_score = s[:B]
        neg_score = s[B:].view(B, k)
        loss, parts = lossfn.compute_margin(pos_score, neg_score)
        loss.backward(); optimizer.step()
        tot += parts["loss_total"]; n_batches += 1
        pos_above.append(parts["pos_above_neg_max"])
    return {
        "train_loss": tot / max(n_batches, 1),
        "n_batches": n_batches,
        "n_batches_skipped": n_skipped,
        "pos_above_neg_max": float(np.mean(pos_above)) if pos_above else float("nan"),
    }


@torch.no_grad()
def _val_auc(model, loader, device):
    """Pointwise val AUC — uniform early-stop signal across all variants."""
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).view(-1)
        preds.append(out.cpu()); labels.append(batch.y.view(-1).cpu())
    preds = torch.cat(preds).numpy(); labels = torch.cat(labels).numpy()
    try:
        return float(roc_auc_score(labels, preds))
    except ValueError:
        return 0.5


class _ValMatrixScorer:
    """Per-epoch val matrix MRR — the RANKING early-stop signal.

    val_global_auc is the shortcut metric the recipe trades away, so selecting
    checkpoints on it can pick ranking-bad models for variants b/c. We instead
    early-stop on matrix MRR over the val split (stable, n>>2 — the
    per-ligand-AUC n=2 objection that made RankBind v4 use val_global_auc does
    NOT apply to matrix MRR). Built once; each epoch only re-encodes (separable
    GCN branches, cheap).
    """

    def __init__(self, config, val_idx):
        vds = GraphDTADataset(config, val_idx)
        smi_to_graph, uni_to_target, pos = {}, {}, []
        for graph, seq_ids, label, smi, uni in vds.items:
            smi_to_graph.setdefault(smi, graph)
            uni_to_target.setdefault(uni, seq_ids)
            if int(label) == 1:
                pos.append((smi, uni))
        self.ligs = list(smi_to_graph.keys())
        self.lig_graphs = [smi_to_graph[s] for s in self.ligs]
        self.prots = list(uni_to_target.keys())
        self.targets = [uni_to_target[u] for u in self.prots]
        self.pos_pairs = pos

    @torch.no_grad()
    def mrr(self, model, device, row_chunk=64):
        if not self.pos_pairs or not _has_separable_gcn(model):
            return float("nan")
        was_training = model.training
        model.eval()
        L = _gcn_encode_ligands(model, self.lig_graphs, device)
        P = _gcn_encode_proteins(model, self.targets, device)
        C = P.shape[0]
        M = np.zeros((len(self.ligs), C), dtype=np.float32)
        for s in range(0, len(self.ligs), row_chunk):
            e = min(s + row_chunk, len(self.ligs)); R = e - s
            xl = L[s:e].unsqueeze(1).expand(R, C, -1).reshape(R * C, -1)
            xt = P.unsqueeze(0).expand(R, C, -1).reshape(R * C, -1)
            M[s:e] = _gcn_head(model, xl, xt).view(R, C).float().cpu().numpy()
        if was_training:
            model.train()
        return float(matrix_ranking_metrics(M, self.ligs, self.prots, self.pos_pairs)["mrr"])


# ── canonical score matrix + test predictions ─────────────────────────────────

@torch.no_grad()
def build_canonical_matrix(model, config, n_matrix, device, chunk=32):
    """[n_lig, n_prot] score matrix over the canonical pool: proteins = first
    n_matrix seqs, ligands = first n_matrix UNIQUE SMILES — verbatim
    ``pairs['substrate_smiles'].unique()[:n_matrix]`` (no validity filtering),
    so the ligand axis is byte-identical to v5_rankbind/eval.py::build_score_matrix
    and evaluation/matrix_per_ligand_auc_all.py::canonical_axes_and_positives.
    A SMILES GraphDTA cannot featurize keeps its slot with a constant-zero row
    (degenerate → dropped by per-ligand AUC, rank-neutral for MRR); dropping it
    would shift every later index off the canonical axis and pull SMILES #201+
    into the pool, breaking the same-200-pool contract with RankBind."""
    from torch_geometric.data import Batch
    pairs = config.load_pairs(); seqs = config.load_sequences()

    proteins = list(seqs.keys())[:n_matrix]
    prot_targets = [seq_to_ids(seqs[p]) for p in proteins]

    ligands = pairs["substrate_smiles"].unique().tolist()[:n_matrix]
    lig_graphs = [smile_to_graph(s) for s in ligands]
    n_invalid = sum(g is None for g in lig_graphs)
    if n_invalid:
        log.warning(f"{n_invalid}/{len(ligands)} canonical SMILES unfeaturizable "
                    f"→ constant-zero placeholder rows (axis stays aligned)")

    n_lig, n_prot = len(ligands), len(proteins)
    M = np.zeros((n_lig, n_prot), dtype=np.float32)
    model.eval()
    # protein-major: one target broadcast across a chunk of ligand graphs.
    for j in range(n_prot):
        tgt = prot_targets[j]
        for s in range(0, n_lig, chunk):
            e = min(s + chunk, n_lig)
            graphs, idxs = [], []
            for li in range(s, e):
                if lig_graphs[li] is None:
                    continue
                g = lig_graphs[li].clone(); g.target = tgt.unsqueeze(0)
                graphs.append(g); idxs.append(li)
            if not graphs:
                continue
            b = Batch.from_data_list(graphs).to(device)
            out = model(b).view(-1).float().cpu().numpy()
            for o, li in enumerate(idxs):
                M[li, j] = out[o]
    return M, ligands, proteins


@torch.no_grad()
def build_test_preds(model, config, test_idx, device, chunk=64):
    """Score every test (ligand, protein) pair → rows (smiles, uniprot, score, label)."""
    from torch_geometric.data import Batch
    test_ds = GraphDTADataset(config, test_idx)
    rows = []
    model.eval()
    items = test_ds.items  # (graph, seq_ids, label, smiles, uniprot)
    for s in range(0, len(items), chunk):
        e = min(s + chunk, len(items))
        graphs = []
        for graph, seq_ids, label, smi, uni in items[s:e]:
            g = graph.clone(); g.target = seq_ids.unsqueeze(0)
            graphs.append(g)
        b = Batch.from_data_list(graphs).to(device)
        out = model(b).view(-1).float().cpu().numpy()
        for off, (graph, seq_ids, label, smi, uni) in enumerate(items[s:e]):
            rows.append((smi, uni, float(out[off]), int(label)))
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    log.info(f"variant={args.variant} device={device} out={args.out_dir}")

    # csv/seq may be project-relative; resolve for the data layer but keep the
    # original (possibly relative) string in the manifest for portability.
    csv_abs = args.csv_path if os.path.isabs(args.csv_path) else os.path.join(PROJECT_ROOT, args.csv_path)
    seq_abs = args.seq_csv if os.path.isabs(args.seq_csv) else os.path.join(PROJECT_ROOT, args.seq_csv)
    config = BRENDADataConfig(seed=args.seed, csv_path=csv_abs, seq_csv=seq_abs)

    train_idx, val_idx, test_idx = config.get_protein_split()
    log.info(f"split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    from torch_geometric.loader import DataLoader as PyGLoader
    from torch.utils.data import DataLoader as TorchLoader

    ModelClass = get_model()
    model = ModelClass().to(device)
    log.info(f"params={sum(p.numel() for p in model.parameters()):,}")

    # Val loader: pointwise, uniform early-stop signal across variants.
    val_ds = GraphDTADataset(config, val_idx)
    val_loader = PyGLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ── variant-specific train loader ─────────────────────────────────────────
    selector = None
    sampler = None
    refresh_rng = np.random.default_rng(args.seed + 1)
    if args.variant == "orig":
        # Controlled within-GraphDTA floor: vanilla random-shuffle BCE, NO
        # balanced sampler. Isolates the sampler's own contribution (orig → a)
        # so the off-the-shelf GraphDTA baseline and the recipe differ in the
        # recipe only, not in featurization or codepath.
        train_ds = GraphDTADataset(config, train_idx)
        train_loader = PyGLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()
    elif args.variant == "a":
        train_ds = GraphDTADataset(config, train_idx)
        sampler = ProteinBalancedSampler(
            train_ds, pairs_per_protein_per_epoch=args.pairs_per_protein,
            pos_neg_ratio=1.0, seed=args.seed)
        train_loader = PyGLoader(train_ds, batch_size=args.batch_size,
                                 sampler=sampler, shuffle=False)
        criterion = nn.BCEWithLogitsLoss()
    else:
        train_ds = GraphDTARecipeDataset(config, train_idx)
        sampler = ProteinBalancedSampler(
            train_ds, pairs_per_protein_per_epoch=args.pairs_per_protein,
            pos_neg_ratio=1.0, seed=args.seed)
        neg_mode = "hard" if args.variant == "c" else "cross_protein_implicit"
        selector = NegativeSelector(
            train_ds, n_negatives=args.n_negatives, seed=args.seed,
            negative_sampling=neg_mode, hard_pool_size=args.hard_pool_size)
        collator = GraphDTATripletCollator(train_ds, selector)
        train_loader = TorchLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            shuffle=False, collate_fn=collator)
        lossfn = RankBindLoss({"type": "margin", "margin": args.margin})

    # Sampler audit (paper supplement) — proves the rebalance, model-free.
    # orig has no balanced sampler (random shuffle), so there is nothing to audit.
    if sampler is not None:
        try:
            sampler.audit(os.path.join(args.out_dir, "sampler_audit.csv"))
        except Exception as e:  # noqa: BLE001
            log.warning(f"sampler audit skipped: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Checkpoint selection + early stop run on the val matrix MRR — the ranking
    # objective the recipe optimises — NOT val_global_auc (the shortcut metric
    # the recipe deliberately trades away; selecting b/c on it would reward the
    # very pathology under test). MRR over the val split is stable (n≫2, unlike
    # the per-ligand-AUC n=2 that forced RankBind v4 onto val_global_auc).
    # val_global_auc is still logged every epoch as a diagnostic. Fall back to
    # AUC only if the val split has no rankable positive pairs.
    bce_variant = args.variant in ("orig", "a")
    val_scorer = _ValMatrixScorer(config, val_idx)
    use_mrr_selection = bool(val_scorer.pos_pairs) and _has_separable_gcn(model)
    sel_metric = "val_matrix_mrr" if use_mrr_selection else "val_global_auc"
    if not use_mrr_selection:
        log.warning("val matrix MRR unavailable → selecting on val_global_auc")
    log.info(f"checkpoint selection metric: {sel_metric} "
             f"(min_epochs={args.early_stop_min_epochs}, patience={args.patience})")

    best_sel, best_path, patience = -1.0, os.path.join(args.out_dir, "best_model.pt"), 0
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(log_path, "w") as lf:
        for epoch in range(1, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            rec = {"epoch": epoch}

            if args.variant == "c" and (epoch - 1) % args.hard_refresh_every == 0:
                info = refresh_scores_graphdta(
                    model, train_ds, selector, device,
                    prot_cap=args.hard_refresh_prot_cap or None,
                    lig_cap=args.hard_refresh_lig_cap or None,
                    rng=refresh_rng)
                rec["hard_refresh"] = info
                if info.get("refreshed") and (
                        info.get("lig_coverage", 1) < 1 or info.get("prot_coverage", 1) < 1):
                    log.warning(f"hard-neg refresh partial coverage: "
                                f"lig={info['lig_coverage']} prot={info['prot_coverage']} "
                                f"(un-scored anchors fall back to random negatives)")

            if bce_variant:
                rec["train_loss"] = _train_epoch_bce(model, train_loader, optimizer,
                                                     criterion, device)
            else:
                rec.update(_train_epoch_margin(model, train_loader, optimizer,
                                               lossfn, device))

            val_auc = _val_auc(model, val_loader, device)
            val_mrr = val_scorer.mrr(model, device) if use_mrr_selection else float("nan")
            rec["val_global_auc"] = val_auc
            rec["val_matrix_mrr"] = val_mrr
            sel = val_mrr if use_mrr_selection else val_auc
            scheduler.step(1.0 - sel)
            lf.write(json.dumps(rec) + "\n"); lf.flush()

            if sel > best_sel:
                best_sel = sel; patience = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
            if epoch == 1 or epoch % 5 == 0:
                extra = "" if bce_variant else \
                    f" pos>negmax={rec.get('pos_above_neg_max', float('nan')):.3f}"
                log.info(f"epoch {epoch:3d} | trainLoss={rec.get('train_loss', float('nan')):.4f} "
                         f"valAUC={val_auc:.4f} valMRR={val_mrr:.4f} "
                         f"best[{sel_metric}]={best_sel:.4f}{extra}")
            # min-epochs guard: never stop before the model has had time to move
            # off the BCE-shortcut basin into ligand-conditional ranking.
            if patience >= args.patience and epoch >= args.early_stop_min_epochs:
                log.info(f"early stop @ {epoch}"); break

    # ── final artifacts from best checkpoint ──────────────────────────────────
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    log.info("building canonical score matrix…")
    M, ligands, proteins = build_canonical_matrix(model, config, args.n_matrix, device)
    assert M.shape == (len(ligands), len(proteins)), (M.shape, len(ligands), len(proteins))
    np.save(os.path.join(args.out_dir, "score_matrix_rankbind.npy"), M)
    json.dump({"axis_0_ligands": ligands, "axis_1_proteins": proteins,
               "shape": list(M.shape)},
              open(os.path.join(args.out_dir, "score_matrix_axes.json"), "w"))

    log.info("scoring test pairs…")
    test_rows = build_test_preds(model, config, test_idx, device)
    import csv as _csv
    with open(os.path.join(args.out_dir, "test_preds_rankbind.csv"), "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["smiles", "uniprot", "score", "label"])
        w.writerows(test_rows)

    positive_pairs = [(smi, uni) for smi, uni, _s, lab in test_rows if lab == 1]
    rank = matrix_ranking_metrics(M, ligands, proteins, positive_pairs)
    json.dump(rank, open(os.path.join(args.out_dir, "test_matrix_ranking.json"), "w"), indent=2)
    log.info(f"matrix MRR={rank['mrr']:.4f} hit@10={rank['hit_at_10']:.4f} "
             f"n_pos={rank['n_positive_pairs_matched']}")

    # Matrix per-ligand AUC on the same axes/positives — the metric on which the
    # off-the-shelf architecture collapses (≈0.5); the a→b→c lift here is the
    # recipe-transfer evidence. (matrix_per_ligand_auc_all.py recomputes this
    # against the single canonical positive source; stored here for the manifest.)
    plig = matrix_per_ligand_auc(M, ligands, proteins, positive_pairs)
    log.info(f"matrix per-ligand AUC={plig['matrix_per_ligand_auc']:.4f} "
             f"(n={plig['n_ligands_counted']})")

    # ── manifest (benchmark_null_eval reads config_resolved.data / seed / name) ─
    name = f"graphdta_recipe_{args.variant}" + (f"_{args.tag}" if args.tag else "")
    manifest = {
        "run_id": os.path.basename(args.out_dir.rstrip("/")),
        "tag": args.tag,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_resolved": {
            "name": name,
            "seed": args.seed,
            "variant": args.variant,
            "data": {
                "csv_path": args.csv_path,
                "seq_csv": args.seq_csv,
                "val_frac": config.val_frac,
                "test_frac": config.test_frac,
            },
            "recipe": {
                "sampler": "protein_balanced",
                "loss": "bce" if args.variant == "a" else "margin",
                "negatives": (None if args.variant == "a"
                              else ("hard" if args.variant == "c"
                                    else "cross_protein_implicit")),
                "margin": args.margin,
                "n_negatives": args.n_negatives,
                "hard_pool_size": args.hard_pool_size,
            },
        },
        "metrics": {
            f"best_{sel_metric}": best_sel,
            "selection_metric": sel_metric,
            "matrix_per_ligand_auc": plig["matrix_per_ligand_auc"],
            "matrix_per_ligand_n": plig["n_ligands_counted"],
            **{f"matrix_{k}": v for k, v in rank.items()},
        },
        "model": "graphdta",
    }
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"), indent=2)
    log.info(f"done → {args.out_dir}")


if __name__ == "__main__":
    main()
