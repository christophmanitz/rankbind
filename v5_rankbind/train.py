"""
v5_rankbind/train.py — Train RankBind with full provenance.

Usage:
    python -m v5_rankbind.train --config v5_rankbind/configs/default.json

Produces under results/v5_rankbind/<run_id>/:
    manifest.json               — provenance record (final on exit)
    best_model.pt               — best-val checkpoint (state_dict)
    train_log.jsonl             — per-epoch metrics
    sampler_audit.csv           — per-protein positives/negatives drawn (first epoch)
    val_preds_best.csv          — val predictions of the best model

The score matrix + test-set evaluation live in eval.py; that's intentional —
training is already long enough without the 200×200 sweep.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v5_rankbind.run_manifest import (
    RunManifest, load_config, set_deterministic_seeds
)
from v5_rankbind.data import (
    ensure_chemberta_cache, build_datasets, collate_pointwise
)
from v5_rankbind.sampler import ProteinBalancedSampler, TripletCollator
from v5_rankbind.model import RankBind
from v5_rankbind.loss import RankBindLoss
from v5_rankbind.metrics import per_ligand_auc, global_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def make_train_loader(
    train_ds, cfg: dict, triplet_collator: TripletCollator | None
) -> tuple[DataLoader, ProteinBalancedSampler | None]:
    """Train loader. If loss is margin, uses triplet_collator (can return None batches)."""
    sampler_cfg = cfg["sampler"]
    if sampler_cfg["type"] == "protein_balanced":
        sampler = ProteinBalancedSampler(
            train_ds,
            pairs_per_protein_per_epoch=sampler_cfg["pairs_per_protein_per_epoch"],
            pos_neg_ratio=sampler_cfg["pos_neg_ratio"],
            seed=cfg["seed"],
        )
        shuffle = False
    elif sampler_cfg["type"] == "random":
        sampler = None
        shuffle = True
    else:
        raise ValueError(f"Unknown sampler type: {sampler_cfg['type']}")

    if triplet_collator is not None:
        collate = triplet_collator
    else:
        collate = collate_pointwise

    loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size_ligands"],
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        collate_fn=collate,
        num_workers=0,
        drop_last=False,
    )
    return loader, sampler


def make_eval_loader(ds, batch_size: int) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pointwise, num_workers=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Eval (during training)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: RankBind, loader: DataLoader, device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    model.eval()
    rows = []
    for batch in loader:
        lig = batch["lig_emb"].to(device)
        prot = batch["prot_emb"].to(device)
        prot_mask = batch.get("prot_mask")
        if prot_mask is not None:
            prot_mask = prot_mask.to(device)
        score = model.score_pairs(lig, prot, prot_mask).float().cpu().numpy()
        for s, uni, sc, lb in zip(batch["smiles"], batch["uniprot"], score, batch["label"]):
            rows.append({"smiles": s, "uniprot": uni, "score": float(sc), "label": int(lb)})
    df = pd.DataFrame(rows)
    scores = df["score"].to_numpy()
    labels = df["label"].to_numpy()
    smiles = df["smiles"].tolist()

    per_lig, n_lig = per_ligand_auc(smiles, scores, labels)
    metrics = {
        "val_per_lig_auc": per_lig,
        "val_n_ligands_counted": n_lig,
        **{f"val_{k}": v for k, v in global_metrics(scores, labels).items()},
    }
    return df, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Train step dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch_margin(
    model: RankBind,
    loader: DataLoader,
    loss_fn: RankBindLoss,
    opt: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
) -> dict:
    model.train()
    accum = {"loss_total": 0.0, "n": 0, "kept_ratio": [],
             "pos_above_neg_max_sum": 0.0, "margin_viol_sum": 0.0,
             "n_hard_active": 0, "n_batches": 0}
    n_none = 0
    for batch in loader:
        if batch is None:
            n_none += 1; continue
        lig = batch["lig_emb"].to(device)
        pos = batch["pos_prot"].to(device)
        neg = batch["neg_prot"].to(device)
        pos_mask = batch.get("pos_mask")
        neg_mask = batch.get("neg_mask")
        if pos_mask is not None:
            pos_mask = pos_mask.to(device)
        if neg_mask is not None:
            neg_mask = neg_mask.to(device)
        pos_s, neg_s = model.score_triplet(lig, pos, neg, pos_mask, neg_mask)
        loss, parts = loss_fn.compute_margin(pos_s, neg_s)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if scheduler is not None: scheduler.step()

        bsz = pos_s.shape[0]
        accum["loss_total"] += parts["loss_total"] * bsz
        accum["n"] += bsz
        accum["kept_ratio"].append(batch["n_anchors_kept"] / max(batch["n_anchors_in"], 1))
        accum["pos_above_neg_max_sum"] += parts["pos_above_neg_max"] * bsz
        accum["margin_viol_sum"] += parts["margin_violation_rate"] * bsz
        accum["n_batches"] += 1
        if batch.get("hard_active"):
            accum["n_hard_active"] += 1
    if accum["n"] == 0:
        return {"train_loss": float("nan"), "n_batches_skipped": n_none}
    return {
        "train_loss":                  accum["loss_total"] / accum["n"],
        "train_keep_ratio_mean":       float(np.mean(accum["kept_ratio"])) if accum["kept_ratio"] else 0.0,
        "train_pos_above_neg_max":     accum["pos_above_neg_max_sum"] / accum["n"],
        "train_margin_violation_rate": accum["margin_viol_sum"] / accum["n"],
        "train_hard_active_batches":   accum["n_hard_active"],
        "n_batches_skipped":           n_none,
    }


def train_epoch_bce(
    model: RankBind,
    loader: DataLoader,
    loss_fn: RankBindLoss,
    opt: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
) -> dict:
    model.train()
    total = 0.0; n = 0
    for batch in loader:
        lig = batch["lig_emb"].to(device)
        prot = batch["prot_emb"].to(device)
        prot_mask = batch.get("prot_mask")
        if prot_mask is not None:
            prot_mask = prot_mask.to(device)
        lab = batch["label"].to(device)
        score = model.score_pairs(lig, prot, prot_mask)
        loss, parts = loss_fn.compute_bce(score, lab)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if scheduler is not None: scheduler.step()

        total += parts["loss_total"] * score.shape[0]
        n += score.shape[0]
    return {"train_loss": total / max(n, 1)}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_root", default=str(PROJECT_ROOT / "results" / "v5_rankbind"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dry_run_epochs", type=int, default=0,
                    help="If >0, cap training to N epochs and skip full early-stop logic.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override cfg['seed']; used for multi-seed ablation sweeps.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_override_note = None
    if args.seed is not None:
        seed_override_note = f"seed override: cfg={cfg['seed']} → cli={args.seed}"
        cfg["seed"] = int(args.seed)
    set_deterministic_seeds(cfg["seed"])
    device = torch.device(args.device)

    manifest = RunManifest.start(
        config_path=args.config, config=cfg, out_root=args.out_root, tag=args.tag
    )
    manifest.note(f"Command: {' '.join(sys.argv)}")
    if seed_override_note is not None:
        manifest.note(seed_override_note)
        print(f"[seed] {seed_override_note}")
    print(f"[manifest] run_id = {manifest.run_id}")
    print(f"[manifest] run_dir = {manifest.run_dir}")

    # Record input provenance
    manifest.record_inputs({
        "csv":    cfg["data"]["csv_path"],
        "seqs":   cfg["data"]["seq_csv"],
    })

    # ── ChemBERTa cache ───────────────────────────────────────────────────
    chemberta_cache_dir = PROJECT_ROOT / "data" / "chemberta_cache"
    # Gather the set of SMILES across all splits
    import pandas as pd
    csv_path = PROJECT_ROOT / cfg["data"]["csv_path"]
    all_smiles = pd.read_csv(csv_path)["substrate_smiles"].dropna().unique().tolist()
    t0 = time.time()
    ensure_chemberta_cache(
        all_smiles, chemberta_cache_dir,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(f"[chemberta] cache ready ({len(all_smiles)} SMILES, {time.time()-t0:.1f}s)")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, split_stats = build_datasets(cfg, chemberta_cache_dir)
    manifest.record_split(**split_stats)
    print(f"[data] train={split_stats['n_train_pairs']} "
          f"val={split_stats['n_val_pairs']} test={split_stats['n_test_pairs']}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = RankBind(cfg).to(device)
    manifest.record_model(**model.count_parameters())
    print(f"[model] head={cfg['model']['head']} "
          f"trainable={model.count_parameters()['n_parameters_trainable']:,}")

    # ── Loss / loaders ────────────────────────────────────────────────────
    loss_fn = RankBindLoss(cfg["loss"])
    triplet_collator = None
    if loss_fn.type == "margin":
        triplet_cfg = cfg["triplet"]
        triplet_collator = TripletCollator(
            train_dataset=train_ds,
            n_negatives=triplet_cfg["n_negatives_per_positive"],
            seed=cfg["seed"],
            negative_sampling=triplet_cfg.get(
                "negative_sampling", "cross_protein_implicit"),
            hard_pool_size=triplet_cfg.get("hard_pool_size", 50),
        )
        manifest.note(
            f"triplet collator: negative_sampling={triplet_collator.negative_sampling} "
            f"hard_pool_size={triplet_collator.hard_pool_size} "
            f"n_negatives={triplet_collator.n_negatives}"
        )
        print(f"[triplet] negative_sampling={triplet_collator.negative_sampling} "
              f"hard_pool_size={triplet_collator.hard_pool_size}")

    train_loader, sampler = make_train_loader(train_ds, cfg, triplet_collator)
    val_loader = make_eval_loader(val_ds, batch_size=64)

    # Audit the sampler (first-epoch draw, deterministic at epoch 0)
    if sampler is not None:
        audit_path = manifest.path("sampler_audit.csv")
        totals = sampler.audit(audit_path)
        manifest.record_output("sampler_audit", str(audit_path))
        manifest.note(f"sampler audit totals: {totals}")
        print(f"[sampler] audit: {totals}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    n_epochs = (args.dry_run_epochs or cfg["train"]["epochs"])
    steps_per_epoch = max(1, math.ceil(len(train_loader)))
    total_steps = steps_per_epoch * n_epochs
    if cfg["train"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=cfg["train"]["min_lr"],
        )
    else:
        scheduler = None

    # ── Train loop ────────────────────────────────────────────────────────
    log_jsonl = manifest.open_jsonl("train_log.jsonl")
    best_ckpt_path = manifest.path("best_model.pt")
    best_metric = -1.0
    best_epoch = -1
    patience_counter = 0
    patience = cfg["train"]["early_stop_patience"]
    min_epochs = cfg["train"].get("early_stop_min_epochs", 0)
    es_key = cfg["train"].get("early_stop_metric", "val_global_auc")

    train_step = (train_epoch_margin if loss_fn.type == "margin"
                  else train_epoch_bce)

    for epoch in range(1, n_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch - 1)
        # Refresh hard-negative score cache at the start of each margin epoch.
        # No-op unless negative_sampling == "hard"; uses the current model
        # parameters to identify the top-hard_pool_size confusers per ligand.
        if triplet_collator is not None and triplet_collator.use_hard:
            t_refresh = time.time()
            ref_info = triplet_collator.refresh_scores(model, device)
            refresh_wall = time.time() - t_refresh
            if epoch == 1:
                print(f"[hard-neg] refreshed scores "
                      f"({ref_info.get('n_lig', 0)} ligands × "
                      f"{ref_info.get('n_prot', 0)} proteins, "
                      f"{refresh_wall:.2f}s)")
        t_start = time.time()
        train_stats = train_step(
            model, train_loader, loss_fn, opt, scheduler, device,
            grad_clip=cfg["train"]["grad_clip"],
        )
        val_df, val_metrics = evaluate(model, val_loader, device)
        wall = time.time() - t_start
        lr_now = opt.param_groups[0]["lr"]

        row = {
            "epoch": epoch, "lr": lr_now, "wall_s": round(wall, 2),
            **train_stats, **val_metrics,
        }
        log_jsonl.write(json.dumps(row) + "\n")
        pos_over = train_stats.get("train_pos_above_neg_max")
        pos_over_str = f" pos>maxneg={pos_over:.3f}" if pos_over is not None else ""
        print(f"[epoch {epoch:3d}] loss={train_stats.get('train_loss', float('nan')):.4f}{pos_over_str} "
              f"val_per_lig_auc={val_metrics['val_per_lig_auc']:.4f} "
              f"val_global_auc={val_metrics['val_global_auc']:.4f} "
              f"({wall:.1f}s, lr={lr_now:.2e})")

        metric = val_metrics.get(es_key, float("nan"))
        if not math.isnan(metric) and metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)
            # Persist val predictions of the current best
            val_df.to_csv(manifest.path("val_preds_best.csv"), index=False)
            patience_counter = 0
            manifest.record_output("checkpoint", str(best_ckpt_path))
            manifest.record_output("val_preds_best", str(manifest.path("val_preds_best.csv")))
        else:
            patience_counter += 1
            if (patience_counter >= patience and args.dry_run_epochs == 0
                    and epoch >= min_epochs):
                manifest.note(f"Early stop at epoch {epoch} (patience={patience}) on {es_key}")
                print(f"[early-stop] triggered at epoch {epoch}")
                break

    log_jsonl.close()

    manifest.record_output("train_log", str(manifest.path("train_log.jsonl")))
    manifest.record_metrics(
        best_val_metric=best_metric,
        best_val_metric_key=es_key,
        best_val_epoch=best_epoch,
    )
    manifest.finish()
    print(f"[done] best {es_key}={best_metric:.4f} @ epoch {best_epoch}")
    print(f"[done] manifest: {manifest.run_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
