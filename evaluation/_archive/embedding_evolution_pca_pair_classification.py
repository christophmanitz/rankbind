"""
evaluation/embedding_evolution_pca.py — visualise how pair representations
evolve through the RankBind architecture.

Forward-passes the test set through the gold-standard model
(v5b/abl_attn_pool seed=7 by default; configurable) and captures pair-level
representations at six architectural milestones:

  0  raw_inputs       concat(ChemBERTa-mean[L=384], ESM2-mean[D=1280])
  1  after_LN         concat(post-LN ligand[384], post-LN protein[1280])
                      -- ligand LN inside LigandProjector,
                         protein LN inside the attention-pool module
  2  after_hidden     concat(post-(Linear+GELU+Dropout) ligand[256],
                             same for protein[256])
  3  proj_concat      concat(fL[256], gP[256])
                      -- the input an MLP-concat head would see
  4  bilinear_low     (fL @ U) * (gP @ V)  [rank=128]
                      -- the actual low-rank interaction term
                         the bilinear head sums to a score
  5  score            scalar pre-bias-and-diag-residual contribution
                      -- the model's final pair output

Per stage:
  - 2-D PCA projection with binding label colouring (binder vs decoy).
  - Linear-probe AUC (logistic regression with 3-fold CV) on the
    high-dimensional representation, summarising how linearly separable
    binders are at that stage.

Output:
  paper/figures/fig_embedding_evolution_pca.png         (6-panel grid)
  evaluation/attractor_results/embedding_evolution_pca.csv
                                                        (one row per stage)

Defaults to a single batch over the full test set; with the v5b model and a
modern GPU this is one forward pass, well under a minute.

Usage:
  python evaluation/embedding_evolution_pca.py
  python evaluation/embedding_evolution_pca.py \\
         --run_dir results/v5_rankbind/20260427-121212_..._abl_attn_pool_v5b_s7 \\
         --max_pairs 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v5_rankbind.data import build_datasets, collate_pointwise  # noqa: E402
from v5_rankbind.model import RankBind  # noqa: E402
from v5_rankbind.run_manifest import load_config  # noqa: E402


DEFAULT_RUN = (
    "results/v5_rankbind/20260427-121212_1746525d51_abl_attn_pool_v5b_s7"
)


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def capture_intermediates(
    model: RankBind,
    batch: dict,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run one forward pass and return per-stage pair-level numpy arrays.

    Returns
    -------
    dict with keys
      raw_inputs    [N, 384+D_prot]
      after_LN      [N, 384+D_prot]
      after_hidden  [N, 256+256]
      proj_concat   [N, 256+256]
      bilinear_low  [N, rank]   (only if bilinear head)
      score         [N]
      labels        [N]
    """
    lig_in = batch["lig_emb"].to(device)                 # [B, 384]
    prot_in = batch["prot_emb"].to(device)               # [B, L, 1280] or [B, 1280]
    prot_mask = batch.get("prot_mask")
    if prot_mask is not None:
        prot_mask = prot_mask.to(device)
    labels = batch["label"].cpu().numpy().astype(int)

    # ── Stage 0: raw inputs ───────────────────────────────────────────────
    if prot_in.dim() == 3:
        # attn_pool input: collapse to a per-pair vector by simple mean for
        # the "raw" comparison. This isolates the "what was on disk" view.
        if prot_mask is not None:
            denom = prot_mask.sum(dim=1, keepdim=True).clamp_min(1)
            prot_raw_pooled = (prot_in * prot_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            prot_raw_pooled = prot_in.mean(dim=1)
    else:
        prot_raw_pooled = prot_in
    raw_inputs = torch.cat([lig_in, prot_raw_pooled], dim=-1)

    # ── Replay LigandProjector explicitly so we can grab intermediates ────
    lig_net = model.lig.net  # nn.Sequential(LN, Linear, GELU, Dropout, Linear)
    L_ln = lig_net[0](lig_in)            # LayerNorm
    L_h = lig_net[1](L_ln)               # Linear
    L_h = lig_net[2](L_h)                # GELU
    L_h = lig_net[3](L_h)                # Dropout
    fL = lig_net[4](L_h)                 # Linear  → [B, d_lig]

    # ── Replay protein encoding to grab the attention-pool output ─────────
    if model.attn_pool is not None:
        # ResidueAttentionPool: LN inside, then weighted mean
        # We expose post-attn_pool (== pre-projection) as the "raw protein"
        # the projector receives.
        prot_pooled, _w = model.attn_pool(prot_in, prot_mask)   # [B, 1280]
    else:
        prot_pooled = prot_in
    prot_net = model.prot.net
    P_ln = prot_net[0](prot_pooled)
    P_h = prot_net[1](P_ln)
    P_h = prot_net[2](P_h)
    P_h = prot_net[3](P_h)
    gP = prot_net[4](P_h)

    after_LN = torch.cat([L_ln, P_ln], dim=-1)
    after_hidden = torch.cat([L_h, P_h], dim=-1)
    proj_concat = torch.cat([fL, gP], dim=-1)

    out = {
        "raw_inputs":   raw_inputs.cpu().numpy(),
        "after_LN":     after_LN.cpu().numpy(),
        "after_hidden": after_hidden.cpu().numpy(),
        "proj_concat":  proj_concat.cpu().numpy(),
        "labels":       labels,
    }

    # ── Stage 4: bilinear interaction term (only if head is bilinear) ─────
    if model.head_type == "bilinear":
        head = model.head
        low = (fL @ head.U) * (gP @ head.V)        # [B, rank]
        out["bilinear_low"] = low.cpu().numpy()
        out["score"] = (low.sum(dim=-1)
                        + (fL * head.d * gP).sum(dim=-1)
                        + head.b).cpu().numpy()
    else:
        # MLP-concat head: capture the head's penultimate representation as
        # the closest analogue to the bilinear interaction term.
        x = torch.cat([fL, gP], dim=-1)
        h = model.head.net[0](x)            # Linear
        h = model.head.net[1](h)            # GELU
        h = model.head.net[2](h)            # Dropout
        out["bilinear_low"] = h.cpu().numpy()
        out["score"] = model.head.net[3](h).squeeze(-1).cpu().numpy()
    return out


def linear_probe_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 3,
                     seed: int = 42) -> float:
    """Stratified k-fold logistic-regression probe AUC.

    Reports the mean ROC-AUC of a logistic regression fit to the embedding,
    evaluated out-of-fold. A single number per stage; rises when the
    embedding becomes more linearly separable in the binding direction.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(X[tr], y[tr])
        scores = clf.decision_function(X[te])
        aucs.append(roc_auc_score(y[te], scores))
    return float(np.mean(aucs))


def pca_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return [N, 2] PCA projection. Pads constant features to avoid
    sklearn's "0 informative components" warning on 1-D inputs.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 1:
        X = np.concatenate([X, np.zeros_like(X)], axis=1)
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X), pca.explained_variance_ratio_


STAGE_ORDER = [
    ("raw_inputs",   "0. Raw inputs\nChemBERTa ⊕ ESM2 (mean)"),
    ("after_LN",     "1. After LayerNorm\nin both projectors"),
    ("after_hidden", "2. After Linear + GELU\n(hidden 256)"),
    ("proj_concat",  "3. fL ⊕ gP\n(final projections)"),
    ("bilinear_low", "4. Bilinear interaction\n(fL @ U) ⊙ (gP @ V)"),
    ("score",        "5. Score\n(model output)"),
]


def make_figure(
    captured: dict[str, np.ndarray],
    auc_table: list[dict],
    out_path: Path,
    run_label: str,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    y = captured["labels"]
    pos = y == 1
    neg = y == 0

    for ax, (key, title) in zip(axes, STAGE_ORDER):
        if key not in captured:
            ax.set_visible(False)
            continue
        X = captured[key]
        auc = next((r["linear_probe_auc"] for r in auc_table if r["stage"] == key),
                   float("nan"))

        if key == "score":
            # 1-D distribution
            ax.hist(X[neg], bins=40, alpha=0.55, color="#d62728", label="non-binder")
            ax.hist(X[pos], bins=40, alpha=0.55, color="#1f77b4", label="binder")
            ax.set_xlabel("score")
            ax.set_ylabel("count")
            ax.legend(fontsize=8, loc="upper right")
        else:
            P, vr = pca_2d(X)
            ax.scatter(P[neg, 0], P[neg, 1], s=8, alpha=0.55, c="#d62728",
                       edgecolors="none", label=f"non-binder (n={int(neg.sum())})")
            ax.scatter(P[pos, 0], P[pos, 1], s=8, alpha=0.7, c="#1f77b4",
                       edgecolors="none", label=f"binder (n={int(pos.sum())})")
            ax.set_xlabel(f"PC1  ({vr[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2  ({vr[1]*100:.1f}%)")
            ax.legend(fontsize=8, loc="best")
        ax.set_title(f"{title}\nlinear-probe AUC = {auc:.3f}", fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"RankBind pair-representation evolution — {run_label}\n"
        f"left→right, top→bottom: each architectural stage",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=DEFAULT_RUN,
                    help="run directory under results/v5_rankbind/ "
                         "(must contain best_model.pt + manifest.json)")
    ap.add_argument("--max_pairs", type=int, default=2000,
                    help="cap test set to first N pairs (default 2000)")
    ap.add_argument(
        "--out_fig", default="paper/figures/fig_embedding_evolution_pca.png",
    )
    ap.add_argument(
        "--out_csv",
        default="evaluation/attractor_results/embedding_evolution_pca.csv",
    )
    args = ap.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    if not (run_dir / "best_model.pt").exists():
        raise FileNotFoundError(
            f"No best_model.pt under {run_dir}. Provided --run_dir might be wrong."
        )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config_resolved"]
    run_label = manifest.get("run_id", run_dir.name)

    device = _select_device()
    print(f"[probe] device      = {device}")
    print(f"[probe] run_dir     = {run_dir}")
    print(f"[probe] run_label   = {run_label}")

    # Build the dataset using the resolved config so the protein-encoder
    # mode + ESM2 dir come straight from the run's manifest.
    chemberta_cache_dir = PROJECT_ROOT / "data" / "chemberta_cache"
    train_ds, val_ds, test_ds, split_stats = build_datasets(
        cfg, chemberta_cache_dir,
    )
    n_test = min(args.max_pairs, len(test_ds))
    indices = list(range(n_test))
    items = [test_ds[i] for i in indices]
    batch = collate_pointwise(items)
    print(f"[probe] test pairs  = {len(items)} "
          f"(positives={int(batch['label'].sum().item())})")

    # Build model + load checkpoint.
    model = RankBind(cfg).to(device).eval()
    sd = torch.load(run_dir / "best_model.pt", map_location=device,
                    weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)

    captured = capture_intermediates(model, batch, device)

    auc_rows = []
    for key, _title in STAGE_ORDER:
        if key not in captured:
            continue
        X = captured[key]
        y = captured["labels"]
        auc = linear_probe_auc(X, y)
        auc_rows.append({
            "stage":             key,
            "n_dim":             int(X.shape[1] if X.ndim > 1 else 1),
            "linear_probe_auc":  auc,
        })
        print(f"[probe] {key:13s} d={auc_rows[-1]['n_dim']:>4}  "
              f"linear_probe_AUC={auc:.3f}")

    out_csv = PROJECT_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(auc_rows).to_csv(out_csv, index=False)

    out_fig = PROJECT_ROOT / args.out_fig
    make_figure(captured, auc_rows, out_fig, run_label=run_label)

    print(f"[probe] saved CSV   = {out_csv}")
    print(f"[probe] saved fig   = {out_fig}")


if __name__ == "__main__":
    main()
