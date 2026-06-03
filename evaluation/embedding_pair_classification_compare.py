"""
evaluation/embedding_pair_classification_compare.py — pair-level linear-probe
AUC through the architecture, for two trained models side by side.

The "before" model is `abl_bce_only` (MLP-concat head, BCE loss, random batches,
no margin, no balanced sampler) — the in-package re-creation of the Phase-1
shortcut pathology. The "after" model is the gold standard
(`abl_attn_pool_v5b_s7`, the residue-level RankBind).

For each model, every test (ligand, protein) pair is forwarded once;
intermediate concat-pair representations are captured at six architectural
stages (raw → after-LN → after-hidden → final projections → head
penultimate → score). For each stage we fit a 3-fold logistic regression
on the concat embedding and report the held-out ROC-AUC of predicting the
binary binder/non-binder label.

The diagnostic the figure makes obvious:
  - The BCE model's pair-classification AUC stays high or *rises* through
    the architecture: the model is learning the per-protein label-prior
    shortcut, exactly as Phase-1 baselines did.
  - The RankBind model's pair-classification AUC starts equally high in
    the raw embedding (BRENDA's decoy construction leaks pair-level
    statistics into ChemBERTa+ESM2 alone) but then *drops* through the
    architecture — the recipe is actively removing that shortcut signal
    in favour of ligand-conditional ranking.

Run on a single A30 GPU: ≈30 s per model, ≈1 min total.

Outputs:
  paper/figures/fig_pair_classification_compare.png
  evaluation/attractor_results/pair_classification_compare.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v5_rankbind.data import build_datasets, collate_pointwise  # noqa: E402
from v5_rankbind.model import RankBind  # noqa: E402


GOLD_RUN = (
    "results/v5_rankbind/20260427-121212_1746525d51_abl_attn_pool_v5b_s7"
)
BEFORE_RUN = (
    "results/v5_rankbind/20260423-135706_9ee7fdbfbc_abl_bce_only_v4_s7"
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def capture_pair_intermediates(
    model: RankBind, batch: dict, device: torch.device,
) -> dict[str, np.ndarray]:
    """Forward one batch and extract pair-level concat representations at
    six architectural milestones, plus the binary label vector."""
    lig_in = batch["lig_emb"].to(device)
    prot_in = batch["prot_emb"].to(device)
    prot_mask = batch.get("prot_mask")
    if prot_mask is not None:
        prot_mask = prot_mask.to(device)
    labels = batch["label"].cpu().numpy().astype(int)

    # Stage 0 — raw inputs (mean-pool the protein side if attn-pool model)
    if prot_in.dim() == 3:
        if prot_mask is not None:
            denom = prot_mask.sum(dim=1, keepdim=True).clamp_min(1)
            P_raw = (prot_in * prot_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            P_raw = prot_in.mean(dim=1)
    else:
        P_raw = prot_in
    raw_inputs = torch.cat([lig_in, P_raw], dim=-1)

    # Replay LigandProjector layer-by-layer.
    ln = model.lig.net
    L_ln = ln[0](lig_in)
    L_h = ln[1](L_ln); L_h = ln[2](L_h); L_h = ln[3](L_h)
    fL = ln[4](L_h)

    # Replay protein path. attn-pool happens BEFORE the projector when present.
    if model.attn_pool is not None:
        prot_pooled, _ = model.attn_pool(prot_in, prot_mask)
    else:
        prot_pooled = prot_in
    pn = model.prot.net
    P_ln = pn[0](prot_pooled)
    P_h = pn[1](P_ln); P_h = pn[2](P_h); P_h = pn[3](P_h)
    gP = pn[4](P_h)

    out = {
        "raw_inputs":   raw_inputs.cpu().numpy(),
        "after_LN":     torch.cat([L_ln, P_ln], dim=-1).cpu().numpy(),
        "after_hidden": torch.cat([L_h, P_h], dim=-1).cpu().numpy(),
        "proj_concat":  torch.cat([fL, gP], dim=-1).cpu().numpy(),
        "labels":       labels,
    }

    # Stage 4 (head penultimate) + Stage 5 (score) — branch by head type so
    # the comparison stays consistent across heads.
    if model.head_type == "bilinear":
        head = model.head
        low = (fL @ head.U) * (gP @ head.V)
        out["head_penultimate"] = low.cpu().numpy()
        out["score"] = (low.sum(dim=-1)
                        + (fL * head.d * gP).sum(dim=-1)
                        + head.b).cpu().numpy()
    else:  # mlp_concat
        x = torch.cat([fL, gP], dim=-1)
        h = model.head.net[0](x); h = model.head.net[1](h); h = model.head.net[2](h)
        out["head_penultimate"] = h.cpu().numpy()
        out["score"] = model.head.net[3](h).squeeze(-1).cpu().numpy()
    return out


def linear_probe_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 3,
                     seed: int = 42) -> float:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    return float(np.mean(aucs))


def run_model(run_dir: Path, n_pairs: int, device: torch.device,
              chemberta_cache_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    if not (run_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"No best_model.pt under {run_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config_resolved"]

    train_ds, val_ds, test_ds, _ = build_datasets(cfg, chemberta_cache_dir)
    n = min(n_pairs, len(test_ds))
    items = [test_ds[i] for i in range(n)]
    batch = collate_pointwise(items)

    model = RankBind(cfg).to(device).eval()
    sd = torch.load(run_dir / "best_model.pt", map_location=device,
                    weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    captured = capture_pair_intermediates(model, batch, device)
    return captured, manifest


STAGES = [
    ("raw_inputs",       "0. Raw\nChemBERTa ⊕ ESM2"),
    ("after_LN",         "1. After LayerNorm\n(both projectors)"),
    ("after_hidden",     "2. After Linear+GELU\n(hidden 256)"),
    ("proj_concat",      "3. fL ⊕ gP\n(final projections)"),
    ("head_penultimate", "4. Head penultimate\n(bilinear-low / MLP-h)"),
    ("score",            "5. Score\n(model output)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_run",   default=GOLD_RUN,
                    help="run_dir of the finished RankBind gold standard")
    ap.add_argument("--before_run", default=BEFORE_RUN,
                    help="run_dir of the pre-RankBind ablation "
                         "(default: abl_bce_only_v4_s7 — the in-package "
                         "re-creation of the Phase-1 shortcut pathology)")
    ap.add_argument("--n_pairs", type=int, default=2000)
    ap.add_argument(
        "--out_fig",
        default="paper/figures/fig_pair_classification_compare.png",
    )
    ap.add_argument(
        "--out_csv",
        default="evaluation/attractor_results/pair_classification_compare.csv",
    )
    args = ap.parse_args()

    device = _device()
    chemberta_cache_dir = PROJECT_ROOT / "data" / "chemberta_cache"
    print(f"[probe] device     = {device}")
    print(f"[probe] gold       = {args.gold_run}")
    print(f"[probe] before     = {args.before_run}")

    rows = []
    captured_all = {}
    for name, rel in [("before", args.before_run), ("gold", args.gold_run)]:
        run_dir = (PROJECT_ROOT / rel).resolve()
        captured, manifest = run_model(
            run_dir, args.n_pairs, device, chemberta_cache_dir,
        )
        cfg_label = manifest["config_resolved"].get("name", run_dir.name)
        seed = manifest["config_resolved"].get("seed")
        n_pos = int((captured["labels"] == 1).sum())
        n_pairs = len(captured["labels"])
        print(f"[{name:>6}] {cfg_label} seed={seed}  pairs={n_pairs}  pos={n_pos}")

        for stage_key, _ in STAGES:
            X = captured[stage_key]
            auc = linear_probe_auc(X, captured["labels"])
            d = X.shape[1] if X.ndim > 1 else 1
            rows.append({
                "model":            name,
                "model_label":      cfg_label,
                "seed":             seed,
                "stage":            stage_key,
                "n_dim":            int(d),
                "linear_probe_auc": auc,
                "test_global_auc":  manifest["metrics"].get("test_global_auc"),
                "matrix_mrr":       manifest["metrics"].get("matrix_mrr"),
            })
            print(f"          {stage_key:18s} d={d:>4}  "
                  f"linear_probe_AUC={auc:.3f}")
        captured_all[name] = (captured, manifest)

    df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[probe] saved CSV  = {out_csv}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, (ax_curve, ax_score) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2, 1]},
    )

    stage_keys = [s for s, _ in STAGES]
    stage_labels = [l for _, l in STAGES]
    x = np.arange(len(stage_keys))
    colors = {"before": "#d62728", "gold": "#1f77b4"}
    markers = {"before": "s", "gold": "o"}

    for name in ("before", "gold"):
        sub = df[df["model"] == name].set_index("stage").loc[stage_keys]
        cfg_label = sub.iloc[0]["model_label"]
        seed = sub.iloc[0]["seed"]
        gauc = sub.iloc[0]["test_global_auc"]
        mrr = sub.iloc[0]["matrix_mrr"]
        legend = (f"{name.upper()}: {cfg_label} (seed {seed})\n"
                  f"  test gAUC={gauc:.3f}  matrix MRR={mrr:.3f}")
        ax_curve.plot(x, sub["linear_probe_auc"].values,
                      marker=markers[name], markersize=11, linewidth=2.4,
                      color=colors[name], label=legend)
        for xi, v in zip(x, sub["linear_probe_auc"].values):
            ax_curve.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 9),
                              textcoords="offset points",
                              ha="center", fontsize=8, color=colors[name])

    ax_curve.set_xticks(x); ax_curve.set_xticklabels(stage_labels, fontsize=9)
    ax_curve.set_ylim(0.45, 1.02)
    ax_curve.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    ax_curve.text(len(stage_keys) - 0.5, 0.51, "chance", color="grey",
                  fontsize=8, ha="right")
    ax_curve.set_ylabel("Pair-level binary classification AUC\n"
                        "(logistic regression, 3-fold CV)")
    ax_curve.set_title("How pair-level shortcut signal is processed by the architecture")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # Right panel: score histogram for both models, on the same x-axis.
    for name in ("before", "gold"):
        captured = captured_all[name][0]
        labels = captured["labels"]
        scores = captured["score"]
        s_norm = (scores - scores.mean()) / (scores.std() + 1e-9)
        for lab, color, alpha, lbl in [
            (0, colors[name], 0.30, f"{name} non-binder"),
            (1, colors[name], 0.65, f"{name} binder"),
        ]:
            mask = labels == lab
            ax_score.hist(
                s_norm[mask], bins=40, density=True,
                color=color, alpha=alpha,
                histtype="stepfilled" if lab == 1 else "step",
                linewidth=1.4 if lab == 0 else 0.0,
                label=lbl,
            )
    ax_score.set_xlabel("score (z-normalised within model)")
    ax_score.set_ylabel("density")
    ax_score.set_title("Score distributions\n(z-normalised; binder ⌒ non-binder)")
    ax_score.grid(alpha=0.3)
    ax_score.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Pair-level classification through the architecture: "
        "shortcut-fitting vs shortcut-avoidant",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_fig = PROJECT_ROOT / args.out_fig
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"[probe] saved fig  = {out_fig}")


if __name__ == "__main__":
    main()
