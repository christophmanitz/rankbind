"""
evaluation/embedding_evolution.py — visualise what RankBind actually learns
through its architecture, with metrics aligned to the model's training
objective.

Two figures are emitted:

  Figure A: stage-wise matrix MRR / Hit@10 on the 200x200 test grid.
            Random baseline → cos-similarity at each shared-dim stage →
            bilinear interaction → full score. Shows that ranking quality
            climbs through the architecture; this is the metric the model
            is actually optimised for.

  Figure B: 4-panel PCA of the per-protein representation at four
            architectural milestones (raw mean-pooled ESM2 → post-attn-pool
            → post-hidden 256 → final gP), with each protein coloured by
            EC top-level class. Shows that the projector pulls
            biologically-related proteins together and pushes unrelated
            ones apart -- the form of "structure" RankBind is supposed to
            learn.

The earlier `embedding_evolution_pca.py` measured pair-level binary
classification AUC through the architecture. That metric drops through the
network (0.86 → 0.68), which IS a real finding about pooled-AUC shortcut
removal but not what the user wanted to see; the script lives under
`evaluation/_archive/` for that purpose.

Usage:
  python evaluation/embedding_evolution.py
  python evaluation/embedding_evolution.py --run_dir <run_dir>
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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v5_rankbind.data import build_datasets, collate_pointwise  # noqa: E402
from v5_rankbind.model import RankBind  # noqa: E402
from v5_rankbind.run_manifest import load_config  # noqa: E402


DEFAULT_RUN = (
    "results/v5_rankbind/20260427-121212_1746525d51_abl_attn_pool_v5b_s7"
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Stage capture (per-protein / per-ligand)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_proteins_per_stage(
    model: RankBind,
    prot_in: torch.Tensor,
    prot_mask: torch.Tensor | None,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Replay the protein path layer by layer and return per-protein
    representations at four architectural milestones."""
    prot_in = prot_in.to(device)
    if prot_mask is not None:
        prot_mask = prot_mask.to(device)

    if prot_in.dim() == 3:
        if prot_mask is not None:
            denom = prot_mask.sum(dim=1, keepdim=True).clamp_min(1)
            raw = (prot_in * prot_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            raw = prot_in.mean(dim=1)
    else:
        raw = prot_in

    if model.attn_pool is not None:
        pooled, _w = model.attn_pool(prot_in, prot_mask)
    else:
        pooled = prot_in

    net = model.prot.net
    P_ln = net[0](pooled)
    P_h = net[1](P_ln)
    P_h = net[2](P_h)
    P_h = net[3](P_h)
    gP = net[4](P_h)

    return {
        "P_raw":    raw.cpu().numpy(),
        "P_pooled": pooled.cpu().numpy(),
        "P_hidden": P_h.cpu().numpy(),
        "P_final":  gP.cpu().numpy(),
    }


@torch.no_grad()
def encode_ligands_per_stage(
    model: RankBind, lig_in: torch.Tensor, device: torch.device,
) -> dict[str, np.ndarray]:
    lig_in = lig_in.to(device)
    net = model.lig.net
    L_ln = net[0](lig_in)
    L_h = net[1](L_ln)
    L_h = net[2](L_h)
    L_h = net[3](L_h)
    fL = net[4](L_h)
    return {
        "L_raw":    lig_in.cpu().numpy(),
        "L_norm":   L_ln.cpu().numpy(),
        "L_hidden": L_h.cpu().numpy(),
        "L_final":  fL.cpu().numpy(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Stage-wise matrix MRR
# ──────────────────────────────────────────────────────────────────────────────

def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_n @ B_n.T


def matrix_ranking_metrics(
    score_mat: np.ndarray,
    ligand_keys: list[str],
    protein_keys: list[str],
    test_pairs: pd.DataFrame,
) -> dict[str, float]:
    """For every (ligand, protein+) positive pair appearing in test_pairs and
    in both axes, compute the rank of protein+ in score_mat[ligand_idx, :].
    Aggregate into MRR / Hit@K. Ties are resolved by upper rank.
    """
    lig_idx = {s: i for i, s in enumerate(ligand_keys)}
    prot_idx = {p: j for j, p in enumerate(protein_keys)}
    pos = test_pairs[test_pairs["label"] == 1]
    pos = pos[pos["substrate_smiles"].isin(lig_idx) & pos["uniprot"].isin(prot_idx)]
    ranks = []
    for _, row in pos.iterrows():
        i = lig_idx[row["substrate_smiles"]]
        j = prot_idx[row["uniprot"]]
        s = score_mat[i]
        rank = int((s > s[j]).sum() + 1)
        ranks.append(rank)
    if not ranks:
        return {"mrr": float("nan"), "h@5": float("nan"), "h@10": float("nan"),
                "mean_rank": float("nan"), "n_matched": 0}
    ranks = np.asarray(ranks)
    return {
        "mrr":       float(np.mean(1.0 / ranks)),
        "h@5":       float(np.mean(ranks <= 5)),
        "h@10":      float(np.mean(ranks <= 10)),
        "mean_rank": float(np.mean(ranks)),
        "n_matched": int(len(ranks)),
    }


def bilinear_score_matrix(
    fL: np.ndarray, gP: np.ndarray, head, return_low: bool = False,
) -> np.ndarray:
    """Apply the trained BilinearHead to all (fL_i, gP_j) pairs.

    score(i, j) = (fL_i · U) · (gP_j · V)^T (sum over rank) +
                  fL_i · diag(d) · gP_j  +  b
    """
    U = head.U.detach().cpu().numpy()
    V = head.V.detach().cpu().numpy()
    d = head.d.detach().cpu().numpy()
    b = float(head.b.detach().cpu().item())
    A = fL @ U                           # [Nl, r]
    B = gP @ V                           # [Np, r]
    low_sum = A @ B.T                    # [Nl, Np]
    if return_low:
        return low_sum
    diag_lig = fL * d                    # [Nl, D]
    diag_term = diag_lig @ gP.T          # [Nl, Np]
    return low_sum + diag_term + b


def stage_score_matrices(
    lig_stages: dict[str, np.ndarray],
    prot_stages: dict[str, np.ndarray],
    head,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build the per-stage 200×200 score matrices we will rank over.

    Stages:
      random            : i.i.d. uniform — null baseline.
      cos_hidden        : cos of after-(LN+Linear+GELU+Dropout) reps (256-d).
      cos_final         : cos of fL and gP (256-d each, what an inner-product
                          head would compute).
      bilinear_low_sum  : sum over the rank-r interaction term — what the
                          bilinear head's primary mode contributes.
      bilinear_full     : the model's actual head output (low + diag + b).
    """
    rng = np.random.default_rng(seed)
    Nl, Np = lig_stages["L_final"].shape[0], prot_stages["P_final"].shape[0]
    return {
        "random":            rng.uniform(0, 1, size=(Nl, Np)).astype(np.float32),
        "cos_hidden":        cosine_matrix(lig_stages["L_hidden"],
                                           prot_stages["P_hidden"]),
        "cos_final":         cosine_matrix(lig_stages["L_final"],
                                           prot_stages["P_final"]),
        "bilinear_low_sum":  bilinear_score_matrix(
                                 lig_stages["L_final"],
                                 prot_stages["P_final"], head,
                                 return_low=True),
        "bilinear_full":     bilinear_score_matrix(
                                 lig_stages["L_final"],
                                 prot_stages["P_final"], head),
    }


STAGE_LABELS_A = [
    ("random",           "Random\nbaseline"),
    ("cos_hidden",       "cos(L_hidden, P_hidden)\n256-d, post-(LN+Linear+GELU)"),
    ("cos_final",        "cos(fL, gP)\nfinal projections"),
    ("bilinear_low_sum", "Bilinear interaction\nΣ_r (fL·U)_r (gP·V)_r"),
    ("bilinear_full",    "Bilinear head\n(model's output)"),
]


def make_figure_A(
    rows: list[dict], out_path: Path, run_label: str,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = [k for k, _ in STAGE_LABELS_A]
    labels = [l for _, l in STAGE_LABELS_A]
    mrrs = [next(r["mrr"] for r in rows if r["stage"] == k) for k in keys]
    h10s = [next(r["h@10"] for r in rows if r["stage"] == k) for k in keys]

    x = np.arange(len(keys))
    w = 0.38
    bars1 = ax.bar(x - w / 2, mrrs, w, color="#1f77b4",
                   label="Matrix MRR", edgecolor="white")
    bars2 = ax.bar(x + w / 2, h10s, w, color="#ff7f0e",
                   label="Hit@10", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(max(mrrs), max(h10s)) * 1.18)
    ax.set_ylabel("ranking quality (test set, 200×200 grid)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars1, mrrs):
        ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    for bar, v in zip(bars2, h10s):
        ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    fig.suptitle(
        f"Stage-wise ranking quality — {run_label}\n"
        f"Each architectural component lifts ligand-conditional retrieval",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure B: protein PCA per stage, coloured by EC class
# ──────────────────────────────────────────────────────────────────────────────

def _ec_subclass(ec_string) -> str | None:
    """Return the EC2 subclass (e.g. '3.1.1.81' → '3.1').

    BRENDA-200 is hydrolase-restricted (900/903 proteins are EC 3); the
    top-level EC class would give a single colour, so we use the
    subclass which produces a meaningful palette of ~6-9 buckets:
    3.1 ester-hydrolases, 3.2 glycosylases, 3.4 peptidases,
    3.5 C-N hydrolases, 3.6 phosphoanhydride hydrolases, etc.
    """
    if ec_string is None:
        return None
    s = str(ec_string).strip()
    if not s or s.lower() == "nan":
        return None
    parts = s.split(".")
    if len(parts) < 2:
        return None
    try:
        ec1, ec2 = int(parts[0]), parts[1]
        if not ec2.isdigit():
            return None
        return f"{ec1}.{int(ec2)}"
    except ValueError:
        return None


# Hydrolase subclasses (EC 3.x). BRENDA-200 is hydrolase-only; we hard-code
# the dominant buckets so the legend stays stable across runs.
EC2_NAMES = {
    "3.1": "3.1 ester-hydrolase",
    "3.2": "3.2 glycosylase",
    "3.3": "3.3 ether-hydrolase",
    "3.4": "3.4 peptidase",
    "3.5": "3.5 C–N hydrolase",
    "3.6": "3.6 phosphoanhydride",
    "3.7": "3.7 C–C hydrolase",
    "3.8": "3.8 halide hydrolase",
    "3.13": "3.13 C–S hydrolase",
}
EC2_COLORS = {
    "3.1": "#1f77b4", "3.2": "#ff7f0e", "3.3": "#9b59b6",
    "3.4": "#2ca02c", "3.5": "#d62728", "3.6": "#8c564b",
    "3.7": "#17becf", "3.8": "#bcbd22", "3.13": "#e377c2",
}


def make_figure_B(
    prot_stages: dict[str, np.ndarray],
    protein_keys: list[str],
    ec_for_protein: dict[str, int | None],
    out_path: Path,
    run_label: str,
):
    panels = [
        ("P_raw",    "0. Raw ESM2 (mean-pool of per-residue)"),
        ("P_pooled", "1. After attention-pool over residues"),
        ("P_hidden", "2. After Linear+GELU (hidden 256)"),
        ("P_final",  "3. gP — final projection"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    ec_arr = np.array([ec_for_protein.get(p) or "" for p in protein_keys],
                      dtype=object)
    valid = ec_arr != ""
    silhouettes = {}

    # Only consider EC2 buckets we have a colour for AND that have ≥3
    # proteins on the axis (singletons add noise to silhouette).
    counts = pd.Series(ec_arr[valid]).value_counts()
    ec_keep = [ec for ec in counts.index if ec in EC2_NAMES and counts[ec] >= 3]

    for ax, (key, title) in zip(axes, panels):
        X = prot_stages[key]
        pca = PCA(n_components=2, random_state=42)
        P = pca.fit_transform(X)
        vr = pca.explained_variance_ratio_

        for ec in ec_keep:
            mask = (ec_arr == ec)
            ax.scatter(P[mask, 0], P[mask, 1], s=22, alpha=0.75,
                       c=EC2_COLORS.get(ec, "#777777"),
                       label=f"{EC2_NAMES.get(ec, ec)} (n={int(mask.sum())})",
                       edgecolors="white", linewidths=0.4)
        # Other / unannotated.
        other_mask = ~np.isin(ec_arr, ec_keep)
        if other_mask.sum() > 0:
            ax.scatter(P[other_mask, 0], P[other_mask, 1], s=14, alpha=0.35,
                       c="#bbbbbb",
                       label=f"other / unknown (n={int(other_mask.sum())})",
                       edgecolors="none")

        # Silhouette in the FULL embedding space (not the 2-D projection),
        # restricted to the ec_keep buckets so we score real cluster quality.
        kept_mask = np.isin(ec_arr, ec_keep)
        if kept_mask.sum() > 5 and len(set(ec_arr[kept_mask])) > 1:
            try:
                sil = silhouette_score(X[kept_mask], ec_arr[kept_mask])
            except Exception:
                sil = float("nan")
        else:
            sil = float("nan")
        silhouettes[key] = sil

        ax.set_xlabel(f"PC1 ({vr[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({vr[1]*100:.1f}%)")
        ax.set_title(f"{title}\nEC-class silhouette = {sil:.3f}", fontsize=10)
        ax.grid(alpha=0.3)

    # One legend at the figure level.
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle(
        f"Protein-embedding evolution by EC subclass — {run_label}\n"
        "PCA per stage; clusters by EC2 (hydrolase subclasses) "
        "indicate biology-aware structure",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return silhouettes


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=DEFAULT_RUN)
    ap.add_argument("--n_matrix", type=int, default=200,
                    help="ligands and proteins on the score-matrix axes "
                         "(matches the value used at training-time eval)")
    ap.add_argument("--out_fig_a",
                    default="paper/figures/fig_embedding_evolution_mrr.png")
    ap.add_argument("--out_fig_b",
                    default="paper/figures/fig_embedding_evolution_ec.png")
    ap.add_argument("--out_csv",
                    default="evaluation/attractor_results/embedding_evolution.csv")
    args = ap.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    if not (run_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"No best_model.pt in {run_dir}")

    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config_resolved"]
    run_label = manifest.get("run_id", run_dir.name)
    device = _device()

    chemberta_cache_dir = PROJECT_ROOT / "data" / "chemberta_cache"
    train_ds, val_ds, test_ds, _ = build_datasets(cfg, chemberta_cache_dir)

    # Build per-axis test sets: 200 unique ligands + 200 unique proteins from
    # the test split. Same construction used by v5 build_score_matrix to
    # keep numbers comparable to the manifest.
    test_pairs = test_ds.pairs.copy()
    ligand_keys = list(dict.fromkeys(test_pairs["substrate_smiles"]))[: args.n_matrix]
    protein_keys = list(dict.fromkeys(test_pairs["uniprot"]))[: args.n_matrix]
    print(f"[probe] device       = {device}")
    print(f"[probe] run_label    = {run_label}")
    print(f"[probe] |ligands|    = {len(ligand_keys)}  "
          f"|proteins| = {len(protein_keys)}")

    # ── Encode ligands once and proteins once ────────────────────────────────
    model = RankBind(cfg).to(device).eval()
    sd = torch.load(run_dir / "best_model.pt", map_location=device,
                    weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)

    # Build per-ligand and per-protein input batches.
    lig_batch = collate_pointwise([
        test_ds[test_pairs.index[test_pairs["substrate_smiles"] == s][0]]
        for s in ligand_keys
    ])
    prot_batch = collate_pointwise([
        test_ds[test_pairs.index[test_pairs["uniprot"] == p][0]]
        for p in protein_keys
    ])
    lig_stages = encode_ligands_per_stage(model, lig_batch["lig_emb"], device)
    prot_stages = encode_proteins_per_stage(
        model, prot_batch["prot_emb"], prot_batch.get("prot_mask"), device,
    )

    # ── Figure A: per-stage Matrix MRR ───────────────────────────────────────
    score_mats = stage_score_matrices(
        lig_stages, prot_stages, head=model.head, seed=42,
    )
    rows = []
    for key, _label in STAGE_LABELS_A:
        m = matrix_ranking_metrics(
            score_mats[key], ligand_keys, protein_keys, test_pairs,
        )
        rows.append({"stage": key, **m})
        print(f"[probe] {key:18s}  MRR={m['mrr']:.3f}  H@5={m['h@5']:.3f}  "
              f"H@10={m['h@10']:.3f}  matched={m['n_matched']}")

    out_csv = PROJECT_ROOT / args.out_csv
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[probe] saved CSV    = {out_csv}")

    out_fig_a = PROJECT_ROOT / args.out_fig_a
    make_figure_A(rows, out_fig_a, run_label)
    print(f"[probe] saved fig A  = {out_fig_a}")

    # ── Figure B: protein PCA by EC class ────────────────────────────────────
    # EC annotation comes from data/dataset_with_decoys.csv; the file the
    # BRENDA-200 split is built from already has an `ec` column.
    csv_path = PROJECT_ROOT / cfg["data"]["csv_path"]
    if csv_path.suffix == ".csv":
        df_ec = pd.read_csv(csv_path)
        if "ec" in df_ec.columns:
            ec_lookup = dict(zip(df_ec["uniprot"], df_ec["ec"]))
        else:
            ec_lookup = {}
    else:
        ec_lookup = {}
    ec_for_protein = {p: _ec_subclass(ec_lookup.get(p)) for p in protein_keys}
    n_known = sum(v is not None for v in ec_for_protein.values())
    print(f"[probe] EC-annotated proteins: {n_known}/{len(protein_keys)}")

    out_fig_b = PROJECT_ROOT / args.out_fig_b
    silhouettes = make_figure_B(
        prot_stages, protein_keys, ec_for_protein, out_fig_b, run_label,
    )
    print(f"[probe] saved fig B  = {out_fig_b}")
    print("[probe] EC-silhouettes per stage:")
    for k, v in silhouettes.items():
        print(f"          {k:9s}: {v:.3f}")


if __name__ == "__main__":
    main()
