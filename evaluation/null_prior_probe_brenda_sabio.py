"""
evaluation/null_prior_probe_brenda_sabio.py — Null-prior probe for the three
v5_rankbind BRENDA+SABIO with_decoys runs (kcat_km / km / turnover).

For each run, builds three null score matrices over the SAME 200×200 geometry
the model evaluated on:

  random      — uniform noise
  prot_prior  — score[i, j] = per-protein training positive rate
                (model that ignores ligand entirely; only knows protein base rate)
  lig_prior   — score[i, j] = per-ligand training positive rate

Then quantifies how close the v5 score matrix is to each null:

  Spearman(model_row, null_row), averaged over ligand rows
  Top-K Jaccard (k=10) per ligand, averaged
  Per-ligand AUC of each null on the test split
  Matrix MRR / Hit@K of each null on the same test split

If model ≈ prot_prior in row-Spearman, the model has just learned the
per-protein training base rate — Phase-1 shortcut, again.

Output:
  evaluation/attractor_results/null_prior_probe_brenda_sabio.csv
  evaluation/attractor_results/null_prior_probe_brenda_sabio.txt
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baselines", "adapters"))

from common import BRENDADataConfig  # noqa: E402

RUNS = [
    ("kcat_km",
     "results/v5_rankbind/20260503-112034_1746525d51_kcat_km_with_decoys_bs_v1"),
    ("km",
     "results/v5_rankbind/20260503-112035_1746525d51_km_with_decoys_bs_v1"),
    ("turnover",
     "results/v5_rankbind/20260503-112035_1746525d51_turnover_with_decoys_bs_v1"),
]


def load_run(run_dir: str):
    M = np.load(os.path.join(run_dir, "score_matrix_rankbind.npy"))
    axes = json.load(open(os.path.join(run_dir, "score_matrix_axes.json")))
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    return M, axes, manifest


def build_nulls(train_df: pd.DataFrame, ligands: list, proteins: list,
                seed: int = 42):
    n_lig, n_prot = len(ligands), len(proteins)
    global_rate = float(train_df["label"].mean())
    prot_rate = train_df.groupby("uniprot")["label"].mean()
    lig_rate = train_df.groupby("substrate_smiles")["label"].mean()

    prot_vec = np.array([prot_rate.get(p, global_rate) for p in proteins],
                        dtype=np.float32)
    lig_vec = np.array([lig_rate.get(s, global_rate) for s in ligands],
                       dtype=np.float32)

    rng = np.random.default_rng(seed)
    return {
        "null_random":     rng.uniform(0, 1, size=(n_lig, n_prot)).astype(np.float32),
        "null_prot_prior": np.broadcast_to(prot_vec, (n_lig, n_prot)).copy(),
        "null_lig_prior":  np.broadcast_to(lig_vec[:, None], (n_lig, n_prot)).copy(),
    }, prot_vec, lig_vec


def row_spearman(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """Mean ± std of per-row Spearman between matrices of identical shape.

    A row with zero variance in either A or B is skipped (Spearman undefined).
    """
    rs = []
    for i in range(A.shape[0]):
        if A[i].std() < 1e-12 or B[i].std() < 1e-12:
            continue
        r, _ = spearmanr(A[i], B[i])
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return float("nan"), float("nan")
    return float(np.mean(rs)), float(np.std(rs))


def topk_row_jaccard(A: np.ndarray, B: np.ndarray, k: int = 10) -> float:
    js = []
    for i in range(A.shape[0]):
        a_top = set(np.argsort(-A[i])[:k])
        b_top = set(np.argsort(-B[i])[:k])
        u = a_top | b_top
        if not u:
            continue
        js.append(len(a_top & b_top) / len(u))
    return float(np.mean(js)) if js else float("nan")


def per_ligand_auc(M: np.ndarray, ligands: list, proteins: list,
                   test_df: pd.DataFrame) -> tuple[float, int]:
    """For each ligand row, compute AUC of M[i, :] vs binary labels from
    test_df entries that hit ligand=ligands[i] and uniprot ∈ proteins.

    Returns (mean_auc_over_eligible_ligands, n_eligible).
    """
    prot_idx = {p: j for j, p in enumerate(proteins)}
    aucs = []
    for i, lig in enumerate(ligands):
        sub = test_df[test_df["substrate_smiles"] == lig]
        if sub.empty:
            continue
        # only proteins that appear on the matrix axis
        sub = sub[sub["uniprot"].isin(prot_idx)]
        if sub["label"].nunique() < 2:
            continue
        scores = sub["uniprot"].map(lambda u: M[i, prot_idx[u]]).values
        labels = sub["label"].values.astype(int)
        try:
            aucs.append(roc_auc_score(labels, scores))
        except Exception:
            continue
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def matrix_ranking(M: np.ndarray, ligands: list, proteins: list,
                   test_df: pd.DataFrame) -> dict:
    """Same matrix-MRR/Hit@K used by v5 eval: for each (lig, prot) positive
    pair in test_df, the rank of `prot` among all proteins for that ligand row.
    """
    prot_idx = {p: j for j, p in enumerate(proteins)}
    lig_idx = {s: i for i, s in enumerate(ligands)}
    test_pos = test_df[test_df["label"] == 1]
    test_pos = test_pos[test_pos["substrate_smiles"].isin(lig_idx)
                        & test_pos["uniprot"].isin(prot_idx)]
    ranks = []
    for _, row in test_pos.iterrows():
        i = lig_idx[row["substrate_smiles"]]; j = prot_idx[row["uniprot"]]
        scores = M[i]
        rank = (scores > scores[j]).sum() + 1  # 1-indexed; ties → upper rank
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


def main():
    out_dir = os.path.join(_HERE, "attractor_results")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    text_lines = []

    for tag, rel_dir in RUNS:
        run_dir = os.path.join(PROJECT_ROOT, rel_dir)
        M, axes, manifest = load_run(run_dir)
        ligands = axes["axis_0_ligands"]
        proteins = axes["axis_1_proteins"]

        cfg = manifest["config_resolved"]
        bconfig = BRENDADataConfig(
            seed=cfg["seed"],
            csv_path=str(os.path.join(PROJECT_ROOT, cfg["data"]["csv_path"])),
            seq_csv=str(os.path.join(PROJECT_ROOT, cfg["data"]["seq_csv"])),
            val_frac=cfg["data"]["val_frac"],
            test_frac=cfg["data"]["test_frac"],
        )
        pairs = bconfig.load_pairs()
        # Same filter the v5 dataloader applies: only keep proteins with
        # sequences AND ESM2 embeddings, so prior rates are computed on the
        # same population the model trained on.
        seqs = bconfig.load_sequences()
        esm2_dir = os.path.join(PROJECT_ROOT, cfg["data"]["esm2_dir"])
        have_esm = {os.path.splitext(f)[0]
                    for f in os.listdir(esm2_dir) if f.endswith(".pt")}
        keep = pairs["uniprot"].isin(seqs) & pairs["uniprot"].isin(have_esm)
        pairs = pairs[keep].reset_index(drop=True)
        train_idx, _, test_idx = bconfig.get_protein_split()
        train_df = pairs[pairs["idx"].isin(set(train_idx))].reset_index(drop=True)
        test_df = pairs[pairs["idx"].isin(set(test_idx))].reset_index(drop=True)

        nulls, prot_vec, lig_vec = build_nulls(
            train_df, ligands, proteins, seed=cfg["seed"]
        )

        # Save null matrices for downstream re-use
        for name, NM in nulls.items():
            np.save(os.path.join(out_dir, f"score_matrix_{tag}_{name}.npy"), NM)

        # Comparisons: model vs each null
        text_lines.append(f"=== {tag} ===")
        text_lines.append(
            f"  model: range=[{M.min():.2f},{M.max():.2f}] "
            f"prot-axis-coverage in train: "
            f"{int(np.isin(proteins, train_df['uniprot'].unique()).sum())}/{len(proteins)}"
        )
        # Per-ligand AUC + matrix-ranking on test split, for the model AND each null
        all_matrices = {"model_v5": M, **nulls}
        for name, NM in all_matrices.items():
            pl_auc, pl_n = per_ligand_auc(NM, ligands, proteins, test_df)
            mr = matrix_ranking(NM, ligands, proteins, test_df)
            row = {
                "dataset": tag, "matrix": name,
                "per_lig_auc": pl_auc, "n_lig_eligible": pl_n,
                **{f"matrix_{k}": v for k, v in mr.items()},
            }
            # Closeness to model (only for nulls)
            if name != "model_v5":
                rs_mean, rs_std = row_spearman(M, NM)
                row["row_spearman_vs_model_mean"] = rs_mean
                row["row_spearman_vs_model_std"] = rs_std
                row["topk_jaccard@10_vs_model"] = topk_row_jaccard(M, NM, k=10)
            rows.append(row)
            text_lines.append(
                f"  {name:18s}  "
                f"per_lig_AUC={pl_auc:.3f} (n={pl_n:>3})  "
                f"MRR={mr['mrr']:.3f}  H@5={mr['h@5']:.3f}  H@10={mr['h@10']:.3f}  "
                f"matched={mr['n_matched']}"
            )
            if name != "model_v5":
                text_lines[-1] += (
                    f"  | rho_row={row['row_spearman_vs_model_mean']:+.3f} "
                    f"±{row['row_spearman_vs_model_std']:.3f}  "
                    f"jac@10={row['topk_jaccard@10_vs_model']:.3f}"
                )
        text_lines.append("")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "null_prior_probe_brenda_sabio.csv")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(out_dir, "null_prior_probe_brenda_sabio.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(text_lines))

    print("\n".join(text_lines))
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved TXT: {txt_path}")


if __name__ == "__main__":
    main()
