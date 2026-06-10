"""
evaluation/benchmark_null_eval.py — §8.3 null-baseline probe for the external
benchmarks (Davis / KIBA / BindingDB / ESP).

The Phase-1 tool ``evaluation/null_baselines.py`` is hard-wired to BRENDA
(``BRENDADataConfig`` defaults) and the four Phase-1 model matrices. This is
the generalised version: given a *finished* v5_rankbind benchmark run dir, it
rebuilds the three null score matrices over **exactly** the same pool of
proteins/ligands that ``v5_rankbind/eval.py::build_score_matrix`` used (the
first ``n_matrix`` entries of ``seqs.keys()`` and the first ``n_matrix`` unique
SMILES), then reports — for RankBind and each null — the §8.3 instruments
pre-registered in ``docs/BENCHMARK_INTEGRATION_PLAN.md``:

  * matrix-level MRR / Hit@k         (ligand-conditional ranking)
  * Gini over attractor scores       (data-geometry descriptor)
  * Gini-residual = gini(model) − gini(null_prot_prior)   (negative = good)
  * Top-10 attractor Jaccard vs null_prot_prior            (low = shortcut-avoidant)

Ground truth for the positive (ligand, protein) pairs is read straight from
the run's own ``test_preds_rankbind.csv`` (label == 1), i.e. the identical set
``eval.py`` used — so the RankBind matrix MRR recomputed here must match the
saved ``test_matrix_ranking.json``. That equality is asserted as a built-in
correctness check; a mismatch means the pool/positives reconstruction drifted
and the null numbers cannot be trusted.

Usage:
  python evaluation/benchmark_null_eval.py RUN_DIR [RUN_DIR ...] \
      --out evaluation/attractor_results/benchmark_null_summary.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "baselines", "adapters"))
sys.path.insert(0, PROJECT_ROOT)

from common import BRENDADataConfig                       # noqa: E402
from attractor_diagnosis import compute_attractor_metrics  # noqa: E402
from v5_rankbind.metrics import matrix_ranking_metrics      # noqa: E402


def _resolve(path: str) -> str:
    """Resolve a (possibly project-relative) path against PROJECT_ROOT."""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def build_null_matrices(cfg_data: dict, seed: int, proteins: list[str],
                        smiles_list: list[str]) -> dict[str, np.ndarray]:
    """Three null matrices over the *given* pool (axes from the saved run).

    Mirrors evaluation/null_baselines.py::build_null_matrices but the protein
    and ligand pools are passed in (taken from the run's score_matrix_axes.json)
    so the geometry is byte-for-byte the one RankBind was scored on.
    """
    bconfig = BRENDADataConfig(
        seed=seed,
        csv_path=_resolve(cfg_data["csv_path"]),
        seq_csv=_resolve(cfg_data["seq_csv"]),
        val_frac=cfg_data["val_frac"],
        test_frac=cfg_data["test_frac"],
    )
    pairs = bconfig.load_pairs()
    train_idx, _, _ = bconfig.get_protein_split()

    train_pairs = pairs[pairs["idx"].isin(set(train_idx))]
    prot_pos_rate = train_pairs.groupby("uniprot")["label"].mean()
    lig_pos_rate = train_pairs.groupby("substrate_smiles")["label"].mean()
    global_rate = float(train_pairs["label"].mean())

    prot_vec = np.array([prot_pos_rate.get(p, global_rate) for p in proteins],
                        dtype=np.float32)
    lig_vec = np.array([lig_pos_rate.get(s, global_rate) for s in smiles_list],
                       dtype=np.float32)

    n_lig, n_prot = len(smiles_list), len(proteins)
    rng = np.random.default_rng(seed)

    # how many of the matrix-pool proteins are actually seen in training
    n_pool_in_train = int(sum(p in prot_pos_rate.index for p in proteins))

    matrices = {
        "null_random":     rng.uniform(0, 1, size=(n_lig, n_prot)).astype(np.float32),
        "null_prot_prior": np.broadcast_to(prot_vec, (n_lig, n_prot)).copy(),
        "null_lig_prior":  np.broadcast_to(lig_vec[:, None], (n_lig, n_prot)).copy(),
    }
    meta = {
        "global_train_pos_rate": global_rate,
        "n_pool_proteins": n_prot,
        "n_pool_proteins_in_train": n_pool_in_train,
        "frac_pool_proteins_in_train": round(n_pool_in_train / max(n_prot, 1), 4),
    }
    return matrices, meta


def top_k_attractors(M: np.ndarray, k: int = 10) -> set[int]:
    """Indices of the top-k 'attractor' proteins (highest mean column score)."""
    scores = compute_attractor_metrics(M)["attractor_scores"]
    return set(int(i) for i in np.argsort(-scores)[:k])


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def row_spearman_vs_prior(M: np.ndarray, prior_vec: np.ndarray) -> tuple[float, float]:
    """Mean +/- std of per-ligand-row Spearman rho between the model's protein
    ordering and the per-protein prior (the same quantity §8.1 reports as
    'rho_row vs null_prot_prior'). Rows where rho is undefined (constant) are
    skipped. prior_vec is constant across ligands, so this measures how much
    each ligand's protein ranking tracks the ligand-agnostic protein prior.
    """
    rhos = []
    for i in range(M.shape[0]):
        r, _ = spearmanr(M[i], prior_vec)
        if not np.isnan(r):
            rhos.append(r)
    if not rhos:
        return float("nan"), float("nan")
    return float(np.mean(rhos)), float(np.std(rhos))


def evaluate_run(run_dir: str) -> dict:
    run_dir = os.path.abspath(run_dir)
    name = os.path.basename(run_dir)

    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    cfg = manifest["config_resolved"]
    cfg_data = cfg["data"]
    seed = cfg.get("seed", 42)

    # Pool + RankBind matrix, exactly as scored
    axes = json.load(open(os.path.join(run_dir, "score_matrix_axes.json")))
    proteins = axes["axis_1_proteins"]
    smiles_list = axes["axis_0_ligands"]
    M_rb = np.load(os.path.join(run_dir, "score_matrix_rankbind.npy"))
    assert M_rb.shape == (len(smiles_list), len(proteins)), "axes/matrix mismatch"

    # Ground-truth positive pairs = the very rows eval.py ranked
    test_df = pd.read_csv(os.path.join(run_dir, "test_preds_rankbind.csv"))
    pos = test_df[test_df["label"] == 1]
    positive_pairs = list(pos[["smiles", "uniprot"]].itertuples(index=False, name=None))

    # ── built-in correctness check: recompute RankBind matrix MRR ───────────
    rb_rank = matrix_ranking_metrics(M_rb, smiles_list, proteins, positive_pairs)
    saved = json.load(open(os.path.join(run_dir, "test_matrix_ranking.json")))
    if saved.get("n_positive_pairs_matched"):
        assert abs(rb_rank["mrr"] - saved["mrr"]) < 1e-6, (
            f"{name}: recomputed MRR {rb_rank['mrr']:.6f} != saved {saved['mrr']:.6f} "
            "— pool/positives reconstruction drifted")
        assert rb_rank["n_positive_pairs_matched"] == saved["n_positive_pairs_matched"]

    nulls, pool_meta = build_null_matrices(cfg_data, seed, proteins, smiles_list)

    gini_rb = compute_attractor_metrics(M_rb)["gini_attractor"]
    gini_pp = compute_attractor_metrics(nulls["null_prot_prior"])["gini_attractor"]
    top10_rb = top_k_attractors(M_rb, 10)
    top10_pp = top_k_attractors(nulls["null_prot_prior"], 10)
    prior_vec = nulls["null_prot_prior"][0]  # per-protein prior (constant across ligands)
    rho_row_mean, rho_row_std = row_spearman_vs_prior(M_rb, prior_vec)

    row = {
        "benchmark": cfg.get("name", name),
        "run": name,
        "n_positive_pairs_matched": rb_rank["n_positive_pairs_matched"],
        # RankBind
        "rb_matrix_mrr": rb_rank["mrr"],
        "rb_matrix_hit5": rb_rank["hit_at_5"],
        "rb_matrix_hit10": rb_rank["hit_at_10"],
        "rb_mean_rank_pct": rb_rank["mean_rank_pct"],
        "rb_gini": gini_rb,
        # nulls
        "null_prot_prior_mrr": matrix_ranking_metrics(
            nulls["null_prot_prior"], smiles_list, proteins, positive_pairs)["mrr"],
        "null_lig_prior_mrr": matrix_ranking_metrics(
            nulls["null_lig_prior"], smiles_list, proteins, positive_pairs)["mrr"],
        "null_random_mrr": matrix_ranking_metrics(
            nulls["null_random"], smiles_list, proteins, positive_pairs)["mrr"],
        "null_prot_prior_gini": gini_pp,
        # §8.3 instruments
        "gini_residual": gini_rb - gini_pp,
        "rho_row_vs_null_prot_prior_mean": rho_row_mean,
        "rho_row_vs_null_prot_prior_std": rho_row_std,
        "top10_jaccard_vs_null_prot_prior": jaccard(top10_rb, top10_pp),
        **pool_meta,
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default=os.path.join(
        _HERE, "attractor_results", "benchmark_null_summary.csv"))
    args = ap.parse_args()

    rows = []
    for rd in args.run_dirs:
        try:
            row = evaluate_run(rd)
            rows.append(row)
            print(f"[ok] {row['benchmark']:14s} "
                  f"MRR rb={row['rb_matrix_mrr']:.4f} "
                  f"prot_prior={row['null_prot_prior_mrr']:.4f} "
                  f"lig_prior={row['null_lig_prior_mrr']:.4f} | "
                  f"rho_row={row['rho_row_vs_null_prot_prior_mean']:+.3f}+/-{row['rho_row_vs_null_prot_prior_std']:.3f} "
                  f"gini_resid={row['gini_residual']:+.4f} "
                  f"jac10={row['top10_jaccard_vs_null_prot_prior']:.3f} | "
                  f"pool_in_train={row['frac_pool_proteins_in_train']:.0%}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {rd}: {type(e).__name__}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nSaved {len(rows)} rows -> {args.out}")
        cols = ["benchmark", "n_positive_pairs_matched", "rb_matrix_mrr",
                "null_prot_prior_mrr", "null_lig_prior_mrr", "null_random_mrr",
                "gini_residual", "top10_jaccard_vs_null_prot_prior",
                "frac_pool_proteins_in_train"]
        print("\n" + df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
