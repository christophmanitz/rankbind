#!/usr/bin/env python3
"""
Matrix-level per-ligand AUC for the four Phase-1 baselines, on the SAME
200×200 geometry and the SAME held-out positive set used by the v5 RankBind
runs — so the comparison is n≈30 vs n≈30, not n=30 vs n=4.

Geometry (identical to evaluation/null_baselines.py and v5 build_score_matrix):
    proteins = list(load_sequences())[:200]
    ligands  = load_pairs()['substrate_smiles'].unique()[:200]
Positive set = TEST-split positives (label==1) over that pool — held out,
no model trained on them (see matrix_per_ligand_auc docstring for the
no-leakage argument: training pairs appear only as distractor columns and
can only depress the AUC).

Writes evaluation/attractor_results/matrix_per_ligand_auc_baselines.csv and
prints a combined table alongside the RankBind numbers for direct comparison.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "adapters"))
from v5_rankbind.metrics import matrix_per_ligand_auc  # noqa: E402
from common import BRENDADataConfig  # noqa: E402

OUT_CSV = ROOT / "evaluation" / "attractor_results" / "matrix_per_ligand_auc_baselines.csv"

BASELINE_MATRICES = {
    "DrugBAN":  "results/original_drugban/score_matrix_DrugBAN.npy",
    "GraphDTA": "results/original_graphdta/score_matrix_graphdta.npy",
    "MolTrans": "results/original_moltrans/score_matrix_moltrans.npy",
    "GEMS":     "results/original_gems/score_matrix_gems.npy",
}

# strict (n=4) per-ligand AUC from Phase-1, for the side-by-side contrast
STRICT_PHASE1 = {
    "DrugBAN": 0.375, "MolTrans": 0.500, "GraphDTA": 0.625, "GEMS": 0.250,
}


def canonical_axes_and_positives(n_matrix: int = 200):
    cfg = BRENDADataConfig()
    pairs = cfg.load_pairs()
    seqs = cfg.load_sequences()
    _, _, test_idx = cfg.get_protein_split()

    proteins = list(seqs.keys())[:n_matrix]
    ligands = pairs["substrate_smiles"].unique()[:n_matrix].tolist()

    test_pos = pairs[(pairs["idx"].isin(set(test_idx))) & (pairs["label"] == 1)]
    positive_pairs = list(
        test_pos[["substrate_smiles", "uniprot"]].itertuples(index=False, name=None)
    )
    return ligands, proteins, positive_pairs


def main() -> None:
    ligands, proteins, positive_pairs = canonical_axes_and_positives()
    print(f"[axes] {len(ligands)} ligands × {len(proteins)} proteins | "
          f"{len(positive_pairs)} held-out test-positive pairs")

    rows = []
    for name, rel in BASELINE_MATRICES.items():
        path = ROOT / rel
        if not path.exists():
            print(f"  [skip] {name}: {rel} missing")
            continue
        # Orientation fix: train_original.py::build_score_matrix_* (and the
        # DrugBAN builder) emit [n_prot, n_lig] — score_matrix[i, j] is
        # protein i × ligand j (train_original.py:164,207). matrix_per_ligand_auc
        # expects the canonical [n_lig, n_prot] (row = ligand; see
        # v5_rankbind.metrics:148-167 and compute_attractor_metrics:78). The pool
        # is square 200×200 so the wrong orientation passes any shape check
        # silently and scrambles the per-ligand AUC. Transpose to canonical.
        M = np.load(path).T
        if M.shape != (len(ligands), len(proteins)):
            print(f"  [warn] {name}: matrix shape {M.shape} != "
                  f"({len(ligands)},{len(proteins)}) — axes may differ, skipping")
            continue
        res = matrix_per_ligand_auc(M, ligands, proteins, positive_pairs)
        rows.append({
            "model":               name,
            "strict_per_lig_auc":  STRICT_PHASE1.get(name, ""),
            "strict_n":            4,
            "matrix_per_lig_auc":  round(res["matrix_per_ligand_auc"], 4),
            "matrix_n":            res["n_ligands_counted"],
            "n_positive_cells":    res["n_positive_cells"],
        })
        print(f"  {name:10s} strict {STRICT_PHASE1.get(name)} (n=4)  ->  "
              f"matrix {res['matrix_per_ligand_auc']:.4f} (n={res['n_ligands_counted']})")

    if not rows:
        print("No baseline rows produced.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} baselines -> {OUT_CSV}")


if __name__ == "__main__":
    main()
