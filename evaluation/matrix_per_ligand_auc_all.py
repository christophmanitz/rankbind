#!/usr/bin/env python3
"""
ONE combined BRENDA head-to-head table for matrix-level per-ligand AUC:
the four Phase-1 baselines + the RankBind variants (v4 / v5b / v6), all
scored on the SAME 200×200 geometry against ONE held-out positive set
(the seed-42 test-split positives, derived once from BRENDADataConfig).

Why a single source: previously baseline positives came from the config and
RankBind positives from each run's test_preds.csv. Both yield the same
seed-42 split, but for a publishable head-to-head every row must use the
identical positive set. This script is that single source of truth.

No-leakage argument (see v5_rankbind.metrics.matrix_per_ligand_auc): the
scored positives are held-out test pairs no model trained on; training pairs
appear only as distractor columns and can only depress the AUC.

Writes evaluation/attractor_results/matrix_per_ligand_auc_all.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines" / "adapters"))
from v5_rankbind.metrics import matrix_per_ligand_auc  # noqa: E402
from common import BRENDADataConfig  # noqa: E402

RUNS_ROOT = ROOT / "results" / "v5_rankbind"
OUT_CSV = ROOT / "evaluation" / "attractor_results" / "matrix_per_ligand_auc_all.csv"

# (label, matrix path, strict-n=4 per-ligand AUC for the contrast column)
BASELINES = [
    ("DrugBAN",  "results/original_drugban/score_matrix_DrugBAN.npy",  0.375),
    ("GEMS",     "results/original_gems/score_matrix_gems.npy",        0.250),
    ("MolTrans", "results/original_moltrans/score_matrix_moltrans.npy", 0.500),
    ("GraphDTA", "results/original_graphdta/score_matrix_graphdta.npy", 0.625),
]

# RankBind BRENDA runs (own axes file present); strict per-lig AUC read live.
RANKBIND_GLOBS = [
    ("RankBind v4 (hard-neg)",  "*default_v4"),
    ("RankBind v5b (attn-pool)", "*abl_attn_pool_v5b_s42"),
    ("RankBind v6 (DeltaField)", "*abl_deltafield_v6_deltafield"),
]

# GraphDTA anti-shortcut recipe (BRENDA). Same canonical [n_lig, n_prot] axes
# file as RankBind runs, so they are scored via the axes path (NOT transposed).
# The progressive a->b->c trend on the very metric where the architecture
# collapses isolates the recipe contribution from the RankBind architecture.
RECIPE_ROOT = ROOT / "results" / "graphdta_recipe"
RECIPE_RUNS = [
    ("GraphDTA-recipe-a (bal+BCE)",   "a_brenda"),
    ("GraphDTA-recipe-b (+margin)",   "b_brenda"),
    ("GraphDTA-recipe-c (+hard-neg)", "c_brenda"),
]


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


def strict_from_run(run_dir: Path):
    s = run_dir / "test_summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        return d.get("per_ligand_auc"), d.get("n_ligands_counted")
    return None, None


def main() -> None:
    ligands, proteins, positive_pairs = canonical_axes_and_positives()
    print(f"[positive source] seed-42 test split: {len(positive_pairs)} positive "
          f"pairs | canonical axes {len(ligands)}×{len(proteins)}")

    rows = []

    # --- baselines: canonical axes ---
    for name, rel, strict in BASELINES:
        p = ROOT / rel
        if not p.exists():
            print(f"  [skip] {name}: missing"); continue
        # Orientation fix: the Phase-1 builders (train_original.py:164,207 +
        # the DrugBAN builder) emit [n_prot, n_lig]; matrix_per_ligand_auc needs
        # canonical [n_lig, n_prot]. The 200×200 square pool hides the swap, so a
        # transposed matrix scrambles the per-ligand AUC silently. Transpose.
        # (RankBind runs below carry their own axes file and are already
        # canonical — they are NOT transposed.)
        M = np.load(p).T
        assert M.shape == (len(ligands), len(proteins)), (
            f"{name}: post-transpose shape {M.shape} != "
            f"canonical ({len(ligands)}, {len(proteins)})")
        res = matrix_per_ligand_auc(M, ligands, proteins, positive_pairs)
        rows.append({"model": name, "family": "baseline",
                     "strict_per_lig_auc": strict, "strict_n": 4,
                     "matrix_per_lig_auc": round(res["matrix_per_ligand_auc"], 4),
                     "matrix_n": res["n_ligands_counted"]})

    # --- RankBind: each run's own axes file, same positive_pairs ---
    for name, glob in RANKBIND_GLOBS:
        hits = sorted(RUNS_ROOT.glob(glob))
        if not hits:
            print(f"  [skip] {name}: no run matches {glob}"); continue
        rd = hits[-1]
        M = np.load(rd / "score_matrix_rankbind.npy")
        ax = json.loads((rd / "score_matrix_axes.json").read_text())
        res = matrix_per_ligand_auc(
            M, ax["axis_0_ligands"], ax["axis_1_proteins"], positive_pairs)
        strict_auc, _ = strict_from_run(rd)
        rows.append({"model": name, "family": "rankbind",
                     "strict_per_lig_auc": round(strict_auc, 4) if strict_auc else "",
                     "strict_n": 4,
                     "matrix_per_lig_auc": round(res["matrix_per_ligand_auc"], 4),
                     "matrix_n": res["n_ligands_counted"]})

    # --- GraphDTA recipe (BRENDA): own canonical axes file, same positive_pairs ---
    for name, tag in RECIPE_RUNS:
        rd = RECIPE_ROOT / tag
        mpath = rd / "score_matrix_rankbind.npy"
        axpath = rd / "score_matrix_axes.json"
        if not (mpath.exists() and axpath.exists()):
            print(f"  [skip] {name}: {tag} not trained yet"); continue
        M = np.load(mpath)
        ax = json.loads(axpath.read_text())
        assert M.shape == (len(ax["axis_0_ligands"]), len(ax["axis_1_proteins"])), (
            f"{name}: matrix shape {M.shape} != axes "
            f"({len(ax['axis_0_ligands'])}, {len(ax['axis_1_proteins'])})")
        res = matrix_per_ligand_auc(
            M, ax["axis_0_ligands"], ax["axis_1_proteins"], positive_pairs)
        rows.append({"model": name, "family": "graphdta_recipe",
                     "strict_per_lig_auc": "", "strict_n": "",
                     "matrix_per_lig_auc": round(res["matrix_per_ligand_auc"], 4),
                     "matrix_n": res["n_ligands_counted"]})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n{'model':28s} {'strict(n=4)':>12s} {'matrix(n=30)':>14s}")
    for r in rows:
        print(f"{r['model']:28s} {str(r['strict_per_lig_auc']):>12s} "
              f"{r['matrix_per_lig_auc']:>10.4f} (n={r['matrix_n']})")
    print(f"\n[done] {len(rows)} models -> {OUT_CSV}")


if __name__ == "__main__":
    main()
