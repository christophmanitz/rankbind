#!/usr/bin/env python3
"""
Recompute matrix-level per-ligand AUC for *existing* v5 run dirs.

The strict ``per_ligand_auc`` in test_summary.json counts only ligands that
have both a positive and a negative *test pair* — under the protein-based
split that is ~4 ligands on BRENDA, statistically empty. This script reuses
the already-saved 200×200 score matrix + held-out positive set to compute a
per-ligand AUC over every ligand row that has an observed binder in the pool,
lifting n from ~4 to ~50 (BRENDA) / hundreds (ESP, turnover). No retraining,
no model load — purely a re-reduction of artifacts already on disk.

Positive set = (smiles, uniprot) pairs with label==1 in test_preds_rankbind.csv,
i.e. the SAME held-out positives that matrix_ranking_metrics already ranks.
A column that is a training positive for the same ligand is treated as a
negative, so this is a conservative (never inflated) estimate.

Usage:
    python evaluation/matrix_per_ligand_auc_recompute.py [RUN_DIR ...]
    # no args -> a curated default set (v4, v5b, v6, ESP, KIBA, topline)
Writes evaluation/attractor_results/matrix_per_ligand_auc.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v5_rankbind.metrics import matrix_per_ligand_auc  # noqa: E402

RUNS_ROOT = ROOT / "results" / "v5_rankbind"
OUT_CSV = ROOT / "evaluation" / "attractor_results" / "matrix_per_ligand_auc.csv"

# Curated default set — one representative run per headline config.
# (glob -> most recent match; benchmark datasets carry their own name token.)
DEFAULT_GLOBS = [
    "*default_v4",        # BRENDA hard-neg headline (seed 42)
    "*abl_attn_pool_v5b_s42",  # BRENDA attn-pool best (seed 42)
    "*abl_deltafield_v6_deltafield",  # DeltaField anti-shortcut arch
    "*conv_esp*",         # ESP-conv (large n)
    "*conv_kiba*",        # KIBA-conv
    "*bindingdb*v4*",     # BindingDB transfer
    "*topline*",          # transductive ceiling
]


def resolve_runs(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a) for a in argv]
    runs: list[Path] = []
    for g in DEFAULT_GLOBS:
        hits = sorted(RUNS_ROOT.glob(g))
        if hits:
            runs.append(hits[-1])  # most recent matching
    return runs


def load_positive_pairs(run_dir: Path) -> list[tuple[str, str]]:
    preds = run_dir / "test_preds_rankbind.csv"
    pairs: list[tuple[str, str]] = []
    with preds.open() as fh:
        for row in csv.DictReader(fh):
            if int(float(row["label"])) == 1:
                pairs.append((row["smiles"], row["uniprot"]))
    return pairs


def process(run_dir: Path) -> dict | None:
    mat_path = run_dir / "score_matrix_rankbind.npy"
    axis_path = run_dir / "score_matrix_axes.json"
    summ_path = run_dir / "test_summary.json"
    if not (mat_path.exists() and axis_path.exists()):
        print(f"  [skip] {run_dir.name}: missing matrix/axes")
        return None

    M = np.load(mat_path)
    ax = json.loads(axis_path.read_text())
    lig_list = ax["axis_0_ligands"]
    prot_list = ax["axis_1_proteins"]
    positive_pairs = load_positive_pairs(run_dir)

    res = matrix_per_ligand_auc(M, lig_list, prot_list, positive_pairs)

    # strict (test-pair) per-ligand AUC for the side-by-side contrast
    old_auc, old_n = float("nan"), 0
    if summ_path.exists():
        s = json.loads(summ_path.read_text())
        old_auc = s.get("per_ligand_auc", float("nan"))
        old_n = s.get("n_ligands_counted", 0)

    return {
        "run":                   run_dir.name,
        "strict_per_lig_auc":    round(old_auc, 4) if old_auc == old_auc else "",
        "strict_n":              old_n,
        "matrix_per_lig_auc":    round(res["matrix_per_ligand_auc"], 4)
                                 if res["matrix_per_ligand_auc"] == res["matrix_per_ligand_auc"] else "",
        "matrix_n":              res["n_ligands_counted"],
        "n_positive_cells":      res["n_positive_cells"],
    }


def main() -> None:
    runs = resolve_runs(sys.argv[1:])
    rows = []
    for rd in runs:
        print(f"[run] {rd.name}")
        r = process(rd)
        if r:
            rows.append(r)
            print(f"  strict: AUC {r['strict_per_lig_auc']} (n={r['strict_n']})"
                  f"  ->  matrix: AUC {r['matrix_per_lig_auc']} (n={r['matrix_n']})")

    if not rows:
        print("No rows produced.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] {len(rows)} runs -> {OUT_CSV}")


if __name__ == "__main__":
    main()
