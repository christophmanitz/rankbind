#!/usr/bin/env python3
"""
evaluation/graphdta_recipe_table.py — assemble the GraphDTA recipe-transfer
head-to-head table.

For each dataset (brenda / turnover) it collects the §8.3 null-baseline
instruments for every trained variant and the RankBind reference, into one
tidy table:

    GraphDTA-BCE (orient-fixed)  ->  recipe-a  ->  recipe-b  ->  recipe-c  ->  RankBind-v4

Sources (all produced upstream — this script only joins them):
  * recipe variants  : evaluation/attractor_results/graphdta_recipe_null_<v>_<ds>.csv
                       (one per `scripts/run_graphdta_recipe.sh` job, written by
                       benchmark_null_eval.py)
  * RankBind ref     : benchmark_null_eval.py run live on the existing v4 run dir
  * BRENDA per-lig-AUC: evaluation/attractor_results/matrix_per_ligand_auc_all.csv
                        (already orientation-fixed; n=30 head-to-head)

The recipe claim holds if MRR / Hit@10 / matrix per-ligand-AUC rise and
gini_residual / Top-10 Jaccard fall along a->b->c, toward RankBind-v4 —
independent of GraphDTA's mediocre architecture.

Usage:  python evaluation/graphdta_recipe_table.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AR = ROOT / "evaluation" / "attractor_results"

# Existing RankBind reference runs per dataset (most recent match wins).
RANKBIND_REF = {
    "brenda":   "results/v5_rankbind/*default_v4",
    "turnover": "results/v5_rankbind/*turnover_with_decoys_hp2000_bs_v2_hp2000",
}

VARIANT_LABEL = {
    "a": "recipe-a (bal+BCE)",
    "b": "recipe-b (+margin)",
    "c": "recipe-c (+hard-neg)",
}
COLS = ["rb_matrix_mrr", "rb_matrix_hit10", "gini_residual",
        "top10_jaccard_vs_null_prot_prior", "rho_row_vs_null_prot_prior_mean"]


def _rankbind_ref_row(dataset: str) -> dict | None:
    """Run benchmark_null_eval live on the existing RankBind run for `dataset`."""
    hits = sorted(glob.glob(str(ROOT / RANKBIND_REF[dataset])))
    if not hits:
        print(f"  [warn] no RankBind ref run for {dataset} ({RANKBIND_REF[dataset]})")
        return None
    sys.path.insert(0, str(ROOT / "evaluation"))
    from benchmark_null_eval import evaluate_run
    try:
        r = evaluate_run(hits[-1])
        r["_variant"] = "RankBind-v4"
        return r
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] RankBind ref eval failed for {dataset}: {e}")
        return None


def _brenda_perlig() -> dict:
    """model-label -> matrix per-ligand AUC from the orientation-fixed table."""
    p = AR / "matrix_per_ligand_auc_all.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["model"], df["matrix_per_lig_auc"]))


def main() -> None:
    perlig = _brenda_perlig()
    any_table = False
    for dataset in ("brenda", "turnover"):
        rows = []
        # recipe variants a/b/c
        for v in ("a", "b", "c"):
            csv = AR / f"graphdta_recipe_null_{v}_{dataset}.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if df.empty:
                continue
            rec = df.iloc[0].to_dict()
            rec["_variant"] = VARIANT_LABEL[v]
            rows.append(rec)
        # RankBind reference
        ref = _rankbind_ref_row(dataset)
        if ref:
            rows.append(ref)

        if not rows:
            print(f"\n[{dataset}] no trained recipe runs yet — skipping.")
            continue

        any_table = True
        out = pd.DataFrame(rows)
        show = ["_variant"] + [c for c in COLS if c in out.columns]
        out_disp = out[show].rename(columns={
            "_variant": "model",
            "rb_matrix_mrr": "MRR",
            "rb_matrix_hit10": "Hit@10",
            "gini_residual": "Gini-res",
            "top10_jaccard_vs_null_prot_prior": "Jac10",
            "rho_row_vs_null_prot_prior_mean": "rho_row",
        })
        if dataset == "brenda" and perlig:
            out_disp["perLigAUC(n30)"] = [
                perlig.get("GraphDTA-recipe-a (bal+BCE)" if "recipe-a" in m else
                           "GraphDTA-recipe-b (+margin)" if "recipe-b" in m else
                           "GraphDTA-recipe-c (+hard-neg)" if "recipe-c" in m else
                           "RankBind v4 (hard-neg)" if "RankBind" in m else m, "")
                for m in out_disp["model"]
            ]
        out_csv = AR / f"graphdta_recipe_table_{dataset}.csv"
        out_disp.to_csv(out_csv, index=False)
        print(f"\n=== GraphDTA recipe transfer — {dataset} ===")
        print(out_disp.to_string(index=False))
        print(f"[saved] {out_csv}")

    if not any_table:
        print("\nNo recipe runs found. Launch scripts/run_graphdta_recipe.sh first,")
        print("then re-run this script to assemble the head-to-head table.")


if __name__ == "__main__":
    main()
