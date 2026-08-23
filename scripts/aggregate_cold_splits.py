#!/usr/bin/env python
"""scripts/aggregate_cold_splits.py — aggregate v7_cold_* runs (skill §5-§8).

Walks results/v5_rankbind/*_v7_cold_*/manifest.json (cold-ligand and
double-cold stress-test runs with matched BCE/RankBind controls), joins the
per-split null baselines from evaluation/null_baseline_firstclass*.csv and
writes:

    evaluation/attractor_results/cold_split_runs.csv        per-run rows
    evaluation/attractor_results/cold_split_multiseed.csv   mean +/- std
    evaluation/COLD_SPLIT_SUMMARY.md                        paper-facing summary

Idempotent; rerun after stragglers finish.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS = os.path.join(_ROOT, "results", "v5_rankbind")
_EVAL = os.path.join(_ROOT, "evaluation")
_OUT = os.path.join(_EVAL, "attractor_results")

METRIC_KEYS = [
    "test_global_auc", "test_global_aupr", "test_per_lig_auc",
    "test_hit_at_1", "test_hit_at_5", "test_hit_at_10",
    "matrix_mrr", "matrix_hit_at_5", "matrix_hit_at_10",
    "matrix_matrix_per_ligand_auc", "matrix_n_positive_pairs_matched",
]
SPLIT_KEYS = [
    "n_train_pairs", "n_val_pairs", "n_test_pairs", "n_train_ligands",
    "n_test_ligands", "test_lig_in_train_frac", "test_prot_in_train_frac",
]
SPLIT_LABEL = {
    "ligand": "cold-ligand (ligand-disjoint; proteins recur)",
    "double_cold": "double-cold (neither axis recurs)",
}


def load_runs() -> pd.DataFrame:
    rows = []
    for mf in sorted(glob.glob(os.path.join(
            _RESULTS, "*_v7_cold_*", "manifest.json"))):
        m = json.load(open(mf))
        if not m.get("metrics", {}).get("test_global_auc"):
            print(f"[agg] SKIP (no eval yet): {m['run_id']}")
            continue
        cr = m.get("config_resolved", {})
        rows.append({
            "run_id": m["run_id"],
            "config": os.path.basename(m["config_path"]).replace(".json", ""),
            "tag": m.get("tag", ""),
            "seed": cr.get("seed"),
            "split_mode": cr.get("data", {}).get("split_mode", "protein"),
            "loss": cr.get("loss", {}).get("type"),
            **{k: m["split"].get(k) for k in SPLIT_KEYS},
            **{k: m["metrics"].get(k) for k in METRIC_KEYS},
        })
    return pd.DataFrame(rows)


def agg_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (smode, cfg, loss), sub in runs.groupby(
            ["split_mode", "config", "loss"], sort=True):
        row = {"split_mode": smode, "config": cfg, "loss": loss,
               "n_seeds": len(sub),
               "seeds": ",".join(str(s) for s in sorted(sub["seed"]))}
        for k in METRIC_KEYS[:-1]:
            row[f"{k}_mean"] = float(sub[k].mean())
            row[f"{k}_std"] = float(sub[k].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def null_rows(split_mode: str) -> pd.DataFrame:
    suffix = "" if split_mode == "protein" else f"_{split_mode}"
    p = os.path.join(_EVAL, f"null_baseline_firstclass{suffix}.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df[["null", "pooled_auc_test_full_split", "mrr_tie_aware",
             "hit_at_10_tie_aware"]]
    df.columns = ["source", "pooled_auc_full_split", "matrix_mrr_tie_aware",
                  "matrix_hit10_tie_aware"]
    return df


def main() -> None:
    runs = load_runs()
    if runs.empty:
        sys.exit("[agg] no finished v7_cold_* runs found")
    os.makedirs(_OUT, exist_ok=True)
    runs.to_csv(os.path.join(_OUT, "cold_split_runs.csv"), index=False)
    agg = agg_table(runs)
    agg.to_csv(os.path.join(_OUT, "cold_split_multiseed.csv"), index=False)
    print(f"[agg] {len(runs)} runs / {len(agg)} config groups")

    md = [
        "# COLD_SPLIT_SUMMARY.md — cold-ligand / double-cold stress tests",
        "",
        "Matched controls: same encoders, capacity class, training budget,",
        "hard-negative logic, seeds {42, 7, 1337}, canonical 200x200 pool;",
        "ONLY the split definition changes (skill §7). Pooled AUC = global",
        "AUC over each split's FULL test set. Matrix MRR/H@10 = within-",
        "ligand ranking on canonical-pool test positives — small-n under",
        "double_cold (~6 pairs), read with the full-split columns. Nulls",
        "per split (skill §8); a null's lack of signal is itself a result.",
    ]
    for smode in ("ligand", "double_cold"):
        sub = runs[runs["split_mode"] == smode]
        if sub.empty:
            continue
        s0 = sub.iloc[0]
        md += ["", f"## Split: {SPLIT_LABEL[smode]} (`{smode}`)", "",
               f"Split structure pinned per run manifest: test ligands seen "
               f"in train {s0['test_lig_in_train_frac']:.1%}, test proteins "
               f"seen in train {s0['test_prot_in_train_frac']:.1%}; pairs "
               f"tr/va/te {int(s0['n_train_pairs'])}/{int(s0['n_val_pairs'])}/"
               f"{int(s0['n_test_pairs'])}.", "",
               "| source | pooled AUC | matrix MRR | H@10 | n |",
               "|---|---:|---:|---:|---:|"]
        for _, r in null_rows(smode).iterrows():
            md.append(f"| {r['source']} (null) "
                      f"| {r['pooled_auc_full_split']:.3f} "
                      f"| {r['matrix_mrr_tie_aware']:.3f} "
                      f"| {r['matrix_hit10_tie_aware']:.3f} | - |")
        for _, r in agg[agg["split_mode"] == smode].iterrows():
            label = ("RankBind (margin + hard negs)" if r["loss"] == "margin"
                     else "BCE control (pairwise)")
            md.append(
                f"| {label} `{r['config']}` "
                f"| {r['test_global_auc_mean']:.3f} ± {r['test_global_auc_std']:.3f} "
                f"| {r['matrix_mrr_mean']:.3f} ± {r['matrix_mrr_std']:.3f} "
                f"| {r['matrix_hit_at_10_mean']:.3f} ± {r['matrix_hit_at_10_std']:.3f} "
                f"| {int(r['n_seeds'])} |")

    md += ["", "## Reference: canonical protein-stratified split", "",
           "| source | pooled AUC | matrix MRR | H@10 | n |",
           "|---|---:|---:|---:|---:|"]
    for _, r in null_rows("protein").iterrows():
        md.append(f"| {r['source']} (null) "
                  f"| {r['pooled_auc_full_split']:.3f} "
                  f"| {r['matrix_mrr_tie_aware']:.3f} "
                  f"| {r['matrix_hit10_tie_aware']:.3f} | - |")
    ms_path = os.path.join(_OUT, "phase2_rankbind_multiseed.csv")
    if os.path.exists(ms_path):
        ms = pd.read_csv(ms_path).set_index("config")
        for cfg, label in (("abl_bce_only", "BCE control (pairwise)"),
                           ("default", "RankBind (margin + hard negs)")):
            if cfg in ms.index:
                r = ms.loc[cfg]
                md.append(f"| {label} | {r['gAUC_mean']:.3f} ± {r['gAUC_std']:.3f} "
                          f"| {r['MRR_mean']:.3f} ± {r['MRR_std']:.3f} "
                          f"| {r['H@10_mean']:.3f} ± {r['H@10_std']:.3f} "
                          f"| {int(r['n_seeds'])} |")
    md += ["",
           "Per-run provenance: `attractor_results/cold_split_runs.csv`; "
           "aggregates: `attractor_results/cold_split_multiseed.csv`. "
           "Regenerate: `python scripts/aggregate_cold_splits.py`."]
    open(os.path.join(_EVAL, "COLD_SPLIT_SUMMARY.md"), "w").write(
        "\n".join(md) + "\n")
    print("[agg] wrote COLD_SPLIT_SUMMARY.md + cold_split_{runs,multiseed}.csv")


if __name__ == "__main__":
    main()
