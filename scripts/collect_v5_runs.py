"""
scripts/collect_v5_runs.py — Walk results/v5_rankbind/*/manifest.json and emit
a consolidated runs_manifest.csv that is the single source of truth for the
paper's ablation table and any published number.

The CSV is designed to paste directly into LaTeX tabular, so it flattens
out the most-quoted fields. The full JSON manifests remain the authoritative
provenance record.

Usage:
    python scripts/collect_v5_runs.py                    # prints + writes CSV
    python scripts/collect_v5_runs.py --json             # also JSON dump
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
RUNS_ROOT = PROJECT_ROOT / "results" / "v5_rankbind"


PRIMARY_COLS = [
    "run_id", "config_name", "sampler", "loss_type", "head",
    "best_val_metric_key", "best_val_metric", "best_val_epoch",
    "test_global_auc", "test_global_aupr", "test_per_lig_auc",
    "test_hit_at_1", "test_hit_at_5", "test_hit_at_10",
    "matrix_mrr", "matrix_mean_rank_pct",
    "matrix_hit_at_1", "matrix_hit_at_5", "matrix_hit_at_10",
    "matrix_n_positive_pairs_matched",
    "n_parameters_trainable",
    "n_train_pairs", "n_val_pairs", "n_test_pairs",
    "started_at", "finished_at", "host", "slurm_job_id",
    "checkpoint_sha256", "source_hash_short",
]


def _get(dct: dict, *path, default=""):
    cur = dct
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def collect() -> pd.DataFrame:
    rows = []
    for mf in sorted(RUNS_ROOT.glob("*/manifest.json")):
        try:
            m = json.loads(mf.read_text())
        except json.JSONDecodeError:
            print(f"WARN: could not parse {mf}")
            continue
        cfg = m.get("config_resolved", {})
        row = {
            "run_id":                 m.get("run_id", ""),
            "config_name":            cfg.get("name", ""),
            "sampler":                _get(cfg, "sampler", "type"),
            "loss_type":              _get(cfg, "loss", "type"),
            "head":                   _get(cfg, "model", "head"),
            "best_val_metric_key":    _get(m, "metrics", "best_val_metric_key"),
            "best_val_metric":        _get(m, "metrics", "best_val_metric"),
            "best_val_epoch":         _get(m, "metrics", "best_val_epoch"),
            "test_global_auc":        _get(m, "metrics", "test_global_auc"),
            "test_global_aupr":       _get(m, "metrics", "test_global_aupr"),
            "test_per_lig_auc":       _get(m, "metrics", "test_per_lig_auc"),
            "test_hit_at_1":          _get(m, "metrics", "test_hit_at_1"),
            "test_hit_at_5":          _get(m, "metrics", "test_hit_at_5"),
            "test_hit_at_10":         _get(m, "metrics", "test_hit_at_10"),
            "matrix_mrr":                        _get(m, "metrics", "matrix_mrr"),
            "matrix_mean_rank_pct":              _get(m, "metrics", "matrix_mean_rank_pct"),
            "matrix_hit_at_1":                   _get(m, "metrics", "matrix_hit_at_1"),
            "matrix_hit_at_5":                   _get(m, "metrics", "matrix_hit_at_5"),
            "matrix_hit_at_10":                  _get(m, "metrics", "matrix_hit_at_10"),
            "matrix_n_positive_pairs_matched":   _get(m, "metrics", "matrix_n_positive_pairs_matched"),
            "n_parameters_trainable": _get(m, "model", "n_parameters_trainable"),
            "n_train_pairs":          _get(m, "split", "n_train_pairs"),
            "n_val_pairs":            _get(m, "split", "n_val_pairs"),
            "n_test_pairs":           _get(m, "split", "n_test_pairs"),
            "started_at":             m.get("started_at", ""),
            "finished_at":            m.get("finished_at", ""),
            "host":                   _get(m, "env", "host"),
            "slurm_job_id":           _get(m, "env", "slurm_job_id"),
            "checkpoint_sha256":      _get(m, "outputs", "checkpoint", "sha256"),
            "source_hash_short":      _get(m, "source_hashes", "_combined_short"),
            "manifest_path":          str(mf),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=PRIMARY_COLS)
    df = pd.DataFrame(rows)
    # order primary columns first
    ordered = [c for c in PRIMARY_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    df = collect()
    out_csv = RUNS_ROOT / "runs_manifest.csv"
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} runs)")
    if len(df):
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df[[c for c in PRIMARY_COLS[:9] if c in df.columns]].to_string(index=False))
    if args.json:
        out_json = RUNS_ROOT / "runs_manifest.json"
        df.to_json(out_json, orient="records", indent=2)
        print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
