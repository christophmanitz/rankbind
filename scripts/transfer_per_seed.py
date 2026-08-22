"""
scripts/transfer_per_seed.py — A5: per-seed uncertainty for the
BRENDA+SABIO transfer rows (skill item A5).

Walks the bs_v1 (seed-42 anchors) and bs_v3_hp* (peak-config extra seeds)
run dirs and emits
evaluation/attractor_results/bs_transfer_per_seed.csv with one row per
(run, seed), including a split-clean guard identical to
aggregate_multiseed.is_split_clean: pre-Protocol-A runs whose training
seed differs from 42 are flagged invalid (they trained on their own split
but were evaluated on the canonical one — commit 6d685af).

km_with_decoys s1337 (SLURM 27295353) was still running when this table
was first generated; rerun this script after it lands to add the row.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_multiseed import is_split_clean, read_seed  # noqa: E402

# dataset -> ordered glob patterns.
# Per dataset: the swept-config seed-42 anchor is the May bs_v2_hp* run
# (untagged s42 -> split-clean by construction, this is what Table 3
# cites); bs_v1_* rows are the superseded pool-50 base config, kept for
# provenance only; bs_v3_hp*_s7/_s1337 are the pinned-split extra seeds.
GROUPS: dict[str, list[dict[str, str]]] = {
    "kcat_km": [
        {"glob": "20260504-115214*kcat_km*_hp1400", "config": "hp1400"},
        {"glob": "20260822-*kcat_km*hp1400_s7", "config": "hp1400"},
        {"glob": "20260822-*kcat_km*hp1400_s1337", "config": "hp1400"},
        {"glob": "20260503-112034*kcat_km_with_decoys_bs_v1", "config": "base_hp50"},
    ],
    "km": [
        {"glob": "20260504-115536*km_with_decoys_hp3400_bs_v2*", "config": "hp3400"},
        {"glob": "20260822-*km_with_decoys_hp3400*_s7", "config": "hp3400"},
        {"glob": "20260822-*km_with_decoys_hp3400*_s1337", "config": "hp3400"},
        {"glob": "20260503-112035*km_with_decoys_bs_v1", "config": "base_hp50"},
    ],
    "turnover": [
        {"glob": "20260504-*turnover_with_decoys_hp2000_bs_v2_hp2000", "config": "hp2000"},
        {"glob": "20260822-*turnover*hp2000_s7", "config": "hp2000"},
        {"glob": "20260822-*turnover*hp2000_s1337", "config": "hp2000"},
        {"glob": "20260503-112035*turnover_with_decoys_bs_v1", "config": "base_hp50"},
    ],
}

METRIC_KEYS = [
    ("mrr", "matrix_mrr"),
    ("hit_at_5", "matrix_hit_at_5"),
    ("hit_at_10", "matrix_hit_at_10"),
    ("mean_rank_pct", "matrix_mean_rank_pct"),
]


def main() -> None:
    runs_root = ROOT / "results" / "v5_rankbind"
    rows = []
    for dataset, entries in GROUPS.items():
        seen: set[str] = set()
        for entry in entries:
            for d in sorted(runs_root.glob(entry["glob"])):
                if d.name in seen:
                    continue
                seen.add(d.name)
                man_p = d / "manifest.json"
                rank_p = d / "test_matrix_ranking.json"
                sum_p = d / "test_summary.json"
                complete = rank_p.exists()
                seed = read_seed(man_p) if man_p.exists() else None
                clean = is_split_clean(man_p) if man_p.exists() else False
                row = {
                    "dataset": dataset,
                    "config": entry["config"],
                    "run": d.name,
                    "seed": seed,
                    "split_clean": clean,
                    "complete": complete,
                    "git_sha": json.loads(man_p.read_text()).get("git_commit", ""),
                    "matrix_mrr": "",
                    "matrix_hit_at_5": "",
                    "matrix_hit_at_10": "",
                    "matrix_mean_rank_pct": "",
                    "test_global_auc": "",
                    "matrix_per_ligand_auc": "",
                }
                if complete:
                    r = json.loads(rank_p.read_text())
                    for src, dst in METRIC_KEYS:
                        row[dst] = round(float(r[src]), 4)
                    pla = r.get("matrix_per_ligand_auc")
                    row["matrix_per_ligand_auc"] = (
                        round(float(pla), 4) if pla is not None else ""
                    )
                    if sum_p.exists():
                        s = json.loads(sum_p.read_text())
                        ga = s.get("global_auc")
                        row["test_global_auc"] = round(float(ga), 4) if ga else ""
                rows.append(row)

    out = ROOT / "evaluation" / "attractor_results" / "bs_transfer_per_seed.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {out} ({len(rows)} rows)")
    for r in rows:
        status = "ok " if r["complete"] and r["split_clean"] else (
            "PEND" if not r["complete"] else "LEAK")
        print(f"  [{status}] {r['dataset']:8s} {r['config']:9s} seed={r['seed']} "
              f"MRR={r['matrix_mrr']} H@10={r['matrix_hit_at_10']}")


if __name__ == "__main__":
    main()
