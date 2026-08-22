"""
scripts/aggregate_multiseed.py — aggregate Phase-2 runs across seeds.

Builds `evaluation/attractor_results/phase2_rankbind_multiseed.csv`:
one row per (config, recipe) showing mean + std over three seeds (42, 7,
1337) for the paper-ready metrics.

Per-config canonical seed-42 tag:

  - default, abl_no_sampler, abl_no_bilinear  → v4   (margin + hard negs)
  - abl_no_margin                             → v3   (BCE, no collator — unchanged by hard-neg mining)
  - abl_bce_only                              → v2   (BCE + MLP — Phase-1 equivalent control)

For every other seed we look for a run tagged `v4_s<seed>`. The seed
itself is read from `config_resolved.seed` in each manifest.json (that
field is set by train.py's `--seed` override and falls back to the JSON
config's seed).

Metrics aggregated (mean, std):

  - matrix_mrr, matrix_hit_at_{1,5,10}, matrix_mean_rank_pct
  - test_global_auc, test_global_aupr
  - gini_residual (from score matrix via evaluation/attractor_diagnosis.py)
  - jac_null      (Top-10 attractor Jaccard vs null_prot_prior)
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "evaluation"))

from attractor_diagnosis import gini, compute_attractor_scores  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# Config: canonical seed-42 tag per config, and common seed-override tags.
# ──────────────────────────────────────────────────────────────────────────────

CANONICAL_SEED42_TAG = {
    "default":         "v4",   # seed-42 anchor on disk (untagged v4 dir)
    "abl_no_sampler":  "v4",
    "abl_no_bilinear": "v4",
    "abl_no_margin":   "v5",   # Protocol-A sweep (2026-08-22): v5_s42 anchor
    "abl_bce_only":    "v5",
    "abl_mrrsel":      "v5",   # A4 condition B: matrix-MRR checkpoint selection
    "abl_attn_pool":   "v5b",  # Stage-(b) of Phase-4 plan: residue attn-pool
}
SWEEP_SEEDS = [42, 7, 1337]

# Canonical head + "cap" label per config — kept in the output for table rendering.
CONFIG_META = {
    "default":         ("bilinear", "128"),
    "abl_no_sampler":  ("bilinear", "128"),
    "abl_no_bilinear": ("mlp_concat", "N/A"),
    "abl_no_margin":   ("bilinear", "128"),
    "abl_bce_only":    ("mlp_concat", "N/A"),
    "abl_mrrsel":      ("bilinear", "128"),
    "abl_attn_pool":   ("bilinear+attn", "128"),
}

METRICS = [
    "matrix_mrr", "matrix_hit_at_1", "matrix_hit_at_5", "matrix_hit_at_10",
    "matrix_mean_rank_pct", "test_global_auc", "test_global_aupr",
    "gini_residual", "jac_null",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"^\d{8}-\d{6}_[0-9a-f]+_(?P<config>[a-z_]+)_(?P<tag>v[a-z0-9]+(?:_s\d+)?)$")


def parse_run_id(run_id: str) -> tuple[str, str] | None:
    """Return (config_name, tag) or None if run_id doesn't parse."""
    m = _TAG_RE.match(run_id)
    if not m:
        return None
    return m.group("config"), m.group("tag")


def read_seed(manifest_path: Path) -> int | None:
    """Return the resolved seed from manifest.json (post --seed override)."""
    m = json.loads(manifest_path.read_text())
    cfg = m.get("config_resolved", {})
    if "seed" in cfg:
        return int(cfg["seed"])
    return None


def read_split_seed(manifest_path: Path) -> int | None:
    """Return data.split_seed if the manifest pins it (Protocol A onward)."""
    m = json.loads(manifest_path.read_text())
    cfg = m.get("config_resolved", {})
    data = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    if "split_seed" in data:
        return int(data["split_seed"])
    return None


def is_split_clean(manifest_path: Path) -> bool:
    """True iff the run trained AND evaluated on the canonical protein split.

    Protocol A (commit 6d685af) pins data.split_seed=42; earlier runs drew
    the training split from --seed while eval always rebuilt seed-42, so
    every *_s7/_s1337 run before the fix leaked ~86% of its 'test' pairs
    into training. Clean = split pinned to 42, or training seed itself 42
    (then train-split and eval-split coincide by construction).
    """
    seed = read_seed(manifest_path)
    if seed == 42:
        return True
    return read_split_seed(manifest_path) == 42


def top_k_attractors(M: np.ndarray, k: int = 10) -> set[int]:
    return set(np.argsort(-compute_attractor_scores(M))[:k].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / max(1, len(a | b))


# ──────────────────────────────────────────────────────────────────────────────
# Main aggregation
# ──────────────────────────────────────────────────────────────────────────────

def find_runs_by_config() -> dict[str, dict[int, Path]]:
    """Return {config_name: {seed: run_dir}} for the canonical 3-seed sweep."""
    runs_root = ROOT / "results" / "v5_rankbind"
    by_config: dict[str, dict[int, Path]] = {c: {} for c in CANONICAL_SEED42_TAG}

    for run_dir in sorted(runs_root.glob("*/manifest.json")):
        parsed = parse_run_id(run_dir.parent.name)
        if parsed is None:
            continue
        cfg_name, tag = parsed
        if cfg_name not in CANONICAL_SEED42_TAG:
            continue

        canonical_42 = CANONICAL_SEED42_TAG[cfg_name]
        # Accept runs whose tag matches {canonical seed-42 tag}, OR a seed-
        # override on the v4 sweep ({v4_s<seed>} — used by the existing
        # multiseed script for v3/v2 configs), OR a seed-override on the
        # config's own canonical sweep ({canonical_42}_s<seed>).
        seed_suffix_re = re.compile(r"_s(?:42|7|1337)$")
        if tag == canonical_42 or seed_suffix_re.search(tag):
            seed = read_seed(run_dir)
            if seed is None or seed not in SWEEP_SEEDS:
                continue
            # Protocol-A guard: skip runs evaluated on a split they trained
            # on (pre-6d685af *_s7/_s1337 runs — see commit 6d685af).
            if not is_split_clean(run_dir):
                continue
            # If duplicates exist for this (config, seed), keep the latest
            # (sorted() above guarantees chronological order, so last write wins).
            by_config[cfg_name][seed] = run_dir.parent

    return by_config


def compute_diagnostics(run_dir: Path, null_gini: float, null_top10: set[int]) -> dict:
    """Extract per-run metrics + Gini residual + Jaccard-vs-null."""
    man = json.loads((run_dir / "manifest.json").read_text())
    met = man.get("metrics", {})
    M = np.load(run_dir / "score_matrix_rankbind.npy")
    g = gini(compute_attractor_scores(M))
    j = jaccard(top_k_attractors(M), null_top10)
    return {
        "matrix_mrr":            float(met["matrix_mrr"]),
        "matrix_hit_at_1":       float(met["matrix_hit_at_1"]),
        "matrix_hit_at_5":       float(met["matrix_hit_at_5"]),
        "matrix_hit_at_10":      float(met["matrix_hit_at_10"]),
        "matrix_mean_rank_pct":  float(met["matrix_mean_rank_pct"]),
        "test_global_auc":       float(met["test_global_auc"]),
        "test_global_aupr":      float(met["test_global_aupr"]),
        "gini_residual":         float(g - null_gini),
        "jac_null":              float(j),
    }


def main() -> None:
    null_M = np.load(ROOT / "evaluation" / "attractor_results" / "score_matrix_null_prot_prior.npy")
    null_gini = gini(compute_attractor_scores(null_M))
    null_top10 = top_k_attractors(null_M)

    by_config = find_runs_by_config()

    rows = []
    for cfg_name, seed_to_dir in by_config.items():
        missing = [s for s in SWEEP_SEEDS if s not in seed_to_dir]
        if missing:
            print(f"[warn] {cfg_name}: missing seeds {missing}  (have {sorted(seed_to_dir)})")
        vals: dict[str, list[float]] = {k: [] for k in METRICS}
        for seed in SWEEP_SEEDS:
            if seed not in seed_to_dir:
                continue
            d = compute_diagnostics(seed_to_dir[seed], null_gini, null_top10)
            for k in METRICS:
                vals[k].append(d[k])
        n = len(vals["matrix_mrr"])
        if n == 0:
            continue
        head, cap = CONFIG_META[cfg_name]
        row = {"config": cfg_name, "head": head, "cap": cap, "n_seeds": n}
        for k in METRICS:
            row[f"{k}_mean"] = mean(vals[k])
            row[f"{k}_std"] = stdev(vals[k]) if n > 1 else 0.0
        rows.append(row)

    # Preserve paper-ordering: margin-recipe configs first, BCE controls,
    # then Phase-4 stage-(b) attn_pool variant.
    order = [
        "default", "abl_no_sampler", "abl_no_bilinear",
        "abl_no_margin", "abl_bce_only", "abl_mrrsel",
        "abl_attn_pool",
    ]
    rows.sort(key=lambda r: order.index(r["config"]))

    out_path = ROOT / "evaluation" / "attractor_results" / "phase2_rankbind_multiseed.csv"
    # Alias headers to the shorter names requested for the paper table.
    header_map = [
        ("config", "config"), ("head", "head"), ("cap", "cap"), ("n_seeds", "n_seeds"),
        ("matrix_mrr_mean",           "MRR_mean"),
        ("matrix_mrr_std",            "MRR_std"),
        ("matrix_hit_at_1_mean",      "H@1_mean"),
        ("matrix_hit_at_1_std",       "H@1_std"),
        ("matrix_hit_at_5_mean",      "H@5_mean"),
        ("matrix_hit_at_5_std",       "H@5_std"),
        ("matrix_hit_at_10_mean",     "H@10_mean"),
        ("matrix_hit_at_10_std",      "H@10_std"),
        ("matrix_mean_rank_pct_mean", "mrp_mean"),
        ("matrix_mean_rank_pct_std",  "mrp_std"),
        ("test_global_auc_mean",      "gAUC_mean"),
        ("test_global_auc_std",       "gAUC_std"),
        ("test_global_aupr_mean",     "gAUPR_mean"),
        ("test_global_aupr_std",      "gAUPR_std"),
        ("gini_residual_mean",        "Gini_resid_mean"),
        ("gini_residual_std",         "Gini_resid_std"),
        ("jac_null_mean",             "Jac_null_mean"),
        ("jac_null_std",              "Jac_null_std"),
    ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([alias for _, alias in header_map])
        for r in rows:
            w.writerow([r[src] for src, _ in header_map])
    print(f"[ok] wrote {out_path} ({len(rows)} configs)")

    # Pretty-print mean ± std for the three primary metrics.
    print()
    print(f"{'config':18s} {'n':>2}  {'MRR':>20s}  {'H@5':>20s}  {'H@10':>20s}  {'gAUC':>20s}  {'Gini_resid':>20s}")
    for r in rows:
        def fmt(k): return f"{r[k+'_mean']:.3f} ± {r[k+'_std']:.3f}"
        print(f"{r['config']:18s} {r['n_seeds']:>2}  "
              f"{fmt('matrix_mrr'):>20s}  {fmt('matrix_hit_at_5'):>20s}  "
              f"{fmt('matrix_hit_at_10'):>20s}  {fmt('test_global_auc'):>20s}  "
              f"{fmt('gini_residual'):>20s}")


if __name__ == "__main__":
    main()
