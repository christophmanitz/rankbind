"""evaluation/attn_pocket_overlap.py — does Stage-(b) attention track the pocket?

The deferred validation from sec:attn (main.tex): the attention-weight audit
showed near-uniform magnitude but a cross-seed-reproducible *rank order*. This
script tests whether that reproducible rank order aligns with the protein's
actual catalytic / binding residues, as annotated by UniProt (active site +
binding site features).

For each protein we treat annotated active/binding residues as positives and
the model's per-residue attention weight as a score, then ask: does the weight
rank annotated residues above background?

Metrics per protein:
  - roc_auc        : ROC-AUC of attention weight predicting annotated residues
                     (0.5 = chance; >0.5 = attention enriched on the pocket)
  - mean_pctile    : mean percentile-rank of annotated residues' weights
                     (0.5 = chance)
  - enrichment     : mean weight on annotated / mean weight overall (1.0 = none)
  - enrichment_top10 : fraction of annotated residues that fall in the protein's
                     top-10% most-attended residues / 0.10 (1.0 = chance)

We compute these (a) per seed and (b) on the cross-seed mean weight (the
"consensus" signal the paper says is the reproducible part). UniProt
1-based residue position p maps to array index p-1 (verified: ESM2 row count
== UniProt length, no CLS/EOS offset).

Output: evaluation/attractor_results/attn_pocket_overlap.csv  (+ stdout summary)

Usage:
  python -m evaluation.attn_pocket_overlap                 # the 6 figure proteins
  python -m evaluation.attn_pocket_overlap --all           # all 60 sampled
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the exact run dirs / helpers from the concentration audit.
from evaluation.attn_weight_inspection import (  # noqa: E402
    RUNS, ESM2_DIR, RNG, N_SAMPLE_PROTEINS,
    load_run_model, get_attn_weights, load_residues, sample_proteins,
)

OUT_DIR = _HERE / "attractor_results"

# The 6 proteins plotted in fig_attn_weight_examples.png are the first 6 of the
# (seed-42) sample order.
FIG_PROTEINS = ["Q77J78", "A0A1L9P8I4", "O25046", "P08311", "Q65MI2", "Q8A6L0"]

# UniProt feature types that mark a residue as part of the catalytic / binding
# machinery (the "pocket" ground truth).
POCKET_TYPES = {"Active site", "Binding site"}


def fetch_pocket_residues(uniprot: str) -> tuple[set[int], int, dict]:
    """Return (0-based positive indices, seq length, per-type counts)."""
    url = (f"https://rest.uniprot.org/uniprotkb/{uniprot}.json"
           f"?fields=length,ft_act_site,ft_binding,ft_site")
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.loads(r.read().decode())
    length = int(d.get("sequence", {}).get("length", 0))
    pos: set[int] = set()
    counts: dict[str, int] = {}
    for f in d.get("features", []):
        t = f["type"]
        if t not in POCKET_TYPES:
            continue
        s = int(f["location"]["start"]["value"])
        e = int(f["location"]["end"]["value"])
        for p in range(s, e + 1):           # 1-based, inclusive
            pos.add(p - 1)                  # -> 0-based array index
        counts[t] = counts.get(t, 0) + (e - s + 1)
    return pos, length, counts


def overlap_metrics(weight: np.ndarray, positives: set[int]) -> dict:
    """weight [L]; positives = set of 0-based indices. Chance baselines noted."""
    L = len(weight)
    pos = sorted(p for p in positives if 0 <= p < L)
    if not pos or len(pos) == L:
        return {}
    y = np.zeros(L, dtype=int)
    y[pos] = 1
    # ROC-AUC of weight predicting annotated residue
    auc = float(roc_auc_score(y, weight))
    # percentile rank of each residue's weight (0..1), then mean over positives
    order = np.argsort(np.argsort(weight))          # rank 0..L-1
    pctile = order / (L - 1)
    mean_pctile = float(pctile[pos].mean())
    # enrichment of mean weight on pocket vs overall
    enrichment = float(weight[pos].mean() / weight.mean())
    # fraction of pocket residues in the top-10% most-attended, / chance (0.10)
    k = max(1, int(round(L * 0.10)))
    top = set(np.argsort(-weight)[:k].tolist())
    frac_top = len(set(pos) & top) / len(pos)
    enrich_top10 = frac_top / 0.10
    return {
        "n_residues": L, "n_pocket": len(pos),
        "roc_auc": auc, "mean_pctile": mean_pctile,
        "enrichment": enrichment, "enrich_top10": enrich_top10,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="use all 60 sampled proteins (default: 6 figure proteins)")
    args = ap.parse_args()

    if args.all:
        proteins = sample_proteins()
        label = f"all {len(proteins)} sampled"
    else:
        proteins = FIG_PROTEINS
        label = f"{len(proteins)} figure proteins"
    print(f"[setup] pocket-overlap on {label}")

    # Per-residue weights, all seeds.
    weights: dict[int, dict[str, np.ndarray]] = {s: {} for s in RUNS}
    for seed, run_dir in RUNS.items():
        model, _ = load_run_model(run_dir)
        for uni in proteins:
            try:
                w = get_attn_weights(model, load_residues(uni))
            except Exception as e:
                print(f"  [skip] s{seed} {uni}: {e}")
                continue
            weights[seed][uni] = w

    rows = []
    for uni in proteins:
        try:
            pos, uni_len, counts = fetch_pocket_residues(uni)
        except Exception as e:
            print(f"  [skip annot] {uni}: {e}")
            continue
        if not pos:
            print(f"  [no annot] {uni}: no active/binding-site features")
            continue
        # consensus = mean weight across the seeds that have this protein
        avail = [s for s in RUNS if uni in weights[s]]
        if not avail:
            continue
        L0 = len(weights[avail[0]][uni])
        if uni_len != L0:
            print(f"  [len mismatch] {uni}: UniProt {uni_len} vs ESM2 {L0} — skip")
            continue
        mean_w = np.mean([weights[s][uni] for s in avail], axis=0)

        m = overlap_metrics(mean_w, pos)
        if not m:
            continue
        m.update({"uniprot": uni, "scope": "consensus",
                  "annot": ";".join(f"{k}:{v}" for k, v in counts.items())})
        rows.append(m)
        for s in avail:
            ms = overlap_metrics(weights[s][uni], pos)
            ms.update({"uniprot": uni, "scope": f"seed{s}", "annot": ""})
            rows.append(ms)

    if not rows:
        print("[done] no proteins had usable annotations.")
        return

    df = pd.DataFrame(rows)
    suffix = "_all" if args.all else ""
    out = OUT_DIR / f"attn_pocket_overlap{suffix}.csv"
    df.to_csv(out, index=False)
    print(f"\n[ok] wrote {out}  ({len(df)} rows)")

    con = df[df.scope == "consensus"]
    print(f"\n=== Consensus (cross-seed mean weight), n={len(con)} proteins "
          f"with annotations ===")
    print(con[["uniprot", "n_residues", "n_pocket", "roc_auc",
               "mean_pctile", "enrichment", "enrich_top10"]]
          .round(3).to_string(index=False))
    print("\nChance baselines: roc_auc 0.50, mean_pctile 0.50, "
          "enrichment 1.00, enrich_top10 1.00")
    print(f"\nMean  roc_auc      = {con.roc_auc.mean():.3f}")
    print(f"Mean  mean_pctile  = {con.mean_pctile.mean():.3f}")
    print(f"Mean  enrichment   = {con.enrichment.mean():.3f}")
    print(f"Mean  enrich_top10 = {con.enrich_top10.mean():.3f}")
    if len(con) >= 5:
        try:
            stat, p = wilcoxon(con.roc_auc - 0.5)
            print(f"\nWilcoxon (roc_auc vs 0.5): p = {p:.4f}  (n={len(con)})")
        except ValueError as e:
            print(f"\nWilcoxon skipped: {e}")


if __name__ == "__main__":
    main()
