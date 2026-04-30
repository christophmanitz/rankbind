"""evaluation/attn_weight_inspection.py — Stage-(b) interpretability check.

Loads each of the 3 attn_pool runs (seeds 42 / 7 / 1337), extracts the
per-residue attention weights from `ResidueAttentionPool` for a sample of
proteins, and quantifies two things:

  1. Concentration: how peaked are the weights? (entropy + top-K% mass)
  2. Cross-seed agreement: do the 3 independently-trained models attend to
     similar residue positions on the *same* protein?

Outputs (under evaluation/attractor_results/):

  - attn_weights_concentration.csv  — per (run, protein): n_residues, entropy,
                                       top1_pos, top5pct_mass, top10pct_mass,
                                       top20pct_mass
  - attn_weights_cross_seed.csv     — per protein: spearman + top-K-overlap
                                       between each pair of seeds
  - fig_attn_weight_examples.png    — 6 proteins × 3 seeds overlaid weight curves
  - fig_attn_concentration_hist.png — top-10% mass histogram (per seed)
  - fig_attn_cross_seed_agreement.png — pairwise spearman + top-10 overlap dist
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v5_rankbind.run_manifest import load_config
from v5_rankbind.model import RankBind

OUT_DIR = _HERE / "attractor_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Three attn_pool runs from the v5b sweep.
RUNS = {
    42:   PROJECT_ROOT / "results" / "v5_rankbind" / "20260427-121113_1746525d51_abl_attn_pool_v5b_s42",
    7:    PROJECT_ROOT / "results" / "v5_rankbind" / "20260427-121212_1746525d51_abl_attn_pool_v5b_s7",
    1337: PROJECT_ROOT / "results" / "v5_rankbind" / "20260427-121250_1746525d51_abl_attn_pool_v5b_s1337",
}

ESM2_DIR = PROJECT_ROOT / "data" / "esm2_embeddings"
N_SAMPLE_PROTEINS = 60       # how many proteins to inspect
N_PLOT_EXAMPLES   = 6        # how many to show on the example figure
RNG = np.random.default_rng(42)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def load_run_model(run_dir: Path, device: str = "cpu") -> tuple[RankBind, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = load_config(manifest["config_path"])
    model = RankBind(cfg).to(device)
    model.load_state_dict(
        torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    )
    model.eval()
    return model, cfg


@torch.no_grad()
def get_attn_weights(model: RankBind, residues: torch.Tensor) -> np.ndarray:
    """residues [L, D] (un-batched). Returns [L] attention weights."""
    x = residues.unsqueeze(0)                                     # [1, L, D]
    mask = torch.ones(1, residues.shape[0], dtype=torch.bool)
    _, w = model.attn_pool(x, mask)
    return w.squeeze(0).cpu().numpy()                             # [L]


def shannon_entropy(p: np.ndarray) -> float:
    """Natural-log entropy. p must be a valid distribution."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def top_k_pct_mass(p: np.ndarray, pct: float) -> float:
    """Fraction of total mass on top-`pct` of positions (e.g. pct=0.10 → top 10%)."""
    n = len(p)
    k = max(1, int(round(n * pct)))
    return float(np.sort(p)[-k:].sum())


def sample_proteins() -> list[str]:
    all_files = sorted(p.stem for p in ESM2_DIR.glob("*.pt"))
    # Reservoir sample for reproducibility.
    take = RNG.choice(all_files, size=min(N_SAMPLE_PROTEINS, len(all_files)),
                      replace=False)
    return [str(x) for x in take]


def load_residues(uniprot: str, max_len: int = 1024) -> torch.Tensor:
    f = ESM2_DIR / f"{uniprot}.pt"
    t = torch.load(f, weights_only=True).to(torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.shape[0] > max_len:
        t = t[:max_len]
    return t


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    proteins = sample_proteins()
    print(f"[setup] sampling {len(proteins)} proteins from {ESM2_DIR}")

    # weights[seed][uniprot] = np.ndarray [L]
    weights: dict[int, dict[str, np.ndarray]] = {s: {} for s in RUNS}
    rows = []
    for seed, run_dir in RUNS.items():
        print(f"[run] loading seed={seed} from {run_dir.name}")
        model, _cfg = load_run_model(run_dir)
        for uni in proteins:
            try:
                resi = load_residues(uni)
            except Exception as e:
                print(f"  [skip] {uni}: {e}")
                continue
            w = get_attn_weights(model, resi)
            weights[seed][uni] = w
            rows.append({
                "seed":           seed,
                "uniprot":        uni,
                "n_residues":     int(len(w)),
                "entropy":        shannon_entropy(w),
                "entropy_uniform":float(np.log(len(w))),  # log L
                "entropy_ratio":  shannon_entropy(w) / float(np.log(len(w))),
                "top1_pos":       int(np.argmax(w)),
                "top1_pos_pct":   float(np.argmax(w) / len(w)),
                "top1_mass":      float(np.max(w)),
                "top5pct_mass":   top_k_pct_mass(w, 0.05),
                "top10pct_mass":  top_k_pct_mass(w, 0.10),
                "top20pct_mass":  top_k_pct_mass(w, 0.20),
            })
    conc_df = pd.DataFrame(rows)
    conc_csv = OUT_DIR / "attn_weights_concentration.csv"
    conc_df.to_csv(conc_csv, index=False)
    print(f"\n[ok] wrote {conc_csv}  ({len(conc_df)} rows)")

    # Per-seed concentration summary
    print("\nPer-seed concentration summary:")
    print(conc_df.groupby("seed")[
        ["entropy_ratio", "top5pct_mass", "top10pct_mass", "top20pct_mass"]
    ].agg(["mean", "median"]).round(3).to_string())

    # ── Cross-seed agreement ────────────────────────────────────────────────
    cross_rows = []
    for uni in proteins:
        if not all(uni in weights[s] for s in RUNS):
            continue
        ws = {s: weights[s][uni] for s in RUNS}
        # Spearman rank-correlation between each pair of seeds' weights
        seed_keys = sorted(RUNS.keys())
        pairs = [(a, b) for i, a in enumerate(seed_keys) for b in seed_keys[i + 1:]]
        row = {"uniprot": uni, "n_residues": len(ws[seed_keys[0]])}
        for a, b in pairs:
            rho = spearmanr(ws[a], ws[b]).correlation
            row[f"spearman_{a}_vs_{b}"] = float(rho)
            # Top-K residue-position overlap
            for K_pct in (0.05, 0.10):
                K = max(1, int(round(len(ws[a]) * K_pct)))
                top_a = set(np.argsort(-ws[a])[:K].tolist())
                top_b = set(np.argsort(-ws[b])[:K].tolist())
                jacc = len(top_a & top_b) / max(1, len(top_a | top_b))
                row[f"top{int(K_pct*100)}pct_jacc_{a}_vs_{b}"] = float(jacc)
        cross_rows.append(row)
    cross_df = pd.DataFrame(cross_rows)
    cross_csv = OUT_DIR / "attn_weights_cross_seed.csv"
    cross_df.to_csv(cross_csv, index=False)
    print(f"\n[ok] wrote {cross_csv}  ({len(cross_df)} rows)")

    spearman_cols = [c for c in cross_df.columns if c.startswith("spearman_")]
    jacc10_cols = [c for c in cross_df.columns if c.startswith("top10pct_jacc_")]
    print("\nCross-seed agreement summary (3 seed-pairs):")
    print(f"  median spearman:    {cross_df[spearman_cols].stack().median():.3f}")
    print(f"  median top-10% jacc:{cross_df[jacc10_cols].stack().median():.3f}")
    print(f"  uniform-baseline jacc (random top-10%): ~ 0.05–0.10")

    # ── Plot 1: example weight curves for 6 proteins, 3 seeds overlaid ─────
    sample_for_plot = list(weights[42].keys())[:N_PLOT_EXAMPLES]
    fig, axes = plt.subplots(N_PLOT_EXAMPLES, 1, figsize=(11, 1.7 * N_PLOT_EXAMPLES),
                             sharex=False, constrained_layout=True)
    if N_PLOT_EXAMPLES == 1:
        axes = [axes]
    seed_colors = {42: "#1f77b4", 7: "#2ca02c", 1337: "#d62728"}
    for ax, uni in zip(axes, sample_for_plot):
        for seed in sorted(RUNS.keys()):
            if uni in weights[seed]:
                w = weights[seed][uni]
                ax.plot(np.arange(len(w)), w, color=seed_colors[seed],
                        label=f"seed={seed}", alpha=0.75, lw=1.2)
        ax.set_title(uni, fontsize=10)
        ax.set_ylabel("attn", fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("residue position", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Attention weights along residue position — 3 seeds per protein",
                 fontsize=12)
    p1 = OUT_DIR / "fig_attn_weight_examples.png"
    plt.savefig(p1, dpi=140); plt.close()
    print(f"[ok] wrote {p1}")

    # ── Plot 2: top-10% mass distribution per seed ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 31)
    for seed in sorted(RUNS.keys()):
        sub = conc_df[conc_df["seed"] == seed]["top10pct_mass"]
        ax.hist(sub, bins=bins, alpha=0.55, label=f"seed={seed} (med={sub.median():.2f})",
                color=seed_colors[seed], edgecolor="black", linewidth=0.4)
    ax.axvline(0.10, color="black", linestyle=":", alpha=0.6,
               label="uniform (0.10)")
    ax.set_xlabel("Fraction of total attention mass on top-10% of residues")
    ax.set_ylabel("Count of proteins")
    ax.set_title("How peaked are the attention weights?  (bigger = more peaked)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    p2 = OUT_DIR / "fig_attn_concentration_hist.png"
    plt.savefig(p2, dpi=140); plt.close()
    print(f"[ok] wrote {p2}")

    # ── Plot 3: pairwise spearman + top-10% jaccard across seeds ───────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    sp_vals = cross_df[spearman_cols].values.flatten()
    jc_vals = cross_df[jacc10_cols].values.flatten()
    axes[0].hist(sp_vals, bins=np.linspace(-0.2, 1.0, 25),
                 color="#888", edgecolor="black", linewidth=0.4)
    axes[0].axvline(np.median(sp_vals), color="red", lw=1.5,
                    label=f"median={np.median(sp_vals):.2f}")
    axes[0].axvline(0.0, color="black", linestyle=":", alpha=0.6,
                    label="independence (0)")
    axes[0].set_xlabel("Spearman ρ between seeds' attention weights")
    axes[0].set_ylabel("Count (per protein × per seed-pair)")
    axes[0].set_title("Cross-seed agreement: rank-correlation of weights")
    axes[0].legend(fontsize=9)
    axes[1].hist(jc_vals, bins=np.linspace(0, 1, 25),
                 color="#888", edgecolor="black", linewidth=0.4)
    axes[1].axvline(np.median(jc_vals), color="red", lw=1.5,
                    label=f"median={np.median(jc_vals):.2f}")
    axes[1].axvline(0.10, color="black", linestyle=":", alpha=0.6,
                    label="random expectation (~0.10)")
    axes[1].set_xlabel("Jaccard of top-10% residue sets")
    axes[1].set_ylabel("Count (per protein × per seed-pair)")
    axes[1].set_title("Cross-seed agreement: top-10% identity overlap")
    axes[1].legend(fontsize=9)
    p3 = OUT_DIR / "fig_attn_cross_seed_agreement.png"
    plt.savefig(p3, dpi=140); plt.close()
    print(f"[ok] wrote {p3}")

    # ── Gate inputs ────────────────────────────────────────────────────────
    print("\n=== INTERPRETABILITY GATE INPUTS (PLAN §13.2) ===")
    print(f"Median top-10% mass per seed (concentration):")
    for seed in sorted(RUNS.keys()):
        sub = conc_df[conc_df["seed"] == seed]
        print(f"  seed={seed}: top-10% mass median = {sub['top10pct_mass'].median():.3f},  "
              f"top-20% = {sub['top20pct_mass'].median():.3f},  "
              f"entropy ratio (1=uniform) = {sub['entropy_ratio'].median():.3f}")
    print(f"\nCross-seed median spearman = {np.median(sp_vals):.3f}  "
          f"(random ~0, identical=1)")
    print(f"Cross-seed median top-10% jaccard = {np.median(jc_vals):.3f}  "
          f"(random ~0.10, identical=1)")
    print("\nGate criterion: weights concentrate on <20% of residues with "
          "qualitative pocket overlap on 2-3 spot-checked proteins.")


if __name__ == "__main__":
    main()
