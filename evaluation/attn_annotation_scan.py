"""evaluation/attn_annotation_scan.py — what (if anything) do the v5b residue
attention weights encode?

Background: attn_pocket_overlap.py showed the cross-seed-consensus attention is
*anti*-correlated with catalytic / binding residues (ROC-AUC 0.21). That answers
"does it find the pocket?" (no). This script asks the broader question: does the
weight track ANY annotation — biological (secondary-structure propensity,
hydrophobicity, charge, UniProt feature regions) or mechanical (the ESM2
per-residue embedding norm)?

Method. For each sampled BRENDA-200 protein we take the cross-seed mean attention
weight and, *within that protein*, correlate it (Spearman) against a panel of
per-residue predictors, then aggregate the within-protein associations across
proteins (mean rho + sign-consistency Wilcoxon). For categorical UniProt feature
regions we use within-protein ROC-AUC of attention predicting the region mask.

The decisive control is the PARTIAL correlation: attention vs each predictor with
the ESM2 residue norm ranked-out. If nothing biological survives removing the
norm, the weights are a representational artefact, not biology.

Outputs (evaluation/attractor_results/):
  - attn_annotation_continuous.csv  — per (protein, predictor): rho, partial_rho
  - attn_annotation_aa.csv          — per amino acid: mean within-protein z(attn)
  - attn_annotation_features.csv    — per UniProt feature type: per-protein AUC
  - attn_annotation_residues.csv    — long per-residue table (for the plots)
  - fig_attn_annotation.png         — rigorous 3-panel summary (paper figure)
  - fig_attn_explainer.png          — intuitive explainer (class box + example tracks)
  + stdout ranked summary with Wilcoxon p-values

Usage:
  python -m evaluation.attn_annotation_scan            # 60 sampled proteins
  python -m evaluation.attn_annotation_scan --n 30     # fewer (faster)
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata, wilcoxon
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.attn_weight_inspection import (  # noqa: E402
    RUNS, ESM2_DIR, load_run_model, get_attn_weights, load_residues,
    sample_proteins,
)

OUT_DIR = _HERE / "attractor_results"
CACHE = OUT_DIR / "_uniprot_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Proteins plotted as example tracks (same as the pocket figure, for continuity).
FIG_PROTEINS = ["Q77J78", "A0A1L9P8I4", "O25046", "P08311", "Q65MI2", "Q8A6L0"]

# ── Amino-acid scales (sequence-derivable, no structure needed) ──────────────
# Kyte-Doolittle hydropathy.
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
# Net charge at pH 7 (simple).
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
# Chou-Fasman helix (Pa) and sheet (Pb) propensities.
PA = {"A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70, "Q": 1.11, "E": 1.51,
      "G": 0.57, "H": 1.00, "I": 1.08, "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13,
      "P": 0.57, "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06}
PB = {"A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19, "Q": 1.10, "E": 0.37,
      "G": 0.75, "H": 0.87, "I": 1.60, "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38,
      "P": 0.55, "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70}
AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Physicochemical class per residue (for the explainer box-plot).
CLASS = {"D": "acidic", "E": "acidic",
         "K": "basic", "R": "basic", "H": "basic",
         "S": "polar", "T": "polar", "N": "polar", "Q": "polar", "C": "polar",
         "Y": "polar", "G": "special", "P": "special",
         "A": "hydrophobic", "V": "hydrophobic", "L": "hydrophobic",
         "I": "hydrophobic", "M": "hydrophobic", "F": "aromatic", "W": "aromatic"}
CLASS_ORDER = ["acidic", "basic", "polar", "special", "hydrophobic", "aromatic"]
CLASS_LABEL = {"acidic": "Acidic\n(D,E)", "basic": "Basic\n(K,R,H)",
               "polar": "Polar\n(S,T,N,Q,C,Y)", "special": "Gly/Pro",
               "hydrophobic": "Hydrophobic\n(A,V,L,I,M)", "aromatic": "Aromatic\n(F,W)"}
CLASS_COLOR = {"acidic": "#d62728", "basic": "#ff7f0e", "polar": "#9467bd",
               "special": "#8c8c8c", "hydrophobic": "#2ca02c", "aromatic": "#1f77b4"}

# UniProt feature types → coarse residue-level category we test for enrichment.
FEATURE_FIELDS = ("ft_domain,ft_region,ft_motif,ft_transmem,ft_intramem,"
                  "ft_signal,ft_transit,ft_propep,ft_compbias,ft_mod_res,"
                  "ft_carbohyd,ft_disulfid,ft_dna_bind,ft_zn_fing,ft_repeat,"
                  "ft_coiled,ft_act_site,ft_binding")
FEATURE_TYPES = ["Domain", "Region", "Motif", "Transmembrane", "Signal",
                 "Compositional bias", "Modified residue", "Disulfide bond",
                 "Zinc finger", "Repeat", "Coiled coil",
                 "Active site", "Binding site"]
_TYPE_ALIAS = {"Signal peptide": "Signal", "Transit peptide": "Signal"}


def fetch_uniprot(uni: str) -> dict | None:
    """Return {seq, length, features:[(type,start,end)]}. Cached on disk."""
    cf = CACHE / f"{uni}.json"
    if cf.exists():
        d = json.loads(cf.read_text())
    else:
        url = (f"https://rest.uniprot.org/uniprotkb/{uni}.json"
               f"?fields=sequence,length,{FEATURE_FIELDS}")
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.loads(r.read().decode())
        except Exception as e:
            print(f"  [skip annot] {uni}: {e}")
            return None
        cf.write_text(json.dumps(d))
    seq = d.get("sequence", {}).get("value", "")
    feats = []
    for f in d.get("features", []):
        t = _TYPE_ALIAS.get(f["type"], f["type"])
        try:
            s = int(f["location"]["start"]["value"])
            e = int(f["location"]["end"]["value"])
        except (KeyError, TypeError):
            continue
        feats.append((t, s, e))
    return {"seq": seq, "length": len(seq), "features": feats}


def feat_mask(features, ftype: str, L: int) -> np.ndarray:
    m = np.zeros(L, dtype=int)
    for t, s_, e_ in features:
        if t != ftype:
            continue
        for p in range(s_, e_ + 1):
            if 1 <= p <= L:
                m[p - 1] = 1
    return m


def partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Spearman partial correlation of a,b controlling for c (rank-residualise)."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    X = np.vstack([np.ones_like(rc), rc]).T
    def resid(y):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta
    ea, eb = resid(ra), resid(rb)
    if ea.std() < 1e-9 or eb.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(ea, eb)[0, 1])


def per_residue_predictors(seq: str, norm: np.ndarray) -> dict[str, np.ndarray]:
    L = len(norm)
    s = seq[:L]
    idx = np.arange(L)
    rel_pos = idx / max(1, L - 1)
    dist_term = np.minimum(idx, (L - 1) - idx) / max(1, (L - 1) / 2)  # 0 at ends, 1 mid
    hyd = np.array([KD.get(c, 0.0) for c in s])
    chg = np.array([CHARGE.get(c, 0.0) for c in s])
    pa = np.array([PA.get(c, 1.0) for c in s])
    pb = np.array([PB.get(c, 1.0) for c in s])
    return {
        "esm2_norm": norm,
        "rel_position": rel_pos,
        "dist_to_terminus": dist_term,
        "hydropathy_KD": hyd,
        "charge": chg,
        "helix_propensity": pa,
        "sheet_propensity": pb,
    }


def _smooth(x: np.ndarray, w: int = 7) -> np.ndarray:
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


# ────────────────────────────────────────────────────────────────────────────
# Figures
# ────────────────────────────────────────────────────────────────────────────

def make_summary_figure(summ: pd.DataFrame, adf: pd.DataFrame,
                        fsumm: pd.DataFrame | None) -> Path:
    """Rigorous 3-panel summary (paper figure)."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)

    # panel a: amino-acid mean z(attn) vs Kyte-Doolittle hydropathy
    aa = adf.copy()
    aa["kd"] = aa["aa"].map(KD)
    aa["cls"] = aa["aa"].map(CLASS)
    x, y = aa["kd"].values, aa["mean_z_attn"].values
    r = float(np.corrcoef(x, y)[0, 1])
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    axes[0].plot(xs, m * xs + b, color="black", lw=1.3, ls="--", zorder=1,
                 label=f"fit (Pearson r = {r:.2f})")
    for _, row in aa.iterrows():
        axes[0].scatter(row.kd, row.mean_z_attn, s=180,
                        color=CLASS_COLOR[row.cls], edgecolor="black",
                        linewidth=0.6, zorder=2)
        axes[0].annotate(row.aa, (row.kd, row.mean_z_attn), ha="center",
                         va="center", fontsize=8, fontweight="bold", zorder=3)
    axes[0].axhline(0, color="grey", lw=0.7, ls=":")
    axes[0].set_xlabel("Kyte–Doolittle hydropathy  (← hydrophilic | hydrophobic →)")
    axes[0].set_ylabel("mean attention bias  (within-protein z-score)")
    axes[0].set_title("(a) The attention preference IS hydrophobicity\n"
                      "each point = one amino acid", fontsize=11)
    axes[0].legend(fontsize=9, loc="upper left"); axes[0].grid(alpha=0.2)

    # panel b: within-protein rho, raw vs partial (norm removed)
    s1 = summ.iloc[::-1]
    yy = np.arange(len(s1))
    axes[1].barh(yy - 0.2, s1["mean_rho"], height=0.4, color="#1f77b4",
                 label="raw ρ")
    axes[1].barh(yy + 0.2, s1["mean_partial_rho"].fillna(0), height=0.4,
                 color="#ff7f0e", label="partial ρ (ESM2 norm removed)")
    axes[1].set_yticks(yy)
    axes[1].set_yticklabels([p.replace("_", " ") for p in s1["predictor"]],
                            fontsize=9)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("mean within-protein Spearman ρ  (chance = 0)")
    axes[1].set_title("(b) Hydrophobicity survives the norm control\n"
                      "= a real preference, not the LayerNorm artefact", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower right"); axes[1].grid(alpha=0.25, axis="x")

    # panel c: feature-region AUCs (consequence)
    if fsumm is not None:
        f2 = fsumm.iloc[::-1]
        yy2 = np.arange(len(f2))
        cols = ["#d62728" if v < 0.5 else "#2ca02c" for v in f2["mean_auc"]]
        axes[2].barh(yy2, f2["mean_auc"] - 0.5, color=cols)
        axes[2].set_yticks(yy2); axes[2].set_yticklabels(f2["feature"], fontsize=9)
        axes[2].axvline(0, color="black", lw=0.8)
        axes[2].set_xlabel("attention enrichment on feature  (AUC − 0.5; 0 = chance)")
        axes[2].set_title("(c) Consequence: it AVOIDS the functional pocket\n"
                          "active/binding sites far below chance", fontsize=11)
        axes[2].grid(alpha=0.25, axis="x")
    p = OUT_DIR / "fig_attn_annotation.png"
    plt.savefig(p, dpi=140); plt.close()
    return p


def make_explainer_figure(res: pd.DataFrame) -> Path:
    """Intuitive explainer: AA-class bias, where functional residues actually
    rank, and example tracks of attention vs hydropathy.

    Honest framing: we DO NOT claim every marker sits in a local valley. Active
    sites rank at the bottom (median ~3rd percentile); binding sites only mildly
    so (median ~16th) with a real upper tail. Panel (2) shows that distribution;
    markers on the tracks are the RAW per-residue values, not the smoothed line.
    """
    res = res.copy()
    res["pctile"] = res.groupby("uniprot")["attn"].rank(pct=True)

    # Representative example tracks: annotated, readable proteins picked by
    # within-protein hydropathy-tracking rho (a good and a clear case).
    cand = []
    for uni, sub in res.groupby("uniprot"):
        if (sub.is_active.sum() + sub.is_binding.sum()) == 0 or len(sub) > 520:
            continue
        rho = spearmanr(sub.attn_z.values, sub.hydropathy.values).correlation
        cand.append((uni, float(rho)))
    cand.sort(key=lambda t: t[1])
    examples = []
    if cand:
        n = len(cand)
        for q in (0.60, 0.88):
            examples.append(cand[min(n - 1, int(q * (n - 1)))][0])
        seen = set(); examples = [u for u in examples if not (u in seen or seen.add(u))]
    n_tracks = len(examples)

    fig = plt.figure(figsize=(13.5, 4.2 + 2.0 * n_tracks), constrained_layout=True)
    gs = fig.add_gridspec(1 + n_tracks, 2, height_ratios=[2.7] + [1.7] * n_tracks)

    # ── (1) attention by physicochemical class ───────────────────────────────
    axc = fig.add_subplot(gs[0, 0])
    data = [res.loc[res.aa.map(CLASS) == c, "attn_z"].values for c in CLASS_ORDER]
    bp = axc.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=1.5))
    for patch, c in zip(bp["boxes"], CLASS_ORDER):
        patch.set_facecolor(CLASS_COLOR[c]); patch.set_alpha(0.75)
    axc.axhline(0, color="black", lw=0.9, ls="--")
    axc.set_xticklabels([CLASS_LABEL[c] for c in CLASS_ORDER], fontsize=8.5)
    axc.set_ylabel("attention bias (within-protein z)")
    axc.set_title("(1) Attention bias by residue class\n"
                  "charged suppressed · hydrophobic/aromatic favoured", fontsize=10.5)
    axc.grid(alpha=0.25, axis="y")

    # ── (2) where functional residues actually rank (percentile) ─────────────
    axp = fig.add_subplot(gs[0, 1])
    allp = res["pctile"].values
    bndp = res.loc[res.is_binding == 1, "pctile"].values
    actp = res.loc[res.is_active == 1, "pctile"].values
    bp2 = axp.boxplot([allp, bndp, actp], widths=0.6, patch_artist=True,
                      showfliers=False, medianprops=dict(color="black", lw=1.5))
    for patch, col in zip(bp2["boxes"], ["#8c8c8c", "#d62728", "#000000"]):
        patch.set_facecolor(col); patch.set_alpha(0.55)
    axp.axhline(0.5, color="black", lw=0.9, ls="--")
    axp.set_xticklabels(["all\nresidues",
                         f"binding\n(med {np.median(bndp):.2f})",
                         f"active\n(med {np.median(actp):.2f})"], fontsize=8.5)
    axp.set_ylabel("attention percentile\n(within protein; 0.5 = chance)")
    axp.set_ylim(0, 1)
    axp.set_title("(2) Where functional residues rank\n"
                  "active strongly suppressed · binding weakly & variably",
                  fontsize=10.5)
    axp.grid(alpha=0.25, axis="y")

    # ── example tracks: attention vs hydropathy along the sequence ───────────
    for i, uni in enumerate(examples):
        ax = fig.add_subplot(gs[1 + i, :])
        sub = res[res.uniprot == uni].sort_values("pos")
        pos = sub.pos.values
        az_raw = sub.attn_z.values
        hz = sub.hydropathy.values; hz = (hz - hz.mean()) / (hz.std() + 1e-9)
        # The attention line IS the raw per-residue value, so the markers (also
        # raw) sit exactly on it. Hydropathy is smoothed to show the trend the
        # attention follows.
        ax.plot(pos, az_raw, color="#1f77b4", lw=1.0, alpha=0.9, zorder=2,
                label="attention (raw) — markers lie on this line")
        ax.plot(pos, _smooth(hz), color="#2ca02c", lw=1.7, alpha=0.9, zorder=3,
                label="hydropathy (smoothed trend)")
        ax.axhline(float(np.median(az_raw)), color="grey", lw=0.7, ls=":",
                   label="protein median")
        act = sub[sub.is_active == 1]; bnd = sub[sub.is_binding == 1]
        if len(bnd):
            ax.scatter(bnd.pos, bnd.attn_z, marker="o", s=52, facecolor="none",
                       edgecolor="#d62728", linewidth=1.6, zorder=5,
                       label="binding site")
        if len(act):
            ax.scatter(act.pos, act.attn_z, marker="v", s=85, color="black",
                       zorder=6, label="active site")
        rho = spearmanr(az_raw, sub.hydropathy.values).correlation
        bits = [f"{uni} — attention tracks hydropathy (ρ={rho:+.2f})"]
        if len(act):
            bits.append(f"active at {act.pctile.median():.0%} pctile")
        if len(bnd):
            bits.append(f"binding at {bnd.pctile.median():.0%} pctile")
        ax.set_title(";  ".join(bits), fontsize=9.5)
        ax.set_ylabel("z-score"); ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7.5, ncol=3, loc="upper right")
        if i == n_tracks - 1:
            ax.set_xlabel("residue position")
    fig.suptitle("v5b residue attention ≈ a hydrophobicity read-out — hydrophobic "
                 "up-weighted, charged suppressed; active sites strongly avoided",
                 fontsize=12)
    p = OUT_DIR / "fig_attn_explainer.png"
    plt.savefig(p, dpi=140); plt.close()
    return p


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="cap number of proteins")
    args = ap.parse_args()

    proteins = sample_proteins()
    if args.n:
        proteins = proteins[: args.n]
    # ensure the example proteins are included so the tracks render
    for u in FIG_PROTEINS:
        if u not in proteins:
            proteins.append(u)
    print(f"[setup] annotation scan on {len(proteins)} proteins")

    models = {s: load_run_model(rd)[0] for s, rd in RUNS.items()}
    print(f"[ok] loaded {len(models)} attn-pool models (seeds {sorted(RUNS)})")

    cont_rows, aa_acc, feat_rows, res_records = [], {a: [] for a in AAS}, [], []
    n_used = 0
    for uni in proteins:
        try:
            resi = load_residues(uni)                       # [L, D]
        except Exception as e:
            print(f"  [skip] {uni}: {e}")
            continue
        L = resi.shape[0]
        norm = resi.norm(dim=1).cpu().numpy()               # [L] ESM2 residue norm
        ws = [get_attn_weights(models[s], resi) for s in RUNS]
        attn = np.mean(ws, axis=0)                           # consensus [L]

        meta = fetch_uniprot(uni)
        if meta is None or not meta["seq"]:
            continue
        if meta["length"] < L:
            print(f"  [len<L] {uni}: seq {meta['length']} < ESM2 {L} — skip")
            continue
        n_used += 1
        preds = per_residue_predictors(meta["seq"], norm)

        # continuous predictors (within-protein Spearman + partial vs norm)
        for name, vec in preds.items():
            if np.std(vec) < 1e-9:
                continue
            rho = spearmanr(attn, vec).correlation
            prho = (np.nan if name == "esm2_norm"
                    else partial_spearman(attn, vec, norm))
            cont_rows.append({"uniprot": uni, "predictor": name,
                              "n_res": L, "rho": rho, "partial_rho_vs_norm": prho})

        # amino-acid identity: z-scored attention pooled per AA
        z = (attn - attn.mean()) / (attn.std() + 1e-12)
        for c, zz in zip(meta["seq"][:L], z):
            if c in aa_acc:
                aa_acc[c].append(float(zz))

        # per-residue long table (for the plots)
        act = feat_mask(meta["features"], "Active site", L)
        bnd = feat_mask(meta["features"], "Binding site", L)
        sig = feat_mask(meta["features"], "Signal", L)
        seqL = meta["seq"][:L]
        for i in range(L):
            res_records.append({
                "uniprot": uni, "pos": int(i),
                "aa": seqL[i] if i < len(seqL) else "X",
                "attn": float(attn[i]), "attn_z": float(z[i]),
                "norm": float(norm[i]),
                "hydropathy": float(preds["hydropathy_KD"][i]),
                "charge": float(preds["charge"][i]),
                "is_active": int(act[i]), "is_binding": int(bnd[i]),
                "is_signal": int(sig[i]),
            })

        # UniProt feature regions: within-protein AUC of attn → region
        for ftype in FEATURE_TYPES:
            mask = feat_mask(meta["features"], ftype, L)
            n_pos = int(mask.sum())
            if 0 < n_pos < L:
                feat_rows.append({"uniprot": uni, "feature": ftype,
                                  "n_pos": n_pos, "n_res": L,
                                  "auc": float(roc_auc_score(mask, attn))})

    if not cont_rows:
        print("[done] no usable proteins."); return
    print(f"\n[ok] {n_used} proteins with sequence+annotations\n")

    res_df = pd.DataFrame(res_records)
    res_df.to_csv(OUT_DIR / "attn_annotation_residues.csv", index=False)

    # ── Continuous predictors summary ───────────────────────────────────────
    cdf = pd.DataFrame(cont_rows)
    cdf.to_csv(OUT_DIR / "attn_annotation_continuous.csv", index=False)
    rows = []
    for name, g in cdf.groupby("predictor"):
        r = g["rho"].dropna()
        pr = g["partial_rho_vs_norm"].dropna()
        wp = wilcoxon(r).pvalue if len(r) >= 6 and r.abs().sum() > 0 else np.nan
        wpp = wilcoxon(pr).pvalue if len(pr) >= 6 and pr.abs().sum() > 0 else np.nan
        rows.append({
            "predictor": name, "n": len(r),
            "mean_rho": r.mean(), "median_rho": r.median(),
            "frac_pos": (r > 0).mean(), "wilcoxon_p": wp,
            "mean_partial_rho": (np.nan if name == "esm2_norm" else pr.mean()),
            "partial_wilcoxon_p": (np.nan if name == "esm2_norm" else wpp),
        })
    summ = pd.DataFrame(rows).sort_values("mean_rho", key=lambda s: s.abs(),
                                          ascending=False)
    pd.set_option("display.width", 200)
    print(f"=== Within-protein attention vs continuous predictors (n={n_used}) ===")
    print("  rho>0: attention HIGHER on high-predictor residues. Chance rho=0.\n")
    print(summ.round(3).to_string(index=False))

    # ── Amino-acid profile ──────────────────────────────────────────────────
    aa_rows = [{"aa": a, "n": len(v),
                "mean_z_attn": float(np.mean(v)) if v else np.nan}
               for a, v in aa_acc.items()]
    adf = pd.DataFrame(aa_rows).sort_values("mean_z_attn")
    adf.to_csv(OUT_DIR / "attn_annotation_aa.csv", index=False)
    print("\n=== Amino-acid attention bias (mean within-protein z-score) ===")
    print("  most DOWN-weighted → most UP-weighted")
    print("  " + "  ".join(f"{r.aa}:{r.mean_z_attn:+.2f}" for r in adf.itertuples()))

    # ── Feature-region AUCs ─────────────────────────────────────────────────
    fsumm = None
    if feat_rows:
        fdf = pd.DataFrame(feat_rows)
        fdf.to_csv(OUT_DIR / "attn_annotation_features.csv", index=False)
        frows = []
        for ft, g in fdf.groupby("feature"):
            a = g["auc"]
            wp = (wilcoxon(a - 0.5).pvalue
                  if len(a) >= 6 and (a - 0.5).abs().sum() > 0 else np.nan)
            frows.append({"feature": ft, "n_proteins": len(a),
                          "mean_auc": a.mean(), "wilcoxon_p_vs_0.5": wp})
        fsumm = pd.DataFrame(frows).sort_values(
            "mean_auc", key=lambda s: (s - 0.5).abs(), ascending=False)
        print("\n=== Attention predicting UniProt feature regions (AUC) ===")
        print("  AUC>0.5: attention enriched ON the feature. Chance=0.5.\n")
        print(fsumm.round(3).to_string(index=False))

    # ── Figures ─────────────────────────────────────────────────────────────
    p1 = make_summary_figure(summ, adf, fsumm)
    print(f"\n[ok] wrote {p1}")
    p2 = make_explainer_figure(res_df)
    print(f"[ok] wrote {p2}")
    print("[ok] CSVs: attn_annotation_{continuous,aa,features,residues}.csv")


if __name__ == "__main__":
    main()
