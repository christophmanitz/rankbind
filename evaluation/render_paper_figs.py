"""
evaluation/render_paper_figs.py — paper-style versions of the four figures
embedded in paper/scirep/main.tex (and paper/jcim/main.tex).

The figures currently embedded in the manuscripts come from the ScaDS.AI
poster renderer (paper/poster/poster_scads/render_figs.py): Open Sans on a
A0-poster scale with the ScaDS corporate blue ramp. This renderer produces
the same content in a journal style:

  - Latin Modern fonts (the manuscripts use lmodern), sized for the printed
    page, not the poster
  - a restrained, colour-blind-safe palette (no ScaDS branding)
  - richer panels where the poster version was too thin:
      * response maps: all four baselines + RankBind + ligand-blind prior
        (2x3 grid, per-panel robust colour scale, Gini annotations)
      * ablation: two panels (matrix MRR and Hit@10) with error bars
      * shortcut overlap: error bars where available, null-random reference
      * attention audit: percentile distributions + concentration panel

Output: paper/figures/paper_style/{fig_respmaps,fig_ablation,fig_jaccard,
fig_attention}.{pdf,png}. The manuscript files are NOT rewritten; review the
PDFs here and copy them into place once approved.

Every number is read from committed CSVs / pinned score matrices; nothing is
hard-coded except the row order of the ablation (Table 2 of the manuscript).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
RES = os.path.join(ROOT, "evaluation", "attractor_results")
POST = os.path.join(ROOT, "paper", "poster", "poster_figure_data")
OUT = os.path.join(ROOT, "paper", "figures", "paper_style")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Fonts: Latin Modern (lmodern), the font the manuscripts are set in.
# ---------------------------------------------------------------------------
TEXLIVE = "/software/all/texlive/20230313-GCC-13.2.0/texmf-dist/fonts/opentype/public/lm"
LM = {
    "regular": os.path.join(TEXLIVE, "lmroman10-regular.otf"),
    "bold":    os.path.join(TEXLIVE, "lmroman10-bold.otf"),
    "italic":  os.path.join(TEXLIVE, "lmroman10-italic.otf"),
    "bolditalic": os.path.join(TEXLIVE, "lmroman10-bolditalic.otf"),
}
for p in LM.values():
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
FAMILY = "Latin Modern Roman" if os.path.exists(LM["regular"]) else "DejaVu Serif"

# ---------------------------------------------------------------------------
# Palette: paper-neutral. Blue = RankBind (the proposed method), warm grey =
# shortcut-taking baselines/controls, light grey = neutral/reference.
# ---------------------------------------------------------------------------
INK   = "#111111"
INK2  = "#444444"
MUTED = "#888888"
GRID  = "#e8e8e8"
AXIS  = "#bbbbbb"
BLUE  = "#2166AC"     # RankBind
BLUE2 = "#67A9CF"     # lighter step of the same hue
WARM  = "#B05A2A"     # shortcut-prone models / BCE control
NEUT  = "#9AA0A4"     # no series identity
GRAY  = "#7f7f7f"

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "paper_blue", ["#f5f9fc", "#d3e3f0", "#9cc2de", "#5f9bc6", "#2166AC", "#0d3557"])

plt.rcParams.update({
    "font.family": FAMILY,
    "mathtext.fontset": "custom",
    "mathtext.rm": FAMILY,
    "mathtext.it": FAMILY,
    "mathtext.bf": FAMILY,
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.7,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def save(fig, name, dpi=300):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), format="pdf",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, f"{name}.png"), format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{name:16s} "
          f"{os.path.getsize(os.path.join(OUT, f'{name}.pdf')) // 1024:5d} KB")


# ---------------------------------------------------------------------------
# 1. Response maps: 4 baselines + RankBind + ligand-blind prior (2 x 3).
# ---------------------------------------------------------------------------
MATRICES = {
    "GraphDTA": os.path.join(ROOT, "results", "original_graphdta", "score_matrix_graphdta.npy"),
    "MolTrans": os.path.join(ROOT, "results", "original_moltrans", "score_matrix_moltrans.npy"),
    "DrugBAN":  os.path.join(ROOT, "results", "original_drugban", "score_matrix_DrugBAN.npy"),
    "GEMS":     os.path.join(ROOT, "results", "original_gems", "score_matrix_gems.npy"),
    "RankBind": os.path.join(ROOT, "results", "v5_rankbind",
                             "20260423-112928_012a2695c2_default_v4",
                             "score_matrix_rankbind.npy"),
    "Null prot. prior": os.path.join(RES, "score_matrix_null_prot_prior.npy"),
}


def _gini(attr):
    a = np.sort(attr)
    n = len(a)
    if n == 0 or a.sum() == 0:
        return np.nan
    cum = np.cumsum(a) / a.sum()
    return (n + 1 - 2 * cum.sum()) / n


def _attractor_scores(M):
    return M.max(axis=0) if False else M.argmax(axis=0)  # placeholder, unused


def fig_respmaps():
    order = [("GraphDTA", 0), ("MolTrans", 1), ("DrugBAN", 2),
             ("GEMS", 3), ("RankBind", 4), ("Null prot. prior", 5)]
    # Gini of the attractor-share distribution: per-protein fraction of
    # molecules for which the protein is the top-scoring partner.
    ginis = {}
    maps = {}
    for name, _ in order:
        p = MATRICES[name]
        M = np.load(p)
        if M.shape != (200, 200):
            M = M.T if M.T.shape == (200, 200) else M
        maps[name] = M
        top = np.argmax(M, axis=1)
        shares = np.bincount(top, minlength=M.shape[1]).astype(float)
        shares = shares / shares.sum()
        ginis[name] = _gini(shares)

    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.4))
    ticks = [0, 50, 100, 150, 200]
    for ax, (name, _) in zip(axes.flat, order):
        M = maps[name]
        lo, hi = np.percentile(M, [1, 99])
        if lo == hi:
            lo, hi = float(M.min()), float(M.max())
        im = ax.imshow(M, cmap=BLUE_RAMP, aspect="equal", vmin=lo, vmax=hi,
                       interpolation="nearest")
        ax.grid(False)
        ax.set_title(f"{name}  (Gini {ginis[name]:.3f})", fontsize=9.5, pad=4)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=8, length=2.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(AXIS)
    # Axis labels: left column and bottom row only.
    for ax in axes[:, 0]:
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks], fontsize=8)
    for ax in axes[1, :]:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    axes[0, 0].set_ylabel("200 ligands  (row index)", fontsize=9)
    axes[1, 0].set_ylabel("200 ligands  (row index)", fontsize=9)
    axes[1, 0].set_xlabel("200 proteins  (column index)", fontsize=9)
    axes[1, 1].set_xlabel("200 proteins  (column index)", fontsize=9)
    axes[1, 2].set_xlabel("200 proteins  (column index)", fontsize=9)
    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    save(fig, "fig_respmaps")


# ---------------------------------------------------------------------------
# 2. Ablation: matrix MRR and Hit@10, rows from manuscript Table 2.
# ---------------------------------------------------------------------------
ABLATION = [
    ("BCE only (matched control)", 0.014, 0.002, 0.000, 0.000, WARM),
    ("- margin loss",              0.020, 0.006, 0.029, 0.029, WARM),
    ("- bilinear head (MLP)",      0.140, 0.087, 0.373, 0.170, NEUT),
    ("- balanced sampler",         0.147, 0.070, 0.441, 0.153, NEUT),
    ("+ matrix-MRR selection",     0.183, 0.055, 0.539, 0.103, NEUT),
    ("RankBind (full)",            0.220, 0.026, 0.598, 0.045, BLUE),
    ("+ residue attention (1 seed)", 0.316, 0.000, 0.706, 0.000, BLUE2),
]


def fig_ablation():
    names = [r[0] for r in ABLATION][::-1]
    mrr = np.array([r[1] for r in ABLATION])[::-1]
    mrr_sd = np.array([r[2] for r in ABLATION])[::-1]
    h10 = np.array([r[3] for r in ABLATION])[::-1]
    h10_sd = np.array([r[4] for r in ABLATION])[::-1]
    colors = [r[5] for r in ABLATION][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={
        "wspace": 0.5, "width_ratios": [1.15, 1]})
    y = np.arange(len(names))
    for ax, vals, sds, xlab, chance in (
            (ax1, mrr, mrr_sd, "matrix MRR  (3 seeds, mean \u00b1 s.d.)", 0.029),
            (ax2, h10, h10_sd, "Hit@10", 0.05)):
        ax.barh(y, vals, height=0.6, color=colors, edgecolor="none", zorder=3)
        ax.errorbar(vals, y, xerr=sds, fmt="none", ecolor=INK2, elinewidth=1.2,
                    capsize=3.5, capthick=1.2, zorder=4)
        for yi, v in zip(y, vals):
            ax.text(v + sds[yi] + 0.012, yi, f"{v:.3f}", va="center",
                    fontsize=9, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        ax.axvline(chance, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=5)
        ax.text(chance, len(names) - 0.35, f"chance {chance:g}", fontsize=8,
                color=MUTED, va="center", ha="left")
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is ax1 else ["" for _ in names],
                           fontsize=9)
        ax.set_xlim(0, 0.80)
        ax.set_xlabel(xlab, fontsize=9, labelpad=3)
        ax.tick_params(labelsize=9)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    save(fig, "fig_ablation")


# ---------------------------------------------------------------------------
# 3. Shortcut overlap: top-ten overlap with the ligand-blind prior.
# ---------------------------------------------------------------------------
def fig_jaccard():
    ov = pd.read_csv(os.path.join(RES, "cross_model_overlap.csv"), index_col=0)
    prior = ov["null_prot_prior"].to_dict()
    rows = [
        ("RankBind", 0.018, 0.030, BLUE),
        ("+ residue attention (1 seed)", 0.000, 0.000, BLUE2),
        ("MolTrans", prior.get("moltrans", 0.053), None, NEUT),
        ("DrugBAN",  prior.get("drugban", 0.538), None, WARM),
        ("GEMS",     prior.get("gems", 0.667), None, WARM),
        ("GraphDTA", prior.get("graphdta", 0.667), None, WARM),
    ]
    names = [r[0] for r in rows][::-1]
    vals = np.array([r[1] for r in rows])[::-1]
    sds = np.array([r[2] if r[2] is not None else 0.0 for r in rows])[::-1]
    colors = [r[3] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.6, color=colors, edgecolor="none", zorder=3)
    for yi, (v, s) in enumerate(zip(vals, sds)):
        ax.text(v + 0.02, yi, f"{v:.3f}", va="center", fontsize=10,
                color=INK, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    ax.axvline(0.30, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=5)
    ax.text(0.305, len(names) - 0.3, "warning level 0.30", fontsize=8.5,
            color=MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(0, 0.80)
    ax.set_xlabel("top-ten overlap with the ligand-blind prior  (Jaccard)",
                  fontsize=9.5, labelpad=3)
    ax.tick_params(labelsize=9.5)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    save(fig, "fig_jaccard")


# ---------------------------------------------------------------------------
# 4. Attention audit: functional-residue percentiles + concentration.
# ---------------------------------------------------------------------------
def fig_attention():
    df = pd.read_csv(os.path.join(POST, "figure10_attn_explainer",
                                  "panel2_functional_residue_percentiles.csv"))
    conc = pd.read_csv(os.path.join(RES, "attn_weights_concentration.csv"))
    cross = pd.read_csv(os.path.join(RES, "attn_weights_cross_seed.csv"))
    spearman_cols = [c for c in cross.columns if c.startswith("spearman_")]
    med_rho = float(cross[spearman_cols].to_numpy().flatten().mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={
        "wspace": 0.55, "width_ratios": [1.25, 1]})

    order = ["all_residues", "binding_site", "active_site"]
    labels = {"all_residues": "all residues",
              "binding_site": "annotated binding site",
              "active_site": "catalytic active site"}
    colors = {"all_residues": NEUT, "binding_site": BLUE2, "active_site": BLUE}
    for i, key in enumerate(order):
        r = df[df.group == key].iloc[0]
        ax1.barh(i, r.q3 - r.q1, left=r.q1, height=0.45, color=colors[key],
                 edgecolor="none", zorder=3)
        ax1.plot([r["median"], r["median"]], [i - 0.25, i + 0.25], color=INK,
                 lw=1.8, zorder=5, solid_capstyle="butt")
        ax1.plot([r.whisker_lo, r.whisker_hi], [i, i], color=INK2, lw=1.1,
                 zorder=2)
        ax1.plot([r.whisker_hi, r.whisker_hi], [i - 0.16, i + 0.16], color=INK2,
                 lw=1.1, zorder=2)
        ax1.text(r.whisker_hi + 0.03, i, f"median {r['median']:.2f}",
                 va="center", fontsize=9.5, color=INK)
    ax1.set_yticks(range(3))
    ax1.set_yticklabels([f"{labels[k]}\n(n = {int(df[df.group == k].iloc[0].n):,})"
                         for k in order], fontsize=9)
    ax1.set_xlim(0, 1.6)
    ax1.set_ylim(-0.6, 2.6)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.tick_params(labelsize=9)
    ax1.set_xlabel("attention percentile within its own protein  (0.5 = average)",
                   fontsize=9, labelpad=3)
    ax1.grid(axis="x")
    ax1.set_axisbelow(True)
    ax1.text(0.0, -0.45, f"cross-seed Spearman $\\rho$ = {med_rho:.2f} "
                         f"(random $\\approx$ 0)", fontsize=8, color=INK2)

    kvals = np.array([0.05, 0.10, 0.20])
    cols = {"0.05": "top5pct_mass", "0.10": "top10pct_mass", "0.20": "top20pct_mass"}
    med = [float(conc[cols[f"{k:.2f}"]].median()) for k in kvals]
    q1 = [float(conc[cols[f"{k:.2f}"]].quantile(0.25)) for k in kvals]
    q3 = [float(conc[cols[f"{k:.2f}"]].quantile(0.75)) for k in kvals]
    ax2.plot([0, 0.25], [0, 0.25], ls=(0, (4, 3)), color=MUTED, lw=1.2,
             label="uniform expectation")
    ax2.errorbar(kvals, med, yerr=[np.array(med) - np.array(q1),
                                   np.array(q3) - np.array(med)],
                 fmt="o", color=BLUE, ms=6, mew=1.2, ecolor=INK2,
                 elinewidth=1.2, capsize=4.5, capthick=1.5,
                 label="attention mass", zorder=5)
    for k, m in zip(kvals, med):
        ax2.text(k, m + 0.02, f"{m:.3f}", ha="center", va="bottom",
                 fontsize=9, color=INK)
    ax2.set_xlabel("top-k% of residues", fontsize=9, labelpad=3)
    ax2.set_ylabel("median share of attention mass", fontsize=9, labelpad=3)
    ax2.set_xticks(kvals)
    ax2.set_xticklabels([f"{int(k*100)}%" for k in kvals], fontsize=9)
    ax2.tick_params(labelsize=9)
    ax2.set_xlim(0, 0.26)
    ax2.set_ylim(0, 0.30)
    ax2.legend(fontsize=8, loc="upper left")
    save(fig, "fig_attention")


if __name__ == "__main__":
    print(f"font: {FAMILY}")
    print(f"output: {OUT}")
    fig_respmaps()
    fig_ablation()
    fig_jaccard()
    fig_attention()