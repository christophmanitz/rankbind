"""Render the RankBind poster figures for the ScaDS.AI A0 template.

Sources: ../poster_figure_data/*.csv (regenerate with
`python -m evaluation.export_poster_figure_data`) plus the headline tables of
../main.tex. Output: figures/*.pdf (vector, placed at 1:1 physical size) and
figures/*.png (screen previews only).

Palette: ScaDS.AI corporate blue for RankBind against #D95F02 for the
shortcut-taking models. Checked with validate_palette.py — worst pair
normal-vision OKLab dE 30.9, dE 20.8 under simulated protan/deutan, both
above 3:1 contrast on white.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "poster_figure_data"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Column width on the A0 sheet: 0.3133 x textwidth(749 mm) = 234.7 mm = 9.24 in.
# Figures are rendered at that exact width, so font sizes below are the point
# sizes that end up on the printed poster.
FIGW = 9.24

for ttf in (HERE / "fonts").glob("OpenSans-*.ttf"):
    font_manager.fontManager.addfont(str(ttf))
FONT = "Open Sans" if any((HERE / "fonts").glob("OpenSans-Regular.ttf")) else "DejaVu Sans"

INK = "#1a1a1a"        # scadsaiblack
INK2 = "#5a5a5a"
MUTED = "#8a8f92"
GRID = "#e4e6e7"
AXIS = "#c3c6c8"
SURF = "#ffffff"
BLUE = "#0074AC"       # scadsaiblue  — RankBind
BLUE_D = "#004B6F"     # scadsaidarkblue — darker step of the same hue
WARM = "#D95F02"       # shortcut-taking models
NEUTRAL = "#9AA0A4"    # no series identity; always directly labelled

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "scads_blue", ["#f2f8fb", "#cfe6f1", "#8dc4dd", "#3d9bc7", "#0074AC", "#00344f"])

plt.rcParams.update({
    "font.family": FONT,
    # Subscripts such as K$_M$ otherwise fall back to DejaVu and read as a
    # different typeface next to the Open Sans labels.
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.cal": FONT,
    "font.size": 21,
    "pdf.fonttype": 42,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{name}.png", format="png", dpi=90, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"{name:20s} {(OUT / f'{name}.pdf').stat().st_size // 1024:4d} KB")


# ------------------------------------------------------- 1. the dissociation
def fig_dissociation():
    pts = [
        ("DrugBAN",   0.954, 0.433, WARM, (0, -44)),
        ("MolTrans",  0.937, 0.470, WARM, (-24, -12)),
        ("GraphDTA",  0.869, 0.533, WARM, (0, 26)),
        ("GEMS",      0.633, 0.514, WARM, (0, 26)),
        ("RankBind",  0.618, 0.878, BLUE, (18, -14)),
        ("+ attn pool (1 seed)", 0.646, 0.930, BLUE_D, (18, 8)),
    ]
    fig, ax = plt.subplots(figsize=(FIGW, FIGW * 0.46))
    ax.axhspan(0.33, 0.5, color=WARM, alpha=0.08, lw=0)
    ax.axhline(0.5, color=AXIS, lw=2, ls=(0, (5, 4)), zorder=1)
    ax.text(0.588, 0.482, "chance level", color=MUTED, fontsize=19, va="top", ha="left")

    for name, x, y, c, off in pts:
        ax.scatter([x], [y], s=430, color=c, edgecolor=SURF, linewidth=3, zorder=4)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off,
                    fontsize=21, color=INK,
                    fontweight="bold" if c != WARM else "normal",
                    ha="center" if abs(off[0]) < 10 else ("right" if off[0] < 0 else "left"))

    ax.set_xlabel("pooled AUC over all pairs   (what the field reports)",
                  labelpad=12, fontsize=21)
    ax.set_ylabel("ranking AUC\nwithin one molecule\n(what the task needs)",
                  labelpad=12, fontsize=21)
    ax.set_xlim(0.58, 1.02)
    ax.set_ylim(0.33, 1.0)
    ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(color=GRID, lw=1.2, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig_dissociation")


# ------------------------------------------------------- 2. response maps
def fig_respmaps():
    files = [
        ("RankBind\n(ours)", "response_map_RankBind.csv", "concentration 0.79"),
        ("GraphDTA\n(baseline)", "response_map_GraphDTA.csv", "concentration 0.99"),
        ("Cheat sheet\n(no molecule)",
         "response_map_Null_prot_prior.csv", "concentration 0.99"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, FIGW * 0.42))
    for ax, (title, fname, sub) in zip(axes, files):
        m = pd.read_csv(SRC / "figure1_summary" / fname).iloc[:, 1:].to_numpy()
        lo, hi = np.percentile(m, [2, 98])
        ax.imshow(m, cmap=BLUE_RAMP, vmin=lo, vmax=hi, aspect="equal",
                  interpolation="nearest")
        ax.set_title(title, fontsize=19, color=INK, pad=10, fontweight="bold")
        ax.text(0.5, -0.11, sub, transform=ax.transAxes, ha="center",
                fontsize=18, color=INK2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(AXIS)
    axes[0].set_ylabel("200 molecules", fontsize=18, color=INK2, labelpad=8)
    axes[1].set_xlabel("200 proteins", fontsize=18, color=INK2, labelpad=34)
    fig.subplots_adjust(wspace=0.10)
    save(fig, "fig_respmaps")


# ------------------------------------------------------- 3. shortcut overlap
def fig_jaccard():
    rows = [
        ("RankBind", 0.018, BLUE),
        ("+ attn pool (1 seed)", 0.000, BLUE_D),
        ("MolTrans", 0.05, WARM),
        ("DrugBAN", 0.54, WARM),
        ("GEMS", 0.67, WARM),
        ("GraphDTA", 0.67, WARM),
    ]
    fig, ax = plt.subplots(figsize=(FIGW, FIGW * 0.42))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.62, color=[r[2] for r in rows],
            edgecolor=SURF, linewidth=2.5)
    for i, (name, v, c) in enumerate(rows):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=21,
                color=INK, fontweight="bold")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=21)
    ax.set_xlim(0, 0.80)
    ax.set_xlabel("top-ten overlap with the cheat sheet   (Jaccard)",
                  labelpad=12, fontsize=20)
    ax.axvline(0.30, color=INK2, lw=2, ls=(0, (5, 4)))
    ax.text(0.318, 5.45, "warning level", color=INK2, fontsize=19, va="center")
    ax.grid(axis="x", color=GRID, lw=1.2)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(AXIS)
    save(fig, "fig_jaccard")


# ------------------------------------------------------- 4. ablation
def fig_ablation():
    rows = [
        ("BCE only (matched control)", 0.014, 0.002, WARM),
        ("− margin loss", 0.020, 0.006, WARM),
        ("− bilinear head (MLP)", 0.140, 0.087, NEUTRAL),
        ("− balanced sampler", 0.147, 0.070, NEUTRAL),
        ("+ matrix-MRR selection", 0.183, 0.055, NEUTRAL),
        ("RankBind (full)", 0.220, 0.026, BLUE),
        ("+ residue attention (1 seed)", 0.316, 0.000, BLUE_D),
    ]
    fig, ax = plt.subplots(figsize=(FIGW, FIGW * 0.42))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.62, color=[r[3] for r in rows],
            edgecolor=SURF, linewidth=2.5, zorder=3)
    ax.errorbar([r[1] for r in rows], y, xerr=[r[2] for r in rows], fmt="none",
                ecolor=INK2, elinewidth=2.5, capsize=8, capthick=2.5, zorder=4)
    for i, (name, v, s, c) in enumerate(rows):
        ax.text(v + s + 0.024, i, f"{v:.3f}", va="center", fontsize=21,
                color=INK, fontweight="bold")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=20)
    ax.set_xlim(0, 0.48)
    ax.set_xlabel("matrix MRR   (3 seeds, mean ± s.d.)",
                  labelpad=12, fontsize=20)
    ax.axvline(0.029, color=INK2, lw=2, ls=(0, (5, 4)), zorder=5)
    ax.text(0.042, 6.55, "chance 0.029", color=INK2, fontsize=19, va="center")
    ax.grid(axis="x", color=GRID, lw=1.2)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(AXIS)
    save(fig, "fig_ablation")


# ------------------------------------------------------- 5. seven datasets
def fig_datasets():
    rows = [
        ("kcat/K$_M$",      0.029, 0.228),
        ("K$_M$",           0.030, 0.219),
        ("turnover",        0.014, 0.344),
        ("ESP",             0.098, 0.387),
        ("BindingDB K$_d$", 0.042, 0.320),
        ("Davis",           0.044, 0.038),
        ("KIBA",            0.035, 0.023),
    ]
    # A blank row between the two dataset families carries the group labels,
    # which reads better than brackets once the figure is this short.
    ypos = [8, 7, 6, 5, 3, 2, 1]
    fig, ax = plt.subplots(figsize=(FIGW, FIGW * 0.47))
    # One bar per dataset, with the control as a tick on the same row. Two
    # stacked bars per row put their value labels closer together than the
    # label height, which is what made the earlier version unreadable.
    for yy, (_, bce, rb) in zip(ypos, rows):
        ax.barh(yy, rb, height=0.52, color=BLUE, edgecolor=SURF, linewidth=2.5,
                zorder=3)
        ax.plot([bce, bce], [yy - 0.36, yy + 0.36], color=WARM, lw=6, zorder=5,
                solid_capstyle="butt")
        ax.text(max(rb, bce) + 0.015, yy, f"{rb:.3f}", va="center",
                fontsize=21, color=INK, fontweight="bold")
    handles = [Patch(facecolor=BLUE, label="RankBind"),
               Line2D([], [], color=WARM, lw=6,
                      label="matched BCE control")]

    for y, text in [(9.0, "enzyme–substrate"), (4.05, "kinase affinity")]:
        ax.text(0.004, y, text, fontsize=19, color=INK2, va="center", ha="left",
                zorder=6, bbox=dict(facecolor=SURF, edgecolor="none", pad=2.5))

    ax.set_yticks(ypos, [r[0] for r in rows], fontsize=21)
    ax.set_ylim(0.30, 9.55)
    ax.axvline(0.029, color=INK2, lw=2, ls=(0, (5, 4)), zorder=5)
    ax.text(0.029, 9.62, "chance", ha="center", va="bottom", fontsize=19,
            color=INK2, clip_on=False)
    ax.set_xlim(0, 0.50)
    ax.set_xlabel("matrix MRR", labelpad=12, fontsize=21)
    ax.grid(axis="x", color=GRID, lw=1.2)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(AXIS)
    leg = ax.legend(handles=handles, loc="lower right", frameon=False,
                    fontsize=20, handlelength=1.2, bbox_to_anchor=(1.0, -0.03))
    for t in leg.get_texts():
        t.set_color(INK2)
    save(fig, "fig_datasets")


# ------------------------------------------------------- 6. attention audit
def fig_attention():
    df = pd.read_csv(SRC / "figure10_attn_explainer" /
                     "panel2_functional_residue_percentiles.csv")
    order = ["all_residues", "binding_site", "active_site"]
    labels = {"all_residues": "all residues",
              "binding_site": "annotated\nbinding site",
              "active_site": "catalytic\nactive site"}
    colors = {"all_residues": NEUTRAL, "binding_site": "#8dc4dd", "active_site": BLUE_D}
    fig, ax = plt.subplots(figsize=(FIGW, FIGW * 0.42))
    for i, key in enumerate(order):
        r = df[df.group == key].iloc[0]
        ax.barh(i, r.q3 - r.q1, left=r.q1, height=0.5, color=colors[key],
                edgecolor=SURF, linewidth=2.5, zorder=3)
        ax.plot([r["median"], r["median"]], [i - 0.29, i + 0.29], color=INK,
                lw=4.5, zorder=5, solid_capstyle="butt")
        ax.plot([r.whisker_lo, r.whisker_hi], [i, i], color=AXIS, lw=2.5, zorder=2)
        ax.text(1.07, i, f"median {r['median']:.2f}   n = {int(r.n):,}",
                va="center", fontsize=20, color=INK)
    ax.set_yticks(range(3), [labels[k] for k in order], fontsize=21)
    ax.set_xlim(0, 1.68)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("attention percentile within its own protein   (0.5 = average)",
                  labelpad=12, fontsize=20)
    ax.grid(axis="x", color=GRID, lw=1.2)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(AXIS)
    save(fig, "fig_attention")


if __name__ == "__main__":
    print(f"font: {FONT}")
    for f in (fig_dissociation, fig_respmaps, fig_jaccard, fig_ablation,
              fig_datasets, fig_attention):
        f()
