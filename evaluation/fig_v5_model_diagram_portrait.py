"""
evaluation/fig_v5_model_diagram_portrait.py — portrait (hochkant) layout of the
RankBind architecture, for single-column / portrait placement in the paper.

Same content and drawing helpers as fig_v5_model_diagram.py; only the layout
differs: the flow runs top -> bottom, the two encoder streams sit side by side
as columns and merge into the bilinear head below, with the score and the
anti-shortcut training band at the bottom.

Outputs (vector + raster):
    paper/figures/fig_v5_model_diagram_portrait.{pdf,svg,png}

Run:
    python evaluation/fig_v5_model_diagram_portrait.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_v5_model_diagram import (  # noqa: E402  (shared helpers + palette)
    rbox, txt, arrow, module, feature_grid, hstrip, vstrip, molecule_image,
    LIG_RAMP, LIG_EDGE, PROT_RAMP, PROT_EDGE, GRN_RAMP, GRN_EDGE, AS_EDGE,
)


# ── vertical variants of the two orientation-specific glyphs ──────────────────
def encoder_stack_v(ax, cx, y_top, w, nbars, edge, label="Transformer layer",
                    bar_h=4.6, gap=2.2):
    """Stacked encoder bars flowing downward; returns the bottom y."""
    shades = ["#FFFFFF", "#F1F1F1", "#E4E4E4", "#D8D8D8"]
    y = y_top
    for k in range(nbars):
        rbox(ax, cx, y - bar_h / 2, w, bar_h, fill=shades[min(k, len(shades) - 1)],
             edge=edge, lw=1.3, radius=1.2, z=4)
        txt(ax, cx, y - bar_h / 2, label, size=7.6, color="#333")
        if k < nbars - 1:
            arrow(ax, cx, y - bar_h, cx, y - bar_h - gap, scale=8, lw=1.0,
                  color="#888")
        y -= bar_h + gap
    return y + gap


def draw_projector_v(ax, cx, in_cy, in_n, out_cy, out_n, c, node, edge, name,
                     name_side="left"):
    """Projector as a 2-layer MLP, drawn vertically (flow top -> bottom):
    input row -> dense fan -> hidden row -> dense fan -> shorter output row."""
    in_y = in_cy - c / 2 - 0.4
    out_y = out_cy + c / 2 + 0.4
    mid_y = (in_y + out_y) / 2
    in_xs = [cx - in_n * c / 2 + (i + 0.5) * c for i in range(in_n)]
    out_xs = [cx - out_n * c / 2 + (i + 0.5) * c for i in range(out_n)]
    n_hid, span = 4, 11.0
    hid_xs = [cx - span / 2 + k * span / (n_hid - 1) for k in range(n_hid)]
    for ix in in_xs:
        for hx in hid_xs:
            ax.plot([ix, hx], [in_y, mid_y], color="#7E868F", lw=1.1, alpha=0.85,
                    zorder=3)
    for hx in hid_xs:
        for ox in out_xs:
            ax.plot([hx, ox], [mid_y, out_y], color="#7E868F", lw=1.1, alpha=0.85,
                    zorder=3)
    ax.scatter(hid_xs, [mid_y] * n_hid, s=34, c=node, edgecolors="white",
               linewidths=0.9, zorder=6)
    if name_side == "left":
        txt(ax, cx - span / 2 - 4, mid_y, name, size=8.0, weight="bold",
            color=edge, ha="right")
    else:
        txt(ax, cx + span / 2 + 4, mid_y, name, size=8.0, weight="bold",
            color=edge, ha="left")


# ──────────────────────────────────────────────────────────────────────────────
def stream(ax, cx, *, mol, ramp, edge, in_dim_label, vec_label, name, name_side):
    """Draw one top-to-bottom encoder stream and return the y of its f/g row."""
    n = in_dim_label[0]
    # encoder module + stacked layers (short title so it stays inside the box)
    module(ax, cx - 16, 110, 32, 21,
           "ChemBERTa (frozen)" if mol else "ESM2-650M (frozen)", edge)
    encoder_stack_v(ax, cx, 126.5, 26, 3, edge, bar_h=4.0, gap=1.8)
    arrow(ax, cx, 136, cx, 131.5, scale=10)               # input box -> encoder
    # mean-pool straight to the embedding row (clear gap, no per-token sub-grid)
    arrow(ax, cx, 110, cx, 106.4, scale=11)
    txt(ax, cx + 4.3, 108.2, "mean-pool", size=7.4, color="#333", ha="left")
    hstrip(ax, cx, 104, n, 2.2, ramp, edge)
    txt(ax, cx + n * 2.2 / 2 + 2.0, 104, in_dim_label[1], size=8.2, color="#333",
        ha="left")
    # projector (vertical MLP) -> f/g row
    draw_projector_v(ax, cx, 104, n, 88, 5, 2.2, ramp[2], edge, name,
                     name_side=name_side)
    hstrip(ax, cx, 88, 5, 2.4, ramp, edge)
    txt(ax, cx, 84.2, vec_label, size=9.4, weight="bold", color=edge)
    return 88


def build() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.6, 16.2))
    ax.set_xlim(0, 116)
    ax.set_ylim(0, 162)
    ax.axis("off")

    cL, cP = 31, 85
    txt(ax, 5, 158, "RankBind (v5): model architecture", size=16, weight="bold",
        ha="left")

    # ── inputs (caption sits INSIDE the box so it never lands on the arrow) ───
    # ligand molecule
    rbox(ax, cL, 146, 22, 19, fill="white", edge=LIG_EDGE, lw=1.6, radius=1.8)
    arr = molecule_image("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
    if arr is not None:
        ax.imshow(arr, extent=[cL - 9.5, cL + 9.5, 141, 155], aspect="auto",
                  zorder=5, origin="upper")
    txt(ax, cL, 138.4, "Ligand · SMILES", size=8.6, weight="bold")
    # protein sequence + ribbon
    rbox(ax, cP, 146, 22, 19, fill="white", edge=PROT_EDGE, lw=1.6, radius=1.8)
    xs = np.linspace(cP - 7.5, cP - 0.5, 80)
    ax.plot(xs, 151 + 1.5 * np.sin((xs - (cP - 7.5)) * 3.2), color=PROT_EDGE,
            lw=2.4, zorder=5, solid_capstyle="round")
    ax.plot([cP - 0.5, cP + 7], [149, 149], color="#9DBDE6", lw=5.5, zorder=4)
    ax.add_patch(plt.Polygon([[cP + 7, 150.6], [cP + 9, 149], [cP + 7, 147.4]],
                             closed=True, facecolor="#9DBDE6", edgecolor=PROT_EDGE,
                             lw=0.8, zorder=4))
    txt(ax, cP, 144, "M K T A Y I A K Q R …", size=7.2, color="#333")
    txt(ax, cP, 138.4, "Protein · AA sequence", size=8.6, weight="bold")

    # ── two streams ──────────────────────────────────────────────────────────
    yfL = stream(ax, cL, mol=True, ramp=LIG_RAMP, edge=LIG_EDGE,
                 in_dim_label=(7, r"$\in\mathbb{R}^{384}$"),
                 vec_label=r"$f(L)\in\mathbb{R}^{256}$", name="Ligand\nProjector",
                 name_side="left")
    yfP = stream(ax, cP, mol=False, ramp=PROT_RAMP, edge=PROT_EDGE,
                 in_dim_label=(9, r"$\in\mathbb{R}^{1280}$"),
                 vec_label=r"$g(P)\in\mathbb{R}^{256}$", name="Protein\nProjector",
                 name_side="right")

    # ── bilinear head (centred, below; the two streams merge here) ────────────
    mcx, mcy = 58, 59
    module(ax, mcx - 19, 44, 38, 34, "Bilinear head  (#3)", GRN_EDGE)
    feature_grid(ax, mcx, mcy, 5, 5, 2.6, GRN_RAMP, GRN_EDGE, depth=0)
    g_top = mcy + 5 * 2.6 / 2       # 65.5
    g_right = mcx + 5 * 2.6 / 2     # 64.5
    hstrip(ax, mcx, g_top + 1.3, 5, 2.6, LIG_RAMP, LIG_EDGE)      # f(L)^T row
    txt(ax, mcx - 9.2, g_top + 1.3, r"$f(L)^{\top}$", size=10.5, weight="bold",
        color=LIG_EDGE, ha="right")
    vstrip(ax, g_right + 1.3, mcy, 5, 2.6, PROT_RAMP, PROT_EDGE)  # g(P) column
    txt(ax, g_right + 4.4, mcy, r"$g(P)$", size=10.5, weight="bold",
        color=PROT_EDGE, ha="left")
    txt(ax, mcx, 71.2, r"$M = UV^{\top}+\mathrm{diag}(d)$   ·   rank 128",
        size=8.6, color=GRN_EDGE)
    # f(L) / g(P) lead symmetrically to the green dashed border of the head
    arrow(ax, cL, 82.5, 48, 78, scale=13, lw=2.4, color=LIG_EDGE)
    arrow(ax, cP, 82.5, 68, 78, scale=13, lw=2.4, color=PROT_EDGE)

    # ── score ────────────────────────────────────────────────────────────────
    arrow(ax, mcx, 45, mcx, 41, scale=13)
    rbox(ax, mcx, 36, 30, 9, fill=GRN_RAMP[1], edge=GRN_EDGE, lw=1.8, radius=1.4)
    txt(ax, mcx, 37.6, "score  s(L,P)", size=10.5, weight="bold", color="#1d5e39")
    txt(ax, mcx, 33.6, "+ b  (learned bias)", size=7.6, color="#444")

    # ── anti-shortcut training band (bottom, full width) ──────────────────────
    ax.add_patch(FancyBboxPatch((6, 4), 104, 21,
                 boxstyle="round,pad=0,rounding_size=2.4", linewidth=2.0,
                 edgecolor=AS_EDGE, facecolor="#FFF3EB", linestyle=(0, (6, 4)),
                 zorder=2))
    txt(ax, 11, 21.0, "Anti-shortcut training (margin objective)", size=12,
        weight="bold", color=AS_EDGE, ha="left")
    txt(ax, 13, 15.6, "#1  protein-balanced sampling", size=10.5, ha="left",
        color="#7a2c0d")
    txt(ax, 13, 11.2, "#2  within-ligand margin loss", size=10.5, ha="left",
        color="#7a2c0d")
    txt(ax, 13, 6.8,
        "#4  hard-negative mining (top-50 confusers, refreshed each epoch)",
        size=10.5, ha="left", color="#7a2c0d")
    # gradient arrow up to the head (routed left of the score box)
    arrow(ax, 32, 25.5, 43, 44, scale=12, color=AS_EDGE, lw=2.0, ls=(0, (5, 3)),
          rad=0.12)
    txt(ax, 28, 34, "gradient\n(f, g, M)", size=8.4, color=AS_EDGE,
        style="italic", ha="right")

    fig.tight_layout(pad=0.5)
    return fig


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build()
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        path = out_dir / f"fig_v5_model_diagram_portrait.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[fig] wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
