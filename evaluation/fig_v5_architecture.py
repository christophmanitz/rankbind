"""
evaluation/fig_v5_architecture.py — RankBind (v5) anti-shortcut schematic.

This figure does NOT just label the four anti-shortcut mechanisms — it draws a
small diagram of the PRINCIPLE behind each one, so a reader sees *why* it
removes the protein-prior shortcut.

The shortcut to defeat: ranking proteins by a protein-only prior beta(P)
("this protein binds everything"), ignoring the ligand. Decompose a score as

    s(L,P) = alpha(L) [ligand-only] + beta(P) [protein prior = shortcut]
                                    + gamma(L,P) [interaction = wanted]

The four mechanisms (numbered as in the figure):

  #1 Protein-balanced sampler   v5_rankbind/sampler.py::ProteinBalancedSampler
        flattens the per-protein positive-rate prior -> beta(P) carries no signal
  #2 Within-ligand margin loss  v5_rankbind/loss.py::margin_loss
        fix L, push s(L,P+) above s(L,P-) by m. L shared -> alpha(L) and bias b
        cancel in the difference -> loss can only fall by ranking on gamma(L,P)
        (NOTE: the protein term beta does NOT cancel inside one triplet; the
        prior is removed by #1 + the head's lack of an additive protein path)
  #3 Bilinear interaction head  v5_rankbind/model.py::BilinearHead
        no additive protein-only term (unlike MLPConcatHead) — structural prior,
        not a hard guarantee; per the ablation the loss is the dominant driver
  #4 Hard-negative mining       v5_rankbind/sampler.py::TripletCollator
        per epoch, re-score all (positive-ligand x train-protein) pairs and draw
        k=4 negatives from the current top-50 confusers -> margin keeps biting

Numbers: v5_rankbind/configs/default.json + v4 3-seed headline in CLAUDE.md /
phase2_rankbind_multiseed.csv. Footer also cites the same-architecture BCE
control that re-introduces the shortcut.

Outputs (vector + raster, matching paper/figures/ convention):
    paper/figures/fig_v5_architecture.pdf
    paper/figures/fig_v5_architecture.png

Run:
    python evaluation/fig_v5_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ──────────────────────────────────────────────────────────────────────────────
# Palette
# ──────────────────────────────────────────────────────────────────────────────
C = {
    "frozen_fill": "#ECECEC", "frozen_edge": "#9AA0A6", "frozen_txt": "#3C4043",
    "train_fill":  "#DCEAF7", "train_edge":  "#2E6DB4", "train_txt":  "#13335c",
    "head_fill":   "#C9DDF2", "head_edge":   "#1F4E8C",
    "score_fill":  "#D8EFD9", "score_edge":  "#2E8B57", "score_txt":  "#1d5e39",
    "input_fill":  "#FFFFFF", "input_edge":  "#5F6368",
    "as_fill":     "#FFFCFA", "as_edge":     "#D9531E", "as_badge":   "#D9531E",
    "pos":         "#2E8B57",  # binder / interaction / good
    "neg":         "#C0392B",  # shortcut / protein-prior / non-binder
    "lig":         "#2E6DB4",  # ligand term alpha(L)
    "inter":       "#2E8B57",  # interaction gamma(L,P)
    "chance":      "#888888",
    "foot_fill":   "#EEF1F8", "foot_edge":  "#5A6072",
    "exp":         "#7A4FB0",
    "ink":         "#202124",
}


# ──────────────────────────────────────────────────────────────────────────────
# Primitive helpers (all in figure data-coords: xlim 0..160, ylim 0..140)
# ──────────────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, *, fill, edge, lw=1.5, radius=2.0, ls="-", z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls,
        zorder=z, alpha=alpha, mutation_aspect=1.0,
    ))
    return (x, y, w, h)


def rect(ax, x, y, w, h, *, fill, edge=None, lw=0.8, z=3, alpha=1.0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill,
                           edgecolor=edge or fill, linewidth=lw,
                           zorder=z, alpha=alpha))


def text(ax, x, y, s, *, size=10, weight="normal", color=C["ink"],
         ha="center", va="center", style="normal", z=6, rot=0):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, style=style, zorder=z, rotation=rot)


def badge(ax, x, y, n, *, size=9.5):
    """Uniform compact numbered anti-shortcut chip."""
    ax.text(x, y, f"#{n}", fontsize=size, fontweight="bold", color="white",
            ha="center", va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.30", fc=C["as_badge"], ec="none"))


def arrow(ax, x1, y1, x2, y2, *, color=C["ink"], lw=1.7, ls="-",
          rad=0.0, z=4, scale=14, alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=scale, linewidth=lw, color=color,
        linestyle=ls, zorder=z, alpha=alpha, shrinkA=1, shrinkB=1,
    ))


def dot(ax, x, y, *, color, s=70, edge="white", z=6, marker="o"):
    ax.scatter([x], [y], s=s, c=color, edgecolors=edge, linewidths=1.0,
               zorder=z, marker=marker)


def panel(ax, cx, cy, w, h, num, title):
    """Anti-shortcut mechanism panel: light frame + numbered chip + bold title.
    Returns the panel center (cx, cy) for convenience."""
    box(ax, cx, cy, w, h, fill=C["as_fill"], edge=C["as_edge"], lw=1.8, radius=2.6)
    left = cx - w / 2
    top = cy + h / 2
    badge(ax, left + 6.0, top - 4.6, num)
    text(ax, left + 12.5, top - 4.6, title, size=11.3, weight="bold",
         color="#7a2c0d", ha="left")
    return cx, cy


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
def build() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16.4, 14.2))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 140)
    ax.axis("off")

    # ── Title + shortcut definition ──────────────────────────────────────────
    text(ax, 4, 137.0,
         "RankBind (v5): how the four anti-shortcut mechanisms work",
         size=18, weight="bold", ha="left")
    text(ax, 4, 133.2,
         "The shortcut to defeat: rank proteins by a protein-only prior "
         "(“this protein binds everything”), ignoring the ligand.  "
         "Every score splits into three parts:",
         size=10.8, ha="left", color="#444")

    # colour-keyed decomposition s = alpha(L) + beta(P) + gamma(L,P)
    text(ax, 4, 129.4, r"$s(L,P)\;=$", size=12, ha="left", weight="bold")
    deco = [
        (22, C["lig"],   r"$\alpha(L)$",     "ligand-only"),
        (58, C["neg"],   r"$\beta(P)$",      "protein prior = the SHORTCUT"),
        (108, C["inter"], r"$\gamma(L,P)$",  "interaction = what we WANT"),
    ]
    parts = ["+", "+"]
    for i, (x0, col, sym, lab) in enumerate(deco):
        text(ax, x0, 129.4, sym, size=12, color=col, weight="bold", ha="left")
        text(ax, x0, 126.4, lab, size=8.6, color=col, ha="left")
        if i < 2:
            text(ax, x0 + 22.5 if i == 0 else x0 + 41.5, 129.4, "+", size=12, ha="left")

    # ===========================================================================
    # Architecture ribbon (compact context, top band y ~ 108..123)
    # ===========================================================================
    text(ax, 4, 122.0, "ARCHITECTURE (forward pass)", size=10.5, weight="bold",
         color=C["head_edge"], ha="left")

    y_lig, y_prot = 117.0, 110.5
    bw, bh = 19, 5.6
    # ligand stream
    box(ax, 13, y_lig, 16, bh, fill=C["input_fill"], edge=C["input_edge"])
    text(ax, 13, y_lig, "Ligand\nSMILES", size=8.4)
    box(ax, 36, y_lig, bw, bh, fill=C["frozen_fill"], edge=C["frozen_edge"])
    text(ax, 36, y_lig, "ChemBERTa  (frozen)\nmean-pool → 384", size=7.9,
         color=C["frozen_txt"])
    box(ax, 60, y_lig, bw, bh, fill=C["train_fill"], edge=C["train_edge"])
    text(ax, 60, y_lig, "LigandProjector\n384 → 256", size=7.9,
         color=C["train_txt"])
    # protein stream
    box(ax, 13, y_prot, 16, bh, fill=C["input_fill"], edge=C["input_edge"])
    text(ax, 13, y_prot, "Protein\nAA-seq", size=8.4)
    box(ax, 36, y_prot, bw, bh, fill=C["frozen_fill"], edge=C["frozen_edge"])
    text(ax, 36, y_prot, "ESM2 (650M)  (frozen)\nmean-pool → 1280", size=7.9,
         color=C["frozen_txt"])
    box(ax, 60, y_prot, bw, bh, fill=C["train_fill"], edge=C["train_edge"])
    text(ax, 60, y_prot, "ProteinProjector\n1280 → 256", size=7.9,
         color=C["train_txt"])
    # f(L), g(P)
    text(ax, 76, y_lig, r"$f(L)$", size=10, weight="bold", color=C["train_txt"])
    text(ax, 76, y_prot, r"$g(P)$", size=10, weight="bold", color=C["train_txt"])
    # head + score
    box(ax, 100, 113.75, 30, 10.5, fill=C["head_fill"], edge=C["head_edge"], lw=1.8)
    text(ax, 100, 116.4, "Bilinear head", size=9.6, weight="bold",
         color=C["head_edge"])
    text(ax, 100, 113.0, r"$s=f(L)^{\top} M\, g(P)+b$", size=9.6)
    badge(ax, 88.5, 118.4, 3)
    box(ax, 133, 113.75, 20, 7, fill=C["score_fill"], edge=C["score_edge"])
    text(ax, 133, 113.75, "score  s(L,P)", size=9.4, weight="bold",
         color=C["score_txt"])
    # ribbon arrows
    for y in (y_lig, y_prot):
        arrow(ax, 21.2, y, 26.3, y, scale=11)
        arrow(ax, 45.6, y, 50.3, y, scale=11)
        arrow(ax, 69.6, y, 73.0, y, scale=11)
    arrow(ax, 79, y_lig, 86, 115.5, scale=11, rad=-0.1)
    arrow(ax, 79, y_prot, 86, 112.0, scale=11, rad=0.1)
    arrow(ax, 115.2, 113.75, 122.8, 113.75, scale=11)

    text(ax, 156, 122.0,
         "frozen = grey   trainable (≈627k) = blue   score = green",
         size=8.4, ha="right", color="#555")

    # ===========================================================================
    # Section header for the mechanism panels
    # ===========================================================================
    text(ax, 4, 104.3,
         "THE FOUR ANTI-SHORTCUT MECHANISMS  —  each drawn as its principle:",
         size=12.5, weight="bold", color=C["as_edge"], ha="left")

    # Panel grid: top row y=80, bottom row y=34; left x=41, right x=119
    W, H = 74, 40
    cL, cR = 41, 119
    yT, yB = 80, 34

    # ──────────────────────────────────────────────────────────────────────
    # PANEL #1 — Protein-balanced sampler : flatten the protein prior
    # ──────────────────────────────────────────────────────────────────────
    panel(ax, cL, yT, W, H, 1, "Protein-balanced sampler")
    base = 71.0          # bar baseline
    sh = 13.0            # bar height scale (height 1.0 -> 13 units)
    bw1, gap1 = 3.0, 1.6
    # axis
    arrow(ax, cL - 33, base, cL - 33, base + sh + 2.5, scale=9, lw=1.1)
    text(ax, cL - 35.5, base + sh / 2, "P(binds)", size=7.6, rot=90, color="#555")
    # raw (spiky) group
    raw = [0.92, 0.15, 0.70, 0.05, 0.85]
    x0 = cL - 30
    for i, hgt in enumerate(raw):
        rect(ax, x0 + i * (bw1 + gap1), base, bw1, hgt * sh, fill=C["neg"], alpha=0.85)
    text(ax, x0 + 2.5 * (bw1 + gap1) - bw1 / 2, base + sh + 1.6,
         "some proteins\n“always bind”", size=7.3, color=C["neg"])
    # arrow
    xa = x0 + 5 * (bw1 + gap1) + 1
    arrow(ax, xa, base + sh * 0.45, xa + 8, base + sh * 0.45, color=C["as_edge"],
          lw=1.8, scale=13)
    text(ax, xa + 4, base + sh * 0.45 + 2.6, "balanced\nsampling", size=7.6,
         color=C["as_edge"], weight="bold")
    # balanced (flat) group
    bal = [0.52, 0.48, 0.50, 0.49, 0.51]
    x1 = xa + 11
    for i, hgt in enumerate(bal):
        rect(ax, x1 + i * (bw1 + gap1), base, bw1, hgt * sh, fill=C["pos"], alpha=0.85)
    # chance line
    ax.plot([x1 - 1, x1 + 5 * (bw1 + gap1)], [base + 0.5 * sh, base + 0.5 * sh],
            ls=(0, (4, 3)), color=C["chance"], lw=1.2, zorder=5)
    text(ax, x1 + 5 * (bw1 + gap1) + 0.5, base + 0.5 * sh, "chance", size=7.2,
         color=C["chance"], ha="left")
    text(ax, x1 + 2.5 * (bw1 + gap1) - bw1 / 2, base + sh + 1.6, "after sampling",
         size=7.3, color=C["pos"])
    # caption
    cap1 = [
        "Each protein is drawn as a binder (+) and a non-binder (−) about",
        "equally often, so the per-protein positive-rate prior — what the",
        "null_prot_prior baseline exploits — drops to chance.",
    ]
    for i, ln in enumerate(cap1):
        text(ax, cL - 35, yT - 11.5 - i * 2.7, ln, size=8.0, ha="left", color="#333")

    # ──────────────────────────────────────────────────────────────────────
    # PANEL #2 — Within-ligand margin loss : rank for a fixed ligand
    # ──────────────────────────────────────────────────────────────────────
    panel(ax, cR, yT, W, H, 2, "Within-ligand margin loss")
    axx = cR - 30          # vertical score axis x
    ybot, ytop = 70.0, 90.0
    arrow(ax, axx, ybot, axx, ytop + 1.0, scale=9, lw=1.1)
    text(ax, axx - 2.6, (ybot + ytop) / 2, "score  s(L, ·)", size=7.6, rot=90,
         color="#555")
    text(ax, axx + 1.5, ytop + 1.8, "fixed ligand L", size=8.2, weight="bold",
         color="#333", ha="left")

    def smap(s):       # score in [0,1] -> y
        return ybot + 1.5 + s * (ytop - ybot - 3)

    colx = axx + 11
    confusers = [0.30, 0.45, 0.55, 0.63]
    for s in confusers:
        dot(ax, colx, smap(s), color=C["neg"], s=58)
    # P+ before (faint, dashed ring) and after (pushed up)
    dot(ax, colx, smap(0.50), color=C["pos"], s=58, edge=C["pos"])
    ax.scatter([colx], [smap(0.50)], s=150, facecolors="none",
               edgecolors=C["pos"], linewidths=1.0, linestyle="--", zorder=5)
    dot(ax, colx, smap(0.93), color=C["pos"], s=95)
    arrow(ax, colx + 2.2, smap(0.55), colx + 2.2, smap(0.90), color=C["pos"],
          lw=1.6, scale=12)
    # margin bracket between top confuser and P+
    bx = colx + 8
    ylo, yhi = smap(0.63), smap(0.93)
    ax.plot([bx, bx], [ylo, yhi], color="#333", lw=1.2, zorder=5)
    ax.plot([bx - 1.2, bx], [ylo, ylo], color="#333", lw=1.2, zorder=5)
    ax.plot([bx - 1.2, bx], [yhi, yhi], color="#333", lw=1.2, zorder=5)
    text(ax, bx + 1.2, (ylo + yhi) / 2, r"$\geq m$", size=9.5, ha="left",
         weight="bold")
    # legend
    dot(ax, cR + 18, smap(0.93), color=C["pos"], s=58)
    text(ax, cR + 20, smap(0.93), r"$P^{+}$ binder", size=7.8, ha="left")
    dot(ax, cR + 18, smap(0.78), color=C["neg"], s=58)
    text(ax, cR + 20, smap(0.78), r"$P^{-}$ confusers", size=7.8, ha="left")
    # caption
    cap2 = [
        r"Fix the ligand L: $P^{+}$ must beat every $P^{-}$ by a margin $m$.",
        r"L is shared $\Rightarrow$ $\alpha(L)$ and bias $b$ cancel in the difference,",
        r"so the loss falls only by ranking proteins on $\gamma(L,P)$.",
    ]
    for i, ln in enumerate(cap2):
        text(ax, cR - 33, yT - 11.5 - i * 2.7, ln, size=8.0, ha="left", color="#333")

    # ──────────────────────────────────────────────────────────────────────
    # PANEL #3 — Bilinear interaction head : no additive protein-only path
    # ──────────────────────────────────────────────────────────────────────
    panel(ax, cL, yB, W, H, 3, "Bilinear interaction head")
    # left mini-graph: MLP-concat (shortcut-prone)
    text(ax, cL - 18, 47.5, "MLP-concat head", size=8.6, weight="bold",
         color="#555")
    fx, gx = cL - 33, cL - 33
    fy, gy = 43.5, 38.5
    mlp = box(ax, cL - 23, 41.0, 9, 6, fill="#EDEDED", edge="#888")
    text(ax, cL - 23, 41.0, "MLP", size=8.0, color="#444")
    sx = cL - 12
    dot(ax, fx, fy, color=C["lig"], s=42); text(ax, fx - 1.5, fy, "f", size=8, ha="right")
    dot(ax, gx, gy, color=C["neg"], s=42); text(ax, gx - 1.5, gy, "g", size=8, ha="right")
    arrow(ax, fx + 1.5, fy, cL - 27.5, 41.8, scale=8, lw=1.1)
    arrow(ax, gx + 1.5, gy, cL - 27.5, 40.2, scale=8, lw=1.1)
    arrow(ax, cL - 18.5, 41.0, sx - 1.5, 41.0, scale=8, lw=1.1)
    dot(ax, sx, 41.0, color=C["score_edge"], s=42); text(ax, sx + 1.5, 41.0, "s", size=8, ha="left")
    # the shortcut path g -> s (protein-only)
    arrow(ax, gx + 1.2, gy - 0.4, sx - 1.0, 39.6, color=C["neg"], lw=1.7,
          ls=(0, (3, 2)), rad=-0.45, scale=11)
    text(ax, cL - 20, 35.0, "can learn  s ≈ β(P)\n(ignores the ligand)",
         size=7.4, color=C["neg"], ha="center")

    # right mini-graph: bilinear (needs both)
    text(ax, cL + 20, 47.5, "Bilinear head (RankBind)", size=8.6, weight="bold",
         color=C["head_edge"])
    fx2, fy2 = cL + 7, 43.5
    gx2, gy2 = cL + 7, 38.5
    mx, my = cL + 19, 41.0
    sx2 = cL + 31
    dot(ax, fx2, fy2, color=C["lig"], s=42); text(ax, fx2 - 1.5, fy2, "f", size=8, ha="right")
    dot(ax, gx2, gy2, color=C["neg"], s=42); text(ax, gx2 - 1.5, gy2, "g", size=8, ha="right")
    ax.add_patch(Circle((mx, my), 1.9, facecolor="white", edgecolor=C["head_edge"],
                        linewidth=1.6, zorder=6))
    text(ax, mx, my, r"$\times$", size=11, color=C["head_edge"], weight="bold")
    arrow(ax, fx2 + 1.5, fy2, mx - 1.8, 41.8, scale=8, lw=1.1)
    arrow(ax, gx2 + 1.5, gy2, mx - 1.8, 40.2, scale=8, lw=1.1)
    arrow(ax, mx + 2.0, my, sx2 - 1.5, my, scale=8, lw=1.1)
    dot(ax, sx2, my, color=C["score_edge"], s=42); text(ax, sx2 + 1.5, my, "s", size=8, ha="left")
    text(ax, mx + 4, 36.6, "+ b", size=8.2, color="#444")
    text(ax, cL + 19, 35.0, "P enters ONLY via × with L;\nonly the scalar b is ligand-free",
         size=7.4, color=C["head_edge"], ha="center")
    # caption
    cap3 = [
        "No additive protein-only term (unlike MLP-concat), so the head",
        "cannot just add a protein prior. A structural bias, not a guarantee:",
        "the margin loss is the dominant driver; DeltaField makes it exact.",
    ]
    for i, ln in enumerate(cap3):
        text(ax, cL - 35, yB - 11.5 - i * 2.7, ln, size=8.0, ha="left", color="#333")

    # ──────────────────────────────────────────────────────────────────────
    # PANEL #4 — Hard-negative mining : sample the model's own confusers
    # ──────────────────────────────────────────────────────────────────────
    panel(ax, cR, yB, W, H, 4, "Hard-negative mining")
    # vertical ranking strip
    sx0, swid = cR - 30, 6
    sy0, shei = 27.0, 20.0
    rect(ax, sx0, sy0, swid, shei, fill="#E6E9F0", edge="#9aa0b0", lw=1.0, z=3)
    # score-direction axis (up = higher score)
    arrow(ax, sx0 - 1.4, sy0, sx0 - 1.4, sy0 + shei + 0.6, scale=8, lw=1.0)
    text(ax, sx0 - 4.6, sy0 + shei / 2, "all proteins ranked\nby current model score",
         size=7.2, rot=90, color="#555")
    # P+ near top
    dot(ax, sx0 + swid / 2, sy0 + shei - 1.5, color=C["pos"], s=70)
    text(ax, sx0 + swid + 1.5, sy0 + shei - 1.5, r"$P^{+}$ (true binder)", size=7.6,
         ha="left", color=C["pos"])
    # top-50 confusers band + k=4 picks
    band_lo, band_hi = sy0 + shei - 6.5, sy0 + shei - 2.8
    rect(ax, sx0, band_lo, swid, band_hi - band_lo, fill=C["as_edge"], edge=None,
         alpha=0.20, z=4)
    for j, yy in enumerate([band_lo + 0.8, band_lo + 1.9, band_lo + 2.6, band_hi - 0.6]):
        dot(ax, sx0 + swid / 2 + (-1.2 if j % 2 else 1.2), yy, color=C["as_edge"], s=34)
    bx4 = sx0 + swid + 1.5
    ax.plot([bx4, bx4], [band_lo, band_hi], color=C["as_edge"], lw=1.3, zorder=6)
    text(ax, bx4 + 1.2, (band_lo + band_hi) / 2 + 0.3,
         "top-50 confusers\n→ draw k=4 here\n(HARD)", size=7.3, ha="left",
         color=C["as_edge"], weight="bold")
    # random easy negatives near bottom
    for yy in [sy0 + 2.0, sy0 + 4.0, sy0 + 6.5, sy0 + 3.0]:
        dot(ax, sx0 + swid / 2 + (1.4 if yy % 2 < 1 else -1.4), yy,
            color="#9aa0b0", s=26, edge="white")
    text(ax, sx0 + swid + 1.5, sy0 + 4.0, "random negatives\n(easy: margin already\nsatisfied)",
         size=7.2, ha="left", color="#777")
    # caption
    cap4 = [
        "Each epoch, re-score all (positive-ligand × train-protein) pairs",
        "and draw the k=4 negatives from the current top-50 confusers,",
        "not random easy ones → the margin keeps biting.",
    ]
    for i, ln in enumerate(cap4):
        text(ax, cR - 33, yB - 11.5 - i * 2.7, ln, size=8.0, ha="left", color="#333")

    # ===========================================================================
    # Footer — net effect + the control that re-introduces the shortcut
    # ===========================================================================
    box(ax, 80, 8.0, 152, 11.5, fill=C["foot_fill"], edge=C["foot_edge"], radius=2.0)
    text(ax, 8, 11.0, "Net effect (v4, 3-seed mean ± std):", size=10.3,
         weight="bold", ha="left", color="#2a2f3d")
    text(ax, 56, 11.0,
         "matrix MRR 0.33±0.07   ·   Hit@10 0.76±0.10   ·   "
         "Gini-residual −0.21±0.02   ·   Top-10 Jaccard vs null ≈ 0.00",
         size=9.6, ha="left", weight="bold", color="#1d3a5c")
    text(ax, 8, 6.0,
         "Causal control: keep this architecture but swap the margin loss for BCE "
         "→ the shortcut returns (global AUC 0.92, but matrix MRR ≈ 0.01, "
         "Jaccard-vs-null 0.43). The loss, not the features, is what removes the prior.",
         size=9.2, ha="left", color="#333")

    fig.tight_layout(pad=0.5)
    return fig


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build()
    for ext, dpi in (("pdf", None), ("png", 200)):
        path = out_dir / f"fig_v5_architecture.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[fig] wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
