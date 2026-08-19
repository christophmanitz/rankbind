"""
evaluation/fig_v5_model_diagram.py — RankBind (v5/v4) architecture, data-flow style.

A clean left-to-right architecture diagram in the visual idiom of the AAM paper
(AAM_paper.pdf, Figs 1 & 3): real molecule sketch + sequence input, the two
pretrained language models drawn as stacked encoder layers inside dashed module
containers, per-element feature-map grids, mean-pooling, projected embedding
vector strips f(L) / g(P), and a bilinear interaction core drawn as an explicit
M-matrix so the reader sees  s = f(L)^T M g(P) + b  as a tensor operation.

Streams:
  Ligand  : SMILES → ChemBERTa (frozen) → per-token 384-d → mean-pool → proj → f(L)∈R^256
  Protein : AA-seq → ESM2-650M (frozen) → per-residue 1280-d → mean-pool → proj → g(P)∈R^256
  Head    : f(L)^T M g(P) + b  (M = U V^T + diag(d), rank 128)  → score s(L,P)
  Trained with the three training-time anti-shortcut mechanisms (#1,#2,#4);
  the bilinear head is the architectural one (#3).

Numbers: v5_rankbind/configs/default.json (= v4 recipe). Outputs:
    paper/figures/fig_v5_model_diagram.pdf
    paper/figures/fig_v5_model_diagram.png

Run:
    python evaluation/fig_v5_model_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ──────────────────────────────────────────────────────────────────────────────
# Palette (AAM idiom: orange = ligand/sequence, blue = protein, green = fusion)
# ──────────────────────────────────────────────────────────────────────────────
LIG_RAMP  = ["#FCEBD2", "#F8D29A", "#F2B056", "#E89020"]
LIG_EDGE  = "#B86A12"
PROT_RAMP = ["#E0EAF7", "#BBD2EE", "#8FB3E0", "#5C8FD0"]
PROT_EDGE = "#2E6DB4"
GRN_RAMP  = ["#DCEFDC", "#AEDDB1", "#79C57F"]
GRN_EDGE  = "#2E8B57"
INK = "#202124"
AS_EDGE = "#D9531E"


# ──────────────────────────────────────────────────────────────────────────────
# Primitive helpers (figure data-coords)
# ──────────────────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, *, fill, edge, lw=1.5, radius=1.6, ls="-", z=3, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls,
        zorder=z, alpha=alpha, mutation_aspect=1.0))


def txt(ax, x, y, s, *, size=10, weight="normal", color=INK, ha="center",
        va="center", style="normal", z=7, rot=0):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            style=style, zorder=z, rotation=rot)


def arrow(ax, x1, y1, x2, y2, *, color=INK, lw=1.8, ls="-", rad=0.0, z=4,
          scale=15, alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=scale, linewidth=lw, color=color,
        linestyle=ls, zorder=z, alpha=alpha, shrinkA=2, shrinkB=2))


def module(ax, x0, y0, w, h, title, color, *, z=1):
    """Dashed rounded module container with a title at the top-left."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0,rounding_size=2.4",
        linewidth=1.8, edgecolor=color, facecolor="none",
        linestyle=(0, (6, 4)), zorder=z))
    txt(ax, x0 + 3, y0 + h - 2.4, title, size=9.6, weight="bold", color=color,
        ha="left", style="italic")


# ── tensor-ish glyphs ──────────────────────────────────────────────────────────
def _shade(ramp, k):
    return ramp[k % len(ramp)]


def feature_grid(ax, cx, cy, ncol, nrow, c, ramp, edge, *, depth=2, off=0.9, z=4):
    """Stacked feature-map: `depth` faint copies behind a front colored grid."""
    x0, y0 = cx - ncol * c / 2, cy - nrow * c / 2
    for d in range(depth, 0, -1):
        ax.add_patch(Rectangle((x0 + d * off, y0 + d * off), ncol * c, nrow * c,
                               facecolor="white", edgecolor=edge, linewidth=0.8,
                               zorder=z - 1, alpha=0.9))
    for i in range(nrow):
        for j in range(ncol):
            ax.add_patch(Rectangle((x0 + j * c, y0 + i * c), c, c,
                                   facecolor=_shade(ramp, i * 3 + j * 2 + i + j),
                                   edgecolor=edge, linewidth=0.7, zorder=z))


def vstrip(ax, cx, cy, n, c, ramp, edge, *, z=5, shades=None):
    """Vertical embedding-vector strip of n colored cells, centred at (cx, cy)."""
    y0 = cy - n * c / 2
    for i in range(n):
        col = shades[i] if shades else _shade(ramp, i * 2 + 1)
        ax.add_patch(Rectangle((cx - c / 2, y0 + i * c), c, c, facecolor=col,
                               edgecolor=edge, linewidth=0.8, zorder=z))
    return (cx - c / 2, cx + c / 2, y0, y0 + n * c)


def hstrip(ax, cx, cy, n, c, ramp, edge, *, z=5):
    """Horizontal vector strip (a row of n cells)."""
    x0 = cx - n * c / 2
    for j in range(n):
        ax.add_patch(Rectangle((x0 + j * c, cy - c / 2), c, c,
                               facecolor=_shade(ramp, j * 2 + 1),
                               edgecolor=edge, linewidth=0.8, zorder=z))
    return (x0, x0 + n * c, cy - c / 2, cy + c / 2)


def encoder_stack(ax, cx, y_bottom, w, n, ramp_edge, *, nbars=4, bar_h=4.2,
                  gap=1.4, label="Encoder layer"):
    """BERT-style stack of rounded encoder bars (light → saturated upward)."""
    ramp = ["#FFFFFF", "#F3F3F3", "#E6E6E6", "#DADADA"]
    cols = ramp_edge[0]
    y = y_bottom
    centers = []
    for k in range(nbars):
        shade = cols[min(k, len(cols) - 1)]
        rbox(ax, cx, y + bar_h / 2, w, bar_h, fill=shade, edge=ramp_edge[1],
             lw=1.3, radius=1.2, z=4)
        txt(ax, cx, y + bar_h / 2, label, size=8.0, color="#333")
        centers.append(y + bar_h / 2)
        # little inter-layer arrows
        if k < nbars - 1:
            arrow(ax, cx, y + bar_h, cx, y + bar_h + gap, scale=8, lw=1.0,
                  color="#888")
        y += bar_h + gap
    return centers[0] - bar_h / 2, centers[-1] + bar_h / 2  # (in_y, out_y)


def draw_projector(ax, cy, in_cx, in_n, in_c, out_cx, out_n, out_c, node, edge,
                   name):
    """Draw the projector as what it IS — a 2-layer MLP — so the viewer sees the
    operation, not a black box:

        input strip --(dense fan = Linear)--> hidden layer (+ GELU curve)
                     --(dense fan = Linear)--> shorter output strip

    More cells in than out shows the dimensionality reduction (384/1280 -> 256);
    the two fans are the two Linear layers; the curve is the nonlinearity.
    """
    in_x = in_cx + in_c / 2 + 0.4
    out_x = out_cx - out_c / 2 - 0.4
    mid_x = (in_x + out_x) / 2
    in_centers = [cy - in_n * in_c / 2 + (i + 0.5) * in_c for i in range(in_n)]
    out_centers = [cy - out_n * out_c / 2 + (i + 0.5) * out_c for i in range(out_n)]
    n_hid, span = 4, 11.0
    hid_ys = [cy - span / 2 + k * span / (n_hid - 1) for k in range(n_hid)]
    # two dense fans = the two Linear layers (thick enough to read clearly)
    for iy in in_centers:
        for hy in hid_ys:
            ax.plot([in_x, mid_x], [iy, hy], color="#7E868F", lw=1.1, alpha=0.85,
                    zorder=3)
    for hy in hid_ys:
        for oy in out_centers:
            ax.plot([mid_x, out_x], [hy, oy], color="#7E868F", lw=1.1, alpha=0.85,
                    zorder=3)
    # hidden-layer nodes
    ax.scatter([mid_x] * n_hid, hid_ys, s=34, c=node, edgecolors="white",
               linewidths=0.9, zorder=6)
    txt(ax, mid_x, cy - span / 2 - 3.4, name, size=8.2, weight="bold", color=edge)


def molecule_image(smiles: str, size=(440, 320)):
    """Render a SMILES to an RGBA numpy array via RDKit (None if unavailable)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        m = Chem.MolFromSmiles(smiles)
        d = rdMolDraw2D.MolDraw2DCairo(*size)
        opts = d.drawOptions()
        opts.bondLineWidth = 2
        opts.clearBackground = True
        rdMolDraw2D.PrepareAndDrawMolecule(d, m)
        d.FinishDrawing()
        import io
        from PIL import Image
        return np.array(Image.open(io.BytesIO(d.GetDrawingText())).convert("RGBA"))
    except Exception as e:  # pragma: no cover - graceful fallback
        print(f"[mol] RDKit render failed ({e}); using placeholder")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────
def build() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(19.5, 10.6))
    ax.set_xlim(0, 214)
    ax.set_ylim(0, 112)
    ax.axis("off")

    yL, yP = 80, 34          # ligand / protein stream centres
    yM = 57                  # bilinear head centre

    txt(ax, 4, 108, "RankBind (v5): model architecture", size=18, weight="bold",
        ha="left")

    # ── module containers ───────────────────────────────────────────────────
    module(ax, 31, yL - 16, 44, 32, "Ligand encoder — ChemBERTa (frozen)", LIG_EDGE)
    module(ax, 31, yP - 16, 44, 32, "Protein encoder — ESM2-650M (frozen)", PROT_EDGE)
    module(ax, 150, yM - 26, 33, 52, "Bilinear interaction head  (#3)", GRN_EDGE)

    # ═══════════════════════ LIGAND STREAM ═══════════════════════════════════
    # input molecule
    rbox(ax, 13, yL, 20, 20, fill="white", edge=LIG_EDGE, lw=1.6, radius=1.8)
    arr = molecule_image("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")  # a sugar substrate
    if arr is not None:
        ax.imshow(arr, extent=[3.5, 22.5, yL - 8.6, yL + 8.6], aspect="auto",
                  zorder=5, origin="upper")
    txt(ax, 13, yL - 12.2, "Ligand · SMILES", size=9.4, weight="bold")

    # SMILES token cells feeding the encoder
    toks = ["O", "C", "C", "1", "O", "…"]
    for j, t in enumerate(toks):
        rbox(ax, 28, yL + 10 - j * 4, 4.6, 3.4, fill=LIG_RAMP[1], edge=LIG_EDGE,
             lw=1.0, radius=0.6, z=5)
        txt(ax, 28, yL + 10 - j * 4, t, size=7.6)
    txt(ax, 28, yL - 14.2, "tokens", size=7.8, color="#555")
    arrow(ax, 23.2, yL, 25.6, yL, scale=9, color="#888", lw=1.1)  # molecule -> tokens

    encoder_stack(ax, 52, yL - 9, 30, 4, (LIG_RAMP, LIG_EDGE), nbars=4,
                  label="Transformer layer")
    arrow(ax, 30.5, yL, 36.5, yL, scale=11)          # tokens -> encoder

    # per-token feature map -> mean-pool -> 384-d embedding
    feature_grid(ax, 84, yL, 3, 4, 2.4, LIG_RAMP, LIG_EDGE)
    txt(ax, 84, yL - 8.2, "per-token\n384-d", size=8.0, color="#555")
    arrow(ax, 67, yL, 78, yL, scale=12)
    arrow(ax, 90, yL, 99, yL, scale=12)
    txt(ax, 94.5, yL + 1.8, "mean\npool", size=7.6, color="#333")

    vstrip(ax, 104, yL, 7, 2.2, LIG_RAMP, LIG_EDGE)
    txt(ax, 104, yL - 9.3, r"$\in\mathbb{R}^{384}$", size=8.4, color="#333")
    vstrip(ax, 132, yL, 5, 2.4, LIG_RAMP, LIG_EDGE)
    txt(ax, 132, yL + 7.6, r"$f(L)\in\mathbb{R}^{256}$", size=9.2, weight="bold",
        color=LIG_EDGE)
    draw_projector(ax, yL, 104, 7, 2.2, 132, 5, 2.4, LIG_RAMP[2], LIG_EDGE,
                   "Ligand Projector")

    # ═══════════════════════ PROTEIN STREAM ══════════════════════════════════
    rbox(ax, 13, yP, 20, 16, fill="white", edge=PROT_EDGE, lw=1.6, radius=1.8)
    # simple ribbon cartoon (helix coil + a strand arrow)
    xs = np.linspace(5.5, 12.5, 80)
    ax.plot(xs, yP + 3 + 1.6 * np.sin((xs - 5.5) * 3.2), color=PROT_EDGE, lw=2.4,
            zorder=5, solid_capstyle="round")
    ax.plot([12.5, 20.0], [yP + 1.0, yP + 1.0], color="#9DBDE6", lw=5.5, zorder=4,
            solid_capstyle="butt")
    ax.add_patch(plt.Polygon([[20.0, yP + 2.6], [22.0, yP + 1.0], [20.0, yP - 0.6]],
                             closed=True, facecolor="#9DBDE6", edgecolor=PROT_EDGE,
                             lw=0.8, zorder=4))
    txt(ax, 13, yP - 5.0, "M K T A Y I A K Q R …", size=7.8, color="#333")
    txt(ax, 13, yP - 9.6, "Protein · amino-acid sequence", size=9.4, weight="bold")

    resi = ["M", "K", "T", "A", "…"]
    for j, t in enumerate(resi):
        rbox(ax, 28, yP + 8 - j * 4, 4.6, 3.4, fill=PROT_RAMP[1], edge=PROT_EDGE,
             lw=1.0, radius=0.6, z=5)
        txt(ax, 28, yP + 8 - j * 4, t, size=7.6)
    txt(ax, 28, yP - 11.0, "residues", size=7.8, color="#555")
    arrow(ax, 23.2, yP, 25.6, yP, scale=9, color="#888", lw=1.1)  # ribbon -> residues

    encoder_stack(ax, 52, yP - 9, 30, 4, (PROT_RAMP, PROT_EDGE), nbars=4,
                  label="Transformer layer")
    arrow(ax, 30.5, yP, 36.5, yP, scale=11)

    feature_grid(ax, 84, yP, 3, 5, 2.2, PROT_RAMP, PROT_EDGE)
    txt(ax, 84, yP - 8.6, "per-residue\n1280-d", size=8.0, color="#555")
    arrow(ax, 67, yP, 78, yP, scale=12)
    arrow(ax, 90, yP, 99, yP, scale=12)
    txt(ax, 94.5, yP + 1.8, "mean\npool", size=7.6, color="#333")

    vstrip(ax, 104, yP, 9, 1.8, PROT_RAMP, PROT_EDGE)
    txt(ax, 104, yP - 9.3, r"$\in\mathbb{R}^{1280}$", size=8.4, color="#333")
    vstrip(ax, 132, yP, 5, 2.4, PROT_RAMP, PROT_EDGE)
    txt(ax, 132, yP + 7.6, r"$g(P)\in\mathbb{R}^{256}$", size=9.2, weight="bold",
        color=PROT_EDGE)
    draw_projector(ax, yP, 104, 9, 1.8, 132, 5, 2.4, PROT_RAMP[2], PROT_EDGE,
                   "Protein Projector")

    # ═══════════════════════ BILINEAR HEAD ═══════════════════════════════════
    # f(L) as a row hugging the top of M, g(P) as a column hugging the right.
    mcx, mcy = 162, yM
    feature_grid(ax, mcx, mcy, 5, 5, 2.6, GRN_RAMP, GRN_EDGE, depth=0)
    g_top = mcy + 5 * 2.6 / 2      # 63.5  (top edge of the M grid)
    g_right = mcx + 5 * 2.6 / 2    # 168.5 (right edge of the M grid)
    hstrip(ax, mcx, g_top + 1.3, 5, 2.6, LIG_RAMP, LIG_EDGE)
    txt(ax, mcx, g_top + 4.0, r"$f(L)^{\top}$", size=11.5, weight="bold",
        color=LIG_EDGE)            # label ABOVE the orange row, large & readable
    vstrip(ax, g_right + 1.3, mcy, 5, 2.6, PROT_RAMP, PROT_EDGE)
    txt(ax, g_right + 4.2, 61.5, r"$g(P)$", size=11.5, weight="bold",
        color=PROT_EDGE, ha="left")   # label to the RIGHT of the blue column
    txt(ax, mcx, 77.6, r"$M = UV^{\top}+\mathrm{diag}(d)$", size=9.6, color=GRN_EDGE)
    txt(ax, mcx, 73.9, "learned weight matrix · rank 128", size=7.6, color="#555",
        style="italic")
    # f(L) / g(P) streams enter the head at the LEFT EDGE of its dashed border
    # (matched, mirror-symmetric pair landing on the green container at x≈150).
    arrow(ax, 134, yL, 149.5, 63, scale=14, lw=2.4, color=LIG_EDGE, rad=-0.28)
    arrow(ax, 134, yP, 149.5, 51, scale=14, lw=2.4, color=PROT_EDGE, rad=0.28)

    # ═══════════════════════ OUTPUT ══════════════════════════════════════════
    arrow(ax, g_right + 4, mcy, 189, mcy, scale=13)
    rbox(ax, 201, mcy, 20, 11, fill=GRN_RAMP[1], edge=GRN_EDGE, lw=1.8, radius=1.4)
    txt(ax, 201, mcy + 1.9, "score s(L,P)", size=10.5, weight="bold",
        color="#1d5e39")
    txt(ax, 201, mcy - 2.6, "+ b  (learned bias)", size=7.6, color="#444")

    # ═══════════════════════ ANTI-SHORTCUT TRAINING BAND ════════════════════
    # Poster-legible: large band, big fonts, clearly separated mechanisms.
    ax.add_patch(FancyBboxPatch((28, 0.5), 158, 16,
                 boxstyle="round,pad=0,rounding_size=2.6", linewidth=2.2,
                 edgecolor=AS_EDGE, facecolor="#FFF3EB", linestyle=(0, (6, 4)),
                 zorder=2))
    txt(ax, 32, 12.3, "Anti-shortcut training (margin objective)", size=14,
        weight="bold", color=AS_EDGE, ha="left")
    txt(ax, 32, 5.2,
        "#1  protein-balanced sampling        #2  within-ligand margin loss        "
        "#4  hard-negative mining (top-50 confusers, refreshed each epoch)",
        size=12.5, ha="left", color="#7a2c0d")
    # gradient arrow up to the head
    arrow(ax, 150, 16.5, 158, yM - 26, scale=13, color=AS_EDGE, lw=2.2,
          ls=(0, (5, 3)), rad=-0.1)
    txt(ax, 162, 25.5, "gradient (f, g, M)", size=9.2, color=AS_EDGE,
        style="italic", ha="left")

    fig.tight_layout(pad=0.5)
    return fig


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here.parent / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build()
    for ext, dpi in (("pdf", None), ("svg", None), ("png", 300)):
        path = out_dir / f"fig_v5_model_diagram.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[fig] wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
