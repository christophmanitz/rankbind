# RankBind: A0 conference poster

Built on the ScaDS.AI LaTeX poster template
(<https://gitlab.hrz.tu-chemnitz.de/scads.ai/latex-template>), A0 portrait.

## Build

```bash
python render_figs.py         # figures/*.pdf from ../poster_figure_data
module load texlive           # provides xelatex; the template needs XeLaTeX
./build.sh                    # two xelatex passes -> main.pdf
```

`build.sh` runs `probe.tex` first. That pass reports how tall each column's
content actually is against the height the page has for it, so a column that
would run into the footer shows up as a number there instead of in print:

```
column fit (available 68.7 cm):
  col1   67.1 cm  fits, 1.7 cm spare
  ...
```

If a column overflows, cut text or shrink a figure until it fits. Do not raise
`\colh` in `poster-setup.tex`: 68.7 cm is what the template's header and footer
leave over, measured from a compiled PDF (content starts 26.75 cm down, the
footer rule sits at 105.4 cm). `\bandh` is likewise measured, not guessed: the
same probe run reports the natural height of the tallest `\stat` box (6.45 cm),
and `\bandh` is set just above it.

## Wording

Terminology is used at full strength, but every term is defined before it is
needed. The `HOW WE SCORE` panel at the top of column 1 defines pooled AUC,
matrix MRR, Hit@10 and the **cheat sheet** (the molecule-blind baseline scoring
each protein by its training positive rate, with *overlap* = Jaccard of the
top-ten sets and *concentration* = Gini). Everything after that (the ingredient
list, the figure axes, the schematic) uses those words without re-explaining
them.
Add new vocabulary to that panel rather than glossing it inline.

Two house rules: no em dashes anywhere in the visible text, and no inline math
for things that are not equations. Fragments like `$200\times200$` or
`$\geq0.30$` are set in the template's math font and read as a different
typeface next to Open Sans; write `200\,$\times$\,200`, `K\textsubscript{M}`,
`\textit{k} = 4` instead, keeping math mode for single symbols only. The same
applies to the figures, where `mathtext` is bound to Open Sans in
`render_figs.py`.

## Files

| File | Contents |
|------|----------|
| `main.tex` | page assembly only: band plus three fixed-height columns |
| `poster-setup.tex` | lengths, colours, `\panel` / `\figcap` / `\stat`, title block |
| `col1.tex` … `col3.tex` | the three columns' content |
| `probe.tex` | measures column heights, writes them to `probe.log` |
| `arch.tex` | the architecture schematic (TikZ, drawn at exactly `\colw`) |
| `render_figs.py` | builds `figures/*.pdf` from `../poster_figure_data` + the paper's tables |
| `validate_palette.py` | the chart-palette check (see below) |
| `preview.png` | low-resolution render of the current `main.pdf` |

## Footer QR codes

`\footerqrcodes` in `poster-setup.tex` adds two codes to the template's footer
row, called from the end of `main.tex`. The template's footer is fixed, so their
positions are absolute and were measured off a compiled PDF: its own QR occupies
x = 4.6-9.3 cm, the contact block's email line ends at x = 26.0 cm and
"in cooperation with" starts at x = 41.8 cm, which leaves the span the two codes
sit in. If the contact details or the footer template change, re-measure before
trusting the placement.

The codes are vector output from `qrcode.sty`, so nothing goes soft at A0. They
need the `nolink` option: with `hyperref`'s `colorlinks`, the package otherwise
wraps them in a link and the modules come out in link colour instead of black.
The live one was checked by decoding it back out of the compiled PDF with
`cv2.QRCodeDetector` rather than by eye; crop it with a generous quiet zone or
detection fails on an otherwise valid code.

The second code is a **placeholder** — there is no repository yet, so it is a
dashed box rather than a scannable code pointing at a dead URL. To make it live,
replace the node body in `\footerqrcodes` with
`\qrcode[height=\fqrsize, level=M, nolink]{<url>}` and change its caption.

## The schematic

`arch.tex` is drawn in TikZ rather than matplotlib so it uses the poster's own
font. Everything below the two encoder boxes is positioned relative to what
precedes it: the boxes size themselves to their text, so absolute `y` values
silently overlap as soon as a label changes length. Avoid the
`(<number> |- <node>)` coordinate form in it — it does not parse as intended and
shifts a whole branch sideways without an error.

Figures are rendered at exactly the column width (234.7 mm), so the point sizes
in `render_figs.py` are the sizes that end up on the printed sheet, and they use
the template's own Open Sans.

## Chart colours

ScaDS.AI blue `#0074AC` for RankBind against `#D95F02` for the shortcut-taking
models. `validate_palette.py` checks the pair: worst-pair OKLab ΔE 30.9 for
normal vision and 20.8 under simulated protanopia/deuteranopia, both above 3:1
contrast on white. Re-run it before changing a series colour.

Grey (`#9AA0A4`) is used only for marks that carry no series identity and always
sit next to a printed value.
