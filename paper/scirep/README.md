# RankBind: Scientific Reports manuscript (rewrite)

This directory contains a rewrite of `../main.tex` (and `../paper.md`) with
one narrative line:

> Pooled AUC on enzyme-substrate benchmarks certifies protein popularity,
> not binding. A molecule-blind null baseline detects the shortcut; a
> four-ingredient ranking recipe (RankBind) breaks it; both transfer to
> seven datasets.

The old papers are untouched. `../main.tex` (the long version) remains the
evidence appendix: every number in this manuscript is reproduced there with
its source run directory, and `REPRODUCIBILITY.md` lists the commands that
emit each table and figure.

## How it maps to the poster

The manuscript follows the ScaDS.AI poster (`../poster_scads/`) panel for
panel:

| Poster column | Manuscript section |
|---------------|--------------------|
| 1. The shortcut | Results 2.1 (BRENDA-200, null baseline, response maps) |
| 2. RankBind     | Results 2.2 (four ingredients, ablation) and 2.3 (residue attention) |
| 3. Does it generalise? | Results 2.4 (seven datasets, matched BCE control, pool scaling, two honest non-wins) |
| 4. Architecture | Methods 4.3 (encoders, projectors, bilinear head) |

Figures are reused from `../poster_scads/figures/` where possible
(`fig_respmaps.pdf`, `fig_ablation.pdf`, `fig_jaccard.pdf`,
`fig_attention.pdf`); the seven-dataset comparison is in Table 3 instead of
the poster's `fig_datasets.pdf`.

## Build

```bash
module load texlive
make
```

11 pages, ~3,900 words of body text, 3 tables, 4 figures. The manuscript is
written in plain `article` class with the Scientific Reports section
structure (Abstract, Introduction, Results, Discussion, Methods, Data
availability, References); moving it into the Springer Nature LaTeX
template (`sn-article`) is a preamble-only change.

## Deliberate choices

- **`null_prot_prior` as the anchor baseline.** The poster calls it "cheat
  sheet"; the manuscript keeps the repo's name and defines it at first use.
- **Per-molecule AUC on the 200x200 matrix (n=30), not the test-pair
  version (n=4).** The n=4 version is reported in the text as the
  continuity metric prior work uses; the n=30 matrix version is what the
  head-to-head against the baselines rests on.
- **Honest non-wins in the main table.** Davis and KIBA stay in Table 3
  with their low MRR, because removing them would overstate the claim.
- **The pool-scaling lesson in the main text.** The fixed-pool drop on
  BRENDA+SABIO (0.134/0.023/0.040) is reported together with the scaled-pool
  values (0.228/0.219/0.344); the transfer rows are single-seed and say so.