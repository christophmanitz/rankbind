# HP-sweep integration plan

This file documents how the in-flight Stage-1 `hard_pool_size` sweep on
BRENDA+SABIO with-decoys datasets will be folded back into the paper,
*before* the sweep results land. It exists to keep authorship honest:
the decision rules below are written down in advance so the eventual
write-up cannot drift toward whatever framing happens to flatter the
numbers we get.

## What is in flight

- Submitted: 2026-05-04 (jobs `21533777`, `21533815`–`21533825`).
- Configs: `v5_rankbind/configs/sweeps/hp_brenda_sabio/` — 12 JSONs,
  one per (dataset, hp value).
- Submission script: `scripts/run_v5_brenda_sabio_hp_sweep.sh`.
- Tag scheme: `bs_v2_hp<value>`.
- Sweep grid (`hard_pool_size` only; everything else stays at the v4
  default that produced the BRENDA-200 headline numbers):

  | Dataset  | n_train_proteins | hp values             | Coverage          |
  |----------|-----------------:|-----------------------|-------------------|
  | kcat_km  |          2,671   | 100 / 300 / 700 / 1400 | 3.7 % … 52 %      |
  | km       |          6,650   | 250 / 700 / 1700 / 3400 | 3.8 % … 51 %     |
  | turnover |          3,972   | 150 / 400 / 1000 / 2000 | 3.8 % … 50 %     |

- Reference points (single seed, hp = 50, the bs_v1 baseline):
  - kcat_km MRR = 0.134, km MRR = 0.023, turnover MRR = 0.040.
- BRENDA-200 anchor (3-seed v4 default): MRR = 0.326 ± 0.072.

## Reading the sweep

The sweep is reduced into one row per (dataset, hp value) by the
existing aggregator `scripts/collect_v5_runs.py` walking
`results/v5_rankbind/*bs_v2_hp*/manifest.json`. Primary metric: matrix
MRR on the 200×200 evaluation pool, since that is what the paper
already gates on.

For each dataset we extract:

1. The hp value with the highest matrix MRR (the **per-dataset peak**).
2. Whether the curve is monotone, plateauing, or unimodal across the
   four sampled values.
3. The headline triple at the peak hp: matrix MRR, Hit@10, and the
   `null_prot_prior` row-Spearman + Top-10 Jaccard from the same probe
   we ran on the bs_v1 runs (`evaluation/null_prior_probe_brenda_sabio.py`,
   re-pointable to the new run dirs by editing the `RUNS` list).

A consolidated CSV `evaluation/attractor_results/hp_sweep_brenda_sabio.csv`
with one row per (dataset, hp) is the artefact this stage produces.

## Decision rules

The four outcomes below are pre-registered. Each names the change to
the paper before any new MRR number is read.

### A. Major win — peak MRR ≥ 0.25 on at least one dataset

This is the threshold below which the paper would still claim no
competitive enzyme-wide result. Reaching it on any of the three
datasets is paper-relevant evidence that the recipe scales when its
core hyperparameter does.

Paper changes:

- §8.1 ("Does the recipe survive larger, enzyme-wide data?") gets a
  follow-up paragraph reporting the peak (dataset, hp, MRR) tuple and
  explicitly retracts the "absolute ranking quality drops" claim for
  that dataset.
- A new minor row is added to the §8.1 table (`bs_v2_hp<peak>`)
  alongside the existing bs_v1 row, so the within-section before/after
  is visible.
- The §8.2 recipe item 4 (hard-negative scaling) cites the actual peak
  coverage rather than the prospective "10–25 %" range.
- Conclusion mentions the enzyme-wide finding in one sentence.
- Abstract is **not** rewritten to claim enzyme-wide competitiveness —
  the abstract stays a BRENDA-200 paper. We avoid abstract scope-creep
  on a single-seed sweep result.

### B. Modest win — peak MRR in [0.15, 0.25) on at least one dataset

Worth reporting; not a headline shift. Same §8.1 paragraph and table
addition as A. §8.2 item 4 is updated to cite the actual coverage at
peak. Conclusion and abstract unchanged. This is the most likely
outcome a priori given the bs_v1 baselines.

### C. Marginal — peak MRR in [0.10, 0.15) on every dataset

The sweep moved the needle but not enough to support a "scaling fixes
it" claim. §8.1 gets a single sentence: "Stage-1 hard_pool_size sweep
(see Appendix B) yields a peak MRR of <X> on <dataset> at hp=<value>;
the qualitative gap to BRENDA-200 persists." A small Appendix B is
added with the four-point curve per dataset and no further narrative
change.

### D. No improvement — peak MRR ≤ bs_v1 baseline on every dataset

The pre-registered hypothesis (hp_pool too small) is refuted. §8.1 is
revised to say so explicitly: "A Stage-1 sweep over `hard_pool_size`
∈ {3.7 % … 52 %} of n_train_proteins did not lift matrix MRR above
the bs_v1 baseline on any of the three datasets, ruling out fixed-pool
sizing as the dominant cause of the absolute-MRR drop." §8.2 item 4
is rewritten to drop the scaling recommendation; the recipe shrinks
to seven steps. Future work in §9 changes from "scale hard-pool" to
"investigate alternative cause: pairs-per-protein, lr scaling, or
loss-rebalancing".

## Multi-seed follow-up

Single-seed sweep numbers are reported in the paper as preliminary
("single-seed, see manifest XYZ for hash") regardless of outcome.
Only after a winning hp value is rerun across seeds {42, 7, 1337} —
matching the three-seed protocol used for the BRENDA-200 headline —
do its numbers replace the single-seed entry. The paper does not get
multi-seed enzyme-wide error bars in the v1 submission unless those
runs land before submission.

## What is *not* swept here

By design this sweep moves only `hard_pool_size`. The four other
candidates (`pairs_per_protein_per_epoch`, `lr`,
`n_negatives_per_positive`, `margin`) are held fixed so the curve is
a 1-D slice the paper can claim is interpretable. If outcome C or D
puts a second sweep on the agenda, the next slice would be `lr` on
the winning hp value of the largest dataset (`km`), again 1-D. We do
not run a 4-D grid.

## When to fold this in

- **Before paper submission**: outcomes A or B trigger §8.1 paragraph
  + table row + §8.2 number tightening + small Appendix B with the
  four-point curve.
- **During revision**: outcome C → small Appendix B only. Outcome D →
  §8.1 + §8.2 + §9 revision per the rule above.
- **After submission, never**: rewriting the abstract on a
  post-submission sweep result. If single-seed enzyme-wide MRR ≥ 0.30
  arrives after submission, it goes in v2 as a new contribution, not
  as an abstract patch on v1.
