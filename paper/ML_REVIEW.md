# ML_REVIEW.md — Phase E, Reviewer B (machine learning / metrics)

Reviewed manuscript: `paper/scirep/main.tex` at commit `c07c9ff`.
Review conducted 2026-08-22 per skill §28. Stance: hostile.

## 1. Strongest claim

> The ranking result does not depend on the checkpoint-selection rule
> (Table 2: pooled-AUC selection 0.220±0.026 vs matrix-MRR selection
> 0.183±0.055), and the margin loss is the dominant lever (removal
> collapses MRR 11×).

## 2. Supporting evidence

- Three seeds per configuration under a pinned split with an eval-side
  guard that refuses split mismatches (§4.2) — this is exactly how it
  should be done, and the paper is candid that an earlier sweep leaked.
- Ablations at matched capacity; BCE control isolates the objective;
  mrrsel variant isolates model selection.
- Synthetic construction with matched marginals demonstrates metric
  non-identifiability in principle.

## 3. Insufficient evidence

1. **Three seeds cannot support the variance claims.** "Cuts the
   seed-to-seed standard deviation from 0.087 to 0.026" (bilinear vs MLP)
   compares two SDs each estimated from n=3. An F-test on 3v3 is
   meaningless; report ranges instead, or run ≥10 seeds for the head
   comparison only (cheap: 1.5 h/run).
2. **"Statistically indistinguishable" for mrrsel vs default** is asserted
   without a test and with visibly different spreads (0.055 vs 0.026).
   Either do a paired test over seeds (n=3, so paired bootstrap over
   molecules aggregated across seeds) or write "overlapping ranges".
3. **Selection-rule sensitivity is confounded**: mrrsel selects on matrix
   MRR computed against *validation* positives on the canonical axes, but
   validation has ~n=? positive rows; if val-positive rows are few, the
   selector is noisy, which explains its wider spread. Report n_val rows.
4. **Chance level of matrix MRR**: 0.029 assumes uniform random ranking,
   but ties and score clustering change E[1/rank]; the tie-aware vs raw
   distinction exists in the repo's null table but is not stated in the
   paper. State which convention Table 1–3 uses.
5. **No confidence intervals anywhere in the main tables.** SD over seeds
   is not a CI on the mean; the paired molecule-level bootstrap exists in
   the repo but its results are not in the manuscript.
6. **Hyperparameter provenance of the transfer rows**: pools were scaled
   post hoc after observing the pool-50 failure ("lifts MRR monotonically").
   This is fine as engineering, but the transfer comparison then benefits
   from a search the BCE control did not receive. Give the control the
   same budget (e.g., BCE + tuned decision threshold or class weight) or
   say explicitly why it is unfair-to-control by design.

## 4. Falsification experiments

- 10-seed rerun of bilinear vs MLP head: if SD ratio →1, the stability
  argument for keeping the bilinear head dies.
- Selection-rule × margin-loss factorial: does mrrsel hurt the no-margin
  configs equally? If mrrsel rescues nothing there either, the
  "selection doesn't matter" claim extends; if it interacts, the current
  one-factor sensitivity check is incomplete.
- Shuffle-label control: RankBind trained on permuted molecule–protein
  assignment should fall to chance MRR; if not, leakage remains.

## 5. Rejection risks

- Reviewers who compute that ±SD over n=3 with values like 0.222±0.385
  (Table 2, MLP top-ten overlap) will call the ablation underpowered and
  the ordering of middle rows arbitrary.
- The pinned-split fix story, while honest, invites the question of what
  else the pipeline got wrong; every number must trace to a manifest, and
  the paper promises REPRODUCIBILITY.md — that file must ship.

## 6. Required changes before submission

1. Replace all "statistically indistinguishable"/variance-ratio language
   with range-based statements, or add the paired tests (data already
   exist).
2. Add CIs (molecule-level bootstrap) to Table 2's \method row and the
   mrrsel row at minimum.
3. State the tie convention for MRR and Hit@K in §4.2.
4. State n_val positive rows used by both selectors.
5. Commit REPRODUCIBILITY.md with the artifact hashes (it is referenced
   three times).

*Verdict: major revision; methodology above the field's bar, statistics
below it.*
