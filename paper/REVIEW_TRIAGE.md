# REVIEW_TRIAGE.md — Phase E follow-up

Triage of the three hostile reviews (JCIM_REVIEW.md, ML_REVIEW.md,
SCIREP_REVIEW.md) against `main.tex` at the review commit `c07c9ff`.
Status as of 2026-08-22.

## Implemented immediately (same day)

| Item | Source | Fix |
|---|---|---|
| Framing contradiction with two-axis finding | all three, implicitly | Abstract + intro mechanism paragraph rewritten ("memorisable dataset-level regularities"; molecules-recurrence + marginal-shaping); §2.1 retitled "Pooled metrics pass where ligand-conditional ranking fails" |
| Gini-residual undefined in paper | B (implied) | Definition added to Methods §4.2 |
| Baselines single-run not stated | JCIM #1, SCIREP #1 | Table 1 caption states one training run per baseline; \method seed ranges added |
| "Statistically indistinguishable" without test | ML #2 | Replaced by range-overlap statement |
| Selection-rule confound unexplained | ML #3 | Protocol-check paragraph now reports n_val = 2 positive rows → matrix-MRR selection is selection on noise; default kept deliberately |
| Tie convention unstated | ML #4 | Metrics §4.2 documents strict-greater counting + constant-scorer caveat |
| Decoy circularity unacknowledged | JCIM #3 | §2.4 provenance sentence: author-generated decoys; synthetic construction carries metric argument to decoy-free settings |

## Deferred — needs new experiments or cluster time

1. **10-seed head comparison** (ML #1): bilinear vs MLP SD ratio from
   3v3 is anecdotal. Cheap (~15 GPU-h). Queue after current jobs.
2. **Paired molecule-level bootstrap CIs in main tables** (ML #2/#5):
   machinery exists (`evaluation/paired_molecule_stats.py`), must be
   refreshed against pinned-split runs and added to Table 2 rows.
3. **Double-cold holdout** (SCIREP #6, JCIM falsification): molecules AND
   proteins unseen — the decisive test whether RankBind learns interaction
   vs molecule identity. New run set, medium cost.
4. **EC-class stratification** of BRENDA-200 dissociation (SCIREP #4).
5. **Decoy-band sensitivity** (JCIM #4): regenerate decoys at narrower/
   wider Tanimoto bands; quantify lig_prior pooled AUC vs band.
6. **BCE-control budget parity on transfer** (ML #6): give controls a
   tuned threshold / class weight.
7. **Chemical example paragraph** (JCIM #5): one molecule's top-5 ranked
   proteins with BRENDA annotations.
8. **Commit score matrices** to satisfy data availability (JCIM #6):
   few MB each, gitignore exception needed.

## Terminology pass (SCIREP #2/#3) — DONE 2026-08-22

Agent line-by-line scan of all term occurrences: title scoped to
"Drug--Target Interaction Benchmarks"; abstract + recipe "not binding" →
"not ligand-conditional discrimination"; "non-binding" → "non-substrate"
in the hard-negative method text; shorthand declaration repeated in §2.1.
All remaining occurrences verified OK-general / OK-shorthand / OK-affinity.
