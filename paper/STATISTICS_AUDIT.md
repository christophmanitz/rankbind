# STATISTICS_AUDIT.md — skill Phase B

Audit date: 2026-08-22. Scope: every quantitative claim in
`paper/scirep/main.tex` at commit `63de30e`.

## Policy (per skill §23)

- No inferential rescue: we do not use significance tests to prop up weak
  effects. Effect size + seed reproducibility are the reporting standard.
- SD, not SEM, everywhere (`±` = SD over seeds unless stated otherwise).
- Exact P values: only in the paired molecule-level bootstrap
  (PAIRED_MOLECULE_STATS.md), reported with the test statistic and n.

## Unit of analysis per headline number

| claim | unit | n | seeds | uncertainty |
|---|---|---:|---:|---|
| baselines' matrix MRR ≈ chance / RankBind 0.220±0.026 | molecule row (30 rows) | 30 molecules × 200 proteins | 3 {42,7,1337} | per-seed SD + paired bootstrap CI over molecules |
| pooled AUC of 4 baselines vs prior cap | pair | full split (~1.9k test pairs) | 1 (pinned ckpts) | construction argument (null table) replaces CIs |
| lig_prior pooled AUC 0.915 / prot_prior 0.500 | pair | full split | deterministic | exact by construction |
| decoy-probe AUCs (0.887/0.833/0.603) | pair | 9632 rows, split-stratified folds | 5 folds | fold SD in probe JSON |
| synthetic dissociation (4.4× MRR @ matched AUC) | simulation | 50 sims/regime | fixed base seed | sim-level spread in CSV |
| seven-dataset transfer | benchmark | per-dataset matrices (n varies) | 1 (+bs peak-seed adds pending) | marked as single-seed in text |

## Checks performed

1. **SD vs SEM**: manuscript uses `mean ± SD` consistently; no SEM anywhere.
   OK.
2. **Bootstrap methodology**: paired over the same 30 molecules across
   models (10k resamples, percentile CIs); documented in
   PAIRED_MOLECULE_STATS.md; generator committed
   (`evaluation/paired_molecule_stats.py`). OK.
3. **Paired tests**: Wilcoxon signed-rank on paired molecule-level MRR
   (RankBind vs best baseline) reported once, two-sided, exact n=30;
   framed as effect-size corroboration, not gatekeeping. OK.
4. **Multiple comparisons**: no family of >3 tests exists; ablation
   reading is descriptive (means + SDs + ranges), not tested. OK.
5. **Chance levels stated**: MRR chance ≈ 0.029 (200 candidates),
   H@K analytic, per-molecule AUC 0.5. Stated at every use. OK.
6. **Deterministic nulls**: null-baseline table is exact
   (prot_prior = 0.500 by construction); tie-aware vs raw MRR distinction
   documented so the constant-scorer MRR=1.0 artifact cannot leak into
   tables. OK.

## Pending items (blocked on cluster jobs)

- [DONE 2026-08-22] A4 selection sensitivity (`abl_mrrsel`, 3 seeds):
  matrix-MRR checkpoint selection gives 0.183±0.055 vs default
  (pooled-AUC selection) 0.220±0.026 — overlapping ranges, conclusion
  selection-invariant; reported as a Table-2 row + protocol paragraph.
- [DONE 2026-08-22] Protocol-A multiseed sweep: **invalidated the old
  table** (referee finding #15, commit 6d685af — pre-fix s7/s1337 runs
  trained on their own split but were evaluated on the canonical one,
  ~86% train-pair leakage). Canonical numbers now from the pinned-split
  v5 sweep + clean seed-42 anchors:
  `phase2_rankbind_multiseed.csv` (regenerated);
  old file preserved as `phase2_rankbind_multiseed_v4_INVALID_leak.csv`.
- [PENDING] Transfer per-seed CSVs (A5): km_with_decoys hp3400 s1337
  (SLURM 27295353) still running; s42/s7 landed for all three bs_v3
  datasets, turnover s1337 landed.
- [PENDING] Phase E reviewer files: run once A5 lands so reviewers see
  final transfer tables.
