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
| baselines' matrix MRR ≈ chance / RankBind 0.326±0.072 | molecule row (30 rows) | 30 molecules × 200 proteins | 3 {42,7,1337} | per-seed SD + paired bootstrap CI over molecules |
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

- [PENDING] Protocol-B model selection (`abl_mrrsel`, SLURM 27295847-49):
  sensitivity table A4 — will state whether headline numbers depend on the
  checkpoint-selection protocol.
- [PENDING] Protocol-A multiseed sweep (split pinned via
  `data.split_seed=42`): refreshes the 3-seed table under pinned splits;
  current Table 2 cites the 20260423 sweep whose split was drawn inside
  each run (same params → identical split, verified for s42 anchors).
- [PENDING] Transfer per-seed CSVs (A5): bs peak-seed additions landing;
  transfer stays labelled single-seed until then.
- [PENDING] Phase E reviewer files (JCIM_REVIEW.md etc.) are deliberately
  deferred until these land, so reviewers see final tables.
