# CLAIM_EVIDENCE_MATRIX.md — skill §29

Every strong scientific statement in Abstract / Introduction / Results /
Discussion of `paper/scirep/main.tex` (commit `63de30e`), mapped to its
evidence. Rule applied: BRENDA-200 claims say BRENDA-200; enzyme-wide
generalisations only where the seven-dataset table supports them.

| # | Claim | Evidence | Dataset | n | Seeds | Limitation | Keep? |
|---|---|---|---|---:|---:|---|---|
| C1 | Four published DTI models pass pooled AUC but fail ligand-conditional ranking (per-row AUC 0.43–0.53) | Table 1; test_set_eval.py matrices | BRENDA-200 | 200×200 matrix, 30 test mols | 1 (pinned ckpts) | single checkpoint per baseline | Yes |
| C2 | Their score structure matches a molecule-blind protein prior (Gini ≈0.995, Jaccard 54–67%) | cross_model_overlap.csv, gini_comparison.csv | BRENDA-200 | 200 proteins | 1 | descriptive geometry, not inference | Yes |
| C3 | Protein-marginal copying cannot lift pooled test AUC past chance on this split (cap 0.500) | null_baseline_table.py construction argument | BRENDA-200 full split | all pairs | deterministic | exact, no sampling error | Yes |
| C4 | The pooled points instead ride molecule-level regularities (lig_prior 0.915; decoys are role-pure) | null table + decoy probe | BRENDA-200 | 9632 pairs / 4574 ligands | 5 folds | frozen-feature probe, linear head only | Yes |
| C5 | Same encoders + BCE vs same encoders + ranking objective → different shortcut behaviour | abl_bce_only vs default (matched capacity) | BRENDA-200 | as C1 | 3 | architecture held at MLP for bce_only (head differs); bilinear+BCE variant in v3 lineage | Yes |
| C6 | \method lifts matrix MRR 0.015→0.326±0.072, H@10 0.03→0.755±0.095 | multiseed CSV | BRENDA-200 | 30 mols | 3 | SD is wide (seed luck); CIs in PAIRED_MOLECULE_STATS.md | Yes |
| C7 | Margin loss is the dominant lever (removal collapses MRR ~8×; pooled AUC rebounds to 0.95) | abl_no_margin row | BRENDA-200 | as C1 | 3 | effect size large vs seed noise | Yes |
| C8 | Balanced sampler secondary positive (+78% MRR vs no_sampler... direction: removal halves it) | abl_no_sampler row | BRENDA-200 | as C1 | 3 | overlaps seed noise on MRR mean | Yes, phrased as secondary |
| C9 | Bilinear head buys seed stability (SD 0.072 vs 0.161), not mean wins | multiseed CSV ranges | BRENDA-200 | as C1 | 3 | 3 seeds is minimal for a variance claim — phrase as observation | Yes, softened |
| C10 | Findings transfer across seven enzyme-substrate benchmarks (direction preserved everywhere) | seven-dataset table | kcat/KM, KM, kcat, ESP + variants | per-benchmark matrices | 1 (+bs peak-seed pending) | single-seed except noted; [PENDING] A5 CSVs | Yes with single-seed caveat |
| C11 | Gini is a dataset-geometry artefact, not a pathology detector | phase-1 null baseline | BRENDA-200 | 200 proteins | deterministic | — | Yes |
| C12 | Pooled AUC validates the wrong property (methodological thesis) | C1–C5 combined | BRENDA-200 primary | — | — | enzyme-family scope; not all DTI | Yes, scoped |
| C13 | Synthetic construction shows matched pooled AUC with 4.4× MRR gap across regimes | synthetic_experiment.py | synthetic | 50 sims/regime | fixed | toy model, calibration documented | Yes |
| C14 | Per-molecule AUC on test pairs rests on n=4 | metrics section states it | BRENDA-200 | 4 | — | demoted, no inference drawn | Yes as demotion note |

Removed/weakened during audit:
- ~~"In this regime pooled AUC certifies the protein marginal, nothing
  else"~~ — contradicted by C3; replaced by two-axis phrasing.
- ~~"a molecule-blind baseline passes [pooled AUC] just as easily"~~
  (Discussion) — false for the protein prior (0.500); rewritten.
