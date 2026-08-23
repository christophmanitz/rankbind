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
| C4 | The pooled points instead ride molecule-level regularities (lig_prior 0.915; decoys are role-pure) | null table + decoy probe | BRENDA-200 | 9632 pairs / 4604 ligands | 5 folds | frozen-feature probe, linear head only | Yes |
| C5 | Same encoders + BCE vs same encoders + ranking objective → different shortcut behaviour | abl_bce_only vs default (matched capacity) | BRENDA-200 | as C1 | 3 | architecture held at MLP for bce_only (head differs); bilinear+BCE variant in v3 lineage | Yes |
| C6 | \method lifts matrix MRR 0.014→0.220±0.026, per-molecule AUC to 0.878±0.020, H@10 0→0.598±0.045 | multiseed CSV (pinned-split v5 sweep) | BRENDA-200 | 30 mols | 3 | replaces invalidated 0.326±0.072 (split leakage in pre-fix sweep); molecule-level paired bootstrap (pinned split, seed-paired runs) gives CI +0.23 [+0.16,+0.32] / +0.20 [+0.11,+0.31] / +0.17 [+0.11,+0.23] for seeds 42/7/1337, all excluding zero | Yes |
| C7 | Margin loss is the dominant lever (removal collapses MRR ~11×; pooled AUC rebounds to 0.91) | abl_no_margin row | BRENDA-200 | as C1 | 3 | effect size large vs seed noise | Yes |
| C8 | Balanced sampler secondary positive (removal costs 33% MRR) | abl_no_sampler row | BRENDA-200 | as C1 | 3 | wide SDs overlap on H@K | Yes, phrased as secondary |
| C9 | Bilinear head buys seed stability (SD 0.026 vs 0.087 = 3.3× narrower) and higher mean under pinned split | multiseed CSV ranges | BRENDA-200 | as C1 | 3 | 3 seeds is minimal for a variance claim — phrase as observation | Yes, softened |
| C9b | Headline ranking result is selection-rule invariant (matrix-MRR selection 0.183±0.055 vs pooled-AUC selection 0.220±0.026) | abl_mrrsel row (A4) | BRENDA-200 | as C1 | 3 | selection variant slightly lower mean; overlapping ranges | Yes |
| C10 | Findings transfer across seven enzyme-substrate benchmarks (direction preserved everywhere) | seven-dataset table | kcat/KM, KM, kcat, ESP + variants | per-benchmark matrices | 3 for BRENDA+SABIO (bs_transfer_per_seed.csv), 1 elsewhere | seed ranges quoted in limitations; DONE 2026-08-22 | Yes with single-seed caveat |
| C11 | Gini is a dataset-geometry artefact, not a pathology detector | phase-1 null baseline | BRENDA-200 | 200 proteins | deterministic | — | Yes |
| C12 | Pooled AUC validates the wrong property (methodological thesis) | C1–C5 combined | BRENDA-200 primary | — | — | enzyme-family scope; not all DTI | Yes, scoped |
| C13 | Synthetic construction shows matched pooled AUC with 4.4× MRR gap across regimes | synthetic_experiment.py | synthetic | 50 sims/regime | fixed | toy model, calibration documented | Yes |
| C14 | Per-molecule AUC on test pairs rests on n=4 | metrics section states it | BRENDA-200 | 4 | — | demoted, no inference drawn | Yes as demotion note |
| C15 | The dissociation does not require ligand recurrence: under a cold-ligand split the BCE control keeps pooled AUC 0.850±0.018 at matrix MRR 0.028±0.013 (= chance), while the ligand prior collapses to 0.500 and the protein prior now carries signal (0.655) | cold_split_multiseed.csv; NULL_BASELINE_ligand.md | BRENDA-200 cold-ligand split | 1495 test pairs / pool 72 matched positives | 3 | one corpus; author-generated decoys shared with canonical construction | Yes |
| C16 | Under a double-cold split (no molecule or protein recurs; both marginals exactly 0.500) the BCE control still posts pooled AUC 0.813±0.010 at MRR 0.025±0.006 — pooled separation persists where memorisation cannot | cold_split_multiseed.csv; NULL_BASELINE_double_cold.md | BRENDA-200 double-cold split | 213 test pairs / 6 matched positive pool pairs | 5 | tiny pool surface → matrix metrics indicative only; mechanism sustaining residual pooled AUC unresolved (similarity structure vs. decoy statistics) — flagged as open question in §2.4 | Yes |
| C17 | RankBind's ranking advantage survives removal of all recurring identities, at reduced magnitude: MRR 0.295±0.064 (cold-ligand), 0.092±0.078 (double-cold, overlapping ranges, widest seed noise) vs control 0.028/0.025 | cold_split_runs.csv + multiseed aggregate | as C15/C16 | as C15/C16 | 3 / 5 | advantage narrows where training signal is thinnest; recovery under double-cold left open | Yes, with explicit reduction reported |

Removed/weakened during audit:
- ~~"In this regime pooled AUC certifies the protein marginal, nothing
  else"~~ — contradicted by C3; replaced by two-axis phrasing.
- ~~"a molecule-blind baseline passes [pooled AUC] just as easily"~~
  (Discussion) — false for the protein prior (0.500); rewritten.
- ~~Old multiseed table (0.326±0.072 etc.)~~ — invalidated by referee
  finding #15 (split leakage in pre-fix s7/s1337 runs, commit 6d685af);
  replaced with pinned-split protocol numbers throughout.
- Residue-attention-pool claim demoted from three-seed (0.427±0.123,
  invalid) to single-seed anchor observation (0.316 vs 0.247 on seed 42).
