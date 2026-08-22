# SCIREP_REVIEW.md — Phase E, Reviewer C (statistics / biology)

Reviewed manuscript: `paper/scirep/main.tex` at commit `c07c9ff`.
Review conducted 2026-08-22 per skill §28. Stance: hostile on n,
terminology and generalisation.

## 1. Strongest claim

> Pooled AUC on enzyme–substrate benchmarks validates the wrong property:
> models pass it via dataset-level regularities while failing per-molecule
> ranking; ranking the task directly (RankBind) fixes behaviour without a
> new architecture.

## 2. Supporting evidence

- The construction argument for the prior's 0.500 cap is exact, not
  sampled — the strongest single piece of evidence in the paper.
- Decoy-role census (1,417 substrate-only / 3,157 decoy-only ligands,
  99.3% pure) with a frozen-feature probe quantifying molecule-side
  memorisation (0.887 ligand-only vs 0.833 full) — appropriate use of a
  simple probe.
- Per-seed transfer CSV acknowledges ±0.05 seed noise rather than hiding
  it.
- Biological audit of attention (de-weights active sites, tracks
  hydrophobicity) — honest negative result, correctly demoted to a
  normalisation finding.

## 3. Insufficient evidence

1. **n=30 molecules carry all BRENDA-200 matrix claims.** MRR, H@K and
   matrix per-molecule AUC rest on 30 rows (34 positive cells). The paper
   says so, but conclusions like "de-striped" (Gini range 0.787–0.824 over
   three seeds of ONE model) are descriptive. No interval for Table 1's
   \method row at all: 0.618/0.878/0.798 appear as points.
2. **n=4 for test-pair per-molecule AUC** is properly demoted — good — but
   Table 1 still prints baselines' matrix AUCs (n=30) without any spread;
   are these one model each? One draw of 30 molecules?
3. **"Substrate" vs "binding" slippage persists in the abstract**
   ("Drug–target interaction models… scored with global AUC") while the
   labels are substrate annotations + generated decoys. Scientific Reports
   readers will not follow the §2.1 shorthand declaration. Every headline
   claim should say "enzyme–substrate annotation", not "interaction".
4. **EC-class composition unreported.** "Hydrolytic enzymes dominate" —
   which classes, what fraction? Generalisation from a hydrolase-heavy
   pool to "enzyme–substrate data" (abstract) needs either class balance
   or an explicit scope limit.
5. **Kinetic-threshold labels**: kcat/KM etc. binarise continuous kinetics;
   threshold choices and their sensitivity are undocumented here (the
   submodule pins them, but the paper must state them).
6. **Causal language outruns the design**: "the margin loss keeps the model
   honest" is a one-dataset, three-seed ablation with correlated
   ingredients (margin+mining share the candidate-set machinery). Fine as
   narrative; not yet causal.

## 4. Falsification experiments

- Stratify BRENDA-200 rows by EC class; if the dissociation concentrates
  in one class, scope the claim.
- Threshold sensitivity on one kinetic dataset (±1 binarisation band):
  do RankBind's wins survive?
- Hold out molecules AND proteins (double-cold): does RankBind retain
  MRR > chance when no molecule recurs? This directly tests whether the
  recipe learns interaction or molecule identity.

## 5. Rejection risks

- "n=30" visible in every table header invites small-sample criticism;
  without intervals the paper looks unaware of its own power.
- Terminology slippage ("binding", "interaction", "DTI" for substrate
  annotation) is the kind of thing Scientific Reports reviewers flag as
  misleading framing even when the authors declared the shorthand.
- The five-point recipe reads as universal; its evidence base is
  enzyme-substrate corpora plus two affinity sets where it mostly fails.

## 6. Required changes before submission

1. Add uncertainty to Table 1 (\method row: min–max or SD across seeds;
   baselines: mark "single run").
2. Global terminology pass: "substrate annotation" in abstract/intro/
   discussion headlines; keep "binds" only inside §2.1 after the
   declaration.
3. Report EC-class distribution of BRENDA-200 and the binarisation bands
   of the kinetic datasets in Methods.
4. Scope sentence for the recipe: "validated on enzyme–substrate corpora;
   affinity sets behave differently".
5. Double-cold experiment (or explicit limitation paragraph acknowledging
   it is missing) before claiming the recipe solves ligand-conditional
   ranking generally.

*Verdict: major revision; honest reporting throughout, but n and
terminology need tightening.*
