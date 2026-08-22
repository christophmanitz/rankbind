# JCIM_REVIEW.md — Phase E, Reviewer A (computational chemistry / DTI)

Reviewed manuscript: `paper/scirep/main.tex` at commit `c07c9ff`
(13 pp., Scientific Reports format). Review conducted 2026-08-22 per
skill §28. Stance: hostile but fair.

## 1. Strongest claim

> On enzyme–substrate benchmarks, four published DTI models pass pooled
> AUC while ranking the true target of each molecule at chance; a matched
> BCE control reproduces this on seven datasets; a ranking-oriented
> objective (RankBind) restores ligand-conditional ranking on every
> catalytic enzyme dataset.

This is well-supported and, to my knowledge, correct.

## 2. Supporting evidence

- Table 1 dissociation with molecule-blind prior and construction
  argument for the 0.500 cap — clean.
- Matched-capacity BCE control inside the authors' own codebase
  (Table 2) — this is the right control and many papers skip it.
- Seven-dataset transfer with per-dataset BCE controls (Table 3),
  including two honest non-wins (Davis metric artefact, KIBA
  non-convergence).
- The synthetic experiment (matched pooled AUC, 4.4× MRR gap) isolates
  the metric critique from the dataset.

## 3. Insufficient evidence

1. **Baseline checkpoints are single-seed and pinned.** DrugBAN/MolTrans/
   GraphDTA/GEMS numbers come from one training run each. The paper's own
   seed-spread data (±0.05 MRR on transfer) suggests single runs can move;
   the same could hold for baselines' pooled AUC (less likely for ranking
   at chance, but state it).
2. **"Substrate of" ≠ "binds".** The paper declares the shorthand, but the
   kinetic-threshold datasets (kcat/KM etc.) inherit thresholding choices
   that make "interaction" label noise heterogeneous across EC classes.
   No sensitivity analysis to the decoy Tanimoto band (0.3–0.8) is given.
3. **Decoy protocol is the paper's own pipeline** (MolTransformer-generated,
   Pareto-frontier). The claim "the decoy protocol makes memorisation
   trivial" is simultaneously the diagnosis and an artefact of the
   benchmark design the authors built. A reader can accept the metric
   argument (synthetic experiment) yet ask how much of the *empirical*
   pooled-AUC inflation transfers to community benchmarks (BindingDB-style,
   no explicit decoys).
4. **Chemical plausibility of RankBind's wins is not probed.** No example
   rankings, no enrichment of known promiscuous scaffolds, nothing that
   connects the 0.220 MRR to chemistry a medicinal chemist would recognise.
5. **Attention-audit section is a curiosity here**, not evidence for the
   core claim; hydropathy correlation (+0.24) is weak material for a
   main-text section.

## 4. Falsification experiments

- Re-run the four baselines under multi-seed; if their matrix ranking
  leaves chance, the headline dissociation collapses.
- Evaluate on a decoy-free benchmark with cold molecules AND cold targets
  (e.g., BindingDB cold-split); if BCE controls pass pooled AUC there via
  molecule memorisation too, the mechanism claim generalises; if not, the
  pathology is partly decoy-injected.
- Vary decoy Tanimoto band; if lig_prior pooled AUC drops toward 0.5 as
  similarity decreases, the "molecular memorisation" axis quantifies away.

## 5. Rejection risks

- "Benchmark constructed by the authors shows pathology in models trained
  on it" — circularity concern (mitigated by ESP/kinase rows, but a
  reviewer may still press).
- Novelty framing: Pahikkala/Wallach already made the evaluation-regime
  point; JCIM reviewers know Wallach & Heifets intimately. The paper must
  be scrupulous that its novelty is the *protein-marginal geometry vs
  molecule-memorisation decomposition* + the recipe, not "pooled metrics
  mislead".
- KIBA non-convergence reads as tuning weakness on the only affinity set
  with real signal structure.

## 6. Required changes before submission

1. State baseline single-seed status in Table 1 caption (currently only
   implied by "pinned checkpoints" language elsewhere).
2. Add one sentence in §2.4 acknowledging that the BRENDA+SABIO decoys are
   author-generated, and that the synthetic result carries the metric-side
   argument to decoy-free settings.
3. Soften "certifies memorisable dataset-level regularities" in the
   abstract to name both axes once (molecule identity chiefly, protein
   marginal in geometry) — currently the abstract names only
   "memorisable regularities", the intro only molecule recurrence +
   marginal shaping; keep the wording symmetric.
4. Move or compress §2.3 (attention audit) to make room for a short
   chemical-example paragraph (one molecule's top-5 ranked proteins with
   BRENDA annotations).
5. Data availability: "available on request" for score matrices will not
   survive review at JCIM; commit the small artefacts (few MB) instead.

*Verdict: major revision; core finding sound.*
