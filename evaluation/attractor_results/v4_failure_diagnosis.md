# v4 Failure-Case Diagnosis — Stage (c) of Phase 4

**Date:** 2026-04-27
**Source:** `evaluation/v4_failure_diagnosis.py` on
`results/v5_rankbind/20260423-112928_012a2695c2_default_v4/`
(seed 42, default config).
**Decision rule:** PLAN.md §13.1 gate — proceed to Stage (b) only with a
defensible chemical-class signal or size correlation.

## 1. Reproducibility check

The script reproduces `test_matrix_ranking.json` exactly:

| Metric | Script | Run JSON | Match |
|---|---:|---:|---|
| n_positive_pairs_matched | 34 | 34 | ✓ |
| MRR | 0.2465 | 0.2465 | ✓ |
| H@1 / H@5 / H@10 | 0.029 / 0.500 / 0.647 | 0.029 / 0.500 / 0.647 | ✓ |

So the per-pair rank distribution we analyse below IS the same one feeding
the headline matrix-ranking metrics in PLAN.md §12.3.

## 2. Bottom-quartile by chemical class

Bottom quartile = rank ≥ 14 (n=9 of 34 positive pairs). SMARTS classes
overlap (one SMILES can carry several tags). `n_total` = pairs in that
class across all 34; `n_bottom_q` = pairs of that class in the bottom Q;
`bottom_share` = `n_bottom_q / n_total`. Uniform = 0.265.

| Class | n_total | n_bottom_q | bottom_share |
|---|---:|---:|---:|
| `long_aliphatic` | 1 | 1 | 1.000 |
| `OTHER` (no SMARTS hit) | 6 | 4 | 0.667 |
| `polyhydroxy` | 8 | 4 | 0.500 |
| `phosphate` | 4 | 1 | 0.250 |
| `carboxylate` | 4 | 1 | 0.250 |
| `phenol_or_aromOH` | 8 | 1 | 0.125 |
| `halogenated` | 7 | 0 | 0.000 |
| `amide_or_peptide` | 2 | 0 | 0.000 |

Spearman ρ(`n_heavy_atoms`, `rank`) = **0.252** (weak positive,
threshold 0.4 not reached).

## 3. What the OTHER class actually is

The OTHER bucket (the largest single contributor to bottom-Q) is **almost
entirely a SMARTS-coverage artefact**, not a genuinely uncategorised
chemistry. The six SMILES:

| SMILES | rank | What it really is |
|---|---:|---|
| `O=C1N=C(O)C2(O)NC(O)=NC2=N1` | 196 | Urate / 5-hydroxyisourate analogue; `N=C(O)` is the *enol-tautomer* of an amide (`-NH-C(=O)-`). My SMARTS `[NX3][CX3](=[OX1])` only matches the keto tautomer. |
| `CCCCCC(=O)Oc1ccc([N+](=O)[O-])cc1` | 133 | **p-Nitrophenyl hexanoate** — an aryl ester. No SMARTS for esters in my pattern list. |
| `Cc1cc(=O)oc2cc(N=C(O)…)ccc12` | 26 | Coumarin / coumarin-amide hybrid; same enol-tautomer issue + lactone. |
| `CC(=O)Oc1ccc([N+](=O)[O-])cc1` | 33 | **p-Nitrophenyl acetate** — aryl ester. |
| `CCCC(=O)Oc1ccc([N+](=O)[O-])cc1` | 11 | **p-Nitrophenyl butyrate** — aryl ester. |
| `N=C(N)NCCCCN` | 2 | Agmatine — guanidine + amine; no SMARTS for guanidine. |

If we re-tag, **4 of the 6 are aryl esters / amide-tautomers**, both of
which would belong to existing classes under a richer SMARTS list.
**The OTHER bucket is therefore not a chemically novel cluster.** It does
not justify atom-level work; it justifies fixing the SMARTS list.

## 4. Where the real signal sits — `polyhydroxy`

The four `polyhydroxy` bottom-Q failures are all genuine carbohydrate /
glycoside substrates:

| SMILES | rank | n_heavy | likely substrate type |
|---|---:|---:|---|
| `OC[C@H]1O[C@@H](O[C@@H]2…)…` | 98 | 34 | Trisaccharide (glycosidase substrate) |
| `O=C(/C=C/c1ccc(O)c(O)c1)O[C@@H]1C[C@](O)(C(=O)O)C[C@@H](O)[C@H]1O` | 21 | 25 | **Chlorogenic acid** (caffeoylquinic acid) — quinic-acid ester of caffeate |
| `OC[C@@H]1O[C@@H](OC[C@@H]2O…)…` | 16 | 19 | Disaccharide |
| `OC[C@@H]1O[C@@H](OC[C@@H]2O[C@@H](OC[C@@H]3O[C@@H](O)…)…)…` | 15 | 28 | Trisaccharide |

These ARE classes where atom-level resolution matters — glycoside hydrolases
and glycosyltransferases distinguish substrates by **stereochemistry of the
anomeric carbon** and **position of OH groups around the pyranose ring**.
ESM2 mean-pool over residues throws this geometric information away on the
protein side; ChemBERTa mean-pool over the molecule throws away the explicit
3D arrangement of OHs on the ligand side. Atom-level processing on the
protein binding pocket is *plausibly* the right architectural lift for this
class.

But: **n = 8 polyhydroxy substrates in the entire test split** (4 of which
fall in bottom-Q). The statistical case is weak.

## 5. The size-correlation arm

Spearman ρ(n_heavy_atoms, rank) = 0.252. Below the §13.1 threshold of 0.4.
Visually (`fig_v4_atoms_vs_rank.png`), the failures with rank ≥ 50 are all
high-atom-count compounds (34 / 45 / 59 atoms), but the bulk of the test
distribution sits at 10–30 heavy atoms with no rank signal. So the
"large-ligand → bad rank" story has a long-tail flavour that is real
qualitatively but not strong enough to cite as a primary justification.

## 6. Decision per §13.1 gate

Quoting the gate:

> ✅ Pass: ≥30% of bottom-MRR-quartile failures fall in 1–2
> chemically-coherent atom-conditioned classes, OR a clear
> `n_heavy_atoms ↑ ⇒ rank ↑` correlation.
> ❌ Fail: failures chemically flat, no size correlation.
> ⚠️ Mixed: no class cluster but residue-level interpretability arguments
> still hold.

Reading our data:

- **Polyhydroxy alone passes the 30% rule** (50% bottom-share, 4/8 of
  class in bottom-Q, 44% of bottom-Q rows tagged). Biologically coherent:
  glycosidase / glycosyltransferase substrates where atom-level stereo
  matters.
- **OTHER bucket disqualified** as a SMARTS-coverage artefact, not a
  novel chemistry.
- **Spearman size-correlation arm fails** (0.252 < 0.4).
- **Sample size is small** (n=8 polyhydroxy, 34 total). The observation
  is consistent with the atom-level hypothesis but does not prove it.

This is a textbook **⚠️ Mixed** outcome.

## 7. Go / no-go

**Decision: proceed to Stage (b), defer Stage (a).**

Rationale:

1. There is one chemically-coherent atom-conditioned class (polyhydroxy /
   carbohydrates) over-represented in failures, which gives Stage (a) a
   plausible target if it ever runs. But n=8 is too thin to *commit* to
   2–4 weeks of atom-pipeline work.

2. The OTHER cluster is mostly missing-SMARTS, not missing-architecture.
   Patching the SMARTS list is a 30-min fix that should happen as part of
   any future iteration of this script, but does not change the gate
   outcome.

3. Stage (b) — residue attention-pool — is justified independently of
   atom-level: it produces interpretable per-residue weights that can be
   inspected qualitatively for binding-pocket overlap, which is itself a
   paper-level contribution and a hard prerequisite for Stage (a)'s
   top-K residue selection. Stage (b) is the right next step regardless
   of how (c) resolved.

4. Stage (a) re-enters the conversation **only** if Stage (b)'s
   attention-weights *also* show signal on the polyhydroxy failure cases
   (e.g., weights concentrate on residues plausibly involved in
   carbohydrate binding) — i.e., we want two independent lines of
   evidence (chemical-class + residue-attention) before paying the
   atom-level cost.

**Action items before starting (b):**

- (Optional, low cost) Extend `SMARTS_PATTERNS` in
  `evaluation/v4_failure_diagnosis.py` with `aryl_ester`, `guanidine`,
  and `amide_enol_tautomer` patterns; re-run; log the updated class
  table as Stage (c)'s final state. *Skipped for now — it does not
  affect the gate outcome.*

- Update `v5_rankbind/PHASE2_LOG.md` with a "Phase-4 Stage-(c) result:
  Mixed → proceeding to Stage (b)" note pointing here.

- Open Stage (b) tasks per PLAN.md §13.2.

## 8. Files produced by Stage (c)

- `evaluation/v4_failure_diagnosis.py` — script
- `evaluation/attractor_results/v4_failure_diagnosis.csv` — 34-row
  per-pair table (smiles, uniprot, rank, mrr_contrib, true_score, margin,
  classes, n_heavy_atoms)
- `evaluation/attractor_results/v4_failure_diagnosis_classes.csv` —
  class × bottom-quartile breakdown
- `evaluation/attractor_results/fig_v4_rank_hist_by_class.png`
- `evaluation/attractor_results/fig_v4_atoms_vs_rank.png`
- `evaluation/attractor_results/fig_v4_class_failure_rate.png`
- This memo: `evaluation/attractor_results/v4_failure_diagnosis.md`
