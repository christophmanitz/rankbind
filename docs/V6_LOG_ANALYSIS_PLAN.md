# v6 / DeltaField — Log-driven optimisation plan

**Purpose.** A closed-loop protocol: read the v6 training logs and eval
outputs, map each measurable signature to a diagnosed failure mode, and trigger
a *specific, novel* mechanism that attacks exactly that mode for our target
task — **ligand-conditional DTI ranking that does not ride the protein-prior
shortcut**. This is not a hyperparameter sweep. Each concept in §3 is a new
architectural / training device, gated by a trigger condition from §2, with a
pre-registered pass/fail check and a one-line novelty caveat.

Target task, restated so every decision below ties back to it:
> rank the *true* ligands of a protein above its non-binders **using the
> ligand→protein interaction only**, measured by matrix-MRR / Hit@k on the
> 200×200 score matrix, with Top-10 Jaccard-vs-`null_prot_prior` ≈ 0 and
> `rho_row` vs prior ≤ 0 as the anti-shortcut guarantee.

Status: **planning only.** Nothing here runs until the v6 baseline
(`abl_deltafield`, job 22205147) lands. The baseline's three pre-registered
checks (A ranking ≥ 0.326 MRR; B Jaccard ≤ 0.11 + zero-field test; C e_res
pocket-AUC > 0.5) are the *entry gate* to this plan — see
`docs/DELTAFIELD_CONCEPT.md` §8.

---

## 1. Instrumentation — what to extract, and what we must add first

### 1.1 Already in the logs (read directly)
| Source | Signal | What it tells us |
|---|---|---|
| `train_log.jsonl` | `margin_loss`, `pos_above_neg_max` per epoch | learning dynamics; separation of own hardest confuser |
| `train_log.jsonl` | `train_keep_ratio_mean`, `n_batches_skipped` | collator health (must be high / zero) |
| `train_log.jsonl` | `val_*` trajectory + early-stop epoch | did it converge or stop early on a noisy metric |
| `test_matrix_ranking.json` | MRR, Hit@5, Hit@10 | check A |
| `test_summary.json` | gAUC, per_lig_auc | shortcut vs ligand-conditional split |
| 200×200 matrix (`score_matrix.npy`) | full score field | null baselines, Jaccard, Gini-residual, `rho_row` (check B) |
| `e_res` / `e_atom` / `C` field outputs | per-residue/atom energy, contact map | check C via `attn_annotation_scan.py` |

### 1.2 New instrumentation to add to DeltaField before cycle 1
These are the load-bearing diagnostics the *difference-field* architecture
needs but a bilinear model never had. Add to `train.py::train_epoch_margin`
logging (cheap, no extra forward):

1. **Field participation ratio** `PR = (Σ e_res)² / (Σ e_res²)` per example,
   logged as epoch mean/median. `PR` = the *effective number of active
   residues*. A realistic pocket is ~5–25 residues; `PR → L` means the field is
   smeared (faking conditionality across the whole protein), `PR → 1` means
   collapse onto a single residue. **This single scalar is the early-warning
   signal for both failure modes the architecture risks.**
2. **Total field energy** `Ē = mean(Σ e_res + Σ e_atom)` per epoch — the
   collapse detector (`Ē → 0` = sparsity collapse, MRR will follow to chance).
3. **Negative-field energy** `Ē⁻ = mean field energy over the batch's hard
   negatives` vs **positive-field energy** `Ē⁺`. The ratio `Ē⁺/Ē⁻` is the
   *direct* measurement of the score-collapse property we want (a non-binder
   should induce a near-zero field). This is the quantity `L_neg` trains; log it
   whether or not `L_neg` is on.
4. **Free-pass leakage probe** `ρ_leak = corr(score, ‖H_free‖)` per epoch — if
   the protein-prior is creeping back into the scalar through the `[H_free; D]`
   re-injection, this rises. Direct guard on open-risk #2 in the concept doc.

These four cost ~4 lines of logging and convert the post-hoc autopsy in §2 into
a live trajectory. **Add them in cycle 0** (before any new concept) so every
later A/B has the same instruments.

---

## 2. Diagnostic decision tree — signature → diagnosis → which §3 concept fires

Read top-down after the v6 baseline lands. The first matching row selects the
cycle-1 concept. (Multiple may match; address in listed order, one per cycle.)

| # | Observable signature in logs/outputs | Diagnosis | Fires |
|---|---|---|---|
| D1 | `Ē → 0` over epochs **and** MRR ≈ chance (≈0.029) | **Sparsity collapse**: field died, score → b0 | **C1** field-energy homeostat |
| D2 | High MRR **but** Jaccard-vs-null > 0.11 / `ρ_leak` rising / `rho_row` > 0 | **Prior leakage**: shortcut re-entered via re-injection | **C2** adversarial prior-orthogonality |
| D3 | `PR` large (≫25, → L) while MRR is mediocre | **Field smear**: protein fakes conditionality by spreading identity over all residues | **C3** field-margin (contrastive Δ) |
| D4 | `pos_above_neg_max` plateaus < 0.9 early; negatives too easy | **Weak negatives**: chosen negatives don't share the anchor's prior, so ranking by identity still works | **C4** isospectral (free-pass) hard negatives |
| D5 | MRR passes (≥0.326) but check C fails (e_res pocket-AUC ≤ 0.5) | **Biophysics, not interface**: field tracks hydrophobicity (v5b inheritance), not the pocket | **C5** annotation-bootstrapped field supervision |
| D6 | Within-protein score rows have tiny dynamic range (scores bunch) | **Rank fragility**: ties make MRR brittle to noise | **C6** rank-spread (within-protein entropy) reg. |
| D7 | MRR passes, all guards green, want robustness | **Validation/over-fit to one ligand chemotype** | **C7** ligand-perturbation field consistency |

A clean baseline (A+B+C all pass, no D-row fires) routes straight to the
**confirmation track**: 3 seeds + matched-capacity head ablation
(deltafield vs bilinear), then turn on hard-neg + `L_sparse`/`L_neg` per the
concept doc. The §3 concepts are for when the baseline reveals a specific
weakness — which, given the open risks, is the likely case.

---

## 3. The novel-concept menu (each gated by a §2 trigger)

Each entry: the idea, why it is new for this task, how it reads the logs, the
loss/architecture change, and the pre-registered check that would *falsify* it.
Run **one concept per cycle**, always A/B against the current best v6 run at
matched capacity, always re-running the full check A/B/C battery so a fix for
one mode cannot silently break another.

### C1 — Field-energy homeostat (closed-loop λ controller)
*Trigger D1.* Instead of annealing `λ_sparse` on a hand-tuned schedule, drive it
with a **controller that holds the measured participation ratio `PR` inside a
target band** (e.g. 6–20 residues): if `PR` drifts above the band, raise
`λ_sparse`; if `Ē` approaches the collapse floor, *lower* it and inject the
energy floor. The schedule is a function of the live diagnostic, not the epoch
counter.
- **Novel because:** sparsity regularisers in DTI/PLI are static or
  cosine-annealed; here the regulariser is a feedback law on a physically
  interpretable state variable (effective pocket size). It makes "pocket
  sparsity" a *controlled* quantity rather than a hoped-for emergent one.
- **Reads:** `PR`, `Ē` from §1.2.
- **Change:** a `FieldEnergyController` in `train.py` updating `λ_sparse`/floor
  each epoch from `PR`; no model change.
- **Falsified if:** controlled runs do not raise MRR vs a best static-λ run, or
  `PR` cannot be held in-band without collapsing `Ē`.

### C2 — Adversarial prior-orthogonality on the difference field
*Trigger D2.* A small **protein-identity probe** is trained to predict the
protein (or its `null_prot_prior` bucket) from the pooled difference field `D̃`;
a **gradient-reversal** layer makes `D̃` *uninformative* about protein identity.
The difference is already supposed to cancel the prior; this enforces it
adversarially on whatever residue survives the re-injection leak.
- **Novel because:** TAPB removes target-prior bias by causal backdoor
  adjustment over a *confounder dictionary*; DeltaField removes it by
  *representation subtraction*. C2 adds a third, complementary mechanism —
  **adversarial invariance applied to the post-subtraction residual** — which,
  composed with the structural zero-field guarantee, is (panel novelty-check
  pending) unseen in DTI ranking.
- **Reads:** `ρ_leak`, Jaccard-vs-null, `rho_row`.
- **Change:** `IdentityProbe` head + GRL on pooled `D̃`; `+ λ_adv` term.
- **Falsified if:** Jaccard/`rho_row` do not drop, or MRR pays > its v4 std
  (0.072) to buy the invariance (the prior carried real ranking signal — would
  reframe the whole shortcut thesis and must be reported).

### C3 — Field-margin: contrastive difference fields (margin in field space)
*Trigger D3.* The within-ligand margin currently lives on the **scalar** score.
C3 lifts it into **field space**: for a protein with true ligand `L⁺` and hard
negative `L⁻`, require the *field* `D̃(L⁺,P)` to be both higher-energy **and
more concentrated** (lower `PR`) at the contact residues than `D̃(L⁻,P)` —
a margin on a field-divergence functional, not on a number. The model can no
longer satisfy the margin by a global offset; it must make the *spatial pattern*
ligand-specific.
- **Novel because:** contrastive learning in DTI operates on pooled embeddings
  or scalar affinities. A margin imposed on the *per-residue perturbation field*
  between two ligands on the same protein — using the difference field as the
  contrastive substrate — is a new objective enabled specifically by this
  architecture.
- **Reads:** `PR` (smear signature), per-residue `e_res` for L⁺ vs L⁻.
- **Change:** `field_margin_loss(D̃⁺, D̃⁻, mask)` in `loss.py`; needs the
  coupled pass for L⁻ (already computed by hard-neg refresh).
- **Falsified if:** field-margin runs do not reduce `PR` *and* improve MRR vs
  scalar-margin only — i.e. spatial specificity was not the bottleneck.

### C4 — Isospectral hard negatives (negatives that share the anchor's prior)
*Trigger D4.* Today's hard negatives are the top-scoring non-binders. C4 selects
negatives whose **free-pass representation `H_free(P⁻)` is closest to the
anchor's `H_free(P⁺)`** — i.e. proteins in the *same shortcut bucket*. These are
the negatives a prior-rider *cannot* separate by identity, so the margin can
only be satisfied by the field. We deliberately mine the confusers the shortcut
would tie.
- **Novel because:** hard-negative mining everywhere ranks by *predicted score*;
  selecting by *similarity in the shortcut's own representation space* (the free
  pass) directly weaponises the architecture's two-pass structure against the
  shortcut. The negative set is chosen in the exact subspace we are trying to
  null out.
- **Reads:** off-diagonal structure of the 200×200 matrix (which proteins the
  model confuses) to validate that isospectral negatives are indeed the
  confusers; `pos_above_neg_max` trajectory.
- **Change:** make `refresh_scores` deltafield-aware (cache `H_free(P)` per
  protein — already required for hard-neg), add a free-pass kNN negative
  sampler in `sampler.py`.
- **Falsified if:** isospectral negatives do not lift MRR / lower Jaccard vs
  score-ranked hard negatives at equal pool size.

### C5 — Annotation-bootstrapped field supervision (semi-supervised pocket)
*Trigger D5.* On the minority of proteins with UniProt binding/active-site
annotations, add a **within-protein ranking loss** that pushes `e_res` up on
annotated pocket residues; then **propagate** that signal to unannotated
homologues via a consistency loss on aligned positions. Turns the sparse,
read-only pocket labels we already use for check C into a *training* signal —
without ever needing holo structures.
- **Novel because:** pocket-localisation supervision in PLI needs 3D
  complexes/contacts (PDBbind). C5 supervises the difference field from
  *sequence-level UniProt feature annotations + homology consistency* — a
  data source no DTI ranker uses for interface localisation — keeping the apo-only
  constraint intact.
- **Reads:** check-C output (`attn_annotation_scan.py` per-protein AUC) to pick
  which proteins are annotated and to measure the lift.
- **Change:** `L_pocket` (margin on `e_res` at annotated sites) + homology
  consistency term; needs the existing UniProt cache + an alignment map.
- **Falsified if:** pocket-AUC rises but **MRR drops** (we taught it biophysics
  at the cost of ranking) — report as a genuine tension, not a win.

### C6 — Rank-spread regulariser (within-protein score entropy)
*Trigger D6.* Penalise low dynamic range in each protein's score row: maximise
the spread/entropy of within-protein scores so true binders are not separated
from non-binders by sub-noise margins. A direct treatment of MRR fragility.
- **Novel because:** a calibration term targeting the *within-protein* score
  distribution (the exact axis MRR ranks on) rather than global calibration —
  small, but unaddressed in the ranking-DTI setting.
- **Reads:** per-row score variance from the 200×200 matrix.
- **Change:** `+ λ_spread · (−Var_within_protein(scores))` (clamped).
- **Falsified if:** Hit@k does not stabilise across seeds (the v4 MLP-vs-bilinear
  variance story is the template — this should narrow the band).

### C7 — Ligand-perturbation field consistency (robustness / falsifiable map)
*Trigger D7.* Using RDKit, delete a functional group from a true ligand; require
the difference field to change **locally near the contact residues** and stay
stable elsewhere (a binding-relevant field is *perturbation-sensitive at the
interface*; a shortcut is insensitive). Doubles as a falsification test of the
interaction-map claim.
- **Novel because:** counterfactual-on-the-ligand consistency for the *interface
  field* (not the scalar) is a new self-supervision signal and a sharp
  interpretability probe unique to the difference-field formulation.
- **Reads:** `e_res`/`C` deltas under ligand perturbation.
- **Change:** RDKit perturbation in the data pipeline + a localised
  consistency loss; off the critical path (robustness cycle, run last).
- **Falsified if:** field changes are delocalised (the map does not track the
  chemistry) — which would itself be a publishable negative result about the
  map's faithfulness.

---

## 4. Closed-loop protocol (how a cycle runs)

1. **Read** the latest run's `train_log.jsonl` + eval outputs through the §1
   instruments; walk the §2 tree; the first firing row names the cycle's
   concept.
2. **Pre-register** the concept's falsification check *before* launching (write
   it into the run tag's notes), so a null result is honest, not reinterpreted.
3. **Implement** the single concept (Fable model for the ML coding, per
   standing preference); add a `configs/abl_<concept>.json` extending the
   current best v6 config — minimal override only.
4. **A/B at matched capacity** vs the current best v6 run, same seed first
   (42), then {7, 1337} only if the single-seed A/B clears the check.
5. **Re-run the full A/B/C battery** + `benchmark_null_eval.py`. A concept is
   kept **only if** it improves its target metric **and** breaks no guard
   (Jaccard ≤ 0.11, zero-field test, `ρ_leak` flat). Log what it cost.
6. **Record** in `PHASE2_LOG.md` + update the relevant memory; one line in the
   §3 table marking the concept accepted/rejected with the number.
7. **Stop rule:** at most one concept per cycle; halt the menu when two
   consecutive cycles fail their pre-registered check (the field-only score has
   reached its ceiling — report it as such, do not keep grafting).

**Ordering rationale:** C1→C2 first (keep the field alive and prior-free — these
protect the core claim), then C3/C4 (sharpen ligand-conditionality — the MRR
levers), then C5 (the interpretability dividend), C6/C7 last (robustness /
falsification). A fix for a downstream mode is meaningless if C1/C2 are red.

---

## 5. What this plan refuses to do

- **No new baseline models.** The menu optimises DeltaField; it does not
  reintroduce GIGN/DeepDTA/etc. (locked after Phase 1).
- **No metric laundering.** Every concept reports its cost; gAUC stays reported,
  never a gate; Gini never a success metric.
- **No holo/PDBbind dependency** on the critical path — C5/C7 use only
  sequence-level UniProt + RDKit, preserving the apo-only constraint.
- **No silent capping.** If a cycle drops proteins, samples, or seeds, it is
  logged. A concept that passes only by narrowing the eval set is rejected.

*Provenance:* extends `docs/DELTAFIELD_CONCEPT.md`; trigger metrics tie to the
project's measured failure modes (`null_prot_prior` Gini ≈ 0.995; v5b pocket
anti-correlation AUC 0.08–0.21; v4 seed variance). Novelty of C1–C7 is claimed
as *composition* and must survive a head-on prior-art check (the DeltaField
panel's standard) before any paper claim.
