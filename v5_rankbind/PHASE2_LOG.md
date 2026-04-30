# Phase 2 — session log (2026-04-27 update)

## Resume state — 2026-04-27, Phase-4 closure

**Phase 2 is fully shipped. Phase-4 Stages (c) and (b) ran 2026-04-27
and are complete. Stage (a) was empirically blocked by the (b)
attention-weight inspection and is deferred indefinitely.** Authoritative
plan addendum: `PLAN.md §14`.

### What changed since 2026-04-23

- **Stage (c) — v4 failure-case diagnosis** (PLAN.md §13.1):
  Per-pair rank breakdown of v4 default's 34 test positive pairs by
  SMARTS class. Polyhydroxy / glycoside substrates over-represented in
  the bottom-quartile (4 of 8 = 50%, n=8 in test). Spearman
  ρ(n_heavy_atoms, rank) = 0.252 (below 0.4). The OTHER bucket was a
  SMARTS-coverage artefact (4 of 6 are aryl esters / amide tautomers).
  **Verdict: ⚠️ Mixed → proceed to (b), defer (a).**
  Memo: `evaluation/attractor_results/v4_failure_diagnosis.md`.
  CSV: `…/v4_failure_diagnosis.csv` (34 rows).
  Plots: `fig_v4_rank_hist_by_class.png`, `fig_v4_atoms_vs_rank.png`,
  `fig_v4_class_failure_rate.png`.

- **Stage (b) — Residue attention-pool** (PLAN.md §13.2):
  New `ResidueAttentionPool` (single-head learned query + LayerNorm) in
  `model.py`, configurable via `model.protein_encoder ∈ {"mean_pool",
  "attn_pool"}`. New config `configs/abl_attn_pool.json`. Per-residue
  tensors threaded through `data.py`, `sampler.py`'s `TripletCollator`
  (`refresh_scores` now encodes proteins chunked through
  `model.encode_protein`), `train.py`, `eval.py`. Smoke-test:
  `v5_rankbind/_smoke_test_attn_pool.py` (all checks pass).
  3-seed sweep submitted on paula (jobs 21392838/839/840), tag `v5b`.
  All three completed 2026-04-27.

  **3-seed numbers (seeds 42, 7, 1337):**

  | Config | MRR (μ ± σ) | H@5 | H@10 | gAUC | Gini-resid |
  |---|---|---|---|---|---|
  | default v4 (mean_pool) | 0.326 ± 0.072 | 0.598 ± 0.090 | 0.755 ± 0.095 | 0.634 ± 0.010 | −0.210 ± 0.022 |
  | **abl_attn_pool v5b** | **0.427 ± 0.123** | **0.686 ± 0.119** | **0.814 ± 0.103** | 0.659 ± 0.028 | −0.216 ± 0.028 |
  | Δ | **+0.101 (+31%)** | +0.088 | +0.059 | +0.025 | ≈ |

  Param-Δ vs v4: **+3,840 (+0.6%)** — a representational change, not a
  capacity change. MRR-arm of §13.2 gate ✅ pass by 2× the threshold.

  **But:** seed-range tells a stability story. attn_pool seeds 42/7/1337
  produce MRR 0.316 / 0.559 / 0.405 — std grew 1.7× (0.072 → 0.123).
  Headline lift is real, but s7 is an outlier-high; s42 is essentially
  on the v4 mean.

- **Stage (b) attention-weight inspection** (the §13.2 interpretability
  arm). Script: `evaluation/attn_weight_inspection.py`. Sample of 60
  proteins, all 3 seeds. Memo:
  `evaluation/attractor_results/attn_weight_inspection.md`.

  | Concentration | Median | Uniform baseline |
  |---|---:|---:|
  | top-10% mass | 0.118 | 0.10 |
  | top-20% mass | 0.228 | 0.20 |
  | entropy / log(L) | 0.999 | 1.00 |

  | Cross-seed agreement | Median | Random expectation |
  |---|---:|---:|
  | Spearman ρ between weights | **0.861** | 0.0 |
  | Top-10% residue Jaccard | **0.500** | ≈ 0.10 |

  **Reading:** weights are essentially uniform in *magnitude*, but their
  *rank-order is highly reproducible across seeds* (ρ 0.86, top-10%
  Jaccard 5× random). The +0.10 MRR lift therefore comes from
  LayerNorm-then-pool, **not** from sharp pocket selection. Three seeds
  converge on a low-magnitude but consistent per-residue preference.

  Plots: `fig_attn_weight_examples.png`,
  `fig_attn_concentration_hist.png`,
  `fig_attn_cross_seed_agreement.png`.

### Stage (a) decision: defer indefinitely

PLAN.md §13.3 specified (a) as: identify top-K=8 residues from
attention, build atom graph on those residues + 4 Å neighbourhood,
2-layer GNN, confidence-gated combination with the residue-level score.

The (b) inspection blocks this mechanism: rank-8 vs rank-50 in attention
mass differ by ~10⁻⁴, and even *consistent* top-10% sets only overlap
50% across seeds. There is no stable top-K to seed an atom graph from.

A redesigned **Option A** (pocket selection from AlphaFold/fpocket
instead of attention) is the only sound path forward. Estimated cost:
~3-4 weeks. Not committed.

Decision: **deferred indefinitely**. Recorded in PLAN.md §14.

### What's still useful to do (none require cluster time)

- **Phase 5 — cross-dataset probe** (the original PLAN.md §10 risk-row
  mitigation): test whether Stage-(b)'s +0.10 MRR lift transfers under
  distribution shift, e.g. RankBind-trained-on-BRENDA evaluated on a
  kcat dataset. Scoping next.
- **Paper-draft start.** Phase-1 + Phase-2 + Stage-(b) is a coherent
  empirical story.

### Key code paths added in 2026-04-27

- `v5_rankbind/model.py::ResidueAttentionPool`
- `v5_rankbind/model.py::RankBind.encode_protein` (dispatches mean/attn)
- `v5_rankbind/data.py::_pad_residues`, residue mode in
  `RankBindDataset` and `collate_pointwise`
- `v5_rankbind/sampler.py::TripletCollator` per-residue collation +
  attn-pool `refresh_scores` chunked encoding
- `v5_rankbind/configs/abl_attn_pool.json`
- `evaluation/v4_failure_diagnosis.py`
- `evaluation/attn_weight_inspection.py`
- `scripts/aggregate_multiseed.py` extended with `abl_attn_pool` (`v5b`
  tag)

---

## Resume state — 2026-04-23 (historical)

**As of 2026-04-23 ~16:00 all cluster-time Phase-2 work is done.** The
remaining work is documentation and a small code patch (figures); no
SLURM jobs need to be submitted to finish Phase 2.

### What is settled, with evidence

- **v4 recipe is the reported number.** 3-seed (42, 7, 1337) mean:
  MRR 0.326 ± 0.072, H@5 0.598 ± 0.090, H@10 0.755 ± 0.095,
  Gini-residual −0.210 ± 0.022, test_gAUC 0.634 ± 0.010.
  Canonical CSV: `evaluation/attractor_results/phase2_rankbind_multiseed.csv`.
  Aggregator: `scripts/aggregate_multiseed.py`.
- **Global AUC ≥ 0.80 is retired as a success gate.** Probe
  `probe_bce_aux` (bce_aux_weight=0.5, seed=42) lifted gAUC by only
  +0.03 (to 0.655) — far below 0.80. Confirms the 0.80 threshold is a
  dataset ceiling under ligand-conditional optimisation, not a
  tunable loss choice. gAUC remains **reported** in every table, just
  not a gate. See the "gAUC trade-off probe — RESULT" section below.
- **Bilinear head is the canonical choice.** Not for a mean-metric win
  (default and abl_no_bilinear means are comparable: 0.326 vs 0.243),
  but for **stability**: 3-seed std is 0.072 for bilinear vs 0.161 for
  MLP — 2.2× wider.
- **Matched-capacity ablations (v4, 3-seed):** margin loss is the
  dominant contribution (removing it: MRR 0.326 → 0.041); balanced
  sampler is a secondary positive (MRR 0.326 → 0.183); bilinear vs MLP
  is a stability argument, not a metric argument.

### What is NOT yet done (none require cluster time)

- **Priority C — PLAN.md addendum.** Append a Phase-2-addendum section
  to `v5_rankbind/PLAN.md` (do NOT edit §1-§9). Must:
  1. Retire `Global AUC ≥ 0.80` as a success gate, cite probe result
     (MRR 0.271 @ gAUC 0.655 at bce_aux=0.5) as evidence that the
     ceiling is structural.
  2. Promote `matrix_mrr` and `matrix_hit_at_10` to primary metrics;
     keep `Gini-residual` and `Top-10 Jaccard vs null_prot_prior` as
     secondary shortcut-avoidance metrics.
  3. Cite `phase2_rankbind_multiseed.csv` as canonical numbers.
- **Priority D — figure regeneration.** Add the v4 default score
  matrix
  (`results/v5_rankbind/20260423-112928_012a2695c2_default_v4/score_matrix_rankbind.npy`)
  to `evaluation/phase_d_figures.py::MATRIX_FILES`. Regenerate
  `evaluation/attractor_results/fig_summary.png` and
  `fig_cross_overlap.png`.
- **Priority E — Phase-2 HTML report.** Pattern after
  `evaluation/attractor_results/phase1_report.html`. Use
  `phase2_rankbind_multiseed.csv` for the error-bar table and the
  narrative from the decision section below.

### First commands for the next session

```bash
module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
source ~/venvs/hieratombind/bin/activate

# Orient (read top-to-bottom of this file for the 2026-04-23 delta).
cat v5_rankbind/PHASE2_LOG.md | head -120

# Verify the reported artifacts.
cat evaluation/attractor_results/phase2_rankbind_multiseed.csv
ls results/v5_rankbind/*_v4/manifest.json results/v5_rankbind/*_v4_s*/manifest.json

# Start Priority C: append a Phase-2 addendum to v5_rankbind/PLAN.md.
# Do NOT edit §1-§9 of PLAN.md — those are the pre-execution reference.
```

No running jobs to wait for. `squeue -u $USER` is empty.

---


## v4 — Hard-negative mining (2026-04-23)

**Status**: code + tests landed; first SLURM run submitted.

### What changed

Priority A from the 2026-04-22 plan. The v3 margin loss saturated on random
easy negatives by epoch ~8 (`pos_above_neg_max` ≈ 0.4%). v4 replaces random
cross-protein sampling with *model-aware* hard negatives.

- `v5_rankbind/sampler.py::TripletCollator`
  - New ctor args: `negative_sampling ∈ {"cross_protein_implicit","hard"}`
    (default in code stays backward-compatible; default JSON config now
    selects `"hard"`) and `hard_pool_size: int` (50).
  - New method `refresh_scores(model, device, lig_chunk=256)`: projects all
    positive-labeled train ligands (1,137) and all train proteins (618)
    once, then computes the full pair-score matrix through the model head
    in ligand chunks. Cached as `self._scores: np.ndarray [N_lig, N_prot]`.
    `model.eval()` during the compute; training mode restored on exit.
  - `__call__` branches on mode. In hard mode, for each anchor:
    mask known-positive proteins + the anchor protein to `-inf`, take
    `argpartition` top-`hard_pool_size` non-positives, sample k uniformly
    from that pool (with replacement only if pool < k).
  - First-epoch fallback: if `_scores is None`, behaves exactly like v3
    (random cross_protein_implicit). This gives the model one epoch of
    noise-ish gradient before hard mining kicks in.
  - Exposes `hard_active: bool` in the collator output for diagnostics.

- `v5_rankbind/loss.py::RankBindLoss.compute_margin`
  - Added `pos_above_neg_max = (pos > neg.max(dim=1).values).float().mean()`
    to the parts dict. This is the saturation diagnostic that motivated
    the whole change — v3's value was 0.4%, v4 is expected to recover into
    the double-digit-percent range.

- `v5_rankbind/train.py`
  - Reads `triplet.negative_sampling` + `triplet.hard_pool_size` from
    config; passes both to `TripletCollator`.
  - Before each margin epoch: `triplet_collator.refresh_scores(model,
    device)` (no-op in non-hard mode). First-epoch refresh is logged with
    wall-time.
  - Aggregates `pos_above_neg_max` and `margin_violation_rate` per-batch
    across the epoch; writes them to `train_log.jsonl` as
    `train_pos_above_neg_max` / `train_margin_violation_rate`. The per-
    epoch stdout line now includes `pos>maxneg=...`.
  - Also logs `train_hard_active_batches` (sanity: should equal the total
    batch count from epoch 2 onward).

- `v5_rankbind/configs/default.json`
  - `triplet.negative_sampling`: `"cross_protein_implicit"` → `"hard"`.
  - Added `triplet.hard_pool_size: 50`.
  - Deep-merge confirmed with `load_config` over all 5 configs: `default`,
    `abl_no_sampler`, `abl_no_bilinear` inherit hard negatives (all three
    are margin configs); `abl_no_margin`, `abl_bce_only` are BCE so the
    collator is never instantiated and the setting is ignored.

- `v5_rankbind/tests/test_sampler.py`
  - Existing 4 tests unchanged and still pass.
  - New: `test_triplet_collator_hard_falls_back_without_scores` — verifies
    `hard_active=False` when `_scores` is None.
  - New: `test_triplet_collator_hard_selects_top_scoring` — injects a
    hand-crafted score matrix and checks the collator picks the top
    eligible protein per anchor.
  - New: `test_triplet_collator_hard_excludes_known_positives` — pumps
    known-positive proteins to huge scores and confirms they are still
    masked out of the negative pool.
  - All 7 sampler tests, loss tests, and model tests pass.

- CPU integration smoke (not committed): `refresh_scores` on real data
  builds a 1137×618 score matrix in 41s on CPU (GPU expected ~2s);
  collator emits `hard_active=True`; at init `pos_above_neg_max = 0.0`
  with margin_violation_rate = 1.0 — the expected "random model loses
  every anchor to its own top confusers" signal.

### v4 default result (SLURM 21302334, paula02, 28 min wall)

Training: 32 epochs, early-stop on `val_global_auc` at epoch 32,
best checkpoint at epoch 22. ~50s/epoch (~12s/epoch of that is the
hard-neg score-cache refresh on GPU — confirmed in the stdout
`[hard-neg] refreshed scores (1137 ligands × 618 proteins, 12.53s)`).

| Metric (test) | v3 default | **v4 default** | Δ vs v3 |
|---|---:|---:|---:|
| matrix MRR ↑          | 0.201 | **0.247** | **+22.9%** |
| matrix Hit@5 ↑        | 0.412 | **0.500** | **+21.4%** |
| matrix Hit@10 ↑       | 0.559 | **0.647** | **+15.8%** |
| matrix Hit@1 ↑        | 0.059 | 0.029     | −51% (n=34, noisy) |
| mean-rank-pct ↓       | 0.123 | **0.100** | −19% (lower = better) |
| test_global_auc       | 0.566 | 0.623     | +10% |
| test_global_aupr      | 0.383 | 0.427     | +12% |
| Gini(attractor) ↓     | 0.871 | **0.787** | −0.084 (lower = more diverse) |
| **Gini-residual** ↓   | −0.124 | **−0.208** | −0.084 (more negative = better shortcut-avoidance) |
| Top-10 Jaccard vs null ↓ | 0.000 | **0.000** | unchanged (trivial-prior overlap = 0) |
| best val epoch        | 3    | 22 | much longer useful training |
| best val_global_auc   | 0.603 | 0.631 | +5% |

v4 wins on all primary metrics and also improves shortcut-avoidance
(Gini-residual). The model now trains for ~20+ useful epochs instead
of plateauing by epoch 3-8.

**Thresholds (from PLAN.md §evaluation):**
- `matrix_mrr ≥ 0.10` → ✅ (0.247)
- `matrix_hit_at_10 ≥ 0.15` → ✅ (0.647)
- `Gini-residual ≤ −0.01` → ✅ (−0.208)
- `Top-10 Jaccard vs null_prot_prior ≤ 0.30` → ✅ (0.000)
- `Global AUC ≥ 0.80` → ❌ (0.623). Gate retired per Priority C
  plan — this is the shortcut metric v4 is trading away.

### pos_above_neg_max trajectory (per epoch, from train_log.jsonl)

Read this bottom-up: v3 post-hoc eval showed 0.4% of anchors had the
positive beating ALL 4 random negatives. v4 logs the live per-epoch
value. Values in v4 reflect hard negatives from epoch 2 onward (epoch 1
uses random — cache not yet built):

- epoch  1 (random): 0.919
- epoch  2 (hard):   0.840   ← harder negatives immediately bite
- epoch  8 (hard):   0.850
- epoch 16 (hard):   0.902
- epoch 22 (hard):   0.941   ← best val
- epoch 32 (hard):   0.966

The dip at epoch 2 and recovery to 0.97 by epoch 32 confirms hard
mining is doing its job: the model keeps finding the current hardest
proteins and learning to separate them. In v3 random-mining the
equivalent fraction was pinned at 0.4% because the max of 4 random
negs kept winning by variance alone; here the model is actually
learning ligand-conditional ranking.

### Artifacts

- Run dir: `results/v5_rankbind/20260423-112928_012a2695c2_default_v4/`
- Score matrix: `.../score_matrix_rankbind.npy`
- `evaluation/attractor_results/phase2_rankbind_summary.csv`: +1 row
  (`default,v4`).
- `results/v5_rankbind/runs_manifest.csv`: regenerated via
  `scripts/collect_v5_runs.py` (11 runs total; v4 row present).

### v4 ablation table (2026-04-23, matched capacity, seed=42)

All three margin configs re-run with hard negatives. BCE configs
inherit no collator, so v3 numbers are canonical for them.

SLURM jobs (both complete):
- `21302697` → `abl_no_sampler_v4` (random batches + margin + biln128 + hard negs), ~11 min wall, early-stop epoch 27 best @17.
- `21302698` → `abl_no_bilinear_v4` (balanced + margin + MLP head + hard negs), ~34 min wall, early-stop epoch 36 best @26.

| Config | tag | head | MRR | H@1 | H@5 | H@10 | mrp | gAUC | Gini-resid | Jac-null |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** (bal+margin+biln) | v3 | biln128 | 0.201 | 0.059 | 0.412 | 0.559 | 0.123 | 0.566 | −0.124 | 0.000 |
| **default** (bal+margin+biln+**hard**) | **v4** | biln128 | **0.247** | 0.029 | **0.500** | **0.647** | **0.100** | 0.623 | **−0.208** | 0.000 |
| abl_no_sampler (rand+margin+biln) | v3 | biln128 | 0.177 | 0.088 | 0.235 | 0.324 | 0.149 | 0.607 | −0.072 | 0.053 |
| abl_no_sampler (rand+margin+biln+**hard**) | **v4** | biln128 | 0.168 | 0.059 | 0.265 | **0.529** | 0.141 | 0.629 | −0.100 | 0.000 |
| abl_no_margin (bal+BCE+biln) — no collator | v3 | biln128 | 0.023 | 0.000 | 0.000 | 0.029 | 0.474 | 0.917 | −0.042 | 0.053 |
| abl_no_bilinear (bal+margin+MLP) | v2 | MLP128 | 0.220 | 0.118 | 0.265 | 0.559 | 0.081 | 0.666 | −0.251 | 0.000 |
| abl_no_bilinear (bal+margin+MLP+**hard**) | **v4** | MLP128 | 0.173 | 0.059 | 0.235 | 0.471 | **0.087** | **0.678** | **−0.276** | 0.000 |
| abl_bce_only (rand+BCE+MLP) — no collator | v2 | MLP128 | 0.013 | 0.000 | 0.000 | 0.000 | 0.493 | 0.921 | −0.002 | 0.429 |

### v4 vs v3 per-ablation — how hard negatives interact with each lever

- **default (bal+margin+biln)**: hard negs lift MRR +23%, H@5 +21%,
  H@10 +16%. Gini-residual improves −0.124 → −0.208. Clear win. This
  is the reported v4 recipe.
- **abl_no_sampler**: MRR about flat (−5%), but H@10 jumps +63% (0.324
  → 0.529) and Gini-residual improves −0.072 → −0.100. Interpretation:
  hard-neg mining sharpens the near-ranking region; the balanced
  sampler is still needed for exact top-1.
- **abl_no_bilinear**: MRR drops 0.220 → 0.173 (−21%), H@5 −11%, H@10
  −16%. But gAUC rises 0.666 → 0.678, Gini-residual deepens −0.251 →
  −0.276, and best val epoch moves 32 → 26 (faster convergence).
  Interpretation: the MLP head over-fits the hard negatives — the
  extra loss signal trades matrix-ranking MRR for a flatter, more
  diverse score distribution. This is a genuine head-vs-head
  difference: bilinear combines well with hard negs, MLP does not.
  Paper framing: *this is the argument for keeping the bilinear
  head*. At v3 matched capacity the two were a wash; under the v4
  loss recipe bilinear is strictly better for primary metrics.

### Overall reading

Hard-neg mining is an unambiguous win for the reported default recipe
(balanced sampler + margin + bilinear head). It compounds well with
the balanced sampler (which handles protein coverage) but does *not*
compound with the MLP head. This is consistent with the Phase-2
narrative: *bilinear is the architecturally appropriate head for
ligand-conditional ranking; loss and sampling choices only shine when
paired with it.*

## Priority B — multi-seed sweep (SUBMITTED 2026-04-23)

Seed 42 is already on disk for every config. This sweep adds seeds 7 and
1337 for all five configs so the paper can report mean ± std.

### Seed-override plumbing (new code)

- `v5_rankbind/train.py` — added `--seed` CLI flag. When set, overrides
  `cfg["seed"]` before `set_deterministic_seeds` and adds a manifest
  note `seed override: cfg=<X> → cli=<Y>` so the actual seed is
  discoverable from the artifact alone. The resolved value is also
  visible at `config_resolved.seed` in `manifest.json`.
- `scripts/run_v5_rankbind.sh` — 4th positional arg `SEED`. When passed,
  `--seed <N>` is threaded to `train.py`, and `_s<N>` is auto-appended
  to the tag (if not already there) so run_ids remain unique across
  seeds. Full arg list now: `CFG_NAME PARTITION TAG SEED`.
- `scripts/run_v5_multiseed.sh` (new) — thin loop: `configs × seeds →
  run_v5_rankbind.sh`. Default matrix: 5 configs × {7, 1337} = 10 jobs.
  Accepts an optional single-config arg. Not used for the first sweep
  (because `default@7` was already smoke-tested separately — see below),
  but kept for future re-runs.

### Submitted jobs (10 total)

All tagged `v4_s<seed>`; seed 42 already exists as v4 (default,
abl_no_sampler, abl_no_bilinear) / v3 (abl_no_margin) / v2
(abl_bce_only). Note: abl_no_margin and abl_bce_only are BCE configs,
so hard-neg mining is a no-op for them — the seed sweep is purely for
error bars.

| JOB ID   | Config          | Seed | Tag    |
|----------|-----------------|-----:|--------|
| 21302962 | default         | 7    | v4_s7  | (smoke-test that verified plumbing)
| 21302985 | default         | 1337 | v4_s1337 |
| 21302986 | abl_no_sampler  | 7    | v4_s7  |
| 21302987 | abl_no_sampler  | 1337 | v4_s1337 |
| 21302988 | abl_no_margin   | 7    | v4_s7  |
| 21302989 | abl_no_margin   | 1337 | v4_s1337 |
| 21302990 | abl_no_bilinear | 7    | v4_s7  |
| 21302991 | abl_no_bilinear | 1337 | v4_s1337 |
| 21302992 | abl_bce_only    | 7    | v4_s7  |
| 21302993 | abl_bce_only    | 1337 | v4_s1337 |

Smoke verification (2026-04-23 13:37): `config_resolved.seed = 7`
recorded in manifest, sampler_audit totals differ from the seed=42
run (2241 vs 2201 positives available — confirming per-sample RNG
reseeded), and the `seed override: cfg=42 → cli=7` note is present.

### Multi-seed results (all 10 jobs finished 2026-04-23)

`scripts/aggregate_multiseed.py` (new) groups runs by `config_name`,
picks the canonical recipe per config (see header of the script), and
emits `evaluation/attractor_results/phase2_rankbind_multiseed.csv`.

**Per-seed raw numbers** (matrix-ranking metrics + test_global_auc + Gini-residual):

| config | seed | MRR | H@5 | H@10 | gAUC | Gini-resid |
|---|---:|---:|---:|---:|---:|---:|
| default          |    7 | 0.346 | 0.618 | 0.794 | 0.636 | −0.233 |
| default          |   42 | 0.247 | 0.500 | 0.647 | 0.623 | −0.208 |
| default          | 1337 | 0.386 | 0.676 | 0.824 | 0.643 | −0.188 |
| abl_no_sampler   |    7 | 0.132 | 0.118 | 0.206 | 0.592 | −0.042 |
| abl_no_sampler   |   42 | 0.168 | 0.265 | 0.529 | 0.629 | −0.100 |
| abl_no_sampler   | 1337 | 0.250 | 0.471 | 0.529 | 0.668 | −0.081 |
| abl_no_bilinear  |    7 | 0.428 | 0.676 | 0.853 | 0.688 | −0.243 |
| abl_no_bilinear  |   42 | 0.173 | 0.235 | 0.471 | 0.678 | −0.276 |
| abl_no_bilinear  | 1337 | 0.130 | 0.176 | 0.235 | 0.615 | −0.026 |
| abl_no_margin    |    7 | 0.033 | 0.029 | 0.118 | 0.971 | −0.029 |
| abl_no_margin    |   42 | 0.023 | 0.000 | 0.029 | 0.917 | −0.042 |
| abl_no_margin    | 1337 | 0.066 | 0.147 | 0.147 | 0.956 | −0.059 |
| abl_bce_only     |    7 | 0.017 | 0.000 | 0.000 | 0.981 | −0.000 |
| abl_bce_only     |   42 | 0.013 | 0.000 | 0.000 | 0.921 | −0.002 |
| abl_bce_only     | 1337 | 0.015 | 0.000 | 0.000 | 0.941 | −0.003 |

**Aggregated (mean ± std over 3 seeds — the paper-ready table):**

| config | head | MRR | H@5 | H@10 | mrp ↓ | gAUC | Gini-resid ↓ | Jac-null ↓ |
|---|---|---|---|---|---|---|---|---|
| **default** (bal+margin+biln+hard) | biln128 | **0.326 ± 0.072** | **0.598 ± 0.090** | **0.755 ± 0.095** | **0.071 ± 0.031** | 0.634 ± 0.010 | **−0.210 ± 0.022** | 0.000 ± 0.000 |
| abl_no_sampler (rand+margin+biln+hard) | biln128 | 0.183 ± 0.060 | 0.284 ± 0.177 | 0.422 ± 0.187 | 0.119 ± 0.023 | 0.630 ± 0.038 | −0.074 ± 0.030 | 0.018 ± 0.030 |
| abl_no_bilinear (bal+margin+MLP+hard) | MLP128 | 0.243 ± 0.161 | 0.363 ± 0.273 | 0.520 ± 0.312 | 0.093 ± 0.059 | 0.660 ± 0.040 | −0.182 ± 0.136 | 0.000 ± 0.000 |
| abl_no_margin (bal+BCE+biln) | biln128 | 0.041 ± 0.023 | 0.059 ± 0.078 | 0.098 ± 0.061 | 0.398 ± 0.080 | 0.948 ± 0.028 | −0.043 ± 0.015 | 0.035 ± 0.030 |
| abl_bce_only (rand+BCE+MLP, Ph1 eq) | MLP128 | 0.015 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.495 ± 0.008 | 0.948 ± 0.030 | −0.002 ± 0.001 | 0.352 ± 0.056 |

CSV: `evaluation/attractor_results/phase2_rankbind_multiseed.csv`.

### What the error bars change vs the single-seed story

1. **Seed=42 was below-average for default.** Single-seed reported MRR
   0.247; 3-seed mean is **0.326 ± 0.072**. Seeds 7 and 1337 both
   produced cleaner matrix-ranking than seed 42 (0.346 and 0.386
   respectively). The paper headline should use the 3-seed mean.
2. **The bilinear-vs-MLP story flips from "MLP is a bit worse" to
   "MLP is high-variance".** Means are close (0.326 vs 0.243), but
   MLP's std (0.161) is **2.2× wider** than bilinear's (0.072). One
   of MLP's seeds (seed=7) got MRR 0.428 — the single best number in
   the whole sweep — but the other two dropped to 0.130 and 0.173.
   Bilinear produces 0.247–0.386 across seeds; MLP produces 0.130–
   0.428. This is now the argument for keeping the bilinear head:
   *comparable mean, strictly tighter distribution*. That is what a
   production inductive bias should buy you.
3. **Balanced sampler is worth it with real error bars.**
   abl_no_sampler mean 0.183 ± 0.060 vs default 0.326 ± 0.072 — a
   ~2σ gap, so the sampler's contribution is robust to seed noise.
4. **Margin loss is essential.** BCE variants sit at MRR 0.015–0.041
   with tight std; they cannot do ligand-conditional ranking
   regardless of seed.
5. **BCE configs overlap on gAUC 0.948.** Both `abl_no_margin` and
   `abl_bce_only` pass the Phase-1 "global AUC ≥ 0.80" gate while
   flunking every matrix-ranking metric — which is precisely why we
   retired that gate.

## gAUC trade-off probe — RESULT (2026-04-23 ~15:51)

**Context.** User pushed back on the "Global AUC ≥ 0.80 retired" framing —
reasonable intuition: "AUC says how good predictions are, I still want
reliable predictions". Phase 1 had already shown this intuition fails on
this dataset (null_prot_prior baseline hits gAUC ≈ 0.95 without ever
looking at the ligand), but we ran an empirical probe to back the framing
with new data rather than old arguments.

**What was run.** Single SLURM job 21303614 on paula. Config:
`v5_rankbind/configs/probe_bce_aux.json` — extends `default.json`, only
change is `loss.bce_aux_weight = 0.5`. Everything else identical to the
reported v4 default (balanced sampler + hard negs + bilinear rank=128,
seed=42, 50-epoch cap / patience 10). Wall-clock ~35 min (early-stopped
epoch 37, best val_gauc 0.6368 at epoch 27).

Run dir:
`results/v5_rankbind/20260423-151536_9ee7fdbfbc_probe_bce_aux_v4_bceaux05/`.
CSV row appended to `evaluation/attractor_results/phase2_rankbind_summary.csv`
with tag `v4_bceaux05`.

**Result vs default_v4 seed=42 baseline:**

| Metric | default_v4 seed=42 | **probe (bce_aux=0.5)** | Δ |
|---|---:|---:|---:|
| test_gAUC        | 0.623  | **0.655**  | **+0.032** |
| test_gAUPR       | 0.427  | **0.514**  | **+0.087** |
| matrix MRR       | 0.247  | **0.271**  | +0.024 |
| matrix H@5       | 0.500  | **0.529**  | +0.029 |
| matrix H@10      | 0.647  | **0.706**  | +0.059 |
| matrix mrp ↓     | 0.100  | 0.115      | +0.015 (slightly worse) |
| Gini-residual ↓  | −0.208 | **−0.241** | −0.033 (better shortcut-avoidance) |
| Jac-null ↓       | 0.000  | 0.000      | = |

All probe numbers sit **inside ±1σ of the 3-seed default mean**
(MRR 0.326 ± 0.072, H@10 0.755 ± 0.095, gAUC 0.634 ± 0.010), so
treating any of these deltas as "better than default" at single-seed
resolution would be over-reading the noise. The only statistically
robust readings are:
- **gAUC is still far below 0.80.** Adding a BCE auxiliary at weight
  0.5 lifts gAUC by ~0.03, nowhere near the DTI-literature standard.
- **Matrix-ranking metrics are not collapsed.** The BCE aux did not
  destroy the margin-loss contribution.

**Decision rule outcome: (c) — gAUC < 0.80 at weight 0.5.**

### Decision (2026-04-23): keep "Global AUC retired as success gate"

The probe confirms the Phase-1 conclusion with fresh data: *the 0.80
gAUC threshold is a **dataset ceiling** under ligand-conditional
optimisation, not a loss-function artefact we can tune away*. A model
that hits gAUC ≥ 0.80 on this dataset is gaming the per-protein positive
rate (as `null_prot_prior` demonstrates at 0.95). `abl_bce_only` does
exactly that and flunks every matrix-ranking metric.

Paper framing therefore stays as currently written across `PLAN.md`,
`CLAUDE.md`, and this log:

> Primary: `matrix_mrr`, `matrix_hit_at_10`. Secondary: Gini-residual,
> Top-10 Jaccard vs null_prot_prior. Reported for reference but not a
> success gate: `test_global_auc`, `test_global_aupr`.

What changes in the wording: *retired as a success gate* is the correct
phrase — not *demoted* or *deprecated*. gAUC is still reported in every
table; it just isn't the metric by which v4 succeeds or fails. The
PLAN.md addendum (Priority C) should cite the probe number (0.655) as
evidence that the ceiling is structural, not an oversight.

**What was NOT done** (deliberately, to keep the reported numbers
stable):
- No weight sweep (1.0 / 2.0). Can be added if a reviewer demands a
  continuous trade-off curve, but the single probe at 0.5 already
  refutes the "just retune the loss" counter-argument.
- No multi-seed sweep for `probe_bce_aux`. Single-seed is enough to
  locate the 0.80 threshold; multi-seed would not change that
  conclusion.
- No change to `default.json`. v4 recipe remains the reported number.


### Pending after Priority B

Priorities A (hard-negative mining), A1 (margin-ablation re-runs with
hard negs), and B (multi-seed) are all complete. Remaining:

- **Priority C — PLAN.md addendum.** Retire `Global AUC ≥ 0.80`,
  promote `matrix_mrr` / `hit_at_10` to primary officially, and cite
  the multi-seed CSV as the canonical numbers. Already reflected in
  practice; PLAN.md just needs the appendix section. No cluster time
  needed.
- **Priority D — figure regeneration.** Add the v4 default score
  matrix at `results/v5_rankbind/20260423-112928_012a2695c2_default_v4/score_matrix_rankbind.npy`
  to `evaluation/phase_d_figures.py::MATRIX_FILES`, regenerate
  `evaluation/attractor_results/fig_summary.png` and
  `fig_cross_overlap.png`. Small patch.
- **Priority E — Phase-2 HTML report.** Phase-2 counterpart of
  `evaluation/attractor_results/phase1_report.html`: glossary,
  ablation figures (MRR bars with error bars by config, Gini-residual
  scatter, top-K Jaccard heatmap). Priority B unlocked this — numbers
  are final.

---

# Phase 2 — session log (2026-04-22)

Companion to `v5_rankbind/PLAN.md` (the original plan — kept untouched as
reference) and `evaluation/attractor_results/phase1_report.html` (Phase 1
narrative). When these conflict, this file is the authoritative delta for
Phase 2 status.

## What got built

`v5_rankbind/` package, stand-alone, reuses `BRENDADataConfig` (seed=42,
protein-based split). Full provenance on every run: source-sha256 tree hash,
input sha256, env capture, metrics, output sha256 written to
`results/v5_rankbind/<run_id>/manifest.json`.

Modules:
- `run_manifest.py`         — manifest start/finish, JSON config loader with
                              `"extends": "parent.json"` single-level
                              inheritance, deterministic seed helper
- `data.py`                 — ChemBERTa cache, ESM2 mean-pool loader,
                              RankBindDataset, protein-split pipeline
- `sampler.py`              — `ProteinBalancedSampler` (per-protein pos/neg
                              balancing) + `TripletCollator` (see Key Fix #1)
- `model.py`                — `LigandProjector` + `ProteinProjector` (frozen
                              embeddings → 256-d), `BilinearHead` (low-rank
                              +diag, configurable `bilinear_rank`),
                              `MLPConcatHead` (ablation)
- `loss.py`                 — `margin_loss(pos[B], neg[B,k], m=1.0)`, BCE,
                              diagnostic parts dict
- `metrics.py`              — `per_ligand_auc`, `hit_at_k`, `global_metrics`,
                              `matrix_ranking_metrics` (see Key Finding #1)
- `train.py` / `eval.py`    — train with cosine LR, bf16, configurable
                              early stop; eval builds 200×200 score matrix
                              over same pool as `evaluation/null_baselines.py`

Configs (all `extends: default.json`):
- `default.json`            — balanced sampler + margin loss + bilinear head
- `abl_no_sampler.json`     — random batch sampler, everything else default
- `abl_no_margin.json`      — pointwise BCE, everything else default
- `abl_no_bilinear.json`    — MLP-concat head, everything else default
- `abl_bce_only.json`       — Phase-1-baseline equivalent (rand + BCE + MLP)

Scripts:
- `scripts/run_v5_rankbind.sh <cfg> <partition> <tag>`  — single SLURM job,
                              captures `run_dir` from train.py stdout (race-
                              safe for concurrent jobs), auto-runs eval
- `scripts/run_v5_ablations.sh`                         — submits all 5
- `scripts/collect_v5_runs.py`                          — walks every
                              `manifest.json`, emits `runs_manifest.csv`

## Key findings from today

### 1. Phase-1 `per_ligand_auc` is an n=4 noisy metric
Only **4 of 1404** test pairs have both classes present per ligand. The
Phase-1 baseline numbers (per-lig AUC 0.25–0.625) are n=4 estimates with
enormous variance. Every v5_rankbind run reports **matrix ranking metrics**
on the full 200×200 score matrix as the primary signal — same geometry as
`evaluation/null_baselines.py`, so direct comparison is meaningful.

Matrix-level metrics are computed via `matrix_ranking_metrics(M, ligs, prots,
positive_pairs)` in `v5_rankbind/metrics.py`: for every observed
`(lig, positive_protein)` pair, compute the rank of the positive protein
among all 200 proteins for that ligand, aggregate to MRR / Hit@K / mean-rank-
percentile.

### 2. BRENDA + decoys dataset pathology (and the fix)
The original `TripletCollator` in `sampler.py` sampled negatives as
`(same_smiles, different_protein, label=0)`. But BRENDA positives are real
substrates while decoys have fundamentally different SMILES — only **16 of
1137** positive SMILES have any matching label-0 pair. Keep-ratio: **3.7%**.
The model trained on ~4% of the data and val AUC never moved past ~0.56.

**Fix**: switched `TripletCollator` to `cross_protein_implicit` negatives —
for anchor `(L+, P+)`, sample `k` random proteins not in L+'s positive set.
Matches evaluation geometry (ligand-conditional ranking over 200 proteins);
false-negative risk ~0.3% given BRENDA density. Keep-ratio jumped to 68%;
MRR went from 0.033 to 0.114 immediately.

### 3. SLURM race condition in eval step
Original `scripts/run_v5_rankbind.sh` located the run dir for the post-train
eval via `ls -td … | head -1`. With concurrent ablation jobs this races;
three of our v2 ablation eval steps all wrote into abl_no_bilinear's dir
(the most recently touched one at their eval time), clobbering each other.
**Fix**: `tee` train.py stdout and parse `[manifest] run_dir = …` for the
correct path. Already re-eval'd all four v2 ablations on their own
checkpoints — data integrity restored.

### 4. Bilinear head was underparameterized
Default rank=32 bilinear had 16,641 head params; MLP-concat ablation had
65,793. At rank=128 (`bilinear_rank: 128` in `default.json` now) the bilinear
head has **exactly** 65,793 params — strictly matched. After this fix the
two heads are a wash: bilinear wins H@5 (0.412 vs 0.265), ties H@10 (0.559),
trails MRR (0.201 vs 0.220). Paper narrative keeps bilinear as canonical
head (interpretability + clean inductive bias).

### 5. Margin loss saturates on easy negatives
v2 default score analysis: pos mean 3.41, neg mean 1.78, std ≈ 2, but only
**0.4%** of positives beat max(neg). Train loss collapses 0.24 → 0.02 over
~8 epochs while val plateaus — classic easy-negative saturation. Hard-
negative mining is the most likely lever for further lift. Not yet
implemented.

## Current matched-capacity ablation table (v3; total_params = 627,201)

All runs: seed=42, 50 epochs max (early-stop patience 10, min-epochs 20),
bf16, cosine LR, evaluated on same 200×200 pool as `null_baselines.py`.

| Config | Head | MRR | H@1 | H@5 | H@10 | mrp | gAUC | Gini | Gini-resid | Jac-vs-null |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **default** (bal+margin+biln) | biln128 | 0.201 | 0.059 | **0.412** | 0.559 | 0.123 | 0.566 | 0.871 | **−0.124** | **0.000** |
| abl_no_sampler (rand+margin+biln) | biln128 | 0.177 | 0.088 | 0.235 | 0.324 | 0.149 | 0.607 | 0.923 | −0.072 | 0.053 |
| abl_no_margin (bal+BCE+biln) | biln128 | 0.023 | 0.000 | 0.000 | 0.029 | 0.474 | 0.917 | 0.953 | −0.042 | 0.053 |
| abl_no_bilinear (bal+margin+MLP) | MLP128 | **0.220** | 0.118 | 0.265 | 0.559 | **0.081** | 0.666 | **0.744** | **−0.251** | 0.000 |
| abl_bce_only (rand+BCE+MLP, Ph1 eq) | MLP128 | 0.013 | 0.000 | 0.000 | 0.000 | 0.493 | 0.921 | 0.993 | −0.002 | 0.429 |

Source: `evaluation/attractor_results/phase2_rankbind_summary.csv` and
`results/v5_rankbind/runs_manifest.csv`.

**Thresholds from PLAN.md §evaluation:**
- `matrix_mrr ≥ 0.10` → ✅ 3 of 3 margin configs pass
- `matrix_hit_at_10 ≥ 0.15` → ✅ 3 of 3 margin configs pass
- `Gini-residual ≤ −0.01` → ✅ all five pass (even abl_bce_only at −0.002)
- `Top-10 Jaccard vs null_prot_prior ≤ 0.30` → ✅ 4 of 5 pass (only Phase-1
  equivalent abl_bce_only fails at 0.429, as expected)
- `Global AUC ≥ 0.80` → ❌ margin configs 0.57–0.67. **This threshold was
  miscalibrated** — Global AUC is the shortcut metric; trading it for
  ligand-conditional ranking is the whole point of Phase 2. Retire.

### Paper reading of the table

- **Margin loss is the dominant Phase-2 contribution.** Remove it → MRR
  drops 9–17× (0.20 → 0.02), H@10 drops ~15× (0.56 → 0.03).
- **Balanced sampler is a secondary positive.** default vs abl_no_sampler:
  +24% MRR, +75% H@5, +72% H@10, and tightens shortcut-avoidance (Gini 0.87
  vs 0.92, Jac-null 0.00 vs 0.05). Consistent but not dramatic.
- **Bilinear head at matched capacity is competitive with MLP, not
  dominant.** Paper framing: bilinear kept for inductive bias and
  interpretability, not for a raw-metric win.
- **BCE configs reproduce Phase-1 pathology regardless of architecture.**
  `abl_bce_only` (Phase-1 equivalent) hits gAUC 0.92 with Jac-null 0.43 and
  matrix MRR 0.013 — i.e., learns the protein-level shortcut perfectly.

## Pending for next session

Ordered by priority. Start here.

### A. Hard-negative mining — IMPLEMENTED 2026-04-23, awaiting v4 results
See the 2026-04-23 section at the top of this file for the implementation
summary. Status as of writing: `refresh_scores` + top-pool sampling wired
into `TripletCollator`, `pos_above_neg_max` diagnostic logged every epoch,
all 5 configs inherit hard negs via deep-merge (BCE configs ignore it).
v4 SLURM run submitted; when it returns, update this section with the v4
vs v3 delta table.

Original planning notes (kept for reference):
> v2/v3 default margin loss saturates on easy negatives by epoch ~8.
> Implement in `sampler.py::TripletCollator`: for each anchor, take the
> top-k proteins the current model scores highest among
> `all_proteins − positive_proteins[L]`. … Tag runs `v4`.
> Sanity check after implementation: `pos_above_neg_max` should rise from
> 0.4% into double-digit percent.

### B. Multi-seed runs
Three seeds × (default, abl_no_sampler, abl_no_margin, abl_no_bilinear) = 12
jobs × ~10 min each ≈ 2 GPU-hours. Adds error bars to the ablation table.
Submit after A lands so we only re-run once. Seeds: 42 (current), 7, 1337.

### C. PLAN.md revision
Append a "Phase-2 addendum" section (do **not** edit §1-§9 — those are the
pre-execution reference):
- Retire `Global AUC ≥ 0.80`
- Note bilinear/MLP head parity at matched capacity
- Explicitly promote matrix MRR and H@10 to primary, per_ligand_auc → supp.

### D. Figure regeneration
Extend `evaluation/phase_d_figures.py::MATRIX_FILES` with the v3 default
score matrix (and abl_no_bilinear if we're comparing heads). Currently the
file references `results/v5_rankbind/current/score_matrix_rankbind.npy` — we
can add a `current` symlink pointing to the chosen v3 run, or hardcode the
path for paper figures.

### E. Report (only after A + B)
Phase-2 HTML report in the style of `phase1_report.html`: glossary,
ablation figures (MRR bars by config, Gini-residual scatter, top-K Jaccard
heatmap including RankBind). Don't write this until numbers are final.

## Artifacts on disk

- `results/v5_rankbind/*/manifest.json`                  — per-run provenance
- `results/v5_rankbind/runs_manifest.csv`                — flat table, paper-ready
- `evaluation/attractor_results/phase2_rankbind_summary.csv` — ablation +
  Gini + null-Jaccard in one CSV
- v2 tag: first round of ablations (bilinear rank=32, 98%-dropped triplets
  pre-fix for default v1, correct triplets for v2). Keep on disk — they
  document the debugging trajectory but are **not** the reported numbers.
- v3 tag: matched-capacity re-runs (bilinear rank=128). These are the
  reported numbers.

## Do / don't (Phase-2 specific, on top of Phase-1 rules in CLAUDE.md)

**Do**
- Tag every SLURM run (`bash scripts/run_v5_rankbind.sh <cfg> paula <tag>`).
  Next tag: `v4` (hard-negative mining).
- Extend configs via `"extends": "default.json"` + minimal overrides.
- After any training change, spot-check `train_keep_ratio_mean` and
  `n_batches_skipped` in the first epoch log — zero skipped / high keep
  ratio is the signal that the collator is healthy.

**Don't**
- Don't edit `v5_rankbind/PLAN.md` §1-§9. Append an addendum section.
- Don't use `ls -td` to find run_dirs in SLURM scripts — race condition.
  The fixed script parses run_dir from train.py stdout.
- Don't early-stop on `val_per_lig_auc` — it's n=2 on the val split, pure
  noise. Default is `val_global_auc`; switching to a proper ligand-
  conditional val metric is a TODO but not urgent.

## Commands to resume

```bash
module load GCC/11.3.0 CUDA/12.4.0 Python/3.10.4-GCCcore-11.3.0
source ~/venvs/hieratombind/bin/activate

# Sanity
ls results/v5_rankbind/*_v3/manifest.json
cat evaluation/attractor_results/phase2_rankbind_summary.csv

# Priority A: implement hard-negative mining in v5_rankbind/sampler.py::TripletCollator
# Then:
bash scripts/run_v5_rankbind.sh default paula v4
```
