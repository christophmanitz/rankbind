# Attention-Weight Inspection, Stage-(b) interpretability diagnosis

**Date:** 2026-04-27
**Source:** `evaluation/attn_weight_inspection.py` over the 3 attn_pool runs
(`v5b_s{42, 7, 1337}`), 60 sampled proteins each.
**Decision rule:** PLAN.md §13.2, the interpretability arm of the b-to-a gate.

## 1. The numbers

Per-seed concentration (median over 60 proteins):

| Seed | top-5% mass | top-10% mass | top-20% mass | entropy / log(L) |
|---|---:|---:|---:|---:|
| 42   | 0.061 | 0.118 | 0.228 | 0.999 |
| 7    | 0.058 | 0.114 | 0.221 | 1.000 |
| 1337 | 0.063 | 0.122 | 0.235 | 0.999 |
| *uniform baseline* | 0.05 | 0.10 | 0.20 | 1.00 |

Cross-seed agreement (median over 60 proteins × 3 seed-pairs):

| Metric | Value | Random expectation |
|---|---:|---:|
| Spearman ρ between weights | 0.861 | 0.0 |
| Top-10% residue Jaccard | 0.500 | ≈ 0.10 |

## 2. What this means

The two findings together tell a non-trivial story:

- The attention weights are essentially uniform in magnitude. Entropy sits at the mathematical ceiling `log(L)`, top-10% of residues hold barely 11.8% of mass, only 1.18× the 10% they would hold under perfect uniform pooling. Visually (`fig_attn_weight_examples.png`), the weight curves are jagged but their dynamic range is tiny, typical span ~0.0015 to 0.0040 on proteins of length ~400.
- But the rank-order of weights is highly reproducible across seeds. Three independent training runs converge on the same residue ranking (ρ 0.86) and 50% of their top-10% residue sets overlap (random would be ~10%). The seeds are not finding "any plausible pocket", they are finding the same plausible pocket.

So the model has learned a consistent but very subtle preference over residues. The structural information is there, but it lives in the relative ordering, not in the magnitude.

## 3. Why does Stage (b) lift MRR by +0.10 then?

If the attention-pool is functionally near-uniform, the +0.10 MRR lift over v4 mean_pool cannot come from sharp pocket selection. The most plausible explanation is the LayerNorm-then-pool architecture: in `ResidueAttentionPool.forward`, residues are LayerNorm-ed before the softmax-weighted average. Even with uniform weights, LayerNorm changes the per-residue feature magnitudes, normalising across residues per protein. v4 mean_pool simply averages raw ESM2 vectors, with no such re-scaling.

This is testable but not yet tested: an "abl_layernorm_only" config that mean-pools after LayerNorm (without learned attention) would isolate the LayerNorm contribution from the (tiny) attention contribution. We do not need that test to make the Stage-(a) decision below; recording it as a follow-up.

## 4. Implications for Stage (a) (atom-level gating)

Stage (a) as specified in PLAN.md §13.3 picks top-K=8 residues from the attention map and builds an atom graph on those residues plus their 4 Å neighbourhood. The current weights cannot support this:

- The difference between rank-8 and rank-50 in attention mass is on the order of 0.0001, well below noise. A "top-8" set is essentially a random sample weighted by sub-percent mass differences.
- Cross-seed Jaccard 0.50 at top-10% means even the consistent top residues only agree half the time across seeds. Picking the same atom graph reproducibly is at best 50/50.

Stage (a) as written is therefore on shaky foundation. Two options:

- **Option A:** redesign (a) to not use attention for pocket selection. Use structural priors instead: for each protein, derive the binding pocket from an external tool (e.g. fpocket on the AlphaFold PDB), or use a fixed physical heuristic (residues within X Å of a known catalytic motif, when EC class is known). Stage (b)'s attention then plays no selection role; it provides a soft over-residue pooling for the residue-level branch only.
- **Option B:** skip (a), invest the time elsewhere. The +0.10 MRR lift is already publishable. The Phase-1, Phase-2 and Stage-(b) story is coherent: shortcut, then margin loss + sampler, then residue-level attention adds normalisation. Stage 5 (cross-dataset probe) is the remaining empirical claim that strengthens the paper. Atom-level may not return any incremental signal that the residue-level path has not already captured.

## 5. Recommendation

Skip (a) for now.

Three reasons:

1. The mechanism (a) needs is not present. Top-K residue selection from attention does not produce a stable graph; the prerequisite has empirically failed even though the outcome (MRR lift) succeeded.

2. The Stage-(c) chemistry signal was already weak. Polyhydroxy / carbohydrate substrates (n=8 in test) showed elevated failure rates, but n was insufficient to commit to 2-4 weeks of atom pipeline work. Now Stage (b) adds the second concern: even if we built the atom pipeline, we don't know which atoms to feed it.

3. The (b) finding is itself a paper-worthy result. "RankBind's residue-level encoder converges across seeds on a low-magnitude but reproducible per-residue preference" is an interpretability claim worth examining, separate from any atom-level extension. The cross-seed Spearman 0.86 deserves a section-level discussion in the paper: what residues do the weights agree on? Is there pocket overlap with AlphaFold confidence regions? With known catalytic-residue databases (M-CSA, UniProt active-site annotations)?

What I would do instead of (a):

- Phase-5-prep: cross-dataset probe (BRENDA to kcat or similar) to test whether the Stage-(b) lift transfers under distribution shift.
- Paper-writing: the empirical story is now complete enough for a submission target. The Phase-2 + Stage-(b) figures are paper-ready.

If the user wants Stage (a) regardless of the diagnostics, Option A above is the only sound path forward; the originally planned Option-A1 (top-K from attention) is empirically blocked.

## 6. §13.2 b-to-a gate verdict

Reading the gate text:

> Pass: matrix MRR mean-lift ≥+0.05 absolute over v4 default (0.326 to ≥0.376), OR attention-weights concentrate on <20% of residues with qualitative pocket overlap on 2-3 spot-checked proteins.
> Fail: no MRR lift AND flat weights.
> Mixed: small lift but flat weights. Still proceed to (a), top-K residue selection can use raw activations rather than learned weights.

By the letter of the gate: pass, the MRR-arm passed by +0.10, which is the OR-clause's first arm.

By the spirit of the gate: the mixed clause's reasoning about "raw activations" is what we'd actually need to invoke. Without learned attention as the selector, Stage (a) needs a redesign, not a continuation. Recording this nuance: technically pass, but the gate under-specified the case where the MRR arm passes but the interpretability arm reveals near-uniform weights.

## 7. Files

- Script: `evaluation/attn_weight_inspection.py`
- CSVs:
  - `evaluation/attractor_results/attn_weights_concentration.csv` (180 rows)
  - `evaluation/attractor_results/attn_weights_cross_seed.csv` (60 rows)
- Figures:
  - `evaluation/attractor_results/fig_attn_weight_examples.png`
  - `evaluation/attractor_results/fig_attn_concentration_hist.png`
  - `evaluation/attractor_results/fig_attn_cross_seed_agreement.png`
- This memo: `evaluation/attractor_results/attn_weight_inspection.md`
