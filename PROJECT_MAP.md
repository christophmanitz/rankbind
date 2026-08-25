# PROJECT MAP, start here

*If you lose the thread, read this top to bottom. Plain language,
jargon explained inline.*

Last updated: 2026-06-04.

---

## 1. What this project is about, in one paragraph

Computers can be trained to predict whether a small molecule (a "ligand")
interacts with a protein. The usual way to judge such a model is a score
called **pooled AUC**. The core finding of this project is that on
enzyme-substrate data, models can get a great pooled AUC by cheating:
instead of learning which molecule fits which protein, they memorise
"this protein tends to react with lots of things" and guess from the
protein alone. We call this the **protein shortcut**. We built RankBind,
a model plus a way to measure the cheating, that forces the model to
actually rank the right molecule for each protein. That is the whole
story.

**One-line thesis:** *DTI models pass pooled AUC by learning a
protein-level shortcut; RankBind is an architecture that enforces
ligand-conditional ranking, measured by ranking metrics instead of
pooled AUC.*

---

## 2. The storyline: how the pieces connect

Read this as a chain; every later piece exists because of the earlier one.

1. The suspicion: maybe DTI models on BRENDA look good but do not really
   understand molecule-protein fit.
2. The diagnosis (Phase 1): train 4 published models and show they all
   get high pooled AUC but rank molecules at chance level within a
   protein. Prove the shortcut with a null baseline, a dumb model that
   only knows each protein's average reactivity; it matches the real
   models. *Pooled AUC is the wrong yardstick; use ranking (matrix MRR /
   Hit@K) instead.*
3. The fix (Phase 2, RankBind): build a model with 4 ingredients that
   together break the shortcut. Ablations show all 4 are needed.
4. The refinement (Phase 4): a smarter way to read the protein,
   attention over residues, adds another lift and gives the strongest
   model.
5. The scale test (§8.1, "enzyme-wide"): re-run on much bigger
   BRENDA+SABIO data (all enzyme classes, not just 200). With one
   hyperparameter scaled, the recipe transfers. Currently being
   finalised across 3 random seeds.
6. The generalisation test (§8.3, external benchmarks): test on
   independent datasets outside BRENDA: ESP (same task) plus
   Davis/KIBA/BindingDB (a related but different task). Currently being
   prepared and run.

Phases 3 and 5 in the old plan were renumbered; the **paper**
(`paper/scirep/main.tex`) is the authoritative narrative now, not the old phase
numbers.

---

## 3. The models: what exists and how they relate

"RankBind" is one codebase (`v5_rankbind/`). The labels v3 / v4 / v5b
are not different programs; they are tagged configurations of that one
codebase, each adding something to the last.

| Label | What it is | Key result (BRENDA-200) | Role |
|---|---|---|---|
| **4 baselines** (GraphDTA, MolTrans, DrugBAN, GEMS) | published DTI models, retrained on our split | pooled AUC 0.63-0.95 but per-ligand AUC ≤ chance | the "they cheat" evidence (Phase 1) |
| **null baselines** (`null_prot_prior` etc.) | dumb probes, not real models | match the baselines' attractor pattern | the *tool* that proves the shortcut |
| **RankBind v3** | matched-capacity base (fair ablation) | MRR 0.201 | first valid RankBind |
| **RankBind v4** | v3 + online hard-negative mining | **MRR 0.326 ± 0.072** | **paper headline model** |
| **RankBind v5b** | v4 + residue attention-pool | **MRR 0.427** | **best model (§7 extension)** |

"mean-pool" vs "attn-pool" is how the protein is summarised. v4 uses
mean-pool; v5b uses the smarter attention-pool. Everything else is
shared.

---

## 4. The datasets: three tiers, three different questions

| Tier | Datasets | Task | Question it answers | Where |
|---|---|---|---|---|
| **1. Headline** | BRENDA-200 (200 prot × 200 lig, hydrolases) | enzyme-substrate | does RankBind beat the shortcut? | `data/` |
| **2. Enzyme-wide** | BRENDA+SABIO: `kcat_km`, `km`, `turnover` (50× scale, all EC classes) | *same* task, big | does the recipe scale up? (§8.1) | `reactionDataFiltering/data/interim/<name>/` |
| **3. External** | **ESP** (same task) + **Davis/KIBA/BindingDB** (kinase *affinity*, different task) | mixed | does it work outside BRENDA? (§8.3) | `/work2/zw93onug-rankbind_bench/benchmarks/` (symlinked into `reactionDataFiltering/data/interim/benchmarks/`) |

"Enzyme-wide" just means tier 2: the *full* enzyme universe instead of
the curated 200-protein slice in tier 1.

---

## 5. The folder map: what every top-level directory is

| Folder | What's in it | Keep / status |
|---|---|---|
| `v5_rankbind/` | **THE model code** + `configs/` (declarative experiment configs) | core |
| `data/` | BRENDA-200 source data (tier-1 dataset) | core |
| `reactionDataFiltering/` | git **submodule**: data pipeline + tier-2/3 data storage | core |
| `evaluation/` | all diagnostic & eval scripts; `attractor_results/` = **main result files**; `suitability_results/` = dataset characterisation; `_archive/` = retired scripts | core |
| `baselines/` | adapters for the 4 comparison models (`adapters/` + `drugban/gems/graphdta/moltrans/`) | keep the 4 |
| `results/` | trained outputs: `original_*` = 4 baselines; `v5_rankbind/` = 55 RankBind runs (see §6) | core |
| `paper/` | the manuscript, **`main.tex` is canonical**; `figures/` | core |
| `scripts/` | SLURM submit + aggregation scripts | core |
| `docs/` | plans & specs (dev plan, phase-1 status, HP-sweep plan, benchmark plan) | reference |
| `external/` | upstream clones of the baseline repos (reference only, not run directly) | reference |
| `logs/` | SLURM job logs | keep (prunable) |
| `_archive/` | retired/superseded material, git-ignored, kept on disk + in history (see `_archive/README.md`): `v4_residue_only/` (12 GB, superseded by v5b) and `baselines_dropped/` (deepdta/dualbind_nvidia/gign) | archive (ignore) |

*(Cleaned 2026-06-04: `v4_residue_only/`, the dropped baselines, and an
empty `training/` were moved out of the active tree into `_archive/`.)*

---

## 6. The 55 RankBind runs: how to read them

`results/v5_rankbind/` holds 55 run directories. The name format is
`<timestamp>_<confighash>_<dataset>_<tag>`, e.g.
`20260603-122734_..._turnover_with_decoys_hp2000_bs_v2_hp2000_s7`.

- The `tag` tells you the experiment: `v3`/`v4` (BRENDA-200 model
  versions), `bs_v1`/`bs_v2_hp<N>` (enzyme-wide sweep), `_s7`/`_s1337`
  (random seed), `abl_*` (ablations), `probe_*` (diagnostics).
- Every run has a `manifest.json` (full config + metrics) and a 200×200
  `score_matrix_*.npy`.
- Flat index of all runs: `results/v5_rankbind/runs_manifest.csv`
  (regenerate with `python scripts/collect_v5_runs.py`).

You almost never need to open a run dir by hand; use the index CSV or
the paper-ready summary CSVs in `evaluation/attractor_results/`.

---

## 7. Paper section, and where its numbers come from

| Paper section | Backed by |
|---|---|
| §3 Diagnosis | `evaluation/{null_baselines,test_set_eval,cross_model_overlap}.py`; `attractor_results/test_summary_all.csv` |
| §4-6 RankBind + ablations | `v5_rankbind/`; `attractor_results/phase2_rankbind_multiseed.csv` |
| §7 Residue extension (v5b) | `configs/abl_attn_pool.json`; `attractor_results/attn_weight_*.csv` |
| §8.1 Enzyme-wide transfer | `results/v5_rankbind/*bs_v2_hp*`; Appendix sweep table; `null_prior_probe_bs_v2_peak.csv` |
| §8.3 External benchmarks (in progress) | `docs/BENCHMARK_INTEGRATION_PLAN.md`; `scripts/prep_benchmark_datasets.py` |
| Reproducibility | `REPRODUCIBILITY.md` maps each number to an exact command |

---

## 8. Current status (2026-06-04)

- Running now: §8.1 3-seed peaks for kcat_km + km (4 jobs). The
  turnover 3-seed is already done, MRR 0.417 ± 0.076 (beats the
  BRENDA-200 headline).
- Benchmark data fully ready: ESP / Davis / KIBA / BindingDB downloaded
  and all ESM2 embeddings computed (on `/work2`, expires ~Jul 4,
  reminder email set). §8.3 runs can start as soon as the GPU frees up.
- Next: finish the §8.1 aggregation (kcat_km + km peaks), then run
  RankBind + null-probe on the 4 benchmarks (v5b for ESP, v4 for the
  kinase sets), then write §8.3.

For the live to-do detail see `CLAUDE.md` (handoff) and the memory
files.

---

## 9. If you only remember three things

1. One model, three labels: `v5_rankbind` is the code; v4 is the
   headline, v5b the best.
2. One thesis: pooled AUC rewards a protein shortcut; RankBind +
   ranking metrics fix that.
3. Three dataset tiers: BRENDA-200 (headline), BRENDA+SABIO (scale),
   ESP/Davis/KIBA/BindingDB (generalise).