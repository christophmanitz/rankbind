# Benchmark integration plan

Pre-registered plan for adding external benchmarks to RankBind, written
before any new number is read, with the same discipline as
`HP_SWEEP_INTEGRATION_PLAN.md`. The goal is to answer the reviewer
question of whether this generalises beyond one BRENDA sub-corpus, along
two axes.

## Two distinct claims, two dataset roles

1. Does the *diagnosis* generalise beyond enzyme-substrate data?
   Davis, KIBA and BindingDB are kinase(-family) affinity benchmarks,
   not enzyme-substrate. We do not claim RankBind beats specialised
   affinity models there. We test whether the shortcut exists: do
   published DTI models pass pooled AUC while per-ligand / matrix
   ranking is weak, and is the attractor distribution recoverable from
   `null_prot_prior`? The `null_prot_prior` probe is the primary
   instrument; RankBind is reported as one of several models, not as
   the headline.

2. Does the *recipe* compete on a second enzyme-substrate corpus?
   ESP (Kroll et al. 2023) is the same task, binary enzyme-substrate
   with constructed negatives. Here RankBind may legitimately target
   competitive matrix MRR / Hit@K, and we compare against ESP's own
   reported numbers.

## Datasets

| Name | Source | Task | Binarisation | Split | ~#prot | ~#lig |
|---|---|---|---|---|---|---|
| `davis` | TDC `DTI('DAVIS')` | kinase Kd | label=1 if pKd ≥ 7 (Kd ≤ 100 nM) | cold-target (= protein split) | 442 | 68 |
| `kiba` | TDC `DTI('KIBA')` | KIBA score | label=1 if KIBA ≥ 12.1 | cold-target | 229 | 2111 |
| `bindingdb_kd` | TDC `DTI('BindingDB_Kd')` | Kd | label=1 if pKd ≥ 7 | cold-target | ~10-20k | ~10k |
| `esp` | github.com/AlexanderKroll/ESP (+ Zenodo) | enzyme-substrate | native binary (1 = substrate, 0 = decoy) | protein split (seed 42) | ~4k | ~10k |

The binarisation thresholds are the DeepDTA / DeepPurpose conventions
(Davis pKd≥7, KIBA≥12.1); they are choices, recorded here so they
cannot drift. Coverage and class balance after binarisation are logged
by the prep script and reported in §5.1 of the paper.

## Storage

Home only. `/work2/<user>` and `/work/<user>` are not writable for this
account, and the one writable `/work2` dir belongs to an unrelated
project. Home has ~180 TB free and already hosts the 21 GB shared ESM2
store.

- Pairs/sequence CSVs: `reactionDataFiltering/data/interim/benchmarks/<name>/`
  (`pairs.csv`, `sequences.csv`, `prep_card.json`).
- ESM2 per-residue `.pt`: written into the shared store
  `reactionDataFiltering/data/interim/esm2_embeddings_shared/` keyed by a
  stable protein id, with a per-benchmark `esm2_embeddings/` symlink dir,
  identical to the existing BRENDA+SABIO scheme (`dedup_embeddings.py`).
  If BindingDB pushes the home footprint uncomfortably high, migrate the
  whole store to a `/work2` allocation later and re-point the symlinks;
  the loader follows symlinks transparently, so no code change is needed.

## Loader schema (do not deviate: `v5_rankbind/data.py`)

- `pairs.csv`: columns `uniprot`, `substrate_smiles`, `label` (0/1). The
  loader mints `idx` itself.
- `sequences.csv`: columns `uniprot`, `sequence` (+ optional `length`).
- `uniprot` is a stable per-protein id. TDC gives amino-acid sequences but
  not UniProt accessions, so we synthesise `id = <name>_<sha1(sequence)[:10]>`
  (collision-safe, deterministic, dedups identical sequences across splits).
- ChemBERTa (`DeepChem/ChemBERTa-77M-MLM`, mean-pool) is cached on first
  access by `data.py`; a pre-cache pass is optional.

## Embeddings

Reuse `reactionDataFiltering/reaction_data/embeddings.py::compute_esm2_embeddings`
(`facebook/esm2_t33_650M_UR50D`, per-residue, 1280-dim, the v5_rankbind
default). One GPU job per benchmark, submitted after the turnover
multi-seed jobs free the queue. Davis/KIBA take minutes; ESP/BindingDB
are the long ones.

## Pre-registered reading: new §8.3 "Does the diagnosis generalise?"

For each benchmark we report, for RankBind and one standard baseline
(re-using a published DTI model where numbers exist, else a BCE
control): pooled AUC, matrix/per-ligand MRR, and Top-10 Jaccard vs
`null_prot_prior`.

- Shortcut present elsewhere (pooled AUC high AND ranking weak AND
  Jaccard ≥ 0.30 for the baseline): the strongest result, the diagnosis
  is not a BRENDA artefact. §8.3 reports it as the headline
  generalisation.
- Shortcut absent on affinity data (baseline ranking already strong,
  Jaccard low): also publishable and honest, it bounds the claim to
  skewed-prior enzyme-substrate regimes. §8.3 says so explicitly and
  the abstract's scope sentence is tightened.
- ESP competitive (RankBind matrix MRR within range of ESP's reported
  performance): promotes ESP to a second headline enzyme-substrate
  result.
- ESP not competitive: reported as a transfer/limitation row, consistent
  with the single-seed enzyme-wide caveat already in §8.1.

These rules are fixed now; the eventual numbers select which paragraph
is written, not how it is framed.

## Execution order

1. ESP first (highest scientific value, same task). 2. Davis (fast, dense
matrix, the cleanest matrix-MRR test). 3. KIBA. 4. BindingDB last
(largest, appendix-grade stress test). Prep (download + CSV) is
CPU/network and runs now; embedding GPU jobs wait for the turnover
queue.
