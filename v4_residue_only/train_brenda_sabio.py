"""
train_brenda_sabio.py — runner for ResidueOnlyBind on the three BRENDA+SABIO
datasets (kcat_km / km / turnover), using the augmented `with_decoys.csv`
variants under reactionDataFiltering/data/interim/.

Reuses train.py / dataset.py / losses.py from the v4_residue_only stack
without modification — only patches module-level paths and skips the
hydrolase-only EC filter (BRENDA+SABIO is enzyme-wide, not hydrolase-only,
so the filter would drop ~70 % of rows).

Usage (SLURM-internal, see run_brenda_sabio.sh):
    python train_brenda_sabio.py --dataset {kcat_km,km,turnover} \
                                 [--epochs N] [--batch-size B] \
                                 [--max-per-protein M]
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Make HierAtomBind/dataset/losses importable (they live in newclaudemodel/).
# Order matters: prepend newclaudemodel FIRST so that `_HERE`-relative
# `import dataset` etc. inside v4 train.py reach newclaudemodel — but THEN
# prepend `_HERE` on top of that, so `import train` resolves to *this*
# directory's v4 train.py (which is what we want to monkey-patch).
sys.path.insert(0, '/home/sc.uni-leipzig.de/zw93onug/newclaudemodel')
sys.path.insert(0, _HERE)

import train as t  # noqa: E402  (imports config + main from v4 train.py)
from dataset_split import SplitCacheDataset  # noqa: E402


PROJECT_ROOT = '/home/sc.uni-leipzig.de/zw93onug/rankbind'

# All three BRENDA+SABIO datasets share the same AlphaFold structures —
# put one protein-graph cache at the v4_residue_only/ level so each
# uniprot is built exactly once across all three runs.
SHARED_PROTEIN_CACHE = os.path.join(
    _HERE, 'cache_brenda_sabio_proteins',
)

DATASET_CSVS = {
    'kcat_km':  'reactionDataFiltering/data/interim/kcat_km_brenda_sabio/with_decoys.csv',
    'km':       'reactionDataFiltering/data/interim/km_brenda_sabio/with_decoys.csv',
    'turnover': 'reactionDataFiltering/data/interim/turnover_brenda_sabio/with_decoys.csv',
}

PDB_DIR = os.path.join(
    PROJECT_ROOT,
    'reactionDataFiltering/data/raw/brenda_sabio_2026-04-29/structures',
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_CSVS))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-per-protein', type=int, default=8)
    ap.add_argument('--early-stop-patience', type=int, default=15)
    ap.add_argument('--tag', default='bs_v1')
    args = ap.parse_args()

    sub = f'{args.dataset}_{args.tag}'

    # Per-dataset run directory: holds processed graphs + checkpoints + logs +
    # plots so the three datasets can train concurrently without colliding.
    run_root = os.path.join(_HERE, 'runs_brenda_sabio', sub)
    os.makedirs(run_root, exist_ok=True)
    for sd in ('checkpoints', 'logs', 'plots'):
        os.makedirs(os.path.join(run_root, sd), exist_ok=True)

    csv_path = os.path.join(PROJECT_ROOT, DATASET_CSVS[args.dataset])

    # ── Patch train.py module-level state ────────────────────────────────────
    # Per-dataset paths.
    t.CSV_PATH        = csv_path
    t.PROTEIN_DIR     = PDB_DIR
    # DATASET_ROOT holds the processed_hieratom/data_*.pt cache.
    t.DATASET_ROOT    = run_root

    # Schema: BRENDA+SABIO uses `ec1` (single-digit top-level EC) — different
    # from the legacy 'ec' column with full hierarchical strings. Setting
    # EC_COL='' disables (a) the hydrolase-only filter inside dataset._process
    # (line ~136) and (b) the same filter inside train.main() (line ~862).
    # _parse_ec_class on missing input returns -1, so the EC auxiliary head
    # gets ignore_index=-1 and effectively trains nothing — acceptable for
    # this run; the remaining BCE + NCE + triplet + Tanimoto + regression
    # signals are intact.
    t.EC_COL          = ''

    # Hyperparams scaled down for the larger BRENDA+SABIO datasets
    # (~43k/57k/?? rows after augmentation, vs ~6k in legacy BRENDA-200).
    t.BATCH_SIZE          = args.batch_size
    t.NUM_EPOCHS          = args.epochs
    t.MAX_PER_PROTEIN     = args.max_per_protein
    t.EARLY_STOP_PATIENCE = args.early_stop_patience

    # Redirect _HERE so per-epoch checkpoint saves
    # (`os.path.join(_HERE, 'checkpoints', f'epoch_{N:03d}.pt')` inline in
    # train.main()) land under the per-dataset run_root.
    t._HERE             = run_root
    t.BEST_MODEL_PATH   = os.path.join(run_root, 'checkpoints', 'best_model.pt')
    t.METRICS_CSV_PATH  = os.path.join(run_root, 'logs', 'metrics.csv')
    t.PLOT_DIR          = os.path.join(run_root, 'plots')

    # Swap the legacy per-row HierAtomBindDataset for the per-uniprot +
    # per-SMILES split cache. Wraps ctor so train.main() can construct it
    # without needing to know about shared_protein_cache.
    os.makedirs(SHARED_PROTEIN_CACHE, exist_ok=True)
    _shared = SHARED_PROTEIN_CACHE

    def _make_split_dataset(*a, **kw):
        kw.setdefault('shared_protein_cache', _shared)
        return SplitCacheDataset(*a, **kw)

    t.HierAtomBindDataset = _make_split_dataset

    print(f'[bs_runner] dataset      = {args.dataset}', flush=True)
    print(f'[bs_runner] csv          = {csv_path}', flush=True)
    print(f'[bs_runner] pdb_dir      = {PDB_DIR}', flush=True)
    print(f'[bs_runner] run_root     = {run_root}', flush=True)
    print(f'[bs_runner] shared_prot  = {SHARED_PROTEIN_CACHE}', flush=True)
    print(f'[bs_runner] epochs       = {args.epochs}', flush=True)
    print(f'[bs_runner] batch_size   = {args.batch_size}', flush=True)
    print(f'[bs_runner] cap/protein  = {args.max_per_protein}', flush=True)

    t.main()


if __name__ == '__main__':
    main()
