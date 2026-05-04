"""
dataset_split.py — drop-in replacement for HierAtomBindDataset that caches:

  proteins/<uniprot>.pt   one residue+atom graph per unique uniprot
  mols/<sha1(smi)>.pt     one molecule graph per unique SMILES
  index.pt                 list of (mol_hash, uniprot, y, tanimoto) per CSV row

Why: the legacy HierAtomBindDataset caches one (mol, residue+atom) tuple
per CSV ROW. For the BRENDA+SABIO with_decoys variants (43k–57k rows,
2.7k–7k unique uniprots, ~10k unique SMILES), this rebuilt the protein
graph for the same uniprot ~14× and inflated cache to ~30 GB / dataset.

Same external API: returns `(mol_graph, residue_graph_with_atom_attrs, y)`
on __getitem__, so the existing collate_fn / HierAtomBindAugmentedDataset
do not need to change.

Per-row metadata (tanimoto, ec_class) is attached at __getitem__ time to
a fresh PyG Data wrapper sharing the cached tensors — does not mutate the
shared cache.
"""

import hashlib
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

# These live in newclaudemodel/; train_brenda_sabio.py prepends that to
# sys.path before importing this module.
from little_sweet_graphs_v2 import molecule_text_to_3d_graph, create_residue_graph  # noqa: E402
from protein_atom_graphs import build_protein_atom_graph  # noqa: E402

log = logging.getLogger(__name__)


def _smiles_hash(smi: str) -> str:
    return hashlib.sha1(smi.encode('utf-8')).hexdigest()[:16]


def _atomic_save(obj, dest_path: str) -> None:
    """torch.save with atomic rename — safe against concurrent writers
    sharing a protein cache directory across SLURM jobs."""
    tmp = f'{dest_path}.tmp.{os.getpid()}'
    torch.save(obj, tmp)
    os.replace(tmp, dest_path)


def _build_protein(pdb_file: str):
    residue_graph = create_residue_graph(pdb_file)
    atom_graph    = build_protein_atom_graph(pdb_file)
    residue_graph.atom_x         = atom_graph.x
    residue_graph.atom_edge_index = atom_graph.edge_index
    residue_graph.atom_edge_attr  = atom_graph.edge_attr
    residue_graph.atom_coords     = atom_graph.atom_coords
    residue_graph.residue_index   = atom_graph.residue_index
    residue_graph.r2a_padded      = atom_graph.r2a_padded
    return residue_graph


class SplitCacheDataset(Dataset):
    """
    Args (drop-in compatible with the legacy HierAtomBindDataset signature):
      root           : per-dataset run dir; mols/ + index.pt go under
                       <root>/cache_split/
      csv_path       : CSV with rows (uniprot, substrate_smiles, value, ...)
      smiles_col     : column for ligand SMILES
      protein_col    : column for uniprot accession
      protein_dir    : dir containing AF-{uniprot}-F1-model_v6.pdb
      target_cols    : list with one column name (e.g. ['value'])
      ec_col         : ignored (always sets ec_class=-1; BRENDA+SABIO ec1
                       is single-digit, _parse_ec_class returns -1 anyway)
      tanimoto_col   : column with TanimotoSimilarity (0 for true positives)
      head           : optional row cap for smoke tests
      resume         : skip building cache files that already exist
      shared_protein_cache : optional path; if set, protein graphs go there
                       (so all three BRENDA+SABIO datasets share the cache)
    """

    def __init__(
        self,
        root,
        csv_path,
        smiles_col,
        protein_col,
        protein_dir,
        target_cols,
        ec_col='',
        tanimoto_col='TanimotoSimilarity',
        transform=None,
        head=None,
        resume=True,
        fast_load=False,
        shared_protein_cache=None,
    ):
        self.csv_path     = csv_path
        self.smiles_col   = smiles_col
        self.protein_col  = protein_col
        self.protein_dir  = protein_dir
        self.target_cols  = target_cols
        self.ec_col       = ec_col
        self.tanimoto_col = tanimoto_col
        self.head         = head
        self.resume       = resume

        # Layout
        cache_root = os.path.join(root, 'cache_split')
        os.makedirs(cache_root, exist_ok=True)
        if shared_protein_cache is None:
            shared_protein_cache = os.path.join(cache_root, 'proteins')
        os.makedirs(shared_protein_cache, exist_ok=True)
        self.prot_cache_dir = shared_protein_cache

        self.mol_cache_dir = os.path.join(cache_root, 'mols')
        os.makedirs(self.mol_cache_dir, exist_ok=True)

        index_path = os.path.join(cache_root, 'index.pt')

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f'CSV not found: {self.csv_path}')

        # ── Build / reuse caches ────────────────────────────────────────────
        df = pd.read_csv(self.csv_path)
        if self.head is not None:
            df = df.head(self.head)
        log.info(f'CSV rows: {len(df)}')

        # Hydrolase-only filter is intentionally skipped here — see
        # train_brenda_sabio.py docstring; keep the full enzyme-wide dataset.

        # 1. Protein graphs (one per unique uniprot)
        unique_ups = sorted(df[self.protein_col].dropna().unique())
        log.info(f'Unique uniprots: {len(unique_ups)}')
        n_built = n_skipped = n_missing_pdb = n_err = 0
        for up in tqdm(unique_ups, desc='Protein graphs'):
            out = os.path.join(self.prot_cache_dir, f'{up}.pt')
            if self.resume and os.path.exists(out):
                n_skipped += 1
                continue
            pdb = os.path.join(self.protein_dir, f'AF-{up}-F1-model_v6.pdb')
            if not os.path.exists(pdb):
                n_missing_pdb += 1
                continue
            try:
                g = _build_protein(pdb)
                _atomic_save(g, out)
                n_built += 1
            except Exception as e:
                n_err += 1
                if n_err <= 5:
                    log.error(f'protein build failed for {up}: {e}')
        log.info(
            f'Proteins: built={n_built} reused={n_skipped} '
            f'missing_pdb={n_missing_pdb} errors={n_err}'
        )

        # 2. Molecule graphs (one per unique SMILES)
        unique_smis = sorted(df[self.smiles_col].dropna().unique())
        log.info(f'Unique SMILES: {len(unique_smis)}')
        smi2hash = {}
        n_built = n_skipped = n_err = 0
        for s in tqdm(unique_smis, desc='Mol graphs'):
            h = _smiles_hash(s)
            smi2hash[s] = h
            out = os.path.join(self.mol_cache_dir, f'{h}.pt')
            if self.resume and os.path.exists(out):
                n_skipped += 1
                continue
            try:
                g = molecule_text_to_3d_graph(s, type='smiles')
                _atomic_save(g, out)
                n_built += 1
            except Exception as e:
                n_err += 1
                if n_err <= 5:
                    log.error(f'mol build failed for {s[:40]}…: {e}')
        log.info(f'Mols: built={n_built} reused={n_skipped} errors={n_err}')

        # 3. Index — one entry per CSV row that has both cache files present.
        # We track the *original CSV row index* so that train.main()'s
        # `df.iloc[file_indices]` alignment trick keeps working.
        entries = []
        csv_row_indices = []
        for csv_idx, row in df.iterrows():
            up = row[self.protein_col]
            s  = row[self.smiles_col]
            if pd.isna(up) or pd.isna(s):
                continue
            prot_path = os.path.join(self.prot_cache_dir, f'{up}.pt')
            if not os.path.exists(prot_path):
                continue
            h = smi2hash.get(s, _smiles_hash(s))
            mol_path = os.path.join(self.mol_cache_dir, f'{h}.pt')
            if not os.path.exists(mol_path):
                continue
            y_arr = pd.to_numeric(row[self.target_cols], errors='coerce') \
                      .values.astype(float)
            tani = 0.0
            if self.tanimoto_col in row.index:
                v = row[self.tanimoto_col]
                if pd.notna(v):
                    tani = float(v)
            if y_arr[0] > 0:
                tani = 0.0  # zero out for true positives (matches legacy)
            entries.append((h, str(up), y_arr, float(tani)))
            csv_row_indices.append(int(csv_idx))

        self.entries = entries
        self.csv_row_indices = csv_row_indices
        torch.save(
            {'entries': entries, 'csv_row_indices': csv_row_indices},
            index_path,
        )
        log.info(
            f'SplitCache index: {len(entries)} usable rows '
            f'(of {len(df)} CSV rows)'
        )

        # files attribute used by train.py main() to extract protein ids
        # via filename. The integers encoded in the names must match the
        # *CSV row indices* so that `df.iloc[file_indices][PROTEIN_COL]`
        # in train.main() returns the correct uniprots.
        self.files = [f'data_{i}.pt' for i in csv_row_indices]

    # ── Dataset interface ───────────────────────────────────────────────────

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        h, up, y_arr, tani = self.entries[idx]
        prot_template = torch.load(
            os.path.join(self.prot_cache_dir, f'{up}.pt'),
            weights_only=False,
        )
        mol = torch.load(
            os.path.join(self.mol_cache_dir, f'{h}.pt'),
            weights_only=False,
        )

        # Wrap protein in a fresh Data so per-row metadata does not
        # mutate the shared cache tensors.
        prot = Data(
            x=prot_template.x,
            edge_index=prot_template.edge_index,
            edge_attr=prot_template.edge_attr,
            coords=prot_template.coords,
            atom_x=prot_template.atom_x,
            atom_edge_index=prot_template.atom_edge_index,
            atom_edge_attr=prot_template.atom_edge_attr,
            atom_coords=prot_template.atom_coords,
            residue_index=prot_template.residue_index,
            r2a_padded=prot_template.r2a_padded,
        )
        prot.tanimoto = torch.tensor(tani, dtype=torch.float)
        prot.ec_class = torch.tensor(-1, dtype=torch.long)

        y = torch.tensor(y_arr, dtype=torch.float)
        return mol, prot, y

    # Helper for train.py's protein_ids_arr construction
    def get_protein_ids(self):
        """Return a list of uniprot strings, one per index — same shape as
        df.iloc[file_indices][PROTEIN_COL].values in the legacy main()."""
        return np.array([e[1] for e in self.entries])
