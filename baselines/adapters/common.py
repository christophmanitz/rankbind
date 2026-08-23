"""
baselines/adapters/common.py — Unified data configuration and splits for all baselines.

All original-repo baselines share:
  - The same (protein, ligand, label) dataset from dataset_with_decoys.csv
  - The same train/val/test split (protein-based, no protein in both train and val)
  - The same evaluation pipeline (score matrix → attractor_diagnosis.py)
"""

import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BRENDADataConfig:
    """Unified data configuration for all baselines.

    All paths are absolute or relative to project root.
    """
    project_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )

    # Data files
    csv_path: str = ""
    seq_csv: str = ""
    struct_dir: str = ""  # AlphaFold PDB structures

    # Split params
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15

    def __post_init__(self):
        if not self.csv_path:
            self.csv_path = os.path.join(
                self.project_root, 'data', 'dataset_with_decoys.csv'
            )
        if not self.seq_csv:
            self.seq_csv = os.path.join(
                self.project_root, 'data', 'sequences', 'sequences.csv'
            )
        if not self.struct_dir:
            self.struct_dir = os.path.expanduser('~/hpc/structures')

    def load_pairs(self) -> pd.DataFrame:
        """Load all (protein, ligand, label) triplets.

        Returns DataFrame with columns:
            uniprot, substrate_smiles, label, idx
        where idx = original CSV row index (matches data_*.pt filenames).
        """
        df = pd.read_csv(self.csv_path)
        df['idx'] = df.index

        # Binary label: matches baseline_dataset.py — y > 0 is a binder.
        # The BRENDA/SABIO CSV stores affinity in the `value` column; zeros are
        # decoys injected by the hieratom pipeline.
        if 'is_decoy' in df.columns:
            df['label'] = (~df['is_decoy'].astype(bool)).astype(int)
        elif 'label' in df.columns:
            pass
        elif 'value' in df.columns:
            df['label'] = (df['value'].astype(float) > 0).astype(int)
        elif 'affinity' in df.columns:
            df['label'] = (df['affinity'].astype(float) > 0).astype(int)
        else:
            raise ValueError(
                f"No label column in {self.csv_path}; expected one of "
                "is_decoy/label/value/affinity"
            )

        return df[['uniprot', 'substrate_smiles', 'label', 'idx']]

    def load_sequences(self) -> dict:
        """Load uniprot → sequence mapping."""
        seq_df = pd.read_csv(self.seq_csv)
        return dict(zip(seq_df['uniprot'], seq_df['sequence']))

    def get_protein_split(self) -> tuple:
        """Protein-based train/val/test split.

        Ensures no protein appears in both train and val/test.
        Returns (train_indices, val_indices, test_indices) as lists of CSV row indices.
        """
        df = self.load_pairs()
        sequences = self.load_sequences()

        # Filter to proteins with sequences
        df = df[df['uniprot'].isin(sequences)].reset_index(drop=True)

        # Get unique proteins
        proteins = df['uniprot'].unique().tolist()
        rng = random.Random(self.seed)
        rng.shuffle(proteins)

        n_prot = len(proteins)
        n_val_prot = max(1, int(n_prot * self.val_frac))
        n_test_prot = max(1, int(n_prot * self.test_frac))

        test_prots = set(proteins[:n_test_prot])
        val_prots = set(proteins[n_test_prot:n_test_prot + n_val_prot])
        train_prots = set(proteins[n_test_prot + n_val_prot:])

        train_idx = df[df['uniprot'].isin(train_prots)]['idx'].tolist()
        val_idx = df[df['uniprot'].isin(val_prots)]['idx'].tolist()
        test_idx = df[df['uniprot'].isin(test_prots)]['idx'].tolist()

        return train_idx, val_idx, test_idx

    def load_canon_smiles(self) -> dict:
        """raw substrate_smiles -> canonical SMILES identity.

        Uses the ``substrate_smiles_canon`` column when present (same molecule
        definition as evaluation/leakage_audit.py L3); falls back to the raw
        string otherwise. Empty/NaN canon values fall back to raw too.
        """
        df = pd.read_csv(self.csv_path)
        if 'substrate_smiles_canon' in df.columns:
            canon = df['substrate_smiles_canon']
            canon = canon.where(canon.notna() & (canon.astype(str).str.len() > 0),
                                df['substrate_smiles'])
            return dict(zip(df['substrate_smiles'], canon))
        return dict(zip(df['substrate_smiles'], df['substrate_smiles']))

    def _partition_keys(self, keys: list, rng: random.Random) -> tuple:
        """Shuffle-split unique keys into (train_set, val_set, test_set)
        using val_frac/test_frac (test first, then val — same order as
        get_protein_split so fractions are drawn identically)."""
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * self.val_frac))
        n_test = max(1, int(len(keys) * self.test_frac))
        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test:n_test + n_val])
        train_keys = set(keys[n_test + n_val:])
        return train_keys, val_keys, test_keys

    def get_ligand_split(self) -> tuple:
        """Ligand-disjoint ("cold-ligand") train/val/test split.

        Partitions unique CANONICAL ligands (same molecule identity as
        leakage_audit L3) at val_frac/test_frac with the project seed; every
        row follows its ligand. Proteins recur freely across splits — this is
        the stress test that removes ligand-level recurrence while keeping the
        protein marginal. Same idx convention as get_protein_split.
        """
        df = self.load_pairs()
        sequences = self.load_sequences()
        df = df[df['uniprot'].isin(sequences)].reset_index(drop=True)

        canon = self.load_canon_smiles()
        keys = df['substrate_smiles'].map(
            lambda s: canon.get(s, s)).unique().tolist()

        rng = random.Random(self.seed)
        train_ligs, val_ligs, test_ligs = self._partition_keys(keys, rng)

        col = df['substrate_smiles'].map(lambda s: canon.get(s, s))
        train_idx = df[col.isin(train_ligs)]['idx'].tolist()
        val_idx = df[col.isin(val_ligs)]['idx'].tolist()
        test_idx = df[col.isin(test_ligs)]['idx'].tolist()
        return train_idx, val_idx, test_idx

    def get_double_cold_split(self) -> tuple:
        """Double-cold split: neither proteins nor ligands recur across
        train/val/test.

        Product partition: proteins are partitioned exactly as in
        get_protein_split and canonical ligands exactly as in
        get_ligand_split (same seed, same fractions). A row (p, l) goes to
        split S iff p is in the protein part AND l is in the ligand part of
        S; rows whose protein and ligand land in different parts are DROPPED
        (~46% under 0.70/0.15/0.15) so both axes stay strictly disjoint
        between every pair of folds. Returns the standard
        (train_idx, val_idx, test_idx) triple.
        """
        df = self.load_pairs()
        sequences = self.load_sequences()
        df = df[df['uniprot'].isin(sequences)].reset_index(drop=True)

        proteins = df['uniprot'].unique().tolist()
        canon = self.load_canon_smiles()
        ligands = df['substrate_smiles'].map(
            lambda s: canon.get(s, s)).unique().tolist()

        rng = random.Random(self.seed)
        tr_p, va_p, te_p = self._partition_keys(proteins, rng)
        rng2 = random.Random(self.seed)
        tr_l, va_l, te_l = self._partition_keys(ligands, rng2)

        col = df['substrate_smiles'].map(lambda s: canon.get(s, s))
        p_part = df['uniprot'].map(
            lambda p: 'train' if p in tr_p else 'val' if p in va_p else 'test')
        l_part = col.map(
            lambda s: 'train' if s in tr_l else 'val' if s in va_l else 'test')

        keep = p_part == l_part
        aligned = df[keep]
        part = p_part[keep]
        train_idx = aligned[part == 'train']['idx'].tolist()
        val_idx = aligned[part == 'val']['idx'].tolist()
        test_idx = aligned[part == 'test']['idx'].tolist()
        return train_idx, val_idx, test_idx

    def get_native_split(self) -> tuple:
        """Dataset-provided native split, read from the ``split`` column of
        the source CSV (values ``train`` / ``test``). A validation fraction
        (``val_frac``) is carved at random (seeded) out of the native train
        rows. Test rows are used verbatim.

        This exists to compare against a benchmark's own published numbers
        on its own held-out test set (e.g. ESP's phylogenetic test split vs
        Kroll et al.'s reported ROC-AUC) rather than the project's
        protein-stratified split. Returns (train_idx, val_idx, test_idx) as
        lists of CSV row indices (same ``idx`` convention as load_pairs).
        """
        df = pd.read_csv(self.csv_path)
        if 'split' not in df.columns:
            raise ValueError(
                f"No 'split' column in {self.csv_path}; native split "
                "unavailable for this dataset"
            )
        df['idx'] = df.index
        sequences = self.load_sequences()
        df = df[df['uniprot'].isin(sequences)]

        test_idx = df[df['split'] == 'test']['idx'].tolist()
        train_pool = df[df['split'] == 'train']['idx'].tolist()

        rng = random.Random(self.seed)
        rng.shuffle(train_pool)
        n_val = max(1, int(len(train_pool) * self.val_frac))
        val_idx = train_pool[:n_val]
        train_idx = train_pool[n_val:]

        return train_idx, val_idx, test_idx

    def get_random_split(self) -> tuple:
        """Transductive (pair-level) train/val/test split.

        Splits individual (protein, ligand) ROWS at random — the same protein
        may appear in train and test. This is NOT the canonical split; it
        exists only to measure the *ceiling* of extractable ligand-conditional
        signal (an upper bound on matrix-MRR under per-protein memorisation),
        as a counterfactual to the protein-disjoint get_protein_split().
        Same seed and val/test fractions so the only changed variable is
        protein disjointness. Returns (train_idx, val_idx, test_idx) row lists.
        """
        df = self.load_pairs()
        sequences = self.load_sequences()
        df = df[df['uniprot'].isin(sequences)].reset_index(drop=True)

        idx = df['idx'].tolist()
        rng = random.Random(self.seed)
        rng.shuffle(idx)

        n = len(idx)
        n_val = max(1, int(n * self.val_frac))
        n_test = max(1, int(n * self.test_frac))

        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        return train_idx, val_idx, test_idx
