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
