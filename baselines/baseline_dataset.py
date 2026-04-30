"""
baseline_dataset.py — Shared dataset utilities for all sequence-based baselines.

Provides:
  ProteinTokenizer  : maps AA sequences → integer tensors [max_len]
  LigandTokenizer   : maps SMILES strings → integer tensors [max_len]
  BaselineDataset   : wraps processed_hieratom .pt files, returns (mol_graph, uniprot, y)
  SeqBaselineDataset: like BaselineDataset but additionally loads protein sequences
                      from data/sequences/sequences.csv

Usage (in baseline train scripts):
  from baseline_dataset import SeqBaselineDataset, ProteinTokenizer, LigandTokenizer
  ds = SeqBaselineDataset(pt_dir, seq_csv, ProteinTokenizer(), LigandTokenizer())
"""

import os
import glob
import string
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizers (shared by all sequence-based baselines)
# ═══════════════════════════════════════════════════════════════════════════════

class ProteinTokenizer:
    """Map amino-acid sequence strings to fixed-length integer tensors.
    Vocab: 20 standard AAs + B,J,O,U,X,Z (ambiguous) + PAD at index 0."""

    VOCAB = "ACDEFGHIKLMNPQRSTVWY" + "BJOUXZ"  # 26 chars

    def __init__(self, max_len: int = 1000):
        self.max_len = max_len
        self._char2idx = {c: i + 1 for i, c in enumerate(self.VOCAB)}
        self.vocab_size = len(self.VOCAB) + 1

    def __call__(self, sequence: str) -> torch.Tensor:
        ids = [self._char2idx.get(c, self._char2idx.get("X", 0))
               for c in sequence.upper()[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class LigandTokenizer:
    """Map SMILES strings to fixed-length integer tensors.
    Vocab: all printable SMILES chars (~65) + PAD at index 0."""

    VOCAB = string.ascii_letters + string.digits + "()[]=#@+-./%\\"

    def __init__(self, max_len: int = 100):
        self.max_len = max_len
        self._char2idx = {c: i + 1 for i, c in enumerate(self.VOCAB)}
        self.vocab_size = len(self.VOCAB) + 1

    def __call__(self, smiles: str) -> torch.Tensor:
        ids = [self._char2idx.get(c, 0) for c in smiles[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class BaselineDataset(Dataset):
    """
    Wraps processed_hieratom .pt files and returns graph data + label.
    Used by GraphDTA (which uses the ligand graph directly).

    Items: (mol_graph, uniprot_str, y_binary)
    """

    def __init__(self, pt_dir: str, max_samples: int = None):
        self.pt_files = sorted(
            glob.glob(os.path.join(pt_dir, 'data_*.pt')),
            key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
        )
        if max_samples:
            self.pt_files = self.pt_files[:max_samples]

        # Filter invalid files at init time
        valid = []
        for p in self.pt_files:
            try:
                mol, prot, y = torch.load(p, weights_only=False)
                if mol.x is not None and torch.isfinite(mol.x).all():
                    valid.append(p)
            except Exception:
                pass
        self.pt_files = valid
        print(f"BaselineDataset: {len(self.pt_files)} valid samples from {pt_dir}")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        mol, prot, y = torch.load(self.pt_files[idx], weights_only=False)
        y_bin = torch.tensor(float(y.item() > 0), dtype=torch.float32)
        uniprot = getattr(prot, 'uniprot', '') or ''
        return mol, uniprot, y_bin


class SeqBaselineDataset(Dataset):
    """
    Like BaselineDataset but also returns tokenised protein and ligand sequences.
    Used by DeepDTA and GraphDTA (protein sequence branch).

    Maps .pt file indices back to the CSV to retrieve SMILES and UniProt IDs
    (these are NOT stored as attributes in the .pt files).

    Items: (prot_ids, lig_ids, mol_graph, y_binary)
      prot_ids : LongTensor [L_prot]  (tokenised, padded)
      lig_ids  : LongTensor [L_lig]   (tokenised, padded)
      mol_graph: PyG Data             (for GraphDTA; ignored by DeepDTA)
      y_binary : FloatTensor scalar
    """

    def __init__(
        self,
        pt_dir: str,
        seq_csv: str,
        prot_tokenizer,
        lig_tokenizer,
        max_samples: int = None,
    ):
        # Load sequence lookup
        seq_df = pd.read_csv(seq_csv)
        self.seq_lookup = dict(zip(seq_df['uniprot'], seq_df['sequence']))
        print(f"Loaded {len(self.seq_lookup)} protein sequences from {seq_csv}")

        # Load the main CSV to map .pt file index → (uniprot, smiles)
        csv_path = os.path.join(os.path.dirname(pt_dir), 'dataset_with_decoys.csv')
        main_df = pd.read_csv(csv_path)
        print(f"Loaded {len(main_df)} rows from {csv_path}")

        self.prot_tokenizer = prot_tokenizer
        self.lig_tokenizer  = lig_tokenizer

        pt_files = sorted(
            glob.glob(os.path.join(pt_dir, 'data_*.pt')),
            key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
        )
        if max_samples:
            pt_files = pt_files[:max_samples]

        valid = []
        skipped_no_seq = 0
        skipped_invalid = 0
        for p in pt_files:
            try:
                # Extract CSV row index from filename: data_1234.pt → row 1234
                idx = int(os.path.basename(p).split('_')[1].split('.')[0])
                if idx >= len(main_df):
                    skipped_invalid += 1
                    continue

                row = main_df.iloc[idx]
                uniprot = str(row['uniprot'])
                smiles  = str(row['substrate_smiles'])

                if uniprot not in self.seq_lookup or not smiles or smiles == 'nan':
                    skipped_no_seq += 1
                    continue

                # Quick validation of .pt file
                mol, prot, y = torch.load(p, weights_only=False)
                if mol.x is None or not torch.isfinite(mol.x).all():
                    skipped_invalid += 1
                    continue

                valid.append((p, uniprot, smiles))
            except Exception:
                skipped_invalid += 1

        self.items = valid
        print(f"SeqBaselineDataset: {len(self.items)} valid samples "
              f"({skipped_no_seq} skipped — no sequence, "
              f"{skipped_invalid} skipped — invalid)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, uniprot, smiles = self.items[idx]
        mol, prot, y = torch.load(path, weights_only=False)

        seq      = self.seq_lookup[uniprot]
        prot_ids = self.prot_tokenizer(seq)
        lig_ids  = self.lig_tokenizer(smiles)
        y_bin    = torch.tensor(float(y.item() > 0), dtype=torch.float32)

        return prot_ids, lig_ids, mol, y_bin


def collate_seq(batch):
    """Collate for SeqBaselineDataset — returns padded tensors + PyG Batch."""
    from torch_geometric.data import Batch
    prot_ids, lig_ids, mols, ys = zip(*batch)
    return (
        torch.stack(prot_ids),    # [B, L_prot]
        torch.stack(lig_ids),     # [B, L_lig]
        Batch.from_data_list(mols),
        torch.stack(ys),          # [B]
    )
