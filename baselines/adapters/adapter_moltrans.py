"""
adapter_moltrans.py — Data adapter for original MolTrans repo.

Converts our BRENDA dataset into the format expected by MolTrans:
  - Drug: BPE-tokenized SMILES (max_d=50, vocab from ChEMBL)
  - Protein: BPE-tokenized sequence (max_p=545, vocab from UniProt)
  - Both include attention masks

Uses the original MolTrans BPE encoding from external/MolTrans/ESPF/.
Requires: subword-nmt (pip install subword-nmt)
"""

import os
import sys
import codecs
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import BRENDADataConfig

# Paths to original MolTrans vocab files
MOLTRANS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'MolTrans')
)
ESPF_DIR = os.path.join(MOLTRANS_ROOT, 'ESPF')


def _load_bpe_encoder():
    """Load BPE encoders from the original MolTrans repo."""
    from subword_nmt.apply_bpe import BPE

    # Protein BPE
    prot_codes = codecs.open(os.path.join(ESPF_DIR, 'protein_codes_uniprot.txt'))
    pbpe = BPE(prot_codes, merges=-1, separator='')
    sub_csv = pd.read_csv(os.path.join(ESPF_DIR, 'subword_units_map_uniprot.csv'))
    words2idx_p = dict(zip(sub_csv['index'].values, range(len(sub_csv))))

    # Drug BPE
    drug_codes = codecs.open(os.path.join(ESPF_DIR, 'drug_codes_chembl.txt'))
    dbpe = BPE(drug_codes, merges=-1, separator='')
    sub_csv = pd.read_csv(os.path.join(ESPF_DIR, 'subword_units_map_chembl.csv'))
    words2idx_d = dict(zip(sub_csv['index'].values, range(len(sub_csv))))

    return pbpe, words2idx_p, dbpe, words2idx_d


# Lazy-loaded global BPE objects
_BPE_CACHE = None


def get_bpe():
    global _BPE_CACHE
    if _BPE_CACHE is None:
        _BPE_CACHE = _load_bpe_encoder()
    return _BPE_CACHE


def protein2emb(sequence, max_p=545):
    """Encode protein sequence via BPE → (ids, mask)."""
    pbpe, words2idx_p, _, _ = get_bpe()
    tokens = pbpe.process_line(sequence).split()
    try:
        ids = np.array([words2idx_p[t] for t in tokens])
    except KeyError:
        ids = np.array([0])

    l = len(ids)
    if l < max_p:
        ids = np.pad(ids, (0, max_p - l), constant_values=0)
        mask = np.array([1] * l + [0] * (max_p - l))
    else:
        ids = ids[:max_p]
        mask = np.ones(max_p, dtype=np.int64)
    return ids, mask


def drug2emb(smiles, max_d=50):
    """Encode SMILES via BPE → (ids, mask)."""
    _, _, dbpe, words2idx_d = get_bpe()
    tokens = dbpe.process_line(smiles).split()
    try:
        ids = np.array([words2idx_d[t] for t in tokens])
    except KeyError:
        ids = np.array([0])

    l = len(ids)
    if l < max_d:
        ids = np.pad(ids, (0, max_d - l), constant_values=0)
        mask = np.array([1] * l + [0] * (max_d - l))
    else:
        ids = ids[:max_d]
        mask = np.ones(max_d, dtype=np.int64)
    return ids, mask


class MolTransDataset(Dataset):
    """Dataset adapter for the original MolTrans model."""

    def __init__(self, config: BRENDADataConfig, indices: list):
        self.sequences = config.load_sequences()
        pairs = config.load_pairs()

        self.items = []
        for idx in indices:
            row = pairs[pairs['idx'] == idx]
            if row.empty:
                continue
            row = row.iloc[0]
            uniprot = row['uniprot']
            smiles = row['substrate_smiles']
            label = row['label']

            if uniprot not in self.sequences:
                continue

            self.items.append((
                self.sequences[uniprot], smiles, float(label)
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, smiles, label = self.items[idx]
        d_ids, d_mask = drug2emb(smiles)
        p_ids, p_mask = protein2emb(seq)
        return (
            torch.tensor(d_ids, dtype=torch.long),
            torch.tensor(d_mask, dtype=torch.long),
            torch.tensor(p_ids, dtype=torch.long),
            torch.tensor(p_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )


def get_model():
    """Import and return the original MolTrans model class."""
    sys.path.insert(0, MOLTRANS_ROOT)
    from models import BIN_Interaction_Flat
    return BIN_Interaction_Flat


def get_model_config(batch_size=32):
    """Return the MolTrans config dict with our dataset parameters.

    The original MolTrans uses BPE vocab sizes from ChEMBL/UniProt.
    We use the same ESPF vocab, so input_dim_drug/target match.
    """
    sys.path.insert(0, MOLTRANS_ROOT)
    from config import BIN_config_DBPE
    config = BIN_config_DBPE()
    # Override batch_size (MolTrans hardcodes this in forward)
    config['batch_size'] = batch_size
    return config
