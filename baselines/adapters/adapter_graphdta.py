"""
adapter_graphdta.py — Data adapter for original GraphDTA repo.

Converts our BRENDA dataset into the format expected by GraphDTA:
  - Ligand: PyG Data with 78-d atom features (from RDKit)
  - Protein: integer-encoded sequence (max_len=1000, 25 AA + pad)

Uses the original GraphDTA featurization code from external/GraphDTA/create_data.py.
"""

import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem

# Add original GraphDTA to path
EXTERNAL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'GraphDTA')
)
sys.path.insert(0, EXTERNAL_ROOT)

from common import BRENDADataConfig


# ── Featurization (from GraphDTA/create_data.py) ─────────────────────────────

SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOC)}
MAX_SEQ_LEN = 1000


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """Compute atom features matching GraphDTA's 78-d encoding."""
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
            'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
            'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge',
            'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg',
            'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()],
        dtype=np.float32
    )


def smile_to_graph(smile):
    """Convert SMILES to PyG Data with 78-d atom features.

    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    features = []
    for atom in mol.GetAtoms():
        features.append(atom_features(atom))
    features = np.array(features)

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


def seq_to_ids(sequence, max_len=MAX_SEQ_LEN):
    """Encode protein sequence as integer tensor (GraphDTA encoding)."""
    ids = [SEQ_DICT.get(aa, 0) for aa in sequence[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ── Dataset ──────────────────────────────────────────────────────────────────

class GraphDTADataset(torch.utils.data.Dataset):
    """Dataset adapter for the original GraphDTA model."""

    def __init__(self, config: BRENDADataConfig, indices: list):
        self.config = config
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

            graph = smile_to_graph(smiles)
            if graph is None:
                continue

            seq_ids = seq_to_ids(self.sequences[uniprot])
            self.items.append((graph, seq_ids, float(label), smiles, uniprot))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        graph, seq_ids, label, smiles, uniprot = self.items[idx]
        # GraphDTA stores target as 2D [1, max_len] so PyG batches to [B, max_len]
        graph = graph.clone()
        graph.target = seq_ids.unsqueeze(0)
        graph.y = torch.tensor([label], dtype=torch.float)
        return graph


def get_model():
    """Import and return the original GraphDTA GCN model class."""
    sys.path.insert(0, os.path.join(EXTERNAL_ROOT, 'models'))
    from gcn import GCNNet
    return GCNNet
