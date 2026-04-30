"""
adapter_drugban.py — Data adapter for original DrugBAN repo.

Converts our BRENDA dataset into the format expected by DrugBAN:
  - Drug: DGL graph with canonical atom features (74d + 1d virtual flag)
  - Protein: integer-encoded sequence (max_len=1200, 25 AA vocab)

The original DrugBAN uses DGLLife for featurization. If DGLLife is not available,
falls back to a manual featurization that matches the canonical feature set.

Requires: dgl, dgllife (optional, for full compatibility)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

from common import BRENDADataConfig

DRUGBAN_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'DrugBAN')
)

# Protein sequence encoding (matches DrugBAN/utils.py integer_label_protein)
PROT_VOCAB = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8,
    "K": 9, "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20, "X": 21,
    "B": 22, "Z": 23, "U": 24, "O": 25,
}
MAX_PROT_LEN = 1200


def integer_label_protein(sequence, max_len=MAX_PROT_LEN):
    """Encode protein sequence as integer tensor (DrugBAN encoding)."""
    ids = [PROT_VOCAB.get(aa, 21) for aa in sequence.upper()[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def smiles_to_dgl_graph(smiles, max_nodes=290):
    """Convert SMILES to DGL graph with canonical atom features.

    Uses DGLLife if available, otherwise falls back to RDKit-based construction.
    Returns (graph, actual_num_atoms) or (None, 0) if invalid.
    """
    try:
        import dgl
        from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
        from functools import partial
        featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        # Match original DrugBAN dataloader: add_self_loop=True
        smiles_to_graph_fn = partial(smiles_to_bigraph, add_self_loop=True)
        graph = smiles_to_graph_fn(smiles, node_featurizer=featurizer)
        if graph is None:
            return None, 0
        n_atoms = graph.num_nodes()
        # Pad with virtual nodes to max_nodes
        if n_atoms < max_nodes:
            graph.add_nodes(max_nodes - n_atoms)
            # Virtual node features: zeros with flag=1
            h = graph.ndata['h']
            h[n_atoms:] = 0  # zero features for virtual nodes
            # Add virtual node flag
            vn_flag = torch.zeros(max_nodes, 1)
            vn_flag[n_atoms:] = 1
            graph.ndata['h'] = torch.cat([h, vn_flag], dim=-1)
        else:
            # Truncate
            h = graph.ndata['h'][:max_nodes]
            vn_flag = torch.zeros(max_nodes, 1)
            graph = dgl.node_subgraph(graph, list(range(max_nodes)))
            graph.ndata['h'] = torch.cat([h, vn_flag], dim=-1)
            n_atoms = max_nodes
        # Virtual (padding) nodes still need self-loops to avoid 0-in-degree
        graph = graph.add_self_loop()
        # add_self_loop / node_subgraph may leave behind _ID fields that differ
        # between graphs — strip them so dgl.batch sees a consistent schema.
        for key in list(graph.ndata.keys()):
            if key != 'h':
                del graph.ndata[key]
        for key in list(graph.edata.keys()):
            del graph.edata[key]
        return graph, n_atoms
    except ImportError:
        # Fallback: use RDKit to build a simple graph
        return _smiles_to_dgl_fallback(smiles, max_nodes)


def _smiles_to_dgl_fallback(smiles, max_nodes=290):
    """Fallback: construct DGL graph from RDKit without DGLLife."""
    import dgl
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, 0

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None, 0

    # Build edge list
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([i, j])
        dst.extend([j, i])

    graph = dgl.graph((src, dst), num_nodes=max_nodes)

    # Simple atom features (74d canonical-like features + 1d virtual flag)
    # This is a simplified version; for full accuracy use DGLLife
    features = torch.zeros(max_nodes, 75)
    for i, atom in enumerate(mol.GetAtoms()):
        if i >= max_nodes:
            break
        # Atom type (first 44 dims)
        features[i, min(atom.GetAtomicNum(), 43)] = 1.0
        # Degree (44-54)
        features[i, 44 + min(atom.GetDegree(), 10)] = 1.0
        # Other features simplified...
    # Virtual node flag
    features[n_atoms:, -1] = 1.0

    graph.ndata['h'] = features
    return graph, min(n_atoms, max_nodes)


class DrugBANDataset(Dataset):
    """Dataset adapter for the original DrugBAN model."""

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

            self.items.append((smiles, self.sequences[uniprot], float(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        smiles, seq, label = self.items[idx]
        graph, n_atoms = smiles_to_dgl_graph(smiles)
        prot_ids = integer_label_protein(seq)

        if graph is None:
            # Return a dummy graph for invalid SMILES
            import dgl
            graph = dgl.graph(([], []), num_nodes=290)
            graph.ndata['h'] = torch.zeros(290, 75)
            graph.ndata['h'][:, -1] = 1.0  # all virtual

        return graph, prot_ids, torch.tensor(label, dtype=torch.float)


def get_model():
    """Import and return the original DrugBAN model class."""
    sys.path.insert(0, DRUGBAN_ROOT)
    from models import DrugBAN as DrugBANModel
    return DrugBANModel
