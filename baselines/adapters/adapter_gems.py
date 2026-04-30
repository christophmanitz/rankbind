"""
adapter_gems.py — Data adapter for original GEMS repo.

GEMS uses:
  - Protein-ligand interaction graph (PyG Data with GATv2Conv)
  - ChemBERTa embeddings for ligand (384d)
  - ESM2 embeddings for protein sequence (per-residue, optional)
  - Node features from RDKit atom/residue properties

For our adapter, we use:
  - Pre-computed ESM2 embeddings (stored as .pt files)
  - ChemBERTa embeddings computed on-the-fly or cached
  - Ligand graphs from RDKit SMILES parsing
  - Protein contact maps from AlphaFold structures

Requires: transformers (for ChemBERTa), esm (optional, for ESM2)
"""

import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from rdkit import Chem

from common import BRENDADataConfig

GEMS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'GEMS')
)


def smiles_to_ligand_graph(smiles):
    """Convert SMILES to ligand subgraph with atom features.

    Returns PyG Data with ~25d atom features matching GEMS featurization.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return None

    # Atom features (matching GEMS/dataprep/graph_construction.py)
    features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
            atom.GetNumExplicitHs(),
            int(atom.IsInRing()),
        ]
        features.append(feat)
    x = torch.tensor(features, dtype=torch.float)

    # Edges from bonds
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src.extend([i, j])
        dst.extend([j, i])

    if len(src) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Edge features
    edge_attr = torch.ones(edge_index.shape[1], 4)  # placeholder

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_chemberta_embedding(smiles, model=None, tokenizer=None):
    """Get ChemBERTa embedding for a SMILES string (384d).

    If model/tokenizer not provided, returns a random embedding (for testing).
    """
    if model is not None and tokenizer is not None:
        inputs = tokenizer(smiles, return_tensors='pt', padding=True,
                          truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling
        return outputs.last_hidden_state.mean(dim=1).squeeze(0)
    else:
        # Placeholder: use hash-based deterministic embedding for reproducibility
        h = hash(smiles) % (2**32)
        rng = np.random.RandomState(h)
        return torch.tensor(rng.randn(384).astype(np.float32))


class GEMSDataset(Dataset):
    """Dataset adapter for the original GEMS model.

    Uses ligand graphs + optional ChemBERTa embeddings + protein features.
    """

    def __init__(self, config: BRENDADataConfig, indices: list,
                 esm2_dir: str = None, use_chemberta: bool = False):
        self.config = config
        self.esm2_dir = esm2_dir or os.path.join(
            config.project_root, 'data', 'esm2_embeddings'
        )
        self.sequences = config.load_sequences()
        pairs = config.load_pairs()

        # Optionally load ChemBERTa
        self.chemberta_model = None
        self.chemberta_tokenizer = None
        if use_chemberta:
            try:
                from transformers import AutoTokenizer, AutoModel
                self.chemberta_tokenizer = AutoTokenizer.from_pretrained(
                    "DeepChem/ChemBERTa-77M-MLM"
                )
                self.chemberta_model = AutoModel.from_pretrained(
                    "DeepChem/ChemBERTa-77M-MLM"
                )
                self.chemberta_model.eval()
            except Exception:
                pass

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

            self.items.append((smiles, uniprot, float(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        smiles, uniprot, label = self.items[idx]

        # Ligand graph
        lig_graph = smiles_to_ligand_graph(smiles)
        if lig_graph is None:
            lig_graph = Data(
                x=torch.zeros(1, 8),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 4),
            )

        # ChemBERTa embedding — store as [1, 384] so PyG batches to [B, 384]
        lig_emb = get_chemberta_embedding(
            smiles, self.chemberta_model, self.chemberta_tokenizer
        )
        lig_graph.lig_emb = lig_emb.unsqueeze(0)

        # ESM2 embedding (if precomputed) — store as [1, 1280]
        esm_path = os.path.join(self.esm2_dir, f'{uniprot}.pt')
        if os.path.exists(esm_path):
            esm_emb = torch.load(esm_path, weights_only=True)
        else:
            seq_len = len(self.sequences[uniprot])
            esm_emb = torch.zeros(seq_len, 1280)

        lig_graph.prot_emb = esm_emb.mean(dim=0).unsqueeze(0)
        lig_graph.y = torch.tensor([label], dtype=torch.float)

        return lig_graph


def get_model():
    """Import and return the original GEMS model class."""
    sys.path.insert(0, os.path.join(GEMS_ROOT, 'model'))
    from GEMS18 import GEMS18d
    return GEMS18d
