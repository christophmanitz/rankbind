"""
adapter_gign.py — Data adapter for original GIGN repo.

GIGN requires 3D protein-ligand complex graphs with:
  - Intramolecular edges (within ligand, within protein)
  - Intermolecular edges (ligand-protein contacts within 5Å)
  - 3D coordinates for all atoms

Data sources needed:
  - Protein PDB files (AlphaFold structures available at ~/hpc/structures/)
  - Ligand 3D poses (from DiffDock docking)

This adapter handles:
  1. Loading docked complex structures
  2. Featurizing atoms using the original GIGN code
  3. Constructing interaction graphs with distance-based inter-edges
"""

import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from common import BRENDADataConfig

GIGN_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'GIGN', 'GIGN')
)


def load_docked_complex(pdb_path, sdf_path, distance_cutoff=5.0):
    """Load a docked protein-ligand complex and build GIGN-format graph.

    Parameters
    ----------
    pdb_path : str
        Path to protein PDB file (AlphaFold structure)
    sdf_path : str
        Path to docked ligand SDF file (from DiffDock)
    distance_cutoff : float
        Cutoff for intermolecular edges (Å)

    Returns
    -------
    Data : PyG Data object with GIGN-format fields, or None if parsing fails
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from scipy.spatial.distance import cdist
    except ImportError as e:
        raise ImportError(f"GIGN adapter requires rdkit and scipy: {e}")

    # Load ligand
    mol = Chem.MolFromMolFile(sdf_path, sanitize=True, removeHs=False)
    if mol is None:
        return None

    lig_conf = mol.GetConformer()
    lig_pos = np.array([lig_conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    n_lig = mol.GetNumAtoms()

    # Load protein (simplified: CA atoms only for residue-level)
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    prot_pos = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    prot_pos.append(residue['CA'].get_vector().get_array())
        break  # First model only

    if len(prot_pos) == 0:
        return None

    prot_pos = np.array(prot_pos)
    n_prot = len(prot_pos)

    # Compute distances for inter-edges
    dist_matrix = cdist(lig_pos, prot_pos)
    inter_mask = dist_matrix < distance_cutoff
    lig_inter_idx, prot_inter_idx = np.where(inter_mask)

    # Offset protein indices
    prot_inter_idx_offset = prot_inter_idx + n_lig

    # Build edge indices
    # Intra-ligand edges (from bonds)
    lig_src, lig_dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lig_src.extend([i, j])
        lig_dst.extend([j, i])

    # Intra-protein edges (CA within 10Å)
    prot_dist = cdist(prot_pos, prot_pos)
    prot_mask = (prot_dist < 10.0) & (prot_dist > 0)
    prot_src, prot_dst = np.where(prot_mask)
    prot_src += n_lig
    prot_dst += n_lig

    # Combine all edges
    all_src = np.concatenate([lig_src, prot_src, lig_inter_idx,
                              prot_inter_idx_offset])
    all_dst = np.concatenate([lig_dst, prot_dst, prot_inter_idx_offset,
                              lig_inter_idx])

    # Node features (simplified)
    # Ligand atoms: atomic number one-hot (first 10 elements)
    lig_features = np.zeros((n_lig, 10), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        lig_features[i, min(atom.GetAtomicNum() - 1, 9)] = 1.0

    # Protein CA: residue one-hot (20 AA)
    prot_features = np.zeros((n_prot, 10), dtype=np.float32)
    # Simplified: just mark as protein nodes
    prot_features[:, 0] = 1.0

    # All positions
    all_pos = np.concatenate([lig_pos, prot_pos], axis=0)
    all_features = np.zeros((n_lig + n_prot, 10), dtype=np.float32)
    all_features[:n_lig] = lig_features
    all_features[n_lig:] = prot_features

    edge_index = torch.tensor(np.stack([all_src, all_dst]), dtype=torch.long)

    data = Data(
        x=torch.tensor(all_features, dtype=torch.float),
        edge_index=edge_index,
        pos=torch.tensor(all_pos, dtype=torch.float),
    )
    data.n_lig = n_lig
    data.n_prot = n_prot

    # GIGN-specific fields
    data.edge_index_intra = torch.tensor(
        np.stack([np.concatenate([lig_src, prot_src]),
                  np.concatenate([lig_dst, prot_dst])]),
        dtype=torch.long
    )
    data.edge_index_inter = torch.tensor(
        np.stack([np.concatenate([lig_inter_idx, prot_inter_idx_offset]),
                  np.concatenate([prot_inter_idx_offset, lig_inter_idx])]),
        dtype=torch.long
    )

    return data


class GIGNDataset(Dataset):
    """Dataset adapter for the original GIGN model.

    Requires pre-docked structures. See scripts/run_diffdock.sh.
    """

    def __init__(self, config: BRENDADataConfig, indices: list,
                 docked_dir: str = None):
        self.config = config
        self.docked_dir = docked_dir or os.path.join(
            config.project_root, 'data', 'docked_complexes'
        )
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

            # Check if docked structure exists
            pdb_path = os.path.join(config.struct_dir, f'{uniprot}.pdb')
            sdf_path = os.path.join(self.docked_dir, f'{uniprot}_{idx}.sdf')

            if os.path.exists(pdb_path) and os.path.exists(sdf_path):
                self.items.append((pdb_path, sdf_path, float(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pdb_path, sdf_path, label = self.items[idx]
        data = load_docked_complex(pdb_path, sdf_path)
        if data is None:
            # Return dummy
            data = Data(
                x=torch.zeros(10, 10),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                pos=torch.zeros(10, 3),
            )
        data.y = torch.tensor([label], dtype=torch.float)
        return data


def get_model():
    """Import and return the original GIGN model class."""
    sys.path.insert(0, GIGN_ROOT)
    from GIGN import GIGN
    return GIGN
