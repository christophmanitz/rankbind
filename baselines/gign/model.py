"""
GIGN — Geometric Interaction Graph Neural Network.

Re-implementation of:
    Gao, Z., et al. (2023). Geometric Interaction Graph Neural Network
    for Predicting Protein–Ligand Binding Affinities from 3D Structures.
    The Journal of Physical Chemistry Letters, 14, 2020–2033.

Simplified architecture (no PDB parsing at runtime):
    Ligand atom GNN:    2-layer GCN on mol_graph (25-d → 128-d)
    Protein residue GNN: 2-layer GCN on prot_graph (33-d → 128-d)
    Interaction GNN:    2 rounds of bidirectional message passing via cross-edges
                        (virtual bipartite edges between mol atoms and prot residues)
    Readout:            global mean pool both → concat → MLP → sigmoid

Adapted for binary binding classification on the BRENDA hydrolase dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class CrossMessagePassing(nn.Module):
    """Bidirectional cross-graph message passing between ligand and protein.

    For each round, ligand nodes receive messages from protein nodes
    (via cross-edges) and vice versa.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        # Ligand ← Protein message
        self.msg_p2l = nn.Linear(hidden_dim, hidden_dim)
        # Protein ← Ligand message
        self.msg_l2p = nn.Linear(hidden_dim, hidden_dim)
        self.norm_l = nn.LayerNorm(hidden_dim)
        self.norm_p = nn.LayerNorm(hidden_dim)

    def forward(self, h_l, h_p, cross_edge_lp):
        """
        Parameters
        ----------
        h_l : Tensor [N_l, D]  ligand node features
        h_p : Tensor [N_p, D]  protein node features
        cross_edge_lp : Tensor [2, E_cross]  edges from ligand to protein
                        row 0 = ligand indices, row 1 = protein indices

        Returns
        -------
        h_l_new, h_p_new : updated features
        """
        lig_idx = cross_edge_lp[0]   # ligand node indices in cross edges
        prot_idx = cross_edge_lp[1]  # protein node indices in cross edges

        # Protein → Ligand messages
        msg_to_l = self.msg_p2l(h_p[prot_idx])  # [E_cross, D]
        agg_l = torch.zeros_like(h_l)
        agg_l.index_add_(0, lig_idx, msg_to_l)

        # Ligand → Protein messages
        msg_to_p = self.msg_l2p(h_l[lig_idx])  # [E_cross, D]
        agg_p = torch.zeros_like(h_p)
        agg_p.index_add_(0, prot_idx, msg_to_p)

        h_l_new = self.norm_l(F.relu(h_l + agg_l))
        h_p_new = self.norm_p(F.relu(h_p + agg_p))
        return h_l_new, h_p_new


class GIGN(nn.Module):
    """Simplified GIGN for binary protein-ligand binding classification.

    Parameters
    ----------
    mol_in_dim : int
        Ligand atom feature dimension (25).
    prot_in_dim : int
        Protein residue feature dimension (33).
    hidden_dim : int
        Hidden dimension for all GNN layers.
    max_cross_edges : int
        Maximum number of cross-edges (randomly sampled if exceeded).
    """

    def __init__(
        self,
        mol_in_dim: int = 25,
        prot_in_dim: int = 33,
        hidden_dim: int = 128,
        max_cross_edges: int = 500,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_cross_edges = max_cross_edges

        # Ligand GCN
        self.mol_conv1 = GCNConv(mol_in_dim, hidden_dim)
        self.mol_conv2 = GCNConv(hidden_dim, hidden_dim)

        # Protein GCN
        self.prot_conv1 = GCNConv(prot_in_dim, hidden_dim)
        self.prot_conv2 = GCNConv(hidden_dim, hidden_dim)

        # Cross message passing (2 rounds)
        self.cross1 = CrossMessagePassing(hidden_dim)
        self.cross2 = CrossMessagePassing(hidden_dim)

        # MLP head
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(0.1)

    def _build_cross_edges(self, n_l, n_p, device):
        """Build virtual cross-edges (complete bipartite, capped).

        Returns
        -------
        Tensor [2, E_cross] on device
        """
        total = n_l * n_p
        if total <= self.max_cross_edges:
            # Complete bipartite
            l_idx = torch.arange(n_l, device=device).repeat_interleave(n_p)
            p_idx = torch.arange(n_p, device=device).repeat(n_l)
        else:
            # Random subset
            indices = torch.randperm(total, device=device)[:self.max_cross_edges]
            l_idx = indices // n_p
            p_idx = indices % n_p
        return torch.stack([l_idx, p_idx], dim=0)

    def forward_single(self, mol_graph, prot_graph):
        """Score a single (mol_graph, prot_graph) pair.

        Parameters
        ----------
        mol_graph : PyG Data with .x, .edge_index
        prot_graph : PyG Data with .x, .edge_index

        Returns
        -------
        score : Tensor scalar
        """
        device = mol_graph.x.device

        # Encode ligand
        h_l = F.relu(self.mol_conv1(mol_graph.x.float(), mol_graph.edge_index))
        h_l = F.relu(self.mol_conv2(h_l, mol_graph.edge_index))

        # Encode protein
        h_p = F.relu(self.prot_conv1(prot_graph.x.float(), prot_graph.edge_index))
        h_p = F.relu(self.prot_conv2(h_p, prot_graph.edge_index))

        # Cross message passing
        cross_edges = self._build_cross_edges(h_l.size(0), h_p.size(0), device)
        h_l, h_p = self.cross1(h_l, h_p, cross_edges)
        h_l, h_p = self.cross2(h_l, h_p, cross_edges)

        # Global mean pool
        pool_l = h_l.mean(dim=0)  # [D]
        pool_p = h_p.mean(dim=0)  # [D]

        # Concat + MLP
        h = torch.cat([pool_l, pool_p], dim=-1)  # [2D]
        h = self.drop(F.relu(self.fc1(h)))
        score = torch.sigmoid(self.fc2(h)).squeeze(-1)
        return score

    def forward(self, mol_graphs, prot_graphs):
        """Score a batch of (mol_graph, prot_graph) pairs.

        Parameters
        ----------
        mol_graphs : list of PyG Data
        prot_graphs : list of PyG Data

        Returns
        -------
        scores : Tensor [B]
        """
        scores = torch.stack([
            self.forward_single(m, p)
            for m, p in zip(mol_graphs, prot_graphs)
        ])
        return scores
