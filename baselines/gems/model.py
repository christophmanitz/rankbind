"""
GEMS — Graph-Enhanced Molecular Screening.

Re-implementation inspired by:
    GEMS (camlab-ethz, 2024): protein language model + ligand GCN.

Simplified architecture (BiLSTM instead of ESM2 for offline use):
    Protein encoder: Embedding(27, 256) → BiLSTM(256, 128, 2 layers) → last hidden → 128-d
    Ligand encoder:  GCN on mol_graph (3 layers, 256-d) → global mean pool → 128-d
    Fusion:          concat → Linear(256, 128) → ReLU → Dropout(0.2) → Linear(128, 1) → sigmoid

Adapted for binary binding classification on the BRENDA hydrolase dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GEMS(nn.Module):
    """GEMS for binary protein-ligand binding classification.

    Parameters
    ----------
    mol_in_dim : int
        Ligand atom feature dimension (25).
    prot_vocab_size : int
        Protein token vocabulary size (including PAD at 0).
    prot_embed_dim : int
        Protein embedding dimension.
    mol_hidden_dim : int
        Ligand GCN hidden dimension.
    out_dim : int
        Output dimension of each encoder branch.
    lstm_layers : int
        Number of BiLSTM layers.
    """

    def __init__(
        self,
        mol_in_dim: int = 25,
        prot_vocab_size: int = 27,
        prot_embed_dim: int = 256,
        mol_hidden_dim: int = 256,
        out_dim: int = 128,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.out_dim = out_dim

        # --- Protein BiLSTM encoder ---
        self.prot_embed = nn.Embedding(prot_vocab_size, prot_embed_dim, padding_idx=0)
        self.prot_lstm = nn.LSTM(
            input_size=prot_embed_dim,
            hidden_size=out_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        # BiLSTM produces 2*out_dim; project back to out_dim
        self.prot_proj = nn.Linear(out_dim * 2, out_dim)

        # --- Ligand GCN encoder ---
        self.mol_conv1 = GCNConv(mol_in_dim, mol_hidden_dim)
        self.mol_conv2 = GCNConv(mol_hidden_dim, mol_hidden_dim)
        self.mol_conv3 = GCNConv(mol_hidden_dim, out_dim)

        # --- Fusion MLP ---
        self.fc1 = nn.Linear(out_dim * 2, out_dim)
        self.fc2 = nn.Linear(out_dim, 1)
        self.drop = nn.Dropout(0.2)

    def _encode_prot(self, prot_seq_ids):
        """Encode protein sequence via BiLSTM → [B, out_dim]."""
        x = self.prot_embed(prot_seq_ids)  # [B, L, embed_dim]
        output, (h_n, _) = self.prot_lstm(x)
        # h_n: [num_layers*2, B, out_dim] — take last layer, both directions
        h_fwd = h_n[-2]  # [B, out_dim]
        h_bwd = h_n[-1]  # [B, out_dim]
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, 2*out_dim]
        h = self.prot_proj(h)  # [B, out_dim]
        return h

    def _encode_mol(self, x, edge_index, batch):
        """GCN encoder for ligand molecular graph → [B, out_dim]."""
        h = F.relu(self.mol_conv1(x, edge_index))
        h = F.relu(self.mol_conv2(h, edge_index))
        h = F.relu(self.mol_conv3(h, edge_index))
        h = global_mean_pool(h, batch)  # [B, out_dim]
        return h

    def forward(self, prot_seq_ids, mol_batch):
        """
        Parameters
        ----------
        prot_seq_ids : Tensor [B, L_prot]
        mol_batch    : PyG Batch with .x, .edge_index, .batch

        Returns
        -------
        bind_prob : Tensor [B]
        """
        prot_vec = self._encode_prot(prot_seq_ids)  # [B, out_dim]
        mol_vec  = self._encode_mol(
            mol_batch.x.float(), mol_batch.edge_index, mol_batch.batch
        )  # [B, out_dim]

        h = torch.cat([prot_vec, mol_vec], dim=-1)  # [B, 2*out_dim]
        h = self.drop(F.relu(self.fc1(h)))
        bind_prob = torch.sigmoid(self.fc2(h)).squeeze(-1)
        return bind_prob
