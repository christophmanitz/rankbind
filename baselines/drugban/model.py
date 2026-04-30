"""
DrugBAN — Bilinear Attention Network for drug-target interaction prediction.

Re-implementation of:
    Bai, P., Miljkovic, F., John, B., & Lu, H. (2023).
    Interpretable bilinear attention network with domain adaptation
    improves drug–target prediction. Nature Machine Intelligence, 5, 126–136.

Architecture:
    Ligand encoder:  GCN on mol_graph (3 layers, 128-d) → global mean pool → 128-d
    Protein encoder: 1D-CNN on AA sequence (3 kernels k=3,5,7, embed_dim=128) → global max pool → 128-d
    BAN:             Bilinear Attention Network (K=8 heads) → 128-d interaction vector
    MLP head:        128 → 64 → 1 → sigmoid

Adapted for binary binding classification on the BRENDA hydrolase dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class BANLayer(nn.Module):
    """Bilinear Attention Network layer.

    Projects drug and target vectors into K heads of dimension D each,
    computes bilinear attention, and pools to a fixed-size vector.
    """

    def __init__(self, input_dim: int = 128, K: int = 8, D: int = 16):
        super().__init__()
        self.K = K
        self.D = D
        self.proj_d = nn.Linear(input_dim, K * D)
        self.proj_t = nn.Linear(input_dim, K * D)
        self.pool = nn.Linear(K, input_dim)

    def forward(self, v_d: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        v_d : Tensor [B, input_dim]  drug embedding
        v_t : Tensor [B, input_dim]  target embedding

        Returns
        -------
        Tensor [B, input_dim]  interaction vector
        """
        B = v_d.size(0)
        # Project to K heads
        h_d = self.proj_d(v_d).view(B, self.K, self.D)  # [B, K, D]
        h_t = self.proj_t(v_t).view(B, self.K, self.D)  # [B, K, D]

        # Bilinear attention: [B, K, K]
        attn = torch.bmm(h_d, h_t.transpose(1, 2)) / (self.D ** 0.5)
        attn = F.softmax(attn.view(B, -1), dim=-1).view(B, self.K, self.K)

        # Weighted sum over target heads for each drug head → [B, K]
        pooled = attn.sum(dim=-1)  # [B, K]

        # Map back to interaction space
        out = self.pool(pooled)  # [B, input_dim]
        return out


class DrugBAN(nn.Module):
    """DrugBAN for binary protein-ligand binding classification.

    Parameters
    ----------
    mol_in_dim : int
        Ligand atom feature dimension (default 25 for the BRENDA dataset).
    prot_vocab_size : int
        Protein token vocabulary size (including PAD at 0).
    embed_dim : int
        Embedding / hidden dimension.
    K : int
        Number of BAN heads.
    """

    def __init__(
        self,
        mol_in_dim: int = 25,
        prot_vocab_size: int = 27,
        embed_dim: int = 128,
        K: int = 8,
    ):
        super().__init__()

        # --- Ligand GCN encoder ---
        self.mol_conv1 = GCNConv(mol_in_dim, embed_dim)
        self.mol_conv2 = GCNConv(embed_dim, embed_dim)
        self.mol_conv3 = GCNConv(embed_dim, embed_dim)

        # --- Protein 1D-CNN encoder ---
        self.prot_embed = nn.Embedding(prot_vocab_size, embed_dim, padding_idx=0)
        self.prot_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        ])

        # --- BAN ---
        self.ban = BANLayer(input_dim=embed_dim, K=K, D=embed_dim // K)

        # --- MLP head ---
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.1)

    def _encode_mol(self, x, edge_index, batch):
        """GCN encoder for ligand molecular graph → [B, embed_dim]."""
        h = F.relu(self.mol_conv1(x, edge_index))
        h = F.relu(self.mol_conv2(h, edge_index))
        h = F.relu(self.mol_conv3(h, edge_index))
        h = global_mean_pool(h, batch)  # [B, embed_dim]
        return h

    def _encode_prot(self, prot_seq_ids):
        """1D-CNN encoder for protein sequence → [B, embed_dim]."""
        x = self.prot_embed(prot_seq_ids)   # [B, L, D]
        x = x.transpose(1, 2)               # [B, D, L]
        conv_outs = []
        for conv in self.prot_convs:
            h = F.relu(conv(x))
            h = h.max(dim=-1).values  # [B, D]
            conv_outs.append(h)
        out = torch.stack(conv_outs, dim=0).max(dim=0).values  # [B, D]
        return out

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
        v_d = self._encode_mol(mol_batch.x.float(), mol_batch.edge_index, mol_batch.batch)
        v_t = self._encode_prot(prot_seq_ids)

        interaction = self.ban(v_d, v_t)  # [B, embed_dim]

        h = self.drop(F.relu(self.fc1(interaction)))
        bind_prob = torch.sigmoid(self.fc2(h)).squeeze(-1)
        return bind_prob
