"""
GraphDTA — GNN on ligand molecular graph + CNN on protein sequence for
binary binding prediction.

Re-implementation of:
    Nguyen, T., Le, H., Quinn, T. P., Nguyen, T., Le, T. D., & Venkatesh, S.
    (2020). GraphDTA: Predicting drug–target binding affinity with graph
    neural networks. Bioinformatics, 37(8), 1140–1147.

Architecture overview:
    Protein encoder:  AA embedding → 3× Conv1D (k=3,5,7) + ReLU → GlobalMaxPool → 128-d
    Ligand encoder:   25-d atom features → Linear(25,128) → 3× GCNConv(128,128) + ReLU + Dropout
                      → global_mean_pool → 128-d
    Fusion MLP:       concat → 256 → ReLU → Dropout → 128 → ReLU → Dropout → 1 → sigmoid

Adapted to binary binding classification for the RankBind attractor bias
analysis.  The ligand encoder operates on PyG molecular graphs (from the
processed data_*.pt files), while the protein encoder uses integer-encoded
amino-acid sequences identical to DeepDTA.
"""

import string

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch


# ---------------------------------------------------------------------------
# Tokenizer (protein only — ligand uses graph features directly)
# ---------------------------------------------------------------------------

class ProteinTokenizer:
    """Map amino-acid sequence strings to fixed-length integer tensors.

    Vocabulary: 20 standard AAs + B, J, O, U, X, Z (ambiguous / rare) + PAD.
    Index 0 is reserved for padding.

    Parameters
    ----------
    max_len : int
        Sequences longer than *max_len* are truncated; shorter ones are
        right-padded with 0.
    """

    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    EXTRA = "BJOUXZ"
    VOCAB = AMINO_ACIDS + EXTRA  # 26 characters

    def __init__(self, max_len: int = 1000) -> None:
        self.max_len = max_len
        self._char2idx = {c: i + 1 for i, c in enumerate(self.VOCAB)}
        self.vocab_size = len(self.VOCAB) + 1  # +1 for PAD at index 0

    def __call__(self, sequence: str) -> torch.Tensor:
        """Encode a single protein sequence string → LongTensor [max_len]."""
        ids = [self._char2idx.get(c, self._char2idx.get("X", 0))
               for c in sequence.upper()[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def batch(self, sequences: list[str]) -> torch.Tensor:
        """Encode a list of sequences → LongTensor [B, max_len]."""
        return torch.stack([self(s) for s in sequences])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphDTA(nn.Module):
    """GraphDTA for binary protein-ligand binding classification.

    The ligand is represented as a molecular graph processed by GCN layers,
    while the protein is represented as an amino-acid character sequence
    processed by multi-kernel 1D convolutions (same as DeepDTA).

    Parameters
    ----------
    prot_vocab_size : int
        Number of tokens in the protein vocabulary (including PAD at 0).
    embed_dim : int
        Embedding / hidden dimension for both encoders.
    ligand_node_dim : int
        Dimensionality of atom-level node features in the molecular graph.
        Matches the mol_graph.x feature width from the processed dataset
        (default 25).
    hidden_dim : int
        Hidden dimension of the fusion MLP.
    """

    def __init__(
        self,
        prot_vocab_size: int = 26,
        embed_dim: int = 128,
        ligand_node_dim: int = 25,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # --- Protein encoder (CNN on AA sequence) ---
        self.prot_embed = nn.Embedding(prot_vocab_size, embed_dim, padding_idx=0)
        self.prot_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        ])

        # --- Ligand encoder (GCN on molecular graph) ---
        self.lig_lin = nn.Linear(ligand_node_dim, embed_dim)
        self.lig_convs = nn.ModuleList([
            GCNConv(embed_dim, embed_dim) for _ in range(3)
        ])
        self.lig_drop = nn.Dropout(0.1)

        # --- Fusion MLP ---
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.out = nn.Linear(embed_dim, 1)
        self.drop = nn.Dropout(0.1)

    # ----- Encoder helpers -----

    def _encode_protein(self, prot_seq_ids: torch.Tensor) -> torch.Tensor:
        """Encode protein sequences via embedding + multi-kernel Conv1D.

        Parameters
        ----------
        prot_seq_ids : Tensor [B, L_prot]

        Returns
        -------
        Tensor [B, embed_dim]
        """
        x = self.prot_embed(prot_seq_ids)           # [B, L, D]
        x = x.transpose(1, 2)                       # [B, D, L]
        conv_outs = []
        for conv in self.prot_convs:
            h = F.relu(conv(x))                      # [B, D, L']
            h = h.max(dim=-1).values                 # [B, D]
            conv_outs.append(h)
        out = torch.stack(conv_outs, dim=0).max(dim=0).values  # [B, D]
        return out

    def _encode_ligand(self, lig_batch: Batch) -> torch.Tensor:
        """Encode molecular graphs via GCN layers + global mean pooling.

        Parameters
        ----------
        lig_batch : torch_geometric.data.Batch
            Batched molecular graphs.  Required attributes:
            - ``x``: node features [N_total, ligand_node_dim]
            - ``edge_index``: COO edge indices [2, E_total]
            - ``batch``: graph membership vector [N_total]

        Returns
        -------
        Tensor [B, embed_dim]
        """
        x = F.relu(self.lig_lin(lig_batch.x))       # [N, D]
        edge_index = lig_batch.edge_index

        for gcn in self.lig_convs:
            x = gcn(x, edge_index)
            x = F.relu(x)
            x = self.lig_drop(x)

        # Global mean pool over nodes → one vector per graph
        out = global_mean_pool(x, lig_batch.batch)   # [B, D]
        return out

    # ----- Forward -----

    def forward(
        self,
        prot_seq_ids: torch.Tensor,
        lig_batch: Batch,
    ) -> torch.Tensor:
        """Predict binding probability.

        Parameters
        ----------
        prot_seq_ids : Tensor [B, L_prot]
            Integer-encoded protein sequences (from ProteinTokenizer).
        lig_batch : torch_geometric.data.Batch
            Batched molecular graphs with node features ``x``,
            ``edge_index``, and ``batch`` attributes.

        Returns
        -------
        bind_prob : Tensor [B]
            Predicted probability of binding (after sigmoid).
        """
        prot_vec = self._encode_protein(prot_seq_ids)    # [B, D]
        lig_vec = self._encode_ligand(lig_batch)          # [B, D]

        h = torch.cat([prot_vec, lig_vec], dim=-1)        # [B, 2D]
        h = self.drop(F.relu(self.fc1(h)))                # [B, hidden_dim]
        h = self.drop(F.relu(self.fc2(h)))                # [B, D]
        bind_prob = torch.sigmoid(self.out(h)).squeeze(-1)  # [B]
        return bind_prob
