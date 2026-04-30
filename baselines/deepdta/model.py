"""
DeepDTA — Purely sequence-based CNN model for binary binding prediction.

Re-implementation of:
    Öztürk, H., Özgür, A., & Ozkirimli, E. (2018).
    DeepDTA: deep drug–target binding affinity prediction.
    Bioinformatics, 34(17), i821–i829.

Architecture overview:
    Protein encoder:  AA character embedding → 3× Conv1D (k=3,5,7) + ReLU → GlobalMaxPool → 128-d
    Ligand encoder:   SMILES character embedding → 3× Conv1D (k=4,6,8) + ReLU → GlobalMaxPool → 128-d
    Fusion MLP:       concat → 256 → ReLU → Dropout → 128 → ReLU → Dropout → 1 → sigmoid

Adapted to binary binding classification (sigmoid output) for the RankBind
attractor bias analysis.  The original DeepDTA predicts continuous affinity
(Kd); here we predict P(binder) since the dataset mixes true binders with
generated decoys (label = 0 for decoy, >0 for binder).
"""

import string
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class ProteinTokenizer:
    """Map amino-acid sequence strings to fixed-length integer tensors.

    Vocabulary: 20 standard AAs + B, J, O, U, X (ambiguous / rare) + PAD token.
    Index 0 is reserved for padding.

    Parameters
    ----------
    max_len : int
        Sequences longer than *max_len* are truncated; shorter ones are
        right-padded with 0.
    """

    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard
    EXTRA = "BJOUXZ"  # ambiguous / non-standard codes
    VOCAB = AMINO_ACIDS + EXTRA  # 26 characters

    def __init__(self, max_len: int = 1000) -> None:
        self.max_len = max_len
        # char → index (1-based; 0 = PAD)
        self._char2idx = {c: i + 1 for i, c in enumerate(self.VOCAB)}
        self.vocab_size = len(self.VOCAB) + 1  # +1 for PAD at index 0

    def __call__(self, sequence: str) -> torch.Tensor:
        """Encode a single protein sequence string → LongTensor [max_len]."""
        ids = [self._char2idx.get(c, self._char2idx.get("X", 0))
               for c in sequence.upper()[:self.max_len]]
        # Pad to max_len
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def batch(self, sequences: list[str]) -> torch.Tensor:
        """Encode a list of sequences → LongTensor [B, max_len]."""
        return torch.stack([self(s) for s in sequences])


class LigandTokenizer:
    """Map SMILES strings to fixed-length integer tensors.

    The vocabulary covers all printable ASCII characters commonly found in
    SMILES notation (letters, digits, brackets, parentheses, ``=``, ``#``,
    ``@``, ``+``, ``-``, ``/``, ``\\``, ``.``, ``%``).  Index 0 is reserved
    for padding.

    Parameters
    ----------
    max_len : int
        SMILES longer than *max_len* are truncated; shorter ones are
        right-padded with 0.
    """

    # Comprehensive SMILES character vocabulary (~65 chars)
    VOCAB = (string.ascii_letters + string.digits +
             "()[]=#@+-./%\\")

    def __init__(self, max_len: int = 100) -> None:
        self.max_len = max_len
        self._char2idx = {c: i + 1 for i, c in enumerate(self.VOCAB)}
        self.vocab_size = len(self.VOCAB) + 1  # +1 for PAD

    def __call__(self, smiles: str) -> torch.Tensor:
        """Encode a single SMILES string → LongTensor [max_len]."""
        ids = [self._char2idx.get(c, 0) for c in smiles[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def batch(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode a list of SMILES → LongTensor [B, max_len]."""
        return torch.stack([self(s) for s in smiles_list])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeepDTA(nn.Module):
    """DeepDTA for binary protein-ligand binding classification.

    Parameters
    ----------
    prot_vocab_size : int
        Number of tokens in the protein vocabulary (including PAD at 0).
    lig_vocab_size : int
        Number of tokens in the ligand/SMILES vocabulary (including PAD).
    embed_dim : int
        Embedding dimension for both protein and ligand encoders.
    hidden_dim : int
        Hidden dimension of the fusion MLP (first linear layer output).
    """

    def __init__(
        self,
        prot_vocab_size: int = 27,
        lig_vocab_size: int = 76,
        embed_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # --- Protein encoder ---
        self.prot_embed = nn.Embedding(prot_vocab_size, embed_dim, padding_idx=0)
        self.prot_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        ])

        # --- Ligand encoder ---
        self.lig_embed = nn.Embedding(lig_vocab_size, embed_dim, padding_idx=0)
        self.lig_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=k // 2)
            for k in (4, 6, 8)
        ])

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
        x = self.prot_embed(prot_seq_ids)          # [B, L, D]
        x = x.transpose(1, 2)                      # [B, D, L]
        conv_outs = []
        for conv in self.prot_convs:
            h = F.relu(conv(x))                     # [B, D, L']
            h = h.max(dim=-1).values                # [B, D]  (global max pool)
            conv_outs.append(h)
        # Element-wise max across the three kernel widths
        out = torch.stack(conv_outs, dim=0).max(dim=0).values  # [B, D]
        return out

    def _encode_ligand(self, lig_seq_ids: torch.Tensor) -> torch.Tensor:
        """Encode SMILES via embedding + multi-kernel Conv1D.

        Parameters
        ----------
        lig_seq_ids : Tensor [B, L_lig]

        Returns
        -------
        Tensor [B, embed_dim]
        """
        x = self.lig_embed(lig_seq_ids)             # [B, L, D]
        x = x.transpose(1, 2)                       # [B, D, L]
        conv_outs = []
        for conv in self.lig_convs:
            h = F.relu(conv(x))
            h = h.max(dim=-1).values
            conv_outs.append(h)
        out = torch.stack(conv_outs, dim=0).max(dim=0).values
        return out

    # ----- Forward -----

    def forward(
        self,
        prot_seq_ids: torch.Tensor,
        lig_seq_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict binding probability.

        Parameters
        ----------
        prot_seq_ids : Tensor [B, L_prot]
            Integer-encoded protein sequences (from ProteinTokenizer).
        lig_seq_ids : Tensor [B, L_lig]
            Integer-encoded SMILES strings (from LigandTokenizer).

        Returns
        -------
        bind_prob : Tensor [B]
            Predicted probability of binding (after sigmoid).
        """
        prot_vec = self._encode_protein(prot_seq_ids)   # [B, D]
        lig_vec = self._encode_ligand(lig_seq_ids)       # [B, D]

        h = torch.cat([prot_vec, lig_vec], dim=-1)       # [B, 2D]
        h = self.drop(F.relu(self.fc1(h)))               # [B, hidden_dim]
        h = self.drop(F.relu(self.fc2(h)))               # [B, D]
        bind_prob = torch.sigmoid(self.out(h)).squeeze(-1)  # [B]
        return bind_prob
