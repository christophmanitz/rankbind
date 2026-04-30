"""
MolTrans — Transformer-based drug-target interaction prediction.

Re-implementation of:
    Huang, K., Xiao, C., Glass, L. M., & Sun, J. (2021).
    MolTrans: Molecular Interaction Transformer for drug–target interaction
    prediction. Bioinformatics, 37(6), 830–836.

Architecture:
    Protein encoder: Embedding(27, 128) → TransformerEncoder (2 layers) → CLS pooling → 128-d
    Ligand encoder:  Embedding(65, 128) → TransformerEncoder (2 layers) → CLS pooling → 128-d
    Fusion:          element-wise product → Linear(128, 64) → ReLU → Linear(64, 1) → sigmoid

Adapted for binary binding classification on the BRENDA hydrolase dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MolTrans(nn.Module):
    """MolTrans for binary protein-ligand binding classification.

    Parameters
    ----------
    prot_vocab_size : int
        Protein vocabulary size (26 AAs + PAD). CLS token added at index = prot_vocab_size.
    lig_vocab_size : int
        Ligand vocabulary size (64 chars + PAD). CLS token added at index = lig_vocab_size.
    d_model : int
        Transformer model dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of TransformerEncoder layers.
    dim_feedforward : int
        Feedforward dimension in Transformer.
    dropout : float
        Dropout rate.
    max_prot_len : int
        Maximum protein sequence length (for positional encoding).
    max_lig_len : int
        Maximum ligand sequence length (for positional encoding).
    """

    def __init__(
        self,
        prot_vocab_size: int = 27,
        lig_vocab_size: int = 76,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_prot_len: int = 1000,
        max_lig_len: int = 100,
    ):
        super().__init__()
        self.d_model = d_model

        # +1 for CLS token appended to vocab
        self.prot_embed = nn.Embedding(prot_vocab_size + 1, d_model, padding_idx=0)
        self.lig_embed  = nn.Embedding(lig_vocab_size + 1, d_model, padding_idx=0)

        # CLS token indices
        self.prot_cls_idx = prot_vocab_size
        self.lig_cls_idx  = lig_vocab_size

        # Positional encodings (learnable)
        self.prot_pos = nn.Embedding(max_prot_len + 1, d_model)  # +1 for CLS
        self.lig_pos  = nn.Embedding(max_lig_len + 1, d_model)

        # Transformer encoders
        prot_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.prot_transformer = nn.TransformerEncoder(prot_layer, num_layers=num_layers)

        lig_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.lig_transformer = nn.TransformerEncoder(lig_layer, num_layers=num_layers)

        # Fusion MLP
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 1)

    def _prepend_cls(self, seq_ids, cls_idx):
        """Prepend CLS token to sequence ids.

        Parameters
        ----------
        seq_ids : Tensor [B, L]
        cls_idx : int

        Returns
        -------
        Tensor [B, L+1] with CLS at position 0
        """
        B = seq_ids.size(0)
        cls_col = torch.full((B, 1), cls_idx, dtype=torch.long, device=seq_ids.device)
        return torch.cat([cls_col, seq_ids], dim=1)

    def _encode_prot(self, prot_seq_ids):
        """Encode protein sequence via Transformer → CLS pooling → [B, d_model]."""
        x = self._prepend_cls(prot_seq_ids, self.prot_cls_idx)  # [B, L+1]
        B, L = x.shape

        # Padding mask: True where token is PAD (index 0), but NOT for CLS
        pad_mask = (x == 0)  # [B, L]
        pad_mask[:, 0] = False  # CLS is never masked

        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.prot_embed(x) + self.prot_pos(pos_ids)  # [B, L, D]
        h = self.prot_transformer(h, src_key_padding_mask=pad_mask)
        return h[:, 0, :]  # CLS token → [B, D]

    def _encode_lig(self, lig_seq_ids):
        """Encode ligand SMILES via Transformer → CLS pooling → [B, d_model]."""
        x = self._prepend_cls(lig_seq_ids, self.lig_cls_idx)  # [B, L+1]
        B, L = x.shape

        pad_mask = (x == 0)
        pad_mask[:, 0] = False

        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.lig_embed(x) + self.lig_pos(pos_ids)
        h = self.lig_transformer(h, src_key_padding_mask=pad_mask)
        return h[:, 0, :]  # [B, D]

    def forward(self, prot_seq_ids, lig_seq_ids):
        """
        Parameters
        ----------
        prot_seq_ids : Tensor [B, L_prot]
        lig_seq_ids  : Tensor [B, L_lig]

        Returns
        -------
        bind_prob : Tensor [B]
        """
        prot_vec = self._encode_prot(prot_seq_ids)  # [B, D]
        lig_vec  = self._encode_lig(lig_seq_ids)     # [B, D]

        # Element-wise product fusion
        fused = prot_vec * lig_vec  # [B, D]
        h = F.relu(self.fc1(fused))
        bind_prob = torch.sigmoid(self.fc2(h)).squeeze(-1)
        return bind_prob
