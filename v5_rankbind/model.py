"""
v5_rankbind/model.py — RankBind model.

  score(L, P) = f(L)^T M g(P) + b     (bilinear head, main model)
  score(L, P) = MLP([f(L); g(P)])     (ablation head)

Encoders are linear projections over pre-computed ChemBERTa / ESM2
embeddings. The encoders do *not* touch the PLM weights: those are frozen
on disk and consumed as fixed-vector inputs, so only the projections and
bilinear core receive gradients.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LigandProjector(nn.Module):
    """ChemBERTa mean-pool (384-d) → d_lig."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProteinProjector(nn.Module):
    """ESM2 mean-pool (1280-d) → d_prot."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidueAttentionPool(nn.Module):
    """Single-head learned-query pool over ESM2 per-residue embeddings.

    Input:
        residues  [B, L, D]   per-residue ESM2 embeddings, padded to L = max_len in batch
        mask      [B, L]      bool, True for real residues, False for padding
    Output:
        pooled    [B, D]      attention-weighted mean over residues
        weights   [B, L]      attention weights (zeroed on padding) — for logging
    """

    def __init__(self, in_dim: int, n_heads: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.n_heads = n_heads
        # Learned query (one per head). Tiny — ~5KB at D=1280.
        self.q = nn.Parameter(torch.zeros(n_heads, in_dim))
        nn.init.normal_(self.q, std=in_dim ** -0.5)
        # LayerNorm over residue features before scoring keeps the dot-product
        # well-scaled across proteins of very different mean activations.
        self.norm = nn.LayerNorm(in_dim)

    def forward(
        self,
        residues: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # residues: [B, L, D]; mask: [B, L]
        x = self.norm(residues)
        # Single-head version: scores [B, L] = (x · q) / sqrt(D)
        # n_heads kept as scaffolding but we only use head 0 here.
        q = self.q[0]                                          # [D]
        scores = (x * q).sum(dim=-1) / (self.in_dim ** 0.5)     # [B, L]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)                 # [B, L]
        # Defensive: rows where mask was all False produce NaN; replace with 0.
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.einsum("bl,bld->bd", weights, residues)  # [B, D]
        return pooled, weights


class BilinearHead(nn.Module):
    """score = f(L)^T M g(P) + b.

    M is parameterised as a low-rank + diagonal factorisation to keep the
    parameter budget tiny:
        M = U V^T + diag(d)
    with U, V ∈ R^{d×r} and d ∈ R^d. For the default config (d_lig=d_prot=256,
    r=32) that is 2 * 256 * 32 + 256 ≈ 16,640 parameters.
    """

    def __init__(self, d_lig: int, d_prot: int, rank: int = 32):
        super().__init__()
        if d_lig != d_prot:
            raise ValueError("BilinearHead assumes d_lig == d_prot for diag term.")
        self.U = nn.Parameter(torch.empty(d_lig, rank))
        self.V = nn.Parameter(torch.empty(d_prot, rank))
        self.d = nn.Parameter(torch.zeros(d_lig))
        self.b = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(self, fL: torch.Tensor, gP: torch.Tensor) -> torch.Tensor:
        # fL: [..., d_lig], gP: [..., d_prot]
        low = (fL @ self.U) * (gP @ self.V)
        lr = low.sum(dim=-1)
        diag = (fL * self.d * gP).sum(dim=-1)
        return lr + diag + self.b


class MLPConcatHead(nn.Module):
    """Ablation head: concat(f(L), g(P)) → 2-layer MLP → scalar.

    Importantly *not* gated by an interaction term — this is the head type
    that lets a model pick up protein-only shortcuts.
    """

    def __init__(self, d_lig: int, d_prot: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_lig + d_prot, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, fL: torch.Tensor, gP: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fL, gP], dim=-1)
        return self.net(x).squeeze(-1)


class RankBind(nn.Module):
    """End-to-end model: projections + head.

    Call either:
        score_pairs(lig_emb, prot_emb)   # [B], pointwise
        score_triplet(lig_emb, pos_prot, neg_prot)  # (pos_score [B], neg_score [B, k])
    """

    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.d_lig = m["d_lig"]
        self.d_prot = m["d_prot"]
        self.head_type = m["head"]
        self.protein_encoder = m.get("protein_encoder", "mean_pool")
        self.lig = LigandProjector(m["lig_input_dim"], m["d_lig"], m["dropout"])
        self.prot = ProteinProjector(m["prot_input_dim"], m["d_prot"], m["dropout"])
        if self.protein_encoder == "attn_pool":
            self.attn_pool = ResidueAttentionPool(m["prot_input_dim"], n_heads=1)
        elif self.protein_encoder == "mean_pool":
            self.attn_pool = None
        else:
            raise ValueError(
                f"Unknown protein_encoder={self.protein_encoder!r}; "
                "expected 'mean_pool' or 'attn_pool'."
            )
        if self.head_type == "bilinear":
            self.head = BilinearHead(
                m["d_lig"], m["d_prot"],
                rank=m.get("bilinear_rank", 32),
            )
        elif self.head_type == "mlp_concat":
            self.head = MLPConcatHead(
                m["d_lig"], m["d_prot"],
                hidden=m.get("mlp_hidden", 128),
                dropout=m["dropout"],
            )
        else:
            raise ValueError(f"Unknown head: {self.head_type}")

    # ── forward helpers ────────────────────────────────────────────────────

    def encode_protein(
        self,
        prot_input: torch.Tensor,
        prot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a protein input to its `d_prot`-dim representation.

        Two input shapes are accepted:
          - mean_pool mode: prot_input is `[B, prot_input_dim]` (already pooled
            on disk by `data.py:load_protein`); `prot_mask` ignored.
          - attn_pool mode: prot_input is `[B, L, prot_input_dim]` per-residue,
            with a `[B, L]` boolean mask marking real residues. Attention pool
            collapses to `[B, prot_input_dim]` first, then the projector runs.
        """
        if self.attn_pool is not None:
            if prot_input.ndim != 3 or prot_mask is None:
                raise ValueError(
                    "attn_pool encoder requires per-residue [B, L, D] input "
                    "with a [B, L] mask; received "
                    f"shape={tuple(prot_input.shape)} mask={None if prot_mask is None else tuple(prot_mask.shape)}."
                )
            pooled, _weights = self.attn_pool(prot_input, prot_mask)
            return self.prot(pooled)
        if prot_input.ndim != 2:
            raise ValueError(
                "mean_pool encoder expects [B, D] input; received "
                f"shape={tuple(prot_input.shape)}."
            )
        return self.prot(prot_input)

    def score_pairs(
        self,
        lig_emb: torch.Tensor,
        prot_emb: torch.Tensor,
        prot_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fL = self.lig(lig_emb)
        gP = self.encode_protein(prot_emb, prot_mask)
        return self.head(fL, gP)

    def score_triplet(
        self,
        lig_emb: torch.Tensor,        # [B, D_lig]
        pos_prot: torch.Tensor,       # [B, D] (mean) or [B, L_pos, D] (attn)
        neg_prot: torch.Tensor,       # [B, k, D] (mean) or [B, k, L_neg, D] (attn)
        pos_mask: torch.Tensor | None = None,  # [B, L_pos] (attn only)
        neg_mask: torch.Tensor | None = None,  # [B, k, L_neg] (attn only)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fL = self.lig(lig_emb)                        # [B, d_lig]
        gP_pos = self.encode_protein(pos_prot, pos_mask)            # [B, d_prot]

        if self.attn_pool is not None:
            B, k, L_neg, D = neg_prot.shape
            gP_neg_flat = self.encode_protein(
                neg_prot.reshape(B * k, L_neg, D),
                neg_mask.reshape(B * k, L_neg) if neg_mask is not None else None,
            )                                                       # [B*k, d_prot]
            gP_neg = gP_neg_flat.reshape(B, k, -1)
        else:
            B, k, _ = neg_prot.shape
            gP_neg = self.prot(neg_prot.reshape(B * k, -1)).reshape(B, k, -1)

        pos_score = self.head(fL, gP_pos)             # [B]
        neg_score = self.head(
            fL.unsqueeze(1).expand(-1, k, -1),
            gP_neg,
        )                                             # [B, k]
        return pos_score, neg_score

    # ── diagnostics ────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            "n_parameters_trainable": int(total_trainable),
            "n_parameters_frozen": int(total_frozen),
            "head_type": self.head_type,
            "d_lig": self.d_lig,
            "d_prot": self.d_prot,
        }
