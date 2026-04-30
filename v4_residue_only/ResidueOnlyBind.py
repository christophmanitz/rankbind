"""
ResidueOnlyBind.py — Residue-level protein-ligand binding model.

Architecture:
  Stage 1  ProteinGraphTransformer  (residue-level, 33-dim → 128, 4 layers)
  Stage 2  LigandGraphTransformer   (ligand atom-level, 25-dim → 128, 4 layers)
  Stage 3  AttentionPool (residues) + AttentionPool (ligand atoms)
  Stage 4  Bilinear fusion → bind_head (classification) + affinity_head (regression)
  Aux      EC classification head
           Contrastive projections (z_prot, z_lig, z_prot_res)

Differences from HierAtomBind:
  - No Gumbel-topK active site selector
  - No atom-level protein GNN
  - No subgraph extraction
  - No cross-attention between protein atoms and ligand atoms
  - Protein is encoded entirely at residue level

Output tuple is API-compatible with HierAtomBind / compute_total_loss:
  (bind_logit, affinity, z_prot, z_lig, z_prot_res, attn_entropy, sel_scores, ec_logit)
  attn_entropy → 0.0  (no cross-attention)
  sel_scores   → zeros [B, N_res]  (no site selector)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from HierAtomBind import (
    FeedForward,
    AttentionPool,
    ProteinGraphTransformer,
    LigandGraphTransformer,
)


class ResidueOnlyBind(nn.Module):
    """
    Residue-only protein-ligand binding predictor.

    Parameters
    ──────────
    protein_node_dim  : residue node feature dim (33)
    protein_edge_dim  : residue edge feature dim (7)
    ligand_node_dim   : ligand atom feature dim (25)
    ligand_edge_dim   : ligand bond feature dim (7)
    hidden_dim        : internal width (128)
    prot_layers       : residue GNN depth (4)
    lig_layers        : ligand GNN depth (4)
    num_heads         : attention heads (8)
    proj_dim          : contrastive projection dim (64)
    num_ec_classes    : EC auxiliary head output dim (10)
    dropout           : dropout probability (0.1)
    """

    def __init__(
        self,
        protein_node_dim=33,
        protein_edge_dim=7,
        ligand_node_dim=25,
        ligand_edge_dim=7,
        hidden_dim=128,
        prot_layers=4,
        lig_layers=4,
        num_heads=8,
        proj_dim=64,
        num_ec_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Stage 1: Residue-level protein GNN
        self.residue_gnn = ProteinGraphTransformer(
            node_dim=protein_node_dim,
            edge_dim=protein_edge_dim,
            hidden_dim=hidden_dim,
            num_layers=prot_layers,
            heads=num_heads,
            dropout=dropout,
        )

        # Stage 2: Ligand atom GNN
        self.ligand_gnn = LigandGraphTransformer(
            node_dim=ligand_node_dim,
            edge_dim=ligand_edge_dim,
            hidden_dim=hidden_dim,
            num_layers=lig_layers,
            heads=num_heads,
            dropout=dropout,
        )

        # Attention pooling (protein residues and ligand atoms)
        self.res_pool = AttentionPool(hidden_dim)
        self.lig_pool = AttentionPool(hidden_dim)

        # Bilinear fusion: [interaction || prot || lig] → 3H → 2H → H
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bind_head     = nn.Linear(hidden_dim, 1)
        self.affinity_head = nn.Linear(hidden_dim, 1)

        # Contrastive projections (H → H → proj_dim, L2-normalised in forward)
        def _proj():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, proj_dim),
            )

        self.prot_proj     = _proj()   # residue-pool → z_prot
        self.lig_proj      = _proj()   # ligand-pool  → z_lig
        self.prot_res_proj = _proj()   # residue-pool → z_prot_res (separate weights)

        # Auxiliary EC classification head
        self.ec_head = nn.Linear(hidden_dim, num_ec_classes)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, ligand_batch, protein_batch, training=None):
        """
        ligand_batch  : PyG Batch  (molecular atom graphs, x dim 25)
        protein_batch : PyG Batch  (residue graphs, x dim 33)

        Returns (API-compatible with HierAtomBind):
          bind_logit   : [B]
          affinity     : [B]
          z_prot       : [B, proj_dim]
          z_lig        : [B, proj_dim]
          z_prot_res   : [B, proj_dim]
          attn_entropy : scalar 0.0
          sel_scores   : [B, N_res] zeros
          ec_logit     : [B, num_ec_classes]
        """
        # ── Sanitize inputs ───────────────────────────────────────────────────
        if torch.isinf(ligand_batch.x).any() or torch.isnan(ligand_batch.x).any():
            ligand_batch.x = torch.nan_to_num(
                ligand_batch.x, nan=0.0, posinf=6.0, neginf=-6.0
            )
        if torch.isinf(protein_batch.x).any() or torch.isnan(protein_batch.x).any():
            protein_batch.x = torch.nan_to_num(
                protein_batch.x, nan=0.0, posinf=1.0, neginf=-1.0
            )

        # ── Stage 1: Residue GNN ──────────────────────────────────────────────
        residue_embs, res_pad = self.residue_gnn(protein_batch)  # [B, N_res, H]
        B, N_res, H = residue_embs.shape

        # ── Stage 2: Ligand GNN ───────────────────────────────────────────────
        lig_emb, lig_pad = self.ligand_gnn(ligand_batch)         # [B, N_lig, H]

        # ── Pool ──────────────────────────────────────────────────────────────
        prot_sum = self.res_pool(residue_embs, res_pad)   # [B, H]
        lig_sum  = self.lig_pool(lig_emb, lig_pad)        # [B, H]

        # ── Bilinear fusion ───────────────────────────────────────────────────
        interaction = prot_sum * lig_sum                  # [B, H]
        fused = self.fusion_mlp(
            torch.cat([interaction, prot_sum, lig_sum], dim=-1)   # [B, 3H]
        )                                                          # [B, H]
        bind_logit = self.bind_head(fused).squeeze(-1)            # [B]
        affinity   = self.affinity_head(fused).squeeze(-1)        # [B]

        # ── Contrastive projections ───────────────────────────────────────────
        z_prot     = F.normalize(self.prot_proj(prot_sum),     dim=-1)
        z_lig      = F.normalize(self.lig_proj(lig_sum),       dim=-1)
        z_prot_res = F.normalize(self.prot_res_proj(prot_sum), dim=-1)

        # ── EC auxiliary head ─────────────────────────────────────────────────
        ec_logit = self.ec_head(prot_sum)                         # [B, num_ec]

        # ── Dummy tensors for loss compatibility ──────────────────────────────
        attn_entropy = torch.tensor(0.0, device=residue_embs.device)
        sel_scores   = torch.zeros(B, N_res, device=residue_embs.device)

        return (bind_logit, affinity, z_prot, z_lig, z_prot_res,
                attn_entropy, sel_scores, ec_logit)
