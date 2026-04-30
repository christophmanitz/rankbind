"""
v5_rankbind/loss.py — Within-ligand margin loss + optional BCE auxiliary.

Margin loss (paper notation):

    L_margin(L, P+, {P_i-}) = 1/k sum_i max(0, m − s(L, P+) + s(L, P_i-))

The training batch delivers (pos_score [B], neg_score [B, k]) tensors.

Optional pointwise BCE is computed either over positive-anchor scores
(auxiliary) or — for the BCE-only ablation — over a flat labeled batch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def margin_loss(
    pos_score: torch.Tensor,   # [B]
    neg_score: torch.Tensor,   # [B, k]
    margin: float = 1.0,
) -> torch.Tensor:
    diff = margin - pos_score.unsqueeze(1) + neg_score   # [B, k]
    return diff.clamp(min=0.0).mean()


def bce_aux_on_triplet(
    pos_score: torch.Tensor,   # [B]
    neg_score: torch.Tensor,   # [B, k]
) -> torch.Tensor:
    """Treat positives as label=1 and negatives as label=0. BCEWithLogits."""
    pos_t = torch.ones_like(pos_score)
    neg_t = torch.zeros_like(neg_score)
    logits = torch.cat([pos_score, neg_score.reshape(-1)])
    targets = torch.cat([pos_t, neg_t.reshape(-1)])
    return F.binary_cross_entropy_with_logits(logits, targets)


def bce_pointwise(score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(score, label.float())


class RankBindLoss:
    """Config-driven loss wrapper.

    type = "margin"  → margin on triplets (+ optional bce_aux_weight * bce_aux)
    type = "bce"     → pointwise BCE on flat batches
    """

    def __init__(self, loss_cfg: dict):
        self.type = loss_cfg["type"]
        self.margin = float(loss_cfg.get("margin", 1.0))
        self.bce_aux_weight = float(loss_cfg.get("bce_aux_weight", 0.0))

    def compute_margin(
        self, pos_score: torch.Tensor, neg_score: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        m = margin_loss(pos_score, neg_score, self.margin)
        total = m
        bce = torch.tensor(0.0, device=m.device)
        if self.bce_aux_weight > 0:
            bce = bce_aux_on_triplet(pos_score, neg_score)
            total = total + self.bce_aux_weight * bce
        with torch.no_grad():
            frac_violating = (pos_score.unsqueeze(1) - neg_score < self.margin).float().mean()
            # Fraction of anchors where the positive outscores every negative.
            # This is the saturation diagnostic that motivates hard-negative
            # mining: random-negative margin-loss runs drive this to ~0.4%
            # even though pos/neg MEANS separate cleanly.
            pos_above_neg_max = (
                pos_score > neg_score.max(dim=1).values
            ).float().mean()
        return total, {
            "loss_margin":    float(m.detach()),
            "loss_bce_aux":   float(bce.detach()),
            "loss_total":     float(total.detach()),
            "margin_violation_rate": float(frac_violating),
            "pos_above_neg_max":     float(pos_above_neg_max),
        }

    def compute_bce(
        self, score: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        loss = bce_pointwise(score, label)
        return loss, {
            "loss_bce":   float(loss.detach()),
            "loss_total": float(loss.detach()),
        }
