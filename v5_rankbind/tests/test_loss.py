"""Smoke tests for margin / BCE loss + gradient flow."""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def test_margin_loss_value():
    from v5_rankbind.loss import margin_loss
    pos = torch.tensor([2.0, 1.0])
    neg = torch.tensor([[0.0, 3.0], [0.5, 0.5]])
    # Batch 0: max(0, 1 - 2 + 0) = 0; max(0, 1 - 2 + 3) = 2 → mean 1.0
    # Batch 1: max(0, 1 - 1 + 0.5) = 0.5; max(0, 1 - 1 + 0.5) = 0.5 → mean 0.5
    # Overall mean over (B, k) = (1.0 * 2 + 0.5 * 2) / 4 = 0.75
    loss = margin_loss(pos, neg, margin=1.0)
    assert abs(loss.item() - 0.75) < 1e-6, loss.item()
    print("  ok: margin loss numerical check")


def test_loss_wrapper_margin():
    from v5_rankbind.loss import RankBindLoss
    loss_fn = RankBindLoss({"type": "margin", "margin": 1.0, "bce_aux_weight": 0.1})
    pos = torch.tensor([2.0, 0.5], requires_grad=True)
    neg = torch.tensor([[0.0, 3.0], [1.0, 2.0]], requires_grad=True)
    loss, parts = loss_fn.compute_margin(pos, neg)
    loss.backward()
    assert pos.grad is not None and neg.grad is not None
    assert set(parts) >= {"loss_margin", "loss_bce_aux", "loss_total",
                          "margin_violation_rate"}
    print("  ok: margin wrapper + gradients")


def test_loss_wrapper_bce():
    from v5_rankbind.loss import RankBindLoss
    loss_fn = RankBindLoss({"type": "bce"})
    score = torch.tensor([1.0, -1.0], requires_grad=True)
    label = torch.tensor([1, 0])
    loss, parts = loss_fn.compute_bce(score, label)
    loss.backward()
    assert score.grad is not None
    assert set(parts) == {"loss_bce", "loss_total"}
    print("  ok: bce wrapper + gradients")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    test_margin_loss_value()
    test_loss_wrapper_margin()
    test_loss_wrapper_bce()
    print("All loss tests passed.")
