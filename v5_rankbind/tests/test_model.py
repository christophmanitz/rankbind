"""Smoke tests for RankBind model shapes and gradient flow."""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def _default_config() -> dict:
    return {
        "model": {
            "head": "bilinear",
            "d_lig": 64,
            "d_prot": 64,
            "lig_input_dim": 384,
            "prot_input_dim": 1280,
            "dropout": 0.0,
        }
    }


def test_bilinear_pairwise_shape():
    from v5_rankbind.model import RankBind
    m = RankBind(_default_config())
    lig = torch.randn(7, 384); prot = torch.randn(7, 1280)
    out = m.score_pairs(lig, prot)
    assert out.shape == (7,), out.shape
    print("  ok: bilinear pairwise shape")


def test_bilinear_triplet_shapes():
    from v5_rankbind.model import RankBind
    m = RankBind(_default_config())
    lig = torch.randn(5, 384); pos = torch.randn(5, 1280); neg = torch.randn(5, 3, 1280)
    p, n = m.score_triplet(lig, pos, neg)
    assert p.shape == (5,); assert n.shape == (5, 3)
    print("  ok: bilinear triplet shapes")


def test_mlp_concat_shapes():
    from v5_rankbind.model import RankBind
    cfg = _default_config(); cfg["model"]["head"] = "mlp_concat"
    m = RankBind(cfg)
    lig = torch.randn(4, 384); prot = torch.randn(4, 1280)
    assert m.score_pairs(lig, prot).shape == (4,)
    print("  ok: mlp_concat pairwise shape")


def test_backprop_flows():
    from v5_rankbind.model import RankBind
    m = RankBind(_default_config())
    lig = torch.randn(3, 384); pos = torch.randn(3, 1280); neg = torch.randn(3, 2, 1280)
    p, n = m.score_triplet(lig, pos, neg)
    loss = (n - p.unsqueeze(1)).relu().mean()
    loss.backward()
    bad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not bad, f"Missing grads in: {bad}"
    print("  ok: all trainable params received gradient")


def test_param_counts_are_small():
    from v5_rankbind.model import RankBind
    cfg = {
        "model": {"head": "bilinear", "d_lig": 256, "d_prot": 256,
                  "lig_input_dim": 384, "prot_input_dim": 1280, "dropout": 0.0}
    }
    m = RankBind(cfg)
    stats = m.count_parameters()
    # Budget guardrail: keep total trainable under 1M so "more capacity"
    # is never the explanation in the paper.
    assert stats["n_parameters_trainable"] < 1_000_000, stats
    print(f"  ok: trainable params = {stats['n_parameters_trainable']:,}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    test_bilinear_pairwise_shape()
    test_bilinear_triplet_shapes()
    test_mlp_concat_shapes()
    test_backprop_flows()
    test_param_counts_are_small()
    print("All model tests passed.")
