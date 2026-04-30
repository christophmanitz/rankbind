"""Smoke-test for Stage (b) attn_pool plumbing. Not a unit test — a fast
end-to-end shape check that exercises:

  - RankBindDataset(attn_pool) returns [L, D] + len
  - collate_pointwise pads correctly + emits prot_mask
  - TripletCollator produces pos/neg residue tensors with masks
  - RankBind.encode_protein dispatches on shape
  - score_pairs and score_triplet forward + backward succeed

Run with: python -m v5_rankbind._smoke_test_attn_pool
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v5_rankbind.run_manifest import load_config
from v5_rankbind.data import build_datasets, collate_pointwise
from v5_rankbind.model import RankBind
from v5_rankbind.sampler import TripletCollator


def main() -> None:
    cfg = load_config(str(_HERE / "configs" / "abl_attn_pool.json"))
    cfg["seed"] = 42
    print(f"[cfg] protein_encoder = {cfg['model'].get('protein_encoder')}")

    chemberta = PROJECT_ROOT / "data" / "chemberta_cache"
    train_ds, val_ds, test_ds, stats = build_datasets(cfg, chemberta)
    print(f"[data] train={stats['n_train_pairs']} val={stats['n_val_pairs']} test={stats['n_test_pairs']}")
    sample = train_ds[0]
    print(f"[item] keys={list(sample.keys())}")
    print(f"[item] prot_emb shape={tuple(sample['prot_emb'].shape)} prot_len={sample['prot_len']}")
    assert sample["prot_emb"].ndim == 2, "attn_pool must yield [L, D] tensors"

    # Pointwise collator with attn_pool
    batch_items = [train_ds[i] for i in range(4)]
    batch = collate_pointwise(batch_items)
    print(f"[collate_pointwise] prot_emb={tuple(batch['prot_emb'].shape)} mask={tuple(batch['prot_mask'].shape)}")
    assert batch["prot_emb"].ndim == 3
    assert batch["prot_mask"].dtype == torch.bool

    # Model forward (mean_pool path NOT used here)
    model = RankBind(cfg)
    print(f"[model] protein_encoder={model.protein_encoder} "
          f"attn_pool={model.attn_pool is not None} "
          f"trainable={model.count_parameters()['n_parameters_trainable']:,}")

    # score_pairs forward
    s = model.score_pairs(batch["lig_emb"], batch["prot_emb"], batch["prot_mask"])
    print(f"[score_pairs] out={tuple(s.shape)}  values={s.detach().numpy()}")
    assert s.shape == (4,)

    # Triplet collator: pick a positive anchor + 4 negatives.
    triplet = TripletCollator(
        train_dataset=train_ds,
        n_negatives=cfg["triplet"]["n_negatives_per_positive"],
        seed=42,
        negative_sampling="cross_protein_implicit",  # warmup mode (no model needed)
        hard_pool_size=cfg["triplet"]["hard_pool_size"],
    )
    # Find a positive in the first 64 items
    pos_items = [it for it in (train_ds[i] for i in range(64)) if it["label"] == 1.0][:4]
    if len(pos_items) < 2:
        raise RuntimeError("Need at least 2 positives in first 64 items for smoke test")
    tb = triplet([*pos_items, train_ds[0]])  # pad with one extra (may be neg)
    assert tb is not None
    print(f"[triplet] B={tb['n_anchors_kept']} "
          f"pos_prot={tuple(tb['pos_prot'].shape)} pos_mask={tuple(tb['pos_mask'].shape)} "
          f"neg_prot={tuple(tb['neg_prot'].shape)} neg_mask={tuple(tb['neg_mask'].shape)}")
    assert tb["pos_prot"].ndim == 3
    assert tb["neg_prot"].ndim == 4

    # score_triplet forward + backward
    pos_s, neg_s = model.score_triplet(
        tb["lig_emb"], tb["pos_prot"], tb["neg_prot"],
        tb["pos_mask"], tb["neg_mask"],
    )
    print(f"[score_triplet] pos_s={tuple(pos_s.shape)} neg_s={tuple(neg_s.shape)}")
    assert pos_s.shape == (tb["n_anchors_kept"],)
    assert neg_s.shape == (tb["n_anchors_kept"], 4)

    loss = (-(pos_s - neg_s.max(dim=1).values).clamp(min=-1)).mean()
    loss.backward()
    # attn_pool query should have grad
    grad_norm = float(model.attn_pool.q.grad.norm())
    print(f"[grad] attn_pool.q grad-norm = {grad_norm:.4e}  (must be > 0)")
    assert grad_norm > 0

    # Sanity: attention weights should NOT all be uniform after init (random query).
    # Build a mixed-length batch on purpose so we can check padding behaviour.
    mixed_idxs = []
    seen_lens = set()
    for i in range(len(train_ds)):
        L = train_ds[i]["prot_len"]
        if L not in seen_lens:
            seen_lens.add(L); mixed_idxs.append(i)
        if len(mixed_idxs) == 4:
            break
    mixed_batch = collate_pointwise([train_ds[i] for i in mixed_idxs])
    with torch.no_grad():
        _, weights = model.attn_pool(mixed_batch["prot_emb"], mixed_batch["prot_mask"])
    print(f"[attn] mixed-batch lens = {mixed_batch['prot_len'].tolist()}  "
          f"weights shape = {tuple(weights.shape)} "
          f"min/max/std = {float(weights.min()):.4e} {float(weights.max()):.4e} {float(weights.std()):.4e}")
    n_pad = int((~mixed_batch["prot_mask"]).sum())
    if n_pad > 0:
        pad_max_abs = float(weights[~mixed_batch["prot_mask"]].abs().max())
        print(f"[attn] padding cells = {n_pad}, padding-weight max abs = {pad_max_abs:.4e}  (must be 0)")
        assert pad_max_abs == 0.0
    else:
        print("[attn] no padding cells in mixed batch (all same length) — skip padding check")
    # Per-row weights must sum to 1 over real residues.
    row_sums = weights.sum(dim=-1)
    print(f"[attn] row sums (should be 1.0): {row_sums.tolist()}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    print("\n[smoke_test] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
