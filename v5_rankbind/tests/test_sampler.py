"""
Smoke tests for ProteinBalancedSampler and TripletCollator.

These are fast offline tests — they use a synthetic dataset so they can run
in any venv (no ESM2 / ChemBERTa / torch_geometric needed). Run:

    python -m v5_rankbind.tests.test_sampler
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _make_synthetic_dataset():
    # Lazy-import so top-level import doesn't drag transformers in.
    from v5_rankbind.data import RankBindDataset

    # 5 proteins, 8 ligands. Strong class imbalance to exercise the sampler.
    rows = []
    for p_i in range(5):
        for l_i in range(8):
            label = int((p_i + l_i) % 3 == 0)
            rows.append({
                "uniprot":          f"P{p_i}",
                "substrate_smiles": f"S{l_i}",
                "label":            label,
                "idx":              len(rows),
            })
    pairs = pd.DataFrame(rows)

    class _FakeDataset(RankBindDataset):
        def __getitem__(self, i: int) -> dict:
            row = self.pairs.iloc[i]
            return {
                "smiles":   row["substrate_smiles"],
                "uniprot":  row["uniprot"],
                "label":    float(row["label"]),
                "lig_emb":  torch.ones(8) * hash(row["substrate_smiles"]) % 7,
                "prot_emb": torch.ones(16) * hash(row["uniprot"]) % 13,
                "pair_idx": int(row["idx"]),
            }

    return _FakeDataset(
        pairs=pairs,
        sequences={f"P{i}": "M" * 10 for i in range(5)},
        esm2_dir=Path("/nonexistent"),
        chemberta_cache_dir=Path("/nonexistent"),
        prot_input_dim=16,
        lig_input_dim=8,
    )


def test_balanced_sampler_yields_balance():
    from v5_rankbind.sampler import ProteinBalancedSampler

    ds = _make_synthetic_dataset()
    sampler = ProteinBalancedSampler(
        ds, pairs_per_protein_per_epoch=10, pos_neg_ratio=1.0, seed=0
    )
    assert len(sampler) == 5 * 10

    counts_pos, counts_neg = {}, {}
    for i in sampler:
        uni = ds.protein_at(i)
        if ds.label_at(i) == 1:
            counts_pos[uni] = counts_pos.get(uni, 0) + 1
        else:
            counts_neg[uni] = counts_neg.get(uni, 0) + 1

    # Each protein with both classes should have ~5 pos + ~5 neg.
    for p in ds.pairs["uniprot"].unique():
        pos = counts_pos.get(p, 0); neg = counts_neg.get(p, 0)
        # allow proteins that are single-class to skip the symmetry check
        n_pos_avail = int(((ds.pairs["uniprot"] == p) & (ds.pairs["label"] == 1)).sum())
        n_neg_avail = int(((ds.pairs["uniprot"] == p) & (ds.pairs["label"] == 0)).sum())
        if n_pos_avail > 0 and n_neg_avail > 0:
            assert abs(pos - neg) <= 1, (p, pos, neg)
        assert pos + neg == 10
    print("  ok: balanced counts per protein")


def test_audit_csv():
    from v5_rankbind.sampler import ProteinBalancedSampler

    ds = _make_synthetic_dataset()
    sampler = ProteinBalancedSampler(ds, pairs_per_protein_per_epoch=6, seed=1)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "audit.csv"
        totals = sampler.audit(out)
        df = pd.read_csv(out)
        assert len(df) == 5
        assert set(df.columns) == {
            "uniprot", "n_pos_drawn", "n_neg_drawn",
            "n_pos_available", "n_neg_available",
        }
        assert totals["total_pos_drawn"] + totals["total_neg_drawn"] == len(sampler)
    print("  ok: audit CSV schema")


def test_triplet_collator_drops_label0_anchors():
    from v5_rankbind.sampler import TripletCollator

    ds = _make_synthetic_dataset()
    coll = TripletCollator(train_dataset=ds, n_negatives=2, seed=7)

    batch = [ds[i] for i in range(len(ds))]
    out = coll(batch)
    assert out is not None
    assert out["pos_prot"].shape[0] == out["neg_prot"].shape[0]
    assert out["neg_prot"].shape[1] == 2
    assert out["n_anchors_pos"] >= 1
    # Anchors kept are a subset of label=1 entries
    assert out["n_anchors_kept"] <= out["n_anchors_pos"]
    print("  ok: triplet collator shape",
          tuple(out["pos_prot"].shape), tuple(out["neg_prot"].shape))


def test_triplet_all_negative_batch_returns_none():
    from v5_rankbind.sampler import TripletCollator

    ds = _make_synthetic_dataset()
    coll = TripletCollator(train_dataset=ds, n_negatives=2, seed=7)
    neg_only = [b for b in (ds[i] for i in range(len(ds))) if b["label"] == 0.0]
    assert coll(neg_only) is None
    print("  ok: neg-only batch returns None")


def test_triplet_collator_hard_falls_back_without_scores():
    from v5_rankbind.sampler import TripletCollator

    ds = _make_synthetic_dataset()
    coll = TripletCollator(
        train_dataset=ds, n_negatives=2, seed=7,
        negative_sampling="hard", hard_pool_size=3,
    )
    assert coll.use_hard
    assert coll._scores is None
    # Without a score cache, hard mode must silently fall back to random.
    batch = [ds[i] for i in range(len(ds))]
    out = coll(batch)
    assert out is not None
    assert out["hard_active"] is False
    assert out["neg_prot"].shape[1] == 2
    print("  ok: hard mode falls back to random before refresh_scores")


def test_triplet_collator_hard_selects_top_scoring():
    """With a hand-crafted score matrix, hard mode should concentrate
    picks on the highest-scoring non-positive proteins for each anchor."""
    from v5_rankbind.sampler import TripletCollator

    ds = _make_synthetic_dataset()
    coll = TripletCollator(
        train_dataset=ds, n_negatives=2, seed=11,
        negative_sampling="hard", hard_pool_size=1,  # strict: only the single hardest
    )
    # Inject a synthetic score cache: for each positive-ligand row, give
    # every protein a unique score so argpartition has a clean winner. We
    # choose scores so that one non-positive protein clearly dominates.
    n_lig = len(coll._smi_to_row)
    n_prot = len(coll._all_proteins)
    scores = np.full((n_lig, n_prot), -1.0, dtype=np.float32)
    # For every ligand row, make the first protein the "hardest" candidate.
    scores[:, 0] = 10.0
    scores[:, 1] = 5.0
    coll._scores = scores

    # Build a batch of positive anchors only (we need labels==1).
    pos_batch = [ds[i] for i in range(len(ds)) if ds.label_at(i) == 1]
    assert len(pos_batch) > 0
    out = coll(pos_batch)
    assert out is not None
    assert out["hard_active"] is True

    # For each anchor, the chosen negatives must not include the ligand's
    # known positives OR the anchor protein itself. And with pool_size=1
    # each negative should be the top-scoring eligible protein.
    all_proteins = coll._all_proteins
    for smi, anchor_uni, neg_list in zip(
        out["smiles"], out["anchor_uniprot"], out["neg_uniprot"]
    ):
        known = coll._pos_prots_by_smi.get(smi, set())
        for n in neg_list:
            assert n not in known, (smi, anchor_uni, n, known)
            assert n != anchor_uni
        # With hard_pool_size=1 and k=2, both negs must be the same top pick
        # (sampled with replacement since pool < k). Verify that pick is the
        # highest-scoring eligible protein.
        eligible_scores = scores[coll._smi_to_row[smi]].copy()
        for p in known:
            eligible_scores[coll._prot_to_col[p]] = -np.inf
        eligible_scores[coll._prot_to_col[anchor_uni]] = -np.inf
        expected = all_proteins[int(np.argmax(eligible_scores))]
        assert all(n == expected for n in neg_list), (smi, neg_list, expected)
    print("  ok: hard mode picks top-scoring eligible proteins")


def test_triplet_collator_hard_excludes_known_positives():
    """Hard mode must never sample a known-positive protein, even when its
    cached score is the highest."""
    from v5_rankbind.sampler import TripletCollator

    ds = _make_synthetic_dataset()
    coll = TripletCollator(
        train_dataset=ds, n_negatives=3, seed=3,
        negative_sampling="hard", hard_pool_size=5,
    )
    # Inject scores where every known-positive protein scores VERY high —
    # the masking is what keeps them out of the negative pool.
    n_lig = len(coll._smi_to_row)
    n_prot = len(coll._all_proteins)
    scores = np.random.default_rng(0).normal(size=(n_lig, n_prot)).astype(np.float32)
    for smi, pos_prots in coll._pos_prots_by_smi.items():
        if smi not in coll._smi_to_row:
            continue
        row = coll._smi_to_row[smi]
        for p in pos_prots:
            scores[row, coll._prot_to_col[p]] = 1e6  # would dominate if not masked
    coll._scores = scores

    pos_batch = [ds[i] for i in range(len(ds)) if ds.label_at(i) == 1]
    out = coll(pos_batch)
    assert out is not None
    for smi, anchor_uni, neg_list in zip(
        out["smiles"], out["anchor_uniprot"], out["neg_uniprot"]
    ):
        known = coll._pos_prots_by_smi.get(smi, set())
        for n in neg_list:
            assert n not in known, ("leaked known positive", smi, n)
            assert n != anchor_uni
    print("  ok: hard mode masks known positives")


if __name__ == "__main__":
    # Make the package importable when run as script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    test_balanced_sampler_yields_balance()
    test_audit_csv()
    test_triplet_collator_drops_label0_anchors()
    test_triplet_all_negative_batch_returns_none()
    test_triplet_collator_hard_falls_back_without_scores()
    test_triplet_collator_hard_selects_top_scoring()
    test_triplet_collator_hard_excludes_known_positives()
    print("All sampler tests passed.")
