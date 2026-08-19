"""
Golden-equivalence test: NegativeSelector == TripletCollator selection.

The GraphDTA recipe-transfer experiment only proves "same recipe, different
architecture" if GraphDTA draws the *identical* negatives as RankBind v4. The
standalone v5_rankbind.negative_selection.NegativeSelector copies the selection
logic out of TripletCollator (which is left untouched, to protect the validated
v4 numbers); this test is the contract that the two stay byte-identical.

Both implementations use a single np.random.default_rng(seed) and perform the
identical sequence of sampling operations per anchor, so seeding them the same
and calling them on the same anchor sequence yields identical draws — even when
called separately (independent, identically-seeded RNGs advance in lockstep
under the same operations).

Run:  python -m v5_rankbind.tests.test_negative_selection
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from v5_rankbind.tests.test_sampler import _make_synthetic_dataset


def _anchor_sequence(ds):
    """Fixed (smiles, uniprot) anchor list over all positive rows, repeated
    twice so the RNG is exercised across many draws."""
    anchors = [(ds.smiles_at(i), ds.protein_at(i))
               for i in range(len(ds)) if ds.label_at(i) == 1]
    return anchors * 2


def test_random_mode_identical():
    from v5_rankbind.sampler import TripletCollator
    from v5_rankbind.negative_selection import NegativeSelector

    ds = _make_synthetic_dataset()
    coll = TripletCollator(train_dataset=ds, n_negatives=3, seed=123)
    sel = NegativeSelector(ds, n_negatives=3, seed=123)

    for smi, uni in _anchor_sequence(ds):
        a = coll._sample_negs_for_anchor(smi, uni)
        b = sel.sample_negs_for_anchor(smi, uni)
        assert a == b, (smi, uni, a, b)
    print("  ok: cross_protein_implicit negatives identical")


def test_hard_mode_identical():
    from v5_rankbind.sampler import TripletCollator
    from v5_rankbind.negative_selection import NegativeSelector

    ds = _make_synthetic_dataset()
    coll = TripletCollator(
        train_dataset=ds, n_negatives=3, seed=77,
        negative_sampling="hard", hard_pool_size=2,
    )
    sel = NegativeSelector(
        ds, n_negatives=3, seed=77,
        negative_sampling="hard", hard_pool_size=2,
    )
    # Both share the same _smi_to_row / _all_proteins ordering, so one score
    # matrix is valid for both.
    assert coll._smi_to_row == sel._smi_to_row
    assert coll._all_proteins == sel._all_proteins

    n_lig = len(coll._smi_to_row)
    n_prot = len(coll._all_proteins)
    scores = np.random.default_rng(0).normal(size=(n_lig, n_prot)).astype(np.float32)
    coll._scores = scores
    sel.set_scores(scores)

    for smi, uni in _anchor_sequence(ds):
        a = coll._sample_negs_for_anchor(smi, uni)
        b = sel.sample_negs_for_anchor(smi, uni)
        assert a == b, (smi, uni, a, b)
    print("  ok: hard-pool negatives identical")


def test_hard_fallback_identical():
    """Hard mode before any score cache must fall back to random identically."""
    from v5_rankbind.sampler import TripletCollator
    from v5_rankbind.negative_selection import NegativeSelector

    ds = _make_synthetic_dataset()
    coll = TripletCollator(
        train_dataset=ds, n_negatives=2, seed=9,
        negative_sampling="hard", hard_pool_size=4,
    )
    sel = NegativeSelector(
        ds, n_negatives=2, seed=9,
        negative_sampling="hard", hard_pool_size=4,
    )
    assert coll._scores is None and sel._scores is None
    for smi, uni in _anchor_sequence(ds):
        a = coll._sample_negs_for_anchor(smi, uni)
        b = sel.sample_negs_for_anchor(smi, uni)
        assert a == b, (smi, uni, a, b)
    print("  ok: hard-mode warmup fallback identical")


def test_hard_partial_coverage_identical():
    """Capped refresh leaves -inf rows (un-scored ligands → random fallback) and
    -inf columns (un-scored proteins → ineligible). The GraphDTA hard-neg refresh
    uses exactly this sentinel (refresh_scores_graphdta seeds the grid with
    -np.inf), so the two implementations must still agree under partial coverage.
    """
    from v5_rankbind.sampler import TripletCollator
    from v5_rankbind.negative_selection import NegativeSelector

    ds = _make_synthetic_dataset()
    coll = TripletCollator(
        train_dataset=ds, n_negatives=3, seed=55,
        negative_sampling="hard", hard_pool_size=3,
    )
    sel = NegativeSelector(
        ds, n_negatives=3, seed=55,
        negative_sampling="hard", hard_pool_size=3,
    )
    n_lig = len(coll._smi_to_row)
    n_prot = len(coll._all_proteins)
    scores = np.random.default_rng(1).normal(size=(n_lig, n_prot)).astype(np.float32)
    # Un-scored ligand rows (lig_cap) → entire row -inf → must hit random fallback.
    scores[0, :] = -np.inf
    if n_lig > 2:
        scores[2, :] = -np.inf
    # Un-scored protein columns (prot_cap) → never eligible as hard confusers.
    scores[:, 1] = -np.inf
    if n_prot > 3:
        scores[:, 3] = -np.inf
    coll._scores = scores
    sel.set_scores(scores)

    for smi, uni in _anchor_sequence(ds):
        a = coll._sample_negs_for_anchor(smi, uni)
        b = sel.sample_negs_for_anchor(smi, uni)
        assert a == b, (smi, uni, a, b)
    print("  ok: partial-coverage (-inf rows/cols) negatives identical")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    test_random_mode_identical()
    test_hard_mode_identical()
    test_hard_fallback_identical()
    test_hard_partial_coverage_identical()
    print("All negative-selection golden tests passed.")
