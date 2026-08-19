"""
v5_rankbind/negative_selection.py — Featurization-agnostic negative selection.

The ligand-conditional negative-sampling logic of
``v5_rankbind.sampler.TripletCollator`` is independent of how proteins and
ligands are featurized — it operates purely on (smiles, uniprot, label)
identity. This module extracts that core so a different architecture
(e.g. GraphDTA, which consumes molecular graphs + integer sequences rather
than ESM2/ChemBERTa embeddings) can train on the *identical* anti-shortcut
data regime as RankBind v4.

``NegativeSelector`` reproduces ``TripletCollator._sample_negs_for_anchor``
byte-for-byte under the same seed and the same cached score matrix. That
equivalence is asserted by ``v5_rankbind/tests/test_negative_selection.py`` —
the guarantee that the GraphDTA recipe sees the same negatives as RankBind,
which is the whole point of the recipe-transfer experiment.

Design note: this is a *standalone* copy of the selection logic, not a
refactor of ``TripletCollator``. The v4 pipeline (and its published numbers)
is left untouched on purpose; the golden test cross-checks the two
implementations instead of sharing code, so a future edit to either side
that breaks identity is caught immediately.

Dataset interface required (duck-typed):
    len(dataset)            -> int
    dataset.smiles_at(i)    -> str
    dataset.protein_at(i)   -> str   (uniprot)
    dataset.label_at(i)     -> int   (1 = binder, 0 = non-binder/decoy)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class NegativeSelector:
    """Ligand-conditional negative sampler (random or hard-pool).

    Mirrors ``TripletCollator``'s negative selection. Two modes via
    ``negative_sampling``:

      * ``"cross_protein_implicit"`` — sample ``n_negatives`` proteins
        uniformly from the valid pool (all proteins minus the ligand's known
        positives minus the anchor protein).
      * ``"hard"`` — requires a cached (positive-ligand × protein) score
        matrix set via ``set_scores``. For each anchor, take the top
        ``hard_pool_size`` proteins from the valid pool by current score and
        sample ``n_negatives`` from that pool. Falls back to random sampling
        for any anchor whose ligand is missing from the cache or before
        ``set_scores`` has ever been called (warmup).

    The column order of the score matrix passed to ``set_scores`` must match
    ``self.all_proteins`` (first-appearance order over the dataset); the row
    order must match ``self.row_to_smi``.
    """

    def __init__(
        self,
        dataset,
        n_negatives: int = 4,
        seed: int = 42,
        negative_sampling: str = "cross_protein_implicit",
        hard_pool_size: int = 50,
    ):
        if negative_sampling not in ("cross_protein_implicit", "hard"):
            raise ValueError(
                f"Unknown negative_sampling={negative_sampling!r}; "
                "expected 'cross_protein_implicit' or 'hard'."
            )
        self.n_negatives = n_negatives
        self._rng = np.random.default_rng(seed)
        self.negative_sampling = negative_sampling
        self.hard_pool_size = int(hard_pool_size)
        self.use_hard = (negative_sampling == "hard")

        # Protein column order: first-appearance over the dataset (identical to
        # TripletCollator's prot_to_idx insertion order).
        prot_to_idx: dict[str, int] = {}
        for i in range(len(dataset)):
            uni = dataset.protein_at(i)
            if uni not in prot_to_idx:
                prot_to_idx[uni] = i
        self._all_proteins = list(prot_to_idx.keys())
        self._prot_to_col = {u: c for c, u in enumerate(self._all_proteins)}

        # Known positive proteins per ligand — excluded from the negative pool.
        self._pos_prots_by_smi: dict[str, set[str]] = defaultdict(set)
        pos_smiles: set[str] = set()
        for i in range(len(dataset)):
            smi = dataset.smiles_at(i)
            if dataset.label_at(i) == 1:
                self._pos_prots_by_smi[smi].add(dataset.protein_at(i))
                pos_smiles.add(smi)

        # Only positive-labeled SMILES can be anchors, so the hard-negative
        # score matrix is restricted to them (same convention as TripletCollator).
        self._smi_to_row: dict[str, int] = (
            {s: r for r, s in enumerate(sorted(pos_smiles))}
            if self.use_hard else {}
        )
        self._row_to_smi: list[str] = list(self._smi_to_row.keys())
        self._scores: np.ndarray | None = None  # [N_lig_rows, N_prot_cols]

    # ── read-only views for the score-matrix builder ──────────────────────
    @property
    def all_proteins(self) -> list[str]:
        return self._all_proteins

    @property
    def row_to_smi(self) -> list[str]:
        return self._row_to_smi

    @property
    def smi_to_row(self) -> dict[str, int]:
        return self._smi_to_row

    def pos_prots(self, smiles: str) -> set[str]:
        return self._pos_prots_by_smi.get(smiles, set())

    def set_scores(self, scores: np.ndarray | None) -> None:
        """Install the (positive-ligand × protein) score cache for hard mode.

        Rows are indexed by ``self.row_to_smi`` (i.e. ``self.smi_to_row``),
        columns by ``self.all_proteins`` (``self._prot_to_col``).
        """
        self._scores = None if scores is None else np.asarray(scores)

    # ── sampling (verbatim from TripletCollator._sample_negs_for_anchor) ───
    def sample_negs_for_anchor(self, smiles: str, anchor_uniprot: str) -> list[str]:
        """Return ``n_negatives`` uniprots to serve as negatives for this anchor."""
        known = self._pos_prots_by_smi.get(smiles, set())
        k = self.n_negatives

        if self.use_hard and self._scores is not None and smiles in self._smi_to_row:
            row = self._smi_to_row[smiles]
            scores_row = self._scores[row].astype(np.float32, copy=True)
            for p in known:
                c = self._prot_to_col.get(p)
                if c is not None:
                    scores_row[c] = -np.inf
            a_col = self._prot_to_col.get(anchor_uniprot)
            if a_col is not None:
                scores_row[a_col] = -np.inf
            valid_mask = np.isfinite(scores_row)
            n_valid = int(valid_mask.sum())
            if n_valid > 0:
                M = min(self.hard_pool_size, n_valid)
                # argpartition selects the top-M by score; order within is irrelevant.
                topM_cols = np.argpartition(-scores_row, M - 1)[:M]
                if M >= k:
                    chosen_cols = self._rng.choice(topM_cols, size=k, replace=False)
                else:
                    chosen_cols = self._rng.choice(topM_cols, size=k, replace=True)
                return [self._all_proteins[int(c)] for c in chosen_cols]
            # else: fall through to random fallback.

        # Random fallback: cross_protein_implicit behaviour.
        pool = [p for p in self._all_proteins
                if p != anchor_uniprot and p not in known]
        if not pool:
            pool = [p for p in self._all_proteins if p != anchor_uniprot]
        replace = len(pool) < k
        return [str(p) for p in self._rng.choice(pool, size=k, replace=replace)]
