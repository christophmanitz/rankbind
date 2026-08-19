"""
v5_rankbind/sampler.py — Balanced and triplet-aware samplers.

Two classes:

  1. ProteinBalancedSampler
       For each protein in the train split, sample approximately equal
       numbers of positives and negatives per epoch. Total length is
       n_proteins * pairs_per_protein_per_epoch.

  2. TripletBatchSampler
       Wraps RankBindDataset to yield triplet batches for margin loss:
       each batch contains B anchors (ligand, positive protein) plus
       k negatives per anchor (same ligand, different protein, label=0).

Both accept a RankBindDataset and behave deterministically for a given seed.

Phase-2 publishability: the sampler writes an audit CSV on the first epoch
(one row per protein: positives seen, negatives seen) so the paper supplement
can show the rebalance concretely.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler

from .data import RankBindDataset, _pad_residues, collate_graph_list


# ──────────────────────────────────────────────────────────────────────────────
# ProteinBalancedSampler
# ──────────────────────────────────────────────────────────────────────────────

class ProteinBalancedSampler(Sampler[int]):
    """Yield indices into a RankBindDataset such that each epoch contains
    roughly equal positives and negatives per protein.

    If a protein has fewer positives than pairs_per_protein_per_epoch * 0.5,
    its positives are sampled with replacement; same for negatives. If a
    protein is missing one class entirely, we draw all pairs from the other
    class (that is unavoidable on decoy-only proteins).
    """

    def __init__(
        self,
        dataset: RankBindDataset,
        pairs_per_protein_per_epoch: int = 16,
        pos_neg_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.pairs_per_protein = pairs_per_protein_per_epoch
        self.pos_neg_ratio = pos_neg_ratio
        self.seed = seed
        self._epoch = 0

        self._by_protein_pos: dict[str, list[int]] = defaultdict(list)
        self._by_protein_neg: dict[str, list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            uni = dataset.protein_at(i)
            if dataset.label_at(i) == 1:
                self._by_protein_pos[uni].append(i)
            else:
                self._by_protein_neg[uni].append(i)

        self.proteins = sorted(
            set(self._by_protein_pos.keys()) | set(self._by_protein_neg.keys())
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.proteins) * self.pairs_per_protein

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch)
        n_pos = max(1, int(round(self.pairs_per_protein * self.pos_neg_ratio
                                 / (1 + self.pos_neg_ratio))))
        n_neg = self.pairs_per_protein - n_pos

        order = list(self.proteins)
        rng.shuffle(order)

        batch_indices: list[int] = []
        for uni in order:
            pos = self._by_protein_pos.get(uni, [])
            neg = self._by_protein_neg.get(uni, [])
            if not pos and not neg:
                continue
            if not pos:
                take_neg = self.pairs_per_protein
                take_pos = 0
            elif not neg:
                take_pos = self.pairs_per_protein
                take_neg = 0
            else:
                take_pos = n_pos
                take_neg = n_neg

            if take_pos > 0:
                idx = rng.choice(pos, size=take_pos, replace=(len(pos) < take_pos))
                batch_indices.extend(int(i) for i in idx)
            if take_neg > 0:
                idx = rng.choice(neg, size=take_neg, replace=(len(neg) < take_neg))
                batch_indices.extend(int(i) for i in idx)

        rng.shuffle(batch_indices)
        return iter(batch_indices)

    # ── audit hook ────────────────────────────────────────────────────────

    def audit(self, out_csv: Path) -> dict:
        """Simulate one epoch, return per-protein counts and write CSV.

        Only reads self.dataset — no model forward pass. Used by tests and
        by train.py's first epoch to produce the paper-supplement table.
        """
        counts_pos: dict[str, int] = defaultdict(int)
        counts_neg: dict[str, int] = defaultdict(int)
        for i in iter(self):
            uni = self.dataset.protein_at(i)
            if self.dataset.label_at(i) == 1:
                counts_pos[uni] += 1
            else:
                counts_neg[uni] += 1

        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for uni in self.proteins:
            p = counts_pos.get(uni, 0); n = counts_neg.get(uni, 0)
            rows.append({
                "uniprot":        uni,
                "n_pos_drawn":    p,
                "n_neg_drawn":    n,
                "n_pos_available":len(self._by_protein_pos.get(uni, [])),
                "n_neg_available":len(self._by_protein_neg.get(uni, [])),
            })
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        totals = {
            "n_proteins":          len(self.proteins),
            "total_pos_drawn":     sum(r["n_pos_drawn"] for r in rows),
            "total_neg_drawn":     sum(r["n_neg_drawn"] for r in rows),
            "total_pos_available": sum(r["n_pos_available"] for r in rows),
            "total_neg_available": sum(r["n_neg_available"] for r in rows),
        }
        return totals


# ──────────────────────────────────────────────────────────────────────────────
# TripletBatchSampler
# ──────────────────────────────────────────────────────────────────────────────

class TripletCollator:
    """Collator that constructs ligand-conditional triplets for margin loss.

    For each positive pair (lig, prot+) in the batch, sample k negative proteins
    `{P_i⁻}` from the training-frame protein pool, excluding any protein that
    is a known positive for this ligand. These are *implicit negatives*:
    unobserved (lig, P⁻) pairs, treated as non-binding with high probability
    (the BRENDA positive density is ≈ 0.3%).

    This mirrors the evaluation geometry — ligand-conditional ranking over
    all proteins — and avoids the SMILES-mismatch pathology of the BRENDA +
    decoys dataset (positives and labeled negatives have essentially disjoint
    SMILES pools, so a strict within-ligand label-0 rule drops 98% of anchors).

    Two sampling modes, selected via `negative_sampling`:

      * "cross_protein_implicit" (v3 default): sample k proteins uniformly at
        random from the valid pool (all train proteins minus known positives
        for this ligand, minus the anchor protein).
      * "hard" (v4): require a cached score matrix (`refresh_scores(model,
        device)` at epoch start). For each anchor, take the top
        `hard_pool_size` proteins from the valid pool by current model score,
        then sample k from that pool uniformly. Falls back to random
        sampling for any anchor whose ligand is missing from the cache or
        before `refresh_scores` has ever been called (epoch 0 / warmup).

    Returns a dict with keys:

        lig_emb   [B, D_lig]             — one per anchor
        pos_prot  [B, D_prot]
        neg_prot  [B, k, D_prot]
        smiles    list[str]
        anchor_uniprot  list[str]
        neg_uniprot     list[list[str]]
        hard_active     bool            — True iff this batch sampled from the
                                          cached score matrix (diagnostic)

    Anchors with label=0 in the incoming batch are dropped — the margin
    loss is undefined without a positive. The training loop logs the ratio
    of kept anchors as `triplet_keep_ratio` (now always 1.0 given ≥1 anchor
    in the batch, since every positive anchor can form a triplet).
    """

    def __init__(
        self,
        train_dataset: RankBindDataset,
        n_negatives: int = 4,
        seed: int = 42,
        negative_sampling: str = "cross_protein_implicit",
        hard_pool_size: int = 50,
    ):
        self.ds = train_dataset
        self.n_negatives = n_negatives
        self._rng = np.random.default_rng(seed)
        self.negative_sampling = negative_sampling
        self.hard_pool_size = int(hard_pool_size)
        self.use_hard = (negative_sampling == "hard")
        self.protein_encoder = getattr(train_dataset, "protein_encoder", "mean_pool")
        self.ligand_encoder = getattr(train_dataset, "ligand_encoder", "mean_pool")
        # gearbind (v7): also emit batched structure + 3D-ligand graphs.
        self.load_graphs = getattr(train_dataset, "load_graphs", False)
        if negative_sampling not in ("cross_protein_implicit", "hard"):
            raise ValueError(
                f"Unknown negative_sampling={negative_sampling!r}; "
                "expected 'cross_protein_implicit' or 'hard'."
            )

        # One dataset index per unique protein (used to fetch its embedding).
        prot_to_idx: dict[str, int] = {}
        for i in range(len(train_dataset)):
            uni = train_dataset.protein_at(i)
            if uni not in prot_to_idx:
                prot_to_idx[uni] = i
        self._prot_to_idx = prot_to_idx
        self._all_proteins = list(prot_to_idx.keys())
        self._prot_to_col = {u: c for c, u in enumerate(self._all_proteins)}

        # Known positive proteins per ligand — excluded from negative pool.
        self._pos_prots_by_smi: dict[str, set[str]] = defaultdict(set)
        # Dataset row per unique SMILES (used for lig embedding fetch).
        smi_to_ds_idx: dict[str, int] = {}
        pos_smiles: set[str] = set()
        for i in range(len(train_dataset)):
            smi = train_dataset.smiles_at(i)
            if smi not in smi_to_ds_idx:
                smi_to_ds_idx[smi] = i
            if train_dataset.label_at(i) == 1:
                self._pos_prots_by_smi[smi].add(train_dataset.protein_at(i))
                pos_smiles.add(smi)
        self._smi_to_ds_idx = smi_to_ds_idx

        # Hard-negative bookkeeping — only the positive-labeled SMILES can
        # ever appear as anchors, so the score matrix is restricted to them.
        self._smi_to_row: dict[str, int] = (
            {s: r for r, s in enumerate(sorted(pos_smiles))}
            if self.use_hard else {}
        )
        self._row_to_smi: list[str] = list(self._smi_to_row.keys())
        self._lig_emb_cache: torch.Tensor | None = None   # [N_lig, D_lig]
        self._prot_emb_cache = None  # mean_pool: [N_prot, D]; attn_pool: list[Tensor]
        self._prot_lens_cache: list[int] | None = None    # attn_pool only
        self._prot_graph_cache: list[dict] | None = None  # gearbind only (static)
        self._scores: np.ndarray | None = None            # [N_lig, N_prot]

    # ── hard-negative score cache ────────────────────────────────────────

    @torch.no_grad()
    def refresh_scores(
        self,
        model,
        device,
        lig_chunk: int = 256,
        prot_chunk: int = 32,
    ) -> dict:
        """Recompute the (positive-ligand × train-protein) score matrix
        using the current model. No-op unless `negative_sampling == 'hard'`.

        For attn_pool encoder: per-residue tensors live on CPU; we encode them
        chunked on GPU through `model.encode_protein` to obtain `gP_all
        [N_prot, d_prot]`, then score against ligands as before.

        DeltaField head: there is no pooled `model.head(fL, gP)` path — scoring
        only happens via the full coupled `forward_field` over per-token
        sequences, so an exact (N_lig × N_prot) score matrix would cost
        O(N_lig·N_prot) coupled transformer forwards (infeasible at turnover
        scale: ~5.7k proteins). For the FIELD_HEADS we therefore use a cheaper
        *model-space protein-similarity* hardness signal: for a positive ligand
        whose true binder(s) are P+, the hard negatives are the proteins whose
        trained protein embedding is most similar to P+. The embedding is
        ligand-independent and static per epoch, but its source differs by head:

          * deltafield → mask-aware mean over residues of `model.prot(residues)`
            (the ProteinProjector — the component actually trained in the field
            path, model.py:forward_field). We deliberately bypass
            `model.encode_protein` / `model.attn_pool`: in the deltafield path
            the attn_pool is instantiated but NEVER trained, so its pooled
            embedding is a random projection and would give meaningless
            similarity.
          * gearbind → `model.prot` is NEVER touched by the gearbind path; its
            trained protein encoder is `model.head.prot_in` + `model.head.prot_gnn`.
            We therefore take the mask-aware mean over residues of the GNN
            free-pass output `U_res = prot_gnn(prot_in(esm2_residues), graph)`.
            This is the free pass only (no ligand, no cross-coupling), so it is
            static per epoch and is the correct protein-identity signal.
        """
        if not self.use_hard:
            return {"refreshed": False}

        # ── Field-head branch: protein-similarity hardness (returns early) ───
        # FIELD_HEADS (deltafield / gearbind) have no pooled model.head(fL, gP)
        # path, so an exact score matrix is infeasible. Both fall back to a cheap
        # model-space protein-similarity signal, but from the head-specific
        # trained protein encoder (deltafield → model.prot mean-pool; gearbind →
        # GNN free-pass residue mean-pool — model.prot is untrained for gearbind).
        from .model import FIELD_HEADS  # local import avoids a cycle at import time
        head_type = getattr(model, "head_type", None)
        if head_type in FIELD_HEADS:
            # Build the per-residue protein cache once (same source as the
            # attn_pool branch below). protein_encoder is "attn_pool" for both
            # field heads, so protein_emb_at returns [L_i, D_in] CPU tensors.
            if self._prot_emb_cache is None:
                prot_residues = [self.ds.protein_emb_at(self._prot_to_idx[u])
                                 for u in self._all_proteins]
                self._prot_emb_cache = prot_residues  # type: ignore[assignment]
                self._prot_lens_cache = [int(t.shape[0]) for t in prot_residues]

            was_training = model.training
            model.eval()
            n_prot = len(self._prot_emb_cache)
            gP_chunks = []

            if head_type == "gearbind":
                # Cache each train protein's structure graph once — static across
                # the run (aligned 1:1 to the cached ESM2 length L_i, so the graph
                # node axis equals the residue axis used by _pad_residues).
                if self._prot_graph_cache is None:
                    self._prot_graph_cache = [
                        self.ds.load_structure(u, int(self._prot_lens_cache[i]))
                        for i, u in enumerate(self._all_proteins)
                    ]
                for s in range(0, n_prot, prot_chunk):
                    e = min(s + prot_chunk, n_prot)
                    chunk_resi = self._prot_emb_cache[s:e]
                    chunk_lens = self._prot_lens_cache[s:e]
                    padded, mask = _pad_residues(
                        [r.to(torch.float32) for r in chunk_resi], chunk_lens
                    )
                    padded = padded.to(device)           # [chunk, L, D_in]
                    mask = mask.to(device)               # [chunk, L]
                    # collate pads the node axis to max(chunk_lens) == L (each
                    # graph's n_res == its ESM2 length), so node_mask aligns.
                    graph = collate_graph_list(
                        self._prot_graph_cache[s:e],
                        has_seqsep=True, node_feat_keys=["plddt"])
                    graph = {k: (v.to(device) if torch.is_tensor(v) else v)
                             for k, v in graph.items()}
                    h0 = model.head.prot_in(padded)              # [chunk, L, d]
                    u = model.head.prot_gnn(h0, graph)           # [chunk, L, d]
                    msum = mask.unsqueeze(-1).to(u.dtype)        # [chunk, L, 1]
                    gP = (u * msum).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
                    gP_chunks.append(gP.float().cpu())           # [chunk, d]
                mode = "gearbind_gnn_freepass"
            else:  # deltafield: ProteinProjector mean-pool (unchanged)
                for s in range(0, n_prot, prot_chunk):
                    e = min(s + prot_chunk, n_prot)
                    chunk_resi = self._prot_emb_cache[s:e]
                    chunk_lens = self._prot_lens_cache[s:e]
                    padded, mask = _pad_residues(
                        [r.to(torch.float32) for r in chunk_resi], chunk_lens
                    )
                    padded = padded.to(device)           # [chunk, L, D_in]
                    mask = mask.to(device)               # [chunk, L]
                    u = model.prot(padded)               # [chunk, L, d] (trained)
                    msum = mask.unsqueeze(-1).to(u.dtype)    # [chunk, L, 1]
                    gP = (u * msum).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
                    gP_chunks.append(gP.float().cpu())       # [chunk, d]
                mode = "deltafield_protein_sim"

            gP_all = torch.cat(gP_chunks, dim=0)         # [N_prot, d] on CPU
            if was_training:
                model.train()

            gP_norm = gP_all / gP_all.norm(dim=1, keepdim=True).clamp(min=1e-8)

            # Per-ligand positive-protein centroid in model embedding space.
            n_lig = len(self._row_to_smi)
            d = gP_norm.shape[1]
            C = torch.zeros(n_lig, d, dtype=torch.float32)
            for r, smi in enumerate(self._row_to_smi):
                cols = [self._prot_to_col[p]
                        for p in self._pos_prots_by_smi.get(smi, set())
                        if p in self._prot_to_col]
                if not cols:
                    # No in-frame positive protein (shouldn't happen for train
                    # anchors); leave row as zeros -> all-equal scores -> the
                    # top-M fallback in _sample_negs_for_anchor still returns a
                    # valid pool.
                    continue
                centroid = gP_norm[cols].mean(dim=0)
                C[r] = centroid / centroid.norm().clamp(min=1e-8)

            # Cosine similarity; higher = more similar = harder, matching the
            # existing top-M selection semantics in _sample_negs_for_anchor.
            self._scores = (C @ gP_norm.t()).numpy()     # [N_lig, N_prot]
            return {"refreshed": True, "n_lig": int(n_lig),
                    "n_prot": int(n_prot), "mode": mode}

        # Lazy-build embedding caches (done once per process).
        if self._lig_emb_cache is None:
            lig_embs = [self.ds[self._smi_to_ds_idx[s]]["lig_emb"]
                        for s in self._row_to_smi]
            self._lig_emb_cache = torch.stack(lig_embs).to(torch.float32)
        if self._prot_emb_cache is None:
            if self.protein_encoder == "attn_pool":
                # Cache per-residue [L_i, D] tensors as a Python list of CPU
                # tensors — stacking with padding upfront is wasteful (would be
                # ~3 GB at L_max=1024 across 618 train proteins).
                prot_residues = [self.ds.protein_emb_at(self._prot_to_idx[u])
                                 for u in self._all_proteins]
                prot_lens = [int(t.shape[0]) for t in prot_residues]
                self._prot_emb_cache = prot_residues  # type: ignore[assignment]
                self._prot_lens_cache = prot_lens
            else:
                prot_embs = [self.ds.protein_emb_at(self._prot_to_idx[u])
                             for u in self._all_proteins]
                self._prot_emb_cache = torch.stack(prot_embs).to(torch.float32)

        was_training = model.training
        model.eval()
        lig_all = self._lig_emb_cache.to(device)

        if self.protein_encoder == "attn_pool":
            n_prot = len(self._prot_emb_cache)
            gP_chunks = []
            for s in range(0, n_prot, prot_chunk):
                e = min(s + prot_chunk, n_prot)
                chunk_resi = self._prot_emb_cache[s:e]
                chunk_lens = self._prot_lens_cache[s:e]
                padded, mask = _pad_residues(
                    [r.to(torch.float32) for r in chunk_resi], chunk_lens
                )
                padded = padded.to(device)
                mask = mask.to(device)
                gP_chunks.append(model.encode_protein(padded, mask))
            gP_all = torch.cat(gP_chunks, dim=0)              # [N_prot, d_prot]
        else:
            prot_all = self._prot_emb_cache.to(device)
            gP_all = model.encode_protein(prot_all)            # [N_prot, d_prot]

        fL_all = model.lig(lig_all)                            # [N_lig, d_lig]
        n_lig, d_lig = fL_all.shape
        n_prot, d_prot = gP_all.shape
        scores = torch.empty(n_lig, n_prot, dtype=torch.float32)
        for start in range(0, n_lig, lig_chunk):
            end = min(start + lig_chunk, n_lig)
            b = end - start
            fL = fL_all[start:end].unsqueeze(1).expand(b, n_prot, d_lig)
            gP = gP_all.unsqueeze(0).expand(b, n_prot, d_prot)
            s = model.head(fL.reshape(b * n_prot, d_lig),
                           gP.reshape(b * n_prot, d_prot))
            scores[start:end] = s.reshape(b, n_prot).float().cpu()

        self._scores = scores.numpy()
        if was_training:
            model.train()
        return {"refreshed": True, "n_lig": int(n_lig), "n_prot": int(n_prot)}

    # ── sampling ─────────────────────────────────────────────────────────

    def _sample_negs_for_anchor(
        self,
        smiles: str,
        anchor_uniprot: str,
    ) -> list[str]:
        """Return `n_negatives` uniprots to serve as negatives for this anchor."""
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

        # Random fallback: v3 cross_protein_implicit behaviour.
        pool = [p for p in self._all_proteins
                if p != anchor_uniprot and p not in known]
        if not pool:
            pool = [p for p in self._all_proteins if p != anchor_uniprot]
        replace = len(pool) < k
        return [str(p) for p in self._rng.choice(pool, size=k, replace=replace)]

    def __call__(self, batch: list[dict]) -> dict | None:
        anchors = [b for b in batch if b["label"] == 1.0]
        if not anchors:
            return None

        if self.ligand_encoder == "per_token":
            # anchors carry [A_i, D] per-token ligand tensors -> pad to [B, A, D]
            lig_tok = [a["lig_emb"] for a in anchors]
            lig_lens = [int(t.shape[0]) for t in lig_tok]
            lig_emb, lig_mask = _pad_residues(lig_tok, lig_lens)
        else:
            lig_emb = torch.stack([a["lig_emb"] for a in anchors])
            lig_mask = None

        hard_active = self.use_hard and self._scores is not None
        neg_unis: list[list[str]] = []
        for a in anchors:
            chosen = self._sample_negs_for_anchor(a["smiles"], a["uniprot"])
            neg_unis.append([str(p) for p in chosen])

        out: dict = {
            "lig_emb":        lig_emb,
            "lig_mask":       lig_mask,
            "smiles":         [a["smiles"] for a in anchors],
            "anchor_uniprot": [a["uniprot"] for a in anchors],
            "neg_uniprot":    neg_unis,
            "n_anchors_in":   len(batch),
            "n_anchors_kept": len(anchors),
            "n_anchors_pos":  len(anchors),
            "n_missing_negs": 0,
            "hard_active":    bool(hard_active),
        }

        if self.protein_encoder == "attn_pool":
            # Pos: pad anchors' [L, D] to [B, L_max_pos, D] + mask.
            pos_residues = [a["prot_emb"] for a in anchors]
            pos_lens = [int(r.shape[0]) for r in pos_residues]
            pos_prot, pos_mask = _pad_residues(pos_residues, pos_lens)

            # Neg: collect for each anchor's k negatives, pad to single global
            # L_max_neg across all (B, k) so the result is [B, k, L_max_neg, D].
            B = len(anchors)
            k = self.n_negatives
            all_neg_residues: list[torch.Tensor] = []
            all_neg_lens: list[int] = []
            all_neg_graphs: list[dict] = []
            for negs in neg_unis:
                for p in negs:
                    r = self.ds.protein_emb_at(self._prot_to_idx[p])
                    all_neg_residues.append(r)
                    all_neg_lens.append(int(r.shape[0]))
                    if self.load_graphs:
                        # neg structure graph aligned to this protein's ESM2 length.
                        all_neg_graphs.append(
                            self.ds.load_structure(p, int(r.shape[0])))
            neg_flat, neg_mask_flat = _pad_residues(all_neg_residues, all_neg_lens)
            L_max_neg, D = neg_flat.shape[-2], neg_flat.shape[-1]
            out["pos_prot"]  = pos_prot                              # [B, L_pos, D]
            out["pos_mask"]  = pos_mask                              # [B, L_pos]
            out["neg_prot"]  = neg_flat.reshape(B, k, L_max_neg, D)  # [B, k, L_neg, D]
            out["neg_mask"]  = neg_mask_flat.reshape(B, k, L_max_neg)  # [B, k, L_neg]

            if self.load_graphs:
                # gearbind: pos protein graphs [B,...], ligand graphs [B,...],
                # neg protein graphs flattened to [B*k,...] then reshaped to [B,k,...].
                out["prot_graph"] = collate_graph_list(
                    [a["prot_graph"] for a in anchors],
                    has_seqsep=True, node_feat_keys=["plddt"])
                out["lig_graph"] = collate_graph_list(
                    [a["lig_graph"] for a in anchors],
                    has_seqsep=False, node_feat_keys=["node_feat"])
                neg_graph_flat = collate_graph_list(
                    all_neg_graphs, has_seqsep=True, node_feat_keys=["plddt"])
                out["neg_prot_graph"] = {
                    key: (val.reshape(B, k, *val.shape[1:]) if torch.is_tensor(val) else val)
                    for key, val in neg_graph_flat.items()
                }
        else:
            out["pos_prot"] = torch.stack([a["prot_emb"] for a in anchors])
            out["neg_prot"] = torch.stack([
                torch.stack(
                    [self.ds.protein_emb_at(self._prot_to_idx[p]) for p in negs]
                )
                for negs in neg_unis
            ])
        return out
