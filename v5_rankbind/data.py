"""
v5_rankbind/data.py — Dataset for RankBind.

Reuses BRENDADataConfig.get_protein_split(seed=42) and the pre-computed
ESM2 per-residue embeddings at data/esm2_embeddings/. ChemBERTa embeddings
are cached on disk on first access (one file per unique SMILES).

Each item exposes:
    {
      'smiles':  str,
      'uniprot': str,
      'label':   float (0/1),
      'lig_emb': Tensor [lig_input_dim]        (ChemBERTa mean-pool),
      'prot_emb':Tensor [prot_input_dim]       (ESM2 mean-pool),
      'pair_idx':int    (original CSV row index)
    }

We do *not* rebuild graphs — Phase 2 is a fixed-encoder experiment.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "adapters"))

from common import BRENDADataConfig  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# ChemBERTa cache
# ──────────────────────────────────────────────────────────────────────────────

_CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"


def _smiles_key(smiles: str) -> str:
    return hashlib.sha1(smiles.encode()).hexdigest()[:16]


def ensure_chemberta_cache(
    smiles_list: Iterable[str],
    cache_dir: Path,
    device: str = "cpu",
    batch_size: int = 64,
) -> None:
    """Precompute and cache ChemBERTa mean-pooled embeddings.

    Writes cache_dir/<sha1(smiles)>.pt per SMILES. Idempotent.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    smiles_list = list(dict.fromkeys(smiles_list))  # dedupe, preserve order
    missing = [s for s in smiles_list
               if not (cache_dir / f"{_smiles_key(s)}.pt").exists()]
    if not missing:
        return

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(_CHEMBERTA_MODEL_NAME)
    mdl = AutoModel.from_pretrained(_CHEMBERTA_MODEL_NAME).to(device)
    mdl.eval()

    with torch.no_grad():
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(device)
            out = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            for s, vec in zip(chunk, pooled.cpu()):
                torch.save(vec.contiguous(), cache_dir / f"{_smiles_key(s)}.pt")


def load_chemberta(smiles: str, cache_dir: Path) -> torch.Tensor:
    p = cache_dir / f"{_smiles_key(smiles)}.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"ChemBERTa embedding missing for SMILES={smiles!r} at {p}. "
            "Run ensure_chemberta_cache first."
        )
    return torch.load(p, weights_only=True)


def ensure_chemberta_token_cache(
    smiles_list: Iterable[str],
    cache_dir: Path,
    device: str = "cpu",
    batch_size: int = 64,
    max_length: int = 128,
) -> None:
    """Precompute and cache ChemBERTa PER-TOKEN embeddings for DeltaField.

    Writes cache_dir/<sha1(smiles)>.pt = float32 [A, 384] of the per-token
    last_hidden_state with the leading [CLS] row dropped, for the A real
    (non-padding) tokens. This is the ligand-atom node sequence the difference
    field operates on (vs. the mean-pooled [384] vector used by the pooled
    heads). Idempotent. Intended cache_dir = a /work2 workspace path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    smiles_list = list(dict.fromkeys(smiles_list))
    missing = [s for s in smiles_list
               if not (cache_dir / f"{_smiles_key(s)}.pt").exists()]
    if not missing:
        return

    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(_CHEMBERTA_MODEL_NAME)
    mdl = AutoModel.from_pretrained(_CHEMBERTA_MODEL_NAME).to(device)
    mdl.eval()

    with torch.no_grad():
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
            out = mdl(**enc).last_hidden_state            # [b, T, 384]
            am = enc["attention_mask"]                    # [b, T]
            for j, s in enumerate(chunk):
                n = int(am[j].sum().item())
                # drop CLS (row 0), keep the remaining real tokens
                toks = out[j, 1:n].contiguous().cpu().to(torch.float32)  # [A, 384]
                torch.save(toks, cache_dir / f"{_smiles_key(s)}.pt")


def load_chemberta_tokens(smiles: str, cache_dir: Path,
                          max_tokens: int = 128) -> torch.Tensor:
    """Load per-token ChemBERTa [A, 384] (clipped to max_tokens)."""
    p = cache_dir / f"{_smiles_key(smiles)}.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"ChemBERTa per-token embedding missing for SMILES={smiles!r} at {p}. "
            "Run ensure_chemberta_token_cache first."
        )
    t = torch.load(p, weights_only=True)
    if t.ndim == 1:                # defensive: a mean-pool file slipped in
        t = t.unsqueeze(0)
    return t[:max_tokens]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class RankBindDataset(Dataset):
    """Indexable dataset. Each index is a (protein, ligand, label) triple.

    Two protein-encoder modes, selected by `protein_encoder` arg:
      - "mean_pool" (default, v4 behaviour): __getitem__["prot_emb"] is [D] —
        the ESM2 mean-pooled vector loaded from disk.
      - "attn_pool" (Stage-(b) of Phase 4 plan, §13.2): __getitem__["prot_emb"]
        is [L, D] per-residue, plus __getitem__["prot_len"] = L. Collators
        pad to max_L per batch and emit a [B, max_L] mask.
    """

    def __init__(
        self,
        pairs: pd.DataFrame,
        sequences: dict,
        esm2_dir: Path,
        chemberta_cache_dir: Path,
        prot_input_dim: int = 1280,
        lig_input_dim: int = 384,
        protein_encoder: str = "mean_pool",
        max_residues: int = 1024,
        ligand_encoder: str = "mean_pool",
        chemberta_token_cache_dir: Path | None = None,
        max_ligand_tokens: int = 128,
        structure_dir: Path | None = None,
        ligand_conformer_dir: Path | None = None,
        load_graphs: bool = False,
        protein_k_cap: int = 32,
        ligand_k_cap: int = 16,
    ):
        if protein_encoder not in ("mean_pool", "attn_pool"):
            raise ValueError(
                f"Unknown protein_encoder={protein_encoder!r}; "
                "expected 'mean_pool' or 'attn_pool'."
            )
        if ligand_encoder not in ("mean_pool", "per_token"):
            raise ValueError(
                f"Unknown ligand_encoder={ligand_encoder!r}; "
                "expected 'mean_pool' or 'per_token'."
            )
        if ligand_encoder == "per_token" and chemberta_token_cache_dir is None:
            raise ValueError("ligand_encoder='per_token' requires chemberta_token_cache_dir.")
        if load_graphs and protein_encoder != "attn_pool":
            raise ValueError(
                "load_graphs=True (gearbind) requires protein_encoder='attn_pool' "
                "(the structure GNN consumes per-residue ESM2 node inputs)."
            )
        self.pairs = pairs.reset_index(drop=True)
        self.sequences = sequences
        self.esm2_dir = Path(esm2_dir)
        self.chemberta_cache_dir = Path(chemberta_cache_dir)
        self.prot_input_dim = prot_input_dim
        self.lig_input_dim = lig_input_dim
        self.protein_encoder = protein_encoder
        self.max_residues = int(max_residues)
        self.ligand_encoder = ligand_encoder
        self.chemberta_token_cache_dir = (
            Path(chemberta_token_cache_dir) if chemberta_token_cache_dir else None
        )
        self.max_ligand_tokens = int(max_ligand_tokens)
        self.structure_dir = Path(structure_dir) if structure_dir else None
        self.ligand_conformer_dir = (
            Path(ligand_conformer_dir) if ligand_conformer_dir else None
        )
        self.load_graphs = bool(load_graphs)
        self.protein_k_cap = int(protein_k_cap)
        self.ligand_k_cap = int(ligand_k_cap)

    def __len__(self) -> int:
        return len(self.pairs)

    def protein_at(self, i: int) -> str:
        return self.pairs.at[i, "uniprot"]

    def smiles_at(self, i: int) -> str:
        return self.pairs.at[i, "substrate_smiles"]

    def label_at(self, i: int) -> int:
        return int(self.pairs.at[i, "label"])

    def protein_emb_at(self, i: int) -> torch.Tensor:
        """Protein embedding for row i WITHOUT touching the ligand cache.

        The triplet collator and the hard-negative refresh only need a
        protein's embedding; going through __getitem__ would also load (and,
        in per_token mode, require a cached) ligand for that row — wasteful and
        a spurious cache dependency. This fetches the protein tensor alone.
        """
        return self.load_protein(self.pairs.at[i, "uniprot"])

    def load_protein(self, uniprot: str) -> torch.Tensor:
        """Load protein embedding in the shape demanded by the encoder mode.

        mean_pool → [D]. attn_pool → [L, D] (clipped to max_residues).
        Missing-file fallback is consistent across both modes (empty tensor with
        a sensible mask in attn_pool, zero vector in mean_pool).
        """
        p = self.esm2_dir / f"{uniprot}.pt"
        if not p.exists():
            if self.protein_encoder == "attn_pool":
                # 1-residue zero "sequence" → mask [True] avoids softmax NaN.
                return torch.zeros(1, self.prot_input_dim, dtype=torch.float32)
            return torch.zeros(self.prot_input_dim, dtype=torch.float32)
        emb = torch.load(p, weights_only=True).to(torch.float32)
        if self.protein_encoder == "mean_pool":
            if emb.ndim == 2:
                emb = emb.mean(dim=0)
            return emb
        # attn_pool: keep [L, D]. Clip to max_residues from the N-terminus —
        # there is no biology rationale for a smarter crop given the small
        # number of cases (max observed length 1022 ≈ max_residues 1024).
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        if emb.shape[0] > self.max_residues:
            emb = emb[: self.max_residues]
        return emb

    # ── GearBind (v7) structure / conformer graph loaders ───────────────────

    def _backbone_fallback_graph(self, n_res: int):
        """Sequence-only backbone graph (self + i±1, relation 0) for `n_res`
        residues — used when a structure cache entry is missing or genuinely
        non-cap-mismatched. Keeps the chain connected so the GNN never NaNs."""
        src: list[int] = []; dst: list[int] = []
        typ: list[int] = []; dist: list[float] = []; sep: list[int] = []
        for i in range(n_res):
            for j in (i - 1, i, i + 1):
                if 0 <= j < n_res:
                    src.append(i); dst.append(j); typ.append(0)
                    dist.append(0.0); sep.append(i - j)
        ei = torch.tensor([src, dst], dtype=torch.long)
        et = torch.tensor(typ, dtype=torch.long)
        ed = torch.tensor(dist, dtype=torch.float32)
        es = torch.tensor(sep, dtype=torch.long)
        return edges_to_neighbors(ei, et, ed, es, n_res, self.protein_k_cap)

    def load_structure(self, uniprot: str, n_res: int) -> dict:
        """Protein structure graph aligned 1:1 to the `n_res` ESM2 rows actually
        loaded (the cache is already ESM2-aligned / ≤1024-cropped; we assert-clip
        defensively). Missing file or a true (non-cap) length mismatch falls back
        to a backbone-only graph. Returns the per-item ragged neighbor format
        (see `edges_to_neighbors`) plus per-node `plddt` (= cache pLDDT / 100)."""
        p = (self.structure_dir / f"{uniprot}.pt") if self.structure_dir else None
        if p is None or not p.exists():
            idx, typ, dist, sep, cnt = self._backbone_fallback_graph(n_res)
            return {
                "plddt": torch.full((n_res,), 0.5, dtype=torch.float32),
                "nbr_idx": idx, "nbr_type": typ, "nbr_dist": dist,
                "nbr_seqsep": sep, "nbr_count": cnt,
                "n_res": int(n_res), "structure_present": False,
            }
        d = torch.load(p, weights_only=True)
        s_n = int(d["n_res"])
        ei = d["edge_index"].long(); et = d["edge_type"].long()
        ed = d["edge_dist"].float(); es = d["edge_seqsep"].long()
        plddt = d["plddt"].float() / 100.0
        if s_n != n_res:
            if s_n > n_res:
                # cap-crop in lockstep: keep first n_res residues, drop edges
                # touching the cropped tail.
                keep = (ei[0] < n_res) & (ei[1] < n_res)
                ei = ei[:, keep]; et = et[keep]; ed = ed[keep]; es = es[keep]
                plddt = plddt[:n_res]
            else:
                # structure shorter than ESM2 -> true mismatch -> backbone fallback.
                idx, typ, dist, sep, cnt = self._backbone_fallback_graph(n_res)
                return {
                    "plddt": torch.full((n_res,), 0.5, dtype=torch.float32),
                    "nbr_idx": idx, "nbr_type": typ, "nbr_dist": dist,
                    "nbr_seqsep": sep, "nbr_count": cnt,
                    "n_res": int(n_res), "structure_present": False,
                }
        idx, typ, dist, sep, cnt = edges_to_neighbors(
            ei, et, ed, es, n_res, self.protein_k_cap
        )
        return {
            "plddt": plddt, "nbr_idx": idx, "nbr_type": typ, "nbr_dist": dist,
            "nbr_seqsep": sep, "nbr_count": cnt,
            "n_res": int(n_res), "structure_present": True,
        }

    def _fallback_ligand_graph(self, chemberta_mean: torch.Tensor) -> dict:
        """1-node ligand graph (missing conformer / parse failure)."""
        node_feat = torch.cat(
            [torch.zeros(1, 39, dtype=torch.float32),
             chemberta_mean.unsqueeze(0).to(torch.float32)], dim=1
        )                                                          # [1, 423]
        return {
            "node_feat": node_feat,
            "nbr_idx": torch.zeros(1, 1, dtype=torch.long),
            "nbr_type": torch.zeros(1, 1, dtype=torch.long),
            "nbr_dist": torch.zeros(1, 1, dtype=torch.float32),
            "nbr_count": torch.ones(1, dtype=torch.long),
            "n_atoms": 1, "conf_ok": False,
        }

    def load_ligand_graph(self, smiles: str) -> dict:
        """3D ligand atom graph from the conformer cache, keyed by `_smiles_key`.

        Unified 3-relation edge set: 0=bond (all bond subtypes collapsed,
        dist=0), 1=spatial-contact (<4.5 Å), 2=spatial-kNN8. Node features =
        `atom_feat[A,39]` ⊕ mean-pool ChemBERTa[384] broadcast to every atom
        → [A, 423]. Missing/unparsable conformer → 1-node fallback."""
        chemberta_mean = load_chemberta(smiles, self.chemberta_cache_dir).to(torch.float32)
        p = (self.ligand_conformer_dir / f"{_smiles_key(smiles)}.pt"
             if self.ligand_conformer_dir else None)
        if p is None or not p.exists():
            return self._fallback_ligand_graph(chemberta_mean)
        d = torch.load(p, weights_only=True)
        if not bool(d.get("parse_ok", False)):
            return self._fallback_ligand_graph(chemberta_mean)
        A = int(d["n_atoms"])
        atom_feat = d["atom_feat"].float()                         # [A, 39]
        bond_index = d["bond_index"].long()                        # [2, Eb]
        spatial_index = d["spatial_index"].long()                  # [2, Es]
        spatial_type = d["spatial_type"].long()                    # 1 or 2
        spatial_dist = d["spatial_dist"].float()                   # [Es]
        Eb = int(bond_index.shape[1])
        ei = torch.cat([bond_index, spatial_index], dim=1)
        et = torch.cat([torch.zeros(Eb, dtype=torch.long), spatial_type], dim=0)
        ed = torch.cat([torch.zeros(Eb, dtype=torch.float32), spatial_dist], dim=0)
        idx, typ, dist, _sep, cnt = edges_to_neighbors(
            ei, et, ed, None, A, self.ligand_k_cap
        )
        node_feat = torch.cat(
            [atom_feat, chemberta_mean.unsqueeze(0).expand(A, -1)], dim=1
        )                                                          # [A, 423]
        return {
            "node_feat": node_feat,
            "nbr_idx": idx, "nbr_type": typ, "nbr_dist": dist,
            "nbr_count": cnt, "n_atoms": A, "conf_ok": bool(d.get("conf_ok", False)),
        }

    def __getitem__(self, i: int) -> dict:
        row = self.pairs.iloc[i]
        if self.ligand_encoder == "per_token":
            lig = load_chemberta_tokens(
                row["substrate_smiles"], self.chemberta_token_cache_dir,
                max_tokens=self.max_ligand_tokens,
            )                                       # [A, lig_input_dim]
        else:
            lig = load_chemberta(row["substrate_smiles"], self.chemberta_cache_dir)  # [D]
        prot = self.load_protein(row["uniprot"])
        item = {
            "smiles":   row["substrate_smiles"],
            "uniprot":  row["uniprot"],
            "label":    float(row["label"]),
            "lig_emb":  lig.to(torch.float32),
            "prot_emb": prot,
            "pair_idx": int(row["idx"]),
        }
        if self.protein_encoder == "attn_pool":
            item["prot_len"] = int(prot.shape[0])
        if self.ligand_encoder == "per_token":
            item["lig_len"] = int(lig.shape[0])
        if self.load_graphs:
            # gearbind (v7): carry the per-residue structure graph (aligned to the
            # ESM2 length actually loaded) and the 3D ligand atom graph.
            item["prot_graph"] = self.load_structure(row["uniprot"], int(prot.shape[0]))
            item["lig_graph"] = self.load_ligand_graph(row["substrate_smiles"])
        return item


def _pad_residues(
    residues: list[torch.Tensor],
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of `[L_i, D]` tensors to `[B, max_L, D]` and produce a mask
    `[B, max_L]` (True = real residue). Lengths must match `[r.shape[0] for r in residues]`."""
    B = len(residues)
    max_L = max(lengths)
    D = residues[0].shape[-1]
    out = residues[0].new_zeros(B, max_L, D)
    mask = torch.zeros(B, max_L, dtype=torch.bool)
    for i, (r, L) in enumerate(zip(residues, lengths)):
        out[i, :L] = r
        mask[i, :L] = True
    return out, mask


# ──────────────────────────────────────────────────────────────────────────────
# GearBind (v7) — structure-graph neighbor format (gather-over-Kmax, no scatter)
# ──────────────────────────────────────────────────────────────────────────────

def edges_to_neighbors(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_dist: torch.Tensor,
    edge_seqsep_or_none: torch.Tensor | None,
    n_nodes: int,
    k_cap: int,
    type_priority: tuple[int, ...] = (0, 1, 2),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert an edge list into a per-node padded neighbor format.

    Groups columns by ``edge_index[0]`` (the node) and lays out each node's
    neighbors along a fixed ``Kg`` axis, where ``Kg`` is the max retained degree
    in this graph (after capping). Returns
        nbr_idx   [N, Kg] long   neighbor node indices (``edge_index[1]``)
        nbr_type  [N, Kg] long   relation type of each retained edge
        nbr_dist  [N, Kg] float  edge distance (Å; 0 for self / bond)
        nbr_seqsep[N, Kg] long   signed seq-separation (zeros if ``…or_none`` None)
        nbr_count [N]     long   number of real neighbors per node

    When a node's degree exceeds ``k_cap``, neighbors are kept by relation
    priority: all of ``type_priority[0]``, then ``type_priority[1]``, ... and the
    overflow of the lowest-priority retained relation is truncated. Padded
    columns (``>= nbr_count[n]``) carry zeros and must be masked downstream.

    Pure torch + a per-node Python pass — no ``torch_scatter`` / ``torch_cluster``.
    """
    edge_index = edge_index.long()
    edge_type = edge_type.long()
    edge_dist = edge_dist.float()
    has_sep = edge_seqsep_or_none is not None
    seqsep = edge_seqsep_or_none.long() if has_sep else None

    prio = {t: r for r, t in enumerate(type_priority)}
    fallback_prio = len(type_priority)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    et = edge_type.tolist()

    per_node: list[list[int]] = [[] for _ in range(n_nodes)]
    for e, s in enumerate(src):
        if 0 <= s < n_nodes:
            per_node[s].append(e)

    kept: list[list[int]] = []
    for n in range(n_nodes):
        es = per_node[n]
        if len(es) > k_cap:
            es = sorted(es, key=lambda e: prio.get(et[e], fallback_prio))[:k_cap]
        kept.append(es)

    Kg = max(1, max((len(k) for k in kept), default=1))
    nbr_idx = torch.zeros(n_nodes, Kg, dtype=torch.long)
    nbr_type = torch.zeros(n_nodes, Kg, dtype=torch.long)
    nbr_dist = torch.zeros(n_nodes, Kg, dtype=torch.float32)
    nbr_seqsep = torch.zeros(n_nodes, Kg, dtype=torch.long)
    nbr_count = torch.zeros(n_nodes, dtype=torch.long)

    dist_list = edge_dist.tolist()
    sep_list = seqsep.tolist() if has_sep else None
    for n, es in enumerate(kept):
        nbr_count[n] = len(es)
        for j, e in enumerate(es):
            nbr_idx[n, j] = dst[e]
            nbr_type[n, j] = et[e]
            nbr_dist[n, j] = dist_list[e]
            if has_sep:
                nbr_seqsep[n, j] = sep_list[e]
    return nbr_idx, nbr_type, nbr_dist, nbr_seqsep, nbr_count


def collate_graph_list(
    graphs: list[dict],
    has_seqsep: bool,
    node_feat_keys: list[str],
) -> dict:
    """Pad a list of per-item ragged neighbor-graphs into a batch.

    Each item dict carries ``nbr_idx/nbr_type/nbr_dist [Ni, Ki]``, ``nbr_count
    [Ni]`` (+ ``nbr_seqsep`` when ``has_seqsep``) and the node-feature arrays
    named in ``node_feat_keys`` (1-D ``[Ni]`` like pLDDT, or 2-D ``[Ni, F]`` like
    the ligand node features). Produces, with ``N = max Ni`` and ``K = max Ki``:
        nbr_idx/nbr_type [B, N, K] long,  nbr_dist [B, N, K] float,
        nbr_seqsep [B, N, K] long (optional),  nbr_mask [B, N, K] bool,
        node_mask [B, N] bool, and each node-feature key padded to [B, N(, F)].
    """
    B = len(graphs)
    maxN = max(int(g["nbr_idx"].shape[0]) for g in graphs)
    maxK = max(int(g["nbr_idx"].shape[1]) for g in graphs)
    out: dict = {
        "nbr_idx":  torch.zeros(B, maxN, maxK, dtype=torch.long),
        "nbr_type": torch.zeros(B, maxN, maxK, dtype=torch.long),
        "nbr_dist": torch.zeros(B, maxN, maxK, dtype=torch.float32),
        "nbr_mask": torch.zeros(B, maxN, maxK, dtype=torch.bool),
        "node_mask": torch.zeros(B, maxN, dtype=torch.bool),
    }
    if has_seqsep:
        out["nbr_seqsep"] = torch.zeros(B, maxN, maxK, dtype=torch.long)
    for key in node_feat_keys:
        sample = graphs[0][key]
        if sample.ndim == 1:
            out[key] = torch.zeros(B, maxN, dtype=torch.float32)
        else:
            out[key] = torch.zeros(B, maxN, int(sample.shape[1]), dtype=torch.float32)

    arK = torch.arange(maxK)
    for b, g in enumerate(graphs):
        Ni = int(g["nbr_idx"].shape[0]); Ki = int(g["nbr_idx"].shape[1])
        out["nbr_idx"][b, :Ni, :Ki] = g["nbr_idx"]
        out["nbr_type"][b, :Ni, :Ki] = g["nbr_type"]
        out["nbr_dist"][b, :Ni, :Ki] = g["nbr_dist"]
        if has_seqsep:
            out["nbr_seqsep"][b, :Ni, :Ki] = g["nbr_seqsep"]
        cnt = g["nbr_count"].long()
        out["nbr_mask"][b, :Ni, :Ki] = arK[:Ki].unsqueeze(0) < cnt.unsqueeze(1)
        out["node_mask"][b, :Ni] = True
        for key in node_feat_keys:
            arr = g[key]
            if arr.ndim == 1:
                out[key][b, :Ni] = arr.float()
            else:
                out[key][b, :Ni, :int(arr.shape[1])] = arr.float()
    return out


def collate_pointwise(batch: list[dict]) -> dict:
    """Pointwise collator. Detects per-residue mode automatically by checking
    whether items carry a `prot_len` key.
    """
    is_attn = "prot_len" in batch[0]
    is_lig_tokens = "lig_len" in batch[0]
    out = {
        "smiles":   [b["smiles"] for b in batch],
        "uniprot":  [b["uniprot"] for b in batch],
        "label":    torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "pair_idx": torch.tensor([b["pair_idx"] for b in batch], dtype=torch.long),
    }
    if is_lig_tokens:
        lig_tok = [b["lig_emb"] for b in batch]
        lig_lens = [b["lig_len"] for b in batch]
        lig, lig_mask = _pad_residues(lig_tok, lig_lens)   # [B, A, D], [B, A]
        out["lig_emb"] = lig
        out["lig_mask"] = lig_mask
    else:
        out["lig_emb"] = torch.stack([b["lig_emb"] for b in batch])
    if is_attn:
        residues = [b["prot_emb"] for b in batch]
        lengths = [b["prot_len"] for b in batch]
        prot, mask = _pad_residues(residues, lengths)
        out["prot_emb"] = prot
        out["prot_mask"] = mask
        out["prot_len"] = torch.tensor(lengths, dtype=torch.long)
    else:
        out["prot_emb"] = torch.stack([b["prot_emb"] for b in batch])
    # gearbind (v7): batch the structure + ligand graphs (padded neighbor tensors
    # + masks). The protein graph node axis is padded to the same max length as
    # prot_emb (both derive from the per-item residue counts), so they align.
    if "prot_graph" in batch[0]:
        out["prot_graph"] = collate_graph_list(
            [b["prot_graph"] for b in batch], has_seqsep=True, node_feat_keys=["plddt"])
        out["lig_graph"] = collate_graph_list(
            [b["lig_graph"] for b in batch], has_seqsep=False, node_feat_keys=["node_feat"])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Loaders from BRENDADataConfig
# ──────────────────────────────────────────────────────────────────────────────

def prepare_frames(config_dict: dict) -> tuple:
    """Return (train_df, val_df, test_df, sequences, bconfig).

    Applies protein-based split. The split is drawn with
    data.split_seed (default 42) so it stays FIXED across training-seed
    sweeps; cfg["seed"] only controls init/shuffling.
    """
    bconfig = BRENDADataConfig(
        seed=int(config_dict["data"].get("split_seed", 42)),
        csv_path=str(PROJECT_ROOT / config_dict["data"]["csv_path"]),
        seq_csv=str(PROJECT_ROOT / config_dict["data"]["seq_csv"]),
        val_frac=config_dict["data"]["val_frac"],
        test_frac=config_dict["data"]["test_frac"],
    )
    pairs = bconfig.load_pairs()
    sequences = bconfig.load_sequences()

    esm2_dir = PROJECT_ROOT / config_dict["data"]["esm2_dir"]
    have_esm = {p.stem for p in esm2_dir.glob("*.pt")}

    # Keep only proteins with sequences AND ESM2 embeddings.
    keep = pairs["uniprot"].isin(sequences) & pairs["uniprot"].isin(have_esm)
    pairs = pairs[keep].reset_index(drop=True)

    split_mode = config_dict["data"].get("split_mode", "protein")
    if split_mode == "random":
        # Transductive ceiling probe only — see common.get_random_split.
        train_idx, val_idx, test_idx = bconfig.get_random_split()
    elif split_mode == "native":
        # Benchmark's own published split (e.g. ESP phylo test vs Kroll).
        train_idx, val_idx, test_idx = bconfig.get_native_split()
    else:
        train_idx, val_idx, test_idx = bconfig.get_protein_split()
    train_idx = set(train_idx); val_idx = set(val_idx); test_idx = set(test_idx)

    train_df = pairs[pairs["idx"].isin(train_idx)].reset_index(drop=True)
    val_df = pairs[pairs["idx"].isin(val_idx)].reset_index(drop=True)
    test_df = pairs[pairs["idx"].isin(test_idx)].reset_index(drop=True)
    return train_df, val_df, test_df, sequences, bconfig


def build_datasets(config_dict: dict, chemberta_cache_dir: Path) -> tuple:
    """Return (train_ds, val_ds, test_ds, split_stats)."""
    train_df, val_df, test_df, sequences, _ = prepare_frames(config_dict)
    esm2_dir = PROJECT_ROOT / config_dict["data"]["esm2_dir"]
    protein_encoder = config_dict["model"].get("protein_encoder", "mean_pool")
    max_residues = config_dict["model"].get("max_residues", 1024)
    ligand_encoder = config_dict["model"].get("ligand_encoder", "mean_pool")
    max_ligand_tokens = config_dict["model"].get("max_ligand_tokens", 128)
    token_cache_dir = None
    if ligand_encoder == "per_token":
        tcd = config_dict["data"]["chemberta_token_cache"]
        token_cache_dir = Path(tcd) if Path(tcd).is_absolute() else PROJECT_ROOT / tcd

    # gearbind (v7): structure + 3D-ligand graphs. The head mandates attn_pool.
    load_graphs = config_dict["model"].get("head") == "gearbind"
    structure_dir = None
    ligand_conformer_dir = None
    if load_graphs:
        protein_encoder = "attn_pool"
        sd = config_dict["data"]["structure_dir"]
        structure_dir = Path(sd) if Path(sd).is_absolute() else PROJECT_ROOT / sd
        lcd = config_dict["data"]["ligand_conformer_dir"]
        ligand_conformer_dir = Path(lcd) if Path(lcd).is_absolute() else PROJECT_ROOT / lcd

    datasets = tuple(
        RankBindDataset(
            pairs=df,
            sequences=sequences,
            esm2_dir=esm2_dir,
            chemberta_cache_dir=chemberta_cache_dir,
            prot_input_dim=config_dict["model"]["prot_input_dim"],
            lig_input_dim=config_dict["model"]["lig_input_dim"],
            protein_encoder=protein_encoder,
            max_residues=max_residues,
            ligand_encoder=ligand_encoder,
            chemberta_token_cache_dir=token_cache_dir,
            max_ligand_tokens=max_ligand_tokens,
            structure_dir=structure_dir,
            ligand_conformer_dir=ligand_conformer_dir,
            load_graphs=load_graphs,
        )
        for df in (train_df, val_df, test_df)
    )
    split_stats = {
        "n_train_pairs":     len(train_df),
        "n_val_pairs":       len(val_df),
        "n_test_pairs":      len(test_df),
        "n_train_proteins":  int(train_df["uniprot"].nunique()),
        "n_val_proteins":    int(val_df["uniprot"].nunique()),
        "n_test_proteins":   int(test_df["uniprot"].nunique()),
        "n_train_positives": int(train_df["label"].sum()),
        "n_val_positives":   int(val_df["label"].sum()),
        "n_test_positives":  int(test_df["label"].sum()),
    }
    return (*datasets, split_stats)
