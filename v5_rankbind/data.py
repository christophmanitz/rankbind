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
    ):
        if protein_encoder not in ("mean_pool", "attn_pool"):
            raise ValueError(
                f"Unknown protein_encoder={protein_encoder!r}; "
                "expected 'mean_pool' or 'attn_pool'."
            )
        self.pairs = pairs.reset_index(drop=True)
        self.sequences = sequences
        self.esm2_dir = Path(esm2_dir)
        self.chemberta_cache_dir = Path(chemberta_cache_dir)
        self.prot_input_dim = prot_input_dim
        self.lig_input_dim = lig_input_dim
        self.protein_encoder = protein_encoder
        self.max_residues = int(max_residues)

    def __len__(self) -> int:
        return len(self.pairs)

    def protein_at(self, i: int) -> str:
        return self.pairs.at[i, "uniprot"]

    def smiles_at(self, i: int) -> str:
        return self.pairs.at[i, "substrate_smiles"]

    def label_at(self, i: int) -> int:
        return int(self.pairs.at[i, "label"])

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

    def __getitem__(self, i: int) -> dict:
        row = self.pairs.iloc[i]
        lig = load_chemberta(row["substrate_smiles"], self.chemberta_cache_dir)
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


def collate_pointwise(batch: list[dict]) -> dict:
    """Pointwise collator. Detects per-residue mode automatically by checking
    whether items carry a `prot_len` key.
    """
    is_attn = "prot_len" in batch[0]
    out = {
        "smiles":   [b["smiles"] for b in batch],
        "uniprot":  [b["uniprot"] for b in batch],
        "label":    torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "lig_emb":  torch.stack([b["lig_emb"] for b in batch]),
        "pair_idx": torch.tensor([b["pair_idx"] for b in batch], dtype=torch.long),
    }
    if is_attn:
        residues = [b["prot_emb"] for b in batch]
        lengths = [b["prot_len"] for b in batch]
        prot, mask = _pad_residues(residues, lengths)
        out["prot_emb"] = prot
        out["prot_mask"] = mask
        out["prot_len"] = torch.tensor(lengths, dtype=torch.long)
    else:
        out["prot_emb"] = torch.stack([b["prot_emb"] for b in batch])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Loaders from BRENDADataConfig
# ──────────────────────────────────────────────────────────────────────────────

def prepare_frames(config_dict: dict) -> tuple:
    """Return (train_df, val_df, test_df, sequences, bconfig).

    Applies protein-based split (seed=42), drops proteins without ESM2 or
    without a sequence entry.
    """
    bconfig = BRENDADataConfig(
        seed=config_dict["seed"],
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
