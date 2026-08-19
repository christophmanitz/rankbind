"""evaluation/deltafield_pocket_roc.py — v6 Check C: does the DeltaField
residue difference-field `e_res` localize the functional binding pocket?

This is the v6 counterpart to the v5b attention-pool pocket audit
(attn_annotation_scan / attn_pocket_overlap), which found the attention pool
*avoids* the pocket (ROC-AUC ~0.21). That attention is ligand-blind. The
DeltaField `e_res` instead comes from the ligand-conditional cross-coupling
(D = H_coupled - H_free), so it has a chance to actually find the pocket.

Method. For each sampled BRENDA protein:
  1. Pick a TRUE-POSITIVE substrate (label==1) for the protein.
  2. Run ONE coupled `model.forward_field` (B=1) on (that ligand, protein).
  3. Take `e_res[0, :L]` over the real residues.
  4. Score within-protein ROC-AUC of e_res predicting the UniProt
     active/binding-site mask (pocket = active OR binding).

Pre-registered Check C: PASS if mean pocket ROC-AUC > 0.5 (e_res localizes
the pocket, unlike v5b attention).

Outputs (evaluation/attractor_results/):
  - v6_deltafield_pocket_roc.csv          (per protein)
  - v6_deltafield_pocket_roc_summary.csv  (aggregate)
  + stdout summary

Usage:
  python -m evaluation.deltafield_pocket_roc                 # CPU, ~60 proteins
  python -m evaluation.deltafield_pocket_roc --device cuda --n 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v5_rankbind.run_manifest import load_config
from v5_rankbind.model import RankBind
# importing v5_rankbind.data also puts baselines/adapters on sys.path
from v5_rankbind.data import (
    ensure_chemberta_token_cache, load_chemberta_tokens, _pad_residues,
)
from evaluation.attn_weight_inspection import load_residues, sample_proteins
from evaluation.attn_annotation_scan import fetch_uniprot, feat_mask

from common import BRENDADataConfig  # noqa: E402  (path added by v5_rankbind.data)

OUT_DIR = _HERE / "attractor_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = (PROJECT_ROOT / "results" / "v5_rankbind" /
           "20260610-121458_9b1c577943_abl_deltafield_v6_deltafield")


def load_model(device: torch.device) -> tuple[RankBind, dict]:
    manifest = json.loads((RUN_DIR / "manifest.json").read_text())
    cfg = load_config(manifest["config_path"])
    model = RankBind(cfg).to(device)
    model.load_state_dict(
        torch.load(RUN_DIR / "best_model.pt", map_location=device,
                   weights_only=True)
    )
    model.eval()
    return model, cfg


@torch.no_grad()
def compute_e_res(model, cfg, smiles, resi, token_cache, device) -> np.ndarray:
    """One coupled forward_field for (ligand, protein). Returns e_res[:L]."""
    max_atoms = cfg["model"].get("max_ligand_tokens", 128)
    lt = load_chemberta_tokens(smiles, token_cache, max_tokens=max_atoms).to(torch.float32)
    lig_pad, lig_mask = _pad_residues([lt], [lt.shape[0]])         # [1,A,Dl],[1,A]
    L = resi.shape[0]
    pr = resi.unsqueeze(0).to(device)                              # [1,L,Dp]
    pm = torch.ones(1, L, dtype=torch.bool, device=device)
    out = model.forward_field(lig_pad.to(device), lig_mask.to(device), pr, pm)
    return out["e_res"][0, :L].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=60, help="cap number of proteins")
    args = ap.parse_args()
    device = torch.device(args.device)

    model, cfg = load_model(device)
    print(f"[ok] loaded DeltaField model from {RUN_DIR.name}")

    # Positive (protein, substrate) pairs from the unified BRENDA config.
    bconfig = BRENDADataConfig(
        seed=cfg["seed"],
        csv_path=str(PROJECT_ROOT / cfg["data"]["csv_path"]),
        seq_csv=str(PROJECT_ROOT / cfg["data"]["seq_csv"]),
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
    )
    pairs = bconfig.load_pairs()
    pos = pairs[pairs["label"] == 1]
    # uniprot -> first positive substrate SMILES
    pos_smiles = (pos.dropna(subset=["substrate_smiles"])
                     .groupby("uniprot")["substrate_smiles"].first().to_dict())

    proteins = sample_proteins()
    if args.n:
        proteins = proteins[: args.n]
    print(f"[setup] DeltaField pocket-ROC on up to {len(proteins)} proteins")

    token_cache = Path(cfg["data"]["chemberta_token_cache"])
    if not token_cache.is_absolute():
        token_cache = PROJECT_ROOT / token_cache

    rows = []
    for uni in proteins:
        smiles = pos_smiles.get(uni)
        if smiles is None:
            continue
        try:
            resi = load_residues(uni, max_len=cfg["model"].get("max_residues", 1024))
        except Exception as e:
            print(f"  [skip] {uni}: residues {e}")
            continue
        L = resi.shape[0]

        meta = fetch_uniprot(uni)
        if meta is None or not meta["seq"]:
            continue
        if meta["length"] < L:
            print(f"  [len<L] {uni}: seq {meta['length']} < ESM2 {L} — skip")
            continue

        act = feat_mask(meta["features"], "Active site", L)
        bnd = feat_mask(meta["features"], "Binding site", L)
        pocket = ((act + bnd) > 0).astype(int)
        if not (0 < int(pocket.sum()) < L):
            continue   # need a usable annotated pocket

        # Make sure the ligand tokens are cached, then run the field forward.
        ensure_chemberta_token_cache([smiles], token_cache, device=str(device),
                                     max_length=cfg["model"].get("max_ligand_tokens", 128))
        e_res = compute_e_res(model, cfg, smiles, resi, token_cache, device)
        if e_res.shape[0] != L or np.std(e_res) < 1e-12:
            continue

        auc_pocket = float(roc_auc_score(pocket, e_res))
        auc_active = (float(roc_auc_score(act, e_res))
                      if 0 < int(act.sum()) < L else np.nan)
        auc_binding = (float(roc_auc_score(bnd, e_res))
                       if 0 < int(bnd.sum()) < L else np.nan)
        rows.append({
            "uniprot": uni, "L": L,
            "n_active": int(act.sum()), "n_binding": int(bnd.sum()),
            "n_pocket": int(pocket.sum()),
            "auc_pocket": auc_pocket, "auc_active": auc_active,
            "auc_binding": auc_binding,
        })
        print(f"  [ok] {uni}: L={L} n_pocket={int(pocket.sum())} "
              f"auc_pocket={auc_pocket:.3f}")

    if not rows:
        print("[done] no usable proteins."); return

    df = pd.DataFrame(rows)
    per_csv = OUT_DIR / "v6_deltafield_pocket_roc.csv"
    df.to_csv(per_csv, index=False)
    print(f"\n[ok] wrote {per_csv}  ({len(df)} proteins)")

    # ── Aggregate ───────────────────────────────────────────────────────────
    def agg(col: str) -> dict:
        a = df[col].dropna().values
        if len(a) == 0:
            return {"n": 0, "mean": np.nan, "median": np.nan,
                    "frac_above_0.5": np.nan, "wilcoxon_p": np.nan}
        wp = (wilcoxon(a - 0.5).pvalue
              if len(a) >= 6 and np.abs(a - 0.5).sum() > 0 else np.nan)
        return {"n": int(len(a)), "mean": float(a.mean()),
                "median": float(np.median(a)),
                "frac_above_0.5": float((a > 0.5).mean()),
                "wilcoxon_p": float(wp) if wp == wp else np.nan}

    summ_rows = []
    for which, col in [("pocket", "auc_pocket"), ("active", "auc_active"),
                       ("binding", "auc_binding")]:
        s = agg(col); s["target"] = which
        summ_rows.append(s)
    summ = pd.DataFrame(summ_rows)[
        ["target", "n", "mean", "median", "frac_above_0.5", "wilcoxon_p"]]
    summ_csv = OUT_DIR / "v6_deltafield_pocket_roc_summary.csv"
    summ.to_csv(summ_csv, index=False)
    print(f"[ok] wrote {summ_csv}\n")

    print("=== v6 DeltaField e_res → pocket ROC-AUC ===")
    print("  AUC>0.5: e_res ENRICHED on functional residues (finds pocket).")
    print("  AUC<0.5: e_res AVOIDS them (like v5b attention's 0.21).\n")
    print(summ.round(4).to_string(index=False))

    pk = agg("auc_pocket")
    mean_p = pk["mean"]
    print("\n=== Check C verdict ===")
    print(f"  n proteins (pocket)     : {pk['n']}")
    print(f"  mean pocket ROC-AUC     : {mean_p:.4f}")
    print(f"  median pocket ROC-AUC   : {pk['median']:.4f}")
    print(f"  frac proteins AUC > 0.5 : {pk['frac_above_0.5']:.3f}")
    print(f"  Wilcoxon p (AUC vs 0.5) : {pk['wilcoxon_p']:.3g}")
    if mean_p > 0.55:
        print("  -> PASS: e_res LOCALIZES the functional pocket "
              "(ligand-conditional field finds it, unlike v5b attention 0.21).")
    elif mean_p >= 0.45:
        print("  -> NEUTRAL: no clear localization, but no active avoidance "
              "either — still better than v5b's 0.21 avoidance.")
    else:
        print("  -> FAIL: e_res AVOIDS the pocket, like v5b attention (0.21).")


if __name__ == "__main__":
    main()
