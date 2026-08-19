"""evaluation/attn_trough_pocket_proximity.py

Question (from the attention-track plots): the annotated binding sites sit at
attention *troughs* (local minima), but not every trough is an annotated
binding site. Are those OTHER troughs nevertheless spatially close (in 3D) to
the real binding sites? If yes, the attention is suppressing the whole 3D
pocket region, not just the annotated residues.

Method (per protein with >=1 UniProt binding site + an AlphaFold structure):
  1. attention per residue = cross-seed-mean already stored in residues_long.csv
  2. troughs = local minima of the attention track (min within +/-W) that are
     below the protein-mean attention (attn_z < 0) -- i.e. as low as the marked
     binding sites, which also sit low.
  3. "other troughs" = troughs that are NOT within +/-NEIGH residues (sequence)
     of any annotated binding site.
  4. 3D: CA coordinates from AF-<UNIPROT>-F1-model_v6.pdb (resnum = seqpos+1).
     For every residue compute min CA-CA distance to ANY binding-site residue.
  5. Compare the min-distance distribution of "other troughs" against a null:
     all residues that are neither binding nor trough (random background).
     Mann-Whitney U + fraction within 8/10/12 A.

Outputs (evaluation/attractor_results/):
  - attn_trough_pocket_proximity.csv       per-residue long table (classified)
  - attn_trough_pocket_proximity_summary.csv   per-protein + pooled stats
  + stdout summary
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from Bio.PDB import PDBParser

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
OUT_DIR = _HERE / "attractor_results"
STRUCT_DIR = Path.home() / "hpc_residue_level" / "data" / "structures"
RES_CSV = OUT_DIR / "attn_annotation_residues.csv"

WINDOW = 4        # local-minimum half-window (residues)
NEIGH = 2         # a trough within +/-NEIGH of a binding site counts as "the site"
THRESH = (8.0, 10.0, 12.0)   # angstrom proximity thresholds reported


def load_ca_coords(uni: str) -> dict[int, np.ndarray]:
    """resnum (1-indexed, == UniProt seq pos) -> CA xyz."""
    pdb = STRUCT_DIR / f"AF-{uni}-F1-model_v6.pdb"
    if not pdb.exists():
        return {}
    s = PDBParser(QUIET=True).get_structure(uni, str(pdb))
    out = {}
    for res in s[0].get_residues():
        if "CA" in res:
            out[res.id[1]] = res["CA"].coord.astype(float)
    return out


def find_troughs(attn: np.ndarray, attn_z: np.ndarray) -> np.ndarray:
    """Boolean mask of local minima (min within +/-WINDOW) that are below mean."""
    L = len(attn)
    mask = np.zeros(L, dtype=bool)
    for i in range(L):
        lo, hi = max(0, i - WINDOW), min(L, i + WINDOW + 1)
        if attn[i] == attn[lo:hi].min() and attn_z[i] < 0:
            mask[i] = True
    return mask


def main() -> None:
    res = pd.read_csv(RES_CSV)
    per_res_rows, prot_rows = [], []
    pooled_trough_d, pooled_null_d = [], []

    for uni, sub in res.groupby("uniprot"):
        sub = sub.sort_values("pos").reset_index(drop=True)
        if sub["is_binding"].sum() < 1:
            continue
        coords = load_ca_coords(uni)
        if not coords:
            continue

        pos = sub["pos"].to_numpy()                 # 0-indexed seq position
        resnum = pos + 1                            # PDB residue number
        has_xyz = np.array([r in coords for r in resnum])
        if has_xyz.sum() < 5:
            continue

        attn = sub["attn"].to_numpy()
        attn_z = sub["attn_z"].to_numpy()
        is_bind = sub["is_binding"].to_numpy().astype(bool)
        troughs = find_troughs(attn, attn_z)

        # binding-site CA coords
        bind_xyz = np.array([coords[r] for r, b in zip(resnum, is_bind)
                             if b and r in coords])
        if len(bind_xyz) == 0:
            continue

        # min CA distance of every residue to any binding-site residue
        min_d = np.full(len(sub), np.nan)
        for i, (r, ok) in enumerate(zip(resnum, has_xyz)):
            if not ok:
                continue
            d = np.linalg.norm(bind_xyz - coords[r], axis=1)
            min_d[i] = float(d.min())

        # "other troughs": trough, not binding, not within NEIGH (seq) of a site
        bind_pos = set(pos[is_bind])
        near_site_seq = np.array([
            any(abs(int(p) - bp) <= NEIGH for bp in bind_pos) for p in pos])
        other_trough = troughs & ~is_bind & ~near_site_seq & has_xyz
        # null background: not binding, not a trough, has coords
        background = ~is_bind & ~troughs & has_xyz

        for i in range(len(sub)):
            cls = ("binding_site" if is_bind[i]
                   else "other_trough" if other_trough[i]
                   else "trough_near_site" if (troughs[i] and not background[i])
                   else "background" if background[i]
                   else "other")
            per_res_rows.append({
                "uniprot": uni, "pos": int(pos[i]), "aa": sub["aa"].iloc[i],
                "attn_z": float(attn_z[i]), "is_trough": bool(troughs[i]),
                "is_binding": bool(is_bind[i]), "klass": cls,
                "min_dist_to_binding_A": min_d[i],
            })

        td = min_d[other_trough]; nd = min_d[background]
        td = td[~np.isnan(td)]; nd = nd[~np.isnan(nd)]
        pooled_trough_d += td.tolist(); pooled_null_d += nd.tolist()
        prot_rows.append({
            "uniprot": uni, "n_binding": int(is_bind.sum()),
            "n_other_troughs": int(len(td)), "n_background": int(len(nd)),
            "med_dist_other_trough_A": float(np.median(td)) if len(td) else np.nan,
            "med_dist_background_A": float(np.median(nd)) if len(nd) else np.nan,
        })

    prdf = pd.DataFrame(per_res_rows)
    prdf.to_csv(OUT_DIR / "attn_trough_pocket_proximity.csv", index=False)
    pdf = pd.DataFrame(prot_rows)

    td = np.array(pooled_trough_d); nd = np.array(pooled_null_d)
    print(f"\n=== Trough -> binding-site 3D proximity (pooled over "
          f"{len(pdf)} proteins) ===")
    print(f"  other troughs (non-binding local minima): n={len(td)}")
    print(f"  background residues (non-binding, non-trough): n={len(nd)}")
    print(f"\n  median min-distance to nearest binding site:")
    print(f"    other troughs : {np.median(td):6.2f} A")
    print(f"    background    : {np.median(nd):6.2f} A")
    for t in THRESH:
        print(f"  within {t:4.0f} A of a binding site:  "
              f"other troughs {100*np.mean(td <= t):5.1f}%  |  "
              f"background {100*np.mean(nd <= t):5.1f}%")
    if len(td) >= 5 and len(nd) >= 5:
        u, p = mannwhitneyu(td, nd, alternative="less")
        print(f"\n  Mann-Whitney U (other troughs closer than background): "
              f"p = {p:.2e}")
        print("  -> " + ("troughs ARE significantly closer to the pocket"
                          if p < 0.05 else
                          "no significant 3D clustering near the pocket"))

    # pooled row appended to summary
    pooled = {"uniprot": "POOLED", "n_binding": int(pdf["n_binding"].sum()),
              "n_other_troughs": len(td), "n_background": len(nd),
              "med_dist_other_trough_A": float(np.median(td)) if len(td) else np.nan,
              "med_dist_background_A": float(np.median(nd)) if len(nd) else np.nan}
    pd.concat([pdf, pd.DataFrame([pooled])], ignore_index=True).to_csv(
        OUT_DIR / "attn_trough_pocket_proximity_summary.csv", index=False)
    print(f"\n[ok] wrote attn_trough_pocket_proximity{{,_summary}}.csv")


if __name__ == "__main__":
    main()
