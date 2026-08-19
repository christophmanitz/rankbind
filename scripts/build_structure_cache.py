#!/usr/bin/env python
"""Build the RankBind v7 protein structure-graph cache from AlphaFold v6 PDBs.

For every UniProt that has an ESM2 per-residue embedding, parse
``AF-<acc>-F1-model_v6.pdb`` into a residue contact graph aligned 1:1 with the
(1024-capped) ESM2 tensor the model actually loads, and write
``<out_dir>/<acc>.pt``. The cache is keyed by UniProt (structures are
ligand-/dataset-independent), so a single shared store covers every dataset —
point each dataset config's ``structure_dir`` at it (mirrors the ESM2 dedup).

Cache entry (per UniProt)::

    {
      'cb':           float16 [Ln, 3]   # CB coord (CA for GLY / missing CB)
      'plddt':        uint8   [Ln]      # per-residue pLDDT (B-factor col)
      'is_gly_ca':    bool    [Ln]      # True where CA was substituted for CB
      'edge_index':   int32   [2, E]
      'edge_type':    int8    [E]       # 0 self+backbone, 1 contact<thr, 2 kNN
      'edge_dist':    float16 [E]       # CB-CB distance (A); 0 for self loops
      'edge_seqsep':  int16   [E]       # i - j (signed sequence separation)
      'n_res':        int     (= Ln)
      'structure_present': bool         # False -> sequence-only fallback graph
    }

Alignment (ESM2 .pt is already N-terminus-cropped to <=1024 at embed time):
    Le = esm_rows;  Lp = pdb residues.
    exact     : Lp == Le
    cap-crop  : Le == max_residues and Lp >= max_residues  -> crop PDB to [:Le]
    mismatch  : otherwise -> sequence-only fallback (backbone edges only)

pLDDT handling: drop contact/kNN edges where min(pLDDT_i, pLDDT_j) < --plddt_min,
but ALWAYS keep self+backbone edges so the chain stays connected.

Emits ``<out_dir>/structure_coverage.csv`` (the data card) and is idempotent
(skips UniProts whose .pt already exists unless --overwrite).
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STRUCTURES = (
    PROJECT_ROOT
    / "reactionDataFiltering/data/raw/brenda_sabio_2026-04-29/structures"
)
DEFAULT_ESM2_DIRS = [
    PROJECT_ROOT / "reactionDataFiltering/data/interim/kcat_km_brenda_sabio/esm2_embeddings",
    PROJECT_ROOT / "reactionDataFiltering/data/interim/km_brenda_sabio/esm2_embeddings",
    PROJECT_ROOT / "reactionDataFiltering/data/interim/turnover_brenda_sabio/esm2_embeddings",
]


# ── PDB parsing ──────────────────────────────────────────────────────────────

def parse_pdb(pdb_path: str):
    """Return (cb [Lp,3] f32, plddt [Lp] f32, is_gly_ca [Lp] bool) for chain 1.

    CB coordinate per residue; CA substituted when CB is absent (every glycine,
    and any residue missing CB). pLDDT read from the CA B-factor (AlphaFold
    stores the per-residue confidence there, identical across a residue's atoms).
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    model = next(iter(structure))
    cb, plddt, gly = [], [], []
    for chain in model:
        for res in chain:
            if res.id[0] != " ":            # skip hetero / water
                continue
            if "CA" not in res:
                continue
            ca = res["CA"]
            if "CB" in res:
                cb.append(res["CB"].coord)
                gly.append(False)
            else:                            # GLY (no side chain) or missing CB
                cb.append(ca.coord)
                gly.append(True)
            plddt.append(ca.get_bfactor())
        break                                # AlphaFold monomer: first chain only
    return (
        np.asarray(cb, dtype=np.float32),
        np.asarray(plddt, dtype=np.float32),
        np.asarray(gly, dtype=bool),
    )


def esm_rows(esm_path: str) -> int:
    t = torch.load(esm_path, weights_only=True, map_location="cpu")
    n = int(t.shape[0]) if t.ndim == 2 else 1
    del t
    return n


# ── edge construction ────────────────────────────────────────────────────────

def build_edges(cb: np.ndarray, plddt: np.ndarray,
                contact_thr: float, knn: int, plddt_min: float):
    """Three relation types, vectorised. Returns (edge_index, edge_type,
    edge_dist, edge_seqsep) as numpy arrays.

    R0 (type 0): self-loop + sequential backbone (i, i±1) — always kept.
    R1 (type 1): CB-CB distance < contact_thr (excl. backbone), pLDDT-gated.
    R2 (type 2): kNN nearest neighbours by CB distance, pLDDT-gated.
    """
    L = cb.shape[0]
    d = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1)  # [L, L]
    idx = np.arange(L)

    # R0: self + backbone
    s0 = [idx, idx[:-1], idx[1:]]
    t0 = [idx, idx[1:], idx[:-1]]
    src0 = np.concatenate(s0)
    dst0 = np.concatenate(t0)
    typ0 = np.zeros(src0.shape[0], dtype=np.int8)

    # R1: contact < thr, exclude |i-j| <= 1, pLDDT gate
    ii, jj = np.where((d < contact_thr) & (d > 0))
    keep = np.abs(ii - jj) > 1
    ii, jj = ii[keep], jj[keep]
    if ii.size:
        gate = np.minimum(plddt[ii], plddt[jj]) >= plddt_min
        ii, jj = ii[gate], jj[gate]
    src1, dst1 = ii, jj
    typ1 = np.ones(src1.shape[0], dtype=np.int8)

    # R2: kNN (exclude self at argsort position 0), pLDDT gate
    if L > 1:
        order = np.argsort(d, axis=1)[:, 1:knn + 1]           # [L, <=knn]
        ki = np.repeat(idx, order.shape[1])
        kj = order.reshape(-1)
        gate = np.minimum(plddt[ki], plddt[kj]) >= plddt_min
        ki, kj = ki[gate], kj[gate]
    else:
        ki = kj = np.empty(0, dtype=np.int64)
    typ2 = np.full(ki.shape[0], 2, dtype=np.int8)

    src = np.concatenate([src0, src1, ki]).astype(np.int32)
    dst = np.concatenate([dst0, dst1, kj]).astype(np.int32)
    etype = np.concatenate([typ0, typ1, typ2])
    edge_index = np.stack([src, dst], axis=0)
    edge_dist = d[src, dst].astype(np.float32)
    edge_seqsep = (src.astype(np.int32) - dst.astype(np.int32)).astype(np.int16)
    return edge_index, etype, edge_dist, edge_seqsep


def _to_entry(cb, plddt, gly, edges, structure_present: bool) -> dict:
    edge_index, etype, edist, eseq = edges
    return {
        "cb": torch.from_numpy(np.ascontiguousarray(cb)).to(torch.float16),
        "plddt": torch.from_numpy(np.clip(np.round(plddt), 0, 255)).to(torch.uint8),
        "is_gly_ca": torch.from_numpy(np.ascontiguousarray(gly)),
        "edge_index": torch.from_numpy(np.ascontiguousarray(edge_index)).to(torch.int32),
        "edge_type": torch.from_numpy(np.ascontiguousarray(etype)).to(torch.int8),
        "edge_dist": torch.from_numpy(np.ascontiguousarray(edist)).to(torch.float16),
        "edge_seqsep": torch.from_numpy(np.ascontiguousarray(eseq)).to(torch.int16),
        "n_res": int(cb.shape[0]),
        "structure_present": bool(structure_present),
    }


def _fallback_entry(Le: int) -> dict:
    """Sequence-only graph: Le nodes, backbone edges, no geometry."""
    cb = np.zeros((Le, 3), dtype=np.float32)
    plddt = np.zeros(Le, dtype=np.float32)
    gly = np.zeros(Le, dtype=bool)
    edges = build_edges(cb, plddt, contact_thr=-1.0, knn=0, plddt_min=-1.0)
    return _to_entry(cb, plddt, gly, edges, structure_present=False)


# ── per-protein worker ───────────────────────────────────────────────────────

def process_one(task):
    acc, pdb_path, esm_path, out_path, opts = task
    max_res = opts["max_residues"]
    try:
        Le = esm_rows(esm_path)
        if not os.path.exists(pdb_path):
            torch.save(_fallback_entry(Le), out_path)
            return (acc, "missing_pdb", -1, Le, 0)

        cb, plddt, gly = parse_pdb(pdb_path)
        Lp = cb.shape[0]

        if Lp == Le:
            status, Ln = "exact", Le
        elif Le == max_res and Lp >= max_res:
            status, Ln = "cap_crop", max_res
            cb, plddt, gly = cb[:Ln], plddt[:Ln], gly[:Ln]
        else:
            torch.save(_fallback_entry(Le), out_path)
            return (acc, "mismatch", Lp, Le, 0)

        edges = build_edges(cb, plddt, opts["contact_thr"], opts["knn"], opts["plddt_min"])
        entry = _to_entry(cb, plddt, gly, edges, structure_present=True)
        torch.save(entry, out_path)
        return (acc, status, Lp, Le, int(entry["edge_index"].shape[1]))
    except Exception as e:                   # noqa: BLE001 - record, never abort the run
        return (acc, f"error:{type(e).__name__}:{str(e)[:60]}", -1, -1, 0)


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structures_dir", type=Path, default=DEFAULT_STRUCTURES)
    ap.add_argument("--esm2_dirs", type=Path, nargs="+", default=DEFAULT_ESM2_DIRS)
    ap.add_argument("--out_dir", type=Path,
                    default=PROJECT_ROOT / "reactionDataFiltering/data/interim/structure_graphs_shared")
    ap.add_argument("--max_residues", type=int, default=1024)
    ap.add_argument("--contact_thr", type=float, default=8.0)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--plddt_min", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap #proteins (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Union of UniProts that have an ESM2 embedding, mapped to one esm path each.
    esm_path = {}
    for d in args.esm2_dirs:
        if not d.exists():
            print(f"[warn] esm2 dir absent: {d}")
            continue
        for p in d.glob("*.pt"):
            esm_path.setdefault(p.stem, str(p))
    accs = sorted(esm_path)
    if args.limit:
        accs = accs[: args.limit]
    print(f"{len(accs)} UniProts with ESM2 | structures: {args.structures_dir} | out: {args.out_dir}")

    opts = dict(max_residues=args.max_residues, contact_thr=args.contact_thr,
                knn=args.knn, plddt_min=args.plddt_min)
    tasks = []
    for acc in accs:
        out_path = args.out_dir / f"{acc}.pt"
        if out_path.exists() and not args.overwrite:
            continue
        pdb_path = str(args.structures_dir / f"AF-{acc}-F1-model_v6.pdb")
        tasks.append((acc, pdb_path, esm_path[acc], str(out_path), opts))
    print(f"{len(tasks)} to build ({len(accs) - len(tasks)} already cached)")

    rows = []
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, t) for t in tasks]
            for k, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if k % 200 == 0 or k == len(tasks):
                    print(f"  {k}/{len(tasks)}")
    else:
        for k, t in enumerate(tasks, 1):
            rows.append(process_one(t))
            if k % 50 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}")

    # Coverage data card (append-friendly: rewrite full set seen this run + reuse prior on rerun).
    cov_path = args.out_dir / "structure_coverage.csv"
    write_header = not cov_path.exists() or args.overwrite
    mode = "w" if write_header else "a"
    with open(cov_path, mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["uniprot", "status", "n_pdb_res", "n_esm_res", "n_edges"])
        for r in rows:
            w.writerow(r)

    from collections import Counter
    cnt = Counter(r[1].split(":")[0] for r in rows)
    print("status counts:", dict(cnt))
    print(f"coverage card -> {cov_path}")


if __name__ == "__main__":
    main()
