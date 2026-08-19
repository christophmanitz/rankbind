#!/usr/bin/env python
"""Build the RankBind v7 ligand 3D-conformer-graph cache (RDKit ETKDGv3).

For every unique ``substrate_smiles`` in the BRENDA+SABIO datasets, embed a 3D
conformer and write a heavy-atom graph to ``<out_dir>/<sha1(smiles)[:16]>.pt``.
The key is ``hashlib.sha1(smiles.encode()).hexdigest()[:16]`` — identical to
``v5_rankbind.data._smiles_key`` and the ChemBERTa caches — so the loader finds
the graph for the same raw SMILES string the dataset row carries. Default
out_dir is a /work2 sibling of the ChemBERTa token cache.

Cache entry (per SMILES)::

    {
      'pos':           float16 [A, 3]    # heavy-atom 3D coords (2D for fallback)
      'z':             int16   [A]       # atomic number
      'atom_feat':     float16 [A, F]    # element/degree/charge/hybr/arom/H/ring/3D-env
      'bond_index':    int32   [2, Eb]
      'bond_type':     int8    [Eb]      # 0 single 1 double 2 triple 3 aromatic
      'spatial_index': int32   [2, Es]
      'spatial_type':  int8    [Es]      # 1 contact (<thr A), 2 kNN
      'spatial_dist':  float16 [Es]
      'n_atoms':       int     (= A)
      'conf_ok':       bool               # False -> ETKDG failed, 2D fallback used
      'parse_ok':      bool               # False -> SMILES unparseable, 1-node stub
    }

Heavy atoms only (A <= --max_atoms, default 128; molecules above are truncated
to the first max_atoms atoms, edges past the cap dropped — rare, observed max
heavy-atom count is 115). Idempotent (skips existing, unless --overwrite).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_GLOB = str(
    PROJECT_ROOT / "reactionDataFiltering/data/interim/*/with_decoys.csv"
)
DEFAULT_OUT = Path("/work2/zw93onug-rankbind_bench/ligand_conformer_cache")

# Heavy-element vocabulary (+OTHER). Order is the one-hot layout — never reorder.
_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se", "As"]
_ELEM_IDX = {e: i for i, e in enumerate(_ELEMENTS)}
_HYBR = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
# feature width: elem(len+1) + degree(6) + charge(5) + hybr(len+1) + arom(1) + H(5) + ring(1) + 3Denv(1)
_F = (len(_ELEMENTS) + 1) + 6 + 5 + (len(_HYBR) + 1) + 1 + 5 + 1 + 1


def _smiles_key(smiles: str) -> str:
    return hashlib.sha1(smiles.encode()).hexdigest()[:16]


def _onehot(i: int, n: int) -> list:
    v = [0.0] * n
    if 0 <= i < n:
        v[i] = 1.0
    return v


def _atom_features(mol, spatial_neighbor_count) -> np.ndarray:
    feats = []
    for a in mol.GetAtoms():
        idx = a.GetIdx()
        sym = a.GetSymbol()
        # element one-hot (+OTHER)
        ei = _ELEM_IDX.get(sym, len(_ELEMENTS))
        f = _onehot(ei, len(_ELEMENTS) + 1)
        f += _onehot(min(a.GetDegree(), 5), 6)
        f += _onehot(int(np.clip(a.GetFormalCharge(), -2, 2)) + 2, 5)
        hi = _HYBR.index(str(a.GetHybridization())) if str(a.GetHybridization()) in _HYBR else len(_HYBR)
        f += _onehot(hi, len(_HYBR) + 1)
        f.append(1.0 if a.GetIsAromatic() else 0.0)
        f += _onehot(min(a.GetTotalNumHs(), 4), 5)
        f.append(1.0 if a.IsInRing() else 0.0)
        f.append(float(spatial_neighbor_count[idx]))
        feats.append(f)
    return np.asarray(feats, dtype=np.float32)


_BOND_TYPE = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}


def _bond_edges(mol):
    src, dst, typ = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        t = _BOND_TYPE.get(str(b.GetBondType()), 0)
        src += [i, j]
        dst += [j, i]
        typ += [t, t]
    return (np.asarray([src, dst], dtype=np.int32),
            np.asarray(typ, dtype=np.int8))


def _spatial_edges(pos: np.ndarray, thr: float, knn: int):
    A = pos.shape[0]
    if A <= 1:
        empty_i = np.zeros((2, 0), dtype=np.int32)
        return empty_i, np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float32), np.zeros(A)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    idx = np.arange(A)
    # type 1: contact < thr (excl self)
    ii, jj = np.where((d < thr) & (d > 0))
    s1, d1 = ii, jj
    t1 = np.ones(s1.shape[0], dtype=np.int8)
    # type 2: kNN (excl self)
    k = min(knn, A - 1)
    order = np.argsort(d, axis=1)[:, 1:k + 1]
    s2 = np.repeat(idx, order.shape[1])
    d2 = order.reshape(-1)
    t2 = np.full(s2.shape[0], 2, dtype=np.int8)
    src = np.concatenate([s1, s2]).astype(np.int32)
    dst = np.concatenate([d1, d2]).astype(np.int32)
    typ = np.concatenate([t1, t2])
    dist = d[src, dst].astype(np.float32)
    # per-atom count of contacts within thr (for the 3D-env atom feature)
    neigh = ((d < thr) & (d > 0)).sum(axis=1)
    return np.stack([src, dst], axis=0), typ, dist, neigh


def _stub_entry() -> dict:
    """Unparseable SMILES -> a single zero node, no edges (loader-safe)."""
    return {
        "pos": torch.zeros(1, 3, dtype=torch.float16),
        "z": torch.zeros(1, dtype=torch.int16),
        "atom_feat": torch.zeros(1, _F, dtype=torch.float16),
        "bond_index": torch.zeros(2, 0, dtype=torch.int32),
        "bond_type": torch.zeros(0, dtype=torch.int8),
        "spatial_index": torch.zeros(2, 0, dtype=torch.int32),
        "spatial_type": torch.zeros(0, dtype=torch.int8),
        "spatial_dist": torch.zeros(0, dtype=torch.float16),
        "n_atoms": 1,
        "conf_ok": False,
        "parse_ok": False,
    }


def process_one(task):
    smiles, out_path, opts = task
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            torch.save(_stub_entry(), out_path)
            return (smiles[:60], "parse_fail", 0)

        mh = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cid = AllChem.EmbedMolecule(mh, params)
        conf_ok = True
        if cid < 0:
            cid = AllChem.EmbedMolecule(mh, useRandomCoords=True, randomSeed=42)
        if cid < 0:
            conf_ok = False
            mol3 = mol                       # heavy-atom mol
            AllChem.Compute2DCoords(mol3)
        else:
            try:
                AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
            except Exception:                # noqa: BLE001
                pass
            mol3 = Chem.RemoveHs(mh)

        A = mol3.GetNumAtoms()
        conf = mol3.GetConformer()
        pos = np.asarray([list(conf.GetAtomPosition(i)) for i in range(A)],
                         dtype=np.float32)
        z = np.asarray([a.GetAtomicNum() for a in mol3.GetAtoms()], dtype=np.int64)

        # cap to max_atoms (rare); drop edges referencing dropped atoms
        max_atoms = opts["max_atoms"]
        if A > max_atoms:
            keep = set(range(max_atoms))
            mol3 = Chem.RWMol(mol3)
            for i in reversed(range(max_atoms, A)):
                mol3.RemoveAtom(i)
            mol3 = mol3.GetMol()
            A = max_atoms
            pos = pos[:A]
            z = z[:A]
            conf_clipped = True
        else:
            conf_clipped = False

        sp_index, sp_type, sp_dist, neigh = _spatial_edges(
            pos, opts["spatial_thr"], opts["spatial_knn"])
        atom_feat = _atom_features(mol3, neigh)
        bond_index, bond_type = _bond_edges(mol3)
        if bond_index.size and bond_index.max() >= A:        # safety after cap
            m = (bond_index[0] < A) & (bond_index[1] < A)
            bond_index, bond_type = bond_index[:, m], bond_type[m]

        entry = {
            "pos": torch.from_numpy(pos).to(torch.float16),
            "z": torch.from_numpy(z).to(torch.int16),
            "atom_feat": torch.from_numpy(atom_feat).to(torch.float16),
            "bond_index": torch.from_numpy(np.ascontiguousarray(bond_index)).to(torch.int32),
            "bond_type": torch.from_numpy(np.ascontiguousarray(bond_type)).to(torch.int8),
            "spatial_index": torch.from_numpy(np.ascontiguousarray(sp_index)).to(torch.int32),
            "spatial_type": torch.from_numpy(np.ascontiguousarray(sp_type)).to(torch.int8),
            "spatial_dist": torch.from_numpy(np.ascontiguousarray(sp_dist)).to(torch.float16),
            "n_atoms": int(A),
            "conf_ok": bool(conf_ok),
            "parse_ok": True,
        }
        torch.save(entry, out_path)
        status = "ok" if conf_ok else "conf_fallback_2d"
        if conf_clipped:
            status += "+clipped"
        return (smiles[:60], status, A)
    except Exception as e:                   # noqa: BLE001
        try:
            torch.save(_stub_entry(), out_path)
        except Exception:
            pass
        return (smiles[:60], f"error:{type(e).__name__}:{str(e)[:50]}", 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_glob", type=str, default=DEFAULT_DATA_GLOB,
                    help="glob of dataset CSVs with a substrate_smiles column")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max_atoms", type=int, default=128)
    ap.add_argument("--spatial_thr", type=float, default=4.5)
    ap.add_argument("--spatial_knn", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    smis = set()
    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise SystemExit(f"no dataset CSVs matched {args.data_glob}")
    for f in files:
        df = pd.read_csv(f, usecols=["substrate_smiles"])
        smis.update(df["substrate_smiles"].dropna().astype(str).tolist())
    smiles_list = sorted(smis)
    if args.limit:
        smiles_list = smiles_list[: args.limit]
    print(f"{len(smiles_list)} unique SMILES from {len(files)} files | out: {args.out_dir}")

    opts = dict(max_atoms=args.max_atoms, spatial_thr=args.spatial_thr,
                spatial_knn=args.spatial_knn)
    tasks = []
    for s in smiles_list:
        out_path = args.out_dir / f"{_smiles_key(s)}.pt"
        if out_path.exists() and not args.overwrite:
            continue
        tasks.append((s, str(out_path), opts))
    print(f"{len(tasks)} to build ({len(smiles_list) - len(tasks)} already cached)")

    rows = []
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_one, t) for t in tasks]
            for k, fut in enumerate(as_completed(futs), 1):
                rows.append(fut.result())
                if k % 500 == 0 or k == len(tasks):
                    print(f"  {k}/{len(tasks)}")
    else:
        for k, t in enumerate(tasks, 1):
            rows.append(process_one(t))
            if k % 50 == 0 or k == len(tasks):
                print(f"  {k}/{len(tasks)}")

    cnt = Counter(r[1].split(":")[0].split("+")[0] for r in rows)
    print("status counts:", dict(cnt))
    cov_path = args.out_dir / "ligand_conformer_coverage.csv"
    write_header = not cov_path.exists() or args.overwrite
    with open(cov_path, "w" if write_header else "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["smiles_prefix", "status", "n_atoms"])
        for r in rows:
            w.writerow(r)
    print(f"coverage card -> {cov_path}")


if __name__ == "__main__":
    main()
