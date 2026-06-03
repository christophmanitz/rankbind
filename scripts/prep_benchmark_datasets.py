#!/usr/bin/env python
"""scripts/prep_benchmark_datasets.py — download + convert external DTI
benchmarks into the v5_rankbind loader schema.

Outputs per benchmark under
``reactionDataFiltering/data/interim/benchmarks/<name>/``:
  - pairs.csv      columns: uniprot, substrate_smiles, label  (loader schema)
  - sequences.csv  columns: uniprot, sequence, length
  - prep_card.json provenance + class balance + binarisation choice

The project's protein-based split (BRENDADataConfig, seed 42) is applied
later by the loader; this script emits the FULL pair list. Binarisation
thresholds are the DeepDTA / DeepPurpose conventions and are recorded in
prep_card.json (see docs/BENCHMARK_INTEGRATION_PLAN.md).

Protein ids: TDC provides amino-acid sequences but no UniProt accessions, so
we synthesise a deterministic id  ``<name>_<sha1(sequence)[:10]>`` which also
dedups identical sequences.

Run in the isolated prep venv:
    source ~/venvs/tdcprep/bin/activate
    python scripts/prep_benchmark_datasets.py davis
    python scripts/prep_benchmark_datasets.py kiba bindingdb_kd
    python scripts/prep_benchmark_datasets.py esp        # no TDC, github/zenodo
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "reactionDataFiltering/data/interim/benchmarks"

# (tdc_name, binarise spec). 'log' converts nM affinity to pK (-log10(M)).
TDC_SPECS = {
    "davis":        dict(tdc="DAVIS",        log=True,  threshold=7.0,  ge=True),
    "kiba":         dict(tdc="KIBA",         log=False, threshold=12.1, ge=True),
    "bindingdb_kd": dict(tdc="BindingDB_Kd", log=True,  threshold=7.0,  ge=True),
}


def _pid(name: str, seq: str) -> str:
    return f"{name}_{hashlib.sha1(seq.encode()).hexdigest()[:10]}"


def _write(name: str, df_pairs: pd.DataFrame, seqs: dict, card: dict) -> None:
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(out / "pairs.csv", index=False)
    seq_df = pd.DataFrame(
        [{"uniprot": u, "sequence": s, "length": len(s)} for u, s in seqs.items()]
    )
    seq_df.to_csv(out / "sequences.csv", index=False)
    card.update(
        n_pairs=int(len(df_pairs)),
        n_proteins=int(df_pairs["uniprot"].nunique()),
        n_ligands=int(df_pairs["substrate_smiles"].nunique()),
        positive_rate=float(df_pairs["label"].mean()),
    )
    (out / "prep_card.json").write_text(json.dumps(card, indent=2))
    print(f"[{name}] {card['n_pairs']} pairs / {card['n_proteins']} prot / "
          f"{card['n_ligands']} lig / pos={card['positive_rate']:.3f} -> {out}")


def prep_tdc(name: str) -> None:
    spec = TDC_SPECS[name]
    from tdc.multi_pred import DTI
    data = DTI(name=spec["tdc"], path=str(OUT_ROOT / name / "_tdc_raw"))
    if spec["log"]:
        data.convert_to_log(form="binding")   # Kd[nM] -> pKd
    df = data.get_data().rename(
        columns={"Drug": "substrate_smiles", "Target": "_seq", "Y": "_y"}
    )[["substrate_smiles", "_seq", "_y"]].dropna()
    df = df[df["_seq"].str.len() > 0].copy()
    df["label"] = ((df["_y"] >= spec["threshold"]) if spec["ge"]
                   else (df["_y"] <= spec["threshold"])).astype(int)
    df["uniprot"] = [_pid(name, s) for s in df["_seq"]]
    seqs = dict(zip(df["uniprot"], df["_seq"]))
    pairs = (df[["uniprot", "substrate_smiles", "label"]]
             .drop_duplicates(subset=["uniprot", "substrate_smiles"]))
    _write(name, pairs, seqs, card=dict(
        source=f"TDC DTI('{spec['tdc']}')",
        binarisation=f"{'pK' if spec['log'] else 'score'} "
                     f"{'>=' if spec['ge'] else '<='} {spec['threshold']}",
        split="project protein-split (BRENDADataConfig, seed 42) applied by loader",
    ))


ESP_RAW = "https://raw.githubusercontent.com/AlexanderKroll/ESP/main/data"
ESP_FILES = {  # the non-LFS CSVs (real data); .pkl siblings are LFS pointers
    "train": f"{ESP_RAW}/enzyme_substrate_data/df_UID_MID_train_exp_1_1.csv",
    "test":  f"{ESP_RAW}/enzyme_substrate_data/df_UID_MID_test_exp_phylo_1_1.csv",
    "chebi": f"{ESP_RAW}/substrate_data/chebiID_to_inchi.tsv",
}


def prep_esp(name: str = "esp") -> None:
    """ESP (Kroll et al. 2023, Nat. Commun.). The enzyme-substrate CSVs carry
    real UniProt IDs + enzyme Sequence + a CHEBI substrate id + binary
    `outcome`, but NO SMILES (molecules are CHEBI id + ECFP). We resolve
    CHEBI -> InChI (chebiID_to_inchi.tsv) -> SMILES via RDKit.

    Run in the MAIN venv (needs rdkit): ~/venvs/hieratombind.
    The ESP phylo test split is preserved in a `split` column so the loader /
    eval can honour ESP's own split when comparing to their reported numbers.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    import re

    def _norm_chebi(x) -> str | None:
        """Any of 'CHEBI:57925', 'ChEBI:57925', '57925', 57925.0 -> '57925'."""
        s = re.sub(r"(?i)chebi:?", "", str(x)).strip()
        try:
            return str(int(float(s)))
        except (ValueError, TypeError):
            return None

    # CHEBI -> SMILES map. The tsv has an 'Input' col ('ChEBI:132511'), a bare
    # 'ChEBI' col ('11805', NaN where unknown), and an 'Inchi' col.
    chebi = pd.read_csv(ESP_FILES["chebi"], sep="\t")
    inchi_col = [c for c in chebi.columns if c.lower() == "inchi"][0]
    id_col = "Input" if "Input" in chebi.columns else \
             [c for c in chebi.columns if "chebi" in c.lower()][0]
    c2s = {}
    for cid, inchi in zip(chebi[id_col], chebi[inchi_col]):
        key = _norm_chebi(cid)
        if key is None or not isinstance(inchi, str) or not inchi.startswith("InChI"):
            continue
        try:
            m = Chem.MolFromInchi(inchi)
            if m is not None:
                c2s[key] = Chem.MolToSmiles(m)
        except Exception:
            pass

    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(ESP_FILES[split])
        df = df.rename(columns={"Uniprot ID": "uniprot", "Sequence": "_seq",
                                "substrate ID": "_chebi", "outcome": "label"})
        df["_chebi"] = df["_chebi"].map(_norm_chebi)
        df["substrate_smiles"] = df["_chebi"].map(c2s)
        df["split"] = split
        frames.append(df[["uniprot", "_seq", "substrate_smiles", "label", "split"]])
    full = pd.concat(frames, ignore_index=True)
    n_pre = len(full)
    full = full.dropna(subset=["substrate_smiles", "_seq"])
    full = full[full["_seq"].str.len() > 0]
    dropped = n_pre - len(full)

    seqs = dict(zip(full["uniprot"], full["_seq"]))
    pairs = full[["uniprot", "substrate_smiles", "label", "split"]].drop_duplicates(
        subset=["uniprot", "substrate_smiles"])
    _write(name, pairs, seqs, card=dict(
        source="github.com/AlexanderKroll/ESP (Kroll et al. 2023); _exp_1_1 train, _exp_phylo_1_1 test",
        binarisation="native binary `outcome` (1 = substrate, 0 = non-substrate, 1:1 sampled)",
        chebi_to_smiles="CHEBI -> InChI (chebiID_to_inchi.tsv) -> SMILES (RDKit)",
        split="ESP native split preserved in `split` column (train/test, phylo test)",
        n_chebi_resolved=len(c2s),
        n_pairs_dropped_unmapped=int(dropped),
    ))


def main(argv: list[str]) -> None:
    targets = argv or list(TDC_SPECS)
    for name in targets:
        if name in TDC_SPECS:
            prep_tdc(name)
        elif name == "esp":
            prep_esp(name)
        else:
            print(f"unknown benchmark: {name}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
