"""
v5_rankbind/eval.py — Test-set evaluation + 200×200 score matrix.

Given a run_dir produced by train.py, load the best checkpoint and produce:

    test_preds_rankbind.csv       — (smiles, uniprot, score, label)
    test_summary.json             — {global_auc, global_aupr, per_lig_auc, hit_at_k}
    score_matrix_rankbind.npy     — [n_lig, n_prot] matrix built from the
                                    same (protein, ligand) pool that Phase-1
                                    null baselines use, so Gini comparisons
                                    are apples-to-apples.

Also extends the manifest in the run_dir with these output paths + metrics.

Usage:
    python -m v5_rankbind.eval --run_dir results/v5_rankbind/<run_id>
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v5_rankbind.run_manifest import RunManifest, load_config, sha256_of
from v5_rankbind.data import (
    ensure_chemberta_cache, ensure_chemberta_token_cache, prepare_frames,
    build_datasets, collate_pointwise, load_chemberta, load_chemberta_tokens,
    _pad_residues, collate_graph_list, RankBindDataset,
)
from v5_rankbind.model import RankBind, FIELD_HEADS


def _graph_to_device(graph: dict | None, device) -> dict | None:
    if graph is None:
        return None
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in graph.items()}


def autocast_ctx(use_bf16: bool):
    """bf16 autocast for eval-time forwards (no GradScaler; eval is no_grad)."""
    if use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
from v5_rankbind.metrics import (
    per_ligand_auc, hit_at_k, global_metrics, matrix_ranking_metrics,
    matrix_per_ligand_auc,
)


@torch.no_grad()
def run_test_set(
    model: RankBind, test_ds, device: torch.device, batch_size: int = 64,
    use_bf16: bool = False,
) -> pd.DataFrame:
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_pointwise, num_workers=0)
    rows = []
    model.eval()
    for batch in loader:
        lig = batch["lig_emb"].to(device)
        prot = batch["prot_emb"].to(device)
        prot_mask = batch.get("prot_mask")
        if prot_mask is not None:
            prot_mask = prot_mask.to(device)
        if model.head_type in FIELD_HEADS:
            lig_mask = batch.get("lig_mask")
            if lig_mask is not None:
                lig_mask = lig_mask.to(device)
            pg = _graph_to_device(batch.get("prot_graph"), device)
            lg = _graph_to_device(batch.get("lig_graph"), device)
            with autocast_ctx(use_bf16):
                score = model.score_pairs_field(
                    lig, lig_mask, prot, prot_mask, prot_graph=pg, lig_graph=lg
                )
            score = score.float().cpu().numpy()
        else:
            with autocast_ctx(use_bf16):
                score = model.score_pairs(lig, prot, prot_mask)
            score = score.float().cpu().numpy()
        for s, uni, sc, lb in zip(batch["smiles"], batch["uniprot"], score, batch["label"]):
            rows.append({"smiles": s, "uniprot": uni, "score": float(sc), "label": int(lb)})
    return pd.DataFrame(rows)


@torch.no_grad()
def build_score_matrix(
    model: RankBind,
    config_dict: dict,
    device: torch.device,
    n_matrix: int = 200,
    chunk: int = 128,
    use_bf16: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a [n_lig, n_prot] score matrix over the SAME pool of proteins and
    SMILES that evaluation/null_baselines.py uses. Critical for Gini to be
    directly comparable between RankBind and the Phase-1 null baselines.
    """
    from common import BRENDADataConfig  # noqa
    bconfig = BRENDADataConfig(
        seed=int(config_dict["data"].get("split_seed", 42)),
        csv_path=str(PROJECT_ROOT / config_dict["data"]["csv_path"]),
        seq_csv=str(PROJECT_ROOT / config_dict["data"]["seq_csv"]),
        val_frac=config_dict["data"]["val_frac"],
        test_frac=config_dict["data"]["test_frac"],
    )
    pairs = bconfig.load_pairs()
    seqs = bconfig.load_sequences()

    # Mirror evaluation/null_baselines.py selection:
    proteins = list(seqs.keys())[:n_matrix]
    smiles_list = pairs["substrate_smiles"].unique().tolist()[:n_matrix]

    # Precompute embeddings
    esm2_dir = PROJECT_ROOT / config_dict["data"]["esm2_dir"]
    chemberta_cache = PROJECT_ROOT / "data" / "chemberta_cache"

    ensure_chemberta_cache(smiles_list, chemberta_cache, device=str(device))

    protein_encoder = config_dict["model"].get("protein_encoder", "mean_pool")
    max_residues = config_dict["model"].get("max_residues", 1024)
    D_in = config_dict["model"]["prot_input_dim"]

    if protein_encoder == "attn_pool":
        # Per-residue tensors live on CPU; encode chunked on GPU.
        from v5_rankbind.data import _pad_residues
        prot_residues: list[torch.Tensor] = []
        for p in proteins:
            f = esm2_dir / f"{p}.pt"
            if f.exists():
                t = torch.load(f, weights_only=True).to(torch.float32)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                if t.shape[0] > max_residues:
                    t = t[:max_residues]
            else:
                t = torch.zeros(1, D_in, dtype=torch.float32)
            prot_residues.append(t)

        chunk = 32
        gP_chunks = []
        model.eval()
        for s in range(0, len(prot_residues), chunk):
            e = min(s + chunk, len(prot_residues))
            chunk_resi = prot_residues[s:e]
            chunk_lens = [int(t.shape[0]) for t in chunk_resi]
            padded, mask = _pad_residues(chunk_resi, chunk_lens)
            padded = padded.to(device); mask = mask.to(device)
            with autocast_ctx(use_bf16):
                gP_chunks.append(model.encode_protein(padded, mask))
        gP_all = torch.cat(gP_chunks, dim=0)                     # [n_prot, d_prot]
    else:
        prot_embs = []
        for p in proteins:
            f = esm2_dir / f"{p}.pt"
            if f.exists():
                t = torch.load(f, weights_only=True)
                if t.ndim == 2:
                    t = t.mean(dim=0)
            else:
                t = torch.zeros(D_in)
            prot_embs.append(t.to(torch.float32))
        prot_embs = torch.stack(prot_embs).to(device)             # [n_prot, D]
        model.eval()
        with autocast_ctx(use_bf16):
            gP_all = model.encode_protein(prot_embs)              # [n_prot, d_prot]

    lig_embs = torch.stack(
        [load_chemberta(s, chemberta_cache).to(torch.float32) for s in smiles_list]
    ).to(device)                                                 # [n_lig, D]
    with autocast_ctx(use_bf16):
        fL_all = model.lig(lig_embs)                             # [n_lig, d_lig]

    n_lig, n_prot = len(smiles_list), len(proteins)
    scores = np.zeros((n_lig, n_prot), dtype=np.float32)
    for i in range(0, n_lig, chunk):
        fL = fL_all[i:i + chunk].unsqueeze(1)                    # [b, 1, d]
        b = fL.shape[0]
        gP = gP_all.unsqueeze(0).expand(b, -1, -1)               # [b, n_prot, d]
        with autocast_ctx(use_bf16):
            row = model.head(fL.expand(-1, n_prot, -1), gP)      # [b, n_prot]
        scores[i:i + chunk] = row.float().cpu().numpy()
    return scores, smiles_list, proteins


@torch.no_grad()
def build_score_matrix_deltafield(
    model: RankBind,
    config_dict: dict,
    device: torch.device,
    n_matrix: int = 200,
    lig_chunk: int = 32,
    use_bf16: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    """[n_lig, n_prot] score matrix for the DeltaField head over the SAME pool
    as build_score_matrix (first n_matrix proteins / unique SMILES). Each
    (ligand, protein) is scored by a full coupled forward_field; we loop over
    proteins and chunk ligands to bound memory. Heavier than the bilinear dot,
    so n_matrix should match the bilinear runs (200) for comparability.
    """
    from common import BRENDADataConfig  # noqa
    bconfig = BRENDADataConfig(
        seed=int(config_dict["data"].get("split_seed", 42)),
        csv_path=str(PROJECT_ROOT / config_dict["data"]["csv_path"]),
        seq_csv=str(PROJECT_ROOT / config_dict["data"]["seq_csv"]),
        val_frac=config_dict["data"]["val_frac"],
        test_frac=config_dict["data"]["test_frac"],
    )
    pairs = bconfig.load_pairs()
    seqs = bconfig.load_sequences()
    proteins = list(seqs.keys())[:n_matrix]
    smiles_list = pairs["substrate_smiles"].unique().tolist()[:n_matrix]

    esm2_dir = PROJECT_ROOT / config_dict["data"]["esm2_dir"]
    max_res = config_dict["model"].get("max_residues", 1024)
    max_atoms = config_dict["model"].get("max_ligand_tokens", 128)
    D_in = config_dict["model"]["prot_input_dim"]
    tcd = config_dict["data"]["chemberta_token_cache"]
    token_cache = Path(tcd) if Path(tcd).is_absolute() else PROJECT_ROOT / tcd
    ensure_chemberta_token_cache(smiles_list, token_cache, device=str(device),
                                 max_length=max_atoms)

    lig_toks = [load_chemberta_tokens(s, token_cache, max_tokens=max_atoms).to(torch.float32)
                for s in smiles_list]
    lig_pad, lig_mask_all = _pad_residues(lig_toks, [t.shape[0] for t in lig_toks])

    prot_res: list[torch.Tensor] = []
    for p in proteins:
        f = esm2_dir / f"{p}.pt"
        if f.exists():
            t = torch.load(f, weights_only=True).to(torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            if t.shape[0] > max_res:
                t = t[:max_res]
        else:
            t = torch.zeros(1, D_in, dtype=torch.float32)
        prot_res.append(t)

    n_lig, n_prot = len(smiles_list), len(proteins)
    scores = np.zeros((n_lig, n_prot), dtype=np.float32)
    model.eval()
    for j, pr in enumerate(prot_res):
        Lj = pr.shape[0]
        pr_dev = pr.to(device)
        for s in range(0, n_lig, lig_chunk):
            e = min(s + lig_chunk, n_lig)
            b = e - s
            lt = lig_pad[s:e].to(device)
            lm = lig_mask_all[s:e].to(device)
            pr_b = pr_dev.unsqueeze(0).expand(b, Lj, D_in).contiguous()
            pm_b = torch.ones(b, Lj, dtype=torch.bool, device=device)
            with autocast_ctx(use_bf16):
                out = model.forward_field(lt, lm, pr_b, pm_b)
            scores[s:e, j] = out["score"].float().cpu().numpy()
    return scores, smiles_list, proteins


@torch.no_grad()
def build_score_matrix_gearbind(
    model: RankBind,
    config_dict: dict,
    device: torch.device,
    n_matrix: int = 200,
    lig_chunk: int = 16,
    use_bf16: bool = False,
) -> tuple[np.ndarray, list[str], list[str]]:
    """[n_lig, n_prot] score matrix for the gearbind head over the SAME pool as
    build_score_matrix. Each (ligand, protein) is scored by a coupled
    forward_field over the structure + 3D-ligand graphs. Loops proteins (outer),
    chunks ligands; per-protein ESM2 + structure graph built once, ligand graphs
    cached across proteins. P2/eval-only path (not exercised by the unit tests).
    """
    from common import BRENDADataConfig  # noqa
    bconfig = BRENDADataConfig(
        seed=int(config_dict["data"].get("split_seed", 42)),
        csv_path=str(PROJECT_ROOT / config_dict["data"]["csv_path"]),
        seq_csv=str(PROJECT_ROOT / config_dict["data"]["seq_csv"]),
        val_frac=config_dict["data"]["val_frac"],
        test_frac=config_dict["data"]["test_frac"],
    )
    pairs = bconfig.load_pairs()
    seqs = bconfig.load_sequences()
    proteins = list(seqs.keys())[:n_matrix]
    smiles_list = pairs["substrate_smiles"].unique().tolist()[:n_matrix]

    esm2_dir = PROJECT_ROOT / config_dict["data"]["esm2_dir"]
    max_res = config_dict["model"].get("max_residues", 1024)
    chemberta_cache = PROJECT_ROOT / "data" / "chemberta_cache"
    ensure_chemberta_cache(smiles_list, chemberta_cache, device=str(device))

    sd = config_dict["data"]["structure_dir"]
    structure_dir = Path(sd) if Path(sd).is_absolute() else PROJECT_ROOT / sd
    lcd = config_dict["data"]["ligand_conformer_dir"]
    ligand_conformer_dir = Path(lcd) if Path(lcd).is_absolute() else PROJECT_ROOT / lcd

    # Throwaway dataset purely as a graph-loader (empty pairs frame).
    ds = RankBindDataset(
        pairs=pd.DataFrame({"uniprot": [], "substrate_smiles": [], "label": [], "idx": []}),
        sequences={}, esm2_dir=esm2_dir, chemberta_cache_dir=chemberta_cache,
        prot_input_dim=config_dict["model"]["prot_input_dim"],
        lig_input_dim=config_dict["model"]["lig_input_dim"],
        protein_encoder="attn_pool", max_residues=max_res,
        ligand_encoder="mean_pool",
        structure_dir=structure_dir, ligand_conformer_dir=ligand_conformer_dir,
        load_graphs=True,
    )

    lig_graphs = [ds.load_ligand_graph(s) for s in smiles_list]
    n_lig, n_prot = len(smiles_list), len(proteins)
    scores = np.zeros((n_lig, n_prot), dtype=np.float32)
    model.eval()
    for j, uni in enumerate(proteins):
        prot_res = ds.load_protein(uni)                     # [L, D_in]
        L = int(prot_res.shape[0])
        sg = ds.load_structure(uni, L)
        for s in range(0, n_lig, lig_chunk):
            e = min(s + lig_chunk, n_lig)
            b = e - s
            lg = collate_graph_list(lig_graphs[s:e], has_seqsep=False,
                                    node_feat_keys=["node_feat"])
            pg = collate_graph_list([sg] * b, has_seqsep=True,
                                    node_feat_keys=["plddt"])
            pr = prot_res.unsqueeze(0).expand(b, L, prot_res.shape[1]).contiguous().to(device)
            pm = torch.ones(b, L, dtype=torch.bool, device=device)
            lig_dummy = torch.zeros(b, config_dict["model"]["lig_input_dim"], device=device)
            with autocast_ctx(use_bf16):
                out = model.forward_field(
                    lig_dummy, None, pr, pm,
                    prot_graph=_graph_to_device(pg, device),
                    lig_graph=_graph_to_device(lg, device),
                )
            scores[s:e, j] = out["score"].float().cpu().numpy()
    return scores, smiles_list, proteins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_matrix", type=int, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest at {manifest_path}")
        sys.exit(1)
    manifest_data = json.loads(manifest_path.read_text())
    cfg = load_config(manifest_data["config_path"])
    # The CLI --seed override applied at TRAINING time lives only in the
    # training process; re-loading the config file here would silently
    # resolve to its base seed and evaluate on a different (often the
    # canonical) split than the model trained on. Always trust
    # config_resolved, which records what training actually used.
    resolved = manifest_data.get("config_resolved", {})
    if "seed" in resolved:
        cfg["seed"] = resolved["seed"]
    if args.n_matrix is not None:
        cfg["eval"]["n_matrix"] = args.n_matrix
    device = torch.device(args.device)

    # Build test dataset from same split
    chemberta_cache = PROJECT_ROOT / "data" / "chemberta_cache"
    train_ds, val_ds, test_ds, split_stats = build_datasets(cfg, chemberta_cache)

    # Guard against silent split drift: the datasets rebuilt here MUST match
    # the split recorded at training time, otherwise all test metrics below
    # compare the model against data it may have trained on.
    m_split = manifest_data.get("split", {})
    for key in ("n_train_pairs", "n_val_pairs", "n_test_pairs"):
        if key in m_split and m_split[key] != split_stats[key]:
            raise SystemExit(
                f"[guard] split mismatch for {key}: manifest={m_split[key]} "
                f"rebuilt={split_stats[key]} — refusing to evaluate. The run "
                "was trained on a different split than eval just rebuilt."
            )

    # DeltaField needs the per-token cache to exist before run_test_set iterates.
    if cfg["model"].get("ligand_encoder") == "per_token":
        tcd = cfg["data"]["chemberta_token_cache"]
        token_cache_dir = Path(tcd) if Path(tcd).is_absolute() else PROJECT_ROOT / tcd
        all_smiles = pd.read_csv(PROJECT_ROOT / cfg["data"]["csv_path"])[
            "substrate_smiles"].dropna().unique().tolist()
        ensure_chemberta_token_cache(all_smiles, token_cache_dir, device=str(device),
                                     max_length=cfg["model"].get("max_ligand_tokens", 128))

    # Load best checkpoint
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"ERROR: no checkpoint at {ckpt_path}")
        sys.exit(1)
    model = RankBind(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # bf16 autocast for eval forwards (read from the run's own cfg; CUDA only).
    use_bf16 = (cfg["train"].get("precision") == "bf16" and device.type == "cuda")
    print(f"[mem] eval bf16_autocast={use_bf16}")

    # ── test-set predictions ──────────────────────────────────────────────
    df = run_test_set(model, test_ds, device, use_bf16=use_bf16)
    test_csv = run_dir / "test_preds_rankbind.csv"
    df.to_csv(test_csv, index=False)

    scores = df["score"].to_numpy(); labels = df["label"].to_numpy()
    smiles = df["smiles"].tolist()
    per_lig, n_lig = per_ligand_auc(smiles, scores, labels)
    glob = global_metrics(scores, labels)
    hits = hit_at_k(smiles, scores, labels, ks=(1, 5, 10))
    summary = {
        "model":                 "rankbind",
        "run_id":                manifest_data["run_id"],
        "checkpoint_sha256":     sha256_of(ckpt_path),
        "n_test_pairs":          int(len(df)),
        "n_positives":           int(labels.sum()),
        "per_ligand_auc":        per_lig,
        "n_ligands_counted":     n_lig,
        **glob, **hits,
    }

    test_summary_path = run_dir / "test_summary.json"
    test_summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ── 200×200 score matrix (matching phase-1 null-baseline geometry) ────
    if model.head_type == "gearbind":
        M, lig_list, prot_list = build_score_matrix_gearbind(
            model, cfg, device, n_matrix=cfg["eval"]["n_matrix"], use_bf16=use_bf16,
        )
    elif model.head_type == "deltafield":
        M, lig_list, prot_list = build_score_matrix_deltafield(
            model, cfg, device, n_matrix=cfg["eval"]["n_matrix"], use_bf16=use_bf16,
        )
    else:
        M, lig_list, prot_list = build_score_matrix(
            model, cfg, device, n_matrix=cfg["eval"]["n_matrix"], use_bf16=use_bf16,
        )
    mat_path = run_dir / "score_matrix_rankbind.npy"
    np.save(mat_path, M)
    axis_path = run_dir / "score_matrix_axes.json"
    axis_path.write_text(json.dumps({
        "axis_0_ligands": lig_list, "axis_1_proteins": prot_list, "shape": list(M.shape)
    }))

    # Ligand-conditional ranking metrics on the FULL matrix (stable signal —
    # unlike per_ligand_auc which is limited to the ~4 test ligands with
    # both classes in the test split).
    positive_pairs = list(df[df["label"] == 1][["smiles", "uniprot"]]
                          .itertuples(index=False, name=None))
    rank_metrics = matrix_ranking_metrics(M, lig_list, prot_list, positive_pairs)
    # Matrix-level per-ligand AUC (n ~50 on BRENDA vs ~4 for the test-pair
    # per_ligand_auc above) — the statistically usable ligand-conditional AUC.
    rank_metrics.update(
        matrix_per_ligand_auc(M, lig_list, prot_list, positive_pairs)
    )
    rank_path = run_dir / "test_matrix_ranking.json"
    rank_path.write_text(json.dumps(rank_metrics, indent=2))
    print("[matrix ranking]", json.dumps(rank_metrics, indent=2))

    # ── Extend manifest ───────────────────────────────────────────────────
    manifest_data["outputs"]["test_preds"] = {
        "path": str(test_csv), "sha256": sha256_of(test_csv),
        "size_bytes": test_csv.stat().st_size,
    }
    manifest_data["outputs"]["test_summary"] = {
        "path": str(test_summary_path), "sha256": sha256_of(test_summary_path),
        "size_bytes": test_summary_path.stat().st_size,
    }
    manifest_data["outputs"]["score_matrix"] = {
        "path": str(mat_path), "sha256": sha256_of(mat_path),
        "size_bytes": mat_path.stat().st_size,
    }
    manifest_data["outputs"]["score_matrix_axes"] = {
        "path": str(axis_path), "sha256": sha256_of(axis_path),
        "size_bytes": axis_path.stat().st_size,
    }
    manifest_data["metrics"]["test_global_auc"] = glob["global_auc"]
    manifest_data["metrics"]["test_global_aupr"] = glob["global_aupr"]
    manifest_data["metrics"]["test_per_lig_auc"] = per_lig
    manifest_data["metrics"]["test_hit_at_1"] = hits.get("hit_at_1")
    manifest_data["metrics"]["test_hit_at_5"] = hits.get("hit_at_5")
    manifest_data["metrics"]["test_hit_at_10"] = hits.get("hit_at_10")
    # Matrix-level ranking metrics (stable ligand-conditional signal)
    for k, v in rank_metrics.items():
        manifest_data["metrics"][f"matrix_{k}"] = v
    manifest_data["outputs"]["test_matrix_ranking"] = {
        "path": str(rank_path), "sha256": sha256_of(rank_path),
        "size_bytes": rank_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2, default=str))
    print(f"[done] manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
