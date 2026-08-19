"""Tests for the DeltaField interaction head.

The headline is test_zero_field_invariant: the anti-shortcut guarantee that
when there is no ligand<->protein coupling the difference field is exactly
zero and the score collapses to the single global bias b0 (identical for every
protein, hence unrankable). This is the architectural property that makes
DeltaField anti-shortcut BY DESIGN — see docs/DELTAFIELD_CONCEPT.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


def _cfg(d: int = 32) -> dict:
    return {
        "model": {
            "head": "deltafield",
            "d_lig": d, "d_prot": d,
            "lig_input_dim": 384, "prot_input_dim": 1280,
            "dropout": 0.0,
            "protein_encoder": "attn_pool",
            "df_n_blocks": 2, "df_n_heads": 4, "df_prop_layers": 0,
        }
    }


def _batch(B=3, L=10, A=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    lig = torch.randn(B, A, 384, generator=g)
    prot = torch.randn(B, L, 1280, generator=g)
    lig_mask = torch.ones(B, A, dtype=torch.bool)
    prot_mask = torch.ones(B, L, dtype=torch.bool)
    # exercise padding when the batch is large enough to spare a row
    if B >= 1 and A >= 3:
        lig_mask[0, -2:] = False
    if B >= 2 and L >= 4:
        prot_mask[1, -3:] = False
    return lig, lig_mask, prot, prot_mask


def test_zero_field_invariant():
    """No coupling => D == 0 exactly => score == b0 for EVERY protein."""
    from v5_rankbind.model import RankBind
    m = RankBind(_cfg()).eval()
    lig, lm, prot, pm = _batch(B=4)
    v = m.lig(lig); u = m.prot(prot)
    out = m.head(u, pm, v, lm, _force_no_coupling=True)
    # field is exactly zero
    assert torch.allclose(out["e_res"], torch.zeros_like(out["e_res"]), atol=1e-6), \
        out["e_res"].abs().max()
    assert torch.allclose(out["e_atom"], torch.zeros_like(out["e_atom"]), atol=1e-6)
    # score is the single global bias for every protein
    b0 = m.head.b0.detach()
    assert torch.allclose(out["score"], b0.expand_as(out["score"]), atol=1e-6), \
        (out["score"], b0)
    # ... and therefore identical across the batch (cannot rank)
    assert out["score"].std() < 1e-6, out["score"]
    print("  ok: zero-field invariant — no coupling => score == b0 for all proteins")


def test_coupling_breaks_the_tie():
    """With coupling on, the field is non-zero and scores actually differ."""
    from v5_rankbind.model import RankBind
    m = RankBind(_cfg()).eval()
    lig, lm, prot, pm = _batch(B=4)
    out = m.forward_field(lig, lm, prot, pm)
    assert out["e_res"].abs().max() > 1e-4, "coupled field should be non-zero"
    assert out["score"].std() > 1e-6, "coupled scores should vary across proteins"
    print("  ok: coupling produces a non-zero field and ranking-capable scores")


def test_ligand_conditional():
    """Same protein, two different ligands -> different scores (not protein-only)."""
    from v5_rankbind.model import RankBind
    m = RankBind(_cfg()).eval()
    _, _, prot, pm = _batch(B=1)
    g = torch.Generator().manual_seed(1)
    A = 5
    ligA = torch.randn(1, A, 384, generator=g)
    ligB = torch.randn(1, A, 384, generator=g)
    lm = torch.ones(1, A, dtype=torch.bool)
    sA = m.forward_field(ligA, lm, prot, pm)["score"]
    sB = m.forward_field(ligB, lm, prot, pm)["score"]
    assert (sA - sB).abs().item() > 1e-5, (sA, sB)
    print("  ok: score is ligand-conditional (same protein, different ligand => different score)")


def test_pairwise_and_triplet_shapes():
    from v5_rankbind.model import RankBind
    m = RankBind(_cfg()).eval()
    lig, lm, prot, pm = _batch(B=6, L=12, A=7)
    assert m.score_pairs_field(lig, lm, prot, pm).shape == (6,)
    # triplet: B=2, k=3 negatives
    B, k, Ln = 2, 3, 9
    g = torch.Generator().manual_seed(2)
    ligt = torch.randn(B, 7, 384, generator=g); lmt = torch.ones(B, 7, dtype=torch.bool)
    pos = torch.randn(B, 11, 1280, generator=g); posm = torch.ones(B, 11, dtype=torch.bool)
    neg = torch.randn(B, k, Ln, 1280, generator=g); negm = torch.ones(B, k, Ln, dtype=torch.bool)
    # score_triplet_field now also returns the pos/neg field dicts (for L_neg).
    p, n, pos_d, neg_d = m.score_triplet_field(ligt, lmt, pos, posm, neg, negm)
    assert p.shape == (B,) and n.shape == (B, k), (p.shape, n.shape)
    assert "e_res" in pos_d and "e_atom" in neg_d
    print("  ok: field pairwise + triplet shapes (+ widened field-dict return)")


def test_backprop_flows():
    from v5_rankbind.model import RankBind
    m = RankBind(_cfg())
    lig, lm, prot, pm = _batch(B=3)
    out = m.forward_field(lig, lm, prot, pm)
    out["score"].sum().backward()
    bad = [n for n, p in m.named_parameters()
           if p.requires_grad and p.grad is None and "head." in n]
    assert not bad, f"Missing grads in head params: {bad}"
    print("  ok: gradients flow to deltafield head params")


def test_param_budget():
    """The shipped config uses d=128 -> ~650k params, matched to the 627k
    bilinear default so 'more capacity' is never the explanation in the paper.
    (d=256 is a deliberately higher-capacity variant at ~2.1M.)"""
    from v5_rankbind.model import RankBind
    m = RankBind(_cfg(d=128))
    n = m.count_parameters()["n_parameters_trainable"]
    assert 550_000 < n < 720_000, n  # within ~15% of the 627k bilinear default
    print(f"  ok: matched-capacity (d=128) trainable params = {n:,} (bilinear default 627,201)")


def test_pairs_path_rejects_pooled_call():
    from v5_rankbind.model import RankBind
    m = RankBind(_cfg())
    try:
        m.score_pairs(torch.randn(2, 384), torch.randn(2, 1280))
    except RuntimeError as e:
        assert "deltafield" in str(e)
        print("  ok: score_pairs rejects deltafield (directs to field path)")
        return
    raise AssertionError("score_pairs should reject deltafield head")


# ──────────────────────────────────────────────────────────────────────────────
# GearBind (v7) — structure-aware GNN head over REAL 3D cache entries
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ESM_DIR = PROJECT_ROOT / "data" / "esm2_embeddings"
CHEMBERTA_DIR = PROJECT_ROOT / "data" / "chemberta_cache"
STRUCT_DIR = (PROJECT_ROOT / "reactionDataFiltering" / "data" / "interim"
              / "structure_graphs_shared")
LIG_DIR = Path("/work2/zw93onug-rankbind_bench/ligand_conformer_cache")
LIG_CSV = (PROJECT_ROOT / "reactionDataFiltering" / "data" / "interim"
           / "kcat_km_brenda_sabio" / "with_decoys.csv")


def _gb_cache_available() -> bool:
    """Real-3D GearBind tests need the structure/ESM2/conformer caches that are
    produced on the Leipzig cluster and are not part of the repository. Skip
    cleanly (rather than fail) on a fresh clone."""
    return (STRUCT_DIR.is_dir() and len(list(STRUCT_DIR.glob("*.pt"))) > 0
            and ESM_DIR.is_dir() and LIG_DIR.is_dir())

requires_gb_cache = pytest.mark.skipif(not _gb_cache_available(),
                                       reason="GearBind caches (structure/ESM2/conformers) not present")


def _gb_cfg(d: int = 128) -> dict:
    return {
        "model": {
            "head": "gearbind",
            "d_lig": d, "d_prot": d,
            "lig_input_dim": 384, "prot_input_dim": 1280,
            "dropout": 0.0,
            "protein_encoder": "attn_pool",
            "df_n_blocks": 2, "df_n_heads": 4, "df_ffn_mult": 4, "df_prop_layers": 0,
            "gnn_layers_prot": 4, "gnn_layers_lig": 3,
        }
    }


def _discover_fixtures(n: int = 3):
    """Find n proteins (structure ∩ ESM2, length-aligned) and n SMILES whose
    conformer + ChemBERTa-mean caches both exist. Globs the live caches per the
    task spec (the ligand cache may still be filling — pick what resolves)."""
    import pandas as pd
    from v5_rankbind.data import _smiles_key

    prots = []
    for f in sorted(STRUCT_DIR.glob("*.pt")):
        u = f.stem
        ef = ESM_DIR / f"{u}.pt"
        if not ef.exists():
            continue
        sd = torch.load(f, weights_only=True)
        e = torch.load(ef, weights_only=True)
        el = e.shape[0] if e.ndim == 2 else 1
        if el == int(sd["n_res"]):
            prots.append(u)
        if len(prots) >= n:
            break

    smis = []
    col = pd.read_csv(LIG_CSV, usecols=["substrate_smiles"])["substrate_smiles"]
    for s in col.dropna().unique().tolist():
        key = _smiles_key(s)
        if (LIG_DIR / f"{key}.pt").exists() and (CHEMBERTA_DIR / f"{key}.pt").exists():
            d = torch.load(LIG_DIR / f"{key}.pt", weights_only=True)
            if bool(d.get("parse_ok", False)):
                smis.append(s)
        if len(smis) >= n:
            break

    assert len(prots) >= n, f"need {n} cached proteins, found {len(prots)}"
    assert len(smis) >= n, f"need {n} cached ligands, found {len(smis)}"
    return prots[:n], smis[:n]


def _gb_dataset(prots, smis):
    import pandas as pd
    from v5_rankbind.data import RankBindDataset
    pairs = pd.DataFrame({
        "uniprot": list(prots),
        "substrate_smiles": list(smis),
        "label": [1.0] * len(prots),
        "idx": list(range(len(prots))),
    })
    return RankBindDataset(
        pairs=pairs, sequences={},
        esm2_dir=ESM_DIR, chemberta_cache_dir=CHEMBERTA_DIR,
        protein_encoder="attn_pool", ligand_encoder="mean_pool",
        structure_dir=STRUCT_DIR, ligand_conformer_dir=LIG_DIR,
        load_graphs=True,
    )


@requires_gb_cache
def test_zero_field_invariant_gearbind():
    """HEADLINE GATE: with full real 3D protein+ligand graphs but no coupling,
    D == 0 exactly => e_res/e_atom == 0 and score == b0 for EVERY protein."""
    from v5_rankbind.model import RankBind
    from v5_rankbind.data import collate_pointwise
    prots, smis = _discover_fixtures(3)
    ds = _gb_dataset(prots, smis)
    batch = collate_pointwise([ds[0], ds[1], ds[2]])
    m = RankBind(_gb_cfg()).eval()
    out = m.forward_field(
        batch["lig_emb"], batch.get("lig_mask"),
        batch["prot_emb"], batch["prot_mask"],
        prot_graph=batch["prot_graph"], lig_graph=batch["lig_graph"],
        _force_no_coupling=True,
    )
    assert torch.allclose(out["e_res"], torch.zeros_like(out["e_res"]), atol=1e-6), \
        out["e_res"].abs().max()
    assert torch.allclose(out["e_atom"], torch.zeros_like(out["e_atom"]), atol=1e-6), \
        out["e_atom"].abs().max()
    b0 = m.head.cross.b0.detach()
    assert torch.allclose(out["score"], b0.expand_as(out["score"]), atol=1e-6), \
        (out["score"], b0)
    assert out["score"].std() < 1e-6, out["score"]
    print("  ok: gearbind zero-field invariant — real 3D graphs, score == b0 for all")


@requires_gb_cache
def test_gearbind_coupling_breaks_tie():
    """Coupling on => non-zero field and ranking-capable (varying) scores."""
    from v5_rankbind.model import RankBind
    from v5_rankbind.data import collate_pointwise
    prots, smis = _discover_fixtures(3)
    ds = _gb_dataset(prots, smis)
    batch = collate_pointwise([ds[0], ds[1], ds[2]])
    m = RankBind(_gb_cfg()).eval()
    out = m.forward_field(
        batch["lig_emb"], batch.get("lig_mask"),
        batch["prot_emb"], batch["prot_mask"],
        prot_graph=batch["prot_graph"], lig_graph=batch["lig_graph"],
    )
    assert out["e_res"].abs().max() > 1e-4, "coupled field should be non-zero"
    assert out["score"].std() > 1e-6, "coupled scores should vary across proteins"
    print("  ok: gearbind coupling produces a non-zero field and varying scores")


@requires_gb_cache
def test_gearbind_triplet_shapes():
    """score_triplet_field over real triplet graphs returns correct shapes and
    the widened pos/neg field dicts."""
    from v5_rankbind.model import RankBind
    from v5_rankbind.sampler import TripletCollator
    prots, smis = _discover_fixtures(3)
    ds = _gb_dataset(prots, smis)
    coll = TripletCollator(ds, n_negatives=2,
                           negative_sampling="cross_protein_implicit")
    batch = coll([ds[0], ds[1], ds[2]])
    m = RankBind(_gb_cfg()).eval()
    pos_s, neg_s, pos_d, neg_d = m.score_triplet_field(
        batch["lig_emb"], batch.get("lig_mask"),
        batch["pos_prot"], batch["pos_mask"],
        batch["neg_prot"], batch["neg_mask"],
        prot_graph=batch["prot_graph"],
        neg_prot_graph=batch["neg_prot_graph"],
        lig_graph=batch["lig_graph"],
    )
    B, k = 3, 2
    assert pos_s.shape == (B,) and neg_s.shape == (B, k), (pos_s.shape, neg_s.shape)
    assert "e_res" in pos_d and "e_atom" in pos_d
    assert "e_res" in neg_d and neg_d["e_res"].shape[0] == B * k
    print("  ok: gearbind triplet shapes + widened field-dict return")


def test_no_torch_scatter_import():
    """model.py must not import torch_scatter / torch_sparse / torch_cluster nor
    call to_dense_batch — the GNN is pure gather+masked-mean. Scans for actual
    import statements / calls (comments may name the banned ops to explain why
    they are avoided), plus a sys.modules guard."""
    import re
    import v5_rankbind.model as model_mod
    src = Path(model_mod.__file__).read_text()
    import_re = re.compile(
        r"^\s*(?:from|import)\s+(torch_scatter|torch_sparse|torch_cluster)\b",
        re.MULTILINE,
    )
    hit = import_re.search(src)
    assert hit is None, f"model.py imports {hit.group(1)}"
    assert "to_dense_batch(" not in src, "model.py calls to_dense_batch"
    for banned in ("torch_scatter", "torch_sparse", "torch_cluster"):
        assert banned not in sys.modules, f"{banned} got imported"
    print("  ok: no torch_scatter / torch_sparse / torch_cluster import; no to_dense_batch call")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    test_zero_field_invariant()
    test_coupling_breaks_the_tie()
    test_ligand_conditional()
    test_pairwise_and_triplet_shapes()
    test_backprop_flows()
    test_param_budget()
    test_pairs_path_rejects_pooled_call()
    test_zero_field_invariant_gearbind()
    test_gearbind_coupling_breaks_tie()
    test_gearbind_triplet_shapes()
    test_no_torch_scatter_import()
    print("All DeltaField + GearBind tests passed.")
