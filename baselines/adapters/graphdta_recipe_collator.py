"""
baselines/adapters/graphdta_recipe_collator.py — GraphDTA on the RankBind
anti-shortcut data regime.

Feeds the off-the-shelf GraphDTA GCN the *identical* ligand-conditional triplet
structure that RankBind v4 trains on, so the recipe-transfer experiment can ask:
is the anti-shortcut advantage the recipe or the RankBind architecture?

Three pieces:

  * GraphDTARecipeDataset — GraphDTADataset subclass whose __getitem__ returns a
    dict {graph, smiles, uniprot, label} (the base adapter is untouched, so the
    Phase-1 BCE trainer keeps working). Adds seq_ids_for / graph_for_smiles
    lookups used by the collator and the hard-negative refresh.

  * GraphDTATripletCollator — collate_fn that, for each positive anchor in the
    batch, draws k negative proteins via the shared NegativeSelector (identical
    to TripletCollator; see v5_rankbind/tests/test_negative_selection.py) and
    builds one flat PyG Batch of B*(1+k) (ligand-graph, target-sequence) pairs.
    The trainer scores it with model(batch).view(-1) and reshapes to
    pos_score[B], neg_score[B,k] for v5_rankbind.loss.margin_loss.

  * refresh_scores_graphdta — (positive-ligand × protein) score matrix installed
    on the selector for hard-negative mining. GCNNet IS separable (independent
    ligand branch / protein branch / head), so each ligand and protein is encoded
    ONCE and the cheap head runs over the full grid — O(n_lig + n_prot) heavy
    forwards, not O(n_lig × n_prot). Verified bit-identical (≤2e-8) to the full
    model(batch) forward, so full coverage is the default; lig_cap / prot_cap
    remain as safety valves and coverage is logged, never silently truncated.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_max_pool as _gmp

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from adapter_graphdta import GraphDTADataset, seq_to_ids  # noqa: E402
from v5_rankbind.negative_selection import NegativeSelector  # noqa: E402


class GraphDTARecipeDataset(GraphDTADataset):
    """GraphDTADataset that yields identity-tagged items for triplet training.

    items[i] == (graph, seq_ids, label, smiles, uniprot) in the base class.
    """

    def __init__(self, config, indices: list):
        super().__init__(config, indices)
        # First bare ligand graph per SMILES (anchor source for refresh).
        self._graph_by_smiles: dict[str, object] = {}
        for graph, _seq, _lab, smi, _uni in self.items:
            if smi not in self._graph_by_smiles:
                self._graph_by_smiles[smi] = graph
        self._seqids_cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, idx):
        graph, _seq_ids, label, smiles, uniprot = self.items[idx]
        return {
            "graph":   graph,            # bare ligand graph (x, edge_index); no target
            "smiles":  smiles,
            "uniprot": uniprot,
            "label":   float(label),
        }

    def seq_ids_for(self, uniprot: str) -> torch.Tensor:
        """Cached integer-encoded sequence [1000] for a protein."""
        t = self._seqids_cache.get(uniprot)
        if t is None:
            t = seq_to_ids(self.sequences[uniprot])
            self._seqids_cache[uniprot] = t
        return t

    def graph_for_smiles(self, smiles: str):
        return self._graph_by_smiles[smiles]


class GraphDTATripletCollator:
    """collate_fn building B*(1+k) (ligand, target) pairs for margin loss.

    Layout of the returned PyG Batch: the first B graphs are the positive
    (anchor-ligand, anchor-protein) pairs; the next B*k are the negatives,
    anchor-major (anchor 0's k negs, then anchor 1's, ...). So after
    ``s = model(batch).view(-1)``:

        pos_score = s[:B]
        neg_score = s[B:].view(B, k)
    """

    def __init__(self, dataset: GraphDTARecipeDataset, selector: NegativeSelector):
        self.ds = dataset
        self.selector = selector
        self.k = selector.n_negatives

    def __call__(self, batch: list[dict]) -> dict | None:
        anchors = [b for b in batch if b["label"] == 1.0]
        if not anchors:
            return None  # margin loss undefined without a positive anchor

        k = self.k
        pos_graphs, neg_graphs = [], []
        smiles, anchor_uni, neg_uni = [], [], []
        for a in anchors:
            smi, uni = a["smiles"], a["uniprot"]
            g_pos = a["graph"].clone()
            g_pos.target = self.ds.seq_ids_for(uni).unsqueeze(0)
            pos_graphs.append(g_pos)

            negs = self.selector.sample_negs_for_anchor(smi, uni)
            for nu in negs:
                g_neg = a["graph"].clone()
                g_neg.target = self.ds.seq_ids_for(nu).unsqueeze(0)
                neg_graphs.append(g_neg)
            smiles.append(smi); anchor_uni.append(uni); neg_uni.append(negs)

        hard_active = bool(self.selector.use_hard and self.selector._scores is not None)
        return {
            "batch":          Batch.from_data_list(pos_graphs + neg_graphs),
            "B":              len(anchors),
            "k":              k,
            "n_anchors_in":   len(batch),
            "n_anchors_kept": len(anchors),
            "smiles":         smiles,
            "anchor_uniprot": anchor_uni,
            "neg_uniprot":    neg_uni,
            "hard_active":    hard_active,
        }


def _has_separable_gcn(model) -> bool:
    """GCNNet exposes a ligand branch, a protein branch and a head as separate
    submodules, so a (ligand × protein) score grid can be computed by encoding
    each ligand and each protein ONCE and running only the cheap head over all
    pairs — O(n_lig + n_prot) heavy forwards instead of O(n_lig × n_prot)."""
    return all(hasattr(model, a) for a in (
        "conv1", "conv2", "conv3", "fc_g1", "fc_g2", "embedding_xt",
        "conv_xt_1", "fc1_xt", "fc1", "fc2", "out", "relu", "dropout"))


@torch.no_grad()
def _gcn_encode_ligands(model, graphs, device, chunk=512) -> torch.Tensor:
    outs = []
    for s in range(0, len(graphs), chunk):
        b = Batch.from_data_list([g.clone() for g in graphs[s:s + chunk]]).to(device)
        x, ei, ba = b.x, b.edge_index, b.batch
        x = model.relu(model.conv1(x, ei))
        x = model.relu(model.conv2(x, ei))
        x = model.relu(model.conv3(x, ei))
        x = _gmp(x, ba)
        x = model.relu(model.fc_g1(x)); x = model.dropout(x)
        x = model.fc_g2(x); x = model.dropout(x)
        outs.append(x)
    return torch.cat(outs, 0)                                   # [n_lig, 128]


@torch.no_grad()
def _gcn_encode_proteins(model, targets, device, chunk=256) -> torch.Tensor:
    outs = []
    for s in range(0, len(targets), chunk):
        t = torch.stack(targets[s:s + chunk]).to(device)       # [b, 1000]
        emb = model.embedding_xt(t)
        c = model.conv_xt_1(emb)
        xt = c.view(-1, 32 * 121)
        xt = model.fc1_xt(xt)
        outs.append(xt)
    return torch.cat(outs, 0)                                   # [n_prot, 128]


@torch.no_grad()
def _gcn_head(model, xl, xt) -> torch.Tensor:
    xc = torch.cat([xl, xt], 1)
    xc = model.relu(model.fc1(xc)); xc = model.dropout(xc)
    xc = model.relu(model.fc2(xc)); xc = model.dropout(xc)
    return model.out(xc).view(-1)


@torch.no_grad()
def refresh_scores_graphdta(
    model,
    dataset: GraphDTARecipeDataset,
    selector: NegativeSelector,
    device,
    prot_cap: int | None = None,
    lig_cap: int | None = None,
    chunk: int = 256,
    rng: np.random.Generator | None = None,
    row_chunk: int = 64,
) -> dict:
    """Recompute the (positive-ligand × protein) score matrix and install it on
    the selector for hard-negative mining. No-op unless selector.use_hard.

    GCNNet is separable (ligand branch / protein branch / head). We exploit that
    to encode each ligand and each protein ONCE and score all pairs through the
    cheap head — exactly why RankBind's separable encoders make full-coverage
    hard-neg refresh cheap. This is verified bit-identical to the full
    model(batch) forward (max abs diff ~2e-8). Full coverage is therefore the
    default; lig_cap / prot_cap remain as safety valves (un-scored rows -> random
    fallback; un-scored cols -> ineligible). Coverage is returned for logging.
    """
    if not selector.use_hard:
        return {"refreshed": False, "reason": "not hard mode"}

    row_smis = selector.row_to_smi
    all_prots = selector.all_proteins
    n_lig, n_prot = len(row_smis), len(all_prots)
    if n_lig == 0:
        selector.set_scores(None)
        return {"refreshed": False, "reason": "no positive anchors"}

    rng = rng if rng is not None else np.random.default_rng(0)
    if lig_cap and n_lig > lig_cap:
        row_sel = np.sort(rng.choice(n_lig, size=lig_cap, replace=False))
    else:
        row_sel = np.arange(n_lig)
    if prot_cap and n_prot > prot_cap:
        col_sel = np.sort(rng.choice(n_prot, size=prot_cap, replace=False))
    else:
        col_sel = np.arange(n_prot)

    was_training = model.training
    model.eval()
    scores = np.full((n_lig, n_prot), -np.inf, dtype=np.float32)
    lig_graphs = [dataset.graph_for_smiles(row_smis[int(r)]) for r in row_sel]
    col_targets = [dataset.seq_ids_for(all_prots[int(c)]) for c in col_sel]

    if _has_separable_gcn(model):
        L = _gcn_encode_ligands(model, lig_graphs, device)     # [R, 128]
        P = _gcn_encode_proteins(model, col_targets, device)   # [C, 128]
        C = P.shape[0]
        for s in range(0, len(row_sel), row_chunk):
            e = min(s + row_chunk, len(row_sel))
            R = e - s
            xl = L[s:e].unsqueeze(1).expand(R, C, -1).reshape(R * C, -1)
            xt = P.unsqueeze(0).expand(R, C, -1).reshape(R * C, -1)
            sc = _gcn_head(model, xl, xt).view(R, C).float().cpu().numpy()
            scores[np.ix_(row_sel[s:e], col_sel)] = sc
    else:
        # Fallback: full per-pair forward (architecture without split branches).
        for ri, r in enumerate(row_sel):
            g0 = lig_graphs[ri]
            vals = np.empty(len(col_sel), dtype=np.float32)
            for s in range(0, len(col_sel), chunk):
                e = min(s + chunk, len(col_sel))
                graphs = []
                for t in col_targets[s:e]:
                    g = g0.clone(); g.target = t.unsqueeze(0); graphs.append(g)
                b = Batch.from_data_list(graphs).to(device)
                vals[s:e] = model(b).view(-1).float().cpu().numpy()[: e - s]
            scores[int(r), col_sel] = vals

    if was_training:
        model.train()
    selector.set_scores(scores)
    return {
        "refreshed":     True,
        "n_lig":         int(n_lig),
        "n_prot":        int(n_prot),
        "rows_scored":   int(len(row_sel)),
        "cols_scored":   int(len(col_sel)),
        "lig_coverage":  round(len(row_sel) / n_lig, 3),
        "prot_coverage": round(len(col_sel) / n_prot, 3),
        "fast_path":     bool(_has_separable_gcn(model)),
    }
