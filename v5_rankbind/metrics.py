"""
v5_rankbind/metrics.py — Evaluation metrics reused across train/eval.

Per-ligand AUC is THE primary metric for Phase 2. The definition matches
evaluation/test_set_eval.py exactly: for each ligand with at least one
positive and one negative test pair, compute ROC-AUC over that ligand's
proteins, then average.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def per_ligand_auc(
    smiles: list[str], scores: np.ndarray, labels: np.ndarray
) -> tuple[float, int]:
    """Return (mean per-ligand AUC, number of ligands counted)."""
    groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for s, sc, lb in zip(smiles, scores, labels):
        groups[s][0].append(float(sc)); groups[s][1].append(int(lb))

    aucs = []
    for sc, lb in groups.values():
        if len(set(lb)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(lb, sc))
        except ValueError:
            continue
    return (float(np.mean(aucs)) if aucs else float("nan"), len(aucs))


def hit_at_k(
    smiles: list[str], scores: np.ndarray, labels: np.ndarray, ks: Iterable[int] = (1, 5, 10)
) -> dict:
    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for s, sc, lb in zip(smiles, scores, labels):
        groups[s].append((float(sc), int(lb)))

    out = {f"hit_at_{k}": np.nan for k in ks}
    out["n_ligands_evaluated"] = 0
    rows = []
    for rows_smi in groups.values():
        if sum(1 for _, l in rows_smi if l == 1) < 1:
            continue
        if len(rows_smi) < 2:
            continue
        rows_smi.sort(key=lambda x: -x[0])
        labels_sorted = [l for _, l in rows_smi]
        pos_rank = labels_sorted.index(1)
        rows.append(pos_rank)
    out["n_ligands_evaluated"] = len(rows)
    if rows:
        for k in ks:
            out[f"hit_at_{k}"] = float(np.mean([1.0 if r < k else 0.0 for r in rows]))
    return out


def global_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = float("nan")
    try:
        aupr = float(average_precision_score(labels, scores))
    except ValueError:
        aupr = float("nan")
    return {"global_auc": auc, "global_aupr": aupr}


def matrix_ranking_metrics(
    score_matrix: np.ndarray,
    lig_list: list[str],
    prot_list: list[str],
    positive_pairs: list[tuple[str, str]],
) -> dict:
    """Ligand-conditional ranking metrics computed directly on the 200×200
    score matrix. Unlike per_ligand_auc (which needs 2+ test measurements per
    ligand), this uses every observed (ligand, positive_protein) pair: rank
    the positive protein among all n_prot proteins in the matrix, aggregate.

    Returns {mrr, mean_rank, mean_rank_pct, hit_at_1, hit_at_5, hit_at_10, n}.
    This is the stable ligand-conditional signal for Phase-2 claims; the
    smaller per_ligand_auc is kept for strict Phase-1 comparability.
    """
    lig_to_row = {s: i for i, s in enumerate(lig_list)}
    prot_to_col = {p: j for j, p in enumerate(prot_list)}
    n_prot = len(prot_list)

    ranks = []
    for lig, prot in positive_pairs:
        if lig not in lig_to_row or prot not in prot_to_col:
            continue
        i = lig_to_row[lig]; j = prot_to_col[prot]
        row = score_matrix[i]
        rank = int((row > row[j]).sum())  # 0-indexed (0 = top)
        ranks.append(rank)

    if not ranks:
        return {k: float("nan") for k in (
            "mrr", "mean_rank", "mean_rank_pct",
            "hit_at_1", "hit_at_5", "hit_at_10",
        )} | {"n_positive_pairs_matched": 0}

    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "mrr":                       float(np.mean(1.0 / (ranks + 1))),
        "mean_rank":                 float(ranks.mean()),
        "mean_rank_pct":             float(ranks.mean() / max(n_prot - 1, 1)),
        "hit_at_1":                  float((ranks < 1).mean()),
        "hit_at_5":                  float((ranks < 5).mean()),
        "hit_at_10":                 float((ranks < 10).mean()),
        "n_positive_pairs_matched":  int(len(ranks)),
    }


def matrix_per_ligand_auc(
    score_matrix: np.ndarray,
    lig_list: list[str],
    prot_list: list[str],
    positive_pairs: list[tuple[str, str]],
) -> dict:
    """Per-ligand AUC computed on the full 200×200 score matrix instead of
    the test-pair table.

    The strict ``per_ligand_auc`` only counts ligands that have BOTH a
    positive and a negative *test pair*; under the protein-based split that
    is ~4 ligands on BRENDA — statistically empty. Here every ligand row of
    the matrix is scored against all ``n_prot`` proteins: a column is a
    positive iff that (ligand, protein) pair is in ``positive_pairs`` (the
    same held-out positive set used by ``matrix_ranking_metrics``), every
    other column is a negative. A ligand counts if its row has ≥1 positive
    and ≥1 negative — which is true for every ligand with an observed
    binder in the pool, lifting n from ~4 to the number of distinct
    positive ligands (~50 on BRENDA, hundreds on ESP/turnover).

    Caveat (shared with matrix_ranking_metrics): columns that are training
    positives for the same ligand are treated as negatives, so this is a
    *conservative* estimate — it can only understate per-ligand AUC, never
    inflate it.

    Returns {matrix_per_ligand_auc, n_ligands_counted, n_positive_cells}.
    """
    lig_to_row = {s: i for i, s in enumerate(lig_list)}
    prot_to_col = {p: j for j, p in enumerate(prot_list)}
    n_prot = len(prot_list)

    # positive column-indices per ligand row, restricted to the matrix axes
    pos_cols: dict[int, set[int]] = defaultdict(set)
    for lig, prot in positive_pairs:
        if lig in lig_to_row and prot in prot_to_col:
            pos_cols[lig_to_row[lig]].add(prot_to_col[prot])

    aucs = []
    n_pos_cells = 0
    for i, cols in pos_cols.items():
        if not cols or len(cols) >= n_prot:
            continue  # need ≥1 positive and ≥1 negative column
        labels = np.zeros(n_prot, dtype=np.int64)
        labels[list(cols)] = 1
        n_pos_cells += len(cols)
        try:
            aucs.append(roc_auc_score(labels, score_matrix[i]))
        except ValueError:
            continue

    return {
        "matrix_per_ligand_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_ligands_counted":     int(len(aucs)),
        "n_positive_cells":      int(n_pos_cells),
    }
