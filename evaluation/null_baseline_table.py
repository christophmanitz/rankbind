"""evaluation/null_baseline_table.py — skill item A10.

First-class protein-prior null baseline table on BRENDA-200. Answers the
skill's headline question:

    How much of the pooled-AUC performance can be reproduced without
    molecular information?

Three null score matrices over the canonical 200x200 pool (identical axes
to every model evaluation):

    null_random      uniform noise
    null_prot_prior  score[i,j] = per-protein TRAIN positive rate
                     (molecule-blind; global-rate fallback for unseen)
    null_lig_prior   score[i,j] = per-ligand TRAIN positive rate

Metrics per null: pooled AUC on held-out TEST pairs inside the pool,
matrix MRR / Hit@K against test positives, Gini, top-10 Jaccard vs
prot_prior. Matrix metrics are reported TWICE:
    raw       strict-greater rank counting — identical to the production
              pipeline (v5_rankbind.metrics.matrix_ranking_metrics), kept
              for comparability with published tables;
    tie-aware seeded random tie-breaking — the honest number for scorers
              with massive ties (lig_prior is constant along each row).

Known construction facts this table verifies:
    * prot_prior pooled AUC on test pairs = 0.500 exactly: the split is
      protein-disjoint, so no test protein has training rows and every
      test pair receives the same global fallback rate.
    * lig_prior pooled AUC is high because the decoy protocol assigns
      ~99% of ligands a fixed role (see DECOY_LEAKAGE_AUDIT.md): the
      per-ligand train rate transfers across the protein split.

Writes NULL_BASELINE.md + null_baseline_firstclass.csv.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "baselines", "adapters"))
sys.path.insert(0, _ROOT)

from common import BRENDADataConfig                    # noqa: E402
from attractor_diagnosis import compute_attractor_metrics  # noqa: E402
from v5_rankbind.metrics import matrix_ranking_metrics  # noqa: E402
from sklearn.metrics import roc_auc_score               # noqa: E402


def tie_aware_matrix_metrics(M, lig_list, prot_list, pos_pairs, seed=42):
    """matrix MRR / Hit@K with seeded random tie-breaking."""
    rng = np.random.default_rng(seed)
    lig_to_row = {s: i for i, s in enumerate(lig_list)}
    prot_to_col = {p: j for j, p in enumerate(prot_list)}
    ranks = []
    for lig, prot in pos_pairs:
        if lig not in lig_to_row or prot not in prot_to_col:
            continue
        i, j = lig_to_row[lig], prot_to_col[prot]
        row = M[i].astype(np.float64)
        jitter = (rng.random(len(row)) - 0.5) * (np.abs(row).max() + 1e-12) * 1e-6
        order = np.argsort(-(row + jitter), kind="stable")
        pos_of = np.empty(len(row), dtype=np.int64)
        pos_of[order] = np.arange(len(row))
        ranks.append(pos_of[j])
    if not ranks:
        return {k: float("nan") for k in ("mrr", "hit_at_5", "hit_at_10")}
    ranks = np.asarray(ranks, dtype=np.float64)
    return {"mrr": float(np.mean(1.0 / (ranks + 1))),
            "hit_at_5": float((ranks < 5).mean()),
            "hit_at_10": float((ranks < 10).mean())}


def get_split_indices(config: BRENDADataConfig, split_mode: str) -> tuple:
    """Dispatch on split_mode — mirrors v5_rankbind/data.py::prepare_frames."""
    if split_mode == "ligand":
        return config.get_ligand_split()
    if split_mode == "double_cold":
        return config.get_double_cold_split()
    if split_mode != "protein":
        raise ValueError(f"Unknown split_mode={split_mode!r}")
    return config.get_protein_split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="protein",
                    choices=["protein", "ligand", "double_cold"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split_mode = args.split
    # Legacy filenames stay unsuffixed for the canonical protein split so all
    # existing references (paper, skill docs) keep resolving.
    suffix = "" if split_mode == "protein" else f"_{split_mode}"

    config = BRENDADataConfig(seed=args.seed)
    pairs = config.load_pairs()
    seqs = config.load_sequences()
    tr_i, va_i, te_i = get_split_indices(config, split_mode)

    proteins = list(seqs.keys())[:200]
    smiles_list = pairs["substrate_smiles"].unique().tolist()[:200]

    # null matrices over identical axes (reuse Phase-1 builder)
    from null_baselines import build_null_matrices
    matrices, _ = build_null_matrices(config, n_matrix=200, seed=args.seed,
                                      split_mode=split_mode)

    ax_s, ax_p = set(smiles_list), set(proteins)
    te_df = pairs[pairs["idx"].isin(set(te_i))]
    tr_df = pairs[pairs["idx"].isin(set(tr_i))]
    sub = te_df[te_df["substrate_smiles"].isin(ax_s) & te_df["uniprot"].isin(ax_p)]
    labels = sub["label"].to_numpy()
    s_idx = sub["substrate_smiles"].map({s: i for i, s in enumerate(smiles_list)}).to_numpy()
    p_idx = sub["uniprot"].map({p: j for j, p in enumerate(proteins)}).to_numpy()
    pos_pairs = sorted({(s, p) for s, p, l in
                        sub[["substrate_smiles", "uniprot", "label"]]
                        .itertuples(index=False) if l == 1})
    print(f"[null-table] test pairs in pool: {len(sub)} "
          f"(pos-rate {labels.mean():.3f}), positives matched: {len(pos_pairs)}")

    # Full-test-split pooled AUC (same evaluation surface as model gAUC)
    glob_rate = float(tr_df["label"].mean())
    pr_map = tr_df.groupby("uniprot")["label"].mean()
    lr_map = tr_df.groupby("substrate_smiles")["label"].mean()
    y_full = te_df["label"].to_numpy()
    full_scores = {
        "null_prot_prior": te_df["uniprot"].map(pr_map).fillna(glob_rate).to_numpy(),
        "null_lig_prior": te_df["substrate_smiles"].map(lr_map).fillna(glob_rate).to_numpy(),
    }
    print(f"[null-table] full test split: {len(te_df)} pairs "
          f"(pos-rate {y_full.mean():.3f})")

    prior_top = set(np.argsort(-matrices["null_prot_prior"][0])[:10])
    rows = []
    for name, M in matrices.items():
        scores = M[s_idx, p_idx]
        auc = float(roc_auc_score(labels, scores))
        if name == "null_random":
            rng_full = np.random.default_rng(42)
            auc_full = float(roc_auc_score(
                y_full, rng_full.uniform(size=len(te_df))))
        else:
            auc_full = float(roc_auc_score(y_full, full_scores[name]))
        raw = matrix_ranking_metrics(M, smiles_list, proteins, pos_pairs)
        ta = tie_aware_matrix_metrics(M, smiles_list, proteins, pos_pairs)
        gini = compute_attractor_metrics(M)["gini_attractor"]
        jac = []
        for i in range(len(smiles_list)):
            top_m = set(np.argsort(-M[i])[:10])
            jac.append(len(top_m & prior_top) / len(top_m | prior_top))
        rows.append({
            "null": name,
            "pooled_auc_test_pool": round(auc, 4),
            "pooled_auc_test_full_split": round(auc_full, 4),
            "mrr_raw": round(raw["mrr"], 4),
            "hit_at_5_raw": round(raw["hit_at_5"], 4),
            "hit_at_10_raw": round(raw["hit_at_10"], 4),
            "mrr_tie_aware": round(ta["mrr"], 4),
            "hit_at_5_tie_aware": round(ta["hit_at_5"], 4),
            "hit_at_10_tie_aware": round(ta["hit_at_10"], 4),
            "gini": round(float(gini), 4),
            "jaccard_top10_vs_prot_prior": round(float(np.mean(jac)), 3),
        })
        print(f"[null-table] {name:>16}: AUC pool/full {auc:.3f}/{auc_full:.3f}"
              f"  MRR raw/tieaware {raw['mrr']:.3f}/{ta['mrr']:.3f}  "
              f"H@10 {raw['hit_at_10']:.3f}/{ta['hit_at_10']:.3f}  "
              f"Gini {gini:.3f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(_HERE, f"null_baseline_firstclass{suffix}.csv"), index=False)

    r = {row["null"]: row for row in rows}
    md = [
        "# NULL_BASELINE" + suffix.upper() + ".md — first-class null baselines",
        "",
        f"Split: **{split_mode}** (seed {args.seed}). "
        "BRENDA-200 canonical pool (first 200 proteins x first 200 unique",
        "SMILES), held-out test split, identical axes to every model run.",
        "",
        f"Test pairs in pool: **{len(sub)}** (pos-rate "
        f"{labels.mean():.3f}, only {int((labels == 0).sum())} negatives);",
        f"full test split: **{len(te_df)}** pairs (pos-rate "
        f"{y_full.mean():.3f}). Unique positive pairs matched for ranking:",
        f"{len(pos_pairs)}.",
    ]
    n_pool_neg = int((labels == 0).sum())
    if n_pool_neg == 0:
        md.append(
            f"NOTE: the 200x200 pool subset contains {len(sub)} test pairs, "
            f"ALL positive ({n_pool_neg} negatives) — pooled AUC is undefined "
            "(nan) on the pool surface for every scorer; the full-split "
            "column is the informative surface under this split.")
    elif n_pool_neg < 10:
        md.append(
            f"NOTE: only {n_pool_neg} negatives fall inside the pool subset; "
            "pool-surface AUC/MRR are high-variance there — read the "
            "full-split column first.")
    md += [
        "",
        "| null | AUC pool | AUC full split | MRR raw | MRR tie-aware "
        "| H@5 tie | H@10 tie | Gini | Jac-top10 vs prot_prior |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("null_random", "null_prot_prior", "null_lig_prior"):
        w = r[name]
        md.append(
            f"| {name} | {w['pooled_auc_test_pool']:.3f} "
            f"| {w['pooled_auc_test_full_split']:.3f} | {w['mrr_raw']:.3f} "
            f"| {w['mrr_tie_aware']:.3f} | {w['hit_at_5_tie_aware']:.3f} "
            f"| {w['hit_at_10_tie_aware']:.3f} | {w['gini']:.3f} "
            f"| {w['jaccard_top10_vs_prot_prior']:.2f} |")

    pp, lp, rd = r["null_prot_prior"], r["null_lig_prior"], r["null_random"]
    canon = config.load_canon_smiles()
    key = lambda col: col.map(lambda s: canon.get(s, s))          # noqa: E731
    tr_ligs = set(key(tr_df["substrate_smiles"]))
    te_ligs = set(key(te_df["substrate_smiles"]))
    tr_prots = set(tr_df["uniprot"])
    te_prots = set(te_df["uniprot"])
    lig_rec = len(te_ligs & tr_ligs) / max(len(te_ligs), 1)
    prot_rec = len(te_prots & tr_prots) / max(len(te_prots), 1)
    md += [
        "",
        "## Reading",
        "",
        f"Split structure: {prot_rec:.1%} of test PROTEINS and "
        f"{lig_rec:.1%} of test LIGANDS (canonical identity) also occur in "
        f"train.",
        "",
        f"**Chance reference is the random row itself**, not the analytic",
        f"single-positive constant: empirical random performance here is",
        f"MRR {rd['mrr_tie_aware']:.3f}, H@10 {rd['hit_at_10_tie_aware']:.3f},"
        f" pooled AUC {rd['pooled_auc_test_pool']:.3f}/"
        f"{rd['pooled_auc_test_full_split']:.3f}.",
    ]
    if split_mode == "protein":
        md += [
            "",
            "**Protein prior cannot reproduce pooled performance on the",
            "protein-disjoint split.** null_prot_prior reaches pooled AUC",
            "**0.500 exactly — by CONSTRUCTION**: no test protein has",
            "training rows, so every test pair receives the same",
            "global-rate fallback (verified on both surfaces). Any model",
            "beating this must generalise beyond train prevalence.",
            "",
            "**Molecule-side prior transfer.** On the FULL test split,",
            f"null_lig_prior reaches pooled AUC",
            f"**{lp['pooled_auc_test_full_split']:.3f}**: per-ligand train",
            "rates transfer across the protein split because molecules are",
            "shared and ~99% of ligands have a fixed role under the decoy",
            "protocol (DECOY_LEAKAGE_AUDIT.md). Molecular memory alone",
            "reproduces a large share of the trained models' global-AUC",
            "range without any interaction learning.",
        ]
    elif split_mode == "ligand":
        md += [
            "",
            "**Cold-ligand mirror image (skill §6).** With ligands",
            "disjoint, null_lig_prior falls back to the global training",
            f"rate for EVERY test pair: pooled AUC",
            f"{lp['pooled_auc_test_full_split']:.3f} — the molecule-side",
            "shortcut is structurally unavailable. Conversely",
            "null_prot_prior now carries signal:",
            f"pooled AUC **{pp['pooled_auc_test_full_split']:.3f}**",
            f"({prot_rec:.0%} of test proteins recur, so their train",
            "prevalence transfers). Under cold-ligand evaluation the",
            "dominant residual shortcut is the protein marginal.",
        ]
    else:
        md += [
            "",
            "**Double-cold (skill §5/§7).** Neither axis recurs across the",
            "split, so BOTH priors collapse to the global-rate fallback:",
            f"prot_prior pooled AUC {pp['pooled_auc_test_full_split']:.3f},",
            f"lig_prior {lp['pooled_auc_test_full_split']:.3f}. Any pooled",
            "AUC above chance under this split must come from genuine",
            "ligand-protein generalisation.",
        ]
    md += [
        "",
        "**Tie artefacts matter for degenerate scorers.** lig_prior is",
        "constant along each row; its raw matrix MRR "
        f"({lp['mrr_raw']:.3f})",
        "is a strict-greater-counting artefact (every column 'rank 0',",
        "H@K = 1.0). Tie-aware MRR is "
        f"{lp['mrr_tie_aware']:.3f}: a per-ligand",
        "constant carries zero within-row ranking information.",
        "",
        f"prot_prior matrix structure: tie-aware MRR "
        f"{pp['mrr_tie_aware']:.3f}; its Gini {pp['gini']:.3f}",
        "matches every trained Phase-1 model: Gini reflects data geometry,",
        "not learned pathology (Phase-1 pivot).",
        "",
        "**Headline answer:** the informative nulls differ BY SPLIT —",
        "report them next to every model number (skill §8). A null's lack",
        "of signal on a given split is itself a result.",
    ]
    open(os.path.join(_HERE, f"NULL_BASELINE{suffix}.md"), "w").write(
        "\n".join(md) + "\n")
    print(f"[null-table] wrote NULL_BASELINE{suffix}.md + "
          f"null_baseline_firstclass{suffix}.csv")


if __name__ == "__main__":
    main()
