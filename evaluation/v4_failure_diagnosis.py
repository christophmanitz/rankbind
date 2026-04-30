"""evaluation/v4_failure_diagnosis.py — Stage (c) of Phase-4 plan (PLAN.md §13.1).

Identifies which test ligands v4 (RankBind default, seed=42) ranks worst and
asks whether they cluster around chemical classes plausibly atom-level
conditioned. Output drives the go/no-go decision for Stage (b)/(a).

Inputs (read-only):
  - results/v5_rankbind/<v4_default_seed42>/score_matrix_rankbind.npy
  - results/v5_rankbind/<v4_default_seed42>/score_matrix_axes.json
  - results/v5_rankbind/<v4_default_seed42>/test_preds_rankbind.csv

Outputs:
  - evaluation/attractor_results/v4_failure_diagnosis.csv
  - evaluation/attractor_results/fig_v4_rank_hist_by_class.png
  - evaluation/attractor_results/fig_v4_atoms_vs_rank.png
  - evaluation/attractor_results/fig_v4_class_failure_rate.png
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
OUT_DIR = os.path.join(_HERE, "attractor_results")
RUN_DIR = os.path.join(
    PROJECT_ROOT,
    "results", "v5_rankbind",
    "20260423-112928_012a2695c2_default_v4",
)


# ---------------------------------------------------------------- SMARTS
# Ten chemically-coherent families covering BRENDA substrate space.
# Each pattern is intentionally broad — we want recall, not precision, since
# the diagnosis is "are atom-level details plausibly necessary here?"
SMARTS_PATTERNS = {
    "phosphate":          "[P](=O)([O,OH])([O,OH])[O,OH]",
    "phosphonate":        "[P](=O)([C,c])([O,OH])[O,OH]",
    "polyhydroxy":        "[CX4]([OH])[CX4]([OH])",          # vicinal diol — sugars/polyols
    "phenol_or_aromOH":   "c[OH]",
    "halogenated":        "[F,Cl,Br,I]",
    "amide_or_peptide":   "[NX3][CX3](=[OX1])",
    "carboxylate":        "[CX3](=O)[OX2H,OX1-]",
    "sulfonate":          "[S](=O)(=O)[O,OH]",
    "nitro":              "[NX3](=O)=O",
    "long_aliphatic":     "[CX4][CX4][CX4][CX4][CX4][CX4][CX4][CX4]",  # ≥8-carbon chain
}


def classify_ligand(smiles: str) -> tuple[list[str], int]:
    """Return (matched_class_list, n_heavy_atoms). Empty list if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], 0
    classes = []
    for name, smarts in SMARTS_PATTERNS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            classes.append(name)
    n_heavy = mol.GetNumHeavyAtoms()
    return classes, n_heavy


def load_run() -> tuple[np.ndarray, list[str], list[str], list[tuple[str, str]]]:
    M = np.load(os.path.join(RUN_DIR, "score_matrix_rankbind.npy"))
    with open(os.path.join(RUN_DIR, "score_matrix_axes.json")) as f:
        axes = json.load(f)
    lig_list = axes["axis_0_ligands"]
    prot_list = axes["axis_1_proteins"]
    preds = pd.read_csv(os.path.join(RUN_DIR, "test_preds_rankbind.csv"))
    pos = preds[preds["label"] == 1][["smiles", "uniprot"]]
    positive_pairs = list(zip(pos["smiles"].tolist(), pos["uniprot"].tolist()))
    return M, lig_list, prot_list, positive_pairs


def per_pair_ranks(
    M: np.ndarray,
    lig_list: list[str],
    prot_list: list[str],
    positive_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Mirrors metrics.matrix_ranking_metrics but exposes per-pair detail."""
    lig_to_row = {s: i for i, s in enumerate(lig_list)}
    prot_to_col = {p: j for j, p in enumerate(prot_list)}
    rows = []
    for lig, prot in positive_pairs:
        if lig not in lig_to_row or prot not in prot_to_col:
            continue
        i = lig_to_row[lig]; j = prot_to_col[prot]
        row = M[i]
        true_score = float(row[j])
        # Rank: 0-indexed (0 = top-1).
        rank = int((row > true_score).sum())
        # Confidence margin: top-1 score minus second-best non-true score.
        sorted_desc = np.sort(row)[::-1]
        top1 = float(sorted_desc[0])
        runner = float(sorted_desc[1])
        # If the true protein IS the top-1, margin = top1 - second.
        # Else margin = true_score - top1 (negative).
        margin = true_score - runner if rank == 0 else true_score - top1
        rows.append({
            "smiles":      lig,
            "uniprot":     prot,
            "rank":        rank,           # 0 = perfect
            "mrr_contrib": 1.0 / (rank + 1),
            "true_score":  true_score,
            "margin":      margin,         # positive if top-1, negative else
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    M, lig_list, prot_list, positive_pairs = load_run()
    df = per_pair_ranks(M, lig_list, prot_list, positive_pairs)
    if df.empty:
        print("No positive pairs intersected the score-matrix axes. Aborting.")
        return

    # SMARTS classification per unique SMILES.
    uniq = df["smiles"].unique()
    cache = {s: classify_ligand(s) for s in uniq}
    df["classes"] = df["smiles"].map(lambda s: ",".join(cache[s][0]) or "OTHER")
    df["n_heavy_atoms"] = df["smiles"].map(lambda s: cache[s][1])

    df = df.sort_values("rank", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(OUT_DIR, "v4_failure_diagnosis.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}  ({len(df)} pairs)")

    # Sanity recap that should match test_matrix_ranking.json.
    print("\nReproducibility check vs test_matrix_ranking.json:")
    print(f"  n_positive_pairs_matched = {len(df)}")
    print(f"  MRR  = {df['mrr_contrib'].mean():.4f}")
    print(f"  H@1  = {(df['rank'] < 1).mean():.4f}")
    print(f"  H@5  = {(df['rank'] < 5).mean():.4f}")
    print(f"  H@10 = {(df['rank'] < 10).mean():.4f}")

    # ---------------- Aggregations for the decision-gate ----------------
    bottom_q = df["rank"].quantile(0.75)
    bot_df = df[df["rank"] >= bottom_q]
    print(f"\nBottom-quartile cutoff: rank ≥ {int(bottom_q)}  (n={len(bot_df)})")

    classes_in_bottom = []
    for clist in bot_df["classes"]:
        for c in (clist.split(",") if clist else ["OTHER"]):
            classes_in_bottom.append(c)
    cls_counts_bot = pd.Series(classes_in_bottom).value_counts()
    classes_in_all = []
    for clist in df["classes"]:
        for c in (clist.split(",") if clist else ["OTHER"]):
            classes_in_all.append(c)
    cls_counts_all = pd.Series(classes_in_all).value_counts()

    cls_table = pd.concat(
        [cls_counts_all.rename("n_total"), cls_counts_bot.rename("n_bottom_q")],
        axis=1,
    ).fillna(0).astype(int)
    cls_table["bottom_share"] = cls_table["n_bottom_q"] / cls_table["n_total"].clip(lower=1)
    cls_table = cls_table.sort_values("bottom_share", ascending=False)
    print("\nClass × bottom-quartile breakdown (sorted by failure share):")
    print(cls_table.to_string())

    cls_table.to_csv(os.path.join(OUT_DIR, "v4_failure_diagnosis_classes.csv"))

    # ---------------- Plots ----------------
    # (1) Rank histogram per class
    classes_sorted = list(cls_table.index)
    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.linspace(0, df["rank"].max() + 1, 25)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(classes_sorted)))
    for color, cls in zip(cmap, classes_sorted):
        ranks_c = []
        for _, r in df.iterrows():
            if cls in r["classes"].split(",") or (cls == "OTHER" and r["classes"] == "OTHER"):
                ranks_c.append(r["rank"])
        if ranks_c:
            ax.hist(ranks_c, bins=bins, alpha=0.55, label=f"{cls} (n={len(ranks_c)})", color=color)
    ax.set_xlabel("Rank of true protein (0 = top-1, 199 = worst)")
    ax.set_ylabel("Count of positive pairs")
    ax.set_title("v4 default: rank distribution per chemical class (overlapping classes counted once each)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "fig_v4_rank_hist_by_class.png")
    plt.savefig(p1, dpi=150); plt.close()
    print(f"Wrote {p1}")

    # (2) Atoms vs rank scatter
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df["n_heavy_atoms"], df["rank"], alpha=0.6, edgecolor="black", linewidth=0.4)
    if len(df) >= 3:
        coef = np.polyfit(df["n_heavy_atoms"], df["rank"], 1)
        xs = np.array([df["n_heavy_atoms"].min(), df["n_heavy_atoms"].max()])
        ax.plot(xs, np.polyval(coef, xs), "r--", lw=1.2, label=f"linear fit slope={coef[0]:.2f}")
    spear = df[["n_heavy_atoms", "rank"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("Heavy atoms in ligand")
    ax.set_ylabel("Rank of true protein (0 = best)")
    ax.set_title(f"v4 default: ligand size vs ranking failure  (Spearman ρ = {spear:.3f})")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "fig_v4_atoms_vs_rank.png")
    plt.savefig(p2, dpi=150); plt.close()
    print(f"Wrote {p2}")

    # (3) Class bottom-quartile failure rate (sanity check for the gate)
    fig, ax = plt.subplots(figsize=(9, 5))
    cls_to_plot = cls_table[cls_table["n_total"] >= 2]  # drop n=1 noise
    ax.barh(cls_to_plot.index[::-1], cls_to_plot["bottom_share"][::-1],
            color=["#b02a37" if x >= 0.5 else "#888" for x in cls_to_plot["bottom_share"][::-1]])
    ax.axvline(0.25, color="black", linestyle=":", alpha=0.6, label="uniform (25%)")
    ax.set_xlabel("Share of class falling in bottom-quartile rank")
    ax.set_title("Which chemical classes are over-represented in v4's worst rankings?")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    for i, (cls, row) in enumerate(cls_to_plot[::-1].iterrows()):
        ax.text(row["bottom_share"] + 0.01, i, f"n={row['n_total']}", va="center", fontsize=8)
    plt.tight_layout()
    p3 = os.path.join(OUT_DIR, "fig_v4_class_failure_rate.png")
    plt.savefig(p3, dpi=150); plt.close()
    print(f"Wrote {p3}")

    # ---------------- Gate inputs printed for the memo ----------------
    print("\n=== INPUTS FOR DECISION-GATE (PLAN §13.1) ===")
    bot_classes_share = (cls_table[cls_table["n_total"] >= 2]
                         .nlargest(3, "bottom_share")[["n_total", "n_bottom_q", "bottom_share"]])
    print("Top-3 over-represented classes in bottom-quartile (n>=2):")
    print(bot_classes_share.to_string())
    print(f"\nSpearman(n_heavy_atoms, rank) = {spear:.3f}")
    n_bot_in_top2 = sum(
        1 for clist in bot_df["classes"]
        for c in clist.split(",")
        if c in bot_classes_share.index[:2].tolist()
    )
    pct_bot_in_top2 = n_bot_in_top2 / max(len(bot_df), 1)
    print(f"Fraction of bottom-quartile rows tagged with top-2 over-represented classes: {pct_bot_in_top2:.2%}")
    print("\nGate criterion: ≥30% bottom-Q in 1-2 atom-conditioned classes  OR  Spearman ρ > 0.4")


if __name__ == "__main__":
    main()
