"""evaluation/export_poster_figure_data.py — dump the raw values behind the two
poster figures (paper Figure 1 = fig_summary, Figure 10 = fig_attn_explainer)
to CSV.

Output tree (paper/poster_figure_data/):
  figure1_summary/
    response_map_RankBind.csv        200x200 score matrix (top-row panel)
    response_map_GraphDTA.csv        200x200 score matrix (top-row panel)
    response_map_Null_prot_prior.csv 200x200 score matrix (top-row panel)
    jaccard_top10_attractors.csv     7x7 Jaccard heatmap (bottom-left panel)
    auc_scatter.csv                  global vs per-ligand AUC + Gini (bottom-mid)
    gini_bars.csv                    Gini per model (bottom-right panel)
  figure10_attn_explainer/
    residues_long.csv                per-residue master table (+ within-protein pctile)
    panel1_class_boxplot_stats.csv   box-plot summary per residue class
    panel2_functional_residue_percentiles.csv  box-plot summary all/binding/active
    aa_attention_bias.csv            mean within-protein z(attn) per amino acid
    example_track_<UNIPROT>.csv      attention vs hydropathy track per example protein
"""
import os
import sys
import json
import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
OUT_DIR = os.path.join(_HERE, "attractor_results")
sys.path.insert(0, _HERE)

from attractor_diagnosis import compute_attractor_scores, gini  # noqa: E402
import phase_d_figures as pdf  # noqa: E402

POSTER_ROOT = os.path.join(PROJECT_ROOT, "paper", "poster_figure_data")

# residue-class map (mirrors attn_annotation_scan.CLASS, kept local to avoid
# importing the heavy torch-dependent module).
CLASS = {"D": "acidic", "E": "acidic",
         "K": "basic", "R": "basic", "H": "basic",
         "S": "polar", "T": "polar", "N": "polar", "Q": "polar", "C": "polar",
         "Y": "polar", "G": "special", "P": "special",
         "A": "hydrophobic", "V": "hydrophobic", "L": "hydrophobic",
         "I": "hydrophobic", "M": "hydrophobic", "F": "aromatic", "W": "aromatic"}
CLASS_ORDER = ["acidic", "basic", "polar", "special", "hydrophobic", "aromatic"]


def _boxstats(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, q1=np.nan, q3=np.nan,
                    whisker_lo=np.nan, whisker_hi=np.nan, min=np.nan, max=np.nan)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    wl = x[x >= lo_fence].min() if np.any(x >= lo_fence) else x.min()
    wh = x[x <= hi_fence].max() if np.any(x <= hi_fence) else x.max()
    return dict(n=len(x), mean=float(x.mean()), median=float(med),
                q1=float(q1), q3=float(q3), whisker_lo=float(wl),
                whisker_hi=float(wh), min=float(x.min()), max=float(x.max()))


def export_figure1():
    out = os.path.join(POSTER_ROOT, "figure1_summary")
    os.makedirs(out, exist_ok=True)

    # Canonical 3-seed means (seeds {42, 7, 1337}) for the RankBind v4
    # headline model, used to override the seed-42 checkpoint values below.
    global ms3
    ms3 = (pd.read_csv(os.path.join(OUT_DIR, "phase2_rankbind_multiseed.csv"))
           .set_index("config").loc["default"])
    plauc = pd.read_csv(os.path.join(OUT_DIR, "matrix_per_ligand_auc.csv"))
    rb_plauc = plauc.loc[plauc.run.str.contains("default_v4"),
                         "matrix_per_lig_auc"].iloc[0]

    matrices = pdf.load_matrices()
    # align all to the RankBind shape used in the figure
    ref = matrices.get("RankBind")
    if ref is not None:
        matrices = pdf.load_matrices(ref_shape=ref.shape)
    test_df = pdf.load_test_summaries()

    # --- top row: three response maps (RankBind, GraphDTA, Null: prot_prior) ---
    fname = {"RankBind": "response_map_RankBind.csv",
             "GraphDTA": "response_map_GraphDTA.csv",
             "Null: prot_prior": "response_map_Null_prot_prior.csv"}
    for name, f in fname.items():
        if name not in matrices:
            print(f"  [warn] matrix missing: {name}")
            continue
        M = matrices[name]
        df = pd.DataFrame(M)
        df.index.name = "ligand_idx"
        df.columns = [f"protein_{j}" for j in range(M.shape[1])]
        df.to_csv(os.path.join(out, f))
        print(f"  wrote {f}  ({M.shape[0]}x{M.shape[1]}, Gini={gini(compute_attractor_scores(M)):.3f})")

    # --- bottom-left: top-10 attractor Jaccard matrix ---
    names = list(matrices.keys())
    attr = {n: compute_attractor_scores(M) for n, M in matrices.items()}
    top = {n: set(np.argsort(-a)[:10].tolist()) for n, a in attr.items()}
    J = np.array([[len(top[a] & top[b]) / max(len(top[a] | top[b]), 1)
                   for b in names] for a in names])
    jdf = pd.DataFrame(J, index=names, columns=names)
    # RankBind headline numbers are 3-seed means over {42, 7, 1337}
    # (phase2_rankbind_multiseed.csv, config "default" = v4). The seed-42
    # checkpoint matrix alone would understate the Jaccard (0.035 vs 0.0
    # for seed 42); the poster's fig_jaccard uses the 3-seed mean.
    jdf.loc["RankBind", "Null: prot_prior"] = ms3["Jac_null_mean"]
    jdf.loc["Null: prot_prior", "RankBind"] = ms3["Jac_null_mean"]
    jdf.index.name = "model"
    jdf.to_csv(os.path.join(out, "jaccard_top10_attractors.csv"))
    print("  wrote jaccard_top10_attractors.csv")

    # --- bottom-middle + bottom-right: AUC scatter + Gini bars ---
    gini_map = {n: float(gini(compute_attractor_scores(M))) for n, M in matrices.items()}
    nm = {"graphdta": "GraphDTA", "moltrans": "MolTrans", "drugban": "DrugBAN",
          "gems": "GEMS", "rankbind": "RankBind"}
    tdf = test_df.copy()
    tdf["name"] = tdf["model"].map(nm)
    tdf["gini"] = tdf["name"].map(gini_map)
    # Same 3-seed override as above: global AUC on the poster is the 3-seed
    # mean (0.634), not the seed-42 checkpoint value (0.623), and the
    # per-ligand AUC is the matrix-level n=30 value the poster reports
    # (0.891), not the n=4 test-split estimate (0.75). Gini stays
    # checkpoint-derived and agrees with the poster (0.787).
    tdf.loc[tdf["model"] == "rankbind", "global_auc"] = ms3["gAUC_mean"]
    tdf.loc[tdf["model"] == "rankbind", "per_ligand_auc"] = rb_plauc
    cols = [c for c in ["name", "model", "global_auc", "per_ligand_auc", "gini"]
            if c in tdf.columns]
    tdf[cols].to_csv(os.path.join(out, "auc_scatter.csv"), index=False)
    print("  wrote auc_scatter.csv")

    gdf = pd.DataFrame({"model": list(gini_map.keys()),
                        "gini_attractor": list(gini_map.values())})
    gdf["is_null"] = gdf["model"].str.contains("Null")
    gdf.to_csv(os.path.join(out, "gini_bars.csv"), index=False)
    print("  wrote gini_bars.csv")


def export_figure10():
    out = os.path.join(POSTER_ROOT, "figure10_attn_explainer")
    os.makedirs(out, exist_ok=True)

    res = pd.read_csv(os.path.join(OUT_DIR, "attn_annotation_residues.csv"))
    # within-protein attention percentile — exactly as the figure computes it
    res["pctile"] = res.groupby("uniprot")["attn"].rank(pct=True)
    res["aa_class"] = res["aa"].map(CLASS)
    res.to_csv(os.path.join(out, "residues_long.csv"), index=False)
    print(f"  wrote residues_long.csv  ({len(res)} residues, "
          f"{res.uniprot.nunique()} proteins)")

    # panel (1): attn_z box-plot per residue class
    rows = []
    for c in CLASS_ORDER:
        st = _boxstats(res.loc[res["aa_class"] == c, "attn_z"].values)
        st["residue_class"] = c
        rows.append(st)
    pd.DataFrame(rows)[["residue_class", "n", "mean", "median", "q1", "q3",
                        "whisker_lo", "whisker_hi", "min", "max"]].to_csv(
        os.path.join(out, "panel1_class_boxplot_stats.csv"), index=False)
    print("  wrote panel1_class_boxplot_stats.csv")

    # panel (2): functional-residue attention percentiles
    rows = []
    for label, mask in [("all_residues", np.ones(len(res), dtype=bool)),
                        ("binding_site", res["is_binding"] == 1),
                        ("active_site", res["is_active"] == 1)]:
        st = _boxstats(res.loc[mask, "pctile"].values)
        st["group"] = label
        rows.append(st)
    pd.DataFrame(rows)[["group", "n", "mean", "median", "q1", "q3",
                        "whisker_lo", "whisker_hi", "min", "max"]].to_csv(
        os.path.join(out, "panel2_functional_residue_percentiles.csv"), index=False)
    print("  wrote panel2_functional_residue_percentiles.csv")

    # amino-acid bias table (already a paper CSV — copy through for self-containment)
    aa = pd.read_csv(os.path.join(OUT_DIR, "attn_annotation_aa.csv"))
    aa.to_csv(os.path.join(out, "aa_attention_bias.csv"), index=False)
    print("  wrote aa_attention_bias.csv")

    # example tracks: reproduce the figure's selection rule, then dump each track
    cand = []
    for uni, sub in res.groupby("uniprot"):
        if (sub.is_active.sum() + sub.is_binding.sum()) == 0 or len(sub) > 520:
            continue
        rho = spearmanr(sub["attn_z"].values, sub["hydropathy"].values).correlation
        cand.append((uni, float(rho)))
    cand.sort(key=lambda t: t[1])
    examples = []
    if cand:
        n = len(cand)
        for q in (0.60, 0.88):
            examples.append(cand[min(n - 1, int(q * (n - 1)))][0])
        seen = set()
        examples = [u for u in examples if not (u in seen or seen.add(u))]
    for uni in examples:
        sub = res[res["uniprot"] == uni].sort_values("pos").copy()
        hz = sub["hydropathy"].values
        sub["hydropathy_z"] = (hz - hz.mean()) / (hz.std() + 1e-9)
        cols = ["uniprot", "pos", "aa", "aa_class", "attn", "attn_z",
                "hydropathy", "hydropathy_z", "pctile",
                "is_active", "is_binding", "is_signal"]
        cols = [c for c in cols if c in sub.columns]
        sub[cols].to_csv(os.path.join(out, f"example_track_{uni}.csv"), index=False)
        rho = spearmanr(sub["attn_z"].values, sub["hydropathy"].values).correlation
        print(f"  wrote example_track_{uni}.csv  (rho_attn_hydro={rho:+.2f})")


def main():
    os.makedirs(POSTER_ROOT, exist_ok=True)
    print("== Figure 1 (fig_summary) ==")
    export_figure1()
    print("\n== Figure 10 (fig_attn_explainer) ==")
    export_figure10()
    print(f"\n[done] poster CSVs under {POSTER_ROOT}")


if __name__ == "__main__":
    main()
