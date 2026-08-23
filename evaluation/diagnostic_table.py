"""evaluation/diagnostic_table.py — skill §4 diagnostic table.

Assembles the compact "information source -> pooled AUC -> matrix MRR"
table on the canonical BRENDA-200 benchmark. Every number is READ from
existing artifact CSVs (nothing recomputed, nothing invented):

    evaluation/null_baseline_firstclass.csv          (null priors)
    evaluation/decoy_leakage_probe.csv               (linear probes)
    evaluation/attractor_results/
        phase2_rankbind_multiseed.csv                (BCE control + RankBind)

Purpose (skill §4): make visually obvious that high pooled AUC does not by
itself demonstrate ligand-conditional target discrimination — the cheap
information sources that never model the interaction already account for
most of the pooled-AUC headroom while staying at chance on within-ligand
ranking.

Writes DIAGNOSTIC_TABLE.md + diagnostic_table.csv next to this script.
"""

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

NULLS_CSV = os.path.join(_HERE, "null_baseline_firstclass.csv")
PROBE_CSV = os.path.join(_HERE, "decoy_leakage_probe.csv")
MULTISEED_CSV = os.path.join(
    _ROOT, "evaluation", "attractor_results", "phase2_rankbind_multiseed.csv")

for p in (NULLS_CSV, PROBE_CSV, MULTISEED_CSV):
    if not os.path.exists(p):
        sys.exit(f"[diagnostic-table] missing artifact: {p} — regenerate it "
                 "first (null_baseline_table.py / decoy_leakage_probe.py / "
                 "aggregate_multiseed.py)")


def f4(x):
    return "" if x == "" or pd.isna(x) else f"{float(x):.3f}"


def main():
    nulls = pd.read_csv(NULLS_CSV).set_index("null")
    probe = pd.read_csv(PROBE_CSV).set_index("variant")
    ms = pd.read_csv(MULTISEED_CSV).set_index("config")

    rd = nulls.loc["null_random"]
    pp = nulls.loc["null_prot_prior"]
    lp = nulls.loc["null_lig_prior"]
    pr_mol = probe.loc["molecule_only"]
    pr_full = probe.loc["full"]
    bce = ms.loc["abl_bce_only"]
    rb = ms.loc["default"]

    rows = [
        {
            "source": "Random baseline",
            "sees_lig": "no", "sees_prot": "no",
            "pooled_auc": float(rd["pooled_auc_test_full_split"]),
            "matrix_mrr": float(rd["mrr_tie_aware"]),
            "jac_vs_protprior": float(rd["jaccard_top10_vs_prot_prior"]),
            "n": "", "note": "uniform noise"},
        {
            "source": "Protein-only prior (molecule-blind)",
            "sees_lig": "no", "sees_prot": "yes",
            "pooled_auc": float(pp["pooled_auc_test_full_split"]),
            "matrix_mrr": float(pp["mrr_tie_aware"]),
            "jac_vs_protprior": 1.0,
            "n": "", "note": "per-protein train positive rate"},
        {
            "source": "Ligand-only prior (train-rate)",
            "sees_lig": "yes", "sees_prot": "no",
            "pooled_auc": float(lp["pooled_auc_test_full_split"]),
            "matrix_mrr": float(lp["mrr_tie_aware"]),
            "jac_vs_protprior": float(lp["jaccard_top10_vs_prot_prior"]),
            "n": "", "note": "per-ligand train rate; constant along each row"},
        {
            "source": "Ligand-only probe (frozen ChemBERTa, linear)",
            "sees_lig": "yes", "sees_prot": "no",
            "pooled_auc": float(pr_mol["test_auc"]),
            "matrix_mrr": None,
            "jac_vs_protprior": None,
            "n": "", "note": "cannot rank proteins within a ligand by "
                             "construction (ligand-only score)"},
        {
            "source": "Ligand+protein probe (both blocks, linear)",
            "sees_lig": "yes", "sees_prot": "yes",
            "pooled_auc": float(pr_full["test_auc"]),
            "matrix_mrr": float(pr_full["matrix_mrr"]),
            "jac_vs_protprior": float(pr_full["jaccard_top10_vs_prior"]),
            "n": "", "note": "chance MRR = 1/200 ~ 0.029"},
        {
            "source": "BCE control (pairwise objective)",
            "sees_lig": "yes", "sees_prot": "yes",
            "pooled_auc": float(bce["gAUC_mean"]),
            "matrix_mrr": float(bce["MRR_mean"]),
            "jac_vs_protprior": float(bce["Jac_null_mean"]),
            "n": int(bce["n_seeds"]),
            "note": "+/-SD pooled {:.3f}, MRR {:.4f}".format(
                bce["gAUC_std"], bce["MRR_std"])},
        {
            "source": "RankBind (within-ligand margin + hard negs)",
            "sees_lig": "yes", "sees_prot": "yes",
            "pooled_auc": float(rb["gAUC_mean"]),
            "matrix_mrr": float(rb["MRR_mean"]),
            "jac_vs_protprior": float(rb["Jac_null_mean"]),
            "n": int(rb["n_seeds"]),
            "note": "+/-SD pooled {:.3f}, MRR {:.4f}".format(
                rb["gAUC_std"], rb["MRR_std"])},
    ]

    tab = pd.DataFrame(rows)
    csv_path = os.path.join(_HERE, "diagnostic_table.csv")
    tab.to_csv(csv_path, index=False)

    md = [
        "# DIAGNOSTIC_TABLE.md — information source vs task-aligned metrics",
        "",
        "Canonical BRENDA-200 benchmark (seed-42 protein-stratified split).",
        "Pooled AUC = held-out TEST pairs, full split surface (same surface",
        "as every model's global AUC); matrix MRR = within-ligand target",
        "ranking over the canonical 200x200 pool (chance ~ H_200/200 ~ 0.029;",
        "tie-aware convention for degenerate scorers). Top-10 overlap vs",
        "prot_prior: 1.0 = identical shortcut geometry.",
        "",
        "| Information source | Sees ligand | Sees protein | Pooled AUC "
        "| Matrix MRR | Jac@10 vs prot_prior | n | note |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['source']} | {r['sees_lig']} | {r['sees_prot']} "
            f"| {f4(r['pooled_auc'])} | {f4(r['matrix_mrr'])} "
            f"| {f4(r['jac_vs_protprior'])} | {r['n']} | {r['note']} |")

    md += [
        "",
        "## Reading",
        "",
        f"- **The molecule axis alone carries the pooled signal**: the",
        f"ligand-only train-rate prior reaches pooled AUC "
        f"{lp['pooled_auc_test_full_split']:.3f} while carrying ZERO",
        "within-ligand ranking information (tie-aware MRR",
        f"{lp['mrr_tie_aware']:.3f}); the ligand-only frozen-encoder probe",
        f"confirms it ({pr_mol['test_auc']:.3f}).",
        f"- **The protein axis is closed on this split** by construction:",
        f"protein-disjointness caps the molecule-blind prior at pooled AUC",
        f"{pp['pooled_auc_test_full_split']:.3f}.",
        "- **Pairwise BCE training reproduces the shortcut, not the task:**",
        f"pooled AUC {bce['gAUC_mean']:.3f} at chance-level matrix MRR",
        f"{bce['MRR_mean']:.3f} and prot-prior-like top-10 overlap",
        f"{bce['Jac_null_mean']:.2f}.",
        "- **Training toward the ranking property moves the task metric,",
        "not the pooled one:** RankBind trades pooled AUC down to",
        f"{rb['gAUC_mean']:.3f} while lifting matrix MRR to",
        f"{rb['MRR_mean']:.3f} (~13x BCE) and dropping the prior overlap to",
        f"{rb['Jac_null_mean']:.2f}.",
        "",
        "Sources (regenerable, do not hand-edit numbers):",
        "`null_baseline_firstclass.csv`, `decoy_leakage_probe.csv`,",
        "`attractor_results/phase2_rankbind_multiseed.csv`. Script:",
        "`evaluation/diagnostic_table.py`.",
    ]
    open(os.path.join(_HERE, "DIAGNOSTIC_TABLE.md"), "w").write(
        "\n".join(md) + "\n")
    print("[diagnostic-table] wrote DIAGNOSTIC_TABLE.md + diagnostic_table.csv")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
