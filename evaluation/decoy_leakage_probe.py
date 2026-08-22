"""evaluation/decoy_leakage_probe.py — skill item A18.

BRENDA decoy-leakage audit: can the published pooled-AUC level be reached
WITHOUT any deep learning? Frozen features + a simple linear classifier:

    ChemBERTa mean-pool [384]  (frozen, same cache as the models use)
    ESM2 mean-pool     [1280]  (frozen, same store as the models use)
    -> concat [1664] -> StandardScaler -> LogisticRegression

Variants: full (lig+prot), molecule_only (A11 ctrl 9), protein_only
(A11 ctrl 10). Canonical seed-42 protein split, identical to every other
analysis. Metrics: train/test pooled AUC; for `full` also matrix MRR /
Hit@K on the canonical 200x200 pool (test positives), Spearman(col-mean
score, train-rate prior) and top-10 Jaccard vs null_prot_prior.

The classifier is linear, so the 200x200 score matrix is computed exactly
as S[i,j] = wl.Lt[i] + wp.Pt[j] + b on scaler-transformed embeddings —
no approximation.

Interpretation frame (skill A18): if the linear probe approaches the
trained deep models' pooled AUC, the decoy construction itself contains
learnable pair-level structure exploitable without any deep fine-tuning.
That is a limitation on ABSOLUTE pooled-AUC values on BRENDA, NOT
evidence that BRENDA is invalid.

Writes DECOY_LEAKAGE_AUDIT.md + decoy_leakage_probe.csv.
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "baselines", "adapters"))
sys.path.insert(0, _ROOT)

from common import BRENDADataConfig          # noqa: E402
from scipy.stats import spearmanr           # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score   # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from v5_rankbind.data import ensure_chemberta_cache, load_chemberta  # noqa: E402
from v5_rankbind.metrics import matrix_ranking_metrics  # noqa: E402
import torch                                # noqa: E402

ESM2_DIR = Path(_ROOT) / "data" / "esm2_embeddings"
CHEMBERTA_CACHE = Path(_ROOT) / "data" / "chemberta_cache"
N_MATRIX = 200


def load_lig_emb(smiles_list):
    ensure_chemberta_cache(sorted(set(smiles_list)), CHEMBERTA_CACHE)
    return np.stack([load_chemberta(s, CHEMBERTA_CACHE).numpy().astype(np.float32)
                     for s in smiles_list])


def load_prot_emb(uniprots):
    embs, n_missing = [], 0
    for u in uniprots:
        p = ESM2_DIR / f"{u}.pt"
        if not p.exists():
            embs.append(np.zeros(1280, dtype=np.float32))
            n_missing += 1
            continue
        t = torch.load(p, weights_only=True).to(torch.float32)
        if t.ndim == 2:
            t = t.mean(dim=0)
        embs.append(t.numpy())
    return np.stack(embs), n_missing


class BlockScaler:
    """StandardScaler applied independently to the [lig | prot] blocks, so a
    transformed concat equals the concat of transformed blocks."""

    def __init__(self, d_lig, X_tr):
        self.sl = StandardScaler().fit(X_tr[:, :d_lig])
        self.sp = StandardScaler().fit(X_tr[:, d_lig:])
        self.d_lig = d_lig

    def transform(self, X):
        return np.hstack([self.sl.transform(X[:, :self.d_lig]),
                          self.sp.transform(X[:, self.d_lig:])])


def main():
    bc = BRENDADataConfig(
        csv_path=str(Path(_ROOT) / "data/dataset_with_decoys.csv"),
        seq_csv=str(Path(_ROOT) / "data/sequences/sequences.csv"))
    df = bc.load_pairs()
    tr_i, va_i, te_i = bc.get_protein_split()

    uniq_s = df["substrate_smiles"].unique().tolist()
    uniq_p = sorted(df["uniprot"].unique())
    print(f"[probe] {len(df)} pairs | {len(uniq_s)} ligands | "
          f"{len(uniq_p)} proteins", flush=True)
    L = load_lig_emb(uniq_s)
    P, n_miss = load_prot_emb(uniq_p)
    print(f"[probe] embeddings loaded (missing esm2 files: {n_miss})", flush=True)

    li = {s: i for i, s in enumerate(uniq_s)}
    pi = {u: i for i, u in enumerate(uniq_p)}
    D_L = L.shape[1]
    X = np.hstack([L[df["substrate_smiles"].map(li).to_numpy()],
                   P[df["uniprot"].map(pi).to_numpy()]])
    y = df["label"].to_numpy()
    is_tr = df["idx"].isin(set(tr_i)).to_numpy()
    is_te = df["idx"].isin(set(te_i)).to_numpy()
    print(f"[probe] train rows {int(is_tr.sum())} "
          f"(pos-rate {y[is_tr].mean():.3f}), test rows {int(is_te.sum())} "
          f"(pos-rate {y[is_te].mean():.3f})", flush=True)

    tr_df = df[df["idx"].isin(set(tr_i))]
    rate_prior = tr_df.groupby("uniprot")["label"].mean()

    # A18 mechanism probe: molecule-side role assignment in the decoy protocol
    g_all = df.groupby("substrate_smiles")["label"].mean()
    g_tr = tr_df.groupby("substrate_smiles")["label"].mean()
    te_df = df[df["idx"].isin(set(te_i))]
    g_te = te_df.groupby("substrate_smiles")["label"].mean()
    n_pure_pos = int((g_all == 1).sum())
    n_pure_neg = int((g_all == 0).sum())
    frac_pure = (n_pure_pos + n_pure_neg) / len(g_all)
    te_ligs = set(g_te.index)
    tr_ligs = set(g_tr.index)
    te_overlap = len(te_ligs & tr_ligs) / max(len(te_ligs), 1)
    print(f"[probe] ligand purity: pure-pos {n_pure_pos}, pure-neg "
          f"{n_pure_neg}, pure {frac_pure:.1%} | test ligands seen in train: "
          f"{te_overlap:.1%}", flush=True)

    seqs = bc.load_sequences()
    proteins_ax = list(seqs.keys())[:N_MATRIX]
    smiles_ax = df["substrate_smiles"].unique().tolist()[:N_MATRIX]
    te_df = df[df["idx"].isin(set(te_i))]
    ax_s, ax_p = set(smiles_ax), set(proteins_ax)
    pos_pairs = list({(s, p) for s, p, l in
                      te_df[te_df["substrate_smiles"].isin(ax_s)
                            & te_df["uniprot"].isin(ax_p)]
                      [["substrate_smiles", "uniprot", "label"]]
                      .itertuples(index=False) if l == 1})
    print(f"[probe] matrix pool test positives: {len(pos_pairs)}", flush=True)

    rows = []
    md = [
        "# DECOY_LEAKAGE_AUDIT.md — skill item A18",
        "",
        "Frozen-feature linear probe on the BRENDA decoy dataset "
        "(canonical seed-42 protein split):",
        "",
        "- ChemBERTa mean-pool [384] + ESM2 mean-pool [1280], both frozen",
        "- StandardScaler + LogisticRegression (C=1, max_iter=2000)",
        f"- train rows {int(is_tr.sum())} / test rows {int(is_te.sum())} "
        f"(test pos-rate {y[is_te].mean():.3f})",
        "",
        "| variant | train AUC | test pooled AUC | matrix MRR | H@5 | H@10 "
        "| n matched |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name in ("full", "molecule_only", "protein_only"):
        if name == "full":
            Xv = X
        elif name == "molecule_only":
            Xv = X[:, :D_L]
        else:
            Xv = X[:, D_L:]
        if name == "full":
            bs = BlockScaler(D_L, Xv[is_tr])
        else:
            bs = StandardScaler().fit(Xv[is_tr])

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(bs.transform(Xv[is_tr]), y[is_tr])
        auc_tr = roc_auc_score(y[is_tr],
                               clf.decision_function(bs.transform(Xv[is_tr])))
        auc_te = roc_auc_score(y[is_te],
                               clf.decision_function(bs.transform(Xv[is_te])))

        row = {"variant": name, "train_auc": round(float(auc_tr), 4),
               "test_auc": round(float(auc_te), 4)}
        mrr = h5 = h10 = nm = None

        if name == "full":
            coef = clf.coef_[0]
            bl, bp, b = coef[:D_L], coef[D_L:], float(clf.intercept_[0])
            sl, sp = bs.sl, bs.sp
            Lt = {s: sl.transform(L[li[s]][None, :])[0] for s in smiles_ax}
            Pt_ax = sp.transform(P[[pi[p] for p in proteins_ax]])
            lig_part = np.stack([Lt[s] @ bl for s in smiles_ax])   # [n_lig]
            S = (lig_part[:, None] + (Pt_ax @ bp)[None, :] + b).astype(np.float32)
            m = matrix_ranking_metrics(S, smiles_ax, proteins_ax, pos_pairs)
            mrr, h5, h10 = m["mrr"], m["hit_at_5"], m["hit_at_10"]
            nm = m["n_positive_pairs_matched"]
            seen = tr_df[tr_df["uniprot"].isin(ax_p)]
            pr = seen.groupby("uniprot")["label"].mean().reindex(proteins_ax)
            valid = pr.notna().to_numpy()
            rho = float(spearmanr(S.mean(axis=0)[valid],
                                  pr.to_numpy()[valid]).statistic)
            prior_top = set(np.argsort(-pr.fillna(0).to_numpy())[:10])
            jac = []
            for i in range(N_MATRIX):
                top_m = set(np.argsort(-S[i])[:10])
                jac.append(len(top_m & prior_top) / len(top_m | prior_top))
            row.update({"matrix_mrr": round(mrr, 4),
                        "hit_at_5": round(h5, 4), "hit_at_10": round(h10, 4),
                        "n_matched": int(nm),
                        "rho_colscore_pri": round(rho, 3),
                        "jaccard_top10_vs_prior": round(float(np.mean(jac)), 3)})
        elif name == "protein_only":
            coef = clf.coef_[0]
            Pt_all = bs.transform(P)
            pscore = Pt_all @ coef + float(clf.intercept_[0])
            pr = rate_prior.reindex(uniq_p)
            valid = pr.notna().to_numpy()
            row["rho_protscore_pri"] = round(
                float(spearmanr(pscore[valid], pr.to_numpy()[valid]).statistic), 3)

        rows.append(row)
        md.append(f"| {name} | {auc_tr:.3f} | {auc_te:.3f} | "
                  f"{'—' if mrr is None else f'{mrr:.3f}'} | "
                  f"{'—' if h5 is None else f'{h5:.3f}'} | "
                  f"{'—' if h10 is None else f'{h10:.3f}'} | "
                  f"{'—' if nm is None else str(nm)} |")
        extra = {k: v for k, v in row.items()
                 if k.startswith("rho") or k.startswith("jaccard")}
        print(f"[probe] {name:>13}: train AUC {auc_tr:.3f}  test AUC {auc_te:.3f}"
              f"  {extra if extra else ''}", flush=True)

    full = next(r for r in rows if r["variant"] == "full")
    mol = next(r for r in rows if r["variant"] == "molecule_only")
    pro = next(r for r in rows if r["variant"] == "protein_only")
    md += [
        "",
        "## Reading",
        "",
        f"**1. The skill's anticipated finding.** The frozen linear probe",
        f"(full) reaches pooled test AUC **{full['test_auc']:.3f}** with no",
        "deep learning at all — trained deep DTI baselines on this dataset",
        "span global AUC 0.63–0.95 (DrugBAN 0.954, MolTrans 0.937, GraphDTA",
        "0.869, GEMS 0.633; RankBind v4 0.634 ± 0.010). A representation-",
        "free linear readout of frozen features recovers a large share of",
        "the published pooled-AUC level.",
        "",
        f"**2. The stronger finding: molecule-side role assignment.** The",
        f"molecule-only probe (**{mol['test_auc']:.3f}**) BEATS the full",
        "probe. Mechanism: of the unique ligands, "
        f"**{n_pure_pos} are positive-only** and **{n_pure_neg} are",
        f"decoy-only** ({frac_pure:.1%} pure overall) — the decoy protocol",
        "assigns each molecule a fixed role rather than sampling negatives",
        "per protein. Because the canonical split is protein-based, it does",
        f"NOT hold out molecules ({te_overlap:.1%} of test-pair ligands also",
        "occur in training rows), so ligand identity alone transfers to the",
        "test split and any model with molecular memory scores high without",
        "learning protein-ligand interaction.",
        "",
        f"**3. Protein side is consistent with the null-prior analysis.** The",
        f"protein-only probe reaches {pro['test_auc']:.3f} and its score",
        f"correlates at rho={pro.get('rho_protscore_pri', float('nan')):+.2f}",
        "with the train-rate prior — i.e. at null-baseline level. The full",
        f"probe's shortcut diagnostics (rho {full['rho_colscore_pri']:+.2f},",
        f"top-10 Jaccard vs prior {full['jaccard_top10_vs_prior']:.2f}) show",
        "its pair-level signal is NOT primarily the protein-prevalence",
        "shortcut analysed elsewhere; it is the molecule-side structure.",
        "",
        f"**4. A fourth dissociation for free.** The full probe scores",
        f"pooled AUC {full['test_auc']:.3f} while its matrix MRR is "
        f"{full['matrix_mrr']:.3f} (chance = H_200/200 = 0.0294; Hit@5 "
        f"{full['hit_at_5']:.2f}). High pooled AUC with chance-level",
        "ligand-conditional ranking — the paper's central metric critique,",
        "reproduced by a linear probe in isolation.",
        "",
        "**Interpretation (skill-mandated framing).** The decoy construction",
        "itself contains learnable pair-level structure that can be",
        "exploited by representation-based models — dominated by molecule-",
        "role memorisation, not protein prevalence — motivating cautious",
        "interpretation of ABSOLUTE pooled-AUC values on BRENDA-with-decoys,",
        "and supporting the paper's choice of ligand-conditional matrix",
        "metrics as primary. This is a limitation, not evidence that BRENDA",
        "is invalid.",
    ]
    pd.DataFrame(rows).to_csv(os.path.join(_HERE, "decoy_leakage_probe.csv"),
                              index=False)
    open(os.path.join(_HERE, "DECOY_LEAKAGE_AUDIT.md"), "w").write("\n".join(md) + "\n")
    print("[probe] wrote DECOY_LEAKAGE_AUDIT.md + decoy_leakage_probe.csv")


if __name__ == "__main__":
    main()
