# Poster figure data: raw CSV values

Raw values behind the two paper figures, for re-plotting on a poster.
Regenerate with: `python -m evaluation.export_poster_figure_data`

## figure1_summary/  (paper Figure 1 = `fig_summary.png`)

Combined Phase-1 + Phase-2 dashboard. RankBind is the only model outside the
baseline/null cluster on every panel.

| File | Figure panel | Contents |
|------|--------------|----------|
| `response_map_RankBind.csv` | top-left | 200×200 score matrix (rows = ligand_idx, cols = protein_0…199). Gini 0.787 |
| `response_map_GraphDTA.csv` | top-middle | same, strongest baseline. Gini 0.995 |
| `response_map_Null_prot_prior.csv` | top-right | same, data-blind protein-prior null. Gini 0.995 |
| `jaccard_top10_attractors.csv` | bottom-left | 7×7 Jaccard of each model's top-10 attractor proteins |
| `auc_scatter.csv` | bottom-middle | per model: `global_auc`, `per_ligand_auc`, `gini` |
| `gini_bars.csv` | bottom-right | Gini(attractor) per model + null flag |
| `figure1_rendered.png` | — | the published figure, for reference |

## figure10_attn_explainer/  (paper Figure 10 = `fig_attn_explainer.png`)

Intuitive view of what the v5b residue attention encodes (≈ a hydrophobicity
read-out; active sites strongly avoided). 60 sampled BRENDA proteins, cross-seed
mean attention.

| File | Figure panel | Contents |
|------|--------------|----------|
| `residues_long.csv` | source for all panels | per-residue master table: `attn`, `attn_z` (within-protein z), `pctile` (within-protein attention percentile), `hydropathy`, `aa`, `aa_class`, `is_active`, `is_binding`, `is_signal` |
| `panel1_class_boxplot_stats.csv` | (1) | box-plot stats of `attn_z` per residue class (acidic to aromatic) |
| `panel2_functional_residue_percentiles.csv` | (2) | box-plot stats of attention percentile for all / binding-site / active-site residues |
| `aa_attention_bias.csv` | (a, related fig) | mean within-protein z(attn) per amino acid |
| `example_track_E2RV69.csv` | bottom track | per-residue attention vs hydropathy along the sequence |
| `example_track_Q8ZRM2.csv` | bottom track | same, second example protein |
| `figure10_rendered.png` | — | the published figure, for reference |

Box-plot stat columns: `n, mean, median, q1, q3, whisker_lo, whisker_hi, min,
max` (whiskers = 1.5×IQR, matching matplotlib defaults).

### figure10_attn_explainer/annotations/  (Quell-Annotationen)

Die UniProt-Annotationen, aus denen `is_active` / `is_binding` / `is_signal`
und Panel (c) abgeleitet wurden; Grundlage sind 60 gesampelte BRENDA-Proteine.

| File | Contents |
|------|----------|
| `functional_residue_annotations.csv` | long: jedes annotierte funktionelle Residuum (`uniprot, position, amino_acid, annotation`): 875 Residuen (478 Signal peptide, 324 Binding site, 73 Active site) über 54 Proteine |
| `feature_region_enrichment.csv` | pro (Protein, Feature-Typ): wie stark die Attention auf der Region angereichert ist (`auc`; 0.5 = Zufall). Quelle für Panel (c) |
| `uniprot_raw_json/<UNIPROT>.json` | rohe UniProt-Records (Sequenz + alle Feature-Regionen), wie von der REST-API geholt und gecacht, die Quelle der Wahrheit |
