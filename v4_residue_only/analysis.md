# ResidueOnlyBind v4 — Analyse & Verbesserungsvorschläge

## Aktuelle Performance

| Metrik | Wert |
|--------|------|
| ROC-AUC | 0.929 (best 0.934 @ Ep.75) |
| Avg Precision | 0.825 |
| Disc. Accuracy | 0.935 |
| Affinity Pearson r | 0.796 |
| Affinity RMSE | 1.34 log₁₀ |

## Schwächen

1. **Deutliches Overfitting**: Train-Loss ~0.27 vs Val-Loss ~0.74 (Faktor 2.7x). Klassifikationsmetriken stagnieren ab Epoch ~25.
2. **Schlechte Kalibrierung**: Reliability Diagram zeigt overconfident predictions — predicted probabilities passen nicht zu tatsächlichen Frequenzen.
3. **Affinity-Regression hat viel Scatter**: RMSE 1.34 log₁₀ bedeutet im Schnitt >10-facher Fehler in kcat/Km. OLS-Linie weicht von y=x ab (systematischer Bias).
4. **Sehr kleiner Datensatz**: ~240 Proteine (nur Hydrolasen) — limitiert alles.

---

## Konkrete Verbesserungen

### A. Overfitting bekämpfen (größter Hebel)
- **Mehr Dropout** (aktuell 0.1 → 0.2-0.3) oder **DropEdge** in den GNN-Layern
- **Weight Decay** hinzufügen (1e-4 bis 1e-3) — fehlt aktuell komplett
- **Label Smoothing** für die BCE (0.05-0.1) — verbessert auch die Kalibrierung
- **Stochastic Depth** / Layer-Drop in den 4-Layer Transformern
- **Datensatz vergrößern**: Über Hydrolasen hinaus expandieren (Transferasen, Oxidoreduktasen) oder aggressiveres Augmentation (Subgraph-Sampling, Feature-Noise)

### B. Kalibrierung verbessern
- **Temperature Scaling** post-hoc (ein einzelner Parameter, auf Val-Set optimiert)
- **Focal Loss** statt BCE — bestraft overconfident predictions
- **Mixup** auf Graph-Ebene (Feature-Interpolation zwischen Samples)

### C. Architektur-Upgrades
- **Cross-Attention statt Bilinear Fusion**: Das elementweise `prot_sum * lig_sum` verliert alle Positionsinformation. Eine Cross-Attention-Schicht (Residue-Embeddings × Ligand-Atom-Embeddings) vor dem Pooling wäre deutlich expressiver — Mittelweg zwischen jetzigem Modell und vollem HierAtomBind
- **Gated Fusion**: `σ(W·[prot||lig]) ⊙ interaction` statt einfacher Konkatenation
- **Virtual Node** im Ligand-GNN: Globale Message-Passing-Information, hilft bei fragmentierten Molekülgraphen

### D. Affinity-Regression stärken
- **Separate Encoder-Köpfe** mit Stop-Gradient: Affinity-Head teilt sich aktuell den gleichen `fused`-Vektor — konkurriert mit Klassifikation
- **Evidential Regression** (Deep Evidential Learning): Gibt Unsicherheitsschätzung mit aus
- **Paarweise Ranking-Loss** (z.B. MarginRankingLoss auf Affinity-Paare desselben Proteins): Robuster als punktweiser L1

---

## Zusätzliche Erkenntnisse aus dem bestehenden Modell

### 1. Protein-Ligand Interaction Fingerprints
Die Attention-Weights aus `res_pool` und `lig_pool` zeigen, welche Residues und Ligand-Atome das Modell als wichtig erachtet:
- Attention-Weights pro Residue extrahieren → auf 3D-Struktur mappen → **predicted binding site** ohne expliziten Site-Selector
- Vergleichen mit bekannten Active Sites aus UniProt/PDB → wie gut findet das Modell die Bindestelle implizit?

### 2. Contrastive Embedding Space analysieren
Die `z_prot` und `z_lig` Projektionen (64-dim) sind bereits trainiert:
- **t-SNE/UMAP** der Protein-Embeddings einfärben nach EC-Subklasse → Wie gut clustern die Hydrolasen?
- **Ligand-Embeddings** clustern → Entdeckst du chemische Familien, die das Modell implizit gelernt hat?
- **Cross-Modal Retrieval**: Für ein neues Protein, welche Liganden haben die höchste Cosine-Similarity im z-Space? → Virtuelles Screening ohne Forward-Pass durch den Fusion-Layer

### 3. Selektivitäts-Profiling
- **Selectivity Matrix**: Alle Protein×Ligand Kombinationen durchrechnen → Welche Liganden sind promiskuitiv? Welche Proteine sind leicht zu targeten?
- **Differential Binding**: Bei Protein-Mutanten (gleiche Sequenz, einzelne Residues verändert) → Wie ändert sich der Score? → **In-silico Mutagenese**

### 4. Unsicherheitsquantifizierung
- **MC-Dropout** (Dropout zur Inferenzzeit an, 20 Forward-Passes) → Varianz = Epistemic Uncertainty
- Predictions mit hoher Unsicherheit = Kandidaten für experimentelle Validierung → **Active Learning Loop**

### 5. Feature Attribution
- **Integrated Gradients** oder **GNNExplainer** auf Input-Features → Welche der 33 Residue-Features und 25 Ligand-Features treiben die Prediction?
- Ergibt das biologisch Sinn? (z.B. sind hydrophobe Features in Binding Pockets dominant?)

---

## Was würde es "besonders" machen?

1. **Interpretierbare Binding-Site-Prediction ohne Supervision** — zeigen, dass Attention-Weights des Residue-Pools bekannte Active Sites recovern
2. **Cross-modal Retrieval** im contrastive Space — Protein rein, passende Liganden raus, ohne klassisches Docking
3. **Uncertainty-guided Virtual Screening** — MC-Dropout-Unsicherheit korreliert mit tatsächlicher Prediction-Qualität
