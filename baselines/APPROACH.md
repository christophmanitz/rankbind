# Baseline-Modelle: Vorgehensweise

## Zwei Ansätze

### Ansatz 1: Re-Implementierungen (aktuell, Smoke-Test)

**Status:** ✅ Alle 7 Modelle abgeschlossen (Jobs 21250020–21250027).

Eigenständige Nachimplementierungen der Paper-Architekturen in `baselines/{model}/model.py`:
- `deepdta/` — CNN×2 auf Sequenzen (Öztürk et al., 2018)
- `graphdta/` — GCN auf Mol-Graph + CNN auf Protein-Seq (Nguyen et al., 2020)
- `drugban/` — GCN + CNN + Bilinear Attention Network (Bai et al., 2023)
- `moltrans/` — Transformer + Element-wise Product (Huang et al., 2021)
- `gign/` — Simplified GIGN mit virtuellen Cross-Edges (Gao et al., 2023)
- `gems/` — BiLSTM + GCN als Proxy für ESM2+GCN (Brocidiacono et al., 2024)

**Limitierungen dieses Ansatzes:**
- Nicht publikationstauglich — Reviewer können Implementierungsfehler beanstanden
- Vereinfachungen (z.B. GIGN ohne 3D, GEMS ohne ESM2)
- Keine verifizierbare Architektur-Äquivalenz zum Original

**Nutzen:**
- Validiert, dass Daten-Pipeline, SLURM-Setup, Score-Matrix-Generierung funktionieren
- Gibt erste Indikation ob Attractor-Bias beobachtbar ist
- Schnell implementiert (kein Dependency-Aufwand)

---

### Ansatz 2: Original-Repos (Paper-Version, in Entwicklung)

Klont die offiziellen GitHub-Repos der Autoren, schreibt Datenadapter, nutzt deren Modellklassen.

**Repos:**
| Modell | Repository | Lizenz |
|--------|-----------|--------|
| DrugBAN | peizhenbai/DrugBAN | MIT |
| GraphDTA + DeepDTA | thinng/GraphDTA | — |
| MolTrans | kexinhuang12345/MolTrans | MIT |
| GIGN | guaguabujianle/GIGN | MIT |
| GEMS | camlab-ethz/GEMS | MIT |

**Vorteile:**
- Wissenschaftlich belastbar: "We use the official implementation of [X]"
- Exakte Architektur wie im Original-Paper
- Reproduzierbar durch Dritte (gleicher Code, gleiche Daten)

**Adapter-Strategie:**
- `baselines/adapters/common.py` — einheitliche Splits + Daten-Config
- `baselines/adapters/adapter_{model}.py` — konvertiert BRENDA-Daten in Modell-Format
- `baselines/train_original.py` — einheitlicher Trainer (gleiche Epochs, Optimizer, Eval)

**3D-Strukturen für GIGN:**
- SKiD-Dataset (Zenodo doi.org/10.5281/zenodo.15355030): vorgedockte BRENDA-Komplexe
- DiffDock für Lücken-Füllung

**ESM2 für GEMS:**
- Vorberechnete Embeddings für alle 903 Proteine

---

## Warum einheitliches Training?

Obwohl wir die Original-Modellarchitekturen nutzen, trainieren wir alle Modelle in **unserem** Loop:

1. **Gleicher Train/Val-Split** — Protein-basiert, kein Leakage
2. **Gleiches Epochs-Budget** — 100 Epochs, Early Stopping patience=10
3. **Gleicher Optimizer** — Adam, lr=1e-4
4. **Zwei Task-Varianten** — Regression (predicted affinity) + Binary (binder vs decoy)
5. **Einheitliche Evaluation** — attractor_diagnosis.py auf Score-Matrizen

Nur so ist ein fairer Vergleich möglich. Jedes Repo hat andere Defaults — direkte Ausführung ihrer Train-Scripts wäre unfair.

---

## Evaluation-Pipeline

```
Trainiertes Modell
    → Score-Matrix [N_lig × N_prot]
    → attractor_diagnosis.py
    → Metrics: Gini(attractor), Hit@k, Rank Displacement, Score Variance
    → Plots: Response Map, Attractor Distribution
    → Comparison Table (alle Modelle)
```

---

## Timeline

1. ✅ Re-Implementierungen geschrieben + SLURM-Jobs submitted (Smoke-Test)
2. ✅ Alle 7 Smoke-Tests erfolgreich — Attractor Bias universell bestätigt (Gini ≥ 0.83)
3. ✅ Original-Repos geklont + Import-Tests bestanden (GraphDTA, MolTrans, GIGN, GEMS)
4. ✅ Adapter geschrieben + getestet (alle 5 Modelle)
5. ✅ `train_original.py` mit vollständigen Training-Loops, Score-Matrix, Diagnose
6. ⏳ DGL-Venv für DrugBAN einrichten (`scripts/setup_drugban_venv.sh`)
7. ⏳ DiffDock für GIGN 3D-Strukturen
8. ⏳ ESM2-Embeddings für GEMS (`scripts/run_esm2_embed.sh`)
9. ⬜ Einheitliches Training aller Orig-Modelle (`scripts/run_original_baselines.sh`)
10. ⬜ Attractor-Vergleichstabelle + Paper-Figuren
