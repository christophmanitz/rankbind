# Zenodo release manifest

Was für einen Zenodo-Record zum Sci-Rep-Manuskript (`paper/scirep/main.tex`)
hochgeladen werden muss, was bereits auf dem System liegt, und was zuerst
regeneriert werden muss. Stand: **2026-08-23** (Repo-Commit `614ee88`;
Papier-Revision, Poster-Sync und Bootstrap-Korrektur committed; Staging
bedarf der Aktualisierung, Upload ausstehend), geprüft gegen das lokale
System.

**Workspace:** `/work2/zw93onug-rankbind_zenodo` (`ws_allocate rankbind_zenodo 30`,
6 Extensions übrig). Staging: `.../zenodo_staging/rankbind-paper-v1/` und
`.../zenodo_staging/rankbind-benchmark-embeddings-v1/` (Stand 2026-08-21,
**vor** den Revisionen unten — vor dem Upload neu aufbauen).

## Ziel

Ein frischer Klon + Zenodo-Download + eine GPU sollen alle Tabellen und
Figuren des Papers regenerieren können (nicht bit-für-bit, aber innerhalb
der berichteten Standardabweichungen, s. §5).

---

## 1. Sofort hochladbar (liegt auf dem System)

| Artefakt | Pfad (lokal) | Größe | Bemerkung |
|---|---|---|---|
| Repo (Code, Configs, CSVs, Figuren) | `~/rankbind` @ `272ef66` | ~11 MB | inkl. Submodul `reactionDataFiltering` @ `4d422d52`; Revision 2026-08-23 (s. §6) |
| BRENDA-200 Paare | `data/dataset_with_decoys.csv` | 967 KB | 9.632 Paare, 3.175 Positiv / 6.457 Decoys |
| BRENDA-200 Sequenzen | `data/sequences/sequences.csv` | 296 KB | |
| BRENDA+SABIO Roh-Snapshot | `reactionDataFiltering/data/raw/brenda_sabio_2026-04-29/` | 923 MB | CSVs (kcat_km 3,2 MB, km 7,9 MB, turnover 4,5 MB) + AlphaFold-Strukturen (907 MB) |
| BRENDA+SABIO interim | `reactionDataFiltering/data/interim/{kcat_km,km,turnover}_brenda_sabio/` | ~5 MB | `with_decoys.csv`, `sequences.csv`, `decoys.csv` + je Stufe `*.manifest.json` (SHA-256, Parameter, Versionen) |
| Phase-1-Checkpoints | `results/original_{graphdta,moltrans,drugban,gems}/` | 255 MB | `best_model.pt` + `score_matrix_*.npy` |
| RankBind-Runs | `results/v5_rankbind/` | 238 MB | 82 `manifest.json`, 77 `best_model.pt`, 75 Score-Matrizen, 75 `test_summary.json`; inkl. Cold-Split-Runs (`v7_cold_*`) |
| Tabellenwerte | `evaluation/attractor_results/*.csv` | committed | Quelle aller Tabellen im Paper; inkl. `cold_split_multiseed.csv`, `cold_split_runs.csv`, `null_baseline_firstclass_{ligand,double_cold}.csv` (Paper Tab. 3) |

## 2. Muss regeneriert werden (fehlt auf dem System)

### a) ESM2-Embeddings (21 GB, 9.912 Dateien) — ✅ regeneriert 2026-08-21

Der gemeinsame Store `reactionDataFiltering/data/interim/esm2_embeddings_shared/`
existiert nicht mehr; alle ~9.500 Symlinks in den Dataset-Ordnern zeigen ins
Leere. Ohne ihn ist kein Training (RankBind oder Baselines) möglich.

**Erledigt:** alle 7 SLURM-Jobs COMPLETED (~25 min auf A30). brenda200: 882,
kcat_km: 3.815, km: 9.500, turnover: 5.672 (= alte Counts), danach
`dedup_embeddings.py --apply`: 9.912 unique im Shared Store, 18.987 Symlinks,
0 SHA-Mismatches. Stichprobe: Tensoren valide [L,1280] float32.
Pfade jetzt: `$WS/embeddings_esm2/{esm2_embeddings_shared,brenda200,<ds>}`;
Repo-Symlinks zeigen dorthin.

Regeneration (deterministisch, frozen `facebook/esm2_t33_650M_UR50D`,
batch_size 1, max_residues 1024):

```bash
cd reactionDataFiltering
bash hpc/setup_venv.sh          # erzeugt ~/venvs/esm2 (CUDA torch + transformers)
sbatch hpc/run_embeddings.sh    # SLURM-Array 0-2 (kcat_km, km, turnover), A30, bis 2 d
```

- Resumierbar: existierende `{uniprot}.pt` werden übersprungen.
- Verifikation: die `esm2_embeddings.manifest.json` je Dataset existieren
  noch und pinnten Eingabe-Sequenzen + Parameter.
- Danach die Symlink-Struktur wiederherstellen
  (`scripts/dedup_embeddings.py` im Haupt-Repo, idempotent).
- Zusätzlich nötig: Embeddings für die Benchmark-Sequenzen (Davis/KIBA/
  BindingDB/ESP, s. b) — dort wurden sie direkt ins jeweilige
  `benchmarks/<ds>/esm2_embeddings/` geschrieben.

### b) Benchmark-Rohdaten Davis / KIBA / BindingDB_Kd / ESP — ✅ neu geladen 2026-08-21

`reactionDataFiltering/data/interim/benchmarks/` existierte nicht mehr.

**Erledigt:** via `scripts/prep_benchmark_datasets.py` (TDC) bzw.
hieratombind-venv (ESP, github.com/AlexanderKroll/ESP) neu geladen:
davis 25.772 Paare/379 Proteine, kiba 117.657/229, bindingdb_kd 46.117/1.413,
esp 28.900/11.436 — je mit `pairs.csv`, `sequences.csv`, `prep_card.json`.
Neue SHA-256 weichen wie erwartet von den historischen Manifests ab
(bit-für-bit-Kette für diese vier Datensätze nicht mehr herstellbar;
Tabellenwerte unberührt, s. committed CSVs). Benchmark-Embeddings ebenfalls
regeneriert (379/229/1.413/11.436 .pt).

## 3. Tatsächliche Record-Struktur (2026-08-21, zwei Records wegen 50-GB-Limit)

**Record 1 `rankbind-paper-v1`** (~28 GB): README.md, .zenodo.json,
Manuskript-PDF, `repo/rankbind-paper-snapshot.tar.gz`, `data/` als Archive
(brenda200.tar.gz, brenda_sabio_raw_2026-04-29.tar [2,9 GB],
brenda_sabio_interim.tar.gz, benchmarks_csvs.tar.gz), `embeddings/`
(brenda200_esm2.tar, brenda_sabio_esm2.tar [Shared Store + Symlinks]),
`results/` (phase1.tar, v5_rankbind.tar, attractor_results_csvs.tar.gz).

**Record 2 `rankbind-benchmark-embeddings-v1`** (~33 GB): README.md,
.zenodo.json, `embeddings/{davis,kiba,bindingdb_kd,esp}_esm2.tar`.

Im Repo liegen die Pfade dann als Symlinks auf den Zenodo-Download
(oder umgekehrt: Zenodo-Struktur spiegelt die Repo-Pfade).

## 4. Nächste Schritte (Checkliste)

1. [x] Backup-Suche — kein Backup gefunden (2026-08-21)
2. [x] ESM2-Embeddings regeneriert + Counts/Stichproben verifiziert (2026-08-21)
3. [x] Davis/KIBA/BindingDB/ESP neu geladen, prep_card.json dokumentiert (2026-08-21)
4. [ ] `REPRODUCIBILITY.md` um Zenodo-DOI ergänzen (Abschnitt „Artefacts on Zenodo")
5. [ ] Paper `Data availability` auf Zenodo-DOI zeigen (statt „on request")
6. [ ] Smoke-Test aus frischem Klon: alle §3-Kommandos der REPRODUCIBILITY.md durchlaufen
7. [ ] Zenodo-Records anlegen (Upload + DOI ausstehend; **Staging zuerst auf `614ee88` neu aufbauen** — der 2026-08-21-Stand enthält die Revisionen aus §6 nicht)

## 5. Grenzen — was „reproduzierbar" hier bedeutet

- **Kein bit-für-bit**: MolTransformer-Decoys haben interne Nichtdeterminismen
  (Submodul-README), bf16-GPU-Rundung variiert leicht zwischen GPU-Typen,
  TDC/BRENDA-Quellen ändern sich über die Zeit.
- **Realistische Erwartung**: 3-Seed-Mittel reproduzieren innerhalb der
  berichteten Standardabweichungen; Monotonie-Aussagen (7–25× vs. Kontrolle)
  sind robust.
- **Was Zenodo nicht bekommt**: venvs (reicht: `requirements.txt` +
  `setup_venv.sh`), MolTransformer (extern, nicht auf PyPI — Installations-
  Anleitung in den Submodul-README), der 650-MB-Tarball des Snapshots
  (nicht auffindbar; das entpackte `raw/` wird hochgeladen).

---

## 6. Revision 2026-08-23 (Commit `272ef66`)

Seit dem Staging-Stand (2026-08-21, `4ff68dc3`) ist folgendes im Repo
gelandet und muss im Snapshot stecken (Staging **neu gebaut 2026-08-23**):

| Was | Commits | Betrifft Paper |
|---|---|---|
| Manuskript-Revision: neuer Titel „When Pooled AUC Lies…", gescopte Claims („not specific to BRENDA-200"), korrigierte Bootstrap-CIs im Intro, explizite Cold-Split-Logik | `d11a521` | Abstract/Intro/§2.4/§2.5/Discussion |
| Paired-Bootstrap-Fix: `paired_molecule_stats.py` baut den Split jetzt aus `data.split_seed=42` statt aus dem Trainings-Seed; CIs +0.23/+0.20/+0.17 (alle exkludieren 0) | `d11a521` | Intro |
| Cold-Split-Code und -Daten: `get_ligand_split`/`get_double_cold_split` (common.py), `split_mode=ligand/double_cold` (data.py), Konfigurationen `v5_rankbind/configs/cold_*.json`, `scripts/aggregate_cold_splits.py`, `cold_split_multiseed.csv`, `null_baseline_firstclass_{ligand,double_cold}.csv` | `a31dd40`, `614ee88` | Tab. 3 + §4.1 |
| Poster-Sync (`paper/poster_scads/`): alle Zahlen auf Pinned-Split-Stand, Titel/Subtitle wie Paper | `d11a521` | Poster |
| Review-Artefakte: `CLAIM_EVIDENCE_MATRIX.md`, `REVIEW_TRIAGE.md`, `skill_jcim_revision.md` aktualisiert/ergänzt | `d11a521`, `614ee88` | — |

Der 2026-08-21-Stand des Stagings reproduziert diese Zahlen **nicht**
(v.a. Tab. 3 und die Intro-CIs); vor dem Upload neu stagen.