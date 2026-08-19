# Bildunterschrift — fig_v5_model_diagram

**Architektur von RankBind (v5).**
Der Ligand kommt als SMILES-String herein, das Protein als Aminosäuresequenz.
Beide werden von je einem vortrainierten Sprachmodell codiert, ChemBERTa für das
Molekül und ESM2 für das Protein. Diese Modelle sind eingefroren und liefern nur
feste Embeddings: einen Vektor pro Token bzw. pro Residuum, 384- bzw.
1280-dimensional. Mittelwert-Pooling fasst sie zu je einem Vektor für das ganze
Molekül und das ganze Protein zusammen.

Jeder Vektor läuft danach durch einen Projektor, ein kleines zweischichtiges Netz
(im Bild als Fan aus Verbindungslinien gezeichnet). Es verdichtet das Embedding
nichtlinear auf 256 Dimensionen und legt Ligand und Protein in denselben Raum,
f(L) und g(P). Das sind die einzigen trainierbaren Teile auf der Encoder-Seite;
an den Sprachmodellen selbst wird nichts verändert.

Im Bilinear-Head treffen f(L) und g(P) zusammen. Über eine gelernte
Gewichtsmatrix M (niedrigrangig plus Diagonale) entsteht der Score
s(L,P) = f(L)ᵀ M g(P) + b. b ist dabei nur ein einzelner konstanter Bias; einen
rein protein-abhängigen additiven Term gibt es nicht, deshalb kann der Head nicht
über einen Protein-Prior abkürzen.

Trainiert wird mit einem Margin-Ziel (Kasten unten): ausgeglichenes Ziehen von
Positiven und Negativen pro Protein (#1), ein Within-Ligand-Margin-Loss (#2) und
Hard-Negative-Mining mit den aktuell ähnlichsten Nicht-Bindern (#4). Der Gradient
aktualisiert nur die Projektoren f, g und die Matrix M.
