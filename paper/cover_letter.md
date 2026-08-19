# Cover letter: RankBind submission

> Draft. Fill the bracketed `[...]` fields at submission time. Tone: short,
> claim-first, pre-empts the one question every reviewer of a single-corpus
> method paper asks.

---

[Date]

To the Editors,
[Journal name]

Dear Editors,

We submit our manuscript, "RankBind: Protein-Invariant Contrastive Learning
for Ligand-Conditional Drug-Target Interaction," for consideration as a
[research article] in [Journal name].

**What the paper shows.** Drug-target interaction (DTI) models on
enzyme-substrate benchmarks are routinely validated with *pooled* metrics such
as global AUC. We show this certifies the wrong property. On the BRENDA
enzyme-substrate corpus, four published baselines (DrugBAN, MolTrans, GraphDTA,
GEMS) reach pooled AUC up to 0.95 while their *per-ligand* ranking is at or
below chance, and their score distribution is statistically indistinguishable
from a data-blind protein-prior baseline (Gini ≈ 0.995 in every case). Pooled
AUC, in this regime, rewards a protein-level shortcut, not ligand-conditional
binding. We then introduce RankBind, a 627k-parameter architecture whose four
ingredients (protein-balanced sampling, a within-ligand margin loss, a
low-rank bilinear interaction head, and online hard-negative mining) break the
shortcut: matrix MRR rises from ≈0.06 (a matched shortcut-prone control) to
0.326 ± 0.072, and top-10 overlap with the data-blind prior falls to 0.000. We
distil the diagnosis and the fix into an eight-step recipe any practitioner can
apply to a new model or dataset.

**Why it matters to your readership.** The contribution is methodological and
portable: a *diagnostic* (the null-prior probe) plus an *architecture* that
passes it. Both are released as runnable code, so any future enzyme-substrate
model can be audited the same way. The shortcut we characterise is a concrete,
measurable instance of the broader shortcut-learning problem now recognised
across machine learning.

**Pre-empting the obvious question: why a single primary corpus?** We
deliberately anchor the headline on one well-controlled enzyme-substrate corpus
(BRENDA-200) because the claim is about *evaluation methodology*, which requires
a clean, fully-audited split rather than breadth. We then provide three
independent generalisation probes precisely to address the breadth concern:
(i) a 50× scale-up to three enzyme-wide BRENDA+SABIO datasets (43-57k pairs),
showing the anti-shortcut property survives the shift; (ii) a cross-dataset
probe on three kinase affinity benchmarks (Davis, KIBA, BindingDB), showing the
enabling prior (Gini ≈ 0.995) is present everywhere and the recipe carries over
to BindingDB; and (iii) a matched shortcut-prone BCE control scored on all seven
corpora, which reproduces the pooled-AUC-vs-ranking dissociation universally and
which RankBind circumvents on every catalytic enzyme dataset (7-25× the
control's ranking). We report the non-wins as they are: affinity corpora
carry no prior ranking signal to circumvent, and two datasets exposed tuning,
not refutation. The enzyme-substrate focus, backed by the enzyme-wide
transferability probe, is the scope we defend.

**Originality and ethics.** This manuscript is original, is not under
consideration elsewhere, and has not been published previously. The author
declares no competing interests. All data sources are public and used under
their respective licences; no human or animal subjects are involved.

We believe this work is a good fit for [Journal name] because [one sentence:
e.g. its focus on rigorous evaluation methodology for computational
drug-target modelling]. We suggest the following potential reviewers with
relevant expertise: [Name, affiliation]; [Name, affiliation]. We request that
[any conflicted reviewers] be excluded.

Thank you for considering our submission.

Sincerely,
Christoph Manitz
Leipzig University
christoph.manitz@uni-leipzig.de
[ORCID]
