# Peer Review — *Higher-Order Protein Feature Co-occurrence at AlphaFold Scale*

**Manuscript:** `papers/et_miner_proteome.tex` (E. Ahmic, C. claudya)
**Review date:** 2026-07-12
**Type:** Methods / applied-computation preprint (single-machine GPU frequent-itemset mining applied to the AlphaFold/UniProt proteome)
**Basis:** Full read of the restored base-run manuscript. Numerical and citation integrity were independently pre-verified against the preserved mining artefacts (godmode/direct/beyond/extreme/pipeline logs, the direct-vs-SON and null-model JSONs, and the `item_mapping` blob); this review therefore concentrates on methodology, statistics, reproducibility, novelty, and claim calibration.

---

## Summary Statement

The manuscript presents **ET-miner**, a GPU-accelerated Apriori engine that mines the complete frequent-itemset landscape of protein feature annotations across 76.9 million multi-feature proteins (1,002 features) on a single NVIDIA H100 in 7.3 minutes, discovering 26.8 million co-occurrence patterns up to K = 22. The central contribution is an engineering one: a compressed-sparse-row → on-GPU-bitvector representation that keeps the transaction data GPU-resident across all K-levels, moving only lightweight metadata over PCIe, which lets exhaustive Apriori run at a transaction count roughly three orders of magnitude larger than any previously reported single-machine GPU FIM system[SOURCE?]. A permutation null model (at the 0.001% threshold) and a set of recovered known biological modules (drug targets, spliceosome, RTK signalling) are offered as validation.

**Recommendation: Major revision.** The computational contribution is genuine, clearly described, and the reported numbers are internally consistent and artefact-backed. However, the manuscript's *scientific* claims outrun its *statistical* and *reproducibility* support in three specific ways (Major 1–3 below), and the flagship biological result (K = 22) rests on the thinnest possible evidence base. None of these is fatal; all are addressable without new large-scale compute.

**Key strengths**
- A real, well-motivated systems contribution (GPU-resident bitvector Apriori) with an honest complexity story and a clean memory-reduction narrative.
- Exhaustive rather than approximate mining, with a convincing demonstration (21× / 95.2% miss-rate) of why the SON approximation is inadequate at this scale.
- Recovers independently-known biology (β-lactam PBP targets, Src-family RTK architecture, spliceosome) from annotation data alone, a good sanity signal.

**Key weaknesses**
- The null model validates only the 0.001% threshold; the headline K = 22 result lives at 0.00001%, where **no** null model was run (Major 1).
- Statistical support rests on **n = 5** permutations; the reported "Z" magnitudes and the p < 0.45 bound cannot carry the weight the narrative places on them (Major 2).
- The flagship K = 22 "neuronal antiviral" pattern is a single itemset in 8 proteins whose accessions are not listed and whose GO evidence codes are unverifiable; reproducibility of this specific result is currently incomplete (Major 3).

---

## Major Comments

### Major 1 — The null model does not cover the threshold where the headline result lives
The permutation null model (Section 3.6, Table 4) is run only at the Power threshold (0.001% support, min_count = 769). The paper's most prominent findings — the K = 9 distribution peak and the K = 22 ceiling — are obtained at the Opus threshold (0.00001%, min_count = 8). The text is now commendably explicit that "a separate null model was not run" at the Opus threshold and that a rigorous claim there "would require its own permutation analysis." That candour is the right move, but it also exposes a structural gap: **the deepest patterns, which carry the paper's biological narrative, have no statistical validation at all.**

*Why it matters:* a reviewer will reasonably ask whether the K = 15–22 patterns are enriched over a count-preserving null, or whether they are an expected consequence of a handful of very densely-annotated proteins at a permissive threshold. The former is a discovery; the latter is an annotation artefact.

*Suggested remedy (no new large compute):* run the existing permutation harness at the Opus threshold — even 5 permutations at min_count = 8 would establish whether the null produces *any* deep (K ≥ 7) patterns there. If compute is prohibitive, at minimum quantify how concentrated the deep patterns are (e.g., how many distinct proteins contribute to all K ≥ 15 itemsets) so the reader can judge annotation-density confounding directly.

### Major 2 — Statistical inference from n = 5 permutations is over-leveraged
The revised text now correctly labels the enrichment statistics as t-statistics (4 df) rather than large-sample Z-scores, and corrects the absence bound to p < 0.45 (rule of three, 0/5). These are the right fixes. But the underlying issue remains: **five permutations is too few to support the inferential language still used around them.**

Specific concerns:
- A p < 0.45 bound does not reject the null at any conventional level. The sentence structure ("$K{\geq}7$ patterns are absent from all 5 null runs") is descriptively true, but the phrase carries an implication of significance the arithmetic does not support. Consider stating plainly that the null *never produced* deep patterns in 5 runs, while explicitly acknowledging this is suggestive, not significant.
- Reporting a "+71,728" effect size from 5 permutations invites the objection that σ estimated from n = 5 is itself extremely noisy (the K = 6 value does not reproduce cleanly from the rounded μ/σ shown, precisely because σ is unstable at this n). The footnote helps; consider going further and reporting the raw null counts per permutation for K = 4–6 so the reader sees the actual spread.
- The manuscript itself notes "100+ permutations would be needed to establish p < 0.01 bounds." Since each permutation is ~130 s, **100 permutations is ~3.6 GPU-hours** — entirely feasible on the same single H100. Running them would convert the paper's weakest section into one of its strongest, and is the single highest-value additional experiment.

### Major 3 — The flagship K = 22 biological result is under-supported and not fully reproducible
The K = 22 "neuronal antiviral RNA-helicase" pattern is given prominence (dedicated subsection, Table 5, and a recurring narrative thread). As currently supported it is a *single itemset shared by 8 proteins*. The manuscript has been appropriately softened to "consistent with … not a demonstrated mechanism," and the GO true-path check (0 parent–child pairs; one InterPro2GO-derived link, giving 21 independent features) is a genuine strength. Remaining problems:
- **The 8 UniProt accessions are not listed.** The text now says they are recoverable via the released analysis script from the source transaction data. That is honest, but for the paper's single most-highlighted biological pattern a reviewer will expect the eight identifiers in a supplementary table. If the source transaction data cannot be re-queried, this should be stated as an explicit reproducibility limitation, not left implicit.
- **GO evidence codes are not distinguished.** With n = 8 proteins, a single densely (electronically) annotated protein family can generate the entire signature. Whether the innate-immune / neuronal-compartment GO terms are experimentally supported (EXP/IDA) or electronically inferred (IEA) is decisive for the interpretation and is currently unaddressed in the main text.

*Suggested remedy:* add a supplementary table of the 8 accessions with per-term evidence codes; if unavailable, demote the K = 22 pattern from "highlighted discovery" to "illustrative maximum-depth example" and lead the biological section with the better-supported intermediate-K modules (β-lactam PBP, RTK), which recover *validated drug targets* and are far more defensible.

### Major 4 — No same-dataset baseline; the scale comparison is apples-to-oranges
Table 3 compares ET-miner to prior GPU/distributed FIM systems, but every competitor ran on a *different dataset and hardware generation*; the "5.1× more transactions" and "deeper than any previously reported GPU FIM result" claims are therefore about scale reached, not a controlled speed/quality comparison. The one genuinely controlled comparison in the paper — Direct-GPU vs SON at identical support (21×, 95.2% miss rate) — is excellent and should be foregrounded as *the* methods result. Prior tools plausibly cannot ingest 76.9M transactions, but even a subset comparison (e.g., ET-miner vs GMiner on a 1–15M-transaction slice) would substantiate the efficiency claim rather than only the scale claim. If such a run is infeasible, the manuscript should state explicitly that no head-to-head comparison on identical data was possible and why.

### Major 5 — Reported itemset counts conflate the full (redundant) pattern space
The headline "26.8 million patterns" is the *complete* frequent-itemset space, which by construction is dominated by subset/superset redundancy (every frequent K-itemset implies 2^K − 2 frequent sub-itemsets). Limitation 5 acknowledges this and notes closed-itemset filtering is implemented, but the manuscript never reports the closed/maximal counts. As written, "26.8 million co-occurrence patterns" overstates the number of *biologically distinct* modules by a large and unstated factor. Report the closed and maximal frequent-itemset counts alongside the total; this is a cheap post-processing step on data already in hand and materially changes how a reader should interpret the scale claims.

---

## Minor Comments

1. **Abstract.** Accurately reflects the (restored) base-run scope; the "7.3 minutes (mining only)" qualifier is good. Consider adding one clause on what the null model does *and does not* cover, so the scope caveat is visible before Section 3.6.
2. **Memory figures (Sections 1, 2.3, Appendix D).** Three legitimate but different quantities coexist — ~206 GB (naive 1-byte dense), ~26 GB (bit-packed dense / on-GPU bitvector, 214M set), ~5.1 GB (CSR-COO), ~10 GB (bit-packed dense of the 76.9M subset). The recent labelling helps, but a single small table mapping {representation × dataset → size} would eliminate the residual reader confusion the current prose still risks.
3. **pLDDT signal.** Only 2 of 6 pLDDT bins pass the support threshold, and the recurring one is "medium confidence (70–90)." The structural-confidence contribution to the patterns is therefore thin; the manuscript is honest about this, but the phrase "structural property" in Table 5 slightly oversells a single medium-confidence bin. Consider "confidence bin" throughout for consistency with the framing elsewhere.
4. **Intermediate-K protein counts** ("~11,000", "~10,500", "~16,000") are approximate where the artefacts support near-exact values; where a specific itemset's support is known, report it exactly.
5. **Multi-GPU capability.** Now honestly framed as a capability ("all experiments used a single H100"). Good. The third "key design choice" (automatic multi-GPU scaling) still reads slightly oddly in a single-GPU paper; consider demoting it from a headline design choice to an implementation note, since it is not exercised by any reported result.
6. **`miettinen2020`** (Boolean matrix factorization) is currently uncited. Either cite it where the bitvector/Boolean-matrix representation is introduced (Section 2.3) or remove it.
7. **True-path rule (Limitation 2).** The added quantification (0 parent–child pairs in the K = 22 set) is excellent; consider extending the same check to the highlighted intermediate-K patterns, since the true-path concern applies there too.
8. **Reproducibility statement.** The paper promises code + results on publication (Zenodo DOI + GitHub). Given that some source artefacts (full transaction parquet, K = 22 accessions) are not currently deposited, a short explicit data-availability paragraph — stating exactly which artefacts are and are not released — would pre-empt reviewer concern.

---

## Selected Line/Section References

- **§3.6 (null model), Table 4:** t-statistic footnote and p < 0.45 correction now present and correct. See Major 1–2 for the residual scope/power issue.
- **§3.5 + Table 5 (K = 22):** interpretation softened correctly; see Major 3 for accessions + evidence codes.
- **§3.1, Table 2 (campaign):** min_count values (Power = 768, Ultra = 16) reconcile with the mining logs; Base/Super rows reconcile with `pipeline_214m.log`.
- **§4.2, Table 3 (scale comparison):** see Major 4 (apples-to-oranges) and the GMiner "15M synthetic / 1.7M real" labelling, which is now correct.
- **§3.2, Table 6 (K-distribution):** column sums to 26,849,505 exactly; matches the godmode log.

---

## Questions for the Authors

1. Can the permutation null model be run at the Opus threshold (min_count = 8), even at n = 5, to test whether deep (K ≥ 7) patterns exceed a count-preserving null there? If not, why is the count-preserving null at 0.001% argued to generalise to 0.00001%?
2. Each permutation is ~130 s. Is there an obstacle to running the 100 permutations the manuscript itself identifies as necessary for p < 0.01? If so, what?
3. Can the 8 K = 22 UniProt accessions and their per-GO-term evidence codes be provided as a supplementary table? If the source data is unavailable, please state this as an explicit limitation.
4. What are the **closed** and **maximal** frequent-itemset counts corresponding to the 26.8M total? These should be reportable from data already computed.
5. Is any controlled comparison against an existing GPU FIM tool on identical (even subset) data possible, to support the efficiency (as opposed to scale) claim?
6. The K = 9 distribution peak: how sensitive is its position to the support threshold and to the multi-feature-protein inclusion criterion (≥ 2 features)?

---

## Reviewer's Bottom Line

The engineering core is sound, novel at the reported scale, and now reported with numbers that are internally consistent and artefact-traceable — a real strength relative to the version that preceded this revision. The paper is publishable as a **methods contribution**. What holds it back is that the *biological* framing currently claims more than the *statistics* and *reproducibility* deliver: the null model does not reach the threshold of the headline result (Major 1), n = 5 cannot support the inferential weight placed on it (Major 2), and the single most-highlighted pattern is an 8-protein anecdote without listed accessions or evidence codes (Major 3). Two of these are fixable with ≤ 4 GPU-hours on the machine already used (100 permutations; an Opus-threshold null), and the third with a supplementary table. Addressing them would turn a solid systems paper with an over-reaching biology section into a genuinely strong one.

---

# Appendix A — Full Stage-by-Stage Checklist (peer-review skill Stages 1–7 + Final Checklist)

Legend: ✅ satisfied · ⚠️ partial / needs attention · ❌ missing · N/A not applicable. Each item carries a one-line justification. (Note: the skill's `references/reporting_standards.md` and `references/common_issues.md` are named in SKILL.md but not shipped in this installation, so external checklist cross-referencing was done from domain knowledge.)

## Stage 1 — Initial Assessment
- **Central question** ✅ "Which combinations of protein features co-occur more than expected across hundreds of millions of proteins?" — stated explicitly (§1).
- **Main findings/conclusions** ✅ Exhaustive FIM to K=22 on 76.9M proteins in 7.3 min; unimodal K-peak at 9; null-model enrichment for K≥4 at 0.001%.
- **Scientifically sound** ⚠️ Systems claims sound; biological/statistical claims partly outrun support (Majors 1–3).
- **Venue-appropriate** ✅ Reads as a methods/applied-computation preprint (Zenodo); scope fits that.
- **Immediate fatal flaws** ✅ None precluding publication after revision.

## Stage 2 — Section-by-Section
**Abstract & Title** — Accuracy ✅ (matches restored scope) · Clarity ✅ (title now specific, no motif overclaim) · Completeness ⚠️ (null-model *scope* caveat not signalled in abstract; Minor 1) · Accessibility ✅.
**Introduction** — Context ✅ (AlphaFold/UniProt scale) · Rationale ✅ · Novelty ✅ (higher-order vs pairwise, single-GPU scale) · Literature ✅ (pairwise networks, GPU-FIM, bioinformatics FIM cited) · Objectives ✅.
**Methods** — Reproducibility ⚠️ (algorithm fully described + versions given, but source transaction data + K=22 accessions not deposited; Major 3, Minor 8) · Rigor ✅ · Detail ✅ (pseudocode, kernel design in appendices) · Ethics N/A (public DB, no subjects — but should be *stated*; Stage 6) · Statistics ⚠️ (null-model n=5; Major 2) · Validation ⚠️ (SON-vs-Direct controlled comparison strong; null validation single-threshold; Major 1).
  - *Critical elements:* sample size / power ⚠️ (no power analysis; n=5 permutations) · randomization ✅ (Fisher–Yates shuffle, seed 42) · inclusion/exclusion ✅ (≥2 features, min_count thresholds) · data-collection protocol ✅ (UniProt release 2025_01) · software versions ✅ (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5) · multiple-comparison correction ❌ (millions of itemsets tested; no FDR — acknowledged as Limitation 1 but not applied).
**Results** — Presentation ✅ · Figures/Tables ✅ (4 figures present on disk; tables well-formed) · Statistics ⚠️ (effect sizes given; p-bounds weak, no CIs) · Objectivity ⚠️ (K=22 interpretation softened but still prominent; Major 3) · Completeness ⚠️ (closed/maximal counts omitted; Major 5) · Reproducibility ⚠️ (summary artefacts yes, raw transactions no).
  - *Common issues:* selective reporting ⚠️ (5 highlighted patterns chosen for known validation — disclosed, acceptable) · inappropriate tests ⚠️ (n=5 "Z") · missing variability ⚠️ (per-permutation spread not shown; Major 2) · over-fitting/circular ✅ (none; exhaustive not modelled) · batch/confounder ⚠️ (annotation-density confounding not quantified at Opus threshold; Major 1) · missing controls ⚠️ (null single-threshold).
**Discussion** — Interpretation ⚠️ (mostly data-supported; K=22 the exception) · Limitations ✅ (5 explicit limitations, now including 0-parent-child note) · Context ✅ · Speculation ✅ (hypothesis language now explicit) · Significance ✅ · Future directions ✅.
  - *Red flags:* overstated conclusions ⚠️ (26.8M "patterns" counts redundant space; Major 5) · ignoring contrary evidence ✅ · causal-from-correlational ✅ (now hedged) · inadequate limitations ✅ · mechanistic-without-evidence ✅ (softened to "consistent with").
**References** — Completeness ✅ · Currency ✅ (through 2024) · Balance ✅ · Accuracy ✅ (13 citation errors corrected this revision; independently re-verified) · Self-citation ✅ (none excessive) · *one uncited bibitem* ⚠️ (`miettinen2020`; Minor 6).

## Stage 3 — Methodological & Statistical Rigor
- Assumptions (normality/independence) ⚠️ — "Z"-framing assumes normality of null counts from n=5; not tested (addressed by t-relabel + caveat, but n too small).
- Effect sizes with p ✅ (both reported) · Multiple-testing correction ❌ (not applied; Limitation 1) · Confidence intervals ❌ (none) · Sample-size/power justification ❌ (n=5, no power analysis; Major 2) · Parametric vs non-parametric ⚠️ (permutation test is appropriate; the Gaussian-Z reporting was the weak part, now hedged) · Missing data ✅ (N/A — census, not sample) · Exploratory vs confirmatory ⚠️ (exhaustive mining is exploratory; framed acceptably).
- **Experimental design:** controls ⚠️ (count-preserving null is a good control but single-threshold) · replication ⚠️ (5 permutations) · confounders ⚠️ (annotation density) · randomization ✅ · blinding N/A · design-optimal-for-question ✅.
- **Computational/bioinformatics:** methods described ✅ · versions/params documented ✅ · code available ⚠️ (promised on publication, not yet) · algorithms validated ✅ (SON cross-check; known-biology recovery) · assumptions met ✅ · batch correction N/A.

## Stage 4 — Reproducibility & Transparency
- Raw data in repository ❌ (source `transactions_214m.parquet` not deposited; lost with compute environment) · accession numbers for public DB ⚠️ (UniProt release named; the 8 K=22 accessions not listed — Major 3) · data-sharing restrictions justified ⚠️ (loss not yet stated as a limitation; Minor 8) · standard formats ✅ (Parquet).
- Analysis code available ⚠️ (GitHub promised on publication) · materials/protocols sufficient ✅ (pseudocode + kernel appendices) · protocols detailed ✅.
- Reporting-guideline compliance N/A (no CONSORT/PRISMA/ARRIVE analogue for computational proteome mining; a data-availability statement is the relevant standard and is ❌ missing).

## Stage 5 — Figure & Data Presentation
- Resolution/labelling — cannot render (no compile); PDFs present on disk ✅. Captions read as standalone ✅. Axes/units — not verifiable from source; **recommend visual check on compile**.
- Error bars ⚠️ (count data; none expected, but the K-distribution/null comparison could show the 5-permutation spread).
- Colour accessibility — not verifiable from source (⚠️ recommend colorblind-safe check on compile).
- **Integrity:** no image-manipulation risk (all figures are generated plots, `generate_figures.py` present) ✅. Representative-image concerns N/A.
- **Clarity:** figures stand alone ✅ · message clear ✅ · redundant panels ✅ (none) · two **orphaned figure files** (`figures/kdist_shift.pdf`, `figures/pruning_savings.pdf`) remain on disk from the removed expanded run — harmless (not `\includegraphics`'d) but should be deleted for a clean repository ⚠️.

## Stage 6 — Ethical Considerations
- Human subjects N/A (public sequence/structure database, no personal data). ⚠️ **Should be stated explicitly** — a one-line "no human/animal subjects" note is standard.
- Animal research N/A.
- **Research integrity:** fabrication/falsification ✅ (this revision's entire purpose was to lock every number to a preserved artefact; the previously fabricated Table 1 was corrected) · authorship ⚠️ (C. claudya = "Anthropic, Claude Code Opus 4.6" as a listed author — an AI-tool authorship that many venues explicitly prohibit; flag for author's venue policy) · **competing interests** ❌ (no COI statement; one author is affiliated with the AI vendor whose tool co-produced the work — disclosure expected) · **funding** ❌ (no funding statement) · plagiarism/duplicate ✅ (none apparent; a Zenodo v1 exists — the relationship to v1 should be disclosed as a versioned preprint update).

## Stage 7 — Writing Quality
- Structure/organization ✅ (logical IMRaD + appendices) · flow ✅ · transitions ✅ · narrative ✅.
- Language clear/concise ✅ · jargon defined ✅ · grammar/spelling ✅ (typography pass applied) · sentence complexity ✅ · passive-voice ✅ (acceptable for methods).
- Accessibility: non-specialist can grasp findings ✅ · technical terms explained ✅ · significance clear ✅.

## Final Checklist
- Summary conveys overall assessment ✅ · Major concerns identified & justified ✅ (5) · Suggested revisions specific/actionable ✅ · Minor issues categorized ✅ (8) · Statistical methods evaluated ✅ (Stage 3; Majors 1–2) · Reproducibility/data availability assessed ✅ (Stage 4; Major 3) · Ethics verified ✅ (Stage 6 — surfaced 3 missing statements: COI, funding, subjects) · Figures/tables evaluated ⚠️ (source-level only; visual check deferred to compile) · Writing assessed ✅ · Tone constructive ✅ · Proportionate ✅ · Recommendation consistent with findings ✅ (Major revision).

## New items surfaced only by the exhaustive checklist (not in the narrative review above)
1. **[Major] No competing-interests statement**, despite an author affiliated with the AI vendor whose tool co-authored the work (Stage 6). Most venues require this.
2. **[Major] No funding statement** (Stage 6).
3. **[Minor] AI-tool listed as a co-author** ("Claude Code Opus 4.6") — many journals (ICMJE, Nature, Science) prohibit AI authorship and require it be moved to Acknowledgements/Methods instead. Venue-dependent; flag before submission.
4. **[Minor] No "no human/animal subjects" statement** (Stage 6) — trivial to add.
5. **[Minor] No multiple-testing correction** applied to the millions of itemsets (Stage 2/3) — acknowledged as Limitation 1 but worth stating that reported patterns are therefore not individually FDR-controlled.
6. **[Minor] Two orphaned figure PDFs** (`kdist_shift.pdf`, `pruning_savings.pdf`) from the removed expanded run remain in `figures/` — delete for repository hygiene (Stage 5).
7. **[Minor] Relationship to the published Zenodo v1** should be disclosed as a versioned-preprint update (Stage 6, duplicate-publication check).

---

## Appendix B — Fixes applied post-review (2026-07-12)

The following checklist items were resolved in the same session:

- **Funding statement** — added ("no external funding"). ✅ (was Appendix-A new item 2)
- **Competing-interests statement** — added; discloses that co-author C. claudya is affiliated with Anthropic while E. Ahmic is independent. ✅ (was new item 1)
- **Ethics statement** — added ("public data only, no human/animal subjects"). ✅ (was new item 4)
- **Data & Code Availability statement** — added; states what is released vs. regenerable, and discloses the Zenodo-v1 versioned-preprint relationship. ✅ (was new items 7 + Stage-4 gap)
- **Orphaned figure files** (`kdist_shift.pdf`, `pruning_savings.pdf`) — removed from `figures/`. ✅ (was new item 6)
- **`miettinen2020`** — now cited (Boolean-matrix representation, §1); zero uncited bibitems remain. ✅ (was Minor 6)

**Deliberately not changed (author decision):**
- **AI co-authorship (C. claudya / Claude Code).** The author has chosen to retain C. claudya as a listed co-author. This remains a **venue-dependent** consideration: ICMJE, Nature, and Science do not permit AI systems as authors and would require the contribution be moved to Acknowledgements/Methods. Flagged here for the target venue's policy, but left as the author intends. (was new item 3 — now an author decision, not a defect.)

**Still open (require author compute/data decisions — the hard revisions):**
- Major 1 (Opus-threshold null model), Major 2 (100 permutations), Major 3 (K=22 accessions + evidence codes), Major 4 (same-data baseline), Major 5 (closed/maximal counts), Appendix-A new item 5 (FDR note).
