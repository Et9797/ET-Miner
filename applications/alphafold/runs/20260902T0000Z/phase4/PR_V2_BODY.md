## Summary

V2 of `paper/et_miner_proteome.tex`: the preprint now reports only the fresh run on this box (single pinned RTX 3090 for the campaign and the SON comparison, two GPUs for the null model and the per-K exports), with every number traced to `RESULTS.md` / `COMPARISON_REPORT.md` (`CHANGELOG_V1_V2.md` maps V1 to V2 value by value). This last commit adds the citation audit: every literature value describing another tool was deleted, every remaining `\cite` was checked against the cited work's text, and the bibliography was pruned and renumbered. Zero H100 references. The PDF builds clean (12 pages, no undefined references, one pre-existing overfull box in the author block).

Full audit report with verbatim quotes: `runs/20260902T0000Z/phase4/citation_audit/CITATION_AUDIT.md` (58 claims over 40 keys: 35 verified, 16 not supported, 6 unverifiable, 1 partly supported). Edit scripts: `revise_tex_v2_pass8_citations.py`, `revise_tex_v2_pass9_citations.py`, `revise_tex_v2_pass10_review.py`, `reorder_bibliography.py` (all anchor-asserting).

## Citations removed (12 bibliography entries dropped)

Deleted with the tool-comparison material (Introduction GPU-FIM sentence, Discussion 4.2 with Table 6, Appendix D Tables 7-8, Appendix C.1 popcount comparison, and the per-level PCIe contrast clauses in Methods 2.5 / Discussion 4.1 / Appendix C.3):
- `chon2018` GMiner, `chon2024` GMiner++, `chon2018b` BIGMiner, `djenouri2019` CGSS, `fang2009`, `borgelt2003`, `acmsurvey2021`, `luna2019`, `zaki1997`, `zaki2000` (Eclat).

Deleted because the cited work does not support the sentence:
- `abramson2024` (AlphaFold 3 paper never mentions the database expansion; replaced by the EMBL-EBI announcement `afdb2026news` and the bioRxiv preprint `han2026afdb`, both verified verbatim).
- `dillingham2008` (RecB is an SF1 helicase in the source; it never links DEAD-box helicases to RecBCD; the "consistent with RecBCD-like repair complexes" clause is gone).

Consequence to be aware of: GPApriori (`zhang2011`) is now the only GPU frequent-itemset-mining system cited, and only for the one statement its text supports (Methods 2.5: it keeps the item bitsets in GPU memory but generates candidates on the CPU, copies them to the GPU for counting and copies the support values back at every level), which replaced an uncited "Traditional approaches ..." sentence. That verified fact also contradicts the contrast V1 drew (it is the reason the "transfer transaction bitmaps at each iteration" clauses were deleted).

## Citations that failed verification (and what was done)

Not supported by the source (deleted or cut back to the source's words):
- Introduction "over 200 million proteins" was cited to `jumper2021`/`varadi2022` (Varadi 2022 reports 360,000 structures across 21 proteomes): `varadi2024` (214 million) added to the cite group; "spanning virtually all known organisms" deleted (no source).
- Introduction "50-350x speedups over Eclat and FP-Growth": no cited source gives such a range (Fang 2009 reports its GPU code 4-16x slower than CPU FP-growth; Zhang 2011 compares against Apriori only; Djenouri's 350x is against the authors' own sequential algorithm on a GPU cluster). Sentence deleted.
- BIGMiner "largest prior scale, 100M transactions": its abstract says "up to 6.5 billion transactions". Borgelt and GMiner table rows: no retrievable support. Deleted.
- Appendix C.3 "implementations that transfer transaction bitmaps ... at each iteration" cited GPApriori, which keeps them resident. Deleted.
- Results 3.3 "DEAD-box helicases such as RIG-I and MDA5": Rehwinkel & Gack call them SF2 helicases with a DECH-box helicase domain ("DEAD" never occurs). Now "The SF2 helicases RIG-I and MDA5".
- Results 3.4 / Discussion 4.3 "beta-lactam antibiotics bind these exact domains": Sauvage et al. state that the penicillin-binding (transpeptidase) domain binds beta-lactams; the transglycosylase domain is the moenomycin target. Cut to the transpeptidase domain.
- Discussion 4.2 "graph mining is NP-hard and scales poorly" attributed to Mrzic et al.: "NP" never occurs. Clause deleted.
- Discussion 4.2 Wang et al. "limited to individual proteomes": they built networks for 417 organisms. Clause deleted.
- Methods 2.5 / Discussion 4.1 "SON ... lowered local threshold" cited to Savasere et al.: their Partition algorithm uses the same support fraction per partition and never uses the name SON. Now "the Partition algorithm of Savasere, Omiecinski and Navathe", the lowered threshold stated as ET-miner's own choice.
- Methods 2.1 "UniProt cross-reference pipeline": not a thing in the UniProt paper; the extractor reads the `DR Pfam` / `DR GO` lines of the flat file, which is what the text now says.
- Future work: the homodimer expansion and its numbers were cited to the AlphaFold 3 paper; they come from the EMBL-EBI announcement of 16 March 2026 (now cited; its 19 May 2026 update replaces "heterodimers ... will follow" with the added heterodimer counts; the interactome quotation is attributed to Janet Thornton as on the page).

Unverifiable (full text closed, cut back to the retrievable abstract / key points):
- `chothia2003`: now "the formation of most proteins by gene duplication, recombination and divergence".
- `rhee2008`: now the Key Points (evidence codes overlooked; propagation through the hierarchy and correlations between GO terms as pitfalls) instead of "electronically propagated GO annotations may create artificial co-occurrence".
- `chon2018` GMiner: the full text could not be retrieved from any source (ScienceDirect blocks non-browser access despite the CC BY-NC-ND licence), so every GMiner statement was unverifiable; all deleted.

Unattributed claims removed at the same time: "`__popcll` ... in a single clock cycle" (no source) and "orders of magnitude less than conventional approaches". "GO annotations are updated monthly" now cites `go2023` (monthly GO releases) and `uniprot2023` (eight-weekly UniProt releases), both verified.

Bibliographic corrections: `varadi2024` full title restored; `coin2009` four authors listed. Found but moot because the entries were removed: `fang2009` carried a ten-author list that belongs to another work (the paper has five authors); `djenouri2019` had the wrong title; `zaki1997` pages 283-286 not 283-296; `chon2024` has two authors; `luna2019` lacked its article number.

Reference numbering: `\bibliographystyle{unsrt}` was inert because the bibliography is hand-written; the 28 remaining `\bibitem`s are now in first-citation order (verified from the `.aux`: [1] = Jumper 2021, none out of order).

## K=13 / K=11 convention

Reported values are the measured level maxima from the RTX 3090 rerun (RESULTS `P-030`): K=13 11,521 proteins, K=11 28,913 proteins, both defined the same way. The convention is stated in Methods 2.6 "Reporting Convention for Highlighted Patterns": counts are read from the per-K itemset tables of the Blitz run; when the description names Pfam identifiers (K=17, K=12) the count is the largest support among the itemsets at that level containing every named identifier, with the number of matching itemsets stated; when no identifier is named (K=19, K=13, K=11) it is the level maximum. The Results footnote was replaced by a pointer to Methods 2.6.

One thing you need to know: the adversarial review decoded the Blitz tables and found that the K=11 level maximum (28,913) is held by a penicillin-binding-protein itemset, not by an AAA+/Clp itemset, and that the K=13 maximum itemset carries repair, recombination and replication but no SOS term. I re-derived this (`decode_intermediate_k.py`, RESULTS `P-031`, INCONSISTENCIES I-016). Per your Decision 2 the level maxima stay as the reported counts, but the text now says which itemset holds each maximum and adds the pattern-specific maxima: K=13 "the 23 itemsets that also carry the SOS-response term reach 10,978"; K=11 "the 491 itemsets that combine an AAA domain (PF00004, PF07724 or PF17871) with a Clp domain (PF10431, PF00574 or PF02861) reach 18,257". If you prefer to pin these two patterns to itemsets instead (the override you left open), the numbers to use are 10,978 and 18,257. K=17 is the single matching itemset (611, tilde removed) and K=19 the single itemset of its level (187); both facts are now stated.

## Adversarial review of the audited tex (pass 10)

A fresh-context reviewer read the final tex against the verdict files and the Blitz tables and found 12 defects, all fixed in `revise_tex_v2_pass10_review.py`:
- Uncited attributions deleted: "the transfer overhead that dominates prior GPU FIM systems" (Introduction); "FP-tree construction is inherently sequential" (Methods 2.2, read as attributed to Han et al.); "None of these approaches perform exhaustive itemset mining beyond K=2" (Discussion 4.2; Meysman mined triplets and Terrapon reports triplets and a quartet, so "again limited to pairs" and "fundamentally" also went); "the bacterial equivalent of the eukaryotic ubiquitin-proteasome system" (K=11).
- K=17: the SH3-SH2-kinase architecture that Boggon and Eck describe is the non-receptor Src/Abl/Tec architecture, and erlotinib's target EGFR carries no SH2/SH3 domain, so the heading "Receptor Tyrosine Kinase Signaling Hub" became "Src-Family Kinase Signaling Module" and the uncited imatinib/dasatinib/erlotinib sentence was deleted. K=13 heading: "Helicase-Recombinase" (no recombinase domain in the itemset) became "Helicase DNA Repair and Recombination Module"; K=11 heading: "AAA+ ATPase Proteasome Complex" became "AAA+ ATPase--Clp Protease Module".
- Scope wording: "across the full AlphaFold universe" and "across the entire known protein universe" now say 76.9 million multi-feature proteins of the AlphaFold Database (the mined subset); "The completion of the AlphaFold Protein Structure Database" lost "completion" (Varadi 2024 lists exclusions).
- Precision: the true-path-rule sentence now distinguishes implied annotation from explicit vocabulary items; Rhee's Key Point wording restored ("indiscriminate propagation", "ignoring the correlations"); the bioRxiv preprint is cited for the expansion only (it reports 1.8M complexes, the announcement 1.7M homodimers); cross-references now point to the K=22 and intermediate-K subsections.

## Wording changes beyond pure deletion (for your eyes)

Per your rule, unverifiable content was deleted rather than rephrased. In eight places the sentence was cut back to the words the source actually states and the citation kept; they are the SF2-helicase, transpeptidase-domain, Partition-algorithm, UniProt flat-file, release-cadence, Chothia, Rhee and homodimer-update edits listed above. All are itemized with the supporting quotes in `CITATION_AUDIT.md`. Naulaerts et al. (a verified primer, formerly in the deleted Discussion 4.2) was re-homed to Discussion 4.2 "Comparison with Protein Domain Mining".

## Verification

- `latexmk` rc 0, 12 pages, 0 undefined references/citations, 0 `??`, 0 H100/Hopper/SXM hits in the PDF text, 0 tool names in the PDF text.
- `runs/20260902T0000Z/phase4/make_changelog_v2.py .` traceability assert passes (every numeric token of the body maps to a RESULTS row, a COMPARISON_REPORT key or a stated role); new §7 summarizes the audit.
- Fresh-context adversarial review of the audited tex: 12 findings, all fixed (pass 10, above).

Do not merge; leave open for review.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Qt98K5LxL87D7oLkvGZDMm
