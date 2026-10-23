"""V2 pass 10: fixes from the fresh-context adversarial review of the audited tex.

Removes the uncited attributions the review found (transfer overhead of "prior GPU FIM systems",
"inherently sequential" FP-tree construction, the receptor-tyrosine-kinase / erlotinib sentence, the
"bacterial equivalent of the ubiquitin-proteasome system" sentence, "beyond K=2"), restores one
verified GPApriori statement with its source, states which itemset holds the K=13 and K=11 level
maxima and adds the pattern-specific maxima (RESULTS P-031), tightens the reporting convention, and
corrects scope wording. Exact-string edits; each anchor must occur exactly once.

Usage:
    python revise_tex_v2_pass10_review.py <path-to-tex>
"""
import sys

path = sys.argv[1]
tex = open(path, encoding="utf-8").read()
orig = tex


def rep(old, new):
    global tex
    n = tex.count(old)
    assert n == 1, f"expected 1 occurrence, found {n}: {old[:90]!r}"
    tex = tex.replace(old, new)


# Introduction: "completion" is no source's word (varadi2024 lists exclusions).
rep(
    r"The completion of the AlphaFold Protein Structure Database~\cite{jumper2021,varadi2022,varadi2024} represents",
    r"The AlphaFold Protein Structure Database~\cite{jumper2021,varadi2022,varadi2024} represents",
)
# Introduction: uncited attribution about prior GPU FIM systems.
rep(
    r"crossing the PCIe bus, greatly reducing the transfer overhead that dominates prior GPU FIM systems. Applied to",
    r"crossing the PCIe bus. Applied to",
)
# Methods 2.2: the sequential-construction clause is unsourced and reads as attributed to han2000.
rep(
    r"We chose Apriori's level-wise structure over FP-Growth~\cite{han2000} specifically because its support counting step maps naturally to GPU SIMD parallelism, while FP-tree construction is inherently sequential.",
    r"We chose Apriori's level-wise structure over FP-Growth~\cite{han2000} because its support counting step maps naturally to GPU SIMD parallelism.",
)
# Methods 2.5: replace the uncited "traditional approaches" sentence by the verified GPApriori description
# ("Only the vertical lists of first generation will be saved in graphics memory"; "candidates are copied from
# main memory to graphic memory by host code ... the support value results are copied back to main memory").
rep(
    r"Traditional approaches generate candidates on the CPU, transfer them to the GPU for counting, return results, and filter on the CPU, repeating this cycle at every level. ET-miner keeps",
    r"GPApriori~\cite{zhang2011}, for example, keeps the item bitsets in GPU memory but generates candidates on the CPU, copies them to the GPU for counting and copies the support values back at every level. ET-miner keeps",
)
rep(
    "\\end{thebibliography}",
    "\\bibitem{zhang2011}\nF.~Zhang, Y.~Zhang, and J.~Bakos.\n\\newblock GPApriori: GPU-accelerated frequent itemset mining.\n\\newblock In \\emph{Proc.\\ IEEE Int.\\ Conf.\\ Cluster Computing}, pp.\\ 590--594, 2011.\n\n\\end{thebibliography}",
)

# Methods 2.6: the convention now covers K=19 and the pattern-specific maxima of K=13 and K=11.
rep(
    r"When the description names Pfam identifiers, the count is the largest support among the itemsets at that $K$-level that contain every named identifier, and the text states how many itemsets match. When the description names no identifier ($K{=}13$ and $K{=}11$), the count is the largest support among all itemsets at that $K$-level. In both cases the count is a maximum over itemsets rather than the support of one named itemset, unless exactly one itemset matches.",
    r"When the description names Pfam identifiers ($K{=}17$ and $K{=}12$), the count is the largest support among the itemsets at that $K$-level that contain every named identifier, and the text states how many itemsets match. When the description names no identifier ($K{=}19$, $K{=}13$ and $K{=}11$), the count is the level maximum, the largest support among all itemsets at that $K$-level; where that maximum is held by an itemset other than the described pattern ($K{=}13$ and $K{=}11$), the text also gives the largest support among the itemsets that carry the described features. A count is the support of a single itemset only when exactly one itemset matches ($K{=}19$ and $K{=}17$).",
)

# Results 3.4: K=19 is the sole itemset of its level.
rep(
    r"\paragraph{$K{=}19$: RNA Spliceosome Processing Hub} (187 proteins). DEAD/DEAH-box helicases",
    r"\paragraph{$K{=}19$: RNA Spliceosome Processing Hub} (187 proteins; the single itemset at this level). DEAD/DEAH-box helicases",
)
# Results 3.4: the SH3-SH2-kinase architecture is the Src-family (non-receptor) architecture of boggon2004;
# the receptor-tyrosine-kinase / drug sentence is uncited and EGFR carries no SH2/SH3 domain.
rep(
    "\\paragraph{$K{=}17$: Receptor Tyrosine Kinase Signaling Hub}\n(611 proteins). Protein tyrosine kinase (PF07714) with SH2 (PF00017)\n",
    "\\paragraph{$K{=}17$: Src-Family Kinase Signaling Module}\n(611 proteins; one itemset at this level contains all three domains). Protein tyrosine kinase (PF07714) with SH2 (PF00017)\n",
)
rep(
    "regulation. This 17-feature combination describes the core receptor tyrosine\nkinase (RTK) signaling machinery, one of the most targeted pathways in cancer\ndrug discovery: imatinib, dasatinib, and erlotinib all target proteins matching\nthis functional signature.\n",
    "regulation.\n",
)
# Results 3.4: K=13 and K=11 level maxima are held by other itemsets (RESULTS P-031).
rep(
    r"\paragraph{$K{=}13$: Helicase-Recombinase DNA Repair Module} (up to 11,521 proteins). DEAD-box helicases co-occurring with DNA repair, recombination, replication, and the SOS response, recovered here from annotation data alone.",
    r"\paragraph{$K{=}13$: Helicase DNA Repair and Recombination Module} (level maximum 11,521 proteins, held by an itemset carrying the DNA repair, recombination and replication terms; the 23 itemsets that also carry the SOS-response term reach 10,978). DEAD-box helicases co-occurring with DNA repair, recombination, replication, and the SOS response, recovered here from annotation data alone.",
)
rep(
    r"\paragraph{$K{=}11$: AAA+ ATPase Proteasome Complex} (up to 28,913 proteins). Protein quality control machinery: AAA-family ATPase with Clp protease domains. The bacterial equivalent of the eukaryotic ubiquitin-proteasome system.",
    r"\paragraph{$K{=}11$: AAA+ ATPase--Clp Protease Module} (level maximum 28,913 proteins, held by a penicillin-binding-protein itemset; the 491 itemsets that combine an AAA domain (PF00004, PF07724 or PF17871) with a Clp domain (PF10431, PF00574 or PF02861) reach 18,257). Protein quality control machinery: AAA-family ATPase with Clp protease domains.",
)

# Discussion 4.2: Meysman mined doublets and triplets and Terrapon reports triplets, so "beyond K=2" is false;
# "fundamentally" and "again limited to pairs" overstate; 76.9M of 214.7M proteins were mined.
rep(
    r"Prior protein domain co-occurrence studies are fundamentally limited in scope.",
    r"Prior protein domain co-occurrence studies are limited in scope.",
)
rep(
    r"Terrapon et al.~\cite{coin2009} used statistical co-occurrence for novel domain detection in \emph{Plasmodium falciparum}, again limited to pairs.",
    r"Terrapon et al.~\cite{coin2009} used statistical co-occurrence of domain pairs for novel domain detection in \emph{Plasmodium falciparum}.",
)
rep(
    r"None of these approaches perform exhaustive itemset mining beyond $K{=}2$. ET-miner's discovery of patterns spanning up to 22 co-occurring features across the full AlphaFold universe~\cite{varadi2024} represents a qualitative advance.",
    r"ET-miner's discovery of patterns spanning up to 22 co-occurring features across 76.9 million multi-feature proteins of the AlphaFold Database~\cite{varadi2024} represents a qualitative advance.",
)
# Discussion 4.3 / Limitations: precise cross-references; the true-path rule concerns implied annotation.
rep(
    r"The $K{=}12$ bacterial cell wall synthase module (Section~\ref{sec:results})",
    r"The $K{=}12$ bacterial cell wall synthase module (Section~\ref{sec:intermediate})",
)
rep(
    r"and propagation of annotations through the hierarchy and correlations between GO terms are known pitfalls~\cite{rhee2008}",
    r"and indiscriminate propagation of annotations through the hierarchy and ignoring the correlations between GO terms are known pitfalls~\cite{rhee2008}",
)
rep(
    r"The Gene Ontology's true-path rule guarantees that any protein annotated with a specific GO term is also annotated with all parent terms, creating structured co-occurrence that is definitional rather than biological.",
    r"Under the Gene Ontology's true-path rule, annotation to a specific GO term implies annotation to all of its parent terms; where such parents are explicit items in our vocabulary, this creates structured co-occurrence that is definitional rather than biological.",
)
rep(
    r"so its depth is not driven by this effect (Section~\ref{sec:results})",
    r"so its depth is not driven by this effect (Section~\ref{sec:k22})",
)
rep("\\subsection{The $K{=}22$ Ceiling}\n", "\\subsection{The $K{=}22$ Ceiling}\n\\label{sec:k22}\n")
# Future work: the preprint reports 1.8M complexes, so it is cited for the expansion only.
rep(
    r"to include predicted protein complexes~\cite{afdb2026news,han2026afdb}: 1.7 million",
    r"to include predicted protein complexes~\cite{afdb2026news} (described in an accompanying preprint~\cite{han2026afdb}): 1.7 million",
)
# Conclusion: 76.9M of the 214.7M proteins were mined.
rep(
    r"across the entire known protein universe---architecture that no pairwise analysis could reveal.",
    r"across 76.9 million multi-feature proteins of the AlphaFold Database---architecture that no pairwise analysis could reveal.",
)

assert tex != orig
open(path, "w", encoding="utf-8").write(tex)
print("pass 10 applied:", len(orig.splitlines()), "->", len(tex.splitlines()), "lines")
