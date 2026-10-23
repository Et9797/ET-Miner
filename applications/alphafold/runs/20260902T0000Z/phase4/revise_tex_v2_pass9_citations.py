"""V2 pass 9: apply the outcomes of the citation audit (runs/20260902T0000Z/phase4/citation_audit/).

Every literature claim of the V2 tex was checked against the cited work's retrieved text. Claims the
source does not support are deleted (or cut back to the words the source does support); attributions
are corrected where the value is stated by a different work already in the bibliography; one web
source that was quoted without a reference gets a bibliography entry. Exact-string edits; each anchor
must occur exactly once.

Usage:
    python revise_tex_v2_pass9_citations.py <path-to-tex>
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


# --- Group B: domain co-occurrence and pattern biology -----------------------------------------

# rehwinkel2020 calls RIG-I and MDA5 SF2 helicases with a DECH-box helicase domain; "DEAD-box" is not in the source.
rep(
    r"DEAD-box helicases such as RIG-I and MDA5 are established viral RNA sensors in innate immunity~\cite{rehwinkel2020}, so this configuration is biologically plausible;",
    r"The SF2 helicases RIG-I and MDA5 are established sensors of viral RNA in innate immunity~\cite{rehwinkel2020}, so this configuration is biologically plausible;",
)

# dillingham2008: RecB/RecD are SF1 helicases; the review never links DEAD-box helicases to RecBCD.
rep(
    r"DEAD-box helicases co-occurring with DNA repair, recombination, replication, and the SOS response, consistent with bacterial RecBCD-like repair complexes~\cite{dillingham2008}, recovered here from annotation data alone.",
    r"DEAD-box helicases co-occurring with DNA repair, recombination, replication, and the SOS response, recovered here from annotation data alone.",
)

# sauvage2008: beta-lactams bind the penicillin-binding (transpeptidase) domain, not the transglycosylase domain.
rep(
    r"This is a \emph{validated drug target}: beta-lactam antibiotics bind these exact domains.",
    r"This is a \emph{validated drug target}: beta-lactam antibiotics bind the penicillin-binding domain that carries the transpeptidase activity~\cite{sauvage2008}.",
)
rep(
    r"beta-lactam antibiotics target these exact domain combinations, yet ET-miner recovered this co-occurrence pattern from annotation data alone",
    r"beta-lactam antibiotics bind the transpeptidase (penicillin-binding) domain of these proteins~\cite{sauvage2008}, yet ET-miner recovered this co-occurrence pattern from annotation data alone",
)

# wang2011 builds one network per organism for hundreds of organisms; "limited to individual proteomes" is not supported.
rep(
    r"but this approach is inherently pairwise ($K{=}2$) and limited to individual proteomes.",
    r"but this approach is inherently pairwise ($K{=}2$).",
)

# mrzic2018 never states NP-hardness; naulaerts2015 (a verified primer) is re-homed here from the deleted Discussion 4.2.
rep(
    r"Mrzic et al.~\cite{mrzic2018} reviewed frequent subgraph mining for biomolecular data, noting that graph mining is NP-hard and scales poorly.",
    r"Mrzic et al.~\cite{mrzic2018} reviewed frequent subgraph mining for biomolecular data, and Naulaerts et al.~\cite{naulaerts2015} give a primer on frequent itemset mining for bioinformatics.",
)

# coin2009: full author list (four authors; verified against the HAL author manuscript and PubMed).
rep(
    "\\bibitem{coin2009}\nN.~Terrapon et al.\n",
    "\\bibitem{coin2009}\nN.~Terrapon, O.~Gascuel, E.~Mar\\'echal, and L.~Br\\'eh\\'elin.\n",
)

# --- Group C: classical frequent itemset mining --------------------------------------------------

# savasere1995 describes the Partition algorithm with the same support fraction per partition; the lowered
# local threshold is ET-miner's own choice and the name SON is conventional, not the paper's.
rep(
    r"For datasets exceeding GPU memory, the SON algorithm~\cite{savasere1995} partitions the data into chunks that are mined independently at a lowered local threshold, followed by a global counting pass that makes the result exact.",
    r"For datasets exceeding GPU memory, the SON algorithm (the Partition algorithm of Savasere, Omiecinski and Navathe~\cite{savasere1995}) partitions the data into chunks that are mined independently at a local threshold, followed by a global counting pass that makes the result exact.",
)
rep(
    r"A two-pass SON with a proportional or lowered local threshold cannot miss a globally frequent itemset~\cite{savasere1995}.",
    r"A two-pass SON with a proportional local threshold cannot miss a globally frequent itemset~\cite{savasere1995}, and lowering the local threshold, as ET-miner does, can only add candidates to the global pass.",
)

# --- Group A: AlphaFold, databases, statistics ---------------------------------------------------

# The 200-million figure is stated by varadi2024 (214 million), not by jumper2021 or varadi2022 (360,000 structures);
# no retrieved source supports "spanning virtually all known organisms".
rep(
    r"The completion of the AlphaFold Protein Structure Database~\cite{jumper2021,varadi2022} represents one of the largest expansions of biological knowledge in history. With predicted structures for over 200 million proteins spanning virtually all known organisms, the database transforms",
    r"The completion of the AlphaFold Protein Structure Database~\cite{jumper2021,varadi2022,varadi2024} represents one of the largest expansions of biological knowledge in history. With predicted structures for over 200 million proteins, the database transforms",
)

# uniprot2023 has no "cross-reference pipeline"; the extractor reads the DR Pfam / DR GO lines of the flat file
# (runs/20260902T0000Z/phase2/scripts/filter_dat_pfamgo.sh).
rep(
    r"obtained their annotations (Pfam domain assignments and Gene Ontology (GO) terms) from the UniProt~\cite{uniprot2023} cross-reference pipeline (TrEMBL release",
    r"obtained their annotations (Pfam domain assignments and Gene Ontology (GO) terms) from the Pfam and GO cross-reference lines of the UniProt~\cite{uniprot2023} flat file (TrEMBL release",
)

# Release cadences as stated by the sources: GO "released on a monthly basis" (go2023); "UniProt releases are
# published every eight weeks" (uniprot2023).
rep(
    r"\textbf{3.}~GO annotations are updated monthly; our results reflect UniProt TrEMBL release 2026\_01 and may vary with different releases (see Section~\ref{sec:methods}).",
    r"\textbf{3.}~The Gene Ontology is released monthly~\cite{go2023} and UniProt every eight weeks~\cite{uniprot2023}; our results reflect UniProt TrEMBL release 2026\_01 and may vary with different releases (see Section~\ref{sec:methods}).",
)

# chothia2003: only the abstract is retrievable; it states duplication, recombination and divergence.
rep(
    r"This is consistent with the modular architecture of protein functional domains~\cite{chothia2003}, where evolution reuses and combines a finite set of building blocks into increasingly specific functional configurations.",
    r"This is consistent with the formation of most proteins by gene duplication, recombination and divergence~\cite{chothia2003}.",
)

# rhee2008: only the abstract and Key Points are retrievable; "artificial co-occurrence" is not among them.
rep(
    r"These biological interpretations carry important caveats: electronically propagated GO annotations may create artificial co-occurrence~\cite{rhee2008}, and many itemsets are subsets or supersets of each other.",
    r"These biological interpretations carry important caveats: evidence codes are often overlooked in GO-based analyses, and propagation of annotations through the hierarchy and correlations between GO terms are known pitfalls~\cite{rhee2008}; in addition, many itemsets are subsets or supersets of each other.",
)

# abramson2024 (the AlphaFold 3 paper) never mentions the database expansion; the numbers and the quotation come
# from the EMBL-EBI announcement of 16 March 2026 (updated 19 May 2026) and the accompanying bioRxiv preprint.
rep(
    r"The AlphaFold database has recently expanded to include predicted protein complexes~\cite{abramson2024}: 1.7 million high-confidence homodimer predictions have been added to the database, with an additional 18 million lower-confidence homodimers available via bulk download. Heterodimer predictions are being assessed and will follow. This expansion---described as ``a first step towards a comprehensive description of the human interactome'' (EMBL-EBI, 2026)---directly motivates extending ET-miner to protein-protein interactions.",
    r"The AlphaFold database expanded in March 2026 to include predicted protein complexes~\cite{afdb2026news,han2026afdb}: 1.7 million high-confidence homodimer predictions have been added to the database, with an additional 18 million lower-confidence homodimers available via bulk download, and an update of 19 May 2026 added almost 80,000 high-confidence heterodimer predictions, with a further 8.1 million lower-confidence heterodimers available for bulk download~\cite{afdb2026news}. This expansion---described by Janet Thornton as ``a first step towards a comprehensive description of the human interactome''~\cite{afdb2026news}---directly motivates extending ET-miner to protein-protein interactions.",
)
rep(
    "\\end{thebibliography}",
    "\\bibitem{afdb2026news}\n"
    "EMBL-EBI.\n"
    "\\newblock Millions of protein complexes added to AlphaFold Database shed light on how proteins interact.\n"
    "\\newblock News release, 16 March 2026, updated 19 May 2026. \\url{https://www.ebi.ac.uk/about/news/technology-and-innovation/first-complexes-alphafold-database/} (accessed 4 September 2026).\n\n"
    "\\bibitem{han2026afdb}\n"
    "Y.~Han et al.\n"
    "\\newblock AlphaFold Database expands to proteome-scale quaternary structures.\n"
    "\\newblock \\emph{bioRxiv}, 2026. doi:10.64898/2026.03.27.714458.\n\n"
    "\\end{thebibliography}",
)

# varadi2024: the bibliography truncated the title.
rep(
    r"\newblock AlphaFold Protein Structure Database in 2024.",
    r"\newblock AlphaFold Protein Structure Database in 2024: providing structure coverage for over 214 million protein sequences.",
)

# --- Group D: GPU frequent itemset mining systems ------------------------------------------------

# zhang2011 keeps its bitsets GPU-resident and moves only candidates and supports per level, chon2018 could not be
# retrieved, djenouri2019 is a single-scan method whose data movement is not retrievable: none supports the
# contrast drawn here, and "orders of magnitude less" has no artifact.
rep(
    r"Because bitvectors remain GPU-resident, the per-level PCIe traffic is limited to prefix group metadata and result indices---orders of magnitude less than conventional approaches that transfer full candidate sets and support arrays each iteration~\cite{zhang2011,chon2018,djenouri2019}. Per-level transfer details",
    r"Because bitvectors remain GPU-resident, the per-level PCIe traffic is limited to prefix group metadata and result indices. Per-level transfer details",
)
rep(
    r"while the 9.6\,GB bitvector matrix is uploaded once and never re-transferred---in contrast to conventional GPU FIM implementations that transfer transaction bitmaps, candidate sets and count arrays at each iteration~\cite{zhang2011,chon2018,djenouri2019}.",
    r"while the 9.6\,GB bitvector matrix is uploaded once and never re-transferred.",
)
rep(
    r"with only lightweight metadata (prefix groups and result indices) crossing the PCIe bus, versus full candidate and support array transfers in conventional approaches.",
    r"with only lightweight metadata (prefix groups and result indices) crossing the PCIe bus.",
)

# \bibliographystyle has no effect on a hand-written thebibliography; numbering follows \bibitem order,
# which reorder_bibliography.py sets to first-citation order.
rep("\\bibliographystyle{unsrt}\n\n", "")

assert tex != orig
open(path, "w", encoding="utf-8").write(tex)
print("pass 9 applied:", len(orig.splitlines()), "->", len(tex.splitlines()), "lines")
