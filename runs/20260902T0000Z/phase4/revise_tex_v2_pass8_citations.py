"""V2 pass 8: remove the tool-comparison material and state the pattern-count convention in Methods.

Removes every value that describes another system or tool (Introduction sentence on GPU FIM
speedups and dataset scale, Discussion 4.2 with Table 6, Appendix D Tables 7 and 8, the popcount
comparison in Appendix C.1), keeps the ET-miner-only memory table of Appendix D under a new title,
and adds the supporting-protein counting convention for the highlighted patterns to Methods.
Exact-string edits; each anchor must occur exactly once.

Usage:
    python revise_tex_v2_pass8_citations.py <path-to-tex>
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


def cut(start, end):
    """Delete the text from the unique anchor `start` up to (not including) the unique anchor `end`."""
    global tex
    assert tex.count(start) == 1, f"start anchor count != 1: {start[:80]!r}"
    assert tex.count(end) == 1, f"end anchor count != 1: {end[:80]!r}"
    i, j = tex.index(start), tex.index(end)
    assert i < j, "anchors out of order"
    tex = tex[:i] + tex[j:]


# 1. Introduction: the sentence exists only to carry other systems' speedups and dataset sizes.
rep(
    r" GPU-accelerated FIM systems~\cite{luna2019,zhang2011,chon2018,djenouri2019,acmsurvey2021} have achieved $50$--$350\times$ speedups over classical CPU algorithms such as Eclat~\cite{zaki2000,zaki1997} and FP-Growth~\cite{han2000}, but all existing GPU implementations were evaluated on datasets of at most a few million real transactions, one to two orders of magnitude smaller than our proteome dataset.",
    "",
)

# 2. Discussion 4.2 (prose, Table 6, the three-choice list and the pointer to Appendix D).
cut(r"\subsection{Comparison with Prior GPU FIM Systems}", r"\subsection{Comparison with Protein Domain Mining}")

# 3. Appendix D: drop Tables 7 and 8 with their lead-in sentences; keep the memory table.
cut(r"Table~\ref{tab:gpu-arch} compares the GPU-level architectural choices", r"Table~\ref{tab:memory-comparison} compares memory requirements.")
rep(
    "\\section{Detailed GPU Architecture Comparison}\n\\label{app:gpu-comparison}",
    "\\section{Memory Footprint of the Dense and CSR Representations}\n\\label{app:memory}",
)
rep(
    r"The bit-packed dense form is smaller (${\sim}26$\,GB for the full set, 9.6\,GB for the mining subset) but still exceeds the CSR footprint.",
    r"The bit-packed dense form is smaller (${\sim}26$\,GB for the full set, 9.6\,GB for the mining subset) but still exceeds the CSR footprint; Appendix~\ref{app:memory} tabulates both representations against density.",
)

# 4. Appendix C.1: unsourced cycle count and the popcount-width comparison with other systems.
rep(
    r"maps to the GPU's native \texttt{\_\_popcll} instruction---a hardware intrinsic that counts set bits in a 64-bit word in a single clock cycle. Each operation processes 64 proteins simultaneously, compared to 32 with the 32-bit \texttt{\_\_popc} used by prior systems~\cite{chon2018,chon2024}.",
    r"maps to the GPU's native \texttt{\_\_popcll} instruction---a hardware intrinsic that counts set bits in a 64-bit word. Each operation processes 64 proteins simultaneously.",
)

# 5. K=17: the matching itemset has support 611 exactly (RESULTS P-030).
rep(r"(${\sim}611$ proteins). Protein tyrosine kinase (PF07714)", r"(611 proteins). Protein tyrosine kinase (PF07714)")

# 6. Methods: the counting convention, stated once, with the footnote replaced by a pointer.
rep(
    "in Appendix~\\ref{app:complexity}.\n",
    "in Appendix~\\ref{app:complexity}.\n\n"
    "\\subsection{Reporting Convention for Highlighted Patterns}\n"
    "\\label{sec:reporting}\n\n"
    "Each intermediate-$K$ pattern highlighted in Section~\\ref{sec:intermediate} is reported with a supporting-protein count read from the per-$K$ itemset tables of the Blitz run (Table~\\ref{tab:campaign}). When the description names Pfam identifiers, the count is the largest support among the itemsets at that $K$-level that contain every named identifier, and the text states how many itemsets match. When the description names no identifier ($K{=}13$ and $K{=}11$), the count is the largest support among all itemsets at that $K$-level. In both cases the count is a maximum over itemsets rather than the support of one named itemset, unless exactly one itemset matches.\n",
)
rep(
    "\\subsection{Biological Discoveries at Intermediate $K$-Levels}\n",
    "\\subsection{Biological Discoveries at Intermediate $K$-Levels}\n\\label{sec:intermediate}\n",
)
rep(
    r"to demonstrate that the mining recovers known biology:",
    r"to demonstrate that the mining recovers known biology (supporting-protein counts follow the convention of Section~\ref{sec:reporting}):",
)
rep(
    r"(itemsets at this level are shared by up to 11,521 proteins\footnote{Supporting-protein counts are read from the per-$K$ itemset tables of the Blitz run. Where the pattern is identified by named domains, the count is the support of the matching itemsets; the $K{=}13$ and $K{=}11$ descriptions name no domain identifiers, so the maximum support at that level is given instead.})",
    r"(up to 11,521 proteins)",
)

assert tex != orig
open(path, "w", encoding="utf-8").write(tex)
print("pass 8 applied:", len(orig.splitlines()), "->", len(tex.splitlines()), "lines")
