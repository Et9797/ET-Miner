"""V2 pass 4: reviewer fixes and the appendix rewrite for the code path actually executed.

Exact-string replacements; each anchor must occur exactly once or the script aborts.

Usage:
    python revise_tex_v2_pass4.py <path-to-tex>
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


def rep_block(start, end, new):
    global tex
    assert tex.count(start) == 1, f"start anchor not unique: {start[:70]!r}"
    i = tex.index(start)
    j = tex.index(end, i)
    tex = tex[:i] + new + tex[j + len(end):]


# introduction
rep(r"far exceeding even the 80\,GB capacity of the highest-end GPUs.", r"far exceeding the 24\,GB of the GPU used here.")

# methods 2.1
rep(
    r"We processed 205,620,298 proteins from UniProt TrEMBL and obtained protein annotations (Pfam domain assignments, Gene Ontology (GO) terms, and predicted local distance difference test (pLDDT) confidence scores) from the UniProt~\cite{uniprot2023} cross-reference pipeline (TrEMBL release 2026\_01, 202,556,314 entries, retrieved from the UniProt archive on 2 September 2026).",
    r"We processed the 205,620,298 AlphaFold DB entries (of 214,683,829) whose mean predicted local distance difference test (pLDDT) confidence is at least 50 and obtained their annotations (Pfam domain assignments and Gene Ontology (GO) terms) from the UniProt~\cite{uniprot2023} cross-reference pipeline (TrEMBL release 2026\_01, 202,556,314 entries, retrieved from the UniProt archive on 2 September 2026).",
)
rep(
    r"(three on the mean pLDDT and three on the fraction of high-confidence residues, of which only the mean-pLDDT bins 50--90 and ${>}90$ are populated under the pLDDT${\geq}50$ filter)",
    r"(three on the mean pLDDT and three on the fraction of high-confidence residues; the low mean-pLDDT bin is removed by the pLDDT${\geq}50$ filter and the three fraction bins are not assigned by the metadata-based extraction, which sees only the per-protein mean, so only the mean-pLDDT bins 50--90 and ${>}90$ are populated)",
)
rep(
    r"and built the transactions in 9.3\,minutes (162,865 proteins/s);",
    r"and built the transactions in 9.3\,minutes of extractor time (10.4\,minutes wall-clock; 162,865 proteins/s);",
)

# methods 2.2 execution paths
rep(
    r"ET-miner implements a GPU-accelerated Apriori algorithm with three execution paths: (1)~a CPU path using boolean matrices with vectorized column operations, (2)~a GPU path with direct CSR-to-GPU bitvector conversion, and (3)~a GPU-accelerated path where bitvectors remain GPU-resident across all $K$-levels, with only lightweight metadata crossing the PCIe bus. The GPU-accelerated path is the primary contribution and was used for all results reported in this paper: on one GPU for the mining campaign, and in its row-split form (transactions partitioned across two GPUs, partial counts summed per candidate) for the permutation null model and the per-level itemset exports.",
    r"ET-miner implements a GPU-accelerated Apriori algorithm with a CPU path (boolean matrices with vectorized column operations, or a sparse variant) and a GPU path that builds bitvectors on-GPU directly from CSR and keeps them GPU-resident across all $K$-levels, with only lightweight metadata crossing the PCIe bus; the GPU path has single-GPU, multi-GPU (candidate-split or row-split) and sparse-CSR variants. The GPU path is the primary contribution and was used for all results reported in this paper: the single-GPU variant for the mining campaign, and the row-split variant (transactions partitioned across two GPUs, partial counts summed per candidate) for the permutation null model and the per-level itemset exports.",
)

# methods 2.3 / 2.4
rep(
    r"The bit-packed dense form is smaller (${\sim}26$\,GB) but still exceeds the CSR footprint.",
    r"The bit-packed dense form is smaller (${\sim}26$\,GB for the full set, 9.6\,GB for the mining subset) but still exceeds the CSR footprint.",
)
rep(
    "\\subsection{GPU-Resident Bitvector Representation}\n",
    "\\subsection{GPU-Resident Bitvector Representation}\n\\label{sec:bitvec}\n",
)
rep(
    r"built directly on-GPU from the CSR column indices via a single host-to-device transfer (${\sim}2.5$\,GB).",
    r"built directly on-GPU from the CSR arrays after a one-time host-to-device copy (${\sim}3.1$\,GB of 64-bit row pointers and column indices).",
)

# results: SON identity wording
rep(
    r"returned exactly the same itemsets in all three cases (5,305; 113,405; 475,865; in 779.6\,s, 1,426.0\,s and 5,673.7\,s)",
    r"returned the same itemsets in all three cases (5,305; 113,405; 475,865; set identity verified against the exhaustive tables at Base and Super, identical count and $K$-distribution at Power; in 779.6\,s, 1,426.0\,s and 5,673.7\,s)",
)

# Table 4 item id
rep(r"& plddt\_mean & Medium confidence (mean pLDDT 50--90) \\", r"& plddt\_mean\_med & Medium confidence (mean pLDDT 50--90) \\")

# discussion 4.2
rep(
    r"Once the bitvector matrix is built on-GPU, the entire Apriori algorithm (candidate generation, support counting, filtering, and sorting) executes without returning data to the CPU. The only per-iteration communication is a single integer indicating how many patterns survived (${\sim}12$\,bytes).",
    r"Once the bitvector matrix is built on-GPU, candidate generation, support counting and threshold filtering execute on the device without touching the database again. The per-iteration communication is limited to the prefix-group arrays of the previous level (uploaded) and the surviving (index, count) records of the current level (downloaded, 12\,bytes each), both proportional to the number of frequent itemsets rather than to the database.",
)
rep(
    r"transferred to the GPU once (${\sim}2.5$\,GB of CSR column indices versus ${\sim}10$\,GB",
    r"transferred to the GPU once (${\sim}3.1$\,GB of CSR arrays versus ${\sim}10$\,GB",
)

# data availability: erratum pointer
rep(
    r"is deposited in the same repository under \texttt{runs/20260902T0000Z/}.",
    r"is deposited in the same repository under \texttt{runs/20260902T0000Z/}. The values of the preprint that this version supersedes (hardware, timings, streaming-path counts and five-permutation statistics) are itemized in \texttt{COMPARISON\_REPORT.md} at the repository root.",
)

# appendix B: pseudocode
rep(
    r"\State $G \gets \text{BUILD\_PREFIX\_GROUPS\_GPU}(\text{freq}_{k-1})$",
    r"\State $G \gets \text{BUILD\_PREFIX\_GROUPS}(\text{freq}_{k-1})$ \Comment{CPU; uploaded}",
)
rep(
    "    \\State $\\text{results.append}(\\text{freq}_k)$\n",
    "    \\State $\\text{results.append}(\\text{freq}_k)$ \\Comment{downloaded per level}\n",
)
rep(
    r"\State $F \gets \text{BULK\_TRANSFER}(\text{results})$ \Comment{Single transfer}",
    r"\State $F \gets \text{ASSEMBLE}(\text{results})$ \Comment{host}",
)

# appendix C
rep(r"The \texttt{popcount} operation in Section~2.4 maps", r"The \texttt{popcount} operation in Section~\ref{sec:bitvec} maps")
rep(
    r"For larger combinations ($K{\geq}3$), candidates sharing a common prefix are grouped via boundary detection on-GPU, and each group is evaluated within a single kernel, with early termination when an intermediate AND result is zero---meaning no protein carries all features in the candidate set.",
    r"For larger combinations ($K{\geq}3$), candidates sharing a common prefix are grouped on the CPU from the previous level's frequent itemsets and uploaded as compact group arrays; each group is evaluated within a single kernel that generates its candidates in-kernel, counts them, and applies the threshold with an atomic compaction, with a warp-level early exit that skips the remaining features of a 32-word window (2,048 proteins) once the running AND is zero in all of its lanes.",
)
rep_block(
    r"    \item $K{=}1$: Popcount of each bitvector yields frequent item counts, stored in GPU memory. Transfer: ${\sim}12$\,bytes (frequent item count and loop control).",
    r"\noindent The total data transfer across all 22 $K$-levels is ${\sim}264$~bytes---compared to gigabytes in conventional GPU FIM implementations that transfer candidate sets and count arrays at each iteration~\cite{zhang2011,chon2018,djenouri2019}.",
    r"""    \item $K{=}1$: Popcount of each bitvector on the GPU; the column counts are downloaded (8\,bytes per item) and thresholded on the host.
    \item $K{=}2$: The frequent-item list is uploaded; the fused pair kernel writes only surviving pairs, which are downloaded as (index, count) records of 12\,bytes each and decoded on the host.
    \item $K{\geq}3$: Prefix groups are built on the host from the previous level's frequent itemsets and uploaded as compact arrays (proportional to $|F_{k-1}|$). The fused kernel generates, counts, and filters candidates; survivors are downloaded as 12-byte (index, count) records, decoded and sorted on the host.
\end{itemize}

\noindent The per-level traffic is therefore proportional to the number of frequent itemsets of adjacent levels (12\,bytes per frequent itemset downloaded, 0.32\,GB for the 26.8 million itemsets of the Opus run), while the 9.6\,GB bitvector matrix is uploaded once and never re-transferred---in contrast to conventional GPU FIM implementations that transfer transaction bitmaps, candidate sets and count arrays at each iteration~\cite{zhang2011,chon2018,djenouri2019}.""",
)

# Table 7
rep(r"Result sorting       & CPU & CPU & \textbf{GPU (\texttt{lexsort})} \\", r"Result sorting       & CPU & CPU & CPU (after download) \\")
rep(r"Early termination    & No & Unknown & \textbf{Yes} (skip when AND $= 0$) \\", r"Early termination    & No & Unknown & \textbf{Warp-level} (skip window when AND $= 0$) \\")
rep(
    r"PCIe per $K$-level   & $O(|TB| + |C_L| + |PS|)$ & $O(\text{bit array blocks})$ & \textbf{${\sim}12$\,bytes} \\",
    r"PCIe per $K$-level   & $O(|TB| + |C_L| + |PS|)$ & $O(\text{bit array blocks})$ & \textbf{$O(|F_{k-1}|)$ up, $12\,\text{B}\times|F_k|$ down} \\",
)
rep(r"Multi-GPU auto-dispatch & Manual & Manual & \textbf{Automatic (memory-aware)} \\", r"Multi-GPU auto-dispatch & Manual & Manual & \textbf{Candidate split by candidate count; row split explicit} \\")

# Table 8
rep(
    r"with CPU involvement limited to loop control ($\sim$4 bytes per $K$-level).}",
    r"with CPU involvement limited to prefix-group construction, decoding and sorting of the survivors, and loop control.}",
)
rep(r"CPU (CSR) $\to$ GPU (\texttt{atomicOr}) & Once ($\sim$2.5\,GB) \\", r"CPU (CSR) $\to$ GPU (\texttt{atomicOr}) & Once ($\sim$3.1\,GB) \\")
rep(r"$K{=}1$ frequency filter  & CPU & GPU (\texttt{cp.where}) & None \\", r"$K{=}1$ frequency filter  & CPU & CPU (after an 8-byte-per-item download) & $8\,\text{B}\times|F_1|$ \\")
rep(r"$K{=}2$ candidate gen     & CPU (enumerate pairs) & GPU (triangular number inverse) & None \\", r"$K{=}2$ candidate gen     & CPU (enumerate pairs) & GPU (triangular number inverse) & frequent items up \\")
rep(r"$K{=}2$ min-support filter & CPU & GPU (\texttt{atomicAdd} sparse output) & None \\", r"$K{=}2$ min-support filter & CPU & GPU (\texttt{atomicAdd} sparse output) & $12\,\text{B}\times|F_2|$ down \\")
rep(r"$K{\geq}3$ candidate gen  & CPU (prefix tree) & GPU (binary search + triangular inverse) & None \\", r"$K{\geq}3$ candidate gen  & CPU (prefix tree) & CPU prefix groups $\to$ GPU (triangular inverse) & $O(|F_{k-1}|)$ up \\")
rep(r"$K{\geq}3$ filter + sort  & CPU & GPU (\texttt{atomicAdd} + \texttt{lexsort}) & None \\", r"$K{\geq}3$ filter + sort  & CPU & GPU filter (\texttt{atomicAdd}); CPU sort & $12\,\text{B}\times|F_k|$ down \\")
rep(r"Loop control              & CPU & CPU reads \texttt{n\_results} & $\sim$4 bytes \\", r"Loop control              & CPU & CPU reads \texttt{n\_results} & 8 bytes \\")
rep(r"Result collection         & CPU (per iteration) & GPU $\to$ CPU (bulk, once) & Once at end \\", r"Result collection         & CPU (per iteration) & GPU $\to$ CPU per level & included above \\")

assert tex != orig
open(path, "w", encoding="utf-8").write(tex)
print("ok pass4:", len(orig), "->", len(tex), "chars")
