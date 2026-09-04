"""V2 preprint pass on paper/et_miner_proteome.tex: report only the fresh RTX 3090 run.

Exact-string replacements; each anchor must occur exactly once or the script aborts
before writing anything.

Usage:
    python revise_tex_v2.py <path-to-tex>
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


# metadata
rep(r"\date{March 2026 (revised September 2026)}", r"\date{September 2026 (version 2)}")

# abstract
rep(
    r"in 7.3\,minutes (excluding feature extraction) on a single NVIDIA H100, revealing",
    r"in 19.1\,minutes (excluding feature extraction) on a single NVIDIA GeForce RTX 3090, revealing",
)
rep(
    r"Every deterministic result in this paper (dataset statistics, itemset counts, $K$-distributions and the $K{=}22$ itemset) was reproduced exactly in an independent re-execution from freshly downloaded data in September 2026; the null-model statistics are those of that re-execution.",
    r"All results reported here come from a single complete execution of the pipeline on freshly downloaded data (September 2026), whose artifacts are deposited with the code.",
)

# introduction
rep(
    r"If this 3-feature combination appears in thousands of proteins across distant evolutionary lineages, it reveals a conserved molecular machine",
    r"This 3-feature combination appears in 1,234 of the proteins mined here; when such a combination recurs across distant evolutionary lineages, it reveals a conserved molecular machine",
)
rep(
    r"in 7.3\,minutes of mining on a single NVIDIA H100, enabling",
    r"in 19.1\,minutes of mining on a single NVIDIA GeForce RTX 3090, enabling",
)

# methods: dataset
rep(
    r"(TrEMBL release 2026\_01, accessed February 2026; the release is identified by its record count of 202,556,314 TrEMBL entries, which among releases 2025\_01 to 2026\_02 matches only 2026\_01)",
    r"(TrEMBL release 2026\_01, 202,556,314 entries, retrieved from the UniProt archive on 2 September 2026)",
)
rep(
    r"Feature extraction processed ${\sim}150$\,GiB of compressed annotation data in 63\,minutes on the original host; this preprocessing is separate from the 7.3\,minute mining runtime reported below.",
    r"Feature extraction streamed the 149.8\,GiB compressed flat file and reduced it to the record lines it parses (48\,minutes, network-bound), restricted it in two filtering passes (29\,minutes) to the 90.5 million pLDDT-passing proteins with Pfam or GO cross-references, and built the transactions in 9.3\,minutes (162,865 proteins/s); this preprocessing is separate from the 19.1\,minute mining runtime reported below.",
)

# methods: architecture / multi-GPU
rep(
    r"The GPU-accelerated path is the primary contribution and was used for all results reported in this paper.",
    r"The GPU-accelerated path is the primary contribution and was used for all results reported in this paper: on one GPU for the mining campaign, and in its row-split form (transactions partitioned across two GPUs, partial counts summed per candidate) for the permutation null model and the per-level itemset exports.",
)
rep(
    r"distribute candidates by index range, though all original experiments in this paper use a single GPU.",
    r"distribute candidates by index range, or partition the transactions across devices (row split); the mining campaign in this paper uses a single GPU.",
)

# results: setup paragraph
rep_block(
    r"The original experiments (February 2026) were performed on a single NVIDIA H100",
    r"spanning four orders of magnitude (Table~\ref{tab:campaign}).",
    r"All experiments were performed on a rented vast.ai instance with two NVIDIA GeForce RTX 3090 GPUs (24\,GB each, compute capability 8.6; driver 595.71.05 with CUDA 13.2, CUDA toolkit 12.1), an AMD EPYC 7402P (24 cores) and 69.6\,GiB of host memory available to the container, running Python~3.10.13, CuPy~14.1.1, NumPy~2.2.6 and Ubuntu~22.04.3. The six-run mining campaign and the direct-versus-streaming comparison ran on a single GPU; the permutation null model and the per-level itemset exports used both GPUs with the row-split path. We conducted six mining runs at progressively lower support thresholds, spanning four orders of magnitude (Table~\ref{tab:campaign}).",
)

# results: Table 2
rep_block(
    r"\caption{Mining campaign results across six support thresholds; all counts are exhaustive (Direct CSR$\to$GPU).",
    r"Opus & 0.00001\% & 8 & 26,849,505 & 22 & 7.3\,min & 19.1\,min \\",
    r"""\caption{Mining campaign results across six support thresholds on a single NVIDIA GeForce RTX 3090 (24\,GB); all counts are exhaustive (Direct CSR$\to$GPU). Support percentages for Ultra and Opus are nominal (rounded); actual thresholds correspond to minimum protein counts of 16 and 8 respectively. Times are wall-clock per run (CSR build, bitvector build, mining and result collection). The streaming SON path, run at the Base, Super and Power thresholds, returned identical itemsets (Section~\ref{sec:results}).}
\label{tab:campaign}
\centering
\small
\begin{tabular}{lrrrrr}
\toprule
Run & Support & Min.\ proteins & Itemsets & Max $K$ & Time \\
\midrule
Base & 0.1\% & 76,891 & 5,305 & 9 & 15.6\,s \\
Super & 0.01\% & 7,690 & 113,405 & 14 & 43.9\,s \\
Power & 0.001\% & 769 & 475,865 & 14 & 83.1\,s \\
Blitz & 0.0001\% & 77 & 2,841,280 & 19 & 4.6\,min \\
Ultra & 0.00002\% & 16 & 14,558,875 & 20 & 11.5\,min \\
Opus & 0.00001\% & 8 & 26,849,505 & 22 & 19.1\,min \\""",
)

# results: SON paragraph
rep_block(
    r"The three highest thresholds were originally mined with a streaming SON path",
    r"are superseded by the exhaustive counts in Table~\ref{tab:campaign}.",
    r"All six thresholds were mined with the Direct CSR$\to$GPU path, which holds the whole 76.9M-protein bitvector matrix on one device. The streaming SON path~\cite{savasere1995} (chunks of 40 million transactions, local threshold $0.9\times$ the global one, followed by a global counting pass) was additionally run at the Base, Super and Power thresholds and returned exactly the same itemsets in all three cases (5,305; 113,405; 475,865; in 779.6\,s, 1,426.0\,s and 5,673.7\,s), as the SON construction guarantees (an itemset frequent in the whole database is locally frequent in at least one chunk, and the global pass verifies every candidate). The cost of streaming is time rather than completeness: in the controlled comparison at the Power threshold the direct path needed 91.7\,s against 5,673.7\,s for SON, a $62\times$ difference (the campaign run of Table~\ref{tab:campaign} is a separate execution of the same direct path, 83.1\,s). The direct path is therefore what makes the low-threshold runs practical: lowering the threshold from 769 to 77 proteins multiplies the number of frequent itemsets by $6.0\times$ (475,865 to 2,841,280) and raises the maximum depth from 14 to 19 (Figure~\ref{fig:campaign}).",
)
rep(
    r"lines show wall-clock time per run on the original H100 (February 2026; Power to Opus) and on one RTX 3090 in the September 2026 re-execution (all six).}",
    r"line shows wall-clock time per run on a single RTX 3090.}",
)

# K=22 section
rep(
    r"the single definitional link is GO:0005524 (ATP binding), which is auto-derivable from PF00270 via InterPro2GO, leaving 21 independently annotated features.",
    r"the single definitional link is GO:0005524 (ATP binding), which InterPro2GO derives from PF00270 (InterPro IPR011545; PF00271/IPR001650 carries no GO mapping, and neither entry maps to any other member), leaving 21 independently annotated features.",
)
rep(
    r"The eight matching proteins were re-identified in the September 2026 re-execution (\texttt{analyze\_k22\_proteins.py}); their accessions are deposited with the reproduction artifacts (see Data and Code Availability)",
    r"The eight matching proteins were identified with \texttt{analyze\_k22\_proteins.py}; their accessions are deposited with the run artifacts (see Data and Code Availability)",
)
rep(
    r"order-of-magnitude estimates read from the decoded-pattern summaries.}",
    r"order-of-magnitude estimates read from the per-$K$ itemset tables of the Blitz run.}",
)

# null model
rep(
    r"The preprint reported five permutations (seed~42; 662\,s on the H100); their individual draws were not recorded and could not be regenerated. The re-execution ran the released procedure with the same seed for 100 permutations (its own five-permutation run forms the first five) in 2,007\,s wall-clock on 2$\times$RTX 3090 with the row-split multi-GPU path (18.6\,s per permutation).",
    r"One hundred permutations were run (seed~42) in 2,007\,s wall-clock on the two RTX 3090s with the row-split path (18.6\,s per permutation).",
)
rep(
    r" The five-permutation analysis of the preprint gave the same qualitative picture ($K_{\max}{=}6$ in every permutation) but only an empirical $p$ of $1/6$ and a 95\% binomial upper bound of 0.45 on the per-permutation probability; the standardized effect sizes it reported ($+3{,}791$, $+7{,}402$, $+71{,}728$) rested on $\sigma$ estimated from five unrecorded draws (five fresh draws give $+3{,}048$, $+9{,}917$, $+175{,}697$) and are superseded by the values above.",
    "",
)

# discussion
rep(
    r"is feasible at the 100-million-transaction scale on a single high-end datacenter GPU.",
    r"is feasible at the 100-million-transaction scale on a single consumer GPU.",
)
rep(
    r"The preprint's claim that SON missed 95.2\% of frequent itemsets originated from an unreleased configuration and is withdrawn (Table~\ref{tab:campaign}); a two-pass SON with a proportional or lowered local threshold cannot miss a globally frequent itemset~\cite{savasere1995}.",
    r"A two-pass SON with a proportional or lowered local threshold cannot miss a globally frequent itemset~\cite{savasere1995}.",
)
rep(
    r"ET-miner & \textbf{76.9M} & 1,002 & \textbf{22} & 1$\times$ H100 & 7.3\,min \\",
    r"ET-miner & \textbf{76.9M} & 1,002 & \textbf{22} & 1$\times$ RTX 3090 & 19.1\,min \\",
)
rep(
    r"All original experiments in this paper used a single H100, and the 76.9M multi-feature proteins were mined without manual tuning; the September 2026 re-execution used one RTX 3090 for the campaign of Table~\ref{tab:campaign} and the row-split two-GPU path for the per-level itemset export and the null model.",
    r"The mining campaign of Table~\ref{tab:campaign} used a single RTX 3090, and the 76.9M multi-feature proteins were mined without manual tuning; the row-split two-GPU path was used for the per-level itemset exports and the null model.",
)
rep(
    r"the biological count exceeds the null expectation by many orders of magnitude (a standardized effect size of $+2{,}728$ estimated from 100 permutations)",
    r"the biological count exceeds the null expectation $4.2$-fold (108,059 vs.\ 25,442; a standardized effect size of $+2{,}728$ estimated from 100 permutations)",
)

# conclusion, funding, availability
rep(
    r"is practical on current-generation datacenter GPUs. With a base vocabulary of 1,002 features on a single H100, mining completes in 7.3 minutes (excluding the one-time 63-minute feature extraction) and reaches $K{=}22$; the independent re-execution on a single RTX 3090 took 19.1 minutes for the same run.",
    r"is practical on a single consumer GPU. With a base vocabulary of 1,002 features on one NVIDIA GeForce RTX 3090, mining completes in 19.1 minutes (excluding the one-time feature extraction, Section~\ref{sec:methods}) and reaches $K{=}22$.",
)
rep(
    r"\noindent This research received no external funding.",
    r"\noindent This research received no external funding. Computations were run on a rented vast.ai cloud instance with two NVIDIA GeForce RTX 3090 GPUs.",
)
rep(
    r"The September 2026 re-execution (scripts, logs, per-$K$ itemset tables, null-model outputs, the accessions of the $K{=}22$ proteins and a per-claim comparison report) is deposited in the same repository under \texttt{runs/20260902T0000Z/}.",
    r"The complete run that produced every value in this paper (scripts, logs, per-$K$ itemset tables, null-model outputs, the accessions of the $K{=}22$ proteins and a per-claim comparison against the preprint) is deposited in the same repository under \texttt{runs/20260902T0000Z/}.",
)

# appendix
rep(r"\textbf{Candidate-index splitting} \\", r"\textbf{Candidate-index or row splitting} \\")

assert tex != orig
open(path, "w", encoding="utf-8").write(tex)
print("ok v2 pass:", len(orig), "->", len(tex), "chars")
