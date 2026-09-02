# Claims extracted from `paper/et_miner_proteome.tex` (V1 preprint) — reproduction audit, phase 1

- Source file: `/root/projects/ET-Miner/paper/et_miner_proteome.tex` (896 lines, 65,595 bytes; read in full, lines 1–896).
- Extraction date: 2026-09-02. Extractor: Claude (automated). Scope: **extraction only** — no number below has been checked for correctness; every value may be wrong or fabricated in the source. Nothing here is a judgment of validity.
- The tex defines **no `\newcommand`/`\def` macros carrying numbers**; every number is literal in the source.
- The referenced figure files (`figures/concept_figure.pdf`, `figures/architecture.pdf`, `figures/mining_campaign.pdf`, `figures/k_distribution.pdf`) do **not** exist anywhere in the repository (no `paper/figures/` directory), so figure *contents* could not be transcribed; only their captions are covered.
- Canonical experiment for the audit ("base214m", base vocabulary over the full AlphaFold DB) corresponds in the paper to the **Opus** run (Table `tab:campaign`, line 230): 0.00001 % nominal support, min_count = 8, 26,849,505 itemsets, K = 22, 7.3 min, Direct CSR→GPU, single H100 — mined over the 76,890,945-protein multi-feature subset of 205,620,298 processed proteins (the paper's own numbers; see Section B).

Conventions used in Section A:
- `tex line` is the physical line number in the .tex file (tables/paragraphs are single long lines in this source, so many rows share a line).
- `category` ∈ {deterministic, hardware-dependent, method-parameter, external-fact, software}; `claim type` ∈ {result, parameter, setup, background}.
- Quoted context is verbatim LaTeX source (≤ 25 words); text in square brackets `[...]` after the quote is the extractor's cross-reference note, not part of the source. `\|` inside quotes is a markdown-escaped `|`.
- Every occurrence of a repeated value is its own row; cross-references cite tex line numbers (stable) rather than row IDs.
- Deliberately **not** tabulated (not claims): LaTeX layout values in the preamble (lines 1–77: `margin=1.8cm`, `columnsep=0.6cm`, `10pt`, `\tolerance=1500`, `\emergencystretch=1.5em`, `\hyphenpenalty=50`, title/caption spacing, `0.92\textwidth`, colour mixes); tabular column spec `p{4.2cm}` (line 293) and `\addlinespace[0.4em]` (line 887); bibliography metadata (lines 538–725: years, volumes, page ranges, `\begin{thebibliography}{37}`); citation years in table headers "GMiner (2018)" / "GMiner++ (2024)" (line 830); section cross-references such as "Section~2.4" (line 797); the placeholder itemset {A, B, C} in the support formula (lines 173–179); algorithm line numbering; and rhetorical enumerations of prose structure ("three innovations" l.120, "two compounding mechanisms" l.238, "Three findings" l.391, "two mechanisms" l.404, "two fundamental limitations" l.408, "three key choices" l.435, "two most closely related" l.821, "first step" l.486, "first systematic map" l.498). The three named drugs (imatinib, dasatinib, erlotinib; l.350) and the K=1/K=2/K≥3 level descriptions (lines 186–190) are non-numeric method prose and are covered in Section B instead.

---

## SECTION A — CLAIMS TABLE (543 rows, document order)

| ID | tex line | section | value | unit | category | claim type | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| T-001 | 83 | front matter | March 2026 | date (manuscript) | external-fact | setup | "\date{March 2026}" |
| T-002 | 91 | abstract | >200 million | predicted protein structures | external-fact | background | "The AlphaFold database contains over 200 million predicted protein structures" [cf. lines 101, 129] |
| T-003 | 91 | abstract | >200 | GB (dense boolean representation) | deterministic | setup | "a standard boolean representation requires over 200\,GB, exceeding GPU memory capacity" [cf. lines 105, 120, 167, 402: ~206 GB] |
| T-004 | 91 | abstract | 1,002 | features (base vocabulary) | deterministic | setup | "With a base vocabulary of 1,002 features" [cf. lines 105, 115, 134, 146, 171, 268, 376, 393, 425, 496] |
| T-005 | 91 | abstract | 76.9 million | proteins (transactions mined) | deterministic | setup | "this enables exhaustive Apriori computation across 76.9 million proteins in 7.3\,minutes (mining only) on a single NVIDIA H100" [cf. 152: 76,890,945] |
| T-006 | 91 | abstract | 7.3 | minutes (mining only) | hardware-dependent | result | "exhaustive Apriori computation across 76.9 million proteins in 7.3\,minutes (mining only) on a single NVIDIA H100" [cf. 120, 129, 230, 425, 496] |
| T-007 | 91 | abstract | 1 × NVIDIA H100 | GPU count and model | hardware-dependent | setup | "in 7.3\,minutes (mining only) on a single NVIDIA H100" [cf. 213] |
| T-008 | 91 | abstract | 26.8 million | co-occurrence patterns (itemsets, Opus run) | deterministic | result | "revealing 26.8 million co-occurrence patterns reaching $K{=}22$" [cf. 230: 26,849,505] |
| T-009 | 91 | abstract | 22 | K (maximum itemset length) | deterministic | result | "revealing 26.8 million co-occurrence patterns reaching $K{=}22$" |
| T-010 | 91 | abstract | >3,700 | Z (standardized effect size, 5 permutations) | deterministic | result | "A permutation-based null model supports biological significance ($Z{>}3{,}700$ for $K{=}4$--$6$ at 0.001\% support" [cf. 379-381, 498] |
| T-011 | 91 | abstract | 4–6 | K range with Z>3,700 | deterministic | result | "($Z{>}3{,}700$ for $K{=}4$--$6$ at 0.001\% support; see Section~\ref{sec:null-model} for scope)" |
| T-012 | 91 | abstract | 0.001 | % support (null-model threshold) | method-parameter | parameter | "($Z{>}3{,}700$ for $K{=}4$--$6$ at 0.001\% support; see Section~\ref{sec:null-model} for scope)" [cf. 226, 362] |
| T-013 | 101 | intro | >200 million | proteins with predicted structures | external-fact | background | "With predicted structures for over 200 million proteins spanning virtually all known organisms" |
| T-014 | 101 | intro | 2 (pairwise) | K of prior domain co-occurrence work | external-fact | background | "Protein domain co-occurrence has been studied primarily through pairwise network approaches~\cite{wang2011,coin2009}" |
| T-015 | 103 | intro | hundreds of millions | proteins | external-fact | background | "which combinations of protein features appear together more often than expected across hundreds of millions of proteins?" |
| T-016 | 103 | intro | 3 | features in illustrative combination (PF00270, GO:0003724, GO:0045087) | external-fact | background | "A protein annotated with a DEAD-box helicase domain (PF00270), RNA helicase activity (GO:0003724), and innate immune response (GO:0045087)" [illustrative example] |
| T-017 | 103 | intro | thousands | proteins (hypothetical support of example) | external-fact | background | "If this 3-feature combination appears in thousands of proteins across distant evolutionary lineages, it reveals a conserved molecular machine" [hypothetical] |
| T-018 | 105 | intro | 50–350× | speedup (prior GPU FIM vs classical CPU algorithms) | external-fact | background | "GPU-accelerated FIM systems~\cite{...} have achieved $50$--$350\times$ speedups over classical CPU algorithms such as Eclat" |
| T-019 | 105 | intro | a few million | transactions (largest prior GPU FIM datasets) | external-fact | background | "all existing GPU implementations were evaluated on datasets of at most a few million transactions" [cf. 408] |
| T-020 | 105 | intro | 3 | orders of magnitude (prior datasets smaller than proteome) | external-fact | background | "three orders of magnitude smaller than our proteome dataset" [cf. 408] |
| T-021 | 105 | intro | ~206 | GB (dense boolean matrix, full set) | deterministic | setup | "The standard dense boolean matrix representation requires ${\sim}206$\,GB for 205.6~million proteins with 1,002 features (one byte per boolean entry)" |
| T-022 | 105 | intro | 205.6 million | proteins (full processed set) | deterministic | setup | "requires ${\sim}206$\,GB for 205.6~million proteins with 1,002 features" [cf. 129: 205,620,298] |
| T-023 | 105 | intro | 1,002 | features | deterministic | setup | "for 205.6~million proteins with 1,002 features (one byte per boolean entry)" |
| T-024 | 105 | intro | 1 | byte per boolean entry (dense representation assumption) | method-parameter | setup | "(one byte per boolean entry)" [cf. 167] |
| T-025 | 105 | intro | 80 | GB (capacity of highest-end GPUs) | external-fact | background | "far exceeding even the 80\,GB capacity of the highest-end GPUs" [cf. 213, 216] |
| T-026 | 114 | figure fig:concept caption | 4 | K (schematic pattern) | deterministic | background | "A $K{=}4$ pattern (yellow highlight) is shared by three proteins." [illustrative schematic] |
| T-027 | 114 | figure fig:concept caption | 3 | proteins sharing schematic pattern | deterministic | background | "A $K{=}4$ pattern (yellow highlight) is shared by three proteins." [illustrative schematic] |
| T-028 | 115 | figure fig:concept caption | 76.9M | proteins (boolean matrix rows) | deterministic | setup | "The transaction database as a boolean matrix (76.9M proteins $\times$ 1,002 features)." |
| T-029 | 115 | figure fig:concept caption | 1,002 | features (boolean matrix columns) | deterministic | setup | "The transaction database as a boolean matrix (76.9M proteins $\times$ 1,002 features)." |
| T-030 | 120 | intro | 206 | GB (dense, before CSR) | deterministic | setup | "a compressed sparse row (CSR) representation constructed directly from the transaction list, reducing 206\,GB to ${\sim}5.1$\,GB" |
| T-031 | 120 | intro | ~5.1 | GB (CSR representation) | deterministic | result | "reducing 206\,GB to ${\sim}5.1$\,GB" [cf. 167, 402] |
| T-032 | 120 | intro | 76.9 million | multi-feature proteins | deterministic | setup | "Applied to 76.9~million multi-feature proteins, ET-miner discovers the complete feature co-occurrence landscape in 7.3\,minutes of mining" |
| T-033 | 120 | intro | 7.3 | minutes (mining) | hardware-dependent | result | "discovers the complete feature co-occurrence landscape in 7.3\,minutes of mining on a single NVIDIA H100" |
| T-034 | 120 | intro | 1 × NVIDIA H100 | GPU count and model | hardware-dependent | setup | "in 7.3\,minutes of mining on a single NVIDIA H100" |
| T-035 | 129 | methods | 214 million | proteins (AlphaFold DB coverage) | external-fact | background | "The AlphaFold Protein Structure Database~\cite{varadi2024} provides predicted structures for 214 million proteins across UniProt." [cf. 878, 885] |
| T-036 | 129 | methods | 205,620,298 | proteins processed (UniProt TrEMBL) | deterministic | setup | "We processed 205,620,298 proteins from UniProt TrEMBL and obtained protein annotations" [cf. 152] |
| T-037 | 129 | methods | 2025_01 | UniProt release (data version) | software | setup | "from the UniProt~\cite{uniprot2023} cross-reference pipeline (release 2025\_01, accessed February 2026)" [cf. 478, 529] |
| T-038 | 129 | methods | February 2026 | data access date | software | setup | "(release 2025\_01, accessed February 2026)" |
| T-039 | 129 | methods | 500 | Pfam domains (most frequent, vocabulary) | method-parameter | parameter | "The feature vocabulary comprises the 500 most frequent Pfam domains and the 500 most frequent GO terms" |
| T-040 | 129 | methods | 500 | GO terms (most frequent, vocabulary) | method-parameter | parameter | "The feature vocabulary comprises the 500 most frequent Pfam domains and the 500 most frequent GO terms" |
| T-041 | 129 | methods | 24,291 | Pfam families observed across corpus | deterministic | result | "(drawn from 24{,}291 Pfam and 25{,}993 GO families observed across the corpus)" |
| T-042 | 129 | methods | 25,993 | GO families observed across corpus | deterministic | result | "(drawn from 24{,}291 Pfam and 25{,}993 GO families observed across the corpus)" |
| T-043 | 129 | methods | 6 | pLDDT confidence bins | method-parameter | parameter | "together with 6 pLDDT confidence bins, for 1{,}006 defined items" |
| T-044 | 129 | methods | 1,006 | defined items (500+500+6) | deterministic | setup | "together with 6 pLDDT confidence bins, for 1{,}006 defined items" |
| T-045 | 129 | methods | 8 | proteins (minimum support threshold, applied during mining) | method-parameter | parameter | "the minimum support threshold of 8 proteins is applied during \emph{mining}, not during vocabulary construction" |
| T-046 | 129 | methods | 150 | GB compressed annotation data (feature extraction input) | deterministic | setup | "Feature extraction processed 150\,GB of compressed annotation data in 63\,minutes" |
| T-047 | 129 | methods | 63 | minutes (feature extraction / preprocessing) | hardware-dependent | result | "Feature extraction processed 150\,GB of compressed annotation data in 63\,minutes" [cf. 496] |
| T-048 | 129 | methods | 7.3 | minutes (mining runtime) | hardware-dependent | result | "this preprocessing is separate from the 7.3\,minute mining runtime reported below" |
| T-049 | 134 | table tab:features caption | 1,006 | items defined | deterministic | setup | "Feature vocabulary composition: 1,006 items are defined (6 pLDDT confidence bins, 500 Pfam domains, 500 GO terms)" |
| T-050 | 134 | table tab:features caption | 6 | pLDDT confidence bins (defined) | method-parameter | parameter | "1,006 items are defined (6 pLDDT confidence bins, 500 Pfam domains, 500 GO terms)" |
| T-051 | 134 | table tab:features caption | 500 | Pfam domains (defined) | method-parameter | parameter | "1,006 items are defined (6 pLDDT confidence bins, 500 Pfam domains, 500 GO terms)" |
| T-052 | 134 | table tab:features caption | 500 | GO terms (defined) | method-parameter | parameter | "1,006 items are defined (6 pLDDT confidence bins, 500 Pfam domains, 500 GO terms)" |
| T-053 | 134 | table tab:features caption | 1,002 | items passing support threshold | deterministic | result | "and 1,002 pass the support threshold of 8 proteins (all 1,000 Pfam/GO items plus 2 pLDDT bins)" |
| T-054 | 134 | table tab:features caption | 8 | proteins (support threshold) | method-parameter | parameter | "1,002 pass the support threshold of 8 proteins" |
| T-055 | 134 | table tab:features caption | 1,000 | Pfam/GO items frequent | deterministic | result | "(all 1,000 Pfam/GO items plus 2 pLDDT bins)" |
| T-056 | 134 | table tab:features caption | 2 | pLDDT bins frequent | deterministic | result | "(all 1,000 Pfam/GO items plus 2 pLDDT bins)" |
| T-057 | 142 | table tab:features | 500 | Pfam domains, Defined | method-parameter | parameter | "Pfam domains & InterPro/Pfam & 500 & 500" |
| T-058 | 142 | table tab:features | 500 | Pfam domains, Frequent | deterministic | result | "Pfam domains & InterPro/Pfam & 500 & 500" |
| T-059 | 143 | table tab:features | 500 | GO terms, Defined | method-parameter | parameter | "GO terms & Gene Ontology & 500 & 500" |
| T-060 | 143 | table tab:features | 500 | GO terms, Frequent | deterministic | result | "GO terms & Gene Ontology & 500 & 500" |
| T-061 | 144 | table tab:features | 6 | pLDDT confidence bins, Defined | method-parameter | parameter | "pLDDT confidence bins & AlphaFold & 6 & 2" |
| T-062 | 144 | table tab:features | 2 | pLDDT confidence bins, Frequent | deterministic | result | "pLDDT confidence bins & AlphaFold & 6 & 2" |
| T-063 | 146 | table tab:features | 1,006 | Total, Defined | deterministic | setup | "\textbf{Total} & & \textbf{1,006} & \textbf{1,002}" |
| T-064 | 146 | table tab:features | 1,002 | Total, Frequent | deterministic | result | "\textbf{Total} & & \textbf{1,006} & \textbf{1,002}" |
| T-065 | 152 | methods | 205,620,298 | total proteins | deterministic | setup | "Of the 205,620,298 total proteins, 76,890,945 (37.4\%) have more than one annotated feature under this vocabulary" |
| T-066 | 152 | methods | 76,890,945 | proteins with >1 feature (transaction set mined) | deterministic | result | "Of the 205,620,298 total proteins, 76,890,945 (37.4\%) have more than one annotated feature under this vocabulary and form the transaction set" |
| T-067 | 152 | methods | 37.4 | % of total proteins with >1 feature | deterministic | result | "76,890,945 (37.4\%) have more than one annotated feature under this vocabulary" [cf. 480: 62.6% excluded] |
| T-068 | 152 | methods | >1 | annotated features (transaction inclusion filter) | method-parameter | parameter | "have more than one annotated feature under this vocabulary and form the transaction set mined in this work" [cf. 480] |
| T-069 | 156 | methods | 3 | execution paths (CPU; GPU direct CSR; GPU-accelerated resident) | software | setup | "ET-miner implements a GPU-accelerated Apriori algorithm with three execution paths: (1)~a CPU path ... (2)~a GPU path ... (3)~a GPU-accelerated path" |
| T-070 | 156 | methods | path (3) GPU-accelerated | execution path used for all reported results | software | setup | "The GPU-accelerated path is the primary contribution and was used for all results reported in this paper." |
| T-071 | 167 | methods | 76.9M | proteins (mining subset) | deterministic | setup | "For the 76.9M-protein mining subset, the resulting matrix contains 316~million non-zero entries" |
| T-072 | 167 | methods | 316 million | non-zero entries (CSR of mining subset) | deterministic | result | "the resulting matrix contains 316~million non-zero entries occupying ${\sim}5.1$\,GB in coordinate format (two 64-bit integers per entry)" |
| T-073 | 167 | methods | ~5.1 | GB (coordinate format, mining subset) | deterministic | result | "316~million non-zero entries occupying ${\sim}5.1$\,GB in coordinate format (two 64-bit integers per entry)" [cf. 120, 402; 171/440: ~3 GB; 886: 19 GB] |
| T-074 | 167 | methods | 2 × 64-bit integers (16 B) | per non-zero entry (COO encoding) | method-parameter | setup | "(two 64-bit integers per entry)" [cf. 878: 8 bytes/entry] |
| T-075 | 167 | methods | ~15× | reduction vs dense representation of same subset | deterministic | result | "a ${\sim}15\times$ reduction from the ${\sim}77$\,GB dense representation of that same subset (one byte per boolean entry)" [cf. 402] |
| T-076 | 167 | methods | ~77 | GB (dense representation of 76.9M subset) | deterministic | setup | "a ${\sim}15\times$ reduction from the ${\sim}77$\,GB dense representation of that same subset (one byte per boolean entry)" |
| T-077 | 167 | methods | 1 | byte per boolean entry | method-parameter | setup | "(one byte per boolean entry)" |
| T-078 | 167 | methods | ~206 | GB (dense matrix, full 205.6M set) | deterministic | setup | "relative to the ${\sim}206$\,GB dense matrix of the full 205.6M-protein set it is ${\sim}40\times$ smaller" |
| T-079 | 167 | methods | 205.6M | proteins (full set) | deterministic | setup | "relative to the ${\sim}206$\,GB dense matrix of the full 205.6M-protein set it is ${\sim}40\times$ smaller" |
| T-080 | 167 | methods | ~40× | smaller than full-set dense matrix | deterministic | result | "relative to the ${\sim}206$\,GB dense matrix of the full 205.6M-protein set it is ${\sim}40\times$ smaller" |
| T-081 | 167 | methods | ~26 | GB (bit-packed dense form) | deterministic | setup | "The bit-packed dense form is smaller (${\sim}26$\,GB) but still exceeds the CSR footprint." [cf. 171, 402, 769, 886] |
| T-082 | 171 | methods | 205.6M | proteins (bitvector rows) | deterministic | setup | "With 205.6M proteins and 1,002 features, the total bitvector matrix occupies ${\sim}26$\,GB of GPU memory" |
| T-083 | 171 | methods | 1,002 | features (bitvectors) | deterministic | setup | "With 205.6M proteins and 1,002 features, the total bitvector matrix occupies ${\sim}26$\,GB of GPU memory" |
| T-084 | 171 | methods | ~26 | GB GPU memory (bitvector matrix) | deterministic | result | "the total bitvector matrix occupies ${\sim}26$\,GB of GPU memory, built directly on-GPU from the CSR column indices" [cf. 886: 27 GB; 440: ~10 GB subset] |
| T-085 | 171 | methods | ~3 | GB (single host-to-device transfer of CSR column indices) | deterministic | result | "built directly on-GPU from the CSR column indices via a single host-to-device transfer (${\sim}3$\,GB)" [cf. 440, 859] |
| T-086 | 171 | methods | 1 | host-to-device transfer | deterministic | setup | "via a single host-to-device transfer (${\sim}3$\,GB)" |
| T-087 | 193 | methods | orders of magnitude less | per-level PCIe traffic vs conventional | deterministic | result | "the per-level PCIe traffic is limited to prefix group metadata and result indices---orders of magnitude less than conventional approaches" |
| T-088 | 196 | methods | once | bitvector load to GPU (count of loads) | deterministic | setup | "the algorithm loads bitvectors onto the GPU once, where they remain resident across all $K$-levels" |
| T-089 | 196 | methods | 1 | GPU (all experiments) | hardware-dependent | setup | "though all experiments in this paper use a single GPU" [cf. 446] |
| T-090 | 200 | methods | 64 | proteins per bitvector operation | deterministic | background | "The bitvector representation processes 64 proteins in a single operation" |
| T-091 | 201 | methods | 64× | speedup over element-wise comparison | deterministic | background | "providing a $64\times$ speedup over element-wise comparison" [cf. 743, 753] |
| T-092 | 213 | results | 1 | GPU (all experiments) | hardware-dependent | setup | "All experiments were performed on a single NVIDIA H100 80\,GB SXM5 with 128\,GB host RAM" |
| T-093 | 213 | results | NVIDIA H100 SXM5 | GPU model | hardware-dependent | setup | "All experiments were performed on a single NVIDIA H100 80\,GB SXM5 with 128\,GB host RAM" |
| T-094 | 213 | results | 80 | GB VRAM | hardware-dependent | setup | "a single NVIDIA H100 80\,GB SXM5" |
| T-095 | 213 | results | 128 | GB host RAM | hardware-dependent | setup | "with 128\,GB host RAM" |
| T-096 | 213 | results | 3.10 | Python version | software | setup | "running Python~3.10, CuPy~13.0, NumPy~1.26, CUDA~12.4, and Ubuntu~22.04" |
| T-097 | 213 | results | 13.0 | CuPy version | software | setup | "running Python~3.10, CuPy~13.0, NumPy~1.26, CUDA~12.4, and Ubuntu~22.04" |
| T-098 | 213 | results | 1.26 | NumPy version | software | setup | "running Python~3.10, CuPy~13.0, NumPy~1.26, CUDA~12.4, and Ubuntu~22.04" |
| T-099 | 213 | results | 12.4 | CUDA version | software | setup | "running Python~3.10, CuPy~13.0, NumPy~1.26, CUDA~12.4, and Ubuntu~22.04" |
| T-100 | 213 | results | 22.04 | Ubuntu version | software | setup | "running Python~3.10, CuPy~13.0, NumPy~1.26, CUDA~12.4, and Ubuntu~22.04" |
| T-101 | 213 | results | 6 | mining runs | method-parameter | setup | "We conducted six mining runs at progressively lower support thresholds, spanning four orders of magnitude" |
| T-102 | 213 | results | 4 | orders of magnitude (support threshold span) | method-parameter | setup | "spanning four orders of magnitude (Table~\ref{tab:campaign})" |
| T-103 | 216 | table tab:campaign caption | 6 | support thresholds | method-parameter | setup | "Mining campaign results across six support thresholds on a single NVIDIA H100 80\,GB." |
| T-104 | 216 | table tab:campaign caption | 1 × NVIDIA H100 | GPU count and model | hardware-dependent | setup | "on a single NVIDIA H100 80\,GB" |
| T-105 | 216 | table tab:campaign caption | 80 | GB VRAM | hardware-dependent | setup | "on a single NVIDIA H100 80\,GB" |
| T-106 | 216 | table tab:campaign caption | 16 | minimum protein count (Ultra) | method-parameter | parameter | "actual thresholds correspond to minimum protein counts of 16 and 8 respectively" |
| T-107 | 216 | table tab:campaign caption | 8 | minimum protein count (Opus) | method-parameter | parameter | "actual thresholds correspond to minimum protein counts of 16 and 8 respectively" |
| T-108 | 224 | table tab:campaign | 0.1 | % support (Base) | method-parameter | parameter | "Base & 0.1\% & 76,891 & 5,305 & 9 & 1.9\,min & Streaming SON" |
| T-109 | 224 | table tab:campaign | 76,891 | min. proteins (Base) | method-parameter | parameter | "Base & 0.1\% & 76,891 & 5,305 & 9 & 1.9\,min & Streaming SON" |
| T-110 | 224 | table tab:campaign | 5,305 | itemsets (Base) | deterministic | result | "Base & 0.1\% & 76,891 & 5,305 & 9 & 1.9\,min & Streaming SON" |
| T-111 | 224 | table tab:campaign | 9 | max K (Base) | deterministic | result | "Base & 0.1\% & 76,891 & 5,305 & 9 & 1.9\,min & Streaming SON" |
| T-112 | 224 | table tab:campaign | 1.9 | min wall-clock (Base) | hardware-dependent | result | "Base & 0.1\% & 76,891 & 5,305 & 9 & 1.9\,min & Streaming SON" |
| T-113 | 224 | table tab:campaign | Streaming SON | method (Base) | method-parameter | setup | "Base & 0.1\% & 76,891 & 5,305 & 9 & 1.9\,min & Streaming SON" |
| T-114 | 225 | table tab:campaign | 0.01 | % support (Super) | method-parameter | parameter | "Super & 0.01\% & 7,689 & 51,124 & 13 & 4.3\,min & Streaming SON" |
| T-115 | 225 | table tab:campaign | 7,689 | min. proteins (Super) | method-parameter | parameter | "Super & 0.01\% & 7,689 & 51,124 & 13 & 4.3\,min & Streaming SON" |
| T-116 | 225 | table tab:campaign | 51,124 | itemsets (Super) | deterministic | result | "Super & 0.01\% & 7,689 & 51,124 & 13 & 4.3\,min & Streaming SON" |
| T-117 | 225 | table tab:campaign | 13 | max K (Super) | deterministic | result | "Super & 0.01\% & 7,689 & 51,124 & 13 & 4.3\,min & Streaming SON" |
| T-118 | 225 | table tab:campaign | 4.3 | min wall-clock (Super) | hardware-dependent | result | "Super & 0.01\% & 7,689 & 51,124 & 13 & 4.3\,min & Streaming SON" |
| T-119 | 225 | table tab:campaign | Streaming SON | method (Super) | method-parameter | setup | "Super & 0.01\% & 7,689 & 51,124 & 13 & 4.3\,min & Streaming SON" |
| T-120 | 226 | table tab:campaign | 0.001 | % support (Power) | method-parameter | parameter | "Power & 0.001\% & 768 & 22,846 & 13 & 18.1\,min & Streaming SON" |
| T-121 | 226 | table tab:campaign | 768 | min. proteins (Power) | method-parameter | parameter | "Power & 0.001\% & 768 & 22,846 & 13 & 18.1\,min & Streaming SON" [cf. 362, 393: min_count=769] |
| T-122 | 226 | table tab:campaign | 22,846 | itemsets (Power, SON) | deterministic | result | "Power & 0.001\% & 768 & 22,846 & 13 & 18.1\,min & Streaming SON" [cf. 238, 404] |
| T-123 | 226 | table tab:campaign | 13 | max K (Power, SON) | deterministic | result | "Power & 0.001\% & 768 & 22,846 & 13 & 18.1\,min & Streaming SON" [cf. 393: Direct at 0.001% reaches K=14] |
| T-124 | 226 | table tab:campaign | 18.1 | min wall-clock (Power) | hardware-dependent | result | "Power & 0.001\% & 768 & 22,846 & 13 & 18.1\,min & Streaming SON" [cf. 238: 1,085.6 s] |
| T-125 | 226 | table tab:campaign | Streaming SON | method (Power) | method-parameter | setup | "Power & 0.001\% & 768 & 22,846 & 13 & 18.1\,min & Streaming SON" |
| T-126 | 228 | table tab:campaign | 0.0001 | % support (Blitz) | method-parameter | parameter | "Blitz & 0.0001\% & 77 & 2,841,280 & 19 & 2.0\,min & Direct CSR$\to$GPU & $9\times$ faster, $124\times$ more itemsets" |
| T-127 | 228 | table tab:campaign | 77 | min. proteins (Blitz) | method-parameter | parameter | "Blitz & 0.0001\% & 77 & 2,841,280 & 19 & 2.0\,min & Direct CSR$\to$GPU" |
| T-128 | 228 | table tab:campaign | 2,841,280 | itemsets (Blitz) | deterministic | result | "Blitz & 0.0001\% & 77 & 2,841,280 & 19 & 2.0\,min & Direct CSR$\to$GPU" |
| T-129 | 228 | table tab:campaign | 19 | max K (Blitz) | deterministic | result | "Blitz & 0.0001\% & 77 & 2,841,280 & 19 & 2.0\,min & Direct CSR$\to$GPU" |
| T-130 | 228 | table tab:campaign | 2.0 | min wall-clock (Blitz) | hardware-dependent | result | "Blitz & 0.0001\% & 77 & 2,841,280 & 19 & 2.0\,min & Direct CSR$\to$GPU" |
| T-131 | 228 | table tab:campaign | Direct CSR→GPU | method (Blitz) | method-parameter | setup | "Blitz & 0.0001\% & 77 & 2,841,280 & 19 & 2.0\,min & Direct CSR$\to$GPU" |
| T-132 | 228 | table tab:campaign | 9× | faster (Blitz vs Power, key transition) | hardware-dependent | result | "$9\times$ faster, $124\times$ more itemsets\textsuperscript{$\dagger$}" [cf. 238, 243] |
| T-133 | 228 | table tab:campaign | 124× | more itemsets (Blitz vs Power) | deterministic | result | "$9\times$ faster, $124\times$ more itemsets\textsuperscript{$\dagger$}" [cf. 238] |
| T-134 | 229 | table tab:campaign | 0.00002 | % support (Ultra, nominal) | method-parameter | parameter | "Ultra & 0.00002\% & 16 & 14,558,875 & 20 & 4.7\,min & Direct CSR$\to$GPU" |
| T-135 | 229 | table tab:campaign | 16 | min. proteins (Ultra) | method-parameter | parameter | "Ultra & 0.00002\% & 16 & 14,558,875 & 20 & 4.7\,min & Direct CSR$\to$GPU" |
| T-136 | 229 | table tab:campaign | 14,558,875 | itemsets (Ultra) | deterministic | result | "Ultra & 0.00002\% & 16 & 14,558,875 & 20 & 4.7\,min & Direct CSR$\to$GPU" |
| T-137 | 229 | table tab:campaign | 20 | max K (Ultra) | deterministic | result | "Ultra & 0.00002\% & 16 & 14,558,875 & 20 & 4.7\,min & Direct CSR$\to$GPU" |
| T-138 | 229 | table tab:campaign | 4.7 | min wall-clock (Ultra) | hardware-dependent | result | "Ultra & 0.00002\% & 16 & 14,558,875 & 20 & 4.7\,min & Direct CSR$\to$GPU" |
| T-139 | 229 | table tab:campaign | Direct CSR→GPU | method (Ultra) | method-parameter | setup | "Ultra & 0.00002\% & 16 & 14,558,875 & 20 & 4.7\,min & Direct CSR$\to$GPU" |
| T-140 | 230 | table tab:campaign | 0.00001 | % support (Opus, nominal) | method-parameter | parameter | "Opus & 0.00001\% & 8 & 26,849,505 & 22 & 7.3\,min & Direct CSR$\to$GPU" |
| T-141 | 230 | table tab:campaign | 8 | min. proteins (Opus) | method-parameter | parameter | "Opus & 0.00001\% & 8 & 26,849,505 & 22 & 7.3\,min & Direct CSR$\to$GPU" |
| T-142 | 230 | table tab:campaign | 26,849,505 | itemsets (Opus) | deterministic | result | "Opus & 0.00001\% & 8 & 26,849,505 & 22 & 7.3\,min & Direct CSR$\to$GPU" [cf. 91, 254, 259: 26.8M] |
| T-143 | 230 | table tab:campaign | 22 | max K (Opus) | deterministic | result | "Opus & 0.00001\% & 8 & 26,849,505 & 22 & 7.3\,min & Direct CSR$\to$GPU" |
| T-144 | 230 | table tab:campaign | 7.3 | min wall-clock (Opus) | hardware-dependent | result | "Opus & 0.00001\% & 8 & 26,849,505 & 22 & 7.3\,min & Direct CSR$\to$GPU" |
| T-145 | 230 | table tab:campaign | Direct CSR→GPU | method (Opus) | method-parameter | setup | "Opus & 0.00001\% & 8 & 26,849,505 & 22 & 7.3\,min & Direct CSR$\to$GPU" |
| T-146 | 235 | table tab:campaign footnote | 0.001 | % support (Power, compared) | method-parameter | parameter | "Compares Power (0.001\%) to Blitz (0.0001\%), varying both method and threshold." |
| T-147 | 235 | table tab:campaign footnote | 0.0001 | % support (Blitz, compared) | method-parameter | parameter | "Compares Power (0.001\%) to Blitz (0.0001\%), varying both method and threshold." |
| T-148 | 235 | table tab:campaign footnote | 21× | controlled same-support speedup (Direct vs SON) | hardware-dependent | result | "Controlled same-support comparison: $21\times$ (\S\ref{sec:results})." [cf. 238, 243, 404] |
| T-149 | 238 | results | 10× | support threshold lowering (Power→Blitz) | method-parameter | parameter | "Despite lowering the support threshold by $10\times$, the Blitz run completes $9\times$ faster and finds $124\times$ more itemsets" |
| T-150 | 238 | results | 9× | faster (Blitz vs Power) | hardware-dependent | result | "the Blitz run completes $9\times$ faster and finds $124\times$ more itemsets (Figure~\ref{fig:campaign})" |
| T-151 | 238 | results | 124× | more itemsets (Blitz vs Power) | deterministic | result | "the Blitz run completes $9\times$ faster and finds $124\times$ more itemsets (Figure~\ref{fig:campaign})" |
| T-152 | 238 | results | 0.001 | % support (controlled comparison) | method-parameter | parameter | "A controlled comparison at identical support (0.001\%) isolates the method effect" |
| T-153 | 238 | results | 475,865 | itemsets (Direct GPU at 0.001%) | deterministic | result | "Direct GPU discovers 475,865 itemsets in 50.7\,s versus SON's 22,846 in 1,085.6\,s---a $21\times$ speedup" [cf. 393, 404] |
| T-154 | 238 | results | 50.7 | s (Direct GPU at 0.001%) | hardware-dependent | result | "Direct GPU discovers 475,865 itemsets in 50.7\,s versus SON's 22,846 in 1,085.6\,s---a $21\times$ speedup" [cf. 404] |
| T-155 | 238 | results | 22,846 | itemsets (SON at 0.001%) | deterministic | result | "versus SON's 22,846 in 1,085.6\,s---a $21\times$ speedup" [cf. 226, 404] |
| T-156 | 238 | results | 1,085.6 | s (SON at 0.001%) | hardware-dependent | result | "versus SON's 22,846 in 1,085.6\,s---a $21\times$ speedup" [cf. 226: 18.1 min; 404] |
| T-157 | 238 | results | 21× | speedup (Direct vs SON, same support) | hardware-dependent | result | "versus SON's 22,846 in 1,085.6\,s---a $21\times$ speedup" |
| T-158 | 238 | results | 95.2 | % of patterns missed by SON | deterministic | result | "SON misses 95.2\% of patterns through two compounding mechanisms." [cf. 404] |
| T-159 | 238 | results | most | K=2 patterns recovered by SON (qualitative) | deterministic | result | "explaining why SON recovers most $K{=}2$ patterns but progressively fewer at higher levels" |
| T-160 | 243 | figure fig:campaign caption | 6 | support thresholds | method-parameter | setup | "Mining campaign across six support thresholds. Bars show itemsets discovered (log scale); line shows wall-clock time." |
| T-161 | 243 | figure fig:campaign caption | 0.001 | % support (SON side of transition) | method-parameter | parameter | "The SON$\to$Direct GPU transition at 0.001\%$\to$0.0001\% yields $9\times$ speedup despite $10\times$ lower support" |
| T-162 | 243 | figure fig:campaign caption | 0.0001 | % support (Direct side of transition) | method-parameter | parameter | "The SON$\to$Direct GPU transition at 0.001\%$\to$0.0001\% yields $9\times$ speedup despite $10\times$ lower support" |
| T-163 | 243 | figure fig:campaign caption | 9× | speedup | hardware-dependent | result | "yields $9\times$ speedup despite $10\times$ lower support" |
| T-164 | 243 | figure fig:campaign caption | 10× | lower support | method-parameter | parameter | "yields $9\times$ speedup despite $10\times$ lower support" |
| T-165 | 243 | figure fig:campaign caption | 21× | controlled same-support speedup | hardware-dependent | result | "a controlled comparison at identical support shows $21\times$ (Section~\ref{sec:results})" |
| T-166 | 249 | results | 9 | K (distribution peak) | deterministic | result | "The distribution is unimodal, peaking at $K{=}9$ with 3,529,257 itemsets (13.14\% of total)" [cf. 254, 276, 462] |
| T-167 | 249 | results | 3,529,257 | itemsets at K=9 | deterministic | result | "peaking at $K{=}9$ with 3,529,257 itemsets (13.14\% of total)" [cf. 276] |
| T-168 | 249 | results | 13.14 | % of total itemsets at K=9 | deterministic | result | "peaking at $K{=}9$ with 3,529,257 itemsets (13.14\% of total)" |
| T-169 | 249 | results | 6 | K (max producible by null model) | deterministic | result | "a permutation-based null model ... cannot produce patterns beyond $K{=}6$" [cf. 364, 391] |
| T-170 | 249 | results | 9 | K peak (second mention) | deterministic | result | "confirming that the $K{=}9$ peak and all deeper patterns reflect genuine biological co-occurrence rather than statistical artifacts" |
| T-171 | 254 | figure fig:kdist caption | 0.00001 | % support | method-parameter | parameter | "$K$-distribution of frequent itemsets at 0.00001\% support (26.8M total)." |
| T-172 | 254 | figure fig:kdist caption | 26.8M | total itemsets | deterministic | result | "$K$-distribution of frequent itemsets at 0.00001\% support (26.8M total)." |
| T-173 | 254 | figure fig:kdist caption | 9 | K (peak) | deterministic | result | "The unimodal distribution peaks at $K{=}9$ (3.53M itemsets)." |
| T-174 | 254 | figure fig:kdist caption | 3.53M | itemsets at peak | deterministic | result | "The unimodal distribution peaks at $K{=}9$ (3.53M itemsets)." |
| T-175 | 254 | figure fig:kdist caption | 1 | itemset at maximum depth | deterministic | result | "A single itemset at maximum depth is shared by exactly 8 proteins." |
| T-176 | 254 | figure fig:kdist caption | 8 | proteins sharing the max-depth itemset | deterministic | result | "A single itemset at maximum depth is shared by exactly 8 proteins." [cf. 286, 289, 335] |
| T-177 | 259 | table tab:kdist caption | 0.00001 | % support | method-parameter | parameter | "$K$-distribution at 0.00001\% support (26.8M total)." |
| T-178 | 259 | table tab:kdist caption | 26.8M | total itemsets | deterministic | result | "$K$-distribution at 0.00001\% support (26.8M total)." |
| T-179 | 268 | table tab:kdist | 1,002 | itemsets at K=1 | deterministic | result | "1 & 1,002 & 0.00 & 12 & 1,996,772 & 7.44" |
| T-180 | 268 | table tab:kdist | 0.00 | % at K=1 | deterministic | result | "1 & 1,002 & 0.00 & 12 & 1,996,772 & 7.44" |
| T-181 | 268 | table tab:kdist | 1,996,772 | itemsets at K=12 | deterministic | result | "1 & 1,002 & 0.00 & 12 & 1,996,772 & 7.44" |
| T-182 | 268 | table tab:kdist | 7.44 | % at K=12 | deterministic | result | "1 & 1,002 & 0.00 & 12 & 1,996,772 & 7.44" |
| T-183 | 269 | table tab:kdist | 73,786 | itemsets at K=2 | deterministic | result | "2 & 73,786 & 0.27 & 13 & 1,259,045 & 4.69" |
| T-184 | 269 | table tab:kdist | 0.27 | % at K=2 | deterministic | result | "2 & 73,786 & 0.27 & 13 & 1,259,045 & 4.69" |
| T-185 | 269 | table tab:kdist | 1,259,045 | itemsets at K=13 | deterministic | result | "2 & 73,786 & 0.27 & 13 & 1,259,045 & 4.69" |
| T-186 | 269 | table tab:kdist | 4.69 | % at K=13 | deterministic | result | "2 & 73,786 & 0.27 & 13 & 1,259,045 & 4.69" |
| T-187 | 270 | table tab:kdist | 452,777 | itemsets at K=3 | deterministic | result | "3 & 452,777 & 1.69 & 14 & 679,471 & 2.53" |
| T-188 | 270 | table tab:kdist | 1.69 | % at K=3 | deterministic | result | "3 & 452,777 & 1.69 & 14 & 679,471 & 2.53" |
| T-189 | 270 | table tab:kdist | 679,471 | itemsets at K=14 | deterministic | result | "3 & 452,777 & 1.69 & 14 & 679,471 & 2.53" |
| T-190 | 270 | table tab:kdist | 2.53 | % at K=14 | deterministic | result | "3 & 452,777 & 1.69 & 14 & 679,471 & 2.53" |
| T-191 | 271 | table tab:kdist | 1,184,461 | itemsets at K=4 | deterministic | result | "4 & 1,184,461 & 4.41 & 15 & 310,527 & 1.16" |
| T-192 | 271 | table tab:kdist | 4.41 | % at K=4 | deterministic | result | "4 & 1,184,461 & 4.41 & 15 & 310,527 & 1.16" |
| T-193 | 271 | table tab:kdist | 310,527 | itemsets at K=15 | deterministic | result | "4 & 1,184,461 & 4.41 & 15 & 310,527 & 1.16" |
| T-194 | 271 | table tab:kdist | 1.16 | % at K=15 | deterministic | result | "4 & 1,184,461 & 4.41 & 15 & 310,527 & 1.16" |
| T-195 | 272 | table tab:kdist | 1,974,126 | itemsets at K=5 | deterministic | result | "5 & 1,974,126 & 7.35 & 16 & 118,659 & 0.44" |
| T-196 | 272 | table tab:kdist | 7.35 | % at K=5 | deterministic | result | "5 & 1,974,126 & 7.35 & 16 & 118,659 & 0.44" |
| T-197 | 272 | table tab:kdist | 118,659 | itemsets at K=16 | deterministic | result | "5 & 1,974,126 & 7.35 & 16 & 118,659 & 0.44" |
| T-198 | 272 | table tab:kdist | 0.44 | % at K=16 | deterministic | result | "5 & 1,974,126 & 7.35 & 16 & 118,659 & 0.44" |
| T-199 | 273 | table tab:kdist | 2,626,332 | itemsets at K=6 | deterministic | result | "6 & 2,626,332 & 9.78 & 17 & 37,261 & 0.14" |
| T-200 | 273 | table tab:kdist | 9.78 | % at K=6 | deterministic | result | "6 & 2,626,332 & 9.78 & 17 & 37,261 & 0.14" |
| T-201 | 273 | table tab:kdist | 37,261 | itemsets at K=17 | deterministic | result | "6 & 2,626,332 & 9.78 & 17 & 37,261 & 0.14" |
| T-202 | 273 | table tab:kdist | 0.14 | % at K=17 | deterministic | result | "6 & 2,626,332 & 9.78 & 17 & 37,261 & 0.14" |
| T-203 | 274 | table tab:kdist | 3,118,459 | itemsets at K=7 | deterministic | result | "7 & 3,118,459 & 11.61 & 18 & 9,375 & 0.03" |
| T-204 | 274 | table tab:kdist | 11.61 | % at K=7 | deterministic | result | "7 & 3,118,459 & 11.61 & 18 & 9,375 & 0.03" |
| T-205 | 274 | table tab:kdist | 9,375 | itemsets at K=18 | deterministic | result | "7 & 3,118,459 & 11.61 & 18 & 9,375 & 0.03" |
| T-206 | 274 | table tab:kdist | 0.03 | % at K=18 | deterministic | result | "7 & 3,118,459 & 11.61 & 18 & 9,375 & 0.03" |
| T-207 | 275 | table tab:kdist | 3,442,954 | itemsets at K=8 | deterministic | result | "8 & 3,442,954 & 12.82 & 19 & 1,818 & 0.01" |
| T-208 | 275 | table tab:kdist | 12.82 | % at K=8 | deterministic | result | "8 & 3,442,954 & 12.82 & 19 & 1,818 & 0.01" |
| T-209 | 275 | table tab:kdist | 1,818 | itemsets at K=19 | deterministic | result | "8 & 3,442,954 & 12.82 & 19 & 1,818 & 0.01" |
| T-210 | 275 | table tab:kdist | 0.01 | % at K=19 | deterministic | result | "8 & 3,442,954 & 12.82 & 19 & 1,818 & 0.01" |
| T-211 | 276 | table tab:kdist | 3,529,257 | itemsets at K=9 (peak, bold) | deterministic | result | "\textbf{9} & \textbf{3,529,257} & \textbf{13.14} & 20 & 255 & $<\!$0.01" [cf. 249, 254, 462] |
| T-212 | 276 | table tab:kdist | 13.14 | % at K=9 (bold) | deterministic | result | "\textbf{9} & \textbf{3,529,257} & \textbf{13.14} & 20 & 255 & $<\!$0.01" |
| T-213 | 276 | table tab:kdist | 255 | itemsets at K=20 | deterministic | result | "\textbf{9} & \textbf{3,529,257} & \textbf{13.14} & 20 & 255 & $<\!$0.01" |
| T-214 | 276 | table tab:kdist | <0.01 | % at K=20 | deterministic | result | "\textbf{9} & \textbf{3,529,257} & \textbf{13.14} & 20 & 255 & $<\!$0.01" |
| T-215 | 277 | table tab:kdist | 3,293,612 | itemsets at K=10 | deterministic | result | "10 & 3,293,612 & 12.27 & 21 & 23 & $<\!$0.01" |
| T-216 | 277 | table tab:kdist | 12.27 | % at K=10 | deterministic | result | "10 & 3,293,612 & 12.27 & 21 & 23 & $<\!$0.01" |
| T-217 | 277 | table tab:kdist | 23 | itemsets at K=21 | deterministic | result | "10 & 3,293,612 & 12.27 & 21 & 23 & $<\!$0.01" |
| T-218 | 277 | table tab:kdist | <0.01 | % at K=21 | deterministic | result | "10 & 3,293,612 & 12.27 & 21 & 23 & $<\!$0.01" |
| T-219 | 278 | table tab:kdist | 2,739,532 | itemsets at K=11 | deterministic | result | "11 & 2,739,532 & 10.20 & 22 & 1 & $<\!$0.01" |
| T-220 | 278 | table tab:kdist | 10.20 | % at K=11 | deterministic | result | "11 & 2,739,532 & 10.20 & 22 & 1 & $<\!$0.01" |
| T-221 | 278 | table tab:kdist | 1 | itemsets at K=22 | deterministic | result | "11 & 2,739,532 & 10.20 & 22 & 1 & $<\!$0.01" [cf. 254, 286] |
| T-222 | 278 | table tab:kdist | <0.01 | % at K=22 | deterministic | result | "11 & 2,739,532 & 10.20 & 22 & 1 & $<\!$0.01" |
| T-223 | 286 | results | 1 | K=22 itemset (count) | deterministic | result | "The single $K{=}22$ itemset is shared by exactly 8 proteins and comprises 22 co-occurring features (Table~\ref{tab:k22})." |
| T-224 | 286 | results | 22 | K (ceiling) | deterministic | result | "The single $K{=}22$ itemset is shared by exactly 8 proteins and comprises 22 co-occurring features" |
| T-225 | 286 | results | 8 | proteins sharing the K=22 itemset | deterministic | result | "The single $K{=}22$ itemset is shared by exactly 8 proteins and comprises 22 co-occurring features" |
| T-226 | 286 | results | 22 | co-occurring features in the itemset | deterministic | result | "is shared by exactly 8 proteins and comprises 22 co-occurring features (Table~\ref{tab:k22})" |
| T-227 | 289 | table tab:k22 caption | 22 | K | deterministic | result | "$K{=}22$ itemset: 22 co-occurring features in 8 proteins." |
| T-228 | 289 | table tab:k22 caption | 22 | co-occurring features | deterministic | result | "$K{=}22$ itemset: 22 co-occurring features in 8 proteins." |
| T-229 | 289 | table tab:k22 caption | 8 | proteins | deterministic | result | "$K{=}22$ itemset: 22 co-occurring features in 8 proteins." |
| T-230 | 297 | table tab:k22 | 2 | Pfam domains in K=22 itemset | deterministic | result | "\multicolumn{3}{l}{\textit{Pfam domains (2)}}" |
| T-231 | 298 | table tab:k22 | PF00270 | Pfam member of K=22 itemset | deterministic | result | "& PF00270 & DEAD/DEAH helicase N-terminal" |
| T-232 | 299 | table tab:k22 | PF00271 | Pfam member of K=22 itemset | deterministic | result | "& PF00271 & Helicase C-terminal" |
| T-233 | 301 | table tab:k22 | 8 | Molecular Function GO terms in K=22 itemset | deterministic | result | "\multicolumn{3}{l}{\textit{Molecular Function (8)}}" |
| T-234 | 302 | table tab:k22 | GO:0005524 | MF member (ATP binding) | deterministic | result | "& GO:0005524 & ATP binding" [cf. 333: the single definitional link] |
| T-235 | 303 | table tab:k22 | GO:0016787 | MF member (Hydrolase activity) | deterministic | result | "& GO:0016787 & Hydrolase activity" |
| T-236 | 304 | table tab:k22 | GO:0000287 | MF member (Magnesium ion binding) | deterministic | result | "& GO:0000287 & Magnesium ion binding" |
| T-237 | 305 | table tab:k22 | GO:0003697 | MF member (ssDNA binding) | deterministic | result | "& GO:0003697 & ssDNA binding" |
| T-238 | 306 | table tab:k22 | GO:0003724 | MF member (RNA helicase activity) | deterministic | result | "& GO:0003724 & RNA helicase activity" |
| T-239 | 307 | table tab:k22 | GO:0003725 | MF member (dsRNA binding) | deterministic | result | "& GO:0003725 & dsRNA binding" |
| T-240 | 308 | table tab:k22 | GO:0003678 | MF member (DNA helicase activity) | deterministic | result | "& GO:0003678 & DNA helicase activity" |
| T-241 | 309 | table tab:k22 | GO:0000978 | MF member (RNA Pol II regulatory binding) | deterministic | result | "& GO:0000978 & RNA Pol II regulatory binding" |
| T-242 | 311 | table tab:k22 | 4 | Biological Process GO terms in K=22 itemset | deterministic | result | "\multicolumn{3}{l}{\textit{Biological Process (4)}}" |
| T-243 | 312 | table tab:k22 | GO:0030154 | BP member (Cell differentiation) | deterministic | result | "& GO:0030154 & Cell differentiation" |
| T-244 | 313 | table tab:k22 | GO:0045087 | BP member (Innate immune response) | deterministic | result | "& GO:0045087 & Innate immune response" |
| T-245 | 314 | table tab:k22 | GO:0051607 | BP member (Defense response to virus) | deterministic | result | "& GO:0051607 & Defense response to virus" |
| T-246 | 315 | table tab:k22 | GO:0034605 | BP member (Cellular response to heat) | deterministic | result | "& GO:0034605 & Cellular response to heat" |
| T-247 | 317 | table tab:k22 | 7 | Cellular Component GO terms in K=22 itemset | deterministic | result | "\multicolumn{3}{l}{\textit{Cellular Component (7)}}" |
| T-248 | 318 | table tab:k22 | GO:0005737 | CC member (Cytoplasm) | deterministic | result | "& GO:0005737 & Cytoplasm" |
| T-249 | 319 | table tab:k22 | GO:0005829 | CC member (Cytosol) | deterministic | result | "& GO:0005829 & Cytosol" |
| T-250 | 320 | table tab:k22 | GO:0005634 | CC member (Nucleus) | deterministic | result | "& GO:0005634 & Nucleus" |
| T-251 | 321 | table tab:k22 | GO:0005739 | CC member (Mitochondrion) | deterministic | result | "& GO:0005739 & Mitochondrion" |
| T-252 | 322 | table tab:k22 | GO:0030424 | CC member (Axon) | deterministic | result | "& GO:0030424 & Axon" |
| T-253 | 323 | table tab:k22 | GO:0030425 | CC member (Dendrite) | deterministic | result | "& GO:0030425 & Dendrite" |
| T-254 | 324 | table tab:k22 | GO:0016607 | CC member (Nuclear speckle) | deterministic | result | "& GO:0016607 & Nuclear speckle" |
| T-255 | 326 | table tab:k22 | 1 | Structural property item in K=22 itemset | deterministic | result | "\multicolumn{3}{l}{\textit{Structural property (1)}}" |
| T-256 | 327 | table tab:k22 | plddt_mean = Medium confidence | structural item member (pLDDT bin) | deterministic | result | "& plddt\_mean & Medium confidence (70--90)" |
| T-257 | 327 | table tab:k22 | 70–90 | pLDDT bin edges (Medium confidence) | method-parameter | parameter | "& plddt\_mean & Medium confidence (70--90)" [only pLDDT bin edges stated anywhere in the paper] |
| T-258 | 333 | results | 22 | features in signature | deterministic | result | "This 22-feature signature is \emph{consistent with} a neuronal antiviral RNA-helicase profile" |
| T-259 | 333 | results | 0 (none) | GO parent–child pairs among the 22 features | deterministic | result | "The 22 features contain \emph{no} GO parent--child pairs, so the depth is not an artifact of the GO true-path rule" [cf. 476] |
| T-260 | 333 | results | 1 | definitional link (GO:0005524 derivable from PF00270 via InterPro2GO) | deterministic | result | "the single definitional link is GO:0005524 (ATP binding), which is auto-derivable from PF00270 via InterPro2GO" |
| T-261 | 333 | results | 21 | independently annotated features | deterministic | result | "leaving 21 independently annotated features" |
| T-262 | 333 | results | 8 | matching proteins (identified during original mining run) | deterministic | result | "The eight matching proteins were identified during the original mining run; because the full transaction matrix is not deposited" |
| T-263 | 335 | results | 8 | proteins (each with exactly 22 vocabulary features) | deterministic | result | "Each of the 8 proteins possesses \emph{exactly} 22 annotated features in our vocabulary, making $K{=}23$ impossible" |
| T-264 | 335 | results | 22 | annotated vocabulary features per K=22 protein (exactly) | deterministic | result | "Each of the 8 proteins possesses \emph{exactly} 22 annotated features in our vocabulary" |
| T-265 | 335 | results | 23 | K (impossible regardless of threshold) | deterministic | result | "making $K{=}23$ impossible regardless of the support threshold" |
| T-266 | 335 | results | 22 | maximum vocabulary features carried by any protein | deterministic | result | "no support threshold can produce a $K{=}23$ itemset when no protein carries more than 22 vocabulary features" |
| T-267 | 335 | results | 23 | K (second mention) | deterministic | result | "no support threshold can produce a $K{=}23$ itemset when no protein carries more than 22 vocabulary features" |
| T-268 | 339 | results | 5 | intermediate patterns highlighted | method-parameter | setup | "We highlight five intermediate patterns, selected to span distinct $K$-levels and biological systems." [cf. 464] |
| T-269 | 341 | results | 19 | K (RNA Spliceosome Processing Hub) | deterministic | result | "\paragraph{$K{=}19$: RNA Spliceosome Processing Hub} (187 proteins)." |
| T-270 | 341 | results | 187 | supporting proteins (K=19 pattern) | deterministic | result | "\paragraph{$K{=}19$: RNA Spliceosome Processing Hub} (187 proteins)." [footnote 353: order-of-magnitude estimates] |
| T-271 | 343 | results | 17 | K (Receptor Tyrosine Kinase Signaling Hub) | deterministic | result | "\paragraph{$K{=}17$: Receptor Tyrosine Kinase Signaling Hub}" |
| T-272 | 344 | results | ~611 | supporting proteins (K=17 pattern) | deterministic | result | "(${\sim}611$ proteins). Protein tyrosine kinase (PF07714) with SH2 (PF00017)" |
| T-273 | 344 | results | PF07714 | Pfam member of K=17 pattern (Protein tyrosine kinase) | deterministic | result | "Protein tyrosine kinase (PF07714) with SH2 (PF00017)" |
| T-274 | 344 | results | PF00017 | Pfam member of K=17 pattern (SH2) | deterministic | result | "Protein tyrosine kinase (PF07714) with SH2 (PF00017)" |
| T-275 | 345 | results | PF00018 | Pfam member of K=17 pattern (SH3) | deterministic | result | "and SH3 (PF00018) adaptor domains---the canonical Src-family kinase" |
| T-276 | 348 | results | 17 | features in RTK combination | deterministic | result | "This 17-feature combination describes the core receptor tyrosine kinase (RTK) signaling machinery" |
| T-277 | 353 | results | 13 | K (Helicase-Recombinase DNA Repair Module) | deterministic | result | "\paragraph{$K{=}13$: Helicase-Recombinase DNA Repair Module} (${\sim}11{,}000$ proteins" |
| T-278 | 353 | results | ~11,000 | supporting proteins (K=13 pattern) | deterministic | result | "(${\sim}11{,}000$ proteins\footnote{...})" |
| T-279 | 353 | footnote | order-of-magnitude estimate | precision of supporting-protein counts for highlighted patterns | deterministic | result | "Supporting-protein counts for the highlighted intermediate-$K$ patterns are order-of-magnitude estimates read from the decoded-pattern summaries." |
| T-280 | 355 | results | 12 | K (Bacterial Cell Wall Synthase) | deterministic | result | "\paragraph{$K{=}12$: Bacterial Cell Wall Synthase} (${\sim}10{,}500$ proteins)." [cf. 464] |
| T-281 | 355 | results | ~10,500 | supporting proteins (K=12 pattern) | deterministic | result | "\paragraph{$K{=}12$: Bacterial Cell Wall Synthase} (${\sim}10{,}500$ proteins)." |
| T-282 | 355 | results | PF00905 | Pfam member of K=12 pattern (transpeptidase) | deterministic | result | "Penicillin-binding protein domains (PF00905 transpeptidase, PF00912 transglycosylase)" |
| T-283 | 355 | results | PF00912 | Pfam member of K=12 pattern (transglycosylase) | deterministic | result | "Penicillin-binding protein domains (PF00905 transpeptidase, PF00912 transglycosylase)" |
| T-284 | 357 | results | 11 | K (AAA+ ATPase Proteasome Complex) | deterministic | result | "\paragraph{$K{=}11$: AAA+ ATPase Proteasome Complex} (${\sim}16{,}000$ proteins)." |
| T-285 | 357 | results | ~16,000 | supporting proteins (K=11 pattern) | deterministic | result | "\paragraph{$K{=}11$: AAA+ ATPase Proteasome Complex} (${\sim}16{,}000$ proteins)." |
| T-286 | 362 | results (sec:null-model) | 0.001 | % support (null-model re-mining threshold) | method-parameter | parameter | "before re-mining at the 0.001\% support threshold ($\text{min\_count}{=}769$)" |
| T-287 | 362 | results (sec:null-model) | 769 | min_count (null-model threshold) | method-parameter | parameter | "re-mining at the 0.001\% support threshold ($\text{min\_count}{=}769$)" [cf. 226: 768; 393] |
| T-288 | 362 | results (sec:null-model) | 5 | permutations | method-parameter | parameter | "Five permutations were performed (seed~42, total runtime 662\,s on the same H100)." |
| T-289 | 362 | results (sec:null-model) | 42 | random seed | method-parameter | parameter | "Five permutations were performed (seed~42, total runtime 662\,s on the same H100)." |
| T-290 | 362 | results (sec:null-model) | 662 | s total runtime (5 permutations) | hardware-dependent | result | "Five permutations were performed (seed~42, total runtime 662\,s on the same H100)." [cf. 393: ~130 s each] |
| T-291 | 362 | results (sec:null-model) | 1 × H100 (same) | GPU used for null model | hardware-dependent | setup | "(seed~42, total runtime 662\,s on the same H100)" |
| T-292 | 364 | results (sec:null-model) | 3 | K (null-model peak) | deterministic | result | "The null model peaks at $K{=}3$ (46.2\% of null itemsets) and generates at most 23 itemsets at $K{=}6$" |
| T-293 | 364 | results (sec:null-model) | 46.2 | % of null itemsets at K=3 | deterministic | result | "The null model peaks at $K{=}3$ (46.2\% of null itemsets)" |
| T-294 | 364 | results (sec:null-model) | 23 | max null itemsets at K=6 (over permutations) | deterministic | result | "and generates at most 23 itemsets at $K{=}6$" [cf. 381: mean 22, sigma 1.1] |
| T-295 | 364 | results (sec:null-model) | 0 | null patterns at K>=7 (any permutation) | deterministic | result | "no permutation produced any pattern at $K{\geq}7$" |
| T-296 | 364 | results (sec:null-model) | 88,745 | biological itemsets at K>=7 (0.001%) | deterministic | result | "In contrast, the biological data contains 88,745 itemsets at $K{\geq}7$ and extends to $K{=}14$." [cf. 382] |
| T-297 | 364 | results (sec:null-model) | 14 | K max (biological, 0.001%) | deterministic | result | "the biological data contains 88,745 itemsets at $K{\geq}7$ and extends to $K{=}14$" [cf. 393] |
| T-298 | 367 | table tab:null-model caption | 0.001 | % support | method-parameter | parameter | "Biological vs.\ null model $K$-distributions at 0.001\% support (5 permutations)." |
| T-299 | 367 | table tab:null-model caption | 5 | permutations | method-parameter | parameter | "Biological vs.\ null model $K$-distributions at 0.001\% support (5 permutations)." |
| T-300 | 367 | table tab:null-model caption | >=4 | K (all strongly enriched) | deterministic | result | "All $K{\geq}4$ patterns are strongly enriched." |
| T-301 | 367 | table tab:null-model caption | 5 | permutations used to estimate sigma | method-parameter | parameter | "The $Z$ column reports standardized effect sizes with $\sigma$ estimated from only 5 permutations" |
| T-302 | 367 | table tab:null-model caption | 4 | degrees of freedom (t-statistics) | method-parameter | parameter | "(equivalently, $t$-statistics with 4 degrees of freedom)" |
| T-303 | 367 | table tab:null-model caption | >=7 | K absent from all null runs | deterministic | result | "$K{\geq}7$ patterns are absent from all 5 null runs ($p < 0.45$" |
| T-304 | 367 | table tab:null-model caption | 5 | null runs (none with K>=7) | deterministic | result | "$K{\geq}7$ patterns are absent from all 5 null runs" |
| T-305 | 367 | table tab:null-model caption | <0.45 | p (K>=7, one-sided binomial bound) | deterministic | result | "($p < 0.45$, the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5)" |
| T-306 | 367 | table tab:null-model caption | 95 | % (one-sided binomial upper bound level) | method-parameter | parameter | "the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5" |
| T-307 | 367 | table tab:null-model caption | 1-0.05^(1/5) | formula for p bound | method-parameter | parameter | "the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5" |
| T-308 | 367 | table tab:null-model caption | 0 of 5 | null runs producing K>=7 | deterministic | result | "the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5" |
| T-309 | 376 | table tab:null-model | 1,002 | Bio itemsets, K=1 | deterministic | result | "1 & 1,002 & 1,002 & 0.0 & 0.0 & 1.0 & preserved" |
| T-310 | 376 | table tab:null-model | 1,002 | Null mu, K=1 | deterministic | result | "1 & 1,002 & 1,002 & 0.0 & 0.0 & 1.0 & preserved" |
| T-311 | 376 | table tab:null-model | 0.0 | Null sigma, K=1 | deterministic | result | "1 & 1,002 & 1,002 & 0.0 & 0.0 & 1.0 & preserved" |
| T-312 | 376 | table tab:null-model | 0.0 | Z, K=1 | deterministic | result | "1 & 1,002 & 1,002 & 0.0 & 0.0 & 1.0 & preserved" |
| T-313 | 376 | table tab:null-model | 1.0 | p, K=1 (preserved) | deterministic | result | "1 & 1,002 & 1,002 & 0.0 & 0.0 & 1.0 & preserved" |
| T-314 | 377 | table tab:null-model | 22,019 | Bio itemsets, K=2 | deterministic | result | "2 & 22,019 & 63,702 & 42.2 & $-987$ & 1.0 & depleted" |
| T-315 | 377 | table tab:null-model | 63,702 | Null mu, K=2 | deterministic | result | "2 & 22,019 & 63,702 & 42.2 & $-987$ & 1.0 & depleted" |
| T-316 | 377 | table tab:null-model | 42.2 | Null sigma, K=2 | deterministic | result | "2 & 22,019 & 63,702 & 42.2 & $-987$ & 1.0 & depleted" |
| T-317 | 377 | table tab:null-model | -987 | Z, K=2 | deterministic | result | "2 & 22,019 & 63,702 & 42.2 & $-987$ & 1.0 & depleted" [cf. 391] |
| T-318 | 377 | table tab:null-model | 1.0 | p, K=2 (depleted) | deterministic | result | "2 & 22,019 & 63,702 & 42.2 & $-987$ & 1.0 & depleted" |
| T-319 | 378 | table tab:null-model | 73,205 | Bio itemsets, K=3 | deterministic | result | "3 & 73,205 & 79,134 & 41.5 & $-143$ & 1.0 & depleted" |
| T-320 | 378 | table tab:null-model | 79,134 | Null mu, K=3 | deterministic | result | "3 & 73,205 & 79,134 & 41.5 & $-143$ & 1.0 & depleted" |
| T-321 | 378 | table tab:null-model | 41.5 | Null sigma, K=3 | deterministic | result | "3 & 73,205 & 79,134 & 41.5 & $-143$ & 1.0 & depleted" |
| T-322 | 378 | table tab:null-model | -143 | Z, K=3 | deterministic | result | "3 & 73,205 & 79,134 & 41.5 & $-143$ & 1.0 & depleted" [cf. 391] |
| T-323 | 378 | table tab:null-model | 1.0 | p, K=3 (depleted) | deterministic | result | "3 & 73,205 & 79,134 & 41.5 & $-143$ & 1.0 & depleted" |
| T-324 | 379 | table tab:null-model | 108,059 | Bio itemsets, K=4 | deterministic | result | "4 & 108,059 & 25,468 & 21.8 & $+3{,}791$ & ${\approx}\,0$ & enriched" |
| T-325 | 379 | table tab:null-model | 25,468 | Null mu, K=4 | deterministic | result | "4 & 108,059 & 25,468 & 21.8 & $+3{,}791$ & ${\approx}\,0$ & enriched" |
| T-326 | 379 | table tab:null-model | 21.8 | Null sigma, K=4 | deterministic | result | "4 & 108,059 & 25,468 & 21.8 & $+3{,}791$ & ${\approx}\,0$ & enriched" |
| T-327 | 379 | table tab:null-model | +3,791 | Z, K=4 | deterministic | result | "4 & 108,059 & 25,468 & 21.8 & $+3{,}791$ & ${\approx}\,0$ & enriched" [cf. 91, 391, 462, 498] |
| T-328 | 379 | table tab:null-model | ~0 | p, K=4 (enriched) | deterministic | result | "4 & 108,059 & 25,468 & 21.8 & $+3{,}791$ & ${\approx}\,0$ & enriched" |
| T-329 | 380 | table tab:null-model | 104,239 | Bio itemsets, K=5 | deterministic | result | "5 & 104,239 & 1,992 & 13.8 & $+7{,}402$ & ${\approx}\,0$ & enriched" |
| T-330 | 380 | table tab:null-model | 1,992 | Null mu, K=5 | deterministic | result | "5 & 104,239 & 1,992 & 13.8 & $+7{,}402$ & ${\approx}\,0$ & enriched" |
| T-331 | 380 | table tab:null-model | 13.8 | Null sigma, K=5 | deterministic | result | "5 & 104,239 & 1,992 & 13.8 & $+7{,}402$ & ${\approx}\,0$ & enriched" |
| T-332 | 380 | table tab:null-model | +7,402 | Z, K=5 | deterministic | result | "5 & 104,239 & 1,992 & 13.8 & $+7{,}402$ & ${\approx}\,0$ & enriched" |
| T-333 | 380 | table tab:null-model | ~0 | p, K=5 (enriched) | deterministic | result | "5 & 104,239 & 1,992 & 13.8 & $+7{,}402$ & ${\approx}\,0$ & enriched" |
| T-334 | 381 | table tab:null-model | 78,596 | Bio itemsets, K=6 | deterministic | result | "6 & 78,596 & 22 & 1.1 & $+71{,}728$ & ${\approx}\,0$ & enriched" |
| T-335 | 381 | table tab:null-model | 22 | Null mu, K=6 | deterministic | result | "6 & 78,596 & 22 & 1.1 & $+71{,}728$ & ${\approx}\,0$ & enriched" [cf. 364: at most 23; 391] |
| T-336 | 381 | table tab:null-model | 1.1 | Null sigma, K=6 | deterministic | result | "6 & 78,596 & 22 & 1.1 & $+71{,}728$ & ${\approx}\,0$ & enriched" [cf. 391] |
| T-337 | 381 | table tab:null-model | +71,728 | Z, K=6 | deterministic | result | "6 & 78,596 & 22 & 1.1 & $+71{,}728$ & ${\approx}\,0$ & enriched" |
| T-338 | 381 | table tab:null-model | ~0 | p, K=6 (enriched) | deterministic | result | "6 & 78,596 & 22 & 1.1 & $+71{,}728$ & ${\approx}\,0$ & enriched" |
| T-339 | 382 | table tab:null-model | 88,745 | Bio itemsets, K=7-14 | deterministic | result | "7--14 & 88,745 & 0 & 0.0 & --- & $<0.45$ & enriched" [cf. 364] |
| T-340 | 382 | table tab:null-model | 0 | Null mu, K=7-14 | deterministic | result | "7--14 & 88,745 & 0 & 0.0 & --- & $<0.45$ & enriched" |
| T-341 | 382 | table tab:null-model | 0.0 | Null sigma, K=7-14 | deterministic | result | "7--14 & 88,745 & 0 & 0.0 & --- & $<0.45$ & enriched" |
| T-342 | 382 | table tab:null-model | --- (undefined) | Z, K=7-14 | deterministic | result | "7--14 & 88,745 & 0 & 0.0 & --- & $<0.45$ & enriched" |
| T-343 | 382 | table tab:null-model | <0.45 | p, K=7-14 (enriched) | deterministic | result | "7--14 & 88,745 & 0 & 0.0 & --- & $<0.45$ & enriched" |
| T-344 | 388 | table tab:null-model footnote | 5 | permutations (Z estimate) | method-parameter | parameter | "Estimated from 5 permutations ($t$-statistic, 4 df); large magnitude indicates strong enrichment, not a calibrated large-sample $Z$." |
| T-345 | 388 | table tab:null-model footnote | 4 | degrees of freedom | method-parameter | parameter | "Estimated from 5 permutations ($t$-statistic, 4 df)" |
| T-346 | 391 | results (sec:null-model) | -987 | Z at K=2 (depleted) | deterministic | result | "$K{=}2$ and $K{=}3$ patterns are \emph{depleted} in biological data relative to the null ($Z{=}{-}987$ and ${-}143$, respectively)" |
| T-347 | 391 | results (sec:null-model) | -143 | Z at K=3 (depleted) | deterministic | result | "($Z{=}{-}987$ and ${-}143$, respectively)" |
| T-348 | 391 | results (sec:null-model) | >=4 | K (biological exceeds null) | deterministic | result | "Second, for $K{\geq}4$, biological co-occurrence massively exceeds the null expectation" |
| T-349 | 391 | results (sec:null-model) | +3,791 | standardized effect size at K=4 | deterministic | result | "the corresponding standardized effect sizes are enormous ($+3{,}791$ at $K{=}4$)" |
| T-350 | 391 | results (sec:null-model) | 5 | permutations (caveat) | method-parameter | parameter | "though we caution that with only 5 permutations these are $t$-statistics (4 df) rather than calibrated large-sample $Z$-scores" |
| T-351 | 391 | results (sec:null-model) | 4 | degrees of freedom | method-parameter | parameter | "these are $t$-statistics (4 df) rather than calibrated large-sample $Z$-scores" |
| T-352 | 391 | results (sec:null-model) | 0 of 5 | permutations with itemsets beyond K=6 | deterministic | result | "In 0 of 5 permutations did the null model produce itemsets beyond $K{=}6$, yielding $p < 0.45$" |
| T-353 | 391 | results (sec:null-model) | <0.45 | p | deterministic | result | "yielding $p < 0.45$ (the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5)" |
| T-354 | 391 | results (sec:null-model) | 95 | % upper bound level | method-parameter | parameter | "(the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5)" |
| T-355 | 391 | results (sec:null-model) | 1-0.05^(1/5) | p-bound formula | method-parameter | parameter | "(the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5)" |
| T-356 | 391 | results (sec:null-model) | 6 | K (null model maximum depth) | deterministic | result | "Third, the null model's maximum depth of $K{=}6$ (with a mean of 22 itemsets, std~1.1, across 5 permutations)" |
| T-357 | 391 | results (sec:null-model) | 22 | mean null itemsets at K=6 | deterministic | result | "(with a mean of 22 itemsets, std~1.1, across 5 permutations)" |
| T-358 | 391 | results (sec:null-model) | 1.1 | std of null itemsets at K=6 | deterministic | result | "(with a mean of 22 itemsets, std~1.1, across 5 permutations)" |
| T-359 | 391 | results (sec:null-model) | 5 | permutations | method-parameter | parameter | "(with a mean of 22 itemsets, std~1.1, across 5 permutations)" |
| T-360 | 393 | results (sec:null-model) | 1,002 | features remaining frequent after shuffling | deterministic | result | "The $K{=}1$ row confirms that all 1,002 features remain frequent after shuffling, verifying that marginal frequencies are preserved." |
| T-361 | 393 | results (sec:null-model) | 0.001 | % support level validated | method-parameter | parameter | "This null model directly validates significance at the 0.001\% support level ($\text{min\_count}{=}769$)" |
| T-362 | 393 | results (sec:null-model) | 769 | min_count at 0.001% | method-parameter | parameter | "at the 0.001\% support level ($\text{min\_count}{=}769$)" [cf. 226: 768; 362] |
| T-363 | 393 | results (sec:null-model) | 475,865 | itemsets (exhaustive Direct CSR->GPU at 0.001%, biological reference) | deterministic | result | "the exhaustive Direct CSR$\to$GPU mining at this threshold (475,865 itemsets, reaching $K{=}14$), not the SON approximation" |
| T-364 | 393 | results (sec:null-model) | 14 | K max (Direct at 0.001%) | deterministic | result | "(475,865 itemsets, reaching $K{=}14$), not the SON approximation listed as ``Power'' in Table~\ref{tab:campaign}" [cf. 226: SON K=13] |
| T-365 | 393 | results (sec:null-model) | 8 | min_count (Opus threshold) | method-parameter | parameter | "The Opus threshold results ($\text{min\_count}{=}8$, $K_\text{max}{=}22$, Table~\ref{tab:kdist}) were obtained at a more permissive threshold" |
| T-366 | 393 | results (sec:null-model) | 22 | K_max (Opus) | deterministic | result | "The Opus threshold results ($\text{min\_count}{=}8$, $K_\text{max}{=}22$, Table~\ref{tab:kdist})" |
| T-367 | 393 | results (sec:null-model) | 100+ | permutations needed for p<0.01 (hypothetical) | method-parameter | background | "A full validation with 100+ permutations would be needed to establish $p < 0.01$ bounds at either threshold" |
| T-368 | 393 | results (sec:null-model) | <0.01 | p bound (hypothetical target) | method-parameter | background | "A full validation with 100+ permutations would be needed to establish $p < 0.01$ bounds at either threshold" |
| T-369 | 393 | results (sec:null-model) | ~130 | s per permutation (full mining run) | hardware-dependent | result | "computational constraints (each permutation requires a full mining run at ${\sim}$130\,s) limited us to 5 trials" [cf. 362: 662 s total] |
| T-370 | 393 | results (sec:null-model) | 5 | trials (permutations) | method-parameter | parameter | "computational constraints (each permutation requires a full mining run at ${\sim}$130\,s) limited us to 5 trials" |
| T-371 | 402 | discussion | 100-million | transaction scale (feasibility claim) | deterministic | result | "ET-miner demonstrates that exact Apriori computation is feasible at the 100-million-transaction scale on a single high-end datacenter GPU." [actual mined set 76.9M] |
| T-372 | 402 | discussion | 1 | GPU (single high-end datacenter GPU) | hardware-dependent | setup | "feasible at the 100-million-transaction scale on a single high-end datacenter GPU" |
| T-373 | 402 | discussion | 206 | GB (full-set dense matrix) | deterministic | setup | "The full-set $206\text{\,GB}$ dense matrix, the $5.1\text{\,GB}$ CSR of the mined subset, and the $26\text{\,GB}$ GPU bitvector matrix" |
| T-374 | 402 | discussion | 5.1 | GB (CSR of mined subset) | deterministic | result | "the $5.1\text{\,GB}$ CSR of the mined subset" |
| T-375 | 402 | discussion | 26 | GB (GPU bitvector matrix) | deterministic | result | "and the $26\text{\,GB}$ GPU bitvector matrix illustrate a compression path" |
| T-376 | 402 | discussion | ~15× | same-subset dense-to-CSR reduction | deterministic | result | "(the same-subset dense-to-CSR reduction is ${\sim}15\times$)" |
| T-377 | 402 | discussion | 22 | iterations (K-levels) with bitvectors on-GPU | deterministic | result | "bitvectors stay on-GPU across all 22 iterations, with only lightweight metadata (prefix groups and result indices) crossing the PCIe bus" [cf. 813] |
| T-378 | 404 | discussion | 0.001 | % support (controlled comparison) | method-parameter | parameter | "A controlled comparison at identical support threshold (0.001\%) reveals a $21\times$ speedup (50.7\,s vs.\ 1,085.6\,s)" |
| T-379 | 404 | discussion | 21× | speedup (Direct vs SON) | hardware-dependent | result | "reveals a $21\times$ speedup (50.7\,s vs.\ 1,085.6\,s)" |
| T-380 | 404 | discussion | 50.7 | s (Direct CSR->GPU at 0.001%) | hardware-dependent | result | "reveals a $21\times$ speedup (50.7\,s vs.\ 1,085.6\,s)" |
| T-381 | 404 | discussion | 1,085.6 | s (SON at 0.001%) | hardware-dependent | result | "reveals a $21\times$ speedup (50.7\,s vs.\ 1,085.6\,s)" |
| T-382 | 404 | discussion | 95.2 | % frequent itemsets missed by SON | deterministic | result | "and shows that SON misses 95.2\% of frequent itemsets (22,846 vs.\ 475,865)" |
| T-383 | 404 | discussion | 22,846 | itemsets (SON) | deterministic | result | "SON misses 95.2\% of frequent itemsets (22,846 vs.\ 475,865)" |
| T-384 | 404 | discussion | 475,865 | itemsets (Direct) | deterministic | result | "SON misses 95.2\% of frequent itemsets (22,846 vs.\ 475,865)" |
| T-385 | 408 | discussion | 100× | speedup (GPApriori, prior work) | external-fact | background | "GPApriori~\cite{zhang2011} achieved up to $100\times$ speedup on FIMI benchmarks using static bitsets." |
| T-386 | 408 | discussion | 100M | transactions (BIGMiner, prior work) | external-fact | background | "BIGMiner~\cite{chon2018b} achieved the largest prior scale at 100M transactions on 30 MapReduce nodes." |
| T-387 | 408 | discussion | 30 | MapReduce nodes (BIGMiner) | external-fact | background | "BIGMiner~\cite{chon2018b} achieved the largest prior scale at 100M transactions on 30 MapReduce nodes." |
| T-388 | 408 | discussion | a few million | transactions (prior systems max) | external-fact | background | "(2)~they were evaluated on datasets of at most a few million transactions, three orders of magnitude smaller than our proteome dataset" |
| T-389 | 408 | discussion | 3 | orders of magnitude | external-fact | background | "three orders of magnitude smaller than our proteome dataset" |
| T-390 | 411 | table tab:scale-comparison caption | 5.1× | more transactions than largest prior single-machine GPU system | deterministic | result | "ET-miner processes $5.1\times$ more transactions than the largest prior single-machine GPU system (GMiner, 15M synthetic transactions; 1.7M real)" |
| T-391 | 411 | table tab:scale-comparison caption | 15M | synthetic transactions (GMiner) | external-fact | background | "(GMiner, 15M synthetic transactions; 1.7M real)" |
| T-392 | 411 | table tab:scale-comparison caption | 1.7M | real transactions (GMiner) | external-fact | background | "(GMiner, 15M synthetic transactions; 1.7M real)" [cf. 843] |
| T-393 | 411 | table tab:scale-comparison caption | 22 | K (deepest GPU FIM result reported) | deterministic | result | "and mines exhaustively to $K{=}22$, deeper than any previously reported GPU FIM result" |
| T-394 | 420 | table tab:scale-comparison | 100K | transactions (Borgelt) | external-fact | background | "Borgelt~\cite{borgelt2003} & 100K & 500 & $\sim$10 & 1$\times$ CPU & 1--100\,s" |
| T-395 | 420 | table tab:scale-comparison | 500 | items (Borgelt) | external-fact | background | "Borgelt~\cite{borgelt2003} & 100K & 500 & $\sim$10 & 1$\times$ CPU & 1--100\,s" |
| T-396 | 420 | table tab:scale-comparison | ~10 | max K (Borgelt) | external-fact | background | "Borgelt~\cite{borgelt2003} & 100K & 500 & $\sim$10 & 1$\times$ CPU & 1--100\,s" |
| T-397 | 420 | table tab:scale-comparison | 1× CPU | hardware (Borgelt) | external-fact | background | "Borgelt~\cite{borgelt2003} & 100K & 500 & $\sim$10 & 1$\times$ CPU & 1--100\,s" |
| T-398 | 420 | table tab:scale-comparison | 1–100 | s (Borgelt time) | external-fact | background | "Borgelt~\cite{borgelt2003} & 100K & 500 & $\sim$10 & 1$\times$ CPU & 1--100\,s" |
| T-399 | 421 | table tab:scale-comparison | 100K | transactions (Fang) | external-fact | background | "Fang~\cite{fang2009} & 100K & 1K & $\sim$5 & 1$\times$ GTX 280 & 1--10\,s" |
| T-400 | 421 | table tab:scale-comparison | 1K | items (Fang) | external-fact | background | "Fang~\cite{fang2009} & 100K & 1K & $\sim$5 & 1$\times$ GTX 280 & 1--10\,s" |
| T-401 | 421 | table tab:scale-comparison | ~5 | max K (Fang) | external-fact | background | "Fang~\cite{fang2009} & 100K & 1K & $\sim$5 & 1$\times$ GTX 280 & 1--10\,s" |
| T-402 | 421 | table tab:scale-comparison | 1× GTX 280 | hardware (Fang) | external-fact | background | "Fang~\cite{fang2009} & 100K & 1K & $\sim$5 & 1$\times$ GTX 280 & 1--10\,s" |
| T-403 | 421 | table tab:scale-comparison | 1–10 | s (Fang time) | external-fact | background | "Fang~\cite{fang2009} & 100K & 1K & $\sim$5 & 1$\times$ GTX 280 & 1--10\,s" |
| T-404 | 422 | table tab:scale-comparison | 15M | transactions (GMiner) | external-fact | background | "GMiner~\cite{chon2018} & 15M & 20K & $\sim$30 & 4$\times$ GTX 1080 & 20--150\,s" |
| T-405 | 422 | table tab:scale-comparison | 20K | items (GMiner) | external-fact | background | "GMiner~\cite{chon2018} & 15M & 20K & $\sim$30 & 4$\times$ GTX 1080 & 20--150\,s" |
| T-406 | 422 | table tab:scale-comparison | ~30 | max K (GMiner) | external-fact | background | "GMiner~\cite{chon2018} & 15M & 20K & $\sim$30 & 4$\times$ GTX 1080 & 20--150\,s" |
| T-407 | 422 | table tab:scale-comparison | 4× GTX 1080 | hardware (GMiner) | external-fact | background | "GMiner~\cite{chon2018} & 15M & 20K & $\sim$30 & 4$\times$ GTX 1080 & 20--150\,s" |
| T-408 | 422 | table tab:scale-comparison | 20–150 | s (GMiner time) | external-fact | background | "GMiner~\cite{chon2018} & 15M & 20K & $\sim$30 & 4$\times$ GTX 1080 & 20--150\,s" |
| T-409 | 423 | table tab:scale-comparison | 100M | transactions (BIGMiner) | external-fact | background | "BIGMiner~\cite{chon2018b} & 100M & 100K & --- & 30$\times$ servers & 1--20K\,s" |
| T-410 | 423 | table tab:scale-comparison | 100K | items (BIGMiner) | external-fact | background | "BIGMiner~\cite{chon2018b} & 100M & 100K & --- & 30$\times$ servers & 1--20K\,s" |
| T-411 | 423 | table tab:scale-comparison | --- (not reported) | max K (BIGMiner) | external-fact | background | "BIGMiner~\cite{chon2018b} & 100M & 100K & --- & 30$\times$ servers & 1--20K\,s" |
| T-412 | 423 | table tab:scale-comparison | 30× servers | hardware (BIGMiner) | external-fact | background | "BIGMiner~\cite{chon2018b} & 100M & 100K & --- & 30$\times$ servers & 1--20K\,s" |
| T-413 | 423 | table tab:scale-comparison | 1–20K | s (BIGMiner time) | external-fact | background | "BIGMiner~\cite{chon2018b} & 100M & 100K & --- & 30$\times$ servers & 1--20K\,s" |
| T-414 | 425 | table tab:scale-comparison | 76.9M | transactions (ET-miner) | deterministic | setup | "ET-miner & \textbf{76.9M} & 1,002 & \textbf{22} & 1$\times$ H100 & 7.3\,min" |
| T-415 | 425 | table tab:scale-comparison | 1,002 | items (ET-miner) | deterministic | setup | "ET-miner & \textbf{76.9M} & 1,002 & \textbf{22} & 1$\times$ H100 & 7.3\,min" |
| T-416 | 425 | table tab:scale-comparison | 22 | max K (ET-miner) | deterministic | result | "ET-miner & \textbf{76.9M} & 1,002 & \textbf{22} & 1$\times$ H100 & 7.3\,min" |
| T-417 | 425 | table tab:scale-comparison | 1× H100 | hardware (ET-miner) | hardware-dependent | setup | "ET-miner & \textbf{76.9M} & 1,002 & \textbf{22} & 1$\times$ H100 & 7.3\,min" |
| T-418 | 425 | table tab:scale-comparison | 7.3 | min (ET-miner time) | hardware-dependent | result | "ET-miner & \textbf{76.9M} & 1,002 & \textbf{22} & 1$\times$ H100 & 7.3\,min" |
| T-419 | 440 | discussion | ~3 | GB (CSR column indices transferred to GPU) | deterministic | result | "transferred to the GPU once (${\sim}3$\,GB of CSR column indices versus ${\sim}10$\,GB for the equivalent bit-packed dense bitmap of the 76.9M mining subset" |
| T-420 | 440 | discussion | ~10 | GB (bit-packed dense bitmap of 76.9M subset) | deterministic | setup | "versus ${\sim}10$\,GB for the equivalent bit-packed dense bitmap of the 76.9M mining subset, a saving that grows with sparsity" |
| T-421 | 440 | discussion | 76.9M | proteins (mining subset) | deterministic | setup | "the equivalent bit-packed dense bitmap of the 76.9M mining subset" |
| T-422 | 440 | discussion | once | CSR transfer to GPU (count) | deterministic | setup | "The transaction database is stored in compressed sparse format (CSR), transferred to the GPU once" |
| T-423 | 443 | discussion | ~12 | bytes per iteration (single integer of surviving-pattern count) | deterministic | result | "The only per-iteration communication is a single integer indicating how many patterns survived (${\sim}12$\,bytes)." [cf. 808-810, 839; 852/868: ~4 bytes] |
| T-424 | 446 | discussion | 1 × H100 | GPU (all experiments) | hardware-dependent | setup | "All experiments in this paper used a single H100; the 76.9M multi-feature proteins were mined without manual tuning." |
| T-425 | 446 | discussion | 76.9M | multi-feature proteins mined | deterministic | setup | "All experiments in this paper used a single H100; the 76.9M multi-feature proteins were mined without manual tuning." |
| T-426 | 454 | discussion | 2 | K (Wang et al. domain co-occurrence networks, pairwise) | external-fact | background | "but this approach is inherently pairwise ($K{=}2$) and limited to individual proteomes" |
| T-427 | 454 | discussion | 2 (pairs) | K (Terrapon et al.) | external-fact | background | "Terrapon et al.~\cite{coin2009} used statistical co-occurrence for novel domain detection in \emph{Plasmodium falciparum}, again limited to pairs." |
| T-428 | 454 | discussion | ~32K | structures (Meysman et al., PDB) | external-fact | background | "but addressed 3D proximity rather than domain-level combinations and was limited to ${\sim}32$K structures" |
| T-429 | 456 | discussion | 2 | K (limit of prior exhaustive itemset mining) | external-fact | background | "None of these approaches perform exhaustive itemset mining beyond $K{=}2$." |
| T-430 | 456 | discussion | 22 | co-occurring features (max pattern span) | deterministic | result | "ET-miner's discovery of patterns spanning up to 22 co-occurring features across the full AlphaFold universe" |
| T-431 | 460 | discussion | 22 | K (neuronal antiviral sentinel) | deterministic | result | "The $K{=}22$ neuronal antiviral sentinel combines four distinct functional roles" |
| T-432 | 460 | discussion | 4 | distinct functional roles (interpretive grouping) | deterministic | result | "combines four distinct functional roles: (1)~RNA/DNA unwinding, (2)~innate immune sensing, (3)~gene regulation, and (4)~neuronal function" |
| T-433 | 460 | discussion | 22 | annotations (co-occurrence, not mechanism) | deterministic | result | "the co-occurrence of these 22 annotations does not by itself demonstrate functional coupling" |
| T-434 | 460 | discussion | 22 | K ceiling (reflects annotation depth) | deterministic | result | "The $K{=}22$ ceiling reflects the depth of current UniProt annotations rather than a fundamental limit of biological complexity" |
| T-435 | 462 | discussion | 9 | K (unimodal peak) | deterministic | result | "The unimodal peak at $K{=}9$ (3.53 million itemsets, 13.14\% of all patterns) occurs where combinatorial growth and support decay intersect." |
| T-436 | 462 | discussion | 3.53 million | itemsets at K=9 | deterministic | result | "The unimodal peak at $K{=}9$ (3.53 million itemsets, 13.14\% of all patterns)" |
| T-437 | 462 | discussion | 13.14 | % of all patterns at K=9 | deterministic | result | "The unimodal peak at $K{=}9$ (3.53 million itemsets, 13.14\% of all patterns)" |
| T-438 | 462 | discussion | >=7 | K (random shuffling produces none) | deterministic | result | "random shuffling cannot produce any patterns at $K{\geq}7$" |
| T-439 | 462 | discussion | 4 | K (biological exceeds null by many orders of magnitude) | deterministic | result | "and even at $K{=}4$ the biological count exceeds the null expectation by many orders of magnitude" |
| T-440 | 462 | discussion | many orders of magnitude | bio/null excess at K=4 (qualitative) | deterministic | result | "the biological count exceeds the null expectation by many orders of magnitude (a standardized effect size of $+3{,}791$ estimated from 5 permutations)" |
| T-441 | 462 | discussion | +3,791 | standardized effect size at K=4 | deterministic | result | "(a standardized effect size of $+3{,}791$ estimated from 5 permutations)" |
| T-442 | 462 | discussion | 5 | permutations | method-parameter | parameter | "(a standardized effect size of $+3{,}791$ estimated from 5 permutations)" |
| T-443 | 464 | discussion | 12 | K (bacterial cell wall synthase module) | deterministic | result | "The $K{=}12$ bacterial cell wall synthase module (Section~\ref{sec:results}) validates the approach" |
| T-444 | 464 | discussion | 5 | highlighted patterns | method-parameter | setup | "The five highlighted patterns ($K{=}11$ to $K{=}19$) were selected to illustrate the diversity of function across the $K$-distribution" |
| T-445 | 464 | discussion | 11 | K (lowest highlighted pattern) | deterministic | result | "The five highlighted patterns ($K{=}11$ to $K{=}19$)" |
| T-446 | 464 | discussion | 19 | K (highest highlighted pattern) | deterministic | result | "The five highlighted patterns ($K{=}11$ to $K{=}19$)" |
| T-447 | 466 | discussion | >=4 | K (higher-order patterns not explained by annotation structure) | deterministic | result | "the null model analysis (Section~\ref{sec:null-model}) demonstrates that the higher-order patterns ($K{\geq}4$) cannot be explained by annotation structure alone" |
| T-448 | 474 | discussion (sec:limitations) | millions | patterns tested (FDR-correction context) | deterministic | background | "with false discovery rate correction for the millions of patterns tested" |
| T-449 | 476 | discussion (sec:limitations) | 22 | K (deepest itemset) | deterministic | result | "We note, however, that the deepest ($K{=}22$) itemset contains 0 parent--child pairs, so its depth is not driven by this effect" |
| T-450 | 476 | discussion (sec:limitations) | 0 | GO parent-child pairs in K=22 itemset | deterministic | result | "the deepest ($K{=}22$) itemset contains 0 parent--child pairs" [cf. 333] |
| T-451 | 478 | discussion (sec:limitations) | monthly | GO annotation update frequency | external-fact | background | "GO annotations are updated monthly; our results reflect UniProt TrEMBL release 2025\_01" |
| T-452 | 478 | discussion (sec:limitations) | 2025_01 | UniProt TrEMBL release | software | setup | "our results reflect UniProt TrEMBL release 2025\_01 and may vary with different releases" |
| T-453 | 480 | discussion (sec:limitations) | 128.7 million | proteins excluded (single annotated feature) | deterministic | result | "Our analysis excludes 128.7 million proteins (62.6\%) with only a single annotated feature." [cf. 152] |
| T-454 | 480 | discussion (sec:limitations) | 62.6 | % of proteins excluded | deterministic | result | "Our analysis excludes 128.7 million proteins (62.6\%) with only a single annotated feature." |
| T-455 | 480 | discussion (sec:limitations) | 1 | annotated feature (exclusion criterion) | method-parameter | parameter | "excludes 128.7 million proteins (62.6\%) with only a single annotated feature" [cf. 152: >1 kept] |
| T-456 | 486 | discussion (sec:limitations) | 1.7 million | high-confidence homodimer predictions (AlphaFold DB) | external-fact | background | "1.7 million high-confidence homodimer predictions have been added to the database" |
| T-457 | 486 | discussion (sec:limitations) | 18 million | lower-confidence homodimers (bulk download) | external-fact | background | "with an additional 18 million lower-confidence homodimers available via bulk download" |
| T-458 | 486 | discussion (sec:limitations) | 2026 | year of quoted EMBL-EBI statement | external-fact | background | "``a first step towards a comprehensive description of the human interactome'' (EMBL-EBI, 2026)" |
| T-459 | 496 | conclusion | 100-million | transaction scale | deterministic | result | "ET-miner demonstrates that exhaustive frequent itemset mining at the 100-million-transaction scale is practical on current-generation datacenter GPUs." |
| T-460 | 496 | conclusion | 1,002 | features (base vocabulary) | deterministic | setup | "With a base vocabulary of 1,002 features on a single H100, mining completes in 7.3 minutes" |
| T-461 | 496 | conclusion | 1 × H100 | GPU | hardware-dependent | setup | "With a base vocabulary of 1,002 features on a single H100" |
| T-462 | 496 | conclusion | 7.3 | minutes (mining) | hardware-dependent | result | "mining completes in 7.3 minutes (excluding the one-time 63-minute feature extraction) and reaches $K{=}22$" |
| T-463 | 496 | conclusion | 63 | minutes (one-time feature extraction) | hardware-dependent | result | "(excluding the one-time 63-minute feature extraction)" |
| T-464 | 496 | conclusion | 22 | K reached | deterministic | result | "mining completes in 7.3 minutes (excluding the one-time 63-minute feature extraction) and reaches $K{=}22$" |
| T-465 | 498 | conclusion | Power threshold (0.001%) | null-model support threshold | method-parameter | parameter | "A permutation-based null model at the Power threshold supports biological significance" |
| T-466 | 498 | conclusion | >3,700 | Z (t-statistics) | deterministic | result | "($Z{>}3{,}700$ for $K{=}4$--$6$, as $t$-statistics from 5 permutations; no null run reached $K{\geq}7$" |
| T-467 | 498 | conclusion | 4–6 | K range | deterministic | result | "($Z{>}3{,}700$ for $K{=}4$--$6$, as $t$-statistics from 5 permutations" |
| T-468 | 498 | conclusion | 5 | permutations | method-parameter | parameter | "as $t$-statistics from 5 permutations; no null run reached $K{\geq}7$" |
| T-469 | 498 | conclusion | >=7 | K (no null run reached) | deterministic | result | "no null run reached $K{\geq}7$; see Section~\ref{sec:null-model}" |
| T-470 | 500 | conclusion | 10.5281/zenodo.18674353 | Zenodo DOI (preprint) | external-fact | setup | "A preprint of this work is archived at Zenodo (DOI: \texttt{10.5281/zenodo.18674353})." |
| T-471 | 529 | back matter (data availability) | 10.5281/zenodo.18674353 | Zenodo DOI (preprint) | external-fact | setup | "This manuscript is a revised version of the preprint archived at Zenodo (DOI \texttt{10.5281/zenodo.18674353})." |
| T-472 | 529 | back matter (data availability) | 2025_01 | UniProt release (regeneration source) | software | setup | "it is regenerable from UniProt release 2025\_01 via the released feature-extraction pipeline" |
| T-473 | 743 | appendix A (app:complexity) | 64 | proteins per CPU/GPU operation | deterministic | background | "The bitvector representation processes 64 proteins per CPU/GPU operation, providing a $64\times$ constant-factor speedup over na\"ive element-wise comparison." |
| T-474 | 743 | appendix A (app:complexity) | 64× | constant-factor speedup | deterministic | background | "providing a $64\times$ constant-factor speedup over na\"ive element-wise comparison" |
| T-475 | 746–751 | appendix A (app:complexity) | 64 | word width in complexity terms (ceil(N/64)) | deterministic | background | "GPU: $O(\|F_1\| \times \lceil N/64 \rceil)$ for bitvectors (persistent across all $K$-levels)" and the per-level $O(\cdot \times \lceil N/64 \rceil / P)$ terms |
| T-476 | 753 | appendix A (app:complexity) | 64× | constant factor improvement (O(N) to O(ceil(N/64))) | deterministic | background | "reduces support counting from $O(N)$ integer comparisons to $O(\lceil N/64 \rceil)$ bitwise AND + popcount operations---a $64\times$ constant factor improvement" |
| T-477 | 766 | appendix B (alg:etminer) | min_count = ceil(sigma * n) | minimum count rule | method-parameter | parameter | "\State $n \gets \|T\|$; $\text{min\_count} \gets \lceil \sigma \cdot n \rceil$" |
| T-478 | 769 | appendix B (alg:etminer) | ~26 | GB VRAM (bitvector matrix B) | deterministic | result | "\State $B \gets \text{GPU\_CSR\_TO\_BITVEC}(\text{CSR})$ \Comment{${\sim}26$\,GB VRAM}" |
| T-479 | 775 | appendix B (alg:etminer) | 3 | starting k for prefix-group loop | method-parameter | parameter | "\State $k \gets 3$" |
| T-480 | 776 | appendix B (alg:etminer) | freq_{k-1} >= k | loop-continuation condition | method-parameter | parameter | "\While{$\|\text{freq}_{k-1}\| \geq k$}" |
| T-481 | 782 | appendix B (alg:etminer) | 1 (single) | bulk result transfer at end | deterministic | setup | "\State $F \gets \text{BULK\_TRANSFER}(\text{results})$ \Comment{Single transfer}" |
| T-482 | 797 | appendix C (app:gpu-details) | 64-bit | popcount word width (__popcll) | software | background | "maps to the GPU's native \texttt{\_\_popcll} instruction---a hardware intrinsic that counts set bits in a 64-bit word in a single clock cycle" |
| T-483 | 797 | appendix C (app:gpu-details) | 1 | clock cycle per __popcll | hardware-dependent | background | "a hardware intrinsic that counts set bits in a 64-bit word in a single clock cycle" |
| T-484 | 797 | appendix C (app:gpu-details) | 64 | proteins processed per operation | deterministic | background | "Each operation processes 64 proteins simultaneously, compared to 32 with the 32-bit \texttt{\_\_popc} used by prior systems" |
| T-485 | 797 | appendix C (app:gpu-details) | 32 | proteins per operation (prior systems) | external-fact | background | "compared to 32 with the 32-bit \texttt{\_\_popc} used by prior systems~\cite{chon2018,chon2024}" |
| T-486 | 797 | appendix C (app:gpu-details) | 32-bit | popcount width (__popc, prior systems) | external-fact | background | "compared to 32 with the 32-bit \texttt{\_\_popc} used by prior systems~\cite{chon2018,chon2024}" |
| T-487 | 801 | appendix C (app:gpu-details) | 1 | candidate pair per GPU thread (K=2) | software | setup | "For pairs ($K{=}2$), each GPU thread evaluates one candidate pair using a triangular-number inverse mapping" |
| T-488 | 801 | appendix C (app:gpu-details) | 0 (AND result) | early-termination condition (K>=3) | method-parameter | parameter | "with early termination when an intermediate AND result is zero---meaning no protein carries all features in the candidate set" |
| T-489 | 808 | appendix C (app:gpu-details) | ~12 | bytes transferred at K=1 | deterministic | result | "$K{=}1$: ... Transfer: ${\sim}12$\,bytes (frequent item count and loop control)." |
| T-490 | 809 | appendix C (app:gpu-details) | ~12 | bytes transferred at K=2 | deterministic | result | "$K{=}2$: Fused pair kernel produces a frequent pair array in GPU memory. Transfer: ${\sim}12$\,bytes (result count and loop control)." |
| T-491 | 810 | appendix C (app:gpu-details) | ~12 | bytes transferred per level at K>=3 | deterministic | result | "$K{\geq}3$: ... Results sorted in GPU memory. Transfer: ${\sim}12$\,bytes (scalar counters)." |
| T-492 | 813 | appendix C (app:gpu-details) | 22 | K-levels (total) | deterministic | result | "The total data transfer across all 22 $K$-levels is ${\sim}264$~bytes" |
| T-493 | 813 | appendix C (app:gpu-details) | ~264 | bytes total CPU-GPU transfer across all K-levels | deterministic | result | "The total data transfer across all 22 $K$-levels is ${\sim}264$~bytes---compared to gigabytes in conventional GPU FIM implementations" [22 x 12 = 264] |
| T-494 | 813 | appendix C (app:gpu-details) | gigabytes | per-campaign transfer in conventional GPU FIM | external-fact | background | "compared to gigabytes in conventional GPU FIM implementations that transfer candidate sets and count arrays at each iteration" |
| T-495 | 834 | table tab:gpu-arch | 32-bit | __popc width (GMiner) | external-fact | background | "Support counting & \texttt{\_\_popc} (32-bit) & \texttt{\_\_popc} (32-bit) & \textbf{\texttt{\_\_popcll} (64-bit)}" |
| T-496 | 834 | table tab:gpu-arch | 32-bit | __popc width (GMiner++) | external-fact | background | "Support counting & \texttt{\_\_popc} (32-bit) & \texttt{\_\_popc} (32-bit) & \textbf{\texttt{\_\_popcll} (64-bit)}" |
| T-497 | 834 | table tab:gpu-arch | 64-bit | __popcll width (ET-miner) | software | setup | "Support counting & \texttt{\_\_popc} (32-bit) & \texttt{\_\_popc} (32-bit) & \textbf{\texttt{\_\_popcll} (64-bit)}" |
| T-498 | 839 | table tab:gpu-arch | ~12 | bytes PCIe per K-level (ET-miner) | deterministic | result | "PCIe per $K$-level & $O(\|TB\| + \|C_L\| + \|PS\|)$ & $O(\text{bit array blocks})$ & \textbf{${\sim}12$\,bytes}" |
| T-499 | 843 | table tab:gpu-arch | 1.7M | transactions, max tested dataset (GMiner, real) | external-fact | background | "Max tested dataset & 1.7M txns (real) & Not reported & \textbf{76.9M txns (proteome)}" |
| T-500 | 843 | table tab:gpu-arch | 76.9M | transactions, max tested dataset (ET-miner, proteome) | deterministic | setup | "Max tested dataset & 1.7M txns (real) & Not reported & \textbf{76.9M txns (proteome)}" |
| T-501 | 852 | table tab:gpu-pipeline caption | ~4 | bytes per K-level (CPU loop control) | deterministic | result | "with CPU involvement limited to loop control ($\sim$4 bytes per $K$-level)" [cf. 443, 808-810, 839: ~12 bytes] |
| T-502 | 859 | table tab:gpu-pipeline | ~3 GB, once | PCIe traffic, database encoding | deterministic | result | "Database encoding & CPU (dense bitmap) & CPU (CSR) $\to$ GPU (\texttt{atomicOr}) & Once ($\sim$3\,GB)" |
| T-503 | 860 | table tab:gpu-pipeline | None (0) | PCIe traffic, K=1 support count | deterministic | result | "$K{=}1$ support count & GPU & GPU (\texttt{popcount}) & None" |
| T-504 | 861 | table tab:gpu-pipeline | None (0) | PCIe traffic, K=1 frequency filter | deterministic | result | "$K{=}1$ frequency filter & CPU & GPU (\texttt{cp.where}) & None" |
| T-505 | 862 | table tab:gpu-pipeline | None (0) | PCIe traffic, K=2 candidate gen | deterministic | result | "$K{=}2$ candidate gen & CPU (enumerate pairs) & GPU (triangular number inverse) & None" |
| T-506 | 863 | table tab:gpu-pipeline | None (0) | PCIe traffic, K=2 support count | deterministic | result | "$K{=}2$ support count & GPU (AND + \texttt{\_\_popc}) & GPU (AND + \texttt{\_\_popcll}) & None" |
| T-507 | 864 | table tab:gpu-pipeline | None (0) | PCIe traffic, K=2 min-support filter | deterministic | result | "$K{=}2$ min-support filter & CPU & GPU (\texttt{atomicAdd} sparse output) & None" |
| T-508 | 865 | table tab:gpu-pipeline | None (0) | PCIe traffic, K>=3 candidate gen | deterministic | result | "$K{\geq}3$ candidate gen & CPU (prefix tree) & GPU (binary search + triangular inverse) & None" |
| T-509 | 866 | table tab:gpu-pipeline | None (0) | PCIe traffic, K>=3 support count | deterministic | result | "$K{\geq}3$ support count & GPU (AND + \texttt{\_\_popc}) & GPU (AND + \texttt{\_\_popcll}) & None" |
| T-510 | 867 | table tab:gpu-pipeline | None (0) | PCIe traffic, K>=3 filter + sort | deterministic | result | "$K{\geq}3$ filter + sort & CPU & GPU (\texttt{atomicAdd} + \texttt{lexsort}) & None" |
| T-511 | 868 | table tab:gpu-pipeline | ~4 | bytes PCIe traffic, loop control | deterministic | result | "Loop control & CPU & CPU reads \texttt{n\_results} & $\sim$4 bytes" |
| T-512 | 869 | table tab:gpu-pipeline | Once at end | PCIe traffic, result collection (bulk) | deterministic | result | "Result collection & CPU (per iteration) & GPU $\to$ CPU (bulk, once) & Once at end" |
| T-513 | 874 | appendix D (app:gpu-comparison) | 7.8× | CSR transfer saving at 0.01% density | deterministic | background | "CSR transfer savings over dense bitmaps depend on dataset sparsity: from $7.8\times$ at $0.01\%$ density to no savings at ${\geq}1\%$ density" |
| T-514 | 874 | appendix D (app:gpu-comparison) | 0.01 | % density (max CSR saving case) | method-parameter | background | "from $7.8\times$ at $0.01\%$ density to no savings at ${\geq}1\%$ density, where CSR overhead exceeds the dense representation" |
| T-515 | 874 | appendix D (app:gpu-comparison) | >=1 | % density (no CSR savings) | deterministic | background | "to no savings at ${\geq}1\%$ density, where CSR overhead exceeds the dense representation" |
| T-516 | 874 | appendix D (app:gpu-comparison) | ~1 | % density (proteome dataset) | deterministic | result | "For our proteome dataset (${\sim}1\%$ density), CSR provides a modest $1.4\times$ reduction." |
| T-517 | 874 | appendix D (app:gpu-comparison) | 1.4× | CSR reduction (proteome dataset) | deterministic | result | "For our proteome dataset (${\sim}1\%$ density), CSR provides a modest $1.4\times$ reduction." [cf. 167: ~15x; 886] |
| T-518 | 878 | table tab:memory-comparison caption | 8 | bytes per CSR entry | method-parameter | setup | "Memory footprint: dense bitmap ($\|D\| \times \|F_1\|$ bits packed) vs.\ CSR (8\,bytes/entry)." [cf. 167: two 64-bit integers per entry] |
| T-519 | 878 | table tab:memory-comparison caption | 214M | proteins (full bitvector matrix in AlphaFold proteome row) | deterministic | setup | "The ``AlphaFold proteome'' row uses the full 214M-protein bitvector matrix" [cf. 129: 214 million; 105/167/171: 205.6M] |
| T-520 | 878 | table tab:memory-comparison caption | 76.9M | multi-feature subset (mining campaign) | deterministic | setup | "the mining campaign (Section~\ref{sec:results}) operates on the 76.9M multi-feature subset" |
| T-521 | 885 | table tab:memory-comparison | 214M | transactions (AlphaFold proteome row header) | deterministic | setup | "\emph{AlphaFold proteome} ($\|D\|{=}214$M, $\|F_1\|{=}1{,}002$)" |
| T-522 | 885 | table tab:memory-comparison | 1,002 | frequent items (AlphaFold proteome row header) | deterministic | setup | "\emph{AlphaFold proteome} ($\|D\|{=}214$M, $\|F_1\|{=}1{,}002$)" |
| T-523 | 886 | table tab:memory-comparison | ~10 | items per transaction (Actual) | deterministic | result | "\quad Actual & $\sim$10 & 27\,GB & 19\,GB & $1.4\times$" |
| T-524 | 886 | table tab:memory-comparison | 27 | GB dense bitmap (Actual, 214M) | deterministic | setup | "\quad Actual & $\sim$10 & 27\,GB & 19\,GB & $1.4\times$" [cf. 171/402/769: ~26 GB] |
| T-525 | 886 | table tab:memory-comparison | 19 | GB CSR (Actual, 214M) | deterministic | result | "\quad Actual & $\sim$10 & 27\,GB & 19\,GB & $1.4\times$" [cf. 167: 5.1 GB; 171/440: ~3 GB] |
| T-526 | 886 | table tab:memory-comparison | 1.4× | Dense/CSR ratio (Actual) | deterministic | result | "\quad Actual & $\sim$10 & 27\,GB & 19\,GB & $1.4\times$" |
| T-527 | 888 | table tab:memory-comparison | 76.9M | transactions (Theoretical rows header) | deterministic | setup | "\emph{Theoretical} ($\|D\|{=}76.9$M, $\|F_1\|{=}1{,}000$)" |
| T-528 | 888 | table tab:memory-comparison | 1,000 | items (Theoretical rows header) | method-parameter | background | "\emph{Theoretical} ($\|D\|{=}76.9$M, $\|F_1\|{=}1{,}000$)" |
| T-529 | 889 | table tab:memory-comparison | 0.01 | % density (Theoretical row) | method-parameter | background | "\quad 0.01\% density & 1 & 9.6\,GB & 1.2\,GB & $7.8\times$" |
| T-530 | 889 | table tab:memory-comparison | 1 | items/txn at 0.01% density | deterministic | background | "\quad 0.01\% density & 1 & 9.6\,GB & 1.2\,GB & $7.8\times$" |
| T-531 | 889 | table tab:memory-comparison | 9.6 | GB dense (0.01% density) | deterministic | background | "\quad 0.01\% density & 1 & 9.6\,GB & 1.2\,GB & $7.8\times$" |
| T-532 | 889 | table tab:memory-comparison | 1.2 | GB CSR (0.01% density) | deterministic | background | "\quad 0.01\% density & 1 & 9.6\,GB & 1.2\,GB & $7.8\times$" |
| T-533 | 889 | table tab:memory-comparison | 7.8× | Dense/CSR ratio (0.01% density) | deterministic | background | "\quad 0.01\% density & 1 & 9.6\,GB & 1.2\,GB & $7.8\times$" |
| T-534 | 890 | table tab:memory-comparison | 0.1 | % density (Theoretical row) | method-parameter | background | "\quad 0.1\% density  & 10  & 9.6\,GB & 6.8\,GB & $1.4\times$" |
| T-535 | 890 | table tab:memory-comparison | 10 | items/txn at 0.1% density | deterministic | background | "\quad 0.1\% density  & 10  & 9.6\,GB & 6.8\,GB & $1.4\times$" |
| T-536 | 890 | table tab:memory-comparison | 9.6 | GB dense (0.1% density) | deterministic | background | "\quad 0.1\% density  & 10  & 9.6\,GB & 6.8\,GB & $1.4\times$" |
| T-537 | 890 | table tab:memory-comparison | 6.8 | GB CSR (0.1% density) | deterministic | background | "\quad 0.1\% density  & 10  & 9.6\,GB & 6.8\,GB & $1.4\times$" |
| T-538 | 890 | table tab:memory-comparison | 1.4× | Dense/CSR ratio (0.1% density) | deterministic | background | "\quad 0.1\% density  & 10  & 9.6\,GB & 6.8\,GB & $1.4\times$" |
| T-539 | 891 | table tab:memory-comparison | 1 | % density (Theoretical row) | method-parameter | background | "\quad 1\% density    & 100 & 9.6\,GB & 62.1\,GB & $0.15\times$" |
| T-540 | 891 | table tab:memory-comparison | 100 | items/txn at 1% density | deterministic | background | "\quad 1\% density    & 100 & 9.6\,GB & 62.1\,GB & $0.15\times$" |
| T-541 | 891 | table tab:memory-comparison | 9.6 | GB dense (1% density) | deterministic | background | "\quad 1\% density    & 100 & 9.6\,GB & 62.1\,GB & $0.15\times$" |
| T-542 | 891 | table tab:memory-comparison | 62.1 | GB CSR (1% density) | deterministic | background | "\quad 1\% density    & 100 & 9.6\,GB & 62.1\,GB & $0.15\times$" |
| T-543 | 891 | table tab:memory-comparison | 0.15× | Dense/CSR ratio (1% density) | deterministic | background | "\quad 1\% density    & 100 & 9.6\,GB & 62.1\,GB & $0.15\times$" |

---

## SECTION B — METHOD RECONSTRUCTION FROM THE PAPER (as stated; not verified)

**B.1 Data source and version**
- Structure database: AlphaFold Protein Structure Database, cited as `varadi2024`, "provides predicted structures for 214 million proteins across UniProt" (l.129). Abstract/intro say "over 200 million" (l.91, l.101).
- Annotation source: "obtained protein annotations (Pfam domain assignments, Gene Ontology (GO) terms, and predicted local distance difference test (pLDDT) confidence scores) from the UniProt~\cite{uniprot2023} cross-reference pipeline (release 2025\_01, accessed February 2026)" (l.129). Repeated: "our results reflect UniProt TrEMBL release 2025\_01" (l.478); "regenerable from UniProt release 2025\_01 via the released feature-extraction pipeline" (l.529).
- Protein set: "We processed 205,620,298 proteins from UniProt TrEMBL" (l.129). Note that Table `tab:memory-comparison` instead describes an "AlphaFold proteome" row with "$|D|{=}214$M" and "the full 214M-protein bitvector matrix" (l.878, l.885), while the bitvector-size statements use "205.6M proteins" (l.171).
- Caveat stated: "Exact itemset counts and $K$-distributions may vary with different UniProt releases due to ongoing annotation updates." (l.129); "GO annotations are updated monthly" (l.478).

**B.2 Number of proteins and selection filter**
- "Of the 205,620,298 total proteins, 76,890,945 (37.4\%) have more than one annotated feature under this vocabulary and form the transaction set mined in this work." (l.152).
- Complement: "Our analysis excludes 128.7 million proteins (62.6\%) with only a single annotated feature." (l.480). No statement is made about proteins with zero vocabulary features (the two percentages 37.4 % + 62.6 % = 100 %).
- Every headline mining figure (76.9M / 76,890,945 transactions) refers to this multi-feature subset (l.91, 115, 120, 152, 167, 425, 440, 446, 843, 878).

**B.3 Base vocabulary definition**
- "The feature vocabulary comprises the 500 most frequent Pfam domains and the 500 most frequent GO terms (drawn from 24{,}291 Pfam and 25{,}993 GO families observed across the corpus), together with 6 pLDDT confidence bins, for 1{,}006 defined items; the minimum support threshold of 8 proteins is applied during \emph{mining}, not during vocabulary construction." (l.129).
- Table `tab:features` (l.134–146): Pfam 500 defined / 500 frequent; GO 500 / 500; pLDDT bins 6 / 2; total 1,006 / 1,002. "1,002 pass the support threshold of 8 proteins (all 1,000 Pfam/GO items plus 2 pLDDT bins)" (l.134).
- pLDDT binning: the only bin edges stated anywhere are for the item in the K=22 itemset: "plddt\_mean & Medium confidence (70--90)" (l.327). The item name is `plddt_mean`. The other five bins' edges, and which second bin is frequent, are **not** stated.
- Items are integer identifiers: "Each protein is represented as a transaction: a set of integer item identifiers corresponding to its annotated features." (l.131). Feature-to-item mapping promised as a released artefact (l.529).
- Frequent-item count 1,002 is reused as the K=1 row of both distributions (l.268, l.376) and as the bitvector column count (l.171).

**B.4 Feature extraction / preprocessing**
- "Feature extraction processed 150\,GB of compressed annotation data in 63\,minutes; this preprocessing is separate from the 7.3\,minute mining runtime reported below." (l.129). Transactions are stored as Parquet (l.161, l.167, l.764). Script named for regenerating the K=22 proteins: `analyze\_k22\_proteins.py` (l.333).

**B.5 Data representation**
- CSR/COO built by streaming the Parquet file: "For the 76.9M-protein mining subset, the resulting matrix contains 316~million non-zero entries occupying ${\sim}5.1$\,GB in coordinate format (two 64-bit integers per entry), a ${\sim}15\times$ reduction from the ${\sim}77$\,GB dense representation of that same subset" (l.167); "${\sim}40\times$ smaller" than the ~206 GB full-set dense matrix; bit-packed dense "${\sim}26$\,GB" (l.167).
- GPU bitvectors: "With 205.6M proteins and 1,002 features, the total bitvector matrix occupies ${\sim}26$\,GB of GPU memory, built directly on-GPU from the CSR column indices via a single host-to-device transfer (${\sim}3$\,GB)." (l.171); "${\sim}3$\,GB of CSR column indices versus ${\sim}10$\,GB for the equivalent bit-packed dense bitmap of the 76.9M mining subset" (l.440); Table `tab:gpu-pipeline`: "Database encoding ... CPU (CSR) $\to$ GPU (\texttt{atomicOr}) & Once ($\sim$3\,GB)" (l.859). Table `tab:memory-comparison` uses "CSR (8\,bytes/entry)" and lists 27 GB dense / 19 GB CSR for 214M proteins (l.878, l.886).
- Support = popcount(AND of bitvectors) (l.173–179); 64-bit `__popcll` (l.797, l.834).

**B.6 Mining algorithm and settings**
- Algorithm: level-wise Apriori (chosen over FP-Growth, l.156), GPU-resident bitvectors across all K-levels (l.183, l.196). Pseudocode `alg:etminer` (l.760–785): `min_count <- ceil(sigma * n)` (l.766); F1 by streaming count (l.767); CSR built over F1 (l.768); bitvectors ~26 GB VRAM (l.769); freq_1 = popcount >= min_count (l.772); freq_2 = FUSED_K2 (l.773); loop `k <- 3`, `while |freq_{k-1}| >= k`: BUILD_PREFIX_GROUPS_GPU, FUSED_K3PLUS (l.775–781); single BULK_TRANSFER at the end (l.782). Prose version (l.186–190): "K=1: Count proteins for each feature; retain those above the support threshold. K=2: Test all feature pairs; retain frequent pairs. K>=3: Group candidates by shared prefix, test each group, retain those meeting the threshold."
- Kernel details (l.799–811): fused candidate-generation/count/filter kernels; K=2 uses a triangular-number inverse mapping, one candidate pair per thread; K>=3 prefix groups by boundary detection on GPU, early termination when an intermediate AND is zero; filtering via `atomicAdd`, sorting via `lexsort` on GPU (l.835–836, l.864–867). Per-level PCIe traffic "~12 bytes" (l.443, 808–810, 839) or "~4 bytes" (l.852, 868); total "~264 bytes" over 22 levels (l.813).
- Support thresholds / min counts (Table `tab:campaign`, l.224–230): Base 0.1 % / 76,891; Super 0.01 % / 7,689; Power 0.001 % / 768; Blitz 0.0001 % / 77; Ultra 0.00002 % (nominal) / 16; Opus 0.00001 % (nominal) / 8. "Support percentages for Ultra and Opus are nominal (rounded); actual thresholds correspond to minimum protein counts of 16 and 8 respectively." (l.216). Null-model section states min_count = 769 for 0.001 % (l.362, l.393).
- Methods per run: Base/Super/Power "Streaming SON"; Blitz/Ultra/Opus "Direct CSR$\to$GPU" (l.224–230). SON: "For datasets exceeding GPU memory, the SON algorithm~\cite{savasere1995} partitions data into chunks mined independently." (l.196). SON is described as approximate/lossy (l.238, l.404); the exhaustive reference at 0.001 % is a separate Direct CSR→GPU run (475,865 itemsets, K=14; l.238, l.393).
- K max: no explicit cap stated; the loop terminates when `|freq_{k-1}| < k` (l.776). Reported maxima: 9, 13, 13, 19, 20, 22 (l.224–230). Structural ceiling claim: "no protein carries more than 22 vocabulary features" (l.335).
- Sparse/closed settings: "ET-miner implements closed itemset filtering as a post-processing step ... We report the complete pattern space here" (l.482) — i.e. reported counts are all frequent itemsets, not closed/maximal.
- Execution path: "three execution paths ... The GPU-accelerated path is the primary contribution and was used for all results reported in this paper." (l.156).
- Multi-GPU: capability described ("replicate bitvectors on each device and distribute candidates by index range", l.196; "memory-aware chunk sizing", l.446; "Candidate-index splitting", l.841) but "all experiments in this paper use a single GPU" (l.196) and "All experiments in this paper used a single H100; the 76.9M multi-feature proteins were mined without manual tuning." (l.446).

**B.7 Hardware**
- "All experiments were performed on a single NVIDIA H100 80\,GB SXM5 with 128\,GB host RAM" (l.213). GPU count = 1 throughout (l.91, 120, 196, 216, 362, 402, 425, 446, 496). **CPU model, core count, storage, and PCIe generation are not stated.** Null model ran "on the same H100" (l.362).

**B.8 Software versions**
- "running Python~3.10, CuPy~13.0, NumPy~1.26, CUDA~12.4, and Ubuntu~22.04" (l.213). No versions given for Polars/Rust/Parquet tooling, InterPro/Pfam release, or GO release; only UniProt release 2025_01 (l.129). Code repository: `https://github.com/Et9797/ET-miner` (l.500, l.529); Zenodo DOI `10.5281/zenodo.18674353` (l.500, l.529).

**B.9 Timings reported**
- Mining wall-clock per run: 1.9, 4.3, 18.1, 2.0, 4.7, 7.3 min (l.224–230); Opus 7.3 min = headline "mining only" time (l.91, 120, 129, 425, 496). Controlled 0.001 % comparison: Direct 50.7 s vs SON 1,085.6 s = 21× (l.238, 404). Feature extraction 63 min (l.129, 496). Null model: 662 s total for 5 permutations, "~130 s" per permutation (l.362, l.393).

**B.10 Statistics reported**
- Null model (l.362): "all transactions were exploded into (protein, feature) pairs, a Fisher--Yates shuffle was applied to the feature column---preserving per-protein feature counts---and duplicate features within each protein were removed before re-mining at the 0.001\% support threshold ($\text{min\_count}{=}769$). Five permutations were performed (seed~42, total runtime 662\,s on the same H100)."
- Reported per K (Table `tab:null-model`, l.376–382): Bio count, null mean μ, null σ, Z = standardized effect "with $\sigma$ estimated from only 5 permutations (equivalently, $t$-statistics with 4 degrees of freedom)" (l.367), and p (1.0 for K=1–3, ≈0 for K=4–6, <0.45 for K=7–14 via "the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5"). Headline: "$Z{>}3{,}700$ for $K{=}4$--$6$" (l.91, l.498), Z = −987 / −143 for K=2/3 (l.391), +3,791 / +7,402 / +71,728 for K=4/5/6 (l.379–381).
- Scope limits stated by the paper: validates only the 0.001 % threshold, not Opus (l.393); aggregate K-distribution only, no per-itemset test (l.474); "A full validation with 100+ permutations would be needed to establish $p < 0.01$ bounds" (l.393).
- Other reported statistics: K-distribution (Table `tab:kdist`), peak K=9 with 3,529,257 (13.14 %) (l.249); SON miss rate 95.2 % (l.238, 404); 124× more itemsets / 9× faster Blitz vs Power (l.228, 238, 243); 5.1× more transactions than GMiner (l.411).

**B.11 K=22 itemset and intermediate patterns (as stated)**
- One K=22 itemset, exactly 8 supporting proteins, each carrying exactly 22 vocabulary features; composition in Table `tab:k22` (2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT bin) (l.286–331); 0 GO parent–child pairs, 1 definitional link (GO:0005524 ← PF00270 via InterPro2GO), 21 independent features (l.333); the 8 proteins are not deposited and are regenerable only via `analyze_k22_proteins.py` from UniProt 2025_01 (l.333).
- Highlighted patterns: K=19 (187 proteins), K=17 (~611; PF07714, PF00017, PF00018), K=13 (~11,000), K=12 (~10,500; PF00905, PF00912), K=11 (~16,000) — "order-of-magnitude estimates read from the decoded-pattern summaries" (l.341–357, footnote l.353). The full member lists of these five itemsets are **not** given.

**B.12 Artefacts promised**
- "the summary result artefacts (per-run mining logs, $K$-distribution tables, null-model outputs, and the feature-to-item mapping) will be released ... The full intermediate transaction matrix is not deposited owing to its size" (l.529).

---

## SECTION C — TABLES AND FIGURES INDEX

Figure files are referenced but absent from the repository (no `paper/figures/` directory; no matching PDFs anywhere under `/root/projects/ET-Miner`), so only captions can be transcribed for figures.

### Figure `fig:concept` — lines 109–118 (`figures/concept_figure.pdf`, missing)
Caption (l.112–116): "Frequent itemset mining applied to protein annotations. (a) Each protein is treated as a transaction whose items are its annotated Pfam domains, GO terms, and structural properties. A $K{=}4$ pattern (yellow highlight) is shared by three proteins. (b) The transaction database as a boolean matrix (76.9M proteins $\times$ 1,002 features). Support counting for any candidate itemset reduces to a column-wise AND followed by a population count."
Numbers: K=4 and 3 proteins (schematic), 76.9M × 1,002 (rows T-026–T-029).

### Figure `fig:architecture` — lines 158–163 (`figures/architecture.pdf`, missing)
Caption (l.161): "ET-miner pipeline: Parquet transactions are compressed to sparse format and converted to GPU-resident bitvectors. Support counting executes on-GPU; prefix group construction and result collection cross the PCIe bus at each $K$-level." No numbers.

### Table `tab:features` — lines 133–150
Caption (l.134): "Feature vocabulary composition: 1,006 items are defined (6 pLDDT confidence bins, 500 Pfam domains, 500 GO terms) and 1,002 pass the support threshold of 8 proteins (all 1,000 Pfam/GO items plus 2 pLDDT bins)."

| Feature Type | Source | Defined | Frequent |
|---|---|---|---|
| Pfam domains | InterPro/Pfam | 500 | 500 |
| GO terms | Gene Ontology | 500 | 500 |
| pLDDT confidence bins | AlphaFold | 6 | 2 |
| **Total** | | **1,006** | **1,002** |

Rows T-049–T-064 (caption T-049–T-056, cells T-057–T-064).

### Table `tab:campaign` — lines 215–236 (table*), footnote l.235
Caption (l.216): "Mining campaign results across six support thresholds on a single NVIDIA H100 80\,GB. Support percentages for Ultra and Opus are nominal (rounded); actual thresholds correspond to minimum protein counts of 16 and 8 respectively."

| Run | Support | Min. proteins | Itemsets | Max K | Time | Method | Key transition |
|---|---|---|---|---|---|---|---|
| Base | 0.1 % | 76,891 | 5,305 | 9 | 1.9 min | Streaming SON | |
| Super | 0.01 % | 7,689 | 51,124 | 13 | 4.3 min | Streaming SON | |
| Power | 0.001 % | 768 | 22,846 | 13 | 18.1 min | Streaming SON | |
| Blitz | 0.0001 % | 77 | 2,841,280 | 19 | 2.0 min | Direct CSR→GPU | 9× faster, 124× more itemsets † |
| Ultra | 0.00002 % | 16 | 14,558,875 | 20 | 4.7 min | Direct CSR→GPU | |
| Opus | 0.00001 % | 8 | 26,849,505 | 22 | 7.3 min | Direct CSR→GPU | |

Footnote † (l.235): "Compares Power (0.001\%) to Blitz (0.0001\%), varying both method and threshold. Controlled same-support comparison: $21\times$ (\S\ref{sec:results})."
Rows T-103–T-148 (caption T-103–T-107, cells T-108–T-145, footnote T-146–T-148).

### Figure `fig:campaign` — lines 240–245 (`figures/mining_campaign.pdf`, missing)
Caption (l.243): "Mining campaign across six support thresholds. Bars show itemsets discovered (log scale); line shows wall-clock time. The SON$\to$Direct GPU transition at 0.001\%$\to$0.0001\% yields $9\times$ speedup despite $10\times$ lower support; a controlled comparison at identical support shows $21\times$ (Section~\ref{sec:results})."
Rows T-160–T-165.

### Figure `fig:kdist` — lines 251–256 (`figures/k_distribution.pdf`, missing)
Caption (l.254): "$K$-distribution of frequent itemsets at 0.00001\% support (26.8M total). The unimodal distribution peaks at $K{=}9$ (3.53M itemsets). A single itemset at maximum depth is shared by exactly 8 proteins."
Rows T-171–T-176.

### Table `tab:kdist` — lines 258–282
Caption (l.259): "$K$-distribution at 0.00001\% support (26.8M total)."

| K | Itemsets | % | K | Itemsets | % |
|---|---|---|---|---|---|
| 1 | 1,002 | 0.00 | 12 | 1,996,772 | 7.44 |
| 2 | 73,786 | 0.27 | 13 | 1,259,045 | 4.69 |
| 3 | 452,777 | 1.69 | 14 | 679,471 | 2.53 |
| 4 | 1,184,461 | 4.41 | 15 | 310,527 | 1.16 |
| 5 | 1,974,126 | 7.35 | 16 | 118,659 | 0.44 |
| 6 | 2,626,332 | 9.78 | 17 | 37,261 | 0.14 |
| 7 | 3,118,459 | 11.61 | 18 | 9,375 | 0.03 |
| 8 | 3,442,954 | 12.82 | 19 | 1,818 | 0.01 |
| **9** | **3,529,257** | **13.14** | 20 | 255 | <0.01 |
| 10 | 3,293,612 | 12.27 | 21 | 23 | <0.01 |
| 11 | 2,739,532 | 10.20 | 22 | 1 | <0.01 |

(Arithmetic note, extraction only: the 22 itemset cells sum to 26,849,505, equal to the Opus "Itemsets" cell at l.230.) Rows T-177–T-222 (caption T-177–T-178, cells T-179–T-222).

### Table `tab:k22` — lines 288–331
Caption (l.289): "$K{=}22$ itemset: 22 co-occurring features in 8 proteins."

| Category | ID | Description |
|---|---|---|
| *Pfam domains (2)* | PF00270 | DEAD/DEAH helicase N-terminal |
| | PF00271 | Helicase C-terminal |
| *Molecular Function (8)* | GO:0005524 | ATP binding |
| | GO:0016787 | Hydrolase activity |
| | GO:0000287 | Magnesium ion binding |
| | GO:0003697 | ssDNA binding |
| | GO:0003724 | RNA helicase activity |
| | GO:0003725 | dsRNA binding |
| | GO:0003678 | DNA helicase activity |
| | GO:0000978 | RNA Pol II regulatory binding |
| *Biological Process (4)* | GO:0030154 | Cell differentiation |
| | GO:0045087 | Innate immune response |
| | GO:0051607 | Defense response to virus |
| | GO:0034605 | Cellular response to heat |
| *Cellular Component (7)* | GO:0005737 | Cytoplasm |
| | GO:0005829 | Cytosol |
| | GO:0005634 | Nucleus |
| | GO:0005739 | Mitochondrion |
| | GO:0030424 | Axon |
| | GO:0030425 | Dendrite |
| | GO:0016607 | Nuclear speckle |
| *Structural property (1)* | plddt_mean | Medium confidence (70–90) |

(2 + 8 + 4 + 7 + 1 = 22 entries.) Rows T-227–T-257 (caption T-227–T-229, cells T-230–T-257).

### Table `tab:null-model` — lines 366–389, footnote l.388
Caption (l.367): "Biological vs.\ null model $K$-distributions at 0.001\% support (5 permutations). All $K{\geq}4$ patterns are strongly enriched. The $Z$ column reports standardized effect sizes with $\sigma$ estimated from only 5 permutations (equivalently, $t$-statistics with 4 degrees of freedom); the very large magnitudes indicate strong enrichment but are not calibrated large-sample $Z$-scores. $K{\geq}7$ patterns are absent from all 5 null runs ($p < 0.45$, the exact one-sided binomial 95\% upper bound $1-0.05^{1/5}$ for 0 of 5)."

| K | Bio | Null μ | Null σ | Z* | p | |
|---|---|---|---|---|---|---|
| 1 | 1,002 | 1,002 | 0.0 | 0.0 | 1.0 | preserved |
| 2 | 22,019 | 63,702 | 42.2 | −987 | 1.0 | depleted |
| 3 | 73,205 | 79,134 | 41.5 | −143 | 1.0 | depleted |
| 4 | 108,059 | 25,468 | 21.8 | +3,791 | ≈0 | enriched |
| 5 | 104,239 | 1,992 | 13.8 | +7,402 | ≈0 | enriched |
| 6 | 78,596 | 22 | 1.1 | +71,728 | ≈0 | enriched |
| 7–14 | 88,745 | 0 | 0.0 | — | <0.45 | enriched |

Footnote * (l.388): "Estimated from 5 permutations ($t$-statistic, 4 df); large magnitude indicates strong enrichment, not a calibrated large-sample $Z$."
(Arithmetic note, extraction only: the Bio column sums to 475,865, equal to the Direct CSR→GPU count at 0.001 % quoted at l.238/l.393/l.404.) Rows T-298–T-345 (caption T-298–T-308, cells T-309–T-343, footnote T-344–T-345).

### Table `tab:scale-comparison` — lines 410–429
Caption (l.411): "Scale comparison of GPU and distributed FIM systems. ET-miner processes $5.1\times$ more transactions than the largest prior single-machine GPU system (GMiner, 15M synthetic transactions; 1.7M real) and mines exhaustively to $K{=}22$, deeper than any previously reported GPU FIM result."

| System | Transactions | Items | Max K | Hardware | Time |
|---|---|---|---|---|---|
| Borgelt [borgelt2003] | 100K | 500 | ~10 | 1× CPU | 1–100 s |
| Fang [fang2009] | 100K | 1K | ~5 | 1× GTX 280 | 1–10 s |
| GMiner [chon2018] | 15M | 20K | ~30 | 4× GTX 1080 | 20–150 s |
| BIGMiner [chon2018b] | 100M | 100K | — | 30× servers | 1–20K s |
| ET-miner | **76.9M** | 1,002 | **22** | 1× H100 | 7.3 min |

Rows T-390–T-418 (caption T-390–T-393, cells T-394–T-418).

### Algorithm `alg:etminer` — lines 760–785 (Appendix B)
Caption (l.761): "ET-miner GPU-Accelerated Apriori". Require: Transactions T (Parquet), minimum support σ. Ensure: frequent itemsets F with support values. Steps: n ← |T|; min_count ← ⌈σ·n⌉ (l.766); F1 ← {i : count(i) ≥ min_count} [Streaming] (l.767); CSR ← BuildCSR(T, F1) [O(nnz) memory] (l.768); B ← GPU_CSR_TO_BITVEC(CSR) [~26 GB VRAM] (l.769); free(CSR); results ← []; freq1 ← {c : popcount(B[c]) ≥ min_count}; freq2 ← FUSED_K2(B, freq1, min_count); k ← 3; while |freq_{k−1}| ≥ k: G ← BUILD_PREFIX_GROUPS_GPU(freq_{k−1}); freq_k ← FUSED_K3PLUS(B, G, min_count); k ← k+1; F ← BULK_TRANSFER(results) [Single transfer]; return F.
Rows T-477–T-481.

### Table `tab:gpu-arch` — lines 823–846 (Appendix D)
Caption (l.825): "Architectural comparison of GPU-based frequent itemset mining systems. ET-miner is the only system that performs candidate generation on-GPU and maintains full GPU residency across Apriori iterations."

| Aspect | GMiner (2018) | GMiner++ (2024) | ET-miner |
|---|---|---|---|
| Data representation | Dense bitmap (\|F1\| × \|D\|) | Dense bitmap | CSR → GPU-built bitvector |
| Candidate generation | CPU only | CPU only | **GPU (in-kernel)** |
| Support counting | `__popc` (32-bit) | `__popc` (32-bit) | **`__popcll` (64-bit)** |
| Min-support filter | CPU (after D2H transfer) | CPU | **GPU (in-kernel `atomicAdd`)** |
| Result sorting | CPU | CPU | **GPU (`lexsort`)** |
| Early termination | No | Unknown | **Yes** (skip when AND = 0) |
| Reduction strategy | Shared-mem `parallelReduction` | Unknown | **Warp shuffle + shared mem** |
| PCIe per K-level | O(\|TB\| + \|C_L\| + \|PS\|) | O(bit array blocks) | **~12 bytes** |
| GPU residency | No (stream each iteration) | No | **Yes (bitvectors GPU-resident)** |
| Multi-GPU strategy | Transaction bitmap sharing | Replicating bit array blocks | **Candidate-index splitting** |
| Multi-GPU auto-dispatch | Manual | Manual | **Automatic (memory-aware)** |
| Max tested dataset | 1.7M txns (real) | Not reported | **76.9M txns (proteome)** |

Numeric rows T-495–T-500; the categorical cells are claims about the implementation (candidate generation in-kernel, `atomicAdd` filtering, `lexsort`, early termination, warp-shuffle reduction, candidate-index multi-GPU splitting, automatic memory-aware dispatch) recorded here for the audit.

### Table `tab:gpu-pipeline` — lines 850–872 (Appendix D)
Caption (l.852): "Pipeline stage execution location. ET-miner performs all compute-intensive stages on-GPU, with CPU involvement limited to loop control ($\sim$4 bytes per $K$-level)."

| Pipeline Stage | GMiner / GMiner++ | ET-miner (GPU-accelerated) | PCIe Traffic (ET-miner) |
|---|---|---|---|
| Database encoding | CPU (dense bitmap) | CPU (CSR) → GPU (`atomicOr`) | Once (~3 GB) |
| K=1 support count | GPU | GPU (`popcount`) | None |
| K=1 frequency filter | CPU | GPU (`cp.where`) | None |
| K=2 candidate gen | CPU (enumerate pairs) | GPU (triangular number inverse) | None |
| K=2 support count | GPU (AND + `__popc`) | GPU (AND + `__popcll`) | None |
| K=2 min-support filter | CPU | GPU (`atomicAdd` sparse output) | None |
| K≥3 candidate gen | CPU (prefix tree) | GPU (binary search + triangular inverse) | None |
| K≥3 support count | GPU (AND + `__popc`) | GPU (AND + `__popcll`) | None |
| K≥3 filter + sort | CPU | GPU (`atomicAdd` + `lexsort`) | None |
| Loop control | CPU | CPU reads `n_results` | ~4 bytes |
| Result collection | CPU (per iteration) | GPU → CPU (bulk, once) | Once at end |

Rows T-501–T-512 (caption T-501, cells T-502–T-512).

### Table `tab:memory-comparison` — lines 876–894 (Appendix D)
Caption (l.878): "Memory footprint: dense bitmap ($|D| \times |F_1|$ bits packed) vs.\ CSR (8\,bytes/entry). Ratio $=$ Dense\,/\,CSR; values ${>}1$ indicate CSR savings. The ``AlphaFold proteome'' row uses the full 214M-protein bitvector matrix; the mining campaign (Section~\ref{sec:results}) operates on the 76.9M multi-feature subset."

| Dataset | Items/txn | Dense | CSR | Ratio |
|---|---|---|---|---|
| *AlphaFold proteome* (\|D\| = 214M, \|F1\| = 1,002) | | | | |
| Actual | ~10 | 27 GB | 19 GB | 1.4× |
| *Theoretical* (\|D\| = 76.9M, \|F1\| = 1,000) | | | | |
| 0.01 % density | 1 | 9.6 GB | 1.2 GB | 7.8× |
| 0.1 % density | 10 | 9.6 GB | 6.8 GB | 1.4× |
| 1 % density | 100 | 9.6 GB | 62.1 GB | 0.15× |

Rows T-518–T-543 (caption T-518–T-520, cells T-521–T-543).

### Other tabular material
- Author-information `tabular` (l.504–506): "E. Ahmic & Indep. Researcher (MSc. Drug Innovation, UU)" — no numeric claims.
- Support-counting display equation (l.174–178): support({A,B,C}) = popcount(bitvec[A] AND bitvec[B] AND bitvec[C]) — notation only.

---

## SECTION D — TOTALS

Total claim rows: **543** (T-001 … T-543).

By category:

| category | rows |
|---|---|
| deterministic | 343 |
| method-parameter | 90 |
| external-fact | 55 |
| hardware-dependent | 41 |
| software | 14 |
| **total** | **543** |

By claim type:

| claim type | rows |
|---|---|
| result | 304 |
| setup | 87 |
| background | 85 |
| parameter | 67 |
| **total** | **543** |

Cross-tabulation (category × claim type):

| category \ type | result | parameter | setup | background |
|---|---|---|---|---|
| deterministic | 278 | 0 | 41 | 24 |
| hardware-dependent | 26 | 0 | 14 | 1 |
| method-parameter | 0 | 67 | 16 | 7 |
| external-fact | 0 | 0 | 3 | 52 |
| software | 0 | 0 | 13 | 1 |

Rows by location (from the `section` column; ID ranges are contiguous in document order):

| section | rows | ID range |
|---|---|---|
| front matter | 1 | T-001–T-001 |
| abstract | 11 | T-002–T-012 |
| intro | 18 | T-013–T-034 |
| figure fig:concept caption | 4 | T-026–T-029 |
| methods | 41 | T-035–T-091 |
| table tab:features caption | 8 | T-049–T-056 |
| table tab:features | 8 | T-057–T-064 |
| results | 58 | T-092–T-285 |
| table tab:campaign caption | 5 | T-103–T-107 |
| table tab:campaign | 38 | T-108–T-145 |
| table tab:campaign footnote | 3 | T-146–T-148 |
| figure fig:campaign caption | 6 | T-160–T-165 |
| figure fig:kdist caption | 6 | T-171–T-176 |
| table tab:kdist caption | 2 | T-177–T-178 |
| table tab:kdist | 44 | T-179–T-222 |
| table tab:k22 caption | 3 | T-227–T-229 |
| table tab:k22 | 28 | T-230–T-257 |
| footnote | 1 | T-279–T-279 |
| results (sec:null-model) | 37 | T-286–T-370 |
| table tab:null-model caption | 11 | T-298–T-308 |
| table tab:null-model | 35 | T-309–T-343 |
| table tab:null-model footnote | 2 | T-344–T-345 |
| discussion | 48 | T-371–T-447 |
| table tab:scale-comparison caption | 4 | T-390–T-393 |
| table tab:scale-comparison | 25 | T-394–T-418 |
| discussion (sec:limitations) | 11 | T-448–T-458 |
| conclusion | 12 | T-459–T-470 |
| back matter (data availability) | 2 | T-471–T-472 |
| appendix A (app:complexity) | 4 | T-473–T-476 |
| appendix B (alg:etminer) | 5 | T-477–T-481 |
| appendix C (app:gpu-details) | 13 | T-482–T-494 |
| table tab:gpu-arch | 6 | T-495–T-500 |
| table tab:gpu-pipeline caption | 1 | T-501–T-501 |
| table tab:gpu-pipeline | 11 | T-502–T-512 |
| appendix D (app:gpu-comparison) | 5 | T-513–T-517 |
| table tab:memory-comparison caption | 3 | T-518–T-520 |
| table tab:memory-comparison | 23 | T-521–T-543 |
| **total** | **543** | |

---

## SECTION E — CROSS-REFERENCE OBSERVATIONS (extraction aid only; no correctness judgment)

Places where the paper states apparently the same quantity with different values or definitions — listed so the auditor knows which variant a reproduction should target:

1. **Power min count**: 768 (Table `tab:campaign`, l.226) vs `min_count = 769` (l.362, l.393).
2. **Results at 0.001 %**: SON "Power" row 22,846 itemsets, K max 13, 18.1 min (l.226) vs exhaustive Direct CSR→GPU 475,865 itemsets, K = 14, 50.7 s (l.238, l.393, l.404). The paper explicitly says the null-model reference is the latter (l.393).
3. **Protein universe size**: "over 200 million" (l.91, 101); "214 million" (l.129); 214M used for the "AlphaFold proteome" memory row (l.878, 885); 205,620,298 / 205.6M processed (l.105, 129, 152, 167, 171); 76,890,945 / 76.9M mined (l.152 and throughout).
4. **CSR / transferred size**: ~5.1 GB "coordinate format (two 64-bit integers per entry)" for 316M nnz (l.120, 167, 402); ~3 GB "CSR column indices" transferred (l.171, 440, 859); 19 GB CSR at "8 bytes/entry" for 214M proteins (l.878, 886). Bytes per entry stated as 16 (l.167) and 8 (l.878).
5. **Bitvector / bit-packed dense size**: ~26 GB (l.167, 171, 402, 769; 205.6M proteins) vs 27 GB (l.886; 214M proteins) vs ~10 GB for the 76.9M subset (l.440) vs 9.6 GB dense for 76.9M × 1,000 (l.889–891).
6. **Dense-to-CSR reduction factor**: ~15× (same subset, l.167, 402), ~40× (vs full-set dense, l.167) vs 1.4× ("modest", l.874, 886) — different bases (1 byte/bool dense vs bit-packed dense).
7. **Per-level PCIe traffic**: "prefix group construction and result collection cross the PCIe bus at each K-level" (l.161, 183, 193, 196, 402) vs "~12 bytes" (l.443, 808–810, 839) vs "~4 bytes" (l.852, 868); total "~264 bytes" over 22 levels (l.813).
8. **Support threshold semantics**: prose "retain those above the support threshold" (l.186–187) vs pseudocode `>= min_count` (l.767, 772) and `min_count = ceil(sigma * n)` (l.766); Ultra/Opus percentages described as nominal, actual thresholds 16 and 8 (l.216).
9. **Z headline**: "Z > 3,700 for K = 4–6" (l.91, 498) vs table values +3,791, +7,402, +71,728 (l.379–381), explicitly described as t-statistics with 4 df (l.367, 388, 391).
10. **Null-model K=6 count**: "at most 23 itemsets at K=6" (l.364) vs mean 22, σ 1.1 (l.381, 391).
11. **Null-model runtime**: 662 s total for 5 permutations (l.362) vs "~130 s" per permutation (l.393).
12. **Theoretical memory rows** use |F1| = 1,000 (l.888) while the actual vocabulary is 1,002 (l.885 and throughout).
13. **"100-million-transaction scale"** (l.402, 496) vs 76.9M transactions actually mined (l.152, 425).
14. **pLDDT bins**: 6 defined, 2 frequent (l.129, 134, 144); only one bin's edges are given ("Medium confidence (70--90)", l.327); the identity of the second frequent bin and the other bin edges are absent.
15. **Supporting-protein counts of intermediate patterns** are stated as approximate order-of-magnitude readings (footnote l.353) except K = 19's "187 proteins" (l.341), which is given without a tilde.
