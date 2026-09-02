# CLAIMS.md — every numeric value/claim extracted from the paper, the reviews, and the log files

All values below are CLAIMS to verify, never ground truth. Each row carries a stable ID
(T-, R1-, R2-, L-, L2- prefixes), the source file and line, the value, its category
(deterministic / hardware-dependent / method-parameter / external-fact / software), and a
verbatim context fragment. The five extraction reports (with their method sections,
tables indexes, review-flagged inconsistencies and totals) are kept verbatim under
`runs/20260902T0000Z/phase1/`; this file concatenates their claim tables and indexes them.
Verdicts live in COMPARISON_REPORT.md, joined through `runs/20260902T0000Z/phase4/`.

## Index

- **T-xxx** — paper/et_miner_proteome.tex (V1 preprint): 543 claim rows (T-001 … T-543) — source report `runs/20260902T0000Z/phase1/claims_tex.md`
- **R1-xxx** — paper reviews part 1 (PAPER_V2_REVIEW.md, peer_review_jul12.md, review_b1_hostile.md): 835 claim rows (R1-001 … R1-835) — source report `runs/20260902T0000Z/phase1/claims_reviews_part1.md`
- **R2-xxx** — paper reviews part 2 (review_b2_results.md, revision_notes_b3.tex, senior_review_jun01.md, senior_review_mar23.md): 505 claim rows (R2-001 … R2-505) — source report `runs/20260902T0000Z/phase1/claims_reviews_part2.md`
- **L-xxx** — surviving result files, bench logs, notebook, READMEs (pre-existing in the tree): 948 claim rows (L-001 … L-948) — source report `runs/20260902T0000Z/phase1/claims_logs.md`
- **L2-xxx** — 11 old mining/pipeline logs pushed in commit e10801c (applications/alphafold/results_214m/logs/): 690 claim rows (L2-001 … L2-690) — source report `runs/20260902T0000Z/phase1/claims_logs_new.md`

Total claim rows: **3521**

---

# SOURCE: paper/et_miner_proteome.tex (V1 preprint)

(verbatim copy of `runs/20260902T0000Z/phase1/claims_tex.md`)

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


---

# SOURCE: paper reviews part 1 (PAPER_V2_REVIEW.md, peer_review_jul12.md, review_b1_hostile.md)

(verbatim copy of `runs/20260902T0000Z/phase1/claims_reviews_part1.md`)

# Phase 1 — Claims extracted from review files (part 1: three review files)

Run: `runs/20260902T0000Z` · Extraction date: 2026-09-02 · Scope: exhaustive, faithful extraction of every numeric value / quantitative claim in three review documents. **No correctness judgement is made here; every number below may be hallucinated in its source file.**

Source files (read in full, every line):

| key | file | lines |
|---|---|---|
| F1 | `/root/projects/ET-Miner/paper/PAPER_V2_REVIEW.md` | 160 |
| F2 | `/root/projects/ET-Miner/paper/peer_review_jul12.md` | 179 |
| F3 | `/root/projects/ET-Miner/paper/review_b1_hostile.md` | 350 |

Conventions:

- One row per **occurrence** of a numeric value / quantitative claim, in file order (F1, F2, F3) then line order. Repeated values on different lines get separate rows so that later phases can map line → value.
- `category`: deterministic = a function of data + parameters (counts, K, percentages, Z/t statistics, sizes derived arithmetically, protein/item counts); hardware-dependent = timings, throughput, speedups, GPU model/count/VRAM, memory-fit feasibility, compute cost; method-parameter = thresholds, support %, min_count, permutation counts, seeds, vocabulary caps, bin bounds, inclusion criteria, confidence levels; external-fact = facts external to the experiment (UniProt release, TrEMBL size, competitor numbers, literature, GO prevalence) **and** review-meta / bibliographic / repository-inventory counts (e.g. '73 claims adjudicated', severity scores, citation counts); software = version numbers.
- `stance`: quotes-paper = the review repeats a value it attributes to the paper; asserts-own = the reviewer states a value of its own (recomputation, value read from a log/JSON artifact, hypothetical, estimate, review-meta count, severity score); disputes = the reviewer says a paper value (or artifact-derived value) is wrong / implausible / invalid; requests = the reviewer asks for a number to be added, changed, listed, or a run to be performed with that parameter.
- Values read by a reviewer from logs/JSON artifacts (not from the paper) are labelled asserts-own; the quoted context names the artifact where the review does.
- **Excluded on purpose** (locators, not claims): paper `.tex` line numbers (e.g. 'line 129'), log/JSON/script line references (e.g. `pipeline_214m.log:2063`), section/table/figure/appendix numbers, review item numbering (Major 1, D1, M3.3, Stage 1–7), timestamps embedded in file names, commit hashes, citation-key years (`miettinen2020`), and the bare shorthand labels '1K'/'35K' when they merely name a vocabulary (the underlying 1,002 / 34,920 are extracted wherever stated; '35K' is kept where it enters arithmetic). Review dates are reported in Section C rather than as claim rows.
- Symbols: `x` = ×, `>=` = ≥, `<=` = ≤, `->` = →, `--` = en dash; commas in numbers are preserved as written.

---

## SECTION A — CLAIMS TABLE

Total rows: **835** (F1: 327, F2: 163, F3: 345).

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R1-001 | F1 | 7 | 9 | verification buckets | external-fact | asserts-own | "Across 9 verification buckets, 73 claims were adjudicated" |
| R1-002 | F1 | 7 | 73 | claims adjudicated | external-fact | asserts-own | "Across 9 verification buckets, 73 claims were adjudicated" |
| R1-003 | F1 | 11 | 62 | claims CONFIRMED | external-fact | asserts-own | "CONFIRMED / 62" (verdict table) |
| R1-004 | F1 | 12 | 4 | claims DISCREPANT | external-fact | asserts-own | "DISCREPANT / 4" (verdict table) |
| R1-005 | F1 | 13 | 7 | claims UNVERIFIABLE | external-fact | asserts-own | "UNVERIFIABLE / 7" (verdict table) |
| R1-006 | F1 | 15 | 6/6 | campaign-table rows verified | external-fact | asserts-own | "the campaign table (6/6)" |
| R1-007 | F1 | 15 | 22 | K-distribution rows | deterministic | quotes-paper | "the full 22-row K-distribution (8/8, line-for-line)" |
| R1-008 | F1 | 15 | 8/8 | K-distribution checks passed | external-fact | asserts-own | "the full 22-row K-distribution (8/8, line-for-line)" |
| R1-009 | F1 | 15 | 6/6 | Direct-vs-SON checks passed | external-fact | asserts-own | "the Direct-vs-SON comparison (6/6)" |
| R1-010 | F1 | 15 | 247 | Pfam items (v1 Table 1) | deterministic | disputes | "The old fabricated v1 Table 1 (247/752/3) has been corrected" |
| R1-011 | F1 | 15 | 752 | GO items (v1 Table 1) | deterministic | disputes | "The old fabricated v1 Table 1 (247/752/3) has been corrected" |
| R1-012 | F1 | 15 | 3 | pLDDT items (v1 Table 1) | deterministic | disputes | "The old fabricated v1 Table 1 (247/752/3) has been corrected" |
| R1-013 | F1 | 15 | 500 | Pfam items (verified) | method-parameter | asserts-own | "corrected to the verified 500/500/6 = 1,006 ground truth" |
| R1-014 | F1 | 15 | 500 | GO items (verified) | method-parameter | asserts-own | "corrected to the verified 500/500/6 = 1,006 ground truth" |
| R1-015 | F1 | 15 | 6 | pLDDT bins (verified) | method-parameter | asserts-own | "corrected to the verified 500/500/6 = 1,006 ground truth" |
| R1-016 | F1 | 15 | 1,006 | total items | deterministic | asserts-own | "corrected to the verified 500/500/6 = 1,006 ground truth" |
| R1-017 | F1 | 19 | >=8 | proteins (feature-retention cutoff, paper wording) | method-parameter | disputes | "'retained all features occurring in at least 8 proteins' does not describe how the vocabulary was built" |
| R1-018 | F1 | 19 | 500 | top-N Pfam frequency cap | method-parameter | asserts-own | "the vocabulary is a top-500-Pfam + top-500-GO frequency cap" |
| R1-019 | F1 | 19 | 500 | top-N GO frequency cap | method-parameter | asserts-own | "the vocabulary is a top-500-Pfam + top-500-GO frequency cap" |
| R1-020 | F1 | 19 | 24,291 | unique Pfam terms | deterministic | asserts-own | "out of 24,291 / 25,993 unique terms" |
| R1-021 | F1 | 19 | 25,993 | unique GO terms | deterministic | asserts-own | "out of 24,291 / 25,993 unique terms" |
| R1-022 | F1 | 19 | >=8 | mining min-support (proteins) | method-parameter | asserts-own | "the '>=8' is the mining min-support, a different quantity" |
| R1-023 | F1 | 20 | 40x | memory reduction (dense to CSR) | deterministic | disputes | "Inflated 40x memory-reduction claim (lines 167, 402)" |
| R1-024 | F1 | 20 | 5.1 | GB (CSR) | deterministic | quotes-paper | "the 5.1 GB CSR is the 76.9M mined subset" |
| R1-025 | F1 | 20 | 76.9M | proteins (mined subset) | deterministic | asserts-own | "the 5.1 GB CSR is the 76.9M mined subset" |
| R1-026 | F1 | 20 | 206 | GB (dense) | deterministic | quotes-paper | "the 206 GB dense is the 205.6M full set" |
| R1-027 | F1 | 20 | 205.6M | proteins (full set) | deterministic | asserts-own | "the 206 GB dense is the 205.6M full set" |
| R1-028 | F1 | 20 | ~15x | memory reduction (same dataset) | deterministic | asserts-own | "Within one dataset the reduction is ~15x, not 40x" |
| R1-029 | F1 | 21 | 8 | proteins (K=22 accessions) | deterministic | quotes-paper | "the paper says the 8 K=22 accessions 'can be recovered by querying the source transaction data'" |
| R1-030 | F1 | 21 | 22 | K (deepest itemset) | deterministic | quotes-paper | "the paper says the 8 K=22 accessions 'can be recovered'" |
| R1-031 | F1 | 22 | <0.45 | p (upper bound) | deterministic | quotes-paper | "the p<0.45 bound is the exact one-sided binomial 95% bound" |
| R1-032 | F1 | 22 | 95% | one-sided confidence level | method-parameter | asserts-own | "the exact one-sided binomial 95% bound (1-0.05^(1/5)=0.451)" |
| R1-033 | F1 | 22 | 1-0.05^(1/5)=0.451 | exact binomial bound | deterministic | asserts-own | "the exact one-sided binomial 95% bound (1-0.05^(1/5)=0.451)" |
| R1-034 | F1 | 22 | 3/5=0.60 | rule-of-three bound | deterministic | asserts-own | "'the rule of three,' which actually gives 3/5=0.60" |
| R1-035 | F1 | 24 | 4 | min_count (verification run) | method-parameter | disputes | "unbacked min_count=4 claim" |
| R1-036 | F1 | 30 | >=8 | support cutoff (heading) | method-parameter | disputes | "D1 - Vocabulary construction misdescribed as an >=8-support cutoff" |
| R1-037 | F1 | 31 | >=8 | proteins (feature retention) | method-parameter | quotes-paper | "We retained all features occurring in at least 8 proteins (the lowest support threshold used)." |
| R1-038 | F1 | 32 | 24291 | unique Pfam | deterministic | asserts-own | "24291 unique Pfam, 25993 unique GO from 205620298 proteins" |
| R1-039 | F1 | 32 | 25993 | unique GO | deterministic | asserts-own | "24291 unique Pfam, 25993 unique GO from 205620298 proteins" |
| R1-040 | F1 | 32 | 205620298 | proteins | deterministic | asserts-own | "24291 unique Pfam, 25993 unique GO from 205620298 proteins" |
| R1-041 | F1 | 32 | 6 | pLDDT items | method-parameter | asserts-own | "Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items" |
| R1-042 | F1 | 32 | 500 | Pfam items | method-parameter | asserts-own | "Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items" |
| R1-043 | F1 | 32 | 500 | GO items | method-parameter | asserts-own | "Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items" |
| R1-044 | F1 | 32 | 1006 | total items | deterministic | asserts-own | "Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items" |
| R1-045 | F1 | 32 | top-500 | per-type frequency cap | method-parameter | asserts-own | "The vocabulary is a top-500-per-type frequency cap, not a >=8-support retention" |
| R1-046 | F1 | 32 | >=8 | mining min_count | method-parameter | asserts-own | "The '>=8' is the mining min_count (godmode_mining.log:2), a separate quantity" |
| R1-047 | F1 | 33 | 205.6M | proteins | deterministic | asserts-own | "At 205.6M proteins, vastly more than 500 Pfam/GO families occur in >=8 proteins" |
| R1-048 | F1 | 33 | >500 | Pfam/GO families occurring in >=8 proteins | deterministic | asserts-own | "vastly more than 500 Pfam/GO families occur in >=8 proteins" |
| R1-049 | F1 | 33 | >=8 | proteins | method-parameter | asserts-own | "vastly more than 500 Pfam/GO families occur in >=8 proteins" |
| R1-050 | F1 | 36 | 40x | reduction (heading) | deterministic | disputes | "D2 - '40x reduction' conflates two different datasets" |
| R1-051 | F1 | 37 | 316 million | non-zero entries | deterministic | quotes-paper | "316 million non-zero entries occupying ~5.1 GB ... a 40x reduction from the ~206 GB naive dense" |
| R1-052 | F1 | 37 | ~5.1 | GB (coordinate format) | deterministic | quotes-paper | "316 million non-zero entries occupying ~5.1 GB" |
| R1-053 | F1 | 37 | 40x | reduction | deterministic | disputes | "a 40x reduction from the ~206 GB naive dense representation" |
| R1-054 | F1 | 37 | ~206 | GB (naive dense) | deterministic | quotes-paper | "a 40x reduction from the ~206 GB naive dense representation" |
| R1-055 | F1 | 38 | 316,421,093 | non-zeros | deterministic | asserts-own | "ties the 316,421,093 non-zeros / 5.1 GB CSR to the 76,890,945-transaction subset" |
| R1-056 | F1 | 38 | 5.1 | GB (CSR) | deterministic | asserts-own | "ties the 316,421,093 non-zeros / 5.1 GB CSR to the 76,890,945-transaction subset" |
| R1-057 | F1 | 38 | 76,890,945 | transactions (subset) | deterministic | asserts-own | "ties the 316,421,093 non-zeros / 5.1 GB CSR to the 76,890,945-transaction subset" |
| R1-058 | F1 | 38 | 206 | GB dense (full set) | deterministic | asserts-own | "the 206 GB dense figure is for the 205.6M full set" |
| R1-059 | F1 | 38 | 205.6M | proteins (full set) | deterministic | asserts-own | "the 206 GB dense figure is for the 205.6M full set" |
| R1-060 | F1 | 38 | 205,620,298 | transactions | deterministic | asserts-own | "205,620,298 x 1002 x 1 B = 206 GB" |
| R1-061 | F1 | 38 | 1002 | items (dense columns) | deterministic | asserts-own | "205,620,298 x 1002 x 1 B = 206 GB" |
| R1-062 | F1 | 38 | 1 | byte per dense entry | method-parameter | asserts-own | "205,620,298 x 1002 x 1 B = 206 GB" |
| R1-063 | F1 | 38 | 206 | GB (computed) | deterministic | asserts-own | "205,620,298 x 1002 x 1 B = 206 GB" |
| R1-064 | F1 | 38 | 76.9M | transactions (subset) | deterministic | asserts-own | "Within the 76.9M subset: dense = 76.9M x 1002 x 1 B = 77 GB" |
| R1-065 | F1 | 38 | 77 | GB dense (subset) | deterministic | asserts-own | "dense = 76.9M x 1002 x 1 B = 77 GB" |
| R1-066 | F1 | 38 | 5.06 | GB (CSR) | deterministic | asserts-own | "so 77/5.06 = ~15x, not 40x" |
| R1-067 | F1 | 38 | ~15x | reduction (same subset) | deterministic | asserts-own | "so 77/5.06 = ~15x, not 40x" |
| R1-068 | F1 | 38 | 40x | reduction | deterministic | disputes | "so 77/5.06 = ~15x, not 40x" |
| R1-069 | F1 | 39 | 40x | reduction | deterministic | disputes | "The 40x figure is manufactured by pairing a smaller-dataset CSR against a larger-dataset dense" |
| R1-070 | F1 | 39 | 1.4x | reduction (bit-packed, appendix) | deterministic | quotes-paper | "already reports the honest same-dataset ratio (1.4x bit-packed, 7.8x at 0.01% density)" |
| R1-071 | F1 | 39 | 7.8x | reduction (at 0.01% density, appendix) | deterministic | quotes-paper | "already reports the honest same-dataset ratio (1.4x bit-packed, 7.8x at 0.01% density)" |
| R1-072 | F1 | 39 | 0.01% | density | deterministic | quotes-paper | "7.8x at 0.01% density" |
| R1-073 | F1 | 40 | 316,421,093 | nz | deterministic | asserts-own | "direct_mining.log (316,421,093 nz on 76,890,945 txns)" |
| R1-074 | F1 | 40 | 76,890,945 | txns | deterministic | asserts-own | "direct_mining.log (316,421,093 nz on 76,890,945 txns)" |
| R1-075 | F1 | 40 | 205,620,298 | total transactions | deterministic | asserts-own | "beyond_mining.log (Total transactions 205,620,298)" |
| R1-076 | F1 | 42 | 8 | K=22 accessions (heading) | deterministic | quotes-paper | "D3 - 8 K=22 accessions 'recoverable' contradicts 'matrix not deposited'" |
| R1-077 | F1 | 42 | 22 | K (heading) | deterministic | quotes-paper | "D3 - 8 K=22 accessions 'recoverable' contradicts 'matrix not deposited'" |
| R1-078 | F1 | 43 | 8 (eight) | matching proteins (K=22) | deterministic | quotes-paper | "The eight matching proteins can be recovered by querying the source transaction data" |
| R1-079 | F1 | 45 | 8 | IDs (K=22 proteins) | deterministic | asserts-own | "The 8 IDs are neither listed in the paper nor regenerable from this repo" |
| R1-080 | F1 | 49 | <0.45 | p (heading) | deterministic | disputes | "D4 - p<0.45 mis-attributed to the 'rule of three'" |
| R1-081 | F1 | 50 | <0.45 | p | deterministic | quotes-paper | "p < 0.45, the 95% one-sided upper bound for 0 of 5 by the rule of three" |
| R1-082 | F1 | 50 | 95% | one-sided upper bound | method-parameter | quotes-paper | "the 95% one-sided upper bound for 0 of 5 by the rule of three" |
| R1-083 | F1 | 50 | 0 of 5 | null runs reaching K>=7 | deterministic | quotes-paper | "the 95% one-sided upper bound for 0 of 5 by the rule of three" |
| R1-084 | F1 | 51 | 0/5 | null runs reaching K>=7 | deterministic | asserts-own | "0/5 null runs reach K>=7 is confirmed" |
| R1-085 | F1 | 51 | >=7 | K (deep-pattern cutoff) | method-parameter | asserts-own | "0/5 null runs reach K>=7 is confirmed" |
| R1-086 | F1 | 51 | 1-0.05^(1/5)=0.4507 | exact binomial 95% bound | deterministic | asserts-own | "The exact one-sided binomial 95% bound is 1-0.05^(1/5)=0.4507" |
| R1-087 | F1 | 51 | 95% | confidence level | method-parameter | asserts-own | "The exact one-sided binomial 95% bound is 1-0.05^(1/5)=0.4507" |
| R1-088 | F1 | 51 | 3/5=0.60 | rule-of-three bound | deterministic | asserts-own | "The rule of three gives 3/n = 3/5 = 0.60, not 0.45" |
| R1-089 | F1 | 51 | 0.45 | p bound (paper value contrasted with rule of three) | deterministic | quotes-paper | "The rule of three gives 3/n = 3/5 = 0.60, not 0.45." |
| R1-090 | F1 | 52 | 0.45 | p bound | deterministic | asserts-own | "The number (0.45) is correct and defensible as the exact binomial bound" |
| R1-091 | F1 | 61 | 7 | confirmed dataset/vocabulary claims | external-fact | asserts-own | "Dataset & vocabulary (7):" |
| R1-092 | F1 | 62 | 214M | proteins | deterministic | quotes-paper | "214M proteins (line 129) = 214,683,829 rows read" |
| R1-093 | F1 | 62 | 214,683,829 | rows read | deterministic | asserts-own | "214M proteins (line 129) = 214,683,829 rows read - pipeline_214m.log:8,2061" |
| R1-094 | F1 | 63 | 205,620,298 | proteins processed | deterministic | quotes-paper | "205,620,298 processed (lines 129,152) = exact 'passed pLDDT filter' / 'Total transactions'" |
| R1-095 | F1 | 64 | 150 | GB (input data) | external-fact | quotes-paper | "150 GB in 63 min (line 129)" |
| R1-096 | F1 | 64 | 63 | min (feature extraction) | hardware-dependent | quotes-paper | "150 GB in 63 min (line 129)" |
| R1-097 | F1 | 64 | 150G | TrEMBL size (log) | external-fact | asserts-own | "'TrEMBL: 150G'; af_extract 3777.0s = 62.95 min" |
| R1-098 | F1 | 64 | 3777.0 | s (af_extract) | hardware-dependent | asserts-own | "af_extract 3777.0s = 62.95 min" |
| R1-099 | F1 | 64 | 62.95 | min (af_extract) | hardware-dependent | asserts-own | "af_extract 3777.0s = 62.95 min" |
| R1-100 | F1 | 65 | 8 | min-support | method-parameter | quotes-paper | "min-support 8 -> 1,002 frequent single items (Table 1 caption)" |
| R1-101 | F1 | 65 | 1,002 | frequent single items | deterministic | quotes-paper | "min-support 8 -> 1,002 frequent single items (Table 1 caption)" |
| R1-102 | F1 | 66 | 1,006 | items | deterministic | quotes-paper | "1,006 items = 6 pLDDT + 500 Pfam + 500 GO (Table 1)" |
| R1-103 | F1 | 66 | 6 | pLDDT items | method-parameter | quotes-paper | "1,006 items = 6 pLDDT + 500 Pfam + 500 GO (Table 1)" |
| R1-104 | F1 | 66 | 500 | Pfam items | method-parameter | quotes-paper | "1,006 items = 6 pLDDT + 500 Pfam + 500 GO (Table 1)" |
| R1-105 | F1 | 66 | 500 | GO items | method-parameter | quotes-paper | "1,006 items = 6 pLDDT + 500 Pfam + 500 GO (Table 1)" |
| R1-106 | F1 | 67 | 1,002 | items passing min-support | deterministic | asserts-own | "1,002 pass (1,000 Pfam/GO + 2 pLDDT bins)" |
| R1-107 | F1 | 67 | 1,000 | Pfam/GO items passing | deterministic | asserts-own | "1,002 pass (1,000 Pfam/GO + 2 pLDDT bins)" |
| R1-108 | F1 | 67 | 2 | pLDDT bins passing | deterministic | asserts-own | "1,002 pass (1,000 Pfam/GO + 2 pLDDT bins)" |
| R1-109 | F1 | 67 | 2 (two) | pLDDT labels in decoded patterns | deterministic | asserts-own | "corroborated by only two pLDDT labels in decoded_top_k_patterns.txt" |
| R1-110 | F1 | 68 | 76,890,945 | multi-feature proteins | deterministic | quotes-paper | "76,890,945 (37.4%) multi-feature proteins (lines 91,120,152)" |
| R1-111 | F1 | 68 | 37.4% | fraction multi-feature | deterministic | quotes-paper | "76,890,945 (37.4%) multi-feature proteins (lines 91,120,152)" |
| R1-112 | F1 | 70 | 6 | confirmed memory claims | external-fact | asserts-own | "Memory (6):" |
| R1-113 | F1 | 70 | 206 | GB dense | deterministic | quotes-paper | "206 GB dense (line 105)" |
| R1-114 | F1 | 70 | 316M | nz | deterministic | quotes-paper | "316M nz / 5.1 GB coordinate (line 167)" |
| R1-115 | F1 | 70 | 5.1 | GB coordinate | deterministic | quotes-paper | "316M nz / 5.1 GB coordinate (line 167)" |
| R1-116 | F1 | 70 | ~26 | GB bit-packed | deterministic | quotes-paper | "~26 GB bit-packed (lines 167,171)" |
| R1-117 | F1 | 70 | ~3 | GB H2D transfer | deterministic | quotes-paper | "~3 GB H2D transfer (line 171)" |
| R1-118 | F1 | 70 | ~264 | B (metadata over 22 K-levels) | deterministic | quotes-paper | "~264 B over 22 K-levels (line 814)" |
| R1-119 | F1 | 70 | 22 | K-levels | deterministic | quotes-paper | "~264 B over 22 K-levels (line 814)" |
| R1-120 | F1 | 70 | 214M | row label (appendix memory table) | deterministic | quotes-paper | "appendix 214M row 27 GB/19 GB/1.4x (line 887)" |
| R1-121 | F1 | 70 | 27 | GB (appendix 214M row) | deterministic | quotes-paper | "appendix 214M row 27 GB/19 GB/1.4x (line 887)" |
| R1-122 | F1 | 70 | 19 | GB (appendix 214M row) | deterministic | quotes-paper | "appendix 214M row 27 GB/19 GB/1.4x (line 887)" |
| R1-123 | F1 | 70 | 1.4x | ratio (appendix 214M row) | deterministic | quotes-paper | "appendix 214M row 27 GB/19 GB/1.4x (line 887)" |
| R1-124 | F1 | 72 | 6/6 | campaign rows verified | external-fact | asserts-own | "Campaign table (6/6):" |
| R1-125 | F1 | 72 | 0.1% | support (Base) | method-parameter | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-126 | F1 | 72 | 76,891 | min_count (Base) | method-parameter | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-127 | F1 | 72 | 5,305 | itemsets (Base) | deterministic | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-128 | F1 | 72 | 9 | K max (Base) | deterministic | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-129 | F1 | 72 | 1.9 | min (Base) | hardware-dependent | quotes-paper | "Base (0.1%/76,891/5,305/K9/1.9min/SON)" |
| R1-130 | F1 | 72 | 0.01% | support (Super) | method-parameter | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-131 | F1 | 72 | 7,689 | min_count (Super) | method-parameter | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-132 | F1 | 72 | 51,124 | itemsets (Super) | deterministic | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-133 | F1 | 72 | 13 | K max (Super) | deterministic | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-134 | F1 | 72 | 4.3 | min (Super) | hardware-dependent | quotes-paper | "Super (0.01%/7,689/51,124/K13/4.3min/SON)" |
| R1-135 | F1 | 72 | 0.001% | support (Power) | method-parameter | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-136 | F1 | 72 | 768 | min_count (Power) | method-parameter | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-137 | F1 | 72 | 22,846 | itemsets (Power) | deterministic | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-138 | F1 | 72 | 13 | K max (Power) | deterministic | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-139 | F1 | 72 | 18.1 | min (Power) | hardware-dependent | quotes-paper | "Power (0.001%/768/22,846/K13/18.1min/SON)" |
| R1-140 | F1 | 72 | 0.0001% | support (Blitz) | method-parameter | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-141 | F1 | 72 | 77 | min_count (Blitz) | method-parameter | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-142 | F1 | 72 | 2,841,280 | itemsets (Blitz) | deterministic | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-143 | F1 | 72 | 19 | K max (Blitz) | deterministic | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-144 | F1 | 72 | 2.0 | min (Blitz) | hardware-dependent | quotes-paper | "Blitz (0.0001%/77/2,841,280/K19/2.0min/Direct)" |
| R1-145 | F1 | 72 | 0.00002% | support (Ultra) | method-parameter | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-146 | F1 | 72 | 16 | min_count (Ultra) | method-parameter | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-147 | F1 | 72 | 14,558,875 | itemsets (Ultra) | deterministic | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-148 | F1 | 72 | 20 | K max (Ultra) | deterministic | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-149 | F1 | 72 | 4.7 | min (Ultra) | hardware-dependent | quotes-paper | "Ultra (0.00002%/16/14,558,875/K20/4.7min/Direct)" |
| R1-150 | F1 | 72 | 0.00001% | support (Opus) | method-parameter | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-151 | F1 | 72 | 8 | min_count (Opus) | method-parameter | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-152 | F1 | 72 | 26,849,505 | itemsets (Opus) | deterministic | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-153 | F1 | 72 | 22 | K max (Opus) | deterministic | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-154 | F1 | 72 | 7.3 | min (Opus) | hardware-dependent | quotes-paper | "Opus (0.00001%/8/26,849,505/K22/7.3min/Direct)" |
| R1-155 | F1 | 74 | 8/8 | K-distribution checks passed | external-fact | asserts-own | "K-distribution (8/8):" |
| R1-156 | F1 | 74 | 26,849,505 | total itemsets | deterministic | quotes-paper | "total 26,849,505; peak K=9 = 3,529,257 (13.14%)" |
| R1-157 | F1 | 74 | 9 | K (distribution peak) | deterministic | quotes-paper | "peak K=9 = 3,529,257 (13.14%)" |
| R1-158 | F1 | 74 | 3,529,257 | itemsets at K=9 | deterministic | quotes-paper | "peak K=9 = 3,529,257 (13.14%)" |
| R1-159 | F1 | 74 | 13.14% | share of itemsets at K=9 | deterministic | quotes-paper | "peak K=9 = 3,529,257 (13.14%)" |
| R1-160 | F1 | 74 | 1,002 | itemsets at K=1 | deterministic | quotes-paper | "K=1=1,002; K=2=73,786; K=22=1 (8 proteins)" |
| R1-161 | F1 | 74 | 73,786 | itemsets at K=2 | deterministic | quotes-paper | "K=1=1,002; K=2=73,786; K=22=1 (8 proteins)" |
| R1-162 | F1 | 74 | 1 | itemsets at K=22 | deterministic | quotes-paper | "K=22=1 (8 proteins)" |
| R1-163 | F1 | 74 | 8 | proteins supporting K=22 | deterministic | quotes-paper | "K=22=1 (8 proteins)" |
| R1-164 | F1 | 74 | 3..21 | K rows verified | deterministic | quotes-paper | "all K=3..21 rows" |
| R1-165 | F1 | 74 | 100.0% | sum of K-distribution percentages | deterministic | asserts-own | "percentages recomputed and sum to 100.0%" |
| R1-166 | F1 | 76 | 2 | K=22 composition checks | external-fact | asserts-own | "K=22 composition (2):" |
| R1-167 | F1 | 76 | 22 | features (K=22 itemset) | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-168 | F1 | 76 | 2 | Pfam features in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-169 | F1 | 76 | 8 | GO:MF features in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-170 | F1 | 76 | 4 | GO:BP features in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-171 | F1 | 76 | 7 | GO:CC features in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-172 | F1 | 76 | 1 | pLDDT feature in K=22 | deterministic | quotes-paper | "22 features = 2 Pfam + 8 MF + 4 BP + 7 CC + 1 pLDDT" |
| R1-173 | F1 | 78 | 17 | confirmed null-model claims | external-fact | asserts-own | "Null model (17):" |
| R1-174 | F1 | 78 | 5 | permutations | method-parameter | quotes-paper | "5 permutations, seed 42, 662 s total, min_count=769" |
| R1-175 | F1 | 78 | 42 | seed | method-parameter | quotes-paper | "5 permutations, seed 42, 662 s total, min_count=769" |
| R1-176 | F1 | 78 | 662 | s (null model total) | hardware-dependent | quotes-paper | "5 permutations, seed 42, 662 s total, min_count=769" |
| R1-177 | F1 | 78 | 769 | min_count (null model) | method-parameter | quotes-paper | "5 permutations, seed 42, 662 s total, min_count=769" |
| R1-178 | F1 | 78 | -987 | Z at K=2 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-179 | F1 | 78 | -143 | Z at K=3 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-180 | F1 | 78 | +3,791 | Z at K=4 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-181 | F1 | 78 | +7,402 | Z at K=5 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-182 | F1 | 78 | +71,728 | Z at K=6 | deterministic | quotes-paper | "Z at K=2 (-987), K=3 (-143), K=4 (+3,791), K=5 (+7,402), K=6 (+71,728)" |
| R1-183 | F1 | 78 | 3 | K (null-model peak) | deterministic | quotes-paper | "null peaks K=3 (46.2%)" |
| R1-184 | F1 | 78 | 46.2% | null share at K=3 | deterministic | quotes-paper | "null peaks K=3 (46.2%)" |
| R1-185 | F1 | 78 | 6 | K (null max depth) | deterministic | quotes-paper | "null max depth K=6 (mean 22, std 1.1)" |
| R1-186 | F1 | 78 | 22 | mean null itemsets at K=6 | deterministic | quotes-paper | "null max depth K=6 (mean 22, std 1.1)" |
| R1-187 | F1 | 78 | 1.1 | std null itemsets at K=6 | deterministic | quotes-paper | "null max depth K=6 (mean 22, std 1.1)" |
| R1-188 | F1 | 78 | 88,745 | biological itemsets at K>=7 | deterministic | quotes-paper | "bio K>=7 = 88,745; real_total 475,865 to K=14" |
| R1-189 | F1 | 78 | 475,865 | real_total itemsets (0.001%) | deterministic | quotes-paper | "bio K>=7 = 88,745; real_total 475,865 to K=14" |
| R1-190 | F1 | 78 | 14 | K max (real, 0.001%) | deterministic | quotes-paper | "real_total 475,865 to K=14" |
| R1-191 | F1 | 78 | 1,002 | K=1 itemsets preserved under permutation | deterministic | quotes-paper | "K=1 preserves 1,002; ~130 s/permutation" |
| R1-192 | F1 | 78 | ~130 | s per permutation | hardware-dependent | quotes-paper | "K=1 preserves 1,002; ~130 s/permutation" |
| R1-193 | F1 | 78 | n=5 | permutations (t-statistic basis) | method-parameter | quotes-paper | "The n=5 Z-scores are appropriately recast as t-statistics (4 df)" |
| R1-194 | F1 | 78 | 4 | df (t-statistics) | deterministic | quotes-paper | "The n=5 Z-scores are appropriately recast as t-statistics (4 df)" |
| R1-195 | F1 | 80 | 6/6 | Direct-vs-SON checks passed | external-fact | asserts-own | "Direct vs SON (6/6):" |
| R1-196 | F1 | 80 | 475,865 | itemsets (Direct) | deterministic | quotes-paper | "Direct 475,865 in 50.7 s; SON 22,846 in 1,085.6 s; 21x speedup; SON misses 95.2%" |
| R1-197 | F1 | 80 | 50.7 | s (Direct) | hardware-dependent | quotes-paper | "Direct 475,865 in 50.7 s" |
| R1-198 | F1 | 80 | 22,846 | itemsets (SON) | deterministic | quotes-paper | "SON 22,846 in 1,085.6 s" |
| R1-199 | F1 | 80 | 1,085.6 | s (SON) | hardware-dependent | quotes-paper | "SON 22,846 in 1,085.6 s" |
| R1-200 | F1 | 80 | 21x | speedup (Direct vs SON) | hardware-dependent | quotes-paper | "21x speedup; SON misses 95.2%" |
| R1-201 | F1 | 80 | 95.2% | SON miss rate | deterministic | quotes-paper | "21x speedup; SON misses 95.2%" |
| R1-202 | F1 | 82 | 3 | confirmed scale-comparison claims | external-fact | asserts-own | "Scale comparison (3):" |
| R1-203 | F1 | 82 | 5.1x | transactions ratio vs GMiner | deterministic | quotes-paper | "5.1x (76.9M/15M); 62.6% = 128.7M excluded" |
| R1-204 | F1 | 82 | 76.9M | transactions (ET-miner) | deterministic | quotes-paper | "5.1x (76.9M/15M)" |
| R1-205 | F1 | 82 | 15M | transactions (GMiner) | external-fact | quotes-paper | "5.1x (76.9M/15M)" |
| R1-206 | F1 | 82 | 62.6% | proteins excluded (single-feature) | deterministic | quotes-paper | "62.6% = 128.7M excluded" |
| R1-207 | F1 | 82 | 128.7M | proteins excluded | deterministic | quotes-paper | "62.6% = 128.7M excluded" |
| R1-208 | F1 | 82 | 76.9M | transactions (ET-miner row) | deterministic | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-209 | F1 | 82 | 1,002 | items (ET-miner row) | deterministic | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-210 | F1 | 82 | 22 | K (ET-miner row) | deterministic | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-211 | F1 | 82 | 1xH100 | GPU count/model | hardware-dependent | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-212 | F1 | 82 | 7.3 | min (ET-miner row) | hardware-dependent | quotes-paper | "ET-miner row 76.9M/1,002/K22/1xH100/7.3min" |
| R1-213 | F1 | 84 | 7 | confirmed citation claims | external-fact | asserts-own | "Citations (7):" |
| R1-214 | F1 | 84 | 47 | \cite commands | external-fact | asserts-own | "47 \cite -> 37 unique keys, all with matching \bibitem and none unused" |
| R1-215 | F1 | 84 | 37 | unique citation keys | external-fact | asserts-own | "47 \cite -> 37 unique keys, all with matching \bibitem and none unused" |
| R1-216 | F1 | 84 | 37 | bibliography entries | external-fact | asserts-own | "\begin{thebibliography}{37} = 37 entries" |
| R1-217 | F1 | 84 | 0 (zero) | cross-paper tool contamination | external-fact | asserts-own | "zero cross-paper (opus_pocket: fpocket, efficient-apriori, scikit-learn, PDBbind) contamination" |
| R1-218 | F1 | 94 | 2025_01 | UniProt release | external-fact | quotes-paper | "UniProt release 2025_01 / line 129, 478 / Log records only input filename" |
| R1-219 | F1 | 94 | Feb 2026 | run date (log) | external-fact | asserts-own | "Log records only input filename uniprot_trembl.dat.gz and Feb 2026 run date; no release string" |
| R1-220 | F1 | 95 | 22 | K | deterministic | quotes-paper | "K=22 shared by exactly 8 proteins / lines 254, 286" |
| R1-221 | F1 | 95 | 8 | proteins (K=22) | deterministic | quotes-paper | "K=22 shared by exactly 8 proteins / lines 254, 286" |
| R1-222 | F1 | 95 | 19 | K max in decoded_top_k_patterns.txt | deterministic | asserts-own | "decoded_top_k_patterns.txt caps at K=19 (187 proteins) and is a different run" |
| R1-223 | F1 | 95 | 187 | proteins (K=19 pattern in decoded file) | deterministic | asserts-own | "caps at K=19 (187 proteins) and is a different run" |
| R1-224 | F1 | 95 | >=20 | K (no artifact contains any) | deterministic | asserts-own | "no artifact contains any K>=20 itemset" |
| R1-225 | F1 | 96 | 22->21 | independent features after GO check | deterministic | quotes-paper | "22->21 independent (0 GO parent-child pairs; InterPro2GO link removed)" |
| R1-226 | F1 | 96 | 0 | GO parent-child pairs | deterministic | quotes-paper | "22->21 independent (0 GO parent-child pairs; InterPro2GO link removed)" |
| R1-227 | F1 | 97 | 4 | min_count (verification run) | method-parameter | quotes-paper | "min_count=4 -> 48M itemsets, same K max / line 335" |
| R1-228 | F1 | 97 | 48M | itemsets (min_count=4 run) | deterministic | disputes | "No min_count=4 run exists. The only '48M' is 'Written 48M transactions' during extraction" |
| R1-229 | F1 | 97 | 48M | transactions written (extraction log) | deterministic | asserts-own | "The only '48M' is 'Written 48M transactions' during extraction - a transaction count, not itemsets" |
| R1-230 | F1 | 97 | 769 | min_count (null-model JSON) | method-parameter | asserts-own | "experiment_null_model_...json (min_count 769)" |
| R1-231 | F1 | 98 | 8 | K=22 UniProt accessions | deterministic | quotes-paper | "8 K=22 UniProt accessions / line 333 / See D3 - not recoverable" |
| R1-232 | F1 | 98 | 22 | K | deterministic | quotes-paper | "8 K=22 UniProt accessions" |
| R1-233 | F1 | 99 | 15M | transactions (GMiner basis) | external-fact | quotes-paper | "Internally used consistently (GMiner 15M basis for 5.1x)" |
| R1-234 | F1 | 99 | 5.1x | transactions ratio | deterministic | quotes-paper | "Internally used consistently (GMiner 15M basis for 5.1x)" |
| R1-235 | F1 | 100 | 37 | references | external-fact | quotes-paper | "External correctness of 37 references / lines 541-724" |
| R1-236 | F1 | 108 | 231 | lines (manuscript rewrite in commit diff) | external-fact | asserts-own | "its diff is a 231-line manuscript rewrite that deletes expanded-run figures" |
| R1-237 | F1 | 110 | 606K | itemsets (expanded run, revision_notes_b3.tex) | deterministic | disputes | "606K itemsets @ 4xH200 in revision_notes_b3.tex vs 16.8B @ 8xH200 in the plan" |
| R1-238 | F1 | 110 | 4xH200 | GPUs (expanded run, revision_notes_b3.tex) | hardware-dependent | disputes | "606K itemsets @ 4xH200 in revision_notes_b3.tex vs 16.8B @ 8xH200 in the plan" |
| R1-239 | F1 | 110 | 16.8B | itemsets (expanded run, plan) | deterministic | disputes | "606K itemsets @ 4xH200 in revision_notes_b3.tex vs 16.8B @ 8xH200 in the plan" |
| R1-240 | F1 | 110 | 8xH200 | GPUs (expanded run, plan) | hardware-dependent | disputes | "606K itemsets @ 4xH200 in revision_notes_b3.tex vs 16.8B @ 8xH200 in the plan" |
| R1-241 | F1 | 111 | n=5 | permutations (caveat) | method-parameter | quotes-paper | "Abstract/conclusion omit the n=5 caveat on Z>3,700 (lines 91, 498)" |
| R1-242 | F1 | 111 | >3,700 | Z (headline) | deterministic | quotes-paper | "Abstract/conclusion omit the n=5 caveat on Z>3,700 (lines 91, 498)" |
| R1-243 | F1 | 111 | 4 | df | deterministic | quotes-paper | "Body (table caption + s4.4) correctly recasts these as t-statistics (4 df)" |
| R1-244 | F1 | 112 | Feb 2026 | date (stale Dutch translation) | external-fact | asserts-own | "Stale Dutch translation et_miner_proteome_nl.tex (old title, Feb 2026 date)" |
| R1-245 | F1 | 113 | 8 | K=22 accessions to list | deterministic | requests | "(a) list the 8 K=22 accessions - see D3" |
| R1-246 | F1 | 113 | ~11,000 | proteins (approximate count) | deterministic | quotes-paper | "replace the '~11,000 / ~10,500 / ~16,000' approximate protein counts (lines 353, 355, 357)" |
| R1-247 | F1 | 113 | ~10,500 | proteins (approximate count) | deterministic | quotes-paper | "replace the '~11,000 / ~10,500 / ~16,000' approximate protein counts (lines 353, 355, 357)" |
| R1-248 | F1 | 113 | ~16,000 | proteins (approximate count) | deterministic | quotes-paper | "replace the '~11,000 / ~10,500 / ~16,000' approximate protein counts (lines 353, 355, 357)" |
| R1-249 | F1 | 113 | 7.3 | min (mining-only) | hardware-dependent | requests | "(d) state explicitly that 7.3 min is mining-only wherever the figure appears standalone" |
| R1-250 | F1 | 122 | >=8 | proteins (feature retention, OLD text) | method-parameter | quotes-paper | "OLD: We retained all features occurring in at least 8 proteins (the lowest support threshold used)." |
| R1-251 | F1 | 123 | 500 | most frequent Pfam domains (proposed text) | method-parameter | requests | "The feature vocabulary comprises the 500 most frequent Pfam domains and the 500 most frequent GO terms" |
| R1-252 | F1 | 123 | 500 | most frequent GO terms (proposed text) | method-parameter | requests | "The feature vocabulary comprises the 500 most frequent Pfam domains and the 500 most frequent GO terms" |
| R1-253 | F1 | 123 | 24,291 | Pfam families observed (proposed text) | deterministic | requests | "(drawn from 24,291 Pfam and 25,993 GO families observed across the corpus)" |
| R1-254 | F1 | 123 | 25,993 | GO families observed (proposed text) | deterministic | requests | "(drawn from 24,291 Pfam and 25,993 GO families observed across the corpus)" |
| R1-255 | F1 | 123 | 6 | pLDDT confidence bins (proposed text) | method-parameter | requests | "together with 6 pLDDT confidence bins, for 1,006 defined items" |
| R1-256 | F1 | 123 | 1,006 | defined items (proposed text) | deterministic | requests | "together with 6 pLDDT confidence bins, for 1,006 defined items" |
| R1-257 | F1 | 123 | 8 | proteins (minimum support, proposed text) | method-parameter | requests | "The minimum support threshold of 8 proteins is applied during mining, not during vocabulary construction" |
| R1-258 | F1 | 125 | 40x | reduction (cross-dataset claim to fix) | deterministic | disputes | "[D2] Line 167 - fix the cross-dataset 40x claim. Make the reduction ratio same-dataset." |
| R1-259 | F1 | 126 | 316 million | non-zero entries (OLD text) | deterministic | quotes-paper | "the resulting matrix contains 316 million non-zero entries occupying ~5.1 GB in coordinate format" |
| R1-260 | F1 | 126 | ~5.1 | GB (OLD text) | deterministic | quotes-paper | "316 million non-zero entries occupying ~5.1 GB in coordinate format" |
| R1-261 | F1 | 126 | 2 x 64-bit | integers per COO entry (OLD text) | method-parameter | quotes-paper | "(two 64-bit integers per entry)" |
| R1-262 | F1 | 126 | 40x | reduction (OLD text) | deterministic | disputes | "a 40x reduction from the ~206 GB naive dense representation (one byte per boolean entry)" |
| R1-263 | F1 | 126 | ~206 | GB (OLD text) | deterministic | quotes-paper | "a 40x reduction from the ~206 GB naive dense representation" |
| R1-264 | F1 | 126 | 1 | byte per boolean entry (OLD text) | method-parameter | quotes-paper | "(one byte per boolean entry)" |
| R1-265 | F1 | 127 | 76.9M | proteins (mining subset, NEW text) | deterministic | requests | "for the 76.9M-protein mining subset the resulting matrix contains 316 million non-zero entries" |
| R1-266 | F1 | 127 | 316 million | non-zero entries (NEW text) | deterministic | requests | "the resulting matrix contains 316 million non-zero entries occupying ~5.1 GB" |
| R1-267 | F1 | 127 | ~5.1 | GB (NEW text) | deterministic | requests | "316 million non-zero entries occupying ~5.1 GB in coordinate format" |
| R1-268 | F1 | 127 | ~15x | reduction (NEW text) | deterministic | requests | "a ~15x reduction from the ~77 GB dense representation of that same subset" |
| R1-269 | F1 | 127 | ~77 | GB dense of subset (NEW text) | deterministic | requests | "a ~15x reduction from the ~77 GB dense representation of that same subset" |
| R1-270 | F1 | 127 | ~206 | GB dense of full set (NEW text) | deterministic | requests | "relative to the ~206 GB dense matrix of the full 205.6M-protein set it is ~40x smaller" |
| R1-271 | F1 | 127 | 205.6M | proteins (full set, NEW text) | deterministic | requests | "relative to the ~206 GB dense matrix of the full 205.6M-protein set it is ~40x smaller" |
| R1-272 | F1 | 127 | ~40x | smaller (cross-dataset, NEW text) | deterministic | requests | "relative to the ~206 GB dense matrix of the full 205.6M-protein set it is ~40x smaller" |
| R1-273 | F1 | 127 | 2 x 64-bit | integers per COO entry (NEW text) | method-parameter | requests | "occupying ~5.1 GB in coordinate format (two 64-bit integers per entry)" |
| R1-274 | F1 | 128 | 40x | reduction (no longer single-dataset) | deterministic | disputes | "stops presenting 40x as a single-dataset reduction" |
| R1-275 | F1 | 131 | 206 | GB (OLD memory path) | deterministic | quotes-paper | "The 206 GB -> 5.1 GB -> 26 GB memory reduction path is applicable to any sparse" |
| R1-276 | F1 | 131 | 5.1 | GB (OLD memory path) | deterministic | quotes-paper | "The 206 GB -> 5.1 GB -> 26 GB memory reduction path" |
| R1-277 | F1 | 131 | 26 | GB (OLD memory path) | deterministic | quotes-paper | "The 206 GB -> 5.1 GB -> 26 GB memory reduction path" |
| R1-278 | F1 | 132 | 206 | GB full-set dense (NEW text) | deterministic | requests | "The full-set 206 GB dense matrix, the 5.1 GB CSR of the mined subset, and the 26 GB GPU bitvector matrix" |
| R1-279 | F1 | 132 | 5.1 | GB CSR of mined subset (NEW text) | deterministic | requests | "the 5.1 GB CSR of the mined subset, and the 26 GB GPU bitvector matrix" |
| R1-280 | F1 | 132 | 26 | GB GPU bitvector matrix (NEW text) | deterministic | requests | "the 26 GB GPU bitvector matrix illustrate a compression path" |
| R1-281 | F1 | 132 | ~15x | same-subset dense-to-CSR reduction (NEW text) | deterministic | requests | "(the same-subset dense-to-CSR reduction is ~15x; see Appendix)" |
| R1-282 | F1 | 135 | 8 | UniProt accessions (to obtain and list) | deterministic | requests | "obtain the 8 UniProt accessions from the authors and list them" |
| R1-283 | F1 | 135 | 8 (eight) | matching proteins (proposed text) | deterministic | requests | "The eight matching proteins (accessions listed in Table) were identified with the released analysis code" |
| R1-284 | F1 | 136 | 2025_01 | UniProt release (fallback text) | external-fact | requests | "they are regenerable only from UniProt release 2025_01 via the released feature-extraction pipeline" |
| R1-285 | F1 | 138 | 0.45 | p bound (correct source to be stated) | deterministic | requests | "replace the phrase by the rule of three with the correct source of 0.45" |
| R1-286 | F1 | 139 | <0.45 | p (OLD caption) | deterministic | quotes-paper | "(p < 0.45, the 95% one-sided upper bound for 0 of 5 by the rule of three)." |
| R1-287 | F1 | 139 | 95% | one-sided upper bound (OLD caption) | method-parameter | quotes-paper | "(p < 0.45, the 95% one-sided upper bound for 0 of 5 by the rule of three)." |
| R1-288 | F1 | 139 | 0 of 5 | null runs (OLD caption) | deterministic | quotes-paper | "(p < 0.45, the 95% one-sided upper bound for 0 of 5 by the rule of three)." |
| R1-289 | F1 | 140 | <0.45 | p (NEW caption) | deterministic | requests | "(p < 0.45, the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5)." |
| R1-290 | F1 | 140 | 95% | binomial upper bound (NEW caption) | method-parameter | requests | "the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5" |
| R1-291 | F1 | 140 | 1-0.05^{1/5} | bound formula (NEW caption) | deterministic | requests | "the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5" |
| R1-292 | F1 | 140 | 0 of 5 | null runs (NEW caption) | deterministic | requests | "the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5" |
| R1-293 | F1 | 141 | <0.45 | p (paper line 391) | deterministic | quotes-paper | "yielding p < 0.45 (the 95% one-sided upper bound for 0 of 5 by the rule of three)" |
| R1-294 | F1 | 141 | 95% | one-sided upper bound (paper line 391) | method-parameter | quotes-paper | "(the 95% one-sided upper bound for 0 of 5 by the rule of three)" |
| R1-295 | F1 | 141 | 0 of 5 | null runs (paper line 391) | deterministic | quotes-paper | "(the 95% one-sided upper bound for 0 of 5 by the rule of three)" |
| R1-296 | F1 | 141 | 1-0.05^{1/5} | bound formula (replacement) | deterministic | requests | "... the exact one-sided binomial 95% upper bound 1-0.05^{1/5} for 0 of 5" |
| R1-297 | F1 | 141 | <0.45 | p (table cell, paper line 382) | deterministic | quotes-paper | "Line 382 table cell shows only <0.45 and needs no change once the caption is fixed" |
| R1-298 | F1 | 143 | 4 | min_count (unverifiable run) | method-parameter | disputes | "No artifact records a min_count=4 run or a 48M-itemset count" |
| R1-299 | F1 | 143 | 48M | itemsets (unverifiable) | deterministic | disputes | "No artifact records a min_count=4 run or a 48M-itemset count" |
| R1-300 | F1 | 143 | 48M | transactions written during extraction | deterministic | asserts-own | "(the only '48M' in logs is transactions written during extraction)" |
| R1-301 | F1 | 144 | 4 | min_count (OLD text) | method-parameter | quotes-paper | "A verification run at min_count=4 discovered 48 million itemsets with identical maximum" |
| R1-302 | F1 | 144 | 48 million | itemsets (OLD text) | deterministic | disputes | "A verification run at min_count=4 discovered 48 million itemsets with identical maximum" |
| R1-303 | F1 | 144 | 8 | proteins (NEW text) | deterministic | requests | "Because each of the 8 proteins carries exactly 22 vocabulary features, K=23 is impossible at any support threshold" |
| R1-304 | F1 | 144 | 22 | vocabulary features per K=22 protein (NEW text) | deterministic | requests | "each of the 8 proteins carries exactly 22 vocabulary features" |
| R1-305 | F1 | 144 | 23 | K (impossible, NEW text) | deterministic | requests | "K=23 is impossible at any support threshold; the ceiling is therefore structural" |
| R1-306 | F1 | 146 | 13 | K (highlighted pattern) | deterministic | quotes-paper | "the highlighted K=13 / K=12 / K=11 patterns from decoded_top_k_patterns.txt" |
| R1-307 | F1 | 146 | 12 | K (highlighted pattern) | deterministic | quotes-paper | "the highlighted K=13 / K=12 / K=11 patterns from decoded_top_k_patterns.txt" |
| R1-308 | F1 | 146 | 11 | K (highlighted pattern) | deterministic | quotes-paper | "the highlighted K=13 / K=12 / K=11 patterns from decoded_top_k_patterns.txt" |
| R1-309 | F1 | 146 | ~11,000 | proteins (K=13 pattern) | deterministic | quotes-paper | "replace (~11,000 proteins), (~10,500 proteins), (~16,000 proteins) with the exact integers" |
| R1-310 | F1 | 146 | ~10,500 | proteins (K=12 pattern) | deterministic | quotes-paper | "replace (~11,000 proteins), (~10,500 proteins), (~16,000 proteins) with the exact integers" |
| R1-311 | F1 | 146 | ~16,000 | proteins (K=11 pattern) | deterministic | quotes-paper | "replace (~11,000 proteins), (~10,500 proteins), (~16,000 proteins) with the exact integers" |
| R1-312 | F1 | 148 | 2025_01 | UniProt/Swiss-Prot release (paper line 478) | external-fact | disputes | "line 478 says UniProt/Swiss-Prot release 2025_01" (vs TrEMBL at line 129) |
| R1-313 | F1 | 149 | 2025_01 | UniProt/Swiss-Prot release (OLD) | external-fact | quotes-paper | "our results reflect UniProt/Swiss-Prot release 2025_01" |
| R1-314 | F1 | 150 | 2025_01 | UniProt TrEMBL release (NEW) | external-fact | requests | "our results reflect UniProt TrEMBL release 2025_01" |
| R1-315 | F1 | 152 | 2025_01 | release string (not in any artifact) | external-fact | disputes | "The '2025_01' release is not recorded in any artifact. Confirm the exact release with the authors" |
| R1-316 | F1 | 152 | February 2026 | access date (softened wording) | external-fact | requests | "soften to 'the UniProt TrEMBL release accessed February 2026'" |
| R1-317 | F1 | 154 | n=5 | permutations (scope to add to headline Z) | method-parameter | requests | "[Integrity 4] Conclusion line 498 - add the n=5 scope to the headline Z." |
| R1-318 | F1 | 155 | >3,700 | Z (OLD conclusion) | deterministic | quotes-paper | "(Z>3,700 for K=4--6; no null run reached K>=7)." |
| R1-319 | F1 | 155 | 4-6 | K range (OLD conclusion) | deterministic | quotes-paper | "(Z>3,700 for K=4--6; no null run reached K>=7)." |
| R1-320 | F1 | 155 | >=7 | K (no null run reached, OLD) | deterministic | quotes-paper | "(Z>3,700 for K=4--6; no null run reached K>=7)." |
| R1-321 | F1 | 156 | >3,700 | Z (NEW conclusion) | deterministic | requests | "(Z>3,700 for K=4--6, as t-statistics from 5 permutations; no null run reached K>=7)" |
| R1-322 | F1 | 156 | 4-6 | K range (NEW conclusion) | deterministic | requests | "(Z>3,700 for K=4--6, as t-statistics from 5 permutations" |
| R1-323 | F1 | 156 | 5 | permutations (NEW conclusion) | method-parameter | requests | "as t-statistics from 5 permutations; no null run reached K>=7" |
| R1-324 | F1 | 156 | >=7 | K (NEW conclusion) | deterministic | requests | "no null run reached K>=7; see Section" |
| R1-325 | F1 | 158 | 7.3 | minutes (paper line 120) | hardware-dependent | quotes-paper | "discovers the complete feature co-occurrence landscape in 7.3 minutes -> ... in 7.3 minutes of mining" |
| R1-326 | F1 | 158 | 7.3 | minutes (paper line 496) | hardware-dependent | quotes-paper | "mining completes in 7.3 minutes -> mining completes in 7.3 minutes (excluding the 63-minute one-time feature extraction" |
| R1-327 | F1 | 158 | 63 | minute feature extraction (NEW text) | hardware-dependent | requests | "(excluding the 63-minute one-time feature extraction, Section)" |
| R1-328 | F2 | 12 | 76.9 million | multi-feature proteins | deterministic | quotes-paper | "across 76.9 million multi-feature proteins (1,002 features) on a single NVIDIA H100 in 7.3 minutes" |
| R1-329 | F2 | 12 | 1,002 | features | deterministic | quotes-paper | "across 76.9 million multi-feature proteins (1,002 features)" |
| R1-330 | F2 | 12 | 1 (single) | NVIDIA H100 GPU | hardware-dependent | quotes-paper | "on a single NVIDIA H100 in 7.3 minutes" |
| R1-331 | F2 | 12 | 7.3 | minutes | hardware-dependent | quotes-paper | "on a single NVIDIA H100 in 7.3 minutes" |
| R1-332 | F2 | 12 | 26.8 million | co-occurrence patterns | deterministic | quotes-paper | "discovering 26.8 million co-occurrence patterns up to K = 22" |
| R1-333 | F2 | 12 | 22 | K (max) | deterministic | quotes-paper | "discovering 26.8 million co-occurrence patterns up to K = 22" |
| R1-334 | F2 | 12 | ~3 orders of magnitude | transaction count vs prior single-machine GPU FIM | external-fact | quotes-paper | "roughly three orders of magnitude larger than any previously reported single-machine GPU FIM system[SOURCE?]" |
| R1-335 | F2 | 12 | 0.001% | null-model threshold | method-parameter | quotes-paper | "A permutation null model (at the 0.001% threshold)" |
| R1-336 | F2 | 14 | 3 (three) | ways claims outrun support | external-fact | asserts-own | "outrun its statistical and reproducibility support in three specific ways (Major 1-3 below)" |
| R1-337 | F2 | 14 | 22 | K (flagship biological result) | deterministic | quotes-paper | "the flagship biological result (K = 22) rests on the thinnest possible evidence base" |
| R1-338 | F2 | 18 | 21x | speedup (Direct vs SON) | hardware-dependent | quotes-paper | "a convincing demonstration (21x / 95.2% miss-rate) of why the SON approximation is inadequate" |
| R1-339 | F2 | 18 | 95.2% | SON miss rate | deterministic | quotes-paper | "a convincing demonstration (21x / 95.2% miss-rate)" |
| R1-340 | F2 | 22 | 0.001% | threshold validated by null model | method-parameter | quotes-paper | "The null model validates only the 0.001% threshold; the headline K = 22 result lives at 0.00001%" |
| R1-341 | F2 | 22 | 22 | K (headline) | deterministic | quotes-paper | "the headline K = 22 result lives at 0.00001%, where no null model was run" |
| R1-342 | F2 | 22 | 0.00001% | threshold (headline result) | method-parameter | quotes-paper | "the headline K = 22 result lives at 0.00001%, where no null model was run" |
| R1-343 | F2 | 23 | n = 5 | permutations | method-parameter | disputes | "Statistical support rests on n = 5 permutations; the reported 'Z' magnitudes and the p < 0.45 bound cannot carry the weight" |
| R1-344 | F2 | 23 | <0.45 | p bound | deterministic | disputes | "the reported 'Z' magnitudes and the p < 0.45 bound cannot carry the weight the narrative places on them" |
| R1-345 | F2 | 24 | 22 | K (flagship pattern) | deterministic | quotes-paper | "The flagship K = 22 'neuronal antiviral' pattern is a single itemset in 8 proteins" |
| R1-346 | F2 | 24 | 8 | proteins (K=22) | deterministic | quotes-paper | "a single itemset in 8 proteins whose accessions are not listed" |
| R1-347 | F2 | 31 | 0.001% | support (Power threshold, null model) | method-parameter | quotes-paper | "run only at the Power threshold (0.001% support, min_count = 769)" |
| R1-348 | F2 | 31 | 769 | min_count (null model) | method-parameter | quotes-paper | "run only at the Power threshold (0.001% support, min_count = 769)" |
| R1-349 | F2 | 31 | 9 | K (distribution peak) | deterministic | quotes-paper | "the K = 9 distribution peak and the K = 22 ceiling - are obtained at the Opus threshold" |
| R1-350 | F2 | 31 | 22 | K (ceiling) | deterministic | quotes-paper | "the K = 9 distribution peak and the K = 22 ceiling - are obtained at the Opus threshold" |
| R1-351 | F2 | 31 | 0.00001% | support (Opus threshold) | method-parameter | quotes-paper | "obtained at the Opus threshold (0.00001%, min_count = 8)" |
| R1-352 | F2 | 31 | 8 | min_count (Opus) | method-parameter | quotes-paper | "obtained at the Opus threshold (0.00001%, min_count = 8)" |
| R1-353 | F2 | 33 | 15-22 | K range (deep patterns lacking null validation) | deterministic | asserts-own | "whether the K = 15-22 patterns are enriched over a count-preserving null" |
| R1-354 | F2 | 35 | 5 | permutations (suggested Opus-threshold null) | method-parameter | requests | "even 5 permutations at min_count = 8 would establish whether the null produces any deep (K >= 7) patterns" |
| R1-355 | F2 | 35 | 8 | min_count (suggested null run) | method-parameter | requests | "even 5 permutations at min_count = 8 would establish" |
| R1-356 | F2 | 35 | >=7 | K (deep-pattern cutoff) | method-parameter | requests | "whether the null produces any deep (K >= 7) patterns there" |
| R1-357 | F2 | 35 | >=15 | K (concentration analysis) | method-parameter | requests | "how many distinct proteins contribute to all K >= 15 itemsets" |
| R1-358 | F2 | 37 | n = 5 | permutations (heading) | method-parameter | disputes | "Major 2 - Statistical inference from n = 5 permutations is over-leveraged" |
| R1-359 | F2 | 38 | 4 | df (t-statistics) | deterministic | quotes-paper | "correctly labels the enrichment statistics as t-statistics (4 df) rather than large-sample Z-scores" |
| R1-360 | F2 | 38 | <0.45 | p (absence bound) | deterministic | quotes-paper | "corrects the absence bound to p < 0.45 (rule of three, 0/5)" |
| R1-361 | F2 | 38 | 0/5 | null runs with deep patterns | deterministic | quotes-paper | "corrects the absence bound to p < 0.45 (rule of three, 0/5)" |
| R1-362 | F2 | 38 | 5 (five) | permutations | method-parameter | disputes | "five permutations is too few to support the inferential language still used around them" |
| R1-363 | F2 | 41 | <0.45 | p | deterministic | disputes | "A p < 0.45 bound does not reject the null at any conventional level" |
| R1-364 | F2 | 41 | >=7 | K | deterministic | quotes-paper | "'K>=7 patterns are absent from all 5 null runs'" |
| R1-365 | F2 | 41 | 5 | null runs | method-parameter | quotes-paper | "'K>=7 patterns are absent from all 5 null runs'" |
| R1-366 | F2 | 42 | +71,728 | effect size (Z at K=6) | deterministic | quotes-paper | "Reporting a '+71,728' effect size from 5 permutations invites the objection" |
| R1-367 | F2 | 42 | 5 | permutations | method-parameter | quotes-paper | "Reporting a '+71,728' effect size from 5 permutations" |
| R1-368 | F2 | 42 | n = 5 | permutations (sigma estimate) | method-parameter | disputes | "sigma estimated from n = 5 is itself extremely noisy" |
| R1-369 | F2 | 42 | 6 | K (value not reproducing from rounded mu/sigma) | deterministic | disputes | "the K = 6 value does not reproduce cleanly from the rounded mu/sigma shown" |
| R1-370 | F2 | 42 | 4-6 | K range (raw null counts requested) | deterministic | requests | "reporting the raw null counts per permutation for K = 4-6" |
| R1-371 | F2 | 43 | 100+ | permutations needed | method-parameter | quotes-paper | "'100+ permutations would be needed to establish p < 0.01 bounds.'" |
| R1-372 | F2 | 43 | <0.01 | p (target bound) | method-parameter | quotes-paper | "'100+ permutations would be needed to establish p < 0.01 bounds.'" |
| R1-373 | F2 | 43 | ~130 | s per permutation | hardware-dependent | quotes-paper | "Since each permutation is ~130 s, 100 permutations is ~3.6 GPU-hours" |
| R1-374 | F2 | 43 | 100 | permutations | method-parameter | asserts-own | "100 permutations is ~3.6 GPU-hours - entirely feasible on the same single H100" |
| R1-375 | F2 | 43 | ~3.6 | GPU-hours | hardware-dependent | asserts-own | "100 permutations is ~3.6 GPU-hours - entirely feasible on the same single H100" |
| R1-376 | F2 | 43 | 1 (single) | H100 | hardware-dependent | quotes-paper | "entirely feasible on the same single H100" |
| R1-377 | F2 | 45 | 22 | K (heading, flagship result) | deterministic | quotes-paper | "Major 3 - The flagship K = 22 biological result is under-supported and not fully reproducible" |
| R1-378 | F2 | 46 | 22 | K (flagship pattern) | deterministic | quotes-paper | "The K = 22 'neuronal antiviral RNA-helicase' pattern is given prominence" |
| R1-379 | F2 | 46 | 1 (single) | itemset | deterministic | quotes-paper | "As currently supported it is a single itemset shared by 8 proteins" |
| R1-380 | F2 | 46 | 8 | proteins | deterministic | quotes-paper | "As currently supported it is a single itemset shared by 8 proteins" |
| R1-381 | F2 | 46 | 0 | GO parent-child pairs | deterministic | quotes-paper | "the GO true-path check (0 parent-child pairs; one InterPro2GO-derived link, giving 21 independent features)" |
| R1-382 | F2 | 46 | 1 (one) | InterPro2GO-derived link | deterministic | quotes-paper | "(0 parent-child pairs; one InterPro2GO-derived link, giving 21 independent features)" |
| R1-383 | F2 | 46 | 21 | independent features | deterministic | quotes-paper | "(0 parent-child pairs; one InterPro2GO-derived link, giving 21 independent features)" |
| R1-384 | F2 | 47 | 8 | UniProt accessions (not listed) | deterministic | requests | "The 8 UniProt accessions are not listed." |
| R1-385 | F2 | 47 | 8 (eight) | identifiers (supplementary table) | deterministic | requests | "a reviewer will expect the eight identifiers in a supplementary table" |
| R1-386 | F2 | 48 | n = 8 | proteins | deterministic | quotes-paper | "With n = 8 proteins, a single densely (electronically) annotated protein family can generate the entire signature" |
| R1-387 | F2 | 48 | 1 (single) | protein family (could generate signature) | deterministic | asserts-own | "a single densely (electronically) annotated protein family can generate the entire signature" |
| R1-388 | F2 | 50 | 8 | accessions (supplementary table) | deterministic | requests | "add a supplementary table of the 8 accessions with per-term evidence codes" |
| R1-389 | F2 | 50 | 22 | K (pattern to demote) | deterministic | quotes-paper | "demote the K = 22 pattern from 'highlighted discovery' to 'illustrative maximum-depth example'" |
| R1-390 | F2 | 53 | 5.1x | more transactions than prior systems | deterministic | disputes | "the '5.1x more transactions' and 'deeper than any previously reported GPU FIM result' claims are therefore about scale reached" |
| R1-391 | F2 | 53 | 21x | speedup (Direct-GPU vs SON) | hardware-dependent | quotes-paper | "Direct-GPU vs SON at identical support (21x, 95.2% miss rate) - is excellent" |
| R1-392 | F2 | 53 | 95.2% | SON miss rate | deterministic | quotes-paper | "Direct-GPU vs SON at identical support (21x, 95.2% miss rate)" |
| R1-393 | F2 | 53 | 76.9M | transactions | deterministic | asserts-own | "Prior tools plausibly cannot ingest 76.9M transactions" |
| R1-394 | F2 | 53 | 1-15M | transaction slice (suggested subset comparison) | method-parameter | requests | "even a subset comparison (e.g., ET-miner vs GMiner on a 1-15M-transaction slice) would substantiate the efficiency claim" |
| R1-395 | F2 | 56 | 26.8 million | patterns (headline) | deterministic | disputes | "The headline '26.8 million patterns' is the complete frequent-itemset space, which by construction is dominated by subset/superset redundancy" |
| R1-396 | F2 | 56 | 2^K - 2 | frequent sub-itemsets per frequent K-itemset | deterministic | asserts-own | "(every frequent K-itemset implies 2^K - 2 frequent sub-itemsets)" |
| R1-397 | F2 | 62 | 7.3 | minutes (mining only) | hardware-dependent | quotes-paper | "the '7.3 minutes (mining only)' qualifier is good" |
| R1-398 | F2 | 63 | 3 (three) | distinct memory quantities coexisting | external-fact | asserts-own | "Three legitimate but different quantities coexist - ~206 GB ... ~26 GB ... ~5.1 GB ... ~10 GB" (four listed) |
| R1-399 | F2 | 63 | ~206 | GB (naive 1-byte dense) | deterministic | quotes-paper | "~206 GB (naive 1-byte dense)" |
| R1-400 | F2 | 63 | 1 | byte per entry (naive dense) | method-parameter | quotes-paper | "~206 GB (naive 1-byte dense)" |
| R1-401 | F2 | 63 | ~26 | GB (bit-packed dense / on-GPU bitvector, 214M set) | deterministic | quotes-paper | "~26 GB (bit-packed dense / on-GPU bitvector, 214M set)" |
| R1-402 | F2 | 63 | 214M | set (proteins) | deterministic | quotes-paper | "~26 GB (bit-packed dense / on-GPU bitvector, 214M set)" |
| R1-403 | F2 | 63 | ~5.1 | GB (CSR-COO) | deterministic | quotes-paper | "~5.1 GB (CSR-COO)" |
| R1-404 | F2 | 63 | ~10 | GB (bit-packed dense of the 76.9M subset) | deterministic | quotes-paper | "~10 GB (bit-packed dense of the 76.9M subset)" |
| R1-405 | F2 | 63 | 76.9M | subset (proteins) | deterministic | quotes-paper | "~10 GB (bit-packed dense of the 76.9M subset)" |
| R1-406 | F2 | 64 | 2 of 6 | pLDDT bins passing support threshold | deterministic | quotes-paper | "Only 2 of 6 pLDDT bins pass the support threshold" |
| R1-407 | F2 | 64 | 70-90 | pLDDT range (medium confidence bin) | method-parameter | quotes-paper | "the recurring one is 'medium confidence (70-90).'" |
| R1-408 | F2 | 65 | ~11,000 | proteins (intermediate-K count) | deterministic | quotes-paper | "Intermediate-K protein counts ('~11,000', '~10,500', '~16,000') are approximate where the artefacts support near-exact values" |
| R1-409 | F2 | 65 | ~10,500 | proteins (intermediate-K count) | deterministic | quotes-paper | "Intermediate-K protein counts ('~11,000', '~10,500', '~16,000') are approximate" |
| R1-410 | F2 | 65 | ~16,000 | proteins (intermediate-K count) | deterministic | quotes-paper | "Intermediate-K protein counts ('~11,000', '~10,500', '~16,000') are approximate" |
| R1-411 | F2 | 66 | 1 (single) | H100 (all experiments) | hardware-dependent | quotes-paper | "Now honestly framed as a capability ('all experiments used a single H100')" |
| R1-412 | F2 | 68 | 0 | GO parent-child pairs (K=22 set) | deterministic | quotes-paper | "The added quantification (0 parent-child pairs in the K = 22 set) is excellent" |
| R1-413 | F2 | 68 | 22 | K | deterministic | quotes-paper | "(0 parent-child pairs in the K = 22 set)" |
| R1-414 | F2 | 69 | 22 | K (accessions not deposited) | deterministic | quotes-paper | "some source artefacts (full transaction parquet, K = 22 accessions) are not currently deposited" |
| R1-415 | F2 | 75 | <0.45 | p (correction present) | deterministic | quotes-paper | "t-statistic footnote and p < 0.45 correction now present and correct" |
| R1-416 | F2 | 76 | 22 | K (Table 5 reference) | deterministic | quotes-paper | "s3.5 + Table 5 (K = 22): interpretation softened correctly; see Major 3 for accessions + evidence codes." |
| R1-417 | F2 | 77 | 768 | min_count (Power) | method-parameter | quotes-paper | "min_count values (Power = 768, Ultra = 16) reconcile with the mining logs" |
| R1-418 | F2 | 77 | 16 | min_count (Ultra) | method-parameter | quotes-paper | "min_count values (Power = 768, Ultra = 16) reconcile with the mining logs" |
| R1-419 | F2 | 78 | 15M | transactions (GMiner synthetic) | external-fact | quotes-paper | "the GMiner '15M synthetic / 1.7M real' labelling, which is now correct" |
| R1-420 | F2 | 78 | 1.7M | transactions (GMiner real) | external-fact | quotes-paper | "the GMiner '15M synthetic / 1.7M real' labelling, which is now correct" |
| R1-421 | F2 | 79 | 26,849,505 | K-distribution column sum | deterministic | asserts-own | "column sums to 26,849,505 exactly; matches the godmode log" |
| R1-422 | F2 | 85 | 8 | min_count (Opus, requested null) | method-parameter | requests | "Can the permutation null model be run at the Opus threshold (min_count = 8), even at n = 5" |
| R1-423 | F2 | 85 | n = 5 | permutations (requested) | method-parameter | requests | "even at n = 5, to test whether deep (K >= 7) patterns exceed a count-preserving null" |
| R1-424 | F2 | 85 | >=7 | K (deep) | method-parameter | requests | "to test whether deep (K >= 7) patterns exceed a count-preserving null there" |
| R1-425 | F2 | 85 | 0.001% | threshold (null model) | method-parameter | quotes-paper | "why is the count-preserving null at 0.001% argued to generalise to 0.00001%?" |
| R1-426 | F2 | 85 | 0.00001% | threshold (headline) | method-parameter | quotes-paper | "why is the count-preserving null at 0.001% argued to generalise to 0.00001%?" |
| R1-427 | F2 | 86 | ~130 | s per permutation | hardware-dependent | quotes-paper | "Each permutation is ~130 s." |
| R1-428 | F2 | 86 | 100 | permutations (requested) | method-parameter | requests | "Is there an obstacle to running the 100 permutations the manuscript itself identifies as necessary for p < 0.01?" |
| R1-429 | F2 | 86 | <0.01 | p | method-parameter | quotes-paper | "the 100 permutations the manuscript itself identifies as necessary for p < 0.01" |
| R1-430 | F2 | 87 | 8 | K=22 UniProt accessions (requested) | deterministic | requests | "Can the 8 K = 22 UniProt accessions and their per-GO-term evidence codes be provided" |
| R1-431 | F2 | 87 | 22 | K | deterministic | quotes-paper | "Can the 8 K = 22 UniProt accessions and their per-GO-term evidence codes be provided" |
| R1-432 | F2 | 88 | 26.8M | total itemsets | deterministic | quotes-paper | "What are the closed and maximal frequent-itemset counts corresponding to the 26.8M total?" |
| R1-433 | F2 | 90 | 9 | K (distribution peak) | deterministic | quotes-paper | "The K = 9 distribution peak: how sensitive is its position to the support threshold" |
| R1-434 | F2 | 90 | >=2 | features (multi-feature inclusion criterion) | method-parameter | quotes-paper | "the multi-feature-protein inclusion criterion (>= 2 features)" |
| R1-435 | F2 | 96 | n = 5 | permutations | method-parameter | disputes | "n = 5 cannot support the inferential weight placed on it (Major 2)" |
| R1-436 | F2 | 96 | 8 | proteins (anecdote) | deterministic | quotes-paper | "the single most-highlighted pattern is an 8-protein anecdote without listed accessions" |
| R1-437 | F2 | 96 | <=4 | GPU-hours (to fix Majors 1-2) | hardware-dependent | asserts-own | "Two of these are fixable with <= 4 GPU-hours on the machine already used (100 permutations; an Opus-threshold null)" |
| R1-438 | F2 | 96 | 100 | permutations | method-parameter | requests | "fixable with <= 4 GPU-hours on the machine already used (100 permutations; an Opus-threshold null)" |
| R1-439 | F2 | 105 | hundreds of millions | proteins | deterministic | quotes-paper | "'Which combinations of protein features co-occur more than expected across hundreds of millions of proteins?'" |
| R1-440 | F2 | 106 | 22 | K | deterministic | quotes-paper | "Exhaustive FIM to K=22 on 76.9M proteins in 7.3 min; unimodal K-peak at 9; null-model enrichment for K>=4 at 0.001%" |
| R1-441 | F2 | 106 | 76.9M | proteins | deterministic | quotes-paper | "Exhaustive FIM to K=22 on 76.9M proteins in 7.3 min" |
| R1-442 | F2 | 106 | 7.3 | min | hardware-dependent | quotes-paper | "Exhaustive FIM to K=22 on 76.9M proteins in 7.3 min" |
| R1-443 | F2 | 106 | 9 | K (peak) | deterministic | quotes-paper | "unimodal K-peak at 9" |
| R1-444 | F2 | 106 | >=4 | K (null-model enrichment) | deterministic | quotes-paper | "null-model enrichment for K>=4 at 0.001%" |
| R1-445 | F2 | 106 | 0.001% | threshold | method-parameter | quotes-paper | "null-model enrichment for K>=4 at 0.001%" |
| R1-446 | F2 | 114 | 22 | K (accessions not deposited) | deterministic | quotes-paper | "but source transaction data + K=22 accessions not deposited; Major 3, Minor 8" |
| R1-447 | F2 | 114 | n=5 | permutations (null model, checklist) | method-parameter | quotes-paper | "Statistics (null-model n=5; Major 2)" |
| R1-448 | F2 | 115 | n=5 | permutations (no power analysis) | method-parameter | quotes-paper | "sample size / power (no power analysis; n=5 permutations)" |
| R1-449 | F2 | 115 | 42 | seed (Fisher-Yates shuffle) | method-parameter | quotes-paper | "randomization (Fisher-Yates shuffle, seed 42)" |
| R1-450 | F2 | 115 | >=2 | features (inclusion) | method-parameter | quotes-paper | "inclusion/exclusion (>=2 features, min_count thresholds)" |
| R1-451 | F2 | 115 | 2025_01 | UniProt release | external-fact | quotes-paper | "data-collection protocol (UniProt release 2025_01)" |
| R1-452 | F2 | 115 | 3.10 | Python version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-453 | F2 | 115 | 13.0 | CuPy version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-454 | F2 | 115 | 1.26 | NumPy version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-455 | F2 | 115 | 12.4 | CUDA version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-456 | F2 | 115 | 22.04 | Ubuntu version | software | quotes-paper | "software versions (Python 3.10/CuPy 13.0/NumPy 1.26/CUDA 12.4/Ubuntu 22.04, H100 80GB SXM5)" |
| R1-457 | F2 | 115 | 80 | GB VRAM (H100 SXM5) | hardware-dependent | quotes-paper | "H100 80GB SXM5" |
| R1-458 | F2 | 115 | millions | itemsets tested without FDR | deterministic | asserts-own | "multiple-comparison correction (millions of itemsets tested; no FDR - acknowledged as Limitation 1 but not applied)" |
| R1-459 | F2 | 116 | 4 | figures present on disk | external-fact | asserts-own | "Figures/Tables (4 figures present on disk; tables well-formed)" |
| R1-460 | F2 | 116 | 22 | K (interpretation still prominent) | deterministic | quotes-paper | "Objectivity (K=22 interpretation softened but still prominent; Major 3)" |
| R1-461 | F2 | 117 | 5 | highlighted patterns | deterministic | quotes-paper | "selective reporting (5 highlighted patterns chosen for known validation - disclosed, acceptable)" |
| R1-462 | F2 | 117 | n=5 | permutations ('Z' inappropriate) | method-parameter | quotes-paper | "inappropriate tests (n=5 'Z')" |
| R1-463 | F2 | 118 | 5 | explicit limitations | external-fact | quotes-paper | "Limitations (5 explicit limitations, now including 0-parent-child note)" |
| R1-464 | F2 | 118 | 0 | parent-child pairs | deterministic | quotes-paper | "(5 explicit limitations, now including 0-parent-child note)" |
| R1-465 | F2 | 118 | 22 | K (interpretation exception) | deterministic | quotes-paper | "Interpretation (mostly data-supported; K=22 the exception)" |
| R1-466 | F2 | 119 | 26.8M | 'patterns' (redundant space) | deterministic | disputes | "overstated conclusions (26.8M 'patterns' counts redundant space; Major 5)" |
| R1-467 | F2 | 120 | 2024 | reference currency (through year) | external-fact | asserts-own | "Currency (through 2024)" |
| R1-468 | F2 | 120 | 13 | citation errors corrected this revision | external-fact | asserts-own | "Accuracy (13 citation errors corrected this revision; independently re-verified)" |
| R1-469 | F2 | 120 | 1 (one) | uncited bibitem | external-fact | asserts-own | "one uncited bibitem (miettinen2020; Minor 6)" |
| R1-470 | F2 | 123 | n=5 | permutations (normality assumption) | method-parameter | quotes-paper | "'Z'-framing assumes normality of null counts from n=5; not tested" |
| R1-471 | F2 | 124 | n=5 | permutations (no power analysis) | method-parameter | quotes-paper | "Sample-size/power justification (n=5, no power analysis; Major 2)" |
| R1-472 | F2 | 125 | 5 | permutations (replication) | method-parameter | quotes-paper | "replication (5 permutations)" |
| R1-473 | F2 | 129 | 8 | K=22 accessions not listed | deterministic | quotes-paper | "the 8 K=22 accessions not listed - Major 3" |
| R1-474 | F2 | 129 | 22 | K | deterministic | quotes-paper | "the 8 K=22 accessions not listed - Major 3" |
| R1-475 | F2 | 135 | 5 | permutation spread (could be shown) | method-parameter | requests | "the K-distribution/null comparison could show the 5-permutation spread" |
| R1-476 | F2 | 138 | 2 (two) | orphaned figure files on disk | external-fact | asserts-own | "two orphaned figure files (figures/kdist_shift.pdf, figures/pruning_savings.pdf) remain on disk from the removed expanded run" |
| R1-477 | F2 | 143 | 4.6 | Claude Code Opus version (listed co-author) | software | quotes-paper | "(C. claudya = 'Anthropic, Claude Code Opus 4.6' as a listed author)" |
| R1-478 | F2 | 143 | 1 (one) | author affiliated with AI vendor | external-fact | asserts-own | "one author is affiliated with the AI vendor whose tool co-produced the work - disclosure expected" |
| R1-479 | F2 | 143 | v1 | Zenodo preprint version | software | asserts-own | "a Zenodo v1 exists - the relationship to v1 should be disclosed as a versioned preprint update" |
| R1-480 | F2 | 151 | 5 | major concerns | external-fact | asserts-own | "Major concerns identified & justified (5)" |
| R1-481 | F2 | 151 | 8 | minor issues | external-fact | asserts-own | "Minor issues categorized (8)" |
| R1-482 | F2 | 151 | 3 | missing statements (COI, funding, subjects) | external-fact | asserts-own | "Ethics verified (Stage 6 - surfaced 3 missing statements: COI, funding, subjects)" |
| R1-483 | F2 | 156 | 4.6 | Claude Code Opus version (AI co-author) | software | quotes-paper | "AI-tool listed as a co-author ('Claude Code Opus 4.6')" |
| R1-484 | F2 | 158 | millions | itemsets (no multiple-testing correction) | deterministic | asserts-own | "No multiple-testing correction applied to the millions of itemsets" |
| R1-485 | F2 | 159 | 2 (two) | orphaned figure PDFs | external-fact | asserts-own | "Two orphaned figure PDFs (kdist_shift.pdf, pruning_savings.pdf) from the removed expanded run remain in figures/" |
| R1-486 | F2 | 160 | v1 | Zenodo version (relationship to disclose) | software | asserts-own | "Relationship to the published Zenodo v1 should be disclosed as a versioned-preprint update" |
| R1-487 | F2 | 171 | v1 | Zenodo version (disclosed post-review) | software | asserts-own | "discloses the Zenodo-v1 versioned-preprint relationship" |
| R1-488 | F2 | 173 | 0 (zero) | uncited bibitems remaining | external-fact | asserts-own | "miettinen2020 - now cited (Boolean-matrix representation, s1); zero uncited bibitems remain" |
| R1-489 | F2 | 179 | 100 | permutations (still open) | method-parameter | requests | "Major 2 (100 permutations)" |
| R1-490 | F2 | 179 | 22 | K (accessions + evidence codes still open) | deterministic | requests | "Major 3 (K=22 accessions + evidence codes)" |
| R1-491 | F3 | 9 | 606K | itemsets (35K expansion) | deterministic | asserts-own | "v2 (2026-02-21): Updated with 35K-feature expansion (606K itemsets, K_max=17, null model, SON infeasibility)" |
| R1-492 | F3 | 9 | 17 | K_max (35K expansion) | deterministic | asserts-own | "35K-feature expansion (606K itemsets, K_max=17, null model, SON infeasibility)" |
| R1-493 | F3 | 15 | 1,002 | features (original vocabulary) | deterministic | quotes-paper | "The authors have expanded from 1,002 features to 34,920 features (35x vocabulary expansion)" |
| R1-494 | F3 | 15 | 34,920 | features (expanded vocabulary) | deterministic | asserts-own | "expanded from 1,002 features to 34,920 features (35x vocabulary expansion)" |
| R1-495 | F3 | 15 | 35x | vocabulary expansion factor | deterministic | asserts-own | "expanded from 1,002 features to 34,920 features (35x vocabulary expansion)" |
| R1-496 | F3 | 15 | 3 (three) | new result files | external-fact | asserts-own | "and produced three new result files:" |
| R1-497 | F3 | 17 | 3 (triplicate) | Direct GPU runs | method-parameter | asserts-own | "Direct GPU triplicate (experiment_direct_vs_son_35k_20260221.json)" |
| R1-498 | F3 | 17 | 606,292 | mean itemsets (triplicate) | deterministic | asserts-own | "606,292 mean itemsets (std=28), K_max=17, 654s mean on 4xH200" |
| R1-499 | F3 | 17 | 28 | std itemsets (triplicate) | deterministic | asserts-own | "606,292 mean itemsets (std=28), K_max=17, 654s mean on 4xH200" |
| R1-500 | F3 | 17 | 17 | K_max | deterministic | asserts-own | "606,292 mean itemsets (std=28), K_max=17, 654s mean on 4xH200" |
| R1-501 | F3 | 17 | 654 | s mean runtime | hardware-dependent | asserts-own | "606,292 mean itemsets (std=28), K_max=17, 654s mean on 4xH200" |
| R1-502 | F3 | 17 | 4xH200 | GPUs | hardware-dependent | asserts-own | "654s mean on 4xH200. SON is INFEASIBLE (175 GB bitvec > 143 GB VRAM)" |
| R1-503 | F3 | 17 | 175 | GB (SON bitvector) | deterministic | asserts-own | "SON is INFEASIBLE (175 GB bitvec > 143 GB VRAM)" |
| R1-504 | F3 | 17 | 143 | GB VRAM (H200) | hardware-dependent | asserts-own | "SON is INFEASIBLE (175 GB bitvec > 143 GB VRAM)" |
| R1-505 | F3 | 18 | 2 | permutations (35K null model) | method-parameter | asserts-own | "Null model (experiment_null_model_20260221_003203.json): 2 permutations" |
| R1-506 | F3 | 18 | 2 | K (null peak) | deterministic | asserts-own | "Null peaks at K=2 (59K itemsets), collapses by K=5 (1 itemset)" |
| R1-507 | F3 | 18 | 59K | null itemsets at K=2 | deterministic | asserts-own | "Null peaks at K=2 (59K itemsets), collapses by K=5 (1 itemset)" |
| R1-508 | F3 | 18 | 5 | K (null collapse) | deterministic | asserts-own | "Null peaks at K=2 (59K itemsets), collapses by K=5 (1 itemset)" |
| R1-509 | F3 | 18 | 1 | null itemsets at K=5 | deterministic | asserts-own | "collapses by K=5 (1 itemset)" |
| R1-510 | F3 | 18 | 17 | K (biological max, 35K) | deterministic | asserts-own | "Biological data extends to K=17." |
| R1-511 | F3 | 18 | 5.45x | total enrichment ratio | deterministic | asserts-own | "5.45x total enrichment ratio. Z=723 at K=3, Z=20,585 at K=4." |
| R1-512 | F3 | 18 | 723 | Z at K=3 (35K) | deterministic | asserts-own | "Z=723 at K=3, Z=20,585 at K=4." |
| R1-513 | F3 | 18 | 20,585 | Z at K=4 (35K) | deterministic | asserts-own | "Z=723 at K=3, Z=20,585 at K=4." |
| R1-514 | F3 | 19 | 0.1% | support (wave 3) | method-parameter | asserts-own | "Mining at 0.1% support reaches K=10+ with millions of itemsets per level" |
| R1-515 | F3 | 19 | 10+ | K reached (wave 3) | deterministic | asserts-own | "Mining at 0.1% support reaches K=10+ with millions of itemsets per level" |
| R1-516 | F3 | 19 | millions | itemsets per level (wave 3) | deterministic | asserts-own | "reaches K=10+ with millions of itemsets per level" |
| R1-517 | F3 | 19 | 743 | s elapsed at K=9 (wave 3) | hardware-dependent | asserts-own | "still running after 743s at K=9 (10M itemsets at that level alone)" |
| R1-518 | F3 | 19 | 9 | K (wave 3 progress) | deterministic | asserts-own | "still running after 743s at K=9 (10M itemsets at that level alone)" |
| R1-519 | F3 | 19 | 10M | itemsets at K=9 (wave 3) | deterministic | asserts-own | "still running after 743s at K=9 (10M itemsets at that level alone)" |
| R1-520 | F3 | 27 | 9/10 | severity (M1) | external-fact | asserts-own | "M1. Two permutations is worse than five -- the null model has regressed [Severity: 9/10]" |
| R1-521 | F3 | 29 | 5 | permutations (1K null model) | method-parameter | quotes-paper | "The original 1K-feature null model used 5 permutations." |
| R1-522 | F3 | 29 | 2 | permutations (35K null model) | method-parameter | asserts-own | "The 35K-feature null model uses only 2 permutations" |
| R1-523 | F3 | 29 | 7138484576005690180 | seed (permutation 1) | method-parameter | asserts-own | "(seeds 7138484576005690180 and 4047939128787533792, experiment_null_model_20260221_003203.json line 8)" |
| R1-524 | F3 | 29 | 4047939128787533792 | seed (permutation 2) | method-parameter | asserts-own | "(seeds 7138484576005690180 and 4047939128787533792, experiment_null_model_20260221_003203.json line 8)" |
| R1-525 | F3 | 33 | n=2 | data points (null) | method-parameter | disputes | "Variance from n=2 is meaningless. With 2 data points, you have exactly 1 degree of freedom." |
| R1-526 | F3 | 33 | 1 | degree of freedom | deterministic | asserts-own | "With 2 data points, you have exactly 1 degree of freedom." |
| R1-527 | F3 | 33 | 123.7 | std of null at K=2 | deterministic | asserts-own | "The reported standard deviations (e.g., K=2: std=123.7; K=3: std=51.6; K=4: std=2.8)" |
| R1-528 | F3 | 33 | 51.6 | std of null at K=3 | deterministic | asserts-own | "(e.g., K=2: std=123.7; K=3: std=51.6; K=4: std=2.8)" |
| R1-529 | F3 | 33 | 2.8 | std of null at K=4 | deterministic | asserts-own | "(e.g., K=2: std=123.7; K=3: std=51.6; K=4: std=2.8)" |
| R1-530 | F3 | 33 | 722.91 | Z at K=3 (35K) | deterministic | disputes | "The 'Z-score' at K=3 of 722.91 is computed as (53235 - 15919.5) / 51.6" |
| R1-531 | F3 | 33 | 53235 | biological itemsets at K=3 (35K) | deterministic | asserts-own | "computed as (53235 - 15919.5) / 51.6 -- but the denominator is estimated from 2 observations" |
| R1-532 | F3 | 33 | 15919.5 | null mean itemsets at K=3 (35K) | deterministic | asserts-own | "computed as (53235 - 15919.5) / 51.6" |
| R1-533 | F3 | 33 | 51.6 | null std at K=3 (denominator) | deterministic | asserts-own | "computed as (53235 - 15919.5) / 51.6" |
| R1-534 | F3 | 33 | 2 | observations | method-parameter | asserts-own | "the denominator is estimated from 2 observations" |
| R1-535 | F3 | 33 | 1 | df (t-statistic equivalent) | deterministic | asserts-own | "A Z-statistic from n=2 is a t-statistic with 1 degree of freedom, which has no finite moments" |
| R1-536 | F3 | 35 | 5 | K (both null runs produce 1 itemset) | deterministic | asserts-own | "At K=5, both null runs produced exactly 1 itemset." |
| R1-537 | F3 | 35 | 1 | null itemsets at K=5 (each run) | deterministic | asserts-own | "At K=5, both null runs produced exactly 1 itemset." |
| R1-538 | F3 | 35 | 0.0 | std at K=5 | deterministic | asserts-own | "The standard deviation is 0.0 from n=2 identical values." |
| R1-539 | F3 | 35 | inf | z_score at K=5 (string) | deterministic | disputes | "The z_score is reported as 'inf' (a string, not a float -- line 104). This is a division-by-zero artifact" |
| R1-540 | F3 | 35 | 10 or 100 | permutations (hypothetical) | method-parameter | asserts-own | "With 10 or 100 permutations, the K=5 null count might range from 0 to 5" |
| R1-541 | F3 | 35 | 0 to 5 | null count range at K=5 (hypothetical) | deterministic | asserts-own | "the K=5 null count might range from 0 to 5, producing a finite and meaningful Z-score" |
| R1-542 | F3 | 35 | n=2 | permutations (identical values) | method-parameter | asserts-own | "The standard deviation is 0.0 from n=2 identical values." |
| R1-543 | F3 | 37 | 1,090 | min_count (35K null model) | method-parameter | asserts-own | "The 35K null uses min_count=1,090 (approximately 0.001% of 109M transactions, line 6-7)" |
| R1-544 | F3 | 37 | 0.001% | support (35K null) | method-parameter | asserts-own | "min_count=1,090 (approximately 0.001% of 109M transactions" |
| R1-545 | F3 | 37 | 109M | transactions (35K dataset) | deterministic | asserts-own | "(approximately 0.001% of 109M transactions, line 6-7)" |
| R1-546 | F3 | 37 | 1,093 | min_count (35K mining campaign) | method-parameter | asserts-own | "The 35K mining campaign uses min_count=1,093 (0.001%, line 9 of the direct_vs_son file)" |
| R1-547 | F3 | 37 | 0.001% | support (35K mining) | method-parameter | asserts-own | "The 35K mining campaign uses min_count=1,093 (0.001%" |
| R1-548 | F3 | 37 | 8 | min_count (paper Opus threshold) | method-parameter | quotes-paper | "neither matches the paper's Opus threshold (min_count=8)" |
| R1-549 | F3 | 39 | 5 | perms (1K null) | method-parameter | quotes-paper | "In the 1K-feature null model (5 perms), K=2 was depleted (Z=-987) and K=3 was depleted (Z=-143)" |
| R1-550 | F3 | 39 | -987 | Z at K=2 (1K) | deterministic | quotes-paper | "K=2 was depleted (Z=-987) and K=3 was depleted (Z=-143)" |
| R1-551 | F3 | 39 | -143 | Z at K=3 (1K) | deterministic | quotes-paper | "K=2 was depleted (Z=-987) and K=3 was depleted (Z=-143)" |
| R1-552 | F3 | 39 | 2 | perms (35K null) | method-parameter | asserts-own | "In the 35K-feature null model (2 perms), K=2 is again depleted (Z=-102)" |
| R1-553 | F3 | 39 | -102 | Z at K=2 (35K) | deterministic | asserts-own | "K=2 is again depleted (Z=-102) but K=3 is now massively enriched (Z=+723)" |
| R1-554 | F3 | 39 | +723 | Z at K=3 (35K) | deterministic | asserts-own | "K=3 is now massively enriched (Z=+723)" |
| R1-555 | F3 | 43 | 100+ | permutations (requested, 35K) | method-parameter | requests | "Run 100+ permutations at min_count=1,093 on the 35K dataset." |
| R1-556 | F3 | 43 | 1,093 | min_count (requested run) | method-parameter | requests | "Run 100+ permutations at min_count=1,093 on the 35K dataset." |
| R1-557 | F3 | 43 | ~655 | s per permutation (35K, 4xH200) | hardware-dependent | asserts-own | "At ~655s per permutation on 4xH200, 100 permutations = ~18 hours, 200 = ~36 hours." |
| R1-558 | F3 | 43 | 4xH200 | GPUs | hardware-dependent | asserts-own | "At ~655s per permutation on 4xH200" |
| R1-559 | F3 | 43 | 100 | permutations | method-parameter | asserts-own | "100 permutations = ~18 hours, 200 = ~36 hours" |
| R1-560 | F3 | 43 | ~18 | hours (100 perms, 35K) | hardware-dependent | asserts-own | "100 permutations = ~18 hours, 200 = ~36 hours" |
| R1-561 | F3 | 43 | 200 | permutations | method-parameter | asserts-own | "100 permutations = ~18 hours, 200 = ~36 hours" |
| R1-562 | F3 | 43 | ~36 | hours (200 perms, 35K) | hardware-dependent | asserts-own | "100 permutations = ~18 hours, 200 = ~36 hours" |
| R1-563 | F3 | 43 | 5 | perms (1K null) | method-parameter | quotes-paper | "The 1K-feature null model (5 perms, ~86s/perm) should also be expanded to 100+ perms." |
| R1-564 | F3 | 43 | ~86 | s per permutation (1K) | hardware-dependent | asserts-own | "The 1K-feature null model (5 perms, ~86s/perm) should also be expanded to 100+ perms." |
| R1-565 | F3 | 43 | 100+ | perms (requested, 1K) | method-parameter | requests | "should also be expanded to 100+ perms" |
| R1-566 | F3 | 45 | 7/10 | severity (M2) | external-fact | asserts-own | "M2. GO true-path inflation: partially addressed by the K-drop but still unquantified [Severity: 7/10]" |
| R1-567 | F3 | 47 | 22 | K_max (1K features) | deterministic | quotes-paper | "The K_max drop from 22 (1K features) to 17 (35K features) is striking." |
| R1-568 | F3 | 47 | 17 | K_max (35K features) | deterministic | asserts-own | "The K_max drop from 22 (1K features) to 17 (35K features) is striking." |
| R1-569 | F3 | 51 | 30% | redundant GO pairs (hypothetical, 1K) | deterministic | asserts-own | "If the 1K vocabulary has 30% redundant pairs and the 35K has 5%, this explains the K-drop" |
| R1-570 | F3 | 51 | 5% | redundant GO pairs (hypothetical, 35K) | deterministic | asserts-own | "If the 1K vocabulary has 30% redundant pairs and the 35K has 5%" |
| R1-571 | F3 | 53 | 22 | K (itemset with GO inflation) | deterministic | quotes-paper | "The K=22 itemset from the 1K-feature analysis contains verified GO hierarchy inflation" |
| R1-572 | F3 | 53 | 19-20 | corrected independent K | deterministic | disputes | "The corrected independent K is approximately 19-20." |
| R1-573 | F3 | 55 | 17 | K (35K itemset needing scrutiny) | deterministic | asserts-own | "The 35K K=17 itemset needs the same scrutiny." |
| R1-574 | F3 | 55 | 17 | co-occurring features (requested listing) | deterministic | requests | "What are the 17 co-occurring features in the K=17 pattern?" |
| R1-575 | F3 | 57 | 5/10 | severity (M3, current) | external-fact | asserts-own | "M3. SON comparison ... [Severity: 5/10, improved from 8/10]" |
| R1-576 | F3 | 57 | 8/10 | severity (M3, v1) | external-fact | asserts-own | "[Severity: 5/10, improved from 8/10]" |
| R1-577 | F3 | 61 | 40M | chunk size (SON, transactions) | method-parameter | asserts-own | "'40M chunk x 35K features = 175 GB bitvec exceeds 143 GB H200 VRAM'" |
| R1-578 | F3 | 61 | 35K | features (bitvec arithmetic) | deterministic | asserts-own | "'40M chunk x 35K features = 175 GB bitvec exceeds 143 GB H200 VRAM'" |
| R1-579 | F3 | 61 | 175 | GB (SON bitvec) | deterministic | asserts-own | "'40M chunk x 35K features = 175 GB bitvec exceeds 143 GB H200 VRAM'" |
| R1-580 | F3 | 61 | 143 | GB (H200 VRAM) | hardware-dependent | asserts-own | "'40M chunk x 35K features = 175 GB bitvec exceeds 143 GB H200 VRAM' (son_status: FAILED_OOM_HUNG)" |
| R1-581 | F3 | 63 | 143 | GB (H200 VRAM) | hardware-dependent | asserts-own | "the highest-end GPU available (NVIDIA H200 at 143 GB)" |
| R1-582 | F3 | 63 | <33M | chunk_size required to fit | deterministic | asserts-own | "you would need chunk_size < 33M, which at 109M transactions means 4+ chunks" |
| R1-583 | F3 | 63 | 109M | transactions | deterministic | asserts-own | "which at 109M transactions means 4+ chunks with proportionally worse miss rates" |
| R1-584 | F3 | 63 | 4+ | chunks | deterministic | asserts-own | "at 109M transactions means 4+ chunks with proportionally worse miss rates" |
| R1-585 | F3 | 67 | 20M | chunk_size (feasible SON) | method-parameter | asserts-own | "chunk_size=20M x 35K features = ~87 GB, fitting on an H200. This would create 6 chunks" |
| R1-586 | F3 | 67 | ~87 | GB (bitvec at 20M chunk) | deterministic | asserts-own | "chunk_size=20M x 35K features = ~87 GB, fitting on an H200" |
| R1-587 | F3 | 67 | 6 | chunks | deterministic | asserts-own | "This would create 6 chunks with terrible miss rates" |
| R1-588 | F3 | 69 | 95.2% | SON miss rate (1K) | deterministic | quotes-paper | "The SON infeasibility at 35K is a STRONGER argument than the 1K 95.2% miss rate." |
| R1-589 | F3 | 73 | 3 (Three) | Direct GPU runs | method-parameter | asserts-own | "Three Direct GPU runs with mean=606,292, std=28, CV=0.0046% demonstrates excellent reproducibility" |
| R1-590 | F3 | 73 | 606,292 | mean itemsets | deterministic | asserts-own | "Three Direct GPU runs with mean=606,292, std=28, CV=0.0046%" |
| R1-591 | F3 | 73 | 28 | std itemsets | deterministic | asserts-own | "Three Direct GPU runs with mean=606,292, std=28, CV=0.0046%" |
| R1-592 | F3 | 73 | 0.0046% | CV of itemset count | deterministic | asserts-own | "Three Direct GPU runs with mean=606,292, std=28, CV=0.0046%" |
| R1-593 | F3 | 73 | 17, 16, 17 | K_max across runs | deterministic | asserts-own | "The slight variation in K_max (17, 16, 17 across runs) at K=17 (with 2, 0, 1 itemsets respectively)" |
| R1-594 | F3 | 73 | 2, 0, 1 | itemsets at K=17 across runs | deterministic | asserts-own | "at K=17 (with 2, 0, 1 itemsets respectively) suggests the boundary itemsets are at the noise floor" |
| R1-595 | F3 | 75 | 8/10 | severity (M4) | external-fact | asserts-own | "M4. The K-max drop (K=22 -> K=17) demands explanation [Severity: 8/10]" |
| R1-596 | F3 | 75 | 22 -> 17 | K-max drop | deterministic | asserts-own | "M4. The K-max drop (K=22 -> K=17) demands explanation" |
| R1-597 | F3 | 77 | 1,002 | features (1K) | deterministic | quotes-paper | "When expanding from 1,002 to 34,920 features:" |
| R1-598 | F3 | 77 | 34,920 | features (35K) | deterministic | asserts-own | "When expanding from 1,002 to 34,920 features:" |
| R1-599 | F3 | 78 | 22 | K_max (1K) | deterministic | quotes-paper | "K_max dropped from 22 to 17" |
| R1-600 | F3 | 78 | 17 | K_max (35K) | deterministic | asserts-own | "K_max dropped from 22 to 17" |
| R1-601 | F3 | 79 | 9 | K (peak, 1K) | deterministic | quotes-paper | "The K-distribution peak shifted from K=9 (1K) to K=7 (35K)" |
| R1-602 | F3 | 79 | 7 | K (peak, 35K) | deterministic | asserts-own | "The K-distribution peak shifted from K=9 (1K) to K=7 (35K)" |
| R1-603 | F3 | 80 | 26.8M | itemsets (1K, min_count=8) | deterministic | quotes-paper | "Total itemsets dropped from 26.8M (at min_count=8) to 606K (at min_count=1,093)" |
| R1-604 | F3 | 80 | 8 | min_count (1K Opus) | method-parameter | quotes-paper | "Total itemsets dropped from 26.8M (at min_count=8) to 606K (at min_count=1,093)" |
| R1-605 | F3 | 80 | 606K | itemsets (35K, min_count=1,093) | deterministic | asserts-own | "Total itemsets dropped from 26.8M (at min_count=8) to 606K (at min_count=1,093)" |
| R1-606 | F3 | 80 | 1,093 | min_count (35K) | method-parameter | asserts-own | "to 606K (at min_count=1,093)" |
| R1-607 | F3 | 82 | 1,093 vs 8 | min_count (support threshold difference) | method-parameter | asserts-own | "The third point is explained by the higher support threshold (1,093 vs 8)" |
| R1-608 | F3 | 84 | 1,093 | min_count (35K experiment) | method-parameter | asserts-own | "The 35K experiment uses min_count=1,093 while the 1K Opus run uses min_count=8." |
| R1-609 | F3 | 84 | 8 | min_count (1K Opus run) | method-parameter | quotes-paper | "The 35K experiment uses min_count=1,093 while the 1K Opus run uses min_count=8." |
| R1-610 | F3 | 84 | 22 | K (pattern supported by 8 proteins) | deterministic | quotes-paper | "The K=22 pattern in the 1K run was supported by exactly 8 proteins -- it would be invisible at min_count=1,093" |
| R1-611 | F3 | 84 | 8 | proteins supporting K=22 | deterministic | quotes-paper | "The K=22 pattern in the 1K run was supported by exactly 8 proteins" |
| R1-612 | F3 | 86 | 109M | transactions (assumed same) | deterministic | asserts-own | "each feature is present in fewer proteins on average (assuming the same 109M transactions)" |
| R1-613 | F3 | 88 | millions | proteins carrying 'cytoplasm' term | deterministic | asserts-own | "removes a term that appeared in millions of proteins, reducing deep co-occurrence potential" |
| R1-614 | F3 | 90 | 4 | GPUs (row-splitting) | hardware-dependent | asserts-own | "The 35K runs use 4-GPU row-splitting with local_min_count=274 (line 2 of power_test_v1_full.log)" |
| R1-615 | F3 | 90 | 274 | local_min_count (power test) | method-parameter | asserts-own | "4-GPU row-splitting with local_min_count=274 (line 2 of power_test_v1_full.log)" |
| R1-616 | F3 | 90 | 14,800 | locally frequent K=2 pairs (union across GPUs) | deterministic | asserts-own | "'K=2: 14,800 locally frequent (union across GPUs), 8,554 frequent'" |
| R1-617 | F3 | 90 | 8,554 | globally frequent K=2 pairs | deterministic | asserts-own | "'K=2: 14,800 locally frequent (union across GPUs), 8,554 frequent'" |
| R1-618 | F3 | 90 | 42% | locally-frequent K=2 pairs failing global recount | deterministic | asserts-own | "shows that 42% of locally-frequent K=2 pairs fail the global recount" |
| R1-619 | F3 | 92 | 8 | min_count (controlled experiment requested) | method-parameter | requests | "A controlled experiment would run the 35K features at min_count=8 on a single GPU (if VRAM allows)" |
| R1-620 | F3 | 92 | 1 (single) | GPU (controlled experiment) | hardware-dependent | requests | "run the 35K features at min_count=8 on a single GPU (if VRAM allows)" |
| R1-621 | F3 | 94 | 7/10 | severity (M5) | external-fact | asserts-own | "M5. 8 vs 3 K=22 proteins -- still unresolved [Severity: 7/10]" |
| R1-622 | F3 | 94 | 8 vs 3 | K=22 proteins (support count vs identified) | deterministic | disputes | "M5. 8 vs 3 K=22 proteins -- still unresolved" |
| R1-623 | F3 | 96 | 8 | proteins (mining result support count) | deterministic | quotes-paper | "The discrepancy between 8 proteins (mining result support count) and 3 proteins identified (analysis script fallback heuristic)" |
| R1-624 | F3 | 96 | 3 | proteins identified (analysis script fallback heuristic) | deterministic | disputes | "and 3 proteins identified (analysis script fallback heuristic) remains unresolved" |
| R1-625 | F3 | 96 | 8 | UniProt accessions (must be named) | deterministic | requests | "All 8 UniProt accessions must be named and verified." |
| R1-626 | F3 | 98 | 6/10 | severity (M6) | external-fact | asserts-own | "M6. Wave 3 partial results hint at massive scale but are incomplete [Severity: 6/10]" |
| R1-627 | F3 | 100 | 0.1% | support (wave 3) | method-parameter | asserts-own | "mining at 0.1% support (min_count=109,225) reaching K=10 with the following per-level counts" |
| R1-628 | F3 | 100 | 109,225 | min_count (wave 3, 0.1%) | method-parameter | asserts-own | "mining at 0.1% support (min_count=109,225) reaching K=10" |
| R1-629 | F3 | 100 | 10 | K reached (wave 3) | deterministic | asserts-own | "mining at 0.1% support (min_count=109,225) reaching K=10" |
| R1-630 | F3 | 104 | 1,164 | itemsets at K=1 (wave 3) | deterministic | asserts-own | table row "1 / 1,164 / 0.3s" |
| R1-631 | F3 | 104 | 0.3 | s cumulative at K=1 (wave 3) | hardware-dependent | asserts-own | table row "1 / 1,164 / 0.3s" |
| R1-632 | F3 | 105 | 8,554 | itemsets at K=2 (wave 3) | deterministic | asserts-own | table row "2 / 8,554 / 1.2s" |
| R1-633 | F3 | 105 | 1.2 | s cumulative at K=2 (wave 3) | hardware-dependent | asserts-own | table row "2 / 8,554 / 1.2s" |
| R1-634 | F3 | 106 | 28,804 | itemsets at K=3 (wave 3) | deterministic | asserts-own | table row "3 / 28,804 / 2.0s" |
| R1-635 | F3 | 106 | 2.0 | s cumulative at K=3 (wave 3) | hardware-dependent | asserts-own | table row "3 / 28,804 / 2.0s" |
| R1-636 | F3 | 107 | 88,561 | itemsets at K=4 (wave 3) | deterministic | asserts-own | table row "4 / 88,561 / 5.3s" |
| R1-637 | F3 | 107 | 5.3 | s cumulative at K=4 (wave 3) | hardware-dependent | asserts-own | table row "4 / 88,561 / 5.3s" |
| R1-638 | F3 | 108 | 280,784 | itemsets at K=5 (wave 3) | deterministic | asserts-own | table row "5 / 280,784 / 17.5s" |
| R1-639 | F3 | 108 | 17.5 | s cumulative at K=5 (wave 3) | hardware-dependent | asserts-own | table row "5 / 280,784 / 17.5s" |
| R1-640 | F3 | 109 | 835,466 | itemsets at K=6 (wave 3) | deterministic | asserts-own | table row "6 / 835,466 / 52.1s" |
| R1-641 | F3 | 109 | 52.1 | s cumulative at K=6 (wave 3) | hardware-dependent | asserts-own | table row "6 / 835,466 / 52.1s" |
| R1-642 | F3 | 110 | 2,207,022 | itemsets at K=7 (wave 3) | deterministic | asserts-own | table row "7 / 2,207,022 / 145.3s" |
| R1-643 | F3 | 110 | 145.3 | s cumulative at K=7 (wave 3) | hardware-dependent | asserts-own | table row "7 / 2,207,022 / 145.3s" |
| R1-644 | F3 | 111 | 5,063,845 | itemsets at K=8 (wave 3) | deterministic | asserts-own | table row "8 / 5,063,845 / 348.5s" |
| R1-645 | F3 | 111 | 348.5 | s cumulative at K=8 (wave 3) | hardware-dependent | asserts-own | table row "8 / 5,063,845 / 348.5s" |
| R1-646 | F3 | 112 | 10,041,611 | itemsets at K=9 (wave 3) | deterministic | asserts-own | table row "9 / 10,041,611 / 743.4s" |
| R1-647 | F3 | 112 | 743.4 | s cumulative at K=9 (wave 3) | hardware-dependent | asserts-own | table row "9 / 10,041,611 / 743.4s" |
| R1-648 | F3 | 113 | 12.5M | locally frequent candidates at K=10 (wave 3) | deterministic | asserts-own | table row "10 / ??? (12.5M locally frequent) / >743s, still running" |
| R1-649 | F3 | 113 | >743 | s at K=10 (still running) | hardware-dependent | asserts-own | table row "10 / ??? (12.5M locally frequent) / >743s, still running" |
| R1-650 | F3 | 115 | 30x | more itemsets per K-level (0.1% vs 0.001% run) | deterministic | asserts-own | "This is 30x more itemsets per K-level than the 0.001% run (e.g., K=7: 2.2M vs 64K)" |
| R1-651 | F3 | 115 | 0.001% | support (comparison run) | method-parameter | asserts-own | "30x more itemsets per K-level than the 0.001% run" |
| R1-652 | F3 | 115 | 2.2M | itemsets at K=7 (0.1% run) | deterministic | asserts-own | "(e.g., K=7: 2.2M vs 64K)" |
| R1-653 | F3 | 115 | 64K | itemsets at K=7 (0.001% run) | deterministic | asserts-own | "(e.g., K=7: 2.2M vs 64K)" |
| R1-654 | F3 | 115 | 10 | K (distribution still growing) | deterministic | asserts-own | "the distribution is still growing at K=10" |
| R1-655 | F3 | 117 | hundreds of millions | itemsets (projected if wave 3 completes) | deterministic | asserts-own | "If this run completes, it could produce hundreds of millions of itemsets at 35K features." |
| R1-656 | F3 | 117 | 12+ | minutes at K=9 | hardware-dependent | asserts-own | "the runtime (already 12+ minutes at K=9) suggests the full campaign may take hours per threshold" |
| R1-657 | F3 | 119 | 1,093 | min_count (power test) | method-parameter | asserts-own | "show mining at min_count=1,093 with K=2 taking 586s alone (vs 1.2s at min_count=109,225)" |
| R1-658 | F3 | 119 | 586 | s (K=2 level at min_count=1,093) | hardware-dependent | asserts-own | "K=2 taking 586s alone (vs 1.2s at min_count=109,225)" |
| R1-659 | F3 | 119 | 1.2 | s (K=2 level at min_count=109,225) | hardware-dependent | asserts-own | "K=2 taking 586s alone (vs 1.2s at min_count=109,225)" |
| R1-660 | F3 | 119 | 109,225 | min_count | method-parameter | asserts-own | "(vs 1.2s at min_count=109,225)" |
| R1-661 | F3 | 119 | 902K | frequent pairs (min_count=1,093) | deterministic | asserts-own | "the lower threshold admits 902K frequent pairs vs 8.5K, creating a vastly larger candidate space at K>=3" |
| R1-662 | F3 | 119 | 8.5K | frequent pairs (min_count=109,225) | deterministic | asserts-own | "admits 902K frequent pairs vs 8.5K" |
| R1-663 | F3 | 119 | >=3 | K (candidate space) | deterministic | asserts-own | "creating a vastly larger candidate space at K>=3" |
| R1-664 | F3 | 119 | 4 | K reached (power test) | deterministic | asserts-own | "The power test reached K=4 with 17.4M itemsets before apparently stalling." |
| R1-665 | F3 | 119 | 17.4M | itemsets at K=4 (power test) | deterministic | asserts-own | "The power test reached K=4 with 17.4M itemsets before apparently stalling." |
| R1-666 | F3 | 121 | 606K | itemsets at 0.001% support | deterministic | asserts-own | "Presenting 606K itemsets at 0.001% support when the 0.1% run alone produces 10M+ itemsets at K=9 undersells" |
| R1-667 | F3 | 121 | 0.001% | support | method-parameter | asserts-own | "Presenting 606K itemsets at 0.001% support" |
| R1-668 | F3 | 121 | 0.1% | support | method-parameter | asserts-own | "when the 0.1% run alone produces 10M+ itemsets at K=9" |
| R1-669 | F3 | 121 | 10M+ | itemsets at K=9 (0.1% run) | deterministic | asserts-own | "the 0.1% run alone produces 10M+ itemsets at K=9" |
| R1-670 | F3 | 121 | 9 | K | deterministic | asserts-own | "the 0.1% run alone produces 10M+ itemsets at K=9" |
| R1-671 | F3 | 129 | 2 | permutations (35K) | method-parameter | asserts-own | "The 35K null model uses 2 permutations instead of the 1K model's 5." |
| R1-672 | F3 | 129 | 5 | permutations (1K) | method-parameter | quotes-paper | "The 35K null model uses 2 permutations instead of the 1K model's 5." |
| R1-673 | F3 | 133 | 22 -> 17 | K drop | deterministic | asserts-own | "The K=22 -> K=17 drop suggests less GO inflation at 35K features, but this is inferential" |
| R1-674 | F3 | 139 | 26.8M | itemsets (provenance gap) | deterministic | quotes-paper | "M4-v1. 26.8M itemsets data provenance gap [RETAINED, unchanged]" |
| R1-675 | F3 | 141 | 7.3 | minute claim (mining vs total pipeline) | hardware-dependent | disputes | "The 7.3-minute claim (mining time vs total pipeline time) is still ambiguous." |
| R1-676 | F3 | 143 | 8 vs 3 | K=22 proteins (heading, retained from v1) | deterministic | disputes | "M5-v1. 8 vs 3 K=22 proteins [RETAINED as M5 above]" |
| R1-677 | F3 | 143 | 22 | K (heading) | deterministic | quotes-paper | "M5-v1. 8 vs 3 K=22 proteins [RETAINED as M5 above]" |
| R1-678 | F3 | 151 | 3/10 | severity (m1) | external-fact | asserts-own | "m1. Triplicate reproducibility reveals boundary instability [Severity: 3/10]" |
| R1-679 | F3 | 153 | 606,319 | itemsets (run 1) | deterministic | asserts-own | "minor variation in total itemsets (606,319 / 606,293 / 606,263) and K_max (17 / 16 / 17)" |
| R1-680 | F3 | 153 | 606,293 | itemsets (run 2) | deterministic | asserts-own | "minor variation in total itemsets (606,319 / 606,293 / 606,263)" |
| R1-681 | F3 | 153 | 606,263 | itemsets (run 3) | deterministic | asserts-own | "minor variation in total itemsets (606,319 / 606,293 / 606,263)" |
| R1-682 | F3 | 153 | 17 / 16 / 17 | K_max per run | deterministic | asserts-own | "and K_max (17 / 16 / 17)" |
| R1-683 | F3 | 153 | 2, 0, 1 | itemsets at K=17 per run | deterministic | asserts-own | "The K=17 level had 2, 0, and 1 itemsets across runs." |
| R1-684 | F3 | 153 | 152 | itemsets at K=16 (stable across runs) | deterministic | asserts-own | "The K=16 level is stable at 152 across all runs." |
| R1-685 | F3 | 155 | 16-17 | K_max (recommended reporting) | deterministic | requests | "Report K_max as '16-17' with a note that K=17 patterns are at the support boundary." |
| R1-686 | F3 | 157 | 5/10 | severity (m2) | external-fact | asserts-own | "m2. The 35K feature vocabulary composition is undocumented [Severity: 5/10]" |
| R1-687 | F3 | 159 | 247 | Pfam items (Table 1) | deterministic | quotes-paper | "The paper documents the 1K feature vocabulary in Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-688 | F3 | 159 | 302 | GO:MF items (Table 1) | deterministic | quotes-paper | "Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-689 | F3 | 159 | 289 | GO:BP items (Table 1) | deterministic | quotes-paper | "Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-690 | F3 | 159 | 161 | GO:CC items (Table 1) | deterministic | quotes-paper | "Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-691 | F3 | 159 | 3 | pLDDT items (Table 1) | deterministic | quotes-paper | "Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT)" |
| R1-692 | F3 | 159 | 34,920 | features (35K vocabulary) | deterministic | asserts-own | "What are the 34,920 features?" |
| R1-693 | F3 | 159 | 34,920 | frequent items at K=1 (35K logs) | deterministic | asserts-own | "The logs show K=1: 34,920 frequent items, suggesting all 34,920 are above the min_count threshold." |
| R1-694 | F3 | 162 | 1,093-2,000 | proteins per feature (hypothetical) | deterministic | asserts-own | "If most new features appear in only 1,093-2,000 proteins, they contribute to K=2 pairs but not deep patterns." |
| R1-695 | F3 | 163 | 1,002 | original features | deterministic | quotes-paper | "Does the 35K vocabulary include all 1,002 original features?" |
| R1-696 | F3 | 165 | 5/10 | severity (m3) | external-fact | asserts-own | "m3. Multi-GPU row-splitting introduces a new approximation [Severity: 5/10]" |
| R1-697 | F3 | 167 | 4 | GPUs (row-splitting) | hardware-dependent | asserts-own | "The 35K results use 4-GPU row-splitting (wave3_base_partial.log line 18: local_min_count=27,307 for global min_count=109,225)" |
| R1-698 | F3 | 167 | 27,307 | local_min_count (wave 3) | method-parameter | asserts-own | "local_min_count=27,307 for global min_count=109,225" |
| R1-699 | F3 | 167 | 109,225 | global min_count (wave 3) | method-parameter | asserts-own | "local_min_count=27,307 for global min_count=109,225" |
| R1-700 | F3 | 167 | 14,800 | locally frequent K=2 (union) | deterministic | asserts-own | "(e.g., line 25: '14,800 locally frequent (union across GPUs), 8,554 frequent')" |
| R1-701 | F3 | 167 | 8,554 | globally frequent K=2 | deterministic | asserts-own | "'14,800 locally frequent (union across GPUs), 8,554 frequent'" |
| R1-702 | F3 | 167 | 42% | candidates pruned by global recount | deterministic | asserts-own | "requiring a global recount that prunes 42% of candidates" |
| R1-703 | F3 | 169 | 1/4 | fraction of transactions per GPU | method-parameter | asserts-own | "each GPU sees 1/4 of the transactions and applies a reduced local threshold" |
| R1-704 | F3 | 169 | 95.2% | SON miss rate (1K analysis) | deterministic | quotes-paper | "The same cascade effect that caused SON to miss 95.2% of patterns in the 1K analysis could be active here" |
| R1-705 | F3 | 169 | global/4 | local_min_count formula | method-parameter | asserts-own | "(local_min_count = global/4 = exact proportional split)" |
| R1-706 | F3 | 173 | 4/10 | severity (m4) | external-fact | asserts-own | "m4. Feature vocabulary expansion justification missing [Severity: 4/10]" |
| R1-707 | F3 | 175 | 1,002 | features | deterministic | quotes-paper | "The jump from 1,002 to 34,920 is not a round number and is not motivated in any documentation." |
| R1-708 | F3 | 175 | 34,920 | features | deterministic | asserts-own | "The jump from 1,002 to 34,920 is not a round number" |
| R1-709 | F3 | 177 | ~60% | of all proteins annotated GO:0005515 protein binding | external-fact | asserts-own | "GO:0005515 protein binding which annotates ~60% of all proteins and creates millions of trivial co-occurrences" |
| R1-710 | F3 | 177 | millions | trivial co-occurrences | deterministic | asserts-own | "annotates ~60% of all proteins and creates millions of trivial co-occurrences" |
| R1-711 | F3 | 179 | <2x min_count | median feature frequency (hypothetical) | deterministic | asserts-own | "If median frequency < 2x min_count, most features are barely above the noise floor." |
| R1-712 | F3 | 181 | 21.4x | speedup (1K SON comparison) | hardware-dependent | quotes-paper | "m5. The '21.4x speedup' from v1 is now secondary [Severity: 2/10, reduced from 4/10]" |
| R1-713 | F3 | 181 | 2/10 | severity (m5, current) | external-fact | asserts-own | "[Severity: 2/10, reduced from 4/10]" |
| R1-714 | F3 | 181 | 4/10 | severity (m5, v1) | external-fact | asserts-own | "[Severity: 2/10, reduced from 4/10]" |
| R1-715 | F3 | 183 | 21.4x | speedup | hardware-dependent | quotes-paper | "The 1K SON comparison (21.4x speedup, 95.2% miss rate) is now secondary to the 35K result" |
| R1-716 | F3 | 183 | 95.2% | SON miss rate | deterministic | quotes-paper | "The 1K SON comparison (21.4x speedup, 95.2% miss rate)" |
| R1-717 | F3 | 185 | 3/10 | severity (m6) | external-fact | asserts-own | "m6. The 'bell curve' terminology [RETAINED from v1, severity 3/10]" |
| R1-718 | F3 | 187 | 7 | K (35K distribution peak) | deterministic | asserts-own | "The 35K K-distribution peaks at K=7 (64,400 itemsets) and has a different shape than the 1K distribution (peak at K=9)" |
| R1-719 | F3 | 187 | 64,400 | itemsets at K=7 (35K) | deterministic | asserts-own | "The 35K K-distribution peaks at K=7 (64,400 itemsets)" |
| R1-720 | F3 | 187 | 9 | K (1K distribution peak) | deterministic | quotes-paper | "a different shape than the 1K distribution (peak at K=9)" |
| R1-721 | F3 | 189 | 3/10 | severity (m7) | external-fact | asserts-own | "m7. Intermediate biological discoveries lack quantitative validation [RETAINED from v1, severity 3/10]" |
| R1-722 | F3 | 191 | ~611 | proteins (approximate count) | deterministic | quotes-paper | "The approximate protein counts ('~611 proteins,' '~11,000 proteins') should be exact." |
| R1-723 | F3 | 191 | ~11,000 | proteins (approximate count) | deterministic | quotes-paper | "The approximate protein counts ('~611 proteins,' '~11,000 proteins') should be exact." |
| R1-724 | F3 | 193 | 4/10 | severity (m8) | external-fact | asserts-own | "m8. No closed/maximal itemset analysis [RETAINED from v1, severity 4/10]" |
| R1-725 | F3 | 195 | 606K | itemsets (35K, likely redundant) | deterministic | asserts-own | "The 606K itemsets at 35K features likely include massive redundancy." |
| R1-726 | F3 | 197 | 3/10 | severity (m9) | external-fact | asserts-own | "m9. Title claims 'AlphaFold Scale' but mines annotations [RETAINED from v1, severity 3/10]" |
| R1-727 | F3 | 201 | 2/10 | severity (m10) | external-fact | asserts-own | "m10. No comparison with cuML/RAPIDS [RETAINED from v1, severity 2/10]" |
| R1-728 | F3 | 205 | 3/10 | severity (m11, current) | external-fact | asserts-own | "m11. Multi-GPU claims are now substantiated but incompletely [Severity: 3/10, improved from 5/10]" |
| R1-729 | F3 | 205 | 5/10 | severity (m11, v1) | external-fact | asserts-own | "[Severity: 3/10, improved from 5/10]" |
| R1-730 | F3 | 207 | 4xH200 | GPUs (35K results) | hardware-dependent | asserts-own | "The 35K results use 4xH200 GPUs, providing the first multi-GPU evidence." |
| R1-731 | F3 | 207 | 8.9-12.9 | s (bitvector build time across runs) | hardware-dependent | asserts-own | "The bitvector build time (8.9-12.9s across runs) and per-level times are reported, but speedup vs 1 GPU is not." |
| R1-732 | F3 | 207 | 1 | GPU (baseline missing) | hardware-dependent | requests | "but speedup vs 1 GPU is not" |
| R1-733 | F3 | 209 | 4/10 | severity (m12) | external-fact | asserts-own | "m12. Runtime comparison across scales is inconsistent [Severity: 4/10]" |
| R1-734 | F3 | 213 | 1,002 | features (1K Opus row) | deterministic | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-735 | F3 | 213 | 76.9M | transactions (1K Opus row) | deterministic | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-736 | F3 | 213 | 0.00001% | threshold (1K Opus row) | method-parameter | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-737 | F3 | 213 | 8 | min_count (1K Opus row) | method-parameter | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-738 | F3 | 213 | 1xH100 | GPUs (1K Opus row) | hardware-dependent | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-739 | F3 | 213 | 7.3 | min (1K Opus row) | hardware-dependent | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-740 | F3 | 213 | 26.8M | itemsets (1K Opus row) | deterministic | quotes-paper | table row "1K Opus / 1,002 / 76.9M / 0.00001% (8) / 1xH100 / 7.3 min / 26.8M" |
| R1-741 | F3 | 214 | 34,920 | features (35K Base row) | deterministic | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-742 | F3 | 214 | 109.2M | transactions (35K Base row) | deterministic | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-743 | F3 | 214 | 0.001% | threshold (35K Base row) | method-parameter | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-744 | F3 | 214 | 1,093 | min_count (35K Base row) | method-parameter | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-745 | F3 | 214 | 4xH200 | GPUs (35K Base row) | hardware-dependent | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-746 | F3 | 214 | 10.9 | min (35K Base row) | hardware-dependent | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-747 | F3 | 214 | 606K | itemsets (35K Base row) | deterministic | asserts-own | table row "35K Base / 34,920 / 109.2M / 0.001% (1,093) / 4xH200 / 10.9 min / 606K" |
| R1-748 | F3 | 216 | 4x | more GPUs (35K vs 1K) | hardware-dependent | asserts-own | "The 35K run uses 4x more GPUs, 100x higher threshold, and still takes 50% longer." |
| R1-749 | F3 | 216 | 100x | higher threshold (35K vs 1K) | method-parameter | asserts-own | "The 35K run uses 4x more GPUs, 100x higher threshold, and still takes 50% longer." |
| R1-750 | F3 | 216 | 50% | longer runtime (35K vs 1K) | hardware-dependent | asserts-own | "and still takes 50% longer" |
| R1-751 | F3 | 216 | 35x | more features / larger bitvectors | deterministic | asserts-own | "This is expected (35x more features = 35x larger bitvectors = memory-bandwidth bound)" |
| R1-752 | F3 | 224 | ~119 | GB bitvector matrix per GPU (35K) | deterministic | asserts-own | "The 35K bitvector matrix is ~119 GB per GPU (line 3 of power_test logs), requiring 4-GPU row-splitting." |
| R1-753 | F3 | 224 | 4 | GPUs (row-splitting required) | hardware-dependent | asserts-own | "requiring 4-GPU row-splitting" |
| R1-754 | F3 | 228 | 9 | K (peak, 1K) | deterministic | quotes-paper | "The peak shifting from K=9 (1K) to K=7 (35K) with a characteristic unimodal shape in both cases" |
| R1-755 | F3 | 228 | 7 | K (peak, 35K) | deterministic | asserts-own | "The peak shifting from K=9 (1K) to K=7 (35K)" |
| R1-756 | F3 | 236 | 2 (two) | vocabulary scales demonstrated | external-fact | asserts-own | "demonstrated at two vocabulary scales (1K and 35K) and two hardware configurations (1xH100 and 4xH200)" |
| R1-757 | F3 | 236 | 2 (two) | hardware configurations (1xH100, 4xH200) | hardware-dependent | asserts-own | "two hardware configurations (1xH100 and 4xH200)" |
| R1-758 | F3 | 236 | 0.1% | support (wave 3) | method-parameter | asserts-own | "mining at 0.1% support can discover tens of millions of itemsets at 35K features" |
| R1-759 | F3 | 236 | tens of millions | itemsets (wave 3 projection) | deterministic | asserts-own | "mining at 0.1% support can discover tens of millions of itemsets at 35K features, though this run is incomplete" |
| R1-760 | F3 | 240 | 175 | GB (SON bitvec) | deterministic | asserts-own | "The 175 GB bitvec calculation (40M chunk x 35K features) is straightforward and verifiable." |
| R1-761 | F3 | 240 | 40M | chunk (SON) | method-parameter | asserts-own | "The 175 GB bitvec calculation (40M chunk x 35K features)" |
| R1-762 | F3 | 244 | 0.0046% | CV (triplicate) | deterministic | asserts-own | "The 35K direct GPU runs demonstrate excellent reproducibility (CV=0.0046%)." |
| R1-763 | F3 | 248 | 5 | perms (1K null) | method-parameter | quotes-paper | "Both the 1K (5 perms) and 35K (2 perms) null models show the same qualitative pattern" |
| R1-764 | F3 | 248 | 2 | perms (35K null) | method-parameter | asserts-own | "Both the 1K (5 perms) and 35K (2 perms) null models show the same qualitative pattern" |
| R1-765 | F3 | 248 | 2 | K (depleted in both nulls) | deterministic | asserts-own | "depleted K=2, enriched K>=3 (35K) or K>=4 (1K), null collapse well before the biological K_max" |
| R1-766 | F3 | 248 | >=3 | K (enriched, 35K) | deterministic | asserts-own | "enriched K>=3 (35K) or K>=4 (1K)" |
| R1-767 | F3 | 248 | >=4 | K (enriched, 1K) | deterministic | quotes-paper | "enriched K>=3 (35K) or K>=4 (1K)" |
| R1-768 | F3 | 258 | 100+ | null-model permutations (both scales) | method-parameter | requests | "R1. Run 100+ null model permutations at BOTH scales [CRITICAL]" |
| R1-769 | F3 | 262 | 100 | perms (35K) | method-parameter | requests | "35K features: 100 perms x ~655s / 4 GPUs = ~4.5 hours. 200 perms = ~9 hours." |
| R1-770 | F3 | 262 | ~655 | s per perm (35K) | hardware-dependent | asserts-own | "35K features: 100 perms x ~655s / 4 GPUs = ~4.5 hours." |
| R1-771 | F3 | 262 | 4 | GPUs | hardware-dependent | asserts-own | "100 perms x ~655s / 4 GPUs = ~4.5 hours" |
| R1-772 | F3 | 262 | ~4.5 | hours (100 perms, 35K) | hardware-dependent | asserts-own | "100 perms x ~655s / 4 GPUs = ~4.5 hours" |
| R1-773 | F3 | 262 | 200 | perms (35K) | method-parameter | requests | "200 perms = ~9 hours" |
| R1-774 | F3 | 262 | ~9 | hours (200 perms, 35K) | hardware-dependent | asserts-own | "200 perms = ~9 hours" |
| R1-775 | F3 | 263 | 100 | perms (1K) | method-parameter | requests | "1K features: 100 perms x ~86s / 1 GPU = ~2.4 hours." |
| R1-776 | F3 | 263 | ~86 | s per perm (1K) | hardware-dependent | asserts-own | "1K features: 100 perms x ~86s / 1 GPU = ~2.4 hours." |
| R1-777 | F3 | 263 | 1 | GPU (1K null) | hardware-dependent | asserts-own | "100 perms x ~86s / 1 GPU = ~2.4 hours" |
| R1-778 | F3 | 263 | ~2.4 | hours (100 perms, 1K) | hardware-dependent | asserts-own | "100 perms x ~86s / 1 GPU = ~2.4 hours" |
| R1-779 | F3 | 267 | ~$50-100 | USD (cloud H200 cost for permutations) | hardware-dependent | asserts-own | "Estimated cost: ~$50-100 on cloud H200 instances. Trivial for a Nature Methods submission." |
| R1-780 | F3 | 271 | 8 | min_count (controlled 35K run) | method-parameter | requests | "Run 35K features at min_count=8 on a single GPU (if ~119 GB fits on H200) or the smallest feasible threshold." |
| R1-781 | F3 | 271 | 1 (single) | GPU | hardware-dependent | requests | "Run 35K features at min_count=8 on a single GPU" |
| R1-782 | F3 | 271 | ~119 | GB (bitvector to fit on H200) | deterministic | asserts-own | "(if ~119 GB fits on H200)" |
| R1-783 | F3 | 271 | ~17 | K_max (if unchanged at low threshold) | deterministic | asserts-own | "If K_max stays at ~17 even at low threshold, the drop is due to feature dilution or GO hierarchy effects" |
| R1-784 | F3 | 271 | 22 | K_max (if it rises toward) | deterministic | asserts-own | "If K_max rises toward 22, the drop is purely a threshold effect" |
| R1-785 | F3 | 279 | 0.1% | support (wave 3) | method-parameter | asserts-own | "The wave 3 partial shows the 35K feature set at 0.1% support producing millions of itemsets per K-level." |
| R1-786 | F3 | 279 | millions | itemsets per K-level | deterministic | asserts-own | "producing millions of itemsets per K-level" |
| R1-787 | F3 | 279 | 0.001% | support (power test) | method-parameter | asserts-own | "lower thresholds (0.001%) produce 900K+ frequent pairs and 17M+ K=4 itemsets" |
| R1-788 | F3 | 279 | 900K+ | frequent pairs (0.001%) | deterministic | asserts-own | "lower thresholds (0.001%) produce 900K+ frequent pairs and 17M+ K=4 itemsets" |
| R1-789 | F3 | 279 | 17M+ | K=4 itemsets (0.001%) | deterministic | asserts-own | "produce 900K+ frequent pairs and 17M+ K=4 itemsets" |
| R1-790 | F3 | 279 | 4xH200 | GPUs (feasibility) | hardware-dependent | asserts-own | "suggesting the full campaign at low support may be computationally infeasible on 4xH200" |
| R1-791 | F3 | 281 | 8 | K=22 proteins to name (heading) | deterministic | requests | "R5. Name the 8 K=22 proteins [HIGH PRIORITY, from v1]" |
| R1-792 | F3 | 281 | 22 | K (heading) | deterministic | quotes-paper | "R5. Name the 8 K=22 proteins [HIGH PRIORITY, from v1]" |
| R1-793 | F3 | 283 | 8 | K=22 proteins (to name) | deterministic | requests | "R5. Name the 8 K=22 proteins [HIGH PRIORITY, from v1] ... Resolve the 8-vs-3 discrepancy." |
| R1-794 | F3 | 283 | 22 | K | deterministic | quotes-paper | "R5. Name the 8 K=22 proteins" |
| R1-795 | F3 | 283 | 8-vs-3 | K=22 protein count discrepancy | deterministic | disputes | "Cross-reference with UniProt. Resolve the 8-vs-3 discrepancy." |
| R1-796 | F3 | 287 | 606K | itemsets (raw, 35K) | deterministic | asserts-own | "The 606K number may collapse to 50K-100K closed itemsets." |
| R1-797 | F3 | 287 | 50K-100K | closed itemsets (projected) | deterministic | asserts-own | "The 606K number may collapse to 50K-100K closed itemsets." |
| R1-798 | F3 | 291 | 42% | K=2 candidates pruned by locally-frequent union | deterministic | asserts-own | "The locally-frequent union step prunes 42% of K=2 candidates -- what is the analogous pruning at K>=5?" |
| R1-799 | F3 | 291 | >=5 | K (pruning analysis requested) | deterministic | requests | "what is the analogous pruning at K>=5?" |
| R1-800 | F3 | 299 | 109M | transactions (35K) | deterministic | asserts-own | "The 109M transactions (35K features) vs 76.9M (1K features) difference suggests the 35K vocabulary includes more organisms." |
| R1-801 | F3 | 299 | 76.9M | transactions (1K) | deterministic | quotes-paper | "The 109M transactions (35K features) vs 76.9M (1K features) difference" |
| R1-802 | F3 | 309 | 1,002 | original features | deterministic | quotes-paper | "Are the 1,002 original features a subset of the 34,920?" |
| R1-803 | F3 | 309 | 34,920 | features (35K) | deterministic | asserts-own | "Are the 1,002 original features a subset of the 34,920?" |
| R1-804 | F3 | 313 | 2 (Two) | data points (vocabulary scales) | external-fact | asserts-own | "Two data points (1K: peak=9, K_max=22; 35K: peak=7, K_max=17) suggest an inverse relationship" |
| R1-805 | F3 | 313 | 9 | K peak (1K) | deterministic | quotes-paper | "(1K: peak=9, K_max=22; 35K: peak=7, K_max=17)" |
| R1-806 | F3 | 313 | 22 | K_max (1K) | deterministic | quotes-paper | "(1K: peak=9, K_max=22; 35K: peak=7, K_max=17)" |
| R1-807 | F3 | 313 | 7 | K peak (35K) | deterministic | asserts-own | "(1K: peak=9, K_max=22; 35K: peak=7, K_max=17)" |
| R1-808 | F3 | 313 | 17 | K_max (35K) | deterministic | asserts-own | "(1K: peak=9, K_max=22; 35K: peak=7, K_max=17)" |
| R1-809 | F3 | 317 | 6 | null K_max (1K, 5 perms) | deterministic | quotes-paper | "The null K_max is 6 (1K, 5 perms) vs 5 (35K, 2 perms)." |
| R1-810 | F3 | 317 | 5 | perms (1K) | method-parameter | quotes-paper | "The null K_max is 6 (1K, 5 perms) vs 5 (35K, 2 perms)." |
| R1-811 | F3 | 317 | 5 | null K_max (35K, 2 perms) | deterministic | asserts-own | "The null K_max is 6 (1K, 5 perms) vs 5 (35K, 2 perms)." |
| R1-812 | F3 | 317 | 2 | perms (35K) | method-parameter | asserts-own | "The null K_max is 6 (1K, 5 perms) vs 5 (35K, 2 perms)." |
| R1-813 | F3 | 317 | ~3.5x | biological/null K_max ratio (1K: 6->22) | deterministic | asserts-own | "If the null-to-biological K_max ratio is approximately constant (~3.5x for 1K: 6->22; ~3.4x for 35K: 5->17)" |
| R1-814 | F3 | 317 | ~3.4x | biological/null K_max ratio (35K: 5->17) | deterministic | asserts-own | "(~3.5x for 1K: 6->22; ~3.4x for 35K: 5->17)" |
| R1-815 | F3 | 321 | 76.9M | transactions (1K run) | deterministic | quotes-paper | "The 1K run processes 76.9M transactions x 1,002 features in 7.3 min on 1xH100." |
| R1-816 | F3 | 321 | 1,002 | features (1K run) | deterministic | quotes-paper | "The 1K run processes 76.9M transactions x 1,002 features in 7.3 min on 1xH100." |
| R1-817 | F3 | 321 | 7.3 | min (1K run) | hardware-dependent | quotes-paper | "The 1K run processes 76.9M transactions x 1,002 features in 7.3 min on 1xH100." |
| R1-818 | F3 | 321 | 1xH100 | GPUs (1K run) | hardware-dependent | quotes-paper | "in 7.3 min on 1xH100" |
| R1-819 | F3 | 321 | 109M | transactions (35K run) | deterministic | asserts-own | "The 35K run processes 109M transactions x 34,920 features in 10.9 min on 4xH200." |
| R1-820 | F3 | 321 | 34,920 | features (35K run) | deterministic | asserts-own | "The 35K run processes 109M transactions x 34,920 features in 10.9 min on 4xH200." |
| R1-821 | F3 | 321 | 10.9 | min (35K run) | hardware-dependent | asserts-own | "The 35K run processes 109M transactions x 34,920 features in 10.9 min on 4xH200." |
| R1-822 | F3 | 321 | 4xH200 | GPUs (35K run) | hardware-dependent | asserts-own | "in 10.9 min on 4xH200" |
| R1-823 | F3 | 321 | ~50x | more feature-transaction-products per GPU-second (35K vs 1K) | hardware-dependent | asserts-own | "Normalized: the 35K run achieves ~50x more feature-transaction-products per GPU-second" |
| R1-824 | F3 | 327 | 2 | permutations (35K null, 'new concern') | method-parameter | asserts-own | "while creating new ones (K-max drop explanation, 2-perm null model, incomplete campaigns, multi-GPU approximation error)" |
| R1-825 | F3 | 329 | 5 | perms (1K null, 'weak') | method-parameter | quotes-paper | "the null model has regressed from weak (5 perms) to indefensible (2 perms)" |
| R1-826 | F3 | 329 | 2 | perms (35K null, 'indefensible') | method-parameter | asserts-own | "the null model has regressed from weak (5 perms) to indefensible (2 perms)" |
| R1-827 | F3 | 333 | 100+ | permutations (condition for minor revision) | method-parameter | requests | "If the authors run 100+ permutations and explain the K-max drop, the paper moves from 'major revision' to 'minor revision.'" |
| R1-828 | F3 | 341 | 9/10 | severity (M1, summary table) | external-fact | asserts-own | severity table "M1: Null model permutation count (now n=2) / 9/10 / ESCALATED" |
| R1-829 | F3 | 341 | n=2 | permutations (35K null) | method-parameter | asserts-own | severity table "M1: Null model permutation count (now n=2)" |
| R1-830 | F3 | 342 | 7/10 | severity (M2, summary table) | external-fact | asserts-own | severity table "M2: GO hierarchy inflation unquantified / 7/10" |
| R1-831 | F3 | 343 | 5/10 | severity (M3, summary table) | external-fact | asserts-own | severity table "M3: SON comparison framing / 5/10" |
| R1-832 | F3 | 344 | 8/10 | severity (M4, summary table) | external-fact | asserts-own | severity table "M4: K-max drop unexplained (NEW) / 8/10" |
| R1-833 | F3 | 345 | 7/10 | severity (M5, summary table) | external-fact | asserts-own | severity table "M5: 8 vs 3 K=22 proteins / 7/10 / Unchanged" |
| R1-834 | F3 | 345 | 8 vs 3 | K=22 proteins (summary table) | deterministic | disputes | severity table "M5: 8 vs 3 K=22 proteins / 7/10 / Unchanged" |
| R1-835 | F3 | 346 | 6/10 | severity (M6, summary table) | external-fact | asserts-own | severity table "M6: Wave 3 incomplete (NEW) / 6/10" |

---

## SECTION B — REVIEW-FLAGGED INCONSISTENCIES

Every place a reviewer says two numbers disagree, a value is implausible, a method detail is missing or contradicts another section, or a value changed between paper versions. Wording in column 4 paraphrases the reviewer; column 5 lists the values/locations the reviewer puts in tension.

| ID | file | line(s) | what the reviewer claims is inconsistent | the two (or more) values/locations involved |
|---|---|---|---|---|
| B-01 | F1 | 19, 30-34, 121-123 | Vocabulary construction misdescribed: paper says features retained if in ≥8 proteins; log shows a top-500-per-type frequency cap; "≥8" is the mining min_count, a different quantity | paper line 129 "at least 8 proteins" vs `pipeline_214m.log:2063-2064` "24291 unique Pfam, 25993 unique GO" → "6 pLDDT + 500 Pfam + 500 GO = 1006"; `godmode_mining.log:2` min_count 8 |
| B-02 | F1 | 20, 36-40, 125-132 | "40× reduction" pairs the CSR of the 76.9M mined subset against the dense matrix of the 205.6M full set; same-dataset ratio is ~15×; body also inconsistent with the paper's own appendix ratios | 5.1 GB CSR (76,890,945 txns, `direct_mining.log`) vs 206 GB dense (205,620,298 × 1002 × 1 B, `beyond_mining.log`); 77 GB / 5.06 GB = ~15×; appendix `tab:memory-comparison` 1.4× bit-packed / 7.8× at 0.01% density vs body 40× (paper lines 167, 402) |
| B-03 | F1 | 21, 42-47, 134-136 | Paper says the 8 K=22 proteins "can be recovered by querying the source transaction data" but also says the transaction matrix is not deposited; script needs an absent parquet; no accession list saved | paper line 333 (recoverable via `analyze_k22_proteins.py`) vs paper line 530 (matrix "not deposited"); `analyze_k22_proteins.py:162-190` requires `transactions_214m_base.parquet` (absent); `item_mapping` absent; no `accessions_deepest_itemset_*.tsv` |
| B-04 | F1 | 22, 49-53, 138-141 | p<0.45 is attributed to the "rule of three", but the rule of three gives 3/5 = 0.60; 0.45 is actually the exact one-sided binomial 95% bound 1−0.05^(1/5) | 0.45 (paper lines 367, 382, 391) vs 0.60 (rule of three) vs 0.4507 (exact binomial); 0/5 null runs reach K≥7 (`experiment_null_model_20260219_061046.json:153,163`) |
| B-05 | F1 | 15 | Value changed between paper versions: v1 Table 1 vocabulary composition was "fabricated" and has been replaced by the verified composition | v1 Table 1 247/752/3 vs verified 500/500/6 = 1,006 |
| B-06 | F1 | 24, 97, 143-144 | Paper's "verification run at min_count=4 discovered 48 million itemsets with identical maximum" has no artifact; the only "48M" in logs is a transaction count during extraction (conflation risk) | min_count=4 / 48M itemsets (paper line 335) vs "Written 48M transactions" (`pipeline_214m.log:2113`); null-model JSON min_count 769 |
| B-07 | F1 | 95 | "K=22 shared by exactly 8 proteins" cannot be verified: the named cross-check file caps at K=19 and is a different run; no artifact contains any K≥20 itemset | K=22 / 8 proteins (paper lines 254, 286) vs `decoded_top_k_patterns.txt` max K=19 (187 proteins) |
| B-08 | F1 | 94, 152 | UniProt release string "2025_01" is not recorded in any artifact; log shows only the input filename and a Feb 2026 run date | "2025_01" (paper lines 129, 478, 530) vs `pipeline_214m.log:3,10` (`uniprot_trembl.dat.gz`, Feb 2026) |
| B-09 | F1 | 148-150 | Data source named inconsistently within the paper: TrEMBL in Methods vs Swiss-Prot in Discussion | paper line 129 "UniProt TrEMBL" (matches log input `uniprot_trembl.dat.gz`) vs paper line 478 "UniProt/Swiss-Prot release 2025_01" |
| B-10 | F1 | 96 | 22→21 independent features (0 GO parent-child pairs; InterPro2GO link removed) is computed at runtime and cannot be reproduced read-only (missing cached `go.obo`, no saved output) | paper line 333 (22→21, 0 pairs) vs `analyze_k22_proteins.py:277-420` (runtime pronto + go.obo, no artifact) |
| B-11 | F1 | 99-100 | Unverifiable from local artifacts: competitor rows (Borgelt/Fang/GMiner/BIGMiner) and the external accuracy of the 37 references — literature-sourced only | paper lines 420-423 (GMiner 15M basis for 5.1×); paper lines 541-724 (37 refs) |
| B-12 | F1 | 72 | Log filenames do not match the paper's run names (mapping had to be verified) | `pipeline/ultra/extreme/direct/beyond/godmode_mining.log` vs paper run names Base/Super/Power/Blitz/Ultra/Opus |
| B-13 | F1 | 108 | Commit subject contradicts its content: "validation produced identical results" vs a 231-line manuscript rewrite that deletes expanded-run figures | commit `b318df5` subject vs its diff (231 lines) |
| B-14 | F1 | 109 | Stray expanded-run artifacts exist although the plan says no artifact of the expanded run was preserved | `results_35k/*.json` present vs `plans/leg-het-vast-in-crispy-glade.md` "geen bewaard artefact" |
| B-15 | F1 | 110 | Expanded/"v2" run reported with two mutually incompatible figures across documents; current .tex contains neither | 606K itemsets @ 4×H200 (`revision_notes_b3.tex`) vs 16.8B itemsets @ 8×H200 (plan) |
| B-16 | F1 | 111, 154-156 | Headline Z>3,700 in abstract/conclusion lacks the n=5 caveat that the body applies (t-statistics, 4 df) | Z>3,700 (paper lines 91, 498) vs table caption / §4.4 t-statistics (4 df), 5 permutations |
| B-17 | F1 | 112 | Dutch translation is stale and diverges from the English paper | `et_miner_proteome_nl.tex` (old title, Feb 2026 date) vs current English .tex |
| B-18 | F1 | 113, 146 | Approximate supporting-protein counts are given where exact values exist in an artifact | ~11,000 / ~10,500 / ~16,000 (paper lines 353, 355, 357; K=13/12/11 patterns) vs exact integers in `decoded_top_k_patterns.txt` |
| B-19 | F1 | 113, 158 | 7.3 min is presented standalone without stating it is mining-only, while extraction alone took 63 min | 7.3 min (paper lines 120, 496) vs 63-minute feature extraction (paper line 129; af_extract 3777.0 s) |
| B-20 | F2 | 22, 31-35, 85 | Null model run only at the Power threshold, but the headline K=9 peak and K=22 ceiling come from the Opus threshold where no null model was run; generalisation across thresholds unjustified | 0.001% / min_count 769 (null model, §3.6, Table 4) vs 0.00001% / min_count 8 (Opus, K=22, K=9 peak) |
| B-21 | F2 | 23, 38-41, 96 | n=5 permutations cannot carry the inferential weight; p<0.45 does not reject the null at any conventional level although the sentence implies significance | n=5; p<0.45 (0/5 null runs with K≥7); "Z" magnitudes relabelled t (4 df) |
| B-22 | F2 | 42 | The K=6 effect size does not reproduce cleanly from the rounded μ/σ shown in the table (σ from n=5 unstable) | +71,728 (K=6) vs rounded μ/σ in Table 4 |
| B-23 | F2 | 43, 86 | Manuscript itself says 100+ permutations are needed for p<0.01, at ~130 s each (≈3.6 GPU-hours) — yet only 5 were run | 100+ permutations / p<0.01 / ~130 s per permutation vs n=5 actually run |
| B-24 | F2 | 24, 47, 69, 129 | The 8 K=22 accessions are not listed; the paper says they are recoverable from source transaction data, but that data (transaction parquet) is not deposited / was lost with the compute environment | 8 accessions (Table 5, §3.5) vs `transactions_214m.parquet` not deposited (Stage 4) |
| B-25 | F2 | 48, 87 | Missing method detail: GO evidence codes (EXP/IDA vs IEA) not distinguished for the K=22 signature, decisive with n=8 proteins | n=8 proteins; per-term evidence codes absent |
| B-26 | F2 | 53 | Scale comparison is apples-to-oranges: every competitor ran on a different dataset and hardware generation, so "5.1× more transactions" is scale reached, not a controlled comparison; no same-data baseline | Table 3 (5.1×, GMiner 15M) vs Direct-vs-SON (21×, 95.2%) the only controlled comparison |
| B-27 | F2 | 56, 88, 119 | "26.8 million patterns" is the full redundant itemset space; closed/maximal counts never reported although filtering is implemented | 26.8M total vs (unreported) closed / maximal counts; 2^K − 2 sub-itemsets per K-itemset |
| B-28 | F2 | 63 | Multiple different memory quantities coexist across sections and risk reader confusion (reviewer says "three" but lists four) | ~206 GB (naive dense), ~26 GB (bit-packed, 214M set), ~5.1 GB (CSR-COO), ~10 GB (bit-packed, 76.9M subset) — Sections 1, 2.3, Appendix D |
| B-29 | F2 | 64 | "Structural property" wording oversells: only 2 of 6 pLDDT bins pass and the recurring one is a single medium-confidence bin | 2 of 6 pLDDT bins; "medium confidence (70–90)" vs Table 5 "structural property" |
| B-30 | F2 | 65 | Approximate intermediate-K protein counts where artefacts support near-exact values | ~11,000 / ~10,500 / ~16,000 vs exact supports in artefacts |
| B-31 | F2 | 66 | Multi-GPU scaling listed as a headline design choice but exercised by no reported result (all experiments single H100) | "automatic multi-GPU scaling" vs "all experiments used a single H100" |
| B-32 | F2 | 67, 120, 173 | Bibliography inconsistency: `miettinen2020` in bibliography but uncited (fixed post-review; zero uncited remain) | 1 uncited bibitem → 0 |
| B-33 | F2 | 78 | Value changed between versions: GMiner row labelling "15M synthetic / 1.7M real" is "now correct" (implying earlier mislabelling) | GMiner 15M synthetic vs 1.7M real (Table 3) |
| B-34 | F2 | 115, 158 | Missing method detail: no multiple-comparison correction applied to millions of itemsets (acknowledged as Limitation 1 only) | millions of itemsets tested; no FDR |
| B-35 | F2 | 120 | Value changed between versions: 13 citation errors corrected in this revision | 13 citation errors (previous version) vs re-verified references (this version) |
| B-36 | F2 | 138, 159, 172 | Repository inconsistent with manuscript: two figure PDFs from the removed expanded run remain on disk (removed post-review) | `figures/kdist_shift.pdf`, `figures/pruning_savings.pdf` vs no `\includegraphics` in .tex |
| B-37 | F2 | 143, 160, 171 | Relationship to the published Zenodo v1 not disclosed (versioned-preprint update); previously fabricated Table 1 corrected in this revision | Zenodo v1 vs current manuscript; "previously fabricated Table 1" |
| B-38 | F2 | 129, 131 | Missing data-availability statement; raw source data not in repository | `transactions_214m.parquet` not deposited; statement ❌ missing (added post-review, line 171) |
| B-39 | F3 | 27-29, 127-129, 329, 341 | Null model regressed: 35K-feature null uses 2 permutations vs 5 in the 1K-feature null | 5 permutations (1K) vs 2 permutations (35K; seeds 7138484576005690180, 4047939128787533792) |
| B-40 | F3 | 33 | Z-scores from n=2 are statistically invalid (t with 1 df, no finite moments); std values from 2 observations have CIs spanning zero to infinity | Z=722.91 at K=3 = (53235 − 15919.5)/51.6; std K=2 123.7, K=3 51.6, K=4 2.8 |
| B-41 | F3 | 35, 43, 265 | Division-by-zero artifact presented as evidence: K=5 null std=0.0 (both runs = 1 itemset) yields z_score "inf" stored as a string | K=5: 1 itemset in both runs, std 0.0, z_score "inf" (JSON line 104) |
| B-42 | F3 | 37 | Thresholds do not match: 35K null model min_count differs from the 35K mining campaign min_count, and neither matches the paper's Opus threshold | min_count 1,090 (null, ≈0.001% of 109M) vs 1,093 (mining, 0.001%) vs 8 (Opus) |
| B-43 | F3 | 39 | Enrichment sign reversal at K=3 between vocabularies is unexplained and unmentioned in the manuscript | 1K (5 perms): K=2 Z=−987, K=3 Z=−143 vs 35K (2 perms): K=2 Z=−102, K=3 Z=+723 |
| B-44 | F3 | 41 | Missing method detail: Fisher–Yates shuffle creates duplicate features within proteins which are then removed; duplication rate undocumented/uncontrolled | permutation dedup (1K and 35K null models) |
| B-45 | F3 | 47-55, 75-92, 133, 271, 313 | K_max drop and peak shift when the vocabulary grew are counter-intuitive and unexplained; four candidate causes (threshold, dilution, GO collapse, multi-GPU local threshold) not disentangled | K_max 22 → 17; peak K=9 → K=7; 26.8M (min_count 8) → 606K (min_count 1,093); 1,002 → 34,920 features |
| B-46 | F3 | 53 | K=22 itemset contains GO hierarchy inflation (cytoplasm/cytosol, cytoplasm/mitochondrion, nucleus/nuclear speckle); corrected independent K is lower than reported | K=22 vs corrected independent K ≈ 19–20 |
| B-47 | F3 | 57-67, 240 | SON "physically impossible" framing needs nuance: 40M-chunk bitvec exceeds VRAM, but a smaller chunk would fit (with more chunks / worse miss rate) | 40M × 35K = 175 GB > 143 GB H200 vs chunk 20M × 35K = ~87 GB (6 chunks); chunk_size < 33M → 4+ chunks at 109M transactions |
| B-48 | F3 | 73, 151-155 | Triplicate runs disagree at the boundary: K_max and K=17 counts vary across runs (support-boundary noise or multi-GPU non-determinism); recommend reporting K_max as 16–17 | totals 606,319 / 606,293 / 606,263; K_max 17 / 16 / 17; K=17 itemsets 2 / 0 / 1; K=16 stable at 152 |
| B-49 | F3 | 90, 165-171, 291 | Multi-GPU row-splitting reintroduces a SON-like approximation: locally-frequent union over-estimates and the global recount prunes 42% at K=2; error at higher K unquantified; no single-GPU baseline | local_min_count 274 (power test) / 27,307 for global 109,225 (wave 3); 14,800 locally frequent vs 8,554 globally frequent K=2 pairs |
| B-50 | F3 | 94-96, 143-145, 281-283, 345 | K=22 support count disagrees with number of proteins the analysis script identified; accessions never named | 8 proteins (mining support) vs 3 proteins (analysis-script fallback heuristic) |
| B-51 | F3 | 98-121, 277-279 | Wave 3 (0.1%) campaign incomplete: log cuts off at K=9 after 743 s with K=10 unknown; per-level counts ~30× the 0.001% run; feasibility of the full 35K campaign at low support unclear | K=9 10,041,611 itemsets at 743.4 s; K=10 "???" (12.5M locally frequent); K=7 2.2M (0.1%) vs 64K (0.001%) |
| B-52 | F3 | 119, 279 | Power test at min_count=1,093 stalls: K=2 alone takes 586 s vs 1.2 s at min_count=109,225 because 902K vs 8.5K frequent pairs; stalled at K=4 with 17.4M itemsets | 586 s vs 1.2 s; 902K vs 8.5K pairs; K=4 17.4M |
| B-53 | F3 | 139-141 | Naming inconsistency and ambiguous timing: 1K result files named `godmode` vs paper run name `Opus`; 7.3-minute claim ambiguous (mining vs total pipeline) | `godmode` vs `Opus`; 7.3 min mining vs total pipeline |
| B-54 | F3 | 157-163, 173-179, 307-309 | 35K vocabulary composition, selection, and overlap with the 1K vocabulary undocumented; 34,920 is not a round number and unmotivated; whether all 1,002 original features are included is unverified | Table 1 (247 Pfam, 302 GO:MF, 289 GO:BP, 161 GO:CC, 3 pLDDT) documented vs 34,920 features undocumented; K=1: 34,920 frequent items |
| B-55 | F3 | 181-183 | 1K SON comparison reported from a single run; triplicate runs and per-K miss rates missing | 21.4× speedup, 95.2% miss rate (single run) |
| B-56 | F3 | 185-187 | "Bell curve" terminology not supported by any formal fit; the two distributions differ in shape and peak | 35K peak K=7 (64,400 itemsets) vs 1K peak K=9 |
| B-57 | F3 | 189-191 | Approximate protein counts for intermediate discoveries should be exact | ~611 proteins; ~11,000 proteins |
| B-58 | F3 | 193-195, 285-287 | Closed/maximal itemset counts not reported; raw 606K likely inflated by GO redundancy | 606K raw vs projected 50K–100K closed |
| B-59 | F3 | 205-207 | Multi-GPU speedup unknown: no single-GPU baseline for 35K; only build times and per-level times reported | bitvector build 8.9–12.9 s across runs; speedup vs 1 GPU not reported |
| B-60 | F3 | 209-216, 319-321 | Runtime comparison across scales is inconsistent / un-normalised: 35K run uses 4× the GPUs and a 100× higher threshold yet takes 50% longer; needs itemsets-per-GPU-second | 1K Opus: 1,002 feat / 76.9M txn / 0.00001% (8) / 1×H100 / 7.3 min / 26.8M vs 35K Base: 34,920 / 109.2M / 0.001% (1,093) / 4×H200 / 10.9 min / 606K |
| B-61 | F3 | 297-299 | Transaction counts differ between the two vocabularies without explanation (suggests different organism coverage) | 109M transactions (35K) vs 76.9M (1K) |
| B-62 | F3 | 315-317 | Null K_max differs across scales; ratio to biological K_max noted as approximately constant but unexplored | null K_max 6 (1K, 5 perms) vs 5 (35K, 2 perms); ratios ~3.5× (6→22) vs ~3.4× (5→17) |
| B-63 | F3 | 43, 258-267 | Compute for adequate permutations is affordable yet not done: 100 perms ≈ 18 h (35K) / ≈ 2.4 h (1K); ~$50–100 cloud cost | ~655 s/perm (35K, 4×H200); ~86 s/perm (1K); 100 / 200 perms |

### B-addendum — extractor-noted cross-file observations (NOT reviewer-flagged; recorded only because they may matter for the audit; no correctness judgement)

| ID | files / lines | observation | values involved |
|---|---|---|---|
| X-01 | F1 L78, F2 L43/L86 vs F3 L43/L263 | Per-permutation time for the 1K null model is stated differently across reviews | ~130 s/permutation (F1, F2) vs ~86 s/perm (F3) |
| X-02 | F1 L80, F2 L18/L53 vs F3 L181/L183 | Direct-vs-SON speedup stated with different precision | 21× (F1, F2) vs 21.4× (F3) |
| X-03 | F3 L159 vs F1 L15 | F3 quotes the paper's Table 1 vocabulary as 247 Pfam / 302 GO:MF / 289 GO:BP / 161 GO:CC / 3 pLDDT (sums to 247/752/3); F1 calls exactly that v1 table "fabricated" and corrected to 500/500/6 = 1,006 | 247/752/3 vs 500/500/6 |
| X-04 | F1 L96, F2 L46 vs F3 L53 | Independent-feature count of the K=22 itemset after GO true-path checking differs | 21 independent (0 parent-child pairs + 1 InterPro2GO link; F1/F2) vs ≈19–20 corrected (F3, parent-child pairs asserted present) |
| X-05 | F1 L72, F2 L77 vs F1 L78, F2 L31 | Power-campaign min_count vs null-model min_count both described as the 0.001% threshold | 768 (Power campaign) vs 769 (null model) |
| X-06 | F1 L110 vs F3 L17/L214 | The 606K @ 4×H200 expanded-run figure that F1 says is contradicted by a 16.8B @ 8×H200 figure in the plan is the same 35K result F3 analyses in detail (606,292 mean, 4×H200) | 606K @ 4×H200 vs 16.8B @ 8×H200 |
| X-07 | F3 L43 vs F3 L262 | Within F3, the wall-clock estimate for 100 permutations at 35K is given two ways (second divides the already-4-GPU per-perm time by 4 again) | 100 perms = ~18 hours (L43) vs ~4.5 hours (L262); 200 perms = ~36 h vs ~9 h |
| X-08 | F2 L63 | Reviewer says "Three legitimate but different quantities coexist" but enumerates four | ~206 GB, ~26 GB, ~5.1 GB, ~10 GB |
| X-09 | F1 L70 / F2 L63 vs F3 L224 | Bit-packed / bitvector matrix sizes quoted for different datasets and vocabularies — not directly comparable across reviews | ~26 GB (1,002 items, 214M set), ~10 GB (76.9M subset), 27/19 GB appendix row vs ~119 GB per GPU (34,920 items, 4-way row split) |
| X-10 | F3 L18 vs F3 L33 | 35K null-model K=3 Z quoted rounded and unrounded in the same file | Z=723 (L18, L39) vs 722.91 (L33) |
| X-11 | F3 L37 vs F3 L214 vs F3 L100 | 35K transaction count given as 109M, 109.2M; 0.1% → min_count 109,225 implies ≈109.2M; 0.001% → 1,093 (mining) but 1,090 (null) | 109M / 109.2M / 109,225 / 1,093 / 1,090 |
| X-12 | F2 L38 vs F1 L22/L49-53 | F2 accepts "p < 0.45 (rule of three, 0/5)" as a correct fix, whereas F1 disputes the "rule of three" attribution (gives 0.60) and credits 0.45 to the exact binomial bound | 0.45 "rule of three" (F2) vs 0.4507 exact binomial / 0.60 rule of three (F1) |
| X-13 | F1 L74/L95 vs F3 L94-96 | F1 confirms K=22 = 1 itemset / 8 proteins against the godmode log but marks the 8 proteins unverifiable (no K≥20 artifact); F3 reports the analysis script found only 3 proteins | 8 (log support) vs 3 (script) vs no artifact with K≥20 |

---

## SECTION C — PER-FILE SUMMARIES

### C.1 F1 — `paper/PAPER_V2_REVIEW.md` ("Adversarial Review — papers/et_miner_proteome.tex")

- **Type / author:** automated adversarial cross-check of every quantitative claim in the .tex against the mining logs, experiment JSONs and decoded-pattern artifacts under `applications/alphafold/results_214m/`, plus "repository integrity seeds from prior analysis". No named human reviewer.
- **Date:** 2026-08-01 (line 3).
- **Paper version reviewed:** the restored base-run manuscript ("V2" per the file name); it explicitly refers to "the old fabricated v1 Table 1 (247/752/3)" having been corrected to 500/500/6 = 1,006, and states the current .tex contains no expanded-run (35K) claims. Paper line numbers cited run to ~887.
- **Scope:** 9 verification buckets, 73 claims adjudicated: 62 CONFIRMED, 4 DISCREPANT, 7 UNVERIFIABLE. Confirmed buckets: dataset/vocabulary (7), memory (6), campaign table (6/6), K-distribution (8/8), K=22 composition (2), null model (17), Direct-vs-SON (6/6), scale comparison (3), citations (7).
- **Discrepancies:** D1 vocabulary described as ≥8-support retention instead of top-500 caps; D2 40× memory reduction conflates 76.9M-subset CSR with 205.6M-set dense (~15× same-dataset); D3 "8 K=22 accessions recoverable" contradicts "matrix not deposited"; D4 p<0.45 mis-attributed to the rule of three (exact binomial 1−0.05^(1/5)=0.4507; rule of three = 0.60).
- **Unverifiable:** UniProt release 2025_01; K=22 in exactly 8 proteins (cross-check file caps at K=19/187 proteins); 22→21 independence check; min_count=4 → 48M itemsets (no such run; "48M" is a transaction count); the 8 accessions; competitor rows; external accuracy of 37 references.
- **Integrity items:** misleading commit `b318df5` (231-line rewrite); stray `results_35k/*.json` vs plan text; self-contradictory expanded run (606K @ 4×H200 vs 16.8B @ 8×H200); missing n=5 caveat on Z>3,700 in abstract/conclusion; stale Dutch translation; approximate counts ~11,000/~10,500/~16,000; 7.3 min not marked mining-only.
- **Verdict:** "The paper's headline numbers are overwhelmingly faithful to the artifacts"; provides an ordered 12-item fix list with exact OLD→NEW text (TrEMBL vs Swiss-Prot at line 478 also flagged). No accept/reject recommendation — it is an audit + fix list; explicitly instructs "Do NOT reintroduce any expanded-run figure."

### C.2 F2 — `paper/peer_review_jul12.md` ("Peer Review — Higher-Order Protein Feature Co-occurrence at AlphaFold Scale")

- **Type / author:** simulated journal-style peer review (methods / applied-computation preprint), unnamed reviewer, with a 7-stage checklist appendix and a post-review fix log. Manuscript `papers/et_miner_proteome.tex` by E. Ahmic and C. claudya ("Anthropic, Claude Code Opus 4.6").
- **Date:** 2026-07-12 (review date and Appendix B fix date).
- **Paper version reviewed:** the "restored base-run manuscript" — the revision in which numbers were re-locked to preserved artefacts after a previous version with a fabricated Table 1 and 13 citation errors; the expanded (35K) run has been removed (two orphaned figure PDFs remain). Title now "Higher-Order Protein Feature Co-occurrence…" (no motif overclaim). A Zenodo v1 exists.
- **Headline numbers as summarised by the reviewer:** 76.9M multi-feature proteins, 1,002 features, single H100, 7.3 min (mining only), 26.8M patterns, K=22, null model at 0.001% (min_count 769, n=5, seed 42, ~130 s/perm), 21× / 95.2% Direct-vs-SON, software Python 3.10 / CuPy 13.0 / NumPy 1.26 / CUDA 12.4 / Ubuntu 22.04 / H100 80GB SXM5.
- **Majors (5):** (1) null model does not cover the Opus threshold where K=22 / K=9 peak live; (2) n=5 permutations over-leveraged (p<0.45 non-significant; +71,728 from unstable σ; 100 permutations ≈ 3.6 GPU-hours feasible); (3) K=22 result is one itemset in 8 unlisted proteins with no evidence codes; (4) no same-dataset baseline (Table 3 apples-to-oranges; 5.1× is scale not speed); (5) 26.8M counts the redundant space — closed/maximal counts missing.
- **Minors (8):** abstract scope caveat; memory-figure table (206 / 26 / 5.1 / 10 GB); pLDDT "structural property" oversell (2 of 6 bins, 70–90); approximate counts; multi-GPU framing; uncited `miettinen2020`; true-path check for intermediate-K; data-availability paragraph. Checklist surfaced: no COI, no funding, no ethics statement, AI co-author policy, no FDR, orphaned figures, Zenodo-v1 relationship.
- **Verdict:** **Major revision** — "publishable as a methods contribution"; engineering sound, biological framing outruns statistics/reproducibility. Appendix B: funding, COI, ethics, data-availability statements added, orphaned figures removed, `miettinen2020` cited; AI co-authorship retained by author decision; Majors 1–5 and the FDR note still open.

### C.3 F3 — `paper/review_b1_hostile.md` ("Hostile Peer Review: Protein Structural Motif Discovery at AlphaFold Scale")

- **Type / author:** "Agent B1 (Senior Paper Reviewer, Nature Methods simulation)" — deliberately hostile review.
- **Date:** 2026-02-21 (review v2, "updated with 35K-feature results"); review v1 dated 2026-02-20 covered only the 1K-feature results. Result files referenced live in `applications/alphafold/results_35k/` and `results_214m/`.
- **Paper version reviewed:** the early (Feb 2026) manuscript still titled "Protein Structural Motif Discovery at AlphaFold Scale", containing the 1K (1,002-feature) results (Table 1 quoted as 247 Pfam / 302 GO:MF / 289 GO:BP / 161 GO:CC / 3 pLDDT; 26.8M itemsets; K=22; 7.3 min on 1×H100; 21.4× / 95.2% SON comparison; 5-perm null), plus three **new, not-yet-in-manuscript** 35K-feature result files: Direct-GPU triplicate (606,292 ± 28 itemsets, K_max 17, 654 s on 4×H200, SON OOM at 175 GB > 143 GB), a 2-permutation null model (min_count 1,090; Z=723 at K=3, 20,585 at K=4, "inf" at K=5), and an incomplete wave-3 0.1% run (10,041,611 itemsets at K=9 after 743 s).
- **Majors (6, severity /10):** M1 null model regressed to n=2 (9); M2 GO true-path inflation unquantified, K=22 corrected to ≈19–20 (7); M3 SON framing — infeasibility stronger than strawman but needs chunk-size nuance (5, from 8); M4 K_max drop 22→17 and peak 9→7 unexplained (8, new); M5 8 vs 3 K=22 proteins unresolved (7); M6 wave 3 incomplete (6, new).
- **Minors (12):** triplicate boundary instability (K_max 17/16/17); undocumented 35K vocabulary; multi-GPU row-splitting approximation (42% of K=2 candidates pruned at recount); unmotivated 34,920; 21.4× now secondary; "bell curve" wording; approximate counts (~611, ~11,000); no closed/maximal counts; title vs annotations; no cuML/RAPIDS comparison; multi-GPU speedup unknown; un-normalised cross-scale runtime comparison (7.3 min 1×H100 vs 10.9 min 4×H200).
- **Strengths (8):** CSR→bitvector pipeline scales to ~119 GB/GPU; multi-scale K-distribution; biology recovery; genuine scale; SON infeasibility as a novel finding; triplicate reproducibility (CV 0.0046%); consistent null enrichment pattern across scales (conditional); honest limitations.
- **Recommended experiments (9):** 100+ permutations at both scales (~18 h or ~4.5 h at 35K, ~2.4 h at 1K; ~$50–100), controlled K_max experiment at min_count=8, GO-hierarchy-aware K, complete the 35K campaign, name the 8 proteins, closed/maximal counts, quantify multi-GPU error, per-K SON miss rate, taxonomic stratification.
- **Verdict:** **Major revision required.** "The paper is stronger with the 35K results, but the null model has regressed from weak (5 perms) to indefensible (2 perms)"; would move to minor revision if 100+ permutations are run and the K-max drop is explained.

---

## SECTION D — TOTALS BY CATEGORY AND STANCE

### D.1 Rows per file

| file | rows |
|---|---|
| F1 `PAPER_V2_REVIEW.md` | 327 |
| F2 `peer_review_jul12.md` | 163 |
| F3 `review_b1_hostile.md` | 345 |
| **all** | **835** |

### D.2 Totals by category (all files, then per file)

| category | all | F1 | F2 | F3 |
|---|---|---|---|---|
| deterministic | 474 | 210 | 76 | 188 |
| hardware-dependent | 91 | 22 | 13 | 56 |
| method-parameter | 175 | 59 | 46 | 70 |
| external-fact | 85 | 36 | 18 | 31 |
| software | 10 | 0 | 10 | 0 |
| **total** | **835** | 327 | 163 | 345 |

### D.3 Totals by stance (all files, then per file)

| stance | all | F1 | F2 | F3 |
|---|---|---|---|---|
| quotes-paper | 338 | 171 | 108 | 59 |
| asserts-own | 372 | 89 | 26 | 257 |
| disputes | 46 | 25 | 11 | 10 |
| requests | 79 | 42 | 18 | 19 |
| **total** | **835** | 327 | 163 | 345 |

### D.4 Category × stance cross-tabulation (all files)

| category \ stance | quotes-paper | asserts-own | disputes | requests | total |
|---|---|---|---|---|---|
| deterministic | 217 | 185 | 31 | 41 | 474 |
| hardware-dependent | 33 | 50 | 3 | 5 | 91 |
| method-parameter | 70 | 65 | 10 | 30 | 175 |
| external-fact | 11 | 69 | 2 | 3 | 85 |
| software | 7 | 3 | 0 | 0 | 10 |
| **total** | 338 | 372 | 46 | 79 | **835** |

Section B contains **63** reviewer-flagged inconsistency rows plus **13** extractor-noted cross-file observations (clearly separated; not reviewer-flagged).



---

# SOURCE: paper reviews part 2 (review_b2_results.md, revision_notes_b3.tex, senior_review_jun01.md, senior_review_mar23.md)

(verbatim copy of `runs/20260902T0000Z/phase1/claims_reviews_part2.md`)

# Claims extraction — review documents, part 2 (B2 results review, B3 revision notes, senior reviews Mar 23 / Jun 01)

Run: `runs/20260902T0000Z/phase1` · Extractor pass over four files, read in full (325 + 659 + 160 + 94 lines).

File labels used in the tables:

| label | file | lines |
|---|---|---|
| F1 | `/root/projects/ET-Miner/paper/review_b2_results.md` | 325 |
| F2 | `/root/projects/ET-Miner/paper/revision_notes_b3.tex` | 659 |
| F3 | `/root/projects/ET-Miner/paper/senior_review_jun01.md` | 160 |
| F4 | `/root/projects/ET-Miner/paper/senior_review_mar23.md` | 94 |

Conventions:

- IDs run R2-001… in file order F1→F4, then first-occurrence line order. The `line` column lists the first occurrence first, then every other line in the same file where the identical value with the identical meaning and stance recurs (exact repeats are consolidated into one row; a different rounding, meaning or stance gets its own row).
- Category: `deterministic` = should reproduce exactly from the same inputs/parameters (counts, Z-scores, fractions, memory footprints derived from data dimensions, arithmetic); `hardware-dependent` = timings, throughput, speedups, costs, GPU model/VRAM/count, host RAM; `method-parameter` = thresholds, min_count, permutation counts, bin definitions, vocabulary cut-offs; `external-fact` = literature/database facts (citations, other systems' scale, InterPro2GO mappings, UniProt sizes, prices); `software` = versions, commits, integer widths.
- Stance: `quotes-paper` (the text repeats a value it attributes to the paper, or to another document it is checking), `asserts-own` (the author's own recomputation, measurement, new result, or prediction), `disputes` (the text says a paper value is wrong, unsupported or implausible), `requests` (asks for a value to be added/changed/run).
- Nothing here is judged for correctness; every value may be wrong in the source.
- Not tabulated (locators, not claims): document dates (given in Section C), paper/bibliography/code line-number references (`line 396`, `r660`, `gpu_dispatch.py:128-131`, `apriori.py:235-268`), table/figure numbers, timestamps embedded in filenames, subagent IDs, LaTeX layout numbers (`\hskip 6pt`), review-process meta counts (3 axes, 3 subagents, 4 earlier review docs), and the purely hypothetical example patterns in F2 §7 (`EC:2.7.11.1`, `LEN:800--1000aa`, `pLDDT:<70` inside example braces — the bin edges themselves are tabulated from §2). `TBD` cells in F2 are listed in Section C, not as rows.

---

## SECTION A — CLAIMS TABLE

### F1 — `review_b2_results.md`

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R2-001 | F1 | 7 | 26.8M | itemsets in godmode parquet | deterministic | asserts-own | `archived/alphafold/results_214m/itemsets_214m_godmode.parquet` (26.8M itemsets) |
| R2-002 | F1 | 18, 42, 191, 195, 196, 304, 308 | 8 | min_count, godmode/main run (paper) | method-parameter | quotes-paper | Godmode (main result): min_count=8 … The godmode run used min_count=8 as claimed. |
| R2-003 | F1 | 22, 30, 38, 57, 61, 88, 130, 288, 317 | 76,890,945 | n_transactions (log / JSON) | deterministic | asserts-own | n_transactions \| 76,890,945 \| godmode log line 2 |
| R2-004 | F1 | 23 | 8 | min_count (godmode log) | method-parameter | asserts-own | min_count \| 8 \| godmode log line 2 |
| R2-005 | F1 | 24, 87 | 1.04e-07 (~0.00001%) | min_support (8 / 76,890,945) | method-parameter | asserts-own | min_support \| 1.04e-07 (~0.00001%) \| 8 / 76,890,945 |
| R2-006 | F1 | 26, 31, 42 | 768 | min_count, direct-vs-SON experiment (JSON) | method-parameter | asserts-own | Direct vs SON experiment: min_count=768 |
| R2-007 | F1 | 32, 40, 131 | 1e-05 (0.001%) | min_support, direct-vs-SON & null (JSON) | method-parameter | asserts-own | min_support \| 1e-05 (0.001%) \| JSON `parameters.min_support` |
| R2-008 | F1 | 34, 39, 42, 194 | 769 | min_count, null model (JSON) | method-parameter | asserts-own | Null model: min_count=769 |
| R2-009 | F1 | 44 | 96× | threshold ratio null/godmode | deterministic | asserts-own | operate at a 96x stricter threshold than the godmode run |
| R2-010 | F1 | 50, 288 | 76.9M | multi-feature proteins (paper) | deterministic | quotes-paper | Claim: 76.9M multi-feature proteins (after dedup), from 205.6M total. |
| R2-011 | F1 | 50 | 205.6M | total proteins (paper) | deterministic | quotes-paper | Claim: 76.9M multi-feature proteins (after dedup), from 205.6M total. |
| R2-012 | F1 | 56, 61, 318 | 205,620,298 | total proteins (paper) | deterministic | quotes-paper | Total proteins \| 205,620,298 \| 205,620,298 \| VERIFIED |
| R2-013 | F1 | 56 | 205,620,298 | total proteins (transactions parquet) | deterministic | asserts-own | Total proteins \| 205,620,298 \| 205,620,298 \| VERIFIED |
| R2-014 | F1 | 57, 61 | 76,890,945 | multi-feature (>1 item) proteins (paper) | deterministic | quotes-paper | Multi-feature (>1 item) \| 76,890,945 \| 76,890,945 \| VERIFIED |
| R2-015 | F1 | 58, 61 | 37.4% | multi-feature share (paper) | deterministic | quotes-paper | Percentage multi-feature \| 37.4% \| 37.39% \| VERIFIED |
| R2-016 | F1 | 58, 73 | 37.39% | multi-feature share (parquet) | deterministic | asserts-own | 37.39% have >1 feature (used in mining) |
| R2-017 | F1 | 59 | 128.7M | single-feature proteins (paper) | deterministic | quotes-paper | Single-feature \| 128.7M (62.6%) \| 128,729,353 (62.61%) \| VERIFIED |
| R2-018 | F1 | 59 | 62.6% | single-feature share (paper) | deterministic | quotes-paper | Single-feature \| 128.7M (62.6%) \| 128,729,353 (62.61%) \| VERIFIED |
| R2-019 | F1 | 59 | 128,729,353 | single-feature proteins (parquet) | deterministic | asserts-own | Single-feature \| 128.7M (62.6%) \| 128,729,353 (62.61%) \| VERIFIED |
| R2-020 | F1 | 59, 72 | 62.61% | single-feature share (parquet) | deterministic | asserts-own | 62.61% of proteins have only 1 feature (excluded from mining) |
| R2-021 | F1 | 67, 75, 289 | 20.45% | "dedup" rate (PROJECT_STATE.md, not the paper) | deterministic | quotes-paper | Claim (PROJECT_STATE.md only): "Dedup affected 20.45%" |
| R2-022 | F1 | 69 | ~20.45% | proteins with duplicate items after null shuffle | deterministic | asserts-own | after shuffling items across proteins, ~20.45% of proteins have duplicate items that must be removed |
| R2-023 | F1 | 79, 103, 189, 217, 290 | K=22 | max itemset depth, main run (paper) | deterministic | quotes-paper | K=22 Validation … the Opus K-distribution (K_max=22) |
| R2-024 | F1 | 85, 90, 111, 290, 323 | 1 | itemsets at K=22 in godmode parquet | deterministic | asserts-own | K=22 itemsets in godmode parquet \| **Exactly 1** |
| R2-025 | F1 | 86 | [1, 13, 23, 507, 509, 510, 513, 519, 529, 533, 569, 632, 651, 677, 766, 784, 795, 850, 887, 917, 965, 966] | item IDs of the K=22 itemset (22 IDs) | deterministic | asserts-own | Items in the itemset \| 22 IDs: [1, 13, 23, 507, 509, 510, 513, …] |
| R2-026 | F1 | 87 | 1.04e-07 | support of the K=22 itemset | deterministic | asserts-own | Support value \| 1.04e-07 |
| R2-027 | F1 | 88, 90, 290, 323 | 8 | proteins supporting the K=22 itemset | deterministic | asserts-own | Estimated protein count \| support * 76,890,945 = **8 proteins** |
| R2-028 | F1 | 96 | 1 (plddt_mean_med) | pLDDT features in K=22 itemset | deterministic | asserts-own | pLDDT \| 1 \| plddt_mean_med |
| R2-029 | F1 | 97 | 2 (PF00270, PF00271) | Pfam features in K=22 itemset | deterministic | asserts-own | Pfam \| 2 \| PF00270 (DEAD/DEAH N-term), PF00271 (Helicase C-term) |
| R2-030 | F1 | 98 | 8 (GO:0005524, GO:0016787, GO:0000287, GO:0003697, GO:0003724, GO:0003725, GO:0003678, GO:0000978) | GO MF features in K=22 itemset | deterministic | asserts-own | GO (MF) \| 8 \| GO:0005524, GO:0016787, GO:0000287, … |
| R2-031 | F1 | 99 | 4 (GO:0030154, GO:0045087, GO:0051607, GO:0034605) | GO BP features in K=22 itemset | deterministic | asserts-own | GO (BP) \| 4 \| GO:0030154, GO:0045087, GO:0051607, GO:0034605 |
| R2-032 | F1 | 100 | 7 (GO:0005737, GO:0005829, GO:0005634, GO:0005739, GO:0030424, GO:0030425, GO:0016607) | GO CC features in K=22 itemset | deterministic | asserts-own | GO (CC) \| 7 \| GO:0005737, GO:0005829, GO:0005634, … |
| R2-033 | F1 | 101, 103 | 22 | decoded features total (= paper Table 5) | deterministic | asserts-own | Feature decode matches Table 5 in the paper exactly. All 22 features confirmed. |
| R2-034 | F1 | 109, 156, 272, 293 | 1,002 | frequent K=1 items (parquet = log) | deterministic | asserts-own | 1 \| 1,002 \| 1,002 \| Yes |
| R2-035 | F1 | 110 | 3,529,257 | K=9 itemsets (parquet = log) | deterministic | asserts-own | 9 \| 3,529,257 \| 3,529,257 \| Yes |
| R2-036 | F1 | 112, 114, 316 | 26,849,505 | total itemsets (parquet = log) | deterministic | asserts-own | Parquet row count matches log exactly: 26,849,505 itemsets. |
| R2-037 | F1 | 124, 134 | 475,865 | Direct GPU itemsets @0.001% | deterministic | asserts-own | Direct GPU \| 475,865 \| 50.72s \| 14 |
| R2-038 | F1 | 124, 211 | 50.72 s | Direct GPU time (JSON) | hardware-dependent | asserts-own | Direct GPU \| 475,865 \| 50.72s \| 14 |
| R2-039 | F1 | 124, 198 | 14 | Direct GPU max K @0.001% | deterministic | asserts-own | Direct GPU \| 475,865 \| 50.72s \| 14 … real extends to K=14 vs null K=6 |
| R2-040 | F1 | 125, 134 | 22,846 | SON itemsets @0.001% | deterministic | asserts-own | SON \| 22,846 \| 1085.6s \| 13 |
| R2-041 | F1 | 125, 127, 210 | 1085.6 s | SON time (JSON) | hardware-dependent | asserts-own | SON \| 22,846 \| 1085.6s \| 13 |
| R2-042 | F1 | 125 | 13 | SON max K | deterministic | asserts-own | SON \| 22,846 \| 1085.6s \| 13 |
| R2-043 | F1 | 127, 136, 291, 319 | 21.4× | speedup 1085.6 / 50.72 | hardware-dependent | asserts-own | **Speedup:** 1085.6 / 50.72 = **21.4x** |
| R2-044 | F1 | 127 | 21× | speedup as rounded in paper caption | hardware-dependent | quotes-paper | paper rounds to "21x" in caption text, reports "21.4x" in controlled comparison |
| R2-045 | F1 | 132 | H100 | GPU model for both direct and SON runs | hardware-dependent | asserts-own | Same GPU: yes (both run on H100) -- **SAME** |
| R2-046 | F1 | 134 | 453,019 | itemsets missed by SON | deterministic | asserts-own | **SON itemset loss:** 475,865 - 22,846 = 453,019 (95.2% lost) |
| R2-047 | F1 | 134 | 95.2% | SON miss rate | deterministic | asserts-own | 453,019 (95.2% lost) -- **VERIFIED**, matches JSON `comparison.itemset_diff` |
| R2-048 | F1 | 146–150, 194 | 6 | null max K, each of 5 runs (JSON) | deterministic | asserts-own | 1 \| 6 \| 20 … 5 \| 6 \| 22 |
| R2-049 | F1 | 146 | 20 | null run 1, K=6 itemset count | deterministic | asserts-own | 1 \| 6 \| 20 |
| R2-050 | F1 | 147 | 22 | null run 2, K=6 count | deterministic | asserts-own | 2 \| 6 \| 22 |
| R2-051 | F1 | 148 | 23 | null run 3, K=6 count | deterministic | asserts-own | 3 \| 6 \| 23 |
| R2-052 | F1 | 149 | 22 | null run 4, K=6 count | deterministic | asserts-own | 4 \| 6 \| 22 |
| R2-053 | F1 | 150 | 22 | null run 5, K=6 count | deterministic | asserts-own | 5 \| 6 \| 22 |
| R2-054 | F1 | 152, 156, 160, 173, 177, 179, 181, 183, 209, 292, 294 | 5 | permutations in the null model | method-parameter | quotes-paper | Max K = 6 across all 5 permutations. Paper claim matches exactly. |
| R2-055 | F1 | 152, 292 | 6 | null max K (paper claim) | deterministic | quotes-paper | Max K = 6 across all 5 permutations. Paper claim matches exactly. |
| R2-056 | F1 | 156, 293 | 1,002 | K=1 count in every null run (marginals preserved) | deterministic | asserts-own | All 5 null runs produce K=1 = 1,002 (identical to real data). |
| R2-057 | F1 | 160 | 0 | null itemsets at K>=7, all 5 runs (JSON) | deterministic | asserts-own | All 5 runs: 0 itemsets at K>=7. |
| R2-058 | F1 | 160 | 0 at K>=7 | paper claim "no permutation produced any pattern at K>=7" | deterministic | quotes-paper | Paper claim "no permutation produced any pattern at K>=7" is correct. |
| R2-059 | F1 | 166 | 22,019 | K=2 biological itemsets (paper) | deterministic | quotes-paper | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-060 | F1 | 166 | 22,019 | K=2 biological itemsets (JSON) | deterministic | asserts-own | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-061 | F1 | 166 | 63,702 | K=2 null μ (paper) | deterministic | quotes-paper | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-062 | F1 | 166 | 63,702.4 | K=2 null μ (JSON) | deterministic | asserts-own | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-063 | F1 | 166 | −987 | K=2 Z (paper) | deterministic | quotes-paper | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-064 | F1 | 166 | −987.08 | K=2 Z (JSON) | deterministic | asserts-own | 2 \| 22,019 \| 22,019 \| 63,702 \| 63,702.4 \| -987 \| -987.08 |
| R2-065 | F1 | 167 | 108,059 | K=4 bio (paper) | deterministic | quotes-paper | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-066 | F1 | 167 | 108,059 | K=4 bio (JSON) | deterministic | asserts-own | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-067 | F1 | 167 | 25,468 | K=4 null μ (paper) | deterministic | quotes-paper | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-068 | F1 | 167 | 25,467.8 | K=4 null μ (JSON) | deterministic | asserts-own | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-069 | F1 | 167 | +3,791 | K=4 Z (paper) | deterministic | quotes-paper | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-070 | F1 | 167 | 3790.74 | K=4 Z (JSON) | deterministic | asserts-own | 4 \| 108,059 \| 108,059 \| 25,468 \| 25,467.8 \| +3,791 \| 3790.74 |
| R2-071 | F1 | 168 | 78,596 | K=6 bio (paper) | deterministic | quotes-paper | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-072 | F1 | 168 | 78,596 | K=6 bio (JSON) | deterministic | asserts-own | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-073 | F1 | 168 | 22 | K=6 null μ (paper) | deterministic | quotes-paper | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-074 | F1 | 168 | 21.8 | K=6 null μ (JSON) | deterministic | asserts-own | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-075 | F1 | 168 | +71,728 | K=6 Z (paper) | deterministic | quotes-paper | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-076 | F1 | 168 | 71728.1 | K=6 Z (JSON) | deterministic | asserts-own | 6 \| 78,596 \| 78,596 \| 22 \| 21.8 \| +71,728 \| 71728.1 |
| R2-077 | F1 | 169, 322 | 88,745 | K=7–14 bio (paper) | deterministic | quotes-paper | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-078 | F1 | 169 | 88,745 | K=7–14 bio (JSON) | deterministic | asserts-own | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-079 | F1 | 169 | 0 | K=7–14 null μ (paper) | deterministic | quotes-paper | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-080 | F1 | 169 | 0 | K=7–14 null μ (JSON) | deterministic | asserts-own | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-081 | F1 | 169, 179 | inf | K=7–14 Z (paper) | deterministic | quotes-paper | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-082 | F1 | 169 | inf | K=7–14 Z (JSON) | deterministic | asserts-own | 7-14 \| 88,745 \| 88,745 \| 0 \| 0 \| inf \| inf |
| R2-083 | F1 | 177 | < 5% | CV of null counts across runs, K=2..6 | deterministic | asserts-own | The null model is remarkably stable across runs (CV < 5% for all K levels). |
| R2-084 | F1 | 179 | 0 events in 5 trials | null K>=7 events | deterministic | asserts-own | With 0 events in 5 trials, the Clopper-Pearson 95% upper confidence bound on P(K>=7 \| null) |
| R2-085 | F1 | 179, 294 | 0.451 (45.1%) | Clopper-Pearson 95% upper bound on P(K>=7 given null) | deterministic | asserts-own | upper confidence bound on P(K>=7 \| null) is **0.451** (45.1%) |
| R2-086 | F1 | 179 | 45% | share of null runs not excludable from producing K>=7 | deterministic | asserts-own | we CANNOT statistically rule out that up to 45% of null runs might produce K>=7 patterns |
| R2-087 | F1 | 179 | p=0 | p-value for K>=7 (paper) | deterministic | disputes | The Z=infinity and p=0 claims for K>=7 are mathematical artifacts (division by zero in standard deviation) |
| R2-088 | F1 | 181, 310 | 100 | permutations needed (minimum) | method-parameter | requests | At least 100 permutations (upper bound: 0.030) |
| R2-089 | F1 | 181 | 0.030 | upper bound on P(K>=7 given null) with 100 permutations | deterministic | asserts-own | At least 100 permutations (upper bound: 0.030) |
| R2-090 | F1 | 181 | 1000 | permutations (ideal) | method-parameter | requests | ideally 1000 permutations (upper bound: 0.003) |
| R2-091 | F1 | 181 | 0.003 | upper bound with 1000 permutations | deterministic | asserts-own | ideally 1000 permutations (upper bound: 0.003) |
| R2-092 | F1 | 183 | 20–23 | K=6 null count range over 5 runs | deterministic | asserts-own | The extreme stability across the 5 runs (K=6 count: 20-23, near-zero variance) |
| R2-093 | F1 | 189, 304 | 769 vs 8 | min_count null vs main (paper line 396 quote) | method-parameter | quotes-paper | run at a higher threshold than the main analysis (min_count=769 vs. 8) for computational efficiency |
| R2-094 | F1 | 194, 198, 200, 304, 322 | 0.001% | null-model support | method-parameter | asserts-own | Null model at 0.001% (min_count=769): max K=6 |
| R2-095 | F1 | 195, 198, 308 | 0.00001% | main-run support | method-parameter | asserts-own | Real data at 0.00001% (min_count=8): max K=22 |
| R2-096 | F1 | 195, 198 | 22 | max K of real data at min_count=8 | deterministic | asserts-own | Real data at 0.00001% (min_count=8): max K=22 |
| R2-097 | F1 | 196 | UNKNOWN (could be K=7, 8, or higher) | null max K if run at min_count=8 | deterministic | disputes | If null model were run at 0.00001% (min_count=8): max K = **UNKNOWN** (could be K=7, 8, or higher) |
| R2-098 | F1 | 198, 304, 308 | K=15–22 | depth range not covered by any null test | deterministic | disputes | It does NOT validate the additional K=15 through K=22 patterns found at the lower threshold. |
| R2-099 | F1 | 200 | K>=7 | depth from which patterns are biological at 0.001% | deterministic | asserts-own | The core finding (K>=7 is biological at 0.001%) is likely sound. |
| R2-100 | F1 | 208, 296, 320 | 7.3 min | godmode run time (paper) | hardware-dependent | quotes-paper | Godmode: 7.3 min \| Log: "440.5s" \| 440.5 / 60 = 7.34 min -- **VERIFIED** |
| R2-101 | F1 | 208, 296, 320 | 440.5 s | godmode run time (log) | hardware-dependent | asserts-own | Godmode: 7.3 min \| Log: "440.5s" |
| R2-102 | F1 | 208 | 7.34 min | 440.5 / 60 | hardware-dependent | asserts-own | 440.5 / 60 = 7.34 min -- **VERIFIED** |
| R2-103 | F1 | 209, 321 | 662 s | null model total time (paper) | hardware-dependent | quotes-paper | Null total: 662s \| JSON: 662.17s |
| R2-104 | F1 | 209 | 662.17 s | null total time (JSON, sum of 5 runs) | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-105 | F1 | 209 | 99.22 s | null run 1 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-106 | F1 | 209 | 135.62 s | null run 2 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-107 | F1 | 209 | 141.97 s | null run 3 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-108 | F1 | 209 | 143.06 s | null run 4 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-109 | F1 | 209 | 142.30 s | null run 5 time | hardware-dependent | asserts-own | Sum of 5 runs: 99.22+135.62+141.97+143.06+142.30 = 662.17s |
| R2-110 | F1 | 210 | 18.1 min | SON time (paper) | hardware-dependent | quotes-paper | SON: 18.1 min \| JSON: 1085.6s \| 1085.6 / 60 = 18.09 min -- **VERIFIED** |
| R2-111 | F1 | 210 | 18.09 min | 1085.6 / 60 | hardware-dependent | asserts-own | 1085.6 / 60 = 18.09 min -- **VERIFIED** |
| R2-112 | F1 | 211 | 50.7 s | Direct GPU time (paper) | hardware-dependent | quotes-paper | Direct GPU: 50.7s \| JSON: 50.72s \| **VERIFIED** |
| R2-113 | F1 | 222–224 | 2 GO terms (GO:0003676, GO:0005524) | InterPro2GO mapping of IPR011545 (PF00270) | external-fact | asserts-own | **IPR011545** (integrates PF00270 DEAD/DEAH helicase) maps to: GO:0003676 … **GO:0005524 (ATP binding)** |
| R2-114 | F1 | 225–226 | 0 GO terms | InterPro2GO mapping of IPR001650 (PF00271) | external-fact | asserts-own | **IPR001650** (integrates PF00271 Helicase C-terminal) maps to: No GO terms |
| R2-115 | F1 | 228, 249, 297 | 1 of 19 | GO terms in K=22 confirmed auto-derived (GO:0005524) | deterministic | asserts-own | Only **1 of 19 GO terms** (GO:0005524, ATP binding) is confirmed auto-derived from Pfam via InterPro2GO. |
| R2-116 | F1 | 228, 237, 241, 249, 297 | 19 | GO terms in the K=22 itemset | deterministic | asserts-own | we cannot determine how many of the 19 GO terms are experimentally validated |
| R2-117 | F1 | 243, 250 | 0 | parent-child GO pairs in the K=22 set | deterministic | asserts-own | **No parent-child pairs exist in the K=22 set.** |
| R2-118 | F1 | 243 | 22 | independent K after GO-hierarchy check (PROJECT_STATE.md) | deterministic | quotes-paper | PROJECT_STATE.md: "GO hierarchy: 0 parent-child pairs -> Independent K = 22" |
| R2-119 | F1 | 254 | 22 → 21 | effective independent feature count | deterministic | asserts-own | This reduces the effective independent feature count from 22 to 21. |
| R2-120 | F1 | 266, 279, 298, 302 | 247 | Pfam domains (paper Table 1) | deterministic | disputes | Pfam domains \| 247 \| **500** |
| R2-121 | F1 | 266, 275, 276, 279, 298, 302 | 500 | Pfam domains (item_mapping parquet) | deterministic | asserts-own | Actually 500 Pfam + 500 GO + 2 pLDDT = 1,002 |
| R2-122 | F1 | 267 | 302 | GO MF terms (paper Table 1) | deterministic | disputes | GO terms (MF) \| 302 \| **cannot split** |
| R2-123 | F1 | 268 | 289 | GO BP terms (paper Table 1) | deterministic | disputes | GO terms (BP) \| 289 \| **cannot split** |
| R2-124 | F1 | 269 | 161 | GO CC terms (paper Table 1) | deterministic | disputes | GO terms (CC) \| 161 \| **cannot split** |
| R2-125 | F1 | 270, 279, 298, 302 | 752 | GO terms total (paper Table 1) | deterministic | disputes | GO terms (total) \| 752 \| **500** |
| R2-126 | F1 | 270, 275, 276, 279, 298, 302 | 500 | GO terms (item_mapping parquet) | deterministic | asserts-own | GO terms (total) \| 752 \| **500** |
| R2-127 | F1 | 271, 276, 277, 302 | 3 | pLDDT bins (paper Table 1) | deterministic | disputes | pLDDT bins \| 3 \| **2** (frequent) |
| R2-128 | F1 | 271, 275, 277, 279, 298, 302 | 2 | pLDDT bins frequent at min_count=8 (plddt_mean_med, plddt_mean_high) | deterministic | asserts-own | only 2 are frequent (plddt_mean_med and plddt_mean_high; plddt_mean_low has < 8 proteins) |
| R2-129 | F1 | 272, 274, 302 | 1,002 | total features (paper Table 1 = parquet) | deterministic | quotes-paper | The total matches (1,002 frequent features), but the breakdown is wrong |
| R2-130 | F1 | 276 | 1,006 | entries in item_mapping_214m.parquet | deterministic | asserts-own | The item_mapping_214m.parquet contains 1,006 entries (500 pfam + 500 go_term + 6 plddt) |
| R2-131 | F1 | 276 | 6 | plddt entries in item mapping | deterministic | asserts-own | 1,006 entries (500 pfam + 500 go_term + 6 plddt) |
| R2-132 | F1 | 276 | 4 | mapping entries not frequent at min_count=8 | deterministic | asserts-own | of which 4 are not frequent at min_count=8 (plddt_mean_low + 3 plddt_fraction items) |
| R2-133 | F1 | 276 | 3 | plddt_fraction items (not frequent) | deterministic | asserts-own | (plddt_mean_low + 3 plddt_fraction items) |
| R2-134 | F1 | 277 | < 8 | proteins carrying plddt_mean_low | deterministic | asserts-own | plddt_mean_low has < 8 proteins |
| R2-135 | F1 | 279, 302 | 500:500:2 (not 247:752:3) | feature-type ratio | deterministic | asserts-own | The actual ratio is 500:500:2, not 247:752:3. |
| R2-136 | F1 | 308 | min_count=8 (0.00001%) | requested null-model threshold | method-parameter | requests | **Run null model at 0.00001% threshold (min_count=8)** to properly validate the K=22 result. |
| R2-137 | F1 | 312 | 21 | independent feature count to be noted in paper | deterministic | requests | "GO:0005524 (ATP binding) is automatically mapped from PF00270 via InterPro2GO, reducing the independent feature count to 21." |

### F2 — `revision_notes_b3.tex`

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R2-138 | F2 | 1, 14, 25, 54, 192, 210, 246, 251, 290, 312, 318, 322, 329, 336, 341, 355, 356, 363, 373, 377, 380, 413, 415, 422, 429, 437, 474, 534, 586, 600, 624, 639 | 35K | features, expanded vocabulary (rounded) | deterministic | asserts-own | LaTeX revision blocks for 35K-feature results |
| R2-139 | F2 | 14, 23, 39, 54, 60, 336, 387, 555, 565, 567, 574, 637 | 0.001% | support of the completed 35K run / controlled comparison | method-parameter | asserts-own | 35K MINING CAMPAIGN RESULTS (0.001% COMPLETE) |
| R2-140 | F2 | 17, 100, 117, 127, 178, 264, 319, 470 | 1,002 | features, initial vocabulary | deterministic | quotes-paper | after the existing Table 2 (1,002-feature campaign) |
| R2-141 | F2 | 21, 60, 99, 117, 127, 201, 215, 240, 264, 320, 388, 470, 628 | 35,012 | features, expanded vocabulary | deterministic | asserts-own | Mining campaign results with the expanded 35,012-feature vocabulary |
| R2-142 | F2 | 22, 177, 185 | 109.2 million | proteins, expanded dataset | deterministic | asserts-own | across 109.2 million proteins on 4$\times$NVIDIA H200 143\,GB (row-split architecture) |
| R2-143 | F2 | 22, 186, 230, 239, 383, 438, 602, 616, 624, 651 | 4 | GPU count (H200) for 35K experiments | hardware-dependent | asserts-own | on 4$\times$NVIDIA H200 143\,GB (row-split architecture) |
| R2-144 | F2 | 22, 186, 220, 230, 383, 438, 602, 616, 624, 651 | H200 (SXM5 at l.624) | GPU model | hardware-dependent | asserts-own | 4$\times$NVIDIA H200 143\,GB … 4$\times$NVIDIA H200 SXM5 141\,GB GPUs |
| R2-145 | F2 | 22, 187, 220, 641 | 143 GB | H200 VRAM | hardware-dependent | asserts-own | This exceeds the 143\,GB VRAM of the NVIDIA H200 |
| R2-146 | F2 | 23, 61 | 3 | replicates of the 0.001% run | method-parameter | asserts-own | The 0.001\% run reports mean $\pm$ std over 3 replicates [ACTUAL] |
| R2-147 | F2 | 37, 47, 585, 600, 605, 654 | 0.1% | Base run support | method-parameter | asserts-own | Base & 0.1\% & 109,225 & \textbf{TBD} & ${\geq}10$ & \textbf{TBD} |
| R2-148 | F2 | 37, 585, 605 | 109,225 | Base min proteins (min_count) | method-parameter | asserts-own | Base & 0.1\% & 109,225 … The Base run (0.1\%, min\_count=109,225) |
| R2-149 | F2 | 37 | ≥10 | Base max K (partial, run ongoing) | deterministic | asserts-own | ${\geq}10$\textsuperscript{$\ast$} … Base run (0.1\%) partial |
| R2-150 | F2 | 38, 655 | 0.01% | Super run support | method-parameter | asserts-own | Super & 0.01\% & 10,923 & \textbf{TBD} |
| R2-151 | F2 | 38 | 10,923 | Super min proteins | method-parameter | asserts-own | Super & 0.01\% & 10,923 |
| R2-152 | F2 | 39, 61, 637 | 606,292 ± 28 | Power itemsets, mean ± std over 3 replicates | deterministic | asserts-own | Power & 0.001\% & 1,093 & 606,292 $\pm$ 28 & 17 & 654.3 $\pm$ 6.7 |
| R2-153 | F2 | 39, 320, 336, 340 | 1,093 | Power min_count (0.001% of 109.2M) | method-parameter | asserts-own | Power & 0.001\% & 1,093 … min\_count$=$1,093 (0.001\% of 109M) |
| R2-154 | F2 | 39, 320, 322, 356, 431, 643, 658 | 17 | Power max K | deterministic | asserts-own | 606,292 $\pm$ 28 & 17 & 654.3 $\pm$ 6.7 … $K{=}17$ (35,012 features, min\_count$=$1,093) |
| R2-155 | F2 | 39 | 654.3 ± 6.7 s | Power time, mean ± std | hardware-dependent | asserts-own | 606,292 $\pm$ 28 & 17 & 654.3 $\pm$ 6.7 & Direct row-split |
| R2-156 | F2 | 40, 566, 655 | 0.0001% | Blitz run support | method-parameter | asserts-own | Blitz & 0.0001\% & 109 & \textbf{TBD} |
| R2-157 | F2 | 40 | 109 | Blitz min proteins | method-parameter | asserts-own | Blitz & 0.0001\% & 109 |
| R2-158 | F2 | 41 | 0.00002% | Ultra run support | method-parameter | asserts-own | Ultra & 0.00002\% & 22 & \textbf{TBD} |
| R2-159 | F2 | 41 | 22 | Ultra min proteins | method-parameter | asserts-own | Ultra & 0.00002\% & 22 |
| R2-160 | F2 | 42, 337 | 0.00001% | Opus run support | method-parameter | asserts-own | Opus & 0.00001\% & 11 & \textbf{TBD} |
| R2-161 | F2 | 42 | 11 | Opus min proteins (35K) | method-parameter | asserts-own | Opus & 0.00001\% & 11 |
| R2-162 | F2 | 48, 596, 606, 610 | 10,041,611 | Base K=9 frequent itemsets (partial) | deterministic | asserts-own | $K{=}9$ reached 10,041,611 frequent itemsets and $K{=}10$ started with 12,491,079 locally frequent candidates before timeout |
| R2-163 | F2 | 49, 597, 607 | 12,491,079 | Base K=10 locally frequent candidates (started, not completed) | deterministic | asserts-own | $K{=}10$ started with 12,491,079 locally frequent candidates before timeout. Run is ongoing. |
| R2-164 | F2 | 61, 79, 362 | K=7 | peak of the 35K 0.001% K-distribution | deterministic | asserts-own | Peak at $K{=}7$ (10.62\%). |
| R2-165 | F2 | 61, 79 | 10.62% | share of itemsets at K=7 | deterministic | asserts-own | Peak at $K{=}7$ (10.62\%). |
| R2-166 | F2 | 62–64 | deterministic at K≤3 and K≥11; variance at K=4..10 | replicate-variance profile | deterministic | asserts-own | Values deterministic at $K{\leq}3$ and $K{\geq}11$; minor variance at intermediate levels from GPU floating-point non-determinism |
| R2-167 | F2 | 73, 400, 448 | 34,920 | K=1 itemsets (35K, 0.001%) | deterministic | asserts-own | 1 & 34,920 & 5.76 & 10 & 43,869 & 7.24 |
| R2-168 | F2 | 73 | 5.76% | K=1 share | deterministic | asserts-own | 1 & 34,920 & 5.76 |
| R2-169 | F2 | 73 | 43,869 | K=10 itemsets | deterministic | asserts-own | 10 & 43,869 & 7.24 |
| R2-170 | F2 | 73 | 7.24% | K=10 share | deterministic | asserts-own | 10 & 43,869 & 7.24 |
| R2-171 | F2 | 74 | 46,853 | K=2 itemsets (kdist table) | deterministic | asserts-own | 2 & 46,853 & 7.73 & 11 & 30,116 & 4.97 |
| R2-172 | F2 | 74 | 7.73% | K=2 share | deterministic | asserts-own | 2 & 46,853 & 7.73 |
| R2-173 | F2 | 74 | 30,116 | K=11 itemsets | deterministic | asserts-own | 11 & 30,116 & 4.97 |
| R2-174 | F2 | 74 | 4.97% | K=11 share | deterministic | asserts-own | 11 & 30,116 & 4.97 |
| R2-175 | F2 | 75 | 53,238 | K=3 itemsets (kdist table) | deterministic | asserts-own | 3 & 53,238 & 8.78 & 12 & 17,355 & 2.86 |
| R2-176 | F2 | 75 | 8.78% | K=3 share | deterministic | asserts-own | 3 & 53,238 & 8.78 |
| R2-177 | F2 | 75 | 17,355 | K=12 itemsets | deterministic | asserts-own | 12 & 17,355 & 2.86 |
| R2-178 | F2 | 75 | 2.86% | K=12 share | deterministic | asserts-own | 12 & 17,355 & 2.86 |
| R2-179 | F2 | 76 | 59,086 | K=4 itemsets (kdist table) | deterministic | asserts-own | 4 & 59,086 & 9.75 & 13 & 8,096 & 1.34 |
| R2-180 | F2 | 76 | 9.75% | K=4 share | deterministic | asserts-own | 4 & 59,086 & 9.75 |
| R2-181 | F2 | 76 | 8,096 | K=13 itemsets | deterministic | asserts-own | 13 & 8,096 & 1.34 |
| R2-182 | F2 | 76 | 1.34% | K=13 share | deterministic | asserts-own | 13 & 8,096 & 1.34 |
| R2-183 | F2 | 77 | 62,226 | K=5 itemsets (kdist table) | deterministic | asserts-own | 5 & 62,226 & 10.26 & 14 & 2,946 & 0.49 |
| R2-184 | F2 | 77 | 10.26% | K=5 share | deterministic | asserts-own | 5 & 62,226 & 10.26 |
| R2-185 | F2 | 77 | 2,946 | K=14 itemsets | deterministic | asserts-own | 14 & 2,946 & 0.49 |
| R2-186 | F2 | 77 | 0.49% | K=14 share | deterministic | asserts-own | 14 & 2,946 & 0.49 |
| R2-187 | F2 | 78 | 63,708 | K=6 itemsets | deterministic | asserts-own | 6 & 63,708 & 10.51 & 15 & 800 & 0.13 |
| R2-188 | F2 | 78 | 10.51% | K=6 share | deterministic | asserts-own | 6 & 63,708 & 10.51 |
| R2-189 | F2 | 78 | 800 | K=15 itemsets | deterministic | asserts-own | 15 & 800 & 0.13 |
| R2-190 | F2 | 78 | 0.13% | K=15 share | deterministic | asserts-own | 15 & 800 & 0.13 |
| R2-191 | F2 | 79 | 64,396 | K=7 itemsets (peak) | deterministic | asserts-own | \textbf{7} & \textbf{64,396} & \textbf{10.62} & 16 & 152 & 0.03 |
| R2-192 | F2 | 79 | 152 | K=16 itemsets | deterministic | asserts-own | 16 & 152 & 0.03 |
| R2-193 | F2 | 79 | 0.03% | K=16 share | deterministic | asserts-own | 16 & 152 & 0.03 |
| R2-194 | F2 | 80 | 62,948 | K=8 itemsets | deterministic | asserts-own | 8 & 62,948 & 10.38 & 17 & 1 & $<\!$0.01 |
| R2-195 | F2 | 80 | 10.38% | K=8 share | deterministic | asserts-own | 8 & 62,948 & 10.38 |
| R2-196 | F2 | 80 | 1 | K=17 itemsets | deterministic | asserts-own | 17 & 1 & $<\!$0.01 |
| R2-197 | F2 | 80 | <0.01% | K=17 share | deterministic | asserts-own | 17 & 1 & $<\!$0.01 |
| R2-198 | F2 | 81 | 55,582 | K=9 itemsets | deterministic | asserts-own | 9 & 55,582 & 9.17 |
| R2-199 | F2 | 81 | 9.17% | K=9 share | deterministic | asserts-own | 9 & 55,582 & 9.17 |
| R2-200 | F2 | 99, 129, 355 | 7 | annotation sources in expanded vocabulary | method-parameter | asserts-own | The expanded vocabulary comprises 35,012 features across seven annotation sources |
| R2-201 | F2 | 100 | 35× | vocabulary size increase (35,012 / 1,002) | deterministic | asserts-own | a $35\times$ increase over the initial 1,002-feature set |
| R2-202 | F2 | 101, 185 | 3,423 | min feature count for vocabulary retention | method-parameter | asserts-own | Features retained at min\_count${\geq}$3,423. |
| R2-203 | F2 | 109, 131, 135, 329 | 12,773 | InterPro domain/family features | deterministic | asserts-own | Protein domains \& families & InterPro & 12,773 & (247) |
| R2-204 | F2 | 109, 134, 326, 329 | 247 | Pfam domains in the 1,002 set | deterministic | quotes-paper | Protein domains \& families & InterPro & 12,773 & (247) |
| R2-205 | F2 | 110, 137, 140 | 5,763 | GO terms (expanded) | deterministic | asserts-own | GO terms (all three ontologies) & Gene Ontology & 5,763 & (752) |
| R2-206 | F2 | 110, 139, 326 | 752 | GO terms in the 1,002 set | deterministic | quotes-paper | GO terms (all three ontologies) & Gene Ontology & 5,763 & (752) |
| R2-207 | F2 | 111, 144, 330 | 1,107 | EC number features (leaf-only) | deterministic | asserts-own | Enzyme Commission numbers & IUBMB (leaf-only) & 1,107 & --- |
| R2-208 | F2 | 112, 151, 330 | 15,162 | UniProt Keyword features | deterministic | asserts-own | UniProt Keywords & UniProtKB & 15,162 & --- |
| R2-209 | F2 | 113, 158 | 175 | taxonomic-class features (leaf-only) | deterministic | asserts-own | Taxonomic lineage & UniProt (class-level, leaf-only) & 175 & --- |
| R2-210 | F2 | 114, 166, 168 | 26 | sequence-length bins (log-scale) | method-parameter | asserts-own | Sequence length bins & Derived (log-scale) & 26 & --- |
| R2-211 | F2 | 115, 172, 173 | 6 | pLDDT confidence bins (expanded) | method-parameter | asserts-own | pLDDT confidence bins & AlphaFold & 6 & (3) |
| R2-212 | F2 | 115, 175 | 3 | pLDDT bins in the 1,002 set | method-parameter | quotes-paper | pLDDT confidence bins & AlphaFold & 6 & (3) … expanded from the 3-bin encoding in the initial vocabulary |
| R2-213 | F2 | 127, 264, 470 | 1,002 → 35,012 | vocabulary expansion (old → new) | deterministic | asserts-own | We expanded the feature vocabulary from 1,002 to 35,012 features |
| R2-214 | F2 | 134–135 | 247 → 12,773 | domain coverage, Pfam → InterPro (old → new) | deterministic | asserts-own | increasing domain coverage from 247 to 12,773 entries while eliminating deduplication concerns |
| R2-215 | F2 | 139–140 | 752 → 5,763 | GO coverage (old → new) | deterministic | asserts-own | expand coverage from 752 to 5,763 terms by lowering the frequency threshold |
| R2-216 | F2 | 145 | 4 | levels in the EC hierarchy | external-fact | asserts-own | EC numbers classify enzyme-catalyzed reactions in a four-level hierarchy (class.subclass.sub-subclass.serial) |
| R2-217 | F2 | 167–168 | 100–150, 150–200, …, >5,000 aa | length-bin edges (log-scale, 26 bins) | method-parameter | asserts-own | log-scale bins (e.g., 100--150~aa, 150--200~aa, $\ldots$, $>$5{,}000~aa), providing 26 discrete size categories |
| R2-218 | F2 | 173–174 | 3 mean-pLDDT bins (<70, 70–90, >90) + 3 fraction bins | pLDDT binning scheme (6 bins) | method-parameter | asserts-own | three for mean pLDDT ($<$70, 70--90, $>$90) and three for the fraction of high-confidence residues |
| R2-219 | F2 | 175 | 3 → 6 | pLDDT bins (old → new) | method-parameter | asserts-own | expanded from the 3-bin encoding in the initial vocabulary |
| R2-220 | F2 | 177–178 | 76.9M → 109.2M | proteins with ≥1 feature (old → new) | deterministic | asserts-own | 109.2~million proteins with at least one annotation feature (up from 76.9~million in the 1,002-feature set) |
| R2-221 | F2 | 178, 337 | 76.9 million (77M at l.337) | proteins in the 1,002-feature set | deterministic | quotes-paper | up from 76.9~million in the 1,002-feature set … (0.00001\% of 77M) |
| R2-222 | F2 | 180 | 150 GB | UniProt .dat.gz input processed | external-fact | asserts-own | Feature extraction processed 150\,GB of UniProt \texttt{.dat.gz} files using a custom Rust parser |
| R2-223 | F2 | 182 | 95.6K records/s | dat_to_tsv.rs parser throughput | hardware-dependent | asserts-own | (\texttt{dat\_to\_tsv.rs}, 95.6K records/s) |
| R2-224 | F2 | 183 | 495K lines/s | build_tx.rs throughput | hardware-dependent | asserts-own | two-pass Rust transaction builder (\texttt{build\_tx.rs}, 495K lines/s) |
| R2-225 | F2 | 184 | 858 s | transaction build time (frequency counting + remapping) | hardware-dependent | asserts-own | performed frequency counting and feature remapping in 858\,s |
| R2-226 | F2 | 185 | ~0.003% | 3,423 as fraction of 109.2M | method-parameter | asserts-own | 3,423 (corresponding to ${\sim}0.003\%$ of 109.2M proteins) |
| R2-227 | F2 | 186, 629 | 477 GB | total bitvector matrix (35K × 109.2M) | deterministic | asserts-own | constrain the bitvector matrix to 477\,GB total across four NVIDIA H200 143\,GB GPUs |
| R2-228 | F2 | 187, 241, 628 | ~119 GB | bitvector shard per device | deterministic | asserts-own | (${\sim}119$\,GB per device with 12\,GB headroom for workspace allocations) |
| R2-229 | F2 | 187 | 12 GB | per-device headroom for workspace | hardware-dependent | asserts-own | ${\sim}119$\,GB per device with 12\,GB headroom for workspace allocations |
| R2-230 | F2 | 207, 215–216, 271, 642 | \|F\| × ⌈N/64⌉ × 8 B ; T_bitvec = \|C_K\|·K·⌈N/64⌉ | bitvector memory / cost model | deterministic | asserts-own | the bitvector matrix for each chunk has dimensions $\|F\| \times \lceil N_{\text{chunk}} / 64 \rceil$ |
| R2-231 | F2 | 212, 215 | 40M | SON chunk size (example) | method-parameter | asserts-own | A single chunk of $N_{\text{chunk}} = 40$M transactions requires: |
| R2-232 | F2 | 216, 641 | 175 GB | chunk bitvector at 40M transactions, 35,012 features | deterministic | asserts-own | 35{,}012 \times \left\lceil \frac{40 \times 10^6}{64} \right\rceil \times 8 \;\text{bytes} = 175\;\text{GB} |
| R2-233 | F2 | 220–221 | H200 = highest-capacity datacenter GPU currently available | hardware ranking claim | external-fact | asserts-own | the 143\,GB VRAM of the NVIDIA H200---the highest-capacity datacenter GPU currently available |
| R2-234 | F2 | 222 | 30M | reduced SON chunk size | method-parameter | asserts-own | (e.g., $N_{\text{chunk}} = 30$M $\rightarrow$ 131\,GB) |
| R2-235 | F2 | 222 | 131 GB | chunk bitvector at 30M | deterministic | asserts-own | (e.g., $N_{\text{chunk}} = 30$M $\rightarrow$ 131\,GB) |
| R2-236 | F2 | 223 | 4 | sequential SON passes ⌈109M/30M⌉ | deterministic | asserts-own | would require at least $\lceil 109\text{M} / 30\text{M} \rceil = 4$ sequential passes |
| R2-237 | F2 | 223, 292, 295, 336 | 109M / 109 million | proteins (rounded) | deterministic | asserts-own | Across 109~million transactions, this produces $8.6 \times 10^{10}$ subset enumerations |
| R2-238 | F2 | 231 | > 22 hours | SON streaming run hung at Pass 1 (4×H200) before termination | hardware-dependent | asserts-own | a SON streaming run on 4$\times$H200 GPUs hung at Pass~1 for over 22 hours before being terminated |
| R2-239 | F2 | 239 | ~27M | proteins per GPU (4 GPUs) | deterministic | asserts-own | Each GPU holds a fraction of the proteins (${\sim}27$M each on 4 GPUs) |
| R2-240 | F2 | 242, 630 | 1 | allreduce per K-level | method-parameter | asserts-own | Local support counts are reduced across GPUs with a single \texttt{allreduce} operation per $K$-level |
| R2-241 | F2 | 250, 542, 553, 567, 575, 649 | 21.4× | SON slowdown / Direct GPU speedup (1K features, 0.001%) | hardware-dependent | asserts-own | ``SON is $21.4\times$ slower'' (Section~\ref{sec:results}, 1K features) |
| R2-242 | F2 | 286 | T_sparse = N · C(\|t\|, K) | hash-tree Apriori cost model | deterministic | asserts-own | T_{\text{sparse}} = N \cdot \binom{\bar{\|t\|}}{K} |
| R2-243 | F2 | 291, 355 | ~12 | average annotated features per protein (35K) | deterministic | asserts-own | With the 35K-feature dataset ($\bar{\|t\|} \approx 12$) at $K{=}5$ |
| R2-244 | F2 | 292 | 792 | C(12,5) subsets per transaction at K=5 | deterministic | asserts-own | each transaction generates $\binom{12}{5} = 792$ subsets |
| R2-245 | F2 | 293 | 8.6 × 10^10 | subset enumerations at K=5 across 109M transactions | deterministic | asserts-own | Across 109~million transactions, this produces $8.6 \times 10^{10}$ subset enumerations at $K{=}5$ alone |
| R2-246 | F2 | 295–296, 300 | ~1.7 × 10^6 | 64-bit ops per candidate, ⌈109×10^6/64⌉ | deterministic | asserts-own | $\lceil 109{\times}10^6 / 64 \rceil \approx 1.7 \times 10^6$ 64-bit operations per candidate at \emph{any} $K$ |
| R2-247 | F2 | 297 | 100 | hypothetical average transaction length | method-parameter | asserts-own | At $\bar{\|t\|}{=}100$ and $K{=}5$: |
| R2-248 | F2 | 298 | 75,287,520 | C(100,5) subsets per transaction | deterministic | asserts-own | sparse traversal generates $\binom{100}{5} = 75{,}287{,}520$ subsets per transaction |
| R2-249 | F2 | 319, 323, 339, 353, 358, 643 | K=22 | 1K max depth (min_count=8) | deterministic | quotes-paper | maximum itemset depth from $K{=}22$ (1,002 features, min\_count$=$8) to $K{=}17$ |
| R2-250 | F2 | 319, 337, 338 | 8 | 1K Opus min_count | method-parameter | quotes-paper | $K{=}22$ (1,002 features, min\_count$=$8) … min\_count$=$8 (0.00001\% of 77M) for the 1K Opus run |
| R2-251 | F2 | 319–320, 643 | K=22 → K=17 | max itemset depth, 1K → 35K (old → new) | deterministic | asserts-own | the \emph{decrease} in maximum itemset depth from $K{=}22$ … to $K{=}17$ (35,012 features, min\_count$=$1,093) |
| R2-252 | F2 | 337 | 137× | absolute threshold ratio 1,093 / 8 | deterministic | asserts-own | a $137\times$ higher absolute threshold |
| R2-253 | F2 | 339, 353 | 8 | proteins sharing the K=22 itemset | deterministic | quotes-paper | (the $K{=}22$ itemset was shared by exactly 8 proteins) |
| R2-254 | F2 | 340 | > 1,000 | proteins a pattern must reach at min_count=1,093 | method-parameter | asserts-own | At min\_count$=$1,093, only patterns present in $>$1,000 proteins qualify. |
| R2-255 | F2 | 346 | 1 (not 4) | features per EC annotation under leaf-only encoding | method-parameter | asserts-own | EC~3.4.21.4 contributes one feature, not four (the leaf plus three ancestors) |
| R2-256 | F2 | 347 | 7–8 | "free" GO co-occurrences per protein from true-path rule (1K vocabulary) | deterministic | asserts-own | GO's true-path rule contributed up to 7--8 ``free'' co-occurrences per protein from hierarchical nesting alone |
| R2-257 | F2 | 353–354 | 8 proteins with exactly 22 features each | K=22 ceiling determined by annotation depth | deterministic | asserts-own | The $K{=}22$ ceiling in the 1K dataset was determined by 8 proteins with exactly 22 features each |
| R2-258 | F2 | 359 | 1 | pLDDT bin in the 1K K=22 pattern | deterministic | asserts-own | combined only Pfam and GO terms with one pLDDT bin, all from two closely related annotation sources |
| R2-259 | F2 | 362 | K=9 | 1K K-distribution peak | deterministic | quotes-paper | The $K$-distribution peak also shifted: from $K{=}9$ (1K features) to $K{=}7$ (35K features) |
| R2-260 | F2 | 362 | K=9 → K=7 | K-distribution peak, 1K → 35K (old → new) | deterministic | asserts-own | from $K{=}9$ (1K features) to $K{=}7$ (35K features) |
| R2-261 | F2 | 370, 373, 382, 388, 644 | 2 | permutations completed (35K null model) | method-parameter | asserts-own | Two permutations were completed on 4$\times$H200 GPUs |
| R2-262 | F2 | 373 | 5 | permutations in the existing (1K) null model | method-parameter | quotes-paper | Replace existing 5-permutation null model with 35K 2-perm data. |
| R2-263 | F2 | 374, 434, 437, 647, 656 | 100 | permutations planned (TBD) | method-parameter | requests | Prepare for 100-permutation update. … 35K null model: 100-permutation results (\textbf{TBD}) |
| R2-264 | F2 | 383 | 1,310 s | total GPU time, 2 permutations | hardware-dependent | asserts-own | (total GPU time: 1,310\,s, wall-clock: 1,456\,s) |
| R2-265 | F2 | 383 | 1,456 s | wall-clock, 2 permutations | hardware-dependent | asserts-own | (total GPU time: 1,310\,s, wall-clock: 1,456\,s) |
| R2-266 | F2 | 388–389, 415, 646 | K=5 | null model max K (35K) | deterministic | asserts-own | The null model collapses at $K{=}5$ (1 itemset per permutation) |
| R2-267 | F2 | 389, 404, 415 | 1 | null itemsets at K=5, per permutation | deterministic | asserts-own | it barely reaches $K{=}5$ (exactly 1 itemset per permutation) |
| R2-268 | F2 | 390, 402, 423, 645 | +722.9 (≈723) | K=3 Z (35K) | deterministic | asserts-own | ($Z{=}723$ at $K{=}3$, $Z{>}20{,}000$ at $K{=}4$, $Z{=}\infty$ for $K{\geq}5$) |
| R2-269 | F2 | 390, 403, 645 | +20,585 (>20,000) | K=4 Z (35K) | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ & ${\approx}\,0$ & enriched |
| R2-270 | F2 | 390–391, 404, 405 | +∞ | Z for K≥5 (35K) | deterministic | asserts-own | $Z{=}\infty$ for $K{\geq}5$ |
| R2-271 | F2 | 400 | 34,920 | K=1 null μ (marginals preserved) | deterministic | asserts-own | 1 & 34,920 & 34,920 & 0.0 & 0.0 & 1.0 & preserved |
| R2-272 | F2 | 400 | 0.0 | K=1 null σ | deterministic | asserts-own | 1 & 34,920 & 34,920 & 0.0 & 0.0 & 1.0 & preserved |
| R2-273 | F2 | 400 | 0.0 | K=1 Z | deterministic | asserts-own | 1 & 34,920 & 34,920 & 0.0 & 0.0 & 1.0 & preserved |
| R2-274 | F2 | 400 | 1.0 | K=1 p | deterministic | asserts-own | 1 & 34,920 & 34,920 & 0.0 & 0.0 & 1.0 & preserved |
| R2-275 | F2 | 401, 449 | 46,847 | K=2 bio (null-model table) | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-276 | F2 | 401 | 59,415 | K=2 null μ | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-277 | F2 | 401 | 123.7 | K=2 null σ | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-278 | F2 | 401 | −101.6 | K=2 Z | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-279 | F2 | 401 | 1.0 | K=2 p | deterministic | asserts-own | 2 & 46,847 & 59,415 & 123.7 & $-101.6$ & 1.0 & depleted |
| R2-280 | F2 | 402, 450 | 53,235 | K=3 bio (null-model table) | deterministic | asserts-own | 3 & 53,235 & 15,920 & 51.6 & $+722.9$ & ${\approx}\,0$ & enriched |
| R2-281 | F2 | 402 | 15,920 | K=3 null μ | deterministic | asserts-own | 3 & 53,235 & 15,920 & 51.6 & $+722.9$ |
| R2-282 | F2 | 402 | 51.6 | K=3 null σ | deterministic | asserts-own | 3 & 53,235 & 15,920 & 51.6 & $+722.9$ |
| R2-283 | F2 | 402 | ≈0 | K=3 p | deterministic | asserts-own | 3 & 53,235 & 15,920 & 51.6 & $+722.9$ & ${\approx}\,0$ & enriched |
| R2-284 | F2 | 403, 451 | 59,095 | K=4 bio (null-model table) | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ & ${\approx}\,0$ & enriched |
| R2-285 | F2 | 403 | 872 | K=4 null μ | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ |
| R2-286 | F2 | 403 | 2.8 | K=4 null σ | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ |
| R2-287 | F2 | 403 | ≈0 | K=4 p | deterministic | asserts-own | 4 & 59,095 & 872 & 2.8 & $+20{,}585$ & ${\approx}\,0$ & enriched |
| R2-288 | F2 | 404, 452 | 62,241 | K=5 bio (null-model table) | deterministic | asserts-own | 5 & 62,241 & 1.0 & 0.0 & $+\infty$ & ${\approx}\,0$ & enriched |
| R2-289 | F2 | 404 | 1.0 | K=5 null μ | deterministic | asserts-own | 5 & 62,241 & 1.0 & 0.0 & $+\infty$ |
| R2-290 | F2 | 404 | 0.0 | K=5 null σ | deterministic | asserts-own | 5 & 62,241 & 1.0 & 0.0 & $+\infty$ |
| R2-291 | F2 | 404 | ≈0 | K=5 p | deterministic | asserts-own | 5 & 62,241 & 1.0 & 0.0 & $+\infty$ & ${\approx}\,0$ & enriched |
| R2-292 | F2 | 405, 432, 453 | 348,853 | K=6–17 bio itemsets | deterministic | asserts-own | 6--17 & 348,853 & 0 & 0.0 & $+\infty$ & ${\approx}\,0$ & enriched |
| R2-293 | F2 | 405, 416 | 0 | K=6–17 null μ (zero itemsets at K≥6) | deterministic | asserts-own | 6--17 & 348,853 & 0 & 0.0 … produces zero itemsets at $K{\geq}6$ |
| R2-294 | F2 | 405 | 0.0 | K=6–17 null σ | deterministic | asserts-own | 6--17 & 348,853 & 0 & 0.0 & $+\infty$ |
| R2-295 | F2 | 405 | ≈0 | K=6–17 p | deterministic | asserts-own | 6--17 & 348,853 & 0 & 0.0 & $+\infty$ & ${\approx}\,0$ & enriched |
| R2-296 | F2 | 414, 646 | K=6 | 1K null-model max K | deterministic | quotes-paper | With 1K features, the null model reached $K{=}6$ (22 itemsets mean) |
| R2-297 | F2 | 414 | 22 | 1K null K=6 mean itemsets | deterministic | quotes-paper | With 1K features, the null model reached $K{=}6$ (22 itemsets mean) |
| R2-298 | F2 | 421 | Z=−143 | 1K K=3 Z (depleted) | deterministic | quotes-paper | With 1K features, $K{=}3$ was depleted ($Z{=}{-}143$) |
| R2-299 | F2 | 423 | 3.3× | bio/null triples ratio at K=3 (35K) | deterministic | asserts-own | biology produces $3.3\times$ more triples than chance |
| R2-300 | F2 | 429 | 5.5× | total bio/null itemset ratio (35K) | deterministic | asserts-own | The 35K dataset produces $5.5\times$ more total itemsets than the null (606K vs.\ 111K) |
| R2-301 | F2 | 430, 645 | 606K | bio total itemsets (rounded) | deterministic | asserts-own | (606K vs.\ 111K) |
| R2-302 | F2 | 430, 645 | 111K | null total itemsets (35K, 2 permutations) | deterministic | asserts-own | (606K vs.\ 111K) |
| R2-303 | F2 | 431 | 12 | K-levels beyond null reach (K=6..17) | deterministic | asserts-own | With 12 levels beyond the null's reach ($K{=}6$ through $K{=}17$) containing 348,853 itemsets |
| R2-304 | F2 | 438 | ~655 s | time per permutation (GPU Fisher–Yates, 4×H200) | hardware-dependent | asserts-own | GPU-accelerated Fisher--Yates shuffle on 4$\times$H200, ${\sim}655$\,s per permutation. |
| R2-305 | F2 | 474–476 | 4 | additional annotation sources vs 1K vocabulary | method-parameter | asserts-own | The 35K vocabulary introduces four additional annotation sources (EC numbers, UniProt Keywords, taxonomic lineage, and sequence length) |
| R2-306 | F2 | 476 | 6 | categories of cross-source patterns | deterministic | asserts-own | creating six categories of cross-source patterns that could not be discovered with any single-source analysis |
| R2-307 | F2 | 548, 552, 558, 576 | 475,865 | Direct GPU itemsets (1K, 0.001%) — retained in replacement text | deterministic | quotes-paper | Direct GPU discovers 475,865 itemsets in 50.7\,s versus SON's 22,846 in 1,085.6\,s |
| R2-308 | F2 | 548, 552, 571, 575 | 50.7 s | Direct GPU time (1K) | hardware-dependent | quotes-paper | Direct GPU discovers 475,865 itemsets in 50.7\,s |
| R2-309 | F2 | 549, 552, 558, 576 | 22,846 | SON itemsets (1K) | deterministic | quotes-paper | versus SON's 22,846 in 1,085.6\,s |
| R2-310 | F2 | 549, 553, 571, 575 | 1,085.6 s | SON time (1K) | hardware-dependent | quotes-paper | versus SON's 22,846 in 1,085.6\,s |
| R2-311 | F2 | 549, 562, 571 | 21× | speedup as currently written in paper (three places) | hardware-dependent | quotes-paper | REPLACE: "… 1,085.6\,s---a $21\times$ speedup." |
| R2-312 | F2 | 548–553 | 21× → 21.4× | Section 3.1 speedup wording (old → new) | hardware-dependent | requests | REPLACE: "…a $21\times$ speedup." WITH: … a $\mathbf{21.4\times}$ speedup (Table~\ref{tab:campaign}, footnote) |
| R2-313 | F2 | 555 | 769 | min_count of the controlled comparison (0.001%) | method-parameter | asserts-own | isolates the method effect at identical support (0.001\%, $\text{min\_count}{=}769$) |
| R2-314 | F2 | 558, 567, 576 | 95.2% | SON miss rate (22,846 of 475,865) | deterministic | asserts-own | SON misses 95.2\% of patterns (22,846 of 475,865) through two compounding mechanisms. |
| R2-315 | F2 | 561–568 | 21× → 21.4× (+ 95.2% miss rate added) | Table 2 footnote (old → new) | hardware-dependent | requests | Replace: "Controlled same-support comparison: $21\times$" With: … $21.4\times$ speedup, 95.2\% SON miss rate |
| R2-316 | F2 | 565–566 | Power (0.001%) vs Blitz (0.0001%) | what the Table 2 † footnote compares (method and threshold both vary) | method-parameter | asserts-own | Compares Power (0.001\%) to Blitz (0.0001\%), varying both method and threshold. |
| R2-317 | F2 | 570–576 | 21× → 21.4× | Discussion 4.1 speedup wording (old → new) | hardware-dependent | requests | Replace: "a $21\times$ speedup (50.7\,s vs.\ 1,085.6\,s)" With: … a $21.4\times$ speedup (50.7\,s vs.\ 1,085.6\,s) |
| R2-318 | F2 | 588, 608 | 1,164 | Base (0.1%) K=1 frequent items | deterministic | asserts-own | K=1:   1,164 frequent items (0.3s) |
| R2-319 | F2 | 588 | 0.3 s | Base K=1 time | hardware-dependent | asserts-own | K=1:   1,164 frequent items (0.3s) |
| R2-320 | F2 | 589, 609 | 8,554 | Base K=2 frequent pairs | deterministic | asserts-own | K=2:   8,554 frequent pairs (1.2s) |
| R2-321 | F2 | 589 | 1.2 s | Base K=2 time | hardware-dependent | asserts-own | K=2:   8,554 frequent pairs (1.2s) |
| R2-322 | F2 | 590, 609 | 28,804 | Base K=3 frequent | deterministic | asserts-own | K=3:  28,804 frequent (2.0s) |
| R2-323 | F2 | 590 | 2.0 s | Base K=3 time | hardware-dependent | asserts-own | K=3:  28,804 frequent (2.0s) |
| R2-324 | F2 | 591, 609 | 88,561 | Base K=4 frequent | deterministic | asserts-own | K=4:  88,561 frequent (5.3s) |
| R2-325 | F2 | 591 | 5.3 s | Base K=4 time | hardware-dependent | asserts-own | K=4:  88,561 frequent (5.3s) |
| R2-326 | F2 | 592, 609 | 280,784 | Base K=5 frequent | deterministic | asserts-own | K=5: 280,784 frequent (17.5s) |
| R2-327 | F2 | 592 | 17.5 s | Base K=5 time | hardware-dependent | asserts-own | K=5: 280,784 frequent (17.5s) |
| R2-328 | F2 | 593, 609 | 835,466 | Base K=6 frequent | deterministic | asserts-own | K=6: 835,466 frequent (52.1s) |
| R2-329 | F2 | 593 | 52.1 s | Base K=6 time | hardware-dependent | asserts-own | K=6: 835,466 frequent (52.1s) |
| R2-330 | F2 | 594, 610 | 2,207,022 | Base K=7 frequent | deterministic | asserts-own | K=7: 2,207,022 frequent (145.3s) |
| R2-331 | F2 | 594 | 145.3 s | Base K=7 time | hardware-dependent | asserts-own | K=7: 2,207,022 frequent (145.3s) |
| R2-332 | F2 | 595, 610 | 5,063,845 | Base K=8 frequent | deterministic | asserts-own | K=8: 5,063,845 frequent (348.5s) |
| R2-333 | F2 | 595 | 348.5 s | Base K=8 time | hardware-dependent | asserts-own | K=8: 5,063,845 frequent (348.5s) |
| R2-334 | F2 | 596 | 743.4 s | Base K=9 time | hardware-dependent | asserts-own | K=9: 10,041,611 frequent (743.4s) |
| R2-335 | F2 | 600–601 | > 15 | expected Base K-max (prediction) | deterministic | asserts-own | At 0.1% support with 35K features, we may see K-max > 15 and total itemsets in the tens of millions. |
| R2-336 | F2 | 601, 611–612 | tens of millions | expected Base total itemsets (prediction) | deterministic | asserts-own | the complete Base run will yield tens of millions of itemsets at high $K$ values |
| R2-337 | F2 | 625 | 141 GB | H200 SXM5 VRAM (hardware section) | hardware-dependent | asserts-own | performed on 4$\times$NVIDIA H200 SXM5 141\,GB GPUs with 1.5\,TB host RAM |
| R2-338 | F2 | 625 | 1.5 TB | host RAM | hardware-dependent | asserts-own | 141\,GB GPUs with 1.5\,TB host RAM |
| R2-339 | F2 | 625 | Python 3.12 | software version | software | asserts-own | running Python~3.12, CuPy~13.4, NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-340 | F2 | 625 | CuPy 13.4 | software version | software | asserts-own | running Python~3.12, CuPy~13.4, NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-341 | F2 | 626 | NumPy 2.0 | software version | software | asserts-own | NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-342 | F2 | 626 | CUDA 12.6 | software version | software | asserts-own | NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-343 | F2 | 626 | Ubuntu 22.04 | OS version | software | asserts-own | NumPy~2.0, CUDA~12.6, and Ubuntu~22.04 |
| R2-344 | F2 | 627 | ~27.3M | proteins per GPU (row-split) | deterministic | asserts-own | The row-split architecture distributes ${\sim}27.3$M proteins per GPU |
| R2-345 | F2 | 630 | ~12 bytes | nccl.allReduce payload per K-level | deterministic | asserts-own | Local support counts are reduced across GPUs via \texttt{nccl.allReduce} with ${\sim}12$\,bytes per $K$-level. |

### F3 — `senior_review_jun01.md`

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R2-346 | F3 | 4 | 968 | lines in `et_miner_proteome.tex` (reviewed version) | deterministic | asserts-own | `/home/et/personal-projects/et-miner/papers/et_miner_proteome.tex` (968 regels) |
| R2-347 | F3 | 18, 94, 98 | 26,849,505 | kdist sum K=1..22 (recounted by hand) | deterministic | asserts-own | kdist K=1..22 = 26,849,505 exact … kdist-som = 26,849,505 (handmatig nageteld ✓) |
| R2-348 | F3 | 18, 98 | 16,812,646,639 | expanded-run cumulative itemsets (sum verified) | deterministic | asserts-own | expanded cumulatief = 16,812,646,639 exact |
| R2-349 | F3 | 19, 100 | 9× | speedup/ratio in paper (verified) | hardware-dependent | quotes-paper | speedup-ratio's (9×, 124×, 21×, 95.2%, 5.1×) kloppen binnen afronding |
| R2-350 | F3 | 19, 100 | 124× | speedup/ratio in paper (verified) | hardware-dependent | quotes-paper | speedup-ratio's (9×, 124×, 21×, 95.2%, 5.1×) kloppen binnen afronding |
| R2-351 | F3 | 19, 100 | 21× | Direct-vs-SON speedup in paper (verified) | hardware-dependent | quotes-paper | speedup-ratio's (9×, 124×, 21×, 95.2%, 5.1×) kloppen binnen afronding |
| R2-352 | F3 | 19, 101 | 95.2% | SON miss rate in paper (verified) | deterministic | quotes-paper | speedup-ratio's (9×, 124×, 21×, 95.2%, 5.1×) kloppen binnen afronding |
| R2-353 | F3 | 19, 55, 101 | 5.1× | scale ratio vs GMiner in paper (verified) | deterministic | quotes-paper | "15M" (scale-tabel, basis van de 5.1×-claim) |
| R2-354 | F3 | 31 | 37 | `\bibitem` entries (r574–757) | deterministic | asserts-own | Bibliografie: 37 `\bibitem` (r574–757). 32 distincte `\cite`-keys, allemaal resolvend |
| R2-355 | F3 | 31 | 32 | distinct `\cite` keys, all resolving | deterministic | asserts-own | 32 distincte `\cite`-keys, allemaal resolvend (geen broken refs) |
| R2-356 | F3 | 39 | 3077 → 3077–3083 | coin2009 (actually Terrapon et al.) page range (old → new) | external-fact | disputes | Pagina's ook incompleet (3077 → 3077–3083) |
| R2-357 | F3 | 40, 149 | 50(D1):D419–D427 → D439–D444 | varadi2022 page range (old → new) | external-fact | disputes | `50(D1):D419--D427` → moet **D439–D444** zijn |
| R2-358 | F3 | 40 | D412–D419 | mistry2021 page range (r712), source of the copy error | external-fact | quotes-paper | collideert exact met de echte range van `mistry2021` (`D412--D419`, r712) → klassieke copy-fout |
| R2-359 | F3 | 41 | 100M | BIGMiner transactions as stated in paper table | external-fact | disputes | Tabel zegt BIGMiner = "100M transactions / 30× servers" |
| R2-360 | F3 | 41 | 30 | BIGMiner servers/nodes as stated in paper (unverifiable) | external-fact | disputes | Het "30 nodes"-getal is niet verifieerbaar uit open bronnen. |
| R2-361 | F3 | 41 | 6.5 billion | BIGMiner transactions per the BIGMiner paper | external-fact | asserts-own | De BIGMiner-paper rapporteert schaling tot **6,5 miljard** transacties |
| R2-362 | F3 | 41 | ≈65× | 6.5B / 100M | external-fact | asserts-own | (≈65× meer dan de geclaimde 100M) |
| R2-363 | F3 | 41 | three orders of magnitude | paper's scale claim vs prior work | deterministic | disputes | verzwak je juist de claim "three orders of magnitude beyond any prior result" |
| R2-364 | F3 | 47 | ~100K | Meysman2015 structures as stated in paper (r499) | external-fact | disputes | Claim-support: "limited to ${\sim}100$K structures" — werkelijk **~32.142** structuren (overschat ~3×) |
| R2-365 | F3 | 47 | ~32,142 | Meysman2015 structures (actual) | external-fact | asserts-own | werkelijk **~32.142** structuren (overschat ~3×) |
| R2-366 | F3 | 47 | ~3× | overestimate factor | external-fact | asserts-own | (overschat ~3×) |
| R2-367 | F3 | 47 | ~32K | suggested replacement value | external-fact | requests | `${\sim}32$K structures` |
| R2-368 | F3 | 48 | 54(7):1–36 → 54(9), Article 179 (179:1–179:35) | acmsurvey2021 issue/pages (old → new) | external-fact | disputes | Issue + pagina's fout: `54(7):1--36` → **54(9), Article 179 (179:1–179:35)** |
| R2-369 | F3 | 49 | 2008 tech-report title vs DaMoN 2009 paper, pp 34–42 | fang2009 title/venue/pages | external-fact | disputes | de titel van het **2008 HKUST tech-report**; de DaMoN-2009 workshop-paper heet "**Frequent itemset mining on graphics processors**" (pp 34–42) |
| R2-370 | F3 | 50 | vol 250, Art. 123928 | chon2024 ESWA volume/article (title wrong) | external-fact | asserts-own | ESWA vol 250, Art. 123928. |
| R2-371 | F3 | 51 | 52(D1):D368 → D368–D375 | varadi2024 page range (old → new) | external-fact | disputes | Ontbrekende eind-pagina: `52(D1):D368` → **D368–D375** |
| R2-372 | F3 | 52 | 21:1507–1521 → 21(3):1507–1520 | chon2018b pages (old → new) | external-fact | disputes | Eind-pagina off-by-one: `21:1507--1521` → **21(3):1507–1520** |
| R2-373 | F3 | 53 | pp. 432–444 | savasere1995 pages (missing in paper) | external-fact | requests | Pagina's ontbreken volledig. \| `pp.\ 432--444` |
| R2-374 | F3 | 54 | 50–350× | GPU speedup range claimed in paper (r106; range verified) | external-fact | quotes-paper | voor "GPU achieved 50–350× speedups" … De range zelf klopt |
| R2-375 | F3 | 54 | 350× | djenouri2019 speedup (confirmed) | external-fact | asserts-own | (350× = djenouri2019, 100× = zhang2011, beide bevestigd) |
| R2-376 | F3 | 54 | 100× | zhang2011 speedup (confirmed) | external-fact | asserts-own | (350× = djenouri2019, 100× = zhang2011, beide bevestigd) |
| R2-377 | F3 | 54 | 2 | CPU algorithms (han2000 FP-Growth, zaki2000 Eclat) wrongly inside the GPU-speedup cite cluster | external-fact | disputes | bundelt twee **CPU-algoritmes** (`han2000` FP-Growth, `zaki2000` Eclat) onder een GPU-speedup-claim |
| R2-378 | F3 | 55 | 15M | GMiner transactions, scale table (r466) | external-fact | disputes | Zelfde systeem krijgt "15M" (scale-tabel, basis van de 5.1×-claim) én "1.7M (real)" |
| R2-379 | F3 | 55 | 1.7M | GMiner transactions, gpu-arch table (r877) | external-fact | quotes-paper | én "1.7M (real)" (gpu-arch-tabel). 1.7M = webdocs FIMI (bevestigd 1,692,082) |
| R2-380 | F3 | 55 | 1,692,082 | webdocs FIMI transaction count | external-fact | asserts-own | 1.7M = webdocs FIMI (bevestigd 1,692,082) |
| R2-381 | F3 | 55 | 4× GTX 1080 | GMiner hardware as stated in paper (unverifiable) | hardware-dependent | quotes-paper | (GPU "4× GTX 1080" niet verifieerbaar — paywall.) |
| R2-382 | F3 | 59–60 | 5 (cited 0×) | orphan bibitems: webb2007, webb2014, abramson2024, zaki1997, miettinen2020 | deterministic | asserts-own | `webb2007` (r724), `webb2014` (r729), `abramson2024` (r734), `zaki1997` (r749), `miettinen2020` (r754) — … **nergens ge-`\cite`d** |
| R2-383 | F3 | 76 | 316M | CSR nnz (r175) | deterministic | quotes-paper | Body: "316M nnz, ~5.1 GB, **two** 64-bit ints/entry" (16 B/entry, COO-paar) |
| R2-384 | F3 | 76 | ~5.1 GB | CSR/COO size (r175) | deterministic | quotes-paper | Body: "316M nnz, ~5.1 GB, **two** 64-bit ints/entry" |
| R2-385 | F3 | 76 | 2 × 64-bit ints = 16 B/entry | COO entry size (r175) | deterministic | quotes-paper | "**two** 64-bit ints/entry" (16 B/entry, COO-paar) |
| R2-386 | F3 | 76 | ~3 GB | H2D transfer size (r179/r485) | deterministic | quotes-paper | Elders: H2D-transfer "~3 GB" (alleen kolom-index, 8 B → 316M×8=2.53 GB) |
| R2-387 | F3 | 76 | 8 B/entry | CSR column-index entry size (appendix r920) | deterministic | quotes-paper | Appendix-tabel: "CSR (**8** bytes/entry)" → 19 GB voor 214M |
| R2-388 | F3 | 76 | 2.53 GB | 316M × 8 B | deterministic | asserts-own | (alleen kolom-index, 8 B → 316M×8=2.53 GB) |
| R2-389 | F3 | 76, 152 | 19 GB | CSR size for 214M proteins (appendix r920) | deterministic | quotes-paper | Appendix-tabel: "CSR (**8** bytes/entry)" → 19 GB voor 214M |
| R2-390 | F3 | 76, 152 | 3 (3 / 5.1 / 19 GB) | distinct CSR sizes given without labels | deterministic | disputes | Drie cijfers (3 / 5.1 / 19 GB) voor "CSR" zonder dat de lezer ze kan rijmen. |
| R2-391 | F3 | 76 | nnz of the 214M set | value missing from paper | deterministic | requests | geef de nnz van de 214M-set (nu nergens vermeld) |
| R2-392 | F3 | 77 | 40× | memory reduction (r175) | deterministic | quotes-paper | 40× = naïeve **byte**-matrix (206 GB, 1 B/boolean) ÷ CSR-COO (5.1 GB) |
| R2-393 | F3 | 77 | 1.4× | memory reduction (r908/r920) | deterministic | quotes-paper | 1.4× = **bitpacked** dense (27 GB) ÷ CSR (19 GB) |
| R2-394 | F3 | 77, 152 | 206 GB | naïve byte matrix (1 B/boolean) | deterministic | quotes-paper | naïeve **byte**-matrix (206 GB, 1 B/boolean) |
| R2-395 | F3 | 77 | 1 B/boolean | byte-matrix convention | deterministic | asserts-own | 206 GB = 1 byte/boolean (niet bitpacked) |
| R2-396 | F3 | 77 | 27 GB | bitpacked dense (appendix) | deterministic | quotes-paper | **bitpacked** dense (27 GB) ÷ CSR (19 GB) |
| R2-397 | F3 | 77, 152 | 26 GB | actual on-GPU bitvector | deterministic | asserts-own | de echte on-GPU bitvector is 26 GB = de "27 GB" appendix-dense, factor ~8 kleiner |
| R2-398 | F3 | 77 | ~8× | 206 GB / 26 GB | deterministic | asserts-own | factor ~8 kleiner |
| R2-399 | F3 | 78 | ~445 GB | expanded bitvector size (r258) | deterministic | disputes | **`~445 GB` expanded bitvector: GB/GiB-mix** |
| R2-400 | F3 | 78 | ~56 GB/device | expanded per-device bitvector (paper) | deterministic | quotes-paper | "~56 GB/device × 8" = 448, niet 445 |
| R2-401 | F3 | 78, 92 | 8 | devices in the expanded run (r258 "× 8"; r965 "× 8") | hardware-dependent | quotes-paper | "~56 GB/device × 8" = 448 … "$332 ≈ $3.50/GPU/uur × 8 × ~12h" |
| R2-402 | F3 | 78 | 448 GB | 56 × 8 | deterministic | asserts-own | "~56 GB/device × 8" = 448, niet 445 |
| R2-403 | F3 | 78 | 478 GB | 109.2M × 35012 / 8 (decimal GB) | deterministic | asserts-own | Decimaal 109.2M×35012/8 = **478 GB** |
| R2-404 | F3 | 78 | 109.2M | expanded protein count | deterministic | quotes-paper | Decimaal 109.2M×35012/8 = **478 GB** |
| R2-405 | F3 | 78 | 35012 | expanded feature count | deterministic | quotes-paper | Decimaal 109.2M×35012/8 = **478 GB** |
| R2-406 | F3 | 78 | 445.19 GiB | same size in GiB | deterministic | asserts-own | 445 klopt **alleen** als **GiB** (445.19 GiB) |
| R2-407 | F3 | 78 | 55.6 GiB | 56 GB per device in GiB | deterministic | asserts-own | als GiB, dan is "56 GB/device" ook fout (→ 55.6 GiB) |
| R2-408 | F3 | 84, 142, 153 | p < 0.17 | one-sided binomial p for 0/5 permutations > K=6 (r414, r429, r435) | deterministic | disputes | "$p < 0.17$ by one-sided binomial test" … **Niet onderbouwd door de standaard one-sided binomiale upper bound.** |
| R2-409 | F3 | 84 | 0/5 | permutations exceeding K=6 | deterministic | quotes-paper | (0/5 permutaties > K=6) |
| R2-410 | F3 | 84 | K=6 | 1K null-model ceiling | deterministic | quotes-paper | (0/5 permutaties > K=6) |
| R2-411 | F3 | 84, 153 | ≈0.45 | 95% upper bound for 0 successes in 5 trials | deterministic | asserts-own | Voor 0 successen in 5 trials is de 95%-upper-bound ≈ **0.45** |
| R2-412 | F3 | 84 | 3/5 = 0.6 | rule-of-three bound | deterministic | asserts-own | (rule-of-three 3/5 = 0.6) |
| R2-413 | F3 | 84 | 1/6 | 0.17 ≈ 1/6, origin unclear | deterministic | asserts-own | 0.17 ≈ 1/6 — herkomst onduidelijk en oogt te optimistisch. |
| R2-414 | F3 | 90 | 1,092 | expanded-run min_count (r260) | method-parameter | disputes | min_count expanded = 1,092 \| 260 \| ⌈0.001%×109,224,173⌉ = ⌈1092.24⌉ = **1093** |
| R2-415 | F3 | 90 | 109,224,173 | expanded-run n (exact) | deterministic | quotes-paper | ⌈0.001%×109,224,173⌉ = ⌈1092.24⌉ = **1093** |
| R2-416 | F3 | 90 | 1092.24 | 0.001% × 109,224,173 | deterministic | asserts-own | ⌈0.001%×109,224,173⌉ = ⌈1092.24⌉ = **1093** |
| R2-417 | F3 | 90 | 1,092 → 1093 | min_count fix (old → new) if ceil is used | method-parameter | requests | 1092→1093, óf vermeld dat round() gebruikt is en pas r800 aan. |
| R2-418 | F3 | 90 | 769 | Power min_count (ceil applied) | method-parameter | quotes-paper | Power gebruikt wél ceil (769). Richting inconsistent. |
| R2-419 | F3 | 91 | $243 | saving stated in paper (r965) | hardware-dependent | disputes | "$243 saving" \| 965 \| 332 − 90 = **242**, niet 243. |
| R2-420 | F3 | 91, 92 | $332 | run cost stated in paper | hardware-dependent | quotes-paper | 332 − 90 = **242** … "$332 ≈ $3.50/GPU/uur × 8 × ~12h" |
| R2-421 | F3 | 91 | $90 | comparison cost stated in paper | hardware-dependent | quotes-paper | 332 − 90 = **242**, niet 243. |
| R2-422 | F3 | 91 | $243 → $242 | saving (old → new) | hardware-dependent | requests | `\$242` |
| R2-423 | F3 | 92 | $3.50/GPU/hour | GPU rental price used in paper (r940) | external-fact | quotes-paper | "$332 ≈ $3.50/GPU/uur × 8 × ~12h" |
| R2-424 | F3 | 92 | ~12 h | run duration used in paper cost formula | hardware-dependent | quotes-paper | "$332 ≈ $3.50/GPU/uur × 8 × ~12h" \| 965 vs 940 |
| R2-425 | F3 | 92 | $336 | 3.50 × 8 × 12 | hardware-dependent | asserts-own | 3.50×8×12 = **$336**; $332 impliceert 11.86 h (≠ "~12 hours") |
| R2-426 | F3 | 92 | 11.86 h | hours implied by $332 | hardware-dependent | asserts-own | $332 impliceert 11.86 h (≠ "~12 hours") |
| R2-427 | F3 | 93, 99 | 62.6% | excluded single-feature proteins (Limitation 4, r525; base run) | deterministic | quotes-paper | "excludes ... (62.6%)" (Limitation 4) \| 525 \| Getal correct voor de **base**-run (37.4+62.6=100) |
| R2-428 | F3 | 93, 99 | 37.4% | multi-feature share (base run) | deterministic | quotes-paper | Getal correct voor de **base**-run (37.4+62.6=100) |
| R2-429 | F3 | 93, 99 | 53.1% | expanded-run coverage claim | deterministic | quotes-paper | spanning met de 53.1%-expanded-claim |
| R2-430 | F3 | 94 | 26.8M | total itemsets as written (r301, r306) | deterministic | disputes | "26.8M total" \| 301, 306 \| 26,849,505 → 26.85M; "26.8M" is **truncatie** i.p.v. afronding |
| R2-431 | F3 | 94 | 26.85M | correct 2-decimal rounding | deterministic | asserts-own | 26,849,505 → 26.85M |
| R2-432 | F3 | 94 | 26.9M | correct 1-decimal rounding | deterministic | asserts-own | "26.8M" is **truncatie** i.p.v. afronding (zou 26.9M zijn) |
| R2-433 | F3 | 99 | 7 | growth factors in paper (all verified) | deterministic | quotes-paper | alle 7 groeifactoren (25.8×…3.4×) ✓ |
| R2-434 | F3 | 99 | 25.8× | largest growth factor | deterministic | quotes-paper | alle 7 groeifactoren (25.8×…3.4×) ✓ |
| R2-435 | F3 | 99 | 3.4× | smallest growth factor | deterministic | quotes-paper | alle 7 groeifactoren (25.8×…3.4×) ✓ |
| R2-436 | F3 | 99 | 76,891 | largest min_count (base run) | method-parameter | quotes-paper | alle min_counts (76,891 … 8) ✓ |
| R2-437 | F3 | 99 | 8 | smallest min_count | method-parameter | quotes-paper | alle min_counts (76,891 … 8) ✓ |
| R2-438 | F3 | 100 | 13.14% | percentage in paper (quantity unspecified; verified) | deterministic | quotes-paper | 37.4% / 53.1% / 13.14% ✓ |
| R2-439 | F3 | 100 | Z > 3,700 for K=4–6 | Z-scores, abstract↔table↔conclusion | deterministic | quotes-paper | Z>3,700 voor K=4–6 consistent abstract↔tabel↔conclusie ✓ |
| R2-440 | F3 | 101 | 475,865 | null-table Bio column total = Direct-GPU total | deterministic | quotes-paper | null Bio-kolom = 475,865 = Direct-GPU-totaal ✓ |
| R2-441 | F3 | 101 | 0.32 | pruning factor, technique 1 | hardware-dependent | quotes-paper | pruning 1−(0.32)(0.88)(0.96) = 73% ✓ |
| R2-442 | F3 | 101 | 0.88 | pruning factor, technique 2 | hardware-dependent | quotes-paper | pruning 1−(0.32)(0.88)(0.96) = 73% ✓ |
| R2-443 | F3 | 101 | 0.96 | pruning factor, technique 3 | hardware-dependent | quotes-paper | pruning 1−(0.32)(0.88)(0.96) = 73% ✓ |
| R2-444 | F3 | 102, 115 | 73% | combined projected K=9 runtime reduction | hardware-dependent | quotes-paper | pruning 1−(0.32)(0.88)(0.96) = 73% ✓ … "are estimated to reduce $K{=}9$ runtime by approximately 73%" |
| R2-445 | F3 | 102 | ~264 B | total allreduce transfer (22 × 12) | deterministic | quotes-paper | ~264 B totaal transfer (22×12) ✓ |
| R2-446 | F3 | 102 | 22 | K-levels in the transfer calculation | deterministic | quotes-paper | ~264 B totaal transfer (22×12) ✓ |
| R2-447 | F3 | 102, 127 | 12 bytes | per-K-level allreduce transfer (r842/r873) | deterministic | quotes-paper | `$\sim$12 bytes` (873) vs `${\sim}12$~bytes` (842) |
| R2-448 | F3 | 109 | 70–90 / 583–589 / K=4–8 | en-dash range examples (pLDDT bin, citation pages, K range) | external-fact | quotes-paper | en-dash-ranges (`70--90`, `583--589`, `$K{=}4$--$8$`) zijn consistent en correct toegepast |
| R2-449 | F3 | 115 | K=9 | level of the projected runtime reduction (r92, r549) | deterministic | quotes-paper | "we estimate would reduce** $K{=}9$ runtime …" |
| R2-450 | F3 | 121 | 40+ | occurrences of "ET-miner" (vs one "ET-Miner", r795) | deterministic | asserts-own | enige uitschieter tussen 40+ `ET-miner` |
| R2-451 | F3 | 123 | 7.3 minutes | run time (r121, r92) | hardware-dependent | quotes-paper | r121 `7.3~minutes` (beschermd), r92/r130/r549 plain spatie ("7.3 minutes", "63 minutes", "4.3 hours") |
| R2-452 | F3 | 123 | 63 minutes | run time (r130) | hardware-dependent | quotes-paper | ("7.3 minutes", "63 minutes", "4.3 hours") |
| R2-453 | F3 | 123, 143 | 4.3 hours | run time (r549; expanded 35K run per l.143) | hardware-dependent | quotes-paper | ("7.3 minutes", "63 minutes", "4.3 hours") … **16,8 miljard itemsets t/m K=8, 4.3 uur** |
| R2-454 | F3 | 123 | 80 GB | house-style unit example (`80\,GB`) | hardware-dependent | quotes-paper | Huis-stijl is `\,` (80\,GB). |
| R2-455 | F3 | 125 | Opus 4.6 | model name in author info (r559 "Opus4.6") | software | quotes-paper | **`Opus4.6`** mist spatie (author-info). \| `Opus 4.6` |
| R2-456 | F3 | 138, 149 | 247/752/3 → 500/500/2 | Table 1 breakdown fixed between versions (r143–150) | deterministic | asserts-own | Table 1 feature-breakdown "fabricated" (247/752/3) \| ✅ **GEFIXT** — toont nu 500/500/2 (r143–150). |
| R2-457 | F3 | 140 | 2 | pLDDT bins, Run 1 (current version, r150) | deterministic | quotes-paper | pLDDT-bins 3 vs 2 \| ✅ **GEFIXT** — Run 1 = 2 bins (r150). |
| R2-458 | F3 | 143 | 606,292 | 35K @0.001% itemsets per revision_notes_b3 | deterministic | quotes-paper | `revision_notes_b3.tex` noteerde voor de 35K-run @ 0.001%: **606,292 itemsets, K_max=17, 654 s** |
| R2-459 | F3 | 143 | K_max=17 | 35K @0.001% max K per revision_notes_b3 | deterministic | quotes-paper | **606,292 itemsets, K_max=17, 654 s** |
| R2-460 | F3 | 143 | 654 s | 35K @0.001% time per revision_notes_b3 | hardware-dependent | quotes-paper | **606,292 itemsets, K_max=17, 654 s** |
| R2-461 | F3 | 143 | 16.8 billion | 35K @0.001% itemsets through K=8 (current paper) | deterministic | quotes-paper | De huidige paper claimt voor dezelfde run **16,8 miljard itemsets t/m K=8, 4.3 uur** |
| R2-462 | F3 | 143 | K=8 | depth reached in current paper's 35K run | deterministic | quotes-paper | **16,8 miljard itemsets t/m K=8, 4.3 uur** |
| R2-463 | F3 | 143 | ~27,000× | ratio between the two 35K results | deterministic | asserts-own | Dat is ~27.000× verschil — vrijwel zeker het effect van de eerder genoemde **NCCL int32-overflow bugfix** |
| R2-464 | F3 | 143 | int32 | NCCL overflow bug (integer width) | software | asserts-own | het effect van de eerder genoemde **NCCL int32-overflow bugfix** (pre-fix undercount) |

### F4 — `senior_review_mar23.md`

| ID | file | line | value | unit | category | stance | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|---|
| R2-465 | F4 | 15, 20, 22 | 247 | Pfam domains, Run 1 (paper Table 1, lines 143–145) | deterministic | disputes | Pfam domains & InterPro/Pfam & 247 & --- \\ … The numbers 247 and 752 do not correspond to any configuration in the codebase. |
| R2-466 | F4 | 16, 20, 22 | 752 | GO terms, Run 1 (paper Table 1) | deterministic | disputes | GO terms & Gene Ontology & 752 & 5,763 \\ |
| R2-467 | F4 | 16 | 5,763 | GO terms, Run 2 (paper Table 1) | deterministic | quotes-paper | GO terms & Gene Ontology & 752 & 5,763 \\ |
| R2-468 | F4 | 17, 20, 70 | 3 | pLDDT bins, Run 1 (paper Table 1) | deterministic | disputes | pLDDT confidence bins & AlphaFold & 3 & 6 \\ … Table 1 claims 3 pLDDT bins for Run 1. |
| R2-469 | F4 | 17, 70 | 6 | pLDDT bins, Run 2 / defined in item mapping | deterministic | quotes-paper | pLDDT confidence bins & AlphaFold & 3 & 6 \\ … The item mapping defines 6 bins but 4 have < 8 proteins. |
| R2-470 | F4 | 20, 22 | 1,002 | Run 1 feature total | deterministic | quotes-paper | The paper claims Run 1 consists of 247 Pfam + 752 GO + 3 pLDDT = 1,002. |
| R2-471 | F4 | 22, 89 | 500 | Pfam domains, actual (item_mapping_214m.parquet) | deterministic | asserts-own | **The actual composition is 500 Pfam + 500 GO + 2 pLDDT = 1,002** |
| R2-472 | F4 | 22, 89 | 500 | GO terms, actual | deterministic | asserts-own | **The actual composition is 500 Pfam + 500 GO + 2 pLDDT = 1,002** |
| R2-473 | F4 | 22, 89 | 2 | pLDDT bins, actual | deterministic | asserts-own | **The actual composition is 500 Pfam + 500 GO + 2 pLDDT = 1,002** |
| R2-474 | F4 | 22 | top_pfam=500, top_go=500 | pipeline parameters (non-default) | method-parameter | asserts-own | Pipeline was run with `top_pfam=500, top_go=500` (non-default parameters). |
| R2-475 | F4 | 27, 29 | 769 vs 8 | min_count null vs main (paper line 438 quote) | method-parameter | quotes-paper | run at a higher threshold than the main analysis (min_count=769 vs. 8) for computational efficiency |
| R2-476 | F4 | 29 | 8 | min_count requested for null-model rerun | method-parameter | requests | Fix: either run the null model at min_count=8, or honestly state that K=15-22 significance is extrapolated. |
| R2-477 | F4 | 29 | K=15–22 | depth range whose significance is extrapolated | deterministic | disputes | or honestly state that K=15-22 significance is extrapolated |
| R2-478 | F4 | 42 | 68% | pruning saving, technique 1 (Table 8, K=9) — projection | hardware-dependent | disputes | Table 8 presents pruning savings of 68%, 12%, 4%, 2% (combined 73%) for K=9. PROJECT_STATE confirms these are projections |
| R2-479 | F4 | 42 | 12% | pruning saving, technique 2 — projection | hardware-dependent | disputes | pruning savings of 68%, 12%, 4%, 2% (combined 73%) for K=9 |
| R2-480 | F4 | 42 | 4% | pruning saving, technique 3 — projection | hardware-dependent | disputes | pruning savings of 68%, 12%, 4%, 2% (combined 73%) for K=9 |
| R2-481 | F4 | 42 | 2% | pruning saving, technique 4 — projection | hardware-dependent | disputes | pruning savings of 68%, 12%, 4%, 2% (combined 73%) for K=9 |
| R2-482 | F4 | 42 | 73% | combined projected K=9 runtime reduction (abstract, unqualified) | hardware-dependent | disputes | The abstract says "reduce projected K=9 runtime by 73%" without qualifying this as an estimate. |
| R2-483 | F4 | 42 | K=9 | level of the pruning projection | deterministic | quotes-paper | (combined 73%) for K=9 |
| R2-484 | F4 | 50 | 12,773 | InterPro entries, Run 2 (replacing Pfam) | deterministic | quotes-paper | Table 1 shows Run 2 has "---" for Pfam domains, replaced by 12,773 InterPro entries. |
| R2-485 | F4 | 52, 54, 56 | 769× | more transactions than next largest GPU system (paper line 456) | deterministic | disputes | "ET-miner processes 769x more transactions than the next largest GPU system." … The 769× appears to compare against CPU systems at 100K transactions. |
| R2-486 | F4 | 56 | 100K | transactions of the CPU systems the 769× apparently compares to | external-fact | quotes-paper | The 769× appears to compare against CPU systems at 100K transactions. |
| R2-487 | F4 | 56 | 100M | BIGMiner transactions (paper table) | external-fact | quotes-paper | BIGMiner processes 100M transactions — MORE than ET-miner's 76.9M. |
| R2-488 | F4 | 56 | 76.9M | ET-miner transactions | deterministic | quotes-paper | BIGMiner processes 100M transactions — MORE than ET-miner's 76.9M. |
| R2-489 | F4 | 56 | 15M | GMiner transactions (paper table) | external-fact | quotes-paper | GMiner is listed at 15M (76.9M/15M = 5.1×, not 769×). |
| R2-490 | F4 | 56 | 5.1× | 76.9M / 15M | deterministic | asserts-own | GMiner is listed at 15M (76.9M/15M = 5.1×, not 769×). |
| R2-491 | F4 | 66 | 5 | permutations in null model (paper line 410) | method-parameter | quotes-paper | The null model uses only 5 permutations (line 410). |
| R2-492 | F4 | 66 | 0.17 | lowest achievable bound on P(K≥7 given null) with 5 trials | deterministic | asserts-own | Cannot bound P(K≥7\|null) below 0.17 with only 5 trials. |
| R2-493 | F4 | 66 | +∞ | Z-scores for K≥7 as reported (paper) | deterministic | disputes | Z-scores for K≥7 reported as "+∞" — misleading with zero observations. |
| R2-494 | F4 | 66 | 0 out of 5 | trials producing K≥7 | deterministic | asserts-own | Can only say "0 out of 5 trials produced K≥7." |
| R2-495 | F4 | 70 | 2 | pLDDT bins frequent at min_count=8 (plddt_mean_med, plddt_mean_high) | deterministic | asserts-own | Only 2 are frequent at min_count=8: `plddt_mean_med` and `plddt_mean_high`. |
| R2-496 | F4 | 70 | 4 | defined bins with < 8 proteins | deterministic | asserts-own | The item mapping defines 6 bins but 4 have < 8 proteins. |
| R2-497 | F4 | 70 | < 8 | proteins in each of the 4 infrequent bins | deterministic | asserts-own | The item mapping defines 6 bins but 4 have < 8 proteins. |
| R2-498 | F4 | 74 | K=1 through K=8 | Run 2 levels reported from the pre-fix (NCCL int32 bug) run | deterministic | disputes | K=1 through K=8 numbers for Run 2 come from a run with the NCCL int32 bug present |
| R2-499 | F4 | 74 | f8c56cb5 | commit fixing the NCCL int32 bug | software | asserts-own | (fixed later in commit f8c56cb5) |
| R2-500 | F4 | 74 | < 2^31 | per-GPU counts (PROJECT_STATE's correctness argument, not re-verified) | software | disputes | PROJECT_STATE argues correctness because per-GPU counts < 2^31, but this was not independently verified by re-running. |
| R2-501 | F4 | 80 | 6 (0 hallucinated) | citations spot-checked: Jumper 2021, Agrawal 1994, Han 2000, Chon 2018, Abramson 2024, Naulaerts 2015 | external-fact | asserts-own | Spot-checked sample: Jumper 2021, Agrawal 1994, Han 2000, Chon 2018, Abramson 2024, Naulaerts 2015 — all verified correct. |
| R2-502 | F4 | 86 | 16.8 billion | itemsets (Run 2 / expanded) | deterministic | quotes-paper | 16.8 billion itemsets across 109M proteins is a real achievement. |
| R2-503 | F4 | 86 | 109M | proteins (Run 2 / expanded) | deterministic | quotes-paper | 16.8 billion itemsets across 109M proteins is a real achievement. |
| R2-504 | F4 | 86 | K=22 | the K=22 finding | deterministic | quotes-paper | The K=22 finding is compelling. |
| R2-505 | F4 | 89 | 500/500/2 | required Table 1 fix | deterministic | requests | 1. Fix Table 1: 500/500/2 |

---

## SECTION B — FLAGGED INCONSISTENCIES

### B.1 — Stated by the texts themselves

| ID | file | line | what is claimed inconsistent | the values/locations involved |
|---|---|---|---|---|
| B-01 | F1 | 44, 185–200, 295, 304 | "A fortiori" argument is logically backwards: a null model run at a stricter threshold cannot validate the more permissive main run | null min_count=769 (0.001%) vs main min_count=8 (0.00001%); null K_max=6; real K_max=14 @0.001% vs 22 @0.00001%; paper line 396 |
| B-02 | F1 | 173–183, 294, 310 | 5 permutations cannot bound P(K≥7 given null); Z=inf and p=0 are division-by-zero artifacts | 0 events / 5 trials → Clopper-Pearson 95% upper bound 0.451; would need 100 (0.030) or 1000 (0.003) permutations |
| B-03 | F1 | 198, 304, 308 | K=15–22 patterns are covered by no null test; should be labelled exploratory or tested at min_count=8 | K=14 (0.001%) vs K=22 (0.00001%) |
| B-04 | F1 | 127 | Two different speedup figures for the same comparison in the paper | "21x" (caption) vs "21.4x" (controlled comparison text) |
| B-05 | F1 | 164–171 | Paper null-table values are rounded relative to the JSON ("minor rounding in paper table") | 63,702 vs 63,702.4; 25,468 vs 25,467.8; 22 vs 21.8; −987 vs −987.08; +3,791 vs 3790.74; +71,728 vs 71728.1 |
| B-06 | F1 | 228–237, 254, 312 | GO:0005524 is auto-derived from PF00270 via InterPro2GO but the paper does not note it; GO evidence codes absent from item mapping (method detail missing) | independent features 22 → 21; 1 of 19 GO terms |
| B-07 | F1 | 262–279, 302 | Table 1 feature breakdown contradicts the item-mapping parquet ("fabricated or refers to a different version of the data") | paper 247 Pfam / 302 MF / 289 BP / 161 CC / 752 GO / 3 pLDDT vs parquet 500 / – / – / – / 500 / 2; total 1,002 in both |
| B-08 | F1 | 276–277 | pLDDT bin count: paper says 3, only 2 are frequent; mapping has 1,006 entries but 1,002 frequent | 3 vs 2; 1,006 vs 1,002 (4 infrequent: plddt_mean_low + 3 plddt_fraction; plddt_mean_low < 8 proteins) |
| B-09 | F1 | 67–75 | "Dedup affected 20.45%" (PROJECT_STATE.md) is not in the paper and describes null-shuffle duplicates, not the source-data split | 20.45% vs 62.61% single-feature / 37.39% multi-feature |
| B-10 | F1 | 26–42 | Two different min_count values for the same nominal 0.001% support in the two experiment JSONs (stated, not flagged) | 768 (direct-vs-SON) vs 769 (null model) |
| B-11 | F2 | 21–23, 39, 60–64 | Replicates of the same 35K run disagree in itemset count and time, attributed to "GPU floating-point non-determinism in row-split reduction" | 606,292 ± 28 itemsets; 654.3 ± 6.7 s; deterministic only at K≤3 and K≥11 |
| B-12 | F2 | 309–366, 643 | Maximum depth decreased between vocabularies ("counterintuitive"); three mechanisms proposed | K=22 (1,002 features, min_count 8) → K=17 (35,012 features, min_count 1,093); 137× threshold ratio |
| B-13 | F2 | 362 | K-distribution peak shifted between vocabularies | K=9 (1K) → K=7 (35K) |
| B-14 | F2 | 413–416, 646 | Null-model ceiling changed between vocabularies | K=6 (22 itemsets mean, 1K) → K=5 (1 itemset per permutation, 35K); zero at K≥6 |
| B-15 | F2 | 420–426 | K=3 flips from depleted to enriched between analyses | Z=−143 (1K) → Z=+723 (35K); 3.3× more triples than chance |
| B-16 | F2 | 429–430 | Comparison sentence omits the 1K ratio it compares against (number missing) | "5.5× … (606K vs. 111K), compared to the 1K ratio at the same support threshold" — 1K ratio never given |
| B-17 | F2 | 547–576 | Speedup wording to be changed in three paper locations (value changed between versions) | 21× → 21.4× in Sec 3.1, Table 2 footnote (adds 95.2% miss rate) and Discussion 4.1 |
| B-18 | F2 | 565–566 | Table 2 † footnote compares runs differing in both method and threshold | Power (0.001%) vs Blitz (0.0001%) |
| B-19 | F2 | 127–178 | Vocabulary/dataset values changed between paper versions | 1,002 → 35,012 features; 247 → 12,773 domains; 752 → 5,763 GO; 3 → 6 pLDDT bins; 76.9M → 109.2M proteins |
| B-20 | F2 | 370–374, 434–457 | Null model to be replaced: 5-permutation (1K) by 2-permutation (35K), 100-permutation still TBD | 5 → 2 → 100 permutations |
| B-21 | F2 | 37–42, 47–49, 653–658 | Five of six campaign rows are TBD; Base run stopped at timeout | Base: ≥10 K, 10,041,611 @K=9, 12,491,079 @K=10 started; Super/Blitz/Ultra/Opus all TBD |
| B-22 | F2 | 383 | Two different totals for the same 2-permutation null run | 1,310 s GPU time vs 1,456 s wall-clock |
| B-23 | F2 | 210–235 | SON declared physically infeasible at 35K: chunk bitvector exceeds VRAM; empirical run hung | 175 GB (40M chunk) > 143 GB; 131 GB (30M chunk) → 4 passes; hung > 22 h at Pass 1 |
| B-24 | F3 | 39 | coin2009: wrong first author (Terrapon), incomplete pages | 3077 → 3077–3083 |
| B-25 | F3 | 40 | varadi2022 page range wrong; collides with mistry2021 (copy error) | D419–D427 → D439–D444; mistry2021 = D412–D419 |
| B-26 | F3 | 41 | BIGMiner scale misreported; "30 nodes" unverifiable; undermines "three orders of magnitude" framing | 100M (paper) vs 6.5 billion (BIGMiner paper), ≈65× |
| B-27 | F3 | 47 | meysman2015 structure count overstated ~3× | ~100K (paper) vs ~32,142 |
| B-28 | F3 | 48 | acmsurvey2021 issue/pages wrong | 54(7):1–36 → 54(9), Article 179 (179:1–179:35) |
| B-29 | F3 | 49 | fang2009 bib title is the 2008 tech-report title, not the DaMoN 2009 paper | pp 34–42 |
| B-30 | F3 | 50 | chon2024 title paraphrased wrongly | ESWA vol 250, Art. 123928 |
| B-31 | F3 | 51 | varadi2024 end page missing | D368 → D368–D375 |
| B-32 | F3 | 52 | chon2018b end page off by one; issue missing | 21:1507–1521 → 21(3):1507–1520 |
| B-33 | F3 | 53 | savasere1995 pages missing | → pp. 432–444 |
| B-34 | F3 | 54 | GPU-speedup citation cluster bundles two CPU algorithms (han2000, zaki2000) | 50–350× (350× djenouri2019, 100× zhang2011 confirmed) |
| B-35 | F3 | 55 | GMiner transaction count differs between two paper tables; hardware unverifiable | 15M (r466, basis of 5.1×) vs 1.7M (r877; webdocs FIMI = 1,692,082); "4× GTX 1080" |
| B-36 | F3 | 57–63 | Five bibitems never cited | webb2007, webb2014, abramson2024, zaki1997, miettinen2020 (0×) |
| B-37 | F3 | 76 | CSR given three sizes under three unlabelled byte conventions; nnz of the 214M set never stated | 316M nnz: ~5.1 GB (16 B/entry, r175) / ~3 GB (r179, r485; 316M × 8 B = 2.53 GB) / 19 GB for 214M (8 B/entry, r920) |
| B-38 | F3 | 77 | "40×" and "1.4×" memory reductions silently compare different representations | 40× = 206 GB (1 B/boolean) ÷ 5.1 GB; 1.4× = 27 GB bitpacked ÷ 19 GB; real on-GPU bitvector 26 GB (~8× below 206) |
| B-39 | F3 | 78 | Expanded bitvector size inconsistent through a GB/GiB mix | ~445 GB (r258) vs 56 GB × 8 = 448 vs 478 GB decimal; 445.19 GiB; 56 GB/device → 55.6 GiB |
| B-40 | F3 | 84, 142, 153 | "p < 0.17 by one-sided binomial test" not supported for 0/5; origin unclear | 0.17 (≈1/6) vs ≈0.45 (rule of three 3/5 = 0.6) |
| B-41 | F3 | 90 | Expanded min_count uses round() while pseudocode (r800, ⌈σ·n⌉) and the Power run (769) use ceil | 1,092 vs ⌈1092.24⌉ = 1093 (n = 109,224,173) |
| B-42 | F3 | 91 | Cost-saving arithmetic off by one dollar | $243 vs 332 − 90 = $242 |
| B-43 | F3 | 92 | Cost formula does not reproduce the stated cost | $332 vs 3.50 × 8 × 12 = $336 (implies 11.86 h, not ~12 h) |
| B-44 | F3 | 93 | 62.6% (Limitation 4) is the base-run figure, unlabelled; tension with expanded coverage claim | 37.4% + 62.6% = 100% vs 53.1% (expanded) |
| B-45 | F3 | 94 | "26.8M" is a truncation, not a rounding | 26,849,505 → 26.85M / 26.9M |
| B-46 | F3 | 115, 150 | Abstract sentence grammatically broken while conclusion (r549) states the same claim correctly | "we estimate would reduce K=9 runtime" vs "are estimated to reduce K=9 runtime by approximately 73%" |
| B-47 | F3 | 123 | Unit spacing for written-out times inconsistent with house style | 7.3 minutes / 63 minutes / 4.3 hours vs `80\,GB` |
| B-48 | F3 | 126 | Same pruning technique named differently in abstract and appendix | "warp-cooperative" (r92) vs "Warp-level" (r951) |
| B-49 | F3 | 127 | "~12 bytes" typeset two ways | `$\sim$12 bytes` (r873) vs `${\sim}12$~bytes` (r842) |
| B-50 | F3 | 136–141 | Earlier-flagged issues confirmed fixed in the current version (values changed between versions) | Table 1 247/752/3 → 500/500/2 (r143–150); "a fortiori" removed (r437); pLDDT 3 → 2 (r150); "GPU-resident" → "GPU-accelerated" |
| B-51 | F3 | 142 | 5-permutation weakness only partly resolved | p<0.17 claim still unsupported (see B-40) |
| B-52 | F3 | 143 | 35K @0.001% result differs ~27,000× between revision_notes_b3 and the current paper; attributed to the NCCL int32-overflow fix; needs confirmation that 16.8B is canonical | 606,292 itemsets / K_max=17 / 654 s (B3) vs 16.8 billion through K=8 / 4.3 h (paper; cumulative 16,812,646,639 internally consistent) |
| B-53 | F4 | 11–22, 86, 89 | Table 1 Run-1 breakdown "fabricated"; flagged 2026-02-20 and never corrected | 247/752/3 vs 500/500/2 (top_pfam=500, top_go=500); total 1,002 |
| B-54 | F4 | 24–29, 91 | "A fortiori" argument logically inverted | min_count 769 (null) vs 8 (main); K=15–22 significance extrapolated |
| B-55 | F4 | 31–38, 90 | "GPU-resident" / "all operations entirely on-GPU" contradicts Council of Copii finding: candidate generation runs on CPU (`gpu_dispatch.py:128-131`; `build_prefix_groups_gpu` not connected) | paper lines 162, 189; should read "GPU-accelerated" |
| B-56 | F4 | 40–42, 92 | Table 8 pruning savings are projections, never measured; abstract does not qualify them | 68% / 12% / 4% / 2% (73% combined, K=9) |
| B-57 | F4 | 48–50, 93 | Run 1 vs Run 2 feature semantics differ (Pfam vs InterPro) yet K-distributions compared side-by-side (Figure 3) | 247 Pfam (Run 1) vs 12,773 InterPro (Run 2, "---" for Pfam) |
| B-58 | F4 | 52–56 | 769× scale claim arithmetic unclear; BIGMiner is larger than ET-miner | 769× (only vs 100K CPU systems); BIGMiner 100M > 76.9M; GMiner 15M → 5.1× |
| B-59 | F4 | 58–62 | "closed itemset pruning" is post-processing (`apriori.py:235-268`, `prune_equal_support`), not in-mining pruning | paper line 528 |
| B-60 | F4 | 64–66, 94 | 5 permutations statistically weak; "+∞" Z-scores misleading with zero observations | cannot bound P(K≥7 given null) below 0.17; 0 of 5 trials |
| B-61 | F4 | 68–70 | pLDDT bin count discrepancy | 3 claimed vs 2 frequent at min_count=8 (6 defined, 4 with < 8 proteins) |
| B-62 | F4 | 72–74 | Run 2 K=1–8 numbers come from a run with the NCCL int32 bug (fixed in f8c56cb5), never re-verified by re-running | "per-GPU counts < 2^31" argument only |

### B.2 — Extractor-observed cross-file / intra-file disagreements (NOT asserted by any of the four texts; listed for the audit, not judged)

| ID | files / lines | observation | values |
|---|---|---|---|
| BX-01 | F1 179 vs F4 66 vs F3 84 | Three different statements of the bound on P(K≥7 given null) for 0/5 permutations | 0.451 (F1, Clopper-Pearson 95%) vs 0.17 (F4, "cannot bound below") vs "0.17 unsupported, ≈0.45" (F3) |
| BX-02 | F2 109–115, 134, 139, 175, 326 vs F1 262–279, F3 138, F4 11–22 | F2 (dated 2026-02-21) still uses the 1K breakdown that F1 (2026-02-20) had disputed and F3/F4 confirm as wrong | 247 / 752 / 3 (F2) vs 500 / 500 / 2 |
| BX-03 | F2 39, 320, 336 vs F3 90 | min_count for 35K @0.001% | 1,093 (F2) vs 1,092 in the current paper (F3 N5) |
| BX-04 | F2 22, 187, 220, 641 vs F2 625 | H200 VRAM stated two ways in the same document | 143 GB vs 141 GB ("H200 SXM5 141 GB") |
| BX-05 | F2 22, 186–187, 239, 627–629 vs F3 78, 92 | Expanded-run GPU count and bitvector size differ between B3 notes and the paper version F3 reviewed | 4 × H200, 477 GB, ~119 GB/device (F2) vs "~56 GB/device × 8" ≈ 445 GB and "$3.50 × 8 GPUs × ~12 h" (paper per F3) |
| BX-06 | F2 401–405 vs F2 73–81 | Bio column of the 2-permutation null table does not equal the kdist table in the same document | 46,847 / 53,235 / 59,095 / 62,241 / K6–17 = 348,853 (sum 605,191) vs 46,853 / 53,238 / 59,086 / 62,226 / K6–17 = 349,969 (sum 606,292) |
| BX-07 | F2 37–42 | Blitz "min proteins" does not follow ceil like the other rows | 109 vs ⌈0.0001% × 109,224,173⌉ = 110 (109,225; 10,923; 1,093; 22; 11 all match ceil) |
| BX-08 | F2 187 | Per-device shard plus stated headroom does not sum to either stated VRAM figure | ~119 GB + 12 GB = 131 GB vs 143 GB / 141 GB |
| BX-09 | F1 57 vs F2 177–178 | The 76.9M protein set is described as "multi-feature (>1 item)" in F1 but the 76.9M → 109.2M comparison in F2 is phrased as proteins "with at least one annotation feature" | 76,890,945 (>1 item) vs 109.2M (≥1 feature) |
| BX-10 | F1 31 vs F2 555 | min_count of the controlled Direct-vs-SON comparison | 768 (experiment JSON, F1) vs 769 (B3 replacement text) |
| BX-11 | F2 547–576 vs F3 19, 100 | F2 asks to replace 21× with 21.4× in three places; F3 (June) still verifies "21×" in the paper | 21× vs 21.4× |
| BX-12 | F2 39 vs F3 143 / F4 74, 86 | 35K @0.001% result (F3 states it, B-52); F4 independently attributes Run 2 K=1–8 to the pre-fix run | 606,292 / K_max 17 / 654 s vs 16.8B through K=8 / 4.3 h |
| BX-13 | F1 132 vs F2 22, 624 | Hardware differs between the 1K experiments and the 35K experiments (not a conflict, but relevant for timing reproduction) | H100 (1K direct-vs-SON, null) vs 4 × H200 (35K) |

---

## SECTION C — PER-FILE SUMMARIES

### F1 — `review_b2_results.md` (325 lines)

- What: "Agent B2: Critical Experimental Result Verification Report" — an internal audit of every quantitative claim in `papers/et_miner_proteome.tex` against the raw artefacts (godmode itemsets parquet, item mapping, direct-vs-SON JSON, null-model JSON, godmode log, `transactions_214m.parquet`).
- Version addressed: the pre-March paper (no version label given); the results are the 1,002-feature ("214m"/godmode) campaign with min_count=8, plus the 0.001% direct-vs-SON and 5-permutation null experiments (JSONs dated 2026-02-19).
- Date: 2026-02-20.
- Verdict: 9 checks; 7 VERIFIED (protein counts, K=22 itemset and its 22-feature decode, 26,849,505 itemsets, 21.4× speedup, null K-distributions, all timings), 2 DISCREPANCY (Table 1 breakdown 247/752/3 vs actual 500/500/2; "a fortiori" argument logically inverted), 3 RISK FLAGs (5 permutations insufficient — Clopper-Pearson bound 0.451; GO:0005524 auto-derived via InterPro2GO → 21 independent features; null threshold 96× stricter than main run).
- Must-fix before submission: Table 1 breakdown; rewrite of the "a fortiori" sentence. Recommended: null model at min_count=8, 100+ permutations, note the GO:0005524 derivation.
- Notable: it establishes the canonical 1K numbers reused everywhere else (76,890,945 / 205,620,298 / 26,849,505 / 475,865 vs 22,846 / 440.5 s / 662.17 s), and records that the two "0.001%" experiments used different min_counts (768 vs 769).

### F2 — `revision_notes_b3.tex` (659 lines)

- What: LaTeX revision blocks ("Agent B3") to be pasted into the paper to add the expanded 35,012-feature / 109.2M-protein results run on 4 × H200; each block carries placement instructions. Values tagged [ACTUAL] come from `results_35k/experiment_direct_vs_son_35k_20260221.json`, `results_35k/experiment_null_model_20260221_003203.json`, `results_35k/wave3_base_partial.log`; [TBD] awaits runs.
- Version addressed: the same pre-March paper as F1 (it proposes a new Table `tab:campaign-35k`, replaces Table 1 with `tab:features-35k`, replaces the 5-permutation null section, and rewrites the 21× speedup text in three places to 21.4×).
- Date: "Updated: 2026-02-21".
- Content: 35K campaign table (only Power 0.001% complete: 606,292 ± 28 itemsets, K_max 17, 654.3 ± 6.7 s over 3 replicates; Base 0.1% partial to K=9 = 10,041,611); full 17-level K-distribution; new vocabulary table (12,773 InterPro / 5,763 GO / 1,107 EC / 15,162 Keywords / 175 taxa / 26 length bins / 6 pLDDT bins = 35,012); SON infeasibility argument (175 GB > 143 GB; run hung > 22 h); bitvector-vs-sparse complexity argument; K_max 22 → 17 explanation; 2-permutation null model (null collapses at K=5; Z=+723 at K=3, +20,585 at K=4); cross-source discovery framework; Wave-3 Base partial log; hardware/software description (Python 3.12, CuPy 13.4, NumPy 2.0, CUDA 12.6, Ubuntu 22.04, 1.5 TB RAM).
- TBD (never filled here): Base/Super/Blitz/Ultra/Opus rows; 100-permutation null table; cross-source pattern examples; K=17 itemset characterisation.
- Notable: still carries the 247/752/3 breakdown for the 1K set that F1 disputed one day earlier; states H200 VRAM as both 143 GB and 141 GB; its 606,292-itemset result is later (F3) reported as superseded by 16.8 billion after an NCCL int32-overflow fix.

### F3 — `senior_review_jun01.md` (160 lines, Dutch)

- What: "Senior-Reviewer Pass" over `et_miner_proteome.tex` (968 lines) along three axes — mis-cited references (web-verified), internal numerical consistency, spelling/grammar — produced by three parallel subagents plus manual verification. Report only; no edits.
- Version addressed: the current (post-March) paper, which already contains the expanded run (16,812,646,639 cumulative itemsets), the fixed Table 1 (500/500/2), the removed "a fortiori" argument, and "GPU-accelerated" wording. It cross-checks against the four earlier review documents (Feb–May 2026).
- Date: 2026-06-01.
- Verdict: the arithmetic core is "opvallend gezond" — both large table sums check exactly (26,849,505; 16,812,646,639), all min_counts, growth factors, Z-scores and ratios (9×, 124×, 21×, 95.2%, 5.1×) hold within rounding. Real problems are elsewhere: 4 MAJOR citation errors (wrong first authors barrio2024→Lau, coin2009→Terrapon; varadi2022 pages; BIGMiner scale 100M vs 6.5B), 9 MINOR bibliographic fixes, 5 orphan references; memory figures using three unlabelled byte conventions (CSR 3 / 5.1 / 19 GB; 206 vs 26 GB dense; 445 GB GB/GiB mix); the "p < 0.17" statistic unsupported (should be ≈0.45); small arithmetic slips ($243 vs $242; $332 vs $336; 1,092 vs 1093; 26.8M truncation); one broken abstract sentence.
- Flags as NEW / TO CONFIRM: the ~27,000× gap between B3's 606,292 itemsets (K_max 17, 654 s) and the current paper's 16.8 billion through K=8 (4.3 h), attributed to the NCCL int32-overflow bugfix.
- Fix priority: R1–R3 citations, G1 abstract, R4 BIGMiner, N1–N2 memory labels, N4 p-value, then the rest.

### F4 — `senior_review_mar23.md` (94 lines)

- What: "Senior Review" by an Opus 4.6 reviewer agent of "Protein Structural Motif Discovery at AlphaFold Scale". Verdict: Major Revisions Required.
- Version addressed: the paper version that already contains Run 2 (16.8 billion itemsets across 109M proteins, 12,773 InterPro entries, 5,763 GO, 6 pLDDT bins) but still has the original Run-1 Table 1 (247/752/3) and the "a fortiori" sentence (line 438).
- Date: 2026-03-23.
- Critical (C1–C4): Table 1 Run-1 breakdown fabricated (actual 500/500/2 from `top_pfam=500, top_go=500`; flagged by B2 on 2026-02-20 and never fixed); "a fortiori" inverted (769 vs 8); "GPU-resident" contradicts the Council of Copii finding that candidate generation is CPU-side; Table 8 pruning savings (68/12/4/2%, 73% combined at K=9) are projections presented as results.
- Important (I1–I6): Run 1 vs Run 2 feature semantics not comparable; 769× scale claim arithmetic unclear (BIGMiner 100M > 76.9M; GMiner 15M → 5.1×); closed-itemset "pruning" is post-processing; 5 permutations weak (cannot bound P(K≥7 given null) below 0.17; "+∞" Z misleading); pLDDT 3 vs 2 bins (6 defined, 4 with < 8 proteins); Run 2 K=1–8 numbers from the NCCL int32-bug run (fixed in f8c56cb5), not re-verified.
- Citations spot-checked (6) all correct.
- Required actions: fix Table 1 (500/500/2), "GPU-resident" → "GPU-accelerated", fix/remove "a fortiori", qualify pruning as projection, discuss Run 1/Run 2 semantics, more permutations or qualified p-values.

---

## SECTION D — TOTALS

Section A rows: **505** (R2-001 … R2-505). Section B: 62 text-stated rows (B-01…B-62) + 13 extractor-observed rows (BX-01…BX-13).

### D.1 — Rows per file

| file | rows |
|---|---|
| F1 | 137 |
| F2 | 208 |
| F3 | 119 |
| F4 | 41 |
| **total** | **505** |

### D.2 — Rows per category

| category | rows |
|---|---|
| deterministic | 330 |
| hardware-dependent | 74 |
| method-parameter | 57 |
| external-fact | 35 |
| software | 9 |
| **total** | **505** |

### D.3 — Rows per stance

| stance | rows |
|---|---|
| quotes-paper | 127 |
| asserts-own | 323 |
| disputes | 40 |
| requests | 15 |
| **total** | **505** |

### D.4 — Category × stance

| category | quotes-paper | asserts-own | disputes | requests | total |
|---|---|---|---|---|---|
| deterministic | 82 | 224 | 21 | 3 | 330 |
| hardware-dependent | 25 | 39 | 6 | 4 | 74 |
| method-parameter | 11 | 39 | 1 | 6 | 57 |
| external-fact | 8 | 14 | 11 | 2 | 35 |
| software | 1 | 7 | 1 | 0 | 9 |
| **total** | **127** | **323** | **40** | **15** | **505** |

### D.5 — File × stance

| file | quotes-paper | asserts-own | disputes | requests | total |
|---|---|---|---|---|---|
| F1 | 33 | 91 | 9 | 4 | 137 |
| F2 | 18 | 186 | 0 | 4 | 208 |
| F3 | 62 | 34 | 18 | 5 | 119 |
| F4 | 14 | 12 | 13 | 2 | 41 |

### D.6 — File × category

| file | deterministic | hardware-dependent | method-parameter | external-fact | software | total |
|---|---|---|---|---|---|---|
| F1 | 104 | 18 | 13 | 2 | 0 | 137 |
| F2 | 135 | 30 | 35 | 3 | 5 | 208 |
| F3 | 65 | 21 | 5 | 26 | 2 | 119 |
| F4 | 26 | 5 | 4 | 4 | 2 | 41 |



---

# SOURCE: surviving result files, bench logs, notebook, READMEs (pre-existing in the tree)

(verbatim copy of `runs/20260902T0000Z/phase1/claims_logs.md`)

# Phase 1 — claims extracted from surviving log / result files

Generated 2026-09-02 by the log-claims subagent. Repo root: `/root/projects/ET-Miner` (all `file` cells in Section A are relative to it).
Scope: every log-like or result-like file that survived in the repo and in `/root` (home). NOT judged for correctness — extraction only.
Paper sources (`paper/*.tex`, `paper/*.md`) were inventoried (Section B) but deliberately NOT extracted here; they are covered by the `claims_tex.md` / `claims_reviews_*.md` subagents.

**Headline:** the ONLY surviving AlphaFold result artifacts are the two `results_214m/*.json` files (2026-02-19, 1K vocab, 76,890,945 transactions), the decoded K=15–19 itemset dump, and hard-coded numbers in the notebook / GLOSSARY / README / RUNBOOK. No mining log (`*.log`, `nohup.out`, `experiment_log_*.txt`), no per-K parquet, no `itemsets_214m_godmode.parquet`, no `item_mapping_214m.parquet`, no `transactions_214m*.parquet`, and no `experiment_full_campaign_*.json` exist anywhere on this machine. The only real logs on the box (`bench/results/**/raw.jsonl`, `env.txt`, `report.md`, `FINDINGS.md`) are the synthetic-preset GPU benchmark campaign on 2× RTX 3090 (2026-08-31 / 2026-09-01), which the paper may cite for kernel throughput but which is not AlphaFold data.

## SECTION A — claim rows

Categories: deterministic | hardware-dependent | method-parameter | external-fact | software. Line refs: `L<n>` = 1-based line in file; `cell N (md/code) L<n>` for the notebook; `(file)` = whole-file property; `(derived)` = value computed by the extractor from the file's own numbers (for cross-checking, not a claim in the file).

| ID | file | line | value | unit | category | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|
| L-001 | `bench/results/2026-08-31-3090x2/env.txt` | (file) | mtime 2026-09-01 22:47 UTC; size 2194 B; git-added 6f789d8 2026-08-31 22:39:15 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-002 | `bench/results/2026-08-31-3090x2/report.md` | (file) | mtime 2026-09-01 22:47 UTC; size 3192 B; git-added 6f789d8 2026-08-31 22:39:15 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-003 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | (file) | mtime 2026-09-01 22:47 UTC; size 4634 B; git-added 6f789d8 2026-08-31 22:39:15 +0000 (follow-up note b1f147e) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-004 | `bench/results/2026-08-31-3090x2/raw.jsonl` | (file) | mtime 2026-09-01 22:47 UTC; size 33942 B; git-added 6f789d8 2026-08-31 22:39:15 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-005 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | (file) | mtime 2026-09-01 22:47 UTC; size 2194 B; git-added b1f147e 2026-09-01 20:11:16 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-006 | `bench/results/2026-09-01-3090x2-sparse/report.md` | (file) | mtime 2026-09-01 22:47 UTC; size 1551 B; git-added b1f147e 2026-09-01 20:11:16 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-007 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | (file) | mtime 2026-09-01 22:47 UTC; size 4375 B; git-added b1f147e 2026-09-01 20:11:16 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-008 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | (file) | mtime 2026-09-01 22:47 UTC; size 18932 B; git-added b1f147e 2026-09-01 20:11:16 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-009 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | (file) | mtime 2026-09-01 22:48 UTC; size 1110 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-010 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | (file) | mtime 2026-09-01 22:48 UTC; size 4676 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-011 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (file) | mtime 2026-09-01 22:48 UTC; size 4749057 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-012 | `applications/alphafold/results_214m/GLOSSARY.md` | (file) | mtime 2026-09-01 22:48 UTC; size 17193 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-013 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | (file) | mtime 2026-09-01 22:48 UTC; size 111866 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-014 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | (file) | mtime 2026-09-01 22:48 UTC; size 7790 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-015 | `bench/results/2026-08-31-3090x2/env.txt` | 2 | 5a8e59f8493622f623eae97daed02d460ec80389 | git sha | software | $ git rev-parse HEAD → 5a8e59f8493622f623eae97daed02d460ec80389 |
| L-016 | `bench/results/2026-08-31-3090x2/env.txt` | 6 | Mon Aug 31 20:52:51 2026 | timestamp | hardware-dependent | nvidia-smi banner: Mon Aug 31 20:52:51 2026 |
| L-017 | `bench/results/2026-08-31-3090x2/env.txt` | 8 | 580.159.03 | driver version | software | NVIDIA-SMI 580.159.03 Driver Version: 580.159.03 CUDA Version: 13.0 |
| L-018 | `bench/results/2026-08-31-3090x2/env.txt` | 8 | 13.0 | CUDA version | software | CUDA Version: 13.0 |
| L-019 | `bench/results/2026-08-31-3090x2/env.txt` | 14 | NVIDIA GeForce RTX 3090 (GPU 0, bus 01:00.0) | GPU model | hardware-dependent | 0  NVIDIA GeForce RTX 3090  On  00000000:01:00.0 Off |
| L-020 | `bench/results/2026-08-31-3090x2/env.txt` | 15 | 24576 | MiB VRAM | hardware-dependent | 1MiB / 24576MiB |
| L-021 | `bench/results/2026-08-31-3090x2/env.txt` | 15 | 360 | W power cap | hardware-dependent | 123W / 360W |
| L-022 | `bench/results/2026-08-31-3090x2/env.txt` | 18 | NVIDIA GeForce RTX 3090 (GPU 1, bus 82:00.0) | GPU model | hardware-dependent | 1  NVIDIA GeForce RTX 3090  On  00000000:82:00.0 Off |
| L-023 | `bench/results/2026-08-31-3090x2/env.txt` | 19 | 24576 | MiB VRAM | hardware-dependent | 1MiB / 24576MiB |
| L-024 | `bench/results/2026-08-31-3090x2/env.txt` | 32 | (no output captured) | — | software | $ /root/projects/ET-Miner/.venv/bin/python3 -m pip freeze — section is empty in file |
| L-025 | `bench/results/2026-08-31-3090x2/report.md` | 3 | 28 ok / 0 failed | runs | deterministic | Runs: 28 ok, 0 failed/timeout. |
| L-026 | `bench/results/2026-08-31-3090x2/report.md` | 9 | 10.4 | s median wall | hardware-dependent | deepk-density-auto \| deep_k \| legacy \| compact \| 2 \| 1 \| 10.4 \| 10.4 \| 416 |
| L-027 | `bench/results/2026-08-31-3090x2/report.md` | 9 | 10.4 | s min wall | hardware-dependent | deepk-density-auto: min s 10.4 |
| L-028 | `bench/results/2026-08-31-3090x2/report.md` | 9 | 416 | MB peak VRAM | hardware-dependent | deepk-density-auto: peak VRAM MB 416; throttled no |
| L-029 | `bench/results/2026-08-31-3090x2/report.md` | 9 | gpus=2; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-density-auto \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-030 | `bench/results/2026-08-31-3090x2/report.md` | 10 | 0.6 | s median wall | hardware-dependent | deepk-legacy-1g \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-031 | `bench/results/2026-08-31-3090x2/report.md` | 10 | 0.6 | s min wall | hardware-dependent | deepk-legacy-1g: min s 0.6 |
| L-032 | `bench/results/2026-08-31-3090x2/report.md` | 10 | 352 | MB peak VRAM | hardware-dependent | deepk-legacy-1g: peak VRAM MB 352; throttled yes |
| L-033 | `bench/results/2026-08-31-3090x2/report.md` | 10 | gpus=1; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-legacy-1g \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-034 | `bench/results/2026-08-31-3090x2/report.md` | 11 | 1.3 | s median wall | hardware-dependent | deepk-legacy-2g \| deep_k \| legacy \| compact \| 2 \| 1 \| 1.3 \| 1.3 \| 416 |
| L-035 | `bench/results/2026-08-31-3090x2/report.md` | 11 | 1.3 | s min wall | hardware-dependent | deepk-legacy-2g: min s 1.3 |
| L-036 | `bench/results/2026-08-31-3090x2/report.md` | 11 | 416 | MB peak VRAM | hardware-dependent | deepk-legacy-2g: peak VRAM MB 416; throttled no |
| L-037 | `bench/results/2026-08-31-3090x2/report.md` | 11 | gpus=2; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-legacy-2g \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-038 | `bench/results/2026-08-31-3090x2/report.md` | 12 | 0.9 | s median wall | hardware-dependent | deepk-nonccl \| deep_k \| legacy \| compact \| 2 \| 1 \| 0.9 \| 0.9 \| 310 |
| L-039 | `bench/results/2026-08-31-3090x2/report.md` | 12 | 0.9 | s min wall | hardware-dependent | deepk-nonccl: min s 0.9 |
| L-040 | `bench/results/2026-08-31-3090x2/report.md` | 12 | 310 | MB peak VRAM | hardware-dependent | deepk-nonccl: peak VRAM MB 310; throttled no |
| L-041 | `bench/results/2026-08-31-3090x2/report.md` | 12 | gpus=2; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-nonccl \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-042 | `bench/results/2026-08-31-3090x2/report.md` | 13 | 0.6 | s median wall | hardware-dependent | deepk-prefilter-off \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-043 | `bench/results/2026-08-31-3090x2/report.md` | 13 | 0.6 | s min wall | hardware-dependent | deepk-prefilter-off: min s 0.6 |
| L-044 | `bench/results/2026-08-31-3090x2/report.md` | 13 | 352 | MB peak VRAM | hardware-dependent | deepk-prefilter-off: peak VRAM MB 352; throttled no |
| L-045 | `bench/results/2026-08-31-3090x2/report.md` | 13 | gpus=1; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-prefilter-off \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-046 | `bench/results/2026-08-31-3090x2/report.md` | 14 | 0.6 | s median wall | hardware-dependent | deepk-shared-1g \| deep_k \| shared \| compact \| 1 \| 3 \| 0.6 \| 0.6 \| 352 |
| L-047 | `bench/results/2026-08-31-3090x2/report.md` | 14 | 0.6 | s min wall | hardware-dependent | deepk-shared-1g: min s 0.6 |
| L-048 | `bench/results/2026-08-31-3090x2/report.md` | 14 | 352 | MB peak VRAM | hardware-dependent | deepk-shared-1g: peak VRAM MB 352; throttled yes |
| L-049 | `bench/results/2026-08-31-3090x2/report.md` | 14 | gpus=1; reps=3; variant=shared; filter=compact; preset=deep_k | params | method-parameter | deepk-shared-1g \| deep_k \| shared \| compact \| 1 \| 3 |
| L-050 | `bench/results/2026-08-31-3090x2/report.md` | 15 | 1.4 | s median wall | hardware-dependent | deepk-shared-2g \| deep_k \| shared \| compact \| 2 \| 3 \| 1.4 \| 1.3 \| 416 |
| L-051 | `bench/results/2026-08-31-3090x2/report.md` | 15 | 1.3 | s min wall | hardware-dependent | deepk-shared-2g: min s 1.3 |
| L-052 | `bench/results/2026-08-31-3090x2/report.md` | 15 | 416 | MB peak VRAM | hardware-dependent | deepk-shared-2g: peak VRAM MB 416; throttled no |
| L-053 | `bench/results/2026-08-31-3090x2/report.md` | 15 | gpus=2; reps=3; variant=shared; filter=compact; preset=deep_k | params | method-parameter | deepk-shared-2g \| deep_k \| shared \| compact \| 2 \| 3 |
| L-054 | `bench/results/2026-08-31-3090x2/report.md` | 16 | 0.6 | s median wall | hardware-dependent | deepk-single-prefilter-on \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-055 | `bench/results/2026-08-31-3090x2/report.md` | 16 | 0.6 | s min wall | hardware-dependent | deepk-single-prefilter-on: min s 0.6 |
| L-056 | `bench/results/2026-08-31-3090x2/report.md` | 16 | 352 | MB peak VRAM | hardware-dependent | deepk-single-prefilter-on: peak VRAM MB 352; throttled yes |
| L-057 | `bench/results/2026-08-31-3090x2/report.md` | 16 | gpus=1; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-single-prefilter-on \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-058 | `bench/results/2026-08-31-3090x2/report.md` | 17 | 1.5 | s median wall | hardware-dependent | skew-nnz \| skewed_rows \| legacy \| compact \| 2 \| 2 \| 1.5 \| 1.5 \| 418 |
| L-059 | `bench/results/2026-08-31-3090x2/report.md` | 17 | 1.5 | s min wall | hardware-dependent | skew-nnz: min s 1.5 |
| L-060 | `bench/results/2026-08-31-3090x2/report.md` | 17 | 418 | MB peak VRAM | hardware-dependent | skew-nnz: peak VRAM MB 418; throttled no |
| L-061 | `bench/results/2026-08-31-3090x2/report.md` | 17 | gpus=2; reps=2; variant=legacy; filter=compact; preset=skewed_rows | params | method-parameter | skew-nnz \| skewed_rows \| legacy \| compact \| 2 \| 2 |
| L-062 | `bench/results/2026-08-31-3090x2/report.md` | 18 | 1.4 | s median wall | hardware-dependent | skew-rows \| skewed_rows \| legacy \| compact \| 2 \| 2 \| 1.4 \| 1.4 \| 416 |
| L-063 | `bench/results/2026-08-31-3090x2/report.md` | 18 | 1.4 | s min wall | hardware-dependent | skew-rows: min s 1.4 |
| L-064 | `bench/results/2026-08-31-3090x2/report.md` | 18 | 416 | MB peak VRAM | hardware-dependent | skew-rows: peak VRAM MB 416; throttled no |
| L-065 | `bench/results/2026-08-31-3090x2/report.md` | 18 | gpus=2; reps=2; variant=legacy; filter=compact; preset=skewed_rows | params | method-parameter | skew-rows \| skewed_rows \| legacy \| compact \| 2 \| 2 |
| L-066 | `bench/results/2026-08-31-3090x2/report.md` | 19 | 89.6 | s median wall | hardware-dependent | stressk2-filter-compact \| stress_k2 \| legacy \| compact \| 2 \| 1 \| 89.6 \| 89.6 \| 6918 |
| L-067 | `bench/results/2026-08-31-3090x2/report.md` | 19 | 89.6 | s min wall | hardware-dependent | stressk2-filter-compact: min s 89.6 |
| L-068 | `bench/results/2026-08-31-3090x2/report.md` | 19 | 6918 | MB peak VRAM | hardware-dependent | stressk2-filter-compact: peak VRAM MB 6918; throttled yes |
| L-069 | `bench/results/2026-08-31-3090x2/report.md` | 19 | gpus=2; reps=1; variant=legacy; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-filter-compact \| stress_k2 \| legacy \| compact \| 2 \| 1 |
| L-070 | `bench/results/2026-08-31-3090x2/report.md` | 20 | 90.2 | s median wall | hardware-dependent | stressk2-filter-cpu \| stress_k2 \| legacy \| cpu \| 2 \| 1 \| 90.2 \| 90.2 \| 6920 |
| L-071 | `bench/results/2026-08-31-3090x2/report.md` | 20 | 90.2 | s min wall | hardware-dependent | stressk2-filter-cpu: min s 90.2 |
| L-072 | `bench/results/2026-08-31-3090x2/report.md` | 20 | 6920 | MB peak VRAM | hardware-dependent | stressk2-filter-cpu: peak VRAM MB 6920; throttled yes |
| L-073 | `bench/results/2026-08-31-3090x2/report.md` | 20 | gpus=2; reps=1; variant=legacy; filter=cpu; preset=stress_k2 | params | method-parameter | stressk2-filter-cpu \| stress_k2 \| legacy \| cpu \| 2 \| 1 |
| L-074 | `bench/results/2026-08-31-3090x2/report.md` | 21 | 89.7 | s median wall | hardware-dependent | stressk2-filter-cupy \| stress_k2 \| legacy \| cupy \| 2 \| 1 \| 89.7 \| 89.7 \| 6998 |
| L-075 | `bench/results/2026-08-31-3090x2/report.md` | 21 | 89.7 | s min wall | hardware-dependent | stressk2-filter-cupy: min s 89.7 |
| L-076 | `bench/results/2026-08-31-3090x2/report.md` | 21 | 6998 | MB peak VRAM | hardware-dependent | stressk2-filter-cupy: peak VRAM MB 6998; throttled yes |
| L-077 | `bench/results/2026-08-31-3090x2/report.md` | 21 | gpus=2; reps=1; variant=legacy; filter=cupy; preset=stress_k2 | params | method-parameter | stressk2-filter-cupy \| stress_k2 \| legacy \| cupy \| 2 \| 1 |
| L-078 | `bench/results/2026-08-31-3090x2/report.md` | 22 | 2238.8 | s median wall | hardware-dependent | stressk2-legacy-1g \| stress_k2 \| legacy \| compact \| 1 \| 1 \| 2238.8 \| 2238.8 \| 17340 |
| L-079 | `bench/results/2026-08-31-3090x2/report.md` | 22 | 2238.8 | s min wall | hardware-dependent | stressk2-legacy-1g: min s 2238.8 |
| L-080 | `bench/results/2026-08-31-3090x2/report.md` | 22 | 17340 | MB peak VRAM | hardware-dependent | stressk2-legacy-1g: peak VRAM MB 17340; throttled yes |
| L-081 | `bench/results/2026-08-31-3090x2/report.md` | 22 | gpus=1; reps=1; variant=legacy; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-legacy-1g \| stress_k2 \| legacy \| compact \| 1 \| 1 |
| L-082 | `bench/results/2026-08-31-3090x2/report.md` | 23 | 1996.7 | s median wall | hardware-dependent | stressk2-legacy-2g \| stress_k2 \| legacy \| compact \| 2 \| 1 \| 1996.7 \| 1996.7 \| 16576 |
| L-083 | `bench/results/2026-08-31-3090x2/report.md` | 23 | 1996.7 | s min wall | hardware-dependent | stressk2-legacy-2g: min s 1996.7 |
| L-084 | `bench/results/2026-08-31-3090x2/report.md` | 23 | 16576 | MB peak VRAM | hardware-dependent | stressk2-legacy-2g: peak VRAM MB 16576; throttled yes |
| L-085 | `bench/results/2026-08-31-3090x2/report.md` | 23 | gpus=2; reps=1; variant=legacy; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-legacy-2g \| stress_k2 \| legacy \| compact \| 2 \| 1 |
| L-086 | `bench/results/2026-08-31-3090x2/report.md` | 24 | 253.3 | s median wall | hardware-dependent | stressk2-shared-1g \| stress_k2 \| shared \| compact \| 1 \| 3 \| 253.3 \| 253.1 \| 17340 |
| L-087 | `bench/results/2026-08-31-3090x2/report.md` | 24 | 253.1 | s min wall | hardware-dependent | stressk2-shared-1g: min s 253.1 |
| L-088 | `bench/results/2026-08-31-3090x2/report.md` | 24 | 17340 | MB peak VRAM | hardware-dependent | stressk2-shared-1g: peak VRAM MB 17340; throttled yes |
| L-089 | `bench/results/2026-08-31-3090x2/report.md` | 24 | gpus=1; reps=3; variant=shared; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-shared-1g \| stress_k2 \| shared \| compact \| 1 \| 3 |
| L-090 | `bench/results/2026-08-31-3090x2/report.md` | 25 | 122.3 | s median wall | hardware-dependent | stressk2-shared-2g \| stress_k2 \| shared \| compact \| 2 \| 3 \| 122.3 \| 122.3 \| 16576 |
| L-091 | `bench/results/2026-08-31-3090x2/report.md` | 25 | 122.3 | s min wall | hardware-dependent | stressk2-shared-2g: min s 122.3 |
| L-092 | `bench/results/2026-08-31-3090x2/report.md` | 25 | 16576 | MB peak VRAM | hardware-dependent | stressk2-shared-2g: peak VRAM MB 16576; throttled yes |
| L-093 | `bench/results/2026-08-31-3090x2/report.md` | 25 | gpus=2; reps=3; variant=shared; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-shared-2g \| stress_k2 \| shared \| compact \| 2 \| 3 |
| L-094 | `bench/results/2026-08-31-3090x2/report.md` | 26 | 1.6 | s median wall | hardware-dependent | twophase-smoke \| smoke \| legacy \| compact \| 2 \| 1 \| 1.6 \| 1.6 \| 408 |
| L-095 | `bench/results/2026-08-31-3090x2/report.md` | 26 | 1.6 | s min wall | hardware-dependent | twophase-smoke: min s 1.6 |
| L-096 | `bench/results/2026-08-31-3090x2/report.md` | 26 | 408 | MB peak VRAM | hardware-dependent | twophase-smoke: peak VRAM MB 408; throttled yes |
| L-097 | `bench/results/2026-08-31-3090x2/report.md` | 26 | gpus=2; reps=1; variant=legacy; filter=compact; preset=smoke | params | method-parameter | twophase-smoke \| smoke \| legacy \| compact \| 2 \| 1 |
| L-098 | `bench/results/2026-08-31-3090x2/report.md` | 32 | 0.97× | speedup shared vs legacy (deep_k, 2 GPUs) | hardware-dependent | deep_k \| 2 \| 1.3 \| 1.4 \| 0.97× |
| L-099 | `bench/results/2026-08-31-3090x2/report.md` | 33 | 16.32× | speedup shared vs legacy (stress_k2, 2 GPUs) | hardware-dependent | stress_k2 \| 2 \| 1996.7 \| 122.3 \| 16.32× |
| L-100 | `bench/results/2026-08-31-3090x2/report.md` | 37 | 10.431 | s (deepk-density-auto#r0) | hardware-dependent | **deep_k** (deepk-density-auto#r0, 10.431s): |
| L-101 | `bench/results/2026-08-31-3090x2/report.md` | 41 | K1: candidates=112; frequent=112 | count | deterministic | deep_k per-level: \| 1 \| 112 \| 112 \| 27 \| |
| L-102 | `bench/results/2026-08-31-3090x2/report.md` | 41 | 27 | ms (K=1, deepk-density-auto#r0) | hardware-dependent | \| 1 \| 112 \| 112 \| 27 \| |
| L-103 | `bench/results/2026-08-31-3090x2/report.md` | 42 | K2: candidates=0; frequent=471 | count | deterministic | deep_k per-level: \| 2 \| 0 \| 471 \| 46 \| |
| L-104 | `bench/results/2026-08-31-3090x2/report.md` | 42 | 46 | ms (K=2, deepk-density-auto#r0) | hardware-dependent | \| 2 \| 0 \| 471 \| 46 \| |
| L-105 | `bench/results/2026-08-31-3090x2/report.md` | 43 | K3: candidates=0; frequent=901 | count | deterministic | deep_k per-level: \| 3 \| 0 \| 901 \| 5 \| |
| L-106 | `bench/results/2026-08-31-3090x2/report.md` | 43 | 5 | ms (K=3, deepk-density-auto#r0) | hardware-dependent | \| 3 \| 0 \| 901 \| 5 \| |
| L-107 | `bench/results/2026-08-31-3090x2/report.md` | 44 | K4: candidates=0; frequent=1407 | count | deterministic | deep_k per-level: \| 4 \| 0 \| 1,407 \| 6 \| |
| L-108 | `bench/results/2026-08-31-3090x2/report.md` | 44 | 6 | ms (K=4, deepk-density-auto#r0) | hardware-dependent | \| 4 \| 0 \| 1,407 \| 6 \| |
| L-109 | `bench/results/2026-08-31-3090x2/report.md` | 45 | K5: candidates=0; frequent=1854 | count | deterministic | deep_k per-level: \| 5 \| 0 \| 1,854 \| 3145 \| |
| L-110 | `bench/results/2026-08-31-3090x2/report.md` | 45 | 3145 | ms (K=5, deepk-density-auto#r0) | hardware-dependent | \| 5 \| 0 \| 1,854 \| 3145 \| |
| L-111 | `bench/results/2026-08-31-3090x2/report.md` | 46 | K6: candidates=0; frequent=1848 | count | deterministic | deep_k per-level: \| 6 \| 0 \| 1,848 \| 2553 \| |
| L-112 | `bench/results/2026-08-31-3090x2/report.md` | 46 | 2553 | ms (K=6, deepk-density-auto#r0) | hardware-dependent | \| 6 \| 0 \| 1,848 \| 2553 \| |
| L-113 | `bench/results/2026-08-31-3090x2/report.md` | 47 | K7: candidates=0; frequent=1320 | count | deterministic | deep_k per-level: \| 7 \| 0 \| 1,320 \| 1817 \| |
| L-114 | `bench/results/2026-08-31-3090x2/report.md` | 47 | 1817 | ms (K=7, deepk-density-auto#r0) | hardware-dependent | \| 7 \| 0 \| 1,320 \| 1817 \| |
| L-115 | `bench/results/2026-08-31-3090x2/report.md` | 48 | K8: candidates=0; frequent=660 | count | deterministic | deep_k per-level: \| 8 \| 0 \| 660 \| 902 \| |
| L-116 | `bench/results/2026-08-31-3090x2/report.md` | 48 | 902 | ms (K=8, deepk-density-auto#r0) | hardware-dependent | \| 8 \| 0 \| 660 \| 902 \| |
| L-117 | `bench/results/2026-08-31-3090x2/report.md` | 49 | K9: candidates=0; frequent=220 | count | deterministic | deep_k per-level: \| 9 \| 0 \| 220 \| 374 \| |
| L-118 | `bench/results/2026-08-31-3090x2/report.md` | 49 | 374 | ms (K=9, deepk-density-auto#r0) | hardware-dependent | \| 9 \| 0 \| 220 \| 374 \| |
| L-119 | `bench/results/2026-08-31-3090x2/report.md` | 50 | K10: candidates=0; frequent=44 | count | deterministic | deep_k per-level: \| 10 \| 0 \| 44 \| 65 \| |
| L-120 | `bench/results/2026-08-31-3090x2/report.md` | 50 | 65 | ms (K=10, deepk-density-auto#r0) | hardware-dependent | \| 10 \| 0 \| 44 \| 65 \| |
| L-121 | `bench/results/2026-08-31-3090x2/report.md` | 51 | K11: candidates=0; frequent=4 | count | deterministic | deep_k per-level: \| 11 \| 0 \| 4 \| 14 \| |
| L-122 | `bench/results/2026-08-31-3090x2/report.md` | 51 | 14 | ms (K=11, deepk-density-auto#r0) | hardware-dependent | \| 11 \| 0 \| 4 \| 14 \| |
| L-123 | `bench/results/2026-08-31-3090x2/report.md` | 53 | 1.517 | s (skew-nnz#r1) | hardware-dependent | **skewed_rows** (skew-nnz#r1, 1.517s): |
| L-124 | `bench/results/2026-08-31-3090x2/report.md` | 57 | K1: candidates=118; frequent=118 | count | deterministic | skewed_rows per-level: \| 1 \| 118 \| 118 \| 26 \| |
| L-125 | `bench/results/2026-08-31-3090x2/report.md` | 57 | 26 | ms (K=1, skew-nnz#r1) | hardware-dependent | \| 1 \| 118 \| 118 \| 26 \| |
| L-126 | `bench/results/2026-08-31-3090x2/report.md` | 58 | K2: candidates=0; frequent=717 | count | deterministic | skewed_rows per-level: \| 2 \| 0 \| 717 \| 47 \| |
| L-127 | `bench/results/2026-08-31-3090x2/report.md` | 58 | 47 | ms (K=2, skew-nnz#r1) | hardware-dependent | \| 2 \| 0 \| 717 \| 47 \| |
| L-128 | `bench/results/2026-08-31-3090x2/report.md` | 59 | K3: candidates=0; frequent=1928 | count | deterministic | skewed_rows per-level: \| 3 \| 0 \| 1,928 \| 7 \| |
| L-129 | `bench/results/2026-08-31-3090x2/report.md` | 59 | 7 | ms (K=3, skew-nnz#r1) | hardware-dependent | \| 3 \| 0 \| 1,928 \| 7 \| |
| L-130 | `bench/results/2026-08-31-3090x2/report.md` | 60 | K4: candidates=0; frequent=2932 | count | deterministic | skewed_rows per-level: \| 4 \| 0 \| 2,932 \| 16 \| |
| L-131 | `bench/results/2026-08-31-3090x2/report.md` | 60 | 16 | ms (K=4, skew-nnz#r1) | hardware-dependent | \| 4 \| 0 \| 2,932 \| 16 \| |
| L-132 | `bench/results/2026-08-31-3090x2/report.md` | 61 | K5: candidates=0; frequent=2683 | count | deterministic | skewed_rows per-level: \| 5 \| 0 \| 2,683 \| 34 \| |
| L-133 | `bench/results/2026-08-31-3090x2/report.md` | 61 | 34 | ms (K=5, skew-nnz#r1) | hardware-dependent | \| 5 \| 0 \| 2,683 \| 34 \| |
| L-134 | `bench/results/2026-08-31-3090x2/report.md` | 62 | K6: candidates=0; frequent=1458 | count | deterministic | skewed_rows per-level: \| 6 \| 0 \| 1,458 \| 40 \| |
| L-135 | `bench/results/2026-08-31-3090x2/report.md` | 62 | 40 | ms (K=6, skew-nnz#r1) | hardware-dependent | \| 6 \| 0 \| 1,458 \| 40 \| |
| L-136 | `bench/results/2026-08-31-3090x2/report.md` | 63 | K7: candidates=0; frequent=443 | count | deterministic | skewed_rows per-level: \| 7 \| 0 \| 443 \| 22 \| |
| L-137 | `bench/results/2026-08-31-3090x2/report.md` | 63 | 22 | ms (K=7, skew-nnz#r1) | hardware-dependent | \| 7 \| 0 \| 443 \| 22 \| |
| L-138 | `bench/results/2026-08-31-3090x2/report.md` | 64 | K8: candidates=0; frequent=67 | count | deterministic | skewed_rows per-level: \| 8 \| 0 \| 67 \| 5 \| |
| L-139 | `bench/results/2026-08-31-3090x2/report.md` | 64 | 5 | ms (K=8, skew-nnz#r1) | hardware-dependent | \| 8 \| 0 \| 67 \| 5 \| |
| L-140 | `bench/results/2026-08-31-3090x2/report.md` | 65 | K9: candidates=0; frequent=4 | count | deterministic | skewed_rows per-level: \| 9 \| 0 \| 4 \| 2 \| |
| L-141 | `bench/results/2026-08-31-3090x2/report.md` | 65 | 2 | ms (K=9, skew-nnz#r1) | hardware-dependent | \| 9 \| 0 \| 4 \| 2 \| |
| L-142 | `bench/results/2026-08-31-3090x2/report.md` | 67 | 1.582 | s (twophase-smoke#r0) | hardware-dependent | **smoke** (twophase-smoke#r0, 1.582s): |
| L-143 | `bench/results/2026-08-31-3090x2/report.md` | 71 | K1 (phase 1): candidates=118; frequent=118 | count | deterministic | smoke two-phase per-level: \| 1 \| 118 \| 118 \| 252 \| |
| L-144 | `bench/results/2026-08-31-3090x2/report.md` | 71 | 252 | ms (K=1, phase 1, twophase-smoke#r0) | hardware-dependent | \| 1 \| 118 \| 118 \| 252 \| |
| L-145 | `bench/results/2026-08-31-3090x2/report.md` | 72 | K2 (phase 1): candidates=0; frequent=290 | count | deterministic | smoke two-phase per-level: \| 2 \| 0 \| 290 \| 48 \| |
| L-146 | `bench/results/2026-08-31-3090x2/report.md` | 72 | 48 | ms (K=2, phase 1, twophase-smoke#r0) | hardware-dependent | \| 2 \| 0 \| 290 \| 48 \| |
| L-147 | `bench/results/2026-08-31-3090x2/report.md` | 73 | K3 (phase 1): candidates=0; frequent=202 | count | deterministic | smoke two-phase per-level: \| 3 \| 0 \| 202 \| 120 \| |
| L-148 | `bench/results/2026-08-31-3090x2/report.md` | 73 | 120 | ms (K=3, phase 1, twophase-smoke#r0) | hardware-dependent | \| 3 \| 0 \| 202 \| 120 \| |
| L-149 | `bench/results/2026-08-31-3090x2/report.md` | 74 | K4 (phase 1): candidates=0; frequent=22 | count | deterministic | smoke two-phase per-level: \| 4 \| 0 \| 22 \| 7 \| |
| L-150 | `bench/results/2026-08-31-3090x2/report.md` | 74 | 7 | ms (K=4, phase 1, twophase-smoke#r0) | hardware-dependent | \| 4 \| 0 \| 22 \| 7 \| |
| L-151 | `bench/results/2026-08-31-3090x2/report.md` | 75 | K5 (phase 1): candidates=0; frequent=0 | count | deterministic | smoke two-phase per-level: \| 5 \| 0 \| 0 \| 0 \| |
| L-152 | `bench/results/2026-08-31-3090x2/report.md` | 75 | 0 | ms (K=5, phase 1, twophase-smoke#r0) | hardware-dependent | \| 5 \| 0 \| 0 \| 0 \| |
| L-153 | `bench/results/2026-08-31-3090x2/report.md` | 76 | K1 (phase 2): candidates=118; frequent=118 | count | deterministic | smoke two-phase per-level: \| 1 \| 118 \| 118 \| 2 \| |
| L-154 | `bench/results/2026-08-31-3090x2/report.md` | 76 | 2 | ms (K=1, phase 2, twophase-smoke#r0) | hardware-dependent | \| 1 \| 118 \| 118 \| 2 \| |
| L-155 | `bench/results/2026-08-31-3090x2/report.md` | 77 | K2 (phase 2): candidates=0; frequent=290 | count | deterministic | smoke two-phase per-level: \| 2 \| 0 \| 290 \| 34 \| |
| L-156 | `bench/results/2026-08-31-3090x2/report.md` | 77 | 34 | ms (K=2, phase 2, twophase-smoke#r0) | hardware-dependent | \| 2 \| 0 \| 290 \| 34 \| |
| L-157 | `bench/results/2026-08-31-3090x2/report.md` | 78 | K3 (phase 2): candidates=0; frequent=202 | count | deterministic | smoke two-phase per-level: \| 3 \| 0 \| 202 \| 87 \| |
| L-158 | `bench/results/2026-08-31-3090x2/report.md` | 78 | 87 | ms (K=3, phase 2, twophase-smoke#r0) | hardware-dependent | \| 3 \| 0 \| 202 \| 87 \| |
| L-159 | `bench/results/2026-08-31-3090x2/report.md` | 79 | K4 (phase 2): candidates=0; frequent=22 | count | deterministic | smoke two-phase per-level: \| 4 \| 0 \| 22 \| 5 \| |
| L-160 | `bench/results/2026-08-31-3090x2/report.md` | 79 | 5 | ms (K=4, phase 2, twophase-smoke#r0) | hardware-dependent | \| 4 \| 0 \| 22 \| 5 \| |
| L-161 | `bench/results/2026-08-31-3090x2/report.md` | 80 | K5 (phase 2): candidates=0; frequent=0 | count | deterministic | smoke two-phase per-level: \| 5 \| 0 \| 0 \| 0 \| |
| L-162 | `bench/results/2026-08-31-3090x2/report.md` | 80 | 0 | ms (K=5, phase 2, twophase-smoke#r0) | hardware-dependent | \| 5 \| 0 \| 0 \| 0 \| |
| L-163 | `bench/results/2026-08-31-3090x2/report.md` | 82 | 2238.846 | s (stressk2-legacy-1g#r0) | hardware-dependent | **stress_k2** (stressk2-legacy-1g#r0, 2238.846s): |
| L-164 | `bench/results/2026-08-31-3090x2/report.md` | 86 | K1: candidates=35,000; frequent=35,000 | count | deterministic | stress_k2 per-level: \| 1 \| 35,000 \| 35,000 \| 72 \| |
| L-165 | `bench/results/2026-08-31-3090x2/report.md` | 86 | 72 | ms (K=1, stressk2-legacy-1g#r0) | hardware-dependent | \| 1 \| 35,000 \| 35,000 \| 72 \| |
| L-166 | `bench/results/2026-08-31-3090x2/report.md` | 87 | K2: candidates=612,482,500; frequent=1,660,332 | count | deterministic | stress_k2 per-level: \| 2 \| 612,482,500 \| 1,660,332 \| 175580 \| |
| L-167 | `bench/results/2026-08-31-3090x2/report.md` | 87 | 175580 | ms (K=2, stressk2-legacy-1g#r0) | hardware-dependent | \| 2 \| 612,482,500 \| 1,660,332 \| 175580 \| |
| L-168 | `bench/results/2026-08-31-3090x2/report.md` | 88 | K3: candidates=1,301,153; frequent=1,301,153 | count | deterministic | stress_k2 per-level: \| 3 \| 1,301,153 \| 1,301,153 \| 2046663 \| |
| L-169 | `bench/results/2026-08-31-3090x2/report.md` | 88 | 2046663 | ms (K=3, stressk2-legacy-1g#r0) | hardware-dependent | \| 3 \| 1,301,153 \| 1,301,153 \| 2046663 \| |
| L-170 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 1 | 2×RTX 3090; 2026-08-31 | hardware/date | hardware-dependent | # GPU campaign findings — 2×RTX 3090, 2026-08-31 |
| L-171 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 4 | 2× RTX 3090 24 GB (sm_86) | GPU | hardware-dependent | Box: 2× RTX 3090 24 GB (sm_86, driver 580.159.03, CuPy 14.1.1, no P2P — NCCL over SHM), vast.ai. |
| L-172 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 5 | 580.159.03 / CuPy 14.1.1 | driver / library | software | driver 580.159.03, CuPy 14.1.1, no P2P — NCCL over SHM |
| L-173 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 10 | 7/7 | tier-equivalence legs | deterministic | **Tier-equivalence chain: 7/7.** Tier 1 Polars == Tier 2 Rust == single-GPU == multi-GPU legacy == shared multi-GPU == efficient-apriori |
| L-174 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 13 | 114 passed / 0 failed | tests | software | **Full GPU test suite: 114 passed / 0 failed** — first-ever on-device run |
| L-175 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 22 | 8.8× / 16.3× | speedup | hardware-dependent | ## Finding 1 — shared/tiled kernel: 8.8× / 16.3× on the stress workload |
| L-176 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 24 | 2,000,000 transactions × 35,000 items; min_count 30; K=3 | preset params | method-parameter | `stress_k2`: 2M transactions × 35K items, min_count 30, mined to K=3 — |
| L-177 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 25 | 612M K=2 candidates; ~76 billion K=3 candidates | count | deterministic | 612M K=2 candidates plus **~76 billion** K=3 candidates, every one exactly counted (no sampling) |
| L-178 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 30 | 2238.8 s legacy → 253.3 s shared (±1 s over 3 reps); 8.8× | s / speedup (1× 3090 fused) | hardware-dependent | \| 1× 3090 (fused) \| 2238.8 s \| 253.3 s (±1 s over 3 reps) \| **8.8×** \| |
| L-179 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 31 | 1996.7 s legacy → 122.3 s shared; 16.3× | s / speedup (2× 3090 row-split) | hardware-dependent | \| 2× 3090 (dense row-split) \| 1996.7 s \| 122.3 s \| **16.3×** \| |
| L-180 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 33 | ≈300M counts/s (1 GPU); ≈630M counts/s (2 GPUs) | support counts per second | hardware-dependent | ≈300M exact support counts/s on one 3090, ≈630M/s on two. |
| L-181 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 36 | 0.97–1.07× | speedup (deep_k/smoke, shared vs legacy) | hardware-dependent | On small/deep data (`deep_k`, `smoke`) shared is neutral (0.97–1.07×) |
| L-182 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 37 | 64 | pairs (sub-64-pair groups routed to legacy kernel) | method-parameter | sub-64-pair groups route to the legacy kernel by design |
| L-183 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 43 | 9,285 fewer K=3 itemsets (−0.7%) | count | deterministic | mined **9,285 fewer K=3 itemsets (−0.7% of that level)** than the seven exact configs |
| L-184 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 45 | 0.7×min_count | prefilter reject threshold | method-parameter | rejects candidates whose sampled estimate is < 0.7×min_count *without exact recount* |
| L-185 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 57 | 10.4 s vs 1.3 s | s (density-auto vs dense) | hardware-dependent | took 10.4 s vs 1.3 s dense: post-transition levels (K=5–9) spend seconds |
| L-186 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 60 | n/32 | crossover (mean support) | method-parameter | The n/32 crossover optimizes memory, not yet time, at this scale. |
| L-187 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 66 | ~2.5 s per level | s (host tidset rebuild) | hardware-dependent | the host tidset rebuild, not the pair loop, cost ~2.5 s per level |
| L-188 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 67 | 1.34 s | s (deepk-density-auto, 2026-09-01 rerun) | hardware-dependent | `deepk-density-auto` runs in 1.34 s (== dense) with the identical signature |
| L-189 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 73 | 89.6 / 89.7 / 90.2 s | s (filter compact/cupy/cpu) | hardware-dependent | **Filter impls** (compact/cupy/cpu) indistinguishable at this scale (89.6/89.7/90.2 s) |
| L-190 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 74 | 2.4 GB | full-array D2H (cpu filter) | hardware-dependent | the cpu impl's full-array D2H is only 2.4 GB here |
| L-191 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 75 | 80 GB | D2H scale (not reachable on box) | method-parameter | structurally required at the 80 GB-D2H scale this box cannot reach |
| L-192 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 76 | 0.9 s vs 1.3 s | s (NCCL fallback vs NCCL) | hardware-dependent | **NCCL fallback** (staged D2D): works, 0.9 s vs 1.3 s with NCCL on tiny data |
| L-193 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 79 | 1.5 vs 1.4 s | s (nnz vs rows balance) | hardware-dependent | `nnz` shows no win over `rows` on the clustered preset (1.5 vs 1.4 s) |
| L-194 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 2 | b8a5032b3e2c492dbc9e185ec24f0aa6b0206b4f | git sha | software | $ git rev-parse HEAD → b8a5032b3e2c492dbc9e185ec24f0aa6b0206b4f |
| L-195 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 6 | Tue Sep  1 20:09:15 2026 | timestamp | hardware-dependent | nvidia-smi banner: Tue Sep  1 20:09:15 2026 |
| L-196 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 8 | 580.159.03 / CUDA 13.0 | driver / CUDA | software | NVIDIA-SMI 580.159.03 Driver Version: 580.159.03 CUDA Version: 13.0 |
| L-197 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 14 | NVIDIA GeForce RTX 3090 (GPU 0) | GPU model | hardware-dependent | 0  NVIDIA GeForce RTX 3090  On  00000000:01:00.0 Off |
| L-198 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 15 | 24576 | MiB VRAM | hardware-dependent | 1MiB / 24576MiB (49W / 360W, 46C, P5) |
| L-199 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 18 | NVIDIA GeForce RTX 3090 (GPU 1) | GPU model | hardware-dependent | 1  NVIDIA GeForce RTX 3090  On  00000000:82:00.0 Off |
| L-200 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 32 | (no output captured) | — | software | pip freeze section empty (file otherwise identical to 2026-08-31 env.txt) |
| L-201 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 3 | 13 ok / 0 failed | runs | deterministic | Runs: 13 ok, 0 failed/timeout. |
| L-202 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 9 | 1.3 | s median wall | hardware-dependent | deepk-density-auto \| deep_k \| legacy \| compact \| 2 \| 1 \| 1.3 \| 1.3 \| 408 |
| L-203 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 9 | 1.3 | s min wall | hardware-dependent | deepk-density-auto: min s 1.3 |
| L-204 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 9 | 408 | MB peak VRAM | hardware-dependent | deepk-density-auto: peak VRAM MB 408; throttled no |
| L-205 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 9 | gpus=2; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-density-auto \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-206 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 10 | 0.7 | s median wall | hardware-dependent | deepk-density-auto-1g \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.7 \| 0.7 \| 680 |
| L-207 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 10 | 0.7 | s min wall | hardware-dependent | deepk-density-auto-1g: min s 0.7 |
| L-208 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 10 | 680 | MB peak VRAM | hardware-dependent | deepk-density-auto-1g: peak VRAM MB 680; throttled no |
| L-209 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 10 | gpus=1; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-density-auto-1g \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-210 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 11 | 0.6 | s median wall | hardware-dependent | deepk-legacy-1g \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-211 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 11 | 0.6 | s min wall | hardware-dependent | deepk-legacy-1g: min s 0.6 |
| L-212 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 11 | 352 | MB peak VRAM | hardware-dependent | deepk-legacy-1g: peak VRAM MB 352; throttled yes |
| L-213 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 11 | gpus=1; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-legacy-1g \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-214 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 12 | 1.3 | s median wall | hardware-dependent | deepk-legacy-2g \| deep_k \| legacy \| compact \| 2 \| 1 \| 1.3 \| 1.3 \| 416 |
| L-215 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 12 | 1.3 | s min wall | hardware-dependent | deepk-legacy-2g: min s 1.3 |
| L-216 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 12 | 416 | MB peak VRAM | hardware-dependent | deepk-legacy-2g: peak VRAM MB 416; throttled yes |
| L-217 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 12 | gpus=2; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-legacy-2g \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-218 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 13 | 0.9 | s median wall | hardware-dependent | deepk-nonccl \| deep_k \| legacy \| compact \| 2 \| 1 \| 0.9 \| 0.9 \| 310 |
| L-219 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 13 | 0.9 | s min wall | hardware-dependent | deepk-nonccl: min s 0.9 |
| L-220 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 13 | 310 | MB peak VRAM | hardware-dependent | deepk-nonccl: peak VRAM MB 310; throttled yes |
| L-221 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 13 | gpus=2; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-nonccl \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-222 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 14 | 0.6 | s median wall | hardware-dependent | deepk-prefilter-off \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-223 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 14 | 0.6 | s min wall | hardware-dependent | deepk-prefilter-off: min s 0.6 |
| L-224 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 14 | 352 | MB peak VRAM | hardware-dependent | deepk-prefilter-off: peak VRAM MB 352; throttled yes |
| L-225 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 14 | gpus=1; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-prefilter-off \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-226 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 15 | 0.6 | s median wall | hardware-dependent | deepk-shared-1g \| deep_k \| shared \| compact \| 1 \| 3 \| 0.6 \| 0.6 \| 352 |
| L-227 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 15 | 0.6 | s min wall | hardware-dependent | deepk-shared-1g: min s 0.6 |
| L-228 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 15 | 352 | MB peak VRAM | hardware-dependent | deepk-shared-1g: peak VRAM MB 352; throttled yes |
| L-229 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 15 | gpus=1; reps=3; variant=shared; preset=deep_k | params | method-parameter | deepk-shared-1g \| deep_k \| shared \| compact \| 1 \| 3 |
| L-230 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 16 | 1.4 | s median wall | hardware-dependent | deepk-shared-2g \| deep_k \| shared \| compact \| 2 \| 3 \| 1.4 \| 1.3 \| 416 |
| L-231 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 16 | 1.3 | s min wall | hardware-dependent | deepk-shared-2g: min s 1.3 |
| L-232 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 16 | 416 | MB peak VRAM | hardware-dependent | deepk-shared-2g: peak VRAM MB 416; throttled yes |
| L-233 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 16 | gpus=2; reps=3; variant=shared; preset=deep_k | params | method-parameter | deepk-shared-2g \| deep_k \| shared \| compact \| 2 \| 3 |
| L-234 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 17 | 0.6 | s median wall | hardware-dependent | deepk-single-prefilter-on \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-235 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 17 | 0.6 | s min wall | hardware-dependent | deepk-single-prefilter-on: min s 0.6 |
| L-236 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 17 | 352 | MB peak VRAM | hardware-dependent | deepk-single-prefilter-on: peak VRAM MB 352; throttled yes |
| L-237 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 17 | gpus=1; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-single-prefilter-on \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-238 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 23 | 0.99× | speedup shared vs legacy (deep_k, 2 GPUs) | hardware-dependent | \| deep_k \| 2 \| 1.3 \| 1.4 \| 0.99× \| |
| L-239 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 27 | 1.382 | s (deepk-shared-2g#r2) | hardware-dependent | **deep_k** (deepk-shared-2g#r2, 1.382s): |
| L-240 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 31 | K1: candidates=112; frequent=112 | count | deterministic | deep_k per-level: \| 1 \| 112 \| 112 \| 25 \| |
| L-241 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 31 | 25 | ms (K=1, deepk-shared-2g#r2) | hardware-dependent | \| 1 \| 112 \| 112 \| 25 \| |
| L-242 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 32 | K2: candidates=0; frequent=471 | count | deterministic | deep_k per-level: \| 2 \| 0 \| 471 \| 50 \| |
| L-243 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 32 | 50 | ms (K=2, deepk-shared-2g#r2) | hardware-dependent | \| 2 \| 0 \| 471 \| 50 \| |
| L-244 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 33 | K3: candidates=0; frequent=901 | count | deterministic | deep_k per-level: \| 3 \| 0 \| 901 \| 7 \| |
| L-245 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 33 | 7 | ms (K=3, deepk-shared-2g#r2) | hardware-dependent | \| 3 \| 0 \| 901 \| 7 \| |
| L-246 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 34 | K4: candidates=0; frequent=1407 | count | deterministic | deep_k per-level: \| 4 \| 0 \| 1,407 \| 9 \| |
| L-247 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 34 | 9 | ms (K=4, deepk-shared-2g#r2) | hardware-dependent | \| 4 \| 0 \| 1,407 \| 9 \| |
| L-248 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 35 | K5: candidates=0; frequent=1854 | count | deterministic | deep_k per-level: \| 5 \| 0 \| 1,854 \| 6 \| |
| L-249 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 35 | 6 | ms (K=5, deepk-shared-2g#r2) | hardware-dependent | \| 5 \| 0 \| 1,854 \| 6 \| |
| L-250 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 36 | K6: candidates=0; frequent=1848 | count | deterministic | deep_k per-level: \| 6 \| 0 \| 1,848 \| 3 \| |
| L-251 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 36 | 3 | ms (K=6, deepk-shared-2g#r2) | hardware-dependent | \| 6 \| 0 \| 1,848 \| 3 \| |
| L-252 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 37 | K7: candidates=0; frequent=1320 | count | deterministic | deep_k per-level: \| 7 \| 0 \| 1,320 \| 3 \| |
| L-253 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 37 | 3 | ms (K=7, deepk-shared-2g#r2) | hardware-dependent | \| 7 \| 0 \| 1,320 \| 3 \| |
| L-254 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 38 | K8: candidates=0; frequent=660 | count | deterministic | deep_k per-level: \| 8 \| 0 \| 660 \| 2 \| |
| L-255 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 38 | 2 | ms (K=8, deepk-shared-2g#r2) | hardware-dependent | \| 8 \| 0 \| 660 \| 2 \| |
| L-256 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 39 | K9: candidates=0; frequent=220 | count | deterministic | deep_k per-level: \| 9 \| 0 \| 220 \| 2 \| |
| L-257 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 39 | 2 | ms (K=9, deepk-shared-2g#r2) | hardware-dependent | \| 9 \| 0 \| 220 \| 2 \| |
| L-258 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 40 | K10: candidates=0; frequent=44 | count | deterministic | deep_k per-level: \| 10 \| 0 \| 44 \| 2 \| |
| L-259 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 40 | 2 | ms (K=10, deepk-shared-2g#r2) | hardware-dependent | \| 10 \| 0 \| 44 \| 2 \| |
| L-260 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 41 | K11: candidates=0; frequent=4 | count | deterministic | deep_k per-level: \| 11 \| 0 \| 4 \| 2 \| |
| L-261 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 41 | 2 | ms (K=11, deepk-shared-2g#r2) | hardware-dependent | \| 11 \| 0 \| 4 \| 2 \| |
| L-262 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 1 | 2×RTX 3090; 2026-09-01 | hardware/date | hardware-dependent | # Sparse-CSR follow-up — 2×RTX 3090, 2026-09-01 |
| L-263 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 4 | 2× RTX 3090 24 GB, sm_86, driver 580.159.03, CuPy 14.1.1 | GPU/driver | hardware-dependent | Same box as the 2026-08-31 campaign (2× RTX 3090 24 GB, sm_86, driver 580.159.03, CuPy 14.1.1, NCCL over SHM) |
| L-264 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 11 | 9/9 | tier-equivalence legs | deterministic | Tier-equivalence chain 9/9 — now extended with single-GPU and multi-GPU sparse CSR legs |
| L-265 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 12 | sparse_from_k=3 | param (smoke preset) | method-parameter | (`sparse_from_k=3` on the smoke preset; CLAUDE.md) |
| L-266 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 13 | 13 | deep_k configs | deterministic | All 13 deep_k configs (dense legacy/shared × 1/2 GPUs, NCCL fallback, prefilter on/off, density-auto × 1/2 GPUs) |
| L-267 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 15 | 8841 | n_itemsets (deep_k) | deterministic | produce the campaign's exact signature: `n_itemsets=8841`, `sum_counts=266261275`, |
| L-268 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 15 | 266261275 | sum_counts (deep_k) | deterministic | `sum_counts=266261275` |
| L-269 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 16 | 4d2c8d28bcd33cd6… | itemset_hash prefix (deep_k) | deterministic | `itemset_hash=4d2c8d28bcd33cd6…` — bit-identical to dense mining. |
| L-270 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 26 | 10.4 s vs 1.3 s | s | hardware-dependent | The campaign measured `deepk-density-auto` at 10.4 s vs 1.3 s dense |
| L-271 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 28 | ~30 ms per level | ms (pair building) | hardware-dependent | pair building was ~30 ms per level, while ~2.5 s |
| L-272 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 29 | ~2.5 s per level (84%) | s / share (host tidset rebuild) | hardware-dependent | ~2.5 s per level (84%) went to rebuilding survivors' tidsets on the host |
| L-273 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 34 | 10.431 s → 1.343 s; legacy-2g 1.338 s; shared-2g 1.35 s; 1.00× | s / ratio (deepk-density-auto 2 GPUs) | hardware-dependent | \| `deepk-density-auto` (2 GPUs) \| 10.431 s \| **1.343 s** \| legacy-2g 1.338 s, shared-2g 1.35 s → **1.00×** \| |
| L-274 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 35 | 0.710 s; legacy-1g 0.584 s; 1.22× | s / ratio (deepk-density-auto-1g) | hardware-dependent | \| `deepk-density-auto-1g` (new) \| — \| **0.710 s** \| legacy-1g 0.584 s → 1.22× \| |
| L-275 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 37 | K=5 | sparse transition level (deep_k) | method-parameter | Per-level, 2 GPUs (ms; the sparse path runs from K=5): |
| L-276 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 41 | K5: frequent=1854; ms 08-31=3145; ms 09-01=48 (transition + first-use NVRTC compile); dense legacy-2g ms=3.6 | count / ms | hardware-dependent | \| 5 \| 1,854 \| 3145 \| 48 (transition + first-use NVRTC compile) \| 3.6 \| |
| L-277 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 42 | K6: frequent=1848; ms 08-31=2553; ms 09-01=4.1; dense legacy-2g ms=2.4 | count / ms | hardware-dependent | \| 6 \| 1,848 \| 2553 \| 4.1 \| 2.4 \| |
| L-278 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 43 | K7: frequent=1320; ms 08-31=1817; ms 09-01=3.5; dense legacy-2g ms=2.5 | count / ms | hardware-dependent | \| 7 \| 1,320 \| 1817 \| 3.5 \| 2.5 \| |
| L-279 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 44 | K8: frequent=660; ms 08-31=902; ms 09-01=3.0; dense legacy-2g ms=3.2 | count / ms | hardware-dependent | \| 8 \| 660 \| 902 \| 3.0 \| 3.2 \| |
| L-280 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 45 | K9: frequent=220; ms 08-31=374; ms 09-01=2.9; dense legacy-2g ms=2.4 | count / ms | hardware-dependent | \| 9 \| 220 \| 374 \| 2.9 \| 2.4 \| |
| L-281 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 46 | K10: frequent=44; ms 08-31=65; ms 09-01=2.7; dense legacy-2g ms=2.2 | count / ms | hardware-dependent | \| 10 \| 44 \| 65 \| 2.7 \| 2.2 \| |
| L-282 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 47 | K11: frequent=4; ms 08-31=14; ms 09-01=2.7; dense legacy-2g ms=1.9 | count / ms | hardware-dependent | \| 11 \| 4 \| 14 \| 2.7 \| 1.9 \| |
| L-283 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 49 | ~700× | per-level speedup vs host rebuild | hardware-dependent | Sparse levels are now single-digit milliseconds — ~700× faster per level than the host rebuild |
| L-284 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 51 | 406/408 MB (dense 414/416 MB) | MB peak VRAM | hardware-dependent | Peak VRAM 406/408 MB (dense 414/416 MB) |
| L-285 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 64 | 156 MB tidsets vs 168 MB bitvecs | MB at transition (deep_k) | deterministic | at deep_k scale it saves almost nothing (156 MB tidsets vs 168 MB bitvecs at the transition) |
| L-286 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 76 | ~11–15 ms sparse vs ~2–11 ms dense (K=6–9, 1 GPU) | ms per level | hardware-dependent | The single-GPU sparse levels cost ~11–15 ms each vs ~2–11 ms dense on one 3090 (K=6–9) |
| L-287 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 80 | 0.7 s | s wall (1-GPU density-auto) | hardware-dependent | Not worth tuning at a 0.7 s wall. |
| L-288 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 81 | 0.90 s vs 1.34 s | s (nonccl vs NCCL) | hardware-dependent | `deepk-nonccl` (0.90 s) is faster than NCCL (1.34 s) on this tiny data |
| L-289 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-legacy-1g#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-legacy-1g_r0.result.json" |
| L-290 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 2238.846 | s wall | hardware-dependent | stressk2-legacy-1g#r0: "status": "ok", "wall_s": 2238.846 |
| L-291 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | GPU0=17340; GPU1=4 | MB peak VRAM | hardware-dependent | stressk2-legacy-1g#r0: "peak_vram_mb": {"0": 17340, "1": 4} |
| L-292 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-legacy-1g#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-293 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 2996485 | n_itemsets | deterministic | stressk2-legacy-1g#r0: "motifs_ok": true, "n_itemsets": 2996485 |
| L-294 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 314055259 | sum_counts | deterministic | stressk2-legacy-1g#r0: "sum_counts": 314055259 |
| L-295 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 0b3e8434997fd3b1ec1e95357857b6d575988d0f4d32f478710738887828aca6 | sha256 itemset_hash | deterministic | stressk2-legacy-1g#r0: "itemset_hash": "0b3e8434997fd3b1…" |
| L-296 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | K1 cand=35000 freq=35000; K2 cand=612482500 freq=1660332; K3 cand=1301153 freq=1301153 | per-level candidates/frequent | deterministic | stressk2-legacy-1g#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-297 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | K1=71.9; K2=175580.2; K3=2046663.2 | ms per level | hardware-dependent | stressk2-legacy-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-298 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-legacy-1g#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-legacy-1g_r0.result.json" |
| L-299 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 0.603 | s wall | hardware-dependent | deepk-legacy-1g#r0: "status": "ok", "wall_s": 0.603 |
| L-300 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-legacy-1g#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-301 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-legacy-1g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-302 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 8841 | n_itemsets | deterministic | deepk-legacy-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-303 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 266261275 | sum_counts | deterministic | deepk-legacy-1g#r0: "sum_counts": 266261275 |
| L-304 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-legacy-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-305 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-legacy-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-306 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | K1=11.7; K2=3.4; K3=6.0; K4=7.9; K5=10.4; K6=12.3; K7=9.0; K8=5.0; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-legacy-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-307 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_FILTER_IMPL': 'compact'}; n_gpus=2; sparse_from_k=None; max_length=2; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "stressk2-filter-compact#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-filter-compact_r0.result.json" |
| L-308 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 89.571 | s wall | hardware-dependent | stressk2-filter-compact#r0: "status": "ok", "wall_s": 89.571 |
| L-309 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | GPU0=6918; GPU1=6918 | MB peak VRAM | hardware-dependent | stressk2-filter-compact#r0: "peak_vram_mb": {"0": 6918, "1": 6918} |
| L-310 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-filter-compact#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-311 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 1695332 | n_itemsets | deterministic | stressk2-filter-compact#r0: "motifs_ok": true, "n_itemsets": 1695332 |
| L-312 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 218250884 | sum_counts | deterministic | stressk2-filter-compact#r0: "sum_counts": 218250884 |
| L-313 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 8d989bfcc6e5c745f67108a9b76aacdbe7ae7929ae1df9a0b96a5d2f9dfba006 | sha256 itemset_hash | deterministic | stressk2-filter-compact#r0: "itemset_hash": "8d989bfcc6e5c745…" |
| L-314 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332 | per-level candidates/frequent | deterministic | stressk2-filter-compact#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-315 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | K1=40.9; K2=86424.3 | ms per level | hardware-dependent | stressk2-filter-compact#r0: per-level "ms" values (rounded to 0.1) |
| L-316 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_FILTER_IMPL': 'cupy'}; n_gpus=2; sparse_from_k=None; max_length=2; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "stressk2-filter-cupy#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-filter-cupy_r0.result.json" |
| L-317 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 89.665 | s wall | hardware-dependent | stressk2-filter-cupy#r0: "status": "ok", "wall_s": 89.665 |
| L-318 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | GPU0=6998; GPU1=6920 | MB peak VRAM | hardware-dependent | stressk2-filter-cupy#r0: "peak_vram_mb": {"0": 6998, "1": 6920} |
| L-319 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-filter-cupy#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-320 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 1695332 | n_itemsets | deterministic | stressk2-filter-cupy#r0: "motifs_ok": true, "n_itemsets": 1695332 |
| L-321 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 218250884 | sum_counts | deterministic | stressk2-filter-cupy#r0: "sum_counts": 218250884 |
| L-322 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 8d989bfcc6e5c745f67108a9b76aacdbe7ae7929ae1df9a0b96a5d2f9dfba006 | sha256 itemset_hash | deterministic | stressk2-filter-cupy#r0: "itemset_hash": "8d989bfcc6e5c745…" |
| L-323 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332 | per-level candidates/frequent | deterministic | stressk2-filter-cupy#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-324 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | K1=41.4; K2=86467.9 | ms per level | hardware-dependent | stressk2-filter-cupy#r0: per-level "ms" values (rounded to 0.1) |
| L-325 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_FILTER_IMPL': 'cpu'}; n_gpus=2; sparse_from_k=None; max_length=2; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "stressk2-filter-cpu#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-filter-cpu_r0.result.json" |
| L-326 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 90.196 | s wall | hardware-dependent | stressk2-filter-cpu#r0: "status": "ok", "wall_s": 90.196 |
| L-327 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | GPU0=6920; GPU1=6920 | MB peak VRAM | hardware-dependent | stressk2-filter-cpu#r0: "peak_vram_mb": {"0": 6920, "1": 6920} |
| L-328 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-filter-cpu#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-329 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 1695332 | n_itemsets | deterministic | stressk2-filter-cpu#r0: "motifs_ok": true, "n_itemsets": 1695332 |
| L-330 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 218250884 | sum_counts | deterministic | stressk2-filter-cpu#r0: "sum_counts": 218250884 |
| L-331 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 8d989bfcc6e5c745f67108a9b76aacdbe7ae7929ae1df9a0b96a5d2f9dfba006 | sha256 itemset_hash | deterministic | stressk2-filter-cpu#r0: "itemset_hash": "8d989bfcc6e5c745…" |
| L-332 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332 | per-level candidates/frequent | deterministic | stressk2-filter-cpu#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-333 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | K1=52.0; K2=86704.7 | ms per level | hardware-dependent | stressk2-filter-cpu#r0: per-level "ms" values (rounded to 0.1) |
| L-334 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_DISABLE_NCCL': '1'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-nonccl#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-nonccl_r0.result.json" |
| L-335 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | 0.863 | s wall | hardware-dependent | deepk-nonccl#r0: "status": "ok", "wall_s": 0.863 |
| L-336 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | GPU0=310; GPU1=310 | MB peak VRAM | hardware-dependent | deepk-nonccl#r0: "peak_vram_mb": {"0": 310, "1": 310} |
| L-337 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | (none) | nvml throttle flags | hardware-dependent | deepk-nonccl#r0: "throttle_reasons": [] |
| L-338 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | 8841 | n_itemsets | deterministic | deepk-nonccl#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-339 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | 266261275 | sum_counts | deterministic | deepk-nonccl#r0: "sum_counts": 266261275 |
| L-340 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-nonccl#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-341 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-nonccl#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-342 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | K1=20.4; K2=12.8; K3=4.3; K4=4.4; K5=3.1; K6=2.2; K7=2.3; K8=2.2; K9=1.8; K10=2.1; K11=1.7 | ms per level | hardware-dependent | deepk-nonccl#r0: per-level "ms" values (rounded to 0.1) |
| L-343 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=auto; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-density-auto#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-density-auto_r0.result.json" |
| L-344 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | 10.431 | s wall | hardware-dependent | deepk-density-auto#r0: "status": "ok", "wall_s": 10.431 |
| L-345 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | GPU0=416; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-density-auto#r0: "peak_vram_mb": {"0": 416, "1": 416} |
| L-346 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | (none) | nvml throttle flags | hardware-dependent | deepk-density-auto#r0: "throttle_reasons": [] |
| L-347 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | 8841 | n_itemsets | deterministic | deepk-density-auto#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-348 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | 266261275 | sum_counts | deterministic | deepk-density-auto#r0: "sum_counts": 266261275 |
| L-349 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-density-auto#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-350 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-density-auto#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-351 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | K1=26.5; K2=45.8; K3=5.3; K4=5.5; K5=3145.4; K6=2552.8; K7=1817.4; K8=902.1; K9=373.7; K10=65.2; K11=13.6 | ms per level | hardware-dependent | deepk-density-auto#r0: per-level "ms" values (rounded to 0.1) |
| L-352 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0', 'ET_MINER_DISABLE_PREFILTER': '1'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-prefilter-off#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-prefilter-off_r0.result.json" |
| L-353 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | 0.609 | s wall | hardware-dependent | deepk-prefilter-off#r0: "status": "ok", "wall_s": 0.609 |
| L-354 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-prefilter-off#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-355 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | (none) | nvml throttle flags | hardware-dependent | deepk-prefilter-off#r0: "throttle_reasons": [] |
| L-356 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | 8841 | n_itemsets | deterministic | deepk-prefilter-off#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-357 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | 266261275 | sum_counts | deterministic | deepk-prefilter-off#r0: "sum_counts": 266261275 |
| L-358 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-prefilter-off#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-359 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-prefilter-off#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-360 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | K1=11.4; K2=3.4; K3=6.0; K4=7.9; K5=12.4; K6=11.5; K7=9.0; K8=5.2; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-prefilter-off#r0: per-level "ms" values (rounded to 0.1) |
| L-361 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-single-prefilter-on#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-single-prefilter-on_r0.result.json" |
| L-362 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 0.592 | s wall | hardware-dependent | deepk-single-prefilter-on#r0: "status": "ok", "wall_s": 0.592 |
| L-363 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-single-prefilter-on#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-364 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-single-prefilter-on#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-365 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 8841 | n_itemsets | deterministic | deepk-single-prefilter-on#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-366 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 266261275 | sum_counts | deterministic | deepk-single-prefilter-on#r0: "sum_counts": 266261275 |
| L-367 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-single-prefilter-on#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-368 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-single-prefilter-on#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-369 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | K1=11.4; K2=2.9; K3=5.9; K4=7.8; K5=10.4; K6=12.2; K7=8.8; K8=4.9; K9=2.2; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-single-prefilter-on#r0: per-level "ms" values (rounded to 0.1) |
| L-370 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | preset=smoke; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=True; timeout_s=1800 | params | method-parameter | "id": "twophase-smoke#r0", "preset": "smoke" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/twophase-smoke_r0.result.json" |
| L-371 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | 1.582 | s wall | hardware-dependent | twophase-smoke#r0: "status": "ok", "wall_s": 1.582 |
| L-372 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | GPU0=408; GPU1=408 | MB peak VRAM | hardware-dependent | twophase-smoke#r0: "peak_vram_mb": {"0": 408, "1": 408} |
| L-373 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | twophase-smoke#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-374 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | 632 | n_itemsets | deterministic | twophase-smoke#r0: "motifs_ok": true, "n_itemsets": 632 |
| L-375 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | 1049580 | sum_counts | deterministic | twophase-smoke#r0: "sum_counts": 1049580 |
| L-376 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | ea17ea26fd0e44f735b62a5699d1227477965905ff34068203ac513aaee6deee | sha256 itemset_hash | deterministic | twophase-smoke#r0: "itemset_hash": "ea17ea26fd0e44f7…" |
| L-377 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | K1 cand=118 freq=118; K2 cand=0 freq=290; K3 cand=0 freq=202; K4 cand=0 freq=22; K5 cand=0 freq=0; K1 cand=118 freq=118; K2 cand=0 freq=290; K3 cand=0 freq=202; K4 cand=0 freq=22; K5 cand=0 freq=0 | per-level candidates/frequent | deterministic | twophase-smoke#r0: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-378 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | K1=251.7; K2=48.4; K3=119.5; K4=6.9; K5=0.1; K1=1.5; K2=33.9; K3=87.1; K4=5.1; K5=0.1 | ms per level | hardware-dependent | twophase-smoke#r0: per-level "ms" values (rounded to 0.1) |
| L-379 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | preset=skewed_rows; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_ROW_BALANCE': 'rows'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "skew-rows#r0", "preset": "skewed_rows" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/skew-rows_r0.result.json" |
| L-380 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | 1.435 | s wall | hardware-dependent | skew-rows#r0: "status": "ok", "wall_s": 1.435 |
| L-381 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | GPU0=416; GPU1=416 | MB peak VRAM | hardware-dependent | skew-rows#r0: "peak_vram_mb": {"0": 416, "1": 416} |
| L-382 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | (none) | nvml throttle flags | hardware-dependent | skew-rows#r0: "throttle_reasons": [] |
| L-383 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | 10350 | n_itemsets | deterministic | skew-rows#r0: "motifs_ok": true, "n_itemsets": 10350 |
| L-384 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | 346834073 | sum_counts | deterministic | skew-rows#r0: "sum_counts": 346834073 |
| L-385 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | 75262446a1c2b29bbc15e6d36afe23300984e6d46f8483511f055f1a510c2405 | sha256 itemset_hash | deterministic | skew-rows#r0: "itemset_hash": "75262446a1c2b29b…" |
| L-386 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | K1 cand=118 freq=118; K2 cand=0 freq=717; K3 cand=0 freq=1928; K4 cand=0 freq=2932; K5 cand=0 freq=2683; K6 cand=0 freq=1458; K7 cand=0 freq=443; K8 cand=0 freq=67; K9 cand=0 freq=4 | per-level candidates/frequent | deterministic | skew-rows#r0: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-387 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | K1=25.7; K2=48.0; K3=6.1; K4=14.7; K5=32.7; K6=41.6; K7=20.8; K8=5.9; K9=2.1 | ms per level | hardware-dependent | skew-rows#r0: per-level "ms" values (rounded to 0.1) |
| L-388 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | preset=skewed_rows; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_ROW_BALANCE': 'nnz'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "skew-nnz#r0", "preset": "skewed_rows" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/skew-nnz_r0.result.json" |
| L-389 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | 1.488 | s wall | hardware-dependent | skew-nnz#r0: "status": "ok", "wall_s": 1.488 |
| L-390 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | GPU0=414; GPU1=418 | MB peak VRAM | hardware-dependent | skew-nnz#r0: "peak_vram_mb": {"0": 414, "1": 418} |
| L-391 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | (none) | nvml throttle flags | hardware-dependent | skew-nnz#r0: "throttle_reasons": [] |
| L-392 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | 10350 | n_itemsets | deterministic | skew-nnz#r0: "motifs_ok": true, "n_itemsets": 10350 |
| L-393 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | 346834073 | sum_counts | deterministic | skew-nnz#r0: "sum_counts": 346834073 |
| L-394 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | 75262446a1c2b29bbc15e6d36afe23300984e6d46f8483511f055f1a510c2405 | sha256 itemset_hash | deterministic | skew-nnz#r0: "itemset_hash": "75262446a1c2b29b…" |
| L-395 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | K1 cand=118 freq=118; K2 cand=0 freq=717; K3 cand=0 freq=1928; K4 cand=0 freq=2932; K5 cand=0 freq=2683; K6 cand=0 freq=1458; K7 cand=0 freq=443; K8 cand=0 freq=67; K9 cand=0 freq=4 | per-level candidates/frequent | deterministic | skew-nnz#r0: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-396 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | K1=26.6; K2=48.2; K3=6.2; K4=16.0; K5=33.7; K6=40.6; K7=22.8; K8=5.5; K9=2.1 | ms per level | hardware-dependent | skew-nnz#r0: per-level "ms" values (rounded to 0.1) |
| L-397 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | preset=skewed_rows; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_ROW_BALANCE': 'rows'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "skew-rows#r1", "preset": "skewed_rows" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/skew-rows_r1.result.json" |
| L-398 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | 1.441 | s wall | hardware-dependent | skew-rows#r1: "status": "ok", "wall_s": 1.441 |
| L-399 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | GPU0=416; GPU1=416 | MB peak VRAM | hardware-dependent | skew-rows#r1: "peak_vram_mb": {"0": 416, "1": 416} |
| L-400 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | (none) | nvml throttle flags | hardware-dependent | skew-rows#r1: "throttle_reasons": [] |
| L-401 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | 10350 | n_itemsets | deterministic | skew-rows#r1: "motifs_ok": true, "n_itemsets": 10350 |
| L-402 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | 346834073 | sum_counts | deterministic | skew-rows#r1: "sum_counts": 346834073 |
| L-403 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | 75262446a1c2b29bbc15e6d36afe23300984e6d46f8483511f055f1a510c2405 | sha256 itemset_hash | deterministic | skew-rows#r1: "itemset_hash": "75262446a1c2b29b…" |
| L-404 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | K1 cand=118 freq=118; K2 cand=0 freq=717; K3 cand=0 freq=1928; K4 cand=0 freq=2932; K5 cand=0 freq=2683; K6 cand=0 freq=1458; K7 cand=0 freq=443; K8 cand=0 freq=67; K9 cand=0 freq=4 | per-level candidates/frequent | deterministic | skew-rows#r1: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-405 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | K1=26.4; K2=49.0; K3=6.7; K4=15.4; K5=32.7; K6=38.5; K7=23.9; K8=5.1; K9=2.1 | ms per level | hardware-dependent | skew-rows#r1: per-level "ms" values (rounded to 0.1) |
| L-406 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | preset=skewed_rows; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_ROW_BALANCE': 'nnz'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "skew-nnz#r1", "preset": "skewed_rows" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/skew-nnz_r1.result.json" |
| L-407 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | 1.517 | s wall | hardware-dependent | skew-nnz#r1: "status": "ok", "wall_s": 1.517 |
| L-408 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | GPU0=414; GPU1=418 | MB peak VRAM | hardware-dependent | skew-nnz#r1: "peak_vram_mb": {"0": 414, "1": 418} |
| L-409 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | (none) | nvml throttle flags | hardware-dependent | skew-nnz#r1: "throttle_reasons": [] |
| L-410 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | 10350 | n_itemsets | deterministic | skew-nnz#r1: "motifs_ok": true, "n_itemsets": 10350 |
| L-411 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | 346834073 | sum_counts | deterministic | skew-nnz#r1: "sum_counts": 346834073 |
| L-412 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | 75262446a1c2b29bbc15e6d36afe23300984e6d46f8483511f055f1a510c2405 | sha256 itemset_hash | deterministic | skew-nnz#r1: "itemset_hash": "75262446a1c2b29b…" |
| L-413 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | K1 cand=118 freq=118; K2 cand=0 freq=717; K3 cand=0 freq=1928; K4 cand=0 freq=2932; K5 cand=0 freq=2683; K6 cand=0 freq=1458; K7 cand=0 freq=443; K8 cand=0 freq=67; K9 cand=0 freq=4 | per-level candidates/frequent | deterministic | skew-nnz#r1: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-414 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | K1=25.7; K2=46.6; K3=6.6; K4=16.5; K5=34.0; K6=39.9; K7=21.9; K8=5.0; K9=2.0 | ms per level | hardware-dependent | skew-nnz#r1: per-level "ms" values (rounded to 0.1) |
| L-415 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-legacy-2g#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-legacy-2g_r0.result.json" |
| L-416 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | 1996.726 | s wall | hardware-dependent | stressk2-legacy-2g#r0: "status": "ok", "wall_s": 1996.726 |
| L-417 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | GPU0=16576; GPU1=16576 | MB peak VRAM | hardware-dependent | stressk2-legacy-2g#r0: "peak_vram_mb": {"0": 16576, "1": 16576} |
| L-418 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | 0x0000000000000001, 0x0000000000000004, 0x0000000000000020, 0x0000000000000024 | nvml throttle flags | hardware-dependent | stressk2-legacy-2g#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004", "0x0000000000000020", "0x0000000000000024"] |
| L-419 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | 3005770 | n_itemsets | deterministic | stressk2-legacy-2g#r0: "motifs_ok": true, "n_itemsets": 3005770 |
| L-420 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | 314350393 | sum_counts | deterministic | stressk2-legacy-2g#r0: "sum_counts": 314350393 |
| L-421 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-legacy-2g#r0: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-422 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332; K3 cand=0 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-legacy-2g#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-423 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | K1=42.5; K2=86439.6; K3=1906847.2 | ms per level | hardware-dependent | stressk2-legacy-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-424 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-1g#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-1g_r0.result.json" |
| L-425 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | 255.303 | s wall | hardware-dependent | stressk2-shared-1g#r0: "status": "ok", "wall_s": 255.303 |
| L-426 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | GPU0=17340; GPU1=4 | MB peak VRAM | hardware-dependent | stressk2-shared-1g#r0: "peak_vram_mb": {"0": 17340, "1": 4} |
| L-427 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | 0x0000000000000001, 0x0000000000000004, 0x0000000000000020, 0x0000000000000024 | nvml throttle flags | hardware-dependent | stressk2-shared-1g#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004", "0x0000000000000020", "0x0000000000000024"] |
| L-428 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | 3005770 | n_itemsets | deterministic | stressk2-shared-1g#r0: "motifs_ok": true, "n_itemsets": 3005770 |
| L-429 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | 314350393 | sum_counts | deterministic | stressk2-shared-1g#r0: "sum_counts": 314350393 |
| L-430 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-1g#r0: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-431 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | K1 cand=35000 freq=35000; K2 cand=612482500 freq=1660332; K3 cand=1310438 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-1g#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-432 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | K1=73.3; K2=18561.1; K3=220167.6 | ms per level | hardware-dependent | stressk2-shared-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-433 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-1g_r0.result.json" |
| L-434 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 0.627 | s wall | hardware-dependent | deepk-shared-1g#r0: "status": "ok", "wall_s": 0.627 |
| L-435 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-436 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-1g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-437 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-438 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r0: "sum_counts": 266261275 |
| L-439 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-440 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-441 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | K1=11.4; K2=6.3; K3=5.3; K4=7.6; K5=11.2; K6=15.0; K7=13.2; K8=9.2; K9=4.7; K10=3.1; K11=2.9 | ms per level | hardware-dependent | deepk-shared-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-442 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-2g#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-2g_r0.result.json" |
| L-443 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | 122.814 | s wall | hardware-dependent | stressk2-shared-2g#r0: "status": "ok", "wall_s": 122.814 |
| L-444 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | GPU0=16576; GPU1=16576 | MB peak VRAM | hardware-dependent | stressk2-shared-2g#r0: "peak_vram_mb": {"0": 16576, "1": 16576} |
| L-445 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-2g#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-446 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | 3005770 | n_itemsets | deterministic | stressk2-shared-2g#r0: "motifs_ok": true, "n_itemsets": 3005770 |
| L-447 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | 314350393 | sum_counts | deterministic | stressk2-shared-2g#r0: "sum_counts": 314350393 |
| L-448 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-2g#r0: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-449 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332; K3 cand=0 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-2g#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-450 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | K1=45.2; K2=7215.2; K3=111994.4 | ms per level | hardware-dependent | stressk2-shared-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-451 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-legacy-2g#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-legacy-2g_r0.result.json" |
| L-452 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | 1.322 | s wall | hardware-dependent | deepk-legacy-2g#r0: "status": "ok", "wall_s": 1.322 |
| L-453 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-legacy-2g#r0: "peak_vram_mb": {"0": 414, "1": 416} |
| L-454 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | (none) | nvml throttle flags | hardware-dependent | deepk-legacy-2g#r0: "throttle_reasons": [] |
| L-455 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | 8841 | n_itemsets | deterministic | deepk-legacy-2g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-456 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | 266261275 | sum_counts | deterministic | deepk-legacy-2g#r0: "sum_counts": 266261275 |
| L-457 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-legacy-2g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-458 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-legacy-2g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-459 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | K1=26.0; K2=44.2; K3=6.0; K4=5.5; K5=4.2; K6=2.6; K7=2.4; K8=2.2; K9=2.2; K10=2.0; K11=2.0 | ms per level | hardware-dependent | deepk-legacy-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-460 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-2g_r0.result.json" |
| L-461 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | 1.391 | s wall | hardware-dependent | deepk-shared-2g#r0: "status": "ok", "wall_s": 1.391 |
| L-462 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r0: "peak_vram_mb": {"0": 414, "1": 416} |
| L-463 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-2g#r0: "throttle_reasons": [] |
| L-464 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-465 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r0: "sum_counts": 266261275 |
| L-466 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-467 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-468 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | K1=26.4; K2=50.6; K3=6.7; K4=8.3; K5=6.0; K6=2.7; K7=2.5; K8=2.8; K9=2.1; K10=4.5; K11=2.0 | ms per level | hardware-dependent | deepk-shared-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-469 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-1g#r1", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-1g_r1.result.json" |
| L-470 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | 253.074 | s wall | hardware-dependent | stressk2-shared-1g#r1: "status": "ok", "wall_s": 253.074 |
| L-471 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | GPU0=17340; GPU1=4 | MB peak VRAM | hardware-dependent | stressk2-shared-1g#r1: "peak_vram_mb": {"0": 17340, "1": 4} |
| L-472 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-1g#r1: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-473 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | 3005770 | n_itemsets | deterministic | stressk2-shared-1g#r1: "motifs_ok": true, "n_itemsets": 3005770 |
| L-474 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | 314350393 | sum_counts | deterministic | stressk2-shared-1g#r1: "sum_counts": 314350393 |
| L-475 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-1g#r1: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-476 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | K1 cand=35000 freq=35000; K2 cand=612482500 freq=1660332; K3 cand=1310438 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-1g#r1: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-477 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | K1=72.9; K2=18099.5; K3=218647.6 | ms per level | hardware-dependent | stressk2-shared-1g#r1: per-level "ms" values (rounded to 0.1) |
| L-478 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r1", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-1g_r1.result.json" |
| L-479 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 0.639 | s wall | hardware-dependent | deepk-shared-1g#r1: "status": "ok", "wall_s": 0.639 |
| L-480 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r1: "peak_vram_mb": {"0": 352, "1": 4} |
| L-481 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-1g#r1: "throttle_reasons": ["0x0000000000000001"] |
| L-482 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r1: "motifs_ok": true, "n_itemsets": 8841 |
| L-483 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r1: "sum_counts": 266261275 |
| L-484 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r1: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-485 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r1: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-486 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | K1=11.4; K2=6.3; K3=5.3; K4=7.7; K5=11.4; K6=14.1; K7=13.3; K8=9.2; K9=4.7; K10=3.1; K11=3.0 | ms per level | hardware-dependent | deepk-shared-1g#r1: per-level "ms" values (rounded to 0.1) |
| L-487 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-2g#r1", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-2g_r1.result.json" |
| L-488 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | 122.257 | s wall | hardware-dependent | stressk2-shared-2g#r1: "status": "ok", "wall_s": 122.257 |
| L-489 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | GPU0=16576; GPU1=16576 | MB peak VRAM | hardware-dependent | stressk2-shared-2g#r1: "peak_vram_mb": {"0": 16576, "1": 16576} |
| L-490 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-2g#r1: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-491 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | 3005770 | n_itemsets | deterministic | stressk2-shared-2g#r1: "motifs_ok": true, "n_itemsets": 3005770 |
| L-492 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | 314350393 | sum_counts | deterministic | stressk2-shared-2g#r1: "sum_counts": 314350393 |
| L-493 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-2g#r1: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-494 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332; K3 cand=0 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-2g#r1: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-495 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | K1=54.9; K2=7204.6; K3=111955.2 | ms per level | hardware-dependent | stressk2-shared-2g#r1: per-level "ms" values (rounded to 0.1) |
| L-496 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r1", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-2g_r1.result.json" |
| L-497 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | 1.316 | s wall | hardware-dependent | deepk-shared-2g#r1: "status": "ok", "wall_s": 1.316 |
| L-498 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r1: "peak_vram_mb": {"0": 414, "1": 416} |
| L-499 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-2g#r1: "throttle_reasons": [] |
| L-500 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r1: "motifs_ok": true, "n_itemsets": 8841 |
| L-501 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r1: "sum_counts": 266261275 |
| L-502 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r1: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-503 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r1: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-504 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | K1=26.3; K2=48.9; K3=6.2; K4=9.3; K5=5.4; K6=2.5; K7=2.5; K8=2.2; K9=2.0; K10=2.0; K11=2.2 | ms per level | hardware-dependent | deepk-shared-2g#r1: per-level "ms" values (rounded to 0.1) |
| L-505 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-1g#r2", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-1g_r2.result.json" |
| L-506 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | 253.285 | s wall | hardware-dependent | stressk2-shared-1g#r2: "status": "ok", "wall_s": 253.285 |
| L-507 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | GPU0=17340; GPU1=4 | MB peak VRAM | hardware-dependent | stressk2-shared-1g#r2: "peak_vram_mb": {"0": 17340, "1": 4} |
| L-508 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-1g#r2: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-509 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | 3005770 | n_itemsets | deterministic | stressk2-shared-1g#r2: "motifs_ok": true, "n_itemsets": 3005770 |
| L-510 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | 314350393 | sum_counts | deterministic | stressk2-shared-1g#r2: "sum_counts": 314350393 |
| L-511 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-1g#r2: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-512 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | K1 cand=35000 freq=35000; K2 cand=612482500 freq=1660332; K3 cand=1310438 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-1g#r2: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-513 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | K1=73.9; K2=18136.5; K3=218798.1 | ms per level | hardware-dependent | stressk2-shared-1g#r2: per-level "ms" values (rounded to 0.1) |
| L-514 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r2", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-1g_r2.result.json" |
| L-515 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 0.603 | s wall | hardware-dependent | deepk-shared-1g#r2: "status": "ok", "wall_s": 0.603 |
| L-516 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r2: "peak_vram_mb": {"0": 352, "1": 4} |
| L-517 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-1g#r2: "throttle_reasons": ["0x0000000000000001"] |
| L-518 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r2: "motifs_ok": true, "n_itemsets": 8841 |
| L-519 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r2: "sum_counts": 266261275 |
| L-520 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r2: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-521 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r2: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-522 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | K1=11.3; K2=6.3; K3=5.3; K4=7.6; K5=11.2; K6=13.9; K7=13.2; K8=9.2; K9=4.7; K10=3.1; K11=3.0 | ms per level | hardware-dependent | deepk-shared-1g#r2: per-level "ms" values (rounded to 0.1) |
| L-523 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-2g#r2", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-2g_r2.result.json" |
| L-524 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | 122.333 | s wall | hardware-dependent | stressk2-shared-2g#r2: "status": "ok", "wall_s": 122.333 |
| L-525 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | GPU0=16576; GPU1=16576 | MB peak VRAM | hardware-dependent | stressk2-shared-2g#r2: "peak_vram_mb": {"0": 16576, "1": 16576} |
| L-526 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-2g#r2: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-527 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | 3005770 | n_itemsets | deterministic | stressk2-shared-2g#r2: "motifs_ok": true, "n_itemsets": 3005770 |
| L-528 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | 314350393 | sum_counts | deterministic | stressk2-shared-2g#r2: "sum_counts": 314350393 |
| L-529 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-2g#r2: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-530 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332; K3 cand=0 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-2g#r2: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-531 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | K1=47.2; K2=7198.9; K3=111923.4 | ms per level | hardware-dependent | stressk2-shared-2g#r2: per-level "ms" values (rounded to 0.1) |
| L-532 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r2", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-2g_r2.result.json" |
| L-533 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | 1.361 | s wall | hardware-dependent | deepk-shared-2g#r2: "status": "ok", "wall_s": 1.361 |
| L-534 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r2: "peak_vram_mb": {"0": 414, "1": 416} |
| L-535 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-2g#r2: "throttle_reasons": [] |
| L-536 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r2: "motifs_ok": true, "n_itemsets": 8841 |
| L-537 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r2: "sum_counts": 266261275 |
| L-538 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r2: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-539 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r2: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-540 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | K1=26.4; K2=49.2; K3=6.8; K4=8.4; K5=5.6; K6=2.9; K7=2.5; K8=2.3; K9=2.1; K10=2.2; K11=2.1 | ms per level | hardware-dependent | deepk-shared-2g#r2: per-level "ms" values (rounded to 0.1) |
| L-541 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_DISABLE_NCCL': '1'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-nonccl#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-nonccl_r0.result.json" |
| L-542 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 0.901 | s wall | hardware-dependent | deepk-nonccl#r0: "status": "ok", "wall_s": 0.901 |
| L-543 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | GPU0=302; GPU1=310 | MB peak VRAM | hardware-dependent | deepk-nonccl#r0: "peak_vram_mb": {"0": 302, "1": 310} |
| L-544 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-nonccl#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-545 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 8841 | n_itemsets | deterministic | deepk-nonccl#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-546 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 266261275 | sum_counts | deterministic | deepk-nonccl#r0: "sum_counts": 266261275 |
| L-547 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-nonccl#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-548 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-nonccl#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-549 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | K1=22.3; K2=12.5; K3=5.1; K4=3.9; K5=2.8; K6=2.1; K7=2.0; K8=1.8; K9=1.6; K10=1.8; K11=1.7 | ms per level | hardware-dependent | deepk-nonccl#r0: per-level "ms" values (rounded to 0.1) |
| L-550 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=auto; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-density-auto#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-density-auto_r0.result.json" |
| L-551 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | 1.343 | s wall | hardware-dependent | deepk-density-auto#r0: "status": "ok", "wall_s": 1.343 |
| L-552 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | GPU0=406; GPU1=408 | MB peak VRAM | hardware-dependent | deepk-density-auto#r0: "peak_vram_mb": {"0": 406, "1": 408} |
| L-553 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | (none) | nvml throttle flags | hardware-dependent | deepk-density-auto#r0: "throttle_reasons": [] |
| L-554 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | 8841 | n_itemsets | deterministic | deepk-density-auto#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-555 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | 266261275 | sum_counts | deterministic | deepk-density-auto#r0: "sum_counts": 266261275 |
| L-556 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-density-auto#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-557 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=2251 freq=1854; K6 cand=1855 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-density-auto#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-558 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | K1=25.7; K2=48.8; K3=5.6; K4=5.3; K5=48.1; K6=4.1; K7=3.5; K8=3.0; K9=2.9; K10=2.7; K11=2.7 | ms per level | hardware-dependent | deepk-density-auto#r0: per-level "ms" values (rounded to 0.1) |
| L-559 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=auto; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-density-auto-1g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-density-auto-1g_r0.result.json" |
| L-560 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | 0.71 | s wall | hardware-dependent | deepk-density-auto-1g#r0: "status": "ok", "wall_s": 0.71 |
| L-561 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | GPU0=680; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-density-auto-1g#r0: "peak_vram_mb": {"0": 680, "1": 4} |
| L-562 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | (none) | nvml throttle flags | hardware-dependent | deepk-density-auto-1g#r0: "throttle_reasons": [] |
| L-563 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | 8841 | n_itemsets | deterministic | deepk-density-auto-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-564 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | 266261275 | sum_counts | deterministic | deepk-density-auto-1g#r0: "sum_counts": 266261275 |
| L-565 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-density-auto-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-566 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=2251 freq=1854; K6 cand=1855 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-density-auto-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-567 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | K1=11.4; K2=2.9; K3=6.5; K4=7.9; K5=74.1; K6=14.7; K7=13.4; K8=11.1; K9=10.8; K10=8.9; K11=2.8 | ms per level | hardware-dependent | deepk-density-auto-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-568 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-prefilter-off#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-prefilter-off_r0.result.json" |
| L-569 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 0.59 | s wall | hardware-dependent | deepk-prefilter-off#r0: "status": "ok", "wall_s": 0.59 |
| L-570 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-prefilter-off#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-571 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-prefilter-off#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-572 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 8841 | n_itemsets | deterministic | deepk-prefilter-off#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-573 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 266261275 | sum_counts | deterministic | deepk-prefilter-off#r0: "sum_counts": 266261275 |
| L-574 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-prefilter-off#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-575 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-prefilter-off#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-576 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | K1=11.4; K2=3.0; K3=6.0; K4=8.2; K5=10.5; K6=11.2; K7=9.6; K8=5.0; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-prefilter-off#r0: per-level "ms" values (rounded to 0.1) |
| L-577 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0', 'ET_MINER_ENABLE_PREFILTER': '1'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-single-prefilter-on#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-single-prefilter-on_r0.result.json" |
| L-578 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 0.595 | s wall | hardware-dependent | deepk-single-prefilter-on#r0: "status": "ok", "wall_s": 0.595 |
| L-579 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-single-prefilter-on#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-580 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-single-prefilter-on#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-581 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 8841 | n_itemsets | deterministic | deepk-single-prefilter-on#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-582 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 266261275 | sum_counts | deterministic | deepk-single-prefilter-on#r0: "sum_counts": 266261275 |
| L-583 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-single-prefilter-on#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-584 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-single-prefilter-on#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-585 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | K1=11.4; K2=2.9; K3=6.0; K4=7.9; K5=10.3; K6=12.4; K7=9.1; K8=4.9; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-single-prefilter-on#r0: per-level "ms" values (rounded to 0.1) |
| L-586 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-legacy-1g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-legacy-1g_r0.result.json" |
| L-587 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 0.584 | s wall | hardware-dependent | deepk-legacy-1g#r0: "status": "ok", "wall_s": 0.584 |
| L-588 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-legacy-1g#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-589 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-legacy-1g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-590 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 8841 | n_itemsets | deterministic | deepk-legacy-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-591 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 266261275 | sum_counts | deterministic | deepk-legacy-1g#r0: "sum_counts": 266261275 |
| L-592 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-legacy-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-593 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-legacy-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-594 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | K1=11.4; K2=2.9; K3=6.1; K4=8.3; K5=10.4; K6=11.2; K7=10.1; K8=4.9; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-legacy-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-595 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-1g_r0.result.json" |
| L-596 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 0.613 | s wall | hardware-dependent | deepk-shared-1g#r0: "status": "ok", "wall_s": 0.613 |
| L-597 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-598 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-1g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-599 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-600 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r0: "sum_counts": 266261275 |
| L-601 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-602 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-603 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | K1=11.4; K2=6.3; K3=5.3; K4=7.7; K5=11.2; K6=14.1; K7=13.2; K8=9.2; K9=4.7; K10=3.1; K11=2.9 | ms per level | hardware-dependent | deepk-shared-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-604 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-legacy-2g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-legacy-2g_r0.result.json" |
| L-605 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 1.338 | s wall | hardware-dependent | deepk-legacy-2g#r0: "status": "ok", "wall_s": 1.338 |
| L-606 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-legacy-2g#r0: "peak_vram_mb": {"0": 414, "1": 416} |
| L-607 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-legacy-2g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-608 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 8841 | n_itemsets | deterministic | deepk-legacy-2g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-609 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 266261275 | sum_counts | deterministic | deepk-legacy-2g#r0: "sum_counts": 266261275 |
| L-610 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-legacy-2g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-611 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-legacy-2g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-612 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | K1=27.9; K2=46.2; K3=5.8; K4=5.4; K5=3.6; K6=2.4; K7=2.5; K8=3.2; K9=2.4; K10=2.2; K11=1.9 | ms per level | hardware-dependent | deepk-legacy-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-613 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-2g_r0.result.json" |
| L-614 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | 1.354 | s wall | hardware-dependent | deepk-shared-2g#r0: "status": "ok", "wall_s": 1.354 |
| L-615 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r0: "peak_vram_mb": {"0": 414, "1": 416} |
| L-616 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-2g#r0: "throttle_reasons": [] |
| L-617 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-618 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r0: "sum_counts": 266261275 |
| L-619 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-620 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-621 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | K1=26.4; K2=49.1; K3=7.2; K4=9.1; K5=5.4; K6=2.5; K7=2.6; K8=2.3; K9=2.6; K10=1.9; K11=1.9 | ms per level | hardware-dependent | deepk-shared-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-622 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r1", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-1g_r1.result.json" |
| L-623 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | 0.628 | s wall | hardware-dependent | deepk-shared-1g#r1: "status": "ok", "wall_s": 0.628 |
| L-624 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r1: "peak_vram_mb": {"0": 352, "1": 4} |
| L-625 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-1g#r1: "throttle_reasons": [] |
| L-626 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r1: "motifs_ok": true, "n_itemsets": 8841 |
| L-627 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r1: "sum_counts": 266261275 |
| L-628 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r1: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-629 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r1: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-630 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | K1=11.4; K2=6.3; K3=5.3; K4=7.6; K5=11.2; K6=14.1; K7=13.2; K8=9.3; K9=4.7; K10=3.1; K11=2.9 | ms per level | hardware-dependent | deepk-shared-1g#r1: per-level "ms" values (rounded to 0.1) |
| L-631 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r1", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-2g_r1.result.json" |
| L-632 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 1.346 | s wall | hardware-dependent | deepk-shared-2g#r1: "status": "ok", "wall_s": 1.346 |
| L-633 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r1: "peak_vram_mb": {"0": 414, "1": 416} |
| L-634 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-2g#r1: "throttle_reasons": ["0x0000000000000001"] |
| L-635 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r1: "motifs_ok": true, "n_itemsets": 8841 |
| L-636 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r1: "sum_counts": 266261275 |
| L-637 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r1: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-638 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r1: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-639 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | K1=26.3; K2=48.8; K3=7.6; K4=9.5; K5=5.4; K6=2.6; K7=2.5; K8=2.2; K9=1.9; K10=2.2; K11=2.0 | ms per level | hardware-dependent | deepk-shared-2g#r1: per-level "ms" values (rounded to 0.1) |
| L-640 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r2", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-1g_r2.result.json" |
| L-641 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | 0.602 | s wall | hardware-dependent | deepk-shared-1g#r2: "status": "ok", "wall_s": 0.602 |
| L-642 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r2: "peak_vram_mb": {"0": 352, "1": 4} |
| L-643 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-1g#r2: "throttle_reasons": [] |
| L-644 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r2: "motifs_ok": true, "n_itemsets": 8841 |
| L-645 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r2: "sum_counts": 266261275 |
| L-646 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r2: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-647 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r2: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-648 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | K1=11.5; K2=6.4; K3=5.3; K4=7.7; K5=11.3; K6=14.1; K7=13.5; K8=9.2; K9=4.7; K10=3.1; K11=3.0 | ms per level | hardware-dependent | deepk-shared-1g#r2: per-level "ms" values (rounded to 0.1) |
| L-649 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r2", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-2g_r2.result.json" |
| L-650 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 1.382 | s wall | hardware-dependent | deepk-shared-2g#r2: "status": "ok", "wall_s": 1.382 |
| L-651 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r2: "peak_vram_mb": {"0": 414, "1": 416} |
| L-652 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-2g#r2: "throttle_reasons": ["0x0000000000000001"] |
| L-653 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r2: "motifs_ok": true, "n_itemsets": 8841 |
| L-654 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r2: "sum_counts": 266261275 |
| L-655 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r2: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-656 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r2: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-657 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | K1=25.3; K2=50.1; K3=6.8; K4=8.5; K5=5.5; K6=2.5; K7=2.5; K8=2.2; K9=2.1; K10=2.0; K11=2.4 | ms per level | hardware-dependent | deepk-shared-2g#r2: per-level "ms" values (rounded to 0.1) |
| L-658 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 2 | direct_gpu_vs_son | experiment id | method-parameter | "experiment": "direct_gpu_vs_son" |
| L-659 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 3 | 2026-02-19T05:03:26.988464+00:00 | timestamp | software | "timestamp": "2026-02-19T05:03:26.988464+00:00" |
| L-660 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 5 | 1e-05 | min_support (fraction) | method-parameter | "min_support": 1e-05 |
| L-661 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 6 | 0.001% | support_pct | method-parameter | "support_pct": "0.001%" |
| L-662 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 7 | 76890945 | n_transactions (proteins) | deterministic | "n_transactions": 76890945 |
| L-663 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 8 | 768 | min_count | method-parameter | "min_count": 768 |
| L-664 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 9 | null | max_length | method-parameter | "max_length": null |
| L-665 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 10 | true | use_gpu | method-parameter | "use_gpu": true |
| L-666 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 13 | 475865 | itemsets (direct GPU) | deterministic | "direct_gpu_result": { "itemsets": 475865 |
| L-667 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 14 | 50.72 | s (direct GPU time) | hardware-dependent | "time_seconds": 50.72 |
| L-668 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 15 | 14 | max_k (direct GPU) | deterministic | "max_k": 14 |
| L-669 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 17 | 1002 | itemsets at K=1 (direct GPU, 0.001%) | deterministic | "k_distribution": … "1": 1002 |
| L-670 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 18 | 22019 | itemsets at K=2 (direct GPU, 0.001%) | deterministic | "k_distribution": … "2": 22019 |
| L-671 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 19 | 73205 | itemsets at K=3 (direct GPU, 0.001%) | deterministic | "k_distribution": … "3": 73205 |
| L-672 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 20 | 108059 | itemsets at K=4 (direct GPU, 0.001%) | deterministic | "k_distribution": … "4": 108059 |
| L-673 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 21 | 104239 | itemsets at K=5 (direct GPU, 0.001%) | deterministic | "k_distribution": … "5": 104239 |
| L-674 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 22 | 78596 | itemsets at K=6 (direct GPU, 0.001%) | deterministic | "k_distribution": … "6": 78596 |
| L-675 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 23 | 48699 | itemsets at K=7 (direct GPU, 0.001%) | deterministic | "k_distribution": … "7": 48699 |
| L-676 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 24 | 25011 | itemsets at K=8 (direct GPU, 0.001%) | deterministic | "k_distribution": … "8": 25011 |
| L-677 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 25 | 10508 | itemsets at K=9 (direct GPU, 0.001%) | deterministic | "k_distribution": … "9": 10508 |
| L-678 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 26 | 3488 | itemsets at K=10 (direct GPU, 0.001%) | deterministic | "k_distribution": … "10": 3488 |
| L-679 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 27 | 869 | itemsets at K=11 (direct GPU, 0.001%) | deterministic | "k_distribution": … "11": 869 |
| L-680 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 28 | 152 | itemsets at K=12 (direct GPU, 0.001%) | deterministic | "k_distribution": … "12": 152 |
| L-681 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 29 | 17 | itemsets at K=13 (direct GPU, 0.001%) | deterministic | "k_distribution": … "13": 17 |
| L-682 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 30 | 1 | itemsets at K=14 (direct GPU, 0.001%) | deterministic | "k_distribution": … "14": 1 |
| L-683 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 34 | SON (Power) | method | method-parameter | "son_reference": { "method": "SON (Power)" |
| L-684 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 35 | 0.001% / 1e-05 | support (SON) | method-parameter | "support_pct": "0.001%", "min_support": 1e-05 |
| L-685 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 37 | 22846 | itemsets (SON) | deterministic | "itemsets": 22846 |
| L-686 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 38 | 1085.6 | s (SON time) | hardware-dependent | "time_seconds": 1085.6 |
| L-687 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 39 | 13 | max_k (SON) | deterministic | "max_k": 13 |
| L-688 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 42 | Direct GPU (Blitz) | method | method-parameter | "blitz_reference": { "method": "Direct GPU (Blitz)" |
| L-689 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 43 | 0.0001% / 1e-06 | support (Blitz) | method-parameter | "support_pct": "0.0001%", "min_support": 1e-06 |
| L-690 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 45 | 2841280 | itemsets (Blitz) | deterministic | "itemsets": 2841280 |
| L-691 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 46 | 119.3 | s (Blitz time) | hardware-dependent | "time_seconds": 119.3 |
| L-692 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 47 | 19 | max_k (Blitz) | deterministic | "max_k": 19 |
| L-693 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 50 | 21.4 | speedup_vs_son | hardware-dependent | "comparison": { "speedup_vs_son": 21.4 |
| L-694 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 51 | false | itemset_match | deterministic | "itemset_match": false |
| L-695 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 52 | 453019 | itemset_diff | deterministic | "itemset_diff": 453019 |
| L-696 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | (derived) | 475865 (sum of k_distribution K1..K14) | itemsets | deterministic | extractor check: sum of direct k_distribution equals itemsets=475865 |
| L-697 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | (derived) | 768.9 → ceil 769 | min_count implied by 1e-05 × 76890945 | deterministic | extractor check: 1e-05 × 76,890,945 = 768.909; file states min_count 768 |
| L-698 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 2 | null_model_permutation_test | experiment id | method-parameter | "experiment": "null_model_permutation_test" |
| L-699 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 3 | 2026-02-19T06:10:46.260818+00:00 | timestamp | software | "timestamp": "2026-02-19T06:10:46.260818+00:00" |
| L-700 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 5 | 1.0001177641918693e-05 | min_support (fraction) | method-parameter | "min_support": 1.0001177641918693e-05 |
| L-701 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 6 | 769 | min_count | method-parameter | "min_count": 769 |
| L-702 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 7 | 76890945 | n_transactions (proteins) | deterministic | "n_transactions": 76890945 |
| L-703 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 8 | 5 | n_permutations | method-parameter | "n_permutations": 5 |
| L-704 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 9 | 42 | seed | method-parameter | "seed": 42 |
| L-705 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 10 | false | recomputed_real | method-parameter | "recomputed_real": false |
| L-706 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 13 | 1002 | real itemsets at K=1 | deterministic | "real_distribution": … "1": 1002 |
| L-707 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 14 | 22019 | real itemsets at K=2 | deterministic | "real_distribution": … "2": 22019 |
| L-708 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 15 | 73205 | real itemsets at K=3 | deterministic | "real_distribution": … "3": 73205 |
| L-709 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 16 | 108059 | real itemsets at K=4 | deterministic | "real_distribution": … "4": 108059 |
| L-710 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 17 | 104239 | real itemsets at K=5 | deterministic | "real_distribution": … "5": 104239 |
| L-711 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 18 | 78596 | real itemsets at K=6 | deterministic | "real_distribution": … "6": 78596 |
| L-712 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 19 | 48699 | real itemsets at K=7 | deterministic | "real_distribution": … "7": 48699 |
| L-713 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 20 | 25011 | real itemsets at K=8 | deterministic | "real_distribution": … "8": 25011 |
| L-714 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 21 | 10508 | real itemsets at K=9 | deterministic | "real_distribution": … "9": 10508 |
| L-715 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 22 | 3488 | real itemsets at K=10 | deterministic | "real_distribution": … "10": 3488 |
| L-716 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 23 | 869 | real itemsets at K=11 | deterministic | "real_distribution": … "11": 869 |
| L-717 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 24 | 152 | real itemsets at K=12 | deterministic | "real_distribution": … "12": 152 |
| L-718 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 25 | 17 | real itemsets at K=13 | deterministic | "real_distribution": … "13": 17 |
| L-719 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 26 | 1 | real itemsets at K=14 | deterministic | "real_distribution": … "14": 1 |
| L-720 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 28 | 475865 | real_total | deterministic | "real_total": 475865 |
| L-721 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 32 | 171401 | total_itemsets (null run 1) | deterministic | "run": 1, "total_itemsets": 171401 |
| L-722 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 33 | 99.22 | s (null run 1 mining time) | hardware-dependent | "run": 1, "time_seconds": 99.22 |
| L-723 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 35 | K1=1002; K2=63737; K3=79166; K4=25482; K5=1994; K6=20 | k_distribution (null run 1) | deterministic | "run": 1, "k_distribution": {"1": 1002, "2": 63737, "3": 79166, "4": 25482, "5": 1994, "6": 20} |
| L-724 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 45 | 171240 | total_itemsets (null run 2) | deterministic | "run": 2, "total_itemsets": 171240 |
| L-725 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 46 | 135.62 | s (null run 2 mining time) | hardware-dependent | "run": 2, "time_seconds": 135.62 |
| L-726 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 48 | K1=1002; K2=63672; K3=79121; K4=25453; K5=1970; K6=22 | k_distribution (null run 2) | deterministic | "run": 2, "k_distribution": {"1": 1002, "2": 63672, "3": 79121, "4": 25453, "5": 1970, "6": 22} |
| L-727 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 58 | 171289 | total_itemsets (null run 3) | deterministic | "run": 3, "total_itemsets": 171289 |
| L-728 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 59 | 141.97 | s (null run 3 mining time) | hardware-dependent | "run": 3, "time_seconds": 141.97 |
| L-729 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 61 | K1=1002; K2=63750; K3=79080; K4=25438; K5=1996; K6=23 | k_distribution (null run 3) | deterministic | "run": 3, "k_distribution": {"1": 1002, "2": 63750, "3": 79080, "4": 25438, "5": 1996, "6": 23} |
| L-730 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 71 | 171395 | total_itemsets (null run 4) | deterministic | "run": 4, "total_itemsets": 171395 |
| L-731 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 72 | 143.06 | s (null run 4 mining time) | hardware-dependent | "run": 4, "time_seconds": 143.06 |
| L-732 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 74 | K1=1002; K2=63703; K3=79185; K4=25475; K5=2008; K6=22 | k_distribution (null run 4) | deterministic | "run": 4, "k_distribution": {"1": 1002, "2": 63703, "3": 79185, "4": 25475, "5": 2008, "6": 22} |
| L-733 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 84 | 171275 | total_itemsets (null run 5) | deterministic | "run": 5, "total_itemsets": 171275 |
| L-734 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 85 | 142.3 | s (null run 5 mining time) | hardware-dependent | "run": 5, "time_seconds": 142.3 |
| L-735 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 87 | K1=1002; K2=63650; K3=79120; K4=25491; K5=1990; K6=22 | k_distribution (null run 5) | deterministic | "run": 5, "k_distribution": {"1": 1002, "2": 63650, "3": 79120, "4": 25491, "5": 1990, "6": 22} |
| L-736 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 98 | real=1002; null_mean=1002.0; null_std=0.0; z_score=0.0; p_value=1.0; direction=depleted; significant=false | statistics K=1 | deterministic | "1": { "real": 1002, "null_mean": 1002.0, "null_std": 0.0, "z_score": 0.0, "p_value": 1.0 |
| L-737 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 107 | real=22019; null_mean=63702.4; null_std=42.2; z_score=-987.08; p_value=1.0; direction=depleted; significant=false | statistics K=2 | deterministic | "2": { "real": 22019, "null_mean": 63702.4, "null_std": 42.2, "z_score": -987.08, "p_value": 1.0 |
| L-738 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 116 | real=73205; null_mean=79134.4; null_std=41.5; z_score=-142.71; p_value=1.0; direction=depleted; significant=false | statistics K=3 | deterministic | "3": { "real": 73205, "null_mean": 79134.4, "null_std": 41.5, "z_score": -142.71, "p_value": 1.0 |
| L-739 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 125 | real=108059; null_mean=25467.8; null_std=21.8; z_score=3790.74; p_value=0.0; direction=enriched; significant=true | statistics K=4 | deterministic | "4": { "real": 108059, "null_mean": 25467.8, "null_std": 21.8, "z_score": 3790.74, "p_value": 0.0 |
| L-740 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 134 | real=104239; null_mean=1991.6; null_std=13.8; z_score=7402.24; p_value=0.0; direction=enriched; significant=true | statistics K=5 | deterministic | "5": { "real": 104239, "null_mean": 1991.6, "null_std": 13.8, "z_score": 7402.24, "p_value": 0.0 |
| L-741 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 143 | real=78596; null_mean=21.8; null_std=1.1; z_score=71728.1; p_value=0.0; direction=enriched; significant=true | statistics K=6 | deterministic | "6": { "real": 78596, "null_mean": 21.8, "null_std": 1.1, "z_score": 71728.1, "p_value": 0.0 |
| L-742 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 152 | real=48699; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=7 | deterministic | "7": { "real": 48699, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-743 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 161 | real=25011; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=8 | deterministic | "8": { "real": 25011, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-744 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 170 | real=10508; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=9 | deterministic | "9": { "real": 10508, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-745 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 179 | real=3488; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=10 | deterministic | "10": { "real": 3488, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-746 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 188 | real=869; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=11 | deterministic | "11": { "real": 869, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-747 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 197 | real=152; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=12 | deterministic | "12": { "real": 152, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-748 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 206 | real=17; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=13 | deterministic | "13": { "real": 17, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-749 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 215 | real=1; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=14 | deterministic | "14": { "real": 1, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-750 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 225 | 475865 | real_total (summary) | deterministic | "summary": { "real_total": 475865 |
| L-751 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 226 | 171320.0 | null_mean_total | deterministic | "null_mean_total": 171320.0 |
| L-752 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 227 | 2.78 | ratio_real_vs_null | deterministic | "ratio_real_vs_null": 2.78 |
| L-753 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 228 | 132.43 | s avg_null_mining_seconds | hardware-dependent | "avg_null_mining_seconds": 132.43 |
| L-754 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 229 | 662.17 | s total_experiment_seconds | hardware-dependent | "total_experiment_seconds": 662.17 |
| L-755 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | (derived) | 88745 | itemsets K>=7 (sum of real_distribution K7..K14) | deterministic | extractor check: 48699+25011+10508+3488+869+152+17+1 = 88,745 (notebook text says 89,566) |
| L-756 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | (derived) | 769.0 | min_count × 1/n (1.0001177641918693e-05 × 76890945) | deterministic | extractor check: min_support here equals 769/76,890,945 exactly |
| L-757 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (file) | 41,665 lines; 4,749,057 bytes; 5,351 itemset blocks (K=15..19); fields per block: Support/Proteins/K, Item IDs, pLDDT features, Pfam domains, GO terms, Full decode | summary | deterministic | structured text dump; not transcribed row-by-row (see summary rows below) |
| L-758 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (file) | 81 distinct item ids used (min 1, max 1002); pLDDT-feature line present in 4,329/5,351 blocks; Pfam line in 5,196/5,351 | summary | deterministic | extractor-computed over all 'Item IDs' lines |
| L-759 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 2 | K=15 to K=19; 214M TrEMBL | scope | method-parameter | DECODED HIGH-K ITEMSETS (K=15 to K=19) FROM 214M TrEMBL MINING |
| L-760 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 3 | 76,890,945 | proteins in dataset | deterministic | Total proteins in dataset: 76,890,945 |
| L-761 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 4 | 5,351 | itemsets K>=15 | deterministic | Total itemsets K>=15: 5,351 |
| L-762 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 8 | 1 | itemsets at K=19 | deterministic | K = 19  \|  1 itemsets |
| L-763 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 12 | support 0.000002; ~187 proteins; K 19 | K=19 itemset 1/1 | deterministic | Support: 0.000002  \|  Proteins: ~187  \|  K: 19 |
| L-764 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 13 | [1, 13, 23, 507, 508, 511, 517, 560, 594, 602, 604, 641, 677, 698, 724, 731, 794, 887, 906] | item ids (K=19 itemset) | deterministic | Item IDs: [1, 13, 23, 507, 508, 511, 517, 560, 594, 602, 604, 641, 677, 698, 724, 731, 794, 887, 906] |
| L-765 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 14 | plddt_mean_med | pLDDT feature (K=19 itemset) | deterministic | pLDDT features:  plddt_mean_med |
| L-766 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 15 | PF00271, PF00270 | Pfam (K=19 itemset) | deterministic | Pfam domains:    PF00271, PF00270 |
| L-767 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 16 | GO:0005524, GO:0046872, GO:0003677, GO:0016887, GO:1990904, GO:0005730, GO:0005654, GO:0006397, GO:0045944, GO:0003724, GO:0043138, GO:0005813, GO:0008380, … (16 GO terms) | GO terms (K=19 itemset) | deterministic | GO terms: GO:0005524 [go_term], GO:0046872 [go_term], GO:0003677 [go_term], GO:0016887 [go_term], … |
| L-768 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 20 | 19 | itemsets at K=18 | deterministic | K = 18  \|  19 itemsets |
| L-769 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 24 | support 0.000002; ~192 proteins | K=18 itemset 1/19 | deterministic | Support: 0.000002  \|  Proteins: ~192  \|  K: 18 |
| L-770 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 175 | 173 | itemsets at K=17 | deterministic | K = 17  \|  173 itemsets |
| L-771 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 179 | support 0.000008; ~611 proteins | K=17 itemset 1/173 | deterministic | Support: 0.000008  \|  Proteins: ~611  \|  K: 17 |
| L-772 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 1544 | 1003 | itemsets at K=16 | deterministic | K = 16  \|  1003 itemsets |
| L-773 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 9400 | 4155 | itemsets at K=15 | deterministic | K = 15  \|  4155 itemsets |
| L-774 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41661 | K=19: 1 itemsets; support [0.000002, 0.000002]; proteins [187 - 187] | summary | deterministic | K=19:     1 itemsets \| support range [0.000002, 0.000002] \| proteins [187 - 187] |
| L-775 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41662 | K=18: 19 itemsets; support [0.000002, 0.000002]; proteins [187 - 192] | summary | deterministic | K=18:    19 itemsets \| support range [0.000002, 0.000002] \| proteins [187 - 192] |
| L-776 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41663 | K=17: 173 itemsets; support [0.000002, 0.000008]; proteins [187 - 611] | summary | deterministic | K=17:   173 itemsets \| support range [0.000002, 0.000008] \| proteins [187 - 611] |
| L-777 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41664 | K=16: 1003 itemsets; support [0.000002, 0.000008]; proteins [187 - 612] | summary | deterministic | K=16:  1003 itemsets \| support range [0.000002, 0.000008] \| proteins [187 - 612] |
| L-778 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41665 | K=15: 4155 itemsets; support [0.000001, 0.000009]; proteins [84 - 664] | summary | deterministic | K=15:  4155 itemsets \| support range [0.000001, 0.000009] \| proteins [84 - 664] |
| L-779 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (derived) | 1+19+173+1003+4155 = 5351; per-K support/protein ranges recomputed from all 5,351 'Support:' lines match the SUMMARY block exactly | consistency | deterministic | extractor check |
| L-780 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (derived) | 84 proteins at K=15 minimum → support 84/76,890,945 = 1.09e-6 | min protein count (K=15) | deterministic | extractor check: lowest 'Proteins: ~84' in file (K=15) |
| L-781 | `applications/alphafold/results_214m/GLOSSARY.md` | 1 | 214M | proteins (title) | deterministic | # Biological Glossary - AlphaFold 214M Mining Results |
| L-782 | `applications/alphafold/results_214m/GLOSSARY.md` | 3 | 214 million | AlphaFold-predicted structures | deterministic | frequent itemset mining of 214 million AlphaFold-predicted protein structures. |
| L-783 | `applications/alphafold/results_214m/GLOSSARY.md` | 9 | K>=10; 2.84M | high-K threshold / itemsets (direct GPU) | deterministic | Domains listed here appear in high-K itemsets (K>=10) from the 2.84M direct GPU mining results. |
| L-784 | `applications/alphafold/results_214m/GLOSSARY.md` | 13 | ~34% | FDA-approved drugs targeting GPCRs | external-fact | ~34% of FDA-approved drugs target GPCRs. |
| L-785 | `applications/alphafold/results_214m/GLOSSARY.md` | 15 | ~100 aa; >100 human proteins | SH2 domain length / count | external-fact | Src Homology 2 domain (~100 aa) … Over 100 human proteins contain SH2 domains. |
| L-786 | `applications/alphafold/results_214m/GLOSSARY.md` | 16 | ~60 aa; ~300 | SH3 domain length / count in human proteome | external-fact | Src Homology 3 domain (~60 aa) … ~300 SH3 domains in the human proteome. |
| L-787 | `applications/alphafold/results_214m/GLOSSARY.md` | 18 | ~50 aa | C1_1 domain length | external-fact | Phorbol ester / diacylglycerol (DAG) binding domain (~50 aa). |
| L-788 | `applications/alphafold/results_214m/GLOSSARY.md` | 21 | 70 | human Dbl-family members | external-fact | 70 human Dbl-family members. |
| L-789 | `applications/alphafold/results_214m/GLOSSARY.md` | 140 | 2.84M frequent itemsets; K=1-19 | itemsets / K range | deterministic | *Generated from et-miner AlphaFold 214M protein mining results (2.84M frequent itemsets, K=1-19). |
| L-790 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | (file) | 29 cells (19 code, 10 markdown); 1 stored output (cell 1 stream); execution_count None on all code cells; kernel '.venv (3.10.12)' | summary | software | notebook stored without executed outputs except one print |
| L-791 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 76.9M--109M | proteins | deterministic | GPU-accelerated frequent itemset mining on 76.9M--109M proteins |
| L-792 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 26.8M itemsets; K=1--22 | God Mode campaign | deterministic | 1K-feature "God Mode" campaign (26.8M itemsets, K=1--22) |
| L-793 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 16.8B itemsets; K=1--8 | 35K Alpha Centauri | deterministic | 35K-feature Alpha Centauri results (16.8B itemsets, K=1--8) |
| L-794 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 76.9M (1K features) / 109.2M (35K features) | proteins | deterministic | UniProt TrEMBL + SwissProt, 76.9M proteins (1K features) / 109.2M proteins (35K features) |
| L-795 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | NVIDIA H100/H200 | GPU | hardware-dependent | **Hardware:** NVIDIA H100/H200 GPUs, CSR bitvector encoding, Apriori with popcount-based support counting |
| L-796 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 7.3 minutes | God Mode mining time | hardware-dependent | 1K God Mode: 26.8M itemsets in 7.3 minutes (K=1--22) |
| L-797 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 16,812,646,639; 12.07B at K=8; 67 GB | itemsets / size | deterministic | 35K Alpha Centauri: **16,812,646,639 itemsets** (K=1--8), including 12.07B K=8 itemsets (67 GB) |
| L-798 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 L40-41 (code) | archived/alphafold/results_214m/itemsets_214m_godmode.parquet; item_mapping_214m.parquet | input paths (NOT present in repo) | software | GODMODE_PATH = REPO_ROOT / "archived" / "alphafold" / "results_214m" / "itemsets_214m_godmode.parquet" |
| L-799 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 L47 (code) | 26.8M rows | God Mode itemsets | deterministic | print("Loading God Mode itemsets (26.8M rows)...") |
| L-800 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 L51 (code) | 1,006 | item-mapping features | deterministic | print("Loading item mapping (1,006 features)...") |
| L-801 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 L67 (code) | 76,890,945 | N_PROTEINS | deterministic | N_PROTEINS = son_data["parameters"]["n_transactions"]  # 76,890,945 |
| L-802 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 (output) | (stream) Loading God Mode itemsets (26.8M rows)... / Loading item mapping (1,006 features)... | only stored output | software | the sole output in the notebook |
| L-803 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 2 L16 (code) | K=22: 1 itemset, ~8 proteins | deepest itemset | deterministic | 22: "Deepest: 22-feature signature (1 itemset, ~8 proteins)" |
| L-804 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 2 L66 (code) | 76.9M proteins; 1,002 features (500 Pfam + 500 GO + 6 pLDDT); support >= 1e-7 | God Mode params | method-parameter | <sub>76.9M proteins, 1,002 features (500 Pfam + 500 GO + 6 pLDDT), support >= 1e-7</sub> |
| L-805 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 3 (md) | K=1--3: 527K; K=4--6: 5.8M; K=7--9: 10.1M (peak); K=10--14: 6.0M; K=15--22: 5.5K | itemsets per K range (God Mode) | deterministic | \| K=1--3 \| 527K \| … \| K=15--22 \| 5.5K \| … found in ~100--200 proteins each |
| L-806 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 3 (md) | K=9: 3.53M | peak K itemsets | deterministic | The **peak at K=9** (3.53M itemsets) |
| L-807 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 3 (md) | 0 at K>=7 | null-model itemsets | deterministic | the null model produces **zero** itemsets at K >= 7. |
| L-808 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 4 L70 (code) | 59.6% | rank-1 item support (share of proteins) | deterministic | text=f"Rank 1: support={supports[0]:.4f} (59.6% of proteins)" |
| L-809 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 5 (md) | dies at K=6; tail to K=22 | null vs real K reach | deterministic | a **fundamentally different** distribution that dies at K=6. The real data's long tail extending to K=22 |
| L-810 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 9 (md) | 16--17 of 19 features shared | K=19 itemsets overlap | deterministic | The K=19 itemsets are all minor variations of a single RNA helicase / spliceosome signature (sharing 16--17 of 19 features) |
| L-811 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 11 L15 (code) | 95.2% (453,019 of 475,865) | SON loss | deterministic | # The comparison shows SON lost 95.2% of itemsets (453,019 out of 475,865) |
| L-812 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 11 L25-26 (code) | SON 22,846 vs direct 475,865; SON max K 13, direct 14 | SON vs direct | deterministic | # SON approximation: it found 22,846 total vs 475,865 direct / # SON max K = 13, Direct max K = 14 |
| L-813 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 12 (md) | 475,865 itemsets in 50.7 s | direct GPU at 0.001% | hardware-dependent | **Direct GPU finds 475,865 itemsets** in 50.7 seconds |
| L-814 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 12 (md) | 22,846 in 1,085.6 s; 21.4x slower | SON at 0.001% | hardware-dependent | **SON finds only 22,846** in 1,085.6 seconds (21.4x slower!) |
| L-815 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 12 (md) | 95.2% | SON pattern loss | deterministic | SON **loses 95.2%** of all patterns, primarily the high-K discoveries |
| L-816 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 13 L110 (code) | Z=3.29 (p<0.001) | significance line | method-parameter | # Significance line at Z=3.29 (p < 0.001) |
| L-817 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | 1,002/1,002 | K=1 features real=null | deterministic | **K=1:** Identical by construction (1,002/1,002 features, sanity check) |
| L-818 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | Z = -987, -143 | K=2--3 z-scores | deterministic | **K=2--3:** Null **exceeds** real data (Z = -987, -143). |
| L-819 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | Z = 3,791 | K=4 z-score | deterministic | **K=4:** Crossover point. Real data starts exceeding null (Z = 3,791). |
| L-820 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | Z = 7,402 | K=5 z-score | deterministic | **K=5:** Strong enrichment (Z = 7,402). |
| L-821 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | Z = 71,728; null 21.8 vs real 78,596 | K=6 | deterministic | **K=6:** Extreme enrichment (Z = 71,728). Null barely reaches K=6 (21.8 itemsets vs 78,596 real). |
| L-822 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | 89,566 | real itemsets at K>=7 | deterministic | Every single one of the 89,566 real itemsets at K >= 7 represents genuine biological organization |
| L-823 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 16,812,646,639; 109.2M proteins; 35,012 features | 35K campaign totals | deterministic | **Mining complete.** 16,812,646,639 frequent itemsets mined across K=1--8 from 109.2M proteins with 35,012 features. |
| L-824 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 28,405 | 35K itemsets at K=1 (file size tiny) | deterministic | \| 1 \| 28,405 \| tiny \| |
| L-825 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 5,506,372 | 35K itemsets at K=2 (file size tiny) | deterministic | \| 2 \| 5,506,372 \| tiny \| |
| L-826 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 176,048,236 | 35K itemsets at K=3 (file size ~3 GB) | deterministic | \| 3 \| 176,048,236 \| ~3 GB \| |
| L-827 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 1,506,508,703 | 35K itemsets at K=4 (file size ~8 GB) | deterministic | \| 4 \| 1,506,508,703 \| ~8 GB \| |
| L-828 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 2,474,423,427 | 35K itemsets at K=5 (file size ~10 GB) | deterministic | \| 5 \| 2,474,423,427 \| ~10 GB \| |
| L-829 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 626,781,137 | 35K itemsets at K=6 (file size ~4 GB) | deterministic | \| 6 \| 626,781,137 \| ~4 GB \| |
| L-830 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 3,507,040,364 | 35K itemsets at K=7 (file size 16.6 GB) | deterministic | \| 7 \| 3,507,040,364 \| 16.6 GB \| |
| L-831 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 12,072,309,005 | 35K itemsets at K=8 (file size 67 GB) | deterministic | \| 8 \| 12,072,309,005 \| 67 GB \| |
| L-832 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 17 L13-19 (code) | InterPro 12.7K; GO 5.8K; EC 1.1K; Keywords 15.2K; Taxonomy 175; Length 26; pLDDT 6 | 35K feature-group sizes | method-parameter | GROUP_LABELS = {"interpro": "InterPro (12.7K)", "go_term": "GO (5.8K)", "ec_number": "EC (1.1K)", "keyword": "Keywords (15.2K)", … |
| L-833 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 21 L94 (code) | 3_507_040_364 | K=7 known count (35K) | deterministic | "n_total": 3_507_040_364,  # known count |
| L-834 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 23 L9-16 (code) | 1: 28_405; 2: 5_506_372; 3: 176_048_236; 4: 1_506_508_703; 5: 2_474_423_427; 6: 626_781_137; 7: 3_507_040_364; 8: 12_072_309_005 | KNOWN_COUNTS (from mining logs) | deterministic | # --- Known counts (from mining logs) --- KNOWN_COUNTS = {…}  # 16,812,646,639 |
| L-835 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 25 (code) | K=2-6; support > 1e-5; min_confidence=0.5 (1K); top 10K K=4, min_confidence=0.3 (35K) | rule-mining params | method-parameter | [1K] Rule candidates … (K=2-6, support > 1e-5) … min_confidence=0.5 … Generating 35K rules (min_confidence=0.3) |
| L-836 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 26.8M itemsets; 7.3 minutes; K=1--22; 76.9M proteins | God Mode scale | hardware-dependent | **Scale:** 26.8M itemsets mined in 7.3 minutes across K=1--22, from 76.9M proteins. |
| L-837 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | K=9: 3.53M | peak | deterministic | The K-distribution peaks at K=9 (3.53M itemsets) |
| L-838 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 89,566 | patterns at K>=7 | deterministic | All 89,566 patterns at K >= 7 are **impossible** under the null hypothesis |
| L-839 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 20.8x more itemsets; 21.4x less time | direct vs SON | hardware-dependent | Direct GPU mining finds 20.8x more itemsets than the SON approximation algorithm, in 21.4x less time. |
| L-840 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | K=19 in ~187 proteins; K=22 in ~8 proteins | deep patterns | deterministic | The K=19 sentinel … found in ~187 proteins. The K=22 apex is a single 22-feature combination found in ~8 proteins |
| L-841 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 16.8B (35K) vs 26.8M (1K); K=8 12.07B, 67 GB, 71.8% | combinatorial explosion | deterministic | 35K features produced 16.8B itemsets at K=1--8, compared to 26.8M at 1K features. K=8 alone (12.07B itemsets, 67 GB) represents 71.8% |
| L-842 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | K=8/K=7 = 3.4x; K=5/K=4 = 1.6x; K=7/K=6 = 5.6x | growth rates | deterministic | K=8/K=7 = 3.4x growth. K=5/K=4 = 1.6x (plateau). K=7/K=6 = 5.6x (re-explosion). |
| L-843 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 30--40B+ | projected K=9 itemsets | deterministic | The 3.4x growth from K=7 to K=8 suggests K=9 could yield 30--40B+ itemsets. |
| L-844 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | February 2026; Prof. Alexandre Bonvin, Utrecht University | date / reviewer | software | *Prepared for review by Prof. Alexandre Bonvin, Utrecht University* *ET-miner \| GPU-accelerated proteome mining \| February 2026* |
| L-845 | `PROGRESS.md` | 6 | 2026-09-02T00:02Z | goal timestamp | software | ## Goal (set 2026-09-02T00:02Z by Et via /goal) |
| L-846 | `PROGRESS.md` | 23 | 2026-09-02T00:04Z | run dir created | software | `runs/20260902T0000Z/` (created 2026-09-02T00:04Z; subdirs logs/, phase1/, phase2/, phase3/, phase4/) |
| L-847 | `PROGRESS.md` | 40 | 2 × NVIDIA GeForce RTX 3090, 24576 MiB, compute cap 8.6 | GPU (this box) | hardware-dependent | GPU: **2 × NVIDIA GeForce RTX 3090, 24576 MiB each, compute cap 8.6** |
| L-848 | `PROGRESS.md` | 41 | driver 595.71.05; CUDA runtime 13.2; nvcc 12.1 | software (this box) | software | Driver 595.71.05, CUDA runtime 13.2, `nvcc` 12.1. **This box is NOT an H100** |
| L-849 | `PROGRESS.md` | 44 | AMD EPYC 7402P, 24 cores / 24 threads, 1 socket, 1 NUMA node | CPU (this box) | hardware-dependent | CPU: AMD EPYC 7402P, 24 cores / 24 threads, 1 socket, 1 NUMA node. |
| L-850 | `PROGRESS.md` | 45 | 125 GiB host; cgroup limit 74,782,343,168 B (~69.6 GiB); ~1.3 GiB used; swap 8 GiB | RAM (this box) | hardware-dependent | RAM: host shows 125 GiB total; container cgroup limit `/sys/fs/cgroup/memory.max` = 74,782,343,168 B (~69.6 GiB) |
| L-851 | `PROGRESS.md` | 49 | 200 GB total, 200 GB available (939 MB used); inodes 195,351,424 total / 191,245,709 free | disk (this box) | hardware-dependent | Disk (container root, overlay): **200 GB total, 200 GB available** (939 MB used). Inodes: 195,351,424 total, 191,245,709 free. |
| L-852 | `PROGRESS.md` | 51 | 1.9 TB NVMe, 850 GB free (not mounted for data) | host disk | hardware-dependent | The host NVMe (/dev/nvme0n1, 1.9 TB, 850 GB free) is NOT mounted for data use |
| L-853 | `PROGRESS.md` | 53 | ~23 TiB; ~644M files | full AlphaFold v4 requirement | external-fact | Disk verdict vs. full AlphaFold v4 requirement (~23 TiB, ~644M files after untar) |
| L-854 | `PROGRESS.md` | 59 | gcloud SDK 583.0.0; bq 2.1.38 | software | software | (SDK 583.0.0, bq 2.1.38), authenticated as etje975@gmail.com. |
| L-855 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 4 | 214M | AlphaFold DB proteins | deterministic | de base-vocab run over de complete AlphaFold DB (214M), die Majors 1/2/3/5 uit `papers/peer_review_jul12.md` sluit |
| L-856 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 9 | 4×H200 (or 8×) | planned GPU box | hardware-dependent | **GPU-box**: 4×H200 (of 8×) op Vast.ai. |
| L-857 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 18 | base214m_20260712 | ET_UPLOAD_TAG | software | `ET_UPLOAD_TAG` staat op `base214m_20260712` |
| L-858 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 38 | 23TB | CIF corpus avoided | external-fact | De pLDDT-via-BigQuery-route (23TB CIF vermijden) matcht `af-extract build-from-metadata --plddt-csv ...` |
| L-859 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 39 | 6 | pLDDT bins | method-parameter | Beide geven de 6 pLDDT-bins bij `include_plddt=true`. |
| L-860 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 44 | 100× | row-split NCCL hit count (null@8) | method-parameter | **Row-split NCCL wordt 100× geraakt door de null@8** → `validate_row_split.py` is de verplichte gate. |
| L-861 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 70 | 300000000 | bq --max_rows | method-parameter | bq query --use_legacy_sql=false --format=csv --max_rows=300000000 |
| L-862 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 76 | 1006 defined / 1002 frequent | base vocab size | method-parameter | ### Fase 1 — Extractie (base vocab 1006 defined / 1002 frequent) |
| L-863 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 82 | --top-pfam 500 --top-go 500 | extraction params | method-parameter | --top-pfam 500 --top-go 500 \ |
| L-864 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 87 | v1: 76.9M of 205.6M | multi-feature proteins (paper v1) | deterministic | Dit zijn de nieuwe Table 1 / abstract-getallen (v1 was 76,9M van 205,6M). |
| L-865 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 93 | --subset-size 1000000 --min-count 50 --n-gpus 2 | row-split validation params | method-parameter | --subset-size 1000000 --min-count 50 --n-gpus 2 -v |
| L-866 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 95 | N_GPUS=4 (green) / N_GPUS=1 (red) | GPU count decision | method-parameter | GREEN (exit 0) → draai de suite met `N_GPUS=4`. RED (exit 1) → `N_GPUS=1` fallback |
| L-867 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 104 | 6×3 campaign; null@769 100-perm | experiment suite | method-parameter | Draait: (1) mining-campagne 6×3, (2) Direct-vs-SON, (3) null@769 100-perm, |
| L-868 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 105 | null@8 100-perm | critical experiment | method-parameter | (4) **null@8 100-perm** (de kritische Major 1+2), (5) deepest-itemset accessions. |
| L-869 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 152 | 4×H200; ~85M multi-feature | compute plan | hardware-dependent | ## Compute & timing (4×H200, ~85M multi-feature) |
| L-870 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 153 | data ~0.5-1h; extraction ~0.5-1h; campaign+Direct/SON ~1h; null@769 ~25min; null@8 ~1.5-2h; closed/maximal+accessions ~0.5h | planned durations | hardware-dependent | Data ~0,5-1u · extractie ~0,5-1u · campagne+Direct/SON ~1u · null@769 ~25min · **null@8 ~1,5-2u** · closed/maximal+accessions ~0,5u |
| L-871 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 154 | ~5-6 h wall-clock; ~$45-90 | planned total | hardware-dependent | → **~5-6u wall-clock**, ~$45-90. |
| L-872 | `applications/alphafold/deploy/run_all_experiments.sh` | 2 | 4×H200 | planned GPUs | hardware-dependent | # Phase 3: 4×H200 Full Experiment Suite |
| L-873 | `applications/alphafold/deploy/run_all_experiments.sh` | 6 | 6 thresholds × 3 runs | campaign design | method-parameter | #   1. Full mining campaign (6 thresholds × 3 runs) |
| L-874 | `applications/alphafold/deploy/run_all_experiments.sh` | 7 | 3 runs each | Direct vs SON design | method-parameter | #   2. Direct GPU vs SON controlled comparison (3 runs each) |
| L-875 | `applications/alphafold/deploy/run_all_experiments.sh` | 8 | min_count=769; 100 permutations; 4 GPUs | null model A | method-parameter | #   3. Null model at min_count=769 (100 permutations, 4 GPUs) |
| L-876 | `applications/alphafold/deploy/run_all_experiments.sh` | 9 | min_count=8; 100 permutations; 4 GPUs | null model B | method-parameter | #   4. Null model at min_count=8 (100 permutations, 4 GPUs) — THE BIG ONE |
| L-877 | `applications/alphafold/deploy/run_all_experiments.sh` | 10 | K=22 | deepest itemset analysis | deterministic | #   5. K=22 protein identification + GO hierarchy analysis |
| L-878 | `applications/alphafold/deploy/run_all_experiments.sh` | 16 | ~4 hours; ~$9/hr | expected runtime / cost (4×H200) | hardware-dependent | # Expected runtime on 4×H200 (~$9/hr): ~4 hours total |
| L-879 | `applications/alphafold/deploy/run_all_experiments.sh` | 17 | 141GB VRAM (564GB total); 4.8 TB/s; 1.4× H100 | H200 spec | external-fact | # H200 has 141GB VRAM (564GB total) and 4.8 TB/s bandwidth (1.4× H100) |
| L-880 | `applications/alphafold/deploy/run_all_experiments.sh` | 26 | /workspace/data/transactions_214m.parquet | default data path | software | DATA="${1:-/workspace/data/transactions_214m.parquet}" |
| L-881 | `applications/alphafold/deploy/run_all_experiments.sh` | 28 | 4 | N_GPUS default | method-parameter | N_GPUS="${N_GPUS:-4}" |
| L-882 | `applications/alphafold/deploy/run_all_experiments.sh` | 82 | ~25 min | estimated null@769 | hardware-dependent | echo "Estimated: ~25 min (fast at high threshold)" |
| L-883 | `applications/alphafold/deploy/run_all_experiments.sh` | 98 | ~100 min (25 batches × ~4 min/perm on H200) | estimated null@8 | hardware-dependent | echo "Estimated: ~100 min (25 batches × ~4 min/perm on H200)" |
| L-884 | `applications/alphafold/deploy/run_all_experiments.sh` | 102 | --min-count 8; --runs 100 | null@8 args | method-parameter | --min-count 8 \ --runs 100 \ --n-gpus "$N_GPUS" \ --perm-per-gpu |
| L-885 | `applications/alphafold/deploy/run_null_model_35k.sh` | 9 | 5 permutations; 0.001% support (min_count=1092) | defaults (35K) | method-parameter | # Defaults: 5 permutations, 0.001% support (min_count=1092) |
| L-886 | `applications/alphafold/deploy/run_null_model_35k.sh` | 10 | ~30-60 min per permutation on 8× H200 | expected runtime | hardware-dependent | # Expected runtime: ~30-60 min per permutation on 8× H200 |
| L-887 | `applications/alphafold/deploy/run_null_model_35k.sh` | 15 | 0.00001 | SUPPORT default | method-parameter | SUPPORT="${2:-0.00001}" |
| L-888 | `applications/alphafold/deploy/run_null_model_35k.sh` | 50 | 42 | seed | method-parameter | --seed 42 \ |
| L-889 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 128 | ~2.3 GB | transactions_35k.parquet size | deterministic | # 35K-feature transactions (the main payload, ~2.3 GB) |
| L-890 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 357 | ~478 GB | bitvec size (35K, row-split across N×H200) | deterministic | echo "  This verifies the ~478 GB bitvec fits across N×H200 with row-split." |
| L-891 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 427 | ~1TB | /dev/shm on vast.ai | hardware-dependent | # /dev/shm is pre-mounted tmpfs on vast.ai (~1TB) |
| L-892 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 456 | --runs 100 --min-count 1090 --n-gpus 8 --seed 42 | null model (35K) args | method-parameter | echo "      --runs 100 --min-count 1090 --n-gpus 8 --seed 42 \\" |
| L-893 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 472 | --support 0.00001 --max-length 8 --use-gpu --n-gpus 8 | mining (35K) args | method-parameter | echo "        --support 0.00001 --max-length 8 --use-gpu --n-gpus 8 \\" |
| L-894 | `applications/alphafold/deploy/deploy_base214m.sh` | 3 | H200 box | target | hardware-dependent | #  BASE-214M DEPLOYMENT — H200 box, base-vocab regeneration |
| L-895 | `applications/alphafold/deploy/deploy_base214m.sh` | 260 | 4 | N_GPUS hint default | method-parameter | N_GPUS_HINT="${N_GPUS:-4}" |
| L-896 | `applications/alphafold/deploy/deploy_base214m.sh` | 279 | 23TB | CIF corpus avoided | external-fact | # pLDDT from BigQuery (avoids the 23TB CIF) |
| L-897 | `applications/alphafold/deploy/deploy_base214m.sh` | 284 | 1006 defined (top-500 Pfam + top-500 GO + 6 pLDDT) | base vocab | method-parameter | # (B) Extraction — base vocab 1006 defined (top-500 Pfam + top-500 GO + 6 pLDDT): |
| L-898 | `applications/alphafold/pipeline/postprocess_tx.py` | 5 | 118K features at min_count=8 | raw 35K TSV | deterministic | Reads the raw TSV from build_tx (118K features at min_count=8), applies a higher |
| L-899 | `applications/alphafold/pipeline/postprocess_tx.py` | 183 | 3423 | --min-count default | method-parameter | parser.add_argument("--min-count", type=int, default=3423, |
| L-900 | `applications/alphafold/pipeline/pipeline_214m.py` | 290 | top_pfam=200; top_go=200; min_plddt=50.0 | run_extract defaults | method-parameter | def run_extract(data_dir: Path, top_pfam: int = 200, top_go: int = 200, … min_plddt: float = 50.0): |
| L-901 | `applications/alphafold/pipeline/extract_features.py` | 187 | pLDDT >90 → 1.0; >70 → 0.5; else 0.1 | frac_high bins | method-parameter | frac_high = 1.0 if global_plddt > 90 else (0.5 if global_plddt > 70 else 0.1) |
| L-902 | `applications/alphafold/pipeline/extract_features.py` | 194 | mean_plddt < 50 / <= 90 | plddt_mean bin edges | method-parameter | if mean_plddt < 50: … elif mean_plddt <= 90: |
| L-903 | `applications/alphafold/experiments/experiment_direct_vs_son.py` | 41 | 3 | --runs default | method-parameter | "--runs", type=int, default=3, |
| L-904 | `applications/alphafold/experiments/experiment_direct_vs_son.py` | 45 | 0.00001 | --min-support default | method-parameter | "--min-support", type=float, default=0.00001, |
| L-905 | `applications/alphafold/experiments/experiment_direct_vs_son.py` | 49 | 40,000,000 | --chunk-size default (SON) | method-parameter | "--chunk-size", type=int, default=40_000_000, |
| L-906 | `applications/alphafold/experiments/experiment_direct_vs_son.py` | 53 | 0.9 | --local-support-factor default (SON) | method-parameter | "--local-support-factor", type=float, default=0.9, |
| L-907 | `applications/alphafold/experiments/experiment_full_campaign.py` | 85 | 3 | --runs default | method-parameter | "--runs", type=int, default=3, |
| L-908 | `applications/alphafold/experiments/experiment_full_campaign.py` | 115 | min_count = ceil(min_support × n_transactions) | threshold rule | method-parameter | min_count = math.ceil(min_support * n_transactions) |
| L-909 | `applications/alphafold/pipeline/run_mining.py` | 193 | 0.01 | --support default | method-parameter | parser.add_argument("--support", type=float, default=0.01, |
| L-910 | `applications/alphafold/pipeline/run_mining.py` | 195 | 4 | --max-length default | method-parameter | parser.add_argument("--max-length", type=int, default=4, |
| L-911 | `applications/alphafold/pipeline/run_mining.py` | 203 | 5 | --baseline-runs default | method-parameter | parser.add_argument("--baseline-runs", type=int, default=5, |
| L-912 | `applications/alphafold/experiments/analyze_k22_proteins.py` | 62 | pLDDT 70-90 | plddt_mean_medium label | method-parameter | ("Struct", "plddt_mean_medium", "pLDDT 70-90"), |
| L-913 | `README.md` | 13 | 80--110x | Rust tier speedup | hardware-dependent | SIMD-vectorized CSR support counting (AVX2/AVX-512). 80--110x speedup. |
| L-914 | `README.md` | 203 | ~264 bytes across 22 levels | PCIe transfer | deterministic | GPU-resident mining: zero PCIe transfers between K-levels (~264 bytes total across 22 levels) |
| L-915 | `README.md` | 204 | n/32 | dense→sparse crossover | method-parameter | switches from dense bitvectors to sparse CSR tidsets when tidsets become the smaller representation (mean support < n/32) |
| L-916 | `README.md` | 205 | 8x H200 | max tested GPUs | hardware-dependent | Multi-GPU support with per-device work distribution (tested up to 8x H200) |
| L-917 | `README.md` | 209 | 26.8 million patterns; ~76M structures; K=22; 7.3 minutes; single H100 | AlphaFold headline | hardware-dependent | ET-Miner discovered **26.8 million co-occurrence patterns** across **~76M predicted protein structures**, reaching feature combinations of size K=22 in 7.3 minutes on a single H100 GPU. |
| L-918 | `README.md` | 211 | over 200 million | AlphaFold DB proteins | external-fact | The AlphaFold Database contains predicted protein structures for over 200 million proteins. |
| L-919 | `README.md` | 213 | ~5 GB CSR; ~26 GB bitvectors | memory footprint | deterministic | ET-Miner constructs a CSR representation directly from transactions (~5 GB), converts to GPU-resident bitvectors (~26 GB) |
| L-920 | `README.md` | 219 | 214M total; 76.9M with multiple annotations | proteins processed | deterministic | \| Proteins processed \| 214M total, 76.9M with multiple annotations \| |
| L-921 | `README.md` | 220 | 1,002 | feature vocabulary | method-parameter | \| Feature vocabulary \| 1,002 items (Pfam domains, GO terms, pLDDT bins) \| |
| L-922 | `README.md` | 221 | 26.8 million | itemsets discovered | deterministic | \| Itemsets discovered \| 26.8 million \| |
| L-923 | `README.md` | 222 | 22 | maximum K | deterministic | \| Maximum K \| 22 (mathematically proven ceiling) \| |
| L-924 | `README.md` | 223 | 7.3 minutes; single H100 | mining time (deepest tier) | hardware-dependent | \| Mining time (deepest tier) \| 7.3 minutes on single H100 \| |
| L-925 | `README.md` | 224 | 0.1% → 0.00001% | support range | method-parameter | \| Support range \| 0.1% down to 0.00001% \| |
| L-926 | `README.md` | 236 | 1,000,000,000 | transactions (billion-scale streaming) | method-parameter | \| Transactions \| 1,000,000,000 \| |
| L-927 | `README.md` | 237 | 25.9 minutes | time | hardware-dependent | \| Time \| 25.9 minutes \| |
| L-928 | `README.md` | 238 | 643,139 tx/sec | throughput | hardware-dependent | \| Throughput \| 643,139 tx/sec \| |
| L-929 | `README.md` | 239 | 14.76 GB | peak memory | hardware-dependent | \| Peak memory \| 14.76 GB \| |
| L-930 | `README.md` | 240 | 326 | itemsets found | deterministic | \| Itemsets found \| 326 \| |
| L-931 | `README.md` | 241 | Intel Core Ultra 9 275HX (24 cores), 134 GB RAM | hardware | hardware-dependent | \| Hardware \| Intel Core Ultra 9 275HX (24 cores), 134 GB RAM \| |
| L-932 | `README.md` | 243 | 819K transactions | efficient-apriori comparison dataset | method-parameter | ### vs. efficient-apriori (819K transactions) |
| L-933 | `README.md` | 245 | AMD Ryzen 5 4600G (12 cores), 30 GB RAM, CPython 3.14 free-threading, Polars 1.37, MKL sparse | system | hardware-dependent | > System: AMD Ryzen 5 4600G (12 cores), 30 GB RAM, CPython 3.14 free-threading build, Polars 1.37, MKL sparse enabled |
| L-934 | `README.md` | 249 | 0.005; 9; 1.21s / 641 MB; 0.24s / 329 MB; 5.0x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (819K tx) | hardware-dependent | \| 0.005 \| 9 \| 1.21s / 641 MB \| 0.24s / 329 MB \| 5.0x \| |
| L-935 | `README.md` | 250 | 0.001; 326; 3.5s / 644 MB; 2.15s / 475 MB; 1.6x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (819K tx) | hardware-dependent | \| 0.001 \| 326 \| 3.5s / 644 MB \| 2.15s / 475 MB \| 1.6x \| |
| L-936 | `README.md` | 251 | 0.0005; 1,151; 16.0s / 691 MB; 6.2s / 1113 MB; 2.6x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (819K tx) | hardware-dependent | \| 0.0005 \| 1,151 \| 16.0s / 691 MB \| 6.2s / 1113 MB \| 2.6x \| |
| L-937 | `README.md` | 252 | 0.0001; 11,159; 214.2s / 2693 MB; 179.9s / 9186 MB; 1.2x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (819K tx) | hardware-dependent | \| 0.0001 \| 11,159 \| 214.2s / 2693 MB \| 179.9s / 9186 MB \| 1.2x \| |
| L-938 | `README.md` | 254 | 2.5M transactions | efficient-apriori comparison dataset | method-parameter | ### vs. efficient-apriori (2.5M transactions) |
| L-939 | `README.md` | 258 | 0.005; 9; 3.9s / 1663 MB; 0.37s / 539 MB; 10.5x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (2.5M tx) | hardware-dependent | \| 0.005 \| 9 \| 3.9s / 1663 MB \| 0.37s / 539 MB \| 10.5x \| |
| L-940 | `README.md` | 259 | 0.001; 336; 12.0s / 1671 MB; 3.5s / 1476 MB; 3.4x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (2.5M tx) | hardware-dependent | \| 0.001 \| 336 \| 12.0s / 1671 MB \| 3.5s / 1476 MB \| 3.4x \| |
| L-941 | `README.md` | 260 | 0.0005; 1,153; 53.7s / 1704 MB; 12.1s / 2369 MB; 4.4x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (2.5M tx) | hardware-dependent | \| 0.0005 \| 1,153 \| 53.7s / 1704 MB \| 12.1s / 2369 MB \| 4.4x \| |
| L-942 | `README.md` | 261 | 0.0001; 10,894; 589.4s / 3758 MB; 262.7s / 12163 MB; 2.2x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (2.5M tx) | hardware-dependent | \| 0.0001 \| 10,894 \| 589.4s / 3758 MB \| 262.7s / 12163 MB \| 2.2x \| |
| L-943 | `bench/README.md` | 3 | 2× RTX 3090, 24 GB, CUDA 12 | design target box | hardware-dependent | Scripted campaign for a rented multi-GPU box (designed for 2× RTX 3090, 24 GB, CUDA 12). |
| L-944 | `bench/README.md` | 15 | ≥ 2 GB | --shm-size | method-parameter | **`--shm-size` ≥ 2 GB** (vast.ai: the "docker options"/shm setting). |
| L-945 | `bench/README.md` | 20 | Disk ≥ 40 GB; host RAM ≥ 32 GB | box requirements | method-parameter | **Disk ≥ 40 GB** (datasets + wheels + rust build), **host RAM ≥ 32 GB** |
| L-946 | `bench/README.md` | 30 | ~10 min | setup_box.sh duration | hardware-dependent | bash bench/setup_box.sh          # env + rust ext + selfcheck + datasets (~10 min) |
| L-947 | `bench/README.md` | 31 | ~30-60 min | run_smoke.sh duration | hardware-dependent | bash bench/run_smoke.sh          # ~30-60 min: tier gate → gpu tests → small matrix |
| L-948 | `bench/README.md` | 33 | ~2-4 h | run_full.sh duration | hardware-dependent | bash bench/run_full.sh           # ~2-4 h: gate → full gpu tests → full matrix |

## SECTION B — file inventory

### B.1 Surviving log-like / result-like files in the repo (all extracted above)

| path | size (B) | mtime (UTC) | type | related to base214m? | one-line description |
|---|---|---|---|---|---|
| `/root/projects/ET-Miner/bench/results/2026-08-31-3090x2/env.txt` | 2194 | 2026-09-01 22:47 | env capture | no (bench, 2×3090) | git HEAD 5a8e59f8; nvidia-smi 2×RTX 3090 driver 580.159.03 CUDA 13.0; pip freeze section empty |
| `/root/projects/ET-Miner/bench/results/2026-08-31-3090x2/report.md` | 3192 | 2026-09-01 22:47 | md report | no (bench, 2×3090) | GPU campaign report: 28 runs ok; wall-time/VRAM per config; kernel A/B; per-level tables |
| `/root/projects/ET-Miner/bench/results/2026-08-31-3090x2/FINDINGS.md` | 4634 | 2026-09-01 22:47 | md report | no (bench, 2×3090) | Findings: tier-equivalence 7/7; 114 GPU tests; shared kernel 8.8×/16.3×; prefilter drops 9,285 itemsets; density-auto 10.4 s |
| `/root/projects/ET-Miner/bench/results/2026-08-31-3090x2/raw.jsonl` | 33942 | 2026-09-01 22:47 | jsonl (28 records) | no (bench, 2×3090) | one JSON record per run: config, wall_s, levels[k,n_candidates,n_frequent,ms], peak_vram_mb, throttle, n_itemsets, sum_counts, itemset_hash |
| `/root/projects/ET-Miner/bench/results/2026-09-01-3090x2-sparse/env.txt` | 2194 | 2026-09-01 22:47 | env capture | no (bench, 2×3090) | git HEAD b8a5032b; nvidia-smi Tue Sep 1 20:09:15 2026; same driver; pip freeze empty |
| `/root/projects/ET-Miner/bench/results/2026-09-01-3090x2-sparse/report.md` | 1551 | 2026-09-01 22:47 | md report | no (bench, 2×3090) | 13 runs ok; deep_k configs only; A/B 0.99×; per-level table |
| `/root/projects/ET-Miner/bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 4375 | 2026-09-01 22:47 | md report | no (bench, 2×3090) | Sparse-CSR follow-up: chain 9/9; signature n_itemsets=8841 sum_counts=266261275; density-auto 1.343 s |
| `/root/projects/ET-Miner/bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 18932 | 2026-09-01 22:47 | jsonl (13 records) | no (bench, 2×3090) | same schema as above, deep_k only, includes density-auto 1g/2g |
| `/root/projects/ET-Miner/applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 1110 | 2026-09-01 22:48 | json result | YES (AlphaFold 1K vocab, 76.9M tx) | Direct GPU vs SON at 0.001% (min_count 768): 475,865 vs 22,846 itemsets; 50.72 s vs 1085.6 s; Blitz 2,841,280 @0.0001% |
| `/root/projects/ET-Miner/applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 4676 | 2026-09-01 22:48 | json result | YES (AlphaFold 1K vocab, 76.9M tx) | Null-model permutation test, 5 perms, seed 42, min_count 769: real 475,865 vs null mean 171,320; per-K z-scores; K>=7 null = 0 |
| `/root/projects/ET-Miner/applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 4749057 | 2026-09-01 22:48 | txt decoded itemsets | YES (AlphaFold; K=15..19 subset) | 5,351 decoded itemsets K=15..19 from '214M TrEMBL mining'; 76,890,945 proteins; per-itemset support/proteins/Pfam/GO |
| `/root/projects/ET-Miner/applications/alphafold/results_214m/GLOSSARY.md` | 17193 | 2026-09-01 22:48 | md glossary | YES (AlphaFold; narrative) | Pfam/GO glossary; states 214 million structures, 2.84M itemsets K=1-19, K>=10 subset |
| `/root/projects/ET-Miner/applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | 111866 | 2026-09-01 22:48 | ipynb (29 cells, 1 output) | YES (AlphaFold 1K + 35K narratives) | Analysis notebook; hard-coded totals (26.8M K=1-22 God Mode; 16.8B K=1-8 35K); references missing archived/ parquets |
| `/root/projects/ET-Miner/applications/alphafold/deploy/RUNBOOK_base214m.md` | 7790 | 2026-09-01 22:48 | md runbook (Dutch) | YES (base214m plan) | Runbook for the base-vocab 214M run on 4×H200: vocab 1006/1002, top-500 Pfam/GO, null@769 & null@8 100-perm, v1 = 76.9M of 205.6M |
| `/root/projects/ET-Miner/applications/alphafold/deploy/run_all_experiments.sh` | 4877 | 2026-09-01 22:48 | bash | YES (base214m suite) | Experiment suite: 6×3 campaign, direct-vs-SON ×3, null@769 & null@8 (100 perms), K=22 analysis; H200 specs/estimates |
| `/root/projects/ET-Miner/applications/alphafold/deploy/deploy_base214m.sh` | 12910 | 2026-09-01 22:48 | bash | YES (base214m deploy) | Deploy script; base vocab 1006 = top-500 Pfam + top-500 GO + 6 pLDDT; N_GPUS default 4 |
| `/root/projects/ET-Miner/applications/alphafold/deploy/run_null_model_35k.sh` | 1702 | 2026-09-01 22:48 | bash | no (35K vocab) | 35K null model defaults: 5 perms, support 0.00001 (min_count=1092), seed 42 |
| `/root/projects/ET-Miner/applications/alphafold/deploy/deploy_project_milky_way.sh` | 17061 | 2026-09-01 22:48 | bash | no (35K vocab) | 35K deploy; ~2.3 GB transactions; ~478 GB bitvec; null --min-count 1090 --runs 100 --n-gpus 8 |
| `/root/projects/ET-Miner/PROGRESS.md` | 7135 | 2026-09-02 00:19 | md (audit state, created 2026-09-02) | meta (this audit) | Phase-0 hardware audit of THIS box (2×3090, driver 595.71.05, 200 GB disk) — not an original experiment log |
| `/root/projects/ET-Miner/README.md` | 12306 | 2026-09-01 22:47 | md | YES (headline numbers) | AlphaFold results table (214M/76.9M, 1,002 items, 26.8M itemsets, K=22, 7.3 min H100) + CPU benchmark tables |
| `/root/projects/ET-Miner/bench/README.md` | 3530 | 2026-09-01 22:47 | md | no | Campaign harness instructions; durations/requirements only |

Note: every repo file's mtime is the `git clone`/checkout time (2026-09-01 22:47–22:48 UTC); the meaningful provenance is the git commit: `results_214m/*`, the notebook, deploy/, pipeline/, experiments/ and paper/ were all added in ONE commit `65d9098` ("add alphafold experiment scripts", 2026-08-31 21:46:33 +0200, author Et9797); bench/results/2026-08-31-3090x2 in `6f789d8` (2026-08-31 22:39 UTC); bench/results/2026-09-01-3090x2-sparse in `b1f147e` (2026-09-01 20:11 UTC).

### B.2 Inventoried but NOT extracted (paper sources — other subagents)

| path | size (B) | mtime (UTC) | type | related to base214m? | one-line description |
|---|---|---|---|---|---|
| `/root/projects/ET-Miner/paper/et_miner_proteome.tex` | 65595 | 2026-09-01 22:48 | md/tex | yes (claims) | V1 paper LaTeX source (claims source) |
| `/root/projects/ET-Miner/paper/PAPER_V2_REVIEW.md` | 19720 | 2026-09-01 22:48 | md/tex | yes (claims) | review (14 mentions of base214m/214M) |
| `/root/projects/ET-Miner/paper/peer_review_jul12.md` | 27049 | 2026-09-01 22:48 | md/tex | yes (claims) | peer review |
| `/root/projects/ET-Miner/paper/review_b1_hostile.md` | 29488 | 2026-09-01 22:48 | md/tex | yes (claims) | review |
| `/root/projects/ET-Miner/paper/review_b2_results.md` | 16203 | 2026-09-01 22:48 | md/tex | yes (claims) | results review (10 mentions of 214M) |
| `/root/projects/ET-Miner/paper/revision_notes_b3.tex` | 32374 | 2026-09-01 22:48 | md/tex | yes (claims) | revision notes |
| `/root/projects/ET-Miner/paper/senior_review_jun01.md` | 15269 | 2026-09-01 22:48 | md/tex | yes (claims) | review |
| `/root/projects/ET-Miner/paper/senior_review_mar23.md` | 5493 | 2026-09-01 22:48 | md/tex | yes (claims) | review |
| `/root/projects/ET-Miner/CLAUDE.md` | 7282 | 2026-09-01 23:57 | md/tex | yes (claims) | correctness policy; no result numbers |

### B.3 Files found in /root (home) and elsewhere on the box — none are experiment logs

| path | size (B) | mtime (UTC) | type | related to base214m? | one-line description |
|---|---|---|---|---|---|
| `/root/.bash_history` | 1152 | 2026-09-01 23:29 | shell history | no | install of gh/gcloud, git clone of ET-Miner, checkout of branch; no data downloads or mining commands |
| `/root/.claude/history.jsonl` | 628 | 2026-09-01 23:39 | claude prompt history | no | 4 slash-command entries from this audit session |
| `/root/.claude/projects/-root-projects-ET-Miner/1bae0ea5-20f3-44f0-a8ba-07a290957c5b.jsonl` | 407052 | 2026-09-02 00:19 | Claude Code session transcript | no (this audit) | transcript of the current audit session — excluded from extraction (would be circular) |
| `/root/.claude/projects/-root-projects-ET-Miner/1bae0ea5-20f3-44f0-a8ba-07a290957c5b/subagents/` | 4096 | 2026-09-02 00:06 | dir: 5 subagent transcripts (*.jsonl) | no (this audit) | parallel Phase-1 subagent transcripts — excluded |
| `/root/.config/gcloud/logs/2026.09.01/` | 4096 | 2026-09-01 23:58 | dir: 7 gcloud debug logs | no (this audit) | gcloud components/init/auth logs from 2026-09-01 23:29–23:58; CONTAIN OAUTH TOKENS — do not copy |
| `/root/.config/gcloud/logs/2026.09.02/` | 4096 | 2026-09-02 00:07 | dir: 4 gcloud debug logs | no (this audit) | `gcloud storage ls gs://public-datasets-deepmind-alphafold-v4/{,proteomes/proteome-tax_id-9606-*,metadata/}` at 00:06 UTC (Phase-2 bucket probe) |
| `/root/projects/downloads/google-cloud-cli-linux-x86_64.tar.gz` | 86740170 | 2026-09-01 23:28 | tarball | no | gcloud SDK download (86.7 MB) |
| `/workspace/ports.log` | 6 | 2026-09-01 22:19 | log | no | contains only '26272' (vast.ai port) |
| `/workspace/onstart.sh` | 82 | 2026-09-01 22:19 | sh | no | vast.ai instance start hook (2 lines) |
| `/var/log/{alternatives,bootstrap,dpkg,jupyter,onstart,ssh_proxy}.log` | — | — | system logs | no | OS/package logs; jupyter.log and onstart.log are 0 bytes |
| `/tmp/claude-0/-root-projects-ET-Miner/1bae0ea5-20f3-44f0-a8ba-07a290957c5b/tasks/*.output` | — | — | 12 task outputs | no (this audit) | background-task outputs of the current session |
| `/root/projects/ET-Miner/runs/20260902T0000Z/RUN_DIR.txt` | 73 | 2026-09-02 00:04 | txt | meta | 'RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z' + '2026-09-02T00:04:43Z' |

### B.4 Deleted files known only by name from git history (NOT checked out, per instructions)

| deleted path | deleted in commit | note |
|---|---|---|
| `bench/results/campaign/smoke-legacy-1g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-legacy-1g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-legacy-2g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-legacy-2g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-shared-1g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-shared-1g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-shared-2g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-shared-2g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/stressk2ml2-legacy-2g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/stressk2ml2-legacy-2g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/stressk2ml2-shared-2g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/stressk2ml2-shared-2g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/raw.jsonl` | b1f147e (2026-09-01 20:11 UTC) | earlier smoke campaign aggregate / log (bench, synthetic presets) |
| `bench/results/campaign/report.md` | b1f147e (2026-09-01 20:11 UTC) | earlier smoke campaign aggregate / log (bench, synthetic presets) |
| `bench/results/smoke_run.log` | b1f147e (2026-09-01 20:11 UTC) | earlier smoke campaign aggregate / log (bench, synthetic presets) |

No deleted file in the entire git history (`git log --all --diff-filter=D`) matches alphafold / base214 / results_214m / experiment_ / parquet / csv — i.e. the AlphaFold mining logs and parquets were never committed to this repository.

Also referenced by code but absent everywhere on disk: `archived/alphafold/results_214m/itemsets_214m_godmode.parquet`, `archived/alphafold/results_214m/item_mapping_214m.parquet` (notebook cell 1), `results_214m/parquet/frequent_k*.parquet`, `results_214m/closed_maximal.json`, `results_214m/experiment_log_*.txt`, `/workspace/data/transactions_214m*.parquet`, `/workspace/data/transactions_35k.parquet`, `/mnt/hdd/research/et-miner-data/transactions_214m.parquet`.

## SECTION C — totals

- Total claim rows: **948** (IDs L-001 … L-948)
- Rows by category: deterministic: 369; external-fact: 10; hardware-dependent: 394; method-parameter: 140; software: 35
- Rows by file:
  - `bench/results/2026-08-31-3090x2/raw.jsonl`: 253
  - `bench/results/2026-08-31-3090x2/report.md`: 146
  - `bench/results/2026-09-01-3090x2-sparse/raw.jsonl`: 118
  - `bench/results/2026-09-01-3090x2-sparse/report.md`: 62
  - `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json`: 60
  - `applications/alphafold/experiments/analysis_alpha_centauri.ipynb`: 56
  - `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json`: 41
  - `README.md`: 30
  - `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md`: 28
  - `bench/results/2026-08-31-3090x2/FINDINGS.md`: 25
  - `applications/alphafold/results_214m/decoded_top_k_patterns.txt`: 25
  - `applications/alphafold/deploy/RUNBOOK_base214m.md`: 18
  - `applications/alphafold/deploy/run_all_experiments.sh`: 13
  - `bench/results/2026-08-31-3090x2/env.txt`: 11
  - `applications/alphafold/results_214m/GLOSSARY.md`: 10
  - `PROGRESS.md`: 10
  - `bench/results/2026-09-01-3090x2-sparse/env.txt`: 8
  - `bench/README.md`: 6
  - `applications/alphafold/deploy/deploy_project_milky_way.sh`: 5
  - `applications/alphafold/deploy/run_null_model_35k.sh`: 4
  - `applications/alphafold/deploy/deploy_base214m.sh`: 4
  - `applications/alphafold/experiments/experiment_direct_vs_son.py`: 4
  - `applications/alphafold/pipeline/run_mining.py`: 3
  - `applications/alphafold/pipeline/postprocess_tx.py`: 2
  - `applications/alphafold/pipeline/extract_features.py`: 2
  - `applications/alphafold/experiments/experiment_full_campaign.py`: 2
  - `applications/alphafold/pipeline/pipeline_214m.py`: 1
  - `applications/alphafold/experiments/analyze_k22_proteins.py`: 1
- Files inventoried: 21 extracted + 9 paper/policy files (not extracted) + 12 non-experiment locations + 15 deleted-by-name
- Surviving AlphaFold/base214m result artifacts: **2 JSON files + 1 decoded-itemset TXT** (all dated/derived from the 2026-02-19 1K-vocab run on 76,890,945 transactions), plus hard-coded numbers in the notebook, GLOSSARY, README and RUNBOOK. **Zero** AlphaFold mining logs survive.

### C.1 Cross-file observations (recorded for the merge step; no verdicts)

- `min_count` at 0.001% on 76,890,945 tx is **768** in `experiment_direct_vs_son_*.json` (L8) but **769** in `experiment_null_model_*.json` (L6, min_support 1.0001177641918693e-05 = 769/76,890,945); ceil(1e-5 × 76,890,945) = 769.
- Notebook cell 14 / cell 28 state **89,566** real itemsets at K≥7; the JSON `real_distribution` (identical in both JSON files) sums to **88,745** for K=7..14.
- Three different itemset totals refer to three different runs/thresholds: **475,865** (K≤14, 0.001%, JSONs), **2,841,280** (K≤19, 0.0001% 'Blitz', JSON L45; GLOSSARY '2.84M, K=1-19'; the decoded K=15–19 dump belongs to this run: 5,351 itemsets K≥15), and **26.8M / K≤22 / support ≥ 1e-7** ('God Mode', notebook + README; its parquet is missing).
- Notebook cell 3 says K=15–22 ≈ **5.5K** itemsets (God Mode); the decoded dump has **5,351** for K=15–19 only (Blitz run) — different runs, similar magnitude.
- Protein counts: `76,890,945` (JSONs, TXT L3) = README '76.9M with multiple annotations' of '214M total'; RUNBOOK L87 says v1 was '76,9M van 205,6M'; GLOSSARY says '214 million'; notebook cell 0 says 76.9M (1K) / 109.2M (35K).
- Hardware in the surviving AlphaFold narratives: single H100 (README L209/L223), 'H100/H200' (notebook), 4×/8× H200 (deploy scripts). All surviving *measured* logs are 2× RTX 3090 (bench). Current box (PROGRESS.md) is 2× RTX 3090, driver 595.71.05 (bench env.txt showed 580.159.03 on 2026-08-31/09-01 → different driver, possibly different box).
- Bench signature values that any reproduction on this box should hit bit-exactly (synthetic presets, deterministic): deep_k `n_itemsets=8841 sum_counts=266261275 hash 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567`; smoke (two-phase) `632 / 1049580 / ea17ea26fd0e44f7…`; skewed_rows `10350 / 346834073 / 75262446a1c2b29b…`; stress_k2 K≤2 `1695332 / 218250884 / 8d989bfcc6e5c745…`; stress_k2 K≤3 exact `3005770 / 314350393 / a6d53e9a5e1b44a7…` vs prefilter-on legacy-1g `2996485 / 314055259 / 0b3e8434997fd3b1…` (the documented −9,285 divergence).
- `env.txt` in both bench dirs has an empty `pip freeze` section — package versions (CuPy 14.1.1 etc.) are only asserted in FINDINGS.md.
- The notebook has no executed outputs (execution_count = null everywhere; one stored print) — every number in it is hard-coded text, not a recorded result.

### C.2 Search commands used (read-only)

```
git log --all --oneline | head -80
git log --all --name-only --diff-filter=D --pretty=format: | grep -iE 'log|result|alphafold|base214|json|csv|parquet|jsonl|txt|out' | sort -u
git show --name-status --diff-filter=D 6f789d8 b1f147e ; git show --stat 65d9098
ls -laR bench/results applications/alphafold runs paper datasets .claude .github
find /root -xdev \( -path /root/projects/downloads/google-cloud-sdk -o -path '*/site-packages' -o -path '*/.cache' -o -path '*/.git' -o -path '*/node_modules' -o -path '*/.venv' \) -prune -o -type f \( -iname '*.log' -o -iname '*.jsonl' -o -iname '*.out' -o -iname 'nohup.out' -o -iname 'results*.json' -o -iname 'results*.csv' -o -iname 'results*.md' -o -iname '*alphafold*' -o -iname '*base214*' -o -iname '*et_miner*' -o -iname '*et-miner*' -o -iname '*.parquet' \) -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n'
find / -xdev \( -path /proc -o -path /sys -o -path /opt/conda -o -path /root/projects/downloads/google-cloud-sdk -o -path '*/site-packages' -o -path '*/.cache' -o -path '*/.git' -o -path /root/projects/ET-Miner -o -path /usr/lib -o -path /usr/share \) -prune -o -type f \( -iname '*base214*' -o -iname '*alphafold*' -o -iname '*et_miner*' -o -iname '*et-miner*' -o -iname 'nohup.out' -o -iname '*.jsonl' -o -iname '*.parquet' \) -print
find / -xdev -type d -name archived   # → none
df -h ; ls -la /workspace /data /mnt /media /srv /tmp /var/log /root/.local /root/.jupyter /root/.ipython
cat /root/.bash_history /root/.claude/history.jsonl ; grep -nE 'gs://|base214|alphafold' /root/.boto /root/.config/gcloud/logs/*/*.log
python3 (json) summaries of bench/results/**/raw.jsonl, results_214m/*.json, decoded_top_k_patterns.txt, analysis_alpha_centauri.ipynb
```


---

# SOURCE: 11 old mining/pipeline logs pushed in commit e10801c (applications/alphafold/results_214m/logs/)

(verbatim copy of `runs/20260902T0000Z/phase1/claims_logs_new.md`)

# Phase 1 — claims extracted from the 11 newly committed AlphaFold mining logs

Generated 2026-09-02 by the log-claims subagent (second pass, "L2-" IDs). Repo root: `/root/projects/ET-Miner`; all `file` cells refer to `applications/alphafold/results_214m/logs/<file>`. Source commit: `e10801c` "Added old Alphafold protein feature co-occurence mining logs" (Et9797, Wed Sep 2 02:35:27 2026 +0200; 12 files, 5,188 insertions).
Scope: every line of every one of the 11 files was read (the two long files were dissected line-by-line: 2,025 strictly periodic `Parsed NNNK DAT records` lines and 205 `Written NM transactions` lines were verified to be monotonic with constant step and are consolidated into one row each; every other digit-bearing line has its own row or an explicit skip-with-reason in Section D). NOT judged for correctness — extraction and reconstruction only.

Conventions for Section A:
- `line` is the 1-based line number as counted by `wc -l` / `sed -n` / `cat -n` (newline-delimited). `watcher.log` line 4 is a single 2,612-byte line whose 45 aria2c progress snapshots are separated by carriage returns; it is expanded into 7 rows.
- Category scheme: **deterministic** = reproducible exactly from the same inputs and parameters (counts, supports, itemsets per K, nnz, rule counts, conf/lift, K_max); **hardware-dependent** = timings, throughput, download rate, disk-usage brackets, GPU names; **method-parameter** = support %, min_count / min proteins, max K, top-N, pLDDT threshold, pLDDT bin count, min_confidence, aria2c connection count, target K; **external-fact** = timestamps of runs and properties of external inputs (TrEMBL file size, DAT record counts, pLDDT CSV row counts, nominal "214M"); **software** = output parquet sizes in bytes, CSV column indices, snapshot bookkeeping. No version strings, exit codes, hostnames, driver strings, or RAM/VRAM figures exist in any of the 11 files.
- Per-K mining lines (`K=n: … candidates -> … frequent (…ms)`) are one row each with value = frequent count; candidates and ms are in the verbatim context. `[tag]` prefixes in the context column are provenance labels added by the extractor; the text after the tag is verbatim (multi-line itemset blocks are joined with ` ; `; contexts are cut at 25 words with `…`).
- `support=X (N proteins)` and `conf=X lift=Yx` lines yield two rows each (both numbers are independently checkable claims).
- Values are transcribed exactly as printed (thousands separators kept where the log has them).

## SECTION A — CLAIMS TABLE

| ID | file | line | value | unit | category | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|
| L2-001 | `beyond_mining.log` | 2 | 0.00002 | % support (nominal) | method-parameter | >>> BEYOND MADMAN — 0.00002% SUPPORT |
| L2-002 | `beyond_mining.log` | 3 | ~15 | min proteins (nominal) | method-parameter | >>> Min proteins: ~15 |
| L2-003 | `beyond_mining.log` | 4 | 25 | max K (max_length) | method-parameter | >>> Max K: 25 |
| L2-004 | `beyond_mining.log` | 6 | 76,890,945 | transactions with >1 item | deterministic | Transactions with >1 item: 76,890,945 |
| L2-005 | `beyond_mining.log` | 7 | 15 | min support count (script-computed) | method-parameter | Min support count: 15 proteins |
| L2-006 | `beyond_mining.log` | 8 | 2026-02-09 06:05:03,735 | timestamp (run start, Direct CSR path) | external-fact | 2026-02-09 06:05:03,735 Direct CSR path: 76,890,945 transactions, min_count=16 |
| L2-007 | `beyond_mining.log` | 8 | 76,890,945 | transactions | deterministic | 2026-02-09 06:05:03,735 Direct CSR path: 76,890,945 transactions, min_count=16 |
| L2-008 | `beyond_mining.log` | 8 | 16 | min_count | method-parameter | 2026-02-09 06:05:03,735 Direct CSR path: 76,890,945 transactions, min_count=16 |
| L2-009 | `beyond_mining.log` | 9 | 1002 | frequent items (K=1) | deterministic | 2026-02-09 06:05:04,142 Direct CSR path: 1002 frequent items |
| L2-010 | `beyond_mining.log` | 10 | 316,421,093 | non-zeros (CSR nnz) | deterministic | 2026-02-09 06:05:04,902 Direct CSR path: 316,421,093 non-zeros |
| L2-011 | `beyond_mining.log` | 11 | 1,002 | frequent itemsets K=1 | deterministic | [per-K] K=1: 1,002 candidates -> 1,002 frequent (80ms) |
| L2-012 | `beyond_mining.log` | 12 | 60,088 | frequent itemsets K=2 | deterministic | [per-K] K=2: 501,501 candidates -> 60,088 frequent (1956ms) |
| L2-013 | `beyond_mining.log` | 13 | 356,691 | frequent itemsets K=3 | deterministic | [per-K] K=3: 0 candidates -> 356,691 frequent (26394ms) |
| L2-014 | `beyond_mining.log` | 14 | 894,903 | frequent itemsets K=4 | deterministic | [per-K] K=4: 0 candidates -> 894,903 frequent (26926ms) |
| L2-015 | `beyond_mining.log` | 15 | 1,421,780 | frequent itemsets K=5 | deterministic | [per-K] K=5: 0 candidates -> 1,421,780 frequent (23562ms) |
| L2-016 | `beyond_mining.log` | 16 | 1,794,852 | frequent itemsets K=6 | deterministic | [per-K] K=6: 0 candidates -> 1,794,852 frequent (14865ms) |
| L2-017 | `beyond_mining.log` | 17 | 2,006,312 | frequent itemsets K=7 | deterministic | [per-K] K=7: 0 candidates -> 2,006,312 frequent (9909ms) |
| L2-018 | `beyond_mining.log` | 18 | 2,053,858 | frequent itemsets K=8 | deterministic | [per-K] K=8: 0 candidates -> 2,053,858 frequent (8260ms) |
| L2-019 | `beyond_mining.log` | 19 | 1,912,672 | frequent itemsets K=9 | deterministic | [per-K] K=9: 0 candidates -> 1,912,672 frequent (7559ms) |
| L2-020 | `beyond_mining.log` | 20 | 1,587,145 | frequent itemsets K=10 | deterministic | [per-K] K=10: 0 candidates -> 1,587,145 frequent (6937ms) |
| L2-021 | `beyond_mining.log` | 21 | 1,148,998 | frequent itemsets K=11 | deterministic | [per-K] K=11: 0 candidates -> 1,148,998 frequent (6317ms) |
| L2-022 | `beyond_mining.log` | 22 | 712,433 | frequent itemsets K=12 | deterministic | [per-K] K=12: 0 candidates -> 712,433 frequent (5460ms) |
| L2-023 | `beyond_mining.log` | 23 | 371,981 | frequent itemsets K=13 | deterministic | [per-K] K=13: 0 candidates -> 371,981 frequent (1803ms) |
| L2-024 | `beyond_mining.log` | 24 | 160,675 | frequent itemsets K=14 | deterministic | [per-K] K=14: 0 candidates -> 160,675 frequent (796ms) |
| L2-025 | `beyond_mining.log` | 25 | 56,221 | frequent itemsets K=15 | deterministic | [per-K] K=15: 0 candidates -> 56,221 frequent (277ms) |
| L2-026 | `beyond_mining.log` | 26 | 15,501 | frequent itemsets K=16 | deterministic | [per-K] K=16: 0 candidates -> 15,501 frequent (80ms) |
| L2-027 | `beyond_mining.log` | 27 | 3,236 | frequent itemsets K=17 | deterministic | [per-K] K=17: 0 candidates -> 3,236 frequent (26ms) |
| L2-028 | `beyond_mining.log` | 28 | 480 | frequent itemsets K=18 | deterministic | [per-K] K=18: 0 candidates -> 480 frequent (12ms) |
| L2-029 | `beyond_mining.log` | 29 | 45 | frequent itemsets K=19 | deterministic | [per-K] K=19: 0 candidates -> 45 frequent (11ms) |
| L2-030 | `beyond_mining.log` | 30 | 2 | frequent itemsets K=20 | deterministic | [per-K] K=20: 0 candidates -> 2 frequent (13ms) |
| L2-031 | `beyond_mining.log` | 32 | 281.0 | s (total, '4.7 min') | hardware-dependent | >>> COMPLETE in 281.0s (4.7 min) |
| L2-032 | `beyond_mining.log` | 33 | 14,558,875 | itemsets (total) | deterministic | >>> Itemsets: 14,558,875 |
| L2-033 | `beyond_mining.log` | 36 | 1,002 | itemsets K=1 | deterministic | [K-dist] K=1: 1,002 |
| L2-034 | `beyond_mining.log` | 37 | 60,088 | itemsets K=2 | deterministic | [K-dist] K=2: 60,088 |
| L2-035 | `beyond_mining.log` | 38 | 356,691 | itemsets K=3 | deterministic | [K-dist] K=3: 356,691 |
| L2-036 | `beyond_mining.log` | 39 | 894,903 | itemsets K=4 | deterministic | [K-dist] K=4: 894,903 |
| L2-037 | `beyond_mining.log` | 40 | 1,421,780 | itemsets K=5 | deterministic | [K-dist] K=5: 1,421,780 |
| L2-038 | `beyond_mining.log` | 41 | 1,794,852 | itemsets K=6 | deterministic | [K-dist] K=6: 1,794,852 |
| L2-039 | `beyond_mining.log` | 42 | 2,006,312 | itemsets K=7 | deterministic | [K-dist] K=7: 2,006,312 |
| L2-040 | `beyond_mining.log` | 43 | 2,053,858 | itemsets K=8 | deterministic | [K-dist] K=8: 2,053,858 |
| L2-041 | `beyond_mining.log` | 44 | 1,912,672 | itemsets K=9 | deterministic | [K-dist] K=9: 1,912,672 |
| L2-042 | `beyond_mining.log` | 45 | 1,587,145 | itemsets K=10 | deterministic | [K-dist] K=10: 1,587,145 |
| L2-043 | `beyond_mining.log` | 46 | 1,148,998 | itemsets K=11 | deterministic | [K-dist] K=11: 1,148,998 |
| L2-044 | `beyond_mining.log` | 47 | 712,433 | itemsets K=12 | deterministic | [K-dist] K=12: 712,433 |
| L2-045 | `beyond_mining.log` | 48 | 371,981 | itemsets K=13 | deterministic | [K-dist] K=13: 371,981 |
| L2-046 | `beyond_mining.log` | 49 | 160,675 | itemsets K=14 | deterministic | [K-dist] K=14: 160,675 |
| L2-047 | `beyond_mining.log` | 50 | 56,221 | itemsets K=15 | deterministic | [K-dist] K=15: 56,221 |
| L2-048 | `beyond_mining.log` | 51 | 15,501 | itemsets K=16 | deterministic | [K-dist] K=16: 15,501 |
| L2-049 | `beyond_mining.log` | 52 | 3,236 | itemsets K=17 | deterministic | [K-dist] K=17: 3,236 |
| L2-050 | `beyond_mining.log` | 53 | 480 | itemsets K=18 | deterministic | [K-dist] K=18: 480 |
| L2-051 | `beyond_mining.log` | 54 | 45 | itemsets K=19 | deterministic | [K-dist] K=19: 45 |
| L2-052 | `beyond_mining.log` | 55 | 2 | itemsets K=20 | deterministic | [K-dist] K=20: 2 |
| L2-053 | `beyond_mining.log` | 57 | 49,989,864 | bytes (output parquet) | software | >>> Saved to /root/alphafold-data/full_214m/itemsets_214m_beyond.parquet (49,989,864 bytes) |
| L2-054 | `direct_mining.log` | 3 | 1e-06 | min_support (0.0001%) | method-parameter | >>> Support: 1e-06 (0.0001%) |
| L2-055 | `direct_mining.log` | 4 | 20 | max K (max_length) | method-parameter | >>> Max K: 20 |
| L2-056 | `direct_mining.log` | 8 | 205,620,298 | transactions (total in parquet) | deterministic | Total transactions: 205,620,298 |
| L2-057 | `direct_mining.log` | 9 | 76,890,945 | transactions with >1 item | deterministic | With >1 item: 76,890,945 |
| L2-058 | `direct_mining.log` | 10 | 76 | min support count (script-computed) | method-parameter | Min support count: 76 proteins |
| L2-059 | `direct_mining.log` | 11 | 0.5 | s (parquet load) | hardware-dependent | Load time: 0.5s |
| L2-060 | `direct_mining.log` | 14 | 2026-02-09 05:46:42,182 | timestamp (run start, Direct CSR path) | external-fact | 2026-02-09 05:46:42,182 Direct CSR path: 76,890,945 transactions, min_count=77 |
| L2-061 | `direct_mining.log` | 14 | 76,890,945 | transactions | deterministic | 2026-02-09 05:46:42,182 Direct CSR path: 76,890,945 transactions, min_count=77 |
| L2-062 | `direct_mining.log` | 14 | 77 | min_count | method-parameter | 2026-02-09 05:46:42,182 Direct CSR path: 76,890,945 transactions, min_count=77 |
| L2-063 | `direct_mining.log` | 15 | 1002 | frequent items (K=1) | deterministic | 2026-02-09 05:46:42,549 Direct CSR path: 1002 frequent items |
| L2-064 | `direct_mining.log` | 16 | 316,421,093 | non-zeros (CSR nnz) | deterministic | 2026-02-09 05:46:43,300 Direct CSR path: 316,421,093 non-zeros |
| L2-065 | `direct_mining.log` | 17 | 1,002 | frequent itemsets K=1 | deterministic | [per-K] K=1: 1,002 candidates → 1,002 frequent (289ms) |
| L2-066 | `direct_mining.log` | 18 | 39,125 | frequent itemsets K=2 | deterministic | [per-K] K=2: 501,501 candidates → 39,125 frequent (1950ms) |
| L2-067 | `direct_mining.log` | 19 | 184,900 | frequent itemsets K=3 | deterministic | [per-K] K=3: 0 candidates → 184,900 frequent (25790ms) |
| L2-068 | `direct_mining.log` | 20 | 361,696 | frequent itemsets K=4 | deterministic | [per-K] K=4: 0 candidates → 361,696 frequent (14559ms) |
| L2-069 | `direct_mining.log` | 21 | 445,661 | frequent itemsets K=5 | deterministic | [per-K] K=5: 0 candidates → 445,661 frequent (10831ms) |
| L2-070 | `direct_mining.log` | 22 | 439,605 | frequent itemsets K=6 | deterministic | [per-K] K=6: 0 candidates → 439,605 frequent (6881ms) |
| L2-071 | `direct_mining.log` | 23 | 387,030 | frequent itemsets K=7 | deterministic | [per-K] K=7: 0 candidates → 387,030 frequent (5693ms) |
| L2-072 | `direct_mining.log` | 24 | 318,349 | frequent itemsets K=8 | deterministic | [per-K] K=8: 0 candidates → 318,349 frequent (1701ms) |
| L2-073 | `direct_mining.log` | 25 | 247,680 | frequent itemsets K=9 | deterministic | [per-K] K=9: 0 candidates → 247,680 frequent (1204ms) |
| L2-074 | `direct_mining.log` | 26 | 179,604 | frequent itemsets K=10 | deterministic | [per-K] K=10: 0 candidates → 179,604 frequent (871ms) |
| L2-075 | `direct_mining.log` | 27 | 117,783 | frequent itemsets K=11 | deterministic | [per-K] K=11: 0 candidates → 117,783 frequent (596ms) |
| L2-076 | `direct_mining.log` | 28 | 67,558 | frequent itemsets K=12 | deterministic | [per-K] K=12: 0 candidates → 67,558 frequent (352ms) |
| L2-077 | `direct_mining.log` | 29 | 32,831 | frequent itemsets K=13 | deterministic | [per-K] K=13: 0 candidates → 32,831 frequent (179ms) |
| L2-078 | `direct_mining.log` | 30 | 13,105 | frequent itemsets K=14 | deterministic | [per-K] K=14: 0 candidates → 13,105 frequent (75ms) |
| L2-079 | `direct_mining.log` | 31 | 4,155 | frequent itemsets K=15 | deterministic | [per-K] K=15: 0 candidates → 4,155 frequent (30ms) |
| L2-080 | `direct_mining.log` | 32 | 1,003 | frequent itemsets K=16 | deterministic | [per-K] K=16: 0 candidates → 1,003 frequent (14ms) |
| L2-081 | `direct_mining.log` | 33 | 173 | frequent itemsets K=17 | deterministic | [per-K] K=17: 0 candidates → 173 frequent (11ms) |
| L2-082 | `direct_mining.log` | 34 | 19 | frequent itemsets K=18 | deterministic | [per-K] K=18: 0 candidates → 19 frequent (12ms) |
| L2-083 | `direct_mining.log` | 35 | 1 | frequent itemsets K=19 | deterministic | [per-K] K=19: 0 candidates → 1 frequent (15ms) |
| L2-084 | `direct_mining.log` | 37 | 119.3 | s (mining, '2.0 min') | hardware-dependent | >>> MINING COMPLETE in 119.3s (2.0 min) |
| L2-085 | `direct_mining.log` | 38 | 120.6 | s (total, '2.0 min') | hardware-dependent | >>> Total time: 120.6s (2.0 min) |
| L2-086 | `direct_mining.log` | 39 | 2,841,280 | itemsets (total) | deterministic | >>> Itemsets found: 2,841,280 |
| L2-087 | `direct_mining.log` | 42 | 1,002 | itemsets K=1 | deterministic | [K-dist] K=1: 1,002 |
| L2-088 | `direct_mining.log` | 43 | 39,125 | itemsets K=2 | deterministic | [K-dist] K=2: 39,125 |
| L2-089 | `direct_mining.log` | 44 | 184,900 | itemsets K=3 | deterministic | [K-dist] K=3: 184,900 |
| L2-090 | `direct_mining.log` | 45 | 361,696 | itemsets K=4 | deterministic | [K-dist] K=4: 361,696 |
| L2-091 | `direct_mining.log` | 46 | 445,661 | itemsets K=5 | deterministic | [K-dist] K=5: 445,661 |
| L2-092 | `direct_mining.log` | 47 | 439,605 | itemsets K=6 | deterministic | [K-dist] K=6: 439,605 |
| L2-093 | `direct_mining.log` | 48 | 387,030 | itemsets K=7 | deterministic | [K-dist] K=7: 387,030 |
| L2-094 | `direct_mining.log` | 49 | 318,349 | itemsets K=8 | deterministic | [K-dist] K=8: 318,349 |
| L2-095 | `direct_mining.log` | 50 | 247,680 | itemsets K=9 | deterministic | [K-dist] K=9: 247,680 |
| L2-096 | `direct_mining.log` | 51 | 179,604 | itemsets K=10 | deterministic | [K-dist] K=10: 179,604 |
| L2-097 | `direct_mining.log` | 52 | 117,783 | itemsets K=11 | deterministic | [K-dist] K=11: 117,783 |
| L2-098 | `direct_mining.log` | 53 | 67,558 | itemsets K=12 | deterministic | [K-dist] K=12: 67,558 |
| L2-099 | `direct_mining.log` | 54 | 32,831 | itemsets K=13 | deterministic | [K-dist] K=13: 32,831 |
| L2-100 | `direct_mining.log` | 55 | 13,105 | itemsets K=14 | deterministic | [K-dist] K=14: 13,105 |
| L2-101 | `direct_mining.log` | 56 | 4,155 | itemsets K=15 | deterministic | [K-dist] K=15: 4,155 |
| L2-102 | `direct_mining.log` | 57 | 1,003 | itemsets K=16 | deterministic | [K-dist] K=16: 1,003 |
| L2-103 | `direct_mining.log` | 58 | 173 | itemsets K=17 | deterministic | [K-dist] K=17: 173 |
| L2-104 | `direct_mining.log` | 59 | 19 | itemsets K=18 | deterministic | [K-dist] K=18: 19 |
| L2-105 | `direct_mining.log` | 60 | 1 | itemsets K=19 | deterministic | [K-dist] K=19: 1 |
| L2-106 | `direct_mining.log` | 63 | 12,177,502 | bytes (output parquet) | software | >>> File size: 12,177,502 bytes |
| L2-107 | `extreme_mining.log` | 1 | 0.001 | % support (on '214M TrEMBL') | method-parameter | >>> EXTREME SUPPORT: 0.001% on 214M TrEMBL |
| L2-108 | `extreme_mining.log` | 3 | 205,620,298 | transactions (total in parquet) | deterministic | Total transactions: 205,620,298 |
| L2-109 | `extreme_mining.log` | 4 | 76,890,945 | transactions with >1 item | deterministic | With >1 item: 76,890,945 |
| L2-110 | `extreme_mining.log` | 5 | 768 | min support count (script-computed; '0.001%') | method-parameter | Min support = 0.001% = 768 proteins |
| L2-111 | `extreme_mining.log` | 7 | 0.001 | % support (banner) | method-parameter | >>> GPU-RESIDENT STREAMING APRIORI — 0.001% SUPPORT |
| L2-112 | `extreme_mining.log` | 8 | 20 | max length (banner) | method-parameter | >>> MAX LENGTH 20 — HUNTING FOR MEGA-COMPLEXES |
| L2-113 | `extreme_mining.log` | 12 | 1085.6 | s (mining) | hardware-dependent | Time: 1085.6s |
| L2-114 | `extreme_mining.log` | 13 | 22,846 | itemsets (total) | deterministic | Itemsets: 22,846 |
| L2-115 | `extreme_mining.log` | 14 | 47 | itemsets K=1 | deterministic | [K-dist] K=1: 47 |
| L2-116 | `extreme_mining.log` | 15 | 990 | itemsets K=2 | deterministic | [K-dist] K=2: 990 |
| L2-117 | `extreme_mining.log` | 16 | 3,392 | itemsets K=3 | deterministic | [K-dist] K=3: 3,392 |
| L2-118 | `extreme_mining.log` | 17 | 5,147 | itemsets K=4 | deterministic | [K-dist] K=4: 5,147 |
| L2-119 | `extreme_mining.log` | 18 | 5,024 | itemsets K=5 | deterministic | [K-dist] K=5: 5,024 |
| L2-120 | `extreme_mining.log` | 19 | 3,847 | itemsets K=6 | deterministic | [K-dist] K=6: 3,847 |
| L2-121 | `extreme_mining.log` | 20 | 2,433 | itemsets K=7 | deterministic | [K-dist] K=7: 2,433 |
| L2-122 | `extreme_mining.log` | 21 | 1,253 | itemsets K=8 | deterministic | [K-dist] K=8: 1,253 |
| L2-123 | `extreme_mining.log` | 22 | 492 | itemsets K=9 | deterministic | [K-dist] K=9: 492 |
| L2-124 | `extreme_mining.log` | 23 | 160 | itemsets K=10 | deterministic | [K-dist] K=10: 160 |
| L2-125 | `extreme_mining.log` | 24 | 48 | itemsets K=11 | deterministic | [K-dist] K=11: 48 |
| L2-126 | `extreme_mining.log` | 25 | 11 | itemsets K=12 | deterministic | [K-dist] K=12: 11 |
| L2-127 | `extreme_mining.log` | 26 | 2 | itemsets K=13 | deterministic | [K-dist] K=13: 2 |
| L2-128 | `extreme_mining.log` | 31 | 2 | itemsets K=13 | deterministic | [deepest-patterns header] --- K=13 (2 itemsets) --- |
| L2-129 | `extreme_mining.log` | 32 | 0.000142 | support (fraction) | deterministic | [K=13 pattern] support=0.000142 (10,916 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0006260 + GO:0009432 … |
| L2-130 | `extreme_mining.log` | 32 | 10,916 | proteins (itemset support count) | deterministic | [K=13 pattern] support=0.000142 (10,916 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0006260 + GO:0009432 … |
| L2-131 | `extreme_mining.log` | 34 | 0.000142 | support (fraction) | deterministic | [K=13 pattern] support=0.000142 (10,913 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006260 + GO:0009432 … |
| L2-132 | `extreme_mining.log` | 34 | 10,913 | proteins (itemset support count) | deterministic | [K=13 pattern] support=0.000142 (10,913 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006260 + GO:0009432 … |
| L2-133 | `extreme_mining.log` | 37 | 11 | itemsets K=12 | deterministic | [deepest-patterns header] --- K=12 (11 itemsets) --- |
| L2-134 | `extreme_mining.log` | 38 | 0.000143 | support (fraction) | deterministic | [K=12 pattern] support=0.000143 (11,007 proteins) — PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0006260 + GO:0009432 … |
| L2-135 | `extreme_mining.log` | 38 | 11,007 | proteins (itemset support count) | deterministic | [K=12 pattern] support=0.000143 (11,007 proteins) — PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0006260 + GO:0009432 … |
| L2-136 | `extreme_mining.log` | 40 | 0.000143 | support (fraction) | deterministic | [K=12 pattern] support=0.000143 (10,978 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0009432 … |
| L2-137 | `extreme_mining.log` | 40 | 10,978 | proteins (itemset support count) | deterministic | [K=12 pattern] support=0.000143 (10,978 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0009432 … |
| L2-138 | `extreme_mining.log` | 42 | 0.000142 | support (fraction) | deterministic | [K=12 pattern] support=0.000142 (10,913 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0016787 + GO:0006310 + GO:0006260 + GO:0009432 + GO:0043138 … |
| L2-139 | `extreme_mining.log` | 42 | 10,913 | proteins (itemset support count) | deterministic | [K=12 pattern] support=0.000142 (10,913 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0016787 + GO:0006310 + GO:0006260 + GO:0009432 + GO:0043138 … |
| L2-140 | `extreme_mining.log` | 44 | 0.000142 | support (fraction) | deterministic | [K=12 pattern] support=0.000142 (10,912 proteins) — plddt_mean_med + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0009432 + GO:0043138 … |
| L2-141 | `extreme_mining.log` | 44 | 10,912 | proteins (itemset support count) | deterministic | [K=12 pattern] support=0.000142 (10,912 proteins) — plddt_mean_med + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0009432 + GO:0043138 … |
| L2-142 | `extreme_mining.log` | 46 | 0.000136 | support (fraction) | deterministic | [K=12 pattern] support=0.000136 (10,494 proteins) — PF00905 + PF00912 + GO:0005886 + GO:0006508 + GO:0071555 + GO:0008360 + GO:0009252 + GO:0030288 + GO:0046677 + GO:0008658 + GO:0009002 … |
| L2-143 | `extreme_mining.log` | 46 | 10,494 | proteins (itemset support count) | deterministic | [K=12 pattern] support=0.000136 (10,494 proteins) — PF00905 + PF00912 + GO:0005886 + GO:0006508 + GO:0071555 + GO:0008360 + GO:0009252 + GO:0030288 + GO:0046677 + GO:0008658 + GO:0009002 … |
| L2-144 | `extreme_mining.log` | 49 | 48 | itemsets K=11 | deterministic | [deepest-patterns header] --- K=11 (48 itemsets) --- |
| L2-145 | `extreme_mining.log` | 50 | 0.000230 | support (fraction) | deterministic | [K=11 pattern] support=0.000230 (17,686 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0043138 + GO:0043590 |
| L2-146 | `extreme_mining.log` | 50 | 17,686 | proteins (itemset support count) | deterministic | [K=11 pattern] support=0.000230 (17,686 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0005524 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0043138 + GO:0043590 |
| L2-147 | `extreme_mining.log` | 52 | 0.000207 | support (fraction) | deterministic | [K=11 pattern] support=0.000207 (15,886 proteins) — plddt_mean_med + PF00004 + PF10431 + PF17871 + PF02861 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0008233 + GO:0034605 |
| L2-148 | `extreme_mining.log` | 52 | 15,886 | proteins (itemset support count) | deterministic | [K=11 pattern] support=0.000207 (15,886 proteins) — plddt_mean_med + PF00004 + PF10431 + PF17871 + PF02861 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0008233 + GO:0034605 |
| L2-149 | `extreme_mining.log` | 54 | 0.000207 | support (fraction) | deterministic | [K=11 pattern] support=0.000207 (15,886 proteins) — PF00004 + PF07724 + PF10431 + PF17871 + PF02861 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0008233 + GO:0034605 |
| L2-150 | `extreme_mining.log` | 54 | 15,886 | proteins (itemset support count) | deterministic | [K=11 pattern] support=0.000207 (15,886 proteins) — PF00004 + PF07724 + PF10431 + PF17871 + PF02861 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0008233 + GO:0034605 |
| L2-151 | `extreme_mining.log` | 56 | 0.000202 | support (fraction) | deterministic | [K=11 pattern] support=0.000202 (15,565 proteins) — PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0043138 + GO:0043590 |
| L2-152 | `extreme_mining.log` | 56 | 15,565 | proteins (itemset support count) | deterministic | [K=11 pattern] support=0.000202 (15,565 proteins) — PF00271 + PF00270 + GO:0005524 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0043138 + GO:0043590 |
| L2-153 | `extreme_mining.log` | 58 | 0.000197 | support (fraction) | deterministic | [K=11 pattern] support=0.000197 (15,167 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0043138 + GO:0043590 |
| L2-154 | `extreme_mining.log` | 58 | 15,167 | proteins (itemset support count) | deterministic | [K=11 pattern] support=0.000197 (15,167 proteins) — plddt_mean_med + PF00271 + PF00270 + GO:0046872 + GO:0005737 + GO:0003677 + GO:0016787 + GO:0006281 + GO:0006310 + GO:0043138 + GO:0043590 |
| L2-155 | `extreme_mining.log` | 61 | 160 | itemsets K=10 | deterministic | [deepest-patterns header] --- K=10 (160 itemsets) --- |
| L2-156 | `extreme_mining.log` | 62 | 0.000240 | support (fraction) | deterministic | [K=10 pattern] support=0.000240 (18,425 proteins) — plddt_mean_med + PF00004 + PF07724 + PF10431 + PF17871 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0034605 |
| L2-157 | `extreme_mining.log` | 62 | 18,425 | proteins (itemset support count) | deterministic | [K=10 pattern] support=0.000240 (18,425 proteins) — plddt_mean_med + PF00004 + PF07724 + PF10431 + PF17871 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0034605 |
| L2-158 | `extreme_mining.log` | 64 | 0.000238 | support (fraction) | deterministic | [K=10 pattern] support=0.000238 (18,313 proteins) — plddt_mean_med + PF00004 + PF07724 + PF10431 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0008233 + GO:0034605 |
| L2-159 | `extreme_mining.log` | 64 | 18,313 | proteins (itemset support count) | deterministic | [K=10 pattern] support=0.000238 (18,313 proteins) — plddt_mean_med + PF00004 + PF07724 + PF10431 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0008233 + GO:0034605 |
| L2-160 | `extreme_mining.log` | 66 | 0.000216 | support (fraction) | deterministic | [K=10 pattern] support=0.000216 (16,614 proteins) — plddt_mean_med + PF00905 + PF00912 + GO:0005886 + GO:0006508 + GO:0071555 + GO:0009252 + GO:0008658 + GO:0009002 + GO:0008955 |
| L2-161 | `extreme_mining.log` | 66 | 16,614 | proteins (itemset support count) | deterministic | [K=10 pattern] support=0.000216 (16,614 proteins) — plddt_mean_med + PF00905 + PF00912 + GO:0005886 + GO:0006508 + GO:0071555 + GO:0009252 + GO:0008658 + GO:0009002 + GO:0008955 |
| L2-162 | `extreme_mining.log` | 68 | 0.000208 | support (fraction) | deterministic | [K=10 pattern] support=0.000208 (16,029 proteins) — plddt_mean_med + PF07724 + PF17871 + PF02861 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0008233 + GO:0034605 |
| L2-163 | `extreme_mining.log` | 68 | 16,029 | proteins (itemset support count) | deterministic | [K=10 pattern] support=0.000208 (16,029 proteins) — plddt_mean_med + PF07724 + PF17871 + PF02861 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0006508 + GO:0008233 + GO:0034605 |
| L2-164 | `extreme_mining.log` | 70 | 0.000207 | support (fraction) | deterministic | [K=10 pattern] support=0.000207 (15,936 proteins) — plddt_mean_med + PF07724 + PF10431 + PF17871 + PF02861 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0008233 + GO:0034605 |
| L2-165 | `extreme_mining.log` | 70 | 15,936 | proteins (itemset support count) | deterministic | [K=10 pattern] support=0.000207 (15,936 proteins) — plddt_mean_med + PF07724 + PF10431 + PF17871 + PF02861 + GO:0005524 + GO:0005737 + GO:0016887 + GO:0008233 + GO:0034605 |
| L2-166 | `godmode_mining.log` | 2 | 2026-02-09 06:42:58,031 | timestamp (run start, Direct CSR path) | external-fact | 2026-02-09 06:42:58,031 Direct CSR path: 76,890,945 transactions, min_count=8 |
| L2-167 | `godmode_mining.log` | 2 | 76,890,945 | transactions | deterministic | 2026-02-09 06:42:58,031 Direct CSR path: 76,890,945 transactions, min_count=8 |
| L2-168 | `godmode_mining.log` | 2 | 8 | min_count | method-parameter | 2026-02-09 06:42:58,031 Direct CSR path: 76,890,945 transactions, min_count=8 |
| L2-169 | `godmode_mining.log` | 3 | 1002 | frequent items (K=1) | deterministic | 2026-02-09 06:42:58,411 Direct CSR path: 1002 frequent items |
| L2-170 | `godmode_mining.log` | 4 | 316,421,093 | non-zeros (CSR nnz) | deterministic | 2026-02-09 06:42:59,144 Direct CSR path: 316,421,093 non-zeros |
| L2-171 | `godmode_mining.log` | 5 | 1,002 | frequent itemsets K=1 | deterministic | [per-K] K=1: 1,002 frequent (84ms) |
| L2-172 | `godmode_mining.log` | 6 | 73,786 | frequent itemsets K=2 | deterministic | [per-K] K=2: 73,786 frequent (2064ms) |
| L2-173 | `godmode_mining.log` | 7 | 452,777 | frequent itemsets K=3 | deterministic | [per-K] K=3: 452,777 frequent (25596ms) |
| L2-174 | `godmode_mining.log` | 8 | 1,184,461 | frequent itemsets K=4 | deterministic | [per-K] K=4: 1,184,461 frequent (37356ms) |
| L2-175 | `godmode_mining.log` | 9 | 1,974,126 | frequent itemsets K=5 | deterministic | [per-K] K=5: 1,974,126 frequent (35019ms) |
| L2-176 | `godmode_mining.log` | 10 | 2,626,332 | frequent itemsets K=6 | deterministic | [per-K] K=6: 2,626,332 frequent (22660ms) |
| L2-177 | `godmode_mining.log` | 11 | 3,118,459 | frequent itemsets K=7 | deterministic | [per-K] K=7: 3,118,459 frequent (14854ms) |
| L2-178 | `godmode_mining.log` | 12 | 3,442,954 | frequent itemsets K=8 | deterministic | [per-K] K=8: 3,442,954 frequent (12055ms) |
| L2-179 | `godmode_mining.log` | 13 | 3,529,257 | frequent itemsets K=9 | deterministic | [per-K] K=9: 3,529,257 frequent (11446ms) |
| L2-180 | `godmode_mining.log` | 14 | 3,293,612 | frequent itemsets K=10 | deterministic | [per-K] K=10: 3,293,612 frequent (10911ms) |
| L2-181 | `godmode_mining.log` | 15 | 2,739,532 | frequent itemsets K=11 | deterministic | [per-K] K=11: 2,739,532 frequent (9730ms) |
| L2-182 | `godmode_mining.log` | 16 | 1,996,772 | frequent itemsets K=12 | deterministic | [per-K] K=12: 1,996,772 frequent (8309ms) |
| L2-183 | `godmode_mining.log` | 17 | 1,259,045 | frequent itemsets K=13 | deterministic | [per-K] K=13: 1,259,045 frequent (6571ms) |
| L2-184 | `godmode_mining.log` | 18 | 679,471 | frequent itemsets K=14 | deterministic | [per-K] K=14: 679,471 frequent (5326ms) |
| L2-185 | `godmode_mining.log` | 19 | 310,527 | frequent itemsets K=15 | deterministic | [per-K] K=15: 310,527 frequent (1836ms) |
| L2-186 | `godmode_mining.log` | 20 | 118,659 | frequent itemsets K=16 | deterministic | [per-K] K=16: 118,659 frequent (674ms) |
| L2-187 | `godmode_mining.log` | 21 | 37,261 | frequent itemsets K=17 | deterministic | [per-K] K=17: 37,261 frequent (217ms) |
| L2-188 | `godmode_mining.log` | 22 | 9,375 | frequent itemsets K=18 | deterministic | [per-K] K=18: 9,375 frequent (61ms) |
| L2-189 | `godmode_mining.log` | 23 | 1,818 | frequent itemsets K=19 | deterministic | [per-K] K=19: 1,818 frequent (18ms) |
| L2-190 | `godmode_mining.log` | 24 | 255 | frequent itemsets K=20 | deterministic | [per-K] K=20: 255 frequent (11ms) |
| L2-191 | `godmode_mining.log` | 25 | 23 | frequent itemsets K=21 | deterministic | [per-K] K=21: 23 frequent (11ms) |
| L2-192 | `godmode_mining.log` | 26 | 1 | frequent itemsets K=22 | deterministic | [per-K] K=22: 1 frequent (13ms) |
| L2-193 | `godmode_mining.log` | 28 | 26,849,505 | itemsets (total) | deterministic | >>> 26,849,505 itemsets in 440.5s |
| L2-194 | `godmode_mining.log` | 28 | 440.5 | s (total) | hardware-dependent | >>> 26,849,505 itemsets in 440.5s |
| L2-195 | `godmode_mining.log` | 29 | 85,131,478 | bytes (output parquet) | software | >>> Saved: 85,131,478 bytes |
| L2-196 | `godmode_mining.log` | 31 | 1 | itemsets K=22 | deterministic | [pattern-block header] === K=22 (1 itemsets) === |
| L2-197 | `godmode_mining.log` | 32 | 8 | proteins (itemset support count) | deterministic | [K=22 pattern #1] #1 (8 proteins) — pLDDT: ['plddt_mean_med'] ; Pfam: ['PF00271', 'PF00270'] ; GO (19): ['GO:0005524', 'GO:0005737', 'GO:0005829', 'GO:0005634', 'GO:0016787', 'GO:0000287', 'GO:0005739', 'GO:0000978', 'GO:0030154', 'GO:0003697', 'GO:0003724', 'GO:0045087', … |
| L2-198 | `godmode_mining.log` | 37 | 23 | itemsets K=21 | deterministic | [pattern-block header] === K=21 (23 itemsets) === |
| L2-199 | `godmode_mining.log` | 38 | 13 | proteins (itemset support count) | deterministic | [K=21 pattern #1] #1 (13 proteins) — pLDDT: ['plddt_mean_med'] ; Pfam: ['PF00271', 'PF00270'] ; GO (18): ['GO:0005524', 'GO:0046872', 'GO:0003677', 'GO:0016887', 'GO:1990904', 'GO:0005730', 'GO:0006260', 'GO:0005654', 'GO:0006397', 'GO:0045944', 'GO:0003724', 'GO:0043138', … |
| L2-200 | `godmode_mining.log` | 42 | 8 | proteins (itemset support count) | deterministic | [K=21 pattern #2] #2 (8 proteins) — pLDDT: ['plddt_mean_med'] ; Pfam: ['PF00271', 'PF00270'] ; GO (18): ['GO:0005524', 'GO:0005737', 'GO:0005829', 'GO:0005634', 'GO:0016787', 'GO:0000287', 'GO:0005739', 'GO:0000978', 'GO:0030154', 'GO:0003697', 'GO:0003724', 'GO:0045087', … |
| L2-201 | `godmode_mining.log` | 46 | 8 | proteins (itemset support count) | deterministic | [K=21 pattern #3] #3 (8 proteins) — pLDDT: ['plddt_mean_med'] ; Pfam: ['PF00271', 'PF00270'] ; GO (18): ['GO:0005524', 'GO:0005737', 'GO:0005829', 'GO:0005634', 'GO:0016787', 'GO:0000287', 'GO:0005739', 'GO:0000978', 'GO:0030154', 'GO:0003697', 'GO:0003724', 'GO:0045087', … |
| L2-202 | `godmode_mining.log` | 51 | 255 | itemsets K=20 | deterministic | [pattern-block header] === K=20 (255 itemsets) === |
| L2-203 | `godmode_mining.log` | 52 | 57 | proteins (itemset support count) | deterministic | [K=20 pattern #1] #1 (57 proteins) — pLDDT: ['plddt_mean_med'] ; Pfam: ['PF00271', 'PF00270'] ; GO (17): ['GO:0005524', 'GO:0005829', 'GO:0016787', 'GO:0000287', 'GO:0005739', 'GO:0000978', 'GO:0030154', 'GO:0003697', 'GO:0003724', 'GO:0045087', 'GO:0030424', 'GO:0030425', … |
| L2-204 | `godmode_mining.log` | 56 | 40 | proteins (itemset support count) | deterministic | [K=20 pattern #2] #2 (40 proteins) — pLDDT: ['plddt_mean_med'] ; GO (19): ['GO:0005886', 'GO:0005829', 'GO:0008270', 'GO:0005739', 'GO:0051301', 'GO:0005730', 'GO:0006633', 'GO:0005874', 'GO:0045944', 'GO:0005694', 'GO:0016042', 'GO:0003682', 'GO:0005813', 'GO:0043161', 'GO:0000122', 'GO:0034599', … |
| L2-205 | `godmode_mining.log` | 59 | 15 | proteins (itemset support count) | deterministic | [K=20 pattern #3] #3 (15 proteins) — pLDDT: ['plddt_mean_med'] ; Pfam: ['PF00271'] ; GO (18): ['GO:0005524', 'GO:0046872', 'GO:0003677', 'GO:0016887', 'GO:1990904', 'GO:0005730', 'GO:0006260', 'GO:0005654', 'GO:0006397', 'GO:0045944', 'GO:0003724', 'GO:0043138', 'GO:0005813', … |
| L2-206 | `holdmybeer.log` | 3 | 4 | proteins (nominal min_count) | method-parameter | support = 4 proteins / 76.9M = 0.000005% |
| L2-207 | `holdmybeer.log` | 3 | 76.9M | transactions (rounded denominator) | deterministic | support = 4 proteins / 76.9M = 0.000005% |
| L2-208 | `holdmybeer.log` | 3 | 0.000005 | % support (nominal, rounded) | method-parameter | support = 4 proteins / 76.9M = 0.000005% |
| L2-209 | `holdmybeer.log` | 4 | 23+ | target K | method-parameter | Target: K=23+ |
| L2-210 | `holdmybeer.log` | 8 | 0.000000052 | min_support (fraction) | method-parameter | >>> Mining at support=0.000000052 (~4 proteins)... |
| L2-211 | `holdmybeer.log` | 8 | ~4 | proteins (nominal min_count) | method-parameter | >>> Mining at support=0.000000052 (~4 proteins)... |
| L2-212 | `holdmybeer.log` | 10 | 18,935,899 | itemsets (total) | deterministic | >>> 18,935,899 itemsets in 573.1s |
| L2-213 | `holdmybeer.log` | 10 | 573.1 | s (total) | hardware-dependent | >>> 18,935,899 itemsets in 573.1s |
| L2-214 | `holdmybeer.log` | 11 | 63,377,870 | bytes (output parquet) | software | >>> Saved: 63,377,870 bytes |
| L2-215 | `holdmybeer.log` | 14 | 1,002 | itemsets K=1 | deterministic | [K-dist] K=1: 1,002 |
| L2-216 | `holdmybeer.log` | 15 | 66,703 | itemsets K=2 | deterministic | [K-dist] K=2: 66,703 |
| L2-217 | `holdmybeer.log` | 16 | 403,157 | itemsets K=3 | deterministic | [K-dist] K=3: 403,157 |
| L2-218 | `holdmybeer.log` | 17 | 1,030,353 | itemsets K=4 | deterministic | [K-dist] K=4: 1,030,353 |
| L2-219 | `holdmybeer.log` | 18 | 1,664,538 | itemsets K=5 | deterministic | [K-dist] K=5: 1,664,538 |
| L2-220 | `holdmybeer.log` | 19 | 2,134,446 | itemsets K=6 | deterministic | [K-dist] K=6: 2,134,446 |
| L2-221 | `holdmybeer.log` | 20 | 2,434,007 | itemsets K=7 | deterministic | [K-dist] K=7: 2,434,007 |
| L2-222 | `holdmybeer.log` | 21 | 2,567,214 | itemsets K=8 | deterministic | [K-dist] K=8: 2,567,214 |
| L2-223 | `holdmybeer.log` | 22 | 2,493,943 | itemsets K=9 | deterministic | [K-dist] K=9: 2,493,943 |
| L2-224 | `holdmybeer.log` | 23 | 2,185,067 | itemsets K=10 | deterministic | [K-dist] K=10: 2,185,067 |
| L2-225 | `holdmybeer.log` | 24 | 1,689,219 | itemsets K=11 | deterministic | [K-dist] K=11: 1,689,219 |
| L2-226 | `holdmybeer.log` | 25 | 1,131,603 | itemsets K=12 | deterministic | [K-dist] K=12: 1,131,603 |
| L2-227 | `holdmybeer.log` | 26 | 646,988 | itemsets K=13 | deterministic | [K-dist] K=13: 646,988 |
| L2-228 | `holdmybeer.log` | 27 | 311,183 | itemsets K=14 | deterministic | [K-dist] K=14: 311,183 |
| L2-229 | `holdmybeer.log` | 28 | 123,916 | itemsets K=15 | deterministic | [K-dist] K=15: 123,916 |
| L2-230 | `holdmybeer.log` | 29 | 40,051 | itemsets K=16 | deterministic | [K-dist] K=16: 40,051 |
| L2-231 | `holdmybeer.log` | 30 | 10,227 | itemsets K=17 | deterministic | [K-dist] K=17: 10,227 |
| L2-232 | `holdmybeer.log` | 31 | 1,983 | itemsets K=18 | deterministic | [K-dist] K=18: 1,983 |
| L2-233 | `holdmybeer.log` | 32 | 274 | itemsets K=19 | deterministic | [K-dist] K=19: 274 |
| L2-234 | `holdmybeer.log` | 33 | 24 | itemsets K=20 | deterministic | [K-dist] K=20: 24 |
| L2-235 | `holdmybeer.log` | 34 | 1 | itemsets K=21 | deterministic | [K-dist] K=21: 1 |
| L2-236 | `holdmybeer.log` | 36 | 21 | K_max | deterministic | >>> MAX K = 21 |
| L2-237 | `holdmybeer.log` | 40 | 18,935,899 | itemsets (total, final line; 'max K=21, 573.1s') | deterministic | >>> HOLD MY BEER MODE COMPLETE. 18,935,899 itemsets, max K=21, 573.1s |
| L2-238 | `holdmybeer_real.log` | 3 | 4 | min_count (proteins) | method-parameter | min_count = 4 proteins (for real) |
| L2-239 | `holdmybeer_real.log` | 5 | 1.945333237480280e-08 | min_support (fraction) | method-parameter | min_support = 1.945333237480280e-08 |
| L2-240 | `holdmybeer_real.log` | 6 | 205620298 | transactions (denominator used in verify) | deterministic | verify: ceil(1.9453332374802804e-08 * 205620298) = 4 |
| L2-241 | `holdmybeer_real.log` | 6 | 4 | min_count (verified ceil) | method-parameter | verify: ceil(1.9453332374802804e-08 * 205620298) = 4 |
| L2-242 | `holdmybeer_real.log` | 8 | 48,007,493 | itemsets (total) | deterministic | >>> 48,007,493 itemsets in 1228.5s |
| L2-243 | `holdmybeer_real.log` | 8 | 1228.5 | s (total) | hardware-dependent | >>> 48,007,493 itemsets in 1228.5s |
| L2-244 | `holdmybeer_real.log` | 9 | 149,590,799 | bytes (output parquet) | software | >>> Saved: 149,590,799 bytes |
| L2-245 | `holdmybeer_real.log` | 12 | 1,002 | itemsets K=1 | deterministic | [K-dist] K=1: 1,002 |
| L2-246 | `holdmybeer_real.log` | 13 | 94,427 | itemsets K=2 | deterministic | [K-dist] K=2: 94,427 |
| L2-247 | `holdmybeer_real.log` | 14 | 603,403 | itemsets K=3 | deterministic | [K-dist] K=3: 603,403 |
| L2-248 | `holdmybeer_real.log` | 15 | 1,649,283 | itemsets K=4 | deterministic | [K-dist] K=4: 1,649,283 |
| L2-249 | `holdmybeer_real.log` | 16 | 2,916,124 | itemsets K=5 | deterministic | [K-dist] K=5: 2,916,124 |
| L2-250 | `holdmybeer_real.log` | 17 | 4,169,081 | itemsets K=6 | deterministic | [K-dist] K=6: 4,169,081 |
| L2-251 | `holdmybeer_real.log` | 18 | 5,318,506 | itemsets K=7 | deterministic | [K-dist] K=7: 5,318,506 |
| L2-252 | `holdmybeer_real.log` | 19 | 6,223,873 | itemsets K=8 | deterministic | [K-dist] K=8: 6,223,873 |
| L2-253 | `holdmybeer_real.log` | 20 | 6,644,170 | itemsets K=9 | deterministic | [K-dist] K=9: 6,644,170 |
| L2-254 | `holdmybeer_real.log` | 21 | 6,362,841 | itemsets K=10 | deterministic | [K-dist] K=10: 6,362,841 |
| L2-255 | `holdmybeer_real.log` | 22 | 5,373,365 | itemsets K=11 | deterministic | [K-dist] K=11: 5,373,365 |
| L2-256 | `holdmybeer_real.log` | 23 | 3,943,668 | itemsets K=12 | deterministic | [K-dist] K=12: 3,943,668 |
| L2-257 | `holdmybeer_real.log` | 24 | 2,484,117 | itemsets K=13 | deterministic | [K-dist] K=13: 2,484,117 |
| L2-258 | `holdmybeer_real.log` | 25 | 1,327,096 | itemsets K=14 | deterministic | [K-dist] K=14: 1,327,096 |
| L2-259 | `holdmybeer_real.log` | 26 | 593,694 | itemsets K=15 | deterministic | [K-dist] K=15: 593,694 |
| L2-260 | `holdmybeer_real.log` | 27 | 219,052 | itemsets K=16 | deterministic | [K-dist] K=16: 219,052 |
| L2-261 | `holdmybeer_real.log` | 28 | 65,356 | itemsets K=17 | deterministic | [K-dist] K=17: 65,356 |
| L2-262 | `holdmybeer_real.log` | 29 | 15,343 | itemsets K=18 | deterministic | [K-dist] K=18: 15,343 |
| L2-263 | `holdmybeer_real.log` | 30 | 2,722 | itemsets K=19 | deterministic | [K-dist] K=19: 2,722 |
| L2-264 | `holdmybeer_real.log` | 31 | 342 | itemsets K=20 | deterministic | [K-dist] K=20: 342 |
| L2-265 | `holdmybeer_real.log` | 32 | 27 | itemsets K=21 | deterministic | [K-dist] K=21: 27 |
| L2-266 | `holdmybeer_real.log` | 33 | 1 | itemsets K=22 | deterministic | [K-dist] K=22: 1 |
| L2-267 | `holdmybeer_real.log` | 35 | 22 | K_max | deterministic | >>> MAX K = 22 |
| L2-268 | `holdmybeer_real.log` | 37 | 1 | itemsets K=22 | deterministic | [pattern-block header] === K=22 (1 itemsets) === |
| L2-269 | `holdmybeer_real.log` | 38 | 8 | proteins (itemset support count) | deterministic | [K=22 pattern #1] #1 (8 proteins) — go_term: ['GO:0005524', 'GO:0005737', 'GO:0005829', 'GO:0005634', 'GO:0016787', 'GO:0000287', 'GO:0005739', 'GO:0000978', 'GO:0030154', 'GO:0003697', 'GO:0003724', 'GO:0045087', 'GO:0030424', 'GO:0030425', 'GO:0051607', 'GO:0003725', 'GO:0034605', 'GO:0016607', 'GO:0003678'] ; … |
| L2-270 | `holdmybeer_real.log` | 43 | 27 | itemsets K=21 | deterministic | [pattern-block header] === K=21 (27 itemsets) === |
| L2-271 | `holdmybeer_real.log` | 44 | 13 | proteins (itemset support count) | deterministic | [K=21 pattern #1] #1 (13 proteins) — go_term: ['GO:0005524', 'GO:0046872', 'GO:0003677', 'GO:0016887', 'GO:1990904', 'GO:0005730', 'GO:0006260', 'GO:0005654', 'GO:0006397', 'GO:0045944', 'GO:0003724', 'GO:0043138', 'GO:0005813', 'GO:0008380', 'GO:0006417', 'GO:0003725', 'GO:0006353', 'GO:0006954'] ; pfam: … |
| L2-272 | `holdmybeer_real.log` | 48 | 8 | proteins (itemset support count) | deterministic | [K=21 pattern #2] #2 (8 proteins) — go_term: ['GO:0005524', 'GO:0005737', 'GO:0005829', 'GO:0005634', 'GO:0016787', 'GO:0000287', 'GO:0005739', 'GO:0000978', 'GO:0030154', 'GO:0003697', 'GO:0003724', 'GO:0045087', 'GO:0030424', 'GO:0030425', 'GO:0051607', 'GO:0003725', 'GO:0034605', 'GO:0016607'] ; pfam: … |
| L2-273 | `holdmybeer_real.log` | 52 | 8 | proteins (itemset support count) | deterministic | [K=21 pattern #3] #3 (8 proteins) — go_term: ['GO:0005524', 'GO:0005737', 'GO:0005829', 'GO:0005634', 'GO:0016787', 'GO:0000287', 'GO:0005739', 'GO:0000978', 'GO:0030154', 'GO:0003697', 'GO:0003724', 'GO:0045087', 'GO:0030424', 'GO:0030425', 'GO:0051607', 'GO:0003725', 'GO:0034605', 'GO:0003678'] ; pfam: … |
| L2-274 | `holdmybeer_real.log` | 57 | 342 | itemsets K=20 | deterministic | [pattern-block header] === K=20 (342 itemsets) === |
| L2-275 | `holdmybeer_real.log` | 58 | 57 | proteins (itemset support count) | deterministic | [K=20 pattern #1] #1 (57 proteins) — go_term: ['GO:0005524', 'GO:0005829', 'GO:0016787', 'GO:0000287', 'GO:0005739', 'GO:0000978', 'GO:0030154', 'GO:0003697', 'GO:0003724', 'GO:0045087', 'GO:0030424', 'GO:0030425', 'GO:0051607', 'GO:0003725', 'GO:0034605', 'GO:0016607', 'GO:0003678'] ; pfam: ['PF00271', … |
| L2-276 | `holdmybeer_real.log` | 62 | 40 | proteins (itemset support count) | deterministic | [K=20 pattern #2] #2 (40 proteins) — go_term: ['GO:0005886', 'GO:0005829', 'GO:0008270', 'GO:0005739', 'GO:0051301', 'GO:0005730', 'GO:0006633', 'GO:0005874', 'GO:0045944', 'GO:0005694', 'GO:0016042', 'GO:0003682', 'GO:0005813', 'GO:0043161', 'GO:0000122', 'GO:0034599', 'GO:0070403', 'GO:0048471', 'GO:0043130'] ; … |
| L2-277 | `holdmybeer_real.log` | 65 | 15 | proteins (itemset support count) | deterministic | [K=20 pattern #3] #3 (15 proteins) — go_term: ['GO:0005524', 'GO:0046872', 'GO:0003677', 'GO:0016887', 'GO:1990904', 'GO:0005730', 'GO:0006260', 'GO:0005654', 'GO:0006397', 'GO:0045944', 'GO:0003724', 'GO:0043138', 'GO:0005813', 'GO:0008380', 'GO:0006417', 'GO:0003725', 'GO:0006353', 'GO:0006954'] ; pfam: … |
| L2-278 | `holdmybeer_real.log` | 70 | 48,007,493 | itemsets (total, final line; 'K=22, 1228.5s') | deterministic | >>> DONE. 48,007,493 itemsets, K=22, 1228.5s |
| L2-279 | `madman_mining.log` | 1 | 0.0001 | % support (on '214M TrEMBL') | method-parameter | >>> MADMAN SUPPORT: 0.0001% on 214M TrEMBL |
| L2-280 | `madman_mining.log` | 3 | 76,890,945 | transactions with >1 item | deterministic | With >1 item: 76,890,945 |
| L2-281 | `madman_mining.log` | 4 | 76 | min support count (script-computed; '0.0001%') | method-parameter | Min support = 0.0001% = 76 proteins |
| L2-282 | `madman_mining.log` | 6 | 0.0001 | % support (banner) | method-parameter | >>> GPU-RESIDENT STREAMING APRIORI — 0.0001% SUPPORT |
| L2-283 | `madman_mining.log` | 7 | 20 | max length (banner) | method-parameter | >>> MAX LENGTH 20 — ABSOLUTE MADMAN MODE |
| L2-284 | `pipeline_214m.log` | 2 | 214M | proteins (nominal, pipeline banner) | external-fact | === FULL 214M AlphaFold Pipeline === |
| L2-285 | `pipeline_214m.log` | 3 | Mon Feb  9 03:40:36 UTC 2026 | timestamp (pipeline start banner) | external-fact | === Mon Feb 9 03:40:36 UTC 2026 === |
| L2-286 | `pipeline_214m.log` | 7 | 150G | file size (uniprot_trembl.dat.gz, du/ls -h style) | external-fact | TrEMBL: 150G |
| L2-287 | `pipeline_214m.log` | 8 | 214683830 | rows (plddt_metadata.csv, incl. header) | external-fact | pLDDT: 214683830 rows |
| L2-288 | `pipeline_214m.log` | 9 | 2026-02-09T03:40:38.963Z | timestamp (af-extract start) | external-fact | [2026-02-09T03:40:38.963Z INFO af_extract] === Build Transactions from Metadata (pLDDT + annotations) === |
| L2-289 | `pipeline_214m.log` | 11–2035 | 100K → 202500K | DAT records parsed (2025 periodic lines, step [100] K, 2026-02-09T03:40:40.224Z → 2026-02-09T04:36:18.840Z) | external-fact | [consolidated periodic] first: [2026-02-09T03:40:40.224Z INFO af_extract::annotations] Parsed 100K DAT records... … last: [2026-02-09T04:36:18.840Z INFO af_extract::annotations] Parsed 202500K DAT records... |
| L2-290 | `pipeline_214m.log` | 2036 | 202556314 | annotation records loaded from DAT | external-fact | [2026-02-09T04:36:19.489Z INFO af_extract::annotations] Loaded 202556314 annotations (25475 Pfam domains, 26536 GO terms) |
| L2-291 | `pipeline_214m.log` | 2036 | 25475 | distinct Pfam domains in DAT | external-fact | [2026-02-09T04:36:19.489Z INFO af_extract::annotations] Loaded 202556314 annotations (25475 Pfam domains, 26536 GO terms) |
| L2-292 | `pipeline_214m.log` | 2036 | 26536 | distinct GO terms in DAT | external-fact | [2026-02-09T04:36:19.489Z INFO af_extract::annotations] Loaded 202556314 annotations (25475 Pfam domains, 26536 GO terms) |
| L2-293 | `pipeline_214m.log` | 2038 | 0 | CSV column index (Accession = 'accession') | software | [2026-02-09T04:36:19.489Z INFO af_extract] Accession column: 'accession' (index 0) |
| L2-294 | `pipeline_214m.log` | 2039 | 1 | CSV column index (pLDDT = 'mean_plddt') | software | [2026-02-09T04:36:19.489Z INFO af_extract] pLDDT column: 'mean_plddt' (index 1) |
| L2-295 | `pipeline_214m.log` | 2040 | 9566439 | proteins passed pLDDT filter after 10M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:21.257Z INFO af_extract] Read 10M rows (9566439 passed filter)... |
| L2-296 | `pipeline_214m.log` | 2041 | 19130885 | proteins passed pLDDT filter after 20M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:23.043Z INFO af_extract] Read 20M rows (19130885 passed filter)... |
| L2-297 | `pipeline_214m.log` | 2042 | 28709742 | proteins passed pLDDT filter after 30M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:24.853Z INFO af_extract] Read 30M rows (28709742 passed filter)... |
| L2-298 | `pipeline_214m.log` | 2043 | 38287797 | proteins passed pLDDT filter after 40M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:26.648Z INFO af_extract] Read 40M rows (38287797 passed filter)... |
| L2-299 | `pipeline_214m.log` | 2044 | 47865323 | proteins passed pLDDT filter after 50M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:28.480Z INFO af_extract] Read 50M rows (47865323 passed filter)... |
| L2-300 | `pipeline_214m.log` | 2045 | 57442488 | proteins passed pLDDT filter after 60M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:30.545Z INFO af_extract] Read 60M rows (57442488 passed filter)... |
| L2-301 | `pipeline_214m.log` | 2046 | 67021898 | proteins passed pLDDT filter after 70M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:32.427Z INFO af_extract] Read 70M rows (67021898 passed filter)... |
| L2-302 | `pipeline_214m.log` | 2047 | 76600156 | proteins passed pLDDT filter after 80M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:34.393Z INFO af_extract] Read 80M rows (76600156 passed filter)... |
| L2-303 | `pipeline_214m.log` | 2048 | 86178558 | proteins passed pLDDT filter after 90M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:36.295Z INFO af_extract] Read 90M rows (86178558 passed filter)... |
| L2-304 | `pipeline_214m.log` | 2049 | 95756671 | proteins passed pLDDT filter after 100M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:38.212Z INFO af_extract] Read 100M rows (95756671 passed filter)... |
| L2-305 | `pipeline_214m.log` | 2050 | 105334770 | proteins passed pLDDT filter after 110M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:40.093Z INFO af_extract] Read 110M rows (105334770 passed filter)... |
| L2-306 | `pipeline_214m.log` | 2051 | 114912567 | proteins passed pLDDT filter after 120M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:41.959Z INFO af_extract] Read 120M rows (114912567 passed filter)... |
| L2-307 | `pipeline_214m.log` | 2052 | 124492183 | proteins passed pLDDT filter after 130M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:43.844Z INFO af_extract] Read 130M rows (124492183 passed filter)... |
| L2-308 | `pipeline_214m.log` | 2053 | 134070226 | proteins passed pLDDT filter after 140M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:45.758Z INFO af_extract] Read 140M rows (134070226 passed filter)... |
| L2-309 | `pipeline_214m.log` | 2054 | 143648378 | proteins passed pLDDT filter after 150M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:47.702Z INFO af_extract] Read 150M rows (143648378 passed filter)... |
| L2-310 | `pipeline_214m.log` | 2055 | 153228159 | proteins passed pLDDT filter after 160M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:49.552Z INFO af_extract] Read 160M rows (153228159 passed filter)... |
| L2-311 | `pipeline_214m.log` | 2056 | 162806075 | proteins passed pLDDT filter after 170M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:51.414Z INFO af_extract] Read 170M rows (162806075 passed filter)... |
| L2-312 | `pipeline_214m.log` | 2057 | 172385565 | proteins passed pLDDT filter after 180M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:53.274Z INFO af_extract] Read 180M rows (172385565 passed filter)... |
| L2-313 | `pipeline_214m.log` | 2058 | 181963510 | proteins passed pLDDT filter after 190M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:55.195Z INFO af_extract] Read 190M rows (181963510 passed filter)... |
| L2-314 | `pipeline_214m.log` | 2059 | 191542750 | proteins passed pLDDT filter after 200M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:57.172Z INFO af_extract] Read 200M rows (191542750 passed filter)... |
| L2-315 | `pipeline_214m.log` | 2060 | 201120299 | proteins passed pLDDT filter after 210M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:59.041Z INFO af_extract] Read 210M rows (201120299 passed filter)... |
| L2-316 | `pipeline_214m.log` | 2061 | 214683829 | rows read (plddt_metadata.csv, data rows) | external-fact | [2026-02-09T04:36:59.926Z INFO af_extract] Read 214683829 rows: 205620298 passed pLDDT filter (>= 50), 9063531 skipped |
| L2-317 | `pipeline_214m.log` | 2061 | 205620298 | proteins passed pLDDT filter | deterministic | [2026-02-09T04:36:59.926Z INFO af_extract] Read 214683829 rows: 205620298 passed pLDDT filter (>= 50), 9063531 skipped |
| L2-318 | `pipeline_214m.log` | 2061 | 50 | min pLDDT threshold (>=) | method-parameter | [2026-02-09T04:36:59.926Z INFO af_extract] Read 214683829 rows: 205620298 passed pLDDT filter (>= 50), 9063531 skipped |
| L2-319 | `pipeline_214m.log` | 2061 | 9063531 | proteins skipped (pLDDT < 50) | deterministic | [2026-02-09T04:36:59.926Z INFO af_extract] Read 214683829 rows: 205620298 passed pLDDT filter (>= 50), 9063531 skipped |
| L2-320 | `pipeline_214m.log` | 2062 | 205620298 | proteins (frequency counting) | deterministic | [2026-02-09T04:36:59.926Z INFO af_extract] Counting Pfam/GO frequencies across 205620298 proteins... |
| L2-321 | `pipeline_214m.log` | 2063 | 24291 | unique Pfam among pLDDT-passing proteins | deterministic | [2026-02-09T04:39:56.574Z INFO af_extract] Frequencies: 24291 unique Pfam, 25993 unique GO from 205620298 proteins |
| L2-322 | `pipeline_214m.log` | 2063 | 25993 | unique GO among pLDDT-passing proteins | deterministic | [2026-02-09T04:39:56.574Z INFO af_extract] Frequencies: 24291 unique Pfam, 25993 unique GO from 205620298 proteins |
| L2-323 | `pipeline_214m.log` | 2064 | 6 | pLDDT bins (items) | method-parameter | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items |
| L2-324 | `pipeline_214m.log` | 2064 | 500 | top Pfam kept (--top-pfam) | method-parameter | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items |
| L2-325 | `pipeline_214m.log` | 2064 | 500 | top GO kept (--top-go) | method-parameter | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items |
| L2-326 | `pipeline_214m.log` | 2064 | 1006 | total items (vocabulary) | deterministic | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items |
| L2-327 | `pipeline_214m.log` | 2065 | 1006 | items in item mapping | deterministic | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Saved item mapping: 1006 items to /root/alphafold-data/full_214m/item_mapping_214m.parquet |
| L2-328 | `pipeline_214m.log` | 2066–2270 | 1M → 205M | transactions written (205 periodic lines, step [1] M, 2026-02-09T04:39:57.791Z → 2026-02-09T04:43:35.342Z) | deterministic | [consolidated periodic] first: [2026-02-09T04:39:57.791Z INFO af_extract::transaction] Written 1M transactions... … last: [2026-02-09T04:43:35.342Z INFO af_extract::transaction] Written 205M transactions... |
| L2-329 | `pipeline_214m.log` | 2271 | 205620298 | transactions written (parquet rows) | deterministic | [2026-02-09T04:43:35.994Z INFO af_extract::transaction] Written 205620298 transactions to /root/alphafold-data/full_214m/transactions_214m.parquet |
| L2-330 | `pipeline_214m.log` | 2272 | 2026-02-09T04:43:35.999Z | timestamp (af-extract Results banner = end of Step 1 binary) | external-fact | [2026-02-09T04:43:35.999Z INFO af_extract] === Results === |
| L2-331 | `pipeline_214m.log` | 2273 | 205620298 | transactions (af-extract summary) | deterministic | [2026-02-09T04:43:35.999Z INFO af_extract] Transactions: 205620298 |
| L2-332 | `pipeline_214m.log` | 2274 | 1006 | total items (af-extract summary) | deterministic | [2026-02-09T04:43:35.999Z INFO af_extract] Total items: 1006 |
| L2-333 | `pipeline_214m.log` | 2277 | 3777.0 | s (af-extract build-from-metadata self-timed) | hardware-dependent | [2026-02-09T04:43:35.999Z INFO af_extract] Time: 3777.0s (54440 proteins/sec) |
| L2-334 | `pipeline_214m.log` | 2277 | 54440 | proteins/sec (af-extract throughput) | hardware-dependent | [2026-02-09T04:43:35.999Z INFO af_extract] Time: 3777.0s (54440 proteins/sec) |
| L2-335 | `pipeline_214m.log` | 2279 | 65m31.040s | real time (bash `time` of Step 1) | hardware-dependent | real 65m31.040s |
| L2-336 | `pipeline_214m.log` | 2280 | 62m38.750s | user time (bash `time` of Step 1) | hardware-dependent | user 62m38.750s |
| L2-337 | `pipeline_214m.log` | 2281 | 2m52.144s | sys time (bash `time` of Step 1) | hardware-dependent | sys 2m52.144s |
| L2-338 | `pipeline_214m.log` | 2286 | 205,620,298 | transactions (Step 2 data stats) | deterministic | Total transactions: 205,620,298 |
| L2-339 | `pipeline_214m.log` | 2287 | 1006 | total items (Step 2 data stats) | deterministic | Total items: 1006 |
| L2-340 | `pipeline_214m.log` | 2289 | 2.2 | items per transaction (mean) | deterministic | Mean: 2.2 |
| L2-341 | `pipeline_214m.log` | 2290 | 46 | items per transaction (max) | deterministic | Max: 46 |
| L2-342 | `pipeline_214m.log` | 2291 | 76,890,945 | transactions with >1 item ('37.4%') | deterministic | With >1 item: 76,890,945 (37.4%) |
| L2-343 | `pipeline_214m.log` | 2294 | 76,890,945 | transactions mined (Step 3) | deterministic | Mining 76,890,945 annotated proteins... |
| L2-344 | `pipeline_214m.log` | 2297 | 113.9 | s (Step 3 mining) | hardware-dependent | Time: 113.9s |
| L2-345 | `pipeline_214m.log` | 2298 | 5,305 | itemsets (total, Step 3) | deterministic | Itemsets: 5,305 |
| L2-346 | `pipeline_214m.log` | 2299 | 667 | itemsets K=1 | deterministic | [K-dist] K=1: 667 |
| L2-347 | `pipeline_214m.log` | 2300 | 1,504 | itemsets K=2 | deterministic | [K-dist] K=2: 1,504 |
| L2-348 | `pipeline_214m.log` | 2301 | 1,378 | itemsets K=3 | deterministic | [K-dist] K=3: 1,378 |
| L2-349 | `pipeline_214m.log` | 2302 | 884 | itemsets K=4 | deterministic | [K-dist] K=4: 884 |
| L2-350 | `pipeline_214m.log` | 2303 | 514 | itemsets K=5 | deterministic | [K-dist] K=5: 514 |
| L2-351 | `pipeline_214m.log` | 2304 | 247 | itemsets K=6 | deterministic | [K-dist] K=6: 247 |
| L2-352 | `pipeline_214m.log` | 2305 | 89 | itemsets K=7 | deterministic | [K-dist] K=7: 89 |
| L2-353 | `pipeline_214m.log` | 2306 | 20 | itemsets K=8 | deterministic | [K-dist] K=8: 20 |
| L2-354 | `pipeline_214m.log` | 2307 | 2 | itemsets K=9 | deterministic | [K-dist] K=9: 2 |
| L2-355 | `pipeline_214m.log` | 2311 | 53,447 | association rules generated | deterministic | Generated 53,447 rules in 0.2s |
| L2-356 | `pipeline_214m.log` | 2311 | 0.2 | s (rule generation) | hardware-dependent | Generated 53,447 rules in 0.2s |
| L2-357 | `pipeline_214m.log` | 2314 | 0.998 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.998 lift= 969x PF01554 => GO:0015297 + GO:0042910 |
| L2-358 | `pipeline_214m.log` | 2314 | 969 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.998 lift= 969x PF01554 => GO:0015297 + GO:0042910 |
| L2-359 | `pipeline_214m.log` | 2315 | 0.999 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.999 lift= 969x GO:0015297 + GO:0042910 => PF01554 |
| L2-360 | `pipeline_214m.log` | 2315 | 969 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.999 lift= 969x GO:0015297 + GO:0042910 => PF01554 |
| L2-361 | `pipeline_214m.log` | 2316 | 1.000 | confidence | deterministic | [TOP 30 BY LIFT] conf=1.000 lift= 921x PF00116 => GO:0004129 + GO:0005507 |
| L2-362 | `pipeline_214m.log` | 2316 | 921 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=1.000 lift= 921x PF00116 => GO:0004129 + GO:0005507 |
| L2-363 | `pipeline_214m.log` | 2317 | 0.945 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 921x GO:0004129 + GO:0005507 => PF00116 |
| L2-364 | `pipeline_214m.log` | 2317 | 921 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 921x GO:0004129 + GO:0005507 => PF00116 |
| L2-365 | `pipeline_214m.log` | 2318 | 0.963 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.963 lift= 845x plddt_mean_med + GO:0043952 => GO:0005886 + GO:0065002 |
| L2-366 | `pipeline_214m.log` | 2318 | 845 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.963 lift= 845x plddt_mean_med + GO:0043952 => GO:0005886 + GO:0065002 |
| L2-367 | `pipeline_214m.log` | 2319 | 0.938 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.938 lift= 845x GO:0005886 + GO:0065002 => plddt_mean_med + GO:0043952 |
| L2-368 | `pipeline_214m.log` | 2319 | 845 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.938 lift= 845x GO:0005886 + GO:0065002 => plddt_mean_med + GO:0043952 |
| L2-369 | `pipeline_214m.log` | 2320 | 0.940 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.940 lift= 835x GO:0043952 => plddt_mean_med + GO:0005886 + GO:0065002 |
| L2-370 | `pipeline_214m.log` | 2320 | 835 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.940 lift= 835x GO:0043952 => plddt_mean_med + GO:0005886 + GO:0065002 |
| L2-371 | `pipeline_214m.log` | 2321 | 0.949 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.949 lift= 835x plddt_mean_med + GO:0005886 + GO:0065002 => GO:0043952 |
| L2-372 | `pipeline_214m.log` | 2321 | 835 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.949 lift= 835x plddt_mean_med + GO:0005886 + GO:0065002 => GO:0043952 |
| L2-373 | `pipeline_214m.log` | 2322 | 0.951 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.951 lift= 834x GO:0043952 => GO:0005886 + GO:0065002 |
| L2-374 | `pipeline_214m.log` | 2322 | 834 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.951 lift= 834x GO:0043952 => GO:0005886 + GO:0065002 |
| L2-375 | `pipeline_214m.log` | 2323 | 0.949 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.949 lift= 834x GO:0005886 + GO:0065002 => GO:0043952 |
| L2-376 | `pipeline_214m.log` | 2323 | 834 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.949 lift= 834x GO:0005886 + GO:0065002 => GO:0043952 |
| L2-377 | `pipeline_214m.log` | 2324 | 0.856 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.856 lift= 827x PF00849 => GO:0003723 + GO:0000455 |
| L2-378 | `pipeline_214m.log` | 2324 | 827 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.856 lift= 827x PF00849 => GO:0003723 + GO:0000455 |
| L2-379 | `pipeline_214m.log` | 2325 | 0.992 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.992 lift= 827x GO:0003723 + GO:0000455 => PF00849 |
| L2-380 | `pipeline_214m.log` | 2325 | 827 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.992 lift= 827x GO:0003723 + GO:0000455 => PF00849 |
| L2-381 | `pipeline_214m.log` | 2326 | 0.902 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.902 lift= 816x GO:0003677 + GO:0000786 => GO:0046982 + GO:0030527 |
| L2-382 | `pipeline_214m.log` | 2326 | 816 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.902 lift= 816x GO:0003677 + GO:0000786 => GO:0046982 + GO:0030527 |
| L2-383 | `pipeline_214m.log` | 2327 | 0.986 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.986 lift= 816x GO:0046982 + GO:0030527 => GO:0003677 + GO:0000786 |
| L2-384 | `pipeline_214m.log` | 2327 | 816 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.986 lift= 816x GO:0046982 + GO:0030527 => GO:0003677 + GO:0000786 |
| L2-385 | `pipeline_214m.log` | 2328 | 0.945 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 809x PF13853 => GO:0005886 + GO:0004930 + GO:0004984 |
| L2-386 | `pipeline_214m.log` | 2328 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 809x PF13853 => GO:0005886 + GO:0004930 + GO:0004984 |
| L2-387 | `pipeline_214m.log` | 2329 | 0.983 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.983 lift= 809x GO:0005886 + GO:0004930 + GO:0004984 => PF13853 |
| L2-388 | `pipeline_214m.log` | 2329 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.983 lift= 809x GO:0005886 + GO:0004930 + GO:0004984 => PF13853 |
| L2-389 | `pipeline_214m.log` | 2330 | 0.940 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.940 lift= 809x PF13853 => plddt_mean_med + GO:0005886 + GO:0004930 + GO:0004984 |
| L2-390 | `pipeline_214m.log` | 2330 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.940 lift= 809x PF13853 => plddt_mean_med + GO:0005886 + GO:0004930 + GO:0004984 |
| L2-391 | `pipeline_214m.log` | 2331 | 0.983 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.983 lift= 809x plddt_mean_med + GO:0005886 + GO:0004930 + GO:0004984 => PF13853 |
| L2-392 | `pipeline_214m.log` | 2331 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.983 lift= 809x plddt_mean_med + GO:0005886 + GO:0004930 + GO:0004984 => PF13853 |
| L2-393 | `pipeline_214m.log` | 2332 | 0.945 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 809x plddt_mean_med + PF13853 => GO:0005886 + GO:0004930 + GO:0004984 |
| L2-394 | `pipeline_214m.log` | 2332 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 809x plddt_mean_med + PF13853 => GO:0005886 + GO:0004930 + GO:0004984 |
| L2-395 | `pipeline_214m.log` | 2333 | 0.978 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.978 lift= 809x GO:0005886 + GO:0004930 + GO:0004984 => plddt_mean_med + PF13853 |
| L2-396 | `pipeline_214m.log` | 2333 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.978 lift= 809x GO:0005886 + GO:0004930 + GO:0004984 => plddt_mean_med + PF13853 |
| L2-397 | `pipeline_214m.log` | 2334 | 0.960 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.960 lift= 802x GO:0000455 => PF00849 + GO:0003723 |
| L2-398 | `pipeline_214m.log` | 2334 | 802 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.960 lift= 802x GO:0000455 => PF00849 + GO:0003723 |
| L2-399 | `pipeline_214m.log` | 2335 | 0.857 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.857 lift= 802x PF00849 + GO:0003723 => GO:0000455 |
| L2-400 | `pipeline_214m.log` | 2335 | 802 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.857 lift= 802x PF00849 + GO:0003723 => GO:0000455 |
| L2-401 | `pipeline_214m.log` | 2336 | 0.856 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.856 lift= 801x PF00849 => GO:0000455 |
| L2-402 | `pipeline_214m.log` | 2336 | 801 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.856 lift= 801x PF00849 => GO:0000455 |
| L2-403 | `pipeline_214m.log` | 2337 | 0.960 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.960 lift= 801x GO:0000455 => PF00849 |
| L2-404 | `pipeline_214m.log` | 2337 | 801 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.960 lift= 801x GO:0000455 => PF00849 |
| L2-405 | `pipeline_214m.log` | 2338 | 0.967 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0015833 => PF08352 + GO:0005524 + GO:0016887 |
| L2-406 | `pipeline_214m.log` | 2338 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0015833 => PF08352 + GO:0005524 + GO:0016887 |
| L2-407 | `pipeline_214m.log` | 2339 | 0.967 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0005524 + GO:0015833 => PF08352 + GO:0016887 |
| L2-408 | `pipeline_214m.log` | 2339 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0005524 + GO:0015833 => PF08352 + GO:0016887 |
| L2-409 | `pipeline_214m.log` | 2340 | 0.967 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0015833 => PF08352 + GO:0016887 |
| L2-410 | `pipeline_214m.log` | 2340 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0015833 => PF08352 + GO:0016887 |
| L2-411 | `pipeline_214m.log` | 2341 | 0.990 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0016887 => PF00005 + GO:0005524 + GO:0015833 |
| L2-412 | `pipeline_214m.log` | 2341 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0016887 => PF00005 + GO:0005524 + GO:0015833 |
| L2-413 | `pipeline_214m.log` | 2342 | 0.990 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0005524 + GO:0016887 => PF00005 + GO:0015833 |
| L2-414 | `pipeline_214m.log` | 2342 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0005524 + GO:0016887 => PF00005 + GO:0015833 |
| L2-415 | `pipeline_214m.log` | 2343 | 0.990 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0016887 => PF00005 + GO:0015833 |
| L2-416 | `pipeline_214m.log` | 2343 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0016887 => PF00005 + GO:0015833 |
| L2-417 | `pipeline_214m.log` | 2346 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 921x PF00116 => GO:0004129 + GO:0005507 |
| L2-418 | `pipeline_214m.log` | 2346 | 921 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 921x PF00116 => GO:0004129 + GO:0005507 |
| L2-419 | `pipeline_214m.log` | 2347 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 776x PF00905 + GO:0071555 => GO:0008658 |
| L2-420 | `pipeline_214m.log` | 2347 | 776 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 776x PF00905 + GO:0071555 => GO:0008658 |
| L2-421 | `pipeline_214m.log` | 2348 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 776x PF00905 => GO:0008658 |
| L2-422 | `pipeline_214m.log` | 2348 | 776 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 776x PF00905 => GO:0008658 |
| L2-423 | `pipeline_214m.log` | 2349 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 735x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0006397 + GO:0008033 |
| L2-424 | `pipeline_214m.log` | 2349 | 735 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 735x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0006397 + GO:0008033 |
| L2-425 | `pipeline_214m.log` | 2350 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 735x PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0006397 + GO:0008033 |
| L2-426 | `pipeline_214m.log` | 2350 | 735 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 735x PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0006397 + GO:0008033 |
| L2-427 | `pipeline_214m.log` | 2351 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x plddt_mean_med + PF01824 + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-428 | `pipeline_214m.log` | 2351 | 734 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x plddt_mean_med + PF01824 + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-429 | `pipeline_214m.log` | 2352 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-430 | `pipeline_214m.log` | 2352 | 734 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-431 | `pipeline_214m.log` | 2353 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x plddt_mean_med + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-432 | `pipeline_214m.log` | 2353 | 734 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x plddt_mean_med + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-433 | `pipeline_214m.log` | 2354 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x PF01824 + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-434 | `pipeline_214m.log` | 2354 | 734 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x PF01824 + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-435 | `pipeline_214m.log` | 2355 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0003723 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-436 | `pipeline_214m.log` | 2355 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0003723 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-437 | `pipeline_214m.log` | 2356 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-438 | `pipeline_214m.log` | 2356 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-439 | `pipeline_214m.log` | 2357 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-440 | `pipeline_214m.log` | 2357 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-441 | `pipeline_214m.log` | 2358 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01348 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-442 | `pipeline_214m.log` | 2358 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01348 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-443 | `pipeline_214m.log` | 2359 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + GO:0003723 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-444 | `pipeline_214m.log` | 2359 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + GO:0003723 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-445 | `pipeline_214m.log` | 2360 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01348 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-446 | `pipeline_214m.log` | 2360 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01348 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-447 | `pipeline_214m.log` | 2361 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-448 | `pipeline_214m.log` | 2361 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-449 | `pipeline_214m.log` | 2362 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01824 + PF01348 + GO:0009507 + GO:0006397 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-450 | `pipeline_214m.log` | 2362 | 663 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01824 + PF01348 + GO:0009507 + GO:0006397 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-451 | `pipeline_214m.log` | 2363 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01348 + GO:0009507 + GO:0006397 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-452 | `pipeline_214m.log` | 2363 | 663 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01348 + GO:0009507 + GO:0006397 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-453 | `pipeline_214m.log` | 2364 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-454 | `pipeline_214m.log` | 2364 | 663 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-455 | `pipeline_214m.log` | 2365 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-456 | `pipeline_214m.log` | 2365 | 663 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-457 | `pipeline_214m.log` | 2368 | 53,447 | association rules (summary) | deterministic | Total rules: 53,447 |
| L2-458 | `pipeline_214m.log` | 2369 | 12,776 | rules with confidence >= 99% | deterministic | Confidence >= 99%: 12,776 |
| L2-459 | `pipeline_214m.log` | 2370 | 31,775 | rules with confidence >= 90% | deterministic | Confidence >= 90%: 31,775 |
| L2-460 | `pipeline_214m.log` | 2371 | 51,126 | rules with lift >= 5.0 | deterministic | Lift >= 5.0: 51,126 |
| L2-461 | `pipeline_214m.log` | 2372 | 40,428 | rules with lift >= 100 | deterministic | Lift >= 100: 40,428 |
| L2-462 | `pipeline_214m.log` | 2373 | 45,286 | cross-domain rules (Pfam<=>GO) | deterministic | Cross-domain (Pfam<=>GO): 45,286 |
| L2-463 | `pipeline_214m.log` | 2376 | Mon Feb  9 04:48:08 UTC 2026 | timestamp (pipeline end banner) | external-fact | === PIPELINE COMPLETE — Mon Feb 9 04:48:08 UTC 2026 === |
| L2-464 | `ultra_mining.log` | 1 | 0.01 | % support (on '214M TrEMBL') | method-parameter | >>> ULTRA LOW SUPPORT: 0.01% on 214M TrEMBL |
| L2-465 | `ultra_mining.log` | 3 | 205,620,298 | transactions (total in parquet) | deterministic | Total transactions: 205,620,298 |
| L2-466 | `ultra_mining.log` | 4 | 76,890,945 | transactions with >1 item | deterministic | With >1 item: 76,890,945 |
| L2-467 | `ultra_mining.log` | 5 | 7,689 | min support count (script-computed; '0.01%') | method-parameter | Min support = 0.01% = 7,689 proteins |
| L2-468 | `ultra_mining.log` | 7 | 0.01 | % support (banner) | method-parameter | >>> GPU-RESIDENT STREAMING APRIORI — 0.01% SUPPORT |
| L2-469 | `ultra_mining.log` | 8 | H100s (plural) | GPU name (only hardware string in any log) | hardware-dependent | >>> RELEASING THE H100s... |
| L2-470 | `ultra_mining.log` | 12 | 257.6 | s (mining) | hardware-dependent | Time: 257.6s |
| L2-471 | `ultra_mining.log` | 13 | 51,124 | itemsets (total) | deterministic | Itemsets: 51,124 |
| L2-472 | `ultra_mining.log` | 14 | 455 | itemsets K=1 | deterministic | [K-dist] K=1: 455 |
| L2-473 | `ultra_mining.log` | 15 | 4,152 | itemsets K=2 | deterministic | [K-dist] K=2: 4,152 |
| L2-474 | `ultra_mining.log` | 16 | 8,936 | itemsets K=3 | deterministic | [K-dist] K=3: 8,936 |
| L2-475 | `ultra_mining.log` | 17 | 10,191 | itemsets K=4 | deterministic | [K-dist] K=4: 10,191 |
| L2-476 | `ultra_mining.log` | 18 | 9,153 | itemsets K=5 | deterministic | [K-dist] K=5: 9,153 |
| L2-477 | `ultra_mining.log` | 19 | 7,177 | itemsets K=6 | deterministic | [K-dist] K=6: 7,177 |
| L2-478 | `ultra_mining.log` | 20 | 5,217 | itemsets K=7 | deterministic | [K-dist] K=7: 5,217 |
| L2-479 | `ultra_mining.log` | 21 | 3,208 | itemsets K=8 | deterministic | [K-dist] K=8: 3,208 |
| L2-480 | `ultra_mining.log` | 22 | 1,651 | itemsets K=9 | deterministic | [K-dist] K=9: 1,651 |
| L2-481 | `ultra_mining.log` | 23 | 707 | itemsets K=10 | deterministic | [K-dist] K=10: 707 |
| L2-482 | `ultra_mining.log` | 24 | 222 | itemsets K=11 | deterministic | [K-dist] K=11: 222 |
| L2-483 | `ultra_mining.log` | 25 | 48 | itemsets K=12 | deterministic | [K-dist] K=12: 48 |
| L2-484 | `ultra_mining.log` | 26 | 7 | itemsets K=13 | deterministic | [K-dist] K=13: 7 |
| L2-485 | `ultra_mining.log` | 32 | 0.07093 | support (fraction) | deterministic | [K=2 pattern] support=0.07093 (5,453,948 proteins) — plddt_mean_high + GO:0046872 |
| L2-486 | `ultra_mining.log` | 32 | 5,453,948 | proteins (itemset support count) | deterministic | [K=2 pattern] support=0.07093 (5,453,948 proteins) — plddt_mean_high + GO:0046872 |
| L2-487 | `ultra_mining.log` | 34 | 0.05538 | support (fraction) | deterministic | [K=2 pattern] support=0.05538 (4,257,942 proteins) — plddt_mean_med + GO:0005737 |
| L2-488 | `ultra_mining.log` | 34 | 4,257,942 | proteins (itemset support count) | deterministic | [K=2 pattern] support=0.05538 (4,257,942 proteins) — plddt_mean_med + GO:0005737 |
| L2-489 | `ultra_mining.log` | 36 | 0.04204 | support (fraction) | deterministic | [K=2 pattern] support=0.04204 (3,232,658 proteins) — plddt_mean_high + GO:0005524 |
| L2-490 | `ultra_mining.log` | 36 | 3,232,658 | proteins (itemset support count) | deterministic | [K=2 pattern] support=0.04204 (3,232,658 proteins) — plddt_mean_high + GO:0005524 |
| L2-491 | `ultra_mining.log` | 40 | 0.10739 | support (fraction) | deterministic | [K=1 pattern] support=0.10739 (8,257,363 proteins) — GO:0005737 |
| L2-492 | `ultra_mining.log` | 40 | 8,257,363 | proteins (itemset support count) | deterministic | [K=1 pattern] support=0.10739 (8,257,363 proteins) — GO:0005737 |
| L2-493 | `ultra_mining.log` | 42 | 0.08286 | support (fraction) | deterministic | [K=1 pattern] support=0.08286 (6,370,846 proteins) — GO:0005829 |
| L2-494 | `ultra_mining.log` | 42 | 6,370,846 | proteins (itemset support count) | deterministic | [K=1 pattern] support=0.08286 (6,370,846 proteins) — GO:0005829 |
| L2-495 | `ultra_mining.log` | 44 | 0.06854 | support (fraction) | deterministic | [K=1 pattern] support=0.06854 (5,270,392 proteins) — GO:0003677 |
| L2-496 | `ultra_mining.log` | 44 | 5,270,392 | proteins (itemset support count) | deterministic | [K=1 pattern] support=0.06854 (5,270,392 proteins) — GO:0003677 |
| L2-497 | `watcher.log` | 4 | 149GiB | aria2c total size (uniprot_trembl.dat.gz) | external-fact | [aria2c progress line, 2612 bytes, 45 snapshots / 25 distinct] first: [145G] 62GiB/149GiB(42%) CN:16 DL:50MiB ETA:29m13s |
| L2-498 | `watcher.log` | 4 | 62GiB/149GiB (42%) | first snapshot (DL 50MiB/s, ETA 29m13s) | hardware-dependent | [aria2c first snapshot] [145G] 62GiB/149GiB(42%) CN:16 DL:50MiB ETA:29m13s |
| L2-499 | `watcher.log` | 4 | 145GiB/149GiB (97%) | last snapshot (DL 55MiB/s, ETA 1m18s); no 100% snapshot present | hardware-dependent | [aria2c last snapshot] [150G] 145GiB/149GiB(97%) CN:16 DL:55MiB ETA:1m18s |
| L2-500 | `watcher.log` | 4 | 16 | aria2c connections (CN) | method-parameter | [aria2c] CN:16 in all 45 snapshots |
| L2-501 | `watcher.log` | 4 | 26–100 | MiB/s download rate range across snapshots | hardware-dependent | [aria2c] DL values observed: 26, 39, 44, 46, 47, 48, 50, 54, 55, 56, 57, 60, 63, 64, 66, 67, 69, 73, 77, 100 MiB |
| L2-502 | `watcher.log` | 4 | 145G → 150G | disk usage bracket ([NNNG] prefix) first → last | hardware-dependent | [aria2c] bracket prefix values: 145, 146, 147, 148, 149, 150 |
| L2-503 | `watcher.log` | 4 | 45 / 25 (CR-separated segments: 46) | snapshots (total / consecutive-distinct) | software | [aria2c] 62GiB(42%) DL:50MiB ETA:29m13s ; 66GiB(44%) DL:57MiB ETA:24m53s ; 70GiB(47%) DL:66MiB ETA:20m24s ; 73GiB(49%) DL:50MiB ETA:25m29s ; 76GiB(51%) DL:46MiB ETA:26m59s ; 83GiB(55%) DL:64MiB ETA:17m26s ; … |
| L2-504 | `watcher.log` | 6 | 150G | file size (downloaded uniprot_trembl.dat.gz) | external-fact | >>> aria2c FINISHED! File size: 150G |
| L2-505 | `watcher.log` | 7 | Mon Feb  9 03:40:36 UTC 2026 | timestamp (watcher hand-off to pipeline) | external-fact | >>> Starting full pipeline at Mon Feb 9 03:40:36 UTC 2026 |
| L2-506 | `watcher.log` | 11 | 214M | proteins (nominal, pipeline banner) | external-fact | === FULL 214M AlphaFold Pipeline === |
| L2-507 | `watcher.log` | 12 | Mon Feb  9 03:40:36 UTC 2026 | timestamp (pipeline start banner) | external-fact | === Mon Feb 9 03:40:36 UTC 2026 === |
| L2-508 | `watcher.log` | 16 | 150G | file size (uniprot_trembl.dat.gz, du/ls -h style) | external-fact | TrEMBL: 150G |
| L2-509 | `watcher.log` | 17 | 214683830 | rows (plddt_metadata.csv, incl. header) | external-fact | pLDDT: 214683830 rows |
| L2-510 | `watcher.log` | 18 | 2026-02-09T03:40:38.963Z | timestamp (af-extract start) | external-fact | [2026-02-09T03:40:38.963Z INFO af_extract] === Build Transactions from Metadata (pLDDT + annotations) === |
| L2-511 | `watcher.log` | 20–2044 | 100K → 202500K | DAT records parsed (2025 periodic lines, step [100] K, 2026-02-09T03:40:40.224Z → 2026-02-09T04:36:18.840Z) | external-fact | [consolidated periodic] first: [2026-02-09T03:40:40.224Z INFO af_extract::annotations] Parsed 100K DAT records... … last: [2026-02-09T04:36:18.840Z INFO af_extract::annotations] Parsed 202500K DAT records... |
| L2-512 | `watcher.log` | 2045 | 202556314 | annotation records loaded from DAT | external-fact | [2026-02-09T04:36:19.489Z INFO af_extract::annotations] Loaded 202556314 annotations (25475 Pfam domains, 26536 GO terms) |
| L2-513 | `watcher.log` | 2045 | 25475 | distinct Pfam domains in DAT | external-fact | [2026-02-09T04:36:19.489Z INFO af_extract::annotations] Loaded 202556314 annotations (25475 Pfam domains, 26536 GO terms) |
| L2-514 | `watcher.log` | 2045 | 26536 | distinct GO terms in DAT | external-fact | [2026-02-09T04:36:19.489Z INFO af_extract::annotations] Loaded 202556314 annotations (25475 Pfam domains, 26536 GO terms) |
| L2-515 | `watcher.log` | 2047 | 0 | CSV column index (Accession = 'accession') | software | [2026-02-09T04:36:19.489Z INFO af_extract] Accession column: 'accession' (index 0) |
| L2-516 | `watcher.log` | 2048 | 1 | CSV column index (pLDDT = 'mean_plddt') | software | [2026-02-09T04:36:19.489Z INFO af_extract] pLDDT column: 'mean_plddt' (index 1) |
| L2-517 | `watcher.log` | 2049 | 9566439 | proteins passed pLDDT filter after 10M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:21.257Z INFO af_extract] Read 10M rows (9566439 passed filter)... |
| L2-518 | `watcher.log` | 2050 | 19130885 | proteins passed pLDDT filter after 20M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:23.043Z INFO af_extract] Read 20M rows (19130885 passed filter)... |
| L2-519 | `watcher.log` | 2051 | 28709742 | proteins passed pLDDT filter after 30M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:24.853Z INFO af_extract] Read 30M rows (28709742 passed filter)... |
| L2-520 | `watcher.log` | 2052 | 38287797 | proteins passed pLDDT filter after 40M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:26.648Z INFO af_extract] Read 40M rows (38287797 passed filter)... |
| L2-521 | `watcher.log` | 2053 | 47865323 | proteins passed pLDDT filter after 50M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:28.480Z INFO af_extract] Read 50M rows (47865323 passed filter)... |
| L2-522 | `watcher.log` | 2054 | 57442488 | proteins passed pLDDT filter after 60M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:30.545Z INFO af_extract] Read 60M rows (57442488 passed filter)... |
| L2-523 | `watcher.log` | 2055 | 67021898 | proteins passed pLDDT filter after 70M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:32.427Z INFO af_extract] Read 70M rows (67021898 passed filter)... |
| L2-524 | `watcher.log` | 2056 | 76600156 | proteins passed pLDDT filter after 80M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:34.393Z INFO af_extract] Read 80M rows (76600156 passed filter)... |
| L2-525 | `watcher.log` | 2057 | 86178558 | proteins passed pLDDT filter after 90M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:36.295Z INFO af_extract] Read 90M rows (86178558 passed filter)... |
| L2-526 | `watcher.log` | 2058 | 95756671 | proteins passed pLDDT filter after 100M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:38.212Z INFO af_extract] Read 100M rows (95756671 passed filter)... |
| L2-527 | `watcher.log` | 2059 | 105334770 | proteins passed pLDDT filter after 110M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:40.093Z INFO af_extract] Read 110M rows (105334770 passed filter)... |
| L2-528 | `watcher.log` | 2060 | 114912567 | proteins passed pLDDT filter after 120M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:41.959Z INFO af_extract] Read 120M rows (114912567 passed filter)... |
| L2-529 | `watcher.log` | 2061 | 124492183 | proteins passed pLDDT filter after 130M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:43.844Z INFO af_extract] Read 130M rows (124492183 passed filter)... |
| L2-530 | `watcher.log` | 2062 | 134070226 | proteins passed pLDDT filter after 140M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:45.758Z INFO af_extract] Read 140M rows (134070226 passed filter)... |
| L2-531 | `watcher.log` | 2063 | 143648378 | proteins passed pLDDT filter after 150M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:47.702Z INFO af_extract] Read 150M rows (143648378 passed filter)... |
| L2-532 | `watcher.log` | 2064 | 153228159 | proteins passed pLDDT filter after 160M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:49.552Z INFO af_extract] Read 160M rows (153228159 passed filter)... |
| L2-533 | `watcher.log` | 2065 | 162806075 | proteins passed pLDDT filter after 170M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:51.414Z INFO af_extract] Read 170M rows (162806075 passed filter)... |
| L2-534 | `watcher.log` | 2066 | 172385565 | proteins passed pLDDT filter after 180M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:53.274Z INFO af_extract] Read 180M rows (172385565 passed filter)... |
| L2-535 | `watcher.log` | 2067 | 181963510 | proteins passed pLDDT filter after 190M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:55.195Z INFO af_extract] Read 190M rows (181963510 passed filter)... |
| L2-536 | `watcher.log` | 2068 | 191542750 | proteins passed pLDDT filter after 200M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:57.172Z INFO af_extract] Read 200M rows (191542750 passed filter)... |
| L2-537 | `watcher.log` | 2069 | 201120299 | proteins passed pLDDT filter after 210M rows | deterministic | [pLDDT CSV progress] [2026-02-09T04:36:59.041Z INFO af_extract] Read 210M rows (201120299 passed filter)... |
| L2-538 | `watcher.log` | 2070 | 214683829 | rows read (plddt_metadata.csv, data rows) | external-fact | [2026-02-09T04:36:59.926Z INFO af_extract] Read 214683829 rows: 205620298 passed pLDDT filter (>= 50), 9063531 skipped |
| L2-539 | `watcher.log` | 2070 | 205620298 | proteins passed pLDDT filter | deterministic | [2026-02-09T04:36:59.926Z INFO af_extract] Read 214683829 rows: 205620298 passed pLDDT filter (>= 50), 9063531 skipped |
| L2-540 | `watcher.log` | 2070 | 50 | min pLDDT threshold (>=) | method-parameter | [2026-02-09T04:36:59.926Z INFO af_extract] Read 214683829 rows: 205620298 passed pLDDT filter (>= 50), 9063531 skipped |
| L2-541 | `watcher.log` | 2070 | 9063531 | proteins skipped (pLDDT < 50) | deterministic | [2026-02-09T04:36:59.926Z INFO af_extract] Read 214683829 rows: 205620298 passed pLDDT filter (>= 50), 9063531 skipped |
| L2-542 | `watcher.log` | 2071 | 205620298 | proteins (frequency counting) | deterministic | [2026-02-09T04:36:59.926Z INFO af_extract] Counting Pfam/GO frequencies across 205620298 proteins... |
| L2-543 | `watcher.log` | 2072 | 24291 | unique Pfam among pLDDT-passing proteins | deterministic | [2026-02-09T04:39:56.574Z INFO af_extract] Frequencies: 24291 unique Pfam, 25993 unique GO from 205620298 proteins |
| L2-544 | `watcher.log` | 2072 | 25993 | unique GO among pLDDT-passing proteins | deterministic | [2026-02-09T04:39:56.574Z INFO af_extract] Frequencies: 24291 unique Pfam, 25993 unique GO from 205620298 proteins |
| L2-545 | `watcher.log` | 2073 | 6 | pLDDT bins (items) | method-parameter | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items |
| L2-546 | `watcher.log` | 2073 | 500 | top Pfam kept (--top-pfam) | method-parameter | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items |
| L2-547 | `watcher.log` | 2073 | 500 | top GO kept (--top-go) | method-parameter | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items |
| L2-548 | `watcher.log` | 2073 | 1006 | total items (vocabulary) | deterministic | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items |
| L2-549 | `watcher.log` | 2074 | 1006 | items in item mapping | deterministic | [2026-02-09T04:39:56.577Z INFO af_extract::transaction] Saved item mapping: 1006 items to /root/alphafold-data/full_214m/item_mapping_214m.parquet |
| L2-550 | `watcher.log` | 2075–2279 | 1M → 205M | transactions written (205 periodic lines, step [1] M, 2026-02-09T04:39:57.791Z → 2026-02-09T04:43:35.342Z) | deterministic | [consolidated periodic] first: [2026-02-09T04:39:57.791Z INFO af_extract::transaction] Written 1M transactions... … last: [2026-02-09T04:43:35.342Z INFO af_extract::transaction] Written 205M transactions... |
| L2-551 | `watcher.log` | 2280 | 205620298 | transactions written (parquet rows) | deterministic | [2026-02-09T04:43:35.994Z INFO af_extract::transaction] Written 205620298 transactions to /root/alphafold-data/full_214m/transactions_214m.parquet |
| L2-552 | `watcher.log` | 2281 | 2026-02-09T04:43:35.999Z | timestamp (af-extract Results banner = end of Step 1 binary) | external-fact | [2026-02-09T04:43:35.999Z INFO af_extract] === Results === |
| L2-553 | `watcher.log` | 2282 | 205620298 | transactions (af-extract summary) | deterministic | [2026-02-09T04:43:35.999Z INFO af_extract] Transactions: 205620298 |
| L2-554 | `watcher.log` | 2283 | 1006 | total items (af-extract summary) | deterministic | [2026-02-09T04:43:35.999Z INFO af_extract] Total items: 1006 |
| L2-555 | `watcher.log` | 2286 | 3777.0 | s (af-extract build-from-metadata self-timed) | hardware-dependent | [2026-02-09T04:43:35.999Z INFO af_extract] Time: 3777.0s (54440 proteins/sec) |
| L2-556 | `watcher.log` | 2286 | 54440 | proteins/sec (af-extract throughput) | hardware-dependent | [2026-02-09T04:43:35.999Z INFO af_extract] Time: 3777.0s (54440 proteins/sec) |
| L2-557 | `watcher.log` | 2288 | 65m31.040s | real time (bash `time` of Step 1) | hardware-dependent | real 65m31.040s |
| L2-558 | `watcher.log` | 2289 | 62m38.750s | user time (bash `time` of Step 1) | hardware-dependent | user 62m38.750s |
| L2-559 | `watcher.log` | 2290 | 2m52.144s | sys time (bash `time` of Step 1) | hardware-dependent | sys 2m52.144s |
| L2-560 | `watcher.log` | 2295 | 205,620,298 | transactions (Step 2 data stats) | deterministic | Total transactions: 205,620,298 |
| L2-561 | `watcher.log` | 2296 | 1006 | total items (Step 2 data stats) | deterministic | Total items: 1006 |
| L2-562 | `watcher.log` | 2298 | 2.2 | items per transaction (mean) | deterministic | Mean: 2.2 |
| L2-563 | `watcher.log` | 2299 | 46 | items per transaction (max) | deterministic | Max: 46 |
| L2-564 | `watcher.log` | 2300 | 76,890,945 | transactions with >1 item ('37.4%') | deterministic | With >1 item: 76,890,945 (37.4%) |
| L2-565 | `watcher.log` | 2303 | 76,890,945 | transactions mined (Step 3) | deterministic | Mining 76,890,945 annotated proteins... |
| L2-566 | `watcher.log` | 2306 | 113.9 | s (Step 3 mining) | hardware-dependent | Time: 113.9s |
| L2-567 | `watcher.log` | 2307 | 5,305 | itemsets (total, Step 3) | deterministic | Itemsets: 5,305 |
| L2-568 | `watcher.log` | 2308 | 667 | itemsets K=1 | deterministic | [K-dist] K=1: 667 |
| L2-569 | `watcher.log` | 2309 | 1,504 | itemsets K=2 | deterministic | [K-dist] K=2: 1,504 |
| L2-570 | `watcher.log` | 2310 | 1,378 | itemsets K=3 | deterministic | [K-dist] K=3: 1,378 |
| L2-571 | `watcher.log` | 2311 | 884 | itemsets K=4 | deterministic | [K-dist] K=4: 884 |
| L2-572 | `watcher.log` | 2312 | 514 | itemsets K=5 | deterministic | [K-dist] K=5: 514 |
| L2-573 | `watcher.log` | 2313 | 247 | itemsets K=6 | deterministic | [K-dist] K=6: 247 |
| L2-574 | `watcher.log` | 2314 | 89 | itemsets K=7 | deterministic | [K-dist] K=7: 89 |
| L2-575 | `watcher.log` | 2315 | 20 | itemsets K=8 | deterministic | [K-dist] K=8: 20 |
| L2-576 | `watcher.log` | 2316 | 2 | itemsets K=9 | deterministic | [K-dist] K=9: 2 |
| L2-577 | `watcher.log` | 2320 | 53,447 | association rules generated | deterministic | Generated 53,447 rules in 0.2s |
| L2-578 | `watcher.log` | 2320 | 0.2 | s (rule generation) | hardware-dependent | Generated 53,447 rules in 0.2s |
| L2-579 | `watcher.log` | 2323 | 0.998 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.998 lift= 969x PF01554 => GO:0015297 + GO:0042910 |
| L2-580 | `watcher.log` | 2323 | 969 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.998 lift= 969x PF01554 => GO:0015297 + GO:0042910 |
| L2-581 | `watcher.log` | 2324 | 0.999 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.999 lift= 969x GO:0015297 + GO:0042910 => PF01554 |
| L2-582 | `watcher.log` | 2324 | 969 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.999 lift= 969x GO:0015297 + GO:0042910 => PF01554 |
| L2-583 | `watcher.log` | 2325 | 1.000 | confidence | deterministic | [TOP 30 BY LIFT] conf=1.000 lift= 921x PF00116 => GO:0004129 + GO:0005507 |
| L2-584 | `watcher.log` | 2325 | 921 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=1.000 lift= 921x PF00116 => GO:0004129 + GO:0005507 |
| L2-585 | `watcher.log` | 2326 | 0.945 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 921x GO:0004129 + GO:0005507 => PF00116 |
| L2-586 | `watcher.log` | 2326 | 921 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 921x GO:0004129 + GO:0005507 => PF00116 |
| L2-587 | `watcher.log` | 2327 | 0.963 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.963 lift= 845x plddt_mean_med + GO:0043952 => GO:0005886 + GO:0065002 |
| L2-588 | `watcher.log` | 2327 | 845 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.963 lift= 845x plddt_mean_med + GO:0043952 => GO:0005886 + GO:0065002 |
| L2-589 | `watcher.log` | 2328 | 0.938 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.938 lift= 845x GO:0005886 + GO:0065002 => plddt_mean_med + GO:0043952 |
| L2-590 | `watcher.log` | 2328 | 845 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.938 lift= 845x GO:0005886 + GO:0065002 => plddt_mean_med + GO:0043952 |
| L2-591 | `watcher.log` | 2329 | 0.940 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.940 lift= 835x GO:0043952 => plddt_mean_med + GO:0005886 + GO:0065002 |
| L2-592 | `watcher.log` | 2329 | 835 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.940 lift= 835x GO:0043952 => plddt_mean_med + GO:0005886 + GO:0065002 |
| L2-593 | `watcher.log` | 2330 | 0.949 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.949 lift= 835x plddt_mean_med + GO:0005886 + GO:0065002 => GO:0043952 |
| L2-594 | `watcher.log` | 2330 | 835 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.949 lift= 835x plddt_mean_med + GO:0005886 + GO:0065002 => GO:0043952 |
| L2-595 | `watcher.log` | 2331 | 0.951 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.951 lift= 834x GO:0043952 => GO:0005886 + GO:0065002 |
| L2-596 | `watcher.log` | 2331 | 834 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.951 lift= 834x GO:0043952 => GO:0005886 + GO:0065002 |
| L2-597 | `watcher.log` | 2332 | 0.949 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.949 lift= 834x GO:0005886 + GO:0065002 => GO:0043952 |
| L2-598 | `watcher.log` | 2332 | 834 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.949 lift= 834x GO:0005886 + GO:0065002 => GO:0043952 |
| L2-599 | `watcher.log` | 2333 | 0.856 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.856 lift= 827x PF00849 => GO:0003723 + GO:0000455 |
| L2-600 | `watcher.log` | 2333 | 827 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.856 lift= 827x PF00849 => GO:0003723 + GO:0000455 |
| L2-601 | `watcher.log` | 2334 | 0.992 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.992 lift= 827x GO:0003723 + GO:0000455 => PF00849 |
| L2-602 | `watcher.log` | 2334 | 827 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.992 lift= 827x GO:0003723 + GO:0000455 => PF00849 |
| L2-603 | `watcher.log` | 2335 | 0.902 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.902 lift= 816x GO:0003677 + GO:0000786 => GO:0046982 + GO:0030527 |
| L2-604 | `watcher.log` | 2335 | 816 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.902 lift= 816x GO:0003677 + GO:0000786 => GO:0046982 + GO:0030527 |
| L2-605 | `watcher.log` | 2336 | 0.986 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.986 lift= 816x GO:0046982 + GO:0030527 => GO:0003677 + GO:0000786 |
| L2-606 | `watcher.log` | 2336 | 816 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.986 lift= 816x GO:0046982 + GO:0030527 => GO:0003677 + GO:0000786 |
| L2-607 | `watcher.log` | 2337 | 0.945 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 809x PF13853 => GO:0005886 + GO:0004930 + GO:0004984 |
| L2-608 | `watcher.log` | 2337 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 809x PF13853 => GO:0005886 + GO:0004930 + GO:0004984 |
| L2-609 | `watcher.log` | 2338 | 0.983 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.983 lift= 809x GO:0005886 + GO:0004930 + GO:0004984 => PF13853 |
| L2-610 | `watcher.log` | 2338 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.983 lift= 809x GO:0005886 + GO:0004930 + GO:0004984 => PF13853 |
| L2-611 | `watcher.log` | 2339 | 0.940 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.940 lift= 809x PF13853 => plddt_mean_med + GO:0005886 + GO:0004930 + GO:0004984 |
| L2-612 | `watcher.log` | 2339 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.940 lift= 809x PF13853 => plddt_mean_med + GO:0005886 + GO:0004930 + GO:0004984 |
| L2-613 | `watcher.log` | 2340 | 0.983 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.983 lift= 809x plddt_mean_med + GO:0005886 + GO:0004930 + GO:0004984 => PF13853 |
| L2-614 | `watcher.log` | 2340 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.983 lift= 809x plddt_mean_med + GO:0005886 + GO:0004930 + GO:0004984 => PF13853 |
| L2-615 | `watcher.log` | 2341 | 0.945 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 809x plddt_mean_med + PF13853 => GO:0005886 + GO:0004930 + GO:0004984 |
| L2-616 | `watcher.log` | 2341 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.945 lift= 809x plddt_mean_med + PF13853 => GO:0005886 + GO:0004930 + GO:0004984 |
| L2-617 | `watcher.log` | 2342 | 0.978 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.978 lift= 809x GO:0005886 + GO:0004930 + GO:0004984 => plddt_mean_med + PF13853 |
| L2-618 | `watcher.log` | 2342 | 809 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.978 lift= 809x GO:0005886 + GO:0004930 + GO:0004984 => plddt_mean_med + PF13853 |
| L2-619 | `watcher.log` | 2343 | 0.960 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.960 lift= 802x GO:0000455 => PF00849 + GO:0003723 |
| L2-620 | `watcher.log` | 2343 | 802 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.960 lift= 802x GO:0000455 => PF00849 + GO:0003723 |
| L2-621 | `watcher.log` | 2344 | 0.857 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.857 lift= 802x PF00849 + GO:0003723 => GO:0000455 |
| L2-622 | `watcher.log` | 2344 | 802 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.857 lift= 802x PF00849 + GO:0003723 => GO:0000455 |
| L2-623 | `watcher.log` | 2345 | 0.856 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.856 lift= 801x PF00849 => GO:0000455 |
| L2-624 | `watcher.log` | 2345 | 801 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.856 lift= 801x PF00849 => GO:0000455 |
| L2-625 | `watcher.log` | 2346 | 0.960 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.960 lift= 801x GO:0000455 => PF00849 |
| L2-626 | `watcher.log` | 2346 | 801 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.960 lift= 801x GO:0000455 => PF00849 |
| L2-627 | `watcher.log` | 2347 | 0.967 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0015833 => PF08352 + GO:0005524 + GO:0016887 |
| L2-628 | `watcher.log` | 2347 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0015833 => PF08352 + GO:0005524 + GO:0016887 |
| L2-629 | `watcher.log` | 2348 | 0.967 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0005524 + GO:0015833 => PF08352 + GO:0016887 |
| L2-630 | `watcher.log` | 2348 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0005524 + GO:0015833 => PF08352 + GO:0016887 |
| L2-631 | `watcher.log` | 2349 | 0.967 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0015833 => PF08352 + GO:0016887 |
| L2-632 | `watcher.log` | 2349 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.967 lift= 792x PF00005 + GO:0015833 => PF08352 + GO:0016887 |
| L2-633 | `watcher.log` | 2350 | 0.990 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0016887 => PF00005 + GO:0005524 + GO:0015833 |
| L2-634 | `watcher.log` | 2350 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0016887 => PF00005 + GO:0005524 + GO:0015833 |
| L2-635 | `watcher.log` | 2351 | 0.990 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0005524 + GO:0016887 => PF00005 + GO:0015833 |
| L2-636 | `watcher.log` | 2351 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0005524 + GO:0016887 => PF00005 + GO:0015833 |
| L2-637 | `watcher.log` | 2352 | 0.990 | confidence | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0016887 => PF00005 + GO:0015833 |
| L2-638 | `watcher.log` | 2352 | 792 | lift (x) | deterministic | [TOP 30 BY LIFT] conf=0.990 lift= 792x PF08352 + GO:0016887 => PF00005 + GO:0015833 |
| L2-639 | `watcher.log` | 2355 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 921x PF00116 => GO:0004129 + GO:0005507 |
| L2-640 | `watcher.log` | 2355 | 921 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 921x PF00116 => GO:0004129 + GO:0005507 |
| L2-641 | `watcher.log` | 2356 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 776x PF00905 + GO:0071555 => GO:0008658 |
| L2-642 | `watcher.log` | 2356 | 776 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 776x PF00905 + GO:0071555 => GO:0008658 |
| L2-643 | `watcher.log` | 2357 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 776x PF00905 => GO:0008658 |
| L2-644 | `watcher.log` | 2357 | 776 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 776x PF00905 => GO:0008658 |
| L2-645 | `watcher.log` | 2358 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 735x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0006397 + GO:0008033 |
| L2-646 | `watcher.log` | 2358 | 735 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 735x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0006397 + GO:0008033 |
| L2-647 | `watcher.log` | 2359 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 735x PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0006397 + GO:0008033 |
| L2-648 | `watcher.log` | 2359 | 735 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 735x PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0006397 + GO:0008033 |
| L2-649 | `watcher.log` | 2360 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x plddt_mean_med + PF01824 + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-650 | `watcher.log` | 2360 | 734 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x plddt_mean_med + PF01824 + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-651 | `watcher.log` | 2361 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-652 | `watcher.log` | 2361 | 734 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-653 | `watcher.log` | 2362 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x plddt_mean_med + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-654 | `watcher.log` | 2362 | 734 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x plddt_mean_med + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-655 | `watcher.log` | 2363 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x PF01824 + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-656 | `watcher.log` | 2363 | 734 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 734x PF01824 + PF01348 + GO:0009507 => GO:0003723 + GO:0006397 + GO:0008033 |
| L2-657 | `watcher.log` | 2364 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0003723 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-658 | `watcher.log` | 2364 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0003723 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-659 | `watcher.log` | 2365 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-660 | `watcher.log` | 2365 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-661 | `watcher.log` | 2366 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-662 | `watcher.log` | 2366 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-663 | `watcher.log` | 2367 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01348 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-664 | `watcher.log` | 2367 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01348 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-665 | `watcher.log` | 2368 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + GO:0003723 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-666 | `watcher.log` | 2368 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + GO:0003723 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-667 | `watcher.log` | 2369 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01348 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-668 | `watcher.log` | 2369 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01348 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-669 | `watcher.log` | 2370 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-670 | `watcher.log` | 2370 | 681 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 681x PF01824 + GO:0003723 + GO:0009507 => plddt_mean_med + GO:0006397 + GO:0008033 |
| L2-671 | `watcher.log` | 2371 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01824 + PF01348 + GO:0009507 + GO:0006397 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-672 | `watcher.log` | 2371 | 663 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01824 + PF01348 + GO:0009507 + GO:0006397 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-673 | `watcher.log` | 2372 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01348 + GO:0009507 + GO:0006397 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-674 | `watcher.log` | 2372 | 663 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01348 + GO:0009507 + GO:0006397 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-675 | `watcher.log` | 2373 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-676 | `watcher.log` | 2373 | 663 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-677 | `watcher.log` | 2374 | 1.000 | confidence | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-678 | `watcher.log` | 2374 | 663 | lift (x) | deterministic | [TOP 20 BY CONFIDENCE] conf=1.000 lift= 663x PF01824 + PF01348 + GO:0009507 => plddt_mean_med + GO:0003723 + GO:0008033 |
| L2-679 | `watcher.log` | 2377 | 53,447 | association rules (summary) | deterministic | Total rules: 53,447 |
| L2-680 | `watcher.log` | 2378 | 12,776 | rules with confidence >= 99% | deterministic | Confidence >= 99%: 12,776 |
| L2-681 | `watcher.log` | 2379 | 31,775 | rules with confidence >= 90% | deterministic | Confidence >= 90%: 31,775 |
| L2-682 | `watcher.log` | 2380 | 51,126 | rules with lift >= 5.0 | deterministic | Lift >= 5.0: 51,126 |
| L2-683 | `watcher.log` | 2381 | 40,428 | rules with lift >= 100 | deterministic | Lift >= 100: 40,428 |
| L2-684 | `watcher.log` | 2382 | 45,286 | cross-domain rules (Pfam<=>GO) | deterministic | Cross-domain (Pfam<=>GO): 45,286 |
| L2-685 | `watcher.log` | 2385 | Mon Feb  9 04:48:08 UTC 2026 | timestamp (pipeline end banner) | external-fact | === PIPELINE COMPLETE — Mon Feb 9 04:48:08 UTC 2026 === |
| L2-686 | `yolo.log` | 2 | 3 | min_count (proteins) | method-parameter | YOLO MODE — min_count = 3 |
| L2-687 | `yolo.log` | 3 | 23 | target K | method-parameter | The DEFINITIVE answer to K=23 |
| L2-688 | `yolo.log` | 5 | 1.458999928110210e-08 | min_support (fraction) | method-parameter | min_support = 1.458999928110210e-08 |
| L2-689 | `yolo.log` | 6 | 205620298 | transactions (denominator used in verify) | deterministic | verify: ceil(1.4589999281102103e-08 * 205620298) = 3 |
| L2-690 | `yolo.log` | 6 | 3 | min_count (verified ceil) | method-parameter | verify: ceil(1.4589999281102103e-08 * 205620298) = 3 |

## SECTION B — WHAT WAS ACTUALLY RUN, per log file

Common facts used below (cited once): the miner's threshold rule in the tree is `min_count = ceil(min_support × n_transactions)` (`src/et_miner/core/result.py:15-17`); the `Direct CSR path:` log lines match `src/et_miner/core/matrix.py:502-548` verbatim; the streaming (SON) entry point is `src/et_miner/streaming/son.py:79` (`apriori_streaming`, defaults chunk_size=40,000,000, local_support_factor=0.9, `gpu_resident` flag) and `streaming/multi_gpu.py:113` (n_gpus=8, chunk_size=10,000,000). None of the banner strings of the mining logs (`EXTREME SUPPORT`, `ULTRA LOW SUPPORT`, `DIRECT GPU MINING`, `BEYOND MADMAN`, `HOLD MY BEER MODE`, `YOLO MODE`, `MADMAN SUPPORT`, `Loading + mining...`) nor the pipeline shell banners (`FULL 214M AlphaFold Pipeline`, `Step 1..4`, `TOP 30 BY LIFT`) occur anywhere in the repository — the driver scripts that produced these logs are not in the tree. Nominal denominators: N1 = 76,890,945 (proteins with >1 item), N0 = 205,620,298 (all pLDDT ≥ 50 proteins in the parquet).

### B.1 `beyond_mining.log` (57 lines) — Direct CSR→GPU, nominal 0.00002 %
- Method: Direct CSR path (L8-10), i.e. `apriori(use_gpu=True)` on the multi-item subset; n_transactions = 76,890,945 (L6, L8).
- Parameters: nominal support 0.00002 % (L2), "Min proteins: ~15" (L3), script-computed "Min support count: 15 proteins" (L7), miner-applied `min_count=16` (L8) [2e-7 × N1 = 15.378; ceil = 16, floor = 15]; Max K 25 (L4); 1002 frequent items (L9); nnz 316,421,093 (L10). Input path not printed.
- Timing: start 2026-02-09 06:05:03,735 (L8); "COMPLETE in 281.0s (4.7 min)" (L32) → end ≈ 06:09:45; per-K ms sum = 141.2 s.
- Result: 14,558,875 itemsets (L33), K_max = 20 (2 itemsets at K=20; L30/L55), per-K K=1…20 (L11-30) identical to K-distribution (L36-55). Saved `/root/alphafold-data/full_214m/itemsets_214m_beyond.parquet`, 49,989,864 bytes (L57).
- Status: completed (natural termination at K=20 < max K 25). No GPU/host/version strings.

### B.2 `direct_mining.log` (64 lines) — Direct CSR→GPU, 1e-06
- Method: "DIRECT GPU MINING — NO STREAMING, NO DENSE MATRIX" (L2); Direct CSR path (L14-16) on the multi-item subset.
- Parameters: min_support 1e-06 (0.0001 %) (L3); Max K 20 (L4); parquet holds 205,620,298 transactions of which 76,890,945 with >1 item (L8-9); script-computed "Min support count: 76 proteins" (L10) vs miner `min_count=77` (L14) [1e-6 × N1 = 76.89]; 1002 frequent items (L15); nnz 316,421,093 (L16).
- Timing: load 0.5 s (L11); start 2026-02-09 05:46:42,182 (L14); mining 119.3 s (L37), total 120.6 s (L38) → end ≈ 05:48:43; per-K ms sum = 71.1 s.
- Result: 2,841,280 itemsets (L39), K_max = 19 (1 itemset; L35/L60); per-K (L17-35) == K-distribution (L42-60). Saved `/root/alphafold-data/full_214m/itemsets_214m_direct.parquet` (L62), 12,177,502 bytes (L63).
- Status: completed (terminated at K=19 < max K 20).

### B.3 `extreme_mining.log` (73 lines) — streaming SON, 0.001 %
- Method: "GPU-RESIDENT STREAMING APRIORI — 0.001% SUPPORT" (L7), "MAX LENGTH 20" (L8) — SON streaming path; chunk size, local-support factor, n_gpus not logged.
- Parameters: 0.001 % "on 214M TrEMBL" (L1); 205,620,298 total / 76,890,945 with >1 item (L3-4); "Min support = 0.001% = 768 proteins" (L5) [1e-5 × N1 = 768.91; floor = 768, ceil = 769]; the min_count actually applied by the streaming code is not logged.
- Timing: "Time: 1085.6s" (L12); no timestamps; no per-K timing.
- Result: 22,846 itemsets (L13), K=1…13 (L14-26), K_max = 13. Saved `/root/alphafold-data/full_214m/itemsets_214m_extreme.parquet` (L27). Deepest patterns listed for K=13 (2), K=12 (11, 5 shown), K=11 (48, 5 shown), K=10 (160, 5 shown) with support fraction and protein count (L31-71); the K=13 itemsets are `plddt_mean_med + PF00271 + PF00270 + 10 GO terms` at 10,916 / 10,913 proteins (L32-35).
- Status: completed ("EXTREME MINING COMPLETE", L73).

### B.4 `godmode_mining.log` (62 lines) — Direct CSR→GPU, min_count = 8
- Method: Direct CSR path (L2-4): 76,890,945 transactions, `min_count=8` (L2) [nominal 1e-7 × N1 = 7.69 → ceil 8; 8/N1 = 1.04e-7]; 1002 frequent items (L3); nnz 316,421,093 (L4). No nominal support %, max K, or input path printed ("Loading + mining...", L1).
- Timing: start 2026-02-09 06:42:58,031 (L2); "26,849,505 itemsets in 440.5s" (L28) → end ≈ 06:50:19; per-K ms sum = 204.8 s.
- Result: 26,849,505 itemsets, K_max = 22 (1 itemset at K=22; L26/L31); per-K K=1…22 (L5-26) without candidate counts; no K-distribution block. Saved 85,131,478 bytes (L29; path not printed). Decoded top-3 itemsets for K=22 (8 proteins; 1 pLDDT + 2 Pfam + 19 GO, L31-35), K=21 (13 / 8 / 8 proteins, L37-49), K=20 (57 / 40 / 15 proteins, L51-62).
- Status: completed (natural termination at K=22).

### B.5 `holdmybeer.log` (40 lines) — nominal "4 proteins"
- Method: not stated; no `Direct CSR path` lines. Input `/root/alphafold-data/full_214m/transactions_214m.parquet` (L7; the full 205,620,298-row file).
- Parameters: "support = 4 proteins / 76.9M = 0.000005%" (L3); "Mining at support=0.000000052 (~4 proteins)" (L8) [5.2e-8 × N1 = 3.998 → ceil 4; 5.2e-8 × N0 = 10.69 → ceil 11 — the log does not say which n the miner used]; target K=23+ (L4); max K not printed.
- Timing: "18,935,899 itemsets in 573.1s" (L10); no timestamps.
- Result: 18,935,899 itemsets, K=1…21 (L14-34), "MAX K = 21" (L36); saved 63,377,870 bytes (L11; path not printed). "Loading item mapping for decode..." (L38) but no decoded patterns are printed.
- Status: completed (L40 "HOLD MY BEER MODE COMPLETE"); target K=23 not reached.

### B.6 `holdmybeer_real.log` (70 lines) — min_count = 4 "for real"
- Method: not stated; no `Direct CSR path` lines; no input path.
- Parameters: `min_count = 4` (L3); `min_support = 1.945333237480280e-08` (L5) = 4/205,620,298 exactly; self-check "ceil(1.9453332374802804e-08 * 205620298) = 4" (L6) → denominator N0 [× N1 would give 1.496 → ceil 2].
- Timing: "48,007,493 itemsets in 1228.5s" (L8); no timestamps.
- Result: 48,007,493 itemsets, K=1…22 (L12-33), "MAX K = 22" (L35); saved 149,590,799 bytes (L9; path not printed). Decoded top-3 for K=22 (8 proteins; same 19 GO + PF00271/PF00270 + plddt_mean_med as godmode, L37-41), K=21 (13 / 8 / 8, L43-55), K=20 (57 / 40 / 15, L57-68).
- Status: completed ("DONE", L70); K=23 not reached.

### B.7 `madman_mining.log` (8 lines) — streaming SON, 0.0001 % (aborted)
- Method: "GPU-RESIDENT STREAMING APRIORI — 0.0001% SUPPORT" (L6), "MAX LENGTH 20 — ABSOLUTE MADMAN MODE" (L7).
- Parameters: 0.0001 % on 214M TrEMBL (L1); 76,890,945 with >1 item (L3; no "Total transactions" line); "Min support = 0.0001% = 76 proteins" (L4) [floor of 76.89].
- Timing/result: none — the file ends at a blank line 8; no error text, no timestamps.
- Status: did not complete (killed, crashed silently, or abandoned); the same threshold was subsequently mined by the direct path (`direct_mining.log`, 05:46 UTC).

### B.8 `pipeline_214m.log` (2,377 lines) — Base run: extraction + 0.1 % SON mining + rules
- Banner: "FULL 214M AlphaFold Pipeline — Mon Feb 9 03:40:36 UTC 2026" (L2-3). End: "PIPELINE COMPLETE — Mon Feb 9 04:48:08 UTC 2026" (L2376) → total wall 4,052 s (67 min 32 s). Completed.
- Step 1 (L6-2281) `af-extract build-from-metadata`, wrapped in bash `time` (real 65m31.040s, user 62m38.750s, sys 2m52.144s; L2279-2281). Inputs: TrEMBL "150G" (L7) = `/root/uniprot_trembl.dat.gz` (L10); pLDDT "214683830 rows" (L8) = `/root/alphafold-data/plddt_metadata.csv` with columns `accession` (index 0) / `mean_plddt` (index 1) (L2037-2039). Inferred command line (flags per `af-extract/src/main.rs:130-172`; values from the log): `time af-extract build-from-metadata --annotations /root/uniprot_trembl.dat.gz --plddt-csv /root/alphafold-data/plddt_metadata.csv --top-pfam 500 --top-go 500 --output /root/alphafold-data/full_214m/transactions_214m.parquet --item-mapping /root/alphafold-data/full_214m/item_mapping_214m.parquet` (`--min-plddt` at its default 50.0; L2061 "passed pLDDT filter (>= 50)"). UniProt release / download URL: not recorded anywhere in the log.
  - DAT parse 03:40:38.963Z → 04:36:19.489Z (3,340.5 s): 2,025 progress lines 100K → 202,500K (L11-2035); "Loaded 202556314 annotations (25475 Pfam domains, 26536 GO terms)" (L2036).
  - pLDDT CSV read 04:36:19.489Z → 04:36:59.926Z (40.4 s): 21 progress lines (L2040-2060); "Read 214683829 rows: 205620298 passed pLDDT filter (>= 50), 9063531 skipped" (L2061).
  - Frequency count 04:36:59.926Z → 04:39:56.574Z (176.6 s): "24291 unique Pfam, 25993 unique GO from 205620298 proteins" (L2063).
  - Encoding + write 04:39:56.577Z → 04:43:35.994Z (219.4 s): "6 pLDDT + 500 Pfam + 500 GO = 1006 total items" (L2064); item mapping saved (L2065); 205 progress lines 1M → 205M (L2066-2270); "Written 205620298 transactions to /root/alphafold-data/full_214m/transactions_214m.parquet" (L2271); Results block: 205620298 transactions, 1006 items, "Time: 3777.0s (54440 proteins/sec)" (L2272-2277) [af-extract wall 03:40:38.963 → 04:43:35.999 = 3,777.0 s; `time real` 3,931.0 s → 154 s of process teardown not in the self-timing].
- Step 2 (L2285-2291) data stats: 205,620,298 transactions, 1006 items, mean 2.2 items, max 46, with >1 item 76,890,945 (37.4 %).
- Step 3 (L2293-2308) "GPU-resident streaming Apriori (0.1% support)" on 76,890,945 annotated proteins (SON path; chunk size / n_gpus / max_length / min_count not logged; 1e-3 × N1 = 76,890.9 → ceil 76,891): Time 113.9 s; 5,305 itemsets; K=1…9 (667 / 1,504 / 1,378 / 884 / 514 / 247 / 89 / 20 / 2); saved `/root/alphafold-data/full_214m/itemsets_214m.parquet`.
- Step 4 (L2310-2373) association rules, min_confidence=0.5: 53,447 rules in 0.2 s; top-30 by lift (max lift 969×, L2314-2343) and top-20 by confidence (all conf=1.000, L2346-2365); summary: 53,447 rules, conf ≥ 99 %: 12,776, conf ≥ 90 %: 31,775, lift ≥ 5.0: 51,126, lift ≥ 100: 40,428, cross-domain Pfam⇔GO: 45,286.
- Steps 2-4 took ≈ 121 s of wall time (04:46:07 → 04:48:08). No GPU/host/version strings anywhere in the file. The in-repo `pipeline/pipeline_214m.py` (L452-463) invokes af-extract with extra `--top-interpro/--top-ec/--top-taxonomy/--min-plddt` flags and writes `alphafold_transactions_214m.parquet` (different filename), so it is not the driver that produced this log.

### B.9 `ultra_mining.log` (45 lines) — streaming SON, 0.01 %
- Method: "GPU-RESIDENT STREAMING APRIORI — 0.01% SUPPORT" (L7); "RELEASING THE H100s..." (L8; only hardware string in any of the 11 logs — plural). Max length, chunk size, n_gpus not logged.
- Parameters: 0.01 % on 214M TrEMBL (L1); 205,620,298 / 76,890,945 (L3-4); "Min support = 0.01% = 7,689 proteins" (L5) [1e-4 × N1 = 7,689.09; floor 7,689, ceil 7,690].
- Timing: "Time: 257.6s" (L12); no timestamps.
- Result: 51,124 itemsets (L13), K=1…13 (L14-26), K_max = 13; saved `/root/alphafold-data/full_214m/itemsets_214m_ultra.parquet` (L27). "DEEPEST PATTERNS" prints only the top-3 of K=2 and K=1 (L31-45): K=2 `plddt_mean_high + GO:0046872` 5,453,948 proteins (0.07093) …; K=1 `GO:0005737` 8,257,363 (0.10739), `GO:0005829` 6,370,846, `GO:0003677` 5,270,392.
- Status: completed (ends after the K=1 top-3; no explicit COMPLETE line).

### B.10 `watcher.log` (2,386 lines) — download watcher + verbatim copy of the pipeline log
- L1-2: "TrEMBL Download Watcher / Monitoring aria2c download...". L4: a single CR-separated aria2c progress line (45 snapshots, 25 distinct): first `[145G] 62GiB/149GiB(42%) CN:16 DL:50MiB ETA:29m13s`, last `[150G] 145GiB/149GiB(97%) CN:16 DL:55MiB ETA:1m18s`; DL 26-100 MiB/s; 16 connections; the bracket prefix (disk usage) rises 145G → 150G; no 100 % snapshot. L6: "aria2c FINISHED! File size: 150G". L7: "Starting full pipeline at Mon Feb 9 03:40:36 UTC 2026".
- L10-2386 are byte-identical to `pipeline_214m.log` L1-2377 (`diff` verified; offset +9), so B.8 applies with line numbers +9.
- Download URL, UniProt release, download start time and aria2c exit status are not recorded. Status: watcher completed its hand-off; pipeline completed as in B.8.

### B.11 `yolo.log` (6 lines) — min_count = 3 (no results)
- "YOLO MODE — min_count = 3 / The DEFINITIVE answer to K=23" (L2-3); `min_support = 1.458999928110210e-08` (L5) = 3/205,620,298 exactly; self-check "ceil(1.4589999281102103e-08 * 205620298) = 3" (L6) [× N1 would give 1.12 → ceil 2].
- No timing, no results, no error text. Status: did not complete (killed / crashed silently / abandoned).

### B.12 Run chronology reconstructable from the logs (all 2026-02-09 UTC)
aria2c download (ETA 29 min at first snapshot) → 03:40:36 pipeline start (watcher L7, pipeline L3) → 04:48:08 pipeline complete → [ultra, extreme, madman: no timestamps; their banners reference the streaming path used in Step 3] → 05:46:42 direct (Blitz) → 06:05:03 beyond → 06:42:58 godmode → [holdmybeer, holdmybeer_real, yolo: no timestamps; their "hold my beer / the real one / yolo" naming and min_count 4→4→3 imply they followed godmode]. The two surviving JSONs (`experiment_direct_vs_son_20260219_050326.json`, `experiment_null_model_20260219_061046.json`) are dated 2026-02-19, ten days later.

## SECTION C — CROSS-LOG OBSERVATIONS (no verdicts)

1. **Script-computed vs miner-applied min_count differ by one in every direct-path log**: `beyond` "Min support count: 15" (L7) vs `min_count=16` (L8); `direct` "76" (L10) vs `min_count=77` (L14). The miner uses `ceil(min_support × n)` (`core/result.py:15-17`); the script headers used the floor. For the SON logs only the floor value is printed: `extreme` 768 (L5; ceil = 769), `ultra` 7,689 (L5; ceil = 7,690), `madman` 76 (L4; ceil = 77); the min_count actually applied inside the streaming code is not logged.
2. **768 vs 769 at 0.001 %**: `extreme_mining.log` L5 says 768; `experiment_direct_vs_son_*.json` `parameters.min_count` = 768 with `min_support` = 1e-05; `experiment_null_model_*.json` `min_count` = 769 with `min_support` = 1.0001177641918693e-05 (= 769/76,890,945). ceil(1e-5 × 76,890,945) = 769.
3. **The JSON's direct-GPU run at 0.001 % (475,865 itemsets, 50.72 s, K=14) has no counterpart among the 11 logs.** The JSON's `son_reference` (22,846 itemsets, 1,085.6 s, K=13) equals `extreme_mining.log` L12-13/L26, and its `blitz_reference` (2,841,280, 119.3 s, K=19) equals `direct_mining.log` L37/L39/L35. `null_model.real_distribution` (K=1…14) is the JSON's own direct@769 run, not any log.
4. **Paper threshold names ↔ log files** (paper `paper/et_miner_proteome.tex` L224-230; mapping also asserted in `paper/PAPER_V2_REVIEW.md` L72):
   - Base (0.1 %, 76,891, 5,305 itemsets, K 9, 1.9 min, Streaming SON) ↔ `pipeline_214m.log` Step 3 (113.9 s = 1.90 min; 5,305; K=9). The log never prints a min_count; 76,891 = ceil(1e-3 × 76,890,945).
   - Super (0.01 %, 7,689, 51,124, K 13, 4.3 min, SON) ↔ `ultra_mining.log` (257.6 s = 4.29 min; 51,124; K=13; "7,689 proteins").
   - Power (0.001 %, 768, 22,846, K 13, 18.1 min, SON) ↔ `extreme_mining.log` (1,085.6 s = 18.09 min; 22,846; K=13; "768 proteins").
   - Blitz (0.0001 %, 77, 2,841,280, K 19, 2.0 min, Direct) ↔ `direct_mining.log` (119.3 s = 1.99 min; 2,841,280; K=19; min_count=77).
   - Ultra (0.00002 %, 16, 14,558,875, K 20, 4.7 min, Direct) ↔ `beyond_mining.log` (281.0 s = 4.68 min; 14,558,875; K=20; min_count=16).
   - Opus (0.00001 %, 8, 26,849,505, K 22, 7.3 min, Direct) ↔ `godmode_mining.log` (440.5 s = 7.34 min; 26,849,505; K=22; min_count=8).
   - Name collision: the file named `ultra_mining.log` is the paper's **Super** row, while the paper's **Ultra** row is `beyond_mining.log`. `madman_mining.log` is an aborted SON attempt at the Blitz threshold (no paper row). `holdmybeer.log` (nominal 4, effective unknown), `holdmybeer_real.log` (min_count 4) and `yolo.log` (min_count 3) are below the Opus threshold and have no paper row.
   - All six paper (itemsets, K_max, minutes) triples match the corresponding log values; the paper's "min proteins" column mixes ceil (76,891; 77; 16; 8) and floor (7,689; 768) conventions relative to 76,890,945.
5. **holdmybeer effective threshold is ambiguous**: it loads the full 205,620,298-row parquet (L7) and mines at `min_support=0.000000052` (L8). 5.2e-8 × 76,890,945 = 3.998 (ceil 4) but 5.2e-8 × 205,620,298 = 10.69 (ceil 11). Its totals (18,935,899 itemsets; K=2 66,703; K_max 21) lie between `beyond` (min_count 16: 14,558,875; K=2 60,088; K_max 20) and `godmode` (min_count 8: 26,849,505; K=2 73,786; K_max 22). `holdmybeer_real.log` then recomputes min_support as 4/205,620,298 and verifies the ceil against 205,620,298 (L5-6), and `yolo.log` does the same with 3/205,620,298 — i.e., those two scripts treated N0 = 205,620,298 as the miner's denominator, whereas the direct-path logs report the miner working on 76,890,945 transactions.
6. **Same deepest itemset across thresholds**: the K=22 itemset (8 proteins; `plddt_mean_med` + PF00271 + PF00270 + 19 GO terms) is identical in `godmode_mining.log` L31-35 and `holdmybeer_real.log` L37-41; K=21 has 23 itemsets at min_count 8 vs 27 at min_count 4; K=20 has 255 vs 342; the top-3 protein counts at K=21 (13/8/8) and K=20 (57/40/15) are identical in both files.
7. **SON runs' K-distributions are non-monotonic in the threshold** while direct runs are monotonic: K=1 count is 667 (`pipeline`, 0.1 %) → 455 (`ultra`, 0.01 %) → 47 (`extreme`, 0.001 %); K=2 is 1,504 → 4,152 → 990. Direct-path logs report K=1 = 1,002 at min_count 77, 16, 8 (and the JSON at 768/769), and K=2 rising 22,019 (JSON@768) → 39,125 (77) → 60,088 (16) → 66,703 (holdmybeer) → 73,786 (8) → 94,427 (4). The paper attributes SON's shortfall to chunk-local pruning (`et_miner_proteome.tex` L238).
8. **Per-K timings sum to roughly half of the reported mining time**: `direct` 71.1 s vs 119.3 s (L37); `beyond` 141.2 s vs 281.0 s (L32); `godmode` 204.8 s vs 440.5 s (L28). The CSR build itself is logged as ~1.1 s (three `Direct CSR path` timestamps, e.g. `direct` L14-16).
9. **Candidate counts**: `direct` and `beyond` print `0 candidates` for every K ≥ 3 (only K=1: 1,002 and K=2: 501,501 = C(1002,2) are non-zero); `godmode` prints no candidate counts at all. The current `gpu/mining.py::_log_level` (L490-497) format (`candidates=… → frequent=… (…s) | cumulative=… | RAM=… VRAM=…`) does not match any log line — the logs' `K=n: … candidates -> … frequent (…ms)` lines came from ad-hoc `level_callback` printers in scripts not in the repo. No RAM or VRAM figure appears in any log.
10. **af-extract format strings differ from the committed source**: the log prints `Loaded … annotations (… Pfam domains, … GO terms)` (L2036), `Frequencies: … unique Pfam, … unique GO from …` (L2063) and `Item encoding: 6 pLDDT + 500 Pfam + 500 GO = 1006 total items` (L2064), whereas `af-extract/src/annotations.rs:228/376`, `main.rs:443` and `transaction.rs:178` in the tree print additional InterPro/EC/taxonomy fields. The log therefore came from an earlier af-extract build than the one committed in `65d9098`; the only history of that source is that single commit.
11. **pLDDT CSV row count**: "pLDDT: 214683830 rows" (L8; presumably `wc -l` incl. header) vs "Read 214683829 rows" (L2061); 205,620,298 passed + 9,063,531 skipped = 214,683,829.
12. **Timing bookkeeping in Step 1**: af-extract self-timed 3,777.0 s (L2277; equals its own first→last timestamp span) vs bash `real` 65m31.040s = 3,931.0 s (L2279): 154 s unaccounted for by the binary's own timer; throughput 54,440 proteins/s = 205,620,298 / 3,777.0.
13. **TrEMBL file size units**: "150G" (`pipeline` L7, `watcher` L6, `du`/`ls -h` style) vs aria2c "149GiB" total (`watcher` L4); the aria2c line's last snapshot is 97 % (145 GiB) yet the watcher declares FINISHED — the final progress redraws were not captured. The download source URL and UniProt release are absent from every log (the later `deploy/RUNBOOK_base214m.md` L66-67 uses `…/current_release/…/uniprot_trembl.dat.gz`; `paper/peer_review_jul12.md` L115 cites "UniProt release 2025_01").
14. **Annotation counts**: 202,556,314 annotation records / 25,475 Pfam domains / 26,536 GO terms in the DAT (L2036) vs 24,291 unique Pfam / 25,993 unique GO among the 205,620,298 pLDDT-passing proteins (L2063), of which only the top 500 + 500 became items (L2064); 1006 items defined, 1002 frequent at every direct-path threshold down to min_count 8 (`godmode` L3).
15. **Denominator of "transactions"**: the parquet holds 205,620,298 rows including 0- and 1-item proteins (L2271, L2286); mining in `pipeline` Step 3, `direct`, `beyond`, `godmode` reports 76,890,945 (37.4 %) transactions; `extreme`/`ultra`/`madman` print both numbers; `holdmybeer`/`holdmybeer_real`/`yolo` reference 205,620,298 (see item 5). Support fractions printed in `extreme` and `ultra` are relative to 76,890,945 (e.g. 10,916 / 76,890,945 = 0.000142, L32; 8,257,363 / 76,890,945 = 0.10739, `ultra` L40).
16. **Hardware / software provenance**: the only hardware string in all 11 files is "RELEASING THE H100s..." (`ultra` L8, plural); the paper's Table caption says "a single NVIDIA H100 80 GB" (`et_miner_proteome.tex` L216). No hostname, driver, CUDA, Python/CuPy/Polars version, git SHA, or exit code appears in any log; the mining logs use Python-logging timestamps (`2026-02-09 05:46:42,182`) for the miner lines and no timestamps for the script lines.
17. **Decoded-pattern dump cross-check**: `results_214m/decoded_top_k_patterns.txt` header "Total itemsets K>=15: 5,351" equals `direct_mining.log` K=15…19 (4,155 + 1,003 + 173 + 19 + 1 = 5,351), tying that dump to the Blitz run, not to the min_count-8 run whose K≥15 total is 310,527 + 118,659 + 37,261 + 9,375 + 1,818 + 255 + 23 + 1 = 477,919.
18. **Output artefacts named in the logs (none survive in the repo)**: `/root/alphafold-data/full_214m/{transactions_214m,item_mapping_214m,itemsets_214m,itemsets_214m_ultra,itemsets_214m_extreme,itemsets_214m_direct,itemsets_214m_beyond}.parquet`; `godmode`, `holdmybeer`, `holdmybeer_real` print byte counts (85,131,478 / 63,377,870 / 149,590,799) but no path; `/root/uniprot_trembl.dat.gz`; `/root/alphafold-data/plddt_metadata.csv`.
19. **Streaming parameters are unrecoverable from the logs**: chunk size, local support factor, n_gpus and max_length for `pipeline` Step 3, `ultra`, `extreme` and `madman` are not printed (current tree defaults: `streaming/son.py:84-86` chunk_size 40,000,000, local_support_factor 0.9; `experiments/experiment_direct_vs_son.py:49` `--chunk-size` default 40,000,000); `extreme`/`madman` banners state "MAX LENGTH 20"; `direct` "Max K: 20"; `beyond` "Max K: 25"; `godmode`, `holdmybeer*`, `yolo` print none.

## SECTION D — TOTALS

Total rows: **690** (IDs L2-001 … L2-690).

| file | rows | deterministic | hardware-dependent | method-parameter | external-fact | software |
|---|---|---|---|---|---|---|
| `beyond_mining.log` | 53 | 45 | 1 | 5 | 1 | 1 |
| `direct_mining.log` | 53 | 44 | 3 | 4 | 1 | 1 |
| `extreme_mining.log` | 59 | 54 | 1 | 4 | 0 | 0 |
| `godmode_mining.log` | 40 | 36 | 1 | 1 | 1 | 1 |
| `holdmybeer.log` | 32 | 25 | 1 | 5 | 0 | 1 |
| `holdmybeer_real.log` | 41 | 36 | 1 | 3 | 0 | 1 |
| `madman_mining.log` | 5 | 1 | 0 | 4 | 0 | 0 |
| `pipeline_214m.log` | 180 | 155 | 7 | 4 | 12 | 2 |
| `ultra_mining.log` | 33 | 28 | 2 | 3 | 0 | 0 |
| `watcher.log` | 189 | 155 | 11 | 5 | 15 | 3 |
| `yolo.log` | 5 | 1 | 0 | 4 | 0 | 0 |
| **total** | **690** | 580 | 28 | 42 | 30 | 10 |

Lines with digits that were deliberately NOT given a row (all other digit-bearing lines are covered; verified by script):

- `direct_mining.log` L62: output path only (/root/alphafold-data/full_214m/itemsets_214m_direct.parquet) — cited in Section B
- `extreme_mining.log` L27: output path only (/root/alphafold-data/full_214m/itemsets_214m_extreme.parquet) — cited in Section B
- `extreme_mining.log` L29: section heading '=== DEEPEST PATTERNS (K=13 down to K=10) ===' (labels only)
- `extreme_mining.log` L33…71 (17 lines): itemset content line (identifiers; captured in context of the support rows)
- `godmode_mining.log` L33…62 (20 lines): itemset content line (identifiers; captured in context of the protein-count row)
- `holdmybeer.log` L7: input path only (/root/alphafold-data/full_214m/transactions_214m.parquet) — cited in Section B
- `holdmybeer_real.log` L39…68 (20 lines): itemset content line (identifiers; captured in context of the protein-count row)
- `pipeline_214m.log` L6, 2283, 2285, 2293, 2310: step label (ordinal only)
- `pipeline_214m.log` L10: input path only (/root/uniprot_trembl.dat.gz) — cited in Section B
- `pipeline_214m.log` L2037: input path only (/root/alphafold-data/plddt_metadata.csv) — cited in Section B
- `pipeline_214m.log` L2275, 2276, 2308: output path only — cited in Section B
- `pipeline_214m.log` L2313: section heading (print-out choice 'top 30')
- `pipeline_214m.log` L2345: section heading (print-out choice 'top 20')
- `ultra_mining.log` L27: output path only (/root/alphafold-data/full_214m/itemsets_214m_ultra.parquet) — cited in Section B
- `ultra_mining.log` L31, 39: section heading ('top 3' is a print-out choice, not a result)
- `ultra_mining.log` L33, 35, 37, 41, 43, 45: itemset content line (identifiers; captured in context of the support rows)
- `watcher.log` L2: tool name only ('aria2c')
- `watcher.log` L15, 2292, 2294, 2302, 2319: step label (ordinal only)
- `watcher.log` L19: input path only (/root/uniprot_trembl.dat.gz) — cited in Section B
- `watcher.log` L2046: input path only (/root/alphafold-data/plddt_metadata.csv) — cited in Section B
- `watcher.log` L2284, 2285, 2317: output path only — cited in Section B
- `watcher.log` L2322: section heading (print-out choice 'top 30')
- `watcher.log` L2354: section heading (print-out choice 'top 20')

Script consistency checks (computed from the log text while generating the table):

- beyond_mining.log: per-K == K-dist: True; sum(K-dist)=14,558,875 vs stated 14,558,875; sum(per-K ms)=141.2s vs stated 281.0s; K_max=20
- direct_mining.log: per-K == K-dist: True; sum(K-dist)=2,841,280 vs stated 2,841,280; sum(per-K ms)=71.1s vs stated 119.3s; K_max=19
- extreme_mining.log: sum(K-dist)=22,846 vs stated 22,846; K_max=13
- godmode_mining.log: sum(per-K)=26,849,505 vs stated 26,849,505; sum(per-K ms)=204.8s vs stated 440.5s; K_max=22 (no K-distribution block; per-K lines carry no candidate counts)
- holdmybeer.log: sum(K-dist)=18,935,899 vs stated 18,935,899; K_max=21
- holdmybeer_real.log: sum(K-dist)=48,007,493 vs stated 48,007,493; K_max=22
- madman_mining.log: no results, no timestamps, no error text — log ends at line 8 (blank)
- pipeline_214m.log: Parsed-lines=2025 (first 100K @ 2026-02-09T03:40:40.224Z, last 202500K @ 2026-02-09T04:36:18.840Z); Written-lines=205; sum(K-dist)=5,305; K_max=9
- ultra_mining.log: sum(K-dist)=51,124 vs stated 51,124; K_max=13
- watcher.log: Parsed-lines=2025 (first 100K @ 2026-02-09T03:40:40.224Z, last 202500K @ 2026-02-09T04:36:18.840Z); Written-lines=205; sum(K-dist)=5,305; K_max=9
- yolo.log: no results, no timestamps — log ends at line 6
