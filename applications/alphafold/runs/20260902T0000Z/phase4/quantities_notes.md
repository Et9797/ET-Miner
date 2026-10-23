# Phase 4 — controlled vocabulary of quantity keys (qkeys)

Files in this directory:

- `quantities.csv` — 336 canonical quantities, columns `qkey,category,unit,description,claimed_value_tex,tex_lines,reproducibility` (ordered by first appearance in the paper).
- `claims_tex_keyed.csv` — all 543 claim rows T-001..T-543 from `phase1/claims_tex.md` (Section A), each with exactly one qkey, columns `claim_id,qkey,value,unit,category,claim_type,tex_line,section,quoted_context`. `value/unit/category/claim_type/tex_line/section/quoted_context` are copied verbatim from the claims file (including the extractor's `[...]` cross-reference notes inside `quoted_context`).
- Generator: `<scratchpad>/build_phase4.py` (explicit one-line-per-claim mapping; nothing is inferred by pattern matching).

Validation printed by the generator: (a) every T-ID appears exactly once — True; (b) every qkey used exists in `quantities.csv` — True (336 used / 336 defined / 0 unused); (c) both files re-parse with Python `csv` at 7 and 9 columns respectively.

Reproducibility classes (count of qkeys): rerun 115, derived 101, setup 64, external 42, hardware 14.

## Conventions

1. **One qkey per distinct quantity.** Repeated statements of the same quantity (abstract, intro, table, discussion, conclusion) share one key; `tex_lines` lists every source line (de-duplicated, semicolon-separated, en-dashes normalised to ASCII).
2. **Prefixes** follow the task specification exactly where it gave one: `dataset_*`, `vocab_*`, `extract_time_min`, `csr_*`/`bitvec_*`/`dense_*`, `run_<base|super|power|blitz|ultra|opus>_<support_pct|min_count|itemsets|kmax|time_min|method>`, `run_power_direct_*`, `run_power_son_*`, `son_speedup`, `son_miss_rate_pct`, `kdist_opus_k<K>_count|pct`, `kdist_opus_peak_*`, `k22_*`, `null_*`, `hw_*`, `sw_*`, `ext_*`, `alg_*`, `mem_*`, `pipe_*`, `arch_*`, `misc_*`.
   - Campaign times are `run_<name>_time_min` (paper unit is minutes); the seconds-precision Direct/SON pair uses `run_power_direct_time_s` / `run_power_son_time_s`.
   - Added prefixes/keys not in the spec, because the claims needed them: `campaign_n_runs`, `campaign_support_span_orders`; `run_blitz_vs_power_{speedup,itemset_ratio,support_ratio}`; `run_opus_vs_gminer_transaction_ratio`; `son_k2_recovery`; `pattern_*` (the five highlighted intermediate-K patterns: `pattern_k<K>_k`, `pattern_k<K>_support`, `pattern_k17_member_1..3`, `pattern_k12_member_1..2`, `pattern_n_highlighted`, `pattern_support_precision`); `dataset_max_features_per_protein`, `dataset_access_date`, `vocab_pfam_go_frequent`, `vocab_plddt_bin_medium_edges`, `csr_h2d_transfer_gb`, `bitvec_subset_gb`, `dense_subset_gb`, `csr_vs_dense_{subset,full}_ratio`.
   - `kdist_opus_k<K>` and `k22_feature_<n>` use **unpadded** integers (`k1`..`k22`, `feature_1`..`feature_22`; `k22_feature_<n>` numbering = tab:k22 row order: 1-2 Pfam, 3-10 GO MF, 11-14 GO BP, 15-21 GO CC, 22 pLDDT bin).
   - The K=7–14 row of tab:null-model uses the suffix `kge7` (`null_kge7_bio|mean|std|z`), matching the spec's `null_p_bound_kge7`.
   - `misc_*` is used for 6 keys only (manuscript date, "100-million-transaction scale" rhetoric, the fig:concept schematic numbers, the introductory illustrative triple).
3. **`category`** = the dominant category of the mapped claim rows (majority; tie → first row in document order). Note that `dataset_metadata_rows` is therefore `external-fact` (4 of 6 rows are phrased as AlphaFold-DB background) while its `reproducibility` is `rerun` (the audit can count its own metadata rows).
4. **`claimed_value_tex`** = every distinct value string of the mapped rows, in document order, joined by ` / `. Most multi-value keys are formatting/rounding variants (see list below); the description says so, or says CONFLICT.
5. **`reproducibility`**: `rerun` = comes out of re-running extraction/mining; `derived` = arithmetic on re-run outputs (percentages, `ceil(s·n)`, ratios, memory-model formulas, Z/p); `hardware` = timings/throughput/PCIe traffic measured on this box; `external` = facts about other work, the world, or illustrations (not re-executable); `setup` = parameters/implementation facts confirmable only as "used as stated" (vocabulary sizes, seeds, versions, pseudocode constants, pipeline-table cells).
   - Base/Super/Power/Blitz: `support_pct` is `setup`, `min_count` is `derived`. Ultra/Opus: the paper says the percentages are nominal and the min counts (16, 8) are the actual thresholds, so `min_count` is `setup` and `support_pct` is `derived`.
6. **Aliases kept on purpose** (spec asked for both key families; a reproduction fills both from one run): `run_power_son_itemsets` ≡ `run_power_itemsets` (22,846), `run_power_son_time_s` ≡ `run_power_time_min` × 60; `kdist_opus_peak_count|pct` coincide with `kdist_opus_k9_count|pct` only while the peak stays at K=9 (they are semantically "value at the argmax"); `k22_n_features` ≡ `run_opus_kmax` by definition; `alg_bitvec_speedup_factor` ≡ `alg_bits_per_word`.
7. **Keys deliberately not created**: `run_power_son_kmax` (the only SON K-max statement is the tab:campaign cell T-123 → `run_power_kmax` = 13); `null_kge7_p` (the 7–14 p cell is the binomial bound → `null_p_bound_kge7`); `pipe_pcie_database_encoding` (T-502 "Once (~3 GB)" → `csr_h2d_transfer_gb`, the "once" being `pipe_h2d_transfer_count`); further `arch_*` keys (the numeric tab:gpu-arch cells are the same facts as elsewhere and are keyed to `arch_support_counting_intrinsic`, `ext_gminer*_support_counting_intrinsic`, `alg_pcie_bytes_per_level`, `ext_gminer_transactions_real`, `dataset_multi_feature`); `alg_k2_pairs` (no claim row states 501,501); hardware bandwidth keys (no claim row).

## qkeys whose paper values conflict internally

**True numeric conflict inside one qkey**

- `alg_pcie_bytes_per_level`: **~12 bytes** (l.443, l.808–810, l.839 tab:gpu-arch) vs **~4 bytes** (l.852 tab:gpu-pipeline caption, l.868 loop-control cell). `alg_pcie_bytes_total` (~264 = 22 × 12) is built on the 12-byte figure.

**Same quantity stated with different numbers under two keys** (kept separate because the spec names both keys)

- `run_power_min_count` = **768** (tab:campaign, l.226) vs `null_min_count` = **769** (l.362, l.393) — both are the min count at 0.001 % of 76,890,945 transactions; `ceil(1e-5 × 76,890,945)` = 769.

**Multi-value keys that are rounding / phrasing variants only (not conflicts)**

`dataset_metadata_rows` (>200 million / hundreds of millions / 214 million / 214M), `dense_gb` (>200 / ~206 / 206), `dataset_multi_feature` (76.9 million / 76.9M / 76,890,945), `dataset_plddt_pass` (205.6 million / 205,620,298 / 205.6M), `run_opus_itemsets` (26.8 million / 26,849,505 / 26.8M / "millions"), `kdist_opus_peak_count` (3,529,257 / 3.53M / 3.53 million), `csr_bytes_gb` (~5.1 / 5.1), `bitvec_gb` (~26 / 26), `csr_h2d_transfer_gb` (~3 / "~3 GB, once"), `hw_gpu_model` (five phrasings of "single NVIDIA H100 SXM5"), `null_support_pct` (0.001 / "Power threshold (0.001%)"), `ext_prior_domain_cooccurrence_kmax` (2 / 2 (pairwise) / 2 (pairs)), `alg_transaction_min_features` (>1 kept / 1 excluded), `pipe_h2d_transfer_count` (1 / once), `pipe_pcie_result_collection` (1 (single) / Once at end), `null_kmax` (6 / ">=7" absent), `null_enriched_from_k` (>=4 / 4), `k22_go_parent_child_pairs` (0 (none) / 0), `ext_bigminer_nodes` (30 / 30× servers), `ext_prior_gpu_fim_popcount_width_bits` (32 / 32-bit).

**Cross-key arithmetic tensions worth knowing before the join** (observations only, no judgement)

- `bitvec_gb` ~26 GB (205.6M proteins) vs `mem_dense_214m_gb` 27 GB (214M proteins): different N, both ≈ N × 1,002 / 8.
- `alg_coo_bytes_per_entry` 16 B (l.167, basis of the 5.1 GB) vs `mem_csr_bytes_per_entry` 8 B (tab:memory-comparison): different representations. The memory table is consistent with CSR = 8 B × nnz + 8 B × N row pointers.
- `mem_items_per_txn_214m` "~10" vs `csr_nnz`/`dataset_multi_feature` = 316M / 76.9M ≈ 4.1 for the mined subset.
- `mem_theoretical_items` 1,000 vs `vocab_items_frequent` 1,002 (the table says so explicitly).
- `misc_transaction_scale_claim` "100-million" vs `dataset_multi_feature` 76.9M.
- `null_k4_enrichment_ratio` "many orders of magnitude" vs `null_k4_bio`/`null_k4_mean` = 108,059 / 25,468 ≈ 4.2.
- `run_power_kmax` 13 (SON) vs `run_power_direct_kmax` 14 (exhaustive) — different methods, not a conflict; likewise `run_power_itemsets` 22,846 vs `run_power_direct_itemsets` 475,865.
- `null_k6_max` "at most 23" vs `null_k6_mean` 22 ± 1.1 — compatible.
- `null_z_min_k4to6` ">3,700" vs table +3,791 / +7,402 / +71,728 — compatible.

## Ambiguous T-rows and the choice made

- **"1 × (NVIDIA) H100" rows** T-007, T-034, T-104, T-291, T-417, T-424, T-461 → `hw_gpu_model` (they name the model; the count is implied). Bare "single GPU" rows T-089, T-092, T-372 → `hw_gpu_count`. T-025 (80 GB "capacity of the highest-end GPUs") → `ext_highend_gpu_vram_gb`, not `hw_gpu_vram_gb` (it is a claim about GPUs in general).
- **0.001 % rows**: T-146, T-152, T-161, T-378 (Power / controlled comparison) → `run_power_support_pct`; T-012, T-286, T-298, T-361, T-465 (null-model threshold) → `null_support_pct`. T-362 (min_count = 769 at l.393) → `null_min_count`.
- **1,002 rows**: T-179 (tab:kdist K=1 cell) → `kdist_opus_k1_count`; T-309 (tab:null-model Bio K=1) → `null_k1_bio`; T-310 and T-360 ("all 1,002 features remain frequent after shuffling") → `null_k1_mean`; all other 1,002 rows → `vocab_items_frequent`.
- **1,000 rows**: T-055 → `vocab_pfam_go_frequent`; T-528 (|F1| of the theoretical memory rows) → `mem_theoretical_items`.
- **"22" rows**: K-labels / ceiling / K-levels (T-224, T-227, T-377, T-431, T-434, T-449, T-492, T-393, T-430, T-464, T-416) → `run_opus_kmax`; "22 co-occurring features / 22-feature signature / 22 annotations" (T-226, T-228, T-258, T-433) → `k22_n_features`; T-264 ("each of the 8 proteins has exactly 22") → `k22_proteins_features_each`; T-266 ("no protein carries more than 22") → `dataset_max_features_per_protein`; T-265/T-267 (K = 23 impossible) → `k22_impossible_k`.
- **"8" rows**: T-045, T-054, T-107, T-141, T-365 (threshold) → `run_opus_min_count`; T-176, T-225, T-229, T-262, T-263 (proteins sharing the K=22 itemset) → `k22_support`.
- **"K ≥ 7 absent from null" rows** T-303, T-438, T-469 (value ">=7") and T-169/T-356 (value 6) → `null_kmax` (the claim is `null_kmax ≤ 6`; the row values differ in form only). T-295 ("no permutation produced any pattern at K≥7", value 0) → `null_kge7_mean`. T-308/T-352 ("0 of 5") → `null_perms_reaching_kge7`. T-305/T-343/T-353 (p < 0.45) → `null_p_bound_kge7`.
- **"K ≥ 4 enriched" rows** T-300, T-348, T-439, T-447 → `null_enriched_from_k`; T-440 ("many orders of magnitude") → `null_k4_enrichment_ratio`.
- **Headline Z rows** T-010/T-466 (">3,700") → `null_z_min_k4to6`; T-011/T-467 ("4–6") → `null_z_headline_k_range`.
- **Per-level PCIe rows** T-423, T-489–T-491, T-498 (tab:gpu-arch cell), T-501 (pipeline caption), T-511 (pipeline loop-control cell) → all `alg_pcie_bytes_per_level`, so the 12-vs-4 conflict is visible inside one key rather than split across `arch_`/`pipe_` keys. T-087 ("orders of magnitude less") → `alg_pcie_reduction_vs_conventional`; T-494 ("gigabytes", conventional systems) → `ext_conventional_gpu_fim_pcie_bytes`.
- **Transfer-count rows** T-086, T-088, T-422 → `pipe_h2d_transfer_count`; T-481 (pseudocode "Single transfer") and T-512 (pipeline "Once at end") → `pipe_pcie_result_collection`; T-502 → `csr_h2d_transfer_gb`.
- **popcount rows** T-482/T-497 → `arch_support_counting_intrinsic`; T-485/T-486 (prose "prior systems") → `ext_prior_gpu_fim_popcount_width_bits`; T-495/T-496 (table cells) → `ext_gminer_support_counting_intrinsic` / `ext_gminerpp_support_counting_intrinsic`; T-483 (1 clock cycle) → `hw_popcll_cycles` with reproducibility `external`.
- **64 rows** T-090, T-473, T-475, T-484 → `alg_bits_per_word`; T-091, T-474, T-476 (64×) → `alg_bitvec_speedup_factor`. T-475's tex line is the range "746–751" in the claims file; it is kept verbatim in `claims_tex_keyed.csv` and appears as the token `746-751` in `quantities.csv` `tex_lines` (the only non-integer token).
- **Memory rows**: T-420 (~10 GB bit-packed subset bitmap, l.440) → `bitvec_subset_gb`, kept separate from the theoretical 9.6 GB cells (`mem_dense_76p9m_gb`, |F1| = 1,000). T-517 ("modest 1.4× reduction", l.874) → `mem_ratio_214m`; T-513 (7.8× at 0.01 %) → `mem_ratio_0p01pct`; T-515 ("no savings at ≥1 % density") → `mem_breakeven_density_pct`; T-516 (~1 % density) → `mem_proteome_density_pct`. T-519/T-521 (214M in the memory table) → `dataset_metadata_rows`.
- **Highlighted patterns**: T-276 ("17-feature combination") → `pattern_k17_k`; T-445/T-446 ("K=11 to K=19" range) → `pattern_k11_k` / `pattern_k19_k`; T-443 → `pattern_k12_k`; T-279 (footnote on precision) → `pattern_support_precision`; T-268/T-444 → `pattern_n_highlighted`.
- **Rhetorical scale rows** T-371/T-459 ("100-million-transaction scale") → `misc_transaction_scale_claim` rather than `dataset_multi_feature`, so the 76.9M key is not polluted with a different number. T-448 ("millions of patterns tested") → `run_opus_itemsets`. T-015 ("hundreds of millions of proteins") → `dataset_metadata_rows`.
- **Illustrations / hypotheticals**: T-016/T-017 (introductory triple, "thousands") → `misc_example_triple_k` / `misc_example_triple_support` (the latter `rerun`: its support can be looked up, the triple is a subset of the K=22 itemset); T-026/T-027 (fig:concept schematic) → `misc_concept_figure_k` / `misc_concept_figure_support` (`external`).
- **Document metadata**: T-001 → `misc_manuscript_date`; T-470/T-471 → `ext_zenodo_doi`; T-037/T-452/T-472 → `dataset_uniprot_release`; T-038 → `dataset_access_date`.
- **Ratios involving external facts**: T-390 (5.1× more transactions than GMiner) → `run_opus_vs_gminer_transaction_ratio` (`derived`); T-020/T-389 (three orders of magnitude) → `ext_prior_gpu_fim_scale_gap_orders` (`derived`, from a vague "few million").
- **Execution paths** T-069/T-070 → `sw_execution_paths` / `sw_execution_path_used`; **campaign meta** T-101/T-103/T-160 → `campaign_n_runs`, T-102 → `campaign_support_span_orders`; **null-model hypotheticals** T-367/T-368 → `null_perms_for_p001` / `null_p_target`; T-432 (four functional roles) → `k22_functional_roles` (`external`, interpretive).
