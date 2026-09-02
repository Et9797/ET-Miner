# Keying notes — review claims part 1 (R1-001..R1-835)

Inputs: `phase1/claims_reviews_part1.md` Section A (835 rows; F1 = `PAPER_V2_REVIEW.md`, F2 = `peer_review_jul12.md`, F3 = `review_b1_hostile.md`), vocabularies `quantities.csv` (336 keys) and `new_quantities_logs_new.csv` (347 keys), conventions `quantities_notes.md`.

Outputs:

- `claims_reviews_part1_keyed.csv` — 835 rows, columns `claim_id,qkey,value,unit,category,stance,file,line,quoted_context` (all non-key fields verbatim from the claims table; no row contained an escaped `\|`).
- `new_quantities_reviews_part1.csv` — 195 new qkeys, same 7 columns as `quantities.csv`; `claimed_value_tex` = `review:` + every distinct value string of the mapped rows in document order joined by ` / `; `tex_lines` = `<review file basename>:<line>` (basenames, not F1/F2/F3, so part-2 files cannot collide); `category` = majority category of the mapped rows (tie → first row).
- Generator: `<scratchpad>/build_reviews_part1.py` with the explicit per-ID maps `map_f1.py`, `map_f2.py`, `map_f3.py` and definitions `newkeys.py`. Nothing is inferred by pattern matching.

Validation printed by the generator: every R1 ID appears exactly once (True); every qkey exists in `quantities.csv`, `new_quantities_logs_new.csv` or the new file (True); every new key is used at least once; both CSVs re-parse with Python `csv` at 9 and 7 columns (836 / 196 lines incl. header).

## Counts

| | rows |
|---|---|
| mapped to existing keys | **498** (487 to `quantities.csv` keys, 11 to `new_quantities_logs_new.csv` keys) |
| mapped to new keys | **337** |
| new qkeys defined | **195** |
| distinct keys used | 343 (148 existing: 143 paper + 5 log keys; 195 new) |

Per file: F1 290 existing / 37 new; F2 137 existing / 26 new; F3 71 existing / 274 new (F3 analyses the 35K-vocabulary result files, which the paper never mentions, plus its severity scores).

Log-vocabulary keys reused: `run_minc4_min_count`, `run_minc4_itemsets` (the paper's old "verification run at min_count=4 → 48M itemsets" = holdmybeer_real.log), `extract_progress_transactions_written` ("Written 48M transactions"), `extract_time_s` (af_extract 3777.0 s), `meta_log_pipeline_start_timestamp` (the "Feb 2026 run date" F1 reads from the log).

New keys by prefix: run_ 59 (v35k campaign, v35k_null, v35k_wave3, v35k_power, v35k_16p8b), meta_review_ 58, kdist_ 19 (v35k, v35k_wave3, v35k_power), null_ 17 (`null_v35k2perm_k<K>_*`, `null_p_bound_rule_of_three`, `null_kmax_bio_ratio`, `null_depleted_k`), misc_ 12, vocab_ 8 (`vocab_v1_*` old Table 1, `vocab_v35k_*`), mem_ 7, cost_ 5, alg_ 2, dataset_ 2, ext_ 2, k22_ 2, hw_ 1, sw_ 1. Reproducibility: external 84, derived 40, rerun 40, setup 16, hardware 15.

## 20 most-used keys

| rows | qkey |
|---|---|
| 34 | run_opus_kmax |
| 28 | null_permutations |
| 25 | k22_support |
| 19 | run_opus_min_count |
| 16 | null_p_bound_kge7 |
| 15 | dataset_multi_feature |
| 12 | vocab_items_frequent |
| 12 | run_opus_itemsets |
| 12 | run_v35k_null_permutations (new) |
| 11 | run_opus_time_min |
| 11 | run_v35k_n_gpus (new) |
| 10 | csr_bytes_gb |
| 10 | dense_gb |
| 9 | csr_vs_dense_full_ratio |
| 9 | run_v35k_itemsets (new) |
| 9 | null_perms_for_p001 |
| 9 | run_v35k_kmax (new) |
| 8 | dataset_plddt_pass |
| 8 | null_kmax |
| 8 | kdist_opus_peak_k |

## Conventions applied

- Rule 1 (same quantity → existing key) was applied regardless of stance: disputed values (40× → `csr_vs_dense_full_ratio`; "rule of three" p<0.45 → `null_p_bound_kge7`; 19–20 "corrected independent K" → `k22_independent_features`), recomputations (`1-0.05^(1/5)=0.451` / `0.4507` → `null_p_bound_kge7`; 62.95 min → `extract_time_min`; 214,683,829 rows read → `dataset_metadata_rows`; 316,421,093 nz → `csr_nnz`; 5.06 GB → `csr_bytes_gb`; 26,849,505 column sum → `run_opus_itemsets`), and requested/proposed text (NEW-text rows carry the same keys as the OLD-text rows they replace).
- Paper-notes conventions reused unchanged: "1×/single (NVIDIA) H100" → `hw_gpu_model`, bare "single GPU"/"1 GPU" → `hw_gpu_count`; "K≥7 absent from null" / "K≥7 deep-pattern cutoff" → `null_kmax`; "0 of 5" / "0/5" → `null_perms_reaching_kge7`; "K≥4 enriched" → `null_enriched_from_k`; ">3,700" → `null_z_min_k4to6`; "4–6" → `null_z_headline_k_range`; "n=5"/"5 permutations"/"5 null runs" → `null_permutations`; "4 df" → `null_t_df`; "100 / 100+ permutations" for the 1K null → `null_perms_for_p001`; "p<0.01" → `null_p_target`; "~130 s" and F3's "~86 s" per permutation → `null_per_perm_time_s` (X-01); 21× and 21.4× → `son_speedup` (X-02); "millions of itemsets tested" → `run_opus_itemsets`; "hundreds of millions of proteins" / "214M" → `dataset_metadata_rows`; "≥2 features" → `alg_transaction_min_features`; "22 K-levels" / "22-row K-distribution" / "K=22 pattern/ceiling/flagship" → `run_opus_kmax`; "22 features = 2 Pfam + …" → `k22_n_features`; "each of the 8 proteins carries exactly 22" → `k22_proteins_features_each`; "K=23 impossible" → `k22_impossible_k`.
- "≥8 proteins" in every phrasing (feature-retention wording the paper used, mining min_count, godmode_mining.log:2, proposed text) → `run_opus_min_count`, as for the paper rows at l.129.
- The Direct-vs-SON block (R1-196..201) uses the alias keys `run_power_son_itemsets` / `run_power_son_time_s`; the campaign-table Power row (R1-135..139) uses `run_power_itemsets` / `run_power_time_min`, mirroring the paper keying.
- "peak K=9 = 3,529,257 (13.14%)" → `kdist_opus_peak_k/_count/_pct` (not `kdist_opus_k9_*`), as for the paper prose rows.
- 35K campaign naming: `run_v35k_*` = the Direct-GPU triplicate / "35K Base" row (606,292 ± 28, K_max 17, 654 s, 4×H200, 109.2M transactions, min_count 1,093); `run_v35k_null_*` and `null_v35k2perm_k<K>_*` = the 2-permutation null (min_count 1,090); `run_v35k_wave3_*` / `kdist_v35k_wave3_*` = the incomplete 0.1% run (min_count 109,225, local 27,307); `run_v35k_power_*` / `kdist_v35k_power_*` = power_test_v1_full.log (min_count 1,093, K=2 586 s, 902K pairs, 17.4M at K=4, stalled at K=4); `run_v35k_16p8b_*` = the plan-file "16.8B @ 8×H200" figure. `vocab_v35k_items` = 34,920 (and "35K" where it enters arithmetic); `kdist_v35k_k1_count` = the "K=1: 34,920 frequent items" log line.
- Per-K cumulative times of the wave-3 table → `run_v35k_wave3_k<K>_cum_time_s` (K=1..10); "743 s", "743.4 s", "12+ minutes" at K=9 are one key; ">743 s, still running" → the K=10 key.
- `run_v35k_time_min` (10.9) is kept as a minutes alias of `run_v35k_time_s` (654), following the paper vocabulary's `time_min`/`time_s` pairs.
- Review meta-information: `meta_review_f1_*` / `meta_review_f2_*` / `meta_review_f3_*`, one key per distinct fact; F3 severities are `meta_review_f3_severity_major<n>` / `_minor<n>` with `_v1` suffix for the review-v1 score; the summary-table repeats (R1-828..835) share the keys of the section headings. Before/after facts share one key with two values (`meta_review_f2_uncited_bibitems` = 1 → 0).
- Hypothetical numbers and parameters of requested experiments with no counterpart in any run → `misc_*` (12 keys, all `external`).

## Ambiguous rows and the choice made

- **R1-007** "22-row K-distribution" → `run_opus_kmax` (number of K rows = K_max, as for the paper's "22 K-levels").
- **R1-033, R1-086** (`1-0.05^(1/5)=0.451` / `=0.4507`) → `null_p_bound_kge7` (the reviewer recomputes the bound value); the bare formula rows R1-291, R1-296 → `null_p_bound_formula`; R1-034, R1-088 ("3/5=0.60") → new `null_p_bound_rule_of_three`.
- **R1-035, R1-227, R1-298, R1-301** (min_count=4 verification run) → `run_minc4_min_count`, and **R1-228, R1-299, R1-302** (48M itemsets) → `run_minc4_itemsets`: the paper's old sentence matches holdmybeer_real.log (48,007,493 itemsets, K_max 22), even though F1 says no such run exists. **R1-229, R1-300** ("Written 48M transactions") → `extract_progress_transactions_written`.
- **R1-045** ("top-500-per-type frequency cap", one row for both caps) → `vocab_pfam_defined` (first of the two equal caps); no separate key for a phrasing variant.
- **R1-048** ("vastly more than 500 families occur in ≥8 proteins") → new `dataset_pfam_go_families_ge8` (rerun-able count, distinct from the 500 caps).
- **R1-093** (214,683,829 rows read) → `dataset_metadata_rows` (its stated reproduction target), not `extract_csv_rows` (214,683,830 wc-l count incl. header).
- **R1-097** ("TrEMBL: 150G" from the log) → `dataset_annotation_gb`, as the logs notes prescribe for du-style figures; **R1-098** 3777.0 s → `extract_time_s`; **R1-099** 62.95 min → `extract_time_min`.
- **R1-109** ("only two pLDDT labels in decoded_top_k_patterns.txt") → `vocab_plddt_frequent` (corroboration of the same count).
- **R1-164** ("all K=3..21 rows" verified) → `meta_review_f1_kdist_rows_range` (a verification-scope fact, not a quantity of the run).
- **R1-165** ("percentages sum to 100.0%") → new `kdist_opus_pct_sum`.
- **R1-219** ("Feb 2026 run date" in the log) → `meta_log_pipeline_start_timestamp` rather than `dataset_access_date` (the row is explicitly about what the log records).
- **R1-222, R1-223** ("decoded_top_k_patterns.txt caps at K=19 (187 proteins)") → `pattern_k19_k` / `pattern_k19_support`: the K=19 / 187-protein pattern is the paper's RNA-spliceosome pattern read from the same decoded file; the "different run" remark is not keyed separately. **R1-224** ("no artifact contains any K≥20 itemset") → `meta_review_f1_artifact_kge20_absent` (a statement about the artifact set; note that godmode_mining.log's decoded K=20/21 itemsets are keyed in the logs vocabulary as `pattern_opus_k20/21_*` — no judgement made here).
- **R1-237, R1-238** (606K @ 4×H200 in revision_notes_b3.tex) → `run_v35k_itemsets` / `run_v35k_n_gpus` (X-06: the same 35K result F3 analyses); **R1-239, R1-240** (16.8B @ 8×H200 in the plan) → `run_v35k_16p8b_itemsets` / `run_v35k_16p8b_n_gpus`.
- **R1-353, R1-357** ("K = 15–22" / "K ≥ 15") → one key `misc_f2_deep_k_range` (same lower bound, reviewer's own frame).
- **R1-354, R1-355, R1-356** (requested 5-permutation null at min_count 8) → `null_permutations` / `run_opus_min_count` / `null_kmax` (the numbers are the existing quantities; the request is carried by `stance`).
- **R1-369** (value "6", "the K=6 value does not reproduce from the rounded μ/σ") → `null_k6_z`: the disputed quantity is the K=6 Z; the row value is only its K label.
- **R1-375, R1-778** (~3.6 GPU-hours / ~2.4 hours for 100 permutations at 1K) → one key `cost_null_100perm_1k_hours` (same computation with 130 s vs 86 s per permutation); **R1-560/R1-772** and **R1-562/R1-774** likewise share `cost_null_100perm_v35k_hours` (~18 / ~4.5) and `cost_null_200perm_v35k_hours` (~36 / ~9) (X-07).
- **R1-387** ("a single … protein family can generate the entire signature") → new `k22_n_families` (checkable once the accessions exist; the row is a hypothetical).
- **R1-401, R1-402** ("~26 GB … 214M set") → `bitvec_gb` + `dataset_metadata_rows` (the paper computes the 26 GB for 205.6M; N difference already noted in the paper notes).
- **R1-406** ("2 of 6 pLDDT bins pass") → `vocab_plddt_frequent`.
- **R1-437** ("≤ 4 GPU-hours to fix Majors 1–2") → new `cost_f2_majors_fix_gpu_hours`, kept apart from the 100-permutation cost it contains.
- **R1-477, R1-483** ("Claude Code Opus 4.6") → new `sw_claude_opus_version`; **R1-479, R1-486, R1-487** (Zenodo v1) → new `ext_zenodo_version` (distinct from `ext_zenodo_doi`).
- **R1-508, R1-536, R1-811** (K=5 "collapse" / "both null runs produced 1 itemset" / "null K_max 5") → `run_v35k_null_kmax`; the K=5 counts themselves → `null_v35k2perm_k5_mean/_std/_z`.
- **R1-540, R1-541** ("10 or 100 permutations … might range from 0 to 5") → `misc_v35k_null_hypothetical_perms` / `_k5_range` (pure hypotheticals, not the requested 100+).
- **R1-555, R1-559, R1-561, R1-769, R1-773** (100+/100/200 permutations for the 35K null) → `run_v35k_null_perms_requested`; the 1K counterparts (R1-565, R1-775) and the "both scales"/"condition for minor revision" rows (R1-768, R1-827) → `null_perms_for_p001` (the paper's own 100+ statement).
- **R1-564, R1-776** ("~86 s/perm" for the 1K null) → `null_per_perm_time_s` (same quantity as the paper's ~130 s; X-01).
- **R1-574** ("the 17 co-occurring features in the K=17 pattern") and **R1-685** (report K_max as "16–17") → `run_v35k_kmax` (no separate `_n_features` alias for the 35K deepest itemset).
- **R1-593/R1-682** ("17, 16, 17") → `run_v35k_kmax_per_run`; **R1-594/R1-683** ("2, 0, 1") → `kdist_v35k_k17_count_per_run`; the three totals R1-679..681 → `run_v35k_run1/2/3_itemsets`.
- **R1-596, R1-673** ("K=22 → K=17") → `run_v35k_vs_opus_kmax_change`; **R1-607** ("1,093 vs 8") → `run_v35k_min_count` (the comparison's new number).
- **R1-613** ("a term that appeared in millions of proteins", cytoplasm) → new `dataset_go_cytoplasm_support` (cf. `pattern_super_son_k1_1_support` = GO:0005737 8,257,363 in the logs vocabulary; not merged because the review row is generic).
- **R1-615** (local_min_count=274 from power_test_v1_full.log:2) → `run_v35k_local_min_count` (the local threshold of the min_count=1,093 4-GPU runs); **R1-657** (min_count=1,093 of the power test) → `run_v35k_min_count` (same threshold, same derivation), while the power test's own results (586 s, 902K pairs, 17.4M at K=4, K reached 4) get `run_v35k_power_*` / `kdist_v35k_power_*` because its counts are incompatible with the triplicate's 606K total.
- **R1-616/617/618** and **R1-700/701/702** (14,800 locally frequent / 8,554 frequent / 42%) → `kdist_v35k_wave3_k2_locally_frequent` / `kdist_v35k_wave3_k2_count` / `run_v35k_wave3_k2_local_prune_pct`: F3 line 90 attributes them to the power-test context but F3 line 167 and the wave-3 table (K=2 = 8,554) place them in wave3_base_partial.log:25; **R1-662** ("8.5K") and **R1-659** ("1.2 s at min_count=109,225") are the same wave-3 K=2 count / cumulative time.
- **R1-620, R1-732, R1-777, R1-781** (bare "single GPU" / "1 GPU") → `hw_gpu_count` by the paper convention, even where the GPU is a hypothetical baseline (R1-620, R1-732, R1-781).
- **R1-622, R1-676, R1-795, R1-834** ("8 vs 3") → new `k22_script_identified_proteins` together with **R1-624** (the 3); the plain "8 proteins" rows stay on `k22_support`.
- **R1-653, R1-719** (64K / 64,400 at K=7, the 35K peak) → `kdist_v35k_k7_count`; the peak position (R1-602, R1-718, R1-755, R1-807) → `kdist_v35k_peak_k`; no `kdist_v35k_peak_count` alias was created.
- **R1-663** ("candidate space at K≥3") → `misc_v35k_power_candidate_space_k` rather than `alg_prefix_loop_start_k` (the row is about candidate-space size, not the loop structure).
- **R1-703** ("each GPU sees 1/4") → new `run_v35k_gpu_row_fraction`; **R1-705** ("local_min_count = global/4") → new `alg_local_min_count_rule`.
- **R1-752, R1-782** (~119 GB "per GPU" / "if ~119 GB fits on H200") → one key `mem_v35k_bitvec_per_gpu_gb`.
- **R1-765** ("depleted K=2" in both nulls) → new `null_depleted_k`; **R1-813/R1-814** (~3.5× / ~3.4× bio-to-null K_max ratios) → `null_kmax_bio_ratio` / `null_v35k2perm_kmax_bio_ratio`.
- **R1-783, R1-784** (K_max "~17" or "22" of a hypothetical 35K min_count=8 run) → `misc_v35k_minc8_hypothetical_kmax`; **R1-780** (its min_count 8) → `run_opus_min_count`.
- **R1-797** (50K–100K closed itemsets, projection) → `run_v35k_closed_itemsets` (`derived`, value marked as a projection in the description).
- **R1-687..691** (Table 1 quoted by F3 as 247/302/289/161/3) and **R1-010..012** (F1's "fabricated v1 Table 1 247/752/3") → `vocab_v1_pfam`, `vocab_v1_go_mf/_bp/_cc`, `vocab_v1_go` (752 = 302+289+161), `vocab_v1_plddt` (X-03), all `external`.
