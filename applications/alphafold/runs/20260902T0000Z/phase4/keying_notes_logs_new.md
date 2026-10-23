# Phase 4 — keying of the log claims (`phase1/claims_logs_new.md`, L2-001..L2-690)

Files written (nothing else touched):

- `claims_logs_new_keyed.csv` — 690 rows, columns `claim_id,qkey,value,unit,category,file,line,quoted_context`; every L2 ID exactly once; `value/unit/category/line/quoted_context` copied verbatim from Section A (`line` keeps the extractor's en-dash ranges such as `11–2035`; `quoted_context` keeps the extractor's `[tag]` provenance prefixes). The only normalisation: the markdown back-ticks around the file name (`` `beyond_mining.log` `` → `beyond_mining.log`).
- `new_quantities_logs_new.csv` — 347 new qkeys, same 7 columns as `quantities.csv`; `claimed_value_tex` = `log:` + the distinct row values joined by ` / `; `tex_lines` = `file:line` of every mapped row (`;`-separated, en-dash → ASCII); `category` = majority category of the mapped rows (tie → first row in document order), `reproducibility` assigned per key.
- Generator: `<scratchpad>/build_phase4_logs_new.py` (explicit one-line-per-ID mapping; the only "loops" are over K for per-K rows, over rank for the rule lists, and the +222 offset for the watcher copy — each loop asserts the row's K / rule identity before assigning).

Validation printed by the generator: every L2 ID exactly once — True; every qkey non-empty and defined in `quantities.csv` or the new file — True; both CSVs re-parse with Python `csv` at 8 and 7 columns; 347 new keys defined / 347 used / 0 unused; 0 name collisions with the 336 existing keys.

## Rows mapped to existing vs new keys

106 rows → 52 existing keys; 584 rows → 347 new keys.

| file | rows | existing | new |
|---|---|---|---|
| beyond_mining.log (Ultra) | 53 | 7 | 46 |
| direct_mining.log (Blitz) | 53 | 7 | 46 |
| extreme_mining.log (Power SON) | 59 | 8 | 51 |
| godmode_mining.log (Opus) | 40 | 31 | 9 |
| holdmybeer.log (hmb) | 32 | 1 | 31 |
| holdmybeer_real.log (minc4) | 41 | 2 | 39 |
| madman_mining.log (madman) | 5 | 1 | 4 |
| pipeline_214m.log (extraction + Base SON + rules) | 180 | 20 | 160 |
| ultra_mining.log (Super SON) | 33 | 7 | 26 |
| watcher.log (download watcher + copy of pipeline_214m.log) | 189 | 21 | 168 |
| yolo.log (minc3) | 5 | 1 | 4 |

Existing keys used (rows): dataset_plddt_pass 15, dataset_multi_feature 13, vocab_items_defined 8, dataset_metadata_rows 4, csr_nnz 3, dataset_annotation_gb 3, vocab_items_frequent 3, dataset_go_families_observed 2, dataset_pfam_families_observed 2, k22_support 2, kdist_opus_k20/k21/k22_count 2 each, run_base_itemsets 2, run_power_support_pct 2, run_super_support_pct 2, vocab_go/pfam/plddt_defined 2 each, and 1 each for hw_gpu_model, kdist_opus_k1..k19_count, pattern_k12_support, run_blitz_itemsets, run_blitz_min_count, run_opus_itemsets, run_opus_min_count, run_power_min_count, run_power_son_itemsets, run_power_son_time_s, run_super_itemsets, run_super_min_count, run_ultra_itemsets, run_ultra_min_count, run_ultra_support_pct.

New keys by prefix: extract_ 123 (of which 98 are the per-rule conf/lift keys of the Step 4 print-outs), kdist_ 117, pattern_ 57, run_ 36, download_ 7, meta_log_ 7. By reproducibility: rerun 274, derived 30, hardware 22, setup 14, external 7.

## The 20 most-used keys (rows)

1. extract_progress_csv_read — 42 (21 pLDDT-CSV progress rows × 2 logs)
2. dataset_plddt_pass — 15
3. dataset_multi_feature — 13
4. vocab_items_defined — 8
5. dataset_metadata_rows — 4
6. extract_rules_count — 4
7. extract_rules_toplift_3_conf — 4 (lift-list #3 and confidence-list #1 are the same rule, × 2 logs)
8. extract_rules_toplift_3_lift — 4
9. vocab_items_frequent — 3
10. csr_nnz — 3
11. meta_log_pipeline_start_timestamp — 3
12. dataset_annotation_gb — 3
13.–20. tied at 2 rows: run_ultra_min_count_nominal, kdist_ultra_k1_count … kdist_ultra_k7_count (and, beyond rank 20, every other kdist_ultra_*/kdist_blitz_* key, run_hmb_min_count, run_hmb_itemsets, run_minc4_min_count, run_minc4_itemsets, kdist_opus_k20/21/22, k22_support, the pipeline/watcher duplicates of every extraction key, etc.).

## Conventions applied

1. **Existing key wins** (rule 1): a log number that is the same underlying quantity as a paper key is keyed to the paper key regardless of formatting (`1002` / `1,002`, `316,421,093` vs "316 million", `214M` / `214683829`, `76.9M`, `150G`).
2. **Run names for new keys**: `base_son`, `super_son`, `power_son` for the three SON runs (so `run_base_son_time_s`, `run_super_son_time_s`, `kdist_power_son_k<K>_count`, `run_power_son_max_length`), `blitz`, `ultra`, `opus`, `hmb`, `minc4`, `minc3`, `madman` for the rest. Existing families (`run_base_*`, `run_super_*`, `run_power_*`, `run_power_son_*`, `kdist_opus_*`) are reused as they are; seconds-precision times get new `run_<run>_time_s` keys (run_opus_time_s 440.5, run_ultra_time_s 281.0, run_blitz_time_s 119.3, run_super_son_time_s 257.6, run_base_son_time_s 113.9; extreme's 1085.6 s → existing run_power_son_time_s).
3. **Per-K rows**: both the `[per-K] K=n: … frequent` line and the `[K-dist] K=n:` line of a run map to the same `kdist_<run>_k<K>_count` (Section D verified they are equal). Deepest-pattern / pattern-block headers (`--- K=13 (2 itemsets) ---`, `=== K=22 (1 itemsets) ===`) map to the same kdist key. The candidates and ms figures inside the per-K contexts have no rows of their own, so no `kcand_*` / `ktime_*` keys were created.
4. **Deepest / decoded patterns**: `pattern_<run>_k<K>_<n>_support` (protein count) and `pattern_<run>_k<K>_<n>_support_frac` (fraction) per listed itemset, n = position in the log's list; the members are spelled out in the description. Two exceptions keyed to paper keys: the single K=22 itemset (see below) and extreme's K=12 #5 (see below).
5. **Progress series**: one key per series, many rows share it (`extract_progress_csv_read` 42 rows, `extract_progress_dat_parsed` 2, `extract_progress_transactions_written` 2).
6. **Rule print-outs**: one key per rule and metric, rank-based (`extract_rules_toplift_<n>_conf|lift`, n = 1..30; `extract_rules_topconf_<n>_conf|lift`, n = 2..20). Lift of toplift #1 is `extract_rules_max_lift` (no separate `toplift_1_lift`). Confidence-list #1 (PF00116 => GO:0004129 + GO:0005507) is the same rule as lift-list #3, so its rows reuse `extract_rules_toplift_3_conf|lift` and there is no `topconf_1_*`. Reverse rules (A=>B / B=>A) keep separate lift keys even though lift is symmetric.
7. **watcher.log** rows 506..685 are the byte-identical copy of pipeline_214m.log rows 284..463 (offset +222, value/unit/category equality asserted) and carry the same keys; every extraction key therefore lists both `pipeline_214m.log:<line>` and `watcher.log:<line>` in `tex_lines`.
8. **Categories/reproducibility of new keys**: timings/throughput → hardware; counts/itemsets/supports/K_max/parquet sizes → rerun; script-computed floor thresholds, support fractions derived from counts, support fractions of patterns → derived; configured max K / target K / actual min_count / column indices / pLDDT threshold → setup; timestamps → external; all `download_*` → hardware (as instructed, although `download_size_gib` and `download_connections` are arguably setup facts).

## Ambiguous rows and the choice made

- **"transactions" counts.** Every `76,890,945` row (Direct CSR path "transactions", "With >1 item", "Mining … annotated proteins", hmb's "76.9M") → `dataset_multi_feature`; every `205,620,298` row ("Total transactions", "Written … transactions", "Transactions:", Step 2 stats, frequency counting) → `dataset_plddt_pass`. This includes the `verify: ceil(min_support * 205620298)` denominators of holdmybeer_real (L2-240) and yolo (L2-689): they were keyed by their number, not to a `run_<run>_n_transactions` key, so the fact that those two scripts used N0 = 205,620,298 as the miner denominator (paper: n = 76,890,945) is visible only in `quoted_context` and in the descriptions of `run_minc4_support` / `run_minc3_support`.
- **Script-computed vs miner-applied min_count.** beyond "Min proteins: ~15" / "Min support count: 15" (L2-002, L2-005) → `run_ultra_min_count_nominal`; "min_count=16" (L2-008) → `run_ultra_min_count`. direct "Min support count: 76" (L2-058) → `run_blitz_min_count_nominal`; "min_count=77" (L2-062) → `run_blitz_min_count`. For the SON logs only the floor is printed and it equals the paper's tab:campaign value: extreme 768 (L2-110) → `run_power_min_count`, ultra_mining 7,689 (L2-467) → `run_super_min_count`; madman 76 (L2-281) → new `run_madman_min_count`.
- **Support as a fraction vs percent.** Percent-valued rows go to the existing `run_<run>_support_pct` (beyond 0.00002, extreme 0.001 ×2, ultra_mining 0.01 ×2) or new ones (`run_madman_support_pct` 0.0001 ×2, `run_hmb_support_pct` 0.000005). Fraction-valued rows get new `run_<run>_support` keys (direct 1e-06 → `run_blitz_support`; hmb 0.000000052; minc4 1.945e-08; minc3 1.459e-08) rather than being folded into the `_pct` keys, so that one key never mixes units.
- **hmb threshold.** "4 proteins" (L2-206) and "~4 proteins" (L2-211) → `run_hmb_min_count` (nominal; the log never prints the miner-applied count). "Target: K=23+" (L2-209) → `run_hmb_target_k`; yolo "The DEFINITIVE answer to K=23" (L2-687) → `run_minc3_target_k`.
- **K_max rows.** Only hmb ("MAX K = 21", L2-236) and minc4 ("MAX K = 22", L2-267) state a K_max as a row → `run_hmb_kmax`, `run_minc4_kmax`. No other log has a K_max row (the K=19/20/22 terminal levels are count rows → `kdist_*_k<K>_count`), so `run_blitz_kmax`, `run_ultra_kmax`, `run_opus_kmax`, `run_power_kmax`, `run_super_kmax`, `run_base_kmax` are not used.
- **"1002 frequent items" vs K=1 counts.** The Direct-CSR-path line "1002 frequent items" (L2-009, L2-063, L2-169) → `vocab_items_frequent`; the K=1 per-K / K-dist rows (1,002 or, for SON runs, 667 / 455 / 47) → `kdist_<run>_k1_count` (mirrors the tex convention T-179 → kdist_opus_k1_count).
- **The K=22 itemset.** godmode "#1 (8 proteins)" at K=22 (L2-197) and holdmybeer_real's K=22 #1 (L2-269) → existing `k22_support`: both logs show exactly one K=22 itemset with the 22 members of tab:k22 (plddt_mean_med + PF00270 + PF00271 + the 19 GO terms). The K=21/K=20 top-3 of the two runs (13/8/8 and 57/40/15 proteins, identical member lists) get per-run keys `pattern_opus_k21_<n>_support` … and `pattern_minc4_k20_<n>_support` (the minc4 descriptions say they coincide with the opus ones).
- **extreme's deepest patterns vs the paper's highlighted patterns.** K=12 #5 (10,494 proteins; PF00905 + PF00912 + GO:0071555 + GO:0009252 + …; L2-143) → existing `pattern_k12_support` (~10,500, "Bacterial Cell Wall Synthase", the paper names exactly those two Pfam members; the only K=12 itemset in the log with them). Its fraction (L2-142) → new `pattern_power_son_k12_5_support_frac`. K=13 #1 (10,916) and #2 (10,913) both fit the paper's `pattern_k13_support` (~11,000, "Helicase-Recombinase DNA Repair Module", no members given) and K=11 #2 and #3 (both 15,886, ClpB-type AAA+ patterns) both fit `pattern_k11_support` (~16,000, no members given); because the paper does not identify which itemset it means, these four keep per-run keys (`pattern_power_son_k13_1|2_support`, `pattern_power_son_k11_2|3_support`) whose descriptions flag them as candidates. Nothing in the logs corresponds to `pattern_k19_support` (187) or `pattern_k17_support` (~611).
- **`extract_items_max` = 46 vs `dataset_max_features_per_protein` = 22.** Kept as the instructed new key (Step 2 "Max: 46" is over the 205.6M-row parquet with the 1,006 defined items, the paper's 22 is over the 1,002 frequent items of the mined set); the description cross-references the paper key. Likewise `extract_items_mean` (2.2) is new, cross-referenced to `mem_items_per_txn_214m` (~10).
- **Row counts of plddt_metadata.csv.** "Read 214683829 rows" (L2-316/538) → `dataset_metadata_rows` (the paper's 214,683,829 expectation, as instructed); the pre-flight "pLDDT: 214683830 rows" (L2-287/509, wc -l incl. header) → new `extract_csv_rows`; the banner "214M" (L2-284/506) → `dataset_metadata_rows` (listed there as a rounded variant).
- **File size of uniprot_trembl.dat.gz.** "TrEMBL: 150G" (L2-286/508) and "aria2c FINISHED! File size: 150G" (L2-504) → existing `dataset_annotation_gb` (150); the aria2c-reported "149GiB" (L2-497) → new `download_size_gib` (as instructed; same file, GiB vs du-style rounding).
- **Pipeline start time** appears three times (pipeline banner L2-285, watcher hand-off L2-505, watcher banner copy L2-507) → one key `meta_log_pipeline_start_timestamp`.
- **Extraction time.** af-extract's self-timed "3777.0s" (L2-333/555) → new `extract_time_s` (paper `extract_time_min` = 63 kept separate as instructed); bash `time` real/user/sys (L2-335..337/557..559) → `extract_wall_real_s` / `extract_wall_user_s` / `extract_wall_sys_s` with the verbatim `65m31.040s`-style values.
- **"RELEASING THE H100s..."** (L2-469, value "H100s (plural)") → `hw_gpu_model` (it names the GPU model); the plural is only recorded in the row value, not keyed to `hw_gpu_count`.
- **madman_mining.log** (aborted SON at the Blitz threshold) gets its own `run_madman_*` keys rather than the Blitz keys, because it is a different (unfinished) run.
- **Two watcher rows describe the same aria2c line from different angles**: L2-498/499 (first/last snapshot strings) → `download_progress_first` / `download_progress_last`; L2-501 (rate range 26–100 MiB/s) → `download_speed_mib_s`; L2-502 (`[145G]`→`[150G]` bracket) → `download_disk_usage_bracket`; L2-503 (45 / 25 snapshots) → `download_snapshot_count`; L2-500 (CN:16) → `download_connections`.

## Keys deliberately not created

`kcand_*`, `ktime_*` (no claim rows; candidates/ms live inside the per-K contexts); `run_<run>_n_transactions` (all transaction counts keyed to `dataset_multi_feature` / `dataset_plddt_pass`); `run_<run>_kmax` for runs without a K_max row; `extract_dat_parse_s`, `extract_pipeline_total_s`, `extract_wall_unaccounted_s` (Section B arithmetic, no rows); `run_power_itemsets` / `run_power_time_min` for extreme (the `run_power_son_*` aliases were used, as instructed); `extract_rules_toplift_1_lift` (= `extract_rules_max_lift`) and `extract_rules_topconf_1_*` (= `extract_rules_toplift_3_*`).
