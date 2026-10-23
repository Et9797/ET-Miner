# Phase 4 (logs) — keying notes for `claims_logs_keyed.csv`

Inputs: `phase1/claims_logs.md` Section A (948 rows L-001..L-948); vocabulary = `phase4/quantities.csv` (336 paper keys) + `phase4/new_quantities_logs_new.csv` (347 keys of the earlier log-claims keying) + `phase4/quantities_notes.md` (conventions). Generator: `<scratchpad>/build_phase4_logs_v2.py` (+ `notes_logs_v2.py`) — an explicit one-line-per-row mapping (`MAP_TXT`) for every non-bench row; the synthetic-benchmark rows (report.md / raw.jsonl of the two 2x RTX 3090 campaigns) are so regular that their run id and metric are parsed from the row and asserted against a fixed run table, then materialised one row at a time. Keys are assigned only; nothing here judges correctness.

## Outputs and validation (printed by the generator)

- `claims_logs_keyed.csv`: 948 rows, columns `claim_id,qkey,value,unit,category,file,line,quoted_context`; every L ID exactly once (True); every qkey non-empty and present in `quantities.csv`, `new_quantities_logs_new.csv` or `new_quantities_logs.csv` (True); re-parses with Python `csv` at 8 columns.
- `new_quantities_logs.csv`: 334 new qkeys, same 7 columns as `quantities.csv`; `claimed_value_tex = "log:<distinct row values in document order, ' / '-joined>"`; `tex_lines = file:line` (semicolon-separated, de-duplicated; line cells such as `(file)`, `(derived)`, `cell 3 (md)` are kept as written); `category` = majority category of the mapped rows (tie -> first row), as in `quantities_notes.md` convention 3; `reproducibility` assigned per key. Re-parses at 7 columns; 0 name collisions with the 336 + 347 existing keys; every new key is used.
- Rows keyed to EXISTING qkeys: 104 — 93 to 56 distinct paper keys (`quantities.csv`), 11 to 6 distinct keys of `new_quantities_logs_new.csv`; rows keyed to NEW qkeys: 844 (334 keys).
- New keys by prefix: `bench_` 149, `cost_` 2, `doc_` 53, `ext_` 1, `hw_` 12, `kdist_` 23, `meta_` 19, `null_` 40, `pattern_` 5, `run_` 21, `sw_` 9.
- New keys by reproducibility: derived 25, external 9, hardware 109, rerun 94, setup 97.
- Verbatim copying: `value`, `unit`, `category`, `line`, `quoted_context` are copied as they stand in the markdown table (the markdown escape `\|` is unescaped to `|`, as `claims_tex_keyed.csv` did); the only normalisation is that the code-formatting backticks around the `file` cell are removed so the column is a plain repo-relative path (same as `claims_logs_new_keyed.csv`).

### Rows per source file (existing paper key / existing logs_new key / new key)

| file | rows | paper | logs_new | new |
|---|---|---|---|---|
| `bench/results/2026-08-31-3090x2/env.txt` | 11 | 0 | 0 | 11 |
| `bench/results/2026-08-31-3090x2/report.md` | 146 | 0 | 0 | 146 |
| `bench/results/2026-08-31-3090x2/FINDINGS.md` | 25 | 0 | 0 | 25 |
| `bench/results/2026-08-31-3090x2/raw.jsonl` | 253 | 0 | 0 | 253 |
| `bench/results/2026-09-01-3090x2-sparse/env.txt` | 8 | 0 | 0 | 8 |
| `bench/results/2026-09-01-3090x2-sparse/report.md` | 62 | 0 | 0 | 62 |
| `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 28 | 0 | 0 | 28 |
| `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 118 | 0 | 0 | 118 |
| `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 41 | 18 | 1 | 22 |
| `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 60 | 22 | 0 | 38 |
| `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 25 | 3 | 10 | 12 |
| `applications/alphafold/results_214m/GLOSSARY.md` | 10 | 3 | 0 | 7 |
| `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | 56 | 27 | 0 | 29 |
| `applications/alphafold/deploy/RUNBOOK_base214m.md` | 18 | 5 | 0 | 13 |
| `PROGRESS.md` | 10 | 0 | 0 | 10 |
| `applications/alphafold/deploy/run_all_experiments.sh` | 13 | 1 | 0 | 12 |
| `applications/alphafold/deploy/run_null_model_35k.sh` | 4 | 0 | 0 | 4 |
| `applications/alphafold/deploy/deploy_project_milky_way.sh` | 5 | 0 | 0 | 5 |
| `applications/alphafold/deploy/deploy_base214m.sh` | 4 | 1 | 0 | 3 |
| `applications/alphafold/pipeline/postprocess_tx.py` | 2 | 0 | 0 | 2 |
| `applications/alphafold/pipeline/pipeline_214m.py` | 1 | 0 | 0 | 1 |
| `applications/alphafold/pipeline/extract_features.py` | 2 | 1 | 0 | 1 |
| `applications/alphafold/experiments/experiment_direct_vs_son.py` | 4 | 0 | 0 | 4 |
| `applications/alphafold/experiments/experiment_full_campaign.py` | 2 | 1 | 0 | 1 |
| `applications/alphafold/pipeline/run_mining.py` | 3 | 0 | 0 | 3 |
| `applications/alphafold/experiments/analyze_k22_proteins.py` | 1 | 1 | 0 | 0 |
| `README.md` | 30 | 10 | 0 | 20 |
| `bench/README.md` | 6 | 0 | 0 | 6 |

### Existing keys used

- Paper keys (rows): `dataset_multi_feature` 7, `dataset_metadata_rows` 4, `run_opus_itemsets` 4, `run_power_direct_itemsets` 4, `null_kge7_bio` 3, `run_opus_time_min` 3, `vocab_items_defined` 3, `kdist_opus_peak_count` 2, `null_k2_z` 2, `null_k4_z` 2, `null_k5_z` 2, `null_k6_z` 2, `null_min_count` 2, `pattern_k19_support` 2, `run_blitz_itemsets` 2, `run_opus_kmax` 2, `run_power_direct_time_s` 2, `run_power_min_count` 2, `run_power_son_itemsets` 2, `run_power_son_time_s` 2, `run_power_support_pct` 2, `son_miss_rate_pct` 2, `vocab_plddt_bin_medium_edges` 2, `alg_min_count_rule` 1, `alg_pcie_bytes_total` 1, `campaign_support_span_orders` 1, `csr_bytes_gb` 1, `hw_gpu_model` 1, `k22_support` 1, `null_k1_bio` 1, `null_k1_mean` 1, `null_k1_z` 1, `null_k2_bio` 1, `null_k3_bio` 1, `null_k3_z` 1, `null_k4_bio` 1, `null_k5_bio` 1, `null_k6_bio` 1, `null_kge7_mean` 1, `null_kmax` 1, `null_per_perm_time_s` 1, `null_permutations` 1, `null_seed` 1, `null_total_time_s` 1, `pattern_k17_support` 1, `run_blitz_kmax` 1, `run_blitz_method` 1, `run_blitz_support_pct` 1, `run_opus_support_pct` 1, `run_power_direct_kmax` 1, `run_power_kmax` 1, `run_power_method` 1, `son_speedup` 1, `vocab_items_frequent` 1, `vocab_pfam_defined` 1, `vocab_plddt_defined` 1.
- `new_quantities_logs_new.csv` keys (rows): `kdist_blitz_k15_count` 2, `kdist_blitz_k16_count` 2, `kdist_blitz_k17_count` 2, `kdist_blitz_k18_count` 2, `kdist_blitz_k19_count` 2, `run_blitz_time_s` 1.

## The 20 most-used keys

| rank | rows | qkey | origin |
|---|---|---|---|
| 1 | 26 | `bench_deep_k_exact_n_itemsets` | new |
| 2 | 26 | `bench_deep_k_exact_sum_counts` | new |
| 3 | 26 | `bench_deep_k_exact_itemset_hash` | new |
| 4 | 20 | `bench_deep_k_density_auto_level_ms` | new |
| 5 | 17 | `bench_deep_k_shared_2g_levels` | new |
| 6 | 17 | `bench_deep_k_shared_2g_level_ms` | new |
| 7 | 14 | `meta_log_file_provenance` | new |
| 8 | 13 | `bench_deep_k_density_auto_levels` | new |
| 9 | 11 | `bench_deep_k_density_auto_wall_s` | new |
| 10 | 11 | `bench_deep_k_shared_2g_wall_s` | new |
| 11 | 11 | `bench_skewed_rows_nnz_levels` | new |
| 12 | 11 | `bench_skewed_rows_nnz_level_ms` | new |
| 13 | 11 | `bench_smoke_twophase_levels` | new |
| 14 | 11 | `bench_smoke_twophase_level_ms` | new |
| 15 | 10 | `bench_deep_k_shared_1g_wall_s` | new |
| 16 | 9 | `hw_bench_gpu_model` | new |
| 17 | 8 | `bench_deep_k_nonccl_wall_s` | new |
| 18 | 8 | `bench_deep_k_shared_1g_peak_vram_mb` | new |
| 19 | 8 | `bench_deep_k_shared_1g_params` | new |
| 20 | 8 | `bench_deep_k_shared_2g_peak_vram_mb` | new |

## Conventions used for the new keys

1. **Naming patterns exactly as specified**: `run_<run>_<metric>`, `kdist_<run>_k<K>_count|pct`, `null_<metric>` / `null_<variant>_<metric>`, `hw_*`, `sw_*`, `cost_*`, `bench_<preset>_<config>_<metric>`, `doc_<what>`, `meta_log_<what>`. Three existing paper families are *extended* rather than duplicated, because the log values are members of those families: `pattern_k19_item_ids|plddt_member|pfam_members|go_members` and `pattern_k18_support` (decoded-itemset members / supports), `null_k7..14_bio|z` (per-K rows of the paper's aggregated K=7-14 null row) and `ext_afdb_v4_corpus_size` (an external AlphaFold-DB fact next to `ext_afdb_homodimers_*`). No `misc_*` key was needed.
2. **Existing key wins** (rule 1), whichever vocabulary defines it: `run_blitz_time_s` (119.3 s) and `kdist_blitz_k15..k19_count` (4,155 / 1,003 / 173 / 19 / 1) already exist in `new_quantities_logs_new.csv` and are reused for the Direct-vs-SON JSON's Blitz block and the decoded-dump section headers / SUMMARY rows; the brief's suggested `run_blitz_k_ge15_itemsets` (5,351) and `kdist_power_direct_k1..14_count` were created because absent. Fraction-valued supports follow the log vocabulary's precedent (`run_blitz_support` next to `run_blitz_support_pct`): new `run_power_support` (1e-05; also the `--min-support` default of experiment_direct_vs_son.py) and `null_support` (1.0001177641918693e-05 = 769 / 76,890,945, the count-derived threshold of the null JSON); percent-phrased rows stay on `run_power_support_pct` / `run_blitz_support_pct` / `null_support_pct`.
3. **Bench (synthetic 2x RTX 3090 campaign, NOT AlphaFold)**. Run ids map to `<preset>_<config>`: `deepk-*` -> `deep_k_<rest>`, `skew-nnz|rows` -> `skewed_rows_nnz|rows`, `stressk2-*` -> `stress_k2_<rest>`, `twophase-smoke` -> `smoke_twophase`. Per-config keys: `_params` (setup), `_wall_s`, `_peak_vram_mb`, `_throttle_flags`, `_level_ms` (hardware), `_levels` (per-K candidates/frequent, rerun). Report.md **median and min wall** rows and raw.jsonl `wall_s` (one per rep) all share `_wall_s` (summary statistics of one measured quantity; for reps=1 they are the same number rounded); report.md per-level count / ms rows share the `_levels` / `_level_ms` key of the run the table describes (deepk-density-auto#r0, skew-nnz#r1, twophase-smoke#r0, stressk2-legacy-1g#r0 on 2026-08-31; deepk-shared-2g#r2 on 2026-09-01). The two campaigns ran different commits (5a8e59f8 vs b8a5032b) but share keys; `tex_lines` / the `file` column disambiguate (this is why `bench_deep_k_density_auto_wall_s` holds both 10.431 s and 1.343 s — the sparse path moved from a host tidset rebuild to the GPU between the campaigns; recorded, not judged).
4. **Bench deterministic signatures** (`n_itemsets`, `sum_counts`, `itemset_hash`) are keyed per *equivalence class* in the `<config>` slot, not per run, because they are the same quantity for every exact config (that is what the tier-equivalence chain asserts): `bench_deep_k_exact_*` (all 13 deep_k configs, both campaigns), `bench_skewed_rows_exact_*`, `bench_smoke_twophase_*` (the only smoke run), `bench_stress_k2_ml2_*` (max_length=2 filter-impl runs), `bench_stress_k2_ml3_*` (exact max_length=3: legacy-2g, shared-1g, shared-2g) and `bench_stress_k2_legacy_1g_*` (the lossy sampled-prefilter run: 2,996,485 vs 3,005,770). Reproducibility: hardware for timings / VRAM / throttle flags, rerun for itemset counts / signatures, as specified; ratios and differences derived.
5. **AlphaFold JSONs**. `kdist_power_direct_k1..14_count` for the Direct-vs-SON k_distribution (per the brief); the null JSON's `real_distribution` goes to `null_k<K>_bio` (existing for K=1..6, new for K=7..14) — the two families are deliberate aliases of one run (the null JSON's `recomputed_real` is false), documented in both descriptions like `run_power_itemsets` = `run_power_son_itemsets` in the paper vocabulary. The per-K `statistics` blocks bundle real/mean/std/z/p in one row; each is keyed to `null_k<K>_z` (the derived statistic the headline claim rests on; bio is keyed separately, mean/std are recoverable from the per-permutation rows). The five per-permutation records get per-run keys `null_perm<i>_total_itemsets`, `null_perm<i>_time_s`, `null_perm<i>_k_distribution` (i = 1..5): with seed 42 each permutation is individually reproducible, so each is a distinct quantity. Summary scalars: `null_mean_total` (171,320.0) and `null_ratio` (2.78) are new; 662.17 s -> existing `null_total_time_s`; **132.43 s -> existing `null_per_perm_time_s`** (the paper's "~130 s = 662/5" is the same quantity), so the brief's suggested `null_avg_perm_time_s` was deliberately NOT created (rule 1 / rule 3).
6. **Planned-but-not-executed experiments** (runbook / deploy scripts: 4x H200, 100-permutation nulls, 35K-feature campaign) are kept out of the executed-run keys: `null_minc769_planned_*`, `null_minc8_planned_*` (variants of the paper null), `run_v35k_*` / `kdist_v35k_*` / `run_v35k_null_*` (the 35K campaign and its null model, named after the brief's run list `v35k`, `v35k_null`), `hw_base214m_planned_gpus`, `cost_*`, `doc_planned_*`, `doc_runbook_*`. `PROGRESS.md` describes the audit box itself -> `hw_audit_box_*`, `sw_audit_box_*`, `meta_log_audit_*`; the bench box (driver 580.159.03, possibly a different vast.ai instance) -> `hw_bench_*`, `sw_bench_*`, `meta_log_bench_*`.
7. **README CPU benchmarks** (billion-scale streaming, efficient-apriori tables) are neither the AlphaFold run nor the GPU bench campaign -> `doc_readme_cpu_stream1b_*`, `doc_readme_ea_*` (one key per table row). GLOSSARY background biology -> `doc_glossary_*` (reproducibility external); code defaults of pipeline scripts -> `doc_pipeline_*`; bench/README expectations -> `doc_bench_readme_*`.
8. **Same-quantity re-use (rule 3)**: rounding / unit / phrasing variants were keyed to the existing key, never to a new one — e.g. 2.84M -> `run_blitz_itemsets`; 26.8M / "26.8 million" -> `run_opus_itemsets`; 3.53M -> `kdist_opus_peak_count`; "over 200 million" / 214M / "214 million" -> `dataset_metadata_rows`; 76,890,945 / 76.9M -> `dataset_multi_feature`; ~264 bytes -> `alg_pcie_bytes_total`; `min_count = ceil(...)` -> `alg_min_count_rule`; "support >= 1e-7" -> `run_opus_support_pct`; 0.1% -> 0.00001% -> `campaign_support_span_orders`; "K=22 protein identification" -> `run_opus_kmax`.

## Rows keyed to existing paper keys that the join should look at first

- Direct-vs-SON JSON: L-666 / L-696 / L-720 / L-750 -> `run_power_direct_itemsets` (475,865); L-667 -> `run_power_direct_time_s` (50.72); L-668 -> `run_power_direct_kmax` (14); L-685 -> `run_power_son_itemsets`; L-686 -> `run_power_son_time_s`; L-687 -> `run_power_kmax` (13, the SON max K; no `run_power_son_kmax`, per the paper notes); L-693 -> `son_speedup` (21.4); L-663 -> `run_power_min_count` (768) and L-697 (extractor: 768.9 -> ceil 769) also -> `run_power_min_count`, so the 768-vs-769 tension stays inside that key; L-701 / L-756 -> `null_min_count` (769); L-661 / L-684 -> `run_power_support_pct`; L-683 / L-688 -> `run_power_method` / `run_blitz_method`.
- Null JSON: L-703 `null_permutations`, L-704 `null_seed`, L-754 `null_total_time_s` (662.17), L-753 `null_per_perm_time_s` (132.43), L-755 `null_kge7_bio` (88,745 — vs the notebook's 89,566 at L-822 / L-838, all three keyed to `null_kge7_bio` so the discrepancy is visible in one key), L-736..L-741 `null_k1..6_z`, L-706..L-711 `null_k1..6_bio`, L-702 `dataset_multi_feature`.
- Blitz / decoded dump: L-690 `run_blitz_itemsets`, L-692 `run_blitz_kmax`, L-689 `run_blitz_support_pct`, L-691 `run_blitz_time_s` (logs_new), L-762 / L-768 / L-770 / L-772 / L-773 and L-774..L-778 `kdist_blitz_k19..15_count` (logs_new), L-763 / L-840 `pattern_k19_support` (187), L-771 `pattern_k17_support` (~611), L-760 `dataset_multi_feature`.
- Opus narrative (notebook / README): `run_opus_itemsets`, `run_opus_kmax`, `run_opus_time_min`, `run_opus_support_pct` (L-804 "support >= 1e-7"), `kdist_opus_peak_count`, `k22_support`, `null_kmax`, `null_kge7_mean`, `null_k1_mean` (L-817 "1,002/1,002 identical by construction"), `son_miss_rate_pct` (L-811 / L-815).
- Vocabulary / dataset: `vocab_items_defined` (1,006: L-800, L-862, L-897), `vocab_items_frequent` (L-921), `vocab_pfam_defined` (L-863), `vocab_plddt_defined` (L-859), `vocab_plddt_bin_medium_edges` (L-902 code edges `< 50 / <= 90` and L-912 label "pLDDT 70-90" — keyed together so the edge discrepancy is visible), `dataset_multi_feature` (76,890,945 in both JSONs, the decoded dump, the notebook, README, runbook), `dataset_metadata_rows` (214M / "over 200 million"), `csr_bytes_gb` (L-919).
- Hardware: L-795 (notebook "NVIDIA H100/H200") -> `hw_gpu_model`, so the H100-vs-H200 wording sits under the paper's key; every 2x RTX 3090 row goes to `hw_bench_gpu_model` / `hw_audit_box_gpu` instead.

## Ambiguous rows and the choice made

- **Composite rows (one row, several quantities)** — the key is the row's leading / most specific quantity; the others are named in the key description:
  - L-736..L-749 statistics blocks -> `null_k<K>_z` (see convention 5).
  - L-812 ("SON 22,846 vs direct 475,865; max K 13 vs 14") -> `run_power_son_itemsets`; L-813 ("475,865 itemsets in 50.7 s", category hardware-dependent) -> `run_power_direct_time_s`; L-814 -> `run_power_son_time_s`; L-839 ("20.8x more itemsets ... 21.4x less time") -> new `run_power_direct_vs_son_itemset_ratio` (20.8x is stated nowhere else; 21.4x already sits under `son_speedup` via L-693 / L-814).
  - L-917 (README headline sentence) -> `run_opus_itemsets`; L-836 (same sentence in the notebook, category hardware-dependent) -> `run_opus_time_min`; L-804 (God Mode params "76.9M proteins, 1,002 features (500 Pfam + 500 GO + 6 pLDDT), support >= 1e-7") -> `run_opus_support_pct` (note the row's own 500+500+6 = 1,006 != 1,002).
  - L-803 ("K=22: 1 itemset, ~8 proteins") -> `k22_support`; L-840 ("K=19 in ~187 proteins; K=22 in ~8") -> `pattern_k19_support`.
  - L-791 ("76.9M--109M proteins"), L-864 ("v1: 76.9M of 205.6M") and L-920 ("214M total, 76.9M with multiple annotations") -> `dataset_multi_feature`; L-794 ("76.9M (1K) / 109.2M (35K)") -> new `run_v35k_n_transactions`; L-823 / L-797 / L-834 (16.8B with 109.2M proteins / 35,012 features / K=8 details) -> `run_v35k_itemsets`; L-841 -> `kdist_v35k_k8_pct` (71.8% is the only number in that row not keyed elsewhere); L-893 (35K mining args) -> `run_v35k_max_length` (the K<=8 cap is the run-specific fact; support 0.00001 / 8 GPUs are noted in the description).
  - L-919 ("~5 GB CSR; ~26 GB bitvectors") -> `csr_bytes_gb` (bitvec_gb is the second number); L-925 ("0.1% -> 0.00001%") -> `campaign_support_span_orders`.
  - L-862 ("1006 defined / 1002 frequent") -> `vocab_items_defined`; L-863 ("--top-pfam 500 --top-go 500") -> `vocab_pfam_defined`.
  - L-818 ("Z = -987, -143") -> `null_k2_z` (first value); L-821 ("Z = 71,728; null 21.8 vs real 78,596") -> `null_k6_z`.
  - L-783 ("K>=10; 2.84M") -> new `doc_glossary_high_k_threshold` (K>=10 is the new information; 2.84M is `run_blitz_itemsets`, also carried by L-789).
  - L-869 ("4xH200; ~85M multi-feature") -> new `doc_runbook_expected_multi_feature` (the ~85M is the new number; 4xH200 is `hw_base214m_planned_gpus` via L-856 / L-872 / L-894); L-867 (runbook suite line) and L-873 / L-907 -> `doc_planned_campaign_design`; L-871 / L-878 (time + money) -> `cost_base214m_planned_total` / `cost_h200_planned_rate`.
  - L-885 ("5 permutations; 0.001% (min_count=1092)") and L-892 ("--runs 100 --min-count 1090 --n-gpus 8 --seed 42") -> `run_v35k_null_min_count` (the 1092-vs-1090 discrepancy is then inside one key).
  - L-757 / L-758 (whole-file extractor summaries of the decoded dump) -> `meta_log_decoded_patterns_file_summary` / `run_blitz_k_ge15_distinct_items`; L-774..L-778 SUMMARY rows (count + support range + protein range per K) -> `kdist_blitz_k<K>_count`; L-779 (extractor sum 5,351) -> `run_blitz_k_ge15_itemsets`; L-780 (84 proteins at K=15) -> new `run_blitz_k15_min_proteins`.
  - L-196 ("580.159.03 / CUDA 13.0") -> `sw_bench_driver_version`; L-172 ("580.159.03 / CuPy 14.1.1") -> `sw_bench_cupy_version` (CuPy is the new information); L-171 / L-263 / L-170 / L-262 (FINDINGS headers with date, driver, CuPy) and L-943 (bench/README "2x RTX 3090, 24 GB, CUDA 12" design target) -> `hw_bench_gpu_model`.
  - L-175 ("8.8x / 16.3x") -> `bench_stress_k2_shared_vs_legacy_1g_speedup` (first number; L-178 / L-179 carry the two separately); L-181 ("0.97-1.07x neutral on deep_k/smoke") -> `bench_deep_k_shared_vs_legacy_2g_speedup`; L-189 ("89.6 / 89.7 / 90.2 s filter compact/cupy/cpu") -> `bench_stress_k2_filter_compact_wall_s`; L-192 / L-288 ("0.9 s vs 1.3 s NCCL fallback vs NCCL") -> `bench_deep_k_nonccl_wall_s`; L-193 ("1.5 vs 1.4 s nnz vs rows") -> `bench_skewed_rows_nnz_wall_s`; L-185 / L-270 / L-273 (density-auto 10.4 s vs dense 1.3 s / 1.343 s) -> `bench_deep_k_density_auto_wall_s`; L-274 -> `bench_deep_k_density_auto_1g_wall_s`; L-276..L-282 (per-K rows comparing 08-31 / 09-01 / dense) -> `bench_deep_k_density_auto_level_ms`; L-284 ("406/408 MB (dense 414/416)") -> `bench_deep_k_density_auto_peak_vram_mb`; L-286 -> `bench_deep_k_density_auto_1g_level_ms`; L-177 ("612M K=2; ~76 billion K=3 candidates") -> new `bench_stress_k2_ml3_candidates_total`.
- **Whole-file provenance rows** L-001..L-014 (mtime / size / git-add commit; the mtime is the checkout time for all) -> a single `meta_log_file_provenance` key; `tex_lines` and the `file` column identify the artefact (one key, 14 rows, like the progress-series keys of the log vocabulary).
- **Report median vs min** (e.g. L-026 / L-027) -> same `_wall_s` key (summary statistics of the same measured quantity), see convention 3.
- **L-266 ("13 deep_k configs")** -> `bench_campaign_0901_runs_status` (same count as "13 ok / 0 failed"); **L-264 (9/9)** shares `bench_tier_equivalence_legs` with L-173 (7/7): the chain grew by two legs between campaigns, not a conflict.
- **L-186 / L-915 (n/32 crossover)** -> one software-constant key `sw_sparse_crossover_rule` (FINDINGS and README state the same rule).
- **Script defaults vs run parameters**: L-904 (`--min-support` default 0.00001 of experiment_direct_vs_son.py) -> `run_power_support` (it is the value the surviving JSON was produced with); L-905 / L-906 -> new `run_power_son_chunk_size` / `run_power_son_local_support_factor` (SON parameters of that experiment); but L-903 / L-874 (`--runs 3`) -> `doc_direct_vs_son_planned_runs` because the surviving JSON holds a single result, and L-909..L-911 (run_mining.py defaults 0.01 / 4 / 5, not used by any surviving run) -> `doc_pipeline_run_mining_*`; L-908 -> `alg_min_count_rule`.
- **L-902 / L-912 (pLDDT bin edges in code vs label)** -> `vocab_plddt_bin_medium_edges`; L-901 (frac_high bins, a different feature) -> `doc_pipeline_frac_high_bins`.
- **L-877 ("K=22 protein identification" planned step)** -> `run_opus_kmax` (the K of the deepest itemset), not a doc_ key.
- **L-853 / L-858 / L-896 (23 TiB / 23TB CIF corpus)** -> one key `ext_afdb_v4_corpus_size` (external fact stated three times; `ext_` because the paper vocabulary keys external facts that way).
- **L-816 (Z=3.29 line)** -> `null_significance_line_z`, a plotting constant kept out of the result keys.
- **L-847 (PROGRESS.md "2 x RTX 3090 ... compute cap 8.6")** -> `hw_audit_box_gpu`, NOT `hw_bench_gpu_model`: the audit box runs driver 595.71.05 while the bench env.txt recorded 580.159.03, so the two are kept apart until proven the same machine.

## Cross-file observations surfaced by the keying (recorded, not judged)

- `run_power_min_count`: 768 (Direct-vs-SON JSON) vs `null_min_count`: 769 (null JSON, whose `null_support` is exactly 769/76,890,945) — same threshold, two files, as already flagged for the paper keys.
- `null_kge7_bio`: 88,745 (JSON sum K=7..14, L-755) vs 89,566 (notebook prose, L-822 / L-838).
- `bench_deep_k_density_auto_wall_s`: 10.431 s (2026-08-31, host tidset rebuild) vs 1.343 s (2026-09-01, GPU-resident sparse) — different commits, both under one key by design.
- `run_v35k_null_min_count`: 1092 vs 1090 across the two 35K scripts.
- `hw_gpu_model`: notebook says "NVIDIA H100/H200" where the paper says a single H100; every surviving *measured* log on this box is 2x RTX 3090 (`hw_bench_gpu_model`, `hw_audit_box_gpu`).
- `vocab_plddt_bin_medium_edges`: code edges `mean_plddt < 50 / <= 90` (L-902) vs label "pLDDT 70-90" (L-912) vs paper 70-90.
- `bench_tier_equivalence_legs`: 7/7 (2026-08-31) vs 9/9 (2026-09-01) — two sparse-CSR legs added, not a conflict.

