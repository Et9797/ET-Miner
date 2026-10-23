# CPU_COMPARISON — CPU in-core / SON vs the GPU run

Built by `scripts/build_comparison.py` from the artifacts named in each row (GPU values read from the GPU run's artifacts under `runs/20260902T0000Z/`, ids from the repo-root RESULTS.md).

| row | what | GPU id | GPU value | CPU value | artifact | verdict |
|---|---|---|---|---|---|---|
| CC-001 | pinned engine | — | — | 37819e2f89c69c1e5bf0c3eb06f3142bfbcc7f68 | env.json | recorded |
| CC-002 | data M-001 | M-001 | 214683829 | 214683829 | data/phase1_verification.json | identical |
| CC-003 | data M-002 | M-002 | 214683829 | 214683829 | data/phase1_verification.json | identical |
| CC-004 | data M-003 | M-003 | 205620298 | 205620298 | data/phase1_verification.json | identical |
| CC-005 | data M-004 | M-004 | 0 | 0 | data/phase1_verification.json | identical |
| CC-006 | data M-005 | M-005 | [20.03, 98.81] | [20.03, 98.81] | data/phase1_verification.json | identical |
| CC-007 | data M-006 | M-006 | [214683829, 'uniprotAccession,globalMetricValue'] | [214683829, 'uniprotAccession,globalMetricValue'] | data/phase1_verification.json | identical |
| CC-008 | data M-007 | M-007 | [205620298, 9063531] | [205620298, 9063531] | data/phase1_verification.json | identical |
| CC-009 | data M-008 | M-008 | 80002801 | 80002801 | data/phase1_verification.json | identical |
| CC-010 | data M-009 | M-009 | 125617497 | 125617497 | data/phase1_verification.json | identical |
| CC-011 | data U-007 | U-007 | 160834306675 | 160834306675 | data/phase1_verification.json | identical |
| CC-012 | data U-009 | U-009 | [160834306675, 161890570993] | [160834306675, 161890570993] | data/phase1_verification.json | identical |
| CC-013 | data X-016 | X-016 | [202556314, 25475, 26536, 232694948, 436931915] | [202556314, 25475, 26536, 232694948, 436931915] | data/phase1_verification.json | identical |
| CC-014 | data X-001 | X-001 | 202556314 | 202556314 | data/phase1_verification.json | identical |
| CC-015 | data X-002 | X-002 | [25475, 26536] | [25475, 26536] | data/phase1_verification.json | identical |
| CC-016 | data X-005 | X-005 | [158700122, 158700122, 202556314] | [158700122, 158700122, 202556314] | data/phase1_verification.json | identical |
| CC-017 | data X-006 | X-006 | 90506840 | 90506840 | data/phase1_verification.json | identical |
| CC-018 | data X-007 | X-007 | 90506840 | 90506840 | data/phase1_verification.json | identical |
| CC-019 | data X-008 | X-008 | [24291, 25993, 1006] | [24291, 25993, 1006] | data/phase1_verification.json | identical |
| CC-020 | data X-009 | X-009 | 90506840 | 90506840 | data/phase1_verification.json | identical |
| CC-021 | data shards | — | [True] | [True] | data/phase1_verification.json | identical |
| CC-022 | data stats.items_per_category | — | {'plddt_mean': 3, 'pfam': 500, 'go_term': 500, 'plddt_fraction': 3} | {'plddt_mean': 3, 'plddt_fraction': 3, 'pfam': 500, 'go_term': 500} | data/phase1_verification.json | identical |
| CC-023 | data stats.items_per_txn_histogram | — | {'1': 128729353, '2': 16588361, '3': 17840950, '4': 16007527, '5': 10554413, '6': 6801712, '7': 3740314, '8': 3484528, '9': 1221451, '10': 354125, '11': 143957, '12': 102523, '13': 26944, '14': 17763, '15': 2926, '16': 913, '17': 1607, '18': 325, '19': 392, '20': 141, '21': 41, '22': 13, '23': 6, '24': 4, '25': 2, '26': 1, '27': 1, '28': 3, '39': 1, '46': 1} | {'1': 128729353, '2': 16588361, '3': 17840950, '4': 16007527, '5': 10554413, '6': 6801712, '7': 3740314, '8': 3484528, '9': 1221451, '10': 354125, '11': 143957, '12': 102523, '13': 26944, '14': 17763, '15': 2926, '16': 913, '17': 1607, '18': 325, '19': 392, '20': 141, '21': 41, '22': 13, '23': 6, '24': 4, '25': 2, '26': 1, '27': 1, '28': 3, '39': 1, '46': 1} | data/phase1_verification.json | identical |
| CC-024 | data stats.items_per_txn_max | — | 46 | 46 | data/phase1_verification.json | identical |
| CC-025 | data stats.items_per_txn_mean_all | — | 2.164914895707427 | 2.164914895707427 | data/phase1_verification.json | identical |
| CC-026 | data stats.items_per_txn_mean_multi | — | 4.115193186922075 | 4.115193186922075 | data/phase1_verification.json | identical |
| CC-027 | data stats.n_csv_pass_full | — | 205620298 | 205620298 | data/phase1_verification.json | identical |
| CC-028 | data stats.n_csv_rows_full | — | 214683829 | 214683829 | data/phase1_verification.json | identical |
| CC-029 | data stats.n_csv_skipped_full | — | 9063531 | 9063531 | data/phase1_verification.json | identical |
| CC-030 | data stats.n_items_defined | — | 1006 | 1006 | data/phase1_verification.json | identical |
| CC-031 | data stats.n_items_present_all | — | 1002 | 1002 | data/phase1_verification.json | identical |
| CC-032 | data stats.n_items_present_multi | — | 1002 | 1002 | data/phase1_verification.json | identical |
| CC-033 | data stats.n_items_with_support_ge8_multi | — | 1002 | 1002 | data/phase1_verification.json | identical |
| CC-034 | data stats.n_multi_feature | — | 76890945 | 76890945 | data/phase1_verification.json | identical |
| CC-035 | data stats.n_multi_written | — | 76890945 | 76890945 | data/phase1_verification.json | identical |
| CC-036 | data stats.n_outside_run_single_item | — | 115113458 | 115113458 | data/phase1_verification.json | identical |
| CC-037 | data stats.n_run_rows | — | 90506840 | 90506840 | data/phase1_verification.json | identical |
| CC-038 | data stats.n_single_feature | — | 128729353 | 128729353 | data/phase1_verification.json | identical |
| CC-039 | data stats.n_transactions_total | — | 205620298 | 205620298 | data/phase1_verification.json | identical |
| CC-040 | data stats.n_zero_feature | — | 0 | 0 | data/phase1_verification.json | identical |
| CC-041 | data stats.nnz_all | — | 445150446 | 445150446 | data/phase1_verification.json | identical |
| CC-042 | data stats.nnz_multi | — | 316421093 | 316421093 | data/phase1_verification.json | identical |
| CC-043 | data stats.pct_multi_feature | — | 37.39462774244204 | 37.39462774244204 | data/phase1_verification.json | identical |
| CC-044 | data stats.pct_single_feature | — | 62.60537225755796 | 62.60537225755796 | data/phase1_verification.json | identical |
| CC-045 | data stats.reconstruction | — | proteins with pLDDT >= min but without any Pfam/GO DAT record are single-item transactions by construction; totals below include them | proteins with pLDDT >= min but without any Pfam/GO DAT record are single-item transactions by construction; totals below include them | data/phase1_verification.json | identical |
| CC-046 | data X-011 | X-011 | 76890945 | 76890945 | data/phase1_verification.json | identical |
| CC-047 | data X-014 | X-014 | [316421093, 445150446] | [316421093, 445150446] | data/phase1_verification.json | identical |
| CC-048 | data X-015 | X-015 | [1006, 1002] | [1006, 1002] | data/phase1_verification.json | identical |
| CC-049 | data mapping feature set | — | 0 | 0 | data/phase1_verification.json | identical |
| CC-050 | data item_support_all | — | [1002, 0] | [1002, 0] | data/phase1_verification.json | identical |
| CC-051 | data item_support_multi | — | [1002, 0] | [1002, 0] | data/phase1_verification.json | identical |
| CC-052 | data plddt_bin_edges.json | — | {'plddt_mean_low': {'item_id': 0, 'n_proteins': 0, 'min_plddt': None, 'max_plddt': None}, 'plddt_mean_med': {'item_id': 1, 'n_proteins': 55232428, 'min_plddt': 50.0, 'max_plddt': 90.0}, 'plddt_mean_high': {'item_id': 2, 'n_proteins': 35274412, 'min_plddt': 90.01, 'max_plddt': 98.81}, 'plddt_frac_high_low': {'item_id': 3, 'n_proteins': 0, 'min_plddt': None, 'max_plddt': None}, 'plddt_frac_high_med': {'item_id': 4, 'n_proteins': 0, 'min_plddt': None, 'max_plddt': None}, 'plddt_frac_high_high': {'item_id': 5, 'n_proteins': 0, 'min_plddt': None, 'max_plddt': None}} | {'plddt_mean_low': {'item_id': 0, 'n_proteins': 0, 'min_plddt': None, 'max_plddt': None}, 'plddt_mean_med': {'item_id': 1, 'n_proteins': 55232428, 'min_plddt': 50.0, 'max_plddt': 90.0}, 'plddt_mean_high': {'item_id': 2, 'n_proteins': 35274412, 'min_plddt': 90.01, 'max_plddt': 98.81}, 'plddt_frac_high_low': {'item_id': 3, 'n_proteins': 0, 'min_plddt': None, 'max_plddt': None}, 'plddt_frac_high_med': {'item_id': 4, 'n_proteins': 0, 'min_plddt': None, 'max_plddt': None}, 'plddt_frac_high_high': {'item_id': 5, 'n_proteins': 0, 'min_plddt': None, 'max_plddt': None}} | data/phase1_verification.json | identical |
| CC-053 | G0 itemsets (first 1M rows, min_count 50): in-core / SON / efficient-apriori | P-001 | 168674 | 167,042 / 167,042 / 167,042 | data/mined/G0/G0.json | identical (three-way); vs GPU: not comparable: the first 1,000,000 rows of the rebuilt mined set are not the GPU run's rows (row order follows the joined pLDDT CSV, whose order comes from 16 parallel Storage Read streams and an unordered polars join; sha256 of the rebuilt multi parquet != EXCLUDED_LARGE_FILES.md; ORIGINAL_DATA_DIR not set) |
| CC-054 | C1 Base itemsets / K_max | P-028 | 5,305 / K9 | 5,305 / K9 | data/mined/C1/meta.json, data/mined/C1/cmp_gpu_opus.json | identical (full outer join vs Opus tables at 76,891, 0 differing) |
| CC-055 | C1 Base K distribution vs the 2-GPU campaign | P-014 | {1: 667, 2: 1504, 3: 1378, 4: 884, 5: 514, 6: 247, 7: 89, 8: 20, 9: 2} | {1: 667, 2: 1504, 3: 1378, 4: 884, 5: 514, 6: 247, 7: 89, 8: 20, 9: 2} | data/mined/C1/meta.json, data/mined/C1/cmp_gpu_opus.json, phase3/2026_01/exp/experiment_full_campaign_20260902_091611.json | identical |
| CC-056 | C1 Base K distribution | P-028 | {1: 667, 2: 1504, 3: 1378, 4: 884, 5: 514, 6: 247, 7: 89, 8: 20, 9: 2} | {1: 667, 2: 1504, 3: 1378, 4: 884, 5: 514, 6: 247, 7: 89, 8: 20, 9: 2} | data/mined/C1/meta.json, data/mined/C1/cmp_gpu_opus.json | identical |
| CC-057 | C1 Base wall (CPU: parquet load + apriori call) | P-028 | 31.1 s (one RTX 3090) | 268.1 s (median of 3, [267.8, 274.9]); peak RSS 11,740 MB | data/mined/C1/meta.json, data/mined/C1/cmp_gpu_opus.json | hardware-comparison |
| CC-058 | C2 Super itemsets / K_max | P-028 | 113,405 / K14 | 113,405 / K14 | data/mined/C2/meta.json, data/mined/C2/cmp_gpu_opus.json | identical (full outer join vs Opus tables at 7,690, 0 differing) |
| CC-059 | C2 Super K distribution vs the 2-GPU campaign | P-014 | {1: 1002, 2: 9151, 3: 19663, 4: 22638, 5: 20439, 6: 16105, 7: 11419, 8: 7121, 9: 3718, 10: 1543, 11: 483, 12: 107, 13: 15, 14: 1} | {1: 1002, 2: 9151, 3: 19663, 4: 22638, 5: 20439, 6: 16105, 7: 11419, 8: 7121, 9: 3718, 10: 1543, 11: 483, 12: 107, 13: 15, 14: 1} | data/mined/C2/meta.json, data/mined/C2/cmp_gpu_opus.json, phase3/2026_01/exp/experiment_full_campaign_20260902_091611.json | identical |
| CC-060 | C2 Super K distribution | P-028 | {1: 1002, 2: 9151, 3: 19663, 4: 22638, 5: 20439, 6: 16105, 7: 11419, 8: 7121, 9: 3718, 10: 1543, 11: 483, 12: 107, 13: 15, 14: 1} | {1: 1002, 2: 9151, 3: 19663, 4: 22638, 5: 20439, 6: 16105, 7: 11419, 8: 7121, 9: 3718, 10: 1543, 11: 483, 12: 107, 13: 15, 14: 1} | data/mined/C2/meta.json, data/mined/C2/cmp_gpu_opus.json | identical |
| CC-061 | C2 Super wall (CPU: parquet load + apriori call) | P-028 | 52.6 s (one RTX 3090) | 6,381.5 s (1 run); peak RSS 12,108 MB | data/mined/C2/meta.json, data/mined/C2/cmp_gpu_opus.json | hardware-comparison |
| CC-062 | C3 Power itemsets / K_max | P-028 | 475,865 / K14 | 475,865 / K14 | data/mined/C3/meta.json, data/mined/C3/cmp_gpu_opus.json | identical (full outer join vs Opus tables at 769, 0 differing) |
| CC-063 | C3 Power K distribution vs the 2-GPU campaign | P-014 | {1: 1002, 2: 22019, 3: 73205, 4: 108059, 5: 104239, 6: 78596, 7: 48699, 8: 25011, 9: 10508, 10: 3488, 11: 869, 12: 152, 13: 17, 14: 1} | {1: 1002, 2: 22019, 3: 73205, 4: 108059, 5: 104239, 6: 78596, 7: 48699, 8: 25011, 9: 10508, 10: 3488, 11: 869, 12: 152, 13: 17, 14: 1} | data/mined/C3/meta.json, data/mined/C3/cmp_gpu_opus.json, phase3/2026_01/exp/experiment_full_campaign_20260902_091611.json | identical |
| CC-064 | C3 Power K distribution | P-028 | {1: 1002, 2: 22019, 3: 73205, 4: 108059, 5: 104239, 6: 78596, 7: 48699, 8: 25011, 9: 10508, 10: 3488, 11: 869, 12: 152, 13: 17, 14: 1} | {1: 1002, 2: 22019, 3: 73205, 4: 108059, 5: 104239, 6: 78596, 7: 48699, 8: 25011, 9: 10508, 10: 3488, 11: 869, 12: 152, 13: 17, 14: 1} | data/mined/C3/meta.json, data/mined/C3/cmp_gpu_opus.json | identical |
| CC-065 | C3 Power wall (CPU: parquet load + apriori call) | P-028 | 74.7 s (one RTX 3090) | 25,604.1 s (1 run); peak RSS 12,438 MB | data/mined/C3/meta.json, data/mined/C3/cmp_gpu_opus.json | hardware-comparison |
| CC-066 | C4 Blitz itemsets / K_max (CPU with count inference) | P-028 | 2,841,280 / K19 | 2,841,280 / K19 | data/mined/C4/meta.json, data/mined/C4/cmp_gpu_opus.json | identical (full outer join vs Opus tables at 77, 0 differing) |
| CC-067 | C4 Blitz K distribution vs the 2-GPU campaign | P-014 | {1: 1002, 2: 39125, 3: 184900, 4: 361696, 5: 445661, 6: 439605, 7: 387030, 8: 318349, 9: 247680, 10: 179604, 11: 117783, 12: 67558, 13: 32831, 14: 13105, 15: 4155, 16: 1003, 17: 173, 18: 19, 19: 1} | {1: 1002, 2: 39125, 3: 184900, 4: 361696, 5: 445661, 6: 439605, 7: 387030, 8: 318349, 9: 247680, 10: 179604, 11: 117783, 12: 67558, 13: 32831, 14: 13105, 15: 4155, 16: 1003, 17: 173, 18: 19, 19: 1} | data/mined/C4/meta.json, data/mined/C4/cmp_gpu_opus.json, phase3/2026_01/exp/experiment_full_campaign_20260902_091611.json | identical |
| CC-068 | C4 Blitz K distribution | P-028 | {1: 1002, 2: 39125, 3: 184900, 4: 361696, 5: 445661, 6: 439605, 7: 387030, 8: 318349, 9: 247680, 10: 179604, 11: 117783, 12: 67558, 13: 32831, 14: 13105, 15: 4155, 16: 1003, 17: 173, 18: 19, 19: 1} | {1: 1002, 2: 39125, 3: 184900, 4: 361696, 5: 445661, 6: 439605, 7: 387030, 8: 318349, 9: 247680, 10: 179604, 11: 117783, 12: 67558, 13: 32831, 14: 13105, 15: 4155, 16: 1003, 17: 173, 18: 19, 19: 1} | data/mined/C4/meta.json, data/mined/C4/cmp_gpu_opus.json | identical |
| CC-069 | C4 Blitz wall (CPU: parquet load + apriori call) (CPU with count inference) | P-028 | 292.9 s (one RTX 3090) | 10,443.8 s (1 run); peak RSS 12,554 MB | data/mined/C4/meta.json, data/mined/C4/cmp_gpu_opus.json | hardware-comparison |
| CC-070 | C5 Ultra itemsets / K_max (CPU with count inference) | P-028 | 14,558,875 / K20 | 14,558,875 / K20 | data/mined/C5/meta.json, data/mined/C5/cmp_gpu_opus.json | identical (full outer join vs Opus tables at 16, 0 differing) |
| CC-071 | C5 Ultra K distribution vs the 2-GPU campaign | P-014 | {1: 1002, 2: 60088, 3: 356691, 4: 894903, 5: 1421780, 6: 1794852, 7: 2006312, 8: 2053858, 9: 1912672, 10: 1587145, 11: 1148998, 12: 712433, 13: 371981, 14: 160675, 15: 56221, 16: 15501, 17: 3236, 18: 480, 19: 45, 20: 2} | {1: 1002, 2: 60088, 3: 356691, 4: 894903, 5: 1421780, 6: 1794852, 7: 2006312, 8: 2053858, 9: 1912672, 10: 1587145, 11: 1148998, 12: 712433, 13: 371981, 14: 160675, 15: 56221, 16: 15501, 17: 3236, 18: 480, 19: 45, 20: 2} | data/mined/C5/meta.json, data/mined/C5/cmp_gpu_opus.json, phase3/2026_01/exp/experiment_full_campaign_20260902_091611.json | identical |
| CC-072 | C5 Ultra K distribution | P-028 | {1: 1002, 2: 60088, 3: 356691, 4: 894903, 5: 1421780, 6: 1794852, 7: 2006312, 8: 2053858, 9: 1912672, 10: 1587145, 11: 1148998, 12: 712433, 13: 371981, 14: 160675, 15: 56221, 16: 15501, 17: 3236, 18: 480, 19: 45, 20: 2} | {1: 1002, 2: 60088, 3: 356691, 4: 894903, 5: 1421780, 6: 1794852, 7: 2006312, 8: 2053858, 9: 1912672, 10: 1587145, 11: 1148998, 12: 712433, 13: 371981, 14: 160675, 15: 56221, 16: 15501, 17: 3236, 18: 480, 19: 45, 20: 2} | data/mined/C5/meta.json, data/mined/C5/cmp_gpu_opus.json | identical |
| CC-073 | C5 Ultra wall (CPU: parquet load + apriori call) (CPU with count inference) | P-028 | 1,996.4 s (one RTX 3090) | 15,199.8 s (1 run); peak RSS 12,676 MB | data/mined/C5/meta.json, data/mined/C5/cmp_gpu_opus.json | hardware-comparison |
| CC-074 | C6 Opus itemsets / K_max (CPU with count inference) | P-028 | 26,849,505 / K22 | 26,849,505 / K22 | data/mined/C6/meta.json, data/mined/C6/cmp_gpu_opus.json | identical (full outer join vs Opus tables at 8, 0 differing) |
| CC-075 | C6 Opus K distribution vs the 2-GPU campaign | P-014 | {1: 1002, 2: 73786, 3: 452777, 4: 1184461, 5: 1974126, 6: 2626332, 7: 3118459, 8: 3442954, 9: 3529257, 10: 3293612, 11: 2739532, 12: 1996772, 13: 1259045, 14: 679471, 15: 310527, 16: 118659, 17: 37261, 18: 9375, 19: 1818, 20: 255, 21: 23, 22: 1} | {1: 1002, 2: 73786, 3: 452777, 4: 1184461, 5: 1974126, 6: 2626332, 7: 3118459, 8: 3442954, 9: 3529257, 10: 3293612, 11: 2739532, 12: 1996772, 13: 1259045, 14: 679471, 15: 310527, 16: 118659, 17: 37261, 18: 9375, 19: 1818, 20: 255, 21: 23, 22: 1} | data/mined/C6/meta.json, data/mined/C6/cmp_gpu_opus.json, phase3/2026_01/exp/experiment_full_campaign_20260902_091611.json | identical |
| CC-076 | C6 Opus K distribution | P-028 | {1: 1002, 2: 73786, 3: 452777, 4: 1184461, 5: 1974126, 6: 2626332, 7: 3118459, 8: 3442954, 9: 3529257, 10: 3293612, 11: 2739532, 12: 1996772, 13: 1259045, 14: 679471, 15: 310527, 16: 118659, 17: 37261, 18: 9375, 19: 1818, 20: 255, 21: 23, 22: 1} | {1: 1002, 2: 73786, 3: 452777, 4: 1184461, 5: 1974126, 6: 2626332, 7: 3118459, 8: 3442954, 9: 3529257, 10: 3293612, 11: 2739532, 12: 1996772, 13: 1259045, 14: 679471, 15: 310527, 16: 118659, 17: 37261, 18: 9375, 19: 1818, 20: 255, 21: 23, 22: 1} | data/mined/C6/meta.json, data/mined/C6/cmp_gpu_opus.json | identical |
| CC-077 | C6 Opus wall (CPU: parquet load + apriori call) (CPU with count inference) | P-028 | 2,631.2 s (one RTX 3090) | 17,670.1 s (1 run); peak RSS 12,807 MB | data/mined/C6/meta.json, data/mined/C6/cmp_gpu_opus.json | hardware-comparison |
| CC-078 | C4 Blitz vs the GPU Blitz tables (`blitz/parquet`) | P-011 | 2,841,280 | 2,841,280 | data/mined/C4/cmp_gpu_blitz.json | identical (0 differing) |
| CC-079 | M4 (min_count 4) itemsets / K_max | P-017 | 48,007,493 / K22 | 48,007,493 / K22 | data/mined/M4/meta.json, data/mined/M4/cmp_gpu_minc4.json | identical (full outer join vs `minc4/parquet`, 0 differing) |
| CC-080 | M4 wall (CPU: parquet load + apriori call) | P-017 | 4,536.9 s (2 GPUs) | 20,653.4 s (1 run); peak RSS 18,653 MB | data/mined/M4/meta.json, data/mined/M4/cmp_gpu_minc4.json | hardware-comparison |
| CC-081 | M4 K distribution vs the GPU log | P-017 | {1: 1002, 2: 94427, 3: 603403, 4: 1649283, 5: 2916124, 6: 4169081, 7: 5318506, 8: 6223873, 9: 6644170, 10: 6362841, 11: 5373365, 12: 3943668, 13: 2484117, 14: 1327096, 15: 593694, 16: 219052, 17: 65356, 18: 15343, 19: 2722, 20: 342, 21: 27, 22: 1} | {1: 1002, 2: 94427, 3: 603403, 4: 1649283, 5: 2916124, 6: 4169081, 7: 5318506, 8: 6223873, 9: 6644170, 10: 6362841, 11: 5373365, 12: 3943668, 13: 2484117, 14: 1327096, 15: 593694, 16: 219052, 17: 65356, 18: 15343, 19: 2722, 20: 342, 21: 27, 22: 1} | data/mined/M4/meta.json, data/sizing/gpu_levels.json (minc4_mining.log) | identical |
| CC-082 | S1 SON itemsets / K_max | P-015 | 5,305 / K9 | 5,305 / K9 | data/mined/S1/meta.json, data/mined/S1/cmp_gpu_son.json, data/mined/S1/cmp_C1.json | identical vs GPU; identical vs C1 (set and counts, P-027) |
| CC-083 | S1 SON wall (CPU: parquet load + apriori_streaming call) | P-015 | 779.6 s (2× RTX 3090 box, `use_gpu=True`; GPUs used not recorded) | 291.1 s (median of 3, [290.8, 291.2]); pass 1 289.8 s, pass 2 0.0 s | data/mined/S1/meta.json, data/mined/S1/cmp_gpu_son.json, data/mined/S1/cmp_C1.json | hardware-comparison |
| CC-084 | S2 SON itemsets / K_max | P-016 | 113,405 / K14 | 113,405 / K14 | data/mined/S2/meta.json, data/mined/S2/cmp_gpu_son.json, data/mined/S2/cmp_C2.json | identical vs GPU; identical vs C2 (set and counts, P-027) |
| CC-085 | S2 SON wall (CPU: parquet load + apriori_streaming call) | P-016 | 1,426.0 s (2× RTX 3090 box, `use_gpu=True`; GPUs used not recorded) | 7,106.3 s (1 run); pass 1 7,104.8 s, pass 2 0.0 s | data/mined/S2/meta.json, data/mined/S2/cmp_gpu_son.json, data/mined/S2/cmp_C2.json | hardware-comparison |
| CC-086 | S3 SON itemsets / K_max | P-029 | 475,865 / K14 | 475,865 / K14 | data/mined/S3/meta.json, data/mined/S3/cmp_gpu_opus.json, data/mined/S3/cmp_C3.json | identical vs GPU; identical vs C3 (set and counts, P-027) |
| CC-087 | S3 SON wall (CPU: parquet load + apriori_streaming call) | P-029 | 6,151.1 s (one RTX 3090) | 29,451.7 s (1 run); pass 1 29,422.2 s, pass 2 28.1 s | data/mined/S3/meta.json, data/mined/S3/cmp_gpu_opus.json, data/mined/S3/cmp_C3.json | hardware-comparison |
| CC-088 | SON / in-core wall ratio at Power | P-029 | 98.81× | 1.15× (29,451.7 / 25,604.1 s) | data/mined/S3/meta.json, data/mined/C3/meta.json | hardware-comparison |
| CC-089 | X1 SON at Power, 4 chunks of 19,222,737 | — | — | 475,865 / K14; wall 16,322.1 s (1 run) vs S3 29,451.7 s | data/mined/X1/meta.json, data/mined/X1/cmp_C3.json | identical vs C3 (set and counts) |
| CC-090 | A1 K=22 itemset: features, supporting proteins, GO parent-child pairs | P-007..P-009 | K22; 8 proteins; 0 pairs | K22; 8 proteins; 0 pairs | data/mined/A1/k22_check.json | identical (feature and protein sets compared) |
| CC-091 | A1 Blitz decode: per-level counts, maxima, member sets, patterns (K=11, 12, 13, 17, 19) | P-030/P-031 | K11 28,913, K12 16,185, K13 11,521, K17 611, K19 187 | K11 28,913, K12 16,185, K13 11,521, K17 611, K19 187 | data/mined/A1/intermediate_k_decode.json | identical |
| CC-092 | P-026 support of {PF00270, GO:0003724, GO:0045087} | P-026 | 1,234 | 1,234 | data/mined/A1/p026_c6.json | identical |
| CC-093 | R1 rules (min_confidence 0.5) all / lift ≥ 5 / lift ≥ 100 | P-015 | 29,540 / 17,499 / 9,283 | 53,447 / 51,126 / 40,428 | data/mined/R1/meta.json, data/mined/R1/adjudication.json | mismatch (adjudicated for the CPU: exact enumeration 53,447; the branch generator on the GPU's non-ascending SON table gives 29,540; INCONSISTENCIES CI-001) |
| CC-094 | R1 max lift | P-015 | 968.9956 | 968.9956 | data/mined/R1/meta.json | identical |
| CC-095 | R1 cross-domain rules (lhs ∪ rhs spans > 1 feature category) | P-015 | 27,072 | 50,977 | data/mined/R1/adjudication.json | mismatch (adjudicated for the CPU; the branch generator gives 27,072) |
| CC-096 | null N5 mean total (K_max per permutation) | P-012 | 171,249.6 (K_max [6] over 5) | 171,290.4 (K_max [6]) | data/mined/N/verdict_n5.json | consistent (criteria: a_kmax_all_6, b_mean_within_3sigma_sqrt5, d_no_perm_reaches_real_at_k456_and_none_k_ge_7) |
| CC-097 | null N100 mean total (K_max per permutation) | P-018 | 171,306.3 (K_max [6] over 100) | 171,300.5 (K_max [6]) | data/mined/N/verdict_n100.json | consistent (criteria: a_kmax_all_6, c_mannwhitney_p_ge_0.01, d_no_perm_reaches_real_at_k456_and_none_k_ge_7) |
| CC-098 | C3G (C3 with use_generator_pruning) vs C3 | — | — | 475,865 itemsets; wall 5,477.5 s vs 25,604.1 s | data/mined/C3G/cmp_C3.json | identical |

## SON pass times at Power (hardware-comparison)

GPU P-029 (one RTX 3090, `phase3/2026_01/single_gpu/direct_vs_son_1gpu.log` timestamps): pass 1 5,300.4 s, pass 2 846.8 s. CPU S3 (`data/mined/S3/meta.json`): pass 1 29,422.2 s, pass 2 28.1 s.

## Per-level counts and times (hardware-comparison for the times)

CPU from `data/mined/<run>/meta.json` (`levels`); GPU from `data/sizing/gpu_levels.json` (`scripts/gpu_levels.py` over the GPU logs). Candidate counts differ by design (the routes generate candidates differently); frequent counts must agree.

| run | K | CPU candidates | CPU frequent | CPU s | GPU candidates | GPU frequent | GPU s | GPU id |
|---|---|---|---|---|---|---|---|---|
| C1 | 1 | 667 | 667 | 0.00 | 667 | 667 | 0.0 | P-028 (single-GPU log) |
| C1 | 2 | 222,111 | 1,504 | 11.15 | 222,111 | 1,504 | 0.4 | P-028 (single-GPU log) |
| C1 | 3 | 2,163 | 1,378 | 23.01 | 109,306 | 1,378 | 0.4 | P-028 (single-GPU log) |
| C1 | 4 | 926 | 884 | 57.49 | 4,677 | 884 | 0.3 | P-028 (single-GPU log) |
| C1 | 5 | 515 | 514 | 80.08 | 688 | 514 | 0.2 | P-028 (single-GPU log) |
| C1 | 6 | 247 | 247 | 49.64 | 262 | 247 | 0.2 | P-028 (single-GPU log) |
| C1 | 7 | 89 | 89 | 27.04 | 89 | 89 | 0.2 | P-028 (single-GPU log) |
| C1 | 8 | 20 | 20 | 7.78 | 20 | 20 | 0.2 | P-028 (single-GPU log) |
| C1 | 9 | 2 | 2 | 0.44 | 2 | 2 | 0.1 | P-028 (single-GPU log) |
| C2 | 1 | 1,002 | 1,002 | 0.00 | 1,002 | 1,002 | 0.2 | P-028 (single-GPU log) |
| C2 | 2 | 501,501 | 9,151 | 10.72 | 501,501 | 9,151 | 0.8 | P-028 (single-GPU log) |
| C2 | 3 | 37,868 | 19,663 | 85.22 | 1,005,655 | 19,663 | 1.7 | P-028 (single-GPU log) |
| C2 | 4 | 24,074 | 22,638 | 454.86 | 261,895 | 22,638 | 2.1 | P-028 (single-GPU log) |
| C2 | 5 | 20,506 | 20,439 | 1,290.09 | 56,985 | 20,439 | 2.6 | P-028 (single-GPU log) |
| C2 | 6 | 16,107 | 16,105 | 1,592.98 | 22,866 | 16,105 | 2.4 | P-028 (single-GPU log) |
| C2 | 7 | 11,419 | 11,419 | 1,649.77 | 12,757 | 11,419 | 1.9 | P-028 (single-GPU log) |
| C2 | 8 | 7,121 | 7,121 | 1,279.65 | 7,401 | 7,121 | 1.3 | P-028 (single-GPU log) |
| C2 | 9 | 3,718 | 3,718 | 1.19 | 3,798 | 3,718 | 0.8 | P-028 (single-GPU log) |
| C2 | 10 | 1,543 | 1,543 | 0.52 | 1,561 | 1,543 | 0.4 | P-028 (single-GPU log) |
| C2 | 11 | 483 | 483 | 0.32 | 485 | 483 | 0.2 | P-028 (single-GPU log) |
| C2 | 12 | 107 | 107 | 0.09 | 107 | 107 | 0.1 | P-028 (single-GPU log) |
| C2 | 13 | 15 | 15 | 0.24 | 15 | 15 | 0.1 | P-028 (single-GPU log) |
| C2 | 14 | 1 | 1 | 0.23 | 1 | 1 | 0.1 | P-028 (single-GPU log) |
| C3 | 1 | 1,002 | 1,002 | 0.00 | 1,002 | 1,002 | 0.0 | P-028 (single-GPU log) |
| C3 | 2 | 501,501 | 22,019 | 10.73 | 501,501 | 22,019 | 0.8 | P-028 (single-GPU log) |
| C3 | 3 | 224,019 | 73,205 | 102.64 | 1,855,572 | 73,205 | 2.8 | P-028 (single-GPU log) |
| C3 | 4 | 120,625 | 108,059 | 1,059.54 | 1,506,134 | 108,059 | 5.8 | P-028 (single-GPU log) |
| C3 | 5 | 105,122 | 104,239 | 4,541.21 | 583,529 | 104,239 | 9.0 | P-028 (single-GPU log) |
| C3 | 6 | 78,659 | 78,596 | 7,212.98 | 186,587 | 78,596 | 9.4 | P-028 (single-GPU log) |
| C3 | 7 | 48,703 | 48,699 | 7,536.81 | 77,638 | 48,699 | 7.4 | P-028 (single-GPU log) |
| C3 | 8 | 25,013 | 25,011 | 5,120.98 | 32,966 | 25,011 | 4.7 | P-028 (single-GPU log) |
| C3 | 9 | 10,508 | 10,508 | 2.03 | 12,519 | 10,508 | 2.3 | P-028 (single-GPU log) |
| C3 | 10 | 3,488 | 3,488 | 0.65 | 3,914 | 3,488 | 0.9 | P-028 (single-GPU log) |
| C3 | 11 | 869 | 869 | 0.36 | 927 | 869 | 0.3 | P-028 (single-GPU log) |
| C3 | 12 | 152 | 152 | 0.10 | 155 | 152 | 0.1 | P-028 (single-GPU log) |
| C3 | 13 | 17 | 17 | 0.24 | 17 | 17 | 0.1 | P-028 (single-GPU log) |
| C3 | 14 | 1 | 1 | 0.23 | 1 | 1 | 0.1 | P-028 (single-GPU log) |
| C4 | 1 | 1,002 | 1,002 | 0.00 | 1,002 | 1,002 | 0.0 | P-028 (single-GPU log) |
| C4 | 2 | 501,501 | 39,125 | 10.94 | 501,501 | 39,125 | 0.9 | P-028 (single-GPU log) |
| C4 | 3 | 796,300 | 184,900 | 107.89 | 3,166,342 | 184,900 | 4.5 | P-028 (single-GPU log) |
| C4 | 4 | 448,959 | 361,696 | 1,751.94 | 4,777,345 | 361,696 | 12.5 | P-028 (single-GPU log) |
| C4 | 5 | 452,265 | 445,661 | 4,722.92 | 3,037,012 | 445,661 | 24.9 | P-028 (single-GPU log) |
| C4 | 6 | 440,013 | 439,605 | 2,725.50 | 1,263,744 | 439,605 | 34.4 | P-028 (single-GPU log) |
| C4 | 7 | 387,055 | 387,030 | 878.47 | 629,087 | 387,030 | 37.0 | P-028 (single-GPU log) |
| C4 | 8 | 318,350 | 318,349 | 222.29 | 388,672 | 318,349 | 34.3 | P-028 (single-GPU log) |
| C4 | 9 | 247,680 | 247,680 | 2.11 | 265,470 | 247,680 | 28.7 | P-028 (single-GPU log) |
| C4 | 10 | 179,604 | 179,604 | 1.71 | 183,438 | 179,604 | 22.2 | P-028 (single-GPU log) |
| C4 | 11 | 117,783 | 117,783 | 1.32 | 118,447 | 117,783 | 15.9 | P-028 (single-GPU log) |
| C4 | 12 | 67,558 | 67,558 | 0.76 | 67,638 | 67,558 | 10.4 | P-028 (single-GPU log) |
| C4 | 13 | 32,831 | 32,831 | 0.61 | 32,836 | 32,831 | 5.8 | P-028 (single-GPU log) |
| C4 | 14 | 13,105 | 13,105 | 0.40 | 13,105 | 13,105 | 2.6 | P-028 (single-GPU log) |
| C4 | 15 | 4,155 | 4,155 | 0.29 | 4,155 | 4,155 | 1.6 | P-028 (single-GPU log) |
| C4 | 16 | 1,003 | 1,003 | 0.10 | 1,003 | 1,003 | 0.3 | P-028 (single-GPU log) |
| C4 | 17 | 173 | 173 | 0.23 | 173 | 173 | 0.1 | P-028 (single-GPU log) |
| C4 | 18 | 19 | 19 | 0.22 | 19 | 19 | 0.1 | P-028 (single-GPU log) |
| C4 | 19 | 1 | 1 | 0.08 | 1 | 1 | 0.1 | P-028 (single-GPU log) |
| C5 | 1 | 1,002 | 1,002 | 0.00 | 1,002 | 1,002 | 0.0 | P-028 (single-GPU log) |
| C5 | 2 | 501,501 | 60,088 | 10.97 | 501,501 | 60,088 | 0.9 | P-028 (single-GPU log) |
| C5 | 3 | 1,919,793 | 356,691 | 122.07 | 5,334,089 | 356,691 | 6.7 | P-028 (single-GPU log) |
| C5 | 4 | 1,158,832 | 894,903 | 2,745.11 | 10,789,453 | 894,903 | 23.1 | P-028 (single-GPU log) |
| C5 | 5 | 1,447,709 | 1,421,780 | 6,968.20 | 9,100,811 | 1,421,780 | 57.9 | P-028 (single-GPU log) |
| C5 | 6 | 1,796,794 | 1,794,852 | 3,875.31 | 5,011,200 | 1,794,852 | 106.5 | P-028 (single-GPU log) |
| C5 | 7 | 2,006,468 | 2,006,312 | 1,137.17 | 3,065,588 | 2,006,312 | 151.9 | P-028 (single-GPU log) |
| C5 | 8 | 2,053,865 | 2,053,858 | 260.73 | 2,379,595 | 2,053,858 | 178.2 | P-028 (single-GPU log) |
| C5 | 9 | 1,912,672 | 1,912,672 | 16.85 | 2,003,749 | 1,912,672 | 222.9 | P-028 (single-GPU log) |
| C5 | 10 | 1,587,145 | 1,587,145 | 14.33 | 1,610,499 | 1,587,145 | 172.9 | P-028 (single-GPU log) |
| C5 | 11 | 1,148,998 | 1,148,998 | 12.40 | 1,154,547 | 1,148,998 | 144.6 | P-028 (single-GPU log) |
| C5 | 12 | 712,433 | 712,433 | 9.12 | 713,608 | 712,433 | 107.3 | P-028 (single-GPU log) |
| C5 | 13 | 371,981 | 371,981 | 5.48 | 372,180 | 371,981 | 61.9 | P-028 (single-GPU log) |
| C5 | 14 | 160,675 | 160,675 | 2.76 | 160,697 | 160,675 | 30.0 | P-028 (single-GPU log) |
| C5 | 15 | 56,221 | 56,221 | 1.21 | 56,222 | 56,221 | 12.2 | P-028 (single-GPU log) |
| C5 | 16 | 15,501 | 15,501 | 0.40 | 15,501 | 15,501 | 4.0 | P-028 (single-GPU log) |
| C5 | 17 | 3,236 | 3,236 | 0.30 | 3,236 | 3,236 | 1.1 | P-028 (single-GPU log) |
| C5 | 18 | 480 | 480 | 0.24 | 480 | 480 | 0.2 | P-028 (single-GPU log) |
| C5 | 19 | 45 | 45 | 0.09 | 45 | 45 | 0.1 | P-028 (single-GPU log) |
| C5 | 20 | 2 | 2 | 0.22 | 2 | 2 | 0.1 | P-028 (single-GPU log) |
| C6 | 1 | 1,002 | 1,002 | 0.00 | 1,002 | 1,002 | 0.0 | P-028 (single-GPU log) |
| C6 | 2 | 501,501 | 73,786 | 11.18 | 501,501 | 73,786 | 0.9 | P-028 (single-GPU log) |
| C6 | 3 | 2,883,292 | 452,777 | 124.13 | 7,151,097 | 452,777 | 8.5 | P-028 (single-GPU log) |
| C6 | 4 | 1,569,632 | 1,184,461 | 3,167.36 | 15,450,673 | 1,184,461 | 29.7 | P-028 (single-GPU log) |
| C6 | 5 | 2,018,549 | 1,974,126 | 8,013.21 | 13,800,022 | 1,974,126 | 75.6 | P-028 (single-GPU log) |
| C6 | 6 | 2,630,524 | 2,626,332 | 4,527.46 | 8,045,174 | 2,626,332 | 144.1 | P-028 (single-GPU log) |
| C6 | 7 | 3,118,876 | 3,118,459 | 1,337.82 | 5,006,188 | 3,118,459 | 217.3 | P-028 (single-GPU log) |
| C6 | 8 | 3,442,988 | 3,442,954 | 298.53 | 4,033,307 | 3,442,954 | 276.5 | P-028 (single-GPU log) |
| C6 | 9 | 3,529,257 | 3,529,257 | 30.26 | 3,697,577 | 3,529,257 | 312.5 | P-028 (single-GPU log) |
| C6 | 10 | 3,293,612 | 3,293,612 | 35.22 | 3,341,726 | 3,293,612 | 324.4 | P-028 (single-GPU log) |
| C6 | 11 | 2,739,532 | 2,739,532 | 35.17 | 2,755,768 | 2,739,532 | 303.5 | P-028 (single-GPU log) |
| C6 | 12 | 1,996,772 | 1,996,772 | 31.01 | 2,003,045 | 1,996,772 | 255.4 | P-028 (single-GPU log) |
| C6 | 13 | 1,259,045 | 1,259,045 | 18.22 | 1,261,325 | 1,259,045 | 183.9 | P-028 (single-GPU log) |
| C6 | 14 | 679,471 | 679,471 | 11.63 | 680,128 | 679,471 | 110.5 | P-028 (single-GPU log) |
| C6 | 15 | 310,527 | 310,527 | 6.05 | 310,661 | 310,527 | 57.5 | P-028 (single-GPU log) |
| C6 | 16 | 118,659 | 118,659 | 2.50 | 118,676 | 118,659 | 24.7 | P-028 (single-GPU log) |
| C6 | 17 | 37,261 | 37,261 | 1.03 | 37,262 | 37,261 | 8.7 | P-028 (single-GPU log) |
| C6 | 18 | 9,375 | 9,375 | 0.46 | 9,375 | 9,375 | 2.4 | P-028 (single-GPU log) |
| C6 | 19 | 1,818 | 1,818 | 0.14 | 1,818 | 1,818 | 0.6 | P-028 (single-GPU log) |
| C6 | 20 | 255 | 255 | 0.23 | 255 | 255 | 0.2 | P-028 (single-GPU log) |
| C6 | 21 | 23 | 23 | 0.22 | 23 | 23 | 0.1 | P-028 (single-GPU log) |
| C6 | 22 | 1 | 1 | 0.22 | 1 | 1 | 0.1 | P-028 (single-GPU log) |
| M4 | 1 | 1,002 | 1,002 | 0.00 | 1,002 | 1,002 | 0.4 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 2 | 501,501 | 94,427 | 10.75 | 501,501 | 94,427 | 0.6 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 3 | 4,668,899 | 603,403 | 132.08 | 10,275,529 | 603,403 | 9.3 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 4 | 2,225,041 | 1,649,283 | 3,781.10 | 23,484,365 | 1,649,283 | 348.5 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 5 | 2,982,216 | 2,916,124 | 9,276.99 | 22,203,427 | 2,916,124 | 1042.6 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 6 | 4,175,330 | 4,169,081 | 5,150.95 | 13,912,871 | 4,169,081 | 1260.3 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 7 | 5,319,068 | 5,318,506 | 1,528.22 | 9,081,564 | 5,318,506 | 979.4 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 8 | 6,223,896 | 6,223,873 | 347.52 | 7,538,368 | 6,223,873 | 503.9 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 9 | 6,644,171 | 6,644,170 | 74.72 | 7,074,028 | 6,644,170 | 192.8 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 10 | 6,362,841 | 6,362,841 | 92.54 | 6,497,043 | 6,362,841 | 71.9 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 11 | 5,373,365 | 5,373,365 | 83.60 | 5,415,551 | 5,373,365 | 39.6 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 12 | 3,943,668 | 3,943,668 | 63.63 | 3,956,938 | 3,943,668 | 27.3 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 13 | 2,484,117 | 2,484,117 | 47.29 | 2,487,961 | 2,484,117 | 17.7 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 14 | 1,327,096 | 1,327,096 | 23.47 | 1,328,022 | 1,327,096 | 9.8 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 15 | 593,694 | 593,694 | 12.23 | 593,860 | 593,694 | 4.6 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 16 | 219,052 | 219,052 | 5.06 | 219,071 | 219,052 | 1.8 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 17 | 65,356 | 65,356 | 1.79 | 65,357 | 65,356 | 0.6 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 18 | 15,343 | 15,343 | 0.64 | 15,343 | 15,343 | 0.1 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 19 | 2,722 | 2,722 | 0.17 | 2,722 | 2,722 | 0.0 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 20 | 342 | 342 | 0.23 | 342 | 342 | 0.0 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 21 | 27 | 27 | 0.22 | 27 | 27 | 0.0 | P-017 (2 GPUs, minc4_mining.log) |
| M4 | 22 | 1 | 1 | 0.22 | 1 | 1 | 0.0 | P-017 (2 GPUs, minc4_mining.log) |
