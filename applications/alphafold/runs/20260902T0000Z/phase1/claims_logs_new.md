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
