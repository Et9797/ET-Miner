# Final independent review (2026-10-23)

Reviewer: a fresh adversarial-verifier subagent that built none of this ("try to break it"), run after every mining
run had finished, over the whole campaign with the focus on N100, M4, X1, C5, the level-at-a-time `compare.py` and
the generated reports. It found no must-fix. It re-counted M4 against `minc4/parquet` with its own pyarrow/numpy code
(all 22 levels, 48,007,493 rows, summed counts 5,584,830,296, ids 458/459 translated), X1 against C3 (14 levels) and
C5 against Opus at 16 (20 levels); direct-counted, by streaming the 76,890,945-row mined set, every single item and
pair (equal to the K1/K2 tables of M4, C6, C5, C4, C3, C2, C1), 14 sampled M4 itemsets at K 3–22 and 6 absent ones;
re-derived both null verdicts (seeds, μ, σ, band, Mann–Whitney p, criterion d); checked every compare report's CPU
signature against its run meta, the empty-side and filtered-side guards, thresholds, environments (127 metas),
provenance, the excluded-file list, and every number of the report's runs, SON and null sections. The interim fixes
1–7, 9, 10 and 12 were confirmed to hold.

Findings, as reported, with what was done:

1. (should) `compare.py`'s level mode kept one file when two files map to the same K (no real directory affected), and
   no test covered the level mode. → Fixed: `level_files` rejects two files for one level;
   `scripts/tests/test_compare_levels.py` adds 7 tests (equality with the whole-table path, the incremental signature,
   a level on one side only, a +1 count, translation, the reference filter, duplicate level files).
2. (should) D-015's verification claims had no artifact. → Fixed: `scripts/check_compare_levels.sh` →
   `logs/check_compare_levels_20261023T0906Z.log` and `data/checks/compare_levels/` (RESULTS CPU-064); D-015 cites them.
3. (should) Some values were typed into the build scripts (G0's 3.5 s and K14, the 98.8× line, "K_max 6 in all").
   → Fixed: read from `validate_row_split.log` (the `__main__` line), the G0 report's GPU K distribution, the direct-vs-SON
   JSON and the verdict JSONs' `k_max_per_perm`.
4. (should) M4's peak memory was understated (12.7 GiB during the engine call; the process peaked at 18.2 GiB during
   table normalization). → Fixed: every run's peak RSS in CPU_COMPARISON.md and CPU_REPORT.md is now the whole
   process's (max of both measurements).
5. (nice) No verdict rows for the K distributions against P-014 (2-GPU campaign) and for M4 against P-017. → Added
   (all identical).
6. (nice) A garbled N5 label in the report; check names ("shards", "mapping feature set") in the GPU-id column; P-029
   given for X1. → Fixed.
7. (nice) RESULTS CPU-058 quoted the 100-permutation median for the 95 permutations. → Corrected (440.93 s).
8. (nice) Interim fix 11 (`--unset-upstream`) does not block a bare `git push` under `push.default=current`. → The
   interim note is corrected; no push happened at any point.
9. (nice) The bundle was one commit behind. → Rewritten at the end.
10. (nice) X1's wall shared the CPU with the low-priority D-015 checks, and P1T ran once although under 15 minutes.
    → The report states the X1 caveat; P1T_r2 and P1T_r3 were run (152.84 s median, RESULTS CPU-063).
