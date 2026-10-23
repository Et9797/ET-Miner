# Interim independent review (2026-10-21)

Reviewer: a fresh adversarial-verifier subagent that built none of this ("try to break it"), run while N100 was at
permutation 7 of 100, over everything finished by then (data, G0, C1–C6, C3G, S1–S3, R1, N5, A1). It re-counted the
itemset tables with its own pyarrow/numpy code (C1–C3 vs Opus filtered, S1/S2 vs the GPU SON tables, C4 vs Blitz on
all 19 levels, C5 vs Opus at 16, C6 vs Opus on all 22 levels; count-recovery residual ≤ 9.3e-10), direct-counted 18
itemsets over the mined parquet (the K=22 set, P-026, boundary rows at 769 and 77, pairs with items 458/459),
mutation-tested `compare.py` (a +1 count, a missing row and an extra row at the threshold were caught), re-derived
R1 exactly, and checked seeds, σ, the band and criterion (d) of N5. Every "identical" itemset verdict held.

Findings, as reported, with what was done:

1. (must-fix) `data/MANIFEST.md` was never committed: the repo-wide `.gitignore` rule `MANIFEST.md` ignored it.
   → Fixed: negation `!runs/20261019T1718Z-cpu/data/MANIFEST.md` in `.gitignore`; the file is committed.
2. (must-fix) M4 and X1 would not start: the running queue had read PLAN.md at 08:17Z, before their re-admission.
   → Fixed: `scripts/after_pid.sh 629549 scripts/queue.sh .` (PID 704457) restarts the queue when it exits;
   PROGRESS.md updated (PLAN.md Re-estimates 18:45Z).
3. (should) The safety factor was not re-derived when the sizing reference changed (C3G, C4, C6); M4's estimate and
   2× limit were understated. → Fixed: `sizing.py` checks every other count-inference run against the reference
   (`data/sizing/sizing_C6_v2.json`: 1.22 / 2.74 / 6.00); M4 re-estimated at 20.4 h with 2.74; the selection still
   fits (PLAN.md 18:45Z). The 6.00 alternative is listed as an open item.
4. (should) Two selection choices go beyond the brief's text: D-011's "skip and continue" reading, and re-admitting
   dropped runs after re-estimation with a reference chosen after each run. → Recorded in PLAN.md and D-011; listed in
   CPU_REPORT.md as an open item for Et's sign-off.
5. (should) CI-001 described the rule-generator bug incompletely (consequent misses keep the rule with lift 0; the
   cross-domain count is affected too; I-012's 45,286 is unchecked). → Fixed: `adjudicate_rules.py` reports lift-0
   and cross-domain counts (exact 50,977; branch generator on the GPU table 27,072 = GPU, 10,502 lift-0 rules);
   CI-001 and RESULTS CPU-056 updated; I-012's cross-domain value recorded as not reproduced.
6. (should) CI-002 said the full tables are "identical in content (CPU-017)" and CPU-020 inferred the row order from
   the sha256 alone. → Fixed: both reworded (aggregate statistics only; the G0 totals are the evidence).
7. (should) The CPU_REPORT draft overstated "every itemset table identical" (not G0 or the null tables) and "streamed
   once". → Fixed in the draft.
8. (should) Values without an artifact: CPU-043, CPU-054, and the GPU null's "K_max 6" typed in `build_comparison.py`.
   → Fixed: `data/mined/A1/p026_c3.json`; `scripts/provenance_scan.sh` with `logs/provenance_scan_20261021T1841Z.log`;
   K_max read from the GPU JSONs.
9. (should) CPU_COMPARISON lacked the per-level times next to P-028/P-029, and some labels were wrong (count inference
   not named for C4–C6, the GPU box for S1/S2, N100 → P-018, "P-028/P-014"). → Fixed in `build_comparison.py`
   (per-level table, SON pass times at Power, labels).
10. (nice) CPU-006 said 9 tests (11 now); RESULTS walls are call-only while CPU_COMPARISON walls include the load;
    CPU-024 compared the CPU call with the GPU's mining loop (its call was 3.5 s); CPU-022's compressor explanation
    (gzip MTIME is likelier); `G0/ea/meta.json` has no `env` block. → Fixed: CPU-006/022/024 corrected, CPU-055 added,
    the wall definition stated in CPU_COMPARISON.md; the oracle's meta records no engine env because it does not use
    the engine (left as is).
11. (nice) The branch tracked `origin/alphafold-experimental-results-reproduction` with `push.default=current`.
    → `git branch --unset-upstream`; no push had happened. Corrected after the final review: with
    `push.default=current` in `~/.gitconfig` a bare `git push` would still push the current branch, so this lowers the
    risk only for an explicit upstream push; no push happened at any point (no remote ref of the branch).
12. (nice) D-012's list of by-hand overlaps missed the 05:18Z P-026 check, and the reviewer's own probes overlapped N100
    permutations 7–8. → Fixed in D-012.
