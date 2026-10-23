# Review of the paper revision with the CPU results (2026-10-23)

Reviewer: a fresh adversarial-verifier subagent that wrote none of the revision ("try to break it"), over the uncommitted
diff of `paper/et_miner_proteome.tex` that adds Section "Reproduction on a CPU" (Table `tab:cpu`) and the sentences
citing it. It re-derived every number of the new table and text from the artifacts (walls, medians, ratios, hours, GiB,
SON ratios, the S3 pass-1 counts, the null means, ranges, Mann–Whitney p and seeds, the K=22 features and accessions,
the intermediate patterns, P-026), re-compared C1, C5 and C6 with the GPU Opus tables itself (0 differences), checked
the bibliography's first-citation order (29 of 29) and built the tex in a scratch copy. Every number held; the findings
concern wording, scope and cause.

Findings, as reported, with what was done:

1. (must-fix) The abstract still called the problem "computationally prohibitive" next to a 4.9 h desktop run. → "computationally demanding".
2. (should) "Every mining run reproduced exactly" overstated: the GPU run saved itemset tables only from its two-GPU
   exports (Opus, Blitz, 4 proteins) and its SON runs at Base and Super; the timed single-GPU runs kept counts and
   K-distributions. → Abstract, introduction, conclusion, table caption and the comparison paragraph now say which
   tables were saved and that the other runs agree in counts and K-distributions.
3. (should) "Its deepest run took 4.9 hours": the 4-protein run is as deep (5.74 h), and the time depends on count
   inference. → "mining the Opus threshold took it 4.9 hours (with count inference, which the GPU run did not use)".
4. (should) "All 51 checks agreed": 55 checks, 4 informational ones differ. → "Of the 55 checks … 51 agreed"; the other
   four (three checksums, the item ids) are tied to the two differences.
5. (should) The SON cause ("the 99× belongs to the streaming implementation") compared different code versions and
   machines; "the released SON implementation" became ambiguous. → Worded as a suggestion with that caveat, using the
   GPU run's own evidence (its chunk pass alone took 85× the direct run); "the SON implementation of the version used
   for the GPU run".
6. (should) Two older sentences conflicted with the CPU result. → "With the streaming implementation used here, the
   direct path …"; "critical to the GPU path's speed".
7. (should) The ratios mix count inference with hardware. → Discussion: "Below Power only the CPU runs used count
   inference, which at Power alone shortened the CPU run 4.7×".
8. (should) 18.2 GiB at 4 proteins was the process after mining (table normalization), not the mining call (12.7 GiB).
   → Table and text give the peak up to the end of the mining call; the caption names the 18.2 GiB afterwards.
9. (nice) The row-order claim had no evidence in the text. → G0's 167,042 vs 168,674 cited; "as far as these checks can show".
10. (nice) Bastide et al. call them key patterns. → "key pattern (or free)"; "the four runs that used it below Power
    match the GPU tables". The bibliographic data of `bastide2000` were not checked online (inbound-data rule).
11. (nice) "two chunks of 40 million" → "at most"; "161.9 GB archive" vs Section 2.1's "149.8 GiB flat file" → the tex
    now names the 149.8 GiB flat file and the archive's MD5; the SON ratios differ from CPU_REPORT.md (medians vs first
    runs) → stated in `PAPER_VALUES.md`; S2 and X1 overlaps → stated in the table caption.
12. (nice) Caption "Sections 3 and 3.3"; "Independent" overstated (same author, software and pipeline). → New label
    `sec:campaign`; the section is "Reproduction on a CPU" and the abstract says "a second execution". The reuse of the
    GPU run's go.obo for the K=22 check (D-009) is not mentioned in the paper.
13. (nice) Underfull boxes and 12 → 14 pages. → Accepted; the build has the same warnings as before plus underfull boxes.

Check after the fixes: `scripts/paper_values.py` → `logs/paper_values_20261023T1032Z.log` (52 values, pass).

# Council review of the revised paper (2026-10-23)

`/council review`: four fresh reviewers (architect, math, alternatives, auditor), two rounds. Math and the auditor
recomputed every number independently from the raw artifacts; all held. Round 2 ended in a unanimous block on wording,
not numbers. What was blocked, and what was done:

1. Scope: "the itemset counts … of its other runs" / "every other run" (abstract, introduction) was contradicted by the
   paper itself (the GPU's first-million-row count 168,674 vs 167,042, CI-002; the null model matched only as a
   distribution), and Data availability still said the GPU run produced "every value in this paper". → "every full-data
   itemset table the GPU run saved … its other full-data mining runs"; introduction adds "matched the null model as a
   distribution"; conclusion "full-data"; "The complete GPU run that produced every value in this paper outside Section 3.6".
2. Hardware: "without a GPU" / "no GPU" for an APU (env.json: "with Radeon Graphics"); `paper_values.py` stripped that
   suffix. → "without a discrete GPU"; "no CUDA device; the processor's integrated graphics unused"; the script now
   requires this wording.
3. Null timing: "14.7 min on the CPU against 18.6 s" set a whole permutation (shuffle, rebuild, mining) against the GPU's
   mining-only time (row-split branch, `experiment_null_model.py` 477–519). → "a whole permutation … 14.7 min on the CPU
   against 20.0 s on the two GPUs, and its mining alone 7.3 min against 18.6 s (medians)", the GPU values parsed from
   `phase3/2026_01/null_model_769_100.log`. Section 3.5's existing "(18.6 s per permutation)" next to 2,007 s for 100
   permutations → "(18.6 s of mining per permutation)" (root RESULTS P-022 calls it "per permutation").

Non-blocking flags applied: `\ref{sec:campaign}` for the campaign/SON references (tab:campaign caption, Discussion 4.1,
Appendix memory caption); the tab:cpu caption no longer cites its own section, says the in-core GPU times exclude reading
the transactions and that peak memory of repeated runs is the largest of three; the Power, CI row has the GPU time
(74.7 s) and ratio (73.4×, from 74.65 s); "The GPU results above come from one execution"; "the counting inference of
Bastide et al."; the checksum differences are attributed to row order and, for the two transaction files, another writer;
the Results paragraph gives the ratio ranges (8.6–343× without, 4.6–73.4× with count inference) instead of repeating
the table; Discussion: "in-core mining … at most 12.7 GiB (the null permutations … peaked at 24.0 GiB)".
`paper_values.py`: GPU times read from the GPU run's JSONs and null log (not from CPU_COMPARISON_rows.json), required
scope and hardware wording, nine rejected phrasings, a compare report required per run, peak memory as the maximum over
repetitions, docstring renamed; two planted errors (a rejected phrasing, a wrong null time) fail the check.

Not applied: the introduction's "requires a computational approach fundamentally different from existing tools" and
Discussion's "feasible … on a single consumer GPU" (framing, left to the author); a CI-versus-hardware decomposition of
the ratios (withdrawn by its proposer: the data cannot separate them); the 4-protein GPU time includes its per-K parquet
flush while the CPU wall excludes writing (no effect on the rounded values); per-run engine sha (not recorded in the run
metas; env.json and the clean checkout at 37819e2 carry it).

Check after the fixes: `logs/paper_values_20261023T1402Z.log` (61 values, 9 rejected phrasings absent, pass); tex builds
with the pre-existing warnings only (14 pages).
