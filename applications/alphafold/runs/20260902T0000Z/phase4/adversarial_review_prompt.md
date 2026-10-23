Adversarial review brief for COMPARISON_REPORT.md (to be given to a fresh-context subagent)

You are an adversarial reviewer of a reproduction audit. Assume every "confirmed" verdict may be wrong
until you have seen the evidence. Files: /root/projects/ET-Miner/COMPARISON_REPORT.md (the report),
/root/projects/ET-Miner/runs/20260902T0000Z/phase4/fresh_values.json (qkey -> fresh value + artifact path),
/root/projects/ET-Miner/runs/20260902T0000Z/phase4/COMPARISON_REPORT_rows.json (one record per claim),
/root/projects/ET-Miner/CLAIMS.md (all extracted claims), /root/projects/ET-Miner/RESULTS.md,
/root/projects/ET-Miner/INCONSISTENCIES.md, and the run directory /root/projects/ET-Miner/runs/20260902T0000Z/.
Checks (report each with evidence):
 1. Coverage: every claim ID in CLAIMS.md (T-, R1-, R2-, L-, L2- prefixes) appears exactly once in section 4
    of the report; no verdict outside {confirmed, hallucinated, inconclusive, expected-hardware-deviation};
    no "TODO"/"pending"/empty verdict cells.
 2. Provenance: for at least 40 randomly chosen "confirmed" rows (covering tex, reviews and logs) open the
    artifact file cited in fresh_values.json for that qkey and recompute/observe the value; flag any row whose
    artifact is missing, whose value cannot be found in it, or whose artifact path lies under
    applications/alphafold/results_214m/ or paper/ (old data — forbidden as a fresh source).
 3. Verdict logic: for 30 random "hallucinated" rows check that the claimed and fresh values really differ
    under the stated rules (exact for integers; rounding at the claim's precision for rounded/approximate
    claims; bounds for ">" claims); for 30 random "inconclusive" rows check that the stated reason is true
    (no fresh artifact, external fact, random-stream statistic, etc.); for hardware rows check the label.
 4. Headline table: re-derive five headline values directly from the parquet/JSON artifacts
    (e.g. total Opus itemsets from the per-K parquet row counts, the K=22 itemset members, the
    multi-feature count from stats.json) and compare with the report.
 5. Consistency: numbers quoted in the header and Section 0 must agree with RESULTS.md and fresh_values.json.
Output: a list of concrete defects (row IDs, what is wrong, evidence), an overall judgement, and the counts you
verified. Do not modify any file.
