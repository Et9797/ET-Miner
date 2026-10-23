# INCONSISTENCIES — CPU reproduction

Discrepancies between the CPU run, the GPU run's artifacts, scripts and logs.
Each entry: what differs, the evidence, the adjudication (direct count where
itemsets are concerned) and the consequence.

## CI-001 — R1 rule counts: the GPU run's 29,540 rules undercount the rules of identical itemsets (verified)

- GPU (RESULTS P-015, `phase3/2026_01/exp/son_base.json`): 29,540 rules at min_confidence 0.5 from the 5,305
  Base SON itemsets; 17,499 with lift >= 5; 9,283 with lift >= 100.
- CPU R1 (`data/mined/R1/meta.json`, pinned engine `generate_rules`): 53,447 / 51,126 / 40,428 from C1's 5,305
  itemsets, which are identical to the GPU tables (CPU-025, CPU-031).
- Adjudication (`scripts/adjudicate_rules.py` -> `data/mined/R1/adjudication.json`):
  1. exact integer enumeration of every rule of the CPU itemsets: 53,447 / 51,126 / 40,428, the same rule set as R1;
  2. the pinned engine's `generate_rules` on the GPU SON table as stored: 53,447 / 51,126 / 40,428, the same rule set
     (after translating ids by feature_name);
  3. this branch's `src/et_miner/core/rules.py::generate_rules` (the generator the GPU run used) on the GPU SON table as
     stored: 29,540 / 17,499 / 9,283, exactly P-015, a strict subset of the exact set;
  4. the same branch generator on the CPU tables (items ascending): 53,447.
- Cause: 1,623 of the 5,305 itemsets in `son_base_itemsets.parquet` are not in ascending item order. The branch's
  `_build_support_lookup` keys supports by the stored tuple while `generate_rules` looks them up by the sorted tuple.
  When the antecedent lookup misses, the rule is dropped silently; when the consequent lookup misses, the rule is kept
  with lift forced to 0.0 (10,502 of the 29,540 GPU rules). The pinned `main` keys by the sorted tuple (its docstring
  records the fix).
- So four P-015 values come from that generator: all rules (29,540 vs 53,447), lift >= 5 (17,499 vs 51,126),
  lift >= 100 (9,283 vs 40,428) and son_run.py's cross-domain count (27,072 vs 50,977; RESULTS CPU-056), each
  reproduced exactly by the branch generator on the GPU table (`data/mined/R1/adjudication.json`).
- Consequence: the CPU values are the correct rule counts; the GPU values are artifacts of the rule generator, not of
  the miner (brief: "a rule-generator issue (see I-012)"). For repo-root I-012 this settles the rule counts: the old
  pipeline log's 53,447 / 51,126 / 40,428 equal the exact counts. Its cross-domain 45,286 is not reproduced (son_run.py's
  definition gives 50,977; the old definition is not in the repository). Verdict for R1: mismatch, adjudicated in favour
  of the CPU.

## CI-002 — G0 total: 167,042 here vs 168,674 on the GPU (not comparable, explained)

- GPU P-001 (`validate_row_split.log`): 168,674 itemsets, K <= 14, on the first 1,000,000 rows of the GPU run's
  `transactions_214m_base_multi.parquet`.
- CPU G0 (`data/mined/G0/G0.json`): 167,042, K <= 14, identical across in-core, SON (4 chunks) and efficient-apriori.
- The two first-million slices hold different proteins: the totals differ while all three CPU routes agree, and the
  rebuilt row order follows the joined pLDDT CSV, whose order comes from 16 parallel Storage Read streams (the sha256
  difference, CPU-020, is consistent with this but would also follow from the different parquet writer). The full
  set agrees with the GPU run in every aggregate statistic (CPU-017) and every full-table mining result matches
  (CPU-025..053); per-protein identity cannot be checked because the GPU file is not on this box. Not an error of
  either route; G0's GPU cross-check is recorded as not comparable.
