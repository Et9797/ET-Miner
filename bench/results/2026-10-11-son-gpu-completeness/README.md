# SON on the GPU at `local_support_factor` 0.95: completeness

The flag PR #31 left open: `use_gpu=True` got the new default 0.95 without a
completeness run (no GPU on that box). This closes the 1-GPU half of it.

- Script: `bench/son_gpu_completeness.py` (untimed). Per workload, it runs
  `apriori_streaming(use_gpu=True, local_support_factor=0.95)` with 4 chunks
  (`chunk_size = ceil(N / 4)`, as `son_stakes.py`). It compares the result as
  `{itemset: absolute count}` with efficient-apriori and with the in-core CPU
  route `apriori()`. It also checks that efficient-apriori and the CPU route
  agree.
  - efficient-apriori is called with the CLAUDE.md convention:
    `(min_count - 0.5) / N`, and an explicit `max_length` (the workload's, else
    the longest transaction).
- Workloads: `son_stakes.SON_WORKLOADS` and dslk2, the seven of Amendment 8.
- Tree: `82e664c`, clean (each row carries `rev`). 2026-10-11, one run.
- Box (`env.txt`): RTX 3060 12 GB (sm_86), Athlon 3000G (4 threads), 31 GB.
  `bench/selfcheck.py`: READY, 13 kernels compiled and launched.

## Result

Every workload matches exactly, with no itemset missing or extra and no count
different (`rows.jsonl`, `FAILS=0`).

| workload | rows | chunk | efficient-apriori | CPU route | GPU SON | GPU vs EA | GPU vs CPU | CPU vs EA | wall s (untimed) |
|---|---|---|---|---|---|---|---|---|---|
| smoke | 60,000 | 15,000 | 694 | 694 | 694 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1.78 |
| deepk | 1,000,000 | 250,000 | 8,841 | 8,841 | 8,841 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 18.74 |
| skew | 1,000,000 | 250,000 | 10,350 | 10,350 | 10,350 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 40.24 |
| wide | 100,000 | 25,000 | 3,872 | 3,872 | 3,872 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 74.85 |
| or005 | 36,422 | 9,106 | 10,488 | 10,488 | 10,488 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 31.43 |
| or0001k2 | 36,422 | 9,106 | 2,678,808 | 2,678,808 | 2,678,808 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 173.15 |
| dslk2 | 20,000,000 | 5,000,000 | not run | 1,143 | 1,143 | – | 0 / 0 / 0 | – | 274.35 |

Differences are given as missing / extra / wrong count.

dslk2 is not run on efficient-apriori. Its 20M transactions, and a tid set per
item, would be held as Python objects, an estimated size well past the box's
31 GB (an estimate, not a measurement). Its GPU rows are compared with the CPU
route only. The tier-equivalence gate ties that route to efficient-apriori on
the smoke preset.

`rows-superseded-939e01b.jsonl` holds the first run, at `939e01b`. That run
compared against the CPU route only, and was replaced because CLAUDE.md makes
efficient-apriori the oracle of every validation run. Its itemset counts are the
same as this run's.

## Not covered

- **Timing.** The wall column is not a measurement: there is no protocol
  amendment and only one rep. It shows the order of magnitude only.
  - At 0.95, F1's CPU SON medians on the 12-thread box
    (`2026-10-08-son-factor/FINDINGS.md`, T1 / T4) were:
    - wide: 0.96 / 0.46 s;
    - or0001k2: 2.75 / 2.84 s;
    - dslk2: 11.58 / 8.43 s.
  - The GPU passes take minutes on these three. They build a Polars boolean
    matrix per chunk, keep the union as a Python `set` of tuples, and count all
    of it in pass 2, with no bound. Adding that bound is handoff item 4.
- **Two GPUs.** `apriori_streaming_multi_gpu` and the 2-GPU tier legs need a
  second device.
