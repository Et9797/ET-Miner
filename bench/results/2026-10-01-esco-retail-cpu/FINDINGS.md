# Partial Retail baseline — stopped at user request

The benchmark was stopped before the second efficient-apriori run completed.
No benchmark child processes remain. These are CPU-only, single-run results
through K=2; they do not measure ESCO, which starts at K≥3. The three-repeat
matrix and GPU comparisons were not completed or gated.

| Support | Minimum count | CPU itemsets through K=2 | CPU sparse seconds | Oracle seconds | Agreement |
|---|---|---|---|---|---|
| 0.0001 | 4 | 2,678,808 | 197.144 | 86.979 | Same itemset/count hash and sum of counts |
| 0.00005 | 2 | 3,973,569 | 217.447 | Not completed | Not checked |

`raw.jsonl` preserves the completed runs, `env.txt` the revision, and
`dataset.json` the prepared dataset hash and full-lattice lower bounds.
CUDA and the Rust extension were unavailable in this environment.

The ESCO restoration and comparison harness remain available. See
`bench/ESCO.md` for the restored API, correctness gate and optional future
GPU runs. CPU regression checks passed; GPU execution remains unverified.
