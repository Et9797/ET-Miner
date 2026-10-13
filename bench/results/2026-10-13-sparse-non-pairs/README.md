# The sparse path's non-pair counter: array miner vs Rust vs scipy

`count_support_batched`'s sparse path counts every itemset that is not a pair
(K=0, K=1, K≥3) in `sparse._count_support_sparse_non_pairs`. PR #33 removed
the Rust counter that path used when the extension was built, which left
scipy (one column slice and row sum per itemset). PR #34 counts those
itemsets on `cpu_miner.count_itemsets` instead. This run times all three on
the same itemsets and checks that they return the same counts.

- Script: `bench/cpu/sparse_non_pairs.py` (seed 7). Three boolean matrices:
  - small: 3,000 × 14, all combinations of K=0, K=1 and K=3..6, plus edge
    cases;
  - mid: 200,000 × 40, 20,000 sampled K=3 and 20,000 sampled K=4 itemsets;
  - wide: 500,000 × 120, 30,000 sampled K=3 itemsets.
- Each counter ran at `n_jobs` 1 and 4, one run each, one after another on an
  otherwise idle box (`env.txt`: Ryzen 5 4600G, 12 threads, 30 GB).
- Trees and counters:
  - `2343acc` (PR #34): the array miner.
  - `76bdaa4` with `et_miner_rust` 0.3.0 built: the Rust counter.
  - `25502a0`, whose `sparse.py` is the one on `main` at `b0cf0a6`: scipy.
- Each row carries a signature of the result: the number of distinct
  itemsets, the sum of counts, and a sha256 of the sorted (itemset, count)
  pairs.

## Result

All three counters return the same counts in every case: one signature per
(case, `n_jobs`) in `rows.jsonl`.

| case | itemsets | n_jobs | scipy s | Rust 0.3.0 s | array miner s |
|---|---|---|---|---|---|
| small | 6,389 | 1 | 1.726 | 0.033 | 0.015 |
| small | 6,389 | 4 | 1.453 | 0.010 | 0.014 |
| mid | 29,880 | 1 | 176.113 | 4.220 | 0.341 |
| mid | 29,880 | 4 | 57.675 | 1.117 | 0.352 |
| wide | 30,000 | 1 | 399.795 | 8.035 | 0.810 |
| wide | 30,000 | 4 | 124.548 | 2.067 | 0.404 |

- The array miner is 164–516× faster than scipy on mid and wide.
- Against the Rust counter it is 3.2–12.4× faster on mid and wide, but slower
  on small at `n_jobs` 4 (0.014 against 0.010 s).
- On mid, `n_jobs` 4 gains nothing: these levels take the per-candidate
  AND+popcount (`count_per_candidate`), which runs on the calling thread; only
  the bitvector build uses the pool.

These are synthetic matrices with uniform density. No route calls
`count_support_batched`, so there are no campaign workloads to compare with.
