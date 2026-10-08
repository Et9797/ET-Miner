# SON on the array miner: Phase S1 (confirmation)

Protocol: `bench/cpu/PROTOCOL.md`, Amendments 3 and 4. Harness:
`bench/cpu/son_stakes.py`, arms `built` (`apriori(streaming=True)`) and
`incore`, 3 reps, cap 600 s, 4 chunks, the S0 box (`env.txt`). Rows:
`raw.jsonl` (72, all ok), every row at `58f1bb6` with a clean tree. The
tier-equivalence chain, the SON, streaming and free-set tests passed on that
tree before the run.

## A first run, superseded

The first S1 run, at `2833775`, is kept in `superseded-2833775.jsonl`. It
matched the S0 stakes everywhere but or0001k2, where `built` took 17.6–17.9 s
against 7.1–7.4 s for the stake: the union of the local results was merged
chunk by chunk, re-sorting the growing union every time (13 of 18 s in
`unique_rows`), and pass 2's K=2 found its columns with `np.unique` over every
pair. `58f1bb6` holds pending rows until they outgrow a budget and the union
(`UNION_PENDING_BYTES`), and finds the columns with a mask. The run below
repeats S1 in full on that tree.

## Correctness

One signature per workload across S0, the superseded run and this one.

## Results

Medians over 3 reps with [min, max]; `current` and the array stakes are S0's
single values.

| regime | S0 current | S0 array | S0 array-pc | S1 built | built peak RSS | S1 incore |
|---|---|---|---|---|---|---|
| smoke T1 | 1.47 s / 177 MB | 0.17 s | 0.16 s | 0.15 s [0.15, 0.16] | 130 MB | 0.08 s |
| smoke T4 | 0.96 s / 198 MB | 0.17 s | 0.14 s | 0.14 s [0.14, 0.15] | 136 MB | 0.07 s |
| deepk T1 | 31.17 s / 374 MB | 4.93 s | 5.18 s | 4.75 s [4.68, 4.76] | 287 MB | 2.07 s |
| deepk T4 | 15.14 s / 386 MB | 4.52 s | 4.66 s | 4.20 s [4.18, 4.20] | 335 MB | 1.64 s |
| skew T1 | 157.01 s / 550 MB | 31.87 s | 42.96 s | 31.15 s [31.05, 31.41] | 382 MB | 2.68 s |
| skew T4 | 78.97 s / 552 MB | 15.37 s | 35.19 s | 15.09 s [15.04, 15.17] | 437 MB | 1.75 s |
| wide T1 | 267.89 s / 1,361 MB | 1.21 s | 1.24 s | 1.15 s [1.14, 1.17] | 250 MB | 0.84 s |
| wide T4 | 235.92 s / 1,732 MB | 0.66 s | 0.65 s | 0.58 s [0.56, 0.62] | 226 MB | 0.37 s |
| or005 T1 | cap | 21.83 s | 13.47 s | 13.88 s [13.87, 13.97] | 258 MB | 0.45 s |
| or005 T4 | cap | 21.18 s | 13.20 s | 13.40 s [13.39, 13.46] | 283 MB | 0.28 s |
| or0001k2 T1 | cap | 7.13 s | 7.12 s | 7.91 s [7.89, 7.96] | 768 MB | 0.83 s |
| or0001k2 T4 | cap | 7.34 s | 7.39 s | 8.01 s [7.98, 8.02] | 853 MB | 0.57 s |

## Rule 4

- **Met.** `built` meets rule 2 against `current` in every regime where S0's go
  held: 3.6× (deepk T4) to 409× (wide T4) faster where `current` finished, and
  7.9–13.9 s where `current` hit the 600 s cap. smoke T4 is outside the go,
  as in S0: 6.6× faster but 0.82 s saved, under the 1 s floor.
- **Memory.** Peak RSS is 0.13–0.87× of `current`'s wherever `current`
  finished. Where it hit the cap no comparison exists; or0001k2 holds 5.4M
  candidate pairs for pass 2 (768–853 MB, against 332–384 MB in-core).
- **The counter dispatch (Amendment 4)** lands on the faster stake per regime:
  skew 31.2/15.1 s against `array`'s 31.9/15.4 s (prefix groups), or005
  13.9/13.4 s against `array-pc`'s 13.5/13.2 s (per candidate). or0001k2 (no
  K≥3) is 0.6–0.8 s above the stakes; a diagnostic run outside the protocol
  puts the difference in the union (4.9 s against the stake's 4.1 s) and in
  pass 2 (1.8 s against 1.6 s).

## What SON costs against the in-core route

SON stays 1.4–48× slower than mining in core (or005: 13.4–13.9 s vs 0.3–0.5 s).
That is SON's own work: four chunks at 0.9 × the threshold give pass 1 1.13M
local candidates on or005 against 10,488 frequent itemsets, and pass 2 counts
every candidate in every chunk. SON earns its place when the data does not fit
in memory, not as a faster route.
