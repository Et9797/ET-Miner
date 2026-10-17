# M2 — Row-wise K=3: report

**Decision: row-wise K=3 stays, behind the r3 dispatch.** It wins 4 of the 20
regimes by rule 2 (sk2ml3 and oom2ml3, on one and two GPUs) and loses none.
On dsl it is 0.03–0.56 s slower, a tie under rule 2's 1 s; the dispatch
leaves dsl on the dense kernels. The final check shows the dispatch picking
rows in exactly the four winning regimes and dense in the other 16. No
signature mismatch, no logged fallback. M10 was not built: row-wise K=3
removed its stake (`PROTOCOL-M2.md`).

Protocol: `PROTOCOL-M2.md`. Data: `../results/2026-10-17-moonshots-m2/`
(`decision.md`, `split.md`, `raw.jsonl`, `k3_crossover.jsonl`). Code:
`313dfc9` (kernel, dispatch, tests) and `902bbad` (the measured crossover),
on `perf/moonshot-m2`, which stacks on M1.

## What changed

- `src/et_miner/gpu/kernels/k3_rows.py` and `_src/k3_rows.cu`: each shard's
  rows count K=3 into a slot layout by pair:
  - pair (u, v) owns one slot per frequent partner w of v;
  - the slot's rank, and the pair's own index, come from a frequent-pair
    bitmap with per-word ranks, built on the device;
  - one warp per row stages the row's active positions and builds 64-bit
    neighbour masks, then spreads the row's frequent pairs over its lanes.
- The row source (M1's `DeviceRows`, or the host CSR's `_HostRows`) keeps its
  entries past K=2 when row-wise K=3 may run.
- Dispatch (`row_split_chunks.k3_counts_rows`): rows when r3 = Σ_rows C(L, 2)
  / (generated candidates × words) is below `K3_ROWS_MAX_R = 2.5e-4` and the
  slots fit one chunk. Rows of more than 64 active items,
  `prune_apriori=False`, `bitvecs=`, ESCO and a resume at K=3 count from the
  bitvecs. `ET_MINER_K3_KERNEL=dense|rows` pins it.
- Tests: `tests/test_k3_rows.py` checks the kernel against brute force:
  - with the bitmap and with the search path;
  - over slot ranges;
  - on rows of 64 items;
  - for shard sums.

  End to end it checks against efficient-apriori, on 1 and 2 GPUs with both
  input layers:
  - ties at min_count, empty rows, repeated items, row counts off 64 / 256;
  - free sets, inference and forced chunks against the dense level.

  Tier legs: rows pinned on 1 and 2 GPUs (plain, forced chunks, count
  inference), and rows by the r3 dispatch.

## Results (2× RTX 3060 / EPYC 7282, median of 3, seconds)

| regime | GPUs | dense [min, max] | rows [min, max] | rows/dense | K=3 level dense → rows | rule 2 |
|---|---|---|---|---|---|---|
| sk2ml3 | 1 | 22.71 [22.70, 22.82] | 1.81 [1.76, 1.81] | 0.080 | 21.57 → 0.62 | rows win |
| sk2ml3 | 2 | 15.38 [15.38, 15.40] | 1.87 [1.87, 2.37] | 0.122 | 13.87 → 0.55 | rows win |
| oom2ml3 | 1 | 1.62 [1.62, 1.63] | 0.38 [0.37, 0.38] | 0.235 | 1.32 → 0.08 | rows win |
| oom2ml3 | 2 | 2.16 [2.16, 2.17] | 1.16 [1.15, 1.17] | 0.534 | 1.14 → 0.09 | rows win |
| dsl | 1 / 2 | 10.62 / 6.86 | 11.18 / 7.34 | 1.05 / 1.07 | 0.14 → 0.73 (1 GPU) | tie (< 1 s) |
| dsl, count inference | 1 / 2 | 5.78 / 4.32 | 6.34 / 4.83 | 1.10 / 1.12 | | tie (< 1 s) |
| dsl, free sets | 1 / 2 | 5.26 / 4.32 | 5.82 / 4.35 | 1.11 / 1.01 | | tie (< 1 s) |
| deepk, skew, smoke, or003, or002 | 1 / 2 | ≤ 1.07 | ≤ 1.06 | 0.99–1.21 | | tie |

On sk2ml3 (1 GPU) the row-wise level is:
- the kernel, 0.39 s;
- graph and bitmap 0.03, filter 0.02, decode 0.08, the rest 0.10.

The dense level was 17.5 s in the tiled kernel and 3.4 s fused. With K=3 at
0.6 s, sk2ml3's wall is now mostly K=2 (0.72 s) and the input (0.31 s).

Peak VRAM falls with rows, because no dense count chunks are allocated:
- oom2ml3: 7.86 → 1.97 GB on one GPU;
- sk2ml3: 11.90 → 10.51 GB on one GPU, 10.81 → 7.28 GB per GPU on two.

Host RSS is unchanged.

Final check (`--mode moonshots-m2-final`, the knob unset): 20 of 20 ok,
signatures consistent with the campaign. Each row's level split shows the
K=3 kernel the dispatch chose: rows for sk2ml3 and oom2ml3 (1.81 / 1.88 and
0.39 / 1.18 s), dense everywhere else.

Budget used: 0.649 GPU-hours of 1 (sweep 0.141, campaign 0.441, final check
0.067).

## The crossover sweep

18 uniform points (N = 1M, F = 50–1,600 items, L = 8–32, nearly every pair
frequent):
- Rows win every point below r3 = 1.69e-4, by 0.007–0.19×.
- Above that the points cross more than once, in r3 order:
  - F800-L32 (3.7e-4) is a loss at 1.05×;
  - F400-L16 (7.3e-4) and F200-L8 (1.4e-3) are wins at 0.79× and 0.27×;
  - F400-L32 (3.0e-3) and F200-L16 (5.9e-3) are losses at 6.0× and 2.0×;
  - F100-L8 (1.1e-2) is a tie at 0.94× (21 against 22 ms);
  - every point from 2.4e-2 is a loss.
- Following the protocol, `K3_ROWS_MAX_R` takes the lowest crossing, 2.5e-4.

The matrix regimes sit far from it: sk2ml3 1.5e-6 and oom2ml3 6.1e-6 (rows),
dsl 0.41 and deepk 0.28 (dense).

## Findings

1. **(should)** r3 misses the row-wise kernel's second cost, the increments.
   Long rows over a dense frequent-pair graph spend their time on row triples
   (F800-L32: 5e9 increments, 3.9 s), which r3 does not count. Two facts known
   before K=3 estimate the increments:
   - the K=2 level's summed counts P2;
   - the row triples T3 = Σ_rows C(L, 3).

   With ρ = P2 / R2, ρ² · T3 predicts sk2ml3's 375 M increments as 409 M and
   oom2ml3's 19.1 M as 18.7 M; in the sweep ρ ≈ 1 and the increments are T3.
   A statistic (R2 + ρ² T3) / (slots × words) orders the sweep's points far
   better than r3. Its single down-crossing is a 21 ms tie. It would admit the
   sweep's mid-range wins that 2.5e-4 leaves dense. It needs its own sweep and
   protocol.
2. **(should)** The rows' input is now the slowest part of a row-wise dsl K=3
   (`count_rows` 0.64 s for 20 M rows) and of the dispatch. Computing r3 over
   dsl's active rows costs 0.04–0.05 s per run in the dense regimes
   (`k2_dispatch` at K=3; 1 % of dsl's wall).
3. **(should)** CuPy's memory pool fragments across calls in one process.
   This predates M1:
   - a second `apriori(stress_k2, max_length=2)` on one GPU raises
     OutOfMemoryError, host and device input layer alike;
   - small allocations split the first run's cached 8.75 GB block, and its
     remainder cannot be released.

   Freeing the pool's cached blocks at the start of a run would fix it. The
   campaigns run one config per process and do not see it.
4. **(nice)** The 2-GPU tail varies again (0.09–0.53 s on dsl-infer, in both
   arms, as in M1's finding 4). The final check's dsl-infer-C2 row (4.85 s)
   carries it: its levels match the dense arm's.
5. **(nice)** Process: the sweep ran before `PROTOCOL-M2.md` existed. Its
   procedure was in `k3_crossover.py`'s docstring, committed at `313dfc9`
   before it ran. The protocol was written during the sweep and committed
   before the campaign.
