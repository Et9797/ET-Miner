# Route optimizations, phase B (O5) — report

Phase B of `SPEC.md` on `perf/route-optimizations-b`. Protocol, fixed before the
first timed run: `PROTOCOL-B.md`. O4 was deferred before it (SPEC, O4
amendment). Evidence: `bench/results/2026-10-06-o5-calibration/`: `raw.jsonl`,
`env.txt` and `bandwidth.jsonl`, all at `210af6f`. The offline stakes are in
`bench/results/2026-10-06-phase-b-stake/`.

Box: 2× RTX A4000 16 GB (sm_86, `NCCL_P2P_DISABLE=1`), Ryzen 5 5600X, 46 GB
RAM, idle throughout.

## Outcome

O5 ends after step 1, on the owner's decision (2026-10-06). ESCO from K=3 lost
all four calibration regimes by 4.7× to 6.8× (reading 3), so there is no
crossover for a cost model to place. `"auto"` stays as it is. O4 stays deferred:
its condition, ESCO levels that win on time but do not fit, did not arise.
Reading 4 applies: the CSR kernel took 26× to 212× its intersection model.

## Calibration (one rep, knobs at their defaults)

| regime | dense | esco | esco / dense |
|---|---|---|---|
| oom2ml3, 1 GPU | 1.79 s | 12.12 s | 6.8× |
| oom2ml3, 2 GPUs | 2.64 s | 15.62 s | 5.9× |
| sk2ml3, 1 GPU | 20.55 s | 97.10 s | 4.7× |
| sk2ml3, 2 GPUs | 16.10 s | 101.72 s | 6.3× |

- Every run was ok, with no fallback and no cap hit. Every ESCO run converted
  at K=3. The signatures agree: 8 configs in 2 groups, every row at `210af6f`.
- The K=3 level's split, in seconds:

| regime | dense K=3 | esco K=3 | conversion (`transition`) | `count_csr` | reduce |
|---|---|---|---|---|---|
| oom2ml3, 1 GPU | 0.97 | 11.23 | 1.16 | 9.97 | – |
| oom2ml3, 2 GPUs | 0.87 | 13.78 | 0.59 | 10.17 | 2.92 |
| sk2ml3, 1 GPU | 17.09 | 93.69 | 18.37 | 74.76 | – |
| sk2ml3, 2 GPUs | 11.88 | 97.47 | 8.94 | 71.41 | 16.53 |

## Where ESCO's K=3 time went (reading 4)

The copy bandwidth was 390.1 GB/s on both GPUs (`bandwidth.jsonl`). The
intersection model (`../results/2026-10-06-phase-b-stake/k3_cost.txt`: sparse
merge bytes, split over the GPUs, at that bandwidth) allows 0.10 s and 0.05 s
on oom2ml3 and 2.90 s and 1.45 s on sk2ml3. `count_csr` took 104×, 212×, 26× and
49× that, past the protocol's 2×.

The pattern points at the enumeration of the generated candidates, not at the
intersections:

- Per generated candidate the kernel costs about the same on both workloads:
  4.8 ns (oom2ml3, 2.08 B generated) and 6.3 ns (sk2ml3, 11.8 B).
- Per candidate that survives the subset test it does not follow the bytes:
  sk2ml3 reads 3.3× the bytes per survivor of oom2ml3 (7.8 KB against 2.4 KB)
  and costs 0.82× as much (516 ns against 630 ns).

## Findings

1. **(should) The CSR kernel spends its time on candidates the subset test
   rejects.** It tests every generated candidate one by one; the tiled dense
   kernel skips whole tile-pairs that are prunable (96.95 % and 96.4 % of them
   at K=3 on oom2ml3 and sk2ml3, phase 0). A skip of prunable candidates in the CSR kernel is its
   own item, per the protocol.
2. **(should) That skip alone would not make ESCO win here.** Two more costs
   stand in the way:
   - The conversion took 18.37 s on sk2ml3 on one GPU, more than the whole
     dense K=3 level (17.09 s), and 8.6× its own bandwidth model on both
     workloads (sk2ml3: 1,660,332 pairs × 2 columns × 31,250 words × 8 B =
     830 GB of AND reads, 2.13 s). The row-wise K=2 already holds each shard's
     rows as a CSR of frequent positions, and the pairs' tidsets could be
     written from it: on sk2ml3 one pass over Σ_rows C(len, 2) = 563 M pair
     occurrences, writing the frequent pairs' 169 M tids (0.68 GB).
   - On two GPUs the ESCO level's reduce took 16.53 s on sk2ml3 against 1.07 s
     for the dense level's compacted reduce (compact 0.53, reduce 0.54): the sparse path reduces whole
     chunk arrays over every generated candidate. O2's compaction does not
     apply to it.
3. **(nice) Thermal slowdown** (reason 0x20) hit the two long ESCO runs on
   sk2ml3, as it hit sk2ml3 in phase A. It cannot account for factors of 4.7
   to 6.8.
4. **(nice) The dense sk2ml3 runs peak near the card's size:** 15,229 MB on one
   GPU and 15,799 MB per GPU on two (`nvidia-smi` sampled every 0.5 s).

## Open (the owner's call)

With all three fixes (the skip, the conversion from the rows, the compacted
reduce on the sparse path), the models leave sk2ml3's ESCO K=3 at the
intersections (≥ 2.9 s on one GPU) plus the conversion from the rows plus an
unknown enumeration cost, against 17.09 s dense. Whether that becomes a SPEC
item ("ESCO on K=3 explosions") or phase C (O6, O7) goes first is open.

## Budget

0.128 GPU-hours for the eight configs (process wall × devices) plus the
bandwidth probe (< 0.01), of the 0.2 agreed. The offline stakes used no GPU
time.
