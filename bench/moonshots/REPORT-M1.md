# M1 — GPU-native input layer: report

**Decision: the device input layer stays and is the dispatch for every column
it serves.** It wins 11 of the 20 regimes by rule 2, loses none and ties the
other 9 (all under rule 2's 1 s). No signature mismatch, no logged fallback.
Integer item columns whose ids span at most 2²² values now reach the GPU
without the host CSR. Other dtypes, wider spans and `bitvecs=` keep today's
path. `ET_MINER_INPUT_LAYER=host|device` pins either path.

Protocol: `PROTOCOL-M1.md`. Data: `../results/2026-10-17-moonshots-m1/`
(`decision.md`, `split.md`, `raw.jsonl`). Code: `6884f78` on
`perf/moonshot-m1`.

## What changed

`src/et_miner/gpu/input_layer.py`: each Polars chunk's Arrow buffers are read
in place, and the ids (minus the smallest) are copied to the shard devices as
int32 through two pinned buffers. On the devices: the K=1 bincount, a lookup
table and the new kernel `ids_to_bitvec`. The row-wise K=2 rows are cut from
the same device entries, sorted and deduplicated there when a row needs it.
The bitvectors are the host path's bytes: tested on every preset and on
null, repeated and unsorted rows, every served dtype, the span limit, and row
counts off word boundaries, on 1 and 2 GPUs (`tests/test_input_layer.py`).
Tier legs pin each layer on 1 and 2 GPUs.

## Results (2× RTX 3060 / EPYC 7282, median of 3, seconds)

| regime | GPUs | host [min, max] | device [min, max] | device/host | input host → device | rule 2 |
|---|---|---|---|---|---|---|
| dsl | 1 | 29.53 [29.52, 30.08] | 10.60 [10.58, 10.65] | 0.36 | 20.69 → 1.91 | device wins |
| dsl, count inference | 1 | 24.80 [24.75, 25.04] | 5.74 [5.72, 5.80] | 0.23 | 20.78 → 1.90 | device wins |
| dsl, free sets | 1 | 24.59 [24.59, 24.73] | 5.25 [5.23, 5.31] | 0.21 | 21.10 → 1.91 | device wins |
| dsl | 2 | 26.35 [25.92, 26.42] | 6.86 [6.82, 6.86] | 0.26 | 20.86 → 1.81 | device wins |
| dsl, count inference | 2 | 24.12 [22.83, 24.20] | 4.32 [4.28, 4.37] | 0.18 | 21.07 → 1.80 | device wins |
| dsl, free sets | 2 | 23.46 [23.28, 23.65] | 4.34 [4.33, 4.36] | 0.19 | 20.91 → 1.81 | device wins |
| sk2ml3 | 1 | 27.52 [27.45, 27.57] | 22.90 [22.82, 22.93] | 0.83 | 4.69 → 0.32 | device wins |
| sk2ml3 | 2 | 20.37 [20.27, 20.40] | 15.39 [15.38, 15.39] | 0.76 | 4.83 → 0.35 | device wins |
| oom2ml3 | 1 | 2.69 [2.68, 2.89] | 1.63 [1.62, 1.63] | 0.61 | 1.00 → 0.09 | device wins |
| skew | 1 | 1.20 [1.16, 1.21] | 0.19 [0.19, 0.19] | 0.16 | 1.04 → 0.10 | device wins |
| oom2ml3 | 2 | 3.14 [3.10, 3.65] | 2.17 [2.17, 2.19] | 0.69 | 1.09 → 0.13 | tie (0.97 s saved) |
| skew | 2 | 1.65 [1.63, 1.66] | 0.68 [0.66, 0.68] | 0.41 | 1.13 → 0.15 | tie (< 1 s) |
| deepk | 1 / 2 | 0.90 / 1.37 | 0.17 / 0.67 | 0.19 / 0.49 | 0.75 → 0.10 / 0.84 → 0.14 | tie (< 1 s) |
| or003 | 1 / 2 | 0.20 / 0.66 | 0.10 / 0.57 | 0.53 / 0.87 | 0.10 → 0.01 / 0.13 → 0.05 | tie |
| or002 | 1 / 2 | 0.36 / 0.64 | 0.25 / 1.06 [0.56, 1.07] | 0.71 / 1.65 | 0.10 → 0.01 / 0.13 → 0.04 | tie |
| smoke | 1 / 2 | 0.08 / 0.62 | 0.05 / 0.58 | 0.63 / 0.94 | 0.04 → 0.01 / 0.08 → 0.05 | tie |

Where the host time went on dsl (1 GPU): host K=1 1.31 s, explode and join
5.64, COO to CSR 10.77, int64 copies 0.37, pageable copies 2.44, bitvector
kernel 0.17. The device path's 1.91 s is the chunk views 0.12, the
pinned copies 0.32, the bincount 0.04, the bitvectors 0.02, and 1.41 s
outside the timed phases (finding 1). The K≥2 levels are unchanged
(8.83 → 8.66 s).

Peak memory (max over reps): dsl's host RSS falls from 10.3 GB to 3.5 GB on
one GPU (10.7 → 3.7 GB on two). Its peak VRAM falls from 2.2 to 1.2 GB (no
int64 CSR on the device). sk2ml3 and oom2ml3 are unchanged within 0.3 GB.

Final check (`--mode moonshots-m1-final`, the layer unset): 20 of 20 ok. The
device layer ran in every one, signatures agree with the campaign, and the
walls match the device medians (dsl 10.68, sk2ml3 22.84 / 15.39). Budget used:
0.725 GPU-hours of 1, campaign and final check together.

## Findings

1. **(should)** `_int_ids` costs 1.0 s of dsl's 1.91 s device input. The span
   check runs Polars' `list.min` / `list.max` per row before the chunks are
   read. A min/max over the in-place Arrow values would take about 0.3 s
   (an `explode().min()/max()` takes 0.33 s here). That saves about 0.7 s on
   dsl (7 % of its 1-GPU wall, 13 % with count inference). It needs its own
   measurement.
2. **(should)** The campaign box changed after Phase 1, from 1× A4000 /
   Pentium G5420 to 2× RTX 3060 / EPYC 7282. Phase 3's "reduced matrix
   against Phase 0" must re-run `main`'s tree on the box it compares on. M1's
   own decision compares both arms on one box and is unaffected.
3. **(nice)** The device arm holds two 64 MB pinned buffers, so the host RSS
   of the small regimes rises by 70–130 MB (smoke 532 → 658 MB). oom2ml3 on two
   GPUs peaks 0.27 GB higher in VRAM, because the entries stay on the device
   until K=2.
4. **(nice)** On two GPUs the tail after the last level varies from 0.05 to
   0.53 s in both arms (smoke and or003: 0.25–0.39 s in both). That variance
   is what gives or002's 2-GPU device arm a higher median (1.06 [0.56, 1.07]
   against 0.64 [0.64, 0.65]): its levels and input are equal or faster, and
   the gap is all in the tail. The variance is not M1's, but the teardown
   could be timed as a span phase.
5. **(nice)** NVML reported the SW power cap (0x4) on 66 of the 140 rows (56 of
   the 120 campaign rows), in both arms, on the heavy regimes (dsl, sk2ml3, oom2ml3): the 3060's 150 W
   limit. It is recorded, not controlled.
6. **(nice)** Test infrastructure: a byte comparison of 8.75 GB bitvectors
   needs pinned staging. CuPy's pinned pool must be emptied between presets,
   or the stress_k2 buffers (17.5 GB locked) stall the next test.
   `tests/test_input_layer.py` does both.
