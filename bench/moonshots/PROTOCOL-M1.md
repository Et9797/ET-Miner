# M1 — GPU-native input layer: measurement protocol

Fixed before M1's first timed run. The configs come from
`bench/consolidation_matrix.py::build_moonshots_m1`; each runs through
`bench/runner.py --mode moonshots-m1` and `bench/consolidation_run.py`.
The design is `SPEC.md` § M1; the code is `src/et_miner/gpu/input_layer.py`.

## Box and budget

- 2× RTX 3060 12 GB (sm_86, `nvidia-smi topo`: NODE), driver 580.173.02
  (CUDA 13.0), CuPy 14.1.1 with NVRTC 12.9. Pinned host-to-device copies
  measured at 12.5 GB/s, pageable at 6.1–6.4 GB/s.
- AMD EPYC 7282 (16 cores, 32 threads, AVX2), 125 GB RAM, `/dev/shm` 25 GB.
  A shared host: the load average is about 6.5 from other tenants before any
  run, and the owner's `watch -n 0.1 nvidia-smi` polls the GPUs. Both are
  recorded, neither is controlled.
- **Not the Phase 0 box** (1× A4000, Pentium G5420). No number here is
  compared with Phase 0's; both arms run on this box in the same campaign.
- Budget: ≤ 1 GPU-hour (the owner's, per idea), counted as the runner counts
  it (process wall × devices). The campaign runs with `--max-gpu-hours 0.95`.
  Estimate: 120 configs, about 0.7 GPU-hours.

## Arms

| arm | pin | path |
|---|---|---|
| `host` | `ET_MINER_INPUT_LAYER=host` | today's: Polars explode + join, scipy COO→CSR, int64 copies, pageable copies, `csr_to_bitvec` |
| `device` | `ET_MINER_INPUT_LAYER=device` | Arrow buffers in place, pinned async copies of int32 ids, device bincount, `ids_to_bitvec`; row-wise K=2's rows cut on the device |

Every config also pins `POLARS_MAX_THREADS`, `RAYON_NUM_THREADS`,
`MKL_NUM_THREADS` and `OMP_NUM_THREADS` to 16 (the physical core count),
`NCCL_P2P_DISABLE=1`, and on one GPU `CUDA_VISIBLE_DEVICES=0` and
`ET_MINER_DISABLE_NCCL=1` (`_cfg`). Every other knob is unset: the kernels are
the measured dispatch's.

## Regimes

The Phase 0 regimes (`MOONSHOT_REGIMES`), on one GPU and on two:
smoke, deepk, skew, or003, or002, dsl, dsl with count inference, dsl with
free sets, oom2ml3, sk2ml3. A regime is (workload, GPUs, options).

## Repetitions and order

Fresh process per config; the child compiles every registered kernel on
every device it uses and runs the same route (same pins) on a 20,000-row
slice of smoke before the timed call, so the device arm's kernel compile and
first pinned allocation are warm. 3 reps, rep-major (every config's rep 0,
then every rep 1, then rep 2). Medians with [min, max]. No tracked file is
edited and nothing is committed while the campaign runs.

Recorded per row (`consolidation_run.py`): the wall of the `apriori()` call
(the dataset read is outside it), the level split and the input span
(`level_split.py`: the host phases, and for the device arm
`input_dev_chunks` / `input_dev_copy` / `input_dev_k1` / `input_dev_bitvec`),
peak VRAM per device, peak RSS, throttle reasons, logged fallbacks and the
result signature.

## Decision rule (the consolidation protocol's rules 1, 2, 4, 5; mechanical)

`bench/moonshots/decide.py RAW --arms host device` applies it.

1. **Signature mismatch or logged fallback** disqualifies the arm
   (must-fix): every row of an equivalence group (dataset, min_support,
   max_length, free sets) must carry one signature, across arms and GPU
   counts.
2. **Winning a regime**: the device arm's median wall is ≥ 10 % below the
   host arm's, the two [min, max] ranges do not overlap, and it saves ≥ 1 s.
4. **Partial winner**: if the device arm wins some regimes and loses
   others (the host arm wins by rule 2's three conditions), it stays only
   behind a dispatch rule from facts known before the build (the dtype, the
   id span, the row and entry counts, free VRAM); the crossover is measured
   and recorded. If it wins some regimes and loses none, it becomes the
   dispatch for every column it serves.
5. **Ties**: if it wins nowhere, the device arm goes (the host path serves
   every column with less code). On a tie, peak VRAM breaks it.

A config that errors (out of memory included) or times out cannot win its
regime; that is not a signature mismatch.

## Final check

After the decision: one rep of every regime with `ET_MINER_INPUT_LAYER` unset
(the chosen dispatch), same pins, on one and two GPUs, signatures checked
against the campaign's.

## Commands

```bash
uv run pytest tests/test_tier_equivalence.py -q && uv run pytest -q -m "gpu and not slow"
uv run python bench/selfcheck.py
OUT=bench/results/$(date +%F)-moonshots-m1
uv run python bench/runner.py --mode moonshots-m1 --out $OUT --max-gpu-hours 0.95
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms host device > $OUT/decision.md
```
