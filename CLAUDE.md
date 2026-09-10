# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Correctness policy (mandatory)

- **efficient-apriori is the canonical correctness oracle** for every smoke
  and validation run, now and in the future.
- The **tier-equivalence chain** must hold on the `smoke` synthetic preset
  (`et_miner.synthetic.PRESETS["smoke"]`):

  Tier 1 Polars == Tier 2 Rust (`sparse=True`) == single-GPU legacy ==
  multi-GPU legacy == shared multi-GPU == single-GPU sparse CSR
  (`sparse_from_k=3`) == multi-GPU sparse CSR == efficient-apriori

  It is enforced by `tests/test_tier_equivalence.py` (exact itemsets AND
  absolute counts) and runs FIRST in `bench/run_smoke.sh` and
  `bench/run_full.sh` — a divergence hard-fails the whole run.
- Never skip, weaken, tolerance-relax, or replace these assertions with
  count-only checks. Never mark them xfail to get a run green.
- Oracle call convention (both are load-bearing):
  - `min_support = (min_count - 0.5) / n_rows` — the miner keeps
    `count >= ceil(s·N)` while efficient-apriori keeps
    `support >= min_support` in float; the −0.5 makes the boundary exact
    for integer counts.
  - an explicit `max_length` — efficient-apriori silently defaults to
    `max_length=8` and would truncate deeper itemsets from the oracle.
- Planted-motif ground truth (`deep_k` and `smoke` presets): assert mined
  support `>=` the planted count, never `==` (background noise can only
  add occurrences). Preset self-checks reject specs whose planted support
  falls below `min_count` (vacuous recovery).

## CUDA kernel constraints (`src/et_miner/gpu/kernels/_src/*.cu`)

- Plain C, `extern "C"`, compiled per-device at first use via
  `cupy.RawKernel`/NVRTC with **no arch flags** — sources must build on
  sm_86 (RTX 3090) as well as sm_90 (H100/H200).
- The rule is the *intent*, not the list: an intrinsic is allowed if it is
  **sm_60+ and NVRTC-compilable with no arch flags**. What is in the tree today
  is `__popcll`, the 32-bit `__popc` (`_src/csr_warp.cu`,
  `_src/compact_threshold.cu`), `__shfl_down_sync`, `__shfl_sync`,
  `__ballot_sync`, `__activemask`, `__ffs` (`_src/compact_threshold.cu`),
  `__ffsll` (`_src/bitvec_extract_tids.cu`), and 64-bit atomics
  (`atomicAdd`/`atomicOr` on `unsigned long long`). No CUB, no templates, no
  cooperative groups, no `memcpy_async`. Check a new one against the rule; do
  not treat this enumeration as closed.
- Static shared memory only, ≤ 48 KB per block (the sm_86 static limit;
  dynamic-shmem opt-in is not used).
- Counting kernels assume `blockDim.x == 256`, coupled to
  `__shared__ unsigned long long warp_sums[8]` — change both or neither.
- Register every new kernel in `gpu/kernels/loader.py::_KERNEL_FILES`;
  `bench/selfcheck.py` compiles and smoke-launches every registered kernel
  on-device before any campaign work.
- Dense row-split count arrays are **int32** (counts are bounded by
  `n_transactions`, guarded to < 2³¹) — keep new dense outputs int32 and
  widen on the host after filtering.

## Dev commands

- Environment: `uv venv && uv sync` (dev group included).
- Rust extension: `cd rust_ext && maturin develop --release`.
- Tests: `uv run pytest -q` — gpu-marked tests auto-skip without a CUDA
  device; `-m "not slow"` skips long ones.
- Lint: `uv run ruff check src tests` (rule set pinned in pyproject).
- All `ET_*` environment knobs are documented in `src/et_miner/_env.py`.
- GPU campaign: `bench/README.md` (setup → selfcheck → smoke gate → matrix).
