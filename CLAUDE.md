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
- Use only intrinsics already present in the tree (all sm_60+):
  `__popcll`, `__shfl_down_sync`/`__shfl_sync`, `__ballot_sync`,
  `__activemask`, `__ffs`, 64-bit atomics. No CUB, no templates, no
  cooperative groups, no `memcpy_async`.
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

## OPERATING RULES — base214m reproduction (apply continuously)

These rules govern the AlphaFold base214m reproduction campaign. They were
copied verbatim from the campaign briefing so they survive context
compaction. Re-read them, then PROGRESS.md and RESULTS.md, after any
compaction.

- Permissions: auto mode is active for unattended runs; do not stop to ask
  for routine file/bash approvals. Only pause for genuinely destructive or
  ambiguous actions.
- Data locality: nothing leaves this machine. Do NOT upload any data or
  artifacts to GCS or any other external storage; the only external
  transfer is the INBOUND AlphaFold download from Google's public bucket.
  Keep all output parquet files, reports, and logs on local disk under a
  single run directory `runs/<UTC-timestamp>/`; Et will rsync them to his
  local machine after the run.
- Durable state: maintain these files on disk and update them as you go,
  because context will compact over a multi-hour run:
  * `PROGRESS.md` — current phase, last completed step, next step, and how
    to resume each long job (exact commands, PIDs, log paths). Use
    `[ ]` / `[x]` checkboxes.
  * `RESULTS.md` — every fresh reproduced value with the artifact
    path/command that produced it and a UTC timestamp.
  * `INCONSISTENCIES.md` — every discrepancy found between scripts, logs,
    the paper reviews, and the .tex.
  * `CLAUDE.md` — these OPERATING RULES live here so they survive
    compaction.
  After any compaction, re-read `PROGRESS.md` and `RESULTS.md` before
  continuing.
- Long jobs: run downloads, extraction, and mining as detached background
  processes writing to timestamped log files (e.g.
  `setsid <cmd> > logs/<step>.log 2>&1 &`), record the PID and log path in
  `PROGRESS.md`, and poll the log rather than blocking. Do not paste
  multi-GB output into the transcript (a background task emitting >5GB is
  auto-killed). Checkpoint so any step is resumable after an interruption.
- Traceability: every value that lands in `COMPARISON_REPORT.md` must point
  to a fresh artifact file or the command that generated it. If a value can
  only be sourced from an old log, mark it inconclusive — do not adopt it.
- Verification: show evidence (command + its output, or the output file
  path), never assert success. Verdicts:
  * deterministic values (pattern counts, support values, K, statistics) →
    must match the claim EXACTLY, else "hallucinated" (or "inconclusive" if
    you can't run that part).
  * hardware-dependent values (timings, throughput, proteins/min) →
    deviation is EXPECTED and must be labeled
    "expected-hardware-deviation", not "hallucinated", because the GPU here
    may not be the paper's H100. Report your value, the claimed value, and
    the hardware for both.
- Documentation style for any script you write or modify: a short
  self-contained module docstring (what it does + Usage + Options);
  implementation detail in function docstrings; NO conversational text or
  genesis/history in code comments.
- Use subagents for (a) codebase exploration so raw file dumps don't flood
  the main context, and (b) a fresh-context adversarial review of
  `COMPARISON_REPORT.md` before you call the goal done.

### Campaign context (from the briefing)

- Running directly on a vast.ai GPU instance inside tmux; no separate box
  to SSH into. No hard cost cap.
- The canonical run is "base214m": AlphaFold base-vocab over the full
  214M-protein set. The original result data was lost to disk corruption.
- Ground truth = fresh re-execution only. Values in the V1 paper
  (.tex/.pdf), the paper reviews, and the logs may all be hallucinated.
  Never treat any pre-existing number as correct; treat them only as
  CLAIMS to verify.
- Phases: 0 inventory & hardware/disk audit → 1 codebase exploration &
  claim extraction (`CLAIMS.md`) → 2 AlphaFold DB re-acquisition +
  base214m re-extraction (resumable; abort to `ABORT_REPORT.md` with `df`
  evidence if the disk plainly cannot hold the data) → 3 mining run
  reproduction → 4 per-value `COMPARISON_REPORT.md` with adversarial
  subagent review.

### Campaign status (updated 2026-09-02)

- The base214m reproduction ran to completion on 2026-09-02. Deliverables at
  the repo root: `COMPARISON_REPORT.md` (+ `COMPARISON_REPORT_rows.json`),
  `CLAIMS.md`, `RESULTS.md`, `INCONSISTENCIES.md`, `PROGRESS.md`; blog draft
  in `paper/blog_post.md`.
- Every fresh artifact, script, log and the verdict machinery is committed
  under `runs/20260902T0000Z/` (start at its `README.md`). Only raw inputs
  above GitHub's 100 MB limit are excluded; they are listed with size and
  sha256 in `runs/20260902T0000Z/EXCLUDED_LARGE_FILES.md` and are re-derivable
  with the scripts in `runs/20260902T0000Z/phase2/scripts/`.
- 2026-09-02 (later): `paper/et_miner_proteome.tex` and `paper/blog_post.md` were revised
  with the fresh values (see PROGRESS.md "Paper and blog revision"); the two result figures are
  regenerated by `paper/figures/make_figures.py` from the campaign JSON. Derived values used in
  the revision are RESULTS.md rows P-021..P-024, X-017..X-019, F-001.
- The operating rules above stay in force for any follow-up reproduction work.
