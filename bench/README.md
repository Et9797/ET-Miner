# GPU benchmark & validation campaign

Scripted campaign for a rented multi-GPU box (designed for 2× RTX 3090,
24 GB, CUDA 12). Everything is driven by three commands; credits only burn
while they run.

## Renting the box (vast.ai)

- **Image**: prefer a CUDA **12.x devel** image (e.g.
  `nvidia/cuda:12.4.1-devel-ubuntu22.04`), host driver **≥ 525**. Runtime or
  driver-only images ship **no CUDA headers**, which NVRTC needs to compile
  the kernels — the gpu extra now installs them via pip
  (`cupy-cuda12x[ctk]`) and `setup_box.sh` probes a trivial kernel compile
  and self-heals, so a runtime image works too; devel just skips that step.
- **`--shm-size` ≥ 2 GB** (vast.ai: the "docker options"/shm setting). Docker's
  default `/dev/shm` is 64 MB, and without P2P NCCL rides its SHM transport —
  too-small shm shows up as hangs or `NCCL WARN SHM` errors. `selfcheck.py`
  warns if `/dev/shm` is small; `NCCL_SHM_DISABLE=1` forces the socket
  transport as a last resort (slower but functional).
  With P2P available through the CPU's host bridge only (`nvidia-smi topo -m`
  shows `PHB`), NCCL's P2P transport hung about one two-GPU run in three at
  the first collective on a Ryzen AM4 box with two RTX A4000s;
  `NCCL_P2P_DISABLE=1` (the SHM transport) fixed it
  (`results/2026-09-28-consolidation-2gpu/nccl-hang/README.md`). On the same
  box plain device-to-device copies do not land either (small ones do, which
  is why no probe can tell); the miner therefore stages every cross-device
  copy through host memory (`gpu/nccl.py::copy_between_devices`,
  `ET_MINER_DIRECT_D2D=1` opts back in) and creates NCCL under
  `NCCL_P2P_LEVEL=NVL` unless you set `NCCL_P2P_LEVEL` or
  `NCCL_P2P_DISABLE` yourself. The campaign runner sets `NCCL_P2P_DISABLE=1`
  explicitly so every row uses one transport; that variable governs NCCL
  only, CuPy copies from older revisions still write P2P.
- **Disk ≥ 40 GB** (datasets + wheels + rust build), **host RAM ≥ 32 GB** —
  a count slice that does not fit the device is filtered on the host
  (up to 24 B/element per 64M-element slice).
- Prefer "dedicated GPU" listings for benchmark stability.

## Quickstart

```bash
git clone https://github.com/Et9797/ET-Miner && cd ET-Miner
git checkout <campaign branch>
bash bench/setup_box.sh          # env + rust ext + selfcheck + datasets (~10 min)
bash bench/run_smoke.sh          # ~30-60 min: tier gate → gpu tests → small matrix
# (report issues back, pull fixes, re-run smoke if needed)
bash bench/run_full.sh           # ~2-4 h: gate → full gpu tests → full matrix
```

Copy `bench/results/` back afterwards — it holds the raw JSONL, the
markdown report, and the environment capture. Datasets under
`datasets/synth/` are seeded and regenerable; don't bother copying them.

## What the gate enforces (see /CLAUDE.md)

`run_smoke.sh` and `run_full.sh` FIRST run `tests/test_tier_equivalence.py`:
Tier 1 Polars == Tier 2 Rust == row-split 1 GPU (measured dispatch, tiled
pinned, per-candidate pinned, forced chunks) == SON 1 GPU == row-split 2 GPUs
(plain and forced chunks) == SON 2 GPUs == **efficient-apriori**, exact
itemsets and counts, on the
`smoke` synthetic preset. Any divergence aborts the run — no benchmark
number is worth recording from a miner that disagrees with the oracle.

## Knobs the matrix drives (documented in `src/et_miner/_env.py`)

ESCO's `sparse_from_k` argument is restored: `None` keeps dense bitvectors,
`"auto"` switches below the measured n/32 mean-count crossover, and an int
fixes the transition level (at least K=3). The transition is one-way.
The `esco` and `esco-retail` comparison matrices and low-support Retail
validation are described in [ESCO.md](ESCO.md).

| Env | Values | Meaning |
|---|---|---|
| `ET_MINER_DISABLE_NCCL` | `1` | force the staged D2D reduce |
| `ET_MINER_MAX_CHUNK_CANDS` | int | force multi-chunk runs |
| `ET_MINER_TILED_MIN_GROUP_PAIRS` | int | pins the pairs a prefix group needs for the tiled kernel (0 = tiled everywhere; unset = the measured crossover per K) |
| `ET_MINER_DISABLE_RUST` | `1` | every Rust role takes its fallback (read once, at import) |

## Consolidation campaign

`bench/runner.py --mode consolidation` runs the GPU-layer consolidation matrix
(`bench/consolidation_matrix.py`; protocol and decision rule in
`bench/consolidation/PROTOCOL.md`). That matrix is the record of the Phase 2
campaign and runs only at its revision (`e4bb3ae`): the routes and parameters
it compared are gone from the tree, and a config that names one fails.
`--mode verify` is the re-run on the consolidated tree (the surviving routes,
each kernel pinned where the dispatch picks); results and the decisions are in
`bench/results/2026-09-2*-consolidation*/` and `bench/consolidation/REPORT.md`.
Each config names its route (C, C-bitvecs, D, E, F — see
`bench/consolidation_run.py`), pins every thread pool and kernel knob it
depends on, warms up on every device it uses, and records
per-level (per-pass for SON) times, per-device peak VRAM, peak RSS, throttle
reasons, the result signature and any logged fallback (which fails the
config). Every config of one (dataset, min_support, max_length, free-sets)
group must produce the same signature.

```bash
uv run python -m et_miner.synthetic --preset all --out datasets/synth
uv run python datasets/prepare_online_retail.py
OUT=bench/results/$(date +%F)-consolidation
uv run python bench/runner.py --mode consolidation --out $OUT --max-gpu-hours 7.5
uv run python bench/consolidation_report.py --out $OUT
```

The Rust host roles are also timed per call on real level arrays:

```bash
export RAYON_NUM_THREADS=6
MB=$(mktemp -d)   # level arrays; tens of MB, not committed
uv run python bench/microbench_rust.py dump deep_sparse_large $MB
uv run python bench/microbench_rust.py dump deep_sparse_large $MB --free-sets
uv run python bench/microbench_rust.py dump stress_k2 $MB --max-length 2 --then-k3 --n-gpus 2
uv run python bench/microbench_rust.py dump stress_k2 $MB --max-length 2 --free-sets --n-gpus 2
uv run python bench/microbench_rust.py time $MB > $OUT/microbench.jsonl
```

`--mode waste` measures what the K≥3 levels count needlessly: each level's
time split and the lattice dumps that `bench/candidate_waste.py` classifies
(`results/2026-10-05-candidate-waste/FINDINGS.md` has the commands).
`--mode pruning` compares the device-side subset test and count inference with
counting every candidate (`pruning/PROTOCOL.md`, `pruning/REPORT.md`).
Phase A of the route optimizations (`optimizations/PROTOCOL.md`) runs in this
order on a frozen tree, every step writing to the same per-revision directory:

```bash
uv run python bench/runner.py --mode optimizations-calibration
uv run python bench/k2_crossover.py            # the K=2 sweep; prints r*
uv run python bench/runner.py --mode optimizations --max-gpu-hours <what is left>
uv run python bench/runner.py --mode optimizations-final   # on the decided tree
uv run python bench/optimizations/decide.py <campaign raw.jsonl> <final raw.jsonl>
```

The outcome is `optimizations/REPORT.md`.

`ET_BENCH_ALPHAFOLD=/path/to/base214m.parquet` (a parquet with an `items`
list column) makes the dataset name `alphafold` available to consolidation
configs; nothing in the matrix uses it unless a config names it.

## Troubleshooting

- `selfcheck.py` failing on kernel compile with "Failed to find CUDA
  headers" = missing toolkit headers: `uv pip install "cupy-cuda12x[ctk]"`
  (or `export CUDA_PATH=/usr/local/cuda` when the image has a toolkit) and
  re-run. Any *other* compile failure is an NVRTC/sm_86 problem: report the
  full log; nothing else is worth running.
- NCCL init warnings with P2P absent are expected on PCIe boxes; the run
  falls back automatically (and one matrix config measures the fallback
  deliberately).
- A run killed by the runner's timeout leaves a `"status": "timeout"` row in
  the JSONL; the runner resumes past completed configs on re-invocation.
