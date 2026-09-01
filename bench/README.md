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
- **Disk ≥ 40 GB** (datasets + wheels + rust build), **host RAM ≥ 32 GB** —
  the survivor filter sorts on the host (~48 B/survivor at peak) and its
  valve stages count slices through host RAM.
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
Tier 1 Polars == Tier 2 Rust == single-GPU legacy == multi-GPU legacy ==
shared multi-GPU == single-GPU sparse CSR (`sparse_from_k=3`) == multi-GPU
sparse CSR == **efficient-apriori**, exact itemsets and counts, on the
`smoke` synthetic preset. Any divergence aborts the run — no benchmark
number is worth recording from a miner that disagrees with the oracle.

## Knobs the matrix drives (documented in `src/et_miner/_env.py`)

| Env | Values | Meaning |
|---|---|---|
| `ET_MINER_KERNEL_VARIANT` | `auto`/`legacy`/`shared` | dense counting kernel A/B |
| `ET_MINER_FILTER_IMPL` | `compact`/`cupy`/`cpu` | survivor filter A/B |
| `ET_MINER_ROW_BALANCE` | `rows`/`nnz` | multi-GPU row split A/B |
| `ET_MINER_DISABLE_NCCL` | `1` | force the staged D2D reduce |
| `ET_MINER_MAX_CHUNK_CANDS` | int | force multi-chunk runs |

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
