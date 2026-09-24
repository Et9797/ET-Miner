# %% [markdown]
# # Tier 3.1: GPU setup and first run
#
# The GPU tier keeps the bit-vector layout of tier 2 and moves it into GPU memory, where CUDA kernels count
# candidates. This notebook lists what the GPU tier needs, detects the devices, runs the smallest possible GPU mine,
# and checks the GPU result against tiers 1 and 2.
#
# **What you will learn**
#
# - the driver, CUDA and package requirements;
# - how the library detects GPUs, and how to check the kernels compile;
# - the smallest GPU call and its equality check against the CPU tiers;
# - what happens when you ask for the GPU on a machine without one.
#
# **Prerequisites**: tiers 1 and 2 ([tier 2, notebook 3](../tier2-rust-pyo3/03-same-results-and-speed.ipynb)).
#
# **How to read the outputs.** The setup cell prints the hardware the committed outputs came from.
# Cells marked *GPU* print `skipped: no CUDA device` on a machine without one; every other cell runs on the CPU.
# The checklist at the end lists what to confirm after running the notebook on a GPU.

# %% [markdown]
# ## 0. Setup and device check
#
# The setup cell sets `SKIP_GPU` when CuPy is missing or no CUDA device is usable, and `SKIP_MULTI` for fewer than two devices.

@@GPU_SETUP@@

# %% [markdown]
# ## 1. Requirements
#
# | Need | Detail | Source |
# |---|---|---|
# | NVIDIA GPU | the kernel sources use only intrinsics available from compute capability 6.0 (sm_60) on | header of `src/et_miner/gpu/kernels/_src/shared_tiled.cu` |
# | Driver with CUDA 12 support | CuPy's `cupy-cuda12x` wheel targets CUDA 12 | `pyproject.toml`, `[project.optional-dependencies] gpu` |
# | `cupy-cuda12x[ctk]` | CuPy plus the CUDA header wheels that NVRTC needs to compile kernels at first use | same |
# | `nvidia-nccl-cu12` | NCCL, used to sum counts across GPUs | same |
# | Rust extension | optional; faster candidate grouping and pruning on the host | [tier 2, notebook 1](../tier2-rust-pyo3/01-build-and-install.md) |
#
# No CUDA toolkit or `nvcc` is needed: kernels are plain CUDA C sources compiled on the device at first use.
# Install the extra into the project environment with
#
# ```bash
# uv sync --inexact --extra gpu
# ```
#
# `--inexact` keeps the Rust extension installed (see tier 2, notebook 1).
# The next cell reads the extra from `pyproject.toml`, so the list above can be checked against HEAD.

# %%
import tomllib

pyproject = tomllib.loads((Path("../..") / "pyproject.toml").read_text())
for extra, packages in pyproject["project"]["optional-dependencies"].items():
    print(f"{extra}: {packages}")

# %% [markdown]
# The `gpu` extra holds exactly CuPy with the CUDA headers and NCCL.

# %% [markdown]
# ## 2. What the library detects
#
# `src/et_miner/backends.py` is the single place that probes for CuPy and the Rust extension.
# The next cell prints the public probes and runs `python -m et_miner info` (the same as `et-miner info`), the command-line view of the same checks.

# %%
import subprocess
import sys

print("HAS_GPU (CuPy importable):   ", et_miner.HAS_GPU)
print("has_cupy() (CuPy + a device):", et_miner.has_cupy())
print("get_gpu_count():             ", et_miner.get_gpu_count())
print("HAS_MULTI_GPU:               ", et_miner.HAS_MULTI_GPU)
print("HAS_RUST:                    ", et_miner.HAS_RUST)
info = subprocess.run([sys.executable, "-m", "et_miner", "info"], capture_output=True, text=True,
                      env={**os.environ, "LOGURU_LEVEL": "WARNING"})
print(info.stdout)

# %% [markdown]
# `HAS_GPU` only says CuPy imports; `has_cupy()` also needs a working device, and that is the probe to gate GPU work on.

# %% [markdown]
# ## 3. The kernels and the on-device self-check
#
# Every CUDA kernel is registered in `src/et_miner/gpu/kernels/loader.py:_KERNEL_FILES`, which maps a kernel name
# to its source file under `gpu/kernels/_src/`. The next cell lists the registry; it needs no GPU.

# %%
from collections import defaultdict

from et_miner.gpu.kernels.loader import _KERNEL_FILES

by_file = defaultdict(list)
for name, source in _KERNEL_FILES.items():
    by_file[source].append(name)
for source in sorted(by_file):
    print(f"{source:28s} {', '.join(by_file[source])}")

# %% [markdown]
# On a GPU machine, `bench/selfcheck.py` compiles every registered kernel on every device and launches the
# critical ones on tiny data; exit code 0 means the device is ready. The next cell (*GPU*) runs it.

# %%
if SKIP_GPU:
    print("skipped: no CUDA device")
else:
    check = subprocess.run([sys.executable, "bench/selfcheck.py"], capture_output=True, text=True, cwd="../..")
    print(check.stdout[-4000:])
    print("exit code", check.returncode)

# %% [markdown]
# A non-zero exit code here means a kernel did not compile or launch; `bench/README.md` has a troubleshooting section.

# %% [markdown]
# ## 4. The smallest GPU run
#
# `use_gpu=True` is the only change to the call. The next cell (*GPU*) mines the hand-made table on the GPU
# and compares it with tier 1. The GPU cells use integer item ids (`a`..`f` become 0..5), because
# `src/et_miner/gpu/mining.py:_build_results_from_gpu` maps columns back through an integer array.

# %%
toy_str = pl.read_parquet(DATA / "toy_8x6.parquet")
code = {name: i for i, name in enumerate("abcdef")}
toy = toy_str.select(pl.col("items").list.eval(pl.element().replace_strict(code, return_dtype=pl.Int64)))
tier1_toy = apriori(toy, min_support=0.25, sparse=False)
print("tier 1:", tier1_toy.height, "itemsets")
if SKIP_GPU:
    print("skipped: no CUDA device")
else:
    gpu_toy = apriori(toy, min_support=0.25, use_gpu=True)
    print(gpu_toy.sort(pl.col("itemset").list.len(), "itemset"))
    assert set(map(tuple, gpu_toy["itemset"].to_list())) == set(map(tuple, tier1_toy["itemset"].to_list()))
    print("GPU == tier 1 on the toy table")

# %% [markdown]
# On a GPU machine the cell prints the GPU result and asserts it has the same itemsets as tier 1.

# %% [markdown]
# ## 5. Same answer as tiers 1 and 2 on the shared sample
#
# The next cell mines `smoke.parquet` with tier 1 and with the whole-loop Rust route of tier 2 (both on the CPU),
# then (*GPU*) with `use_gpu=True`, and asserts all three give the same itemsets and counts.

# %%
import numpy as np

from et_miner import apriori_from_csr

smoke = pl.read_parquet(DATA / "smoke.parquet")
N, S = smoke.height, 0.01


def counted(result):
    return {(tuple(s), round(v * N)) for s, v in zip(result["itemset"].to_list(), result["support"].to_list())}


tier1 = counted(apriori(smoke, min_support=S, sparse=False))
rows = smoke["items"].to_list()  # the sample's item ids are 0..n-1, sorted and unique within each row
indptr = np.cumsum([0] + [len(r) for r in rows]).astype(np.int64)
indices = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
its, cnt = apriori_from_csr(indptr, indices, N, int(indices.max()) + 1, S, 0)
tier2 = {(tuple(s), int(c)) for s, c in zip(its, cnt)}
assert tier1 == tier2
print("tier 1 == tier 2:", len(tier1), "itemsets")
if SKIP_GPU:
    print("GPU: skipped: no CUDA device")
else:
    tier3 = counted(apriori(smoke, min_support=S, use_gpu=True))
    assert tier3 == tier1
    print("tier 3 (use_gpu=True) == tier 1 == tier 2:", len(tier3), "itemsets")

# %% [markdown]
# The CPU tiers agree; on a GPU machine the assertion extends the chain to the GPU tier.
# The repository enforces the full chain, including multi-GPU routes and efficient-apriori, in `tests/test_tier_equivalence.py`.

# %% [markdown]
# ## 6. Asking for the GPU without one
#
# The GPU path does not fall back to the CPU silently. The next cell shows what `use_gpu=True` raises when CuPy is
# missing; on a GPU machine it prints that the call succeeded instead.

# %%
try:
    apriori(toy, min_support=0.25, use_gpu=True)
    print("use_gpu=True ran on the GPU")
except Exception as err:  # the error type depends on what is missing
    print(f"{type(err).__name__}: {err}")

# %% [markdown]
# Without CuPy the call stops with an error instead of returning a CPU result, so a GPU benchmark cannot quietly run on the CPU.

# %% [markdown]
# ## Summary and next step
#
# - The GPU tier needs an NVIDIA driver with CUDA 12 and the `gpu` extra; kernels compile on first use.
# - Gate GPU work on `et_miner.has_cupy()`; check kernels with `bench/selfcheck.py`.
# - `use_gpu=True` returns the same itemsets and counts as tiers 1 and 2.
#
# **Checklist when you run this notebook on a GPU**
#
# - [ ] Setup cell prints the GPU name and CUDA runtime version.
# - [ ] Section 3: `bench/selfcheck.py` exits with code 0.
# - [ ] Section 4: the toy result prints and the assertion passes.
# - [ ] Section 5: the line `tier 3 (use_gpu=True) == tier 1 == tier 2` prints.
# - [ ] Section 6: prints `use_gpu=True ran on the GPU`.
#
# Next: [02-gpu-resident-mining](02-gpu-resident-mining.ipynb).
