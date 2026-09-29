"""GPU acceleration: CUDA kernels, CSR/bitvec builders, the row-split miner.

Modules here are import-safe without CuPy — cupy itself is imported inside
functions, so failures surface at call time with a clear error rather than
at import. Import the concrete submodule you need
(e.g. ``from et_miner.gpu.kernels import count_tiled_fused``); this
package ``__init__`` deliberately re-exports nothing to keep GPU modules
out of CPU-only import chains.
"""
