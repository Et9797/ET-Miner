"""#4 -- _sparse_matmul casts int32 -> float32 for MKL; k=2 counts saturate at 2^24.

core/sparse.py:118-121 casts any non-float input to float32 before handing it to
dot_product_mkl. The CSR data is np.int32 (core/matrix.py:464, chosen
deliberately -- its comment notes "uint8 overflows at 256"), so the cast always
fires. float32 has a 24-bit significand: once the accumulator reaches
2^24 = 16,777,216, `acc + 1` rounds back to `acc` and the sum sticks there.

It is an UNDER-count, and the level filter is `count >= min_count`, so genuinely
frequent pairs are silently dropped and the loss cascades into every higher K.
Between 2^24 and 2^25 the representable spacing is 2, so counts in that range
are wrong by +-1 even without saturating.

Fires only when sparse_dot_mkl imports. It does on this box, so the MKL branch
is the live one here. The scipy fallback in the same function (:123-124) keeps
int32 and is exact -- two branches of one function disagreeing at scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

N_ROWS = 20_000_000


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    import numpy as np
    from scipy.sparse import csr_matrix

    try:
        import sparse_dot_mkl  # noqa: F401
    except ImportError:
        return False, "sparse_dot_mkl not importable; the MKL branch is unreachable here"

    from et_miner.core.sparse import _sparse_matmul

    # two fully populated columns -> the true pair count is exactly N_ROWS
    indptr = np.arange(0, 2 * N_ROWS + 1, 2, dtype=np.int32)
    indices = np.tile(np.array([0, 1], dtype=np.int32), N_ROWS)
    data = np.ones(2 * N_ROWS, dtype=np.int32)
    m = csr_matrix((data, indices, indptr), shape=(N_ROWS, 2))

    got = int(_sparse_matmul(m.T.tocsr(), m)[0, 1])
    exact = int((m.T.tocsr() @ m)[0, 1])  # scipy in int32

    live = got != N_ROWS
    return live, (
        f"true pair count {N_ROWS:,}; _sparse_matmul -> {got:,} "
        f"(error {got - N_ROWS:+,}, {100 * (got - N_ROWS) / N_ROWS:+.1f}%); "
        f"scipy int32 -> {exact:,}"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
