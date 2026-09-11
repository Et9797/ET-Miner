"""#23/#24 -- GPU result buffers truncate silently, and non-deterministically.

Four multi-GPU kernels clamped with a bare `n = min(n, gpu_max)` where their
single-GPU siblings called _warn_result_truncation. And that helper itself
tolerated up to 5% loss with only a logger.warning, so calling it at four more
sites would merely have extended a 5% silent-loss window to all eight. The two
are not independent fixes.

The dropped set is non-deterministic -- the kernels append via atomicAdd -- so
these routes disagreed with the row-split path, with the CPU tiers, and with
THEMSELVES RUN TWICE. There is no percentage of silently-lost frequent itemsets
that is acceptable for a miner whose contract is exactness.

Reproduced with a reduced max_results, which exercises the identical code path:
the production trigger is >10M survivors in one GPU's slice at one level.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    try:
        import cupy as cp
    except ImportError:
        return False, "no CUDA device"

    import numpy as np

    from et_miner.gpu.kernels import count_itemsets_fused_k3plus
    from et_miner.gpu.kernels.loader import _warn_result_truncation

    findings = []

    # (a) the helper itself: a 3% overflow must not be tolerated
    try:
        got = _warn_result_truncation(9880, 9583, "probe")
        findings.append(f"3% overflow tolerated: returned {got:,} of 9,880 (297 lost silently)")
    except RuntimeError:
        pass

    # (b) a real kernel run with a buffer one short of the survivor count
    rng = np.random.default_rng(0)
    n_rows, n_cols = 4096, 40
    dense = (rng.random((n_rows, n_cols)) < 0.6)
    n_u64s = (n_rows + 63) // 64
    bits = np.zeros((n_cols, n_u64s), dtype=np.uint64)
    for c in range(n_cols):
        for r in np.flatnonzero(dense[:, c]):
            bits[c, r // 64] |= np.uint64(1) << np.uint64(r % 64)
    bv = cp.asarray(bits)

    cands = [(a, b, c) for a in range(n_cols) for b in range(a + 1, n_cols)
             for c in range(b + 1, n_cols)][:4000]
    full, _ = count_itemsets_fused_k3plus(bv, cands, n_u64s, 1)
    n_true = len(full)
    if n_true < 10:
        return False, f"fixture produced only {n_true} survivors; cannot overflow a buffer"

    try:
        got, _ = count_itemsets_fused_k3plus(bv, cands, n_u64s, 1, max_results=n_true - 1)
        findings.append(
            f"kernel returned {len(got):,} of {n_true:,} survivors with no error "
            f"({n_true - len(got)} lost silently)"
        )
    except RuntimeError:
        pass

    return bool(findings), "; ".join(findings) if findings else "any overflow raises"


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
