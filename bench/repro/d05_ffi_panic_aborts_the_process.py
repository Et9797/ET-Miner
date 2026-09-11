"""#5 -- panic = "abort" turns every FFI panic into an uncatchable SIGABRT.

rust_ext/Cargo.toml sets `panic = "abort"` under [profile.release], and
CLAUDE.md's documented build is `maturin develop --release`, so the shipped
extension always aborts. PyO3's error model depends on unwinding to produce a
PanicException; abort removes it. Ordinary-looking inputs then exit 134 with no
traceback and nothing catchable by try/except -- a long GPU campaign loses every
unflushed level with no diagnostic.

This is a bug rather than an accepted trade-off because the codebase already
depends on it not being true: lib.rs implements an explicit non-contiguous copy
fallback five times and omits it six times; lib.rs raises a proper PyValueError
for an unsorted prev_flat and then aborts for a length mismatch one call later;
and gpu/mining.py:324-329 wraps a Rust call in a stale-wheel recovery path that
is dead code under abort.

Each vector runs in its own subprocess, because a live one kills the runner.

Note: PanicException derives from **BaseException**, not Exception, so a plain
`except Exception` does not catch it -- which matters for the recovery path at
gpu/mining.py:324-329. The subprocess below catches BaseException deliberately.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

VECTORS: dict[str, str] = {
    "noncontig": """
        import numpy as np, et_miner_rust
        idx = np.arange(0, 20, dtype=np.int64)[::2]          # non-C-contiguous view
        et_miner_rust.build_column_bitvecs_u64(
            np.arange(11, dtype=np.int64), idx, 10, 4)
    """,
    "ncols_small": """
        import numpy as np, et_miner_rust
        et_miner_rust.build_column_bitvecs_u64(
            np.array([0, 2], dtype=np.int64),
            np.array([0, 5], dtype=np.int64), 1, 1)          # n_cols smaller than max index
    """,
    "nrows_big": """
        import numpy as np, et_miner_rust
        et_miner_rust.build_column_bitvecs_u64(
            np.array([0, 1, 2], dtype=np.int64),
            np.array([0, 1], dtype=np.int64), 99, 4)         # n_rows > len(indptr)-1
    """,
}


def _run(body: str) -> tuple[int, str]:
    script = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        try:
    """) % str(REPO / "src") + textwrap.indent(textwrap.dedent(body), "    ") + textwrap.dedent("""
        except BaseException as e:
            print("CAUGHT:" + type(e).__name__)
            sys.exit(0)
        print("NO_ERROR")
    """)
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    return p.returncode, (p.stdout.strip().splitlines() or [""])[-1]


def reproduce() -> tuple[bool, str]:
    try:
        import et_miner_rust  # noqa: F401
    except ImportError:
        return False, "et_miner_rust not built; nothing to test"

    aborted, caught = [], []
    for name, body in VECTORS.items():
        rc, last = _run(body)
        if rc < 0 or rc == 134:
            aborted.append(f"{name}(rc={rc})")
        elif last.startswith("CAUGHT:"):
            caught.append(f"{name}->{last[7:]}")

    live = bool(aborted)
    return live, (
        f"aborted uncatchably: {aborted or 'none'}; "
        f"raised catchable: {caught or 'none'}"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
