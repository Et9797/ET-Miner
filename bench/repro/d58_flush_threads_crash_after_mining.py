"""#58 -- ET_FLUSH_THREADS=0 divides by zero; -1 indexes an empty `bounds`.

_env.flush_threads() (_env.py:68-69) is a bare int() with no validation. The
min() at flush.py:141-144 floors only the CPU term, not the env term, so any
value <= 0 wins and propagates. Both crashes land AFTER the level is fully
mined, so a long campaign loses the level's work.

Note: the report cites the IndexError at flush.py:229; it is at :227 (:229 is
the `schema` argument). Recorded as N18.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _flush(n_threads_env: str, tmpdir: str):
    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    from et_miner.io.flush import _flush_k_parquet

    os.environ["ET_FLUSH_THREADS"] = n_threads_env
    os.environ["ET_MINER_FLUSH_PARALLEL_THRESHOLD"] = "1"  # force the partitioned branch
    items = np.arange(20, dtype=np.int32).reshape(10, 2)
    sup = np.full(10, 0.5, dtype=np.float64)
    _flush_k_parquet(items, sup, 2, output_dir=tmpdir, is_remote=False,
                     uploader=None, backup_dir=None)


def reproduce() -> tuple[bool, str]:
    saved = {k: os.environ.get(k) for k in
             ("ET_FLUSH_THREADS", "ET_MINER_FLUSH_PARALLEL_THRESHOLD")}
    seen = {}
    try:
        for val in ("0", "-1"):
            with tempfile.TemporaryDirectory() as td:
                try:
                    _flush(val, td)
                    seen[val] = "no error"
                except Exception as e:  # noqa: BLE001 -- the point is which one
                    seen[val] = type(e).__name__
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    live = seen.get("0") == "ZeroDivisionError" or seen.get("-1") == "IndexError"
    return live, f"ET_FLUSH_THREADS=0 -> {seen.get('0')}; =-1 -> {seen.get('-1')}"


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
