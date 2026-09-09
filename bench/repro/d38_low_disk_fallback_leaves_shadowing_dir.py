"""#38 -- the low-disk fallback leaves an empty directory that shadows its own output.

os.makedirs(local_part_dir, exist_ok=True) at flush.py:159 runs unconditionally
and BEFORE the disk check at :161-166. When the check fails, the fallback writes
the real data to output_dir/frequent_k{k}.parquet and returns at :194, leaving
output_dir/frequent_k{k}/ in place and empty.

Both downstream consumers dispatch on isdir and prefer the directory:
  io/gcs.py:108-110    resolve_k_parquet
  core/rules.py:122-126  the part_*.parquet glob
so rule generation for that K yields zero rows, silently, and resume_from_k
reads an empty dataset.

The branch is NOT remote-gated (the docstring at :76-80 says gs://; the code at
:176-177 is a local ParquetWriter), so this reproduces with no bucket at all.
Recorded as N8.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    import numpy as np

    sys.path.insert(0, str(REPO / "src"))
    from et_miner.io import flush as flush_mod
    from et_miner.io.gcs import resolve_k_parquet

    # `shutil` is imported INSIDE _flush_k_parquet (flush.py:162), so it is not
    # a module attribute -- patch the stdlib function itself.
    import shutil as _shutil

    real_usage = _shutil.disk_usage

    class _NoSpace:
        free = 0
        total = 1 << 40
        used = 1 << 40

    with tempfile.TemporaryDirectory() as td:
        items = np.arange(20, dtype=np.int32).reshape(10, 2)
        sup = np.full(10, 0.5, dtype=np.float64)
        # force the partitioned branch, then starve it of disk
        import os

        old = os.environ.get("ET_MINER_FLUSH_PARALLEL_THRESHOLD")
        os.environ["ET_MINER_FLUSH_PARALLEL_THRESHOLD"] = "1"
        _shutil.disk_usage = lambda p: _NoSpace()
        try:
            flush_mod._flush_k_parquet(items, sup, 2, output_dir=td, is_remote=False,
                                       uploader=None, backup_dir=None)
        finally:
            _shutil.disk_usage = real_usage
            if old is None:
                os.environ.pop("ET_MINER_FLUSH_PARALLEL_THRESHOLD", None)
            else:
                os.environ["ET_MINER_FLUSH_PARALLEL_THRESHOLD"] = old

        single = Path(td) / "frequent_k2.parquet"
        partdir = Path(td) / "frequent_k2"
        parts = sorted(partdir.glob("part_*.parquet")) if partdir.is_dir() else []
        resolved = resolve_k_parquet(td, 2)

        live = single.exists() and partdir.is_dir() and not parts and str(resolved) == str(partdir)
        return live, (
            f"data written to {single.name} (exists={single.exists()}); "
            f"empty dir left: {partdir.is_dir()} with {len(parts)} parts; "
            f"resolve_k_parquet -> {'the EMPTY DIR' if str(resolved) == str(partdir) else Path(resolved).name}"
        )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
