"""#62 -- get_ramdisk_info parses `df` without -k, so sizes are environment-dependent.

ramdisk.py:234-259 shells out to `df <path>` and reads column 1 as 1K-blocks.
Coreutils honours DF_BLOCK_SIZE / BLOCK_SIZE / BLOCKSIZE and switches to
512-byte blocks under POSIXLY_CORRECT, so the reported GB figures change with
the ambient environment -- and they feed the space check at :349.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _df_blocks(path: str, env_extra: dict[str, str]) -> int | None:
    env = {**os.environ, **env_extra}
    out = subprocess.run(["df", path], capture_output=True, text=True, env=env, timeout=10)
    lines = out.stdout.strip().splitlines()
    if len(lines) < 2:
        return None
    return int(lines[1].split()[1])


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    from et_miner.streaming.ramdisk import get_ramdisk_info

    with tempfile.TemporaryDirectory() as td:
        plain = _df_blocks(td, {})
        b512 = _df_blocks(td, {"BLOCK_SIZE": "512"})

        old = os.environ.get("BLOCK_SIZE")
        os.environ["BLOCK_SIZE"] = "512"
        try:
            info_512 = get_ramdisk_info(Path(td))
        finally:
            if old is None:
                os.environ.pop("BLOCK_SIZE", None)
            else:
                os.environ["BLOCK_SIZE"] = old
        info_plain = get_ramdisk_info(Path(td))

    if info_512 is None or info_plain is None:
        return False, "get_ramdisk_info returned None; cannot compare"

    size_differs = abs(info_512["size_gb"] - info_plain["size_gb"]) > 1e-9
    df_differs = plain is not None and b512 is not None and plain != b512
    live = size_differs and df_differs
    return live, (
        f"df blocks: plain={plain}, BLOCK_SIZE=512 -> {b512}; "
        f"get_ramdisk_info size_gb: {info_plain['size_gb']:.2f} vs {info_512['size_gb']:.2f}"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
