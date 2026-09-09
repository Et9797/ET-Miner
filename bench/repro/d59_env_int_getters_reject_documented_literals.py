"""#59 -- `_env` integer getters reject the literals their own docstring prints.

_env.py:9-14 documents ET_MINER_FLUSH_PARALLEL_THRESHOLD "(default 1e8)" and
ET_FLUSH_CHUNK_SIZE "(default 5e7)"; README.md:175 prints 5e7 too. The getters
at :53 and :65 are a bare int(), and int("1e8") raises ValueError. The module
even has an _int_env helper (:104-111) that neither uses.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from et_miner import _env

    failures = []
    for var, getter, literal in (
        ("ET_MINER_FLUSH_PARALLEL_THRESHOLD", _env.flush_parallel_threshold, "1e8"),
        ("ET_FLUSH_CHUNK_SIZE", _env.flush_chunk_size, "5e7"),
    ):
        old = os.environ.get(var)
        os.environ[var] = literal
        try:
            getter()
        except ValueError as e:
            failures.append(f"{var}={literal} -> {type(e).__name__}")
        finally:
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old

    doc = _env.__doc__ or ""
    documented = "1e8" in doc and "5e7" in doc
    live = bool(failures) and documented
    return live, f"docstring prints 1e8/5e7; {'; '.join(failures) or 'both parse'}"


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
