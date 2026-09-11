"""#21 -- compute_self_sufficiency crashes instead of returning its documented empty frame.

The function has an empty-result path that returns a correctly typed empty
DataFrame. It is dead code: the chunk result is appended unconditionally, so
result_chunks is never empty once any chunk has been read. `combined` is then a
0-row frame, .min() on its ratio column returns None, and the logging f-string
raises TypeError from inside a log statement -- after all the mining work.

The sibling function in the same file gets this right: generate_rules_drop1
guards with `if n_rules > 0`. The two parquet consumers differ on exactly that
one line.

Trigger: any call where no K-itemset retains a surviving (K-1)-subset.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    import polars as pl

    from et_miner.core.rules import compute_self_sufficiency

    with tempfile.TemporaryDirectory() as td:
        k3 = Path(td) / "k3.parquet"
        k2 = Path(td) / "k2.parquet"
        # no (K-1)-subset of {1,2,3} is present, so every inner join is empty
        pl.DataFrame({"itemset": [[1, 2, 3]], "support": [0.30]}).write_parquet(k3)
        pl.DataFrame({"itemset": [[7, 8]], "support": [0.50]}).write_parquet(k2)
        try:
            out = compute_self_sufficiency(k3, k2)
        except TypeError as e:
            return True, f"raised TypeError from inside the log statement: {e}"
        except Exception as e:  # noqa: BLE001
            return True, f"raised {type(e).__name__}: {e}"

    ok = out.height == 0 and "self_sufficiency_ratio" in out.columns
    return not ok, (
        f"returned {out.height}-row frame with columns {out.columns}"
        if ok else f"returned an unexpected frame: {out}"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
