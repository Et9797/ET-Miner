"""#46 -- an environment override equal to the default value is silently discarded.

config.py:376-388 infers "was this explicitly set?" by comparing the parsed env
value against a freshly constructed default, not by checking presence in
os.environ. So exporting a variable whose value happens to equal the default is
indistinguishable from not exporting it, and the TOML file value wins -- the
opposite of the precedence documented at config.py:335.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    from et_miner.config import Config, load_config

    default = Config().apriori.min_support  # 0.01
    file_value = 0.25

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "et.toml"
        cfg_path.write_text(f"[apriori]\nmin_support = {file_value}\n")

        old = os.environ.get("ET_MINER_APRIORI_MIN_SUPPORT")
        os.environ["ET_MINER_APRIORI_MIN_SUPPORT"] = str(default)
        try:
            got = load_config(cfg_path, use_env=True).apriori.min_support
        finally:
            if old is None:
                os.environ.pop("ET_MINER_APRIORI_MIN_SUPPORT", None)
            else:
                os.environ["ET_MINER_APRIORI_MIN_SUPPORT"] = old

    live = got != default
    return live, (
        f"env={default} (==default), file={file_value} -> load_config gave {got} "
        f"({'env discarded' if live else 'env honoured'})"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
