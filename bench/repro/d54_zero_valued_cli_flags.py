"""#54 -- `--min-support 0`, `--max-length 0`, `--chunk-size 0` are silently
replaced by config values.

cli.py:170-171 and :178 use `or` as a None-coalesce. `or` tests falsiness, so a
legitimately-supplied 0 is indistinguishable from "not supplied". 0.0 is a valid
min_support per AprioriConfig.validate (config.py:65-66, accepts 0.0 <= s <= 1.0).
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "et_miner" / "cli.py"
WATCH = {"min_support", "max_length", "chunk_size"}


def reproduce() -> tuple[bool, str]:
    tree = ast.parse(SRC.read_text())
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.BoolOp):
            continue
        if not isinstance(node.value.op, ast.Or):
            continue
        left = node.value.values[0]
        if isinstance(left, ast.Attribute) and left.attr in WATCH:
            hits.append((left.attr, node.lineno))

    # and the semantic proof, independent of the source scan
    args_min_support = 0.0
    config_min_support = 0.01
    effective = args_min_support or config_min_support
    swallowed = effective != args_min_support

    live = bool(hits) and swallowed
    where = ", ".join(f"{n}@:{ln}" for n, ln in hits) or "none"
    return live, f"`or`-coalesce at {where}; 0.0 -> {effective} (supplied value discarded)"


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
