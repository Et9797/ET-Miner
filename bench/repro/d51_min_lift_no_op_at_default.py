"""#51 -- `--min-lift` filters nothing at or below its own documented default.

cli.py:107-112 advertises `default=1.0` and "Minimum lift for rule filtering
(default: 1.0)", but cli.py:203-204 gates the filter on `> 1.0`, so at the
advertised default no filtering happens and negatively correlated rules
(lift < 1.0) are emitted anyway.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "et_miner" / "cli.py"


def reproduce() -> tuple[bool, str]:
    tree = ast.parse(SRC.read_text())

    # the advertised default
    default = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            if node.args and getattr(node.args[0], "value", None) == "--min-lift":
                for kw in node.keywords:
                    if kw.arg == "default":
                        default = ast.literal_eval(kw.value)

    # the comparison actually used
    op = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute):
            if node.left.attr == "min_lift":
                op = type(node.ops[0]).__name__

    strict = op == "Gt"
    live = strict and default is not None
    ev = f"default={default!r}, filter uses `{op}` -> at the default the filter is a no-op"
    if not live:
        ev = f"default={default!r}, filter uses `{op}` -> applied at the default"
    return live, ev


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
