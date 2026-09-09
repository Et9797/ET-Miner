"""#50 -- config fields are parsed, validated and exposed, and never reach the miner.

config.py declares apriori.sparse (:57), apriori.n_jobs (:60),
streaming.memory_budget_mb (:78) and streaming.prefetch_chunks (:81). Three of
the four have a real destination -- apriori(sparse=, n_jobs=) and
apriori_streaming(memory_budget_gb=) -- and cli.py:170-193 forwards none of
them. prefetch_chunks has no counterpart anywhere in the codebase.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "src" / "et_miner" / "cli.py"
WATCH = ["sparse", "n_jobs", "memory_budget_mb", "prefetch_chunks"]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    from et_miner.config import Config

    cfg = Config()
    declared = [f for f in WATCH
                if hasattr(cfg.apriori, f) or hasattr(cfg.streaming, f)]

    src = CLI.read_text()
    read_by_cli = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Attribute) and node.attr in WATCH:
            # config.apriori.sparse / config.streaming.n_jobs ...
            if isinstance(node.value, ast.Attribute) and getattr(
                node.value.value, "id", ""
            ) == "config":
                read_by_cli.append(node.attr)

    dropped = [f for f in declared if f not in read_by_cli]
    live = len(dropped) == len(declared) and bool(declared)
    return live, f"declared+validated: {declared}; read by cli.py: {read_by_cli or 'none'}"


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
