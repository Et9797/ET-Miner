"""#40 -- the CLI removes every terminal log sink, so -v, -q and all error
messages produce no visible output.

_logging.py:106 unconditionally calls logger.remove(), stripping loguru's
built-in stderr sink, then adds a file sink and only adds terminal sinks when
stderr=/stdout= are true (:123-127, both default False at :64-65). All three
configure_logging calls in cli.py:277-282 omit them, so every logger.* call in
the CLI -- including the error paths at :150, :160, :164, :297 and :306 -- goes
only to a file under ~/.cache/et-miner/logs/.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "src" / "et_miner" / "cli.py"


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    from et_miner._logging import configure_logging

    sig = inspect.signature(configure_logging)
    stderr_default = sig.parameters["stderr"].default
    stdout_default = sig.parameters["stdout"].default

    calls = []
    for node in ast.walk(ast.parse(CLI.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "configure_logging":
            kws = {kw.arg for kw in node.keywords}
            calls.append((node.lineno, sorted(kws)))

    silent = [ln for ln, kws in calls if "stderr" not in kws and "stdout" not in kws]
    live = bool(calls) and len(silent) == len(calls) and not stderr_default and not stdout_default
    return live, (
        f"configure_logging(stderr={stderr_default}, stdout={stdout_default}); "
        f"{len(silent)}/{len(calls)} CLI calls pass neither -> lines {silent}"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
