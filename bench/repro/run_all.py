#!/usr/bin/env python3
"""Run every defect reproduction and report which are still live.

Exit code 0 when every repro ran; 1 when a repro itself errored. Use
--expect-live to make a *fixed* defect the failure (that is how a repro is
checked before its fix lands).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def discover(ids: set[str] | None) -> list[Path]:
    out = []
    for p in sorted(HERE.glob("[dn][0-9][0-9]_*.py")):
        num = re.match(r"[dn](\d+)_", p.name)
        if ids and (not num or num.group(1).lstrip("0") not in ids):
            continue
        out.append(p)
    return out


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", action="append", help="defect number (repeatable)")
    ap.add_argument("--expect-live", action="store_true",
                    help="exit non-zero if any defect reproduces as ALREADY FIXED")
    args = ap.parse_args()

    ids = {i.lstrip("0#") for i in args.id} if args.id else None
    scripts = discover(ids)
    if not scripts:
        print("no reproductions matched", file=sys.stderr)
        return 1

    live, fixed, broken = [], [], []
    for path in scripts:
        try:
            is_live, evidence = load(path).reproduce()
        except Exception:
            broken.append(path.stem)
            print(f"  {path.stem:34} ERROR")
            traceback.print_exc(limit=3)
            continue
        (live if is_live else fixed).append(path.stem)
        print(f"  {path.stem:34} {'LIVE ' if is_live else 'FIXED'}  {evidence}")

    print(f"\n{len(live)} live, {len(fixed)} fixed, {len(broken)} broken")
    if broken:
        return 1
    if args.expect_live and fixed:
        print(f"expected all live, but these are fixed: {', '.join(fixed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
