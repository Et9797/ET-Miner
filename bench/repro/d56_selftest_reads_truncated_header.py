"""#56 -- the ramdisk self-test reads a 17-byte header from a 21-byte format.

The writers pack '<4sQQB' (ramdisk.py:388-394, :400-406). struct.calcsize of
that is 21. The self-test at ramdisk.py:636 reads 17, so the np.fromfile that
follows starts 4 bytes into the payload: every value is garbage and the element
count is off. The test then prints OK.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "et_miner" / "streaming" / "ramdisk.py"

FMT = "<4sQQB"


def reproduce() -> tuple[bool, str]:
    real = struct.calcsize(FMT)
    text = SRC.read_text()

    # every f.read(N) in the file, with its line number
    reads = [(i + 1, int(m.group(1)))
             for i, line in enumerate(text.splitlines())
             if (m := re.search(r"\.read\((\d+)\)", line))]
    wrong = [(ln, n) for ln, n in reads if n != real]

    # confirm the format really is the one packed
    packs = re.findall(r"struct\.pack\(\s*['\"]([^'\"]+)['\"]", text)
    fmt_ok = FMT in packs

    live = bool(wrong) and fmt_ok
    return live, (
        f"struct.calcsize({FMT!r})={real}; packs={sorted(set(packs))}; "
        f"header reads of the wrong size: {wrong or 'none'}"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
