"""#55 -- setup_ramdisk dereferences get_ramdisk_info's documented None return.

get_ramdisk_info is documented as returning None (ramdisk.py:220-222). At
ramdisk.py:105 the guard `if stat_info and stat_info['size_gb'] >= size_gb`
short-circuits safely, but the else branch at :109 subscripts stat_info
unconditionally -> TypeError. That TypeError is then swallowed by the blanket
`except Exception` at :140-141 and re-raised as
RuntimeError("Ramdisk setup failed: ..."), which blames the mount.

Note: the report cites :104-110; the fault is at :109 only (recorded as N16).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    from et_miner.streaming import ramdisk

    orig_info, orig_run = ramdisk.get_ramdisk_info, ramdisk.subprocess.run
    orig_geteuid = ramdisk.os.geteuid

    class _Mounted:
        returncode = 0
        stdout = "is a mountpoint\n"
        stderr = ""

    ramdisk.get_ramdisk_info = lambda p: None          # its documented return
    ramdisk.subprocess.run = lambda *a, **k: _Mounted()  # "already mounted"
    ramdisk.os.geteuid = lambda: 0
    try:
        # mount_point MUST exist, or setup_ramdisk takes the mkdir branch at
        # :91 and never reaches the already-mounted check. The default
        # /mnt/ramdisk does not exist on a typical box, which is how a first
        # attempt at this repro "passed" without executing the code under test.
        with tempfile.TemporaryDirectory() as td:
            ramdisk.setup_ramdisk(size_gb=1, mount_point=td)
        outcome = "returned normally"
    except TypeError as e:
        outcome = f"TypeError: {e}"
    except Exception as e:  # noqa: BLE001
        outcome = f"{type(e).__name__}: {e}"
    finally:
        ramdisk.get_ramdisk_info, ramdisk.subprocess.run = orig_info, orig_run
        ramdisk.os.geteuid = orig_geteuid

    # The TypeError from :109 is caught by the blanket `except Exception` at
    # :140-141 and re-raised as RuntimeError("Ramdisk setup failed: ..."), which
    # blames the mount rather than the None deref. Either shape is the defect.
    live = "not subscriptable" in outcome or outcome.startswith("TypeError")
    return live, f"get_ramdisk_info()->None on a mounted path: setup_ramdisk {outcome}"


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
