"""List the run's uncommitted files (gitignored, or above GitHub's 100 MB limit) with size and sha256.

Writes RUN_DIR/EXCLUDED_LARGE_FILES.md and fails if a file above 100 MB is
not ignored by git (so none can be committed by accident). Hashes are cached
in RUN_DIR/state/sha256_cache.json by (path, size, mtime).

Usage:
    python3 large_files.py RUN_DIR

Options: none.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LIMIT = 100 * 10**6


def sha256(path):
    """Return the sha256 hex digest of a file."""
    s = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            s.update(b)
    return s.hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    repo = Path(subprocess.run(["git", "-C", str(run), "rev-parse", "--show-toplevel"], capture_output=True,
                               text=True, check=True).stdout.strip())
    cache_p = run / "state/sha256_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    listed = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--others", "--ignored", "--exclude-standard",
                             "--", str(run)], capture_output=True, check=True).stdout.decode().split("\0")
    ignored_set = {(repo / f).resolve() for f in listed if f}
    rows = []
    for p in sorted(run.rglob("*")):
        if not p.is_file() or (p.stat().st_size <= LIMIT and p.resolve() not in ignored_set):
            continue
        if p.is_relative_to(run / "state") or p.suffix == ".partial" or "__pycache__" in p.parts:
            continue
        st = p.stat()
        key = f"{p.relative_to(run)}|{st.st_size}|{int(st.st_mtime)}"
        if key not in cache:
            cache[key] = sha256(p)
        rows.append((str(p.relative_to(run)), st.st_size, cache[key], p.resolve() in ignored_set))
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(cache, indent=1))
    lines = ["# Files not committed (above 100 MB, or re-derivable intermediates)", "",
             f"Listed by `scripts/large_files.py` at {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}. "
             "Each is gitignored, stays on local disk under this run directory, and is re-derivable with the "
             "scripts in `scripts/` (see `data/MANIFEST.md`).", "",
             "| file | bytes | sha256 |", "|---|---|---|"]
    lines += [f"| `{f}` | {n:,} | `{h}` |" for f, n, h, _ in rows]
    (run / "EXCLUDED_LARGE_FILES.md").write_text("\n".join(lines) + "\n")
    bad = [f for f, _, _, ign in rows if not ign]
    print(f"{len(rows)} files listed ({sum(n for _, n, _, _ in rows) / 1e9:.1f} GB); > 100 MB and not ignored: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    os.umask(0o022)
    main()
