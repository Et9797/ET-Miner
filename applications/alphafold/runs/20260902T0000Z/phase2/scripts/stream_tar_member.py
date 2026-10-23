#!/usr/bin/env python3
"""Stream one member out of a remote .tar.gz without storing the archive.

The archive is fetched as fixed-size HTTP byte ranges by a bounded pool of
parallel workers, consumed strictly in order by a streaming tar reader, and
the selected member's raw bytes are copied to stdout. Only a small rolling
window of range files ever touches the disk, and reading stops as soon as
the member has been copied.

Usage:
    stream_tar_member.py URL MEMBER [options] > member_bytes

Options:
    --chunk-mib N     size of one byte range in MiB (default 512)
    --parallel N      ranges fetched concurrently (default 12)
    --window N        extra fetched-but-unconsumed ranges kept on disk
                      beyond --parallel (default 4)
    --workdir DIR     directory for the rolling window of range files
                      (default ./stream_work)
    --size BYTES      archive size; taken from a HEAD request when omitted
    --log PATH        progress log file (default: stderr)
    --list            list member names/sizes as they stream by (no copy)
"""

import argparse
import io
import os
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def head_size(url):
    """Return the Content-Length of ``url`` from a HEAD request."""
    out = subprocess.run(
        ["curl", "-sSI", "--max-time", "60", url],
        check=True, capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("no Content-Length in HEAD response")


class RangeFetcher:
    """Fetch byte ranges of ``url`` into ``workdir`` with bounded look-ahead."""

    def __init__(self, url, size, chunk, parallel, window, workdir, log):
        self.url, self.size, self.chunk = url, size, chunk
        self.n_chunks = (size + chunk - 1) // chunk
        self.workdir, self.log = workdir, log
        self.slots = threading.Semaphore(parallel + window)
        self.pool = ThreadPoolExecutor(max_workers=parallel)
        self.futures = {}
        self.stop = threading.Event()
        os.makedirs(workdir, exist_ok=True)
        self.feeder = threading.Thread(target=self._feed, daemon=True)
        self.feeder.start()

    def path(self, i):
        return os.path.join(self.workdir, f"chunk_{i:06d}")

    def _feed(self):
        for i in range(self.n_chunks):
            if self.stop.is_set():
                return
            self.slots.acquire()
            if self.stop.is_set():
                return
            self.futures[i] = self.pool.submit(self._fetch, i)

    def _fetch(self, i):
        start = i * self.chunk
        end = min(self.size, start + self.chunk) - 1
        expected = end - start + 1
        tmp = self.path(i) + ".part"
        for attempt in range(1, 31):
            if self.stop.is_set():
                return
            proc = subprocess.run(
                ["curl", "-sS", "--fail", "--max-time", "1800",
                 "--retry", "5", "--retry-all-errors", "--retry-delay", "5",
                 "-r", f"{start}-{end}", "-o", tmp, self.url],
                capture_output=True, text=True,
            )
            if proc.returncode == 0 and os.path.getsize(tmp) == expected:
                os.replace(tmp, self.path(i))
                return
            self.log.write(f"chunk {i} attempt {attempt} failed rc={proc.returncode} "
                           f"{proc.stderr.strip()[:200]}\n")
            self.log.flush()
            time.sleep(min(60, 5 * attempt))
        raise RuntimeError(f"chunk {i} failed after 30 attempts")

    def take(self, i):
        """Block until chunk ``i`` is on disk and return its path."""
        fut = None
        while fut is None:
            fut = self.futures.get(i)
            if fut is None:
                time.sleep(0.05)
        fut.result()
        return self.path(i)

    def release(self, i):
        os.remove(self.path(i))
        del self.futures[i]
        self.slots.release()

    def shutdown(self):
        self.stop.set()
        self.slots.release()
        self.pool.shutdown(wait=False, cancel_futures=True)
        for name in os.listdir(self.workdir):
            if name.startswith("chunk_"):
                try:
                    os.remove(os.path.join(self.workdir, name))
                except OSError:
                    pass


class SequentialStream(io.RawIOBase):
    """Read-only file object that concatenates fetched chunks in order."""

    def __init__(self, fetcher, log):
        self.f, self.log = fetcher, log
        self.i, self.fh = 0, None
        self.consumed = 0
        self.t0 = self.t_last = time.time()

    def readable(self):
        return True

    def _open_next(self):
        if self.fh is not None:
            self.fh.close()
            self.f.release(self.i)
            self.i += 1
        if self.i >= self.f.n_chunks:
            self.fh = None
            return False
        self.fh = open(self.f.take(self.i), "rb")
        return True

    def readinto(self, b):
        while True:
            if self.fh is None and not self._open_next():
                return 0
            n = self.fh.readinto(b)
            if n:
                self.consumed += n
                now = time.time()
                if now - self.t_last >= 30:
                    rate = self.consumed / (now - self.t0) / 1e6
                    self.log.write(f"consumed {self.consumed/1e9:.2f} GB of "
                                   f"{self.f.size/1e9:.2f} GB  avg {rate:.1f} MB/s\n")
                    self.log.flush()
                    self.t_last = now
                return n
            if not self._open_next():
                return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url")
    ap.add_argument("member")
    ap.add_argument("--chunk-mib", type=int, default=512)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--workdir", default="./stream_work")
    ap.add_argument("--size", type=int)
    ap.add_argument("--log")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    log = open(a.log, "a") if a.log else sys.stderr
    size = a.size or head_size(a.url)
    log.write(f"start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
              f"url={a.url} size={size} member={a.member}\n")
    log.flush()

    fetcher = RangeFetcher(a.url, size, a.chunk_mib << 20, a.parallel,
                           a.window, a.workdir, log)
    stream = io.BufferedReader(SequentialStream(fetcher, log), 16 << 20)
    out = sys.stdout.buffer
    copied = 0
    found = False
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                log.write(f"member {member.name} size={member.size}\n")
                log.flush()
                if a.list or member.name != a.member:
                    continue
                found = True
                src = tar.extractfile(member)
                while True:
                    buf = src.read(8 << 20)
                    if not buf:
                        break
                    out.write(buf)
                    copied += len(buf)
                out.flush()
                if copied != member.size:
                    raise RuntimeError(f"copied {copied} != member size {member.size}")
                break
    finally:
        fetcher.shutdown()
    log.write(f"end {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
              f"found={found} copied={copied} consumed={stream.raw.consumed}\n")
    log.flush()
    if not (found or a.list):
        sys.exit(2)


if __name__ == "__main__":
    main()
