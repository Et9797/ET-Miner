"""One benchmark config, executed in a fresh process (fresh CUDA context).

Invoked by bench/runner.py with a JSON config as argv[1]. Mining logs go to
stderr (loguru default); the single RESULT json line goes to stdout.
Exit codes: 0 ok, 3 correctness failure (motif recovery), other = crash.

Configs with ``"mode": "consolidation"`` name their route explicitly (C,
C-bitvecs, D, E, F; see ``run_consolidation``), warm up on every device they
use, and record per-level (or per-pass) times, per-device peak VRAM, peak RSS,
throttle reasons, the result signature, and any fallback the run logged.
"""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


#: NVML clock-event reasons that slow a running kernel (SW power cap, HW
#: slowdown, SW/HW thermal, HW power brake). GpuIdle (0x1) and the
#: application/display clock settings are not throttling.
THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80


class VramSampler(threading.Thread):
    """Polls nvidia-smi (context-free — sees true device peaks) every 0.5 s,
    and this process's RSS alongside it."""

    #: Field name for active throttle reasons differs across nvidia-smi
    #: versions; probe from newest to oldest, fall back to memory-only.
    _QUERIES = (
        "index,memory.used,clocks_event_reasons.active",
        "index,memory.used,clocks_throttle_reasons.active",
        "index,memory.used",
    )

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mb: dict[int, int] = {}
        self.throttle_reasons: set[str] = set()
        self.reason_mask: dict[int, int] = {}
        self.peak_rss_mb = 0.0
        self._halt = threading.Event()
        self._query_idx = 0
        try:
            import psutil

            self._proc = psutil.Process()
        except ImportError:
            self._proc = None

    def _poll_once(self) -> bool:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={self._QUERIES[self._query_idx]}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return False
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[1].isdigit():
                idx, used = int(parts[0]), int(parts[1])
                self.peak_mb[idx] = max(self.peak_mb.get(idx, 0), used)
                if len(parts) > 2 and parts[2] not in ("0x0000000000000000", "[N/A]", "", "[Not Supported]"):
                    self.throttle_reasons.add(parts[2])
                    try:
                        self.reason_mask[idx] = self.reason_mask.get(idx, 0) | int(parts[2], 16)
                    except ValueError:
                        pass
        return True

    def run(self):
        while not self._halt.is_set():
            try:
                if not self._poll_once() and self._query_idx < len(self._QUERIES) - 1:
                    self._query_idx += 1  # field unsupported — degrade the query
            except Exception:
                pass
            if self._proc is not None:
                try:
                    self.peak_rss_mb = max(self.peak_rss_mb, self._proc.memory_info().rss / 2**20)
                except Exception:
                    pass
            self._halt.wait(0.5)

    def throttled_devices(self) -> list[int]:
        return sorted(d for d, m in self.reason_mask.items() if m & THROTTLE_MASK)

    def stop(self):
        self._halt.set()


def result_signatures(res, n_rows: int) -> dict:
    """Order-independent signatures of the mined result."""
    itemsets = res["itemset"].to_list()
    supports = res["support"].to_list()
    n = len(itemsets)
    total = 0
    hasher = hashlib.sha256()
    lines = []
    for s, sup in zip(itemsets, supports):
        cnt = round(sup * n_rows)
        total += cnt
        lines.append((tuple(sorted(int(i) for i in s)), cnt))
    lines.sort()
    for key, cnt in lines:
        hasher.update((",".join(map(str, key)) + f":{cnt};").encode())
    return {"n_itemsets": n, "sum_counts": int(total), "itemset_hash": hasher.hexdigest()}


#: Config keys whose apriori() parameters were removed; a config that sets one
#: measured a route that no longer exists, so it fails instead of running another.
REMOVED_KEYS = ("gpu_resident", "prune_apriori")


def main() -> int:
    cfg = json.loads(sys.argv[1])
    import os

    removed = sorted(k for k in REMOVED_KEYS if k in cfg)
    if removed:
        raise ValueError(f"config {cfg.get('id')} sets removed keys {removed}: its route no longer exists")
    if cfg.get("sparse_from_k") is not None and cfg.get("route") in ("F", "EA", "cpu", "D", "E"):
        raise ValueError("sparse_from_k requires an in-core GPU route; this config would ignore it")

    for k, v in cfg.get("env", {}).items():
        os.environ[k] = str(v)
    if cfg.get("mode") == "consolidation":
        from consolidation_run import run_consolidation

        return run_consolidation(cfg)

    import polars as pl

    from et_miner.core.apriori import apriori
    from et_miner.synthetic import PRESETS

    spec = PRESETS[cfg["preset"]]
    data_path = REPO / "datasets" / "synth" / f"{cfg['preset']}.parquet"
    sidecar = json.loads((REPO / "datasets" / "synth" / f"{cfg['preset']}.json").read_text())
    df = pl.read_parquet(data_path)

    min_support = cfg.get("min_support") or spec.min_support
    levels: list[dict] = []

    def level_cb(k, n_cand, n_freq, ms):
        levels.append({"k": k, "n_candidates": int(n_cand), "n_frequent": int(n_freq), "ms": float(ms)})

    sampler = VramSampler()
    sampler.start()
    t0 = time.perf_counter()
    status = "ok"
    signatures: dict = {}
    motifs_ok = True
    try:
        if cfg.get("two_phase"):
            import tempfile

            from et_miner.gpu.row_split import mine_two_phase

            with tempfile.TemporaryDirectory() as td:
                mine_two_phase(
                    df,
                    phase1_support=min_support,
                    # Distinct supports are the ONLY configuration in which the
                    # anchor filter does anything; phase2 == phase1 makes every
                    # frequent item an anchor and the mask all-True (the vacuous
                    # shape tests/test_row_split_e2e.py:102 also has).
                    phase2_support=cfg.get("phase2_support", min_support),
                    max_length=cfg.get("max_length"),
                    item_col="items",
                    n_gpus=cfg.get("n_gpus", 2),
                    output_dir=td,
                    sparse_from_k=cfg.get("sparse_from_k"),
                    level_callback=level_cb,
                )
                parts = sorted((Path(td) / "phase2").glob("frequent_k*.parquet"))
                res = pl.concat([pl.read_parquet(p) for p in parts]) if parts else pl.DataFrame(
                    {"itemset": [], "support": []}
                )
        elif cfg.get("route") == "cpu":
            # CPU tier: the path #4 (MKL float cast), #13 (batch length filter)
            # and #18 (n_jobs) change. Not reachable through the GPU branch.
            res = apriori(
                df,
                min_support=min_support,
                max_length=cfg.get("max_length"),
                item_col="items",
                use_gpu=False,
                sparse=cfg.get("sparse"),
                n_jobs=cfg.get("n_jobs", 1),
                level_callback=level_cb,
            )
        else:
            res = apriori(
                df,
                min_support=min_support,
                max_length=cfg.get("max_length"),
                item_col="items",
                use_gpu=True,
                n_gpus=cfg.get("n_gpus", 2),
                sparse_from_k=cfg.get("sparse_from_k"),
                level_callback=level_cb,
            )
        wall_s = time.perf_counter() - t0
        signatures = result_signatures(res, spec.n_rows)

        # Planted-motif recovery: oracle-independent correctness at depth.
        if sidecar.get("planted") and not cfg.get("max_length") and not cfg.get("two_phase"):
            counts = {}
            for s, sup in zip(res["itemset"].to_list(), res["support"].to_list()):
                counts[tuple(sorted(int(i) for i in s))] = round(sup * spec.n_rows)
            for entry in sidecar["planted"]:
                key = tuple(sorted(entry["items"]))
                got = counts.get(key)
                if got is None or got < entry["planted_count"]:
                    motifs_ok = False
                    print(
                        f"MOTIF FAILURE: {key} mined={got} planted={entry['planted_count']}",
                        file=sys.stderr,
                    )
    except Exception as e:  # noqa: BLE001 — recorded, parent decides
        status = f"error: {type(e).__name__}: {e}"
        wall_s = time.perf_counter() - t0
    finally:
        sampler.stop()
        sampler.join(timeout=3)

    payload = json.dumps(
        {
            "id": cfg["id"],
            "config": cfg,
            "status": status,
            "wall_s": round(wall_s, 3),
            "levels": levels,
            "peak_vram_mb": sampler.peak_mb,
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "throttle_reasons": sorted(sampler.throttle_reasons),
            "motifs_ok": motifs_ok,
            **signatures,
        }
    )
    # Primary channel is a file: NCCL (and anything else writing raw to
    # fd 1) interleaves with Python's buffered stdout and can splice the
    # JSON mid-line, so stdout is only a debug fallback.
    if cfg.get("result_path"):
        Path(cfg["result_path"]).write_text(payload + "\n")
    print(payload)
    if status != "ok":
        return 1
    return 0 if motifs_ok else 3


if __name__ == "__main__":
    sys.exit(main())
