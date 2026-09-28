"""Typed, call-time accessors for every ET_* environment variable.

All ad-hoc environment reads in the package go through this module so the
full knob surface is documented in one place. Getters read the environment
at call time so `monkeypatch.setenv` in tests and late exports in job scripts
both behave as expected. One exception: ET_MINER_DISABLE_RUST is read once,
when et_miner.backends is imported.

Variables:
    ET_MINER_FLUSH_PARALLEL_THRESHOLD  int, rows above which parquet flush
                                       partitions in parallel (default 1e8)
    ET_MINER_LEGACY_WRITE              "1" enables the legacy single-table
                                       parquet write path
    ET_FLUSH_COMPRESSION               parquet compression codec (zstd)
    ET_FLUSH_CHUNK_SIZE                rows per flush chunk (default 5e7)
    ET_FLUSH_THREADS                   flush writer threads (default 4)
    ET_PARQUET_TMPDIR                  spill directory for parquet staging
                                       (default: system temp dir)
    ET_UPLOAD_TAG                      run prefix for uploaded artifacts
    ET_PARQUET_BACKUP_DIR              optional local backup dir for flushes
    ET_UPLOAD_GCS                      "1" enables GCS upload of results
    ET_MINER_GCS_BUCKET                destination bucket URI
    ET_MINER_GCS_CREDENTIALS           path to a service-account JSON
    GCS_TOKEN                          raw OAuth token (alternative to creds)
    ET_MINER_LOG_DIR                   directory for optional file logging
    ET_MINER_FILTER_IMPL               dense-count threshold filter impl:
                                       "compact" (default) | "cupy" | "cpu"
                                       (see et_miner.gpu.kernels.filter)
    ET_MINER_MAX_CHUNK_CANDS           caps the measured dense-chunk budget
                                       (candidates per chunk) — lets tests
                                       force multi-chunk runs on small data
    ET_MINER_DISABLE_NCCL              "1" skips NCCL init and forces the
                                       staged D2D reduce fallback
    ET_MINER_ROW_BALANCE               multi-GPU row-split mode: "rows"
                                       (default, equal row counts) or "nnz"
                                       (equal cumulative nnz cuts)
    ET_MINER_KERNEL_VARIANT            removed: setting it raises ValueError
                                       (the kernel is chosen per prefix
                                       group; pin it with the next knob)
    ET_MINER_TILED_MIN_GROUP_PAIRS     int, pins the candidate pairs a
                                       prefix group needs for the tiled
                                       kernel; smaller groups run on the
                                       per-candidate kernel. 0 = tiled for
                                       every group. Unset: the measured
                                       crossover per K (see
                                       et_miner.gpu.row_split_chunks)
    ET_MINER_DISABLE_RUST              "1" runs as if the Rust extension were
                                       not built: every Rust role takes its
                                       fallback. Read once, at import of
                                       et_miner.backends, so set it before
                                       importing et_miner.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def flush_parallel_threshold() -> int:
    return int(os.environ.get("ET_MINER_FLUSH_PARALLEL_THRESHOLD", "100000000"))


def legacy_write() -> bool:
    return os.environ.get("ET_MINER_LEGACY_WRITE") == "1"


def flush_compression() -> str:
    return os.environ.get("ET_FLUSH_COMPRESSION", "zstd")


def flush_chunk_size() -> int:
    return int(os.environ.get("ET_FLUSH_CHUNK_SIZE", "50000000"))


def flush_threads() -> int:
    return int(os.environ.get("ET_FLUSH_THREADS", "4"))


def parquet_tmpdir() -> str:
    return os.environ.get("ET_PARQUET_TMPDIR", tempfile.gettempdir())


def upload_tag(default: str) -> str:
    return os.environ.get("ET_UPLOAD_TAG", default)


def parquet_backup_dir() -> str | None:
    return os.environ.get("ET_PARQUET_BACKUP_DIR")


def upload_gcs_enabled() -> bool:
    return os.environ.get("ET_UPLOAD_GCS", "").strip() == "1"


def gcs_bucket() -> str:
    return os.environ.get("ET_MINER_GCS_BUCKET", "gs://et-miner-results")


def gcs_credentials_path() -> str | None:
    return os.environ.get("ET_MINER_GCS_CREDENTIALS")


def gcs_token() -> str | None:
    return os.environ.get("GCS_TOKEN")


def filter_impl() -> str:
    return os.environ.get("ET_MINER_FILTER_IMPL", "compact").strip().lower()


def _int_env(name: str, default: int | None) -> int | None:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {v!r}") from None


def max_chunk_candidates() -> int | None:
    return _int_env("ET_MINER_MAX_CHUNK_CANDS", None)


def disable_nccl() -> bool:
    return os.environ.get("ET_MINER_DISABLE_NCCL", "").strip() == "1"


def row_balance() -> str:
    return os.environ.get("ET_MINER_ROW_BALANCE", "rows").strip().lower()


def reject_removed_knobs() -> None:
    """Raise for a knob that no longer exists rather than ignore it."""
    if "ET_MINER_KERNEL_VARIANT" in os.environ:
        raise ValueError(
            "ET_MINER_KERNEL_VARIANT was removed: the row-split miner picks the tiled or the "
            "per-candidate kernel per prefix group, at the measured crossover. "
            "ET_MINER_TILED_MIN_GROUP_PAIRS pins that choice (0 = tiled for every group). Unset it."
        )


def tiled_min_group_pairs() -> int | None:
    return _int_env("ET_MINER_TILED_MIN_GROUP_PAIRS", None)


def log_dir() -> Path:
    configured = os.environ.get("ET_MINER_LOG_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "et-miner" / "logs"


def disable_rust() -> bool:
    return os.environ.get("ET_MINER_DISABLE_RUST", "").strip() == "1"
