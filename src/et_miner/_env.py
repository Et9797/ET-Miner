"""Typed, call-time accessors for every ET_* environment variable.

All ad-hoc environment reads in the package go through this module so the
full knob surface is documented in one place. Getters read the environment
at call time (never at import time) so `monkeypatch.setenv` in tests and
late exports in job scripts both behave as expected.

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


def max_chunk_candidates() -> int | None:
    v = os.environ.get("ET_MINER_MAX_CHUNK_CANDS")
    return int(v) if v else None


def disable_nccl() -> bool:
    return os.environ.get("ET_MINER_DISABLE_NCCL", "").strip() == "1"


def log_dir() -> Path:
    configured = os.environ.get("ET_MINER_LOG_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "et-miner" / "logs"
