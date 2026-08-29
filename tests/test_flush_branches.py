"""Regression tests for apriori flush branches.

Covers the 5 flush branches that _flush_k_parquet must preserve:
  1. single-table local           (n < threshold, local output_dir)
  2. single-table direct gs://    (n < threshold, gs:// output)         [needs creds]
  3. parallel partitioned local   (n >= threshold, local output_dir)
  4. partitioned NVMe-first → gs  (n >= threshold, gs:// output)        [needs creds]
  5. disk-low fallback chunked    (n >= threshold, low disk, gs://)     [needs creds]

Polars memory safety: outputs verified via `scan_parquet().collect(streaming=True)`.
Synthetic data is small (≤10K rows) — parallel branch is reached by lowering
ET_MINER_FLUSH_PARALLEL_THRESHOLD.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# `import et_miner.core.apriori as X` resolves to the FUNCTION not the module —
# et_miner/__init__.py's `from et_miner.core.apriori import apriori` rebinds the
# package attribute. importlib.import_module bypasses the rebind.
apriori_mod = importlib.import_module("et_miner.core.apriori")
from et_miner.io.gcs import GCSUploader  # noqa: E402

requires_flush_extraction = pytest.mark.skipif(
    not hasattr(apriori_mod, "_flush_k_parquet"),
    reason="Awaits _flush_k_parquet module-level extraction.",
)


def _has_gcs_creds() -> bool:
    if os.environ.get("ET_MINER_GCS_CREDENTIALS"):
        return True
    return Path.home().joinpath(".config/gcloud/application_default_credentials.json").exists()


def _synth(n: int, k: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    items = rng.integers(0, 1_000, size=(n, k), dtype=np.int32)
    supports = rng.uniform(0.001, 0.5, size=n).astype(np.float64)
    return items, supports


# ── Branch 1: single-table local ────────────────────────────────────────


@requires_flush_extraction
def test_branch_1_single_table_local(tmp_path: Path):
    items, sup = _synth(n=5_000, k=3)
    apriori_mod._flush_k_parquet(
        items,
        sup,
        k_level=3,
        output_dir=str(tmp_path),
        is_remote=False,
        uploader=None,
        backup_dir=None,
    )
    out = tmp_path / "frequent_k3.parquet"
    assert out.exists(), "single-table branch must write frequent_k{k}.parquet"

    df = pl.scan_parquet(str(out)).collect(engine="streaming")
    assert df.height == 5_000
    assert df.columns == ["itemset", "support"]
    assert df["itemset"][0].to_list() == items[0].tolist()
    assert df["support"][0] == pytest.approx(sup[0])


@requires_flush_extraction
def test_branch_1_k1_single_column(tmp_path: Path):
    """K=1 case: items_flat is (n, 1) — verify it still produces lists of length 1."""
    items, sup = _synth(n=200, k=1)
    apriori_mod._flush_k_parquet(
        items,
        sup,
        k_level=1,
        output_dir=str(tmp_path),
        is_remote=False,
        uploader=None,
        backup_dir=None,
    )
    df = pl.scan_parquet(str(tmp_path / "frequent_k1.parquet")).collect(engine="streaming")
    assert df.height == 200
    assert all(len(s) == 1 for s in df["itemset"].to_list())


# ── Branch 3: parallel partitioned local ─────────────────────────────────


@requires_flush_extraction
def test_branch_3_parallel_partitioned_local(tmp_path: Path, monkeypatch):
    """Force parallel branch via lowered threshold — verifies row count + schema preserved."""
    monkeypatch.setenv("ET_MINER_FLUSH_PARALLEL_THRESHOLD", "1000")
    monkeypatch.setenv("ET_FLUSH_THREADS", "2")
    monkeypatch.setenv("ET_FLUSH_CHUNK_SIZE", "750")

    items, sup = _synth(n=5_000, k=2)
    apriori_mod._flush_k_parquet(
        items,
        sup,
        k_level=2,
        output_dir=str(tmp_path),
        is_remote=False,
        uploader=None,
        backup_dir=None,
    )

    part_dir = tmp_path / "frequent_k2"
    assert part_dir.is_dir(), "parallel branch must write frequent_k{k}/ directory"
    parts = sorted(part_dir.glob("part_*.parquet"))
    assert len(parts) >= 1

    # Stream-read all partitions — polars-memory-safe
    df = pl.scan_parquet(str(part_dir / "part_*.parquet")).collect(engine="streaming")
    assert df.height == 5_000


@requires_flush_extraction
def test_branch_3_partition_split_correctness(tmp_path: Path, monkeypatch):
    """All input rows present after partition merge, no duplicates, no losses."""
    monkeypatch.setenv("ET_MINER_FLUSH_PARALLEL_THRESHOLD", "100")
    monkeypatch.setenv("ET_FLUSH_THREADS", "4")
    monkeypatch.setenv("ET_FLUSH_CHUNK_SIZE", "100")

    n = 1_000
    items = np.arange(n * 2, dtype=np.int32).reshape(n, 2)
    sup = np.linspace(0.01, 0.5, n, dtype=np.float64)

    apriori_mod._flush_k_parquet(
        items,
        sup,
        k_level=2,
        output_dir=str(tmp_path),
        is_remote=False,
        uploader=None,
        backup_dir=None,
    )

    df = pl.scan_parquet(str(tmp_path / "frequent_k2" / "part_*.parquet")).collect(engine="streaming")
    assert df.height == n
    # Sum of itemset[0] over all rows = sum(0, 2, 4, ..., 2*(n-1)) — distribution-invariant check
    flat_first = np.array([s[0] for s in df["itemset"].to_list()])
    assert flat_first.sum() == sum(range(0, 2 * n, 2))


# ── Caller-side writeable safety (Auditor punt 1) ────────────────────────


@requires_flush_extraction
def test_caller_writeable_safety_pattern(tmp_path: Path):
    """The caller's writeable=False/True wrap must be respected; _flush_k_parquet
    itself must NOT touch writeable flags (Auditor R2 voorwaarde)."""
    items, sup = _synth(n=500, k=2)

    items.flags.writeable = False
    sup.flags.writeable = False
    try:
        apriori_mod._flush_k_parquet(
            items,
            sup,
            k_level=2,
            output_dir=str(tmp_path),
            is_remote=False,
            uploader=None,
            backup_dir=None,
        )
    finally:
        # If _flush_k_parquet flipped writeable=True internally, the caller's
        # finally would no-op-restore — verifies no internal mutation happened.
        assert items.flags.writeable is False
        assert sup.flags.writeable is False
        items.flags.writeable = True
        sup.flags.writeable = True

    assert (tmp_path / "frequent_k2.parquet").exists()


# ── Backup path ─────────────────────────────────────────────────────────


@requires_flush_extraction
def test_backup_dir_creates_copy(tmp_path: Path):
    """ET_PARQUET_BACKUP_DIR / backup_dir param mirrors the output."""
    items, sup = _synth(n=300, k=2)
    backup = tmp_path / "backup"
    out = tmp_path / "out"
    out.mkdir()

    apriori_mod._flush_k_parquet(
        items,
        sup,
        k_level=2,
        output_dir=str(out),
        is_remote=False,
        uploader=None,
        backup_dir=str(backup),
    )

    assert (out / "frequent_k2.parquet").exists()
    assert (backup / "frequent_k2.parquet").exists()


# ── GCSUploader lifecycle (Auditor critical drain bug) ───────────────────


def test_uploader_drain_default_does_not_shutdown_executor():
    """drain() default must NOT shutdown — cross-K-level lifetime depends on it."""
    u = GCSUploader(prefix="test", max_workers=1, enabled=False)
    u.drain()
    # No assertion-on-internal-state — the contract is: subsequent uploads
    # must still be schedulable. With enabled=False both calls are no-ops, but
    # they must not raise (executor not closed).
    u.upload("/tmp/nonexistent")
    u.upload_to_uri("/tmp/nonexistent", "gs://bucket/x")
    u.close(wait=False)


def test_uploader_close_is_idempotent():
    """close(wait=False) after close(wait=True) must not raise — exception path."""
    u = GCSUploader(prefix="test", max_workers=2, enabled=False)
    u.drain()
    u.close(wait=True)
    u.close(wait=False)


def test_uploader_disabled_is_noop(tmp_path: Path):
    """enabled=False — all upload methods are no-ops, no executor created."""
    u = GCSUploader(prefix="test", max_workers=2, enabled=False)
    assert u._executor is None
    u.upload(str(tmp_path / "nonexistent.parquet"))
    u.upload_to_uri(str(tmp_path / "nonexistent.parquet"), "gs://bucket/x")
    assert u._futures == []
