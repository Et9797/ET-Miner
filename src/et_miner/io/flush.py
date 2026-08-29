"""Parquet flush-to-disk for per-K frequent-itemset results.

Streams (itemset, support) arrays to frequent_k{k}.parquet (single table) or
frequent_k{k}/part_*.parquet (parallel partitioned) with optional GCS upload
and local backup. Behavior knobs are the ET_FLUSH_* / ET_MINER_* environment
variables documented in et_miner._env, read at call time.
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

from et_miner import _env
from et_miner.io.gcs import (
    GCSUploader,
    join_gs_uri,
    parse_gs_url,
    pyarrow_gcs_filesystem,
)


def _write_partition(
    items_flat,
    supports,
    pidx,
    pstart,
    pend,
    local_part_dir,
    schema,
    comp_kw,
    chunk_size,
    k_width,
):
    """Write rows [pstart, pend) of items_flat/supports to part_{pidx:03d}.parquet."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    part_path = os.path.join(local_part_dir, f"part_{pidx:03d}.parquet")
    writer = pq.ParquetWriter(part_path, schema, **comp_kw)
    try:
        for cs in range(pstart, pend, chunk_size):
            ce = min(cs + chunk_size, pend)
            cn = ce - cs
            chunk_flat = items_flat[cs:ce].ravel()
            chunk_offsets = np.arange(0, (cn + 1) * k_width, k_width, dtype=np.int64)
            chunk_list = pa.LargeListArray.from_arrays(chunk_offsets, chunk_flat)
            chunk_table = pa.table(
                {"itemset": chunk_list, "support": supports[cs:ce]},
                schema=schema,
            )
            writer.write_table(chunk_table, row_group_size=cn)
            del chunk_list, chunk_table, chunk_offsets, chunk_flat
    finally:
        writer.close()
    return part_path


def _flush_k_parquet(
    items_flat,
    supports,
    k_level: int,
    output_dir: str,
    is_remote: bool,
    uploader: GCSUploader | None,
    backup_dir: str | None = None,
) -> None:
    """Flush K-level results to parquet — module-level, closure-free.

    Five branches preserved (K=1-4 single-table proof):
      single-table local           (n < threshold, local)
      single-table direct gs://    (n < threshold, gs://)
      parallel partitioned local   (n >= threshold, local — 7.6× speedup)
      partitioned NVMe-first → gs  (n >= threshold, gs:// — async upload)
      disk-low fallback chunked    (n >= threshold, gs://, low-disk single-stream)

    Caller must wrap with writeable=False/True try/finally for the parallel
    branch (Auditor R2 voorwaarde 1). This function does NOT touch flags.

    Threshold via ET_MINER_FLUSH_PARALLEL_THRESHOLD (default 100M rows).
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor as _FlushTPE

    n = items_flat.shape[0]
    k_width = items_flat.shape[1] if items_flat.ndim == 2 else 1

    chunk_threshold = _env.flush_parallel_threshold()
    legacy = _env.legacy_write()
    use_parallel = n >= chunk_threshold and not legacy

    flush_comp = _env.flush_compression()
    comp_kw: dict[str, Any] = {"compression": flush_comp, "use_dictionary": False}
    if flush_comp == "zstd":
        comp_kw["compression_level"] = 2

    if not use_parallel:
        # ── Single-table path: small outputs (<threshold) or legacy override ──
        out_path = join_gs_uri(output_dir, f"frequent_k{k_level}.parquet")
        flat_items = items_flat.ravel()
        offsets = np.arange(0, (n + 1) * k_width, k_width, dtype=np.int64)
        list_arr = pa.LargeListArray.from_arrays(offsets, flat_items)
        arrow_table = pa.table({"itemset": list_arr, "support": supports})
        if is_remote:
            bucket, blob = parse_gs_url(out_path)
            pq.write_table(
                arrow_table,
                f"{bucket}/{blob}",
                filesystem=pyarrow_gcs_filesystem(),
                **comp_kw,
            )
        else:
            pq.write_table(arrow_table, out_path, **comp_kw)
        del arrow_table

        if is_remote:
            logger.info(f"  → Flushed {n:,} itemsets to {out_path} ({flush_comp}, direct gs://)")
        else:
            size_mb = os.path.getsize(out_path) / (1024**2)
            logger.info(f"  → Flushed {n:,} itemsets to {out_path} ({size_mb:.0f} MB, {flush_comp})")

        if uploader is not None and not is_remote:
            uploader.upload(out_path)

        if backup_dir and not is_remote:
            import subprocess

            os.makedirs(backup_dir, exist_ok=True)
            subprocess.run(["cp", out_path, backup_dir + "/"], check=True)
            logger.info(f"  → Backup: {out_path} → {backup_dir}/")
        return

    # ── Parallel partitioned write ──
    chunk_size = _env.flush_chunk_size()
    n_threads = min(
        _env.flush_threads(),
        max(1, (os.cpu_count() or 4) // 4),
    )

    schema = pa.schema(
        [
            ("itemset", pa.large_list(pa.from_numpy_dtype(items_flat.dtype))),
            ("support", pa.from_numpy_dtype(supports.dtype)),
        ]
    )

    if is_remote:
        tmpdir = _env.parquet_tmpdir()
        os.makedirs(tmpdir, exist_ok=True)
        local_part_dir = os.path.join(tmpdir, f"frequent_k{k_level}")
    else:
        local_part_dir = os.path.join(output_dir, f"frequent_k{k_level}")
    os.makedirs(local_part_dir, exist_ok=True)

    # Disk pre-check — fall back to single-stream gs:// write when low
    import shutil

    est_compressed_gb = (n * (k_width * 4 + 8)) / (2.8 * 1024**3)
    free_gb = shutil.disk_usage(local_part_dir).free / 1024**3
    if free_gb < est_compressed_gb * 1.2:
        logger.warning(
            f"  ⚠ Low disk: {free_gb:.0f} GB free, ~{est_compressed_gb:.0f} GB needed. "
            f"Falling back to single-threaded direct write."
        )
        out_path = join_gs_uri(output_dir, f"frequent_k{k_level}.parquet")
        if is_remote:
            fs = pyarrow_gcs_filesystem()
            bucket, blob = parse_gs_url(out_path)
            writer = pq.ParquetWriter(f"{bucket}/{blob}", schema, filesystem=fs, **comp_kw)
        else:
            writer = pq.ParquetWriter(out_path, schema, **comp_kw)
        try:
            for cs in range(0, n, chunk_size):
                ce = min(cs + chunk_size, n)
                cn = ce - cs
                chunk_flat = items_flat[cs:ce].ravel()
                chunk_offs = np.arange(0, (cn + 1) * k_width, k_width, dtype=np.int64)
                chunk_list = pa.LargeListArray.from_arrays(chunk_offs, chunk_flat)
                chunk_table = pa.table(
                    {"itemset": chunk_list, "support": supports[cs:ce]},
                    schema=schema,
                )
                writer.write_table(chunk_table, row_group_size=cn)
                del chunk_list, chunk_table, chunk_offs, chunk_flat
        finally:
            writer.close()
        logger.info(f"  → Flushed {n:,} itemsets to {out_path} (fallback)")
        return

    psz = (n + n_threads - 1) // n_threads
    bounds = [(i * psz, min((i + 1) * psz, n)) for i in range(n_threads) if i * psz < n]
    bounds = [(s, e) for s, e in bounds if e > s]

    t_flush = time.perf_counter()
    try:
        if len(bounds) > 1:
            with _FlushTPE(max_workers=len(bounds)) as pool:
                futs = [
                    pool.submit(
                        _write_partition,
                        items_flat,
                        supports,
                        i,
                        s,
                        e,
                        local_part_dir,
                        schema,
                        comp_kw,
                        chunk_size,
                        k_width,
                    )
                    for i, (s, e) in enumerate(bounds)
                ]
                part_paths = [f.result() for f in futs]
        else:
            part_paths = [
                _write_partition(
                    items_flat,
                    supports,
                    0,
                    *bounds[0],
                    local_part_dir,
                    schema,
                    comp_kw,
                    chunk_size,
                    k_width,
                )
            ]
    except Exception:
        for pidx in range(len(bounds)):
            p = os.path.join(local_part_dir, f"part_{pidx:03d}.parquet")
            if os.path.exists(p):
                os.unlink(p)
        raise
    flush_dt = time.perf_counter() - t_flush

    total_mb = sum(os.path.getsize(p) for p in part_paths) / (1024**2)
    logger.info(
        f"  → Flushed {n:,} itemsets to {local_part_dir}/ "
        f"({len(part_paths)} parts, {total_mb:.0f} MB, {flush_comp}, "
        f"{flush_dt:.1f}s, {len(bounds)} threads)"
    )

    if uploader is not None:
        if is_remote:
            for p in part_paths:
                gs_dest = join_gs_uri(
                    output_dir,
                    f"frequent_k{k_level}/{os.path.basename(p)}",
                )
                uploader.upload_to_uri(p, gs_dest)
        else:
            for p in part_paths:
                uploader.upload(p)

    if backup_dir and not is_remote:
        backup_dest = os.path.join(backup_dir, f"frequent_k{k_level}")
        shutil.copytree(local_part_dir, backup_dest, dirs_exist_ok=True)
        logger.info(f"  → Backup: {local_part_dir}/ → {backup_dest}/")


