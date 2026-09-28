"""Row-split GPU mining (the in-core GPU miner) and two-phase (anchor + zoom) mining.

Splits transactions across one or more GPUs by row range, mines each shard with
the bitvec kernels, and sums the per-GPU counts (NCCL reduce, or the staged
D2D fallback; nothing to reduce on one GPU). mine_two_phase drives two
apriori() passes (anchor discovery, then neighborhood zoom). Import-safe
without CuPy; GPU work happens only at call time.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from et_miner import _env
from et_miner.core.profiling import ProfilingSession
from et_miner.core.result import (
    _build_result_df,
    _empty_result,
    _min_count,
)
from et_miner.gpu.mining import (
    _anchor_keep_mask,
    _prune_non_free_mask,
    _rows_sorted,
)
from et_miner.gpu.nccl import _init_nccl
from et_miner.gpu.row_split_chunks import (
    compute_chunk_budget,
    plan_group_chunks,
    run_chunked_dense_level,
)
from et_miner.io.flush import _flush_k_parquet
from et_miner.io.gcs import (
    GCSUploader,
    is_gs_uri,
    is_upload_enabled,
    polars_storage_options,
)

if TYPE_CHECKING:
    # numpy is imported inside the functions that use it, alongside the
    # optional cupy import; this binds the name for annotations only.
    import numpy as np


def shard_prebuilt_bitvecs(bitvecs_gpu, n_transactions: int, n_gpus: int, devices=None):
    """Row-split a caller's (n_cols, n_u64s) bitvecs across up to ``n_gpus`` devices.

    Shards are cut at 64-row word boundaries and copied onto their devices (the
    caller's array is read-only to the engine; the copy stages through the host
    where devices have no peer access). On one device the caller's array is the
    only shard. ``devices`` names the shard devices explicitly, repeats allowed.

    Returns:
        ``bitvecs_list`` for ``_apriori_row_split_multi_gpu``: (array, device, rows).
    """
    import cupy as cp
    import numpy as np

    n_u64s = bitvecs_gpu.shape[1]
    if devices is None:
        devices = list(range(max(1, min(int(n_gpus), cp.cuda.runtime.getDeviceCount(), n_u64s))))
    n_dev = len(devices)
    if n_dev == 1 and devices[0] == bitvecs_gpu.device.id:
        return [(bitvecs_gpu, int(bitvecs_gpu.device.id), n_transactions)]
    cuts = np.linspace(0, n_u64s, n_dev + 1).astype(np.int64)
    shards = []
    for i, dev in enumerate(devices):
        lo, hi = int(cuts[i]), int(cuts[i + 1])
        with cp.cuda.Device(bitvecs_gpu.device.id):
            src = cp.ascontiguousarray(bitvecs_gpu[:, lo:hi])
        with cp.cuda.Device(dev):
            dst = cp.empty(src.shape, dtype=src.dtype)
            dst.data.copy_from_device(src.data, src.nbytes)
            cp.cuda.Device().synchronize()
        del src
        shards.append((dst, dev, min(hi * 64, n_transactions) - lo * 64))
    logger.info(f"  Pre-built bitvecs: split into {n_dev} row shards")
    return shards


def _apriori_row_split_multi_gpu(
    csr,  # scipy CSR matrix (None when bitvecs_list provided)
    col_to_item: dict[int, int],
    n_transactions: int,
    min_support: float,
    max_length: int | None,
    n_gpus: int,
    level_callback=None,
    bitvecs_list=None,  # Pre-built row-split bitvecs: list[(gpu_array, dev_id, n_rows)]
    output_dir=None,  # Per-K Parquet flush: write frequent_k{k}.parquet per level
    resume_from_k: int | None = None,  # Resume from K=N+1, loading K=N from parquet
    prune_non_free: bool = False,  # keep only free-sets (generators) per level
    anchor_items: set | None = None,  # V3 B6: two-phase anchor filtering
    profile: bool = False,
    max_ram_gb: float | None = None,
    max_vram_gb: float | None = None,
) -> "pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]":
    """Mine frequent itemsets using row-split bitvecs across multiple GPUs.

    Splits the bitvec by transaction rows across GPUs. Each GPU holds ~1/n_gpus
    of the data.

    GPU-resident dense counting architecture, per VRAM-budgeted candidate
    chunk (both K=2 and K>=3 — see gpu.row_split_chunks):
      1. Each GPU runs the dense kernel on the chunk → int32 partial counts
      2. ncclReduce to GPU 0 (or the bounded staged D2D fallback)
      3. compact_threshold on GPU 0 → only survivors cross PCIe to CPU

    Since all GPUs share the same prev_frequent, they generate the same
    candidates in the same deterministic order. The dense output at index i
    from GPU 0 is the partial count for the same candidate as index i from
    GPU 1. Element-wise sum = exact global counts.

    PCIe transfer per chunk: only survivors × 12 bytes (int64 index +
    int32 count). For K=2 with 35K features: ~600K frequent pairs × 12 =
    7.2 MB instead of the 2.4 GB dense array. (The default `compact`
    filter keeps this guarantee at any survivor count; the `cpu` A/B
    baseline impl deliberately re-enacts the historical full-array D2H.)

    ``prune_non_free`` keeps two populations per level, and which one each
    consumer gets is the whole correctness story:

      * ``prev_frequent_flat`` — the free-sets, i.e. what this level EMITS and
        what the next level's prefix join generates from. Emitting a level and
        then generating from a smaller one is what silently dropped frequent,
        apriori-valid itemsets from K=5 on: the output advertised itemsets the
        run would never extend.
      * ``prev_full_flat`` — the complete frequent level, which every subset
        test of the next level resolves against (both the apriori prune and the
        free-set test). Testing against the pruned level instead misses subsets
        that were themselves pruned, and under-prunes.

    With ``prune_non_free=False`` the two are the same array, so the unpruned
    path carries no extra cost and its output is the complete lattice.

    On one GPU a level whose dense counts would need more than one chunk (the
    K=2 pair space, or a prefix group larger than the chunk budget) is counted
    with the fused tiled kernel instead, which keeps only the survivors.

    ``profile`` returns ``(frame, ProfilingSession)`` with one phase per level.
    ``max_ram_gb`` / ``max_vram_gb`` raise MemoryError between levels once host
    RSS or the largest device pool exceeds them (None disables the guard).
    """
    import numpy as np

    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy required for multi-GPU mining")

    from concurrent.futures import ThreadPoolExecutor

    from et_miner.gpu.csr_bitvec import build_bitvecs_row_split
    from et_miner.gpu.kernels import (
        get_popcount_kernel,
        count_pairs_k2_allcounts,
        count_k3plus_allcounts,
        count_tiled_fused,
        k2_groups,
        select_k3plus_groups,
        upload_k3plus_groups,
        build_k3plus_groups_from_flat,
        decode_k2_pairs_flat,
        decode_k3plus_flat,
    )

    if n_transactions > np.iinfo(np.int32).max:
        raise ValueError(
            f"n_transactions={n_transactions:,} exceeds int32 max ({np.iinfo(np.int32).max:,}). "
            f"CSR tidset indices are int32 — upgrade to int64 before running at this scale."
        )

    min_count_threshold = _min_count(min_support, n_transactions)
    session = ProfilingSession() if profile else None

    def _result(df):
        return (df, session) if profile else df

    logger.info(
        f"  Row-split multi-GPU: {n_gpus} GPUs, min_count={min_count_threshold:,} (GPU-resident dense counting)"
    )

    # Phase 0: Build row-split bitvecs across GPUs.
    if bitvecs_list is None:
        t0 = time.perf_counter()
        bitvecs_list = build_bitvecs_row_split(csr, n_gpus)
        del csr
        build_time = time.perf_counter() - t0
        logger.info(f"  Bitvec build: {build_time:.1f}s across {len(bitvecs_list)} GPUs")
    else:
        logger.info(f"  Pre-built bitvecs: {len(bitvecs_list)} GPUs")

    n_cols = max(col_to_item.keys()) + 1 if col_to_item else 0
    effective_max_length = min(
        max_length if max_length else float("inf"),
        n_cols,
    )

    # int32 vocab IDs (V5=149, AlphaFold=35K) — halves items_flat in the parquet flush
    if col_to_item:
        _max_item = max(col_to_item.values())
        assert _max_item < 2**31, f"item ID {_max_item} exceeds int32 range"
    col_to_item_arr = np.zeros(n_cols, dtype=np.int32)
    for c, item in col_to_item.items():
        col_to_item_arr[c] = item

    # V3 B6: Convert anchor item IDs → column indices for fast filtering
    anchor_col_arr = None
    if anchor_items is not None:
        item_to_col = {int(col_to_item_arr[i]): i for i in range(n_cols)}
        anchor_cols = sorted(item_to_col[aid] for aid in anchor_items if aid in item_to_col)
        if anchor_cols:
            anchor_col_arr = np.array(anchor_cols, dtype=np.int32)
            logger.info(f"  Anchor filter: {len(anchor_col_arr)} of {len(anchor_items)} anchor items mapped to columns")
        else:
            logger.warning("  No anchor items mapped to columns — anchor filter disabled")

    device_ids = [did for _, did, _ in bitvecs_list]
    _one_device = len(device_ids) == 1
    if _one_device:
        # One shard is already the sum: no communicator, and the chunk budget
        # reserves no reduce workspace (the in-place case).
        nccl_comms, _use_nccl = None, True
    else:
        nccl_comms, _use_nccl = _init_nccl(device_ids)
        if _use_nccl:
            logger.info(f"  NCCL: {len(device_ids)} communicators (ring all-reduce)")
        else:
            logger.info("  NCCL unavailable, using sequential D2D")

    def _memory_gb() -> tuple[float, float]:
        """(host RSS, largest device-pool use) in GB."""
        import psutil

        vram = 0.0
        for did in device_ids:
            with cp.cuda.Device(did):
                vram = max(vram, cp.get_default_memory_pool().used_bytes() / (1 << 30))
        return psutil.Process(os.getpid()).memory_info().rss / (1 << 30), vram

    def _check_memory_guard(k_level: int) -> None:
        if max_ram_gb is None and max_vram_gb is None:
            return
        ram_gb, vram_gb = _memory_gb()
        for used, limit, what in ((ram_gb, max_ram_gb, "RAM"), (vram_gb, max_vram_gb, "VRAM")):
            if limit is not None and used > limit:
                raise MemoryError(
                    f"Memory guard tripped after K={k_level}: {what}={used:.1f}GB exceeds "
                    f"max_{what.lower()}_gb={limit}GB. The lattice is INCOMPLETE at this point, so "
                    f"it is not returned. Raise max_{what.lower()}_gb, lower max_length, or pass "
                    f"output_dir so each level is flushed as it completes."
                )

    # Per-K Parquet flush: write each K level to disk immediately.
    # Prevents 680 GB CPU RAM accumulation at K=7+ scale.
    _output_is_remote = bool(output_dir) and is_gs_uri(output_dir)
    if output_dir and not _output_is_remote:
        os.makedirs(output_dir, exist_ok=True)

    # Deferred results: accumulate numpy arrays per level,
    # build Polars DataFrame at the end via PyArrow. No .tolist() overhead.
    # When output_dir is set, arrays flush to Parquet per K and are NOT accumulated.
    deferred_itemsets_np: list[np.ndarray] = []  # (n, k) int32 arrays (col_to_item_arr dtype)
    deferred_supports: list[np.ndarray] = []

    # GCSUploader: single instance for the whole K-loop.
    # enabled when local output + ET_UPLOAD_GCS=1, OR when output_dir is gs://
    # (NVMe-first strategy needs uploads regardless of the global gate).
    _upload_local = bool(output_dir) and not _output_is_remote
    _need_upload = (is_upload_enabled() and _upload_local) or _output_is_remote
    uploader = GCSUploader(
        prefix=_env.upload_tag(f"run_{int(time.time())}"),
        max_workers=2,
        enabled=_need_upload,
    )

    def _flush_or_defer(items_flat, supports, k_level):
        """Flush via module-level _flush_k_parquet (output_dir set) or accumulate.

        Caller-side writeable safety wraps the flush call (Auditor R2 voorwaarde 1):
        _flush_k_parquet itself does not touch flags.
        """
        if output_dir is None:
            deferred_itemsets_np.append(items_flat)
            deferred_supports.append(supports)
            return

        items_flat.flags.writeable = False
        supports.flags.writeable = False
        try:
            _flush_k_parquet(
                items_flat,
                supports,
                k_level,
                output_dir=output_dir,
                is_remote=_output_is_remote,
                uploader=uploader,
                backup_dir=_env.parquet_backup_dir(),
            )
        finally:
            items_flat.flags.writeable = True
            supports.flags.writeable = True

    # ── RESUME: skip K=1..resume_from_k, load prev_frequent from parquet ──
    _resume_active = bool(resume_from_k and resume_from_k >= 2 and output_dir)

    if _resume_active:
        from et_miner.io.gcs import clear_resolve_cache, resolve_k_parquet

        clear_resolve_cache()
        resume_path = resolve_k_parquet(output_dir, resume_from_k)
        logger.info(f"  RESUME: Loading K={resume_from_k} from {resume_path}")
        t_resume = time.perf_counter()

        if _output_is_remote:
            table = pl.scan_parquet(resume_path, storage_options=polars_storage_options()).collect().to_arrow()
        else:
            import pyarrow.parquet as pq

            table = pq.read_table(resume_path)
        itemsets_col = table.column("itemset")

        # Vectorized item_id → col_idx conversion via numpy lookup array
        max_item_id = max(col_to_item.values())
        item_to_col = np.full(max_item_id + 1, -1, dtype=np.int32)
        for col_idx, item_id in col_to_item.items():
            item_to_col[item_id] = col_idx

        # Extract flat values from ChunkedArray<LargeList> — combine chunks first
        flat_item_ids = itemsets_col.combine_chunks().values.to_numpy()
        flat_col_ids = item_to_col[flat_item_ids]

        n_loaded = len(itemsets_col)
        prev_frequent_flat = flat_col_ids.reshape(n_loaded, resume_from_k).astype(np.int32)

        # Reconstruct raw counts from the flushed support column, so the first
        # resumed level keeps the free-set prune. count → support → count
        # round-trips exactly through float64 for any int32-range count.
        if "support" in table.column_names:
            supports_np = table.column("support").combine_chunks().to_numpy(zero_copy_only=False)
            prev_counts_flat = np.rint(supports_np.astype(np.float64) * n_transactions).astype(np.int64)
        else:
            prev_counts_flat = None  # counts unknown — the prune and auto transition wait one level
        del table, itemsets_col, flat_item_ids, flat_col_ids, item_to_col
        if prune_non_free and not _rows_sorted(prev_frequent_flat):
            # Parquet order is not row-sorted; the free-set binary search needs it.
            _order = np.lexsort(prev_frequent_flat[:, ::-1].T)
            prev_frequent_flat = prev_frequent_flat[_order]
            if prev_counts_flat is not None:
                prev_counts_flat = prev_counts_flat[_order]

        # The parquet holds the EMITTED level, which under prune_non_free is the
        # free subset — the complete level of K=resume_from_k was never written.
        # Generation resumes exactly (it reads the free-sets), but the first
        # resumed level's free-set test resolves against the free subset instead
        # of the complete level, which can only under-prune (keep a few extra).
        prev_full_flat = prev_frequent_flat
        prev_full_counts = prev_counts_flat
        if prune_non_free:
            logger.warning(
                f"  RESUME: K={resume_from_k} parquet holds the free-sets, not the complete level — "
                f"the K={resume_from_k + 1} free-set test may under-prune slightly. "
                "Levels after that are exact."
            )

        resume_time = time.perf_counter() - t_resume
        logger.info(f"  RESUME: {n_loaded:,} itemsets → col indices in {resume_time:.1f}s")

        k = resume_from_k + 1
        logger.info(f"  RESUME: Jumping to K={k} ({len(prev_frequent_flat):,} itemsets)")

    # ── K=1: parallel popcount across GPUs, sum ────────────────────────
    if not _resume_active:
        _k1_start = time.perf_counter()
        if session:
            session.start_phase("k1_support")
        popcount_kernel = get_popcount_kernel()

        CHUNK_COLS = 4096  # ~14 GB temp per chunk — fits in remaining VRAM

        def _k1_popcount_on_gpu(bitvec_gpu, device_id):
            """Chunked popcount on one GPU — runs in thread for parallelism."""
            with cp.cuda.Device(device_id):
                local_counts = np.zeros(n_cols, dtype=np.int64)
                for c_start in range(0, n_cols, CHUNK_COLS):
                    c_end = min(c_start + CHUNK_COLS, n_cols)
                    chunk = bitvec_gpu[c_start:c_end]
                    popcounts = popcount_kernel(chunk.view(cp.uint64))
                    local_counts[c_start:c_end] = cp.sum(
                        popcounts.reshape(c_end - c_start, -1), axis=1, dtype=cp.int64
                    ).get()
                    del popcounts
                return local_counts

        with ThreadPoolExecutor(max_workers=len(bitvecs_list)) as pool:
            futures = [pool.submit(_k1_popcount_on_gpu, bv, did) for bv, did, _ in bitvecs_list]
            global_col_counts = sum(f.result() for f in futures)

        # Vectorized K=1 filtering — no Python loop
        freq_mask_k1 = global_col_counts >= min_count_threshold
        freq_col_indices = np.where(freq_mask_k1)[0]
        freq_col_counts = global_col_counts[freq_mask_k1]

        # K=1 frequent columns as flat (n, 1) array — already numpy
        prev_full_flat = freq_col_indices.astype(np.int32).reshape(-1, 1)
        prev_full_counts = freq_col_counts.astype(np.int64)  # preserved for the free-set test

        # Free-set semantics start at K=1: an item in EVERY transaction has the
        # empty set's support, so it is not a generator. Dropping it here is what
        # makes the kept level exactly the free-sets at every K.
        prev_frequent_flat = prev_full_flat
        prev_counts_flat = prev_full_counts
        if prune_non_free:
            _k1_free = prev_full_counts < n_transactions
            if not _k1_free.all():
                prev_frequent_flat = prev_full_flat[_k1_free]
                prev_counts_flat = prev_full_counts[_k1_free]
                logger.debug(
                    f"    Free-set pruning K=1: {len(prev_full_flat):,} → {len(prev_frequent_flat):,} "
                    "(items present in every transaction)"
                )

        if len(prev_frequent_flat) > 0:
            # K=1 is emit-filtered too. _anchor_keep_mask used to return None
            # for k<2, so a two-phase run emitted every frequent item at K=1 --
            # a separate inconsistency that disappears once anchoring is purely
            # an output selector. It is free: generation reads
            # prev_frequent_flat, not the emitted array.
            _k1_flat, _k1_counts = prev_frequent_flat, prev_counts_flat
            _k1_keep = _anchor_keep_mask(prev_frequent_flat, anchor_col_arr, 1)
            if _k1_keep is not None:
                _k1_flat = prev_frequent_flat[_k1_keep]
                _k1_counts = prev_counts_flat[_k1_keep]
            if len(_k1_flat) > 0:
                _flush_or_defer(
                    col_to_item_arr[_k1_flat], _k1_counts / n_transactions, 1
                )

        k1_time = time.perf_counter() - _k1_start
        if session:
            session.end_phase(n_candidates=n_cols, n_frequent=len(prev_frequent_flat))
        if level_callback:
            level_callback(1, n_cols, len(prev_frequent_flat), k1_time * 1000)
        logger.info(f"  K=1: {len(prev_frequent_flat):,} frequent items in {k1_time:.1f}s")

        if len(prev_frequent_flat) == 0:
            return _result(_build_result_df([]))
        _check_memory_guard(1)

        k = 2

    # ── K>=2: GPU-resident dense counting ────────────────────────────────
    # State: prev_frequent_flat — numpy (n_freq, k-1) array of column indices.
    # No Python tuples in the hot path. Results decoded at end of each level
    # via vectorized numpy. Next-level groups built from flat arrays directly.
    try:
        while k <= effective_max_length and prev_frequent_flat.shape[0] >= k:
            _k_start = time.perf_counter()
            if session:
                session.start_phase(f"k{k}")
            _n_cands_cb = 0

            if k == 2:
                freq_cols = sorted(prev_frequent_flat[:, 0])
                n_pairs = len(freq_cols) * (len(freq_cols) - 1) // 2
                _n_cands_cb = n_pairs

                # Chunked by measured VRAM budget — big cards get one chunk,
                # small (or pool-limited) cards split the pair space. The
                # pair space is one synthetic group: whole-in-one-chunk runs
                # may use the shared/tiled kernel, multi-chunk runs are
                # legacy sub-chunks (a partial pair range can't be tiled).
                _k2_budget = compute_chunk_budget(device_ids, group_data_bytes=0, use_nccl=_use_nccl)
                if _one_device and n_pairs > _k2_budget:
                    bv0, did0, _ = bitvecs_list[0]
                    logger.info(
                        f"  K=2: {n_pairs:,} total pairs exceed one dense chunk ({_k2_budget:,}); "
                        "fused tiled count on the one GPU"
                    )
                    with cp.cuda.Device(did0):
                        freq_pair_indices, freq_pair_counts = count_tiled_fused(
                            bv0, k2_groups(freq_cols), bv0.shape[1], min_count_threshold
                        )
                else:
                    _k2_cp = np.array([0, n_pairs], dtype=np.int64)
                    k2_chunks = plan_group_chunks(_k2_cp, _k2_budget)
                    logger.info(
                        f"  K=2: {n_pairs:,} total pairs, {len(k2_chunks)} chunk(s), "
                        f"dense output {min(_k2_budget, n_pairs) * 4 / (1 << 30):.2f} GB/GPU per chunk"
                    )

                    def _k2_chunk_on_gpu(bitvec_gpu, device_id, chunk):
                        with cp.cuda.Device(device_id):
                            return count_pairs_k2_allcounts(
                                bitvec_gpu,
                                freq_cols,
                                bitvec_gpu.shape[1],
                                chunk_start=chunk.start,
                                chunk_size=chunk.size,
                                variant="legacy" if chunk.use_legacy else None,
                            )

                    freq_pair_indices, freq_pair_counts = run_chunked_dense_level(
                        bitvecs_list,
                        k2_chunks,
                        _k2_chunk_on_gpu,
                        min_count_threshold,
                        nccl_comms,
                        _use_nccl,
                        level_label="K=2",
                    )

                n_freq = len(freq_pair_indices)
                if n_freq > 0:
                    current_flat = decode_k2_pairs_flat(freq_pair_indices, freq_cols)
                    current_counts_raw = freq_pair_counts  # already int64


                    # Pair cache disabled — not yet wired to K>=3 kernels (saves ~13.6 GB VRAM)
                else:
                    current_flat = np.empty((0, 2), dtype=np.int32)
                    current_counts_raw = np.empty(0, dtype=np.int64)

            else:
                # K>=3: build groups from flat array — no Python tuple grouping
                groups_info = build_k3plus_groups_from_flat(prev_frequent_flat)

                n_freq = 0
                current_flat = np.empty((0, k), dtype=np.int32)
                current_counts_raw = np.empty(0, dtype=np.int64)

                if groups_info is not None:
                    tc = groups_info.total_candidates
                    _n_cands_cb = tc

                    # VRAM budget for candidate-range chunking. Group data
                    # stays resident across all chunks; only the dense int32
                    # output (chunk_size × 4 bytes) varies per chunk.
                    group_data_bytes = (
                        len(groups_info.prefix_items) * 4
                        + len(groups_info.prefix_offsets) * 8
                        + len(groups_info.suffixes) * 4
                        + len(groups_info.suffix_offsets) * 8
                        + len(groups_info.cumulative_pairs) * 8
                    )

                    max_cands_per_chunk = compute_chunk_budget(
                        device_ids, group_data_bytes=group_data_bytes, use_nccl=_use_nccl
                    )

                    # (groups, survivor indices into them, counts) per counted part
                    parts = []
                    dense_groups = groups_info
                    if _one_device:
                        mega = np.diff(groups_info.cumulative_pairs) > max_cands_per_chunk
                        if mega.any():
                            fused_groups = select_k3plus_groups(groups_info, mega)
                            dense_groups = None if mega.all() else select_k3plus_groups(groups_info, ~mega)
                            logger.info(
                                f"  K={k}: {int(mega.sum()):,} prefix group(s), "
                                f"{fused_groups.total_candidates:,} candidates, exceed one dense chunk "
                                f"({max_cands_per_chunk:,}); fused tiled count on the one GPU"
                            )
                            bv0, did0, _ = bitvecs_list[0]
                            with cp.cuda.Device(did0):
                                parts.append(
                                    (fused_groups, *count_tiled_fused(bv0, fused_groups, bv0.shape[1], min_count_threshold))
                                )

                    if dense_groups is not None:
                        k3_chunks = plan_group_chunks(dense_groups.cumulative_pairs, max_cands_per_chunk)

                        logger.debug(
                            f"  K={k}: {dense_groups.total_candidates:,} candidates, {len(k3_chunks)} chunk(s), "
                            f"group data {group_data_bytes / (1 << 30):.1f} GB, "
                            f"budget {max_cands_per_chunk:,} cands/chunk"
                        )

                        # Upload group data to all GPUs ONCE — stays resident across chunks
                        all_groups_gpu = {}
                        for bv, did, _ in bitvecs_list:
                            all_groups_gpu[did] = upload_k3plus_groups(dense_groups, did)

                        def _k3plus_chunk_on_gpu(bitvec_gpu, device_id, chunk):
                            with cp.cuda.Device(device_id):
                                return count_k3plus_allcounts(
                                    bitvec_gpu,
                                    dense_groups,
                                    bitvec_gpu.shape[1],
                                    chunk_start=chunk.start,
                                    chunk_size=chunk.size,
                                    groups_gpu=all_groups_gpu[device_id],
                                    variant="legacy" if chunk.use_legacy else None,
                                )

                        # try/finally, because this is ~40 GB on a wide level and
                        # run_chunked_dense_level can raise. Without it the group
                        # arrays stayed resident on every device for the rest of
                        # the run.
                        try:
                            freq_cand_indices, freq_cand_counts = run_chunked_dense_level(
                                bitvecs_list,
                                k3_chunks,
                                _k3plus_chunk_on_gpu,
                                min_count_threshold,
                                nccl_comms,
                                _use_nccl,
                                level_label=f"K={k}",
                            )
                        finally:
                            # Free group data from all GPUs
                            for did in list(all_groups_gpu):
                                with cp.cuda.Device(did):
                                    del all_groups_gpu[did]
                                    cp.get_default_memory_pool().free_all_blocks()
                        parts.append((dense_groups, freq_cand_indices, freq_cand_counts))

                    parts = [part for part in parts if len(part[1]) > 0]
                    if parts:
                        current_flat = np.concatenate([decode_k3plus_flat(idx, g, k) for g, idx, _ in parts])
                        current_counts_raw = np.concatenate([cnt for _, _, cnt in parts])
                        n_freq = len(current_flat)


            # ── Level end: sort, split the two populations, emit ─────────
            # The lexsort runs BEFORE the free-set prune so that the COMPLETE
            # level is sorted too: the next level binary-searches it, and the
            # prune below is order-preserving, so one sort serves both. Neither
            # decode is lex-sorted by itself — K=2 enumerates pairs
            # triangularly and K>=3 emits each prefix group's pairs in j-major
            # order ((0,1),(0,2),(1,2),(0,3),...), which is not lex order once a
            # group has >= 4 suffixes. Skipped when already sorted
            # (_rows_sorted is O(n·k), no sort), and when nothing needs it.
            if n_freq > 0 and (k == 2 or prune_non_free) and not _rows_sorted(current_flat):
                sort_idx = np.lexsort(current_flat[:, ::-1].T)
                current_flat = current_flat[sort_idx]
                current_counts_raw = current_counts_raw[sort_idx]

            # The complete frequent level — what every subset test of K+1
            # resolves against. Same object as the emitted level when the flag
            # is off, so the unpruned path pays nothing for this.
            full_flat = current_flat
            full_counts = current_counts_raw

            # Free-set (generator) pruning: drop itemsets whose count equals a
            # (k-1)-subset's, tested against the COMPLETE previous level. What
            # survives is both what this level emits and what K+1 generates
            # from — those must be the same population, or the output
            # advertises itemsets the run will never extend.
            if prune_non_free and n_freq > 0:
                _keep = _prune_non_free_mask(current_flat, current_counts_raw, prev_full_flat, prev_full_counts)
                current_flat = current_flat[_keep]
                current_counts_raw = current_counts_raw[_keep]
                n_freq = len(current_flat)

            if n_freq > 0:
                # V3 B6: anchoring is an OUTPUT SELECTOR, applied here and
                # nowhere else. It used to filter `current_flat`, and that one
                # array then became BOTH downstream populations: full_flat (the
                # subset oracle every K+1 apriori test resolves against) and
                # prev_frequent_flat (the generation base). That is unsound in
                # two independent ways at once -- the oracle needs the
                # (k-1)-subsets that DROP the anchor, which are unanchored by
                # construction, and the prefix-join needs the family closed
                # under its two prefix-parents, which an anchored candidate's
                # parents need not be. Measured loss: 95-99% of the anchored
                # itemsets, in every configuration the public API can produce.
                #
                # This is the same defect class the free-set prune documents as
                # fixed above; the difference in outcome is one property.
                # Freeness is anti-monotone. Anchoredness is not.
                #
                # A pruning-preserving variant does not exist: hoist+remap was
                # implemented and measured to lose 129 of 321, because it can
                # only repair the level immediately below the first filtered
                # one. The candidate-space reduction and the apriori prune are
                # mutually exclusive. See mine_two_phase's docstring.
                _emit_flat, _emit_counts = current_flat, current_counts_raw
                _keep = _anchor_keep_mask(current_flat, anchor_col_arr, k)
                if _keep is not None:
                    _emit_flat = current_flat[_keep]
                    _emit_counts = current_counts_raw[_keep]
                    logger.debug(
                        f"    Anchor filter K={k}: emitting {len(_emit_flat):,} of "
                        f"{n_freq:,} (generation base left intact)"
                    )
                if len(_emit_flat) > 0:
                    items_flat = col_to_item_arr[_emit_flat]  # (n, k) int32
                    _flush_or_defer(items_flat, _emit_counts / n_transactions, k)

            k_time = time.perf_counter() - _k_start
            if session:
                session.end_phase(n_candidates=_n_cands_cb, n_frequent=n_freq)
            if level_callback:
                level_callback(k, _n_cands_cb, n_freq, k_time * 1000)
            logger.info(f"  K={k}: {n_freq:,} frequent in {k_time:.1f}s")

            if n_freq == 0:
                break
            _check_memory_guard(k)

            prev_frequent_flat = current_flat
            prev_counts_flat = current_counts_raw
            prev_full_flat = full_flat
            prev_full_counts = full_counts
            k += 1
    finally:
        # Emergency uploader shutdown (happy path does ordered drain below)
        try:
            uploader.close(wait=False)
        except Exception:
            pass

    # When output_dir is set, results are already flushed to Parquet per K level.
    # Return empty DataFrame — caller reads from output_dir instead.
    if output_dir:
        uploader.drain()
        uploader.close(wait=True)
        logger.info(f"  All results flushed to {output_dir}/frequent_k*.parquet")
        return _result(_empty_result())

    # Build DataFrame from numpy arrays — zero .tolist() overhead
    if not deferred_itemsets_np:
        return _result(_empty_result())

    all_supports = np.concatenate(deferred_supports)
    return _result(_build_deferred_frame(deferred_itemsets_np, all_supports))


def _build_deferred_frame(
    deferred_itemsets_np: list[np.ndarray],
    all_supports: np.ndarray,
) -> pl.DataFrame:
    """Build the result frame from the deferred per-level itemset arrays.

    Split out of `_apriori_row_split_multi_gpu` for one reason: the host-RAM
    peak of this block is what decides whether a campaign survives its last
    level, and inside that function it is unreachable without a GPU. Here it is
    a pure function of its two arguments.

    `all_supports` is built by the CALLER and passed in, so a peak measured
    across this call is this block's own allocation and nothing else's. The
    identity below, and the test that pins it, both depend on that split.
    """
    import numpy as np

    # PyArrow path: O(1) Python overhead via Arrow ListArray from numpy.
    #
    # The int64 cast is load-bearing, not cosmetic. FOUR places return a frame
    # on this route and they used to disagree on the itemset dtype. Two are in
    # THIS function -- this arrow path and the list fallback in its `except
    # ImportError`; the other two are the `_empty_result()` calls in the CALLER,
    # `_apriori_row_split_multi_gpu`, which were left there when this block was
    # split out. The count said "three return paths" of "this function" while
    # naming four sites across two, which is the defect this comment block keeps
    # being rewritten for: say which set, then count that set.
    #
    # Both `_empty_result()` calls give List(Int64) (core/result.py), the list
    # fallback gives List(Int64) via Python ints, and this path gave
    # List(Int32) -- because this file builds `col_to_item_arr` as np.int32
    # (guarded by the `_max_item < 2**31` assert beside it) and Arrow preserves
    # it. So the ONE path that normally runs was the odd one out.
    #
    # Deliberately asymmetric with the flushed parquet, which stays
    # large_list<int32> (io/flush.py builds its own list array from the same
    # int32 items_flat): widening it would double the itemset bytes of every
    # artifact already on disk, and nothing reads it in a dtype-sensitive way --
    # the resume reader indexes `item_to_col[flat_item_ids]`, the mine_two_phase
    # anchor read goes through `df["itemset"].to_list()`, and both core/rules.py
    # consumers are parquet-to-parquet so their join keys are int32 on both
    # sides. Widening those would cost ~84 GB on a K=7 K-1 frame -- an
    # order-of-magnitude estimate, not a measurement: the arithmetic is
    # rows x 6 items x 4 extra bytes, so it stands on a K=6 frame of ~3.5e9
    # rows, and that row count is recorded nowhere in this tree. The only
    # scale figure that is written down is core/rules.py's "K=8: 12B rows,
    # 67 GB". Quoted with its basis so the next reader can reject it.
    # tests/test_row_split_dtypes.py pins each of those boundaries.
    #
    # These references name symbols, not line numbers. Four of the five numbers
    # this block used to carry were wrong; corrected, two went stale again
    # inside the same session, because any edit above them moves them and
    # nothing checks. Scoped to this block deliberately: it is what was fixed
    # here, not a project-wide convention -- there is no check that would make
    # it one, and stating it as a rule while the rest of the tree keeps its
    # line numbers is the kind of claim this block exists to stop making.
    try:
        import pyarrow as pa

        # Both arrays are preallocated and filled chunk by chunk. This is the
        # `output_dir=None` route -- the one that keeps every level in host RAM
        # instead of flushing it -- so it is the route where N is largest, and
        # the peak is what decides whether a campaign survives its last level.
        #
        # `np.concatenate(...).astype(np.int64, copy=False)` read as two cheap
        # steps and was not: int32 -> int64 can never satisfy copy=False, so the
        # concatenated int32 buffer (4N) and the int64 result (8N) were both
        # live across the cast, on top of the int32 sources (4N) that stay alive
        # to the end of this block. #26 (17d8574) added that cast for dtype
        # consistency and doubled the peak, 8N -> 16N, without saying so. The
        # dtype fix stands; the undeclared cost is what is fixed here.
        #
        # `widths` had the same shape one line down -- a list of per-chunk
        # arrays plus the concatenated copy, both live -- and it is dropped
        # entirely: it existed only to be cumsum'd into `offsets`, so the widths
        # are written straight into `offsets[1:]` and summed in place. A scalar
        # broadcast into a slice needs no temporary at all.
        #
        # WHAT THE FIGURES BELOW ARE CONDITIONAL ON. `deferred_itemsets_np`
        # holds ONE ENTRY PER K LEVEL -- appended once per level (see the
        # `_flush_or_defer` calls for K=1 and for the level loop), so entry j
        # has shape (n_j, j): both the row count and k vary, and an apriori
        # lattice is strongly peaked with a tiny tail. The measurements below
        # use a FIXTURE -- equal chunks, uniform k -- which is a property of
        # the harness, not of this code. Five successive measurements in the
        # review of this comment each corrected the one before by varying a
        # dimension it had held fixed (shape, then distribution, then k); a
        # harness only varies what its author knows is a variable. So the
        # totals are stated as an identity, and anything distribution-
        # dependent is stated as a condition rather than a constant.
        #
        # IDENTITY, item arrays only. N = total items, R = |offsets| =
        # (rows+1)*8B, kbar = N/rows the mean itemset length -- so R = 8N/kbar
        # to within the one extra element, and N and R are one variable, not
        # two:
        #   floor     = 4N                 sources only; offsets does not exist yet
        #   old       = 12N + max(4N, 2R)  TWO candidate peaks, whichever is
        #                                  higher: the int32->int64 cast (4N
        #                                  sources + 4N concat + 8N result) or
        #                                  the `widths` pair (12N + 2R)
        #   flat-only = 12N + 2R           fixing the cast alone leaves `widths`
        #                                  -- the per-chunk list AND its
        #                                  concatenation, R each
        #   new       = 12N + R            4N sources + 8N flat + offsets
        #
        # CROSSOVER at kbar = 4, where 4N = 2R. ABOVE it the cast sets the old
        # peak and `old` collapses to 16N; BELOW it the `widths` pair already
        # set the peak, and 16N under-states it. So "fixing the cast relocates
        # the peak to `widths`" is true only above the crossover -- below it the
        # peak was never at the cast to be moved from.
        #
        # The transition is `old` -> 12N + R, NOT 16N -> 12N: `offsets` is
        # absent from the cast peak (which precedes `widths`/`offsets` existing
        # at all) and present once here, so it does not cancel under
        # differencing. Hence
        #
        #   saving = max(4N, 2R) - R = max(4N - R, R) >= R > 0
        #
        # which is strictly positive at every kbar: this change cannot regress
        # the peak, at any shape. Reading the saving as 4N - R holds only above
        # the crossover, and there is no single "the real lattice" to read it
        # against -- the in-tree presets fall on BOTH sides of kbar = 4:
        #
        #   preset        itemsets    items      kbar     4N vs 2R
        #   smoke              694     1,664   2.3977   2R > 4N  (below)
        #   skewed_rows     10,350    44,900   4.3382   4N > 2R  (above)
        #   deep_k           8,841    46,727   5.2853   4N > 2R  (above)
        #
        # Measured by mining each preset at its own min_support (smoke on the
        # CPU tier, the other two with use_gpu=True); the k-histograms are in
        # the session record and `smoke`'s is reproduced as `SMOKE_K_HIST` in
        # tests/test_row_split_memory.py, which builds a fixture from it.
        #
        # ABOVE the crossover the naive reading is EXACT: at deep_k, 4N - R and
        # max(4N, 2R) - R are both 116,172 B. BELOW it the naive reading
        # understates -- at smoke the true saving is R = 5,560 B against a naive
        # 4N - R = 1,096 B, so 5.07x low. Below kbar = 2 it is SIGN-INVERTED
        # (4N < R), predicting a regression where the true saving is R; that is
        # a property of the formula, asserted here about the formula and not
        # about any preset.
        #
        # The figure this replaced was "kbar = 2.365, measured", with a "5.5x
        # LOW" derived from it. Neither had an artifact anywhere in the tree,
        # and 5.5x is exactly what 2.365 yields -- so the consequence could not
        # corroborate the premise, it only restated it.
        #
        # ONE WORKED INSTANCE, above the crossover, so `old` = 16N in this
        # branch only. At the fixture shape N=200M items / k=5 / 8 equal chunks,
        # VmHWM: 4N = 0.745, R = 0.298, interpreter floor ~0.02 ->
        # 0.77 / 3.00 / 2.85 / 2.56 GiB, a measured saving of 0.447 GiB.
        # Reproduced independently at 0.767 / 3.003 / 2.853 / 2.555. Below the
        # crossover, same N: kbar = 3 measures old = 3.251 against the 16N form
        # 2.980, and kbar = 1 measures 5.238 -- the form that collapses to 16N
        # is the one that fails here, not the max().
        #
        # RE-MEASURED on a SECOND instrument, because everything above is VmHWM
        # and VmHWM stops being true -- not merely noisy -- below N ~= 10M,
        # where glibc does not return sub-mmap-threshold blocks. tracemalloc,
        # run against the pre-fix code itself (`0a21f35^`: the
        # `np.concatenate(...).astype(np.int64)` line and the `widths` pair
        # below it, not a paraphrase of them -- a paraphrase that keeps the
        # per-chunk list alive across `offsets` measures 3R and disagrees),
        # puts `old` at ratio 1.0000 of 12N + max(4N, 2R) at kbar = 1, 2, 3, 4,
        # 5 and 8, with the crossover landing on kbar = 4 exactly. The two
        # instruments agree once the 4N sources are counted on both sides:
        # kbar = 1 is 28N/16N = 1.75 there against 5.238/2.980 = 1.758 here.
        #
        # The in-place `np.cumsum(offsets[1:], out=offsets[1:])` below aliases
        # input and output, and that is a documented contract, not tolerated
        # behaviour: NumPy >= 1.13 defines an overlapping ufunc operation to
        # give the non-overlapping result, and `accumulate` participates --
        # `np.add.accumulate(x[:-1], out=x[1:])` returns the no-overlap answer,
        # which a naive in-place loop cannot.
        #
        # The PEAK, separately, rides on a narrower property than that. The
        # copy-on-overlap path is not a future risk; it exists and runs today,
        # and it materialises a FULL-SIZE temporary as soon as two operands
        # overlap without being identical. Measured on a 1.49 GiB int64 array,
        # identical on 1.26.4 and 2.2.6:
        #   np.cumsum(x, out=x)            0.000 GiB   exact alias
        #   np.cumsum(x[1:], out=x[1:])    0.000 GiB   <- the form used here
        #   np.cumsum(x[1:], out=x[:-1])   1.490 GiB   <- one element of shift
        #   np.cumsum(x[:n/2], out=y)      0.745 GiB   disjoint: output only
        # The line below evaluates `offsets[1:]` twice, producing two DISTINCT
        # view objects sharing base, offset, shape and strides. Both cost
        # nothing, so NumPy classifies on what the views describe rather than
        # on object identity -- and the saving depends on that classification
        # continuing to treat two identical views as identical rather than
        # merely overlapping. One decision wide, not one feature.
        #
        # Nothing pins that. The equivalence to the old concatenate/cumsum
        # form was verified ad hoc over randomised chunk shapes and holds, but
        # that form is no longer in the tree for a test to compare against, so
        # a future defensive copy would keep the RESULT right and revert the
        # peak with nothing red. Same status as the note at the top of this
        # block: verified, not pinned.
        total_rows = sum(a.shape[0] for a in deferred_itemsets_np)
        total_items = sum(a.size for a in deferred_itemsets_np)

        flat_values = np.empty(total_items, dtype=np.int64)
        offsets = np.empty(total_rows + 1, dtype=np.int64)
        offsets[0] = 0
        _item_pos = 0
        _row_pos = 1
        for _a in deferred_itemsets_np:
            _nr, _k = _a.shape
            flat_values[_item_pos : _item_pos + _a.size] = _a.ravel()  # int32 -> int64, one chunk
            _item_pos += _a.size
            offsets[_row_pos : _row_pos + _nr] = _k  # scalar broadcast, no temporary
            _row_pos += _nr
        np.cumsum(offsets[1:], out=offsets[1:])
        arrow_list = pa.LargeListArray.from_arrays(offsets, flat_values)
        return pl.DataFrame(
            {
                "itemset": pl.Series("itemset", arrow_list),
                "support": all_supports,
            }
        )
    except ImportError:
        # Fallback: single-pass list construction.
        # Narrowed from `except (ImportError, Exception)`, which collapses to
        # Exception and swallowed every PyArrow failure -- silently taking a
        # path with a different dtype and different memory behaviour. A missing
        # pyarrow is a fallback; a broken one is a bug and must surface.
        all_itemsets = []
        for arr in deferred_itemsets_np:
            for i in range(arr.shape[0]):
                all_itemsets.append(arr[i].tolist())
        return pl.DataFrame(
            {
                "itemset": all_itemsets,
                "support": all_supports.tolist(),
            }
        )


def mine_two_phase(
    transactions,
    phase1_support: float = 0.001,
    phase2_support: float = 0.0005,
    max_length: int | None = None,
    item_col: str = "items",
    n_gpus: int = 1,
    output_dir: str | None = None,
    sparse_from_k: int | str | None = None,
    level_callback=None,
) -> tuple:
    """Two-phase mining: anchor discovery, then an anchor-restricted REPORT.

    Phase 1 mines at phase1_support to find anchor items — the items that
    participate in frequent patterns at reasonable support thresholds.

    Phase 2 mines at phase2_support (lower) and **reports only the itemsets
    containing at least one anchor**.

    .. warning::
       **Phase 2 mines the full lattice at phase2_support and post-filters. It
       does not reduce the candidate space, and it cannot.**

       An earlier version applied the anchor mask to the mining state, which
       did shrink the next level's generation base — and was unsound, losing
       95-99% of the anchored itemsets in every configuration this function can
       produce. Two independent reasons: the apriori oracle must test the
       (k-1)-subsets that DROP the anchor, and those are unanchored by
       construction; and the prefix-join needs the surviving family closed
       under its two prefix-parents, which an anchored candidate's parents need
       not be. Anchoredness is not anti-monotone, which is exactly why the same
       code shape is sound for the free-set prune and catastrophic here.

       A pruning-preserving variant was sought and does not exist. hoist+remap
       was implemented and measured to lose 129 of 321: it repairs only the
       level immediately below the first filtered one, because level k-1 was
       itself generated from a restricted base. No column ordering repairs it —
       ordering changes which subsets go missing, never whether they do.

       So budget Phase 2 as a full run at phase2_support. The default was
       0.00001, chosen when the filter was believed to cut the search space;
       at that threshold a post-filtering Phase 2 is likely intractable, so it
       is now 0.0005. Set it lower deliberately, having sized the full lattice.

       If genuine candidate reduction is needed, the one sound shape is
       per-anchor conditional databases: for each anchor `a`, mine the
       projection onto transactions containing `a` — downward-closed within
       itself — then union and dedupe. One run per anchor is the cost.

    Args:
        transactions: Transaction data (DataFrame or LazyFrame).
        phase1_support: Support threshold for anchor discovery (higher).
        phase2_support: Support threshold for phase 2 (lower). Phase 2 mines the
            FULL lattice at this threshold and then reports the anchored subset,
            so this is the cost driver — see the warning above.
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name containing item lists.
        n_gpus: Number of GPUs to use.
        output_dir: Directory for per-K Parquet output. Phase 1 writes to
            output_dir/phase1/, Phase 2 to output_dir/phase2/.
        sparse_from_k: Removed; any value other than None raises ValueError.
        level_callback: Optional callback(k, n_candidates, n_frequent, ms).

    Returns:
        Tuple of (phase1_result, phase2_result, anchor_items) where
        anchor_items is the set of item IDs found in Phase 1.
    """
    import shutil
    import tempfile
    from pathlib import Path

    _cleanup_phase1 = False  # track if we need to clean up temp dirs
    # Phase 1: Anchor mining
    if output_dir:
        phase1_dir = os.path.join(output_dir, "phase1")
    else:
        phase1_dir = tempfile.mkdtemp(prefix="etminer_p1_")
        _cleanup_phase1 = True
    os.makedirs(phase1_dir, exist_ok=True)

    logger.info(
        f"TWO-PHASE MINING: Phase 1 {phase1_support:.6%} support (anchor discovery), Phase 2 {phase2_support:.6%} support (neighborhood zoom)"
    )

    from et_miner.core.apriori import apriori

    phase1_result = apriori(
        transactions,
        min_support=phase1_support,
        max_length=max_length,
        item_col=item_col,
        use_gpu=True,
        n_gpus=n_gpus,
        output_dir=phase1_dir,
        sparse_from_k=sparse_from_k,
        prune_equal_support=True,
        level_callback=level_callback,
    )

    # Extract anchor items from ALL K-levels
    anchor_items = set()
    for k_file in sorted(Path(phase1_dir).glob("frequent_k*.parquet")):
        df = pl.scan_parquet(k_file).collect(engine="streaming")
        if "items" in df.columns:
            for item_list in df["items"].to_list():
                anchor_items.update(int(x) for x in item_list)
        elif "itemset" in df.columns:
            for item_list in df["itemset"].to_list():
                anchor_items.update(int(x) for x in item_list)

    logger.info(f"  Phase 1 complete: {len(anchor_items)} anchor items")

    # Clean up Phase 1 temp dir (anchors extracted, parquets no longer needed)
    if _cleanup_phase1:
        shutil.rmtree(phase1_dir, ignore_errors=True)

    if not anchor_items:
        logger.warning("  Phase 1 found 0 anchor items — Phase 2 will run unfiltered")

    # Phase 2: Neighborhood zoom with anchor filtering
    _cleanup_phase2 = output_dir is None
    phase2_dir = os.path.join(output_dir, "phase2") if output_dir else tempfile.mkdtemp(prefix="etminer_p2_")
    os.makedirs(phase2_dir, exist_ok=True)

    phase2_result = apriori(
        transactions,
        min_support=phase2_support,
        max_length=max_length,
        item_col=item_col,
        use_gpu=True,
        n_gpus=n_gpus,
        output_dir=phase2_dir,
        sparse_from_k=sparse_from_k,
        prune_equal_support=True,
        anchor_items=anchor_items if anchor_items else None,
        level_callback=level_callback,
    )

    logger.info("  Phase 2 complete")

    if _cleanup_phase2:
        shutil.rmtree(phase2_dir, ignore_errors=True)

    logger.info("TWO-PHASE MINING DONE")
    return phase1_result, phase2_result, anchor_items


