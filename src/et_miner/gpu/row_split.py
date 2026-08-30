"""Multi-GPU row-split mining and two-phase (anchor + zoom) mining.

Splits transactions across GPUs by row range, mines each shard with the
bitvec kernels, and combines counts via NCCL all-reduce. mine_two_phase
drives two apriori() passes (anchor discovery, then neighborhood zoom).
Import-safe without CuPy; GPU work happens only at call time.
"""

from __future__ import annotations

import os
import time

import polars as pl
from loguru import logger

from et_miner import _env
from et_miner.core.result import (
    _build_result_df,
    _empty_result,
    _min_count,
)
from et_miner.gpu.density import DENSITY_CROSSOVER, SPARSE_AUTO, should_transition_to_sparse
from et_miner.gpu.mining import (
    _SparseState,
    _apply_anchor_filter,
    _convert_to_tidsets,
    _prune_closed_flat,
    _prune_groups_apriori,
)
from et_miner.gpu.nccl import _init_nccl, _nccl_allreduce_sum
from et_miner.io.flush import _flush_k_parquet
from et_miner.io.gcs import (
    GCSUploader,
    is_gs_uri,
    is_upload_enabled,
    polars_storage_options,
)


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
    prune_closed: bool = False,  # V3: prune non-closed itemsets between K-levels
    prune_apriori: bool = False,  # V3: Apriori subset pruning on candidate groups
    sparse_from_k: int | str | None = None,  # V3: CSR from this K level, or "auto" = measured density
    anchor_items: set | None = None,  # V3 B6: two-phase anchor filtering
) -> "pl.DataFrame":
    """Mine frequent itemsets using row-split bitvecs across multiple GPUs.

    Splits the bitvec by transaction rows across GPUs. Each GPU holds ~1/n_gpus
    of the data.

    GPU-resident dense counting architecture (no recount, no D2H for arrays):
      1. Each GPU runs fused kernel on ALL candidates → dense count array in VRAM
      2. Device-to-device sum on GPU 0 (NVLink if available) → global counts
      3. Filter + where on GPU 0 → only freq_indices cross PCIe to CPU

    Since all GPUs share the same prev_frequent, they generate the same
    candidates in the same deterministic order. The dense output at index i
    from GPU 0 is the partial count for the same candidate as index i from
    GPU 1. Element-wise sum = exact global counts.

    PCIe transfer: only len(freq_indices) × 16 bytes (index + count).
    For K=2 with 35K features: ~600K pairs × 16 = 9.6 MB instead of 19.5 GB.
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
        count_csr_intersections,
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

    # Local state for sparse CSR mode (replaces old function-attribute mutation)
    sparse_state = _SparseState()

    logger.info(
        f"  Row-split multi-GPU: {n_gpus} GPUs, min_count={min_count_threshold:,} (GPU-resident dense counting)"
    )

    # Phase 0: Build row-split bitvecs across GPUs
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

    # Initialize NCCL for multi-GPU all-reduce (ring topology)
    device_ids = [did for _, did, _ in bitvecs_list]
    nccl_comms, _use_nccl = _init_nccl(device_ids)
    if _use_nccl:
        logger.info(f"  NCCL: {len(device_ids)} communicators (ring all-reduce)")
    else:
        logger.info("  NCCL unavailable, using sequential D2D")

    # Per-K Parquet flush: write each K level to disk immediately.
    # Prevents 680 GB CPU RAM accumulation at K=7+ scale.
    _output_is_remote = bool(output_dir) and is_gs_uri(output_dir)
    if output_dir and not _output_is_remote:
        os.makedirs(output_dir, exist_ok=True)

    # Deferred results: accumulate numpy arrays per level,
    # build Polars DataFrame at the end via PyArrow. No .tolist() overhead.
    # When output_dir is set, arrays flush to Parquet per K and are NOT accumulated.
    deferred_itemsets_np: list[np.ndarray] = []  # (n, k) int64 arrays
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

        # V3: reconstruct raw counts from the flushed support column, so the
        # first resumed level keeps closed pruning and the "auto" density
        # transition. count → support → count round-trips exactly through
        # float64 for any int32-range count.
        if "support" in table.column_names:
            supports_np = table.column("support").combine_chunks().to_numpy(zero_copy_only=False)
            prev_counts_flat = np.rint(supports_np.astype(np.float64) * n_transactions).astype(np.int64)
        else:
            prev_counts_flat = None  # counts unknown — closed pruning and auto transition wait one level
        del table, itemsets_col, flat_item_ids, flat_col_ids, item_to_col

        resume_time = time.perf_counter() - t_resume
        logger.info(f"  RESUME: {n_loaded:,} itemsets → col indices in {resume_time:.1f}s")

        k = resume_from_k + 1
        prev_live_mgpu = set(prev_frequent_flat.ravel().tolist())
        logger.info(f"  RESUME: Jumping to K={k} ({len(prev_live_mgpu)} live columns)")

    # ── K=1: parallel popcount across GPUs, sum ────────────────────────
    if not _resume_active:
        _k1_start = time.perf_counter()
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

        if len(freq_col_indices) > 0:
            freq_items_k1 = col_to_item_arr[freq_col_indices]
            k1_supports = freq_col_counts / n_transactions
            _flush_or_defer(freq_items_k1.reshape(-1, 1), k1_supports, 1)

        k1_time = time.perf_counter() - _k1_start
        if level_callback:
            level_callback(1, n_cols, len(freq_col_indices), k1_time * 1000)
        logger.info(f"  K=1: {len(freq_col_indices):,} frequent items in {k1_time:.1f}s")

        if len(freq_col_indices) == 0:
            return _build_result_df([])

        # K=1 frequent columns as flat (n, 1) array — already numpy
        prev_frequent_flat = freq_col_indices.astype(np.int32).reshape(-1, 1)
        prev_counts_flat = freq_col_counts.astype(np.int64)  # V3: preserve for closed pruning
        k = 2
        prev_live_mgpu = set(freq_col_indices.tolist())

    # ── K>=2: GPU-resident dense counting ────────────────────────────────
    # State: prev_frequent_flat — numpy (n_freq, k-1) array of column indices.
    # No Python tuples in the hot path. Results decoded at end of each level
    # via vectorized numpy. Next-level groups built from flat arrays directly.
    try:
        while k <= effective_max_length and prev_frequent_flat.shape[0] >= k:
            _k_start = time.perf_counter()

            # V3: Sparse CSR mode — fixed K-level or measured density ("auto").
            # Sticky once entered: the transition frees the bitvecs, so later
            # levels must never fall back to the dense path.
            _mean_count = None
            if (
                sparse_from_k == SPARSE_AUTO
                and not sparse_state.active
                and prev_counts_flat is not None
                and len(prev_counts_flat) > 0
            ):
                _mean_count = float(prev_counts_flat.mean())
            _sparse_mode = sparse_state.active or should_transition_to_sparse(
                sparse_from_k, k, n_transactions=n_transactions, mean_count=_mean_count
            )

            if _sparse_mode:
                # ═══ V3 SPARSE CSR PATH ═══
                # Density transition: bitvecs → CSR tid-sets at the K boundary.
                # First time entering sparse mode: convert bitvecs → tidsets, free VRAM.
                if not sparse_state.active:
                    _trigger = (
                        f"measured mean support {_mean_count / n_transactions:.4%} "
                        f"< {DENSITY_CROSSOVER:.4%} crossover"
                        if sparse_from_k == SPARSE_AUTO
                        else f"fixed sparse_from_k={sparse_from_k}"
                    )
                    logger.info(f"  ═══ DENSITY TRANSITION at K={k} ({_trigger}): dense bitvec → sparse CSR ═══")
                    bv0, did0, _ = bitvecs_list[0]
                    with cp.cuda.Device(did0):
                        n_u64s_local = bv0.shape[1]
                    # Adaptive batch_size: fit in VRAM headroom (H100=81GB, H200=141GB)
                    with cp.cuda.Device(did0):
                        free_mem = cp.cuda.Device(did0).mem_info[0]
                    bytes_per_row = n_u64s_local * 8  # uint64 words → bytes
                    # 2× bytes_per_row: one for and_results + one for bv[col_indices] gather
                    max_batch = max(100, int(free_mem * 0.5 / (2 * bytes_per_row)))
                    tidset_offsets, tidset_indices = _convert_to_tidsets(
                        bitvecs_list,
                        prev_frequent_flat,
                        n_u64s_local,
                        batch_size=min(max_batch, 10_000),
                        verify=True,
                    )
                    # Store device IDs for multi-GPU CSR (bitvecs_list about to be cleared)
                    sparse_state.device_ids = [did for _, did, _ in bitvecs_list]

                    # Free ALL bitvec VRAM across all GPUs
                    for bv, did, _ in bitvecs_list:
                        with cp.cuda.Device(did):
                            del bv
                            cp.get_default_memory_pool().free_all_blocks()
                    bitvecs_list.clear()
                    logger.debug("    Freed bitvec VRAM across all GPUs")
                    # Store state for subsequent K-levels
                    sparse_state.active = True

                # Build groups from prev_frequent
                groups_info = build_k3plus_groups_from_flat(prev_frequent_flat)

                # Apriori pruning (reuse A3) — Rust fast path with Python fallback
                if prune_apriori and groups_info is not None:
                    tc_before = groups_info.total_candidates
                    prev_freq_set = set(map(tuple, prev_frequent_flat.tolist()))
                    groups_info = _prune_groups_apriori(groups_info, prev_freq_set, k, prev_flat_np=prev_frequent_flat)
                    tc_after = groups_info.total_candidates if groups_info is not None else 0
                    if tc_before > tc_after:
                        logger.debug(
                            f"    Apriori pruning K={k}: {tc_before:,} → {tc_after:,} ({100 * (1 - tc_after / tc_before):.1f}% pruned)"
                        )

                n_freq = 0
                current_flat = np.empty((0, k), dtype=np.int32)
                current_counts_raw = np.empty(0, dtype=np.int64)

                if groups_info is not None and groups_info.total_candidates > 0:
                    tc = groups_info.total_candidates
                    logger.info(f"  K={k}: {tc:,} candidates (CSR sparse mode)")

                    # Build candidate pair arrays: for each group, enumerate suffix pairs
                    # Each pair (i, j) maps to indices in prev_frequent_flat
                    pair_a_list, pair_b_list = [], []
                    so = groups_info.suffix_offsets
                    cp_arr = groups_info.cumulative_pairs

                    for g in range(len(so) - 1):
                        sstart, send = int(so[g]), int(so[g + 1])
                        gsuf = groups_info.suffixes[sstart:send]
                        prefix = tuple(
                            int(x)
                            for x in groups_info.prefix_items[
                                int(groups_info.prefix_offsets[g]) : int(groups_info.prefix_offsets[g + 1])
                            ]
                        )

                        # Map suffix → index in prev_frequent_flat
                        # Build lookup: tuple(itemset) → index
                        if sparse_state.prev_idx_lookup is None:
                            sparse_state.prev_idx_lookup = {
                                tuple(int(x) for x in prev_frequent_flat[i]): i for i in range(len(prev_frequent_flat))
                            }
                        idx_lookup = sparse_state.prev_idx_lookup

                        for si_pos in range(len(gsuf)):
                            for sj_pos in range(si_pos + 1, len(gsuf)):
                                si, sj = int(gsuf[si_pos]), int(gsuf[sj_pos])
                                key_i = prefix + (si,)
                                key_j = prefix + (sj,)
                                idx_i = idx_lookup.get(key_i)
                                idx_j = idx_lookup.get(key_j)
                                if idx_i is not None and idx_j is not None:
                                    pair_a_list.append(idx_i)
                                    pair_b_list.append(idx_j)

                    if pair_a_list:
                        n_pairs = len(pair_a_list)
                        pair_a_np = np.array(pair_a_list, dtype=np.int64)
                        pair_b_np = np.array(pair_b_list, dtype=np.int64)

                        device_ids_csr = sparse_state.device_ids
                        n_gpus_avail = len(device_ids_csr)

                        if n_gpus_avail <= 1 or n_pairs < 1000:
                            # Single GPU: direct (avoid ThreadPool overhead for small workloads)
                            with cp.cuda.Device(device_ids_csr[0] if device_ids_csr else 0):
                                offsets_gpu = cp.array(tidset_offsets, dtype=cp.int64)
                                indices_gpu = cp.array(tidset_indices, dtype=cp.int32)
                                pa_gpu = cp.array(pair_a_np, dtype=cp.int64)
                                pb_gpu = cp.array(pair_b_np, dtype=cp.int64)
                                counts_gpu = count_csr_intersections(offsets_gpu, indices_gpu, pa_gpu, pb_gpu, n_pairs)
                                counts_cpu = counts_gpu.get()
                                del offsets_gpu, indices_gpu, pa_gpu, pb_gpu, counts_gpu
                                cp.get_default_memory_pool().free_all_blocks()
                        else:
                            # Multi-GPU: partition pairs across GPUs
                            pairs_per_gpu = (n_pairs + n_gpus_avail - 1) // n_gpus_avail

                            def _csr_on_gpu(device_id, p_start, p_end):
                                with cp.cuda.Device(device_id):
                                    off_gpu = cp.array(tidset_offsets, dtype=cp.int64)
                                    idx_gpu = cp.array(tidset_indices, dtype=cp.int32)
                                    pa_gpu = cp.array(pair_a_np[p_start:p_end], dtype=cp.int64)
                                    pb_gpu = cp.array(pair_b_np[p_start:p_end], dtype=cp.int64)
                                    result = count_csr_intersections(off_gpu, idx_gpu, pa_gpu, pb_gpu, p_end - p_start)
                                    result_cpu = result.get()
                                    del off_gpu, idx_gpu, pa_gpu, pb_gpu, result
                                    cp.get_default_memory_pool().free_all_blocks()
                                    return result_cpu

                            with ThreadPoolExecutor(max_workers=n_gpus_avail) as pool:
                                futures = []
                                for i, did in enumerate(device_ids_csr):
                                    ps = i * pairs_per_gpu
                                    pe = min(ps + pairs_per_gpu, n_pairs)
                                    if ps < pe:
                                        futures.append(pool.submit(_csr_on_gpu, did, ps, pe))
                                counts_cpu = np.concatenate([f.result() for f in futures])

                            logger.debug(f"    CSR intersect: {n_pairs:,} pairs across {n_gpus_avail} GPUs")

                        # Filter frequent candidates
                        freq_mask = counts_cpu >= min_count_threshold
                        freq_pair_indices = np.where(freq_mask)[0]
                        freq_cand_counts = counts_cpu[freq_mask]
                        n_freq = len(freq_pair_indices)

                        if n_freq > 0:
                            # Decode candidate pairs → flat itemset array
                            decoded = []
                            for pi in freq_pair_indices:
                                a_idx, b_idx = pair_a_list[pi], pair_b_list[pi]
                                itemset_a = tuple(int(x) for x in prev_frequent_flat[a_idx])
                                itemset_b = tuple(int(x) for x in prev_frequent_flat[b_idx])
                                # Join: shared prefix + two suffixes
                                prefix = itemset_a[:-1]
                                new_itemset = prefix + (itemset_a[-1], itemset_b[-1])
                                decoded.append(new_itemset)

                            current_flat = np.array(decoded, dtype=np.int32)
                            current_counts_raw = freq_cand_counts.astype(np.int64)

                            # V3 B6: Anchor filter (sparse CSR path)
                            if anchor_col_arr is not None:
                                _n_before = len(current_flat)
                                current_flat, current_counts_raw, n_freq = _apply_anchor_filter(
                                    current_flat, current_counts_raw, anchor_col_arr, k
                                )
                                if n_freq < _n_before:
                                    # Rebuild freq_pair_indices to match filtered current_flat
                                    anchor_mask_csr = np.zeros(_n_before, dtype=bool)
                                    _tmp_flat = np.array(decoded, dtype=np.int32)
                                    for _col in range(_tmp_flat.shape[1]):
                                        anchor_mask_csr |= np.isin(_tmp_flat[:, _col], anchor_col_arr)
                                    freq_pair_indices = freq_pair_indices[anchor_mask_csr]
                                    freq_cand_counts = freq_cand_counts[anchor_mask_csr]
                                    del _tmp_flat, anchor_mask_csr

                            items_flat = col_to_item_arr[current_flat]
                            _flush_or_defer(items_flat, current_counts_raw / n_transactions, k)

                            # Skip tidset building at max_length — no K+1 iteration needed
                            if k >= effective_max_length:
                                break

                            # Build new tid-sets for K+1: intersect parent tid-sets on CPU
                            # Pre-allocate numpy buffer using known intersection sizes
                            total_tids_new = int(np.sum(freq_cand_counts))
                            new_offsets = np.empty(n_freq + 1, dtype=np.int64)
                            new_offsets[0] = 0
                            new_indices = np.empty(total_tids_new, dtype=np.int32)
                            write_pos = 0
                            for out_i, pi in enumerate(freq_pair_indices):
                                a_idx, b_idx = pair_a_list[pi], pair_b_list[pi]
                                a_start = int(tidset_offsets[a_idx])
                                a_end = int(tidset_offsets[a_idx + 1])
                                b_start = int(tidset_offsets[b_idx])
                                b_end = int(tidset_offsets[b_idx + 1])
                                tids_a = tidset_indices[a_start:a_end]
                                tids_b = tidset_indices[b_start:b_end]
                                isect = np.intersect1d(tids_a, tids_b, assume_unique=True)
                                n_isect = len(isect)
                                assert write_pos + n_isect <= total_tids_new, (
                                    f"CSR buffer overrun at itemset {out_i}: {write_pos} + {n_isect} > {total_tids_new}"
                                )
                                new_indices[write_pos : write_pos + n_isect] = isect
                                write_pos += n_isect
                                new_offsets[out_i + 1] = write_pos

                            assert write_pos <= total_tids_new, f"CSR buffer overrun: {write_pos} > {total_tids_new}"
                            tidset_offsets = new_offsets
                            tidset_indices = new_indices[:write_pos]

                            total_tids = len(tidset_indices)
                            avg_sup = total_tids / n_freq if n_freq > 0 else 0
                            tidset_mb = (len(tidset_offsets) * 8 + len(tidset_indices) * 4) / 1024**2
                            logger.debug(
                                f"    New tidsets: {n_freq:,} itemsets, avg support {avg_sup:.0f}, {tidset_mb:.1f} MB"
                            )

                # Clean up per-K state (NOT device_ids — persists across K-levels)
                sparse_state.prev_idx_lookup = None

            elif k == 2:
                freq_cols = sorted(prev_frequent_flat[:, 0])
                n_pairs = len(freq_cols) * (len(freq_cols) - 1) // 2
                _mem_gb = n_pairs * 4 / (1 << 30)  # int32 dense counts
                logger.info(f"  K=2: {n_pairs:,} total pairs, dense output {_mem_gb:.2f} GB/GPU")

                def _k2_dense_on_gpu(bitvec_gpu, device_id):
                    with cp.cuda.Device(device_id):
                        n_u64s = bitvec_gpu.shape[1]
                        return count_pairs_k2_allcounts(
                            bitvec_gpu,
                            freq_cols,
                            n_u64s,
                        )

                with ThreadPoolExecutor(max_workers=len(bitvecs_list)) as pool:
                    futures = [pool.submit(_k2_dense_on_gpu, bv, did) for bv, did, _ in bitvecs_list]
                    gpu_results = [f.result() for f in futures]

                # Multi-GPU reduction: NCCL ring all-reduce or sequential D2D
                if _use_nccl:
                    _nccl_allreduce_sum(gpu_results, nccl_comms, device_ids)

                with cp.cuda.Device(bitvecs_list[0][1]):
                    global_counts = gpu_results[0]
                    if not _use_nccl:
                        for i in range(1, len(gpu_results)):
                            cp.add(global_counts, cp.asarray(gpu_results[i]), out=global_counts)
                    for i in range(1, len(gpu_results)):
                        gpu_results[i] = None
                    del gpu_results
                    # Free memory pool on ALL GPUs, not just GPU 0 (P2: VRAM zombie fix)
                    for _, did, _ in bitvecs_list:
                        with cp.cuda.Device(did):
                            cp.get_default_memory_pool().free_all_blocks()

                    from et_miner.gpu.kernels.filter import compact_threshold_filter

                    freq_pair_indices, freq_pair_counts = compact_threshold_filter(global_counts, min_count_threshold)
                    del global_counts
                    cp.get_default_memory_pool().free_all_blocks()

                n_freq = len(freq_pair_indices)
                if n_freq > 0:
                    current_flat = decode_k2_pairs_flat(freq_pair_indices, freq_cols)
                    current_counts_raw = freq_pair_counts.astype(np.int64)  # V3: preserve

                    # V3 B6: Anchor filter (K=2 dense path)
                    current_flat, current_counts_raw, n_freq = _apply_anchor_filter(
                        current_flat, current_counts_raw, anchor_col_arr, k
                    )

                    items_flat = col_to_item_arr[current_flat]  # (n, 2) int32
                    _flush_or_defer(items_flat, current_counts_raw / n_transactions, k)

                    # Pair cache disabled — not yet wired to K>=3 kernels (saves ~13.6 GB VRAM)
                else:
                    current_flat = np.empty((0, 2), dtype=np.int32)
                    current_counts_raw = np.empty(0, dtype=np.int64)

            else:
                # K>=3: build groups from flat array — no Python tuple grouping
                groups_info = build_k3plus_groups_from_flat(prev_frequent_flat)

                # V3: Apriori subset pruning — Rust fast path with Python fallback
                if prune_apriori and groups_info is not None:
                    tc_before = groups_info.total_candidates
                    prev_freq_set = set(map(tuple, prev_frequent_flat.tolist()))
                    groups_info = _prune_groups_apriori(groups_info, prev_freq_set, k, prev_flat_np=prev_frequent_flat)
                    tc_after = groups_info.total_candidates if groups_info is not None else 0
                    if tc_before > tc_after:
                        logger.debug(
                            f"    Apriori pruning K={k}: {tc_before:,} → {tc_after:,} candidates ({100 * (1 - tc_after / tc_before):.1f}% pruned)"
                        )

                n_freq = 0
                current_flat = np.empty((0, k), dtype=np.int32)
                current_counts_raw = np.empty(0, dtype=np.int64)

                if groups_info is not None:
                    tc = groups_info.total_candidates

                    # VRAM budget for candidate-range chunking.
                    # Group data stays resident across all chunks; only the dense
                    # output array (chunk_size × 8 bytes, int64) varies per chunk.
                    group_data_bytes = (
                        len(groups_info.prefix_items) * 4
                        + len(groups_info.prefix_offsets) * 8
                        + len(groups_info.suffixes) * 4
                        + len(groups_info.suffix_offsets) * 8
                        + len(groups_info.cumulative_pairs) * 8
                    )

                    # Estimate free VRAM from first GPU (all GPUs have same bitvecs)
                    with cp.cuda.Device(bitvecs_list[0][1]):
                        cp.get_default_memory_pool().free_all_blocks()  # flush cached blocks for accurate reading
                        free_vram, _ = cp.cuda.Device().mem_info
                    dense_budget = free_vram - group_data_bytes - 6 * (1 << 30)  # 6 GB safety
                    # Assumes NCCL in-place reduce; non-NCCL fallback may need // 16
                    max_cands_per_chunk = max(1, int(dense_budget // 10))  # 8B result + 2B margin for reduction temps
                    n_chunks = max(1, (tc + max_cands_per_chunk - 1) // max_cands_per_chunk)

                    logger.debug(
                        f"  K={k}: {tc:,} candidates, {n_chunks} chunk(s), group data {group_data_bytes / (1 << 30):.1f} GB, dense budget {dense_budget / (1 << 30):.1f} GB"
                    )

                    # Upload group data to all GPUs ONCE — stays resident across chunks
                    all_groups_gpu = {}
                    for bv, did, _ in bitvecs_list:
                        all_groups_gpu[did] = upload_k3plus_groups(groups_info, did)

                    from et_miner.gpu.kernels.filter import compact_threshold_filter

                    all_freq_indices = []
                    all_freq_counts = []

                    _chunk_pool = ThreadPoolExecutor(max_workers=len(bitvecs_list))

                    for chunk_idx in range(n_chunks):
                        chunk_start = chunk_idx * max_cands_per_chunk
                        chunk_size = min(max_cands_per_chunk, tc - chunk_start)

                        if n_chunks > 1:
                            logger.debug(
                                f"    chunk {chunk_idx + 1}/{n_chunks}: candidates [{chunk_start:,}, {chunk_start + chunk_size:,})"
                            )

                        # Per-GPU counting with pre-uploaded groups
                        def _k3plus_chunk_on_gpu(bitvec_gpu, device_id, _cs=chunk_start, _csz=chunk_size):
                            with cp.cuda.Device(device_id):
                                return count_k3plus_allcounts(
                                    bitvec_gpu,
                                    groups_info,
                                    bitvec_gpu.shape[1],
                                    chunk_start=_cs,
                                    chunk_size=_csz,
                                    groups_gpu=all_groups_gpu[device_id],
                                )

                        futures = [_chunk_pool.submit(_k3plus_chunk_on_gpu, bv, did) for bv, did, _ in bitvecs_list]
                        gpu_results = [f.result() for f in futures]

                        # Multi-GPU reduction: NCCL ring all-reduce (chunk-sized arrays)
                        if _use_nccl:
                            _nccl_allreduce_sum(gpu_results, nccl_comms, device_ids)

                        # Reduce across GPUs, then threshold on GPU (sparse transfer)
                        with cp.cuda.Device(bitvecs_list[0][1]):
                            global_counts = gpu_results[0]
                            if not _use_nccl:
                                for i in range(1, len(gpu_results)):
                                    cp.add(global_counts, cp.asarray(gpu_results[i]), out=global_counts)

                            # Survivor compaction on GPU 0 — only survivors cross
                            # PCIe (see gpu.kernels.filter for the impl choices)
                            freq_indices_chunk, freq_counts = compact_threshold_filter(global_counts, min_count_threshold)
                            n_freq_chunk = len(freq_indices_chunk)
                            pass_rate = 100 * n_freq_chunk / chunk_size if chunk_size > 0 else 0
                            logger.info(
                                f"  Chunk {chunk_idx + 1}/{n_chunks} filtering: "
                                f"{chunk_size:,} candidates → {n_freq_chunk:,} frequent "
                                f"({pass_rate:.1f}% pass rate, min_count={min_count_threshold:,})"
                            )

                            if n_freq_chunk > 0:
                                all_freq_indices.append(freq_indices_chunk + chunk_start)
                                all_freq_counts.append(freq_counts)

                            # Free all GPU memory from this chunk
                            del global_counts
                            for i in range(len(gpu_results)):
                                gpu_results[i] = None
                            del gpu_results
                            # Free ALL GPUs, not just primary — prevents pool fragmentation
                            for _, did, _ in bitvecs_list:
                                with cp.cuda.Device(did):
                                    cp.get_default_memory_pool().free_all_blocks()

                    _chunk_pool.shutdown(wait=False)

                    # Free group data from all GPUs
                    for did in list(all_groups_gpu):
                        with cp.cuda.Device(did):
                            del all_groups_gpu[did]
                            cp.get_default_memory_pool().free_all_blocks()
                    del all_groups_gpu

                    # Combine chunk results
                    if all_freq_indices:
                        freq_cand_indices = np.concatenate(all_freq_indices)
                        freq_cand_counts = np.concatenate(all_freq_counts)
                        n_freq = len(freq_cand_indices)
                        current_flat = decode_k3plus_flat(
                            freq_cand_indices,
                            groups_info,
                            k,
                        )
                        current_counts_raw = freq_cand_counts.astype(np.int64)  # V3: preserve

                        # V3 B6: Anchor filter (K>=3 dense path)
                        current_flat, current_counts_raw, n_freq = _apply_anchor_filter(
                            current_flat, current_counts_raw, anchor_col_arr, k
                        )

                        items_flat = col_to_item_arr[current_flat]  # (n, k) int32
                        _flush_or_defer(
                            items_flat,
                            current_counts_raw / n_transactions,
                            k,
                        )

            k_time = time.perf_counter() - _k_start
            if level_callback:
                level_callback(k, 0, n_freq, k_time * 1000)
            logger.info(f"  K={k}: {n_freq:,} frequent in {k_time:.1f}s")

            if n_freq == 0:
                break

            # Progressive bitvector deallocation across all GPUs (skip in sparse mode)
            # Replace single-threaded np.unique on
            # (n, k) int32 (~30-90s on K=6's 2.58B elements) with Rust parallel
            # bitset extract (sub-second). Falls back to np.unique if older wheel.
            try:
                from et_miner.backends import get_rust_ext

                et_miner_rust = get_rust_ext()
                if hasattr(et_miner_rust, "unique_columns_from_flat"):
                    current_live_mgpu = set(
                        int(x)
                        for x in et_miner_rust.unique_columns_from_flat(
                            np.ascontiguousarray(current_flat, dtype=np.int32).ravel()
                        )
                    )
                else:
                    current_live_mgpu = set(np.unique(current_flat).tolist())
            except (ImportError, AttributeError):
                current_live_mgpu = set(np.unique(current_flat).tolist())
            dead_cols = prev_live_mgpu - current_live_mgpu
            if dead_cols and bitvecs_list:
                dead_arr = sorted(dead_cols)
                for bv, did, _ in bitvecs_list:
                    with cp.cuda.Device(did):
                        dead_idx = cp.array(dead_arr, dtype=cp.int64)
                        bv[dead_idx] = 0
                freed_mb = len(dead_cols) * bitvecs_list[0][0].shape[1] * 8 / 1024**2
                logger.debug(
                    f"    Pruning: zeroed {len(dead_cols)} dead bitvecs at K={k} ({freed_mb:.0f} MB logical × {len(bitvecs_list)} GPUs)"
                )
            prev_live_mgpu = current_live_mgpu

            # V3: Closed itemset pruning — remove non-closed before next-level candidate gen
            if prune_closed and n_freq > 0:
                current_flat, current_counts_raw = _prune_closed_flat(
                    current_flat, current_counts_raw, prev_frequent_flat, prev_counts_flat
                )
                n_freq = len(current_flat)

            # R2 invariant: binary search in prune_closed_flat_raw requires
            # prev_flat to be sorted-by-row. K>=3 decode produces sorted output
            # (groups sorted by prefix), but K=2 decode (triangular enumeration)
            # is NOT lex-sorted. Sort here at K=2 to guarantee the invariant.
            # Cost: ~10K rows at K=2 = sub-millisecond.
            if n_freq > 0 and k == 2:
                sort_idx = np.lexsort(current_flat[:, ::-1].T)
                current_flat = current_flat[sort_idx]
                current_counts_raw = current_counts_raw[sort_idx]

            prev_frequent_flat = current_flat
            prev_counts_flat = current_counts_raw if n_freq > 0 else np.empty(0, dtype=np.int64)
            k += 1
    finally:
        # sparse_state is local — GC handles cleanup, no manual delattr needed
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
        return _empty_result()

    # Build DataFrame from numpy arrays — zero .tolist() overhead
    if not deferred_itemsets_np:
        return _empty_result()

    all_supports = np.concatenate(deferred_supports)

    # PyArrow path: O(1) Python overhead via Arrow ListArray from numpy
    try:
        import pyarrow as pa

        flat_values = np.concatenate([a.ravel() for a in deferred_itemsets_np])
        widths = np.concatenate([np.full(a.shape[0], a.shape[1], dtype=np.int64) for a in deferred_itemsets_np])
        offsets = np.empty(len(widths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(widths, out=offsets[1:])
        arrow_list = pa.LargeListArray.from_arrays(offsets, flat_values)
        return pl.DataFrame(
            {
                "itemset": pl.Series("itemset", arrow_list),
                "support": all_supports,
            }
        )
    except (ImportError, Exception):
        # Fallback: single-pass list construction
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
    phase2_support: float = 0.00001,
    max_length: int | None = None,
    item_col: str = "items",
    n_gpus: int = 1,
    output_dir: str | None = None,
    sparse_from_k: int | str | None = SPARSE_AUTO,
    level_callback=None,
) -> tuple:
    """Two-phase mining: anchor discovery + neighborhood zoom.

    Phase 1 mines at phase1_support to find anchor items — the items that
    participate in frequent patterns at reasonable support thresholds.

    Phase 2 mines at phase2_support (much lower), restricting candidates to
    those containing >=1 anchor item from Phase 1. This reduces candidate
    explosion from billions to a tractable search space focused on the
    neighborhoods of known-interesting items.

    Args:
        transactions: Transaction data (DataFrame or LazyFrame).
        phase1_support: Support threshold for anchor discovery (higher).
        phase2_support: Support threshold for neighborhood zoom (lower).
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name containing item lists.
        n_gpus: Number of GPUs to use.
        output_dir: Directory for per-K Parquet output. Phase 1 writes to
            output_dir/phase1/, Phase 2 to output_dir/phase2/.
        sparse_from_k: Dense→sparse CSR transition. "auto" (default) switches
            when the previous level's measured mean support drops below the
            n/32 byte-cost crossover; an int fixes the K-level; None never
            switches.
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


