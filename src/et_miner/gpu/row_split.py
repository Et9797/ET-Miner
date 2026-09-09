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
    _anchor_keep_mask,
    _prune_groups_apriori,
    _prune_non_free_mask,
    _rows_sorted,
)
from et_miner.gpu.nccl import _init_nccl
from et_miner.gpu.sparse_csr import (
    SparseMiningState,
    convert_shards_to_csr,
    free_groups,
    log_new_shards,
    materialize_survivors,
    run_sparse_level,
    upload_groups_to_shards,
)
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


def _release_level_state(groups_gpu, sparse_state) -> None:
    """Release both level-scoped GPU allocations, independently.

    These are two unrelated allocations -- the current level's group arrays
    (tens of GB on a dense K>=3 level) and the resident CSR shards -- and they
    used to share one `try` with a bare `except Exception: pass`. Any failure in
    `free_groups` therefore skipped `sparse_state.release()` entirely, leaking
    the shards, and the `pass` meant nothing recorded that it had happened.

    The bare catch itself is deliberate and stays: this runs from a `finally`,
    frequently while an exception is already propagating, and a cleanup failure
    must never replace the original error. What changes is that a failure in one
    cannot cancel the other, and that both are logged instead of swallowed.

    Extracted to module scope so the isolation is testable directly, rather than
    by driving a whole mining run to failure at the right moment.
    """
    for label, release in (
        ("group arrays", lambda: free_groups(groups_gpu)),
        ("CSR shards", sparse_state.release),
    ):
        try:
            release()
        except Exception as exc:  # noqa: BLE001 -- see docstring
            logger.warning(f"    Cleanup failed while releasing {label}: {exc!r}")


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
    prune_apriori: bool = False,  # Apriori subset pruning on candidate groups
    sparse_from_k: int | str | None = None,  # V3: CSR from this K level, or "auto" = measured density
    anchor_items: set | None = None,  # V3 B6: two-phase anchor filtering
) -> "pl.DataFrame":
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

    # Sparse CSR mode state: one GPU-resident shard per device (gpu.sparse_csr),
    # sticky once the transition fires.
    sparse_state = SparseMiningState()
    _sparse_groups_gpu = None

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
        # resumed level keeps the free-set prune and the "auto" density
        # transition. count → support → count round-trips exactly through
        # float64 for any int32-range count.
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
        if level_callback:
            level_callback(1, n_cols, len(prev_frequent_flat), k1_time * 1000)
        logger.info(f"  K=1: {len(prev_frequent_flat):,} frequent items in {k1_time:.1f}s")

        if len(prev_frequent_flat) == 0:
            return _build_result_df([])

        k = 2

    # ── K>=2: GPU-resident dense counting ────────────────────────────────
    # State: prev_frequent_flat — numpy (n_freq, k-1) array of column indices.
    # No Python tuples in the hot path. Results decoded at end of each level
    # via vectorized numpy. Next-level groups built from flat arrays directly.
    try:
        while k <= effective_max_length and prev_frequent_flat.shape[0] >= k:
            _k_start = time.perf_counter()
            # Sparse mode bookkeeping: the survivor candidate indices (filtered and
            # permuted in lockstep with current_flat by every later step, so the
            # shards are materialized in the final row order), the resident group
            # arrays of this level, and the candidate count for the level callback.
            _surv = None
            _sparse_groups_gpu = None
            _n_cands_cb = 0

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
                # ═══ SPARSE CSR PATH — GPU-resident row-split shards (gpu.sparse_csr) ═══
                if not sparse_state.active:
                    _trigger = (
                        f"measured mean support {_mean_count / n_transactions:.4%} "
                        f"< {DENSITY_CROSSOVER:.4%} crossover"
                        if sparse_from_k == SPARSE_AUTO
                        else f"fixed sparse_from_k={sparse_from_k}"
                    )
                    logger.info(f"  ═══ DENSITY TRANSITION at K={k} ({_trigger}): dense bitvec → sparse CSR ═══")
                    # Each GPU converts its own bitvec shard on-device (shard-local
                    # tids, no host merge); per-shard row lengths are verified
                    # against the dense counts of the previous level exactly.
                    sparse_state.shards = convert_shards_to_csr(bitvecs_list, prev_frequent_flat, prev_counts_flat)

                    # Free ALL bitvec VRAM across all GPUs
                    for bv, did, _ in bitvecs_list:
                        with cp.cuda.Device(did):
                            del bv
                            cp.get_default_memory_pool().free_all_blocks()
                    bitvecs_list.clear()
                    logger.debug("    Freed bitvec VRAM across all GPUs")

                # Build groups from prev_frequent, with the suffix-slot → row
                # permutation the CSR kernels enumerate candidates from.
                groups_info = build_k3plus_groups_from_flat(prev_frequent_flat, with_src_rows=True)

                # Apriori pruning — resolved against the COMPLETE previous level.
                # Against the free subset it rejects candidates whose (k-1)-subsets
                # are frequent but not free, which loses frequent itemsets; against
                # the complete level it only drops candidates that cannot be
                # frequent, so it is lossless.
                if prune_apriori and groups_info is not None:
                    tc_before = groups_info.total_candidates
                    prev_freq_set = set(map(tuple, prev_full_flat.tolist()))
                    groups_info = _prune_groups_apriori(groups_info, prev_freq_set, k, prev_flat_np=prev_full_flat)
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
                    _n_cands_cb = tc
                    logger.info(f"  K={k}: {tc:,} candidates (CSR sparse mode)")

                    # Count on every shard (in-kernel candidate enumeration), reduce
                    # the int32 partials, compact survivors — the dense chunk loop.
                    _sparse_groups_gpu = upload_groups_to_shards(groups_info, sparse_state.shards)
                    _surv, current_counts_raw = run_sparse_level(
                        sparse_state.shards,
                        groups_info,
                        _sparse_groups_gpu,
                        min_count_threshold,
                        nccl_comms=nccl_comms,
                        use_nccl=_use_nccl,
                        level_label=f"K={k}",
                    )
                    n_freq = len(_surv)

                    if n_freq > 0:
                        current_flat = decode_k3plus_flat(_surv, groups_info, k)
                        current_counts_raw = current_counts_raw.astype(np.int64)


            elif k == 2:
                freq_cols = sorted(prev_frequent_flat[:, 0])
                n_pairs = len(freq_cols) * (len(freq_cols) - 1) // 2
                _n_cands_cb = n_pairs

                # Chunked by measured VRAM budget — big cards get one chunk,
                # small (or pool-limited) cards split the pair space. The
                # pair space is one synthetic group: whole-in-one-chunk runs
                # may use the shared/tiled kernel, multi-chunk runs are
                # legacy sub-chunks (a partial pair range can't be tiled).
                _k2_budget = compute_chunk_budget(device_ids, group_data_bytes=0, use_nccl=_use_nccl)
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

                # Apriori subset pruning — resolved against the COMPLETE previous
                # level (see the sparse branch above for why that matters).
                if prune_apriori and groups_info is not None:
                    tc_before = groups_info.total_candidates
                    prev_freq_set = set(map(tuple, prev_full_flat.tolist()))
                    groups_info = _prune_groups_apriori(groups_info, prev_freq_set, k, prev_flat_np=prev_full_flat)
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
                    k3_chunks = plan_group_chunks(groups_info.cumulative_pairs, max_cands_per_chunk)

                    logger.debug(
                        f"  K={k}: {tc:,} candidates, {len(k3_chunks)} chunk(s), "
                        f"group data {group_data_bytes / (1 << 30):.1f} GB, "
                        f"budget {max_cands_per_chunk:,} cands/chunk"
                    )

                    # Upload group data to all GPUs ONCE — stays resident across chunks
                    all_groups_gpu = {}
                    for bv, did, _ in bitvecs_list:
                        all_groups_gpu[did] = upload_k3plus_groups(groups_info, did)

                    def _k3plus_chunk_on_gpu(bitvec_gpu, device_id, chunk):
                        with cp.cuda.Device(device_id):
                            return count_k3plus_allcounts(
                                bitvec_gpu,
                                groups_info,
                                bitvec_gpu.shape[1],
                                chunk_start=chunk.start,
                                chunk_size=chunk.size,
                                groups_gpu=all_groups_gpu[device_id],
                                variant="legacy" if chunk.use_legacy else None,
                            )

                    freq_cand_indices, freq_cand_counts = run_chunked_dense_level(
                        bitvecs_list,
                        k3_chunks,
                        _k3plus_chunk_on_gpu,
                        min_count_threshold,
                        nccl_comms,
                        _use_nccl,
                        level_label=f"K={k}",
                    )

                    # Free group data from all GPUs
                    for did in list(all_groups_gpu):
                        with cp.cuda.Device(did):
                            del all_groups_gpu[did]
                            cp.get_default_memory_pool().free_all_blocks()
                    del all_groups_gpu

                    n_freq = len(freq_cand_indices)
                    if n_freq > 0:
                        current_flat = decode_k3plus_flat(
                            freq_cand_indices,
                            groups_info,
                            k,
                        )
                        current_counts_raw = freq_cand_counts  # already int64


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
                if _surv is not None:
                    _surv = _surv[sort_idx]

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
                # Mask form on both branches: the sparse path also carries the
                # survivor index array and must filter it in lockstep, so the
                # shards materialize exactly the kept rows.
                _keep = _prune_non_free_mask(current_flat, current_counts_raw, prev_full_flat, prev_full_counts)
                current_flat = current_flat[_keep]
                current_counts_raw = current_counts_raw[_keep]
                if _surv is not None:
                    _surv = _surv[_keep]
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
            if level_callback:
                level_callback(k, _n_cands_cb, n_freq, k_time * 1000)
            logger.info(f"  K={k}: {n_freq:,} frequent in {k_time:.1f}s")

            if n_freq == 0:
                break

            if sparse_state.active and _sparse_groups_gpu is not None:
                # Rebuild the shards for K+1 in the final row order (skipped when no
                # next level can follow); per-shard lengths are verified against
                # the survivor counts exactly. Row i of the new shards ≡ row i of
                # prev_frequent_flat on every GPU, by construction.
                if n_freq >= k + 1 and k < effective_max_length and _surv is not None:
                    sparse_state.replace(
                        materialize_survivors(
                            sparse_state.shards, _sparse_groups_gpu, _surv, current_counts_raw, level_label=f"K={k}"
                        )
                    )
                    log_new_shards(sparse_state.shards, n_freq)
                free_groups(_sparse_groups_gpu)
                _sparse_groups_gpu = None

            prev_frequent_flat = current_flat
            prev_counts_flat = current_counts_raw
            prev_full_flat = full_flat
            prev_full_counts = full_counts
            k += 1
    finally:
        # Release the resident CSR shards and any group arrays of an aborted
        # level -- independently, so one failure cannot leak the other.
        _release_level_state(_sparse_groups_gpu, sparse_state)
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

    # PyArrow path: O(1) Python overhead via Arrow ListArray from numpy.
    #
    # The int64 cast is load-bearing, not cosmetic. This function has three
    # return paths and they used to disagree on the itemset dtype: both
    # _empty_result() calls above give List(Int64) (core/result.py:55), the
    # list fallback below gives List(Int64) via Python ints, and this path gave
    # List(Int32) -- because col_to_item_arr is np.int32 (see :165, guarded to
    # item IDs < 2**31) and Arrow preserves it. So the ONE path that normally
    # runs was the odd one out, and _apriori_from_bitvecs (gpu/mining.py:599)
    # builds the same lookup as int64, so the two GPU routes disagreed as well.
    #
    # Deliberately asymmetric with the flushed parquet, which stays
    # large_list<int32> (io/flush.py builds its own list array from the same
    # int32 items_flat): widening it would double the itemset bytes of every
    # artifact already on disk, and nothing reads it in a dtype-sensitive way --
    # the resume reader indexes item_to_col[flat_item_ids] (:265), the
    # mine_two_phase anchor read goes through .to_list() (:883), and both
    # core/rules.py consumers are parquet-to-parquet so their join keys are
    # int32 on both sides. Widening those would cost ~84 GB on a K=7 K-1 frame.
    # tests/test_row_split_dtypes.py pins each of those boundaries.
    try:
        import pyarrow as pa

        flat_values = np.concatenate([a.ravel() for a in deferred_itemsets_np]).astype(np.int64, copy=False)
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
    sparse_from_k: int | str | None = SPARSE_AUTO,
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


