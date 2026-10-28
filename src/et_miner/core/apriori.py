"""apriori(): parameter validation and the routing to a mining route.

Routes: the CPU miner (et_miner.core.cpu_miner, the default), the row-split
GPU miner (et_miner.gpu.row_split; use_gpu=True or bitvecs=), and SON
streaming on one or more devices (et_miner.streaming; streaming=True). This
module validates the parameters and rejects combinations a route cannot
honour.

Usage:
    from et_miner import apriori

    itemsets = apriori(df, min_support=0.01)
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Callable
from math import comb
from typing import Any, Literal

import polars as pl

from et_miner import _env
from et_miner.gpu.density import validate_sparse_from_k

from .profiling import ProfilingSession
from .result import (
    _INT64_ITEM_DTYPES,
    _INT_ITEM_DTYPES,
    _emitted_items,
    _empty_result,
    _item_dtype,
)

def _warn_complexity(
    n_frequent_items: int,
    min_support: float,
) -> None:
    """Warn user if computation might be expensive."""
    # Estimate k=2 candidates (n choose 2)
    n_pairs = comb(n_frequent_items, 2) if n_frequent_items > 1 else 0

    # Warning thresholds (tuned from benchmarks)
    if n_pairs > 10_000_000:
        warnings.warn(
            f"Very large computation ahead!\n"
            f"  Frequent items: {n_frequent_items:,}\n"
            f"  Candidate pairs: {n_pairs:,}\n"
            f"  Estimated time: >30 minutes\n"
            f"  Consider increasing min_support (currently {min_support})",
            UserWarning,
            stacklevel=3,
        )
    elif n_pairs > 1_000_000:
        warnings.warn(
            f"Large computation: {n_pairs:,} candidate pairs.\n  This may take several minutes.",
            UserWarning,
            stacklevel=3,
        )


def _validate_parameters(
    min_support: float,
    max_length: int | None,
    batch_size: int | None,
    sparse_from_k: int | str | None = None,
    prune_apriori: bool = True,
) -> None:
    """Validate apriori parameters with clear error messages."""
    # min_support: must be numeric in [0.0, 1.0]
    if not isinstance(min_support, (int, float)):
        raise TypeError(f"min_support must be numeric, got {type(min_support).__name__}")
    if not 0.0 <= min_support <= 1.0:
        raise ValueError(
            f"min_support must be in range [0.0, 1.0], got {min_support}\n"
            f"  Hint: Support is a proportion (e.g., 0.01 = 1%)"
        )

    # max_length: must be positive int or None
    if max_length is not None:
        if not isinstance(max_length, int):
            raise TypeError(f"max_length must be int or None, got {type(max_length).__name__}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")

    # batch_size: must be positive int or None
    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be int or None, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    validate_sparse_from_k(sparse_from_k)
    if not isinstance(prune_apriori, bool):
        raise TypeError(f"prune_apriori must be bool, got {type(prune_apriori).__name__}")


def _validate_route_support(
    *,
    streaming: bool,
    use_gpu: bool,
    has_bitvecs: bool,
    n_gpus: int,
    profile: bool,
    gpu_resident: bool,
    prune_equal_support: bool,
    use_generator_pruning: bool,
    anchor_items: set | None,
    output_dir: str | None,
    resume_from_k: int | None,
    memory_budget_gb: float | None,
    prune_apriori: bool = True,
) -> None:
    """Reject parameter/route combinations the chosen route cannot honour.

    apriori() takes a wide parameter set and dispatches to one of four routes
    (the row-split GPU miner, single- and multi-GPU SON streaming, the CPU
    miner), not all of which implement all of it. Every mismatch below used to be
    SILENT: the caller passed the parameter, the route dropped it, and nothing
    in the return value or the logs said so. Measured on a 400-row fixture:

      streaming=True + prune_equal_support  -> the complete 214-itemset lattice
                                               where the gated answer is 176
      anchor_items on the CPU route         -> 214 rows, identical to unanchored
      output_dir on the CPU route           -> files written: []
      profile=True + streaming + n_gpus>1   -> a bare DataFrame, which then
                                               UNPACKS into two Series, so
                                               `result, session = apriori(...)`
                                               succeeds and hands back a column
                                               of itemsets and a column of floats

    The file already raised for one such combination (profile + pruning on a GPU
    path); this generalises that to every one of them. Raising is the right
    answer rather than best-effort forwarding: a miner whose contract is
    exactness should not quietly answer a different question.

    Where a route CAN honour a parameter it is forwarded instead -- output_dir
    and resume_from_k on the bitvecs route, and memory_budget_gb on multi-GPU
    streaming -- so this only fires where the capability genuinely does not
    exist.
    """
    if gpu_resident:
        raise ValueError(
            "gpu_resident was removed: use_gpu=True (or bitvecs=) mines on the row-split miner, "
            "which keeps candidates and counts on the GPU. Drop the argument."
        )

    # prune_equal_support / use_generator_pruning: the row-split miner alone
    # implements the gates, and it is only reachable via use_gpu or bitvecs=.
    if streaming and (prune_equal_support or use_generator_pruning):
        which = " and ".join(
            n for n, v in (("prune_equal_support", prune_equal_support),
                           ("use_generator_pruning", use_generator_pruning)) if v
        )
        raise ValueError(
            f"streaming=True cannot be combined with {which}: the SON streaming "
            "paths do not implement the pruning gates, and would return the "
            "complete frequent lattice instead of the free-sets. Drop one of "
            "the two, or mine without streaming."
        )

    # anchor_items is implemented by the row-split miner for transactions
    # input only. The bitvecs route REFUSES it rather than forwarding: that
    # route forwards output_dir and resume_from_k, so anchoring there would put
    # anchoring on a route that also persists and resumes -- and an anchored
    # per-K parquet is not a resumable mining state (see the resume_from_k
    # check below). Refusing keeps everything that route flushes a complete
    # level for a structural reason, rather than only while a second guard
    # holds. Nothing in the tree pairs bitvecs= with anchor_items.
    if anchor_items is not None and not (use_gpu and not streaming and not has_bitvecs):
        raise ValueError(
            "anchor_items requires the row-split miner, reached with "
            "use_gpu=True, streaming=False and transactions (not bitvecs=): "
            "only that miner implements anchor filtering, and every other route "
            "would silently return the complete, differently-shaped lattice."
        )

    # profile builds a ProfilingSession, which every route but multi-GPU SON does.
    if profile and streaming and n_gpus > 1 and not has_bitvecs:
        raise ValueError(
            "profile=True cannot be combined with streaming=True and "
            "n_gpus>1: the multi-GPU streaming path builds no "
            "ProfilingSession and would return a bare DataFrame, which "
            "unpacks silently into two Series."
        )

    # The row-split miner serves bitvecs= (that branch runs first) and use_gpu
    # without streaming.
    reaches_row_split = has_bitvecs or (use_gpu and not streaming)

    # The CPU route generates candidates with the subset test built in, and SON
    # mines its chunks with the test on; only the row-split miner can skip it.
    if use_generator_pruning and not prune_apriori:
        raise ValueError(
            "use_generator_pruning needs prune_apriori=True: on the row-split miner the count "
            "inference runs inside the same subset test, and without it every candidate is counted."
        )
    if not prune_apriori and not reaches_row_split:
        raise ValueError(
            "prune_apriori=False requires the row-split miner, reached with bitvecs= or with "
            "use_gpu=True and streaming=False: the CPU route and SON always test every "
            "candidate's (k-1)-subsets, so the flag would be ignored there."
        )

    # output_dir / resume_from_k are the row-split miner's per-K parquet flush.
    if output_dir is not None or resume_from_k is not None:
        which = " and ".join(
            n for n, v in (("output_dir", output_dir), ("resume_from_k", resume_from_k))
            if v is not None
        )
        if not reaches_row_split:
            raise ValueError(
                f"{which} requires the row-split miner, reached with bitvecs= or "
                "with use_gpu=True and streaming=False. On this route the per-K "
                "parquet flush does not run: output_dir would stay empty and "
                "resume_from_k would silently re-mine from K=1."
            )


    # A persisted K-level is a valid resume artifact IFF it is the complete
    # frequent level at that K -- resume reloads it as BOTH the generation base
    # and the subset oracle for every level above.
    #
    # Anchoring emits a strict subset of the level, so an anchored parquet is a
    # REPORT, not a mining state. Resuming from one reconstructs exactly the
    # restricted generation base that made anchoring unsound in the first place:
    # measured 57 itemsets against 838 for the same call without resume, a 93%
    # silent loss, every level after the resume point wrong.
    #
    # The free-set prune violates the same rule in the SAFE direction (it emits
    # a subset that is closed under the operations the next level needs) and
    # says so where it resumes; anchoring violates it in the unsafe direction,
    # because anchoredness is not anti-monotone. Only the unsafe one is refused.
    #
    # Refusing the READ rather than the write is deliberate: mine_two_phase
    # already produces anchored artifacts on disk, so guarding the writer would
    # break that feature, while guarding the reader closes the hole whoever
    # wrote the file -- including a user resuming by hand from a directory
    # mine_two_phase left behind.
    if resume_from_k is not None and anchor_items is not None:
        raise ValueError(
            "resume_from_k cannot be combined with anchor_items: an anchored "
            "per-K parquet holds only the itemsets containing an anchor, so it "
            "is a report rather than a resumable mining state. Resuming from it "
            "would rebuild the next level from a restricted base and silently "
            "lose most of the lattice. Re-mine without resume_from_k, or resume "
            "from a run that was not anchored."
        )

    # memory_budget_gb sizes the SON chunk; nothing else reads it.
    if memory_budget_gb is not None and not streaming:
        raise ValueError(
            "memory_budget_gb requires streaming=True: it overrides chunk_size "
            "for the SON paths and is read nowhere else."
        )


def _beyond_int64(items: pl.Series) -> bool:
    """Whether an item id lies outside the Int64 range (nulls are no items and do not count)."""
    return items.cast(pl.Int64, strict=False).null_count() > items.null_count()


def _as_item_dtype(result: Any, items: pl.Series, column_ids: bool) -> Any:
    """The GPU route's result (a frame, or (frame, session)) with its itemsets as the CPU route emits ``items``.

    With ``column_ids`` the route mined column ids, which map to ``items``
    (cast as ``_emitted_items`` does); otherwise it emitted the item ids as
    Int64, which are cast back. Either way the itemsets are copied once.
    """
    df, session = result if isinstance(result, tuple) else (result, None)
    items = _emitted_items(items)
    if column_ids:
        cols = pl.Series(range(len(items)), dtype=pl.Int64)
        itemset = pl.col("itemset").list.eval(pl.element().replace_strict(cols, items, return_dtype=items.dtype))
    else:
        itemset = pl.col("itemset").cast(pl.List(items.dtype))
    df = df.with_columns(itemset)
    return df if session is None else (df, session)


def _coded_items(column: pl.Series) -> tuple[pl.Series, pl.Series]:
    """(the list column with every item replaced by its rank among the distinct items as Int32, those items sorted).

    Nulls stay null. The ranks follow the item order, so ascending codes are ascending items.
    """
    items = column.explode().drop_nulls().unique().sort()
    ranks = pl.Series(range(len(items)), dtype=pl.Int32)
    return column.list.eval(pl.element().replace_strict(items, ranks, default=None)), items


def _release_cached_blocks(n_gpus: int, device: int = 0) -> None:
    """Hand CuPy's cached blocks back to the driver on the devices a GPU run uses.

    An earlier call in this process leaves its blocks cached in the pool. This
    run's first small allocations would split them, a split block cannot be
    released, and the run's largest allocation then fails on a device that has
    the room for it. One GPU means ``device``; more mean devices 0..n-1.
    """
    import cupy as cp

    devices = [device] if n_gpus <= 1 else range(min(n_gpus, cp.cuda.runtime.getDeviceCount()))
    for d in devices:
        with cp.cuda.Device(d):
            cp.get_default_memory_pool().free_all_blocks()


def apriori(
    transactions: pl.LazyFrame | pl.DataFrame | None = None,
    min_support: float = 0.5,
    max_length: int | None = None,
    item_col: str = "items",
    use_gpu: bool = False,
    batch_size: int | None = 10_000,
    profile: bool = False,
    show_progress: bool = False,
    warn_complexity: bool = True,
    prune_equal_support: bool = False,
    use_generator_pruning: bool = False,
    prune_apriori: bool = True,
    sparse: bool | None = None,
    n_jobs: int = 1,
    enable_length_filter: bool = True,
    # Streaming parameters (SON algorithm)
    streaming: bool = False,
    chunk_size: int = 10_000_000,
    n_gpus: int = 1,
    memory_budget_gb: float | None = None,
    progress_callback: Callable[[str, int, int, dict[str, Any]], None] | None = None,
    level_callback: Callable[[int, int, int, float], None] | None = None,
    # Removed: True raises ValueError
    gpu_resident: bool = False,
    # Pre-built bitvector input (GPU fast path)
    bitvecs: tuple | None = None,
    # Per-K Parquet flush (prevents CPU RAM OOM at K=7+ scale)
    output_dir: str | None = None,
    # Resume from K=N+1 by loading K=N frequent itemsets from parquet
    resume_from_k: int | None = None,
    # Memory guard limits for exhaustive mining
    max_ram_gb: float = 800.0,
    max_vram_gb: float = 70.0,
    # ESCO: dense→sparse CSR transition, fixed K or measured density
    sparse_from_k: int | Literal["auto"] | None = None,
    # V3 B6: restrict candidates to anchor neighborhoods (two-phase mining)
    anchor_items: set | None = None,
) -> pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]:
    """Find the frequent itemsets (or free-sets) of a transaction list column.

    The CPU route (the default) builds one CSR of the frequent items and mines
    every level from it; use_gpu, bitvecs= and streaming select the other routes.

    Args:
        transactions: Transaction data with item lists.
        min_support: Minimum support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name with item lists.
        use_gpu: Mine on the GPU with the row-split miner (requires CuPy).
        batch_size: Validated (a positive int or None) but read by no route: SON's
            CPU passes count from a CSR and no longer batch candidates.
        profile: If True, return profiling metrics alongside results.
        show_progress: If True, display progress bars where a route has them
            (requires tqdm).
        warn_complexity: If True, warn when candidate pairs > 1M.
        prune_equal_support: Mine frequent FREE-SETS (generators) instead of the
            complete lattice. An itemset is free when no proper subset has the
            same support [Bastide et al. 2000]; free-sets are anti-monotone, so
            each level is generated from the previous level's free-sets alone —
            3-50x fewer candidates on dense data. The returned lattice is
            exactly the free-sets: what is emitted is what the next level is
            generated from, so the support of every omitted frequent itemset
            equals that of one of its subsets. False (default) returns the
            complete frequent lattice. This is NOT closed-itemset mining, which
            asks about equal-support supersets.
        use_generator_pruning: Infer support from (k-1)-subsets instead of counting.
            Pascal [Bastide et al. 2000]: a candidate with a non-free (k-1)-subset
            is itself non-free and its support is exactly the minimum of its
            (k-1)-subset supports, so it never has to be counted. Exact, no
            impact on results. CPU route and the row-split GPU miner; on the
            GPU it needs ``prune_apriori`` and applies to complete-lattice
            runs (in a free-set run such candidates are not free and are
            skipped anyway).
        prune_apriori: Skip candidates with an infrequent (k-1)-subset before
            counting them (Apriori's subset test; exact, no impact on results).
            The row-split GPU miner tests every K>=3 candidate on the device
            against the previous level and leaves the ones it settles
            uncounted; in a free-set run the test runs against the free level,
            which also skips every candidate that cannot be free. False counts
            every candidate the prefix groups generate, and is honoured only by
            the row-split miner (the CPU route and SON always test).
        sparse: Removed; only None is accepted (any other value raises).
        n_jobs: Parallel workers for counting. 1 = sequential, -1 = all CPUs.
        enable_length_filter: Skip transactions shorter than k when counting
            k-itemsets. Set to False to disable (results are the same).
        streaming: Use SON algorithm for chunked processing. Memory holds one
            chunk's CSR and the candidates instead of the whole dataset.
        chunk_size: Transactions per chunk when streaming=True. Default 10M.
        n_gpus: GPUs for the row-split miner and for streaming (default 1).
            Requires CuPy.
        memory_budget_gb: Auto-calculate chunk_size to fit this budget.
            Overrides chunk_size. Only used when streaming=True.
        progress_callback: Streaming progress callback
            (phase, chunk_idx, n_chunks, metrics).
        level_callback: Per-level callback
            (k, n_candidates, n_frequent, duration_ms).

            **n_candidates is route-dependent and the routes do not agree.**
            The CPU path applies the full per-candidate subset test before
            counting, so it reports candidates that survived it. The GPU path
            reports every candidate its prefix groups generate, including the
            ones its device-side subset test (``prune_apriori``) then leaves
            uncounted, so it reports more candidates than the CPU path for the
            same input at the same level. Both are honest counts of what that
            route enumerated; neither is "the" candidate count. A consumer
            doing per-level cost accounting should treat the figure as
            comparable within a route and not across routes.

            n_frequent and duration_ms mean the same thing everywhere.
        bitvecs: Pre-built GPU bitvectors tuple (bitvecs_gpu, col_to_item, n_transactions).
            Skips DataFrame conversion; transactions must be None when provided.
            The array is READ-ONLY to the engine: ET-Miner writes only to
            bitvectors it builds itself, never to one it is handed, so the
            caller may reuse it after the call without copying it first.
        sparse_from_k: In-core GPU paths only — an int switches to sparse CSR
            tidsets from that K (at least 3); "auto" switches when the previous
            level's mean count falls below n_transactions/32. None (default)
            keeps dense bitvectors. The transition is one-way. CPU mining
            ignores this GPU setting; SON streaming rejects it.
        max_ram_gb / max_vram_gb: Memory guards for the GPU routes
            (use_gpu=True without streaming, or bitvecs=): MemoryError between
            K-levels once host RSS or the largest device's CuPy pool exceeds
            them. Candidate chunks are sized from measured free VRAM either way.
        gpu_resident: Removed; True raises ValueError. The row-split miner
            keeps candidates and counts on the GPU.

    Returns:
        DataFrame with itemset and support (Float64) columns. Integer item ids
        come back as List(Int64), except UInt64, Int128 and UInt128 ids, which
        keep their dtype (an empty result too).

        Every itemset is emitted as an **ascending tuple of item ids**, on every
        route. That is a contract, not an accident: both parquet consumers in
        core.rules join K against K-1 positionally, and _build_support_lookup is
        keyed on the itemset, so a producer emitting [10, 2] where another emits
        [2, 10] silently loses rows and corrupts numbers across routes. The
        order is also stable across min_support values, so a stored artifact or
        a frozen digest keyed on the emitted list stays valid when the threshold
        moves.

        Row order within the frame is NOT part of the contract -- only the order
        of items within each itemset.

        If profile=True: tuple of (DataFrame, ProfilingSession).

    Example:
        >>> result = apriori(df, min_support=0.5)
        >>> result = apriori(df, min_support=0.001, use_gpu=True)
        >>> result, session = apriori(df, min_support=0.1, profile=True)
        >>> result = apriori(df, min_support=0.0001, n_jobs=-1)
        >>> result = apriori(huge_df, min_support=0.001, streaming=True, n_gpus=8)
    """
    _validate_parameters(min_support, max_length, batch_size, sparse_from_k, prune_apriori)
    _env.reject_removed_knobs()

    if streaming and bitvecs is None and sparse_from_k is not None:
        raise ValueError("sparse_from_k requires in-core GPU mining; SON streaming does not implement the transition")

    # Every parameter/route mismatch is rejected here, ABOVE the routing, so a
    # route cannot silently drop something the caller asked for. This has to run
    # before `if streaming:` — that branch returns long before the GPU routing
    # is reached, which is how streaming came to swallow prune_equal_support.
    _validate_route_support(
        streaming=streaming,
        use_gpu=use_gpu,
        has_bitvecs=bitvecs is not None,
        n_gpus=n_gpus,
        profile=profile,
        gpu_resident=gpu_resident,
        prune_equal_support=prune_equal_support,
        use_generator_pruning=use_generator_pruning,
        prune_apriori=prune_apriori,
        anchor_items=anchor_items,
        output_dir=output_dir,
        resume_from_k=resume_from_k,
        memory_budget_gb=memory_budget_gb,
    )

    # Route to bitvecs fast path if pre-built GPU bitvectors provided
    if bitvecs is not None:
        # Validate: cannot provide both bitvecs and transactions
        if transactions is not None:
            raise ValueError(
                "Cannot provide both 'transactions' and 'bitvecs'. "
                "Use 'bitvecs' for pre-built GPU bitvectors or 'transactions' for DataFrame input."
            )

        # Unpack and validate bitvecs tuple
        if not isinstance(bitvecs, tuple) or len(bitvecs) != 3:
            raise ValueError("bitvecs must be a tuple of (bitvecs_gpu, col_to_item, n_transactions)")

        bitvecs_gpu, col_to_item, n_trans = bitvecs

        # Validate bitvecs_gpu is a CuPy array
        if not hasattr(bitvecs_gpu, "__cuda_array_interface__"):
            raise TypeError(
                f"bitvecs_gpu must be a CuPy array (has __cuda_array_interface__), got {type(bitvecs_gpu).__name__}"
            )

        # Validate shape
        if bitvecs_gpu.ndim != 2:
            raise ValueError(f"bitvecs_gpu must be 2D (n_cols, n_u64s), got {bitvecs_gpu.ndim}D")

        # Validate n_u64s matches n_transactions
        expected_n_u64s = math.ceil(n_trans / 64)
        if bitvecs_gpu.shape[1] != expected_n_u64s:
            raise ValueError(
                f"bitvecs_gpu.shape[1] ({bitvecs_gpu.shape[1]}) must equal "
                f"ceil(n_transactions/64) = ceil({n_trans}/64) = {expected_n_u64s}"
            )

        # Validate col_to_item keys match n_cols
        if len(col_to_item) != bitvecs_gpu.shape[0]:
            raise ValueError(
                f"col_to_item has {len(col_to_item)} keys but bitvecs_gpu has {bitvecs_gpu.shape[0]} columns"
            )

        # The route holds item ids as 64-bit integers; a float would be truncated.
        if not all(isinstance(v, numbers.Integral) for v in col_to_item.values()):
            raise ValueError(
                "col_to_item must map every column to an integer item id: the bitvecs route holds item ids as "
                "64-bit integers. Map the items to integer ids first."
            )

        # The kernels index the array as col * n_u64s + word.
        if bitvecs_gpu.dtype != "uint64" or not bitvecs_gpu.flags.c_contiguous:
            raise ValueError(
                f"bitvecs_gpu must be a C-contiguous uint64 array (cupy.ascontiguousarray), got "
                f"{bitvecs_gpu.dtype}, c_contiguous={bitvecs_gpu.flags.c_contiguous}"
            )

        from et_miner.gpu.row_split import _apriori_row_split_multi_gpu, shard_prebuilt_bitvecs

        _release_cached_blocks(n_gpus, int(bitvecs_gpu.device.id))
        return _apriori_row_split_multi_gpu(
            None,
            col_to_item,
            n_trans,
            min_support,
            max_length,
            max(1, n_gpus),
            level_callback,
            bitvecs_list=shard_prebuilt_bitvecs(bitvecs_gpu, n_trans, max(1, n_gpus)),
            output_dir=output_dir,
            resume_from_k=resume_from_k,
            prune_non_free=prune_equal_support,
            prune_apriori=prune_apriori,
            infer_counts=use_generator_pruning,
            sparse_from_k=sparse_from_k,
            profile=profile,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
        )

    # Validate transactions is provided when bitvecs is not
    if transactions is None:
        raise ValueError("Either 'transactions' or 'bitvecs' must be provided")

    if sparse is not None:
        raise ValueError(
            "sparse= was removed with the scipy sparse counter (route consolidation C4): no route reads it. "
            "Drop the argument."
        )

    # Route to streaming implementation if requested
    if streaming:
        # Multi-GPU streaming if n_gpus > 1
        if n_gpus > 1:
            from et_miner.streaming.multi_gpu import apriori_streaming_multi_gpu

            return apriori_streaming_multi_gpu(
                transactions,
                min_support=min_support,
                max_length=max_length,
                item_col=item_col,
                n_gpus=n_gpus,
                chunk_size=chunk_size,
                memory_budget_gb=memory_budget_gb,
                batch_size=batch_size,
                show_progress=show_progress,
                sparse=sparse,
                n_jobs=n_jobs,
                progress_callback=progress_callback,
            )

        # Single-GPU/CPU streaming
        from et_miner.streaming.son import apriori_streaming

        return apriori_streaming(
            transactions,
            min_support=min_support,
            max_length=max_length,
            item_col=item_col,
            chunk_size=chunk_size,
            memory_budget_gb=memory_budget_gb,
            use_gpu=use_gpu,
            gpu_resident=gpu_resident,
            batch_size=batch_size,
            profile=profile,
            show_progress=show_progress,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
        )

    lf = transactions.lazy() if isinstance(transactions, pl.DataFrame) else transactions

    # ── GPU direct path: transactions → CSR → GPU bitvecs ──────────────
    # Bypasses the dense boolean matrix entirely.
    # For 205M × 1006: ~5 GB CPU + ~25 GB GPU  instead of  206 GB CPU.
    if use_gpu:
        from .matrix import _build_csr_from_transactions

        # The route holds item ids as 64-bit integers. Other items (strings,
        # floats, categoricals) and ids of a wider dtype that lie outside Int64
        # mine as column ids, and the result maps them back.
        item_dtype = _item_dtype(lf, item_col)
        integer = item_dtype in _INT_ITEM_DTYPES
        stores = output_dir or resume_from_k is not None or anchor_items is not None
        if stores and not integer:
            raise ValueError(
                f"items of {item_dtype} are not integer ids: output_dir, resume_from_k and anchor_items store "
                "item ids as 64-bit integers on the GPU route. Map the items to integer ids, or mine without "
                "those options."
            )

        def stores_ids(items: pl.Series) -> None:
            if stores and item_dtype not in _INT64_ITEM_DTYPES and _beyond_int64(items):
                raise ValueError(
                    f"item ids of {item_dtype} lie outside the Int64 range: output_dir, resume_from_k and "
                    "anchor_items store item ids in 64 bits on the GPU route. Map the ids to Int64, or mine "
                    "without those options."
                )

        # Input layer: the device layer for the integer ids it serves; any other column coded to integer ids
        # (Polars, item order) and mined there too; the host CSR only where neither can (more distinct items
        # than INT_SPAN_LIMIT, or output_dir / resume_from_k / anchor_items on a column the device layer does
        # not serve), or when ET_MINER_INPUT_LAYER=host pins it.
        layer = _env.input_layer()
        if layer != "host" and item_dtype in _INT64_ITEM_DTYPES:
            from et_miner.gpu.input_layer import NOT_SERVED, build_device_input

            column = lf.select(pl.col(item_col)).collect(engine="in-memory").get_column(item_col)
            _release_cached_blocks(n_gpus)
            built = build_device_input(
                column,
                min_support,
                max(1, n_gpus),
                keep_rows=_env.k2_kernel() != "dense" or _env.k3_kernel() != "dense",
                row_order="lex",
            )
            if built is None:
                return (_empty_result(item_dtype), ProfilingSession()) if profile else _empty_result(item_dtype)
            if built is not NOT_SERVED:
                from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

                return _apriori_row_split_multi_gpu(
                    None,
                    built.col_to_item,
                    built.n_transactions,
                    min_support,
                    max_length,
                    max(1, n_gpus),
                    level_callback,
                    output_dir=output_dir,
                    resume_from_k=resume_from_k,
                    prune_non_free=prune_equal_support,
                    prune_apriori=prune_apriori,
                    infer_counts=use_generator_pruning,
                    sparse_from_k=sparse_from_k,
                    anchor_items=anchor_items,
                    profile=profile,
                    max_ram_gb=max_ram_gb,
                    max_vram_gb=max_vram_gb,
                    device_input=built,
                )
            lf = column.to_frame().lazy()

        if layer != "host" and not stores:
            from et_miner.gpu.input_layer import NOT_SERVED, build_device_input

            from .cpu_miner import INT_SPAN_LIMIT

            column = lf.select(pl.col(item_col)).collect(engine="in-memory").get_column(item_col)
            coded, items = _coded_items(column)
            if len(items) <= INT_SPAN_LIMIT:
                _release_cached_blocks(n_gpus)
                built = build_device_input(
                    coded,
                    min_support,
                    max(1, n_gpus),
                    keep_rows=_env.k2_kernel() != "dense" or _env.k3_kernel() != "dense",
                    row_order="lex",
                )
                if built is None:
                    return (_empty_result(item_dtype), ProfilingSession()) if profile else _empty_result(item_dtype)
                if built is not NOT_SERVED:
                    from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

                    result = _apriori_row_split_multi_gpu(
                        None,
                        built.col_to_item,
                        built.n_transactions,
                        min_support,
                        max_length,
                        max(1, n_gpus),
                        level_callback,
                        prune_non_free=prune_equal_support,
                        prune_apriori=prune_apriori,
                        infer_counts=use_generator_pruning,
                        sparse_from_k=sparse_from_k,
                        profile=profile,
                        max_ram_gb=max_ram_gb,
                        max_vram_gb=max_vram_gb,
                        device_input=built,
                    )
                    return _as_item_dtype(result, items, column_ids=True)
            lf = column.to_frame().lazy()

        csr_result = _build_csr_from_transactions(lf, min_support, item_col, check_items=stores_ids)
        if csr_result is None:
            if profile:
                return _empty_result(item_dtype), ProfilingSession()
            return _empty_result(item_dtype)

        csr, idx_to_item, n_trans = csr_result

        items = pl.Series(list(idx_to_item.values()), dtype=item_dtype) if item_dtype not in _INT64_ITEM_DTYPES else None
        column_ids = items is not None and (not integer or _beyond_int64(items))
        if column_ids:
            idx_to_item = {c: c for c in idx_to_item}

        from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

        _release_cached_blocks(n_gpus)
        result = _apriori_row_split_multi_gpu(
            csr,
            idx_to_item,
            n_trans,
            min_support,
            max_length,
            max(1, n_gpus),
            level_callback,
            output_dir=output_dir,
            resume_from_k=resume_from_k,
            prune_non_free=prune_equal_support,  # free-sets: emit == generate
            prune_apriori=prune_apriori,
            infer_counts=use_generator_pruning,
            sparse_from_k=sparse_from_k,
            anchor_items=anchor_items,  # V3 B6: two-phase anchor filtering
            profile=profile,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
        )
        return result if items is None else _as_item_dtype(result, items, column_ids)

    # ── CPU path: one CSR of the frequent items, mined level by level ──
    from .cpu_miner import mine_cpu

    return mine_cpu(
        lf,
        min_support,
        max_length,
        item_col,
        prune_equal_support=prune_equal_support,
        use_generator_pruning=use_generator_pruning,
        enable_length_filter=enable_length_filter,
        n_jobs=n_jobs,
        level_callback=level_callback,
        profile=profile,
        warn_complexity=warn_complexity,
    )

