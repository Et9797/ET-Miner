"""Apriori algorithm implementation for association rule mining.

Fast and memory-efficient implementation using boolean matrix representation and
vectorized operations, with optional GPU acceleration via CUDA kernels.

Support counting uses column bitwise AND operations:

    support({A,B}) = (col("A") & col("B")).sum()

Key ideas:
1. Boolean matrix representation (not TID-lists)
2. Column bitwise AND operations for support (no Python loops in counting)
3. GPU-ready via CUDA bitvector kernels (CuPy)

The GPU inner loops live in et_miner.gpu (mining, row_split); parquet flush
and GCS upload in et_miner.io; candidate generation in .candidates. This
module keeps parameter validation, the CPU mining loop, and the routing in
apriori() that picks a tier.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable
from math import comb
from typing import Any, Literal

import polars as pl
from loguru import logger

from et_miner.gpu.density import validate_sparse_from_k

from .candidates import _generate_candidates
from .matrix import (
    build_boolean_matrix,
    count_support_batched,
)
from .profiling import ProfilingSession
from .result import (
    _build_result_df,
    _empty_result,
    _min_count,
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

    # sparse_from_k: int K-level, "auto", or None
    validate_sparse_from_k(sparse_from_k)


def _prune_equal_support(
    current_frequent: list[tuple[str, ...]],
    current_counts: dict[tuple[str, ...], int],
    prev_counts: dict[tuple[str, ...], int],
) -> list[tuple[str, ...]]:
    """Keep only the free-sets (generators) of a level.

    An itemset is a *free-set* when no proper subset has the same support
    [Bastide et al. 2000, Pascal]. If support({A,B,C}) == support({A,B}) then C
    is implied by {A,B} and {A,B,C} is not free. Free-sets are anti-monotone —
    every subset of a free-set is free — so the next level can be generated
    from the survivors alone without losing a single free-set.

    NOT closed-itemset mining: closed asks about equal-support *supersets*,
    this asks about equal-support *subsets*, and the two select different
    itemsets. ``prev_counts`` must be the COMPLETE previous level (it is never
    pruned in the caller) or the test misses subsets that were themselves
    pruned and under-prunes.

    Compares raw integer counts, exactly as the GPU path does — a float
    comparison with a tolerance made the two tiers disagree at the boundary.

    Args:
        current_frequent: Frequent k-itemsets from current iteration.
        current_counts: Integer counts for the current k-itemsets.
        prev_counts: Integer counts for the COMPLETE (k-1)-itemset level.

    Returns:
        The free-sets, in input order.
    """
    if not prev_counts:
        return current_frequent

    free = []
    for itemset in current_frequent:
        is_free = True
        current_count = current_counts[itemset]

        # Not free if the count equals any (k-1)-subset's count
        for i in range(len(itemset)):
            subset = itemset[:i] + itemset[i + 1 :]
            if prev_counts.get(subset) == current_count:
                is_free = False
                break

        if is_free:
            free.append(itemset)

    return free




def _infer_count_from_subsets(
    candidate: tuple[str, ...],
    prev_counts: dict[tuple[str, ...], int],
    prev_free: set[tuple[str, ...]] | None,
) -> int | None:
    """Infer a candidate's exact count from its (k-1)-subsets, or None.

    Pascal [Bastide et al. 2000]: if a (k-1)-subset Y of X is *not* free — some
    Z ⊊ Y has rows(Z) = rows(Y) — then for X = Y ∪ {a} we have
    rows(X) = rows(Z ∪ {a}), and any (k-1)-subset W with Z ∪ {a} ⊆ W ⊂ X
    satisfies rows(W) = rows(X). So X is not free either and

        count(X) = min{count(W) : W ⊂ X, |W| = k-1}

    exactly — no counting pass needed. When every (k-1)-subset is free nothing
    can be inferred and the candidate must be counted.

    This replaces an earlier rule that inferred whenever ALL subsets happened to
    share a count, which is unsound: support({A,B}) = support({A,C}) =
    support({B,C}) = 30 with support({A,B,C}) = 15 was its own documented
    counterexample. Here all three pairs are free, so no inference is made.

    Args:
        candidate: Candidate k-itemset to check.
        prev_counts: Integer counts for the COMPLETE (k-1)-itemset level.
        prev_free: The free (generator) (k-1)-itemsets. None disables inference.

    Returns:
        The exact count, or None when the candidate has to be counted.
    """
    k = len(candidate)
    if k < 2 or prev_free is None:
        return None

    subset_counts: list[int] = []
    licensed = False
    for i in range(k):
        subset = candidate[:i] + candidate[i + 1 :]
        count = prev_counts.get(subset)
        if count is None:
            # A subset is not frequent — the candidate cannot be either, and we
            # have no minimum to take. Let the caller count it.
            return None
        subset_counts.append(count)
        if subset not in prev_free:
            licensed = True

    if not licensed:
        return None

    inferred = min(subset_counts)
    logger.debug(f"INFERRED {candidate}: non-free subset licenses count={inferred}")
    return inferred




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
) -> None:
    """Reject parameter/route combinations the chosen route cannot honour.

    apriori() takes a wide parameter set and dispatches to one of six routes,
    not all of which implement all of it. Every mismatch below used to be
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
      gpu_resident + prune_equal_support    -> the row-split result, because
                                               _route_for_pruning is tested first

    The file already raised for one such combination (profile + pruning on a GPU
    path); this generalises that to every one of them. Raising is the right
    answer rather than best-effort forwarding: a miner whose contract is
    exactness should not quietly answer a different question.

    Where a route CAN honour a parameter it is forwarded instead -- output_dir
    and resume_from_k on the bitvecs+pruning route, and memory_budget_gb on
    multi-GPU streaming -- so this only fires where the capability genuinely
    does not exist.
    """
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

    _routes_to_row_split = prune_equal_support and (use_gpu or has_bitvecs)

    # gpu_resident is not implemented by the row-split miner, and pruning wins
    # the routing decision, so the combination silently ignores gpu_resident.
    if gpu_resident and _routes_to_row_split:
        raise ValueError(
            "gpu_resident=True cannot be combined with prune_equal_support: the "
            "pruning gates are implemented by the row-split miner, which has no "
            "GPU-resident mode, so gpu_resident would be silently ignored. Drop "
            "one of the two."
        )

    # anchor_items reaches the row-split miner only through the use_gpu branch.
    if anchor_items is not None and (streaming or not use_gpu):
        raise ValueError(
            "anchor_items requires use_gpu=True and streaming=False: only the "
            "row-split miner implements anchor filtering, and every other route "
            "would silently return the complete, differently-shaped lattice."
        )

    # profile builds a ProfilingSession, which only the CPU loop, the single-GPU
    # bitvec miner and single-GPU SON streaming do.
    if profile:
        if streaming and n_gpus > 1:
            raise ValueError(
                "profile=True cannot be combined with streaming=True and "
                "n_gpus>1: the multi-GPU streaming path builds no "
                "ProfilingSession and would return a bare DataFrame, which "
                "unpacks silently into two Series."
            )
        if _routes_to_row_split or anchor_items is not None or (use_gpu and n_gpus > 1):
            raise ValueError(
                "profile=True cannot be combined with a row-split GPU run "
                "(n_gpus>1, anchor_items, or prune_equal_support): the row-split "
                "miner builds no ProfilingSession and would return a bare "
                "DataFrame, which unpacks silently into two Series."
            )

    # output_dir / resume_from_k are the row-split miner's per-K parquet flush.
    if output_dir is not None or resume_from_k is not None:
        which = " and ".join(
            n for n, v in (("output_dir", output_dir), ("resume_from_k", resume_from_k))
            if v is not None
        )
        reaches_row_split = _routes_to_row_split or (
            use_gpu and not streaming and (n_gpus > 1 or anchor_items is not None)
        )
        if not reaches_row_split:
            raise ValueError(
                f"{which} requires the row-split miner, reached with use_gpu=True "
                "and one of n_gpus>1, anchor_items, or prune_equal_support (or "
                "with bitvecs= and prune_equal_support). On this route the "
                "per-K parquet flush does not run: output_dir would stay empty "
                "and resume_from_k would silently re-mine from K=1."
            )

    # memory_budget_gb sizes the SON chunk; nothing else reads it.
    if memory_budget_gb is not None and not streaming:
        raise ValueError(
            "memory_budget_gb requires streaming=True: it overrides chunk_size "
            "for the SON paths and is read nowhere else."
        )


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
    # GPU-resident mode (zero PCIe round trips)
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
    # V3: dense→sparse CSR transition — int K-level, "auto" = measured density
    sparse_from_k: int | Literal["auto"] | None = None,
    # V3 B6: restrict candidates to anchor neighborhoods (two-phase mining)
    anchor_items: set | None = None,
) -> pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]:
    """Find frequent itemsets using fully vectorized boolean matrix operations.

    Transactions become a boolean matrix; support counting becomes vectorized AND + sum.

    Args:
        transactions: Transaction data with item lists.
        min_support: Minimum support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name with item lists.
        use_gpu: Use GPU acceleration if available (requires polars[gpu]).
        batch_size: Candidates per batch for memory control. None = no batching.
        profile: If True, return profiling metrics alongside results.
        show_progress: If True, display progress bar (requires tqdm).
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
            asks about equal-support supersets. GPU runs take the row-split
            path when this is on — the only one that implements the gates.
        use_generator_pruning: Infer support from (k-1)-subsets instead of counting.
            Pascal [Bastide et al. 2000]: a candidate with a non-free (k-1)-subset
            is itself non-free and its support is exactly the minimum of its
            (k-1)-subset supports, so it never has to be counted. Exact, no
            impact on results. CPU path only.
        sparse: Scipy CSR matrix usage. True = force, False = Polars, None = auto
            (switches at >100K k=2 candidates or >500 items <10% density).
        n_jobs: Parallel workers for sparse k>2 counting. 1=sequential, -1=all CPUs.
            Only active when sparse mode is used.
        enable_length_filter: Filter transactions shorter than k before counting
            k-itemset support. Set to False to disable.
        streaming: Use SON algorithm for chunked processing. Memory becomes
            O(chunk_size × n_items) instead of O(total × n_items).
        chunk_size: Transactions per chunk when streaming=True. Default 10M.
        n_gpus: GPUs for multi-GPU streaming (default 1). Requires CuPy.
        memory_budget_gb: Auto-calculate chunk_size to fit this budget.
            Overrides chunk_size. Only used when streaming=True.
        progress_callback: Streaming progress callback
            (phase, chunk_idx, n_chunks, metrics).
        level_callback: Per-level callback
            (k, n_candidates, n_frequent, duration_ms).

            **n_candidates is route-dependent and the routes do not agree.**
            The CPU path applies the full per-candidate subset test before
            counting, so it reports candidates that survived it. The GPU group
            path prunes at suffix granularity, which over-approximates, so it
            reports more candidates than the CPU path for the same input at the
            same level. Both are honest counts of what that route was about to
            count; neither is "the" candidate count. A consumer doing per-level
            cost accounting should treat the figure as comparable within a
            route and not across routes.

            n_frequent and duration_ms mean the same thing everywhere.
        bitvecs: Pre-built GPU bitvectors tuple (bitvecs_gpu, col_to_item, n_transactions).
            Skips DataFrame conversion; transactions must be None when provided.
            The array is READ-ONLY to the engine: ET-Miner writes only to
            bitvectors it builds itself, never to one it is handed, so the
            caller may reuse it after the call without copying it first.
        sparse_from_k: GPU paths only — when to switch support counting from
            dense bitvectors to sparse CSR tidsets. An int fixes the K-level;
            "auto" transitions when the previous level's measured mean support
            drops below n_transactions/32 (the point where tidsets become
            smaller than bitvectors); None (default) never switches.
        max_ram_gb / max_vram_gb: Memory guards for the single-GPU
            exhaustive mining loop only (checked between K-levels via RSS
            and the CuPy pool). The multi-GPU row-split path does not read
            them — it sizes candidate chunks from measured free VRAM and
            honors CuPy memory-pool limits instead.

    Returns:
        DataFrame with itemset (List[Int64]) and support (Float64) columns.

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
        >>> result = apriori(df, min_support=0.0001, sparse=True, n_jobs=-1)
        >>> result = apriori(huge_df, min_support=0.001, streaming=True, n_gpus=8)
    """
    _validate_parameters(min_support, max_length, batch_size, sparse_from_k)

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
        anchor_items=anchor_items,
        output_dir=output_dir,
        resume_from_k=resume_from_k,
        memory_budget_gb=memory_budget_gb,
    )

    # prune_equal_support is implemented by the row-split miner alone — both
    # gates live there. Every other GPU route accepted the flag and dropped it
    # on the floor, handing back the complete lattice while the caller believed
    # it had asked for the free-sets. Route those calls instead of ignoring
    # them; the row-split path runs on one GPU as happily as on many.
    _route_for_pruning = bool(prune_equal_support) and (use_gpu or bitvecs is not None)

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

        # Pruning requested → the only route that implements it.
        if _route_for_pruning:
            from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

            return _apriori_row_split_multi_gpu(
                None,
                col_to_item,
                n_trans,
                min_support,
                max_length,
                n_gpus,
                level_callback,
                bitvecs_list=[(bitvecs_gpu, int(bitvecs_gpu.device.id), n_trans)],
                prune_non_free=True,
                prune_apriori=True,
                sparse_from_k=sparse_from_k,
                # The sibling call below passes both; this one dropped them, so
                # output_dir produced no flush and resume_from_k silently
                # re-mined from K=1 on a multi-day campaign.
                output_dir=output_dir,
                resume_from_k=resume_from_k,
            )

        # Route to GPU-resident or standard bitvecs fast path
        if gpu_resident:
            from et_miner.gpu.mining import _apriori_from_bitvecs_gpu_resident

            return _apriori_from_bitvecs_gpu_resident(
                bitvecs_gpu,
                col_to_item,
                n_trans,
                min_support,
                max_length,
                profile,
                level_callback,
                n_gpus=n_gpus,
            )
        from et_miner.gpu.mining import _apriori_from_bitvecs

        return _apriori_from_bitvecs(
            bitvecs_gpu,
            col_to_item,
            n_trans,
            min_support,
            max_length,
            batch_size,
            profile,
            level_callback,
            n_gpus=n_gpus,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
            sparse_from_k=sparse_from_k,
        )

    # Validate transactions is provided when bitvecs is not
    if transactions is None:
        raise ValueError("Either 'transactions' or 'bitvecs' must be provided")

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
            sparse=sparse,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
        )

    lf = transactions.lazy() if isinstance(transactions, pl.DataFrame) else transactions

    # ── GPU direct path: transactions → CSR → GPU bitvecs ──────────────
    # Bypasses the dense boolean matrix entirely.
    # For 205M × 1006: ~5 GB CPU + ~25 GB GPU  instead of  206 GB CPU.
    if use_gpu:
        from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

        from .matrix import _build_csr_from_transactions

        csr_result = _build_csr_from_transactions(lf, min_support, item_col)
        if csr_result is None:
            if profile:
                return _empty_result(), ProfilingSession()
            return _empty_result()

        csr, idx_to_item, n_trans = csr_result

        # Row-split path: multiple GPUs, anchor filtering, or free-set pruning
        # (the single-GPU bitvec miner implements neither gate).
        if n_gpus > 1 or anchor_items is not None or _route_for_pruning:
            from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

            return _apriori_row_split_multi_gpu(
                csr,
                idx_to_item,
                n_trans,
                min_support,
                max_length,
                n_gpus,
                level_callback,
                output_dir=output_dir,
                resume_from_k=resume_from_k,
                prune_non_free=prune_equal_support,  # free-sets: emit == generate
                prune_apriori=prune_equal_support,  # Apriori subset pruning
                sparse_from_k=sparse_from_k,  # V3: density transition K-level
                anchor_items=anchor_items,  # V3 B6: two-phase anchor filtering
            )

        bitvecs_gpu = _build_gpu_bitvec_matrix(csr)
        del csr  # free CPU CSR memory

        if gpu_resident:
            from et_miner.gpu.mining import _apriori_from_bitvecs_gpu_resident

            return _apriori_from_bitvecs_gpu_resident(
                bitvecs_gpu,
                idx_to_item,
                n_trans,
                min_support,
                max_length,
                profile,
                level_callback,
                n_gpus=n_gpus,
            )
        from et_miner.gpu.mining import _apriori_from_bitvecs

        return _apriori_from_bitvecs(
            bitvecs_gpu,
            idx_to_item,
            n_trans,
            min_support,
            max_length,
            batch_size,
            profile,
            level_callback,
            n_gpus=n_gpus,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
            sparse_from_k=sparse_from_k,
        )

    # ── CPU path: dense boolean matrix ─────────────────────────────────
    session = ProfilingSession() if profile else None

    # Phase 1: Build boolean matrix (includes frequent 1-itemset mining)
    if session:
        session.start_phase("matrix_build")

    matrix, col_to_item, n_trans = build_boolean_matrix(lf, min_support, item_col)

    if session:
        session.end_phase(n_frequent_items=len(col_to_item), n_transactions=n_trans)

    # Compute max possible k from transaction lengths
    max_tx_length = (lf.select(pl.col(item_col).list.len().max()).collect(engine="streaming").item()) or 0

    effective_max_length = min(
        max_length if max_length else float("inf"),
        max_tx_length,
        len(col_to_item),  # Can't have more items than columns
    )

    if not col_to_item:
        if profile:
            return _empty_result(), session
        return _empty_result()

    item_cols = list(col_to_item.keys())

    # Phase 2: Get exact 1-itemset supports from matrix column sums
    _k1_start = time.perf_counter()
    if session:
        session.start_phase("k1_support")

    # 1-itemset counting: always use streaming engine (simple column sums, no GPU benefit)
    one_itemset_exprs = [pl.col(c).sum().alias(c) for c in item_cols]
    one_counts = matrix.lazy().select(one_itemset_exprs).collect(engine="streaming")

    results: list[tuple[list[int], float]] = []
    prev_frequent: list[tuple[str, ...]] = []

    # Filter on integer count to avoid float precision issues at boundaries.
    # math.ceil ensures we don't include items below threshold.
    min_count_threshold = _min_count(min_support, n_trans)
    for col in item_cols:
        count = one_counts.get_column(col).item()
        if count >= min_count_threshold:
            # Free-set semantics start at K=1: an item in EVERY transaction has
            # the empty set's support, so it is not a generator.
            if prune_equal_support and count == n_trans:
                continue
            support = count / n_trans
            results.append(([col_to_item[col]], support))
            prev_frequent.append((col,))

    if session:
        session.end_phase(n_frequent=len(prev_frequent))

    # Call level callback for k=1
    if level_callback:
        _k1_duration_ms = (time.perf_counter() - _k1_start) * 1000
        level_callback(1, len(item_cols), len(prev_frequent), _k1_duration_ms)

    if not prev_frequent:
        if profile:
            return _empty_result(), session
        return _empty_result()

    # Warn about potentially expensive computation
    if warn_complexity:
        _warn_complexity(len(prev_frequent), min_support)

    # The COMPLETE K=1 level (including any item pruned above as non-free):
    # every subset test of K=2 resolves against this, not against prev_frequent.
    prev_supports: dict[tuple[str, ...], float] = {
        (col,): one_counts.get_column(col).item() / n_trans
        for col in item_cols
        if one_counts.get_column(col).item() >= min_count_threshold
    }

    # Integer counts of the same complete level — what the free-set test uses.
    prev_counts: dict[tuple[str, ...], int] = {
        (col,): one_counts.get_column(col).item()
        for col in item_cols
        if one_counts.get_column(col).item() >= min_count_threshold
    }

    # The free (generator) 1-itemsets: everything frequent except items present
    # in every transaction, which carry the empty set's support. Tracked
    # independently of prune_equal_support because Pascal inference needs it.
    prev_free: set[tuple[str, ...]] | None = {
        itemset for itemset, count in prev_counts.items() if count < n_trans
    }

    # Phase 3: k >= 2 with vectorized support counting
    k = 2

    while k <= effective_max_length and len(prev_frequent) >= k:
        _k_start = time.perf_counter()

        # Generate candidates using Apriori join
        if session:
            session.start_phase(f"k{k}_candidate_gen")

        candidates = _generate_candidates(prev_frequent, k)

        if session:
            session.end_phase(n_candidates=len(candidates))

        if not candidates:
            break

        # Support inference + counting
        if session:
            session.start_phase(f"k{k}_support_count")

        # Phase A: Try to infer count from subsets (Pascal/generator pruning)
        inferred_counts: dict[tuple[str, ...], int] = {}
        needs_counting: list[tuple[str, ...]] = []

        if use_generator_pruning and prev_counts and k >= 2:
            for candidate in candidates:
                inferred_count = _infer_count_from_subsets(candidate, prev_counts, prev_free)
                if inferred_count is not None:
                    # Count can be inferred - no need to count!
                    inferred_counts[candidate] = inferred_count
                else:
                    needs_counting.append(candidate)
        else:
            needs_counting = candidates

        # Log inference results
        if use_generator_pruning and len(inferred_counts) > 0:
            skip_rate = 100 * len(inferred_counts) / len(candidates)
            logger.debug(
                f"[k={k}] Generator pruning: {len(inferred_counts)}/{len(candidates)} "
                f"candidates ({skip_rate:.1f}%) skipped via count inference"
            )

        # Phase B: Count support for remaining candidates
        if needs_counting:
            counted = count_support_batched(
                matrix,
                needs_counting,
                n_trans,
                batch_size,
                use_gpu,
                show_progress,
                sparse,
                n_jobs,
                enable_length_filter=enable_length_filter,
            )
        else:
            counted = {}

        # Combine inferred and counted (all are integer counts now)
        all_counts = {**inferred_counts, **counted}

        # Filter to frequent itemsets using integer count comparison.
        # This avoids float precision issues at boundaries.
        current_frequent: list[tuple[str, ...]] = []
        current_supports: dict[tuple[str, ...], float] = {}
        current_counts: dict[tuple[str, ...], int] = {}
        n_skipped = len(inferred_counts)

        for itemset in candidates:
            count = all_counts[itemset]

            if count >= min_count_threshold:
                current_frequent.append(itemset)
                current_supports[itemset] = count / n_trans
                current_counts[itemset] = count

        # Free-set (generator) pruning, resolved against the COMPLETE previous
        # level — prev_counts is never pruned, only prev_frequent is. What
        # survives is both what this level emits and what K+1 generates from:
        # emitting a level and then generating from a smaller one advertises
        # itemsets the run will never extend, which is how apriori-valid
        # itemsets went missing from K=5 on.
        if prune_equal_support and prev_counts:
            current_frequent = _prune_equal_support(current_frequent, current_counts, prev_counts)
            current_free = set(current_frequent)
        elif use_generator_pruning and prev_counts:
            # Not pruning, but Pascal needs to know which itemsets are free.
            current_free = set(_prune_equal_support(current_frequent, current_counts, prev_counts))
        else:
            current_free = None

        for itemset in current_frequent:
            results.append(([col_to_item[c] for c in itemset], current_supports[itemset]))

        if session:
            session.end_phase(
                n_frequent=len(current_frequent),
                n_inferred=n_skipped if use_generator_pruning else 0,
            )

        # Call level callback for k>=2
        if level_callback:
            _k_duration_ms = (time.perf_counter() - _k_start) * 1000
            level_callback(k, len(candidates), len(current_frequent), _k_duration_ms)

        if not current_frequent:
            break

        # The complete level feeds the next level's subset tests; the free
        # subset feeds its candidate generation.
        prev_supports = current_supports
        prev_counts = current_counts
        prev_free = current_free
        prev_frequent = current_frequent
        k += 1

    result_df = _build_result_df(results)
    if profile:
        return result_df, session
    return result_df


