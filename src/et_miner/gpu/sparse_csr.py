"""GPU-resident, row-split sparse-CSR levels — the dense→sparse transition target.

Each GPU keeps a :class:`CsrShard`: the sorted-unique **shard-local**
transaction ids of every current frequent itemset, in the same row order as
``prev_frequent_flat``. It is built on-device from the GPU's own bitvec shard
at the transition and rebuilt on-device for the survivors of every later
level, so the sparse path keeps the dense path's memory model (each GPU
holds its own rows, nothing is replicated or re-uploaded per level).

Per level: candidates are enumerated in-kernel from the resident group
arrays (``kernels/csr_warp.py``), per-GPU int32 partial counts go through
the same reduce + compact filter as the dense chunk loop
(``row_split_chunks.run_chunked_dense_level``), and the survivors' tidsets
are materialized by a second kernel pass in survivor order — row *i* of the
new shard ≡ row *i* of the survivor table on every GPU, by construction.
Single-GPU mining is the one-shard case (no reduce).

Two exact checks guard the row alignment on every run and raise instead of
warning: at the transition the per-shard row lengths must sum to the dense
counts of the previous level, and at every materialization the per-shard
survivor lengths must sum to the survivor counts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from loguru import logger

from et_miner.gpu.kernels.csr_warp import count_csr_gather, count_csr_range, write_csr_gather
from et_miner.gpu.kernels.k3plus import upload_k3plus_groups
from et_miner.gpu.row_split_chunks import (
    MARGIN_FLOOR_BYTES,
    MARGIN_VRAM_FRACTION,
    _device_available_bytes,
    compute_chunk_budget,
    plan_candidate_chunks,
    run_chunked_dense_level,
)


@dataclass
class CsrShard:
    """Device-resident CSR of shard-local tids, one row per current itemset."""

    device_id: int
    n_rows_local: int  # transactions in this shard — every tid is < this
    offsets: object  # cupy int64 (n_itemsets + 1)
    indices: object  # cupy int32 (nnz,), strictly increasing within a row

    @property
    def n_itemsets(self) -> int:
        return int(self.offsets.shape[0]) - 1

    @property
    def nnz(self) -> int:
        return int(self.indices.shape[0])

    @property
    def nbytes(self) -> int:
        return 8 * int(self.offsets.shape[0]) + 4 * self.nnz

    def free(self) -> None:
        import cupy as cp

        with cp.cuda.Device(self.device_id):
            self.offsets = None
            self.indices = None
            cp.get_default_memory_pool().free_all_blocks()


@dataclass
class SparseMiningState:
    """Sticky sparse-mode state of one mining run (replaces the host CSR)."""

    shards: list[CsrShard] | None = None

    @property
    def active(self) -> bool:
        return self.shards is not None

    @property
    def device_ids(self) -> list[int]:
        return [s.device_id for s in (self.shards or [])]

    def replace(self, new_shards: list[CsrShard] | None) -> None:
        old = self.shards
        self.shards = new_shards
        for s in old or []:
            s.free()

    def release(self) -> None:
        self.replace(None)


def _group_bytes(groups_info) -> int:
    """Device bytes of the uploaded group arrays (with suffix_src_rows and ctp)."""
    gsr = getattr(groups_info, "suffix_src_rows", None)
    return (
        4 * len(groups_info.prefix_items)
        + 8 * len(groups_info.prefix_offsets)
        + 4 * len(groups_info.suffixes)
        + 8 * len(groups_info.suffix_offsets)
        + 8 * len(groups_info.cumulative_pairs)
        + 8 * len(groups_info.suffix_offsets)  # ctp
        + (8 * len(gsr) if gsr is not None else 0)
    )


def convert_shards_to_csr(bitvecs_list, prev_frequent_flat, prev_counts_flat=None, *, batch_cap: int = 10_000):
    """Build one CsrShard per GPU from its bitvec shard, entirely on-device.

    Args:
        bitvecs_list: ``[(bitvec_gpu uint64 (n_cols, n_u64s_local), device_id, n_rows_local), ...]``.
        prev_frequent_flat: numpy int32 ``(n_freq, k)`` — the itemsets whose
            tidsets are wanted, in the row order the shards must keep.
        prev_counts_flat: numpy int64 ``(n_freq,)`` dense counts of the same
            rows, or None (resume without counts). When given, the per-shard
            row lengths must sum to it exactly — a RuntimeError otherwise.
        batch_cap: upper bound on itemsets per AND batch (VRAM-bounded below).

    Returns:
        ``[CsrShard, ...]`` in ``bitvecs_list`` order.
    """
    import cupy as cp

    from et_miner.gpu.kernels import get_cuda_kernel, get_popcount_kernel

    prev_frequent_flat = np.ascontiguousarray(prev_frequent_flat, dtype=np.int32)
    n_freq, k = prev_frequent_flat.shape

    def _one(bv, did, n_rows_local):
        with cp.cuda.Device(did):
            n_u64s = int(bv.shape[1])
            pool = cp.get_default_memory_pool()
            free_b, _ = _device_available_bytes(did)
            # AND temporaries: the accumulator plus the gathered column rows.
            batch = max(100, min(int(batch_cap), int(free_b * 0.4 // max(1, 2 * n_u64s * 8))))
            popcount = get_popcount_kernel()
            extract = get_cuda_kernel("bitvec_extract_tids")
            count_parts, idx_parts = [], []
            for b0 in range(0, n_freq, batch):
                b1 = min(b0 + batch, n_freq)
                nb = b1 - b0
                batch_flat = prev_frequent_flat[b0:b1]
                acc = cp.full((nb, n_u64s), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=cp.uint64)
                for col in range(k):
                    acc &= bv[batch_flat[:, col].astype(np.int64)]
                row_counts = popcount(acc.ravel().view(cp.uint64)).reshape(nb, n_u64s).sum(axis=1, dtype=cp.int64)
                boff = cp.zeros(nb + 1, dtype=cp.int64)
                cp.cumsum(row_counts, out=boff[1:])
                total = int(boff[-1])
                bidx = cp.empty(total, dtype=cp.int32)
                if total > 0:
                    extract(
                        ((nb + 255) // 256,),
                        (256,),
                        (acc, boff, bidx, np.int64(nb), np.int64(n_u64s), np.int64(0)),
                    )
                count_parts.append(row_counts)
                idx_parts.append(bidx)
                del acc, boff
            row_counts_all = cp.concatenate(count_parts) if count_parts else cp.zeros(0, dtype=cp.int64)
            offsets = cp.zeros(n_freq + 1, dtype=cp.int64)
            if n_freq:
                cp.cumsum(row_counts_all, out=offsets[1:])
            indices = cp.concatenate(idx_parts) if idx_parts else cp.zeros(0, dtype=cp.int32)
            cp.cuda.Device(did).synchronize()
            host_counts = row_counts_all.get()
            del count_parts, idx_parts, row_counts_all
            pool.free_all_blocks()
            return CsrShard(int(did), int(n_rows_local), offsets, indices), host_counts

    with ThreadPoolExecutor(max_workers=len(bitvecs_list)) as pool:
        results = list(pool.map(lambda t: _one(*t), bitvecs_list))
    shards = [r[0] for r in results]

    if prev_counts_flat is not None and n_freq:
        total = np.zeros(n_freq, dtype=np.int64)
        for _, hc in results:
            total += hc.astype(np.int64)
        expected = np.asarray(prev_counts_flat, dtype=np.int64)
        bad = np.nonzero(total != expected)[0]
        if len(bad):
            for s in shards:
                s.free()
            i = int(bad[0])
            raise RuntimeError(
                f"dense→sparse transition: tidset lengths disagree with dense counts for {len(bad):,} of "
                f"{n_freq:,} itemsets (row {i}: {int(total[i])} vs {int(expected[i])})"
            )
    nnz = sum(s.nnz for s in shards)
    mb = sum(s.nbytes for s in shards) / 1024**2
    logger.info(
        f"    Dense→Sparse: {n_freq:,} itemsets, avg support {nnz / n_freq if n_freq else 0:.0f}, "
        f"{mb:.1f} MB tidset across {len(shards)} shard(s), row lengths verified exactly"
    )
    return shards


def upload_groups_to_shards(groups_info, shards) -> dict[int, dict]:
    """Resident group arrays (with suffix_src_rows) per device: {device_id: dict}."""
    return {s.device_id: upload_k3plus_groups(groups_info, s.device_id, with_src_rows=True) for s in shards}


def free_groups(groups_gpu: dict[int, dict] | None) -> None:
    import cupy as cp

    for did, g in (groups_gpu or {}).items():
        with cp.cuda.Device(did):
            g.clear()
            cp.get_default_memory_pool().free_all_blocks()


def run_sparse_level(shards, groups_info, groups_gpu, min_count, *, nccl_comms, use_nccl, level_label=""):
    """Count every candidate on every shard, reduce, filter: ``(surv, counts)``.

    ``surv`` are ascending int64 candidate indices into the level's candidate
    space and ``counts`` their exact global counts (int64), both on host —
    the same contract as the dense chunk loop.
    """
    device_ids = [s.device_id for s in shards]
    tc = int(groups_info.total_candidates)
    # The shard is already resident (measured free VRAM excludes it); reserving
    # it again is deliberate headroom for materialization, when old and new
    # shards coexist.
    budget = compute_chunk_budget(
        device_ids,
        group_data_bytes=_group_bytes(groups_info) + max(s.nbytes for s in shards),
        use_nccl=use_nccl,
    )
    chunks = plan_candidate_chunks(tc, budget)
    if len(chunks) > 1:
        logger.debug(f"    {level_label} CSR: {tc:,} candidates in {len(chunks)} chunks (budget {budget:,})")

    def launch(shard, did, chunk):
        import cupy as cp

        with cp.cuda.Device(did):
            return count_csr_range(shard.offsets, shard.indices, groups_gpu[did], chunk.start, chunk.size)

    pseudo_bitvecs = [(s, s.device_id, s.n_rows_local) for s in shards]
    return run_chunked_dense_level(
        pseudo_bitvecs, chunks, launch, min_count, nccl_comms, use_nccl, level_label=f"{level_label} CSR"
    )


def _preflight(shards, need_bytes: int, n_surv: int, level_label: str) -> None:
    for s in shards:
        free_b, total_b = _device_available_bytes(s.device_id)
        margin = min(max(MARGIN_FLOOR_BYTES, int(total_b * MARGIN_VRAM_FRACTION)), max(0, free_b) // 4)
        if need_bytes + margin > free_b:
            raise RuntimeError(
                f"{level_label}: materializing {n_surv:,} survivors' tidsets needs up to "
                f"{need_bytes / 1e9:.2f} GB on GPU {s.device_id} but only {free_b / 1e9:.2f} GB is available "
                f"beside the current shard ({s.nbytes / 1e9:.2f} GB) and the group arrays — old and new "
                f"shards must coexist for the write. Lower max_length, raise min_support, or add GPUs."
            )


def materialize_survivors(shards, groups_gpu, surv, expected_counts, *, level_label=""):
    """New shards holding the survivors' tidsets, in ``surv`` order, on every GPU.

    ``expected_counts`` are the survivors' global counts; the per-shard
    lengths written must sum to them exactly (RuntimeError otherwise — the
    kernel/host candidate decode would have drifted).
    """
    import cupy as cp

    surv = np.ascontiguousarray(surv, dtype=np.int64)
    expected = np.asarray(expected_counts, dtype=np.int64)
    n = int(len(surv))
    if len(expected) != n:
        raise ValueError(f"{level_label}: {n} survivors but {len(expected)} counts")
    # Exact upper bound: a shard's nnz is at most the global count sum.
    _preflight(shards, 4 * int(expected.sum()) + 8 * (n + 1), n, level_label)

    def _one(shard):
        did = shard.device_id
        with cp.cuda.Device(did):
            ids = cp.asarray(surv)
            cnt = count_csr_gather(shard.offsets, shard.indices, groups_gpu[did], ids)
            new_off = cp.zeros(n + 1, dtype=cp.int64)
            if n:
                cp.cumsum(cnt.astype(cp.int64), out=new_off[1:])
            nnz = int(new_off[-1]) if n else 0
            free_b, _ = _device_available_bytes(did)
            if 4 * nnz > free_b:
                raise RuntimeError(
                    f"{level_label}: GPU {did} needs {4 * nnz / 1e9:.2f} GB for the new shard "
                    f"but only {free_b / 1e9:.2f} GB is free"
                )
            new_idx = cp.empty(nnz, dtype=cp.int32)
            if n and nnz:
                write_csr_gather(shard.offsets, shard.indices, groups_gpu[did], ids, new_off, new_idx)
            cp.cuda.Device(did).synchronize()
            host_cnt = cnt.get()
            del cnt, ids
            return CsrShard(did, shard.n_rows_local, new_off, new_idx), host_cnt

    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        results = list(pool.map(_one, shards))
    new_shards = [r[0] for r in results]

    total = np.zeros(n, dtype=np.int64)
    for _, hc in results:
        total += hc.astype(np.int64)
    bad = np.nonzero(total != expected)[0]
    if len(bad):
        for s in new_shards:
            s.free()
        i = int(bad[0])
        raise RuntimeError(
            f"{level_label}: survivor tidset lengths disagree with counts for {len(bad):,} of {n:,} survivors "
            f"(survivor {i}: {int(total[i])} vs {int(expected[i])}) — kernel/host candidate decode drift"
        )
    return new_shards


def log_new_shards(shards, n_freq: int) -> None:
    nnz = sum(s.nnz for s in shards)
    mb = sum(s.nbytes for s in shards) / 1024**2
    logger.debug(
        f"    New tidsets: {n_freq:,} itemsets, avg support {nnz / n_freq if n_freq else 0:.0f}, "
        f"{mb:.1f} MB across {len(shards)} shard(s)"
    )
