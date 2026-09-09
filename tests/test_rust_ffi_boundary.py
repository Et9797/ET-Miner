"""The Rust FFI error contract.

`panic = "abort"` used to turn every panic in the extension into an
uncatchable SIGABRT. PyO3's error model depends on unwinding to produce a
PanicException, and CLAUDE.md's documented build is `maturin develop
--release`, so the shipped extension always aborted: five ordinary-looking
inputs exited 134 with no traceback, losing every unflushed level of a long
campaign.

It was never a consistent choice either. `lib.rs` implemented a non-contiguous
copy fallback five times and omitted it six times; it raised a proper
PyValueError for an unsorted `prev_flat` and then aborted one call later for a
length mismatch; and `gpu/mining.py`'s stale-wheel recovery path was dead code
under abort.

Two things are asserted here: that a bad input raises rather than aborting, and
that it raises something *useful* -- a ValueError naming the offending value,
not a PanicException from an index-out-of-bounds deep inside a kernel.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

rust = pytest.importorskip("et_miner_rust")


def _indptr(n_rows: int, per_row: int) -> np.ndarray:
    return np.arange(0, per_row * n_rows + 1, per_row, dtype=np.int64)


class TestBadShapesRaiseValueError:
    def test_n_cols_smaller_than_the_largest_column_index(self):
        with pytest.raises(ValueError, match=r"column index 5 is out of range for n_cols=1"):
            rust.build_column_bitvecs_u64(
                np.array([0, 2], dtype=np.int64), np.array([0, 5], dtype=np.int64), 1, 1
            )

    def test_n_rows_larger_than_the_indptr(self):
        with pytest.raises(ValueError, match=r"n_rows=99 requires csr_indptr of length >= 100"):
            rust.build_column_bitvecs_u64(
                np.array([0, 1, 2], dtype=np.int64), np.array([0, 1], dtype=np.int64), 99, 4
            )

    def test_empty_indptr(self):
        with pytest.raises(ValueError, match="at least one element"):
            rust.build_column_bitvecs_u64(
                np.array([], dtype=np.int64), np.array([], dtype=np.int64), 0, 4
            )

    def test_indptr_promising_more_nnz_than_indices_holds(self):
        with pytest.raises(ValueError, match="exceeds len"):
            rust.build_column_bitvecs_u64(
                np.array([0, 99], dtype=np.int64), np.array([0, 1], dtype=np.int64), 1, 4
            )

    def test_zero_rows_is_a_legitimate_degenerate_input(self):
        """build_boolean_matrix can produce an empty matrix. n_u64s is then 0 and
        `chunks(0)` panics, so this needs an explicit path rather than a guard."""
        out = rust.build_column_bitvecs_u64(
            np.array([0], dtype=np.int64), np.array([], dtype=np.int64), 0, 4
        )
        assert out.shape == (4, 0)


class TestNonContiguousInputIsCopiedNotRejected:
    """Half the entry points already copied a non-contiguous view; the other half
    called `as_slice().unwrap()` and panicked. One rule now.

    Each variant below builds a NON-CONTIGUOUS view whose logical contents are
    the same valid CSR -- rows of {0, 1}. That matters: an earlier version tiled
    [0, 1] and took `[::2]`, which yields rows of [0, 0], a duplicate-index CSR.
    It passed only because it compared a view against a copy of the same
    malformed input and both agreed; the row-monotonicity guard now rejects it.
    """

    @staticmethod
    def _strided(values: np.ndarray) -> np.ndarray:
        """Interleave a filler and take every other element."""
        padded = np.empty(values.size * 2, dtype=np.int64)
        padded[0::2] = values
        padded[1::2] = -999
        return padded[0::2]

    @staticmethod
    def _column_of_2d(values: np.ndarray) -> np.ndarray:
        """A column of a 2-D array: contiguous only if the array has one column."""
        arr = np.empty((values.size, 2), dtype=np.int64)
        arr[:, 0] = values
        arr[:, 1] = -999
        return arr[:, 0]

    @pytest.mark.parametrize("make", [_strided, _column_of_2d], ids=["strided", "column"])
    def test_a_view_gives_the_same_answer_as_its_copy(self, make):
        n_rows, per_row, n_cols = 8, 2, 6
        indptr = _indptr(n_rows, per_row)
        values = np.tile(np.array([0, 1], dtype=np.int64), n_rows)

        view = make(values)
        assert not view.flags["C_CONTIGUOUS"], "fixture must actually be non-contiguous"
        assert view.tolist() == values.tolist(), "the view must hold the valid CSR"

        from_view = rust.build_column_bitvecs_u64(indptr, view, n_rows, n_cols)
        from_copy = rust.build_column_bitvecs_u64(
            indptr, np.ascontiguousarray(view), n_rows, n_cols
        )
        assert np.array_equal(from_view, from_copy)


def test_a_panic_is_catchable_rather_than_a_process_abort():
    """The core of #5, in a subprocess because a live defect kills the runner.

    Note PanicException derives from **BaseException**, not Exception, so a
    plain `except Exception` will not catch it -- which matters for
    gpu/mining.py's recovery path.
    """
    script = textwrap.dedent(
        """
        import numpy as np, et_miner_rust as r
        try:
            # prefix_len far past any real lattice: an internal panic, not a
            # shape we validate at the boundary
            r.prune_non_free_flat(
                np.zeros((1, 2), dtype=np.int32), np.zeros(1, dtype=np.int64),
                np.zeros((1, 1), dtype=np.int32), np.zeros(99, dtype=np.int64))
        except BaseException as e:
            print("CAUGHT", type(e).__name__)
        else:
            print("NO_ERROR")
        """
    )
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, (
        f"process died with rc={p.returncode} "
        f"({'SIGABRT — panic=abort is back' if p.returncode in (-6, 134) else 'unexpected'}); "
        f"stderr: {p.stderr[-400:]}"
    )
    assert p.stdout.split()[0] in {"CAUGHT", "NO_ERROR"}


class TestValidateCsrCoversEveryMalformedShape:
    """Five holes the first version of `validate_csr` left open.

    Its docstring promises "a Python exception naming the offending value rather
    than an index-out-of-bounds panic from somewhere inside the kernel". For
    these shapes it did not deliver: four panicked, and one returned a silently
    wrong number.

    Catch with `BaseException`, not `Exception` -- `PanicException.__mro__` is
    `['PanicException', 'BaseException', 'object']`, so a `pytest.raises(Exception)`
    written for the pre-fix behaviour passes vacuously.
    """

    @staticmethod
    def _i(a):
        return np.array(a, dtype=np.int64)

    def test_n_rows_at_usize_max_does_not_wrap_the_guard(self):
        """`n_rows + 1` wrapped to 0 at usize::MAX -- release sets no
        overflow-checks -- so `0 > indptr.len()` was false and the guard was
        skipped entirely."""
        with pytest.raises(ValueError, match="requires csr_indptr of length"):
            rust.count_itemsets_sparse(self._i([0, 2]), self._i([0, 1]), 2**64 - 1, [[0]])

    @pytest.mark.parametrize(
        "indptr,indices,n_rows",
        [
            pytest.param([0, 5, 2], [0, 1], 2, id="indptr-jumps-then-drops"),
            pytest.param([0, 2, 1], [0, 1], 2, id="indptr-decreasing"),
            pytest.param([0, -1, 2], [0, 1], 2, id="indptr-negative-interior"),
        ],
    )
    def test_non_monotonic_indptr(self, indptr, indices, n_rows):
        """Only `indptr[n_rows]` was bounds-checked, so the kernels' per-row
        slice `indices[indptr[i]..indptr[i+1]]` panicked on rows 0..n_rows-1.
        A negative interior entry was never cast-checked and wrapped."""
        with pytest.raises(ValueError, match="non-decreasing"):
            rust.count_itemsets_sparse(self._i(indptr), self._i(indices), n_rows, [[0]])

    def test_the_same_shape_is_rejected_by_every_guarded_entry_point(self):
        """It panicked on all four, not just the one it was first found on."""
        bad = (self._i([0, 5, 2]), self._i([0, 1]), 2)
        with pytest.raises(ValueError):
            rust.count_itemsets_sparse(*bad, [[0]])
        with pytest.raises(ValueError):
            rust.count_itemsets_simd(bad[0], bad[1], bad[2], 2, [[0]])
        with pytest.raises(ValueError):
            rust.build_column_bitvecs_u64(bad[0], bad[1], bad[2], 2)
        with pytest.raises(ValueError):
            rust.apriori_from_csr(bad[0], bad[1], bad[2], 2, 0.5, 3)

    def test_a_negative_index_beside_a_positive_one(self):
        """A logic bug rather than a gap: the guard tested `max_idx < 0` over the
        whole array's MAXIMUM, so it rejected only when the largest index was
        negative. `[-1, 1]` with n_cols=2 cleared it (max = 1 < 2) and panicked
        in-kernel."""
        with pytest.raises(ValueError, match="negative"):
            rust.count_itemsets_simd(self._i([0, 2]), self._i([-1, 1]), 1, 2, [[0]])
        # control: an all-negative row was already caught
        with pytest.raises(ValueError):
            rust.count_itemsets_simd(self._i([0, 2]), self._i([-1, -2]), 1, 2, [[0]])

    def test_an_unsorted_row_is_rejected_rather_than_undercounted(self):
        """The only hole that produced a WRONG NUMBER instead of a panic.

        `count_itemsets_sparse_raw` binary-searches each row -- its own comment
        says "the indices are sorted, so we can use binary search" -- and nothing
        validated it. Three rows all equal to {0,1,2} with row 1 stored
        descending gave sparse=2 against a truth of 3, a 33% undercount, while
        the SIMD path returned 3. Which number a caller got was decided by
        `hasattr(rust, "count_itemsets_simd")`, so Tier 2 of the mandated chain
        disagreed with itself depending on how the wheel was built.
        """
        indptr = self._i([0, 3, 6, 9])
        indices = self._i([0, 1, 2, 2, 1, 0, 0, 1, 2])  # row 1 descending
        for itemset in ([[0, 1]], [[1, 2]], [[0, 1, 2]]):
            with pytest.raises(ValueError, match="strictly increasing"):
                rust.count_itemsets_sparse(indptr, indices, 3, itemset)

    def test_a_valid_csr_is_unaffected(self):
        """The guard must not reject well-formed input; scipy CSR is monotonic
        and row-sorted by construction, which is why this was reachable only
        through direct FFI use."""
        got = rust.count_itemsets_sparse(
            self._i([0, 2, 4]), self._i([0, 1, 0, 1]), 2, [[0, 1]]
        )
        assert int(got[0]) == 2

    def test_an_empty_row_is_allowed(self):
        """indptr[i] == indptr[i+1] is a legitimate empty transaction."""
        got = rust.count_itemsets_sparse(
            self._i([0, 2, 2, 4]), self._i([0, 1, 0, 1]), 3, [[0, 1]]
        )
        assert int(got[0]) == 2
