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
    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(lambda a: a[::2], id="strided"),
            pytest.param(lambda a: np.asfortranarray(a.reshape(-1, 1))[:, 0], id="f_order"),
        ],
    )
    def test_a_view_gives_the_same_answer_as_its_copy(self, make):
        """Half the entry points already copied; the other half panicked. One rule now."""
        n_rows, per_row, n_cols = 8, 2, 6
        indptr = _indptr(n_rows, per_row)
        base = np.tile(np.array([0, 1], dtype=np.int64), n_rows * 2)
        view = make(base)[: per_row * n_rows]
        assert not view.flags["C_CONTIGUOUS"] or view.base is not None

        from_view = rust.build_column_bitvecs_u64(indptr, view, n_rows, n_cols)
        from_copy = rust.build_column_bitvecs_u64(indptr, np.ascontiguousarray(view), n_rows, n_cols)
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
