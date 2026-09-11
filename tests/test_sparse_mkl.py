"""MKL setup and the sparse matmul's numeric range.

These paths are only reachable when `sparse_dot_mkl` imports, and the same
function is exact or catastrophically wrong depending on whether that optional
import resolves -- so the tests that need MKL skip rather than pass vacuously
when it is absent. Absent means not installed: `mkl` is a declared dependency,
so an installed-but-unloadable MKL is a defect, and `importorskip` is left
strict about it (an `ImportError` that is not a `ModuleNotFoundError` fails
collection here rather than skipping). That is how the CI runner, which has no
system MKL, showed that `_setup_mkl_library_path` was not making the venv's
copy loadable: the import is what `TestThePinnedMklIsTheOneLoaded` pins.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from et_miner.core import sparse as sp

mkl = pytest.importorskip("sparse_dot_mkl", reason="MKL branch unreachable without sparse_dot_mkl")


def _two_full_columns(n_rows: int) -> csr_matrix:
    """Both columns set in every row, so the true pair count is exactly n_rows."""
    indptr = np.arange(0, 2 * n_rows + 1, 2, dtype=np.int32)
    indices = np.tile(np.array([0, 1], dtype=np.int32), n_rows)
    data = np.ones(2 * n_rows, dtype=np.int32)
    return csr_matrix((data, indices, indptr), shape=(n_rows, 2))


class TestSparseMatmulRange:
    @pytest.mark.slow
    def test_counts_above_2_pow_24_are_exact(self):
        """#4 -- float32 has a 24-bit significand, so an accumulator sticks at
        2**24 = 16,777,216 and `acc + 1` rounds back to `acc`. The level filter
        is `count >= min_count`, so the resulting UNDER-count silently drops
        genuinely frequent pairs and the loss cascades into every higher K.

        Measured before the fix: 20,000,000 -> 16,777,216, a 16.1% error.
        """
        n_rows = 20_000_000
        m = _two_full_columns(n_rows)
        got = int(sp._sparse_matmul(m.T.tocsr(), m)[0, 1])
        assert got == n_rows, f"saturated at {got:,} (float32 sticks at {2**24:,})"

    def test_the_two_branches_of_one_function_agree(self):
        """The MKL branch and the scipy fallback must not disagree -- before the
        fix they did, at scale, depending purely on an optional import."""
        m = _two_full_columns(100_000)
        mkl_result = int(sp._sparse_matmul(m.T.tocsr(), m)[0, 1])
        scipy_result = int((m.T.tocsr() @ m)[0, 1])
        assert mkl_result == scipy_result == 100_000

    def test_integer_input_is_widened_to_float64_not_float32(self):
        """The mechanism, not just the outcome: float32 would be a silent
        regression that this fixture is too small to expose."""
        m = _two_full_columns(1000)
        out = sp._sparse_matmul(m.T.tocsr(), m)
        assert out.dtype == np.float64, f"expected float64 accumulation, got {out.dtype}"


class TestMklThreadsAreNotHijacked:
    def test_configure_and_restore_round_trips(self):
        """#16 -- _restore_mkl_threads used to re-derive a count from the
        environment rather than restoring the one it displaced, so the `finally`
        blocks that call it did not actually restore anything."""
        from sparse_dot_mkl import mkl_get_max_threads, mkl_set_num_threads

        original = mkl_get_max_threads()
        try:
            mkl_set_num_threads(2)
            sp._saved_mkl_threads = None  # forget any earlier capture
            before = mkl_get_max_threads()
            sp._configure_mkl_for_parallel(8)
            sp._restore_mkl_threads()
            assert mkl_get_max_threads() == before
        finally:
            mkl_set_num_threads(original)
            sp._saved_mkl_threads = None

    def test_import_does_not_configure_threads(self):
        """The module is imported lazily from inside count_support_batched, so a
        module-scope mkl_set_num_threads() fired mid-run and silently overrode a
        host application's own configuration (measured: 2 became 24)."""
        src = (
            "import sys; "
            "from sparse_dot_mkl import mkl_set_num_threads, mkl_get_max_threads; "
            "mkl_set_num_threads(2); "
            "before = mkl_get_max_threads(); "
            "import et_miner.core.sparse; "
            "print(before, mkl_get_max_threads())"
        )
        import subprocess

        out = subprocess.run(
            [os.sys.executable, "-c", src], capture_output=True, text=True, timeout=180
        )
        assert out.returncode == 0, out.stderr[-800:]
        before, after = (int(x) for x in out.stdout.strip().split()[-2:])
        assert after == before, f"importing core.sparse changed MKL threads {before} -> {after}"


class TestMklPathDiscovery:
    @staticmethod
    def _capture(fn):
        """Collect loguru output -- it does not route through pytest's caplog."""
        from loguru import logger

        lines: list[str] = []
        sink = logger.add(lines.append, level="DEBUG", format="{message}")
        try:
            fn()
        finally:
            logger.remove(sink)
        return "".join(lines)

    def test_a_miss_is_logged_rather_than_silent(self, monkeypatch, tmp_path):
        """#17 -- the probe tested one filename at one location and returned at
        :47 with NO log line, so nothing distinguished "the path was already
        fine" from "the probe found nothing". The numerics then ran against
        whichever unpinned system MKL happened to load, which is exactly what
        #4's accuracy depends on."""
        empty = tmp_path / "no-mkl-here"
        empty.mkdir()
        monkeypatch.setattr(sp, "_mkl_search_dirs", lambda: [str(empty)])
        monkeypatch.setenv("LD_LIBRARY_PATH", "")
        monkeypatch.delenv("MKL_RT", raising=False)
        out = self._capture(sp._setup_mkl_library_path)
        assert "no libmkl_rt.so*" in out, f"a miss must say so; got {out!r}"
        assert "MKL_RT" not in os.environ, "a miss must not invent an MKL_RT"

    def test_a_hit_is_logged_sets_mkl_rt_and_extends_the_path(self, monkeypatch, tmp_path):
        """And the .so.3 case the old probe could not see: it hardcoded
        libmkl_rt.so.2, while a venv may ship .so.3. The highest version is the
        one `MKL_RT` names when a directory holds more than one, and the
        unversioned symlink ranks below every versioned file."""
        libdir = tmp_path / "lib"
        libdir.mkdir()
        (libdir / "libmkl_rt.so.2").write_bytes(b"")
        (libdir / "libmkl_rt.so.3").write_bytes(b"")
        (libdir / "libmkl_rt.so").write_bytes(b"")
        monkeypatch.setattr(sp, "_mkl_search_dirs", lambda: [str(libdir)])
        monkeypatch.setenv("LD_LIBRARY_PATH", "/pre/existing")
        monkeypatch.delenv("MKL_RT", raising=False)
        out = self._capture(sp._setup_mkl_library_path)
        assert "libmkl_rt.so.3" in out
        assert os.environ["MKL_RT"] == str(libdir / "libmkl_rt.so.3")
        assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep) == [str(libdir), "/pre/existing"]

    def test_a_preset_mkl_rt_is_kept(self, monkeypatch, tmp_path):
        """`MKL_RT` is a default, not an override: a caller who pinned a
        different MKL keeps it, and the log says so."""
        libdir = tmp_path / "lib"
        libdir.mkdir()
        (libdir / "libmkl_rt.so.3").write_bytes(b"")
        monkeypatch.setattr(sp, "_mkl_search_dirs", lambda: [str(libdir)])
        monkeypatch.setenv("LD_LIBRARY_PATH", "")
        monkeypatch.setenv("MKL_RT", "/their/own/libmkl_rt.so.2")
        out = self._capture(sp._setup_mkl_library_path)
        assert os.environ["MKL_RT"] == "/their/own/libmkl_rt.so.2"
        assert "preset" in out and "/their/own/libmkl_rt.so.2" in out

    def test_an_already_present_dir_is_not_added_twice(self, monkeypatch, tmp_path):
        libdir = tmp_path / "lib"
        libdir.mkdir()
        (libdir / "libmkl_rt.so.2").write_bytes(b"")
        monkeypatch.setattr(sp, "_mkl_search_dirs", lambda: [str(libdir)])
        monkeypatch.setenv("LD_LIBRARY_PATH", str(libdir))
        monkeypatch.delenv("MKL_RT", raising=False)
        out = self._capture(sp._setup_mkl_library_path)
        assert "already on LD_LIBRARY_PATH" in out
        assert os.environ["MKL_RT"] == str(libdir / "libmkl_rt.so.2"), "MKL_RT is set on this branch too"
        assert os.environ["LD_LIBRARY_PATH"] == str(libdir)

    def test_search_dirs_are_real_existing_directories(self):
        dirs = sp._mkl_search_dirs()
        assert all(os.path.isdir(d) for d in dirs)
        assert len(dirs) == len(set(dirs)), "search dirs must be deduplicated"


class TestThePinnedMklIsTheOneLoaded:
    """The reason `MKL_RT` is set, measured rather than reasoned. The dynamic
    loader reads LD_LIBRARY_PATH once at process start, so extending it from
    inside the interpreter -- all `_setup_mkl_library_path` did before -- left
    sparse_dot_mkl to find whatever MKL the loader could see: on the dev box,
    conda's `/opt/conda/lib/libmkl_rt.so.2` rather than the venv's; on the CI
    runner, nothing (`mkl_rt not found` at collection). Both read off
    `/proc/self/maps`, which is the object the process mapped and not the path
    a variable names."""

    @staticmethod
    def _mapped_libmkl_rt(maps_text: str) -> set[str]:
        return {os.path.realpath(line.split()[-1]) for line in maps_text.splitlines() if "libmkl_rt" in line}

    def test_this_process_mapped_the_copy_mkl_rt_names(self):
        if not os.path.exists("/proc/self/maps"):
            pytest.skip("needs /proc/self/maps")
        assert "MKL_RT" in os.environ, "importing core.sparse before sparse_dot_mkl must have set it"
        want = os.path.realpath(os.environ["MKL_RT"])
        assert self._mapped_libmkl_rt(open("/proc/self/maps").read()) == {want}

    def test_a_fresh_interpreter_with_no_loader_path_loads_the_venv_copy(self):
        """The CI shape: no LD_LIBRARY_PATH, no MKL_RT, and the import order the
        package relies on (core.sparse first). The copy that loads must be the
        one under `sys.prefix` -- the pinned dependency -- and the only one."""
        if not os.path.exists("/proc/self/maps"):
            pytest.skip("needs /proc/self/maps")
        src = (
            "import os; import et_miner.core.sparse; import sparse_dot_mkl; "
            "print(os.environ['MKL_RT']); print(open('/proc/self/maps').read())"
        )
        env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "MKL_RT")}
        out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=180, env=env)
        assert out.returncode == 0, out.stderr[-800:]
        mkl_rt, _, maps = out.stdout.partition("\n")
        want = os.path.realpath(mkl_rt.strip())
        assert want.startswith(os.path.realpath(sys.prefix) + os.sep), f"MKL_RT points outside the venv: {want}"
        assert self._mapped_libmkl_rt(maps) == {want}

class TestNJobsIsHonouredOnTheRustPath:
    def test_thread_budget_changes_wall_time_and_not_results(self):
        """#18 -- _should_use_rust ignored both its arguments and the Rust path
        was taken before n_jobs was ever resolved, so rayon took every core
        regardless. A caller who set n_jobs=1 -- documented as "sequential
        execution" -- was silently oversubscribed, which on a shared box is the
        difference between a bounded job and one that takes the machine."""
        rust = pytest.importorskip("et_miner_rust")
        if rust.get_num_threads() < 2:
            pytest.skip("needs a multi-thread rayon pool to observe a budget")

        import time

        rng = np.random.default_rng(0)
        n_rows, n_cols = 200_000, 200
        # sorted(): a CSR row must be strictly increasing, which the Rust
        # boundary now validates. rng.choice returns them unsorted, so this
        # fixture was building a malformed CSR and the binary-search path
        # silently undercounted on it.
        rows = [np.sort(rng.choice(n_cols, size=12, replace=False)) for _ in range(n_rows)]
        indptr = np.arange(0, 12 * n_rows + 1, 12, dtype=np.int64)
        indices = np.concatenate(rows).astype(np.int64)
        itemsets = [
            [a, b, c] for a in range(30) for b in range(a + 1, 32) for c in range(b + 1, 34)
        ][:3000]

        def run(n_threads):
            t = time.perf_counter()
            out = rust.count_itemsets_simd(indptr, indices, n_rows, n_cols, itemsets, n_threads)
            return time.perf_counter() - t, int(out.sum())

        t_unbounded, sum_unbounded = run(0)
        t_one, sum_one = run(1)

        assert sum_one == sum_unbounded, "a thread budget must not change the counts"
        assert t_one > t_unbounded * 1.5, (
            f"n_threads=1 took {t_one:.2f}s vs {t_unbounded:.2f}s unbounded -- "
            "the budget does not appear to be applied"
        )

    def test_rust_min_itemsets_fossil_is_gone(self):
        """The constant that used to gate this was declared and never read."""
        assert not hasattr(sp, "_RUST_MIN_ITEMSETS")
