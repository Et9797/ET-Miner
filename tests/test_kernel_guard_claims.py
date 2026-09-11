"""What is CLAIMED about the gpu-resident input guards, checked without a device.

`test_kernel_input_guards.py` exercises the guards on a card: it builds inputs
on device 0 and asserts each entry point rejects the wrong rank or dtype. This
file checks the sentences written around those guards -- which entry points are
guarded, who calls `_assert_home`, where `build_prefix_groups_gpu` is reached
unchecked, whether the K=1 seed cast is still in `gpu/mining.py`. Every test
here reads a tuple, a docstring or a module's source. None needs cupy.

That is why it is a separate file. These tests used to sit in
`test_kernel_input_guards.py` under its `pytestmark = pytest.mark.gpu`, its
module-level `importorskip("cupy")` and its device skip, beneath a comment
saying they "run on a box with no device". Measured: `CUDA_VISIBLE_DEVICES=""
pytest -q tests/test_kernel_input_guards.py` collected that file as ONE skip
and all seven were hidden. Here they carry no marker and no gate.
`test_the_modules_under_test_import_without_cupy` pins the premise -- the three
modules read below import with cupy blocked -- so the premise stays measured
rather than remembered.

Two families of claim:

1. THE SET CLAIMS. `gpu_resident.py`'s module docstring and
   `loader.py::_assert_home`'s both quantify over "the entry points that guard
   their inputs". Successive rewordings of that sentence were each false about
   `build_prefix_groups_gpu`, which the package exports and which calls no
   guard. `_GUARDED_ENTRY_POINTS` and `_EXEMPT_ENTRY_POINTS` make the tuples
   the claim; the tests hold the tuples to the source in both directions, so a
   new export cannot join the family by being described as part of it. Call
   sites are read off the AST (`_call_sites`), so a docstring that mentions
   `_assert_home(` is not a caller and two calls on one line are two.

2. THE K=1 SEED CAST. `cp.where(freq_mask)[0]` in `gpu/mining.py` returns
   int64 natively and is `.astype(cp.int32)`'d on the spot to build the K=1
   seed. WHAT CATCHES A DROPPED `.astype`, measured rather than reasoned:
   deleting it inside `_apriori_from_bitvecs_gpu_resident` fails
   `tests/test_gpu_resident_e2e.py::test_gpu_resident_matches_cpu_reference`,
   where the `_assert_dtype` guard on `count_pairs_fused_k2_gpu_resident`
   raises on the seed -- the guard working as designed, since an end-to-end
   run is what supplies it a real seed. The BEHAVIOURAL backstop is that e2e
   test. `test_the_k1_seed_cast_is_still_in_mining` is a SOURCE check: it
   reads the module's own source, so a reader of this file is not told a
   cast is checked here when it is not. An earlier version rebuilt the
   expression locally, never imported `gpu.mining`, and stayed green under
   the mutation.

TALLY UNDER THAT MUTATION, re-measured after the split: this file goes
1 failed, 7 passed; `test_kernel_input_guards.py` stays 8 passed, which is the
point of its docstring's pointer here -- nothing in the device file sees the
cast, and it says so.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


_GUARD_CALLS = ("_assert_home", "_assert_rank", "_assert_dtype")


def test_the_modules_under_test_import_without_cupy():
    """The premise of this file, held in a fresh interpreter: `None` in
    `sys.modules` makes `import cupy` raise, and the three modules read below
    must import anyway. If one grows a module-level cupy import, every test
    here would start needing a device and the file's reason to exist is gone --
    fail here, loudly, rather than hide seven tests behind a skip again."""
    code = (
        "import sys; sys.modules['cupy'] = None\n"
        "import et_miner.gpu.mining, et_miner.gpu.kernels.gpu_resident, et_miner.gpu.dispatch\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]


# --------------------------------------------------------------------------
# The set claims
# --------------------------------------------------------------------------


def _exported_from_gpu_resident() -> set[str]:
    """Names `kernels/__init__.py` re-exports that are DEFINED in
    `gpu_resident`, read off the objects rather than parsed out of the import
    statement.

    `__module__` is the load-bearing filter, not `dir()`: `gpu_resident` itself
    imports `get_cuda_kernel` and the `_assert_*` helpers from `loader`, so
    `dir()` sees them and identity against `kernels` matches -- the first
    version of this helper reported `get_cuda_kernel` as an unclassified
    gpu-resident entry point. It says "defined in", so it must test that."""
    from et_miner.gpu import kernels
    from et_miner.gpu.kernels import gpu_resident

    return {
        n
        for n in kernels.__all__
        if getattr(getattr(kernels, n, None), "__module__", None) == gpu_resident.__name__
    }


def test_every_exported_entry_point_is_classified_exactly_once():
    from et_miner.gpu.kernels import gpu_resident

    guarded = set(gpu_resident._GUARDED_ENTRY_POINTS)
    exempt = set(gpu_resident._EXEMPT_ENTRY_POINTS)
    exported = _exported_from_gpu_resident()

    assert exported, "premise: the package must re-export something from gpu_resident"
    assert guarded & exempt == set(), f"classified both ways: {sorted(guarded & exempt)}"
    assert exported - (guarded | exempt) == set(), (
        f"exported but unclassified: {sorted(exported - (guarded | exempt))}. Add each to "
        "_GUARDED_ENTRY_POINTS (and give it the three guards) or to _EXEMPT_ENTRY_POINTS "
        "with the reason -- the module docstring quantifies over those two tuples."
    )
    assert (guarded | exempt) - exported == set(), (
        f"classified but not exported: {sorted((guarded | exempt) - exported)}. The tuples "
        "describe this module's PUBLIC surface; a renamed or withdrawn export must leave them."
    )


def test_the_guarded_entry_points_really_call_all_three_guards():
    """Membership checked against the function's source, not declared. Adding a
    name to `_GUARDED_ENTRY_POINTS` without adding the guards fails here."""
    import inspect

    from et_miner.gpu.kernels import gpu_resident

    for name in gpu_resident._GUARDED_ENTRY_POINTS:
        src = inspect.getsource(getattr(gpu_resident, name))
        for guard in _GUARD_CALLS:
            assert f"{guard}(" in src, f"{name} is in _GUARDED_ENTRY_POINTS but its source has no {guard}("


def test_the_exempt_entry_points_call_no_guard_and_say_why():
    """The other direction: an exempt name that HAS acquired guards is a stale
    exemption, and belongs in the guarded tuple instead."""
    import inspect

    from et_miner.gpu.kernels import gpu_resident

    for name, reason in gpu_resident._EXEMPT_ENTRY_POINTS.items():
        src = inspect.getsource(getattr(gpu_resident, name))
        for guard in _GUARD_CALLS:
            assert f"{guard}(" not in src, (
                f"{name} is listed exempt but calls {guard}( -- move it to _GUARDED_ENTRY_POINTS."
            )
        assert reason.strip(), f"{name} is exempt with no stated reason"


def test_the_k2_and_k3_families_are_both_represented():
    """A guarded tuple that lost its K=2 half, or its multi-GPU half, would
    still pass every test above while the docstring's "differing only in the
    rank each requires" stopped covering anything."""
    from et_miner.gpu.kernels import gpu_resident

    guarded = set(gpu_resident._GUARDED_ENTRY_POINTS)
    assert any("k2" in n for n in guarded) and any("k3plus" in n for n in guarded)
    assert any(n.endswith("_multi_gpu") for n in guarded)
    assert any(not n.endswith("_multi_gpu") for n in guarded)


def _call_sites(symbol: str) -> set[tuple[str, int, str]]:
    """`(path relative to the package, lineno, enclosing top-level def)` for
    every call to `symbol` in every `.py` under `et_miner`, read off the AST.

    A call is an `ast.Call` whose callee is the bare name or an attribute of
    that name, so `symbol(...)` and `module.symbol(...)` both count and the
    `def` line, imports, comments and strings do not -- no exclusion list. The
    previous version was a regex over text lines: a docstring containing
    `_assert_home(` turned the suite red, two calls on one line counted as one,
    and the test comparing it against `str.count` was counting lines on one
    side and occurrences on the other. `ast.Call` is the unit on both sides.

    The enclosing name is the OUTERMOST `def` that contains the call, so a call
    made from a nested helper is attributed to the entry point that owns it --
    the question the exemption prose answers."""
    import et_miner

    root = Path(et_miner.__file__).parent
    out: set[tuple[str, int, str]] = set()

    def visit(node: ast.AST, rel: str, owner: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, rel, owner or child.name)
                continue
            if isinstance(child, ast.Call):
                fn = child.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name == symbol:
                    out.add((rel, child.lineno, owner or "<module>"))
            visit(child, rel, owner)

    for f in sorted(root.rglob("*.py")):
        visit(ast.parse(f.read_text(), filename=str(f)), str(f.relative_to(root)), "")
    return out


def _names_quoted_in(prose: str, names: set[str]) -> set[str]:
    """Which of `names` appear in `prose` as whole identifiers. Substring
    matching would let `count_k3plus_gpu_resident` be found inside
    `_count_k3plus_gpu_resident_impl`."""
    return {n for n in names if re.search(rf"(?<![A-Za-z0-9_]){re.escape(n)}(?![A-Za-z0-9_])", prose)}


def test_assert_home_is_called_only_from_the_guarded_entry_points():
    """`loader.py::_assert_home` says its callers are EXACTLY
    `_GUARDED_ENTRY_POINTS`. One set equality carries both halves of
    "exactly": every guarded entry point calls it, and nothing else does.
    Asserting only the first is how a set claim stays half-true."""
    from et_miner.gpu.kernels import gpu_resident

    sites = _call_sites("_assert_home")
    callers = {owner for _, _, owner in sites}

    assert callers == set(gpu_resident._GUARDED_ENTRY_POINTS), (
        f"_assert_home is called from {sorted(callers)} but _GUARDED_ENTRY_POINTS is "
        f"{sorted(gpu_resident._GUARDED_ENTRY_POINTS)}; sites: {sorted(sites)}. If a new caller "
        "is legitimate, the 'callers are exactly _GUARDED_ENTRY_POINTS' sentence in loader.py "
        "is now false."
    )
    assert {path for path, _, _ in sites} == {"gpu/kernels/gpu_resident.py"}, sorted(sites)


def test_build_prefix_groups_gpu_call_sites_match_its_exemption():
    """The exemption reason for `build_prefix_groups_gpu` counts its call sites
    and names each by its enclosing function, saying one of them --
    `gpu/dispatch.py::dispatch_k3plus_gpu_resident` -- reaches it BEFORE any
    guard. A counted claim in prose goes stale the moment a caller is added,
    and a named one the moment a caller moves; both are checked here.

    Named by function, not by line: the call in `dispatch.py` moves every time
    a line is inserted above it, and pinning the number would turn every such
    edit into a red test about an exemption that has not changed."""
    from et_miner.gpu.kernels import gpu_resident

    reason = gpu_resident._EXEMPT_ENTRY_POINTS["build_prefix_groups_gpu"]
    sites = _call_sites("build_prefix_groups_gpu")
    owners = {owner for _, _, owner in sites}

    assert {path for path, _, _ in sites} == {"gpu/kernels/gpu_resident.py", "gpu/dispatch.py"}, sorted(sites)
    assert len(sites) == 3, (
        f"the exemption reason says THREE call sites; found {len(sites)}: {sorted(sites)}. "
        "Update _EXEMPT_ENTRY_POINTS['build_prefix_groups_gpu'] with it."
    )
    assert _names_quoted_in(reason, owners) == owners, (
        f"call sites are owned by {sorted(owners)} but the exemption reason names only "
        f"{sorted(_names_quoted_in(reason, owners))}"
    )
    dispatch_owner = {owner for path, _, owner in sites if path == "gpu/dispatch.py"}
    assert dispatch_owner == {"dispatch_k3plus_gpu_resident"}, sorted(sites)
    assert "gpu/dispatch.py::dispatch_k3plus_gpu_resident" in reason


# --------------------------------------------------------------------------
# The K=1 seed cast
# --------------------------------------------------------------------------


def test_the_k1_seed_cast_is_still_in_mining():
    """Read off `gpu.mining`'s own source, because that is the file the claim is
    about. A local rebuild of the expression -- what this test used to do --
    stays green when the cast is deleted from the module.

    Source inspection and not behaviour: the cast is 60 lines inside
    `_apriori_from_bitvecs_gpu_resident`, reachable only with a real bitvec
    matrix, and the behavioural check already exists one layer out
    (`test_gpu_resident_e2e.py`, via the `_assert_dtype` guard). The positive
    control -- that an int32 (n, 1) seed built this way passes both guards --
    needs a device and is `test_kernel_input_guards.py::
    test_the_guards_accept_a_correctly_built_k1_seed`."""
    import inspect

    from et_miner.gpu import mining

    src = inspect.getsource(mining._apriori_from_bitvecs_gpu_resident)
    assert "cp.where(freq_mask)[0].astype(cp.int32)" in src, (
        "the K=1 seed cast is gone or reshaped. `cp.where` is int64 natively, so "
        "without it the seed reaches count_pairs_fused_k2_gpu_resident as int64 "
        "and _assert_dtype raises there -- see test_gpu_resident_e2e.py."
    )
