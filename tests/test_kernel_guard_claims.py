"""Which kernel wrappers run the bitvec input guard, checked without a device.

`test_kernel_input_guards.py` exercises `loader._assert_bitvecs` on a card.
This file reads source only: every exported wrapper whose first parameter is
`bitvecs_gpu` must call the guard itself or delegate to one that does, so a
new wrapper cannot launch on unchecked bitvecs by being added to the exports.
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys

#: Exported wrappers that take bitvecs but launch no raw-pointer kernel.
_EXEMPT = {
    "column_popcounts": "an ElementwiseKernel, whose argument types CuPy checks itself",
}


def test_the_kernel_package_imports_without_cupy():
    code = "import sys; sys.modules['cupy'] = None\nimport et_miner.gpu.kernels\n"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]


def _calls(fn) -> set[str]:
    tree = ast.parse(inspect.getsource(fn).lstrip())
    return {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }


def test_every_bitvec_wrapper_runs_the_guard():
    from et_miner.gpu import kernels

    wrappers = {
        name: getattr(kernels, name)
        for name in kernels.__all__
        if inspect.isfunction(getattr(kernels, name))
        and next(iter(inspect.signature(getattr(kernels, name)).parameters), None) == "bitvecs_gpu"
    }
    assert wrappers, "no wrapper takes bitvecs_gpu; the export list or this check is wrong"
    guarded = {name for name, fn in wrappers.items() if "_assert_bitvecs" in _calls(fn)}
    for _ in wrappers:  # delegation chains are short; iterate to a fixed point
        guarded |= {name for name, fn in wrappers.items() if _calls(fn) & guarded}
    unguarded = set(wrappers) - guarded - set(_EXEMPT)
    assert not unguarded, f"bitvec wrappers that never reach _assert_bitvecs: {sorted(unguarded)}"
    assert set(_EXEMPT) <= set(wrappers), "an exemption names a wrapper that is gone"
