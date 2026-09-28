"""psutil is a runtime dependency, not a dev one.

Two modules under ``src/`` measure host memory through psutil and swallow
ImportError, degrading silently rather than failing (the row-split miner's
``max_ram_gb`` guard imports it unguarded):

- ``gpu.kernels.filter._host_ram_available`` returns None, which collapses
  the compact filter's feasibility test onto a fixed SLICE_ELEMS survivor
  cap and sends every larger level to the sliced CPU valve;
- ``streaming.son._get_memory_gb`` reports 0.0, so the streaming memory
  budget reads as empty.

Declaring psutil only in the dev group therefore changes behaviour for
installed users without raising anything. These tests fail if it slips back
out of ``[project].dependencies``.
"""

import importlib

import pytest


def test_psutil_is_importable():
    """The dependency itself is present in a plain install."""
    assert importlib.import_module("psutil") is not None


def test_streaming_reports_real_process_memory():
    """_get_memory_gb measures RSS instead of returning its 0.0 fallback."""
    from et_miner.streaming.son import _get_memory_gb

    assert _get_memory_gb() > 0.0


def test_host_ram_available_returns_a_measurement():
    """The compact filter can size its host workspace.

    None here is what degrades the feasibility test to SLICE_ELEMS.
    """
    pytest.importorskip("cupy", reason="gpu.kernels.filter requires cupy")
    from et_miner.gpu.kernels.filter import _host_ram_available

    available = _host_ram_available()
    assert available is not None
    assert available > 0
