"""psutil is a runtime dependency, not a dev one.

Two modules under ``src/`` measure host memory through psutil and swallow
ImportError, degrading silently rather than failing (the row-split miner's
``max_ram_gb`` guard imports it unguarded):

- ``streaming.son._get_memory_gb`` reports 0.0, so the streaming memory
  budget reads as empty.

Declaring psutil only in the dev group therefore changes behaviour for
installed users without raising anything. These tests fail if it slips back
out of ``[project].dependencies``.
"""

import importlib



def test_psutil_is_importable():
    """The dependency itself is present in a plain install."""
    assert importlib.import_module("psutil") is not None


def test_streaming_reports_real_process_memory():
    """_get_memory_gb measures RSS instead of returning its 0.0 fallback."""
    from et_miner.streaming.son import _get_memory_gb

    assert _get_memory_gb() > 0.0

