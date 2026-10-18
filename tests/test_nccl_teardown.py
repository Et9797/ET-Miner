"""The row-split miner destroys its NCCL communicators on a background thread.

NCCL's destroy waits for its proxy thread, which wakes every 500 ms when idle,
so a call ended up to about 0.5 s after its work
(bench/moonshots/REPORT-TEARDOWN.md).
The miner now hands the communicators to `nccl._destroy_nccl` once its levels
are done, and returns without waiting.
"""

from __future__ import annotations

import subprocess
import sys
import threading

import numpy as np
import polars as pl
import pytest

def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


needs_two = pytest.mark.skipif(_gpu_count() < 2, reason="needs 2 CUDA devices")

THREAD = "et-miner-nccl-destroy"


def _rows(seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [sorted(set(rng.integers(0, 25, size=int(rng.integers(2, 9))).tolist())) for _ in range(6_000)]
    return pl.DataFrame({"items": rows})


def _mined(res: pl.DataFrame) -> set:
    return {(tuple(s), round(sup, 9)) for s, sup in zip(res["itemset"].to_list(), res["support"].to_list())}


def _join_destroys() -> None:
    for thread in threading.enumerate():
        if thread.name == THREAD:
            thread.join(timeout=30)
            assert not thread.is_alive(), "an NCCL destroy did not finish within 30 s"


def test_no_communicators_start_no_thread():
    from et_miner.gpu.nccl import _destroy_nccl

    assert _destroy_nccl(None) is None and _destroy_nccl([]) is None


@pytest.mark.gpu
@pytest.mark.multigpu
@needs_two
def test_destroy_runs_on_its_own_thread():
    from et_miner.gpu.nccl import _destroy_nccl, _init_nccl

    comms, ok = _init_nccl([0, 1])
    assert ok
    thread = _destroy_nccl(comms)
    assert thread.name == THREAD and thread is not threading.current_thread() and not thread.daemon
    thread.join(timeout=30)
    assert [c.comm for c in comms] == [0, 0]


@pytest.mark.gpu
@pytest.mark.multigpu
@needs_two
def test_the_miner_hands_its_communicators_off(monkeypatch):
    """Every communicator the miner built is destroyed once its thread ends, and the result is the one-GPU result."""
    from et_miner import apriori
    from et_miner.gpu import row_split

    built = []
    init = row_split._init_nccl

    def spy(device_ids):
        comms, ok = init(device_ids)
        built.extend(comms or [])
        return comms, ok

    handed = []
    destroy = row_split._destroy_nccl

    def hand_off(comms):
        handed.append(list(comms or []))
        return destroy(comms)

    monkeypatch.setattr(row_split, "_init_nccl", spy)
    monkeypatch.setattr(row_split, "_destroy_nccl", hand_off)
    df = _rows()
    two = _mined(apriori(df, min_support=0.01, use_gpu=True, n_gpus=2))
    _join_destroys()
    assert len(built) == 2 and handed == [built] and [c.comm for c in built] == [0, 0]
    assert two == _mined(apriori(df, min_support=0.01, use_gpu=True, n_gpus=1)) and len(two) > 0


@pytest.mark.gpu
@pytest.mark.multigpu
@needs_two
def test_calls_in_a_row_while_the_last_destroy_runs():
    """A call's init may overlap the previous call's destroy: NCCL serves distinct communicators on distinct threads."""
    from et_miner import apriori

    df = _rows(11)
    results = [_mined(apriori(df, min_support=0.01, use_gpu=True, n_gpus=2)) for _ in range(3)]
    _join_destroys()
    assert results[0] == results[1] == results[2] and len(results[0]) > 0


@pytest.mark.gpu
@pytest.mark.multigpu
@needs_two
def test_an_error_leaves_the_communicators_to_the_frame(monkeypatch):
    """After an error the miner starts no destroy thread: the communicators go with its frame, as before."""
    from et_miner import apriori
    from et_miner.gpu import row_split

    started = []
    monkeypatch.setattr(row_split, "_destroy_nccl", lambda comms: started.append(comms))

    def fail(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(row_split, "run_chunked_dense_level", fail)
    with pytest.raises(RuntimeError, match="injected"):
        apriori(_rows(), min_support=0.01, use_gpu=True, n_gpus=2)
    assert started == []


@pytest.mark.gpu
@pytest.mark.multigpu
@needs_two
def test_a_process_exits_cleanly_right_after_a_call():
    """The interpreter waits for the destroy thread before CUDA's exit handlers run."""
    code = (
        "import numpy as np, polars as pl\n"
        "from et_miner import apriori\n"
        "rng = np.random.default_rng(7)\n"
        "rows = [sorted(set(rng.integers(0, 25, size=int(rng.integers(2, 9))).tolist())) for _ in range(6_000)]\n"
        "res = apriori(pl.DataFrame({'items': rows}), min_support=0.01, use_gpu=True, n_gpus=2)\n"
        "print(res.height)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert int(done.stdout.strip().splitlines()[-1]) > 0
