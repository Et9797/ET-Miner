"""#6-#10, N1-N4 -- parameters accepted by apriori() and silently dropped by the route.

One failure shape, nine instances. `apriori()` takes the parameter, picks a
route, and the route either does not accept it or never reaches the code that
reads it. Nothing in the return value or the logs says so.

  #6  streaming=True drops prune_equal_support / use_generator_pruning:
      the `if streaming:` branch is evaluated BEFORE either consumer of
      _route_for_pruning, and neither callee has the parameter. The caller asks
      for free-sets and gets the complete lattice.
  #7  profile=True returns a bare DataFrame on three routes. Worse than an
      exception: a result frame has exactly two columns, so
      `result, session = apriori(...)` unpacks into two Series.
  #8  anchor_items is ignored on the CPU and streaming routes -- it appears in
      the signature, the routing test and the forwarded kwarg, and nowhere in
      the CPU loop.
  #9  output_dir / resume_from_k are dropped outside the row-split route.
  #10 memory_budget_gb is dropped on the multi-GPU streaming route.
  N1  gpu_resident is dropped whenever prune_equal_support is set, because
      _route_for_pruning is tested before the `if gpu_resident:` branch.
      Not in the 62; found while planning the remediation.
  N2  gpu_resident is dropped on ANY row-split run, not only a pruning one:
      the same test reads `n_gpus > 1 or anchor_items is not None or
      _route_for_pruning`, so a plain multi-GPU call with no pruning anywhere
      in it lands on row-split too. The N1 guard keyed on prune_equal_support
      and missed both other doors.
  N3  gpu_resident is dropped on multi-GPU streaming: apriori_streaming_multi_gpu
      has no such parameter, while the single-GPU sibling one branch below is
      handed one.
  N4  gpu_resident is dropped on the CPU route, which never reads the flag.

The file already demonstrates the right pattern: an explicit ValueError for the
same class of unsupported combination.
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _fixture():
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(0)
    rows = [sorted(rng.choice(12, size=int(rng.integers(3, 7)), replace=False).tolist())
            for _ in range(400)]
    # nested vocabulary so the free-set prune actually engages
    rows = [r + [100] if 0 in r else r for r in rows]
    return pl.DataFrame({"items": rows})


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    import polars as pl

    from et_miner import apriori

    df = _fixture()
    live: list[str] = []

    def check(label: str, call, is_silently_wrong) -> None:
        """LIVE when the call returns something the caller did not ask for.

        FIXED when it raises -- refusing an unsupported combination is the
        contract here, because the route genuinely cannot honour the parameter.
        """
        try:
            got = call()
        except (ValueError, TypeError, NotImplementedError):
            return
        verdict = is_silently_wrong(got)
        if verdict:
            live.append(f"{label}: {verdict}")

    full = apriori(df, min_support=0.05)
    gated = apriori(df, min_support=0.05, use_gpu=True, prune_equal_support=True)
    plain_resident = apriori(df, min_support=0.05, use_gpu=True, gpu_resident=True)

    check("#6 streaming+prune",
          lambda: apriori(df, min_support=0.05, streaming=True, chunk_size=100,
                          prune_equal_support=True),
          lambda r: (f"{r.height} rows = the full lattice, not the {gated.height} free-sets"
                     if r.height == full.height else None))

    check("#7 profile on multi-GPU streaming",
          lambda: apriori(df, min_support=0.05, streaming=True, chunk_size=100,
                          n_gpus=2, profile=True),
          lambda r: (f"bare {type(r).__name__}; unpacks to two "
                     f"{type(next(iter(r))).__name__}" if isinstance(r, pl.DataFrame) else None))

    check("#8 anchor_items on CPU",
          lambda: apriori(df, min_support=0.05, anchor_items={0, 1}),
          lambda r: (f"{r.height} rows, identical to unanchored"
                     if r.height == full.height else None))

    def _output_dir_call():
        with tempfile.TemporaryDirectory() as td:
            apriori(df, min_support=0.05, output_dir=td)
            return sorted(p.name for p in Path(td).iterdir())

    check("#9 output_dir on CPU", _output_dir_call,
          lambda files: f"files written: {files}" if not files else None)

    # #10 needs a spy, not an output comparison: the budget only changes the
    # derived chunk size, and on a 400-row fixture every chunking lands on one
    # chunk. So assert the value actually crosses the route boundary.
    from et_miner.streaming import multi_gpu as mg

    seen: dict[str, object] = {}
    real = mg.apriori_streaming_multi_gpu

    def spy(*a, **kw):
        seen.update(kw)
        raise RuntimeError("stop after capturing the forwarded kwargs")

    mg.apriori_streaming_multi_gpu = spy
    try:
        apriori(df, min_support=0.05, streaming=True, chunk_size=100, n_gpus=2,
                memory_budget_gb=0.25)
    except (RuntimeError, ValueError, TypeError):
        pass
    finally:
        mg.apriori_streaming_multi_gpu = real

    accepts = "memory_budget_gb" in inspect.signature(real).parameters
    forwarded = seen.get("memory_budget_gb")
    if not accepts or forwarded != 0.25:
        live.append(
            f"#10 memory_budget_gb: callee accepts={accepts}, "
            f"forwarded={forwarded!r} (expected 0.25)"
        )

    # N2-N4 need the #10 spy, not a height comparison. Every route here is
    # obliged by the tier-equivalence chain to return the SAME itemsets, so
    # comparing heights compares two numbers that are equal by contract and
    # cannot see a dropped flag. Written that way first, N2 measured 214 against
    # a plain_resident of 214 and reported itself FIXED on a tree where the
    # defect was live -- the headline case, clearing its own test. What does
    # differ is which miner ran, so capture that instead.
    from et_miner.gpu import mining as gm
    from et_miner.gpu import row_split as rs

    _ROUTES = [(rs, "_apriori_row_split_multi_gpu", "row_split"),
               (gm, "_apriori_from_bitvecs_gpu_resident", "gpu_resident"),
               (gm, "_apriori_from_bitvecs", "bitvecs"),
               (mg, "apriori_streaming_multi_gpu", "streaming_multi_gpu")]

    def _route_taken(**kw):
        """The label of the miner apriori() actually dispatched to, or "cpu".

        Patching the module attribute works because apriori() imports each
        route function lazily inside its branch, which is the same reason the
        #10 spy above works.
        """
        taken: list[str] = []
        saved = [(m, n, getattr(m, n)) for m, n, _ in _ROUTES]
        for m, n, label in _ROUTES:
            def stop(*a, _l=label, **k):
                taken.append(_l)
                raise RuntimeError("stop after capturing the route")
            setattr(m, n, stop)
        try:
            apriori(df, min_support=0.05, gpu_resident=True, **kw)
        except RuntimeError:
            pass  # the spy's own stop signal, raised once the route is known
        finally:
            for m, n, orig in saved:
                setattr(m, n, orig)
        # ValueError deliberately propagates: a guard refusing the call is the
        # FIXED state, and check() reads it as such. Swallowing it here would
        # leave `taken` empty and report the refusal as a dispatch to the CPU
        # miner -- the defect, on the tree that fixed it.
        return taken[0] if taken else "cpu"

    def _not_resident(route):
        return f"ran {route}, not the GPU-resident miner" if route != "gpu_resident" else None

    check("N2 gpu_resident+multi-GPU",
          lambda: _route_taken(use_gpu=True, n_gpus=2), _not_resident)

    check("N2b gpu_resident+anchor_items",
          lambda: _route_taken(use_gpu=True, anchor_items={0, 1}), _not_resident)

    check("N3 gpu_resident+multi-GPU streaming",
          lambda: _route_taken(streaming=True, chunk_size=100, n_gpus=2), _not_resident)

    check("N4 gpu_resident on the CPU route",
          lambda: _route_taken(), _not_resident)

    check("N1 gpu_resident+prune",
          lambda: apriori(df, min_support=0.05, use_gpu=True, prune_equal_support=True,
                          gpu_resident=True),
          lambda r: (f"{r.height} rows = the row-split result, not gpu_resident's "
                     f"{plain_resident.height}"
                     if r.height == gated.height != plain_resident.height else None))

    return bool(live), ("; ".join(live) if live
                        else "every unsupported combination is rejected or honoured")


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
