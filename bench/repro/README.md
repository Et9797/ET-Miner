# Reproduction harness

One script per defect. Each script **exits non-zero while the defect is live and
zero once it is fixed**, so the same file is both the evidence and the
acceptance gate.

```
uv run python bench/repro/run_all.py            # run every repro
uv run python bench/repro/run_all.py --id 51    # one defect
uv run python bench/repro/run_all.py --expect-live   # CI-style: fail if a defect is already fixed
```

## Why this exists

`BUGS_FOUND.md` pastes a reproduction for most of #1–#39. Defects **#40–#62**
were filed as one-line summary-table rows with a `path:line` and no
reproduction. Six factual errors have already been found in that range
(N2, N8, N15, N16, N18, N19), so ground rule 10 of the remediation plan holds
them to the same bar as the rest: **a defect that cannot be reproduced gets
reclassified, not fixed.**

## Naming

`d<NN>_<slug>.py` for a numbered defect, `n<NN>_<slug>.py` for a finding from
the new register. Each exposes:

```python
def reproduce() -> tuple[bool, str]:
    """Return (defect_is_live, one_line_evidence)."""
```

`run_all.py` imports and calls it; a script may also be run directly.
