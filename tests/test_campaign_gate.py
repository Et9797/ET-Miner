"""The campaign gate's NOT-GATED branch, and the revision stamp it compares on.

`bench/runner.py`'s coverage report is unreachable from a green campaign: a run
that gates successfully returns at the ✓ and never executes the branch that
counts and names the failures. Every defect found there so far shipped
unobserved for exactly that reason, and `bash bench/run_smoke.sh` exiting 0 is
no evidence about it -- a fresh single-revision run has nothing missing, bad or
stale to report. These tests drive `_coverage` and `_git_rev` directly.

CONTROL, both measured, not asserted. Replace `ok_here` with the old
`len(all_ids) - len(missing) - len(bad) - len(stale)` and FOUR tests go red --
`test_ok_count_survives_a_bad_and_stale_overlap` at `assert (0 == 1)`, plus the
complementarity, latest-row and outside-the-matrix cases, which the subtraction
also gets wrong. Drop the digest from `_git_rev` and exactly ONE goes red,
`test_two_distinct_dirty_trees_get_distinct_revs`, at
`assert '13f486b-dirty' != '13f486b-dirty'`; the other five `_git_rev` tests
still pass, because a bare `-dirty` suffix is right about everything except
telling two dirty trees apart.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_runner():
    """Import `bench/runner.py` by path -- `bench/` is not a package and is
    outside `testpaths`, so a plain import does not reach it."""
    spec = importlib.util.spec_from_file_location("_bench_runner", REPO_ROOT / "bench" / "runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()


def _row(id_, status="ok", rev="abc1234"):
    return {"id": id_, "config": {"id": id_}, "status": status, "rev": rev}


# --------------------------------------------------------------------------
# _coverage: the classification the printed count is read off
# --------------------------------------------------------------------------


def test_ok_count_survives_a_bad_and_stale_overlap():
    """The defect: `bad` and `stale` are not disjoint, so subtracting both from
    the total counted the overlap twice. Two configs, one ok here and one that
    is BOTH non-ok and at another revision -- the truth is 1, the subtraction
    said 0."""
    rows = [_row("a", rev="here"), _row("b", status="timeout", rev="there")]
    _, missing, bad, stale, ok_here = runner._coverage({"a", "b"}, rows, "here")

    assert bad == ["b"] and stale == ["b"], "premise: b must land in both lists"
    assert 2 - len(missing) - len(bad) - len(stale) == 0, "premise: the old expression says 0"
    assert len(ok_here) == 1 and ok_here == ["a"]


def test_the_ok_count_never_goes_negative():
    """Enough overlapping ids and the subtraction underflows: three configs
    each both non-ok and stale gives 3 - 0 - 3 - 3 = -3 printed as a count."""
    rows = [_row(i, status="crash", rev="there") for i in ("a", "b", "c")]
    _, missing, bad, stale, ok_here = runner._coverage({"a", "b", "c"}, rows, "here")

    assert len(ok_here) == 0
    assert 3 - len(missing) - len(bad) - len(stale) == -3, "premise: the old form underflows"


def test_ok_here_and_the_failure_lists_are_complementary():
    """The only partition property the printed count may assume: every id is
    either in `ok_here` or in at least one of the three failure lists, and
    never in both."""
    rows = [
        _row("ok", rev="here"),
        _row("badonly", status="timeout", rev="here"),
        _row("staleonly", rev="there"),
        _row("both", status="timeout", rev="there"),
    ]
    all_ids = {"ok", "badonly", "staleonly", "both", "never-ran"}
    _, missing, bad, stale, ok_here = runner._coverage(all_ids, rows, "here")

    failed = set(missing) | set(bad) | set(stale)
    assert set(ok_here) | failed == all_ids
    assert set(ok_here) & failed == set()
    assert missing == ["never-ran"]


def test_missing_is_disjoint_from_bad_and_stale():
    """An absent row has no status and no rev, so it must not be reported as
    the wrong status or the wrong revision as well."""
    _, missing, bad, stale, _ = runner._coverage({"a"}, [], "here")
    assert missing == ["a"] and bad == [] and stale == []


def test_latest_row_wins_over_an_earlier_one():
    """raw.jsonl is append-only: a re-run appends, so the LAST row for an id is
    the current one. A stale row followed by a fresh one is covered."""
    rows = [_row("a", status="crash", rev="there"), _row("a", rev="here")]
    _, missing, bad, stale, ok_here = runner._coverage({"a"}, rows, "here")
    assert ok_here == ["a"] and not (missing or bad or stale)


def test_rows_outside_the_matrix_are_ignored():
    """`check_equivalence` runs unscoped over every row, but coverage is a
    claim about the MATRIX -- a stray id must not gate it either way."""
    rows = [_row("a", rev="here"), _row("not-in-matrix", status="crash", rev="there")]
    _, missing, bad, stale, ok_here = runner._coverage({"a"}, rows, "here")
    assert ok_here == ["a"] and not (missing or bad or stale)


# --------------------------------------------------------------------------
# _git_rev: what the staleness compare above can actually separate
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_repo(tmp_path, monkeypatch):
    """A throwaway git repo with one committed file, with `runner.REPO` pointed
    at it. `_git_rev` shells out with `cwd=REPO`, so this is the whole seam."""
    if not subprocess.run(["git", "--version"], capture_output=True).returncode == 0:
        pytest.skip("git not available")
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
    )
    monkeypatch.setattr(runner, "REPO", tmp_path)
    return tmp_path


def test_a_clean_tree_gets_the_bare_sha(tiny_repo):
    rev = runner._git_rev()
    assert "-dirty" not in rev and len(rev) >= 7


def test_two_distinct_dirty_trees_get_distinct_revs(tiny_repo):
    """The defect: `f"{sha}-dirty"` was ONE string for every distinct tree at a
    commit, so the staleness compare could not separate rows produced by two
    different edits and called them consistent. Both edits touch the same file,
    which is what a porcelain-status hash would also have failed to separate."""
    (tiny_repo / "f.txt").write_text("b\n")
    first = runner._git_rev()
    (tiny_repo / "f.txt").write_text("c\n")
    second = runner._git_rev()

    assert first != second
    assert first.split("-dirty.")[0] == second.split("-dirty.")[0], "same commit, different tree"


def test_the_same_edit_reproduces_the_same_rev(tiny_repo):
    """A resume must land in the campaign directory it started in -- the digest
    has to be a function of the content and nothing else (not time, not order)."""
    (tiny_repo / "f.txt").write_text("b\n")
    first = runner._git_rev()
    (tiny_repo / "f.txt").write_text("c\n")
    runner._git_rev()
    (tiny_repo / "f.txt").write_text("b\n")
    assert runner._git_rev() == first


def test_reverting_the_edit_returns_the_bare_sha(tiny_repo):
    (tiny_repo / "f.txt").write_text("b\n")
    assert "-dirty" in runner._git_rev()
    (tiny_repo / "f.txt").write_text("a\n")
    assert "-dirty" not in runner._git_rev()


def test_untracked_files_do_not_change_the_rev(tiny_repo):
    """Documented scope, pinned so it cannot drift silently: `git diff HEAD`
    does not see untracked files, matching the `--untracked-files=no` this
    replaced. A reader sizing the residual risk needs this to stay true."""
    (tiny_repo / "f.txt").write_text("b\n")
    before = runner._git_rev()
    (tiny_repo / "u.txt").write_text("untracked\n")
    assert runner._git_rev() == before


def test_a_staged_edit_counts_as_dirty(tiny_repo):
    """`git diff HEAD` covers staged and unstaged alike -- a `git add`ed change
    is still not the committed code."""
    (tiny_repo / "f.txt").write_text("b\n")
    subprocess.run(["git", "add", "f.txt"], cwd=tiny_repo, check=True)
    assert "-dirty" in runner._git_rev()


def test_the_rev_is_usable_as_a_directory_name(tiny_repo):
    """`_campaign_out` is `DEFAULT_OUT / _git_rev()`, so the stamp doubles as a
    path component."""
    (tiny_repo / "f.txt").write_text("b\n")
    rev = runner._git_rev()
    assert "/" not in rev and "\\" not in rev and rev == rev.strip()
    (tiny_repo / rev).mkdir()
