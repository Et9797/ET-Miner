"""The campaign gate's NOT-GATED branch, and the revision stamp it compares on.

`bench/runner.py`'s coverage report is unreachable from a green campaign: a run
that gates successfully returns at the ✓ and never executes the branch that
counts and names the failures. Every defect found there so far shipped
unobserved for exactly that reason, and `bash bench/run_smoke.sh` exiting 0 is
no evidence about it -- a fresh single-revision run has nothing missing, bad or
stale to report. These tests drive `_coverage` and `_git_rev` directly, and
`main` end to end with git, the device count, the environment capture and the
child process stubbed -- which is also the first time `main`'s green path has
run without a GPU.

CONTROL, both measured, not asserted, re-measured after the tests below were
added. Replace `ok_here` with the old
`len(all_ids) - len(missing) - len(bad) - len(stale)` and FOUR tests go red --
`test_ok_count_survives_a_bad_and_stale_overlap`, plus the complementarity,
latest-row and outside-the-matrix cases, which the subtraction also gets wrong.
Drop the digest from `_git_rev` and exactly ONE goes red,
`test_two_distinct_dirty_trees_get_distinct_revs`, at
`assert '97bb2c0-dirty' != '97bb2c0-dirty'` -- that sha is reproducible because
`tiny_repo` commits at a fixed date -- while the other seven `_git_rev` tests
still pass, because a bare `-dirty` suffix is right about everything except
telling two dirty trees apart.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
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


def test_an_unknown_here_covers_nothing():
    """The false green. `here` and every row's rev come from the same
    function, so when git fails they are all "unknown" -- EQUAL -- and the
    equality compare called every row fresh, printed the tick and exited 0.
    The sentinel is not a revision and must be stale on both sides."""
    rows = [_row("a", rev="unknown"), _row("b", rev="unknown")]
    _, missing, bad, stale, ok_here = runner._coverage({"a", "b"}, rows, "unknown")

    assert ok_here == [] and stale == ["a", "b"]
    assert missing == [] and bad == [], "premise: the rows are present and ok; only the rev is wrong"


def test_an_unknown_row_is_stale_at_a_real_rev():
    """The other side: a row stamped "unknown" is stale whatever `here` is, so
    a campaign that lost git for one config cannot count that config."""
    _, _, _, stale, ok_here = runner._coverage({"a"}, [_row("a", rev="unknown")], "abc1234")
    assert stale == ["a"] and ok_here == []


# --------------------------------------------------------------------------
# _git_rev: what the staleness compare above can actually separate
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_repo(tmp_path, monkeypatch):
    """A throwaway git repo with one committed file, with `runner.REPO` pointed
    at it. `_git_rev` shells out with `cwd=REPO`, so this is the whole seam.

    The commit is made at a fixed author and committer date, so its sha is the
    same on every box and every day and the module docstring can quote it. A
    previous version set no date, and the literal it quoted was whatever the
    clock said when it was measured.

    `shutil.which`, not a `subprocess.run(["git", "--version"])`: with git
    absent the latter raises `FileNotFoundError` before there is a returncode
    to test, so the skip it guarded could never fire and the seven tests below
    errored instead."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("a\n")
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
        env={**os.environ, "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
             "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000"},
    )
    monkeypatch.setattr(runner, "REPO", tmp_path)
    return tmp_path


def test_a_clean_tree_gets_the_bare_sha(tiny_repo):
    """A hex abbreviation and nothing else. `len(rev) >= 7` was the previous
    assertion, and `"unknown"` is seven characters long."""
    rev = runner._git_rev()
    assert re.fullmatch(r"[0-9a-f]{7,40}", rev), rev


def test_git_failure_yields_the_sentinel_and_says_why(tmp_path, monkeypatch, capsys):
    """Not a repository: `git rev-parse` exits 128. The sentinel comes back --
    and the reason goes to stderr, because a bare "unknown" in a campaign
    directory name was undiagnosable after the fact."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    monkeypatch.setattr(runner, "REPO", tmp_path)
    assert runner._git_rev() == "unknown"
    err = capsys.readouterr().err
    assert "_git_rev" in err and "not a git repository" in err


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


# --------------------------------------------------------------------------
# main(): the sentinel refusal and the mid-run drift check, driven with stubs
# --------------------------------------------------------------------------


def _drive_main(monkeypatch, tmp_path, revs: list[str], *, row_rev: str = "A") -> int:
    """Run `main --mode smoke --out tmp` with git, the device count, the
    environment capture and the child process all stubbed. `revs` is what
    successive `_git_rev()` calls return (the last value repeats); rows come
    back ok, stamped `row_rev`, with one shared signature so
    `check_equivalence` has nothing to say."""
    it = iter(revs)
    last = revs[-1]
    monkeypatch.setattr(runner, "_git_rev", lambda: next(it, last))
    monkeypatch.setattr(runner, "_gpu_count", lambda: 2)
    monkeypatch.setattr(runner, "capture_environment", lambda out_dir: None)
    monkeypatch.setattr(
        runner, "run_config",
        lambda cfg, out_dir: {"id": cfg["id"], "config": cfg, "status": "ok", "rev": row_rev,
                              "n_itemsets": 1, "sum_counts": 1, "itemset_hash": "h"},
    )
    monkeypatch.setattr(sys, "argv", ["runner.py", "--mode", "smoke", "--out", str(tmp_path)])
    return runner.main()


def test_main_gates_a_fresh_run_on_a_frozen_tree(monkeypatch, tmp_path, capsys):
    """Positive control for the two refusals below: same stubs, nothing
    moves, the tick is emitted and the exit code is 0."""
    rc = _drive_main(monkeypatch, tmp_path, ["A"])
    out = capsys.readouterr().out
    assert rc == 0 and "consistent ✓" in out and "every row at A" in out


def test_main_refuses_to_start_at_an_unknown_rev(monkeypatch, tmp_path, capsys):
    """Fail before the first config, not after the last: a matrix run at the
    sentinel could never gate, and the smoke matrix is not free."""
    monkeypatch.setattr(runner, "run_config", lambda cfg, out_dir: pytest.fail("a config ran"))
    it = iter(["unknown"])
    monkeypatch.setattr(runner, "_git_rev", lambda: next(it, "unknown"))
    monkeypatch.setattr(runner, "_gpu_count", lambda: 2)
    monkeypatch.setattr(runner, "capture_environment", lambda out_dir: None)
    monkeypatch.setattr(sys, "argv", ["runner.py", "--mode", "smoke", "--out", str(tmp_path)])

    rc = runner.main()
    out = capsys.readouterr().out
    assert rc == 2 and "cannot determine the revision" in out and "consistent ✓" not in out


def test_main_refuses_the_tick_when_the_tree_moved_after_the_last_config(monkeypatch, tmp_path, capsys):
    """The case `_coverage` cannot see: every row IS at `here`, because the
    edit landed after the last config finished -- so nothing is stale, and by
    the row predicate alone the tick would be printed about a tree that no
    longer exists. `here` is read once before the loop, `now` once after."""
    rc = _drive_main(monkeypatch, tmp_path, ["A", "B"], row_rev="A")
    out = capsys.readouterr().out
    assert rc == 1
    assert "NOT GATED at A" in out and "started at A and stands at B" in out
    assert "6 of 6 configs are ok at A" in out, "premise: no row is stale; only the tree moved"
    assert "consistent ✓" not in out


def test_main_lists_rows_produced_after_a_mid_run_edit_as_stale(monkeypatch, tmp_path, capsys):
    """The case `_coverage` does see, still driven end to end: the tree moved
    and the configs that ran afterwards carry the new rev."""
    rc = _drive_main(monkeypatch, tmp_path, ["A", "B"], row_rev="B")
    out = capsys.readouterr().out
    assert rc == 1
    assert "0 of 6 configs are ok at A" in out
    assert "6 were produced at a revision other than A (B)" in out
    assert "consistent ✓" not in out
