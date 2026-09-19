"""The CI contract: exit codes, stream discipline and clean failure messages.

These run the real entry point in a subprocess. Calling the Typer app in-process
would bypass ``main()``, which is exactly where the exit-code mapping lives.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalforge import __version__
from evalforge.cli import ExitCode

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
SUITE = FIXTURES / "test_suite.json"
V1 = FIXTURES / "prompt_v1.txt"
V2 = FIXTURES / "prompt_v2.txt"


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI offline and deterministically, whatever the host env holds."""
    return subprocess.run(
        [sys.executable, "-m", "evalforge", *args],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "EVALFORGE_OFFLINE_TEST": "1"},
        timeout=300,
    )


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cli")


def test_version_is_single_sourced() -> None:
    result = run("--version")
    assert result.returncode == ExitCode.OK
    assert __version__ in result.stdout


def test_help_lists_every_command() -> None:
    result = run("--help")
    assert result.returncode == ExitCode.OK
    for command in ("generate", "compare", "check", "show"):
        assert command in result.stdout


def test_improvement_exits_zero(out: Path) -> None:
    result = run(
        "check", "--candidate", str(V2), "--baseline", str(V1), "--suite", str(SUITE),
        "--report", str(out / "ok.md"), "--results", str(out / "ok.json"),
    )
    assert result.returncode == ExitCode.OK, result.stderr
    assert "CANDIDATE WINS" in result.stdout


def test_regression_exits_one(out: Path) -> None:
    result = run(
        "check", "--candidate", str(V1), "--baseline", str(V2), "--suite", str(SUITE),
        "--report", str(out / "bad.md"), "--results", str(out / "bad.json"),
    )
    assert result.returncode == ExitCode.REGRESSION
    assert "CI gate failed" in result.stderr


def test_bad_flag_exits_two() -> None:
    assert run("check", "--not-a-flag").returncode == ExitCode.USAGE


@pytest.mark.parametrize(
    ("make", "fragment"),
    [
        (lambda p: p.write_text("not json", encoding="utf-8"), "not a valid TestSuite"),
        (lambda p: None, "cannot read"),
        (lambda p: p.write_text("{}", encoding="utf-8"), "Field required"),
    ],
)
def test_broken_suite_exits_three_not_one(
    tmp_path: Path, make: object, fragment: str
) -> None:
    # The whole point of a separate code: a typo in a path must not look like
    # "your prompt regressed" to whoever reads the failed build.
    path = tmp_path / "suite.json"
    make(path)  # type: ignore[operator]
    result = run(
        "compare", "--suite", str(path),
        "--baseline", str(FIXTURES / "baseline_run.json"),
        "--candidate", str(FIXTURES / "candidate_run.json"),
    )
    assert result.returncode == ExitCode.INPUT
    # Errors must stay on one line so CI log greps find them.
    assert fragment in result.stderr
    assert all(len(line) < 400 for line in result.stderr.splitlines())


def test_disjoint_completion_sets_exit_three(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"label": "x", "completions": {}}), encoding="utf-8")
    result = run(
        "compare", "--suite", str(SUITE), "--baseline", str(empty),
        "--candidate", str(FIXTURES / "candidate_run.json"),
    )
    assert result.returncode == ExitCode.INPUT
    assert "no test case ids are present in both" in result.stderr


def test_empty_prompt_file_exits_three(tmp_path: Path) -> None:
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n", encoding="utf-8")
    result = run("check", "--candidate", str(blank), "--baseline", str(V1), "--suite", str(SUITE))
    assert result.returncode == ExitCode.INPUT
    assert "is empty" in result.stderr


def test_unexpected_failure_exits_four(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalforge import cli

    def explode() -> None:
        raise RuntimeError("something unforeseen")

    monkeypatch.setattr(cli, "app", explode)
    monkeypatch.delenv("EVALFORGE_TRACEBACK", raising=False)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == ExitCode.INTERNAL


def test_traceback_escape_hatch_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalforge import cli

    def explode() -> None:
        raise RuntimeError("something unforeseen")

    monkeypatch.setattr(cli, "app", explode)
    monkeypatch.setenv("EVALFORGE_TRACEBACK", "1")
    with pytest.raises(RuntimeError, match="unforeseen"):
        cli.main()


def test_exit_codes_are_all_distinct() -> None:
    values = [code.value for code in ExitCode]
    assert len(values) == len(set(values))


def test_scorecard_goes_to_stdout_and_warnings_to_stderr(tmp_path: Path) -> None:
    result = run(
        "check", "--candidate", str(V2), "--baseline", str(V1), "--suite", str(SUITE),
        "--count", "4", "--report", str(tmp_path / "r.md"), "--results", str(tmp_path / "r.json"),
    )
    assert "candidate win rate" in result.stdout
    assert "input error" not in result.stdout


def test_generate_then_compare_round_trips(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    assert run(
        "generate", "--spec", str(FIXTURES / "task_spec.json"),
        "--count", "8", "--offline", "--out", str(suite),
    ).returncode == ExitCode.OK
    assert len(json.loads(suite.read_text(encoding="utf-8"))["cases"]) == 8

    result = run(
        "check", "--candidate", str(V2), "--baseline", str(V1), "--suite", str(suite),
        "--report", str(tmp_path / "r.md"), "--results", str(tmp_path / "r.json"),
    )
    assert result.returncode == ExitCode.OK
    assert (tmp_path / "r.md").read_text(encoding="utf-8").startswith("# eval-forge scorecard")

    shown = run("show", "--results", str(tmp_path / "r.json"))
    assert shown.returncode == ExitCode.OK
    assert "candidate win rate" in shown.stdout


def test_runs_are_reproducible_for_a_seed(tmp_path: Path) -> None:
    def once(tag: str) -> str:
        run(
            "check", "--candidate", str(V2), "--baseline", str(V1), "--suite", str(SUITE),
            "--seed", "123", "--report", str(tmp_path / f"{tag}.md"),
            "--results", str(tmp_path / f"{tag}.json"),
        )
        return (tmp_path / f"{tag}.md").read_text(encoding="utf-8")

    assert once("a") == once("b")


def test_partial_completion_set_is_reported_not_silently_scored(tmp_path: Path) -> None:
    # Scoring a subset without saying so understates n and quietly widens every
    # interval, which reads as "under-powered" for no visible reason.
    full = json.loads((FIXTURES / "baseline_run.json").read_text(encoding="utf-8"))
    kept = dict(list(full["completions"].items())[:10])
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({**full, "completions": kept}), encoding="utf-8")

    result = run(
        "compare", "--suite", str(SUITE), "--baseline", str(partial),
        "--candidate", str(FIXTURES / "candidate_run.json"),
        "--report", str(tmp_path / "r.md"), "--results", str(tmp_path / "r.json"),
    )
    assert result.returncode == ExitCode.OK
    assert "were skipped" in result.stderr
    assert "The report covers 10 cases" in result.stderr
    assert "| test cases | 10 |" in (tmp_path / "r.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("alpha", ["0", "1.5", "-0.2"])
def test_alpha_outside_a_probability_is_rejected(alpha: str) -> None:
    result = run(
        "check", "--candidate", str(V2), "--baseline", str(V1),
        "--suite", str(SUITE), "--alpha", alpha,
    )
    assert result.returncode == ExitCode.USAGE
