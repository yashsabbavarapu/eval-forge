"""Command line interface: suite generation, comparison and the CI gate."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Final, TypeVar

import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .generator import gemini_available, generate, load_spec
from .judge import DeterministicJudge, GeminiJudge, PairwiseJudge
from .models import BenchmarkReport, CaseResult, CompletionSet, TaskSpec, TestSuite
from .runner import dump_json, run_comparison, run_completions
from .stats import BIAS_ALERT, reliability_warnings

app = typer.Typer(add_completion=False, help="Synthetic benchmark + debiased LLM judge for CI.")
# Report on stdout, diagnostics on stderr: a CI job can pipe the scorecard
# somewhere useful without swallowing the reason it failed.
console = Console()
# soft_wrap keeps a diagnostic on one line. Rich otherwise hard-wraps at 80
# columns when stderr is not a TTY, which silently breaks `grep` in CI logs.
err_console = Console(stderr=True, soft_wrap=True)

M = TypeVar("M", bound=BaseModel)


class ExitCode(IntEnum):
    """Exit contract for CI.

    A gate is only useful if "your prompt got worse" is distinguishable from "the
    harness broke". Collapsing both onto 1 means a typo in a path fails the build
    with a regression message and sends someone hunting a regression that does not
    exist - so failures get their own codes.
    """

    OK = 0
    REGRESSION = 1
    USAGE = 2  # emitted by the argument parser itself
    INPUT = 3
    INTERNAL = 4


class InputError(Exception):
    """A file is missing, unreadable, or does not contain what it should."""


def _load(path: Path, model: type[M]) -> M:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc.strerror or exc}") from exc
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        raise InputError(
            f"{path} is not a valid {model.__name__} ({exc.error_count()} problem(s)); "
            f"first: {exc.errors()[0].get('msg', 'unknown')}"
        ) from exc
    except ValueError as exc:
        raise InputError(f"{path} is not valid JSON: {exc}") from exc


def _spec(path: Path) -> TaskSpec:
    try:
        return load_spec(str(path))
    except OSError as exc:
        raise InputError(f"cannot read spec {path}: {exc.strerror or exc}") from exc
    except (ValueError, ValidationError) as exc:
        raise InputError(f"{path} is not a valid TaskSpec: {exc}") from exc


def _read_prompt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read prompt file {path}: {exc.strerror or exc}") from exc
    if not text.strip():
        raise InputError(f"prompt file {path} is empty")
    return text


def _resolve_workers(requested: int | None, *, online: bool) -> int:
    """Default concurrency.

    Online runs are I/O-bound and quota-limited, so a small pool helps and a big
    one just earns 429s. The offline path is pure CPU: under the GIL a pool adds
    a lock acquisition per task and no parallelism, so it runs serially.
    """
    if requested is not None:
        return requested
    return 2 if online else 1


def _version(value: bool) -> None:
    if value:
        console.print(f"evalforge {__version__}")
        raise typer.Exit(code=ExitCode.OK)


@app.callback()
def _root(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """Synthetic benchmark generation, debiased LLM judging and CI regression gating."""


def _verdict(report: BenchmarkReport) -> tuple[str, str]:
    if report.positional_bias_rate > BIAS_ALERT:
        return "JUDGE UNRELIABLE (positional bias)", "magenta"
    if not report.is_statistically_significant:
        return "NO SIGNIFICANT CHANGE", "yellow"
    return ("CANDIDATE WINS", "green") if report.mean_score_delta > 0 else ("REGRESSION", "red")


CATEGORY_HEADER: Final = ("category", "n", "candidate", "baseline", "tie", "mean delta")


def _metrics(report: BenchmarkReport) -> list[tuple[str, str]]:
    """The executive summary rows, shared by the terminal and markdown renderers."""
    return [
        ("test cases", str(report.total_cases)),
        (
            "candidate win rate",
            f"{report.candidate_win_rate:.1%} "
            f"(95% CI {report.win_rate_ci_low:.1%} - {report.win_rate_ci_high:.1%})",
        ),
        ("baseline win rate", f"{report.baseline_win_rate:.1%}"),
        ("tie rate", f"{report.tie_rate:.1%}"),
        (
            "mean rubric delta",
            f"{report.mean_score_delta:+.3f} "
            f"(95% bootstrap CI {report.delta_ci_low:+.3f} - {report.delta_ci_high:+.3f})",
        ),
        ("permutation p-value", f"{report.p_value:.4f} (alpha = {report.alpha})"),
        ("statistically significant", "yes" if report.is_statistically_significant else "no"),
        ("Cohen's kappa (order vs swapped)", f"{report.cohen_kappa:.3f}"),
        ("positional bias rate", f"{report.positional_bias_rate:.1%}"),
        ("offline-fallback rate", f"{report.fallback_rate:.1%}"),
    ]


def _category_rows(report: BenchmarkReport) -> list[tuple[str, ...]]:
    return [
        (
            name,
            f"{s['n']:.0f}",
            f"{s['candidate_win_rate']:.0%}",
            f"{s['baseline_win_rate']:.0%}",
            f"{s['tie_rate']:.0%}",
            f"{s['mean_delta']:+.2f}",
        )
        for name, s in report.category_breakdown.items()
    ]


def _render(report: BenchmarkReport, title: str) -> None:
    headline, colour = _verdict(report)
    console.print(f"\n[bold {colour}]{headline}[/] - {title}\n")
    table = Table(show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for name, value in _metrics(report):
        table.add_row(name, value)
    console.print(table)
    if report.category_breakdown:
        cat = Table(show_header=True, header_style="bold", title="by category")
        for column in CATEGORY_HEADER:
            cat.add_column(column, justify="left" if column == "category" else "right")
        for row in _category_rows(report):
            cat.add_row(*row)
        console.print(cat)
    for warning in reliability_warnings(report):
        err_console.print(f"[yellow]![/] {warning}")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _markdown(report: BenchmarkReport, results: Sequence[CaseResult], title: str) -> str:
    headline, _ = _verdict(report)
    top: dict[tuple[str, str], int] = {}
    for failure in report.failures:
        key = (failure.side, failure.constraint)
        top[key] = top.get(key, 0) + 1
    lines = [
        f"# eval-forge scorecard - {title}",
        "",
        f"**Verdict: {headline}** (permutation p = {report.p_value:.4f}, alpha = {report.alpha})",
        "",
        "## Executive summary",
        "",
        *_table(("metric", "value"), _metrics(report)),
        "",
        "## Breakdown by category",
        "",
        *_table(CATEGORY_HEADER, _category_rows(report)),
    ]
    if caveats := reliability_warnings(report):
        lines += ["", "## Reliability caveats", "", *(f"- {c}" for c in caveats)]
    lines += ["", "## Constraint failures", ""]
    if top:
        lines += _table(
            ("side", "constraint", "violations"),
            [(side, constraint, str(n)) for (side, constraint), n in
             sorted(top.items(), key=lambda kv: -kv[1])],
        )
    else:
        lines.append("No machine-checkable constraint violations on either side.")
    losses = sorted((r for r in results if r.winner.value == "BASELINE"), key=lambda r: r.delta)[:5]
    lines += ["", "## Worst candidate regressions", ""]
    if losses:
        for result in losses:
            lines.append(
                f"- `{result.case_id}` ({result.category.value}, delta {result.delta:+.2f}) "
                f"- {result.reasoning}"
            )
    else:
        lines.append("None: the candidate did not lose any case after debiasing.")
    lines += [
        "",
        "---",
        "",
        "_Two judge passes per case with positions swapped; a preference that tracks the slot "
        "rather than the text is nullified to a TIE._",
        "",
    ]
    return "\n".join(lines)


def _judge(use_gemini: bool, tie_epsilon: float, position_prior: float) -> PairwiseJudge:
    backend = (
        GeminiJudge(tie_epsilon=tie_epsilon)
        if use_gemini and gemini_available()
        else DeterministicJudge(tie_epsilon=tie_epsilon, position_prior=position_prior)
    )
    return PairwiseJudge(backend, tie_epsilon=tie_epsilon)


def _finish(
    results: Sequence[CaseResult],
    report: BenchmarkReport,
    title: str,
    out_md: Path,
    out_json: Path | None,
    fail_on_regression: bool,
) -> None:
    _render(report, title)
    out_md.write_text(_markdown(report, results, title), encoding="utf-8")
    console.print(f"\n[dim]scorecard ->[/] {out_md}")
    if out_json is not None:
        dump_json(
            str(out_json),
            {
                "report": report.model_dump(mode="json"),
                "results": [r.model_dump(mode="json") for r in results],
            },
        )
        console.print(f"[dim]results   ->[/] {out_json}")
    if fail_on_regression and report.is_statistically_significant and report.mean_score_delta < 0:
        err_console.print("[bold red]CI gate failed: statistically significant regression.[/]")
        raise typer.Exit(code=ExitCode.REGRESSION)


@app.command("generate")
def generate_suite(
    spec: Annotated[Path, typer.Option("--spec", help="Seed task specification JSON.")],
    count: Annotated[int, typer.Option("--count", min=1, help="Number of test cases.")] = 20,
    out: Annotated[
        Path, typer.Option("--out", help="Where to write the suite.")
    ] = Path("test_suite.json"),
    seed: Annotated[int, typer.Option("--seed")] = 1337,
    offline: Annotated[
        bool, typer.Option("--offline/--online", help="Force the deterministic generator.")
    ] = False,
) -> None:
    """Synthesise a test suite from a task specification."""
    suite = generate(_spec(spec), count, seed=seed, offline=offline)
    dump_json(str(out), suite.model_dump(mode="json"))
    source = "gemini" if (not offline and gemini_available()) else "offline deterministic"
    console.print(f"[green]wrote[/] {len(suite.cases)} cases to {out} [dim]({source})[/]")


@app.command()
def compare(
    suite: Annotated[Path, typer.Option("--suite")],
    baseline: Annotated[Path, typer.Option("--baseline", help="Baseline completions JSON.")],
    candidate: Annotated[Path, typer.Option("--candidate", help="Candidate completions JSON.")],
    report_path: Annotated[
        Path, typer.Option("--report", help="Markdown scorecard path.")
    ] = Path("eval_report.md"),
    results_path: Annotated[
        Path, typer.Option("--results", help="Structured JSON results path.")
    ] = Path("eval_results.json"),
    alpha: Annotated[
        float, typer.Option("--alpha", min=1e-6, max=0.5, help="Significance threshold.")
    ] = 0.05,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, help="Default: 1 offline (CPU-bound), 2 online.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seeds the bootstrap and permutation test.")] = 7,
    tie_epsilon: Annotated[float, typer.Option("--tie-epsilon")] = 0.15,
    simulate_position_bias: Annotated[
        float,
        typer.Option(
            "--simulate-position-bias",
            help="Inject an artificial slot-A preference to demonstrate debiasing.",
        ),
    ] = 0.0,
    use_gemini: Annotated[bool, typer.Option("--gemini/--no-gemini")] = True,
) -> None:
    """Judge two pre-computed completion sets and emit a scorecard."""
    loaded = _load(suite, TestSuite)
    base = _load(baseline, CompletionSet)
    cand = _load(candidate, CompletionSet)
    online = use_gemini and gemini_available()
    try:
        results, report = run_comparison(
            loaded,
            base,
            cand,
            judge=_judge(use_gemini, tie_epsilon, simulate_position_bias),
            workers=_resolve_workers(workers, online=online),
            alpha=alpha,
            seed=seed,
        )
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    _finish(results, report, f"{cand.label} vs {base.label}", report_path, results_path, True)


@app.command()
def check(
    candidate: Annotated[Path, typer.Option("--candidate", help="Candidate system-prompt file.")],
    baseline: Annotated[Path, typer.Option("--baseline", help="Baseline system-prompt file.")],
    spec: Annotated[Path, typer.Option("--spec")] = Path("fixtures/task_spec.json"),
    suite: Annotated[
        Path | None, typer.Option("--suite", help="Reuse a suite instead of generating one.")
    ] = None,
    count: Annotated[int, typer.Option("--count", min=1)] = 24,
    report_path: Annotated[Path, typer.Option("--report")] = Path("eval_report.md"),
    results_path: Annotated[Path, typer.Option("--results")] = Path("eval_results.json"),
    alpha: Annotated[
        float, typer.Option("--alpha", min=1e-6, max=0.5, help="Significance threshold.")
    ] = 0.05,
    workers: Annotated[
        int | None, typer.Option("--workers", min=1, help="Default: 1 offline (CPU-bound), 2 online.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seeds the bootstrap and permutation test.")] = 7,
    tie_epsilon: Annotated[float, typer.Option("--tie-epsilon")] = 0.15,
    offline: Annotated[bool, typer.Option("--offline/--online")] = False,
    use_gemini: Annotated[bool, typer.Option("--gemini/--no-gemini")] = True,
) -> None:
    """CI gate: compare two system prompts end to end. Exits 1 on a regression."""
    loaded = (
        _load(suite, TestSuite)
        if suite is not None
        else generate(_spec(spec), count, seed=seed, offline=offline)
    )
    if not loaded.cases:
        raise InputError("the test suite contains no cases")
    online = use_gemini and gemini_available()
    resolved = _resolve_workers(workers, online=online)
    base = run_completions(_read_prompt(baseline), loaded, baseline.name, workers=resolved)
    cand = run_completions(_read_prompt(candidate), loaded, candidate.name, workers=resolved)
    try:
        results, report = run_comparison(
            loaded,
            base,
            cand,
            judge=_judge(use_gemini, tie_epsilon, 0.0),
            workers=resolved,
            alpha=alpha,
            seed=seed,
        )
    except ValueError as exc:
        raise InputError(str(exc)) from exc
    _finish(results, report, f"{candidate.name} vs {baseline.name}", report_path, results_path, True)


@app.command()
def show(
    results: Annotated[Path, typer.Option("--results")] = Path("eval_results.json"),
) -> None:
    """Re-render a saved results file."""
    try:
        payload = json.loads(results.read_text(encoding="utf-8"))
        report = BenchmarkReport.model_validate(payload["report"])
    except OSError as exc:
        raise InputError(f"cannot read {results}: {exc.strerror or exc}") from exc
    except (ValueError, KeyError, ValidationError) as exc:
        raise InputError(f"{results} is not an evalforge results file: {exc}") from exc
    _render(report, str(results))


def main() -> None:
    """Entry point that maps failures onto the documented exit contract.

    Click only intercepts its own exceptions, so anything else would otherwise
    reach the shell as a traceback and exit 1 - indistinguishable from the
    regression signal this tool exists to produce.
    """
    try:
        app()
    except SystemExit:
        raise
    except InputError as exc:
        err_console.print(f"[bold red]input error:[/] {escape(str(exc))}")
        raise SystemExit(ExitCode.INPUT) from exc
    except KeyboardInterrupt:
        err_console.print("[yellow]interrupted[/]")
        raise SystemExit(ExitCode.INTERNAL) from None
    except Exception as exc:  # noqa: BLE001 - last resort, mapped to its own code
        if os.environ.get("EVALFORGE_TRACEBACK"):
            raise
        err_console.print(f"[bold red]internal error:[/] {escape(repr(exc))}")
        err_console.print("[dim]set EVALFORGE_TRACEBACK=1 for the full traceback[/]")
        raise SystemExit(ExitCode.INTERNAL) from exc


if __name__ == "__main__":
    main()
