"""Batch execution harness: completion synthesis, concurrency, retry, reporting."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Final, TypeVar

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .generator import warn_once
from .judge import (
    FORBID_RE,
    MAX_WORDS_RE,
    REQUIRE_RE,
    PairwiseJudge,
    constraint_failures,
    extract_terms,
    words,
)
from .models import BenchmarkReport, CaseResult, CompletionSet, TestCase, TestSuite
from .stats import build_report

T = TypeVar("T")

_BODY: Final = (
    "Hi there - I work with teams shipping into the same corner you are in with {topic}.",
    "Three of your peers told us the same thing last quarter, so I dug into it.",
    "We handled {topic} for a team your size and cut the manual loop down hard.",
    "No deck, no pitch: just the two changes that moved the needle for them.",
)
_FILLER: Final = (
    "To give the full picture, here is the longer version of how we got there and why it "
    "matters for a team operating at your scale today.",
    "It is worth walking through each step in order, because the sequencing is what makes "
    "the difference between a one-off fix and something durable.",
    "I can also share the internal write-up, the rollout checklist and the follow-up notes "
    "our team kept while the change was landing.",
)
# Strips the template's imperative lead-in so the mock body reads as prose.
_LEAD_RE: Final = re.compile(
    r"^(?:respond about|draft the deliverable for|produce output covering|"
    r"handle this request about|handle)\s+",
    re.IGNORECASE,
)

_STOP_DIRECTIVE: Final = frozenset(
    "do not don't never avoid mention use include contain reference discuss quote state must "
    "should always the a an of to in on for and or under over at least most more than fewer "
    "no phrase word words exact refrain from end with maximum minimum".split()
)


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16)


def _covered(system_prompt: str, constraint: str) -> bool:
    """Does the system prompt actually encode this constraint?"""
    low = system_prompt.lower()
    plain = constraint.lower().replace("'", " ").replace('"', " ")
    terms = [w for w in words(plain) if w not in _STOP_DIRECTIVE]
    if not terms:
        return len(system_prompt) > 200
    return all(t in low for t in terms)


def simulate_completion(system_prompt: str, case: TestCase, *, jitter: int = 0) -> str:
    """Deterministic offline stand-in for a real model call.

    A constraint is obeyed with high probability when the system prompt actually
    states it, and rarely otherwise - so a richer prompt genuinely outperforms a
    thin one, without any network access.
    """
    rng = random.Random(_seed(system_prompt, case.id, str(jitter)))
    topic = _LEAD_RE.sub("", case.prompt.split(".")[0]).strip()[:70]
    sentences = [s.format(topic=topic) for s in _BODY]
    limit: int | None = None
    obey_limit = True
    for constraint in case.expected_constraints:
        low = constraint.lower()
        obey = rng.random() < (0.92 if _covered(system_prompt, constraint) else 0.25)
        if (m := MAX_WORDS_RE.search(low)) is not None:
            found = int(m.group(1))
            if limit is None or found < limit:
                limit, obey_limit = found, obey
        elif (m := FORBID_RE.search(low)) is not None:
            terms = extract_terms(m.group(1), quoted_only=False)
            if terms and not obey:
                sentences.insert(1, f"On {terms[0]}: happy to walk through the {terms[0]} detail up front.")
        elif (m := REQUIRE_RE.search(low)) is not None:
            terms = extract_terms(m.group(1), quoted_only=True)
            if terms and obey:
                sentences.insert(1, f"Worth {terms[0]} of your time - open to it?")
    if limit is not None and not obey_limit:
        sentences += list(_FILLER)
        while len(words(" ".join(sentences))) <= limit:
            sentences.append(_FILLER[len(sentences) % len(_FILLER)])
    text = " ".join(sentences)
    if limit is not None and obey_limit and len(words(text)) > limit:
        text = " ".join(words(text)[:limit])
    return text


def _retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 0.4) -> T:
    """Retry with exponential backoff; re-raises the final failure."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * 2.0**attempt)
    raise AssertionError("unreachable")


def _map(
    fn: Callable[[TestCase], T],
    cases: Sequence[TestCase],
    workers: int,
    label: str,
    show: bool,
) -> list[T]:
    # The offline path is pure CPU, so a thread pool buys nothing under the GIL
    # and costs a lock acquisition per task. Only pay for it when asked.
    if workers <= 1:
        if not show:
            return [fn(case) for case in cases]
    elif not show:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, cases))
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(label, total=len(cases))
        out: list[T] = []
        if workers <= 1:
            for case in cases:
                out.append(fn(case))
                progress.advance(task)
            return out
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for item in pool.map(fn, cases):
                out.append(item)
                progress.advance(task)
        return out


def run_completions(
    system_prompt: str,
    suite: TestSuite,
    label: str,
    *,
    workers: int = 8,
    progress: bool = True,
) -> CompletionSet:
    """Execute every test case against one system prompt."""
    texts = _map(
        lambda case: _retry(lambda: simulate_completion(system_prompt, case)),
        suite.cases,
        workers,
        f"running {label}",
        progress,
    )
    return CompletionSet(
        label=label,
        system_prompt=system_prompt,
        completions={c.id: t for c, t in zip(suite.cases, texts, strict=True)},
    )


def run_comparison(
    suite: TestSuite,
    baseline: CompletionSet,
    candidate: CompletionSet,
    *,
    judge: PairwiseJudge | None = None,
    workers: int = 8,
    progress: bool = True,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[list[CaseResult], BenchmarkReport]:
    """Judge every case in both orderings and aggregate into a report."""
    pair = judge if judge is not None else PairwiseJudge()
    cases = [c for c in suite.cases if c.id in baseline.completions and c.id in candidate.completions]
    if not cases:
        raise ValueError("no test case ids are present in both completion sets")
    # Silently scoring a subset would understate n, widen every interval and make
    # the gate look under-powered for reasons nobody can see in the output.
    if (dropped := len(suite.cases) - len(cases)) > 0:
        missing = [c.id for c in suite.cases if c not in cases][:3]
        warn_once(
            f"{dropped} of {len(suite.cases)} suite cases are missing from one or both "
            f"completion sets and were skipped (e.g. {', '.join(missing)}). "
            f"The report covers {len(cases)} cases."
        )

    def judge_case(case: TestCase) -> CaseResult:
        return _retry(
            lambda: pair.evaluate(case, candidate.completions[case.id], baseline.completions[case.id])
        )

    results = _map(judge_case, cases, workers, "judging (2 passes/case)", progress)
    failures = [
        f
        for case in cases
        for f in constraint_failures(case, candidate.completions[case.id], baseline.completions[case.id])
    ]
    return results, build_report(results, failures, alpha=alpha, seed=seed)


def dump_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
