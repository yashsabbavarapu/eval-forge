"""Debiased pairwise LLM-as-a-judge.

Every case is judged twice with the two candidates in opposite slots. A verdict
is only trusted when it survives the swap; a preference that tracks the *slot*
rather than the *text* is reported as positional bias and collapsed to a TIE.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Final, Protocol, cast

from .generator import GEMINI_ERRORS, gemini_available, gemini_json, warn_once
from .models import (
    CaseResult,
    Category,
    ConstraintFailure,
    EvaluationRubric,
    JudgeVerdict,
    RubricScore,
    TestCase,
    Winner,
)

DEFAULT_RUBRICS: Final[tuple[EvaluationRubric, ...]] = (
    EvaluationRubric(
        criteria_name="accuracy",
        description="Claims are grounded in the reference context; nothing is invented.",
        weight=1.0,
    ),
    EvaluationRubric(
        criteria_name="constraint_compliance",
        description="Every stated constraint (negative, style and length) is respected.",
        weight=1.5,
    ),
    EvaluationRubric(
        criteria_name="conciseness",
        description="Says what is needed without padding, at the requested length.",
        weight=0.75,
    ),
)

WORD_RE: Final = re.compile(r"[A-Za-z0-9'$%]+")
QUOTED_RE: Final = re.compile(r"['\"]([^'\"]+)['\"]")
MAX_WORDS_RE: Final = re.compile(
    r"(?:under|at most|no more than|fewer than|max(?:imum)?(?: of)?)\s+(\d+)\s+words"
)
MIN_WORDS_RE: Final = re.compile(r"(?:at least|more than|min(?:imum)?(?: of)?)\s+(\d+)\s+words")
FORBID_RE: Final = re.compile(
    r"(?:do not|do n't|don't|never|avoid|refrain from)\s+"
    r"(?:mention|use|includ|referenc|discuss|quot|stat)\w*\s+(.+)"
)
REQUIRE_RE: Final = re.compile(
    r"(?:must|should|always)\s+(?:mention|include|use|reference|contain|end with)\s+(.+)"
)

_STOP: Final = frozenset(
    "the a an any of to in on for and or that this it its their with as is are be "
    "phrase word words exact".split()
)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def extract_terms(tail: str, *, quoted_only: bool) -> list[str]:
    """Pull checkable literal terms out of a constraint's object clause."""
    quoted = [t.strip().lower() for t in QUOTED_RE.findall(tail) if t.strip()]
    if quoted or quoted_only:
        return quoted
    content = [w for w in words(tail.lower()) if w not in _STOP]
    return content if 1 <= len(content) <= 3 else []


def check_constraint(constraint: str, text: str) -> tuple[bool | None, str]:
    """Check one constraint. Returns ``(passed, detail)``; ``None`` = not checkable."""
    return _check(constraint, text.lower(), len(words(text)))


def _check(constraint: str, body: str, n: int) -> tuple[bool | None, str]:
    """Inner check over a pre-lowered body and pre-computed word count.

    ``compliance`` walks every constraint of a case against the same answer, so
    tokenising the answer once per constraint was quadratic in the constraint count.
    """
    low = constraint.strip().rstrip(".").lower()
    if (m := MAX_WORDS_RE.search(low)) is not None:
        limit = int(m.group(1))
        return n <= limit, f"{n} words (limit {limit})"
    if (m := MIN_WORDS_RE.search(low)) is not None:
        floor = int(m.group(1))
        return n >= floor, f"{n} words (floor {floor})"
    if (m := FORBID_RE.search(low)) is not None:
        if terms := extract_terms(m.group(1), quoted_only=False):
            hits = [t for t in terms if t in body]
            return not hits, "absent" if not hits else f"found {hits}"
    if (m := REQUIRE_RE.search(low)) is not None:
        if terms := extract_terms(m.group(1), quoted_only=True):
            missing = [t for t in terms if t not in body]
            return not missing, "present" if not missing else f"missing {missing}"
    return None, "not machine-checkable"


@lru_cache(maxsize=4096)
def _compliance(constraints: tuple[str, ...], text: str) -> tuple[float, tuple[tuple[str, str], ...]]:
    body = text.lower()
    n = len(words(text))
    checked = 0
    passed = 0
    failures: list[tuple[str, str]] = []
    for constraint in constraints:
        ok, detail = _check(constraint, body, n)
        if ok is None:
            continue
        checked += 1
        if ok:
            passed += 1
        else:
            failures.append((constraint, detail))
    return (passed / checked if checked else 1.0), tuple(failures)


def compliance(case: TestCase, text: str) -> tuple[float, list[tuple[str, str]]]:
    """Fraction of machine-checkable constraints satisfied, plus the failures.

    Memoised: the position swap scores each answer twice and the failure
    breakdown asks a third time, always with identical inputs.
    """
    ratio, failures = _compliance(tuple(case.expected_constraints), text)
    return ratio, list(failures)


@lru_cache(maxsize=1024)
def _reference_terms(reference: str) -> frozenset[str]:
    """Content terms of a case's reference context; stable across all its answers."""
    return frozenset(w for w in words(reference.lower()) if w not in _STOP and len(w) > 3)


def _grounding(text: str, reference: str) -> float:
    ref = _reference_terms(reference)
    if not ref:
        return 0.6
    return len(ref & set(words(text.lower()))) / len(ref)


@lru_cache(maxsize=4096)
def _target_length(constraints: tuple[str, ...]) -> int:
    """The strictest word budget the constraints impose, or a default if none do."""
    limits = [
        int(m.group(1))
        for constraint in constraints
        if (m := MAX_WORDS_RE.search(constraint.lower())) is not None
    ]
    return min(limits) if limits else 110


def _clamp(value: float, low: float = 1.0, high: float = 5.0) -> float:
    return max(low, min(high, value))


@lru_cache(maxsize=4096)
def _rubric_scores(constraints: tuple[str, ...], reference: str, text: str) -> tuple[float, float, float]:
    ratio, _ = _compliance(constraints, text)
    n = len(words(text))
    target = _target_length(constraints)
    length_ratio = n / target if target else 1.0
    if n == 0:
        conciseness = 1.0
    elif length_ratio <= 1.0:
        conciseness = 5.0 - 4.0 * max(0.0, 0.3 - length_ratio) / 0.3
    else:
        conciseness = 5.0 - 4.0 * min(1.0, length_ratio - 1.0)
    return (
        _clamp(1.0 + 4.0 * min(1.0, _grounding(text, reference) / 0.6)),
        _clamp(1.0 + 4.0 * ratio),
        _clamp(conciseness),
    )


def rubric_scores(case: TestCase, text: str) -> dict[str, float]:
    """Deterministic 1-5 scores per rubric dimension for a single answer.

    Memoised on the values that actually determine the result, so the second
    (position-swapped) pass reuses the first pass's work instead of repeating it.
    """
    accuracy, compliance_score, conciseness = _rubric_scores(
        tuple(case.expected_constraints), case.reference_context, text
    )
    return {
        "accuracy": accuracy,
        "constraint_compliance": compliance_score,
        "conciseness": conciseness,
    }


def weighted(scores: dict[str, float], rubrics: Sequence[EvaluationRubric]) -> float:
    total = sum(r.weight for r in rubrics) or 1.0
    return sum(scores.get(r.criteria_name, 3.0) * r.weight for r in rubrics) / total


class JudgeBackend(Protocol):
    """Scores one ordered presentation ``(answer_a, answer_b)``."""

    def score(
        self,
        case: TestCase,
        answer_a: str,
        answer_b: str,
        rubrics: Sequence[EvaluationRubric],
    ) -> JudgeVerdict: ...


class DeterministicJudge:
    """Offline rubric judge.

    ``position_prior`` is a deliberate thumb on the scale for slot A. It is 0.0
    in real runs and only raised to *manufacture* positional bias so the
    debiasing machinery can be exercised in tests and demos. It is added after
    clamping, so a large prior can push ``score_a`` past the rubric ceiling.
    """

    def __init__(self, *, tie_epsilon: float = 0.15, position_prior: float = 0.0) -> None:
        self.tie_epsilon = tie_epsilon
        self.position_prior = position_prior

    def score(
        self,
        case: TestCase,
        answer_a: str,
        answer_b: str,
        rubrics: Sequence[EvaluationRubric],
    ) -> JudgeVerdict:
        sa_dims = rubric_scores(case, answer_a)
        sb_dims = rubric_scores(case, answer_b)
        sa = _clamp(weighted(sa_dims, rubrics)) + self.position_prior
        sb = _clamp(weighted(sb_dims, rubrics))
        margin = sa - sb
        winner = (
            Winner.TIE
            if abs(margin) < self.tie_epsilon
            else (Winner.CANDIDATE if margin > 0 else Winner.BASELINE)
        )
        detail = " ".join(
            f"{r.criteria_name}={sa_dims[r.criteria_name]:.2f}/{sb_dims[r.criteria_name]:.2f}"
            for r in rubrics
            if r.criteria_name in sa_dims
        )
        return JudgeVerdict(
            winner=winner,
            score_a=round(sa, 4),
            score_b=round(sb, 4),
            reasoning=f"A/B {detail} margin={margin:+.2f}",
            rubric_scores=[
                RubricScore(
                    criteria_name=r.criteria_name,
                    score_a=sa_dims[r.criteria_name],
                    score_b=sb_dims[r.criteria_name],
                )
                for r in rubrics
                if r.criteria_name in sa_dims
            ],
        )


_VERDICT_SCHEMA: Final[dict[str, Any]] = {
    "type": "OBJECT",
    "properties": {
        "score_a": {"type": "NUMBER"},
        "score_b": {"type": "NUMBER"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["score_a", "score_b", "reasoning"],
}


class JudgeUnavailable(RuntimeError):
    """The online judge could not score a presentation.

    Raised rather than quietly substituting another judge: the two passes of a
    comparison must come from the *same* rater, or the swap measures the
    difference between two judges instead of the effect of position.
    """


class GeminiJudge:
    """Gemini Flash rubric judge.

    With no API key configured at all it delegates to :class:`DeterministicJudge`
    for the whole run, which is consistent. A mid-run API failure instead raises
    :class:`JudgeUnavailable` so the caller can redo *both* passes offline.
    """

    def __init__(self, *, tie_epsilon: float = 0.15) -> None:
        self.tie_epsilon = tie_epsilon
        self._fallback = DeterministicJudge(tie_epsilon=tie_epsilon)

    def score(
        self,
        case: TestCase,
        answer_a: str,
        answer_b: str,
        rubrics: Sequence[EvaluationRubric],
    ) -> JudgeVerdict:
        if not gemini_available():
            return self._fallback.score(case, answer_a, answer_b, rubrics)
        rubric_text = "\n".join(f"- {r.criteria_name} (weight {r.weight}): {r.description}" for r in rubrics)
        prompt = (
            "You are a strict evaluation judge. Score two candidate answers on a calibrated "
            "1-5 scale per dimension, then return the weighted mean for each.\n"
            "Judge only the text. Length is not quality. Never prefer an answer for being first.\n\n"
            f"RUBRIC:\n{rubric_text}\n\nTASK PROMPT:\n{case.prompt}\n\n"
            f"REFERENCE CONTEXT:\n{case.reference_context}\n\n"
            f"CONSTRAINTS:\n{json.dumps(case.expected_constraints)}\n\n"
            f"ANSWER A:\n{answer_a}\n\nANSWER B:\n{answer_b}\n\n"
            "Return score_a and score_b in [1,5] plus one sentence of reasoning."
        )
        try:
            raw = cast(dict[str, Any], gemini_json(prompt, _VERDICT_SCHEMA, temperature=0.0))
            sa, sb = _clamp(float(raw["score_a"])), _clamp(float(raw["score_b"]))
        except GEMINI_ERRORS as exc:
            warn_once(f"Gemini judge failed ({exc!r}); those cases fall back to the offline judge.")
            raise JudgeUnavailable(str(exc)) from exc
        margin = sa - sb
        winner = (
            Winner.TIE
            if abs(margin) < self.tie_epsilon
            else (Winner.CANDIDATE if margin > 0 else Winner.BASELINE)
        )
        return JudgeVerdict(winner=winner, score_a=sa, score_b=sb, reasoning=str(raw["reasoning"]))


def _slot(verdict: JudgeVerdict) -> str:
    """Which *slot* the raw verdict preferred, independent of which text sat there."""
    if verdict.winner is Winner.TIE:
        return "TIE"
    return "A" if verdict.score_a > verdict.score_b else "B"


class PairwiseJudge:
    """Runs both orderings of a comparison and reconciles them."""

    def __init__(
        self,
        backend: JudgeBackend | None = None,
        *,
        rubrics: Sequence[EvaluationRubric] = DEFAULT_RUBRICS,
        tie_epsilon: float = 0.15,
        fallback: JudgeBackend | None = None,
    ) -> None:
        self.backend = backend if backend is not None else DeterministicJudge(tie_epsilon=tie_epsilon)
        self.fallback = fallback if fallback is not None else DeterministicJudge(tie_epsilon=tie_epsilon)
        self.rubrics = list(rubrics)
        self.tie_epsilon = tie_epsilon

    def evaluate(self, case: TestCase, candidate: str, baseline: str) -> CaseResult:
        used_fallback = False
        try:
            v1 = self.backend.score(case, candidate, baseline, self.rubrics)
            raw2 = self.backend.score(case, baseline, candidate, self.rubrics)
        except JudgeUnavailable:
            # Redo BOTH passes on the fallback. Keeping a surviving online pass and
            # pairing it with an offline one would make the swap a comparison of two
            # raters, which is exactly the confound this class exists to remove.
            used_fallback = True
            v1 = self.fallback.score(case, candidate, baseline, self.rubrics)
            raw2 = self.fallback.score(case, baseline, candidate, self.rubrics)
        v2 = raw2.model_copy(update={"position_swapped": True})

        # Project slot-space scores back onto entities.
        cand = (v1.score_a + v2.score_b) / 2.0
        base = (v1.score_b + v2.score_a) / 2.0
        s1, s2 = _slot(v1), _slot(v2)

        # Pass 1 puts the candidate in slot A, pass 2 puts the baseline there.
        e1 = Winner.TIE if s1 == "TIE" else (Winner.CANDIDATE if s1 == "A" else Winner.BASELINE)
        e2 = Winner.TIE if s2 == "TIE" else (Winner.BASELINE if s2 == "A" else Winner.CANDIDATE)

        # Same slot preferred in both orderings => the preference is the position.
        biased = s1 == s2 != "TIE"
        delta = cand - base
        if biased:
            winner, note = Winner.TIE, f"positional bias: slot {s1} preferred in both orderings"
        elif e1 is e2:
            winner, note = e1, f"both orderings agree on {e1.value}"
        else:
            winner = (
                Winner.TIE
                if abs(delta) < self.tie_epsilon
                else (Winner.CANDIDATE if delta > 0 else Winner.BASELINE)
            )
            note = f"one ordering tied; resolved on debiased margin {delta:+.2f}"
        return CaseResult(
            case_id=case.id,
            category=case.category,
            winner=winner,
            candidate_score=round(cand, 4),
            baseline_score=round(base, 4),
            delta=round(delta, 4),
            positional_bias=biased,
            used_fallback=used_fallback,
            reasoning=f"{note} | pass1: {v1.reasoning} | pass2: {v2.reasoning}",
            pass1=v1,
            pass2=v2,
        )


def constraint_failures(case: TestCase, candidate: str, baseline: str) -> list[ConstraintFailure]:
    """Per-side constraint violations, for the scorecard's failure breakdown."""
    out: list[ConstraintFailure] = []
    for side, text in (("candidate", candidate), ("baseline", baseline)):
        _, failures = compliance(case, text)
        out += [
            ConstraintFailure(
                case_id=case.id,
                category=Category(case.category),
                side=cast(Any, side),
                constraint=constraint,
                detail=detail,
            )
            for constraint, detail in failures
        ]
    return out
