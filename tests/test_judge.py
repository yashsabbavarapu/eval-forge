"""Judge: rubric scoring, constraint checking and positional-swap debiasing."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import pytest

from evalforge import generator
from evalforge.generator import GeminiCallError
from evalforge.judge import (
    DEFAULT_RUBRICS,
    DeterministicJudge,
    GeminiJudge,
    JudgeUnavailable,
    PairwiseJudge,
    check_constraint,
    compliance,
    constraint_failures,
)
from evalforge.models import EvaluationRubric, JudgeVerdict, TestCase, Winner

CASE = TestCase(
    id="t-1",
    prompt="Email the VP about flaky CI pipelines.",
    reference_context="Account: Acme. Problem: flaky pipelines block the release train.",
    expected_constraints=[
        "Under 40 words",
        "Do not mention pricing",
        "Must include the phrase '15 minutes'",
    ],
)

GOOD = (
    "Acme's flaky pipelines block the release train. "
    "Two changes fixed that for a peer team. Worth 15 minutes?"
)
BAD = "Our pricing is flexible and I can walk you through the full pricing sheet today. " + (
    "We also have a very long list of things to discuss in detail together. " * 6
)


class AlwaysPrefersA:
    """A judge with a pure slot-A preference and zero regard for the text."""

    def score(
        self,
        case: TestCase,
        answer_a: str,
        answer_b: str,
        rubrics: Sequence[EvaluationRubric],
    ) -> JudgeVerdict:
        return JudgeVerdict(winner=Winner.CANDIDATE, score_a=4.5, score_b=2.5, reasoning="slot A")


class AlwaysPrefersB:
    def score(
        self,
        case: TestCase,
        answer_a: str,
        answer_b: str,
        rubrics: Sequence[EvaluationRubric],
    ) -> JudgeVerdict:
        return JudgeVerdict(winner=Winner.BASELINE, score_a=1.0, score_b=4.0, reasoning="slot B")


class LongerWins:
    """Verbosity bias: whichever answer has more words wins."""

    def score(
        self,
        case: TestCase,
        answer_a: str,
        answer_b: str,
        rubrics: Sequence[EvaluationRubric],
    ) -> JudgeVerdict:
        a, b = float(len(answer_a.split())), float(len(answer_b.split()))
        return JudgeVerdict(
            winner=Winner.CANDIDATE if a > b else Winner.BASELINE,
            score_a=min(5.0, 1.0 + a / 40),
            score_b=min(5.0, 1.0 + b / 40),
            reasoning="length",
        )


# --- constraint checking ---------------------------------------------------


@pytest.mark.parametrize(
    ("constraint", "text", "expected"),
    [
        ("Under 40 words", "three short words", True),
        ("Under 3 words", "one two three four", False),
        ("At least 5 words", "too short", False),
        ("Do not mention pricing", "no forbidden term here", True),
        ("Do not mention pricing", "our PRICING is great", False),
        ("Never mention 'discount'", "a big discount awaits", False),
        ("Must include the phrase '15 minutes'", "got 15 minutes?", True),
        ("Must include the phrase '15 minutes'", "got a moment?", False),
    ],
)
def test_check_constraint(constraint: str, text: str, expected: bool) -> None:
    assert check_constraint(constraint, text)[0] is expected


def test_unverifiable_constraints_are_reported_as_none() -> None:
    assert check_constraint("Sound warm and human", "anything")[0] is None


def test_compliance_ignores_unverifiable_constraints() -> None:
    case = CASE.model_copy(update={"expected_constraints": ["Sound warm", "Under 3 words"]})
    ratio, failures = compliance(case, "one two three")
    assert ratio == 1.0
    assert failures == []


def test_compliance_reports_each_violation() -> None:
    ratio, failures = compliance(CASE, BAD)
    assert ratio == 0.0
    assert len(failures) == 3


def test_constraint_failures_attribute_the_right_side() -> None:
    failures = constraint_failures(CASE, candidate=GOOD, baseline=BAD)
    assert {f.side for f in failures} == {"baseline"}
    assert all(f.case_id == CASE.id for f in failures)


# --- rubric scoring --------------------------------------------------------


def test_deterministic_judge_prefers_the_compliant_answer() -> None:
    verdict = DeterministicJudge().score(CASE, GOOD, BAD, DEFAULT_RUBRICS)
    assert verdict.winner is Winner.CANDIDATE
    assert verdict.score_a > verdict.score_b
    assert {r.criteria_name for r in verdict.rubric_scores} == {
        "accuracy",
        "constraint_compliance",
        "conciseness",
    }
    assert all(1.0 <= s <= 5.0 for r in verdict.rubric_scores for s in (r.score_a, r.score_b))


def test_identical_answers_tie() -> None:
    assert DeterministicJudge().score(CASE, GOOD, GOOD, DEFAULT_RUBRICS).winner is Winner.TIE


# --- positional swap debiasing --------------------------------------------


def test_swap_catches_artificial_slot_a_preference() -> None:
    result = PairwiseJudge(AlwaysPrefersA()).evaluate(CASE, GOOD, BAD)
    assert result.positional_bias is True
    assert result.winner is Winner.TIE
    # The averaged scores cancel exactly, so no phantom margin survives.
    assert result.candidate_score == pytest.approx(result.baseline_score)
    assert result.delta == pytest.approx(0.0)
    assert "positional bias" in result.reasoning


def test_swap_catches_artificial_slot_b_preference() -> None:
    result = PairwiseJudge(AlwaysPrefersB()).evaluate(CASE, GOOD, BAD)
    assert result.positional_bias is True
    assert result.winner is Winner.TIE


def test_second_pass_is_flagged_as_swapped() -> None:
    result = PairwiseJudge(AlwaysPrefersA()).evaluate(CASE, GOOD, BAD)
    assert result.pass1.position_swapped is False
    assert result.pass2.position_swapped is True


def test_content_driven_preference_survives_the_swap() -> None:
    result = PairwiseJudge().evaluate(CASE, GOOD, BAD)
    assert result.positional_bias is False
    assert result.winner is Winner.CANDIDATE
    assert result.delta > 0


def test_direction_is_not_inverted_by_the_swap() -> None:
    result = PairwiseJudge().evaluate(CASE, BAD, GOOD)
    assert result.winner is Winner.BASELINE
    assert result.delta < 0


def test_verbosity_bias_is_content_driven_not_positional() -> None:
    # A length-biased judge is consistent across orderings: it is a real
    # (if wrong-headed) preference, so debiasing must not mask it as a tie.
    result = PairwiseJudge(LongerWins()).evaluate(CASE, GOOD, BAD)
    assert result.positional_bias is False
    assert result.winner is Winner.BASELINE


def test_injected_position_prior_flips_a_genuine_preference() -> None:
    clean = PairwiseJudge(DeterministicJudge()).evaluate(CASE, GOOD, BAD)
    assert clean.winner is Winner.CANDIDATE

    # A prior larger than the true margin turns the raw judge into a slot-A
    # voter; the swap must detect that and refuse to call a winner.
    biased = PairwiseJudge(DeterministicJudge(position_prior=4.5)).evaluate(CASE, GOOD, BAD)
    assert biased.positional_bias is True
    assert biased.winner is Winner.TIE


def test_tie_epsilon_widens_the_tie_band() -> None:
    narrow = PairwiseJudge(DeterministicJudge(tie_epsilon=0.01)).evaluate(CASE, GOOD, BAD)
    wide = PairwiseJudge(DeterministicJudge(tie_epsilon=4.9), tie_epsilon=4.9).evaluate(
        CASE, GOOD, BAD
    )
    assert narrow.winner is Winner.CANDIDATE
    assert wide.winner is Winner.TIE


# --- online-path degradation ----------------------------------------------


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/x")
    return httpx.HTTPStatusError(
        f"{status}", request=request, response=httpx.Response(status, request=request)
    )


@pytest.mark.parametrize("status", [429, 500, 404])
def test_transport_errors_are_wrapped_at_the_boundary(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    # The original bug: httpx.HTTPStatusError is not an OSError, so a handler
    # naming only builtins let a 429 escape and kill the whole run. gemini_json
    # now converts every transport failure into one type, so callers cannot miss
    # a case - and nothing outside this module needs to import httpx.
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("evalforge.generator.time.sleep", lambda _s: None)

    def post(*_args: object, **_kwargs: object) -> httpx.Response:
        return httpx.Response(status, request=httpx.Request("POST", "https://x.invalid"))

    monkeypatch.setattr("httpx.post", post)
    with pytest.raises(GeminiCallError):
        generator.gemini_json("p", {"type": "OBJECT"}, attempts=1)


@pytest.mark.parametrize("status", [429, 500, 404])
def test_gemini_judge_signals_unavailable_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")

    def boom(*_args: object, **_kwargs: object) -> object:
        raise GeminiCallError(f"HTTP {status}")

    monkeypatch.setattr("evalforge.judge.gemini_json", boom)
    with pytest.raises(JudgeUnavailable):
        GeminiJudge().score(CASE, GOOD, BAD, DEFAULT_RUBRICS)


def test_gemini_judge_signals_unavailable_on_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")

    def boom(*_args: object, **_kwargs: object) -> object:
        raise GeminiCallError("transport failure: no route")

    monkeypatch.setattr("evalforge.judge.gemini_json", boom)
    with pytest.raises(JudgeUnavailable):
        GeminiJudge().score(CASE, GOOD, BAD, DEFAULT_RUBRICS)


def test_unparseable_payload_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")

    def post(*_args: object, **_kwargs: object) -> httpx.Response:
        return httpx.Response(
            200, json={"unexpected": "shape"}, request=httpx.Request("POST", "https://x.invalid")
        )

    monkeypatch.setattr("httpx.post", post)
    with pytest.raises(GeminiCallError, match="unparseable"):
        generator.gemini_json("p", {"type": "OBJECT"})


def test_gemini_judge_signals_unavailable_on_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("evalforge.judge.gemini_json", lambda *a, **k: {"nonsense": 1})
    with pytest.raises(JudgeUnavailable):
        GeminiJudge().score(CASE, GOOD, BAD, DEFAULT_RUBRICS)


def test_no_key_uses_the_offline_judge_for_the_whole_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No key is a configuration choice, not a mid-run failure: it is consistent
    # across every case, so it needs no fallback bookkeeping.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert GeminiJudge().score(CASE, GOOD, BAD, DEFAULT_RUBRICS).winner is Winner.CANDIDATE


def test_gemini_judge_uses_the_live_verdict_when_the_call_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(
        "evalforge.judge.gemini_json",
        lambda *a, **k: {"score_a": 2.0, "score_b": 4.5, "reasoning": "B is tighter"},
    )
    verdict = GeminiJudge().score(CASE, GOOD, BAD, DEFAULT_RUBRICS)
    assert verdict.winner is Winner.BASELINE
    assert verdict.reasoning == "B is tighter"


def test_gemini_judge_clamps_out_of_range_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(
        "evalforge.judge.gemini_json",
        lambda *a, **k: {"score_a": 99.0, "score_b": -4.0, "reasoning": "r"},
    )
    verdict = GeminiJudge().score(CASE, GOOD, BAD, DEFAULT_RUBRICS)
    assert (verdict.score_a, verdict.score_b) == (5.0, 1.0)


def test_gemini_json_retries_transient_status_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("evalforge.generator.time.sleep", lambda _s: None)
    calls: list[int] = []

    def post(*_args: object, **_kwargs: object) -> httpx.Response:
        calls.append(1)
        request = httpx.Request("POST", "https://example.invalid")
        if len(calls) < 3:
            return httpx.Response(429, request=request)
        body = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr("httpx.post", post)
    assert generator.gemini_json("p", {"type": "OBJECT"}) == {"ok": True}
    assert len(calls) == 3


def test_gemini_json_gives_up_after_the_attempt_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("evalforge.generator.time.sleep", lambda _s: None)

    def post(*_args: object, **_kwargs: object) -> httpx.Response:
        request = httpx.Request("POST", "https://example.invalid")
        return httpx.Response(429, request=request)

    monkeypatch.setattr("httpx.post", post)
    with pytest.raises(GeminiCallError):
        generator.gemini_json("p", {"type": "OBJECT"}, attempts=2)


def test_gemini_json_does_not_retry_a_permanent_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("evalforge.generator.time.sleep", lambda _s: None)
    calls: list[int] = []

    def post(*_args: object, **_kwargs: object) -> httpx.Response:
        calls.append(1)
        request = httpx.Request("POST", "https://example.invalid")
        return httpx.Response(404, request=request)

    monkeypatch.setattr("httpx.post", post)
    with pytest.raises(GeminiCallError):
        generator.gemini_json("p", {"type": "OBJECT"})
    assert len(calls) == 1, "a retired model or bad key must not burn the retry budget"


def test_gemini_json_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # GeminiCallError subclasses RuntimeError, so existing handlers still catch it.
    with pytest.raises(RuntimeError):
        generator.gemini_json("p", {"type": "OBJECT"})


def _quota_response(quota_id: str, retry_delay: str | None = None) -> httpx.Response:
    details: list[dict[str, object]] = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id, "quotaValue": "20"}],
        }
    ]
    if retry_delay is not None:
        details.append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
        )
    return httpx.Response(
        429,
        json={"error": {"code": 429, "details": details}},
        request=httpx.Request("POST", "https://example.invalid"),
    )


def test_daily_quota_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    # A per-day cap cannot clear inside a run; sleeping through it wastes minutes
    # and still ends in the offline fallback.
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("evalforge.generator.time.sleep", lambda _s: None)
    calls: list[int] = []

    def post(*_args: object, **_kwargs: object) -> httpx.Response:
        calls.append(1)
        return _quota_response("GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    monkeypatch.setattr("httpx.post", post)
    with pytest.raises(GeminiCallError):
        generator.gemini_json("p", {"type": "OBJECT"})
    assert len(calls) == 1


def test_per_minute_quota_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("evalforge.generator.time.sleep", lambda _s: None)
    calls: list[int] = []

    def post(*_args: object, **_kwargs: object) -> httpx.Response:
        calls.append(1)
        if len(calls) < 2:
            return _quota_response("GenerateRequestsPerMinutePerProject-FreeTier", "3s")
        body = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", "https://x.invalid"))

    monkeypatch.setattr("httpx.post", post)
    assert generator.gemini_json("p", {"type": "OBJECT"}) == {"ok": True}
    assert len(calls) == 2


def test_retry_delay_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("evalforge.generator.time.sleep", lambda s: slept.append(s))
    assert generator._backoff(_quota_response("PerMinute", "7s"), 0) == 7.0
    assert generator._backoff(_quota_response("PerDay"), 0) is None
    assert generator._backoff(
        httpx.Response(200, request=httpx.Request("POST", "https://x.invalid")), 0
    ) is None


def test_warn_once_deduplicates(capsys: pytest.CaptureFixture[str]) -> None:
    generator._warned.clear()
    for _ in range(5):
        generator.warn_once("same message")
    generator.warn_once("different message")
    assert capsys.readouterr().err.count("same message") == 1


class FailsOnSecondPass:
    """Succeeds once, then goes down - the shape of a mid-run quota exhaustion."""

    def __init__(self) -> None:
        self.calls = 0

    def score(
        self,
        case: TestCase,
        answer_a: str,
        answer_b: str,
        rubrics: Sequence[EvaluationRubric],
    ) -> JudgeVerdict:
        self.calls += 1
        if self.calls > 1:
            raise JudgeUnavailable("quota exhausted")
        return JudgeVerdict(winner=Winner.CANDIDATE, score_a=5.0, score_b=1.0, reasoning="online")


def test_a_half_failed_case_is_redone_entirely_offline() -> None:
    # Pairing one surviving online pass with an offline one would make the swap a
    # comparison of two raters rather than two orderings.
    backend = FailsOnSecondPass()
    result = PairwiseJudge(backend).evaluate(CASE, GOOD, BAD)
    assert result.used_fallback is True
    assert "online" not in result.pass1.reasoning
    assert result.pass1.reasoning.startswith("A/B ")
    assert result.pass2.reasoning.startswith("A/B ")
    assert result.winner is Winner.CANDIDATE


def test_a_fully_online_case_is_not_marked_as_fallback() -> None:
    assert PairwiseJudge(DeterministicJudge()).evaluate(CASE, GOOD, BAD).used_fallback is False


def test_fallback_rate_reaches_the_report() -> None:
    from evalforge.stats import build_report, reliability_warnings

    results = [
        PairwiseJudge(FailsOnSecondPass()).evaluate(CASE, GOOD, BAD) for _ in range(4)
    ] + [PairwiseJudge(DeterministicJudge()).evaluate(CASE, GOOD, BAD) for _ in range(4)]
    report = build_report(results)
    assert report.fallback_rate == pytest.approx(0.5)
    assert any("offline judge" in w for w in reliability_warnings(report))
