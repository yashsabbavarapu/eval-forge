"""Typed data contracts for eval-forge (Pydantic v2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Category(StrEnum):
    """Bucket a synthetic test case belongs to."""

    STANDARD = "standard"
    BOUNDARY = "boundary"
    ADVERSARIAL = "adversarial"


class Winner(StrEnum):
    """Outcome of a single debiased pairwise comparison."""

    BASELINE = "BASELINE"
    CANDIDATE = "CANDIDATE"
    TIE = "TIE"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class TaskSpec(_Base):
    """Seed specification the synthetic generator expands into a suite."""

    name: str
    purpose: str
    audience: str
    hard_constraints: list[str] = Field(default_factory=list)
    seed_topics: list[str] = Field(default_factory=list)
    tone: str = ""


class TestCase(_Base):
    id: str
    prompt: str
    reference_context: str = ""
    expected_constraints: list[str] = Field(default_factory=list)
    category: Category = Category.STANDARD


class TestSuite(_Base):
    spec_name: str
    cases: list[TestCase]


class CompletionSet(_Base):
    """Completions produced by one system prompt, keyed by test-case id."""

    label: str
    system_prompt: str = ""
    completions: dict[str, str] = Field(default_factory=dict)


class EvaluationRubric(_Base):
    criteria_name: str
    description: str
    weight: float = 1.0
    max_score: int = 5


class RubricScore(_Base):
    criteria_name: str
    score_a: float
    score_b: float


class JudgeVerdict(_Base):
    """A single (A, B) presentation judged by one pass of the judge."""

    winner: Winner
    score_a: float
    score_b: float
    reasoning: str
    position_swapped: bool = False
    rubric_scores: list[RubricScore] = Field(default_factory=list)


class ConstraintFailure(_Base):
    case_id: str
    category: Category
    side: Literal["candidate", "baseline"]
    constraint: str
    detail: str


class CaseResult(_Base):
    """Position-debiased outcome for one test case (two judge passes)."""

    case_id: str
    category: Category
    winner: Winner
    candidate_score: float
    baseline_score: float
    delta: float
    positional_bias: bool
    # True when the online judge failed and BOTH passes were redone offline.
    used_fallback: bool = False
    reasoning: str
    pass1: JudgeVerdict
    pass2: JudgeVerdict


class BenchmarkReport(_Base):
    total_cases: int
    candidate_win_rate: float
    baseline_win_rate: float
    tie_rate: float
    cohen_kappa: float
    p_value: float
    is_statistically_significant: bool
    positional_bias_rate: float
    fallback_rate: float = 0.0
    mean_score_delta: float
    delta_ci_low: float
    delta_ci_high: float
    win_rate_ci_low: float
    win_rate_ci_high: float
    alpha: float
    category_breakdown: dict[str, dict[str, float]] = Field(default_factory=dict)
    failures: list[ConstraintFailure] = Field(default_factory=list)
