"""eval-forge: synthetic benchmark generation + debiased LLM-as-a-judge."""

from __future__ import annotations

from .judge import (
    DEFAULT_RUBRICS,
    DeterministicJudge,
    GeminiJudge,
    JudgeUnavailable,
    PairwiseJudge,
)
from .models import (
    BenchmarkReport,
    CaseResult,
    Category,
    CompletionSet,
    EvaluationRubric,
    JudgeVerdict,
    TaskSpec,
    TestCase,
    TestSuite,
    Winner,
)
from .stats import build_report, cohen_kappa, permutation_test

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_RUBRICS",
    "BenchmarkReport",
    "CaseResult",
    "Category",
    "CompletionSet",
    "DeterministicJudge",
    "EvaluationRubric",
    "GeminiJudge",
    "JudgeUnavailable",
    "JudgeVerdict",
    "PairwiseJudge",
    "TaskSpec",
    "TestCase",
    "TestSuite",
    "Winner",
    "__version__",
    "build_report",
    "cohen_kappa",
    "permutation_test",
]
