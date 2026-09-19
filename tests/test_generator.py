"""Synthetic generator: structure, determinism and constraint propagation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from evalforge.generator import generate, generate_offline, load_spec
from evalforge.models import Category, TaskSpec

SPEC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "task_spec.json"


@pytest.fixture()
def spec() -> TaskSpec:
    return load_spec(str(SPEC_PATH))


def test_fixture_spec_parses(spec: TaskSpec) -> None:
    assert spec.name == "gtm-cold-outbound"
    assert spec.hard_constraints
    assert spec.seed_topics


def test_generates_exact_count_and_unique_ids(spec: TaskSpec) -> None:
    suite = generate_offline(spec, 20)
    assert suite.spec_name == spec.name
    assert len(suite.cases) == 20
    assert len({c.id for c in suite.cases}) == 20
    assert all(c.prompt.strip() for c in suite.cases)


@pytest.mark.parametrize("count", [1, 3, 7, 20, 51])
def test_count_is_honoured_for_awkward_sizes(spec: TaskSpec, count: int) -> None:
    assert len(generate_offline(spec, count).cases) == count


def test_all_three_categories_are_covered(spec: TaskSpec) -> None:
    tally = Counter(c.category for c in generate_offline(spec, 20).cases)
    assert set(tally) == set(Category)
    # Adversarial and boundary coverage must not collapse to a token case.
    assert tally[Category.ADVERSARIAL] >= 4
    assert tally[Category.BOUNDARY] >= 4


def test_hard_constraints_propagate_to_every_case(spec: TaskSpec) -> None:
    for case in generate_offline(spec, 24).cases:
        assert set(spec.hard_constraints) <= set(case.expected_constraints)
        assert len(case.expected_constraints) == len(set(case.expected_constraints))


def test_edge_categories_add_extra_constraints(spec: TaskSpec) -> None:
    suite = generate_offline(spec, 24)
    for case in suite.cases:
        if case.category is not Category.STANDARD:
            extra = set(case.expected_constraints) - set(spec.hard_constraints)
            assert extra, f"{case.id} added no category-specific constraint"


def test_reference_context_carries_spec_signal(spec: TaskSpec) -> None:
    for case in generate_offline(spec, 12).cases:
        assert spec.audience in case.reference_context
        assert spec.purpose in case.reference_context


def test_generation_is_deterministic_per_seed(spec: TaskSpec) -> None:
    a = generate_offline(spec, 16, seed=99)
    b = generate_offline(spec, 16, seed=99)
    c = generate_offline(spec, 16, seed=100)
    assert a == b
    assert [x.prompt for x in a.cases] != [x.prompt for x in c.cases]


def test_offline_flag_bypasses_gemini(monkeypatch: pytest.MonkeyPatch, spec: TaskSpec) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    suite = generate(spec, 6, offline=True)
    assert len(suite.cases) == 6
    assert all(c.id.startswith(spec.name) for c in suite.cases)


def test_missing_key_falls_back_to_offline(monkeypatch: pytest.MonkeyPatch, spec: TaskSpec) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert generate(spec, 5) == generate_offline(spec, 5)


def test_gemini_failure_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, spec: TaskSpec
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")

    def boom(*_args: object, **_kwargs: object) -> object:
        raise OSError("network down")

    monkeypatch.setattr("evalforge.generator.gemini_json", boom)
    suite = generate(spec, 5)
    assert len(suite.cases) == 5


def test_gemini_payload_is_validated_and_merged(
    monkeypatch: pytest.MonkeyPatch, spec: TaskSpec
) -> None:
    payload = [
        {
            "prompt": "Email the VP about a stalled migration.",
            "reference_context": "Account: Acme.",
            "expected_constraints": ["Under 60 words"],
            "category": "adversarial",
        }
    ]
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr("evalforge.generator.gemini_json", lambda *a, **k: payload)
    suite = generate(spec, 1)
    case = suite.cases[0]
    assert case.category is Category.ADVERSARIAL
    assert "Under 60 words" in case.expected_constraints
    # Spec-level hard constraints are still enforced on model-authored cases.
    assert set(spec.hard_constraints) <= set(case.expected_constraints)


def test_suite_round_trips_through_json(spec: TaskSpec, tmp_path: Path) -> None:
    from evalforge.models import TestSuite

    suite = generate_offline(spec, 8)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite.model_dump(mode="json")), encoding="utf-8")
    assert TestSuite.model_validate_json(path.read_text(encoding="utf-8")) == suite
