"""Statistics: Cohen's kappa, intervals and the permutation test."""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence

import pytest

from evalforge.models import CaseResult, Category, JudgeVerdict, Winner
from evalforge.stats import (
    _CHUNK,
    _MASK,
    _subset_sum_tables,
    bootstrap_ci,
    build_report,
    cohen_kappa,
    pass_agreement_kappa,
    permutation_test,
    reliability_warnings,
    wilson_interval,
    win_counts,
)


def verdict(winner: Winner, a: float = 4.0, b: float = 2.0, swapped: bool = False) -> JudgeVerdict:
    return JudgeVerdict(
        winner=winner, score_a=a, score_b=b, reasoning="x", position_swapped=swapped
    )


def _mirror(winner: Winner) -> Winner:
    if winner is Winner.TIE:
        return Winner.TIE
    return Winner.BASELINE if winner is Winner.CANDIDATE else Winner.CANDIDATE


def result(
    idx: int,
    winner: Winner,
    delta: float,
    *,
    bias: bool = False,
    p1: Winner | None = None,
    p2: Winner | None = None,
    category: Category = Category.STANDARD,
) -> CaseResult:
    return CaseResult(
        case_id=f"c{idx}",
        category=category,
        winner=winner,
        candidate_score=3.0 + delta / 2,
        baseline_score=3.0 - delta / 2,
        delta=delta,
        positional_bias=bias,
        reasoning="x",
        pass1=verdict(p1 if p1 is not None else winner),
        # pass2 reports a slot, not an entity: a judge that consistently
        # names the same winner emits the mirrored label on the swap.
        pass2=verdict(p2 if p2 is not None else _mirror(winner), swapped=True),
    )


# --- Cohen's kappa ---------------------------------------------------------


def test_kappa_matches_hand_computed_2x2() -> None:
    # Confusion matrix: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no (n = 50).
    # p_o = 35/50 = 0.70; p_e = (30/50)(25/50) + (20/50)(25/50) = 0.50.
    # kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40.
    a = ["yes"] * 25 + ["no"] * 25
    b = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.40)


def test_kappa_is_one_for_perfect_agreement_with_spread() -> None:
    labels = ["a", "b", "c", "a", "b", "c"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_kappa_is_one_when_both_raters_are_constant_and_agree() -> None:
    assert cohen_kappa(["a"] * 6, ["a"] * 6) == pytest.approx(1.0)


def test_kappa_is_zero_when_one_rater_is_constant_and_disagrees() -> None:
    assert cohen_kappa(["a"] * 6, ["b"] * 6) == pytest.approx(0.0)


def test_kappa_is_negative_for_systematic_disagreement() -> None:
    a = ["yes", "no"] * 10
    b = ["no", "yes"] * 10
    assert cohen_kappa(a, b) == pytest.approx(-1.0)


def test_kappa_near_zero_for_independent_raters() -> None:
    rng = random.Random(0)
    a = [rng.choice("xy") for _ in range(4000)]
    b = [rng.choice("xy") for _ in range(4000)]
    assert abs(cohen_kappa(a, b)) < 0.06


def test_kappa_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


def test_pass_agreement_kappa_maps_slots_back_to_entities() -> None:
    # A judge that always picks slot A: pass1 says CANDIDATE, pass2 (swapped)
    # also reports slot A, which in entity space is the BASELINE. The two passes
    # never agree, and because each is constant the chance agreement is 0 too,
    # so kappa collapses to exactly 0 - no reliability at all.
    biased = [
        result(i, Winner.TIE, 0.0, bias=True, p1=Winner.CANDIDATE, p2=Winner.CANDIDATE)
        for i in range(20)
    ]
    assert pass_agreement_kappa(biased) == pytest.approx(0.0)

    # A consistent judge agrees across orderings in entity space: pass2's raw
    # slot label is BASELINE, which flips back to CANDIDATE.
    honest = [
        result(i, Winner.CANDIDATE, 1.0, p1=Winner.CANDIDATE, p2=Winner.BASELINE)
        for i in range(10)
    ] + [result(i, Winner.BASELINE, -1.0, p1=Winner.BASELINE, p2=Winner.CANDIDATE) for i in range(10)]
    assert pass_agreement_kappa(honest) == pytest.approx(1.0)


def test_pass_agreement_kappa_is_zero_for_empty_input() -> None:
    assert pass_agreement_kappa([]) == 0.0


# --- intervals -------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate() -> None:
    lo, hi = wilson_interval(40, 50)
    assert lo < 0.8 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_wilson_interval_stays_in_range_at_the_boundaries() -> None:
    lo, hi = wilson_interval(0, 20)
    assert lo == pytest.approx(0.0)
    assert 0.0 < hi < 0.25
    lo, hi = wilson_interval(20, 20)
    assert hi == pytest.approx(1.0)
    assert 0.75 < lo < 1.0


def test_wilson_interval_narrows_as_n_grows() -> None:
    small = wilson_interval(8, 10)
    large = wilson_interval(800, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_covers_the_mean_and_is_seed_stable() -> None:
    values = [0.5] * 30 + [0.1] * 30
    lo, hi = bootstrap_ci(values, seed=3)
    assert lo < 0.30 < hi
    assert bootstrap_ci(values, seed=3) == (lo, hi)


def test_bootstrap_ci_of_a_constant_is_degenerate() -> None:
    assert bootstrap_ci([2.0] * 25) == pytest.approx((2.0, 2.0))


def test_bootstrap_ci_of_nothing_is_zero() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)


# --- permutation test ------------------------------------------------------


def test_permutation_test_detects_a_strong_consistent_effect() -> None:
    assert permutation_test([0.8] * 30) < 0.001


def test_permutation_test_finds_nothing_in_symmetric_noise() -> None:
    rng = random.Random(11)
    raw = [rng.gauss(0.0, 1.0) for _ in range(200)]
    centre = sum(raw) / len(raw)
    # Centre the sample, then add an offset far inside the noise floor
    # (0.02 against a standard error of ~0.07) so the full permutation path
    # runs rather than the zero-statistic shortcut.
    deltas = [d - centre + 0.02 for d in raw]
    assert permutation_test(deltas, permutations=4000, seed=5) > 0.05


def test_permutation_test_is_two_sided() -> None:
    positive = permutation_test([0.6] * 25, seed=5)
    negative = permutation_test([-0.6] * 25, seed=5)
    assert positive == pytest.approx(negative)


def test_permutation_test_p_value_is_bounded_and_never_zero() -> None:
    p = permutation_test([5.0] * 40, permutations=500)
    assert 0.0 < p <= 1.0
    assert p == pytest.approx(1.0 / 501.0)


def test_permutation_test_degenerate_inputs() -> None:
    assert permutation_test([]) == 1.0
    assert permutation_test([0.0] * 10) == 1.0


def test_permutation_test_needs_a_large_enough_sample() -> None:
    # Three identical deltas cannot beat 1/8 under sign-flipping, so no
    # three-case suite can ever clear alpha = 0.05.
    assert permutation_test([1.0] * 3, permutations=20000) > 0.05
    assert permutation_test([1.0] * 8, permutations=20000) < 0.05


def test_permutation_test_is_deterministic_for_a_seed() -> None:
    deltas = [0.3, -0.1, 0.5, 0.2, -0.4, 0.9, 0.1, 0.2]
    assert permutation_test(deltas, seed=42) == permutation_test(deltas, seed=42)


# --- report assembly -------------------------------------------------------


def _mixed() -> Sequence[CaseResult]:
    wins = [result(i, Winner.CANDIDATE, 0.9) for i in range(14)]
    losses = [result(100 + i, Winner.BASELINE, -0.4, category=Category.ADVERSARIAL) for i in range(3)]
    ties = [result(200 + i, Winner.TIE, 0.0, bias=True, category=Category.BOUNDARY) for i in range(3)]
    return [*wins, *losses, *ties]


def test_win_counts_and_rates_sum_to_one() -> None:
    results = _mixed()
    assert win_counts(results) == (14, 3, 3)
    report = build_report(results)
    total = report.candidate_win_rate + report.baseline_win_rate + report.tie_rate
    assert total == pytest.approx(1.0)
    assert report.total_cases == 20
    assert report.candidate_win_rate == pytest.approx(0.7)


def test_report_flags_a_significant_improvement() -> None:
    report = build_report(_mixed())
    assert report.mean_score_delta > 0
    assert report.p_value < 0.05
    assert report.is_statistically_significant is True
    assert report.delta_ci_low < report.mean_score_delta < report.delta_ci_high


def test_report_flags_a_significant_regression() -> None:
    report = build_report([result(i, Winner.BASELINE, -0.7) for i in range(25)])
    assert report.is_statistically_significant is True
    assert report.mean_score_delta < 0


def test_report_stays_quiet_on_noise() -> None:
    rng = random.Random(4)
    noisy = [result(i, Winner.TIE, rng.gauss(0.0, 0.5)) for i in range(60)]
    report = build_report(noisy)
    assert report.is_statistically_significant is False
    assert report.p_value > report.alpha


def test_report_tracks_positional_bias_and_categories() -> None:
    report = build_report(_mixed())
    assert report.positional_bias_rate == pytest.approx(3 / 20)
    assert set(report.category_breakdown) == {"standard", "boundary", "adversarial"}
    assert report.category_breakdown["adversarial"]["baseline_win_rate"] == pytest.approx(1.0)
    assert report.category_breakdown["standard"]["n"] == pytest.approx(14.0)


def test_report_handles_an_empty_run() -> None:
    report = build_report([])
    assert report.total_cases == 0
    assert report.p_value == 1.0
    assert report.is_statistically_significant is False
    assert report.cohen_kappa == 0.0


def test_alpha_controls_the_gate() -> None:
    results = [result(i, Winner.CANDIDATE, 0.5) for i in range(6)]
    lenient = build_report(results, alpha=0.05)
    strict = build_report(results, alpha=0.001)
    assert lenient.p_value == strict.p_value
    assert lenient.is_statistically_significant is True
    assert strict.is_statistically_significant is False


# --- reliability caveats ---------------------------------------------------


def test_clean_run_raises_no_caveats() -> None:
    # Decisive (27/30), order-consistent, unbiased and adequately powered.
    clean = [result(i, Winner.CANDIDATE, 0.9) for i in range(27)] + [
        result(90 + i, Winner.TIE, 0.0) for i in range(3)
    ]
    assert reliability_warnings(build_report(clean)) == []


def test_a_70_percent_win_rate_at_n20_is_not_resolved() -> None:
    # 14/20 has a Wilson CI of roughly [48%, 86%], which still straddles a
    # coin flip - the caveat must fire even though the margin test passes.
    report = build_report(_mixed())
    assert report.is_statistically_significant is True
    assert any("spans" in w for w in reliability_warnings(report))


def test_pervasive_positional_bias_is_called_out() -> None:
    biased = [
        result(i, Winner.TIE, 0.9, bias=True, p1=Winner.CANDIDATE, p2=Winner.CANDIDATE)
        for i in range(20)
    ]
    warnings = reliability_warnings(build_report(biased))
    assert any("Positional bias" in w for w in warnings)
    assert any("kappa" in w for w in warnings)


def test_unresolved_win_rate_is_called_out() -> None:
    split = [result(i, Winner.CANDIDATE, 0.2) for i in range(10)] + [
        result(50 + i, Winner.BASELINE, -0.2) for i in range(10)
    ]
    assert any("spans" in w for w in reliability_warnings(build_report(split)))


def test_underpowered_run_is_called_out() -> None:
    tiny = [result(i, Winner.CANDIDATE, 0.9) for i in range(4)]
    report = build_report(tiny)
    assert report.is_statistically_significant is False
    assert any("cannot reach" in w for w in reliability_warnings(report))


# --- the optimised permutation kernel --------------------------------------


@pytest.mark.parametrize("n", [1, 7, 8, 9, 16, 17, 23])
def test_subset_sum_tables_match_naive_enumeration(n: int) -> None:
    # The chunked table is a pure speed trick; it must be bit-for-bit equivalent
    # to summing the flipped subset directly, including across chunk boundaries.
    rng = random.Random(n)
    deltas = [rng.gauss(0.0, 1.0) for _ in range(n)]
    tables = _subset_sum_tables(deltas)
    for _ in range(200):
        bits = rng.getrandbits(n)
        naive = sum(d for i, d in enumerate(deltas) if bits >> i & 1)
        remaining, fast = bits, 0.0
        for table in tables:
            fast += table[remaining & _MASK]
            remaining >>= _CHUNK
        assert fast == pytest.approx(naive, abs=1e-9)


@pytest.mark.parametrize("n", [8, 10, 12])
def test_permutation_test_matches_the_exact_null_distribution(n: int) -> None:
    # For small n the null can be enumerated exactly (2**n sign vectors). The
    # sampled p-value must land on it.
    rng = random.Random(n * 7)
    deltas = [rng.gauss(0.35, 1.0) for _ in range(n)]
    observed = abs(sum(deltas) / n)
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, deltas, strict=True)) / n) >= observed - 1e-12
    )
    exact = hits / 2**n
    assert permutation_test(deltas, permutations=40000, seed=11) == pytest.approx(exact, abs=0.02)
