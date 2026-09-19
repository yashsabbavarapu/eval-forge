"""Statistical engine: agreement, confidence intervals and significance.

Everything here is pure-Python and seeded, so a CI gate is reproducible.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from math import fsum
from typing import Final

from .models import BenchmarkReport, CaseResult, ConstraintFailure, Winner

ALPHA: Final = 0.05
_Z: Final = 1.959963984540054  # two-sided 95% normal quantile
BIAS_ALERT: Final = 0.15  # positional-bias rate above which a run is untrustworthy
KAPPA_ALERT: Final = 0.4  # inter-pass agreement below which a run is untrustworthy
MIN_CASES: Final = 6  # below this, a sign-flip test cannot reach alpha = 0.05


def win_counts(results: Sequence[CaseResult]) -> tuple[int, int, int]:
    """``(candidate_wins, baseline_wins, ties)`` over debiased verdicts."""
    tally = Counter(r.winner for r in results)
    return tally[Winner.CANDIDATE], tally[Winner.BASELINE], tally[Winner.TIE]


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over the same items.

    kappa = (p_o - p_e) / (1 - p_e), where p_e is the chance agreement implied by
    each rater's own marginal label distribution. Returns 1.0 when the raters
    agree perfectly on a single label (p_e == 1), else 0.0 for that degenerate case.
    """
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        raise ValueError("kappa needs two equal-length, non-empty label sequences")
    p_o = sum(x == y for x, y in zip(labels_a, labels_b, strict=True)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    p_e = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if math.isclose(p_e, 1.0):
        return 1.0 if math.isclose(p_o, 1.0) else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def pass_agreement_kappa(results: Sequence[CaseResult]) -> float:
    """Kappa between the original ordering and the position-swapped ordering.

    Both passes are mapped into *entity* space (who won, not which slot won), so a
    judge that only tracks position lands near kappa = 0.
    """
    if not results:
        return 0.0
    first = [r.pass1.winner.value for r in results]
    second: list[str] = []
    for r in results:
        raw = r.pass2.winner
        if raw is Winner.TIE:
            second.append(Winner.TIE.value)
        else:
            # pass 2 presents the baseline in slot A, so slot labels invert.
            flipped = Winner.BASELINE if raw is Winner.CANDIDATE else Winner.CANDIDATE
            second.append(flipped.value)
    return cohen_kappa(first, second)


def wilson_interval(successes: int, n: int, z: float = _Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (stable at 0 and 1)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = 4000, alpha: float = ALPHA, seed: int = 7
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired per-case deltas.

    ``rng.choices`` draws a whole resample in one call using ``random()``
    directly; ``randrange`` per element costs an ``int.bit_length`` round trip
    each time, which dominated this function at realistic suite sizes. Drawing
    all resamples in a single call is marginally faster still, but allocates
    ``resamples * n`` floats at once - not a trade worth making on a large suite.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    pool = list(values)
    choices = rng.choices
    # Plain sum, not fsum: the ~1e-15 difference cannot move a percentile here.
    means = sorted(sum(choices(pool, k=n)) / n for _ in range(resamples))
    lo = means[max(0, int(math.floor((alpha / 2) * resamples)))]
    hi = means[min(resamples - 1, int(math.ceil((1 - alpha / 2) * resamples)) - 1)]
    return lo, hi


_CHUNK: Final = 8
_MASK: Final = (1 << _CHUNK) - 1


def _subset_sum_tables(deltas: Sequence[float]) -> list[list[float]]:
    """Precompute every subset sum for each 8-element chunk of ``deltas``.

    A sign-flip permutation only needs ``sum(d_i)`` over the flipped subset, and
    there are just 256 distinct subsets per chunk. Building the tables costs
    ``32 * n`` additions once; each of the 10,000 permutations then costs ``n/8``
    table lookups instead of ``n`` branch-and-add steps.
    """
    tables: list[list[float]] = []
    for start in range(0, len(deltas), _CHUNK):
        chunk = deltas[start : start + _CHUNK]
        table = [0.0] * (1 << len(chunk))
        for mask in range(1, 1 << len(chunk)):
            lowest = mask & -mask
            table[mask] = table[mask ^ lowest] + chunk[lowest.bit_length() - 1]
        tables.append(table)
    return tables


def permutation_test(
    deltas: Sequence[float], *, permutations: int = 10000, seed: int = 7
) -> float:
    """Two-sided paired permutation (sign-flip) test on per-case score deltas.

    Under H0 the candidate and baseline labels are exchangeable within a case, so
    each delta's sign is equally likely to flip. p = (1 + #{|T*| >= |T|}) / (B + 1),
    the add-one correction that keeps p strictly positive.

    Implementation note: flipping a subset F of the deltas gives
    ``sum(eps_i d_i) = S - 2 * sum(d_i for i in F)``, so a permutation reduces to
    one subset sum. Drawing all n signs as a single ``getrandbits(n)`` and reading
    the subset sum out of precomputed 8-bit chunk tables replaces ``B * n`` RNG
    calls with ``B``, and ``B * n`` adds with ``B * n/8`` lookups.
    """
    n = len(deltas)
    if n == 0:
        return 1.0
    total = fsum(deltas)
    observed = abs(total / n)
    if math.isclose(observed, 0.0):
        return 1.0
    threshold = (observed - 1e-12) * n
    tables = _subset_sum_tables(deltas)
    getrandbits = random.Random(seed).getrandbits
    extreme = 0
    for _ in range(permutations):
        bits = getrandbits(n)
        flipped = 0.0
        for table in tables:
            flipped += table[bits & _MASK]
            bits >>= _CHUNK
        if abs(total - 2.0 * flipped) >= threshold:
            extreme += 1
    return (1.0 + extreme) / (permutations + 1.0)


def _category_breakdown(results: Sequence[CaseResult]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cat in sorted({r.category.value for r in results}):
        subset = [r for r in results if r.category.value == cat]
        wins, losses, ties = win_counts(subset)
        out[cat] = {
            "n": float(len(subset)),
            "candidate_win_rate": wins / len(subset),
            "baseline_win_rate": losses / len(subset),
            "tie_rate": ties / len(subset),
            "mean_delta": sum(r.delta for r in subset) / len(subset),
        }
    return out


def build_report(
    results: Sequence[CaseResult],
    failures: Sequence[ConstraintFailure] = (),
    *,
    alpha: float = ALPHA,
    permutations: int = 10000,
    seed: int = 7,
) -> BenchmarkReport:
    """Aggregate debiased case results into the CI-gate scorecard."""
    n = len(results)
    wins, losses, ties = win_counts(results)
    deltas = [r.delta for r in results]
    p_value = permutation_test(deltas, permutations=permutations, seed=seed)
    ci_lo, ci_hi = bootstrap_ci(deltas, alpha=alpha, seed=seed)
    wr_lo, wr_hi = wilson_interval(wins, n)
    return BenchmarkReport(
        total_cases=n,
        candidate_win_rate=wins / n if n else 0.0,
        baseline_win_rate=losses / n if n else 0.0,
        tie_rate=ties / n if n else 0.0,
        cohen_kappa=pass_agreement_kappa(results),
        p_value=p_value,
        is_statistically_significant=p_value < alpha,
        positional_bias_rate=(sum(r.positional_bias for r in results) / n) if n else 0.0,
        fallback_rate=(sum(r.used_fallback for r in results) / n) if n else 0.0,
        mean_score_delta=(sum(deltas) / n) if n else 0.0,
        delta_ci_low=ci_lo,
        delta_ci_high=ci_hi,
        win_rate_ci_low=wr_lo,
        win_rate_ci_high=wr_hi,
        alpha=alpha,
        category_breakdown=_category_breakdown(results),
        failures=list(failures),
    )


def reliability_warnings(report: BenchmarkReport) -> list[str]:
    """Reasons to distrust a run, independent of which side "won"."""
    out: list[str] = []
    if report.positional_bias_rate > BIAS_ALERT:
        out.append(
            f"Positional bias on {report.positional_bias_rate:.0%} of cases "
            f"(> {BIAS_ALERT:.0%}): the judge is reading slot order, not text. Those verdicts "
            "were nullified to ties, which is why the win rates collapse; the score margin "
            "survives because averaging both orderings cancels an additive position effect."
        )
    if report.fallback_rate:
        out.append(
            f"{report.fallback_rate:.0%} of cases were scored by the offline judge after the "
            "online judge failed. Both passes of those cases were redone offline so each "
            "swap stays self-consistent, but they are not calibrated against the cases the "
            "online judge scored - treat this run as two populations, not one."
        )
    if report.total_cases and report.cohen_kappa < KAPPA_ALERT:
        out.append(
            f"Inter-pass Cohen's kappa is {report.cohen_kappa:.2f} (< {KAPPA_ALERT}): the two "
            "orderings barely agree. Tighten the rubric before acting on this run."
        )
    if report.win_rate_ci_low < 0.5 < report.win_rate_ci_high:
        out.append(
            f"Win-rate CI [{report.win_rate_ci_low:.0%}, {report.win_rate_ci_high:.0%}] spans "
            f"50%: the win rate is unresolved at n = {report.total_cases}."
        )
    if 0 < report.total_cases < MIN_CASES:
        out.append(
            f"n = {report.total_cases}: a sign-flip permutation test cannot reach "
            f"alpha = 0.05 with fewer than {MIN_CASES} cases."
        )
    return out
