# eval-forge scorecard - prompt_v2.txt vs prompt_v1.txt

**Verdict: CANDIDATE WINS** (permutation p = 0.0001, alpha = 0.05)

## Executive summary

| metric | value |
| --- | --- |
| test cases | 24 |
| candidate win rate | 91.7% (95% CI 74.2% - 97.7%) |
| baseline win rate | 4.2% |
| tie rate | 4.2% |
| mean rubric delta | +1.637 (95% bootstrap CI +1.270 - +1.962) |
| permutation p-value | 0.0001 (alpha = 0.05) |
| statistically significant | yes |
| Cohen's kappa (order vs swapped) | 1.000 |
| positional bias rate | 0.0% |
| offline-fallback rate | 0.0% |

## Breakdown by category

| category | n | candidate | baseline | tie | mean delta |
| --- | --- | --- | --- | --- | --- |
| adversarial | 7 | 86% | 0% | 14% | +1.56 |
| boundary | 7 | 86% | 14% | 0% | +1.67 |
| standard | 10 | 100% | 0% | 0% | +1.67 |

## Constraint failures

| side | constraint | violations |
| --- | --- | --- |
| baseline | Under 120 words | 20 |
| baseline | Do not mention pricing | 18 |
| baseline | Must include the phrase '15 minutes' | 18 |
| baseline | Never mention 'discount' | 16 |
| baseline | Do not mention 'revenue' | 6 |
| baseline | Under 80 words | 5 |
| baseline | Under 100 words | 5 |
| candidate | Must include the phrase '15 minutes' | 3 |
| candidate | Under 120 words | 3 |
| candidate | Never mention 'discount' | 2 |
| candidate | Do not mention pricing | 1 |
| candidate | Under 80 words | 1 |
| candidate | Under 100 words | 1 |

## Worst candidate regressions

- `gtm-cold-outbound-bou-015` (boundary, delta -0.81) - both orderings agree on BASELINE | pass1: A/B accuracy=2.46/2.25 constraint_compliance=3.67/3.67 conciseness=1.20/5.00 margin=-0.81 | pass2: A/B accuracy=2.25/2.46 constraint_compliance=3.67/3.67 conciseness=5.00/1.20 margin=+0.81

---

_Two judge passes per case with positions swapped; a preference that tracks the slot rather than the text is nullified to a TIE._
