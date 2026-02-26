# Results
- Total instances: 120
- Parse rate: 1.000
- Accuracy: 0.433
- Macro accuracy: 0.433
- Macro F1: 0.421
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 44 | 0.367 |
| do_gt_obs | 54 | 0.450 |
| approx_equal | 22 | 0.183 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 40 | 0.475 | 0.452 |
| do_gt_obs | 40 | 0.575 | 0.489 |
| approx_equal | 40 | 0.250 | 0.323 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 15 | 0.800 |
| backdoor_adjustable | 15 | 0.333 |
| collider | 15 | 0.133 |
| confounding | 15 | 0.533 |
| confounding_only | 15 | 0.800 |
| instrumental_variable | 15 | 0.600 |
| mediation | 15 | 0.133 |
| no_confounding | 15 | 0.133 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 40 | 0.250 |
| medium (tol <= gap < 0.08) | 39 | 0.205 |
| easy (gap >= 0.08) | 41 | 0.829 |

## Breakdown by n_in_band reliability

| n_in_band bucket | n | acc |
|---|---:|---:|
| < 100 | 8 | 0.125 |
| 100-199 | 100 | 0.470 |
| >= 200 | 12 | 0.333 |
| missing | 0 | 0.000 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 19 | 14 | 7 |
| do_gt_obs | 12 | 23 | 5 |
| approx_equal | 13 | 17 | 10 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
