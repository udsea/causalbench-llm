# Results
- Total instances: 120
- Parse rate: 1.000
- Accuracy: 0.408
- Macro accuracy: 0.408
- Macro F1: 0.325
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 49 | 0.408 |
| do_gt_obs | 0 | 0.000 |
| approx_equal | 71 | 0.592 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 40 | 0.525 | 0.472 |
| do_gt_obs | 40 | 0.000 | 0.000 |
| approx_equal | 40 | 0.700 | 0.505 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 15 | 0.267 |
| backdoor_adjustable | 15 | 0.533 |
| collider | 15 | 0.400 |
| confounding | 15 | 0.400 |
| confounding_only | 15 | 0.533 |
| instrumental_variable | 15 | 0.333 |
| mediation | 15 | 0.533 |
| no_confounding | 15 | 0.267 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 40 | 0.700 |
| medium (tol <= gap < 0.08) | 39 | 0.231 |
| easy (gap >= 0.08) | 41 | 0.293 |

## Breakdown by n_in_band reliability

| n_in_band bucket | n | acc |
|---|---:|---:|
| < 100 | 8 | 0.250 |
| 100-199 | 100 | 0.400 |
| >= 200 | 12 | 0.583 |
| missing | 0 | 0.000 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 21 | 0 | 19 |
| do_gt_obs | 16 | 0 | 24 |
| approx_equal | 12 | 0 | 28 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
