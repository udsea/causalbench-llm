# Results
- Total instances: 60
- Parse rate: 1.000
- Accuracy: 0.367
- Macro accuracy: 0.367
- Macro F1: 0.284
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 44 | 0.733 |
| do_gt_obs | 2 | 0.033 |
| approx_equal | 14 | 0.233 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 20 | 0.800 | 0.500 |
| do_gt_obs | 20 | 0.000 | 0.000 |
| approx_equal | 20 | 0.300 | 0.353 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 12 | 0.417 |
| backdoor_adjustable | 9 | 0.556 |
| collider | 4 | 0.500 |
| confounding | 13 | 0.231 |
| confounding_only | 7 | 0.286 |
| instrumental_variable | 7 | 0.571 |
| mediation | 4 | 0.250 |
| no_confounding | 4 | 0.000 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 20 | 0.300 |
| medium (tol <= gap < 0.08) | 10 | 0.700 |
| easy (gap >= 0.08) | 30 | 0.300 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 16 | 1 | 3 |
| do_gt_obs | 15 | 0 | 5 |
| approx_equal | 13 | 1 | 6 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
