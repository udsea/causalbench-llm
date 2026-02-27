# Results
- Total instances: 120
- Parse rate: 1.000
- Accuracy: 0.417
- Macro accuracy: 0.417
- Macro F1: 0.414
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 43 | 0.358 |
| do_gt_obs | 50 | 0.417 |
| approx_equal | 27 | 0.225 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 40 | 0.425 | 0.410 |
| do_gt_obs | 40 | 0.500 | 0.444 |
| approx_equal | 40 | 0.325 | 0.388 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 15 | 0.867 |
| backdoor_adjustable | 15 | 0.267 |
| collider | 15 | 0.067 |
| confounding | 15 | 0.400 |
| confounding_only | 15 | 0.800 |
| instrumental_variable | 15 | 0.667 |
| mediation | 15 | 0.133 |
| no_confounding | 15 | 0.133 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 40 | 0.325 |
| medium (tol <= gap < 0.08) | 35 | 0.171 |
| easy (gap >= 0.08) | 45 | 0.689 |

## Breakdown by n_in_band reliability

| n_in_band bucket | n | acc |
|---|---:|---:|
| < 100 | 7 | 0.143 |
| 100-199 | 105 | 0.438 |
| >= 200 | 8 | 0.375 |
| missing | 0 | 0.000 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 17 | 15 | 8 |
| do_gt_obs | 14 | 20 | 6 |
| approx_equal | 12 | 15 | 13 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
