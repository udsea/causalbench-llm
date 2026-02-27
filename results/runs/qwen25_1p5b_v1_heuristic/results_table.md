# Results
- Total instances: 120
- Parse rate: 1.000
- Accuracy: 0.442
- Macro accuracy: 0.442
- Macro F1: 0.438
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 43 | 0.358 |
| do_gt_obs | 48 | 0.400 |
| approx_equal | 29 | 0.242 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 40 | 0.500 | 0.482 |
| do_gt_obs | 40 | 0.500 | 0.455 |
| approx_equal | 40 | 0.325 | 0.377 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 15 | 0.800 |
| backdoor_adjustable | 15 | 0.333 |
| collider | 15 | 0.133 |
| confounding | 15 | 0.600 |
| confounding_only | 15 | 0.800 |
| instrumental_variable | 15 | 0.600 |
| mediation | 15 | 0.133 |
| no_confounding | 15 | 0.133 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 40 | 0.325 |
| medium (tol <= gap < 0.08) | 42 | 0.238 |
| easy (gap >= 0.08) | 38 | 0.789 |

## Breakdown by n_in_band reliability

| n_in_band bucket | n | acc |
|---|---:|---:|
| < 100 | 6 | 0.167 |
| 100-199 | 101 | 0.446 |
| >= 200 | 13 | 0.538 |
| missing | 0 | 0.000 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 20 | 12 | 8 |
| do_gt_obs | 12 | 20 | 8 |
| approx_equal | 11 | 16 | 13 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
