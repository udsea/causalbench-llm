# Results
- Total instances: 120
- Parse rate: 1.000
- Accuracy: 0.333
- Macro accuracy: 0.333
- Macro F1: 0.167
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 0 | 0.000 |
| do_gt_obs | 120 | 1.000 |
| approx_equal | 0 | 0.000 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 40 | 0.000 | 0.000 |
| do_gt_obs | 40 | 1.000 | 0.500 |
| approx_equal | 40 | 0.000 | 0.000 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 15 | 0.333 |
| backdoor_adjustable | 15 | 0.333 |
| collider | 15 | 0.333 |
| confounding | 15 | 0.333 |
| confounding_only | 15 | 0.333 |
| instrumental_variable | 15 | 0.333 |
| mediation | 15 | 0.333 |
| no_confounding | 15 | 0.333 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 40 | 0.000 |
| medium (tol <= gap < 0.08) | 35 | 0.486 |
| easy (gap >= 0.08) | 45 | 0.511 |

## Breakdown by n_in_band reliability

| n_in_band bucket | n | acc |
|---|---:|---:|
| < 100 | 7 | 0.429 |
| 100-199 | 105 | 0.343 |
| >= 200 | 8 | 0.125 |
| missing | 0 | 0.000 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 0 | 40 | 0 |
| do_gt_obs | 0 | 40 | 0 |
| approx_equal | 0 | 40 | 0 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
