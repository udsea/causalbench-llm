# Results
- Total instances: 120
- Parse rate: 1.000
- Accuracy: 0.283
- Macro accuracy: 0.283
- Macro F1: 0.281
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 35 | 0.292 |
| do_gt_obs | 43 | 0.358 |
| approx_equal | 42 | 0.350 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 40 | 0.200 | 0.213 |
| do_gt_obs | 40 | 0.350 | 0.337 |
| approx_equal | 40 | 0.300 | 0.293 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 15 | 0.200 |
| backdoor_adjustable | 15 | 0.267 |
| collider | 15 | 0.333 |
| confounding | 15 | 0.467 |
| confounding_only | 15 | 0.267 |
| instrumental_variable | 15 | 0.333 |
| mediation | 15 | 0.267 |
| no_confounding | 15 | 0.133 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 40 | 0.300 |
| medium (tol <= gap < 0.08) | 42 | 0.286 |
| easy (gap >= 0.08) | 38 | 0.263 |

## Breakdown by n_in_band reliability

| n_in_band bucket | n | acc |
|---|---:|---:|
| < 100 | 6 | 0.333 |
| 100-199 | 101 | 0.297 |
| >= 200 | 13 | 0.154 |
| missing | 0 | 0.000 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 8 | 15 | 17 |
| do_gt_obs | 13 | 14 | 13 |
| approx_equal | 14 | 14 | 12 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
