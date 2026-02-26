# Results
- Total instances: 60
- Parse rate: 1.000
- Accuracy: 0.350
- Macro accuracy: 0.350
- Macro F1: 0.268
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 45 | 0.750 |
| do_gt_obs | 3 | 0.050 |
| approx_equal | 12 | 0.200 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 20 | 0.800 | 0.492 |
| do_gt_obs | 20 | 0.000 | 0.000 |
| approx_equal | 20 | 0.250 | 0.312 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 13 | 0.385 |
| backdoor_adjustable | 9 | 0.556 |
| collider | 4 | 0.250 |
| confounding | 11 | 0.182 |
| confounding_only | 8 | 0.375 |
| instrumental_variable | 7 | 0.571 |
| mediation | 4 | 0.250 |
| no_confounding | 4 | 0.000 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 14 | 0.214 |
| medium (tol <= gap < 0.08) | 15 | 0.467 |
| easy (gap >= 0.08) | 31 | 0.355 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 16 | 1 | 3 |
| do_gt_obs | 16 | 0 | 4 |
| approx_equal | 13 | 2 | 5 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
