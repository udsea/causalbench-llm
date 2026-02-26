# Results
- Total instances: 60
- Parse rate: 1.000
- Accuracy: 0.350
- Macro accuracy: 0.350
- Macro F1: 0.280
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 28 | 0.467 |
| do_gt_obs | 0 | 0.000 |
| approx_equal | 32 | 0.533 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 20 | 0.500 | 0.417 |
| do_gt_obs | 20 | 0.000 | 0.000 |
| approx_equal | 20 | 0.550 | 0.423 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 12 | 0.167 |
| backdoor_adjustable | 8 | 0.500 |
| collider | 4 | 1.000 |
| confounding | 12 | 0.167 |
| confounding_only | 8 | 0.250 |
| instrumental_variable | 8 | 0.375 |
| mediation | 4 | 0.750 |
| no_confounding | 4 | 0.250 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 20 | 0.550 |
| medium (tol <= gap < 0.08) | 9 | 0.111 |
| easy (gap >= 0.08) | 31 | 0.290 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 10 | 0 | 10 |
| do_gt_obs | 9 | 0 | 11 |
| approx_equal | 9 | 0 | 11 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
