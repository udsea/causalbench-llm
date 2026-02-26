# Results
- Total instances: 60
- Parse rate: 1.000
- Accuracy: 0.300
- Macro accuracy: 0.300
- Macro F1: 0.183
- Invalid/unknown predictions: 0

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 20 | 0.850 | 0.479 |
| do_gt_obs | 20 | 0.050 | 0.069 |
| approx_equal | 20 | 0.000 | 0.000 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 8 | 0.625 |
| backdoor_adjustable | 7 | 0.429 |
| collider | 8 | 0.125 |
| confounding | 8 | 0.250 |
| confounding_only | 7 | 0.429 |
| instrumental_variable | 8 | 0.250 |
| mediation | 7 | 0.286 |
| no_confounding | 7 | 0.000 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 20 | 0.000 |
| medium (tol <= gap < 0.08) | 23 | 0.391 |
| easy (gap >= 0.08) | 17 | 0.529 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 17 | 3 | 0 |
| do_gt_obs | 19 | 1 | 0 |
| approx_equal | 15 | 5 | 0 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
