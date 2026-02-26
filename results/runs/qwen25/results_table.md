# Results
- Total instances: 60
- Parse rate: 1.000
- Accuracy: 0.333
- Macro accuracy: 0.333
- Macro F1: 0.173
- Invalid/unknown predictions: 0

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 20 | 1.000 | 0.519 |
| do_gt_obs | 20 | 0.000 | 0.000 |
| approx_equal | 20 | 0.000 | 0.000 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 7 | 0.571 |
| backdoor_adjustable | 7 | 0.714 |
| collider | 8 | 0.375 |
| confounding | 8 | 0.375 |
| confounding_only | 8 | 0.375 |
| instrumental_variable | 7 | 0.143 |
| mediation | 7 | 0.143 |
| no_confounding | 8 | 0.000 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 20 | 0.000 |
| medium (tol <= gap < 0.08) | 23 | 0.478 |
| easy (gap >= 0.08) | 17 | 0.529 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 20 | 0 | 0 |
| do_gt_obs | 20 | 0 | 0 |
| approx_equal | 17 | 3 | 0 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
