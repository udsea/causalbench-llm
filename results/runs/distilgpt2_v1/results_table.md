# Results
- Total instances: 12
- Parse rate: 0.000
- Accuracy: 0.000
- Macro accuracy: 0.000
- Macro F1: 0.000
- Invalid/unknown predictions: 12

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 0 | 0.000 |
| do_gt_obs | 0 | 0.000 |
| approx_equal | 0 | 0.000 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 4 | 0.000 | 0.000 |
| do_gt_obs | 4 | 0.000 | 0.000 |
| approx_equal | 4 | 0.000 | 0.000 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 2 | 0.000 |
| backdoor_adjustable | 2 | 0.000 |
| collider | 1 | 0.000 |
| confounding | 2 | 0.000 |
| confounding_only | 3 | 0.000 |
| mediation | 1 | 0.000 |
| no_confounding | 1 | 0.000 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 4 | 0.000 |
| medium (tol <= gap < 0.08) | 1 | 0.000 |
| easy (gap >= 0.08) | 7 | 0.000 |

## Breakdown by n_in_band reliability

| n_in_band bucket | n | acc |
|---|---:|---:|
| < 100 | 0 | 0.000 |
| 100-199 | 5 | 0.000 |
| >= 200 | 7 | 0.000 |
| missing | 0 | 0.000 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 0 | 0 | 0 |
| do_gt_obs | 0 | 0 | 0 |
| approx_equal | 0 | 0 | 0 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
