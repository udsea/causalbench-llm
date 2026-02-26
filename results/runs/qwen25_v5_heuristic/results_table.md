# Results
- Total instances: 60
- Parse rate: 1.000
- Accuracy: 0.550
- Macro accuracy: 0.550
- Macro F1: 0.524
- Invalid/unknown predictions: 0

## Predicted label distribution

| predicted label | n | share |
|---|---:|---:|
| obs_gt_do | 28 | 0.467 |
| do_gt_obs | 25 | 0.417 |
| approx_equal | 7 | 0.117 |

## Breakdown by gold label

| label | n | acc | f1 |
|---|---:|---:|---:|
| obs_gt_do | 20 | 0.750 | 0.625 |
| do_gt_obs | 20 | 0.650 | 0.578 |
| approx_equal | 20 | 0.250 | 0.370 |

## Breakdown by SCM motif

| scm_kind | n | acc |
|---|---:|---:|
| anti_causal | 11 | 0.818 |
| backdoor_adjustable | 8 | 0.125 |
| collider | 4 | 0.000 |
| confounding | 12 | 0.667 |
| confounding_only | 7 | 1.000 |
| instrumental_variable | 10 | 0.600 |
| mediation | 4 | 0.250 |
| no_confounding | 4 | 0.250 |

## Breakdown by gap difficulty

| difficulty bucket | n | acc |
|---|---:|---:|
| borderline (gap < tol) | 20 | 0.250 |
| medium (tol <= gap < 0.08) | 9 | 0.667 |
| easy (gap >= 0.08) | 31 | 0.710 |

## Breakdown by n_in_band reliability

| n_in_band bucket | n | acc |
|---|---:|---:|
| < 100 | 0 | 0.000 |
| 100-199 | 0 | 0.000 |
| >= 200 | 0 | 0.000 |
| missing | 60 | 0.550 |

## Confusion matrix (gold x pred)

| gold \ pred | obs_gt_do | do_gt_obs | approx_equal |
|---|---:|---:|---:|
| obs_gt_do | 15 | 4 | 1 |
| do_gt_obs | 6 | 13 | 1 |
| approx_equal | 7 | 8 | 5 |

Invalid/unknown predictions are excluded from the 3x3 matrix and counted above.
