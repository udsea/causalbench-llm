# causalbench-llm

`causalbench-llm` is a synthetic structural causal model (SCM) benchmark for testing whether LLMs can distinguish observational associations from interventional effects under common causal traps. It generates linear-Gaussian SCM instances, asks models to predict the relationship between `P(Y>0 | X~1)` and `P(Y>0 | do(X=1))`, and scores strict JSON outputs.

## Tasks Implemented

- `intervention_compare_confounding`: `U -> X, U -> Y, X -> Y`
- `intervention_compare_confounding_only`: `U -> X, U -> Y`
- `intervention_compare_no_confounding`: `X -> Y`
- `intervention_compare_mediation`: `X -> M -> Y`
- `intervention_compare_collider`: `X -> Y, X -> Z, Y -> Z` (collider at `Z`)
- `intervention_compare_instrumental_variable`: `Z -> X -> Y, U -> X, U -> Y`
- `intervention_compare_anti_causal`: `Y -> X, U -> X, U -> Y`
- `intervention_compare_backdoor_adjustable`: `W -> X, W -> Y, X -> Y` with observed `W`

By default, instance generation uses rejection sampling to target a balanced gold-label mix:

- ~1/3 `obs_gt_do`
- ~1/3 `do_gt_obs`
- ~1/3 `approx_equal`

## One-Command Run

```bash
uv run python -m causalbench.eval.run_eval \
  --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
  --device cpu \
  --n-instances 240 \
  --seed 0 \
  --scm-kinds all \
  --balance-labels \
  --stratify-motif-label \
  --n-prompt-obs-samples 2000 \
  --x-band 0.25 \
  --eq-margin 0.06 \
  --dir-margin 0.06 \
  --discard-ambiguous \
  --out-dir results/runs/dev
```

Then build a markdown summary table:

```bash
uv run python -m causalbench.eval.summarize \
  results/runs/dev/results.jsonl \
  --out-table results/runs/dev/results_table.md
```

## Example Results Table

Example format (values shown are from `results/runs/qwen05b/results_table.md` in this repo):

| label | n | acc |
|---|---:|---:|
| do_gt_obs | 10 | 1.000 |
| obs_gt_do | 13 | 0.000 |
| approx_equal | 7 | 0.000 |

To reproduce this format, run the two commands above and open `results/runs/dev/results_table.md`.

`summarize` also reports difficulty buckets by `gap = |obs_prob - do_prob|`:
- `gap < tol` (borderline)
- `tol <= gap <= 0.08` (medium)
- `gap > 0.08` (easy)

## Roadmap

- [x] Add multiple causal motifs beyond confounding
- [x] Balance labels by construction via rejection sampling
- [x] Strict JSON scoring
- [x] GitHub Actions CI (`pytest`, `ruff`, optional `mypy`)
- [ ] Add motif-specific prompts that test adjustment/selection reasoning explicitly
- [ ] Add calibration metrics and confidence-aware scoring
- [ ] Add non-linear SCM families and discrete variable variants
- [ ] Publish a fixed benchmark split + leaderboard script
