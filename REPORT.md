# causalbench-llm Report

## Setup

- Codebase: `causalbench-llm` with synthetic linear-Gaussian SCM tasks for intervention-vs-observation comparison.
- Evaluation pipeline: `python -m causalbench.eval.run_eval` plus `python -m causalbench.eval.summarize`.
- Task framing: model predicts whether `P(Y>0 | X~1)` is greater than, less than, or approximately equal to `P(Y>0 | do(X=1))`.
- Labels: `obs_gt_do`, `do_gt_obs`, `approx_equal`.
- Motifs covered: confounding, confounding-only, no-confounding, mediation, collider, instrumental variable, anti-causal, and backdoor-adjustable.
- Current aggregate source used in this report: `results/summary_model_grid.md` (19 runs discovered).

## Models Tested

Representative runs from the grid include:
- `Qwen/Qwen2.5-0.5B-Instruct` variants (`qwen25`, `qwen25_v2`, `qwen25_v3`, `qwen25_v4`, `qwen25_v5`, `qwen25_v6`, `qwen25_v7`, `qwen25_v8`)
- `Qwen/Qwen2.5-1.5B-Instruct` variants (`qwen25_1p5b_v1`, `qwen25_1p5b_v1_heuristic`)
- `Qwen/Qwen2.5-3B-Instruct` variants (`qwen25_3b_v1`, `qwen25_3b_v1_heuristic`)
- `distilgpt2_v1` smoke-style run
- `deepseek_r1_1p5b_v1`
- Heuristic baselines (run labels ending in `_heuristic`)

## Key Findings

1. Best observed accuracy in the aggregated table is `0.550` (`qwen25_v5_heuristic`, `n=60`, `parse_rate=1.000`, `macro_f1=0.524`).
2. A non-heuristic model variant reaches similar top-line accuracy (`qwen25_v7`: `acc=0.542`) but with skewed class behavior (`recall_do=0.000`, `recall_eq=0.659`), indicating uneven causal-class competence.
3. Several runs remain near chance-level for 3-way labels (`~0.333`), including explicit collapse patterns:
- `qwen25` predicts `obs_gt_do` for `95%` of outputs (`collapse=obs_gt_do`).
- `qwen25_3b_v1` predicts `do_gt_obs` for `100%` of outputs (`collapse=do_gt_obs`).
4. Parse robustness is model-dependent: multiple runs have `parse_rate=0.000` (`distilgpt2_v1`, `deepseek_r1_1p5b_v1`, `_smoketest_distilgpt2`, `dev`), making task performance uninterpretable there.
5. Overall, the benchmark is exposing both reasoning failures and output-format reliability failures, which is useful for practical model selection.

## Failure Modes

- Class-collapse behavior: some models default to one answer category regardless of motif context.
- Near-equality confusion: `approx_equal` often has weak recall outside a few runs, suggesting brittle handling of small causal gaps.
- Directional asymmetry: some models handle `obs_gt_do` much better than `do_gt_obs` (or vice versa), implying learned directional bias rather than structural reasoning.
- Parsing/schema failures: strict JSON outputs fail completely for some models, producing zero usable predictions.
- Potential shortcut reliance: high scores in a few runs can coexist with poor macro-F1 balance, indicating non-causal heuristics.

## Limitations + Next Steps

### Limitations

- Synthetic linear-Gaussian SCMs are controlled but narrow; external validity to real-world text tasks is limited.
- Run sizes vary (`n` ranges from very small to 120), so direct cross-run comparison is noisy.
- Current headline metrics are mostly classification-focused; calibration and uncertainty quality are not yet first-class.
- Some reported results mix different prompt/model variants under similar naming, which complicates attribution.

### Next steps

1. Standardize all comparisons on one fixed split and one fixed `n` to improve fairness and confidence in deltas.
2. Add confusion matrices and motif-level breakdowns to every run artifact by default.
3. Track calibration metrics (ECE/Brier-style proxies for categorical confidence) when models expose confidence.
4. Add robust JSON repair/retry instrumentation to separate reasoning errors from formatting errors.
5. Extend beyond linear SCMs (non-linear and discrete variants) to reduce overfitting to one synthetic family.
