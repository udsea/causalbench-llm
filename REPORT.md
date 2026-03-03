# CausalBench-LLM — Report (v0)

## TL;DR
CausalBench-LLM evaluates whether models can distinguish **observational association** from **interventional effects** by predicting the relation between:
- **P(Y>0 | X=1)** vs **P(Y>0 | do(X=1))**

Across current runs, the benchmark surfaces two practical bottlenecks:
1) **Reasoning failures** (class collapse, directional bias, weak handling of near-equality)
2) **Reliability failures** (strict JSON parsing breaks entirely for some models)

This makes it useful both for causal-capability measurement and for real-world model selection under structured-output constraints.

---

## Setup
- **Codebase:** `causalbench-llm` (synthetic linear-Gaussian SCM tasks; intervention vs observation)
- **Eval:** `python -m causalbench.eval.run_eval`
- **Summarize:** `python -m causalbench.eval.summarize`
- **Task:** predict whether `P(Y>0 | X=1)` is **greater than**, **less than**, or **≈ equal** to `P(Y>0 | do(X=1))`
- **Labels:** `obs_gt_do`, `do_gt_obs`, `approx_equal`
- **Motifs:** confounding, confounding-only, no-confounding, mediation, collider, instrumental variable, anti-causal, backdoor-adjustable
- **Aggregate source:** `results/summary_model_grid.md` (19 runs discovered)

---

## Models tested (representative)
- Qwen2.5 Instruct: 0.5B / 1.5B / 3B variants (multiple prompt/model variants)
- `distilgpt2_v1` (smoke-style run)
- `deepseek_r1_1p5b_v1`
- Heuristic baselines (`*_heuristic`)

---

## Key findings
- Best observed accuracy in the aggregated table: **0.550**
  - `qwen25_v5_heuristic`: `n=60`, `parse_rate=1.000`, `macro_f1=0.524`
- A non-heuristic run reaches similar top-line accuracy but is unbalanced:
  - `qwen25_v7`: `acc=0.542` with **skewed class behavior**
    - `recall_do=0.000`, `recall_eq=0.659` → uneven competence across causal classes
- Several runs remain near chance (~0.333 for 3-way labels) with explicit **class collapse**:
  - `qwen25`: predicts `obs_gt_do` for **95%** of outputs
  - `qwen25_3b_v1`: predicts `do_gt_obs` for **100%** of outputs
- Parse robustness is highly model-dependent:
  - some runs have `parse_rate=0.000` (e.g., `distilgpt2_v1`, `deepseek_r1_1p5b_v1`, `_smoketest_distilgpt2`, `dev`)
  - when parsing fails completely, task performance is **not interpretable**
- Overall: the benchmark distinguishes **reasoning** vs **formatting** failure modes, and both matter in practice.

> Interpretation: Many failures look like robustness failures rather than “no capability”: models can appear competitive on accuracy while collapsing on    minority classes or failing strict formatting. This suggests evaluating causal reasoning requires both balanced metrics and reliability-aware scoring under distribution shift.

---

## Observed failure modes
1) **Class collapse**: defaulting to one label irrespective of motif/context  
2) **Near-equality brittleness**: `approx_equal` often has weak recall  
3) **Directional asymmetry**: models handle `obs_gt_do` much better than `do_gt_obs` (or vice versa), suggesting directional bias / shortcutting  
4) **Schema/JSON failures**: strict structured output breaks → zero usable predictions  
5) **Shortcut reliance**: high accuracy can coexist with poor macro-F1 balance

---

## Limitations
- Linear-Gaussian SCMs are controlled but narrow; external validity to real-world text settings is limited
- Run sizes vary (`n` ranges from very small to 120), so cross-run comparisons are noisy
- Metrics are mostly classification-focused; calibration/uncertainty quality is not yet first-class
- Some results mix prompt/model variants under similar naming, complicating attribution

---

## Next steps
1) Standardize: fixed split + fixed `n` for fair deltas  
2) Always export: confusion matrices + motif-level breakdowns per run  
3) Add calibration: ECE / Brier-style proxies when confidence is available  
4) Separate reasoning vs formatting: add JSON repair/retry instrumentation and report both “raw” and “post-repair” scores  
5) Expand SCM families: non-linear + discrete variants to reduce overfitting to a single synthetic family