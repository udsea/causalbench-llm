# CausalBench Technical Note (Nightly Alpha)

## Scope
This note documents the benchmark hypothesis, the structural assumptions behind the synthetic SCMs, and why observational and interventional probabilities can diverge even when both are estimated from the same generative system.

## Main Hypothesis
LLMs default to observational reasoning when causal evidence is ambiguous.

Operationally, this appears as:
- high mass on one or two "safe" labels,
- near-zero recall on one direction class (`do_gt_obs` or `obs_gt_do`),
- better performance in buckets where prompt evidence is more decisive.

## Core Quantity
Each instance asks the model to compare:
- \(A = P(Y>0 \mid X \approx x_1)\) (observational), and
- \(B = P(Y>0 \mid do(X=x_1))\) (interventional).

The benchmark label is one of:
- `obs_gt_do` when \(A>B\),
- `do_gt_obs` when \(B>A\),
- `approx_equal` when \(|A-B|\) is within a margin.

## Why Observational and Interventional Can Differ
Consider a linear SCM with unobserved confounding:
- \(X = aU + \epsilon_x\)
- \(Y = bX + cU + \epsilon_y\)
- \(U, \epsilon_x, \epsilon_y\) mean-zero and mutually independent.

### Interventional Mean
Under intervention, incoming arrows into \(X\) are removed and \(X\) is fixed to \(x\):
\[
E[Y \mid do(X=x)] = bx.
\]

### Observational Mean
Conditioning on \(X=x\) does not break confounding:
\[
E[Y \mid X=x] = b x + c\,E[U\mid X=x].
\]
For jointly Gaussian \((U, X)\),
\[
E[U\mid X=x] = \frac{\operatorname{Cov}(U,X)}{\operatorname{Var}(X)}x,
\]
so
\[
E[Y\mid X=x] = \left(b + c\frac{\operatorname{Cov}(U,X)}{\operatorname{Var}(X)}\right)x.
\]

The additional term
\[
\Delta_{conf}(x) = c\frac{\operatorname{Cov}(U,X)}{\operatorname{Var}(X)}x
\]
is observational bias from confounding.

Sign of \(\Delta_{conf}(x)\) determines whether observational association over- or under-estimates causal effect, and therefore whether `obs_gt_do` or `do_gt_obs` should hold.

## Identifiability Discussion
From coarse observational summaries alone (single-band conditional probability, baseline rate, and global correlation), direction of \(A-B\) is not always identifiable.

Implications:
- some instances are fundamentally ambiguous for text-only predictors,
- "approx_equal" and directional classes may overlap in feature space,
- stronger evidence (e.g., adjusted estimates, multiple conditioning anchors, or proxy variables) can improve identifiability.

## Formal Counterexample
Two SCMs can share similar observational summaries but imply opposite interventional ordering:

1. Positive confounding bias:
- choose \(c>0\), \(\operatorname{Cov}(U,X)>0\), giving \(\Delta_{conf}(x)>0\), often pushing toward `obs_gt_do`.

2. Negative confounding bias:
- choose \(c<0\), \(\operatorname{Cov}(U,X)>0\) (or equivalent sign flip), giving \(\Delta_{conf}(x)<0\), often pushing toward `do_gt_obs`.

If summary statistics compress away the confounding direction signal, both can look similar to an LLM while requiring opposite labels.

## Evaluation Design in This Repo
To make model behavior diagnosable, this repo uses:
- balanced labels by construction,
- motif-aware generation,
- fixed split export for reproducibility,
- per-label metrics, confusion matrices, and reliability buckets.

## Reproducible Fixed Split
Generate once and reuse for all models:

```bash
uv run python -m causalbench.eval.export_split \
  --out-jsonl experiments/fixed_split_v1.jsonl \
  --out-meta-json experiments/fixed_split_v1_meta.json \
  --n-instances 120 \
  --seed 0 \
  --scm-kinds all \
  --balance-labels \
  --stratify-motif-label
```

Evaluate models on exactly this split via:

```bash
uv run python -m causalbench.eval.run_eval \
  --instances-jsonl experiments/fixed_split_v1.jsonl \
  --backend hf \
  --model-name "Qwen/Qwen2.5-3B-Instruct" \
  --out-dir results/runs/qwen25_3b_fixed_v1
```

