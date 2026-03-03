# causalbench-llm

## What It Is / Why It Matters
`causalbench-llm` is a synthetic benchmark for testing whether language models can reason causally, not just correlate.
It generates linear-Gaussian SCM instances with known structure and known ground-truth intervention effects.
Each prompt asks models to compare `P(Y>0 | X~1)` against `P(Y>0 | do(X=1))` under causal motifs that induce common reasoning traps.
Strict JSON output parsing plus deterministic scoring makes model behavior easy to compare and reproduce.
This matters because many real-world decisions depend on intervention-aware reasoning, not observational pattern matching.

## Install + Run (3 commands)

```bash
uv sync
uv run python -m causalbench.eval.run_eval --backend hf --model-name "Qwen/Qwen2.5-0.5B-Instruct" --device cpu --n-instances 120 --seed 0 --scm-kinds all --balance-labels --stratify-motif-label --out-dir results/runs/dev
uv run python -m causalbench.eval.summarize results/runs/dev/results.jsonl --out-table results/runs/dev/results_table.md
```

## What's Inside

### SCM motifs
- confounding (`U -> X`, `U -> Y`, `X -> Y`)
- confounding-only (`U -> X`, `U -> Y`)
- no-confounding direct effect (`X -> Y`)
- mediation (`X -> M -> Y`)
- collider (`X -> Y`, `X -> Z`, `Y -> Z`)
- instrumental variable (`Z -> X -> Y`, `U -> X`, `U -> Y`)
- anti-causal (`Y -> X`, `U -> X`, `U -> Y`)
- backdoor-adjustable (`W -> X`, `W -> Y`, `X -> Y`)

### Metrics
- parse rate (strict JSON validity)
- overall accuracy and macro F1
- per-label recall for `obs_gt_do`, `do_gt_obs`, and `approx_equal`
- prediction mix and collapse detection
- difficulty and reliability buckets in summary reports

### Baselines
- Hugging Face local-model backend
- OpenRouter API-model backend
- non-LLM heuristic baseline: `python -m causalbench.eval.heuristic_baseline`

## Screenshot (Results Table)

![Results table screenshot](docs/results-table-screenshot.svg)

## Technical Note

See [`docs/technical_note.md`](docs/technical_note.md) for derivations, assumptions, and benchmark framing.
