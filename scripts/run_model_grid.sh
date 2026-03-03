#!/usr/bin/env bash
set -euo pipefail

SPLIT_PATH=${1:-experiments/fixed_split_v1.jsonl}
OUT_ROOT=${2:-results/runs/model_grid_v1}
BACKEND=${BACKEND:-hf}
DEVICE=${DEVICE:-cpu}
MODELS_CSV=${MODELS:-Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct}

if [[ ! -f "$SPLIT_PATH" ]]; then
  echo "Missing split file: $SPLIT_PATH"
  echo "Generate one first with: uv run python -m causalbench.eval.export_split --out-jsonl $SPLIT_PATH"
  exit 1
fi

mkdir -p "$OUT_ROOT"
IFS=',' read -r -a MODELS <<< "$MODELS_CSV"

for MODEL in "${MODELS[@]}"; do
  SLUG=$(echo "$MODEL" | tr '/:.' '_')
  RUN_DIR="$OUT_ROOT/$SLUG"
  echo "=== Running $MODEL ($BACKEND) ==="

  uv run python -m causalbench.eval.run_eval \
    --backend "$BACKEND" \
    --model-name "$MODEL" \
    --device "$DEVICE" \
    --instances-jsonl "$SPLIT_PATH" \
    --out-dir "$RUN_DIR"

  uv run python -m causalbench.eval.summarize \
    "$RUN_DIR/results.jsonl" \
    --out-table "$RUN_DIR/results_table.md"

done

uv run python -m causalbench.eval.aggregate_reports \
  --results-root "$OUT_ROOT" \
  --out-table "$OUT_ROOT/summary_model_grid.md"

echo "Wrote $OUT_ROOT/summary_model_grid.md"
