#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_NAME=${DATASET_NAME:-princeton-nlp/SWE-bench_Lite}
PREDICTIONS_PATH=${PREDICTIONS_PATH:-$SCRIPT_DIR/outputs/qwen3.5-9b__SWE-bench_Lite__test.jsonl}
MAX_WORKERS=${MAX_WORKERS:-8}
RUN_ID=${RUN_ID:-qwen35_9b_eval}
MODAL=${MODAL:-false}

# use --predictions_path 'gold' to verify the gold patches
# use --run_id to name the evaluation run
# use --modal true to run on Modal
if [[ "$MODAL" == "true" ]]; then
  python -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET_NAME" \
    --predictions_path "$PREDICTIONS_PATH" \
    --max_workers "$MAX_WORKERS" \
    --run_id "$RUN_ID" \
    --modal true
else
  python -m swebench.harness.run_evaluation \
    --dataset_name "$DATASET_NAME" \
    --predictions_path "$PREDICTIONS_PATH" \
    --max_workers "$MAX_WORKERS" \
    --run_id "$RUN_ID"
fi