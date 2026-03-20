#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1,3

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"

export DEEPSPEED_LOCAL_PATH="${DEEPSPEED_LOCAL_PATH:-${PROJECT_ROOT}/DeepSpeed}"
export LOCAL_PROFILE_FLOPS="${LOCAL_PROFILE_FLOPS:-1}"
export LOCAL_MAX_TOKENS="${LOCAL_MAX_TOKENS:-8096}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-8}"


cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-glm-4.7-flash-local}"
DATA_FILENAME="${DATA_FILENAME:-dataset/gpqa_main.csv}"
PROMPT_TYPE="${PROMPT_TYPE:-zero_shot}"
SEED="${SEED:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

CMD=(
  python
  baselines/run_baseline.py
  main
  --model_name "$MODEL_NAME"
  --data_filename "$DATA_FILENAME"
  --prompt_type "$PROMPT_TYPE"
  --seed "$SEED"
)

if [[ -n "$MAX_EXAMPLES" ]]; then
  CMD+=(--max_examples "$MAX_EXAMPLES")
fi

"${CMD[@]}"