#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPSPEED_LOCAL_PATH="${DEEPSPEED_LOCAL_PATH:-/home/rzh/jialu/sys/DeepSpeed}"
export PYTHONPATH="${DEEPSPEED_LOCAL_PATH}:/home/rzh/jialu/sys/transformers/src:${PYTHONPATH}"
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# Keep Triton/temporary files off home/NFS mounts to avoid DeepSpeed cleanup races.
LOCAL_CACHE_ROOT="${LOCAL_CACHE_ROOT:-/tmp/${USER}/gpqa_runtime_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_CACHE_ROOT}/triton_autotune}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOCAL_CACHE_ROOT}/torchinductor}"
export TMPDIR="${TMPDIR:-${LOCAL_CACHE_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"

MODEL_NAME="${MODEL_NAME:-qwen3.5-27b-local}"
DATA_FILENAME="${DATA_FILENAME:-dataset/gpqa_main.csv}"
PROMPT_TYPE="${PROMPT_TYPE:-zero_shot}"
SEED="${SEED:-0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
export LOCAL_MAX_TOKENS="${LOCAL_MAX_TOKENS:-4096}"
export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-32}"
export LOCAL_PROFILE_FLOPS="${LOCAL_PROFILE_FLOPS:-1}"
export LOCAL_FORCE_BATCH1_FOR_FLOPS="${LOCAL_FORCE_BATCH1_FOR_FLOPS:-0}"



CMD=(
  /home/rzh/conda-environment/sys/bin/python baselines/run_baseline.py main
  --model_name "$MODEL_NAME"
  --data_filename "$DATA_FILENAME"
  --prompt_type "$PROMPT_TYPE"
  --seed "$SEED"
)

if [[ -n "$MAX_EXAMPLES" ]]; then
  CMD+=(--max_examples "$MAX_EXAMPLES")
fi

"${CMD[@]}"
