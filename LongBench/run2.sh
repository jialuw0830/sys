#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen3.5-9B-Instruct}
PORT=${PORT:-8001}
N_PROC=${N_PROC:-1}
BATCH_SIZE=${BATCH_SIZE:-4}

export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:${PORT}/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-token-abc123}

python pred.py --model "$MODEL" -n "$N_PROC" --batch_size "$BATCH_SIZE"