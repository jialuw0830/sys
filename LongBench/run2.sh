#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen3.5-27B}
PORT=${PORT:-8002}
N_PROC=${N_PROC:-1}

export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:${PORT}/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-token-abc123}

python pred.py --model "$MODEL" -n "$N_PROC"