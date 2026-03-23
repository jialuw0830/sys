#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-GLM-4.7-Flash}
PORT=${PORT:-8003}
N_PROC=${N_PROC:-1}

export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:${PORT}/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-token-abc123}

python pred.py --model "$MODEL" -n "$N_PROC"