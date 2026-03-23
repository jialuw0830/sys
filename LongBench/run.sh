#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MODEL_PATH=${MODEL_PATH:-$HOME/models/GLM-4.7-Flash}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$MODEL_PATH}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8003}

vllm serve "$MODEL_PATH" \
	--host "$HOST" \
	--port "$PORT" \
	--served-model-name "$SERVED_MODEL_NAME" \
	--api-key token-abc123 \
	--tensor-parallel-size 1 \
	--gpu-memory-utilization 0.92 \
	--max-model-len 131072 \
	--trust-remote-code