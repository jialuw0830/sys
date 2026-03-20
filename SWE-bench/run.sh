export CUDA_VISIBLE_DEVICES=5
MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen3.5-9B}
vllm serve "$MODEL_PATH" \
  --served-model-name qwen3.5-9b \
  --api-key token-abc123 \
  --host 127.0.0.1 \
  --port 8002 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 131072 \
  --trust-remote-code