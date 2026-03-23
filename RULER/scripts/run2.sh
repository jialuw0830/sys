#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run2.sh [model_name] [benchmark_name]
# Optional env:
#   STREAM=1  # enable streaming mode in call_api
export CUDA_VISIBLE_DEVICES="2"
MODEL_NAME="${1:-qwen3.5-9b}"
BENCHMARK="${2:-synthetic}"

GPUS="${GPUS:-1}"
ROOT_DIR="${ROOT_DIR:-benchmark_root}"
MODEL_DIR="${MODEL_DIR:-/home/cc/models}"
ENGINE_DIR="${ENGINE_DIR:-.}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8006}"
API_KEY="${API_KEY:-token-abc123}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b}"

source config_models.sh
source config_tasks.sh

STREAM="${STREAM:-1}"
THREADS_LIST=(2 8 16 32 64 128)
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"

# config_models.sh was authored without nounset; predefine variables it probes.
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
TOKENIZER_TYPE="${TOKENIZER_TYPE:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"
AZURE_ID="${AZURE_ID:-}"
AZURE_SECRET="${AZURE_SECRET:-}"
AZURE_ENDPOINT="${AZURE_ENDPOINT:-}"

MODEL_CONFIG=$(MODEL_SELECT "${MODEL_NAME}" "${MODEL_DIR}" "${ENGINE_DIR}")
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
	echo "Model: ${MODEL_NAME} is not supported"
	exit 1
fi

declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
	echo "Benchmark: ${BENCHMARK} is not supported"
	exit 1
fi

SERVER_PID=""

cleanup() {
	if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
		kill "${SERVER_PID}" >/dev/null 2>&1 || true
	fi
}
trap cleanup EXIT

for THREADS in "${THREADS_LIST[@]}"; do
	echo "Starting vLLM server via official entrypoint with --max-num-seqs=${VLLM_MAX_NUM_SEQS} (threads=${THREADS})..."

	vllm serve "${MODEL_PATH}" \
		--host "${HOST}" \
		--port "${PORT}" \
		--tensor-parallel-size "${GPUS}" \
		--trust-remote-code \
		--served-model-name "${SERVED_MODEL_NAME}" \
		--generation-config vllm \
		--max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
		&
	SERVER_PID=$!

	sleep 10

	total_time=0
	for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
		RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/threads_${THREADS}/${MAX_SEQ_LENGTH}"
		DATA_DIR="${RESULTS_DIR}/data"
		PRED_DIR="${RESULTS_DIR}/pred"
		mkdir -p "${DATA_DIR}" "${PRED_DIR}"

		for TASK in "${TASKS[@]}"; do
			python data/prepare.py \
				--save_dir "${DATA_DIR}" \
				--benchmark "${BENCHMARK}" \
				--task "${TASK}" \
				--tokenizer_path "${TOKENIZER_PATH}" \
				--tokenizer_type "${TOKENIZER_TYPE}" \
				--max_seq_length "${MAX_SEQ_LENGTH}" \
				--model_template_type "${MODEL_TEMPLATE_TYPE}" \
				--num_samples "${NUM_SAMPLES}" \
				${REMOVE_NEWLINE_TAB}

			start_time=$(date +%s)
			python pred/call_api.py \
				--data_dir "${DATA_DIR}" \
				--save_dir "${PRED_DIR}" \
				--benchmark "${BENCHMARK}" \
				--task "${TASK}" \
				--server_type openai \
				--server_host 127.0.0.1 \
				--server_port "${PORT}" \
				--base_url "http://127.0.0.1:${PORT}/v1" \
				--api_key "${API_KEY}" \
				--model_name_or_path "${SERVED_MODEL_NAME}" \
				--temperature "${TEMPERATURE}" \
				--top_k "${TOP_K}" \
				--top_p "${TOP_P}" \
				--threads "${THREADS}" \
				--batch_size "${BATCH_SIZE}" \
				$( [ "${STREAM}" = "1" ] && echo "--stream" ) \
				${STOP_WORDS}
			end_time=$(date +%s)
			total_time=$((total_time + end_time - start_time))
		done

		python eval/evaluate.py \
			--data_dir "${PRED_DIR}" \
			--benchmark "${BENCHMARK}"
	done

	echo "Total time spent on call_api with threads=${THREADS}: $total_time seconds"

	SUMMARY_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/threads_${THREADS}"
	SUMMARY_FILE="${SUMMARY_DIR}/summary.json"
	mkdir -p "${SUMMARY_DIR}"

	SEQ_LENGTHS_JSON=$(printf '"%s",' "${SEQ_LENGTHS[@]}")
	SEQ_LENGTHS_JSON="[${SEQ_LENGTHS_JSON%,}]"

	cat > "${SUMMARY_FILE}" <<EOF
{
	"model_name": "${MODEL_NAME}",
	"served_model_name": "${SERVED_MODEL_NAME}",
	"benchmark": "${BENCHMARK}",
	"threads": ${THREADS},
	"vllm_max_num_seqs": ${VLLM_MAX_NUM_SEQS},
	"total_time_seconds": ${total_time},
	"stream": ${STREAM},
	"seq_lengths": ${SEQ_LENGTHS_JSON},
	"generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

	echo "Summary written to ${SUMMARY_FILE}"

	cleanup
	SERVER_PID=""
done