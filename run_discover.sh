#!/bin/bash
# run_discover.sh — Single-node TTT-Discover launcher for RunAI / any 8xH100 node.
#
# Usage:
#   MODEL_NAME=Qwen/Qwen3-8B EXPERIMENT=my_exp bash run_discover.sh
#
# Environment variables (all optional, have defaults):
#   MODEL_NAME          HuggingFace model ID or local path   (default: Qwen/Qwen3-8B)
#   EXPERIMENT          Experiment name for logging           (default: discover_$(date))
#   VLLM_GPU            GPU index for vLLM inference          (default: 0)
#   TRAIN_GPU           GPU index for LoRA training           (default: 1)
#   VLLM_PORT           vLLM server port                      (default: 8000)
#   LORA_RANK           LoRA rank                             (default: 32)
#   NUM_EPOCHS          Training epochs                       (default: 50)
#   CHECKPOINT_DIR      Directory for checkpoints             (default: ./checkpoints)

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
EXPERIMENT="${EXPERIMENT:-discover_$(date +%Y%m%d_%H%M%S)}"
VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"
VLLM_PORT="${VLLM_PORT:-8000}"
LORA_RANK="${LORA_RANK:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

VLLM_URL="http://localhost:${VLLM_PORT}/v1"

echo "=========================================="
echo "TTT-Discover Single-Node Launcher"
echo "  Model:      ${MODEL_NAME}"
echo "  Experiment: ${EXPERIMENT}"
echo "  vLLM GPU:   ${VLLM_GPU}   (port ${VLLM_PORT})"
echo "  Train GPU:  ${TRAIN_GPU}"
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Step 1: Start vLLM inference server on VLLM_GPU
# ---------------------------------------------------------------------------
echo "[1/3] Starting vLLM server on GPU ${VLLM_GPU}..."

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --enable-lora \
    --max-lora-rank "${LORA_RANK}" \
    --max-loras 4 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --port "${VLLM_PORT}" \
    --trust-remote-code \
    &

VLLM_PID=$!

# ---------------------------------------------------------------------------
# Step 2: Wait for vLLM to be healthy
# ---------------------------------------------------------------------------
echo "[2/3] Waiting for vLLM to be ready..."
VLLM_HEALTH_URL="http://localhost:${VLLM_PORT}/health"
MAX_WAIT=300  # 5 minutes
ELAPSED=0
until curl -sf "${VLLM_HEALTH_URL}" > /dev/null 2>&1; do
    if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s"
        kill "${VLLM_PID}" 2>/dev/null || true
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ... still waiting (${ELAPSED}s elapsed)"
done
echo "  vLLM is ready."

# ---------------------------------------------------------------------------
# Step 3: Run TTT-Discover training (Ray on all remaining CPUs)
# ---------------------------------------------------------------------------
echo "[3/3] Starting TTT-Discover training on GPU ${TRAIN_GPU}..."

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python -m ttt_discover.discovery \
    model_name="${MODEL_NAME}" \
    lora_rank="${LORA_RANK}" \
    num_epochs="${NUM_EPOCHS}" \
    experiment_name="${EXPERIMENT}" \
    checkpoint_dir="${CHECKPOINT_DIR}" \
    vllm_url="${VLLM_URL}" \
    training_device="cuda:0" \
    "$@"

# Cleanup vLLM on exit
echo "Training complete. Stopping vLLM server..."
kill "${VLLM_PID}" 2>/dev/null || true
