#!/usr/bin/env bash
# ==============================================================================
# start_vllm_mlx.sh
# Starts vllm-mlx on Mac Mini M4 Pro (64 GB) serving Qwen3.6-35B-A3B (8-bit or 6-bit MLX)
# with Radix/Trie prefix caching, 4096-token fused prefill step size, NVMe SSD
# KV-cache tiering, OpenAI /v1/chat/completions, and Anthropic /v1/messages.
# ==============================================================================

set -euo pipefail

# Default to Full 8-Bit on 64 GB Mac Mini (~36.8 GB weights, leaving ~20 GB free)
# Override with MODEL_ID="mlx-community/Qwen3.6-35B-A3B-6bit" for dual-model setups
MODEL_ID="${MODEL_ID:-mlx-community/Qwen3.6-35B-A3B-8bit}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
PREFILL_STEP_SIZE="${PREFILL_STEP_SIZE:-4096}"
SSD_CACHE_DIR="${SSD_CACHE_DIR:-$HOME/.vllm-mlx/prefix-cache}"

mkdir -p "${SSD_CACHE_DIR}"

echo "==> Starting vllm-mlx on ${HOST}:${PORT} (64 GB Mac Mini M4 Pro Profile)"
echo "    Model:              ${MODEL_ID}"
echo "    Max Context:        ${MAX_MODEL_LEN} tokens"
echo "    Prefill Step Size:  ${PREFILL_STEP_SIZE} tokens (TTFT optimized)"
echo "    SSD Cache Dir:      ${SSD_CACHE_DIR}"

exec vllm-mlx serve "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --prefill-step-size "${PREFILL_STEP_SIZE}" \
  --enable-prefix-cache \
  --ssd-cache-dir "${SSD_CACHE_DIR}" \
  --warm-prompts
