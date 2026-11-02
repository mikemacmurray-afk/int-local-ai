#!/usr/bin/env bash
# ==============================================================================
# start_vllm_mlx.sh
# Starts vllm-mlx on Mac Mini M4 Pro (48 GB) serving Qwen3.6-35B-A3B (6-bit MLX)
# with trie-based prefix caching, SSD cache spill, OpenAI /v1/chat/completions,
# and Anthropic /v1/messages compatibility.
# ==============================================================================

set -euo pipefail

MODEL_ID="${MODEL_ID:-mlx-community/Qwen3.6-35B-A3B-6bit}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
SSD_CACHE_DIR="${SSD_CACHE_DIR:-$HOME/.vllm-mlx/prefix-cache}"

mkdir -p "${SSD_CACHE_DIR}"

echo "==> Starting vllm-mlx on ${HOST}:${PORT}"
echo "    Model:          ${MODEL_ID}"
echo "    Max Context:    ${MAX_MODEL_LEN} tokens"
echo "    SSD Cache Dir:  ${SSD_CACHE_DIR}"

exec vllm-mlx serve "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --ssd-cache-dir "${SSD_CACHE_DIR}" \
  --warm-prompts

