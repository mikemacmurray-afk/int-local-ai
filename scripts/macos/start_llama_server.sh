#!/usr/bin/env bash
# ==============================================================================
# start_llama_server.sh
# Starts llama.cpp's llama-server on Mac Mini M4 Pro (48 GB) with Metal GPU
# offload, Flash Attention, Q8_0 KV Cache, Prefix Slot Reuse, and Qwen3.6
# recommended sampling parameters.
# ==============================================================================

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/ai-models/gguf/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf}"
ALIAS="${ALIAS:-qwen/qwen3.6-35b-a3b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-131072}"     # 128K total context across parallel slots
PARALLEL="${PARALLEL:-2}"          # 2 concurrent slots (64K context per slot)

echo "==> Starting llama-server on ${HOST}:${PORT}"
echo "    Model:       ${MODEL_PATH}"
echo "    Alias:       ${ALIAS}"
echo "    Context:     ${CTX_SIZE} (${PARALLEL} parallel slots)"

exec llama-server \
  --model "${MODEL_PATH}" \
  --alias "${ALIAS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-gpu-layers 99 \
  --ctx-size "${CTX_SIZE}" \
  --parallel "${PARALLEL}" \
  --cont-batching \
  --flash-attn \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-reuse 256 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.1 \
  --repeat-penalty 1.05

