#!/usr/bin/env bash
# ==============================================================================
# start_llama_server.sh
# Starts llama.cpp's llama-server on Mac Mini M4 Pro (64 GB) with:
# - Full 8-bit (Q8_0) or 6-bit (UD-Q6_K_XL) Qwen3.6-35B-A3B weights
# - Metal Prefill Kernel Tuning (--batch-size 4096 --ubatch-size 2048 --flash-attn on)
# - Prefix Cache Reuse (--cache-reuse 256) & Persistent NVMe Slot Cache (--slot-save-path)
# - 4 Dedicated Parallel Slots (Curator, Expert, SGC, DIDS Coding)
# ==============================================================================

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/ai-models/gguf/Qwen3.6-35B-A3B-Q8_0.gguf}"
ALIAS="${ALIAS:-qwen/qwen3.6-35b-a3b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-262144}"       # 256K total context across 4 parallel slots (64K per slot)
PARALLEL="${PARALLEL:-4}"            # 4 dedicated slots (0: Curator, 1: Expert, 2: SGC, 3: Coding)
BATCH_SIZE="${BATCH_SIZE:-4096}"     # Logical prefill batch size
UBATCH_SIZE="${UBATCH_SIZE:-2048}"   # Physical Metal GPU micro-batch size (saturates 20 GPU cores)
SLOT_SAVE_PATH="${SLOT_SAVE_PATH:-$HOME/.cache/llama-slots}"

mkdir -p "${SLOT_SAVE_PATH}"

echo "==> Starting llama-server on ${HOST}:${PORT} (64 GB Mac Mini M4 Pro Profile)"
echo "    Model:          ${MODEL_PATH}"
echo "    Alias:          ${ALIAS}"
echo "    Context:        ${CTX_SIZE} (${PARALLEL} parallel slots)"
echo "    Prefill Batch:  -b ${BATCH_SIZE} -ub ${UBATCH_SIZE} --flash-attn on --cache-reuse 256"
echo "    Slot Cache Dir: ${SLOT_SAVE_PATH}"

exec llama-server \
  --model "${MODEL_PATH}" \
  --alias "${ALIAS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --n-gpu-layers 99 \
  --ctx-size "${CTX_SIZE}" \
  --parallel "${PARALLEL}" \
  --cont-batching \
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}" \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-reuse 256 \
  --slot-save-path "${SLOT_SAVE_PATH}" \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.1 \
  --repeat-penalty 1.05
