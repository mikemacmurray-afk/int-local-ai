# `int-local-ai` — Local AI Inference Architecture & Playbook

**Hardware Target:** Apple Mac Mini M4 Pro (48 GB Unified Memory · 273 GB/s Memory Bandwidth)  
**Primary Model Class:** Sparse Mixture-of-Experts (MoE) — **`Qwen3.6-35B-A3B`** (35B total / 3B active parameters, 3:1 Gated DeltaNet + GQA hybrid attention)  
**Repository:** [mikemacmurray-afk/int-local-ai](https://github.com/mikemacmurray-afk/int-local-ai)

---

## 1. Executive Summary

This repository contains the architectural research, video transcript analyses, model & parameter tuning guide, integration blueprints, and macOS provisioning scripts for standing up a dedicated **Mac Mini Pro (48 GB Unified Memory)** as a local AI inference appliance (`za-local`).

The appliance is purpose-built to provide zero-marginal-cost, POPIA-compliant, low-latency local LLM and embedding inference across three primary engineering workspaces:

1. **QDR (`D:\Work\Agents\iOCO-vodacom-qdr`)** — *Queryable Design Repository* (Oracle PL/SQL & APEX structural graph, hybrid BM25 + ChromaDB `bge-m3` retrieval, and the D15 Mediated Conversation Curator/Expert pipeline).
2. **SGC (`D:\Work\Agents\int-standards-guardrails-compliance`)** — *Standards, Guardrails & Compliance* (Multi-tenant enterprise AI compliance gateway, `forbid_egress` / `route_local` policy enforcement, POPIA/King V data-residency control, and P3 AI policy ingestion).
3. **DIDS & General Oracle/APEX Engineering (`D:\Work\Agents\iOCO-vodacom-dids-work`)** — Production incident triage (`ONTREADY`, `MSPI`, `FTTB`), SQL/PLSQL performance optimization, POPIA database encryption, and autonomous local coding loops (`Cline` + `Serena` LSP + `Playwright MCP`).

---

## 2. Repository Structure & Documentation Index

| Document / Script | Purpose |
|---|---|
| [`docs/01-youtube-video-transcripts-analysis.md`](file:///d:/Work/Agents/int-local-ai/docs/01-youtube-video-transcripts-analysis.md) | **Detailed analysis & step-by-step instruction summaries** of the 4 reference YouTube videos (`KVXaHQu32Rw`, `IRyiP_U7RuA`, `x_qmqqp6d7s`, `T0N4rPwVL4M`), including exact chapter breakdowns, commands, and architectural takeaways. |
| [`docs/02-mac-mini-48gb-hardware-and-runtime-architecture.md`](file:///d:/Work/Agents/int-local-ai/docs/02-mac-mini-48gb-hardware-and-runtime-architecture.md) | **Hardware & Runtime Architecture** for the Mac Mini M4 Pro 48 GB: memory bandwidth math (273 GB/s), `iogpu.wired_limit_mb` kernel tuning (unlocking 42–44 GB for Metal/MLX), thermal management, and comparing `vllm-mlx`, `llama-server` (`llama.cpp`), `LM Studio`, and `Ollama`. |
| [`docs/03-moe-models-and-qwen36-35b-a3b-deep-dive.md`](file:///d:/Work/Agents/int-local-ai/docs/03-moe-models-and-qwen36-35b-a3b-deep-dive.md) | **Deep-Dive Model & Parameter Guide**: Why Sparse MoE + Hybrid Linear Attention (`Qwen3.6-35B-A3B`) transforms 48 GB unified memory; KV-cache math (~20 KB/token via 3:1 Gated DeltaNet); quantization comparison (`MLX 4-bit/6-bit` vs `Unsloth UD-Q4_K_XL/UD-Q6_K_XL`); and exact sampling & server flags per workload. |
| [`docs/04-project-integration-blueprint-qdr-sgc-dids.md`](file:///d:/Work/Agents/int-local-ai/docs/04-project-integration-blueprint-qdr-sgc-dids.md) | **Project Integration Blueprint**: Concrete code-level and `.env` configurations connecting the Mac Mini Pro 48 GB to **QDR** (`config.py`, `llm_provider.py`, `model_manager.py`), **SGC** (`guardrails/upstream/local.py`, `route_local`), and **DIDS** (`Cline`, `Serena`, `Twinny`, `Playwright MCP`). |
| [`docs/05-macos-provisioning-and-operations-runbook.md`](file:///d:/Work/Agents/int-local-ai/docs/05-macos-provisioning-and-operations-runbook.md) | **Day-1 Provisioning & Operations Runbook**: End-to-end setup guide for the Mac Mini Pro from first boot to headless LAN service, `launchd` daemons, health checks, and benchmarking. |
| [`scripts/macos/configure_vram_limit.sh`](file:///d:/Work/Agents/int-local-ai/scripts/macos/configure_vram_limit.sh) | Script + `LaunchDaemon` generator to raise macOS Metal wired memory ceiling from ~36 GB to **42 GB (`43008 MB`)** or **43 GB (`44032 MB`)**. |
| [`scripts/macos/start_vllm_mlx.sh`](file:///d:/Work/Agents/int-local-ai/scripts/macos/start_vllm_mlx.sh) | Production startup script for `vllm-mlx` serving `Qwen3.6-35B-A3B` with prefix caching, SSD cache tiering, tool calling, and vision enabled. |
| [`scripts/macos/start_llama_server.sh`](file:///d:/Work/Agents/int-local-ai/scripts/macos/start_llama_server.sh) | Production startup script for `llama-server` (`llama.cpp`) serving `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` with Flash Attention (`-fa`), quantized KV cache, and continuous batching. |
| [`configs/sampling-profiles.json`](file:///d:/Work/Agents/int-local-ai/configs/sampling-profiles.json) | Machine-readable sampling and context profiles for QDR Curator, QDR Expert, SGC Policy Evaluation, and DIDS PL/SQL Coding. |

---

## 3. High-Level Guidance at a Glance

### Why `Qwen3.6-35B-A3B` on Mac Mini M4 Pro 48 GB is the "Golden Pairing"

1. **Solves the Memory Bandwidth Bottleneck (273 GB/s):**
   * On Apple Silicon, LLM token generation (decode) is strictly memory-bandwidth bound: $\text{TPS} \approx \frac{\text{Memory Bandwidth}}{\text{Active Weights Read per Token}}$.
   * A **dense** 32B–35B model in 4-bit (~20 GB) reads ~20 GB per token, capping an M4 Pro (273 GB/s) at **~11–13 tok/s**.
   * **`Qwen3.6-35B-A3B`** holds **35B total parameters** in unified memory (fitting easily in 48 GB at 4-bit, 5-bit, or 6-bit quantization), but routes each token through only **9 of 256 experts (~3B active parameters)**. The M4 Pro only reads ~1.8–2.5 GB per token, delivering **45–65 tok/s decode** with 35B-class frontier reasoning and coding intelligence (`73.4%` SWE-bench Verified).
2. **Solves the Long-Context KV Cache Wall (3:1 Gated DeltaNet Hybrid):**
   * Traditional transformers require $O(N)$ KV cache across all layers (often 150–250 KB/token, eating 10–16 GB for 64K context).
   * `Qwen3.6-35B-A3B` uses **30 Gated DeltaNet (linear attention / constant-memory state) layers** and only **10 full Grouped-Query Attention (GQA) layers** ($h_{kv}=2, d_{head}=256$).
   * Its KV cache footprint is only **~20 KB/token in FP16** (or **~10 KB/token in Q8_0**). A **128K token context window consumes only ~2.5 GB of RAM** (or ~1.3 GB in Q8_0)!
3. **Replaces Cloud Dependencies Across All Three Projects:**
   * **QDR:** Runs `text-embedding-bge-m3` (1024-dim, preserving the immutable 6,207 `chunk_id` embeddings) alongside `Qwen3.6-35B-A3B` simultaneously in 48 GB RAM with zero model-unloading thrash. Powers both the fast JSON **Curator** (`mediated_curator_model`) and the deep PL/SQL **Expert**.
   * **SGC:** Acts as the high-speed `za-local` upstream for `route_local` and `forbid_egress` verdicts, keeping regulated Vodacom/iOCO data, POPIA PII, and proprietary PL/SQL code strictly on-premises.

---

## 4. Recommended Model Roster for the 48 GB Mac Mini

| Tier / Role | Primary Model Choice | Quantization & Format | Memory Footprint (Weights) | Expected Speed (M4 Pro) | Target Workloads |
|---|---|---|---|---|---|
| **Tier 1: Primary Workhorse (Always Loaded)** | **`Qwen3.6-35B-A3B`** | **MLX 5-bit / 6-bit** (`mlx-community`) or **GGUF `UD-Q5_K_XL`** (`unsloth`) | ~21 GB (4-bit) / ~25 GB (5-bit) / ~28 GB (6-bit) | **45–65 tok/s** decode<br>350–650 tok/s prefill | QDR Curator & Expert, SGC `route_local` upstream, DIDS Cline/PLSQL refactoring, Vision diagram parsing |
| **Tier 2: Dense High-Precision Specialist (Optional Swap)** | **`Qwen3.5-27B`** (Dense) | **GGUF `Q6_K` or `Q8_0`** / **MLX 6-bit** | ~22 GB (`Q6_K`) / ~28.5 GB (`Q8_0`) | **14–19 tok/s** decode | Deep single-pass SOW synthesis, formal compliance document drafting where dense attention nuance is preferred |
| **Tier 3: Ultra-Fast Sidecar / Autocomplete** | **`Qwen2.5-Coder-3B`** or **`Qwen3-4B`** | **MLX 4-bit / 8-bit** or **GGUF `Q8_0`** | ~2.5–4.0 GB | **95–130+ tok/s** decode | IDE inline autocomplete (`Twinny`), ultra-fast SGC regex/classification pre-checks |
| **Tier 4: Locked Embedding Engine (Always Loaded)** | **`BAAI/bge-m3`** (`text-embedding-bge-m3`) | **`F16` / `Q8_0` GGUF** | ~1.2 GB | Sub-50ms query embed | **Mandatory for QDR** (matches existing 1024-dim ChromaDB collection of 6,207 chunks) |

---

## 5. Quick-Start Parameter Cheat Sheet (`Qwen3.6-35B-A3B`)

| Parameter | Coding / PL/SQL / JSON Curator | Deep Reasoning / Architecture / SOW | SGC Policy / Deterministic Eval |
|---|---|---|---|
| `temperature` | `0.2` – `0.3` (Curator JSON) / `0.6` (Coding) | `0.6` | `0.1` |
| `top_p` | `0.95` | `0.95` | `0.90` |
| `top_k` | `20` | `20` – `40` | `10` |
| `min_p` | `0.0` (or `0.05` if `top_k` disabled) | `0.0` | `0.05` |
| `presence_penalty` | `0.0` – `0.1` | `0.1` – `0.2` | `0.0` |
| `repetition_penalty` | `1.0` – `1.05` | `1.05` | `1.0` |
| `enable_thinking` | `false` for Curator JSON / `true` for Expert | `true` (`preserve_thinking: true`) | `false` (for low-latency gate) |
| `ctx_size` (`-c`) | `65536` (64K) | `131072` (128K) | `32768` (32K) |

See [`docs/03-moe-models-and-qwen36-35b-a3b-deep-dive.md`](file:///d:/Work/Agents/int-local-ai/docs/03-moe-models-and-qwen36-35b-a3b-deep-dive.md) for the full mathematical and empirical breakdown.

