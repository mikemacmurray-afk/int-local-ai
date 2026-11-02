# `int-local-ai` — Local AI Inference Architecture & Playbook

**Hardware Target:** Apple Mac Mini M4 Pro (**64 GB Unified Memory** · 273 GB/s Memory Bandwidth · 14-Core CPU / 20-Core GPU)  
**Primary Model Class:** Sparse Mixture-of-Experts (MoE) — **`Qwen3.6-35B-A3B`** (**Full 8-Bit** or **6-Bit**, 35B total / 3B active parameters, 3:1 Gated DeltaNet + GQA hybrid attention)  
**Repository:** [mikemacmurray-afk/int-local-ai](https://github.com/mikemacmurray-afk/int-local-ai)

---

## 1. Executive Summary

This repository contains the architectural research, video transcript analyses, model & parameter tuning guide, prefill (TTFT) optimization playbook, integration blueprints, and macOS provisioning scripts for standing up a dedicated **Mac Mini M4 Pro (64 GB Unified Memory)** as a local AI inference appliance (`za-local`).

The appliance is purpose-built to provide zero-marginal-cost, POPIA-compliant, low-latency local LLM and embedding inference across three primary engineering workspaces:

1. **QDR (`D:\Work\Agents\iOCO-vodacom-qdr`)** — *Queryable Design Repository* (Oracle PL/SQL & APEX structural graph, hybrid BM25 + ChromaDB `bge-m3` retrieval, and the D15 Mediated Conversation Curator/Expert pipeline).
2. **SGC (`D:\Work\Agents\int-standards-guardrails-compliance`)** — *Standards, Guardrails & Compliance* (Multi-tenant enterprise AI compliance gateway, `forbid_egress` / `route_local` policy enforcement, POPIA/King V data-residency control, and P3 AI policy ingestion).
3. **DIDS & General Oracle/APEX Engineering (`D:\Work\Agents\iOCO-vodacom-dids-work`)** — Production incident triage (`ONTREADY`, `MSPI`, `FTTB`), SQL/PLSQL performance optimization, POPIA database encryption, and autonomous local coding loops (`Cline` + `Serena` LSP + `Playwright MCP`).

---

## 2. Repository Structure & Documentation Index

| Document / Script | Purpose |
|---|---|
| [`docs/01-youtube-video-transcripts-analysis.md`](file:///d:/Work/Agents/int-local-ai/docs/01-youtube-video-transcripts-analysis.md) | **Detailed analysis & step-by-step instruction summaries** of the 4 reference YouTube videos (`KVXaHQu32Rw`, `IRyiP_U7RuA`, `x_qmqqp6d7s`, `T0N4rPwVL4M`), including exact chapter breakdowns, commands, and architectural takeaways. |
| [`docs/02-mac-mini-48gb-hardware-and-runtime-architecture.md`](file:///d:/Work/Agents/int-local-ai/docs/02-mac-mini-48gb-hardware-and-runtime-architecture.md) | **Hardware & Runtime Architecture** for the Mac Mini M4 Pro: memory bandwidth math (273 GB/s), `iogpu.wired_limit_mb` kernel tuning, thermal management, and comparing `vllm-mlx`, `llama-server` (`llama.cpp`), `LM Studio`, and `Ollama`. |
| [`docs/03-moe-models-and-qwen36-35b-a3b-deep-dive.md`](file:///d:/Work/Agents/int-local-ai/docs/03-moe-models-and-qwen36-35b-a3b-deep-dive.md) | **Deep-Dive Model & Parameter Guide**: Why Sparse MoE + Hybrid Linear Attention (`Qwen3.6-35B-A3B`) transforms Apple Silicon unified memory; KV-cache math (~20 KB/token via 3:1 Gated DeltaNet); quantization comparison; and exact sampling & server flags per workload. |
| [`docs/04-project-integration-blueprint-qdr-sgc-dids.md`](file:///d:/Work/Agents/int-local-ai/docs/04-project-integration-blueprint-qdr-sgc-dids.md) | **Project Integration Blueprint**: Concrete code-level and `.env` configurations connecting the Mac Mini Pro to **QDR** (`config.py`, `llm_provider.py`, `model_manager.py`), **SGC** (`guardrails/upstream/local.py`, `route_local`), and **DIDS** (`Cline`, `Serena`, `Twinny`, `Playwright MCP`). |
| [`docs/05-macos-provisioning-and-operations-runbook.md`](file:///d:/Work/Agents/int-local-ai/docs/05-macos-provisioning-and-operations-runbook.md) | **Day-1 Provisioning & Operations Runbook**: End-to-end setup guide for the Mac Mini Pro from first boot to headless LAN service, `launchd` daemons, health checks, and benchmarking. |
| [`docs/06-64gb-upgrade-8bit-moe-and-prefill-optimization.md`](file:///d:/Work/Agents/int-local-ai/docs/06-64gb-upgrade-8bit-moe-and-prefill-optimization.md) | **64 GB Upgrade Deep-Dive**: Full **8-Bit vs. 6-Bit** analysis for `Qwen3.6-35B-A3B` (router stability, inference & PL/SQL coding benefits), **Catalog of Sparse/MoE Models for 64 GB** (`Gemma-4-26B-A4B`, `GLM-4.7-Flash`, `Qwen3-Next-80B-A3B`), and the **6-Layer Prefill (TTFT) Optimization Playbook**. |
| [`scripts/macos/configure_vram_limit.sh`](file:///d:/Work/Agents/int-local-ai/scripts/macos/configure_vram_limit.sh) | Script + `LaunchDaemon` generator to raise macOS Metal wired memory ceiling on 64 GB to **56 GB (`57344 MB`)** or **57 GB (`58368 MB`)**. |
| [`scripts/macos/start_vllm_mlx.sh`](file:///d:/Work/Agents/int-local-ai/scripts/macos/start_vllm_mlx.sh) | Production startup script for `vllm-mlx` serving `Qwen3.6-35B-A3B-8bit` with `--prefill-step-size 4096`, Radix Trie prefix caching, and NVMe SSD cache tiering. |
| [`scripts/macos/start_llama_server.sh`](file:///d:/Work/Agents/int-local-ai/scripts/macos/start_llama_server.sh) | Production startup script for `llama-server` (`llama.cpp`) serving `Qwen3.6-35B-A3B-Q8_0.gguf` with `-b 4096 -ub 2048 --flash-attn on --cache-reuse 256 --slot-save-path`. |
| [`configs/sampling-profiles.json`](file:///d:/Work/Agents/int-local-ai/configs/sampling-profiles.json) | Machine-readable sampling and context profiles for QDR Curator, QDR Expert, SGC Policy Evaluation, and DIDS PL/SQL Coding. |

---

## 3. High-Level Guidance at a Glance

### Why `Qwen3.6-35B-A3B` (Full 8-Bit / 6-Bit) on Mac Mini M4 Pro 64 GB is the "Golden Pairing"

1. **Solves the Memory Bandwidth Bottleneck (273 GB/s) Even at Full 8-Bit:**
   * On Apple Silicon, LLM token generation (decode) is strictly memory-bandwidth bound: $\text{TPS} \approx \frac{\text{Memory Bandwidth}}{\text{Active Weights Read per Token}}$.
   * **`Qwen3.6-35B-A3B`** holds **35B total parameters** in unified memory, but routes each token through only **9 of 256 experts (~3B active parameters)**.
   * With **64 GB Unified Memory** (`56 GB` Metal allocation), you can run `Qwen3.6-35B-A3B` at **Full 8-Bit (`~36.8 GB`)** — preserving 100% BF16 router stability and specialized PL/SQL tail-expert precision — while still reading only ~3.1 GB per token (**40–46 tok/s** decode) and leaving **~19 GB of free GPU memory**! Or run **6-Bit (`~28.2 GB`)** at **48–54 tok/s** and keep a second 25B–30B MoE model co-resident in RAM at the same time.
2. **Solves the Long-Context KV Cache Wall (3:1 Gated DeltaNet Hybrid):**
   * `Qwen3.6-35B-A3B` uses **30 Gated DeltaNet (linear attention / constant-memory state) layers** and only **10 full Grouped-Query Attention (GQA) layers** ($h_{kv}=2, d_{head}=256$).
   * Its KV cache footprint is only **~20 KB/token in FP16** (or **~10 KB/token in Q8_0**). Four parallel 64K-token slots (`256K` total context) consume only **~2.6 GB of RAM**!
3. **Eliminates the 80%+ Prefill Bottleneck (TTFT Optimization):**
   * By combining `-b 4096 -ub 2048 --flash-attn on` (or `vllm-mlx --prefill-step-size 4096`), Radix/Slot KV Prefix Caching (`--cache-reuse 256` / `--ssd-cache-dir`), and canonical prompt ordering (`cacheable_prefix` first), multi-turn QDR and coding prefill times drop from **~40+ seconds down to < 2 seconds**.

---

## 4. Recommended Model Roster for the 64 GB Mac Mini M4 Pro

| Tier / Role | Primary Model Choice | Quantization & Format | Memory Footprint (Weights) | Expected Speed (M4 Pro) | Target Workloads |
|---|---|---|---|---|---|
| **Tier 1A: Primary Workhorse (Max Fidelity)** | **`Qwen3.6-35B-A3B`** | **MLX 8-bit** or **GGUF `Q8_0`** | **~36.8 GB** (leaves ~19 GB free) | **40–46 tok/s** decode<br>950–1,350 tok/s prefill | Gold-standard single-model mode: QDR Expert/Curator, Oracle PL/SQL & APEX coding, SGC policy eval, Vision diagrams |
| **Tier 1B: Primary Workhorse (Dual-Model Co-Resident)** | **`Qwen3.6-35B-A3B`** | **MLX 6-bit** or **GGUF `UD-Q6_K_XL`** | **~28.2 GB** (leaves ~28 GB free) | **48–54 tok/s** decode<br>1,050–1,400 tok/s prefill | Near-lossless quality while leaving 28 GB free to co-host a second MoE specialist simultaneously |
| **Tier 2: Co-Resident Compliance & Diversity MoE** | **`Google Gemma-4-26B-A4B-IT`** or **`GLM-4.7-Flash` (30B-A3B)** | **MLX 6-bit / 8-bit** or **GGUF `Q6_K` / `Q8_0`** | **~19.8 – 26.5 GB** | **42–58 tok/s** decode | Fast QDR Curator co-resident partner, SGC compliance classification, cross-family architecture review |
| **Tier 3: Heavyweight 80B Sparse Frontier (Swap Mode)** | **`Qwen3-Next-80B-A3B`** | **MLX 4-bit** or **GGUF `UD-Q4_K_XL`** | **~45.5 GB** (fits in 56 GB pool!) | **35–45 tok/s** decode | 80B-parameter deep architectural synthesis & complex multi-system root-cause analysis running at 3B active speed |
| **Tier 4: Ultra-Fast Sidecar & Locked Embeddings** | **`BAAI/bge-m3`** + **`Qwen2.5-Coder-3B`** | **`F16` / `Q8_0`** | **~1.2 GB + ~2.8 GB** | Sub-50ms embed / **110+ tok/s** FIM | **Mandatory `bge-m3` for QDR** (1024-dim ChromaDB lock) + `Twinny` inline IDE autocomplete |

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
| `ctx_size` (`-c`) | `65536` (64K per slot) | `131072` (128K) | `32768` (32K) |

See [`docs/03-moe-models-and-qwen36-35b-a3b-deep-dive.md`](file:///d:/Work/Agents/int-local-ai/docs/03-moe-models-and-qwen36-35b-a3b-deep-dive.md) and [`docs/06-64gb-upgrade-8bit-moe-and-prefill-optimization.md`](file:///d:/Work/Agents/int-local-ai/docs/06-64gb-upgrade-8bit-moe-and-prefill-optimization.md) for the full mathematical and empirical breakdown.
