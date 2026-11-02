# Deep-Dive Model & Parameter Guide: `Qwen3.6-35B-A3B` & Sparse MoE on 48 GB Unified Memory

This document provides the complete architectural analysis, quantization matrix, memory/KV-cache math, and exact runtime parameter settings for running **`Qwen3.6-35B-A3B`** (and its peer models) on your **Mac Mini M4 Pro (48 GB)**.

---

## 1. Architectural Anatomy of `Qwen3.6-35B-A3B`

Released under the permissive **Apache 2.0 license**, Alibaba Cloud's **`Qwen3.6-35B-A3B`** represents a major leap over earlier Qwen3-30B-A3B and Qwen2.5-Coder-32B models by combining **three architectural breakthroughs** inside a single unified checkpoint:

### 1.1 Fine-Grained Sparse Mixture-of-Experts (12:1 Sparsity Ratio)
* **Total Parameters:** **~35 Billion**
* **Active Parameters per Token:** **~3.0 Billion**
* **Expert Routing Topology:**
  * **256 Total Routed Experts** per MoE layer.
  * **8 Routed Experts + 1 Shared Expert = 9 Active Experts** fired per token.
  * Why this matters: Instead of coarse 8-expert layouts (like early Mixtral 8x7B), splitting knowledge across 256 fine-grained experts with 1 always-active shared expert prevents routing collapse, preserves deep domain knowledge (such as Oracle PL/SQL package syntax, `DBMS_SCHEDULER`, and APEX 24.1 APIs), and keeps the active memory read per token to just ~3B parameters.

### 1.2 3:1 Hybrid Attention: Gated DeltaNet + Grouped-Query Attention (GQA)
Across its **40 transformer layers** (hidden dimension $d = 2048$), `Qwen3.6-35B-A3B` does **not** use expensive quadratic full self-attention on every layer. Instead, it repeats a **4-layer block ten times** ($10 \times 4 = 40\text{ layers}$):
* **3 Layers of Gated DeltaNet (Linear Attention):**
  * Accounts for **75% of the network (30 of 40 layers)**.
  * Compresses historical sequence context into a **fixed-size recurrent state matrix** using delta-rule associative memory updates.
  * **Memory cost per additional token = $0\text{ bytes}$!** It does not grow with sequence length and computes in $O(1)$ per step during decode.
* **1 Layer of Gated Full Attention (GQA):**
  * Accounts for **25% of the network (10 of 40 layers)**.
  * Uses **2 KV heads ($h_{kv}=2$)** with head dimension **$d_{head}=256$**.
  * Provides exact needle-in-a-haystack token retrieval across the **262,144-token (256K)** native context window (extensible to 1M tokens via YaRN).

### 1.3 Native Multimodality, Thinking Preservation, & Multi-Token Prediction
* **Native Vision Encoder:** Unlike `Qwen3-30B-A3B` (which was text-only), `Qwen3.6-35B-A3B` natively processes **Text, Images (diagrams, UI screenshots, scanned PDFs), and Video** when loaded with its multimodal projector (`mmproj` in GGUF or `mlx-vlm` / `vllm-mlx`).
* **Preserved Thinking (`preserve_thinking` / `<think>`):** Supports controllable chain-of-thought (`enable_thinking: true | false`). When enabled in multi-turn agent loops, retaining prior reasoning traces across tool calls prevents the agent from losing track of its plan over 10+ steps.
* **Benchmark Dominance:**
  * **SWE-bench Verified:** `73.4%` (surpassing much larger dense open-weight models)
  * **Terminal-Bench 2.0:** `51.5%`
  * **GPQA Diamond:** `84.1%`

---

## 2. Exact KV Cache & Memory Math on a 48 GB Mac Mini Pro

Let us calculate the exact KV cache footprint per token for `Qwen3.6-35B-A3B`:

$$\text{KV Bytes / Token} = 2 \text{ (K and V)} \times L_{\text{full}} \times h_{kv} \times d_{\text{head}} \times \text{Bytes per Element}$$

Given $L_{\text{full}} = 10$, $h_{kv} = 2$, $d_{\text{head}} = 256$:

$$\text{Elements / Token} = 2 \times 10 \times 2 \times 256 = 10,240 \text{ elements/token}$$

| KV Cache Precision | Bytes / Element | KV Memory / Token | Memory for **32K Context** | Memory for **64K Context** | Memory for **128K Context** | Memory for **256K Context** |
|---|---|---|---|---|---|---|
| **`FP16` / `BF16`** (Unquantized) | 2.0 bytes | **20.48 KB** | **~0.64 GB** | **~1.28 GB** | **~2.56 GB** | **~5.12 GB** |
| **`Q8_0` / `FP8`** (Recommended) | ~1.05 bytes | **~10.75 KB** | **~0.34 GB** | **~0.67 GB** | **~1.34 GB** | **~2.68 GB** |
| **`Q4_0` / `KV4`** (Ultra-compact) | ~0.55 bytes | **~5.63 KB** | **~0.18 GB** | **~0.35 GB** | **~0.70 GB** | **~1.40 GB** |

*(Plus ~0.4 GB fixed overhead for the 30 Gated DeltaNet recurrent state matrices and compute buffers).*

> [!TIP]
> **Why This Is a Game-Changer for QDR & DIDS:**
> A standard dense 32B model (`Qwen2.5-Coder-32B`) uses 64 layers with 8 KV heads ($2 \times 64 \times 8 \times 128 \times 2 = 262\text{ KB/token}$), meaning a 128K context window consumes **~33.5 GB of RAM just for KV cache**—blowing past a 48 GB Mac Mini!  
> By contrast, **`Qwen3.6-35B-A3B` at 128K context consumes only ~1.34 GB (in `Q8_0` KV) or ~2.56 GB (in `FP16` KV)**!

---

## 3. Quantization Selection Matrix for 48 GB Unified Memory

Because the KV cache of `Qwen3.6-35B-A3B` is so light (~1.3–2.5 GB at 128K context), **you do NOT need to settle for aggressive 3-bit or low-end 4-bit quantization** on a 48 GB Mac Mini! You have the luxury of running **5-bit or 6-bit quantization**, which recovers ~99.5% of full BF16 perplexity and significantly sharpens strict JSON schema adherence and PL/SQL syntax accuracy.

| Quantization Variant | Format & Source | Weight Size on Disk/RAM | Total RAM Used at **128K Context** (`FP16` KV + `bge-m3` 1.2 GB) | Remaining Free RAM (of 48 GB) | Quality vs. Speed Verdict |
|---|---|---|---|---|---|
| **`mlx-community/Qwen3.6-35B-A3B-4bit`** | Native MLX 4-bit | ~19.8 GB | ~24.0 GB | ~24.0 GB | Fastest decode (**~58–65 tok/s**). Ideal if running multiple models simultaneously. |
| **`unsloth/Qwen3.6-35B-A3B-UD-Q4_K_XL`** | GGUF Unsloth Dynamic 4-bit | ~21.2 GB | ~25.4 GB | ~22.6 GB | Keeps router gates, shared expert, and attention layers in 6/8-bit while 4-bitting routed experts. |
| **`mlx-community/Qwen3.6-35B-A3B-6bit`** *(or 5-bit)* | **Native MLX 6-bit** | **~28.2 GB** | **~32.4 GB** | **~15.6 GB** | **⭐ Primary Recommendation (MLX):** Near-lossless quality, **~48–55 tok/s**, fits effortlessly inside our 43 GB Metal ceiling with 15+ GB to spare. |
| **`unsloth/Qwen3.6-35B-A3B-UD-Q5_K_XL`** *(or `UD-Q6_K_XL`)* | **GGUF Unsloth Dynamic 5/6-bit** | **~25.0 – 28.8 GB** | **~29.2 – 33.0 GB** | **~15.0 – 18.8 GB** | **⭐ Primary Recommendation (GGUF / LM Studio / `llama-server`):** Best balance of PL/SQL coding precision and speed (**~44–50 tok/s**). |
| **`mlx-community/Qwen3.6-35B-A3B-8bit`** / **`Q8_0`** | 8-bit MLX / GGUF | ~36.8 GB | ~41.0 GB | ~7.0 GB | Fits if `iogpu.wired_limit_mb=44032`, but leaves tight headroom for macOS apps and increases memory bandwidth load by ~30%. **6-bit / `UD-Q5_K_XL` is the sweet spot.** |

> [!IMPORTANT]
> **Critical Rule for MoE Quantization:** Never use naive uniform `Q4_0` on a Mixture-of-Experts model. The router gating layer (`ffn_gate_inp`) and shared expert are tiny but hyper-sensitive to quantization noise. Always use **Unsloth Dynamic (`UD-Q5_K_XL` / `UD-Q4_K_XL`)** for GGUF or **`mlx-community` 5-bit / 6-bit** for MLX.

---

## 4. Candidate Model Comparison: How Peers Stack Up on 48 GB

In addition to `Qwen3.6-35B-A3B`, here is how other top models in your codebase (`llm_provider.py` / `config.py`) and the 2026 open-weight ecosystem compare on the 48 GB Mac Mini Pro:

| Model | Architecture | Total / Active Params | Weight RAM (`Q5`/`6-bit`) | Speed on M4 Pro | Best Role in Your Stack |
|---|---|---|---|---|---|
| **`Qwen3.6-35B-A3B`** | Hybrid DeltaNet + GQA Sparse MoE (Multimodal) | 35B / **3B** | ~25–28 GB | **48–60 tok/s** | **#1 All-Rounder:** QDR Curator & Expert, SGC `route_local`, DIDS PL/SQL & APEX agent, Vision figure description. |
| **`Qwen3.5-27B`** | Hybrid Dense (Multimodal) | 27B / **27B** | ~23 GB (`Q6_K`) / ~28.5 GB (`Q8_0`) | **14–18 tok/s** | **High-Precision Dense Writer:** Great when writing long formal SOWs or executive briefings where you want every parameter active on every token and 15 tok/s is plenty fast. |
| **`Google Gemma-4-26B-A4B-IT`** | Sparse MoE (Multimodal) | 25.2B / **3.8B** | ~18–21 GB | **45–55 tok/s** | **Secondary Opinion / Critic:** Already in QDR's `_FLIP_STATIC_FALLBACK_MODELS`! Excellent for QDR's `mediated_challenge_enabled` counter-evidence / review step or SGC policy cross-checking. |
| **`GLM-4.7-Flash`** | Sparse MoE | 30B / **3B** | ~21 GB | **48–58 tok/s** | Strong multi-turn browser/web agent alternative, though Qwen3.6 leads on Oracle SQL/PLSQL and Python/TS. |
| **`Qwen2.5-Coder-3B-Base`** | Dense FIM Base | 3B / **3B** | ~2.5 GB (`Q8_0`) | **100+ tok/s** | Dedicated zero-latency inline autocomplete in VS Code (`Twinny`). |
| **`BAAI/bge-m3`** | Dense Bi-Encoder Embedding | 0.57B | ~1.2 GB (`F16`) | < 50ms | **Mandatory Locked Embedder** for QDR's 6,207 ChromaDB vectors (1024-dim). |

---

## 5. Deep-Dive Sampling & Runtime Parameter Guide for `Qwen3.6-35B-A3B`

Sparse MoE models with hybrid linear attention behave slightly differently from older dense Llama/Qwen2.5 models when sampling. Specifically:
1. **Temperature sensitivity:** If `temperature` is set to `0.0` (greedy) during long chain-of-thought (`<think>`) sequences, MoE models can occasionally enter repetitive reasoning loops. Conversely, if `temperature > 0.85` without `top_k` or `min_p`, low-probability tokens can trigger sub-optimal expert routing on subsequent steps.
2. **Thinking mode control (`enable_thinking`):**
   * For **fast JSON classification/routing calls** (such as QDR's `judge_sufficiency`, `reformulate_query`, or the Mediated Conversation **Curator** `frame`/`call` actions), **disable thinking (`enable_thinking: false`)** or cap thinking budget so the model outputs valid JSON in `< 1.0 second` instead of spending 500 tokens deliberating first!
   * For **complex synthesis, bug root-cause analysis, and PL/SQL refactoring** (QDR **Expert**, Code Workbench, DIDS `ONTREADY` analysis), **enable thinking (`enable_thinking: true`)** and set `preserve_thinking: true` across multi-turn tool loops.

### Workload-Specific Parameter Profiles

#### Profile A: QDR Mediated **Curator** & Fast Orchestrator Side-Calls (`query_assist`)
* **Used by:** `query_orchestrator.py` (`judge_sufficiency`, `reformulate_query`, `generate_interpretations`) and D15 Mediator Curator JSON actions (`frame`, `plan`, `call`, `brief`, `present`).
* **Goal:** Instant, deterministic, schema-valid JSON output with zero wasted `<think>` latency.
* **Settings:**
  * `temperature`: **`0.2`**
  * `top_p`: **`0.90`**
  * `top_k`: **`20`**
  * `min_p`: **`0.05`**
  * `presence_penalty`: **`0.0`**
  * `repetition_penalty`: **`1.0`**
  * `max_tokens`: **`2048`**
  * `chat_template_kwargs`: **`{"enable_thinking": false}`** *(or prepend `/no_think` in system prompt if using an older template)*

#### Profile B: QDR Mediated **Expert**, Subsystem Synthesis & DIDS Oracle PL/SQL Coding
* **Used by:** QDR's Expert `DRAFT` / `CHALLENGE` stages, `subsystem_synthesis.py`, `sow_sections.py`, Code Workbench (`workbench_service.py`), and `Cline` / `Claude Code` on DIDS.
* **Goal:** Deep multi-hop reasoning over Oracle DDL/PLSQL packages, exact citation grounding (`E1`..`E#`), and zero truncation.
* **Settings:**
  * `temperature`: **`0.6`** *(Official Qwen3.6 recommended sweet spot for thinking + coding)*
  * `top_p`: **`0.95`**
  * `top_k`: **`20`**
  * `min_p`: **`0.0`**
  * `presence_penalty`: **`0.1`** *(Prevents repetitive boilerplate in long PL/SQL package bodies or 10K-token SOW drafts)*
  * `repetition_penalty`: **`1.05`**
  * `max_tokens`: **`16384`** *(Matches QDR's `LlmPredictionConfig(max_tokens=16000)` in `llm_provider.py:760`)*
  * `chat_template_kwargs`: **`{"enable_thinking": true, "preserve_thinking": true}`**

#### Profile C: SGC Local Route (`route_local` / `forbid_egress`) & P3 Policy Document Ingestion
* **Used by:** `int-standards-guardrails-compliance` (`guardrails/upstream/local.py`) when sensitive/regulated POPIA or proprietary code traffic is routed locally, and for upcoming Roadmap Wave 2 **P3 AI Document Ingestion**.
* **Goal:** Faithful compliance instruction following, zero hallucinated policy clauses, and responsive gateway latency (note: increase SGC's `local.py` timeout from `30.0s` to `120.0s`—see `docs/04-project-integration-blueprint-qdr-sgc-dids.md`).
* **Settings:**
  * `temperature`: **`0.3`** *(or `0.1` for structured YAML/JSON policy extraction)*
  * `top_p`: **`0.90`**
  * `top_k`: **`20`**
  * `min_p`: **`0.05`**
  * `presence_penalty`: **`0.0`**
  * `max_tokens`: **`8192`**
  * `chat_template_kwargs`: **`{"enable_thinking": false}`** for interactive gateway chat (`< 30s` SLA); **`{"enable_thinking": true}`** for offline P3 policy pack extraction.

#### Profile D: QDR Multimodal Figure & Diagram Describer (`figure_describer.py`)
* **Used by:** `services/figure_describer.py` (describing architecture diagrams, scanned PDF pages, and APEX screenshots with exact `SCHEMA.OBJECT` label transcription).
* **Goal:** High-precision OCR and visual relationship extraction without inventing labels.
* **Settings:**
  * `temperature`: **`0.2`**
  * `top_p`: **`0.90`**
  * `top_k`: **`20`**
  * `max_tokens`: **`1024`** *(Capped by `MAX_DESCRIPTION_CHARS = 4000` / 300 words in `figure_describer.py:20-30`)*
  * `chat_template_kwargs`: **`{"enable_thinking": false}`**

---

## 6. Engine-Level Server Flags (`llama-server`, `LM Studio`, `vllm-mlx`)

When launching the server on the Mac Mini M4 Pro 48 GB, use these exact engine flags:

### 6.1 `llama-server` / LM Studio GGUF Engine Flags
| Flag / Setting | Value | Why It Matters on M4 Pro 48 GB |
|---|---|---|
| `--model` / `-m` | `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` | Best Unsloth dynamic quant for 48 GB unified memory (~25 GB). |
| `--mmproj` | `mmproj-Qwen3.6-35B-A3B-f16.gguf` | Enables native vision input for QDR `figure_describer.py` and pasted APEX screenshots. |
| `--n-gpu-layers` / `-ngl` | `99` (All layers) | Forces 100% of layers and all 256 experts into Metal Unified Memory. |
| `--ctx-size` / `-c` | `131072` (128K) or `65536` (64K) | Per-slot context capacity. At `Q8_0` KV cache, 128K uses only ~1.34 GB! |
| `--parallel` / `-np` | `2` | Allows 2 concurrent requests (e.g. QDR background worker + interactive query) without queueing stalls. |
| `--cont-batching` / `-cb` | Enabled (Default) | Continuous batching across parallel slots. |
| `--flash-attn` / `-fa` | `on` (`true`) | Mandatory for fast Metal prefill and KV cache quantization. |
| `--cache-type-k` / `-ctk` | `q8_0` | Halves Key cache memory with zero measurable perplexity degradation. |
| `--cache-type-v` / `-ctv` | `q8_0` | Halves Value cache memory (requires `--flash-attn`). |
| `--cache-reuse` | `256` | Enables automatic prefix-cache reuse when prompts share a 256+ token prefix (huge win for QDR Curator & Cline). |
| `--no-mmap` | Omit (Keep `mmap` **ON**) | On Apple Silicon Unified Memory, `mmap` allows zero-copy page sharing between the OS file cache and Metal GPU. |

### 6.2 `vllm-mlx` Native Apple Silicon Server Flags
```bash
vllm-mlx serve mlx-community/Qwen3.6-35B-A3B-6bit \
  --host 0.0.0.0 \
  --port 8080 \
  --max-num-seqs 4 \
  --max-model-len 131072 \
  --ssd-cache-dir ~/.vllm-mlx/prefix-cache \
  --warm-prompts
```
* **Why `--ssd-cache-dir` is huge:** In multi-turn QDR conversations or Cline coding sessions across `iOCO-vodacom-dids-work`, `vllm-mlx` hashes prompt blocks into a prefix trie and persists evicted KV blocks to the Mac Mini's internal NVMe SSD (which reads at ~5,000–6,000 MB/s), making multi-turn context switching nearly instantaneous.

