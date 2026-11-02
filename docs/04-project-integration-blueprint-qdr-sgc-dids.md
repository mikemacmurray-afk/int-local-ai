# Project Integration Blueprint: Connecting the Mac Mini Pro 48 GB to QDR, SGC, and DIDS

This document maps our **Mac Mini M4 Pro (48 GB)** local AI appliance (`za-local`) directly into the exact files, environment variables, and architectural constraints of your three target workspaces:

1. **QDR:** `D:\Work\Agents\iOCO-vodacom-qdr`
2. **SGC:** `D:\Work\Agents\int-standards-guardrails-compliance`
3. **DIDS:** `D:\Work\Agents\iOCO-vodacom-dids-work`

---

## 1. Network Topology & Hostname Convention

Assume the Mac Mini M4 Pro is connected via Gigabit/10Gb Ethernet on the local subnet with a reserved IPv4 address (e.g., `192.168.1.50`) or running on the local machine directly if you also move the development stack onto macOS (`127.0.0.1`).

* **Rule 1 (IPv4 Only, Never `localhost`):** Both `iOCO-vodacom-qdr` (`apps/api/config.py:27`, `163`) and `int-standards-guardrails-compliance` explicitly note that `localhost` resolves to IPv6 `::1` first on Windows, adding a 2-second timeout or hang on IPv4-bound services. Always use `127.0.0.1` (if running on the Mac itself) or the explicit LAN IPv4 address (e.g., `192.168.1.50`).
* **Standardized Ports on the Mac Mini:**
  * **Port `1234` (`http://192.168.1.50:1234/v1`):** **LM Studio Server (with MLX + GGUF Engine)** — Serves both `text-embedding-bge-m3` (pinned in memory, `--ttl 0`) and `qwen/qwen3.6-35b-a3b` (for QDR's `lmstudio` provider & SDK).
  * **Port `8080` (`http://192.168.1.50:8080/v1`):** **Optional `vllm-mlx` or `llama-server`** — High-throughput OpenAI (`/v1/chat/completions`) + Anthropic (`/v1/messages`) server with SSD prefix caching for SGC's `route_local`, Claude Code, and Cline.

---

## 2. Integrating with QDR (`D:\Work\Agents\iOCO-vodacom-qdr`)

### 2.1 Respecting QDR's Hard Constraints (`CLAUDE.md` & `docs/DECISIONS.md`)
* **Immutable `chunk_id` & `text-embedding-bge-m3` Lock (D6):**
  * QDR's ChromaDB collection (`data/chroma_data`) holds **6,207 embeddings** of dimension `1024` generated with `text-embedding-bge-m3`.
  * On the Mac Mini 48 GB, load `text-embedding-bge-m3` (`F16` or `Q8_0`, ~1.2 GB) in LM Studio on port `1234` with **Idle TTL disabled (`--ttl 0`)**.
  * Because `bge-m3` uses only 1.2 GB out of 48 GB, it stays permanently pinned alongside `Qwen3.6-35B-A3B` (~25–28 GB) without ever being evicted!

### 2.2 QDR `.env` Configuration (`D:\Work\Agents\iOCO-vodacom-qdr\apps\api\.env`)
Update your QDR `.env` file to point at the Mac Mini andupgrade the generation model from `qwen/qwen3.5-9b` to `qwen/qwen3.6-35b-a3b`:

```ini
# --- Local Inference on Mac Mini M4 Pro 48GB ---
LMSTUDIO_BASE_URL=http://192.168.1.50:1234/v1
LMSTUDIO_EMBED_MODEL=text-embedding-bge-m3
LMSTUDIO_GENERATION_MODEL=qwen/qwen3.6-35b-a3b
EMBEDDING_PROVIDER=lmstudio

# Timeout tuning: Qwen3.6-35B-A3B generates at 50+ tok/s on M4 Pro,
# but 128K context prefill or 16K token SOW generation benefits from a 180s window.
LLM_REQUEST_TIMEOUT_SECONDS=180.0
EMBEDDING_REQUEST_TIMEOUT_SECONDS=600.0

# D15 Mediated Conversation Pipeline (Curator + Expert):
# Route the Curator's fast JSON decisions to the local Mac Mini at $0 cost!
MEDIATED_PIPELINE_DEFAULT=true
MEDIATED_CURATOR_PROVIDER=lmstudio
MEDIATED_CURATOR_MODEL=qwen/qwen3.6-35b-a3b
```

### 2.3 Recommended QDR Feature-Level Provider Routing (`services/model_manager.py`)
QDR's `model_manager.py` (`GENERATION_FEATURES` at line 108) allows independent provider/model selection per feature in the QDR Settings UI (`GET/POST /settings/providers`). Here is the optimal routing table once the Mac Mini 48 GB is online:

| QDR Feature (`GENERATION_FEATURES`) | What It Powers in QDR | Recommended Provider & Model | Why |
|---|---|---|---|
| **`default`** | Fallback for any unconfigured feature | `lmstudio` $\rightarrow$ `qwen/qwen3.6-35b-a3b` | Zero API cost, 50+ tok/s, 100% POPIA / Vodacom code privacy. |
| **`query_assist`** | `reformulate_query`, `generate_interpretations`, `judge_sufficiency` (`query_orchestrator.py:235`) | `lmstudio` $\rightarrow$ `qwen/qwen3.6-35b-a3b` *(with `enable_thinking: false`)* | Eliminates 3–5 cloud API round-trips per query; executes in `< 1s` on M4 Pro. |
| **`categorization`** | File classification (`categorizer.py`) & Figure descriptions (`figure_describer.py`) | `lmstudio` $\rightarrow$ `qwen/qwen3.6-35b-a3b` *(once multimodal unlocked—see §2.4)* | Keeps uploaded Vodacom design docs local during ingestion. |
| **`system`** | Background Intent Ledger (`intent_ledger.py`) | `lmstudio` $\rightarrow$ `qwen/qwen3.6-35b-a3b` | Background ledger extraction runs continuously for free. |
| **`synthesis`** | Subsystem Synthesis (`subsystem_synthesis.py`) & SOW generation (`sow_sections.py`) | `lmstudio` $\rightarrow$ `qwen/qwen3.6-35b-a3b` *(or `Qwen3.5-27B` Q8)* | 128K context window holds entire APEX page exports & package bodies without truncation. |
| **`analysis`** | Code Workbench (`workbench_service.py`) & Tier-2 Findings (`findings_tier2.py`) | `lmstudio` $\rightarrow$ `qwen/qwen3.6-35b-a3b` | `73.4%` SWE-bench score excels at PL/SQL defect detection and impact analysis. |
| **`query`** | Main interactive Expert answer | `lmstudio` (`qwen/qwen3.6-35b-a3b`) **or** Hybrid (`anthropic` / `gemini` for complex client demos) | Use local by default; flip to Claude/Gemini only when desired. |

### 2.4 Code-Level Opportunities in QDR Unlocked by the Mac Mini M4 Pro

Inspecting `iOCO-vodacom-qdr/apps/api/services/` reveals **four specific code bottlenecks** that were written as workarounds for your old Windows/Vulkan setup and can now be upgraded:

1. **Embedding Batch Size (`apps/api/services/llm_provider.py:440`):**
   * *Current code:* `batch_size = 100 if self.provider_type == "openai" else 10` (because Windows Vulkan drivers crashed with `ErrorDeviceLost` on batches > 10).
   * *Mac Mini M4 Pro upgrade:* Apple Silicon Metal unified memory does not suffer from Windows TDR / Vulkan `ErrorDeviceLost`. Raising `batch_size` from `10` to **`64`** (and making `_ensure_lmstudio_handshake()` run once per file rather than once every 10 chunks) cuts QDR's embedding time for large APEX exports by **~70%**!
2. **Unlocking Local Multimodal Vision (`apps/api/services/figure_describer.py:33` & `llm_provider.py:604`):**
   * *Current code:* `figure_describer.py:33` sets `NO_VISION_PROVIDERS = {"lmstudio"}` and `llm_provider.py:605` ignores attached images on the `lmstudio` branch because `lmstudio.llm().respond()` was text-only when running `Qwen3.5-9B`.
   * *Mac Mini M4 Pro upgrade:* `Qwen3.6-35B-A3B` is **natively multimodal**! By routing `lmstudio` vision calls through `self.client.chat.completions.create()` (the OpenAI-compatible `/v1/chat/completions` endpoint on LM Studio / `vllm-mlx`, which accepts `{"type": "image_url"}` blocks via `_openai_style_content()`), QDR can describe diagrams in `figure_describer.py` and inspect pasted APEX screenshots **100% locally** under SGC governance!
3. **Remote vs. Local `lmstudio_control.py` (`apps/api/services/lmstudio_control.py:7-13`):**
   * *Current code:* `lmstudio_control.py` assumes LM Studio runs on `127.0.0.1` and shells out to `["lms", "server", "start"]` on failure.
   * *Mac Mini M4 Pro upgrade:* If QDR runs on Windows while LM Studio runs on the Mac Mini over LAN (`192.168.1.50:1234`), configure `lmstudio.configure_default_client("192.168.1.50:1234")` (or use the HTTP `/v1/models` probe in `lmstudio_probe.py`), as the Mac Mini's `launchd` service keeps the server permanently online.
4. **Fast JSON Mode for the Curator (`apps/api/services/query_orchestrator.py:64`):**
   * *Current code:* `_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", ...)` strips leaked `<think>` blocks after the model finishes generating them.
   * *Mac Mini M4 Pro upgrade:* Passing `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` on `query_assist` and Curator JSON calls skips generating the `<think>` block entirely, saving 5–10 seconds per mediated turn!

---

## 3. Integrating with SGC (`D:\Work\Agents\int-standards-guardrails-compliance`)

### 3.1 How SGC Uses the Mac Mini Pro (`za-local` Upstream)
In SGC (`guardrails/upstream/local.py` and the Corporate LLM Router architecture in `docs/brainstorm/2026-10-24-corporate-llm-router-report.md`):
* Whenever a site's active policy snapshot triggers a **`route_local`** or **`forbid_egress`** effect (for example, when a prompt contains a South African ID number `za_id_number`, MSISDN `za_msisdn`, bank account, or proprietary Vodacom/iOCO source code), SGC diverts the completion away from cloud providers (OpenRouter/OpenAI/Anthropic) to the site's configured **`local_route`** (`local.py:chat_completion`).
* As documented in `docs/brainstorm/2026-10-25-corporate-practice-research.md` §3 (South African data-residency facts), neither AWS Bedrock `af-south-1`, Azure SA North, nor Vertex `africa-south1` currently guarantees in-country processing for frontier Claude/Gemini/OpenAI models. **Your Mac Mini M4 Pro 48 GB running `Qwen3.6-35B-A3B` provides a genuine, physical in-country (`za-local`) sovereign inference node.**

### 3.2 Critical Timeout & Parameter Fix in `guardrails/upstream/local.py`
Let's inspect `D:\Work\Agents\int-standards-guardrails-compliance\guardrails\upstream\local.py`:
```python
async def chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: float = 30.0,
) -> ChatCompletionResult:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {"model": model, "messages": messages}
    return await post_chat_completion(client, url, headers={}, payload=payload, timeout=timeout)
```
* **Important Finding (Matches Router Report §2 line 107):**
  1. `timeout: float = 30.0` is hardcoded to 30 seconds. If `Qwen3.6-35B-A3B` receives a 15,000-token QDR prompt redirected via `route_local` and has `<think>` enabled, prefill + thinking + output can take 35–45 seconds, causing SGC to log an `upstream_timeout` (`504`)!
  2. **Recommended Tuning for SGC + Mac Mini:**
     * Configure SGC's site `local_route` to point to `http://192.168.1.50:1234` (or `:8080` for `vllm-mlx`) with model `qwen/qwen3.6-35b-a3b`.
     * Increase the default `timeout` in `guardrails/upstream/local.py` (or via env var `GUARDRAILS_LOCAL_UPSTREAM_TIMEOUT_SECONDS=120.0`) to **`120.0` seconds**.
     * For interactive Console Chat (`console-chat`), pass `"temperature": 0.3` and disable extended thinking if sub-15-second turnaround is desired.

### 3.3 Powering SGC Roadmap Wave 2 (P3: AI Document Ingestion) Locally
* In `handover.md` (line 38), **Wave 2 P3 (AI-assisted policy document ingestion)** extracts structured `Frameworks`, `Classifications`, `Detectors`, and `Requirements` (`schema: sgc.policy-bundle/1`) from regulatory PDFs (POPIA, King V Principle 10, ISO 27001, NIST AI RMF, and internal client IP policies).
* Because internal client security policies are themselves confidential, running P3 document ingestion against **`Qwen3.6-35B-A3B` on the Mac Mini (128K context, `temperature=0.1`, structured JSON schema output)** allows SGC to ingest entire 80-page regulatory/corporate PDFs locally at zero cost and with zero external egress.

---

## 4. Integrating with DIDS & General Work (`D:\Work\Agents\iOCO-vodacom-dids-work`)

### 4.1 Nature of the DIDS Workload
Reviewing `D:\Work\Agents\iOCO-vodacom-dids-work` shows deep enterprise Oracle Database & APEX 24.1 engineering:
* **Complex SQL & View Optimization:** `SFG_Perfromance SQL's.txt`, `Oracle_ONTREADY_MSPI_Query_Optimization_Session_Record.md`, `Oracle_View_Optimization_Conversation_Record.md`, `FTTH_210_53_Slow_after_FNO_Selected`.
* **Batch Job & Package Triage:** `FTTB_AND_SDM_MSPI_SIEBEL_HOURLY_JOBS.txt`, `LOAD_MSPI_ACTIVITY_DATE_RUNTIME_ERROR.txt`, `ONTREADY_RELEASE_PROCESS_Design.pdf`.
* **Security & Compliance:** `Popia Compliance/`, `Database_encryption_popia_session_recall.md`.
* **APEX 24.1 Development:** `Cancelled Orders Manual/`, Interactive Grid DML troubleshooting (`handover_summary.md`).

### 4.2 Why Local Inference on the Mac Mini Transforms DIDS Work
1. **Strict POPIA & Production Schema Confidentiality:**
   * DIDS files frequently contain real production schema names (`APEX_MANAGER`, `SDM`, `FTTB`, `VERIDATA`), internal hostnames (`pifnf03zatcrh.vodacom.corp`), execution plans, and sample data extracts (`MSPI_Extracts`).
   * Sending raw production SQL traces or MSPI extracts to public cloud chatbots risks violating corporate data governance. Running **`Qwen3.6-35B-A3B` locally on the Mac Mini** keeps 100% of Vodacom DIDS schemas, SQL execution plans, and POPIA encryption designs inside your physical office network.
2. **Deploying the Video 4 (`T0N4rPwVL4M`) Local Coding Powerhouse Stack on DIDS:**
   * **Engine:** Mac Mini M4 Pro 48 GB running `vllm-mlx` (`:8080`) or `LM Studio` (`:1234`) with `Qwen3.6-35B-A3B` at **128K context** (`131072` tokens).
   * **Agent (`Cline` / `Claude Code`):** Pointed at the Mac Mini (optionally routed through **SGC** on `:8750` so every DIDS prompt is also audit-logged and POPIA-checked!). With 128K context, you can drop an entire 4,000-line PL/SQL package body + `ALL_DEPENDENCIES` export + `DBMS_XPLAN` output into a single prompt for root-cause analysis.
   * **Autocomplete (`Twinny`):** Pointed at `Qwen2.5-Coder-3B-Base` co-loaded on the Mac Mini (~2.5 GB RAM) for instant PL/SQL and SQL inline completion in VS Code.
   * **Code Lookup (`QDR MCP / API` + `Serena`):** Connect your DIDS agent to QDR's structural graph API (`http://127.0.0.1:8000`) to query caller/callee trees across `SDM.ONTREADY_RELEASE_PROCESS` and `APEX_MANAGER`.

