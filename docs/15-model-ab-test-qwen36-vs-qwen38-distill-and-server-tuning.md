# 15 — Model A/B Test: Qwen3.6-35B-A3B vs Qwen3.8-35B-A3B-Distill, and LM Studio Server Tuning

**Date:** 2026-11-08 · **Server:** Mac mini M5 Pro, 64 GB, macOS 27.2, LM Studio (mlx-llm runtime 1.14.0) at `http://192.168.3.175:1234` · **Status:** tests complete; no production change made (server restored to its original configuration).

---

## 1. Summary

| Question | Answer |
|---|---|
| Is there a newer model with the same structure as Qwen3.6-35B-A3B? | **No official one.** Alibaba's Qwen3.8 open weights are a 2.4T-A95B flagship and a **27B dense** model. The only same-architecture successor is **Qwen3.8-35B-A3B-Distill**, a community fine-tune of Qwen3.6-35B-A3B on Qwen3.8 teacher traces (empero-ai, Apache-2.0). |
| Is the distill more accurate? | **Equal.** 168/190 vs 167/190 on 190 automatically graded calls covering SGC, QDR, chat, reasoning, coding, instruction following, extraction and tool calling. |
| Is it faster? | **Yes, for whole jobs: 648 s vs 1,734 s (2.7×)** across the same quality suites, because it writes ~3× fewer tokens (33,194 vs 98,570). Per-token speed is identical (~65 tok/s decode, same prompt reading). With thinking on it uses **5.1× fewer thinking tokens** and was more accurate (26/26 vs 23/26). |
| Any regressions? | **One real one:** with thinking off it never emits **parallel tool calls** (0/6 vs 6/6). With thinking on it does. One minor QDR formatting miss (claim labels per paragraph instead of per sentence, 1 of 2 runs). |
| Does its MTP head speed things up in LM Studio? | **No.** LM Studio's MLX runtime ignores `--speculative-draft-mtp` for this architecture (decode 65.5–66.4 tok/s with it on vs 64.5–65.1 off). |
| Context and parallelism? | 131,072 context is cheap in memory (~20 KB/token KV) and both models recall correctly at 107K tokens, but a 107K prompt takes ~100 s to read. 8 parallel slots raise total throughput 21% over 4 (156 vs 129 tok/s) at 20 tok/s per stream. |
| Biggest operational finding | **A long prompt blocks every other request until it has been read.** A 30K-token prompt held an SGC meaning check for ~17 s (60K → ~45 s). |

**Recommendation:** the distill is a credible default candidate: same accuracy, much shorter jobs, especially with thinking on. Before switching, (1) run one real QDR multi-turn session and SGC's own suite against it, and (2) decide how tool-calling agents run (thinking on, or accept sequential tool calls). Separately, fix the head-of-line blocking for SGC and consider raising the context to 131,072. Details and the full option list are in §9.

---

## 2. Background: what prompted the test

- The current local model, `qwen/qwen3.6-35b-a3b` (lmstudio-community MLX 8-bit), dates from April 2026.
- The video *"Everyone Says Local AI Is Getting Smarter. We Tested It on Our Mac"* (<https://youtu.be/98ljuFRK2_Y>) compared an April and a November model on the same Mac. **Note:** it compared the *dense* Qwen3.6-27B against the *dense* Qwen3.8-27B (both Q4 on Ollama), not the 35B-A3B MoE. Its results:
  - blind-graded average 7.54 → 8.19 /10; completed 17/18 → 18/18 runs;
  - the newer model won scheduling, the safe recovery guide and recovering decisions from a contradictory meeting;
  - it **lost** the decision email (invented a saving), data extraction (wrong field types) and the evidence audit (added unsupported assumptions);
  - time to first word (1.42 vs 1.55 s) and writing speed (38.6 vs 38.0 tok/s) were the same, yet the median job took **65 s instead of 116 s** because the newer model wrote about half as many tokens.
- This test checks whether the same pattern (better results mainly through shorter output, with risk on grounded and structured tasks) holds for our MoE model and our workloads.

---

## 3. Candidate survey (web research, 2026-11-08)

| Model | Structure | Status | Verdict |
|---|---|---|---|
| Qwen3.6-35B-A3B (current) | MoE, 35B total / ~3B active, 256 experts (8 routed + 1 shared), hybrid Gated DeltaNet + full attention every 4th layer, MTP head | Official, Apr 2026 | Baseline |
| **Qwen3.8-35B-A3B-Distill** (empero-ai; MLX build NovaeonStudio `oQ8-fp16-mtp`) | **Identical architecture** (`qwen3_5_moe`, 40 layers, 256/8 experts, MTP head) | Community SFT, Sep 2026 | **Tested** |
| Qwen3.8-27B | Dense 27B | Official, Aug 2026 | Not like-for-like: reads all 27B weights per token, ~4–6× slower decode on this Mac (others measured 11–20 tok/s on similar machines) |
| Qwen3.7 Max / Plus | — | API only | Not runnable locally |
| Gemma-4-26B-A4B, Nemotron 3.5 Lightning | Different MoE families | Official | Out of scope (different structure); third-party tests put both slower than Qwen3.6-35B-A3B on Macs |

What other testers found for the distill:
- **Model card:** MMLU 0.838 → 0.834 (flat); ARC-Challenge +3.4 to +4.4 points. Trained on 8,192-token examples, so the card warns of shorter outputs and possible long-context degradation.
- **InsiderLLM, RTX 3090, 47-item routing task:** accuracy 22/47 vs 23/47 (a tie); identical speed; a fifth of the reasoning tokens; but **6/47 answers came back as chat instead of the requested JSON**.
- **NovaeonStudio (M5 Max, oMLX engine):** IFEval average 84.86; ~120 tok/s decode with MTP; 9/9 and 13/13 on its agent suites.

---

## 4. Review of the previous day's server traffic

LM Studio server logs, 2026-11-07 15:42 to 2026-11-08 05:00: **610 requests**, all to `qwen/qwen3.6-35b-a3b`. Decode was a steady 61–66 tok/s for every caller.

| Caller | Calls | Latency (median / p90) | Findings |
|---|---|---|---|
| QDR curator (`multiturn-v2` worktree) | 224 + 8 embeddings | 5 s / 10 s | Prompts 7–11K tokens; 49% of prompt tokens served from the prefix cache; **20/224 (9%) cut off at `max_tokens` 600** |
| SGC meaning checks | 128 | < 1 s | **26 empty answers**, all from calls that did not send `reasoning_effort: "none"` (18 short probes plus 8 from a deliberate thinking-on run). All 90 calls with `"none"` answered. |
| SGC bench scripts (`think_bench*.py`) | 171 | 2 s / 31 s | 5 context-overflow errors during the 64K tests; 12 log-counting answers cut off (known) |
| Manual tests / health checks | 79 | — | — |

Also seen: LM Studio's prompt-cache disk store was full (16,999 of 17,000 MiB, 13.9 GB already evicted).

---

## 5. Test method

### 5.1 Setup

- **Same server, same settings for both models:** 65,536 context, 4 parallel slots, MLX 8-bit weights. Each model was loaded on its own; they don't fit in memory together.
- **Harness:** `scripts/model_bench/bench.py`, which calls the server from the Windows laptop over the LAN.
  - It reuses SGC's graded tasks unchanged (`int-standards-guardrails-compliance/scripts/local_ai_bench/think_bench.py` and `think_bench_hard.py`), so results compare directly with SGC spec §5.6.
  - The other suites are new.
- **Recording:** every call writes one row of JSON lines. Timings come from LM Studio's `/api/v0` `stats`: time to first token and server-side tokens/s.
- **Grading is automatic:**
  - fixed answer keys;
  - SQL run against a real SQLite table;
  - code run against unit tests in a separate isolated Python process (10 s limit; anything importing os/sys/subprocess or calling open/eval/exec is refused);
  - JSON parsed and field-checked;
  - tool calls compared by name and arguments.
  - All test cases and checkers were validated against reference solutions before use.
- **Sampling (Qwen's presets, as SGC uses them):**
  - thinking off: `reasoning_effort: "none"`, temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5;
  - thinking on: temperature 0.6, top_p 0.95, top_k 20.
  - Seeds 42 and 7 where a suite runs twice. LM Studio is not fully deterministic even with a seed, so totals matter, not single runs.

### 5.2 Suites

| Suite | What it covers | Calls per model |
|---|---|---|
| `sgc` | SGC meaning check: 17 labelled texts (codename, Afrikaans, buried line, implied deal, injection attempt, public deals, false friends), thinking off, `max_tokens` 8, ×2 | 34 |
| `chat` | SGC round 1: 11 tasks (fact, JSON extraction, summary, trick question, 2 logic puzzles, weekday count, crew schedule, SQL, code fix, counting in a 6K-token log), in three settings: off ×2 seeds, off + "work it out briefly, then `ANSWER:`" ×2, thinking on with an 8K budget ×1 | 55 |
| `hard` | SGC round 2: 6 harder tasks (counting in a 20K-token log, SQL window function, scheduling puzzle, subtle median bug, invoice with VAT, 21-business-day notice period), same three settings (the log count is skipped with thinking on) | 29 |
| `qdr` | Replica of the QDR curator, using QDR's own preamble, digest/next instructions, JSON action schemas and claim-labelling rule, over **20.5K tokens of `[E#] <evidence>` blocks** (22 generated PL/SQL packages plus 4 planted objects). Five stages: digest that must list written tables (and not call the read-only table "written"); digest where the asked fact is **absent** (must answer `no`/`partly`, fill `missing`, invent no schedule); digest that must list all 6 routines of a package with no invented ones; `next` step (valid capability, right sub-query, no repeated call); expert answer with `[proven: E#]` labels. QDR settings: thinking off, `max_tokens` 600 (1,500 for the expert answer). ×2 seeds | 10 |
| `code` | 10 tasks: parse_duration, merge_intervals, LRU cache, top-k words, South African ID validation (date + Luhn), RLE encode/decode, flatten dict, binary-search bug fix, running-total SQL (window function), Inventory class with validation. Thinking off ×2, thinking on 8K ×1 | 30 |
| `generic` | 6 verifiable instruction-following checks (exactly 3 bullets; < 50 words with no commas; all lowercase; a word exactly twice; title + exact ending; two paragraphs separated by `***`), JSON extraction with `null` when the source is silent, vendor claim vs measured value, meeting notes with corrections (must use the corrected date and budget), grounded "not stated" answer. ×2 | 20 |
| `tools` | OpenAI tool calling with 4 tools: single call, **parallel calls**, no tool needed, enum + integer arguments, argument formatting (date/time), choosing the right tool. ×2 | 12 |
| `longctx` | 3 linked facts planted at 12%, 55% and 88% of a generated log; question needs all three. Sizes 8K, 16K, 32K, 60K tokens (plus 107K at 128K context) | 4 |
| `perf` | Decode speed on a ~700-token answer, ×3 | 3 |
| `conc` | Aggregate throughput at 1, 2 and 4 concurrent requests (and 6 and 8); head-of-line test: 4 meaning checks fired 2 s after a 30K-token prompt | — |

### 5.3 Model-specific setup needed for the distill

- **Thinking switch.** On first load, `reasoning_effort: "none"` was **ignored** by the distill (it still thought, about 50 tokens even for one-word answers). `chat_template_kwargs.enable_thinking`, `enable_thinking`, `reasoning.effort` and `reasoning_effort: "minimal"` were ignored too.
  - **Cause:** LM Studio only maps `reasoning_effort` onto a model whose `model.yaml` declares an `enableThinking` custom field. The server log warned "No valid custom reasoning fields found in model …".
  - **Fix:** added `~/.lmstudio/hub/models/local/qwen3.8-35b-a3b-distill/{manifest.json,model.yaml}`, mirroring Qwen3.6's definition, and loaded the model as `local/qwen3.8-35b-a3b-distill`. After that, `reasoning_effort: "none"` gives 0 thinking tokens.
  - **Without this, SGC's meaning checks would return empty on the distill.**
- **MTP head.** The build includes MTP weights (42 tensors). The lmstudio-community Qwen3.6 build has none. See §7.4.

---

## 6. Results: quality and job time

### 6.1 Totals (both models: 65,536 context, 4 slots, MTP off)

| Suite / setting | Qwen3.6 right | Qwen3.6 time | Distill right | Distill time | Output tokens (3.6 → distill) |
|---|---|---|---|---|---|
| SGC meaning checks | 34/34 | 14 s | 34/34 | 13 s | 68 → 68 |
| Chat, thinking off | 14/22 | 38 s | 14/22 | 30 s | 2,034 → 1,422 |
| Chat, off + brief working | 21/22 | 85 s | 21/22 | 58 s | 4,682 → 3,068 |
| Chat, thinking on (8K) | 10/11 | 272 s | **11/11** | **61 s** | 17,150 → 3,722 |
| Hard, thinking off | 6/12 | 65 s | 6/12 | 26 s | 3,316 → 1,020 |
| Hard, off + brief working | 9/12 | 103 s | 10/12 | 73 s | 5,724 → 3,917 |
| Hard, thinking on (8K) | 4/5 | 281 s | **5/5** | **61 s** | 17,764 → 3,860 |
| QDR curator replica | 10/10 | 76 s | 9/10 | 73 s | 1,785 → 1,534 |
| Coding, thinking off | 19/20 | 91 s | 19/20 | 64 s | 5,140 → 3,472 |
| Coding, thinking on (8K) | 9/10 | 641 s | **10/10** | **146 s** | 38,797 → 9,063 |
| Generic | 19/20 | 51 s | 19/20 | 28 s | 1,603 → 1,530 |
| Tool calling | 12/12 | 17 s | **10/12** | 13 s | 507 → 518 |
| **All quality suites** | **167/190** | **1,734 s** | **168/190** | **648 s** | **98,570 → 33,194** |

- **Thinking-mode cost:** thinking tokens totalled 71,182 on Qwen3.6 and 14,036 on the distill (5.1× fewer).
- **Median time per thinking-on call:**

  | Suite | Qwen3.6 | Distill |
  |---|---|---|
  | Chat | 20.9 s | 3.7 s |
  | Hard | 37.5 s | 8.9 s |
  | Coding | 49.7 s | 9.0 s |

- **Run-outs at the 8K thinking budget:** Qwen3.6 had 2 (SQL window function at 130 s; binary-search fix at 133 s). The distill had none.

### 6.2 Tasks where the two models differed (right / attempts)

| Task | Qwen3.6 | Distill | Comment |
|---|---|---|---|
| tools / parallel calls | 2/2 | **0/2** | Distill called only `get_ticket(INC-1)` |
| qdr / expert labelled answer | 2/2 | 1/2 | Distill put one `[proven: E4]` tag at each paragraph end instead of on every sentence: a real instruction miss that QDR's mechanical checks would flag |
| generic / meeting corrections | 2/2 | 1/2 | **Checker artefact:** the distill's answer was correct but mentioned the old values in "(corrected from …)" notes |
| chat / crew schedule (off; off + working) | 2/2; 2/2 | 1/2; 1/2 | Date arithmetic slip |
| chat / liar puzzle (off) | 0/2 | 1/2 | |
| chat / JSON extraction (off + working) | 1/2 | 2/2 | |
| chat / code fix (thinking on) | 0/1 | 1/1 | |
| code / binary-search fix (thinking on) | 0/1 (ran out) | 1/1 | |
| hard / SQL window function (thinking on) | 0/1 (ran out) | 1/1 | |
| hard / notice period (off + working) | 1/2 | 2/2 | |
| generic / word exactly twice | 1/2 | 2/2 | |

Both models failed the same things otherwise:
- counting rows in a 6K- or 20K-token log, which belongs in SQL or code, not the model;
- the plain-"off" reasoning puzzles, which pass with the brief-working hint;
- South African ID validation (thinking off, 1 of 2 each).

### 6.3 Parallel tool-call probe (follow-up)

With 3 phrasings × 2 seeds and thinking off:

| Model | Thinking | Parallel calls made |
|---|---|---|
| Qwen3.6 | off | **6/6** |
| Distill | off | **0/6** (always only the first call, including two `create_incident` calls in one request) |
| Distill | on | 2/2 |

Consequence:
- **QDR is unaffected**, because the curator returns JSON actions rather than native tool calls.
- Native tool-calling agents with thinking off would need one extra round trip per additional call.

### 6.4 QDR replica detail

Both models:
- returned valid JSON in 8 of 8 action stages;
- never invented a schedule for the "absent" question (they answered `status: no` with `missing` filled);
- listed all 6 package routines with nothing invented;
- did not describe the read-only `CUSTOMER_MASTER` as written;
- chose a valid, non-repeated next call (`structural_graph`).

Timing for a 20.5K-token evidence prompt:

| Read | Time to first token |
|---|---|
| Cold | 10.6–11.4 s |
| Same evidence again (prefix cache) | 0.25–0.36 s |

The prefix cache works on both models.

---

## 7. Results: speed, context and concurrency

### 7.1 Decode speed

| Model | Decode, ~700-token answer |
|---|---|
| Qwen3.6 | 64.6–64.8 tok/s |
| Distill | 64.5–65.1 tok/s |
| Distill with `--speculative-draft-mtp` | 65.5–66.4 tok/s (no gain, see §7.4) |

### 7.2 Long context: recall and reading speed

| Prompt tokens | Qwen3.6 time to first token | Reading speed | Distill time to first token | Reading speed | Recall (both) |
|---|---|---|---|---|---|
| ~8.2K | 3.8 s | 2,157 tok/s | 4.1 s | 2,007 tok/s | ✔ |
| ~16.5K | 8.0 s | 2,047 tok/s | 8.9 s | 1,862 tok/s | ✔ |
| ~33K | 18.9 s | 1,723 tok/s | 20.2 s | 1,637 tok/s | ✔ |
| ~61K | 43.8 s | 1,397 tok/s | 45.5 s | 1,358 tok/s | ✔ |
| ~107K (context 131,072) | 98.1 s | 1,095 tok/s | 100.0 s | 1,074 tok/s | ✔ |

- The distill's 8K-token training examples did **not** hurt recall at 107K on this test.
- A larger prompt (about 136K tokens) was rejected cleanly with a context-overflow error, and the server stayed up.

### 7.3 Concurrency (aggregate tok/s, ~500-token answers)

| Concurrent requests | Qwen3.6, 4 slots | Distill, 4 slots | Distill, 8 slots | Per request |
|---|---|---|---|---|
| 1 | 62.9 | 62.9 | 57.4 | 65 |
| 2 | 92.4 | 92.8 | — | 48–49 |
| 4 | 115.0 | 129.0 | 128.5 | 33–34 |
| 6 | — | — | 139.6 | 23–24 |
| 8 | — | — | 156.3 | 20–21 |

### 7.4 MTP speculative decoding

- **Distill:** `lms load … --speculative-draft-mtp` loads without error, but the server log shows no draft or speculative activity, and decode speed is unchanged. These models load through LM Studio's vision-model (VLM) path, which appears to ignore the flag.
- **Qwen3.6:** the lmstudio-community 8-bit build has no MTP weights, so MTP isn't possible with it in any case.
- **Engines that do use MTP or speculative decoding on this model family:**
  - oMLX: NovaeonStudio reports ~40% gain on short context;
  - llama.cpp MTP GGUFs: ~1.5× reported on an M1 Max;
  - Splash with DFlash2 drafts: vendor reports 1.7× decode on an M5 Pro; an independent Mac mini test measured 2–3.5× on JSON and code and 0.8–1.7× on prose.

### 7.5 Head-of-line blocking

- **Test:** four SGC meaning checks sent one after another, starting 2 s after a ~30K-token prompt.

  | Run | Meaning-check latency (4 checks) | Big request |
  |---|---|---|
  | Distill, 4 slots | **17.07 s**, 0.50 s, 0.45 s, 0.44 s | 21.2 s |
  | Distill, 8 slots | 17.33 s, 0.55 s, 0.53 s, 0.51 s | 22.1 s |
  | Qwen3.6, 4 slots | 0.44 s, **27.14 s**, 0.57 s, 3.06 s | 36.1 s (SGC traffic was hitting the server at the same time, see §8) |

- **Reading:** a request that arrives while a long prompt is being read waits until reading has finished. Generation then interleaves normally.
- **Practical effect:**

  | Long request sharing the server | Meaning check can be delayed by |
  |---|---|
  | QDR curator call (7–11K tokens) | 4–7 s |
  | 60K-token chat | ~45 s |

  This directly affects SGC-4 ("meaning checks must not queue behind long local chats").

### 7.6 Memory cost of context

- Only 10 of the 40 layers use full attention (2 KV heads × head_dim 256), so the KV cache costs about **20 KB per token**:

  | Context | KV cache per full-length sequence |
  |---|---|
  | 65,536 | 1.25 GiB |
  | 131,072 | 2.5 GiB |
  | 262,144 | 5 GiB |

- The linear-attention state per sequence is constant and small.
- Weights resident: 35.2 GiB (Qwen3.6) or 36.8 GiB (distill). The GPU wired limit is already 57,344 MB.
- `lms load --estimate-only` isn't useful for MLX models: it reports the same low-confidence 49.22 GiB and "will fail" for every setting.

---

## 8. Issues found while testing

1. **Distill thinking switch:** see §5.3. A local `model.yaml` is required.
2. **Other clients share the server.**
   - Real SGC traffic (meaning checks, local-route chats with the working hint, ticket-log questions) ran during the tests, mainly 08:43–09:10. It overlapped the Qwen3.6 generic, tools, long-context and concurrency suites, so a few Qwen3.6 timings in that window, including the 27 s head-of-line figure, may be inflated. Accuracy is unaffected.
   - At 09:15–09:18 that client requested the distill by its file key, which **auto-loaded a second copy** (8K context, 1-hour idle timeout). It later blocked a reload with "insufficient system resources" until it was unloaded.
   - **Rule for future tests:** check the server log for foreign traffic, and unload auto-loaded copies when swapping models.
3. **Download stalls.** LM Studio's own download of the distill stopped on 2026-11-07 at 3 of 8 shards. A resumable `curl` with a stall timeout (`-Y 200000 -y 30 -C -`, 40 retries) completed it, with one reset along the way.
4. **SSH to the Mac** intermittently timed out during the banner exchange under load. Retries work.

---

## 9. Options and recommendations

### 9.1 Model

| Option | For | Against |
|---|---|---|
| **A. Keep Qwen3.6** | Proven with QDR and SGC; parallel tool calls with thinking off; official release | Long outputs: 2.7× more job time; thinking mode often runs out at 8K |
| **B. Switch to the distill (recommended for trial)** | Same accuracy; 2.7× less job time; thinking mode 4–5× cheaper and more reliable; same memory, context and concurrency behaviour; drop-in architecture | No parallel tool calls with thinking off; one QDR labelling miss; community fine-tune (provenance; trained partly on prompts from a free community endpoint); needs the local `model.yaml` |
| C. Run both on demand | Choose per workload | Can't hold both in 64 GB at once; each swap takes 11–23 s |

**Gate before making B the default:**
1. Run one real QDR multi-turn session (multiturn-v2 golden questions) against the distill.
2. Run SGC's own test suite against the distill.
3. For tool-calling agents, either turn thinking on for tool turns or accept sequential calls.
4. Update callers that name the model key (`qwen/qwen3.6-35b-a3b`) or load with an identifier alias.

### 9.2 Context

- **Raise the loaded context from 65,536 to 131,072.**
  - It costs at most ~1.25 GiB more, and both models recall correctly at 107K.
  - Keep large prompts for jobs that can wait: about 1 s per 1,100–1,400 prompt tokens, during which the server serves no one else.
- 262,144 fits in memory (~5 GiB KV) but would mean 4+ minutes of reading per full prompt. It isn't useful for interactive work.

### 9.3 Parallelism

- **Keep 4 slots for interactive use:** 34 tok/s per stream at full load.
- Use 8 slots only for batch runs, such as QDR's parallel sub-query digests: +21% total throughput, but 20 tok/s per stream.

### 9.4 Head-of-line blocking (highest-value fix)

| Option | Effect |
|---|---|
| Cap prompt size on interactive routes (SGC local route, QDR curator) and send larger documents only when the server is otherwise idle | No code in the engine needed |
| Run SGC meaning checks on a separate small model (its own LM Studio process) | Likely removes the wait; **not measured** |
| Use an engine that interleaves decoding with prefill (Splash `--decode-share`, default 0.5) | Designed for exactly this |

### 9.5 Decode speed (beyond the model choice)

| Option | Expected gain | Effort / risk |
|---|---|---|
| **Splash engine** (Inco AI; DFlash2 speculative decoding; LM Studio "Bionic" runtime or standalone `splash serve`; supports Qwen3.6-35B-A3B MLX 2–8-bit, so our existing 8-bit weights could be served with `--model-dir`) | Vendor: 1.7× decode on an M5 Pro. Independent Mac mini test: 2–3.5× on JSON and code, 0.8–1.7× on prose; batches concurrent requests | Requires installing third-party software (needs approval; blocked in this session). Separate server and port (8000 clashes with QDR). Draft acceptance on the distill unknown |
| llama.cpp backend with a Qwen3.6 MTP GGUF (e.g. `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) | ~1.5× reported on an M1 Max; mixed reports for the 35B MoE in LM Studio | Different quantisation; must re-check quality and prefill speed |
| oMLX server with the NovaeonStudio build | ~40% decode gain from MTP (vendor) | Another server to run |

Also:
- The VLM prompt-cache disk budget was full at 17 GB on Qwen3.6 (the distill's load set 16.6 GiB for 65K context). Watch it if QDR cache hit rates matter.

---

## 10. How to reproduce

```powershell
# from d:\Work\Agents\int-local-ai\scripts\model_bench  (needs httpx; SGC repo path is hard-coded in bench.py)
python bench.py qwen/qwen3.6-35b-a3b q36_base                    # all suites, ~45 min
python bench.py distill d38 sgc,qdr,tools                        # selected suites
$env:LONG_SIZES="100000"; python bench.py distill d38_long longctx
$env:CONC="1,4,8";        python bench.py distill d38_conc conc
python compare.py results\2026-11-08\q36_base.jsonl results\2026-11-08\d38_nomtp.jsonl
```

Model swap on the Mac:

```bash
lms unload qwen/qwen3.6-35b-a3b
lms load local/qwen3.8-35b-a3b-distill -c 65536 --parallel 4 --identifier distill -y
# restore
lms unload distill
lms load qwen/qwen3.6-35b-a3b -c 65536 --parallel 4 -y
```

Log analysis: `parse_lms.py <log...> <out.csv>` turns LM Studio server logs into one row per request; `stats.py <csv>` groups the rows by caller.

**Files:**
- `scripts/model_bench/bench.py`, `compare.py`, `parse_lms.py`, `stats.py`
- `scripts/model_bench/results/2026-11-08/`:
  - `q36_base`, `d38_nomtp`: full runs (`.jsonl` and `.log`)
  - `q36_long`: corrected long-context run
  - `q36_ctx128`, `d38_ctx128`: 107K recall
  - `d38_par8`: 8-slot concurrency

---

## 11. Changes left on the Mac

- **Restored:** `qwen/qwen3.6-35b-a3b` loaded with 65,536 context and 4 slots, as before the tests (smoke-tested OK).
- **Added:**
  - the distill at `~/.lmstudio/models/NovaeonStudio/Qwen3.8-35B-A3B-Distill-oQ8-fp16-mtp/` (39.5 GB, complete);
  - its LM Studio definition at `~/.lmstudio/hub/models/local/qwen3.8-35b-a3b-distill/`.
- **Temporary, safe to delete:** `/tmp/ytv` (transcript tool environment), `/tmp/dl_distill.sh`, `/tmp/dl_distill.log`.

---

## 12. Sources

- Video: <https://www.youtube.com/watch?v=98ljuFRK2_Y>
- Distill model card: <https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill>
- MLX build used: <https://huggingface.co/NovaeonStudio/Qwen3.8-35B-A3B-Distill-oQ8-fp16-mtp>
- InsiderLLM head-to-head: <https://insiderllm.com/guides/qwen-3-8-35b-a3b-distill-vs-qwen-3-6-35b/>
- Qwen 3.8 vs 3.6 (dense 27B): <https://codersera.com/blog/qwen-3-8-vs-qwen-3-6-2026/>
- Mac mini engine benchmarks (MLX, llama.cpp, Splash): <https://github.com/HDZucht/Mac-Mini-M6-LLM-benchmarks>
- Splash: <https://lmstudio.ai/blog/splash-engine>, <https://github.com/incoai/splash>
- LM Studio MTP: <https://x.com/lmstudio/status/2057889028578455905>, <https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1951>
- LM Studio parallel requests: <https://lmstudio.ai/docs/app/advanced/parallel-requests>
- Qwen (Wikipedia): <https://en.wikipedia.org/wiki/Qwen>
- SGC baseline experiments: `int-standards-guardrails-compliance/docs/superpowers/specs/2026-11-06-sgc-local-ai-prefill-and-query-structuring-design.md` §5.6
