# Day-1 Provisioning & Operations Runbook — Mac Mini M4 Pro (48 GB)

Follow this step-by-step runbook when unboxing and configuring your **Mac Mini M4 Pro (48 GB)** to turn it into a reliable, headless or semi-headless local AI inference appliance (`za-local`).

---

## Phase 1: Base macOS System & Power Preparation

### 1.1 Network & SSH Setup
1. Connect the Mac Mini to your LAN via **wired Ethernet** (Gigabit or 10GbE).
2. Assign a **Static IPv4 Reservation** on your router (e.g., `192.168.1.50`) so your Windows workstation (`D:\Work\Agents\*`) always reaches it at a fixed IPv4 address.
3. Enable **Remote Login (SSH)** and **Screen Sharing**:
   * *System Settings* $\rightarrow$ *General* $\rightarrow$ *Sharing* $\rightarrow$ toggle on **Remote Login** and **Screen Sharing**.

### 1.2 Disable Sleep & Enable Headless Auto-Recovery
Open Terminal on the Mac Mini and run:
```bash
# Prevent sleep, disk sleep, and enable Wake-on-LAN + auto-restart on power loss
sudo pmset -a sleep 0 disksleep 0 displaysleep 15 womp 1 autorestart 1
```

### 1.3 Unlock 42–43 GB of Unified Memory for Metal / GPU
Run the included provisioning script [`scripts/macos/configure_vram_limit.sh`](file:///d:/Work/Agents/int-local-ai/scripts/macos/configure_vram_limit.sh) to raise macOS's default ~36 GB Metal limit to **43 GB (`44032 MB`)** and persist it across reboots via `launchd`:
```bash
chmod +x scripts/macos/configure_vram_limit.sh
sudo ./scripts/macos/configure_vram_limit.sh 44032
```

---

## Phase 2: Toolchain Installation (Homebrew, `uv`, MLX, `llama.cpp`, LM Studio)

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install core CLI tools, llama.cpp (Metal-accelerated), and uv (fast Python manager)
brew install git wget curl jq htop btop llama.cpp uv

# 3. Install LM Studio (includes GUI + headless 'lms' CLI + built-in MLX & llama.cpp engines)
brew install --cask lm-studio

# 4. Bootstrap LM Studio's CLI ('lms') into your PATH
~/.lmstudio/bin/lms bootstrap

# 5. Install native MLX serving stack (mlx-lm, mlx-vlm, vllm-mlx, huggingface-hub)
uv tool install --force mlx-lm
uv tool install --force vllm-mlx
uv tool install --force huggingface-hub
```

---

## Phase 3: Downloading the Core Model Roster

Create a clean, dedicated models directory (`~/ai-models`) or use LM Studio's cache alongside Hugging Face CLI:

### 3.1 Option A: Via LM Studio CLI (`lms`) — Best for QDR (`:1234`)
```bash
# 1. Pull the mandatory 1024-dim BGE-M3 embedding model for QDR (6,207 ChromaDB chunks)
lms get text-embedding-bge-m3

# 2. Pull Qwen3.6-35B-A3B (choose MLX 6-bit for max Apple Silicon speed, or Unsloth UD-Q5_K_XL GGUF)
lms get mlx-community/Qwen3.6-35B-A3B-6bit
# Or GGUF dynamic quant:
lms get unsloth/Qwen3.6-35B-A3B-GGUF -q UD-Q5_K_XL

# 3. Optional: Pull fast 3B FIM model for VS Code Twinny inline autocomplete
lms get qwen2.5-coder-3b-base
```

### 3.2 Option B: Direct Hugging Face Download for `vllm-mlx` / `llama-server` (`:8080`)
```bash
mkdir -p ~/ai-models/gguf

# Download MLX 6-bit weights (~28 GB) into HF cache for vllm-mlx / mlx_lm.server
hf download mlx-community/Qwen3.6-35B-A3B-6bit

# Or download Unsloth Dynamic 5-bit GGUF (~25 GB) + Vision Projector for llama-server
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf \
  --local-dir ~/ai-models/gguf
```

---

## Phase 4: Starting the Inference Services

### 4.1 Starting LM Studio Headless Server on Port `1234` (Primary for QDR + SGC)
```bash
# Start LM Studio API server bound to LAN (0.0.0.0:1234) with CORS enabled
lms server start --port 1234 --cors

# Load and PIN text-embedding-bge-m3 permanently (--ttl 0 prevents idle unload!)
lms load text-embedding-bge-m3 --identifier text-embedding-bge-m3 --ttl 0 --gpu max

# Load and PIN Qwen3.6-35B-A3B with 64K-128K context window
lms load mlx-community/Qwen3.6-35B-A3B-6bit \
  --identifier qwen/qwen3.6-35b-a3b \
  --context-length 65536 \
  --ttl 0 \
  --gpu max

# Verify loaded models and memory usage
lms ps
```

### 4.2 Starting `vllm-mlx` or `llama-server` on Port `8080`
Use the provided production startup scripts in [`scripts/macos/`](file:///d:/Work/Agents/int-local-ai/scripts/macos/):
* **For `vllm-mlx` (Native MLX + SSD Prefix Cache + Anthropic `/v1/messages` + OpenAI `/v1`):**
  ```bash
  ./scripts/macos/start_vllm_mlx.sh
  ```
* **For `llama-server` (Direct `llama.cpp` + Flash Attention + `Q8_0` KV Cache):**
  ```bash
  ./scripts/macos/start_llama_server.sh
  ```

---

## Phase 5: End-to-End Verification & Benchmarking

Run these verification checks from your **Windows workstation (`PowerShell`)** to confirm QDR, SGC, and DIDS can reach the Mac Mini Pro:

```powershell
$MAC_IP = "192.168.1.50"

# 1. Verify OpenAI-compatible /v1/models list
Invoke-RestMethod -Uri "http://$($MAC_IP):1234/v1/models" | ConvertTo-Json -Depth 4

# 2. Verify 1024-dim BGE-M3 Embedding (QDR smoke test)
$embedBody = @{
    model = "text-embedding-bge-m3"
    input = @("SELECT * FROM SDM.ONTREADY_RELEASE_PROCESS;")
} | ConvertTo-Json
$embedResp = Invoke-RestMethod -Method Post -Uri "http://$($MAC_IP):1234/v1/embeddings" -ContentType "application/json" -Body $embedBody
Write-Host "Embedding dimension (must equal 1024): $($embedResp.data[0].embedding.Count)"

# 3. Verify Qwen3.6-35B-A3B Chat Completion (Fast Non-Thinking JSON Curator Test)
$chatBody = @{
    model = "qwen/qwen3.6-35b-a3b"
    messages = @(
        @{ role = "system"; content = "Reply with ONLY a JSON object." },
        @{ role = "user"; content = 'Return {"status": "ready", "arch": "MoE-A3B"}' }
    )
    temperature = 0.2
    top_p = 0.90
    top_k = 20
    chat_template_kwargs = @{ enable_thinking = $false }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri "http://$($MAC_IP):1234/v1/chat/completions" -ContentType "application/json" -Body $chatBody | ConvertTo-Json -Depth 5
```

