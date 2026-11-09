#!/bin/sh
# Recreates the LM Studio local definition local/qwen3.8-distill-thinkswitch on the Mac.
# It points at the NovaeonStudio MLX build and adds the enableThinking switch,
# without which reasoning_effort:"none" is ignored. Run ON THE MAC after the
# model download (~/.lmstudio/models/NovaeonStudio/Qwen3.8-35B-A3B-Distill-oQ8-fp16-mtp/).
# From the laptop: ssh mikem@192.168.3.175 'sh -s' < scripts/lmstudio/install_distill_definition.sh
set -e
D=$HOME/.lmstudio/hub/models/local/qwen3.8-distill-thinkswitch; mkdir -p $D; cat > $D/manifest.json <<EOF
{
  "type": "model",
  "owner": "local",
  "name": "qwen3.8-distill-thinkswitch",
  "dependencies": [
    {
      "type": "model",
      "purpose": "baseModel",
      "modelKeys": ["novaeonstudio/qwen3.8-35b-a3b-distill-oq8-fp16-mtp"],
      "sources": [{"type": "huggingface", "user": "NovaeonStudio", "repo": "Qwen3.8-35B-A3B-Distill-oQ8-fp16-mtp"}]
    }
  ],
  "revision": 1
}
EOF
cat > $D/model.yaml <<EOF
model: local/qwen3.8-distill-thinkswitch
base:
  - key: novaeonstudio/qwen3.8-35b-a3b-distill-oq8-fp16-mtp
    sources:
      - type: huggingface
        user: NovaeonStudio
        repo: Qwen3.8-35B-A3B-Distill-oQ8-fp16-mtp
metadataOverrides:
  domain: llm
  architectures:
    - qwen35moe
  compatibilityTypes:
    - safetensors
  paramsStrings:
    - 35B
  minMemoryUsageBytes: 20400000000
  contextLengths:
    - 262144
  vision: true
  reasoning: true
  trainedForToolUse: true
config:
  operation:
    fields:
      - key: llm.prediction.temperature
        value: 0.6
      - key: llm.prediction.topKSampling
        value: 20
      - key: llm.prediction.topPSampling
        value:
          checked: true
          value: 0.95
customFields:
  - key: enableThinking
    displayName: Enable Thinking
    description: Controls whether the model will think before replying
    type: boolean
    defaultValue: true
    effects:
      - type: setJinjaVariable
        variable: enable_thinking
  - key: preserveThinking
    displayName: Preserve Thinking
    description: Preserve reasoning content in all prior assistant turns instead of only the most recent one
    type: boolean
    defaultValue: false
    effects:
      - type: setJinjaVariable
        variable: preserve_thinking
EOF
~/.lmstudio/bin/lms ls | grep -i -E 'distill|qwen3.6'
