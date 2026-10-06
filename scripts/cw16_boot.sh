#!/usr/bin/env bash
# CW-16 pod: workspace-bench env (for the model class), then download the NLA reconstructor only.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
set -a; . /workspace/.hf_env; set +a
mkdir -p /workspace/hf; cd /workspace
{ command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH="$HOME/.local/bin:$PATH"
  [ -d workspace-bench ] || git clone --depth 1 https://github.com/camilablank/workspace-bench
  cd workspace-bench && uv python install 3.12 -q && uv sync --extra gpu --extra dev -q && uv pip install -q "huggingface_hub[cli,hf_transfer]" flash-linear-attention
  uv run hf download ceselder/qwen3.6-27b-nla-rl --include "ar_reconstructor/*" --include "nla_meta.yaml" --include "av_base/tokenizer*" --include "av_base/chat_template*" --include "av_base/special_tokens*"
  echo BOOTSTRAP_DONE; } > /workspace/bootstrap.log 2>&1
