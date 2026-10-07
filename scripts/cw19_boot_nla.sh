#!/usr/bin/env bash
# CW-14 pod B: NLA only, then an NLA worker (K=2) on /workspace/jobs_nla. The job waits in that folder until placed.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
mkdir -p /workspace/out /workspace/jobs_nla /workspace/hf; cd /workspace
{ command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH="$HOME/.local/bin:$PATH"
  [ -d workspace-bench ] || git clone --depth 1 https://github.com/camilablank/workspace-bench
  cd workspace-bench && uv python install 3.12 -q && uv sync --extra gpu --extra dev -q && uv pip install -q "huggingface_hub[cli,hf_transfer]" flash-linear-attention
  uv run hf download ceselder/qwen3.6-27b-nla-rl --include "av_rl_adapters/iter_000400/*" --include "nla_meta.yaml" --include "av_base/*"
  echo BOOTSTRAP_DONE; } > /workspace/bootstrap.log 2>&1
cd /workspace/workspace-bench && MODE=nla K=1 JOBS=/workspace/jobs_nla uv run python /workspace/worker.py > /workspace/worker_nla.log 2>&1
