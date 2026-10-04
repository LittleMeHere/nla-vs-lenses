#!/usr/bin/env bash
# Runs on the Runpod pod. Installs uv + workspace-bench, downloads every model needed for the
# smoke test into /workspace/hf. Idempotent; log in /workspace/bootstrap.log.
set -euo pipefail
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /workspace/hf
cd /workspace
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"
[ -d workspace-bench ] || git clone --depth 1 https://github.com/camilablank/workspace-bench
cd workspace-bench && uv python install 3.12 -q && uv sync --extra gpu --extra dev -q
uv pip install -q "huggingface_hub[cli,hf_transfer]"
dl() { uv run hf download "$@"; }
dl Qwen/Qwen3.6-27B
dl camilablank/workspace-lenses --include "qwen3.6-27b/j-lens/*" --include "qwen3.6-27b/r-lens/*"
dl agu18dec/local-workspace --repo-type dataset --include "ckpts/ao/rl/s3d.ddp600.s0/iter_000600/*" || echo OLENS_GATED_SKIPPED
dl ceselder/olens-rl-grpo-trajectory-qwen36-27b --include "step_00390/*"
dl neuronpedia/jacobian-lens --include "qwen3.6-27b/jlens/Salesforce-wikitext/*"
dl ceselder/qwen3.6-27b-nla-rl --include "av_rl_adapters/iter_000400/*" --include "nla_meta.yaml" --include "av_base/*"
echo BOOTSTRAP_DONE
