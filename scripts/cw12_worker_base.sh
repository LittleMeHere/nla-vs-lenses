#!/usr/bin/env bash
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
cd /workspace/workspace-bench && MODE=base uv run python /workspace/worker.py > /workspace/worker_base.log 2>&1
