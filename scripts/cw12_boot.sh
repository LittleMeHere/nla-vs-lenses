#!/usr/bin/env bash
# Pod setup for CW-12: bootstrap, J-Lens memory patch, then a base-model worker that waits for job files.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
mkdir -p /workspace/out /workspace/jobs
bash /workspace/pod_bootstrap.sh > /workspace/bootstrap.log 2>&1
cd /workspace/workspace-bench && uv pip install -q flash-linear-attention && python3 /workspace/patch_jlens_mem.py src/wsbench/produce/methods.py > /workspace/patch.log 2>&1
MODE=base uv run python /workspace/worker.py > /workspace/worker_base.log 2>&1
