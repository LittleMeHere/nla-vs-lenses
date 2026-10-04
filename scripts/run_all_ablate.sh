#!/usr/bin/env bash
# bootstrap (idempotent), re-apply the J-Lens patch, install fla, then the layer sweep chain
export PATH=$HOME/.local/bin:$PATH
bash /workspace/pod_bootstrap.sh > /workspace/bootstrap.log 2>&1
cd /workspace/workspace-bench && uv pip install -q flash-linear-attention && python3 /workspace/patch_jlens_mem.py src/wsbench/produce/methods.py
set -a; . /workspace/.hf_env; set +a
export HF_HOME=/workspace/hf
uv run hf download agu18dec/local-workspace --repo-type dataset --include "ckpts/ao/rl/s3d.ddp600.s0/iter_000600/*" > /dev/null 2>&1
bash /workspace/ablate_chain.sh
