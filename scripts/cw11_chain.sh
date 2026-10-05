#!/usr/bin/env bash
# Runs on the pod inside tmux: bootstrap, J-Lens memory patch, then CW-11 pass A and pass B.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
mkdir -p /workspace/out
bash /workspace/pod_bootstrap.sh > /workspace/bootstrap.log 2>&1
cd /workspace/workspace-bench && uv pip install -q flash-linear-attention && python3 /workspace/patch_jlens_mem.py src/wsbench/produce/methods.py > /workspace/patch.log 2>&1
uv run python /workspace/cw11_pass_a.py /workspace/out/cw11 /workspace/resid_L42.pt /workspace/items.json 50 > /workspace/cw11_a.log 2>&1
uv run python /workspace/cw11_pass_b.py /workspace/out/cw11 > /workspace/cw11_b.log 2>&1
echo CHAIN_DONE > /workspace/cw11.done
