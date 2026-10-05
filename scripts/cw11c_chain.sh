#!/usr/bin/env bash
# Add-on to CW-11: waits for the main chain, then runs the cue_orth condition through both passes.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
until [ -f /workspace/cw11.done ]; do sleep 30; done
cd /workspace/workspace-bench
uv run python /workspace/cw11c_pass_a.py /workspace/out/cw11 /workspace/resid_L42.pt /workspace/items.json 50 > /workspace/cw11c_a.log 2>&1
SUF=cw11c uv run python /workspace/cw11_pass_b.py /workspace/out/cw11 > /workspace/cw11c_b.log 2>&1
echo CHAIN_DONE > /workspace/cw11c.done
