cd /workspace/workspace-bench
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
until [ -f /workspace/chain56.done ]; do sleep 30; done
COS=0.8 SWAP_SCALES=1,2,4,8,16 uv run python /workspace/ablate2_pass.py /workspace/out/abl3 /workspace/resid_abl50_L42.pt /workspace/items.json 50 > /workspace/abl3_a.log 2>&1
uv run python /workspace/smoke_pass_b.py /workspace/out/abl3 resid_abl2.pt > /workspace/abl3_b.log 2>&1
echo CHAIN_DONE > /workspace/abl3.done
