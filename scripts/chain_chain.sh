cd /workspace/workspace-bench
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
uv run python /workspace/chain_pass.py /workspace/out/chain60 60 > /workspace/chain_a.log 2>&1
uv run python /workspace/smoke_pass_b.py /workspace/out/chain60 > /workspace/chain_b.log 2>&1
echo CHAIN_DONE > /workspace/chain.done
