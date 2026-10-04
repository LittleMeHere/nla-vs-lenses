cd /workspace/workspace-bench
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
until [ -f /workspace/abl2.done ]; do sleep 30; done
READ_LAYER=56 SKIP_LENSES=1 uv run python /workspace/chain_pass.py /workspace/out/chain60_L56 60 > /workspace/chain56_a.log 2>&1
uv run python /workspace/smoke_pass_b.py /workspace/out/chain60_L56 56 > /workspace/chain56_b.log 2>&1
echo CHAIN_DONE > /workspace/chain56.done
