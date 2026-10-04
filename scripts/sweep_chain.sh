cd /workspace/workspace-bench
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
METHODS=jlens,olens LAYERS=16,28,36 RESID_LAYERS=16,28,36 uv run python /workspace/smoke_pass_a.py /workspace/out/sweep50 50 > /workspace/sweep_a.log 2>&1
uv run python /workspace/smoke_pass_b.py /workspace/out/sweep50 16,28,36 > /workspace/sweep_b.log 2>&1
echo CHAIN_DONE > /workspace/sweep.done
