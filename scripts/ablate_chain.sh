cd /workspace/workspace-bench
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
# the L42 residuals from the first pod were not copied off it; recapture (logit lens only, cheap)
METHODS=logit_lens LAYERS=42 RESID_LAYERS=42 uv run python /workspace/smoke_pass_a.py /workspace/out/abl50 50 > /workspace/abl_cap.log 2>&1
uv run python /workspace/ablate_pass.py /workspace/out/abl50 /workspace/out/abl50/resid_L42.pt 50 > /workspace/abl_a.log 2>&1
uv run python /workspace/smoke_pass_b.py /workspace/out/abl50 resid_abl.pt > /workspace/abl_b.log 2>&1
echo CHAIN_DONE > /workspace/abl.done
