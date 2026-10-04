# brew at the last render token (":" after the "Answer" prefill), L42: originals, then removal/swap/random
# controls, then the NLA on everything, then the missing NLA pass of the multihop revised swap (#7).
cd /workspace/workspace-bench
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
FAMILY=brew_intermediates uv run python /workspace/chain_pass.py /workspace/out/brew50 50 > /workspace/brew_a.log 2>&1
COS=0.8 SWAP_SCALES=1,2,4,8,16 uv run python /workspace/ablate2_pass.py /workspace/out/brew50_abl /workspace/out/brew50/resid_L42.pt /workspace/brew_items.json 50 > /workspace/brew_abl.log 2>&1
uv run python /workspace/smoke_pass_b.py /workspace/out/brew50 > /workspace/brew_b.log 2>&1
uv run python /workspace/smoke_pass_b.py /workspace/out/brew50_abl resid_abl2.pt > /workspace/brew_abl_b.log 2>&1
mkdir -p /workspace/out/abl3 && cp /workspace/resid_abl3.pt /workspace/out/abl3/resid_abl2.pt
uv run python /workspace/smoke_pass_b.py /workspace/out/abl3 resid_abl2.pt > /workspace/abl3_b.log 2>&1
echo CHAIN_DONE > /workspace/brew.done
