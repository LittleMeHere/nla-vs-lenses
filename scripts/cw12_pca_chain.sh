#!/usr/bin/env bash
# After the NLA split job: stop that worker, run the PCA comparison on a base worker, then the NLA on its parts.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
until [ -f /workspace/jobs_nla/cw12_c_nla.done ] || [ -f /workspace/jobs_nla/cw12_c_nla.err ]; do sleep 10; done
touch /workspace/jobs_nla/STOP
while tmux has-session -t switch 2>/dev/null; do sleep 5; done
cd /workspace/workspace-bench
( until [ -f /workspace/jobs_pca/cw12_d_pca.done ] || [ -f /workspace/jobs_pca/cw12_d_pca.err ]; do sleep 10; done; touch /workspace/jobs_pca/STOP ) &
MODE=base JOBS=/workspace/jobs_pca uv run python /workspace/worker.py > /workspace/worker_base2.log 2>&1
( until [ -f /workspace/jobs_nla2/cw12_e_nla.done ] || [ -f /workspace/jobs_nla2/cw12_e_nla.err ]; do sleep 10; done; touch /workspace/jobs_nla2/STOP ) &
MODE=nla K=2 JOBS=/workspace/jobs_nla2 uv run python /workspace/worker.py > /workspace/worker_nla2.log 2>&1
echo CHAIN_DONE > /workspace/cw12_pca.done
