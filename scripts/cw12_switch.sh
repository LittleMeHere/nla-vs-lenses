#!/usr/bin/env bash
# Waits for the base worker's split job, stops that worker, then starts an NLA worker on its own job folder.
export PATH=$HOME/.local/bin:$PATH HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a; . /workspace/.hf_env; set +a
until [ -f /workspace/jobs/cw12_b_split.done ] || [ -f /workspace/jobs/cw12_b_split.err ]; do sleep 10; done
touch /workspace/jobs/STOP
while tmux has-session -t base 2>/dev/null; do sleep 5; done
cd /workspace/workspace-bench && MODE=nla K=2 JOBS=/workspace/jobs_nla uv run python /workspace/worker.py > /workspace/worker_nla.log 2>&1
