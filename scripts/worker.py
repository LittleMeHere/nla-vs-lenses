"""Keep one model loaded on the pod and run job files as they arrive, so trying variants does not cost a reload.
MODE=base : loads Qwen3.6-27B with J-Lens (and switches to the oracle lens on request). Jobs get `p` (the Producer).
MODE=nla  : loads the NLA reader. Jobs get `m` (the NLA) and `Sampling`.
A job is a Python file dropped into /workspace/jobs/. It is run once with those names defined, its output goes to
/workspace/jobs/<name>.log, and <name>.done or <name>.err is written. The job file itself is the record of what ran.
Stop the worker by creating /workspace/jobs/STOP.  Usage (in tmux): MODE=base uv run python /workspace/worker.py"""
import contextlib, io, os, sys, time, traceback, torch
from pathlib import Path
JOBS = Path(os.environ.get("JOBS", "/workspace/jobs")); JOBS.mkdir(parents=True, exist_ok=True)
MODE = os.environ.get("MODE", "base"); torch.set_grad_enabled(False)
t0 = time.time(); ns = {"torch": torch, "Path": Path}
if MODE == "base":
    from wsbench.produce.producer import Producer
    ns["p"] = Producer.load("Qwen/Qwen3.6-27B", "jlens")
else:
    from types import SimpleNamespace
    from wsbench.produce.methods import NLA, Sampling
    m = NLA(sampling=Sampling(k=int(os.environ.get("K", "4")))); m.bind(SimpleNamespace(device="cuda"))
    ns.update(m=m, Sampling=Sampling)
print(f"worker {MODE} ready after {time.time()-t0:.0f}s, mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB, watching {JOBS}", flush=True)
(JOBS / f"READY_{MODE}").write_text(str(time.time()))
while not (JOBS / "STOP").exists():
    todo = sorted(f for f in JOBS.glob("*.py") if not (f.with_suffix(".done").exists() or f.with_suffix(".err").exists()))
    if not todo:
        time.sleep(5); continue
    f = todo[0]; t = time.time(); print(f"job {f.name} start", flush=True)
    with open(f.with_suffix(".log"), "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            exec(compile(f.read_text(), str(f), "exec"), dict(ns)); ok = True
        except Exception:
            traceback.print_exc(); ok = False
    f.with_suffix(".done" if ok else ".err").write_text(f"{time.time()-t:.1f}s")
    print(f"job {f.name} {'done' if ok else 'ERROR'} {time.time()-t:.0f}s", flush=True)
print("worker stopped", flush=True)
