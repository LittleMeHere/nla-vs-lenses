"""CW-11 pass B: the NLA (K samples per cell, fixed seed per cell) on the vectors saved by cw11_pass_a.py.
Usage: uv run python cw11_pass_b.py OUT_DIR"""
import json, os, sys, time, zlib, torch
from pathlib import Path
from types import SimpleNamespace
from wsbench.produce.methods import NLA, Sampling
out = Path(sys.argv[1]); K = int(os.environ.get("K", "4"))
SUF = os.environ.get("SUF", "cw11")
cells = torch.load(out / f"resid_{SUF}.pt")
t0 = time.time(); m = NLA(sampling=Sampling(k=K)); m.bind(SimpleNamespace(device="cuda"))
print(f"nla loaded {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB repo {m.repo} adapter {m.adapter} k {K}", flush=True)
with (out / f"nla_{SUF}.jsonl").open("w") as fh:
    for c in cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|nla".encode()))
        t = time.time(); r = m.read(c["h"], c["layer"])
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "layer": c["layer"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} n={len(r.samples)} {time.time()-t:.1f}s: {r.samples[0][:120]!r}", flush=True)
print(f"maxmem {torch.cuda.max_memory_allocated()/1e9:.1f}GB CW11_PASS_B_DONE", flush=True)
