"""Pass B: the NLA reader alone (av_base + RL adapter) on the residuals saved by pass A.
Usage: uv run python smoke_pass_b.py OUT_DIR"""
import json, sys, time, torch
from pathlib import Path
from types import SimpleNamespace
from wsbench.produce.methods import NLA
out = Path(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else "42"
cells = torch.load(out / (f"resid_L{tag}.pt" if not tag.endswith(".pt") else tag))
t0 = time.time(); m = NLA(); m.bind(SimpleNamespace(device="cuda"))
print(f"nla loaded {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
name = "nla.multihop.jsonl" if tag == "42" else (f"nla_{Path(tag).stem}.jsonl" if tag.endswith(".pt") else f"nla_L{tag}.multihop.jsonl")
with (out / name).open("w") as fh:
    for c in cells:
        t = time.time(); r = m.read(c["h"], c["layer"])
        fh.write(json.dumps({"id": c["id"], "layer": c["layer"], "cond": c.get("cond"), "partner": c.get("partner"), "pos": c["pos"], "token": c["token"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} L{c['layer']} {time.time()-t:.1f}s: {r.samples[0][:160]!r}", flush=True)
print(f"maxmem {torch.cuda.max_memory_allocated()/1e9:.1f}GB PASS_B_DONE", flush=True)
