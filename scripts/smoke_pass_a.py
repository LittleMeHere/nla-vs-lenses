"""Pass A of the compute check. Base Qwen3.6-27B bf16 on one GPU: logit_lens, jlens, rlens, olens on
the first N multihop items via the wsbench producer, then save the layer-42 residuals for the same
cells so pass B (the NLA, a second 27B) can run after the base is freed.
Usage: uv run python smoke_pass_a.py OUT_DIR N"""
import json, sys, time, torch
from pathlib import Path
from wsbench import readplan
from wsbench.produce.producer import Producer, _positions
from wsbench.produce.render import render

out, n = Path(sys.argv[1]), int(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
t0 = time.time()
p = Producer.load("Qwen/Qwen3.6-27B", "logit_lens")
print(f"base loaded {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
import os
for m in os.environ.get("METHODS", "logit_lens,jlens,rlens,olens").split(","):
    t = time.time(); p.use(m)
    layers = [int(x) for x in os.environ["LAYERS"].split(",")] if os.environ.get("LAYERS") else None
    tag = f"{m}_L{os.environ['LAYERS']}" if layers else m
    p.run_family("multihop", out / f"{tag}.multihop.jsonl", limit=n, layers=layers)
    print(f"{m} done {time.time()-t:.0f}s maxmem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
cells = []
RL = [int(x) for x in os.environ.get("RESID_LAYERS", "42").split(",")]
for spec in readplan.plan("multihop")[:n]:
    r = render(spec, p.backend.tokenizer)
    pos = _positions(spec.positions, r.tokens); idx = [q % len(r) for q in pos]
    acts = p.backend.capture(r.ids, RL, idx)
    with torch.no_grad():
        logits = p.backend.model(torch.tensor([r.ids], device=p.backend.device)).logits[0]
    for i, (q, j) in enumerate(zip(pos, idx)):
        top = logits[j].float().topk(10)
        nxt = [p.backend.tokenizer.decode([int(t)]) for t in top.indices]
        for L in RL:
            cells.append({"id": spec.id, "layer": L, "pos": q, "token": r.decoded[j], "h": acts[L][i].clone(),
                          "next_top10": nxt, "next_scores": [round(float(v), 2) for v in top.values]})
torch.save(cells, out / f"resid_L{os.environ.get('RESID_LAYERS', '42')}.pt")
with (out / "next_top10.jsonl").open("w") as fh:
    for c in cells:
        fh.write(json.dumps({"id": c["id"], "pos": c["pos"], "tokens": c["next_top10"], "scores": c["next_scores"]}, ensure_ascii=False) + "\n")
print(f"saved {len(cells)} L42 cells; PASS_A_DONE", flush=True)
