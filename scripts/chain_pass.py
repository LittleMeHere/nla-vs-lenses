"""chain_intermediates (or FAMILY): read every method at the LAST token of the chat render (the bank's read
cell, assistant onset after the empty think block). J-Lens + logit lens over layers 20..60 step 4
plus 42; oracle lens at 42; save L42 residuals for the NLA pass (smoke_pass_b.py).
Usage: uv run python chain_pass.py OUT N"""
import json, os, sys, time, torch
from pathlib import Path
from wsbench import readplan
from wsbench.produce.producer import Producer
from wsbench.produce.render import render

out, n = Path(sys.argv[1]), int(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
FAMILY = os.environ.get("FAMILY", "chain_intermediates")
specs = readplan.plan(FAMILY)[:n]
RL = int(os.environ.get("READ_LAYER", "42"))  # layer the prose readers get
GRID = sorted(set(range(20, 61, 4)) | {42, RL})
t0 = time.time(); p = Producer.load("Qwen/Qwen3.6-27B", "logit_lens")
print(f"base loaded {time.time()-t0:.0f}s", flush=True)
rendered = []
for spec in specs:
    r = render(spec, p.backend.tokenizer); j = len(r) - 1
    acts = p.backend.capture(r.ids, GRID, [j])
    logits = p.backend.model(torch.tensor([r.ids], device=p.backend.device)).logits[0, j].float()
    top = logits.topk(10)
    rendered.append((spec, r, j, acts, [p.backend.tokenizer.decode([int(t)]) for t in top.indices]))
print(f"captured {len(rendered)} items; read tokens: {sorted(set(repr(r.decoded[j]) for _, r, j, _, _ in rendered))}", flush=True)
for m in ([] if os.environ.get("SKIP_LENSES") else ["logit_lens", "jlens"]):
    p.use(m)
    with (out / f"{m}.{FAMILY}.jsonl").open("w") as fh:
        for spec, r, j, acts, _ in rendered:
            for L in GRID:
                o = p.method.read(acts[L][0], L)
                fh.write(json.dumps({"id": spec.id, "layer": L, "pos": j, "token": r.decoded[j], "tokens": o.tokens, "scores": o.scores}, ensure_ascii=False) + "\n")
    print(f"{m} done", flush=True)
with (out / "next_top10.jsonl").open("w") as fh:
    for spec, r, j, acts, nxt in rendered:
        fh.write(json.dumps({"id": spec.id, "pos": j, "tokens": nxt}, ensure_ascii=False) + "\n")
cells = [{"id": s.id, "layer": RL, "pos": j, "token": r.decoded[j], "h": a[RL][0].clone()} for s, r, j, a, _ in rendered]
torch.save(cells, out / f"resid_L{RL}.pt"); print(f"saved L{RL} residuals", flush=True)
p.use("olens")
with (out / f"olens_L{RL}.{FAMILY}.jsonl").open("w") as fh:
    for c in cells:
        t = time.time(); o = p.method.read(c["h"], RL)
        fh.write(json.dumps({"id": c["id"], "layer": RL, "pos": c["pos"], "token": c["token"], "samples": o.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"olens {c['id']} {time.time()-t:.1f}s", flush=True)
print("CHAIN_PASS_DONE", flush=True)
