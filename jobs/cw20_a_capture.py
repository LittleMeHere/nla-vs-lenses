# CW-20 (base worker; designs/CW-20_olens_rebuild.md): capture L44 at the read token for multihop items 31-100, save
# the activation and the rendered ids, run the oracle lens at L44 with 2 samples.
import json, time, zlib
from wsbench import readplan
from wsbench.produce.producer import _positions
from wsbench.produce.render import render
OUT = Path("/workspace/out/cw20"); OUT.mkdir(parents=True, exist_ok=True); L = 44
tok = p.backend.tokenizer
caps = []
for spec in readplan.plan("multihop")[30:100]:
    r = render(spec, tok); pos = _positions(spec.positions, r.tokens); idx = [q % len(r) for q in pos]
    assert len(idx) == 1, (spec, pos)
    h = p.backend.capture(r.ids, [L], idx)[L][0].float().cpu()
    ids = r.ids[0].tolist() if hasattr(r.ids, "tolist") and getattr(r.ids, "ndim", 1) == 2 else list(map(int, r.ids))
    caps.append({"id": spec.id, "layer": L, "pos": idx[0], "token": r.decoded[idx[0]], "ids": ids, "h": h})
torch.save(caps, OUT / "resid_L44.tmp"); (OUT / "resid_L44.tmp").rename(OUT / "resid_L44.pt")
print("CAPTURED", len(caps), caps[0]["id"], caps[-1]["id"], "ids len", len(caps[0]["ids"]), "pos", caps[0]["pos"], flush=True)
p.use("olens"); p.method.sampling.k = 2
with open(OUT / "olens_L44.jsonl", "w") as fh:
    for c in caps:
        torch.manual_seed(zlib.crc32(f"{c['id']}|full|olens|L44".encode())); t = time.time()
        r = p.method.read(c["h"], L)
        fh.write(json.dumps({"id": c["id"], "layer": L, "pos": c["pos"], "token": c["token"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {time.time()-t:.1f}s", flush=True)
print("OLENS_L44_DONE", flush=True)
