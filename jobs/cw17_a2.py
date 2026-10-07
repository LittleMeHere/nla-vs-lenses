# CW-17 (base worker, pod O2): the oracle lens (L44) on poetry prompts 51-100, from the parts built by cw17_a1.py.
import json, time, zlib
OUT = Path("/workspace/out/cw17"); OUT.mkdir(parents=True, exist_ok=True); cells = torch.load("/workspace/resid_poetry_L44.pt")
ids = []; [ids.append(c["id"]) for c in cells if c["id"] not in ids]; mine = set(ids[50:])
p.use("olens"); p.method.sampling.k = 2
with open(OUT / "olens_poetry_b.jsonl", "w") as fh:
    for c in cells:
        if c["id"] not in mine: continue
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|olens".encode())); t = time.time(); r = p.method.read(c["h"], 44)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "layer": 44, "share": c["share"], "tokens": c["tokens"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} {time.time()-t:.1f}s", flush=True)
print("OLENS_B_DONE", flush=True)
