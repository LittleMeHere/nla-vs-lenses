# CW-17 (NLA worker, K=2): the NLA (L42) on one half of the poetry prompts, from the parts built by cw17_a1.py.
import json, time, zlib
OUT = Path("/workspace/out/cw17"); OUT.mkdir(parents=True, exist_ok=True); cells = torch.load("/workspace/resid_poetry_L42.pt")
ids = []; [ids.append(c["id"]) for c in cells if c["id"] not in ids]; mine = set(ids[:50])
with open(OUT / "nla_poetry_a.jsonl", "w") as fh:
    for c in cells:
        if c["id"] not in mine: continue
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|nla".encode())); t = time.time(); r = m.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "layer": 42, "share": c["share"], "tokens": c["tokens"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} n={len(r.samples)} {time.time()-t:.1f}s", flush=True)
print("NLA_a_DONE", flush=True)
