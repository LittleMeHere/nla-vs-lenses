# CW-14 extension (base worker): the oracle lens on the parts built by scripts/cw14_ext_cells.py.
import json, time, zlib
OUT = Path("/workspace/out/cw14"); cells = torch.load("/workspace/resid_ext.pt")
p.use("olens"); p.method.sampling.k = 2
with open(OUT / "olens_ext.jsonl", "w") as fh:
    for c in cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|olens".encode())); t = time.time()
        r = p.method.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "share": c["share"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} {time.time()-t:.1f}s", flush=True)
print("EXT_OLENS_DONE", flush=True)
