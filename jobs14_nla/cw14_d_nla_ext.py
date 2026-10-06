# CW-14 (NLA worker, K=2): the NLA on the extension parts built by scripts/cw14_ext_cells.py.
import json, time, zlib
OUT = Path("/workspace/out/cw14"); OUT.mkdir(parents=True, exist_ok=True); cells = torch.load("/workspace/resid_ext.pt")
with open(OUT / "nla_ext.jsonl", "w") as fh:
    for c in cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|nla".encode())); t = time.time()
        r = m.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "share": c["share"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} n={len(r.samples)} {time.time()-t:.1f}s", flush=True)
print("EXT_NLA_DONE", flush=True)
