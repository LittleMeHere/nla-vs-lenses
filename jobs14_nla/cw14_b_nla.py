# CW-14 (NLA worker, K=2): the NLA on the parts saved by cw14_a_split.py.
import json, time, zlib
OUT = Path("/workspace/out/cw14"); OUT.mkdir(parents=True, exist_ok=True); cells = torch.load("/workspace/resid_split.pt")
with open(OUT / "nla_split.jsonl", "w") as fh:
    for c in cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|nla".encode())); t = time.time()
        r = m.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "share": c["share"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} n={len(r.samples)} {time.time()-t:.1f}s", flush=True)
print("SPLIT_NLA_DONE", flush=True)
