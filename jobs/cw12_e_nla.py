# CW-12 step 2 (NLA worker, K samples set by the worker): the NLA on the parts saved by cw12_d_pca.py.
import json, time, zlib
OUT = Path("/workspace/out/cw12"); cells = torch.load(OUT / "resid_pca.pt")
with open(OUT / "nla_pca.jsonl", "w") as fh:
    for c in cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|nla".encode())); t = time.time()
        r = m.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "share": c["share"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} n={len(r.samples)} {time.time()-t:.1f}s", flush=True)
print("PCA_NLA_DONE", flush=True)
