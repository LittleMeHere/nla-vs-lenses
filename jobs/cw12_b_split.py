# CW-12 step 2 (base worker): 30 items, split at k=1024 (chosen from the 5-item check), J-Lens + oracle lens per part.
import json, time, zlib
OUT = Path("/workspace/out/cw12"); N_ITEMS, K = 30, 1024
seen, sel = set(), []
for c in torch.load("/workspace/resid_L42.pt"):
    if c["layer"] == 42 and c["id"] not in seen: seen.add(c["id"]); sel.append(c)
dev = p.backend.device
Q = torch.load(OUT / "jbasis.pt")["Q"].float().to(dev)[:, :K]
g = torch.Generator(device="cpu").manual_seed(0); R, _ = torch.linalg.qr(torch.randn(Q.shape[0], K, generator=g)); R = R.to(dev)
# J-Lens top-10 computed directly (the producer's read moves the Jacobian to the GPU on every call, about 20 s each)
from wsbench.produce.methods import _load_jacobians
p.use("jlens"); tok = p.backend.tokenizer
J = _load_jacobians(p.method.repo, p.method.filename, "cpu")[42].to(dev); W = p.backend.unembed.float()
den = torch.cat([(W[i:i + 16384] @ J).norm(dim=1) for i in range(0, W.shape[0], 16384)]).clamp_min(1e-9)
def jl(h):
    s = (W @ (J @ h.to(dev).float())) / den
    return [tok.decode([i]) for i in torch.topk(s, 10).indices.tolist()]
cells = []
for c in sel[:N_ITEMS]:
    h = c["h"].to(dev).float(); hj = Q @ (Q.T @ h); hr = R @ (R.T @ h)
    for cond, v in (("full", h), ("J1024", hj), ("N1024", h - hj), ("R1024", hr), ("RN1024", h - hr)):
        cells.append({"id": c["id"], "cond": cond, "h": v.cpu(), "layer": 42, "share": float((v.norm() / h.norm()) ** 2), "tokens": jl(v)})
torch.save(cells, OUT / "resid_split.pt")
p.use("olens"); p.method.sampling.k = 2
with open(OUT / "olens_split.jsonl", "w") as fh:
    for c in cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|olens".encode())); t = time.time()
        r = p.method.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "share": c["share"], "tokens": c["tokens"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} {time.time()-t:.1f}s", flush=True)
print("SPLIT_OLENS_DONE", flush=True)
