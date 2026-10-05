# CW-12 step 1 (base worker): build the J basis at L42, split 5 items at several k, J-Lens + oracle lens on each part.
import json, re, time, zlib, random
from wsbench.produce.methods import _load_jacobians
OUT = Path("/workspace/out/cw12"); OUT.mkdir(parents=True, exist_ok=True)
items = {it["name"]: it for it in json.load(open("/workspace/items.json"))["items"]}
seen, sel = set(), []
for c in torch.load("/workspace/resid_L42.pt"):
    if c["layer"] == 42 and c["id"] not in seen: seen.add(c["id"]); sel.append(c)
tok, dev = p.backend.tokenizer, p.backend.device
J = _load_jacobians(p.method.repo, p.method.filename, "cpu")[42].to(dev)
W = p.backend.unembed.float(); V, d = W.shape
t0 = time.time(); G = torch.zeros(d, d, device=dev, dtype=torch.float64)
for i in range(0, V, 16384):
    D = W[i:i + 16384] @ J; D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-9)
    G += (D.T @ D).double()
ev, Q = torch.linalg.eigh(G); ev, Q = ev.flip(0), Q.flip(1).float()
print(f"basis {time.time()-t0:.0f}s; share of J-direction variance in top k: " + ", ".join(f"{k}:{float(ev[:k].sum()/ev.sum()):.3f}" for k in (16, 64, 256, 1024, 2560)), flush=True)
torch.save({"Q": Q[:, :2560].half().cpu(), "ev": ev.cpu()}, OUT / "jbasis.pt")
g = torch.Generator(device="cpu").manual_seed(0); R, _ = torch.linalg.qr(torch.randn(d, 1024, generator=g)); R = R.to(dev)
def norm(s): return re.sub(r"[^a-z0-9]+", "", s.lower())
def jl(h, k=10):
    p.use("jlens"); p.method.k = k; r = p.method.read(h.cpu(), 42); p.method.k = 10; return r.tokens
cells = []
for c in sel[:5]:
    h = c["h"].to(dev).float(); parts = {"full": h}
    for k in (64, 256, 1024, 2560):
        hj = Q[:, :k] @ (Q[:, :k].T @ h); parts[f"J{k}"] = hj; parts[f"N{k}"] = h - hj
    hr = R @ (R.T @ h); parts["R1024"] = hr; parts["RN1024"] = h - hr
    for cond, v in parts.items():
        cells.append({"id": c["id"], "cond": cond, "h": v.cpu(), "layer": 42, "share": float((v.norm() / h.norm()) ** 2), "tokens": jl(v)})
torch.save(cells, OUT / "resid_sweep.pt")
p.use("olens"); p.method.sampling.k = 2
with open(OUT / "sweep.jsonl", "w") as fh:
    for c in cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|olens".encode())); t = time.time()
        r = p.method.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "share": c["share"], "tokens": c["tokens"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} share {c['share']:.3f} {time.time()-t:.1f}s top {c['tokens'][:5]}", flush=True)
print("SWEEP_DONE", flush=True)
