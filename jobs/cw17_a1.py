# CW-17 (base worker, pod O1): L44 J basis; capture poetry at L42/L44 and build parts; capture directed modulation by layer;
# then the oracle lens (L44) on the first 50 poetry prompts.
import json, time, zlib
from wsbench import readplan
from wsbench.produce.producer import _positions
from wsbench.produce.render import render
from wsbench.produce.methods import _load_jacobians
OUT = Path("/workspace/out/cw17"); OUT.mkdir(parents=True, exist_ok=True); K = 1024
tok, dev = p.backend.tokenizer, p.backend.device
p.use("jlens"); JAC = _load_jacobians(p.method.repo, p.method.filename, "cpu"); W = p.backend.unembed.float()
def basis(L):
    J = JAC[L].to(dev).float(); G = torch.zeros(5120, 5120, device=dev, dtype=torch.float64)
    for i in range(0, W.shape[0], 16384):
        D = W[i:i + 16384] @ J; D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-9); G += (D.T @ D).double()
    ev, Q = torch.linalg.eigh(G); return Q.flip(1).float()[:, :2560], ev.flip(0)
Q44, ev44 = basis(44); torch.save({"Q": Q44.half().cpu(), "ev": ev44.cpu()}, OUT / "jbasis_L44.pt")
Q42 = torch.load("/workspace/jbasis.pt")["Q"].float().to(dev)
chk, _ = basis(42); ov = float(((Q42[:, :K].T @ chk[:, :K]) ** 2).sum() / K); print(f"L42 basis rebuilt here vs saved: overlap {ov:.4f}; L44 top-1024 share {float(ev44[:K].sum()/ev44.sum()):.3f}; L42-L44 overlap {float(((Q42[:, :K].T @ Q44[:, :K]) ** 2).sum()/K):.3f}", flush=True)
assert ov > 0.99
g = torch.Generator(device="cpu").manual_seed(0); Rr, _ = torch.linalg.qr(torch.randn(5120, K, generator=g)); Rr = Rr.to(dev)
den = {L: torch.cat([(W[i:i + 16384] @ JAC[L].to(dev).float()).norm(dim=1) for i in range(0, W.shape[0], 16384)]).clamp_min(1e-9) for L in (42, 44)}
def jl(h, L): return [tok.decode([i]) for i in torch.topk((W @ (JAC[L].to(dev).float() @ h.to(dev).float())) / den[L], 10).indices.tolist()]
cells = {42: [], 44: []}
for spec in readplan.plan("poetry")[:100]:
    r = render(spec, tok); pos = _positions(spec.positions, r.tokens); idx = [q % len(r) for q in pos]; assert len(idx) == 1
    acts = p.backend.capture(r.ids, [42, 44], idx)
    for L, Q in ((42, Q42[:, :K]), (44, Q44[:, :K])):
        h = acts[L][0].float().to(dev); hj = Q @ (Q.T @ h); hr = Rr @ (Rr.T @ h)
        for cond, v in (("full", h), ("J1024", hj), ("N1024", h - hj), ("R1024", hr)):
            cells[L].append({"id": spec.id, "cond": cond, "h": v.cpu(), "layer": L, "pos": pos[0], "token": r.decoded[idx[0]], "share": float((v.norm() / h.norm()) ** 2), "tokens": jl(v, L)})
for L in (42, 44): torch.save(cells[L], OUT / f"resid_poetry_L{L}.tmp"); (OUT / f"resid_poetry_L{L}.tmp").rename(OUT / f"resid_poetry_L{L}.pt")
print("POETRY_CELLS_SAVED", len(cells[42]), len(cells[44]), flush=True)
LAYERS = list(range(20, 61, 4)) + [42]; dm = []
for spec in readplan.plan("directed_modulation")[:100]:
    try:
        r = render(spec, tok); pos = _positions(spec.positions, r.tokens); idx = [q % len(r) for q in pos]
        acts = p.backend.capture(r.ids, LAYERS, idx)
        dm.append({"id": spec.id, "pos": pos, "tokens": [r.decoded[j] for j in idx], "h": {L: acts[L].half() for L in LAYERS}})
    except Exception as e: print("dm skip", getattr(spec, "id", "?"), repr(e)[:150], flush=True)
torch.save(dm, OUT / "resid_dm_layers.pt"); print("DM_SAVED", len(dm), flush=True)
p.use("olens"); p.method.sampling.k = 2
ids = []; [ids.append(c["id"]) for c in cells[44] if c["id"] not in ids]; mine = set(ids[:50])
with open(OUT / "olens_poetry_a.jsonl", "w") as fh:
    for c in cells[44]:
        if c["id"] not in mine: continue
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|olens".encode())); t = time.time(); r = p.method.read(c["h"], 44)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "layer": 44, "share": c["share"], "tokens": c["tokens"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} {time.time()-t:.1f}s", flush=True)
print("OLENS_A_DONE", flush=True)
