# CW-14 (base worker; designs/CW-14_split_confirmation.md): capture L42 for all 100 multihop items, check the first 30
# against the saved CW-12 activations, build whole / J / rest / random for items 31-100, save them, run the oracle lens.
import json, time, zlib
from wsbench import readplan
from wsbench.produce.producer import _positions
from wsbench.produce.render import render
from wsbench.produce.methods import _load_jacobians
OUT = Path("/workspace/out/cw14"); OUT.mkdir(parents=True, exist_ok=True); K = 1024
tok, dev = p.backend.tokenizer, p.backend.device
old = {}
for c in torch.load("/workspace/resid_L42.pt"):
    if c["layer"] == 42 and c["id"] not in old: old[c["id"]] = c["h"].float()
used = list(old)[:30]
caps = []
for spec in readplan.plan("multihop")[:100]:
    r = render(spec, tok); pos = _positions(spec.positions, r.tokens); idx = [q % len(r) for q in pos]
    assert len(idx) == 1, (spec, pos)
    h = p.backend.capture(r.ids, [42], idx)[42][0].float().cpu()
    caps.append({"id": spec.id, "pos": pos[0], "token": r.decoded[idx[0]], "h": h})
cs = [float(torch.nn.functional.cosine_similarity(c["h"], old[c["id"]], dim=0)) for c in caps if c["id"] in old]
print(f"capture check: {len(cs)} items also in the saved file, cosine min {min(cs):.5f} mean {sum(cs)/len(cs):.5f}", flush=True)
assert min(cs) > 0.999, "capture does not reproduce the saved activations"
assert [c["id"] for c in caps[:30]] == used, "first 30 of the plan are not the CW-12 items"
sel = caps[30:]; print("fresh items", len(sel), sel[0]["id"], sel[-1]["id"], flush=True)
Q = torch.load("/workspace/jbasis.pt")["Q"].float().to(dev)[:, :K]
g = torch.Generator(device="cpu").manual_seed(0); R, _ = torch.linalg.qr(torch.randn(Q.shape[0], K, generator=g)); R = R.to(dev)
p.use("jlens")
J = _load_jacobians(p.method.repo, p.method.filename, "cpu")[42].to(dev); W = p.backend.unembed.float()
den = torch.cat([(W[i:i + 16384] @ J).norm(dim=1) for i in range(0, W.shape[0], 16384)]).clamp_min(1e-9)
def jl(h):
    s = (W @ (J @ h.to(dev).float())) / den
    return [tok.decode([i]) for i in torch.topk(s, 10).indices.tolist()]
cells = []
for c in sel:
    h = c["h"].to(dev); hj = Q @ (Q.T @ h); hr = R @ (R.T @ h)
    for cond, v in (("full", h), ("J1024", hj), ("N1024", h - hj), ("R1024", hr)):
        cells.append({"id": c["id"], "cond": cond, "h": v.cpu(), "layer": 42, "pos": c["pos"], "token": c["token"],
                      "share": float((v.norm() / h.norm()) ** 2), "tokens": jl(v)})
torch.save(cells, OUT / "resid_split.tmp"); (OUT / "resid_split.tmp").rename(OUT / "resid_split.pt")
print("CELLS_SAVED", len(cells), flush=True)
p.use("olens"); p.method.sampling.k = 2
with open(OUT / "olens_split.jsonl", "w") as fh:
    for c in cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|olens".encode())); t = time.time()
        r = p.method.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "share": c["share"], "tokens": c["tokens"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} {time.time()-t:.1f}s", flush=True)
print("SPLIT_OLENS_DONE", flush=True)
