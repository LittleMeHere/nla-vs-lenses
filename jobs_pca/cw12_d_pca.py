# CW-12 comparison added after the first results (base worker): a PCA subspace of the same size as the J part.
# PCA is fitted on L42 activations at every token position of benchmark prompts (multihop, association, basic readout, multilingual, chain, brew; a family is skipped if it fails to load), centred.
import json, time, zlib
from wsbench import readplan
from wsbench.produce.render import render
OUT = Path("/workspace/out/cw12"); N_ITEMS, K = 30, 1024
tok, dev = p.backend.tokenizer, p.backend.device
vecs = []
for fam, n in (("multihop", 100), ("association", 100), ("basic_readout", 100), ("multilingual", 100), ("chain_intermediates", 60), ("brew_intermediates", 40)):
    try:
        specs = readplan.plan(fam)[:n]
    except Exception as e:
        print("skip", fam, repr(e)[:120], flush=True); continue
    for spec in specs:
        r = render(spec, tok); pos = list(range(1, len(r.ids)))
        vecs.append(p.backend.capture(r.ids, [42], pos)[42])
    print(fam, "prompts", len(specs), "vectors so far", sum(len(v) for v in vecs), flush=True)
X = torch.cat(vecs).float(); g = torch.Generator().manual_seed(0)
if len(X) > 16000: X = X[torch.randperm(len(X), generator=g)[:16000]]
mu = X.mean(0); _, S, Vh = torch.linalg.svd((X - mu).to(dev), full_matrices=False)
P = Vh[:K].T.contiguous(); var = (S ** 2 / (S ** 2).sum()).cpu()
Q = torch.load(OUT / "jbasis.pt")["Q"].float().to(dev)[:, :K]
overlap = float(((Q.T @ P) ** 2).sum() / K)          # 1 = same subspace; K/d = 0.2 for unrelated subspaces
xj = float((((X.to(dev) - mu.to(dev)) @ Q) ** 2).sum() / ((X.to(dev) - mu.to(dev)) ** 2).sum())
print(f"PCA fit on {len(X)} vectors; variance in top {K} PCs {float(var[:K].sum()):.3f}; share of centred variance inside the J part {xj:.3f}; "
      f"overlap J vs PCA subspace {overlap:.3f} (unrelated subspaces: {K / X.shape[1]:.3f})", flush=True)
torch.save({"P": P.half().cpu(), "mu": mu, "var": var, "overlap": overlap, "var_in_J": xj, "n": len(X)}, OUT / "pcabasis.pt")
seen, sel = set(), []
for c in torch.load("/workspace/resid_L42.pt"):
    if c["layer"] == 42 and c["id"] not in seen: seen.add(c["id"]); sel.append(c)
from wsbench.produce.methods import _load_jacobians
p.use("jlens"); J = _load_jacobians(p.method.repo, p.method.filename, "cpu")[42].to(dev); W = p.backend.unembed.float()
den = torch.cat([(W[i:i + 16384] @ J).norm(dim=1) for i in range(0, W.shape[0], 16384)]).clamp_min(1e-9)
def jl(h): return [tok.decode([i]) for i in torch.topk((W @ (J @ h.to(dev).float())) / den, 10).indices.tolist()]
cells = []
for c in sel[:N_ITEMS]:
    h = c["h"].to(dev).float(); hp = P @ (P.T @ h)
    for cond, v in (("P1024", hp), ("PN1024", h - hp)):
        cells.append({"id": c["id"], "cond": cond, "h": v.cpu(), "layer": 42, "share": float((v.norm() / h.norm()) ** 2), "tokens": jl(v)})
torch.save(cells, OUT / "resid_pca.pt")
p.use("olens"); p.method.sampling.k = 2
with open(OUT / "olens_pca.jsonl", "w") as fh:
    for c in cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|olens".encode())); t = time.time()
        r = p.method.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "share": c["share"], "tokens": c["tokens"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"{c['id']} {c['cond']} share {c['share']:.2f} {time.time()-t:.1f}s", flush=True)
print("PCA_OLENS_DONE", flush=True)
