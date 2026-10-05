"""CW-11 pass A (designs/CW-11_cue_removal.md): build intact / bridge / cue / both / rand versions of each item's L42
activation, save J-Lens top-10 and the bridge's J-Lens score per condition, run the oracle lens (K samples), save the
vectors for the NLA pass. Adapted from ablate2_pass.py.
Usage: uv run python cw11_pass_a.py OUT RESID.pt ITEMS.json N"""
import json, os, random, re, sys, time, zlib, torch
from pathlib import Path
from wsbench.produce.producer import Producer
from wsbench.produce.methods import _load_jacobians

COS = float(os.environ.get("COS", "0.8")); K = int(os.environ.get("K", "4"))
out, resid, items_path, n = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
items = {it["name"]: it for it in json.load(open(items_path))["items"]}
seen, sel = set(), []
for c in torch.load(resid):
    if c["layer"] == 42 and c["id"] not in seen: seen.add(c["id"]); sel.append(c)
sel = sel[:n]
p = Producer.load("Qwen/Qwen3.6-27B", "jlens")
tok, dev = p.backend.tokenizer, p.backend.device
J = _load_jacobians(p.method.repo, p.method.filename, "cpu")[42].to(dev)
W = p.backend.unembed.float(); V = W.shape[0]
STOP = set("the a an of to in on at for and or is are was were be by with as that this it its from into about fact "
           "what which who whose".split())

def norm(s): return re.sub(r"[^a-z0-9]+", "", s.lower())
def tids_for(strings):
    o = set()
    for s in strings:
        for v in (s, " " + s, s.lower(), " " + s.lower(), s.capitalize(), " " + s.capitalize()):
            t = tok.encode(v, add_special_tokens=False)
            if t: o.add(t[0])
    return sorted(o)
def dirs(tids): return (W[tids] @ J)
def unit(x): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
def neighbours(tids):
    B = unit(dirs(tids)); hits = set(tids)
    for i in range(0, V, 16384):
        C = unit(W[i:i + 16384] @ J) @ B.T
        hits.update(((C.max(dim=1).values > COS).nonzero().flatten() + i).tolist())
    return sorted(hits)
def project_out(h, tids):
    if not tids: return h.clone()
    Q, _ = torch.linalg.qr(dirs(tids).T)
    return h - Q @ (Q.T @ h)
def jl_top(h, k):
    p.method.k = k; r = p.method.read(h.cpu(), 42); p.method.k = 10; return r.tokens
def matches(t, inters):
    a = norm(t)
    return any(a and (a == norm(x) or (len(a) >= 3 and (a in norm(x) or norm(x).startswith(a)))) for x in inters)
def strong_set(h, inters):
    T = neighbours(tids_for(inters))
    for _ in range(3):
        extra = [tok.encode(t, add_special_tokens=False)[0] for t in jl_top(project_out(h, T), 50)
                 if matches(t, inters) and tok.encode(t, add_special_tokens=False)]
        new = sorted(set(extra) - set(T))
        if not new: break
        T = sorted(set(T) | set(neighbours(new)))
    return T
def cue_words(it):
    drop = set(re.findall(r"[a-z0-9]+", " ".join(it["intermediates"] + [it["target"]]).lower()))
    return sorted({w for w in re.findall(r"[a-z0-9]+", it["prompt"].lower()) if w not in STOP and w not in drop and len(w) > 1})

rng = random.Random(0)
new_cells, jrows = [], []
for c in sel:
    i = c["id"]; it = items[i]; h = c["h"].to(dev).float()
    Tb = strong_set(h, it["intermediates"])
    cw = cue_words(it)
    Tc = sorted(set(neighbours(tids_for(cw))) - set(Tb)) if cw else []
    u_b = unit(dirs(tids_for(it["intermediates"][:1]))[0])
    vs = {"intact": h, "bridge": project_out(h, Tb), "cue": project_out(h, Tc),
          "both": project_out(h, sorted(set(Tb) | set(Tc))), "rand": project_out(h, rng.sample(range(V), max(len(Tc), 1)))}
    for cond, v in vs.items():
        top50 = jl_top(v, 50)
        meta = {"id": i, "cond": cond, "n_bridge_dirs": len(Tb), "n_cue_dirs": len(Tc), "cue_words": cw,
                "bridge_score": float(u_b @ v), "norm_ratio": float(v.norm() / h.norm()),
                "cue_in_top50": sum(any(norm(t) == norm(w) for w in cw) for t in top50)}
        new_cells.append({**c, "h": v.cpu(), **{k: meta[k] for k in ("cond", "n_bridge_dirs", "n_cue_dirs")}})
        jrows.append({**meta, "tokens": top50[:10]})
    print(f"built {i}: bridge dirs {len(Tb)}, cue dirs {len(Tc)}, cue words {cw}", flush=True)
(out / "jlens_cw11.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in jrows))
torch.save(new_cells, out / "resid_cw11.pt"); print("JLENS_DONE", flush=True)
p.use("olens")
if hasattr(p.method, "sampling"): p.method.sampling.k = K
print("olens sampling", getattr(p.method, "sampling", None), flush=True)
with (out / "olens_cw11.jsonl").open("w") as fh:
    for c in new_cells:
        torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|olens".encode()))
        t = time.time(); r = p.method.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"olens {c['id']} {c['cond']} n={len(r.samples)} {time.time()-t:.1f}s", flush=True)
print("CW11_PASS_A_DONE", flush=True)
