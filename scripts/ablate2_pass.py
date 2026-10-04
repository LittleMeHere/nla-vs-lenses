"""Stronger bridge ablation + swap positive control on the 50 multihop items at L42.
For each item: d_t = J^T W_U[t] is token t's J-Lens direction (J-Lens score of t is unit(d_t).h).
  strong: remove the bridge tokens' directions AND every vocab token whose direction has cosine
          > COS with a bridge direction; then up to 3 rounds of: J-Lens top-50 on the ablated
          vector, add any token string-matching the bridge, remove again.
  swap:   strong-ablate own AND partner item's bridge sets, then add s * unit(d_partner_main),
          where s = unit(d_own_main).h, so J-Lens should now score the partner bridge as it
          scored the own bridge. A reader that follows the swap is reading; one that keeps
          naming its own bridge is recomputing from context.
  rand:   remove as many random-vocab-token directions as the strong set (control).
J-Lens top-10 is saved per condition; oracle lens (4 samples) on all; vectors saved for the NLA.
Usage: uv run python ablate2_pass.py OUT RESID.pt ITEMS.json N"""
import json, os, random, re, sys, time, torch
from pathlib import Path
from wsbench.produce.producer import Producer
from wsbench.produce.methods import _load_jacobians

COS = float(os.environ.get("COS", "0.5"))
SWAP_SCALES = [float(x) for x in os.environ.get("SWAP_SCALES", "1").split(",")]  # try in order until J-Lens shows the partner
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

def norm(s): return re.sub(r"[^a-z0-9]+", "", s.lower())
def tids_for(strings):
    out = set()
    for s in strings:
        for v in (s, " " + s, s.lower(), " " + s.lower(), s.capitalize(), " " + s.capitalize()):
            t = tok.encode(v, add_special_tokens=False)
            if t: out.add(t[0])
    return sorted(out)
def dirs(tids): return (W[tids] @ J)                        # [k, d]
def unit(x): return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
def neighbours(tids):
    B = unit(dirs(tids)); hits = set(tids)
    for i in range(0, V, 16384):
        C = unit(W[i:i + 16384] @ J) @ B.T                     # [chunk, k]
        idx = (C.max(dim=1).values > COS).nonzero().flatten() + i
        hits.update(idx.tolist())
    return sorted(hits)
def project_out(h, tids):
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

rng = random.Random(0)
partner = {c["id"]: sel[(k + 1) % len(sel)]["id"] for k, c in enumerate(sel)}
for k, c in enumerate(sel):   # make sure partner bridges differ
    j = k + 1
    while set(map(norm, items[sel[j % len(sel)]["id"]]["intermediates"])) & set(map(norm, items[c["id"]]["intermediates"])): j += 1
    partner[c["id"]] = sel[j % len(sel)]["id"]
strong_cache = {}
for c in sel:
    strong_cache[c["id"]] = strong_set(c["h"].to(dev).float(), items[c["id"]]["intermediates"])
print(f"strong sets: mean {sum(map(len, strong_cache.values()))/len(sel):.1f} tokens", flush=True)
new_cells, jrows = [], []
for c in sel:
    i, o = c["id"], partner[c["id"]]
    h = c["h"].to(dev).float(); Ti, To = strong_cache[i], strong_cache[o]
    h_strong = project_out(h, Ti)
    u_own = unit(dirs(tids_for(items[i]["intermediates"][:1]))[0]); u_oth = unit(dirs(tids_for(items[o]["intermediates"][:1]))[0])
    s = float(u_own @ h)
    base = project_out(h, sorted(set(Ti) | set(To)))
    oth_t = items[o]["intermediates"]
    for k_scale in SWAP_SCALES:
        h_swap = base + k_scale * abs(s) * u_oth
        if any(matches(t, oth_t) for t in jl_top(h_swap, 10)): break
    h_rand = project_out(h, rng.sample(range(V), len(Ti)))
    for cond, v in (("strong", h_strong), ("swap", h_swap), ("rand", h_rand)):
        v = v.cpu()
        new_cells.append({**c, "h": v, "cond": cond, "partner": o, "n_dirs": len(Ti), "s_own": s, "swap_scale": k_scale})
        jrows.append({"id": i, "cond": cond, "partner": o, "n_dirs": len(Ti), "s_own": s, "swap_scale": k_scale, "tokens": jl_top(v, 10)})
(out / "jlens_abl2.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in jrows))
torch.save(new_cells, out / "resid_abl2.pt"); print("JLENS_DONE", flush=True)
p.use("olens")
with (out / "olens_abl2.jsonl").open("w") as fh:
    for c in new_cells:
        t = time.time(); r = p.method.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "partner": c["partner"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"olens {c['id']} {c['cond']} {time.time()-t:.1f}s", flush=True)
print("ABLATE2_PASS_DONE", flush=True)
