"""Bridge ablation. For each saved L42 residual, remove the directions that carry the bank's bridge
tokens under the J-Lens linear map (d_t = J^T W_U[t]), so J-Lens no longer shows the bridge, and
also a control that removes as many random-token directions. Run J-Lens and the oracle lens on
both; save the vectors for the NLA pass. Usage: uv run python ablate_pass.py OUT RESID.pt N"""
import json, os, random, sys, time, torch
from pathlib import Path
from wsbench.produce.producer import Producer
from wsbench.produce.methods import _load_jacobians

out, resid, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
items = {it["name"]: it for it in json.load(open("/workspace/items.json"))["items"]}
cells = [c for c in torch.load(resid) if c["layer"] == 42]
ids_seen, sel = set(), []
for c in cells:
    if c["id"] not in ids_seen: ids_seen.add(c["id"]); sel.append(c)
sel = sel[:n]
p = Producer.load("Qwen/Qwen3.6-27B", "jlens")
tok, dev = p.backend.tokenizer, p.backend.device
J = _load_jacobians(p.method.repo, p.method.filename, "cpu")[42].to(dev)      # [d, d] fp32
W = p.backend.unembed.float()                                                  # [V, d]
V = W.shape[0]
def token_ids(strings):
    out = []
    for s in strings:
        for v in (s, " " + s, s.lower(), " " + s.lower(), s.capitalize(), " " + s.capitalize()):
            t = tok.encode(v, add_special_tokens=False)
            if t: out.append(t[0])
    return sorted(set(out))
def project_out(h, tids):
    D = (W[tids] @ J).T                                                        # [d, k]: d_t = J^T W_U[t]
    Q, _ = torch.linalg.qr(D)
    return h - Q @ (Q.T @ h)
rng = random.Random(0)
new_cells, rows = [], {"abl": [], "ctrl": []}
for c in sel:
    it = items[c["id"]]; tids = token_ids(it["intermediates"])
    h = c["h"].to(dev).float()
    rand = rng.sample(range(V), len(tids))
    for cond, t in (("abl", tids), ("ctrl", rand)):
        h2 = project_out(h, t).cpu()
        new_cells.append({**c, "h": h2, "cond": cond, "n_dirs": len(t)})
        p.use("jlens"); r = p.method.read(h2, 42)
        rows[cond].append({"id": c["id"], "layer": 42, "pos": c["pos"], "token": c["token"], "cond": cond,
                           "n_dirs": len(t), "tokens": r.tokens, "scores": r.scores})
for cond in rows:
    (out / f"jlens_{cond}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[cond]))
torch.save(new_cells, out / "resid_abl.pt")
print(f"ablated {len(sel)} items; mean dirs {sum(c['n_dirs'] for c in new_cells)/len(new_cells):.1f}; JLENS_DONE", flush=True)
p.use("olens")
with (out / "olens_ablctrl.jsonl").open("w") as fh:
    for c in new_cells:
        t = time.time(); r = p.method.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "layer": 42, "pos": c["pos"], "token": c["token"], "cond": c["cond"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
        print(f"olens {c['id']} {c['cond']} {time.time()-t:.1f}s", flush=True)
print("ABLATE_PASS_DONE", flush=True)
