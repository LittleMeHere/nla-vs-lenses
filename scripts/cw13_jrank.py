"""CW-13: where does the bridge rank in J-Lens's full list? No GPU. Uses the saved L42 activations (50 multihop items),
the neuronpedia J-Lens map and the model's output-word matrix (read from the Hub by byte range, cached locally).
J-Lens score of token t on activation h, as in the benchmark's cosine readout: (W_U[t] . (J h)) / ||W_U[t] J||.
Usage: python scripts/cw13_jrank.py  -> runs/jrank/ranks.json, analysis.txt"""
import json, re, struct, time
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download, HfFileSystem
R = Path(__file__).parent.parent / "runs"; OUT = R / "jrank"; OUT.mkdir(exist_ok=True)
t0 = time.time()
wp = OUT / "lm_head_fp16.npy"
if not wp.exists():
    fs = HfFileSystem()
    with fs.open("Qwen/Qwen3.6-27B/model-00008-of-00015.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hd = json.loads(f.read(n)); info = hd["lm_head.weight"]
        a, b = info["data_offsets"]; f.seek(8 + n + a); buf = bytearray()
        while len(buf) < b - a: buf += f.read(min(1 << 26, b - a - len(buf)))
    dt = {"BF16": None, "F16": np.float16, "F32": np.float32}[info["dtype"]]
    if dt is None: W = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(info["shape"]).float().half().numpy()
    else: W = np.frombuffer(buf, dtype=dt).reshape(info["shape"]).astype(np.float16)
    np.save(wp, W); print("lm_head", info["dtype"], info["shape"], round(time.time() - t0), "s", flush=True)
W = torch.from_numpy(np.load(wp)).float(); V = W.shape[0]
obj = torch.load(hf_hub_download("neuronpedia/jacobian-lens", "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"), map_location="cpu", weights_only=False)
jac = obj["J"] if "J" in obj else obj["jacobians"]
J = (jac[42] if not isinstance(jac, dict) else (jac[42] if 42 in jac else jac["42"])).float(); del obj, jac
print("J", tuple(J.shape), "W", tuple(W.shape), round(time.time() - t0), "s", flush=True)
den = torch.cat([(W[i:i + 8192] @ J).norm(dim=1) for i in range(0, V, 8192)]).clamp_min(1e-9)
# token texts straight from tokenizer.json (byte-level BPE: "Ġ" marks a leading space); enough for matching ASCII words
tj = json.load(open(hf_hub_download("Qwen/Qwen3.6-27B", "tokenizer.json"), encoding="utf-8"))
dec = [""] * V
for t, i in tj["model"]["vocab"].items():
    if i < V: dec[i] = t.replace("Ġ", " ").replace("Ċ", "\n")
for a in tj.get("added_tokens", []):
    if a["id"] < V: dec[a["id"]] = a["content"]
def norm(s): return re.sub(r"[^a-z0-9]+", "", s.lower())
nd = np.array([norm(s) for s in dec])
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
def ranks_for(h, inter):
    s = (W @ (J @ h.float())) / den; order = torch.argsort(s, descending=True); rk = torch.empty(V, dtype=torch.long); rk[order] = torch.arange(1, V + 1)
    exact = np.isin(nd, [norm(x) for x in inter])
    return (int(rk[torch.from_numpy(exact)].min()) if exact.any() else None), [dec[i] for i in order[:10].tolist()]
out = {"whole50": {}, "parts30": {}}
for c in torch.load(R / "abl50/resid_L42.pt"):
    if c["layer"] == 42 and c["id"] not in out["whole50"]:
        r, top = ranks_for(c["h"], bank[c["id"]]["intermediates"]); out["whole50"][c["id"]] = {"rank": r, "top10": top}
for f in ("resid_split.pt", "resid_pca.pt"):
    for c in torch.load(R / "cw12" / f):
        r, _ = ranks_for(c["h"], bank[c["id"]]["intermediates"]); out["parts30"].setdefault(c["cond"], {})[c["id"]] = r
json.dump(out, open(OUT / "ranks.json", "w"), indent=1)
L = [f"vocabulary {V}; items {len(out['whole50'])}; bridge = any token whose text equals a bank intermediate (letters and digits only, case-insensitive)"]
rk = np.array([v["rank"] for v in out["whole50"].values() if v["rank"]]); n = len(out["whole50"])
L.append(f"bridge has a matching token in {len(rk)}/{n} items; median rank {int(np.median(rk))}")
L.append("bridge within J-Lens's top k (of all %d items): " % n + ", ".join(f"k={k}: {int((rk <= k).sum())}" for k in (1, 10, 50, 200, 1000, 10000)))
L.append("\nparts (30 items): part | median rank | within top 10 / 50 / 200 / 1000")
for cond, d in out["parts30"].items():
    r = np.array([v for v in d.values() if v]); L.append(f"  {cond}: {int(np.median(r))} | " + " / ".join(str(int((r <= k).sum())) for k in (10, 50, 200, 1000)) + f" (of {len(d)})")
open(OUT / "analysis.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("done", round(time.time() - t0), "s")
