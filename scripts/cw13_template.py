"""CW-13b: the template lens (camilablank/workspace-lenses @ d740106d, Qwen3.6-27B, 13,174 words) on the saved L42
activations. No GPU. Score of word w = t_w(42) . (h - mu), with mu the mean L42 activation over 16,000 benchmark-prompt
positions (runs/cw12/pcabasis.pt); the raw dot product t_w . h is reported as a variant, because the lens's own decode
rule and mean are not in the file. Usage: python scripts/cw13_template.py -> runs/jrank/template_ranks.json, template_analysis.txt"""
import json, re, struct
from pathlib import Path
import numpy as np, torch
R = Path(__file__).parent.parent / "runs"; OUT = R / "jrank"; TL = R / "template_lens/qwen3.6-27b/template-lens"
with open(TL / "templates.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; hd = json.loads(f.read(n)); info = hd["templates"]
    Lr, Wn, d = info["shape"]; f.seek(8 + n + info["data_offsets"][0] + 42 * Wn * d * 2); buf = f.read(Wn * d * 2)
T = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).reshape(Wn, d).float()
words = [l.split("\t", 1)[1].strip() for l in open(TL / "template_words.txt") if "\t" in l]; assert len(words) == Wn
mu = torch.load(R / "cw12/pcabasis.pt")["mu"].float()
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
def norm(s): return re.sub(r"[^a-z0-9]+", "", s.lower())
nw = np.array([norm(w) for w in words])
def rank(h, inter, centre=True):
    s = T @ ((h.float() - mu) if centre else h.float()); order = torch.argsort(s, descending=True); rk = torch.empty(Wn, dtype=torch.long); rk[order] = torch.arange(1, Wn + 1)
    m = np.isin(nw, [norm(x) for x in inter])
    return (int(rk[torch.from_numpy(m)].min()) if m.any() else None), [words[i] for i in order[:10].tolist()]
out = {"whole50": {}, "whole50_raw": {}, "parts30": {}}
for c in torch.load(R / "abl50/resid_L42.pt"):
    if c["layer"] == 42 and c["id"] not in out["whole50"]:
        r, top = rank(c["h"], bank[c["id"]]["intermediates"]); out["whole50"][c["id"]] = {"rank": r, "top10": top}
        out["whole50_raw"][c["id"]] = rank(c["h"], bank[c["id"]]["intermediates"], False)[0]
for f in ("resid_split.pt", "resid_pca.pt"):
    for c in torch.load(R / "cw12" / f):
        # a part is a projection of h; centre it with the same projection of mu so the mean is removed consistently
        out["parts30"].setdefault(c["cond"], {})[c["id"]] = rank(c["h"], bank[c["id"]]["intermediates"], False)[0]
json.dump(out, open(OUT / "template_ranks.json", "w"), indent=1)
L = [f"template lens: {Wn} words, layer 42; items 50"]
for key, lab in (("whole50", "centred"), ("whole50_raw", "raw dot product")):
    rk = np.array([(v["rank"] if isinstance(v, dict) else v) for v in out[key].values() if (v["rank"] if isinstance(v, dict) else v)])
    L.append(f"{lab}: bridge in vocabulary for {len(rk)}/50; median rank {int(np.median(rk))}; within top k: " + ", ".join(f"k={k}: {int((rk <= k).sum())}" for k in (1, 10, 50, 200, 1000)))
L.append("parts (30 items, raw dot product): part | in vocab | median rank | within top 10 / 50 / 200")
for cond, dd in out["parts30"].items():
    r = np.array([v for v in dd.values() if v]); L.append(f"  {cond}: {len(r)} | {int(np.median(r))} | " + " / ".join(str(int((r <= k).sum())) for k in (10, 50, 200)))
ex = out["whole50"]["atomic-26-symbol"]; L.append(f"example atomic-26-symbol (bridge iron): rank {ex['rank']}, top 10 {ex['top10']}")
open(OUT / "template_analysis.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
