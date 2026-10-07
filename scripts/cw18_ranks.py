"""CW-18: where is each family's content readable? J-Lens rank of the bank's intermediate word, by layer, best over the
read positions. No reader, CPU. Only items whose intermediate is a single token are scored.
Usage: .venv/bin/python scripts/cw18_ranks.py -> runs/cw18/ranks.txt, ranks.json"""
import json, re
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download
R = Path(__file__).parent.parent / "runs"; D = R / "cw18"
W = torch.from_numpy(np.load(R / "jrank/lm_head_fp16.npy")).float(); V = W.shape[0]
obj = torch.load(hf_hub_download("neuronpedia/jacobian-lens", "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"), map_location="cpu", weights_only=False)
jac = obj["J"] if "J" in obj else obj["jacobians"]; nJ = len(jac) if not isinstance(jac, dict) else 10**9
tj = json.load(open(hf_hub_download("Qwen/Qwen3.6-27B", "tokenizer.json"), encoding="utf-8")); dec = [""] * V
for t, i in tj["model"]["vocab"].items():
    if i < V: dec[i] = t.replace("Ġ", " ").replace("Ċ", "\n")
norm = lambda s: re.sub(r"[^a-z0-9]+", "", str(s).lower()); nd = np.array([norm(s) for s in dec])
FAMS = [f for f in ("multihop", "association", "typo", "multilingual", "basic_readout") if (D / f"layers_{f}.pt").exists()]
data = {f: torch.load(D / f"layers_{f}.pt") for f in FAMS}; bank = {f: {i.get("name", i.get("id")): i for i in json.load(open(D / f"items_{f}.json"))["items"]} for f in FAMS}
masks = {f: {r["id"]: torch.from_numpy(np.isin(nd, [norm(w) for w in (bank[f].get(r["id"]) or {}).get("intermediates") or [] if norm(w)])) for r in data[f]} for f in FAMS}
LAYERS = [L for L in sorted(data[FAMS[0]][0]["h"]) if L < nJ]; best = {f: {L: [] for L in LAYERS} for f in FAMS}
for L in LAYERS:
    J = (jac[L] if not isinstance(jac, dict) else (jac[L] if L in jac else jac[str(L)])).float()
    den = torch.cat([(W[i:i + 8192] @ J).norm(dim=1) for i in range(0, V, 8192)]).clamp_min(1e-9); WJ = None
    for f in FAMS:
        for r in data[f]:
            m = masks[f][r["id"]]
            if not m.any(): continue
            S = (r["h"][L].float() @ J.T) @ W.T / den; rk = (S > S[:, m].max(dim=1, keepdim=True).values).sum(1) + 1; best[f][L].append(int(rk.min()))
    print("layer", L, flush=True)
out = {}; Ls = ["family (items scored of items captured; read positions per item) | per layer: median rank of the intermediate word / share in top 10 / share in top 200"]
for f in FAMS:
    n = len(best[f][LAYERS[0]]); out[f] = {L: {"median": float(np.median(v)), "top10": float(np.mean([x <= 10 for x in v])), "top200": float(np.mean([x <= 200 for x in v])), "n": len(v)} for L, v in best[f].items() if v}
    Ls.append(f"{f} ({n} of {len(data[f])}; {sorted({len(r['pos']) for r in data[f]})}) | " + "; ".join(f"L{L}: {int(o['median'])} / {o['top10']:.2f} / {o['top200']:.2f}" for L, o in out[f].items()))
open(D / "ranks.txt", "w").write("\n".join(Ls) + "\n"); json.dump(out, open(D / "ranks.json", "w"), indent=1); print("\n".join(Ls))
