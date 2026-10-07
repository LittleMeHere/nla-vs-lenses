"""CW-17 layer check for directed modulation: J-Lens rank of the concept word at the writing positions, by layer and by
instruction. No reader, CPU. Usage: .venv/bin/python scripts/cw17_dm_layers.py -> runs/cw17/dm_layers.txt, dm_layers.json"""
import json, re
from pathlib import Path
import numpy as np, torch
from huggingface_hub import hf_hub_download
R = Path(__file__).parent.parent / "runs"; D = R / "cw17"
W = torch.from_numpy(np.load(R / "jrank/lm_head_fp16.npy")).float(); V = W.shape[0]
obj = torch.load(hf_hub_download("neuronpedia/jacobian-lens", "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"), map_location="cpu", weights_only=False)
jac = obj["J"] if "J" in obj else obj["jacobians"]
tj = json.load(open(hf_hub_download("Qwen/Qwen3.6-27B", "tokenizer.json"), encoding="utf-8")); dec = [""] * V
for t, i in tj["model"]["vocab"].items():
    if i < V: dec[i] = t.replace("Ġ", " ").replace("Ċ", "\n")
norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower()); nd = np.array([norm(s) for s in dec])
items = {i["name"]: i for i in json.load(open(D / "dm_items.json"))["items"]}
dm = torch.load(D / "resid_dm_layers.pt"); LAYERS = sorted(dm[0]["h"])
res = {}; skipped = 0
for L in LAYERS:
    J = (jac[L] if not isinstance(jac, dict) else (jac[L] if L in jac else jac[str(L)])).float()
    den = torch.cat([(W[i:i + 8192] @ J).norm(dim=1) for i in range(0, V, 8192)]).clamp_min(1e-9)
    for d in dm:
        it = items[d["id"]]; mask = torch.from_numpy(nd == norm(it["concept"]))
        if not mask.any():
            skipped += (L == LAYERS[0]); continue
        H = d["h"][L].float(); S = (H @ J.T) @ W.T / den            # [positions, V]
        rk = (S > S[:, mask].max(dim=1, keepdim=True).values).sum(1) + 1   # best rank of any token spelling the concept
        res.setdefault(it["polarity"], {}).setdefault(L, []).append(rk.tolist())
    print("layer", L, "done", flush=True)
out = {}; Ls = ["instruction | prompts | per layer: share of prompts with the concept in J-Lens top 10 at any writing position / median over prompts of the best rank"]
for pol, byL in res.items():
    out[pol] = {}; row = []
    for L in LAYERS:
        best = [min(r) for r in byL[L]]; out[pol][L] = {"top10_any": float(np.mean([b <= 10 for b in best])), "median_best_rank": float(np.median(best)), "n": len(best)}
        row.append(f"L{L}: {out[pol][L]['top10_any']:.2f} / {int(out[pol][L]['median_best_rank'])}")
    Ls.append(f"{pol} | {len(byL[LAYERS[0]])} | " + "; ".join(row))
Ls.append(f"prompts skipped because the concept is not a single token: {skipped}")
open(D / "dm_layers.txt", "w").write("\n".join(Ls) + "\n"); json.dump(out, open(D / "dm_layers.json", "w"), indent=1); print("\n".join(Ls))
