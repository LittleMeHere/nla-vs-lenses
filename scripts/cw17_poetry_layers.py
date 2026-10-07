"""CW-17 follow-up: J-Lens rank of the rhyme word by layer, at the newline ending line one and at the last prompt token.
CPU. Usage: .venv/bin/python scripts/cw17_poetry_layers.py -> runs/cw17/poetry_layers.txt"""
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
bank = {i["name"]: i for i in json.load(open(D / "poetry_items.json"))["items"]}
rows = torch.load(D / "resid_poetry_layers.pt"); LAYERS = sorted(rows[0]["h"]); out = {0: {}, 1: {}}; skipped = 0
for L in LAYERS:
    if L >= (len(jac) if not isinstance(jac, dict) else 10**9): continue
    J = (jac[L] if not isinstance(jac, dict) else (jac[L] if L in jac else jac[str(L)])).float()
    den = torch.cat([(W[i:i + 8192] @ J).norm(dim=1) for i in range(0, V, 8192)]).clamp_min(1e-9)
    for r in rows:
        mask = torch.from_numpy(np.isin(nd, [norm(w) for w in bank[r["id"]]["intermediates"]]))
        if not mask.any(): skipped += (L == LAYERS[0]); continue
        S = (r["h"][L].float() @ J.T) @ W.T / den; rk = (S > S[:, mask].max(dim=1, keepdim=True).values).sum(1) + 1
        for k in (0, 1): out[k].setdefault(L, []).append(int(rk[k]))
Ls = ["position | per layer: share of prompts with the rhyme word in J-Lens top 10 / median rank   (%d prompts; %d skipped, word not a single token)" % (len(rows) - skipped, skipped)]
for k, lab in ((0, "newline ending line one (the bank's read position)"), (1, "last prompt token")):
    Ls.append(lab + " | " + "; ".join(f"L{L}: {np.mean([x <= 10 for x in out[k][L]]):.2f} / {int(np.median(out[k][L]))}" for L in sorted(out[k])))
open(D / "poetry_layers.txt", "w").write("\n".join(Ls) + "\n"); print("\n".join(Ls))
