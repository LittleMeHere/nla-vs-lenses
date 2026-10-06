"""CW-16b analysis. Usage: python scripts/cw16b_analyze.py RECON.pt [--examples N] -> prints; with --save writes runs/cw16b/analysis.txt
FVE as in CW-16: vectors scaled to sqrt(5120), mean of the 70 targets as baseline. Paired over prompts."""
import json, math, sys
from pathlib import Path
import numpy as np, torch
R = Path(__file__).parent.parent / "runs"; D = R / "cw16b"; SC = math.sqrt(5120)
sc = lambda v: v.float() / v.float().norm() * SC
rec = torch.load(sys.argv[1]); T = {json.loads(l)["key"]: json.loads(l)["text"] for l in open(D / "texts.jsonl")}
G = {c["id"]: sc(c["h"]) for c in torch.load(R / "cw14/resid_split.pt") if c["cond"] == "full"}; mu = torch.stack(list(G.values())).mean(0)
Q = torch.load(R / "cw12/jbasis.pt")["Q"].float()[:, :1024]; pj = lambda v: Q @ (Q.T @ v)
def m(k):
    g = G[k.split("|")[1]]; d = sc(rec[k]) - g; gj = pj(g - mu); dj = pj(d); gr = (g - mu) - gj; dr = d - dj
    return np.array([1 - float(d.pow(2).sum() / (g - mu).pow(2).sum()), 1 - float(dj.pow(2).sum() / gj.pow(2).sum()), 1 - float(dr.pow(2).sum() / gr.pow(2).sum())])
def boot(x):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1); return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
per = {}
for k in rec:
    v, i, s = k.split("|"); per.setdefault(v, {}).setdefault(i, {})[s] = m(k)
def item(v, i):   # mean over the two write-ups (and, for controls, over seeds) of one prompt
    vs = [v] if not v.startswith("rand_") else [f"{v}_{s}" for s in range(3)]
    xs = [per[u][i][s] for u in vs if u in per and i in per[u] for s in per[u][i]]
    return np.mean(xs, axis=0) if xs else None
ids = sorted(per["orig"]); L = [f"prompts: {len(ids)}", "version | FVE | FVE in J part | FVE in rest   (mean over prompts)"]
NAMES = [("orig", "original"), ("corrected", "wrong parts corrected in place"), ("drop_content_wrong", "wrong facts removed"), ("rand_content_wrong", "  control: random spans, same length"),
         ("drop_restate_wrong", "wrong restatements of the prompt removed"), ("rand_restate_wrong", "  control"), ("drop_all_wrong", "everything wrong removed"), ("rand_all_wrong", "  control"),
         ("drop_restate", "all restatements removed"), ("rand_restate", "  control"), ("drop_format", "format statements removed"), ("rand_format", "  control"), ("drop_content", "all content claims removed"), ("rand_content", "  control")]
for v, lab in NAMES:
    xs = [item(v, i) for i in ids]; xs = np.array([x for x in xs if x is not None])
    if len(xs): L.append(f"  {lab} ({len(xs)}): " + " | ".join("%+.3f" % a for a in xs.mean(0)))
L.append("\npaired differences (mean over prompts [95% bootstrap]): FVE | FVE in J part | FVE in rest")
PAIRS = [("corrected", "orig")] + [(f"drop_{g}", f"rand_{g}") for g in ("content_wrong", "restate_wrong", "all_wrong", "restate", "format", "content")]
for a, b in PAIRS:
    c = [(item(a, i), item(b, i)) for i in ids]; c = [(x, y) for x, y in c if x is not None and y is not None]
    if c: L.append(f"  {a} - {b} ({len(c)}): " + " | ".join("%+.3f [%+.3f, %+.3f]" % boot([x[j] - y[j] for x, y in c]) for j in range(3)))
print("\n".join(L))
if "--save" in sys.argv: open(D / "analysis.txt", "w").write("\n".join(L) + "\n")
if "--examples" in sys.argv:
    n = int(sys.argv[sys.argv.index("--examples") + 1]); bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
    for i in ids[:n]:
        print("\n#####", bank[i]["prompt"], "| answer", bank[i]["target"])
        for v in ("orig", "corrected", "drop_content_wrong", "drop_restate_wrong", "drop_restate"):
            k = f"{v}|{i}|0"
            if k in rec: print(f"  [{v} {100*m(k)[0]:+.0f}%] " + T[k].replace("\n", " / ")[:420])
