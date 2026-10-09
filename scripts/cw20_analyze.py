"""CW-20: rebuilt variance for each version of the oracle lens write-ups, and paired differences against the
length-matched random removals. Predictions are averaged over a write-up's bullets. Main score: whitened FVE around
the corpus mean, with one global scale fitted on the unedited write-ups and held fixed; mean cosine beside it.
Usage: cw20_analyze.py runs/cw20/recon.pt"""
import collections
import sys

import numpy as np
import torch

d = torch.load(sys.argv[1])
T = {k: v.double() for k, v in d["targets"].items()}
empty = set(d.get("empty", []))
acc = collections.defaultdict(list)
for i, (k, w) in enumerate(zip(d["keys"], d["white"].double())):
    if i in empty:
        continue
    v, pid, s = k.split("|")[:3]
    acc[(v, pid, s)].append(w)
P = {k: torch.stack(v).mean(0) for k, v in acc.items()}
orig = [k for k in P if k[0] == "orig"]
a = float(sum((P[k] * T[k[1]]).sum() for k in orig) / sum((P[k] * P[k]).sum() for k in orig))
fve = {k: 1 - float(((T[k[1]] - a * p) ** 2).sum() / (T[k[1]] ** 2).sum()) for k, p in P.items()}
cos = {k: float(torch.nn.functional.cosine_similarity(p, T[k[1]], dim=0)) for k, p in P.items()}
per = {m: collections.defaultdict(lambda: collections.defaultdict(list)) for m in ("fve", "cos")}
for k in P:
    per["fve"][k[0]][k[1]].append(fve[k])
    per["cos"][k[0]][k[1]].append(cos[k])
ids = sorted(per["fve"]["orig"])


def item(m, v, i):
    vs = [v] if not v.startswith("rand_") else [f"{v}_{s}" for s in range(3)]
    xs = [x for u in vs if i in per[m][u] for x in per[m][u][i]]
    return np.mean(xs) if xs else None


def mean(m, v):
    xs = [item(m, v, i) for i in ids]
    xs = [x for x in xs if x is not None]
    return np.mean(xs), len(xs)


def diff(m, x, y):
    dd = np.array([item(m, x, i) - item(m, y, i) for i in ids if item(m, x, i) is not None and item(m, y, i) is not None])
    bs = dd[np.random.default_rng(0).integers(0, len(dd), (10000, len(dd)))].mean(1)
    return dd.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(dd)


print(f"prompts: {len(ids)}; global scale fitted on the unedited write-ups: {a:.3f}")
print("version | scaled FVE | cosine   (mean over prompts)")
for v, lab in [("orig", "original"), ("corrected", "wrong parts corrected in place")] + [x for g in ("answer", "wrong", "about", "invented", "format") for x in ((f"drop_{g}", f"{g} removed"), (f"rand_{g}", "  control: random removal, same length"))]:
    f, n = mean("fve", v)
    c, _ = mean("cos", v)
    print(f"  {lab} ({n}): {f:+.3f} | {c:.3f}")
print("\npaired differences (mean over prompts [95% bootstrap]): scaled FVE | cosine")
for x, y in [("corrected", "orig")] + [(f"drop_{g}", f"rand_{g}") for g in ("answer", "wrong", "about", "invented", "format")]:
    f = diff("fve", x, y)
    c = diff("cos", x, y)
    print(f"  {x} - {y} ({f[3]}): {f[0]:+.3f} [{f[1]:+.3f}, {f[2]:+.3f}] | {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]")
