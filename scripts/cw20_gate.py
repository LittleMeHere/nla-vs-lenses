"""CW-20 check 2 and 3: scaled whitened FVE of the unedited oracle lens write-ups under three ways of feeding them to
the reconstructor, and the floor (another prompt's write-up). Usage: cw20_gate.py runs/cw20/recon_gate.pt"""
import collections
import sys

import torch

d = torch.load(sys.argv[1])
T = {k: v.double() for k, v in d["targets"].items()}
ids = list(T)
mu = torch.stack([T[i] for i in ids]).mean(0)
print("capture check cosine min %.5f mean %.5f" % (min(d["capture_cos"]), sum(d["capture_cos"]) / len(d["capture_cos"])))
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for k, w in zip(d["keys"], d["white"].double()):
    conv, pid, s = k.split("|")[:3]
    groups[conv][(pid, s)].append(w)


def scaled_fve(P, Y):
    a = float((P * Y).sum() / (P * P).sum())
    return 1 - float(((Y - a * P) ** 2).sum() / ((Y - mu) ** 2).sum()), a


for conv, g in groups.items():
    keys = sorted(g)
    P = torch.stack([torch.stack(g[k]).mean(0) for k in keys])
    Y = torch.stack([T[k[0]] for k in keys])
    f, a = scaled_fve(P, Y)
    cos = torch.nn.functional.cosine_similarity(P - mu * 0, Y, dim=1).mean()
    shift = [ids[(ids.index(k[0]) + 1) % len(ids)] for k in keys]
    Yf = torch.stack([T[i] for i in shift])
    ff, _ = scaled_fve(P, Yf)
    cosf = torch.nn.functional.cosine_similarity(P, Yf, dim=1).mean()
    ret = float((torch.nn.functional.normalize(P, dim=1) @ torch.nn.functional.normalize(torch.stack([T[i] for i in ids]), dim=1).T).argmax(1).eq(torch.tensor([ids.index(k[0]) for k in keys])).double().mean())
    print(f"{conv:16} n {len(keys)}  scaled FVE {f:+.3f} (scale {a:.2f})  cosine {cos:.3f}  retrieval@1 of 70 {ret:.2f}  | other prompt's write-up: FVE {ff:+.3f} cosine {cosf:.3f}")
