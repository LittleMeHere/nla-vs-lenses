"""CW-12 side check, no GPU: is prompt information linearly decodable from the J part and from the rest of the L42
activation? 50 multihop items. Labels: answer is a number; prompt contains a given frequent word. Leave-one-out
logistic regression on each part. Usage: python scripts/cw12_probe.py -> runs/cw12/probe.txt"""
import json, re, collections
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
R = Path(__file__).parent.parent / "runs"
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
cells = [c for c in torch.load(R / "abl50/resid_L42.pt") if c["layer"] == 42]
ids = [c["id"] for c in cells]; H = torch.stack([c["h"].float() for c in cells])
Q = torch.load(R / "pod_sync/jnkgruewkya74v/cw12/jbasis.pt")["Q"].float()[:, :1024]
g = torch.Generator().manual_seed(0); Rn, _ = torch.linalg.qr(torch.randn(H.shape[1], 1024, generator=g))
parts = {"whole": H, "J part (top 1024)": H @ Q, "rest": H - (H @ Q) @ Q.T, "random 1024": H @ Rn, "all but random": H - (H @ Rn) @ Rn.T}
STOP = set("the a an of to in on at for and or is are was were be by with as that this it its from into about fact what which who whose".split())
words = collections.Counter(w for i in ids for w in set(re.findall(r"[a-z]+", bank[i]["prompt"].lower())) - STOP)
labels = {"answer is a number": np.array([bank[i]["target"].strip().isdigit() for i in ids])}
for w, n in words.most_common():
    if 10 <= n <= 40 and len(w) > 2: labels[f'prompt has "{w}"'] = np.array([w in re.findall(r"[a-z]+", bank[i]["prompt"].lower()) for i in ids])
def loo_auc(X, y, seed=None):
    X = X.numpy(); X = (X - X.mean(0)) / (X.std() + 1e-9); y = y.copy()
    if seed is not None: np.random.default_rng(seed).shuffle(y)
    s = np.zeros(len(y))
    for k in range(len(y)):
        m = np.arange(len(y)) != k
        s[k] = LogisticRegression(C=0.05, max_iter=2000).fit(X[m], y[m]).decision_function(X[k:k + 1])[0]
    return roc_auc_score(y, s)
L = [f"items {len(ids)}; share of squared norm: " + ", ".join(f"{k} {float((v.norm(dim=1)**2 / (H.norm(dim=1)**2)).mean()) if v.shape[1] == H.shape[1] else float(((v**2).sum(1) / (H**2).sum(1)).mean()):.2f}" for k, v in parts.items())]
L.append("label (positives of 50) | " + " | ".join(parts) + " | shuffled labels, whole (mean of 5)")
tot = {k: [] for k in parts}
for name, y in labels.items():
    a = {k: loo_auc(v, y) for k, v in parts.items()}; sh = np.mean([loo_auc(H, y, s) for s in range(5)])
    for k in parts: tot[k].append(a[k])
    L.append(f"{name} ({int(y.sum())}) | " + " | ".join(f"{a[k]:.2f}" for k in parts) + f" | {sh:.2f}")
L.append("mean over labels | " + " | ".join(f"{np.mean(tot[k]):.2f}" for k in parts))
open(R / "cw12/probe.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
