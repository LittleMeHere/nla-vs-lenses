"""CW-12 analysis (designs/CW-12_j_split.md). Usage: python scripts/cw12_analyze.py -> runs/cw12/analysis.txt"""
import json, re
from pathlib import Path
import numpy as np
R = Path(__file__).parent.parent / "runs"; D = R / "cw12"
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
STOP = set("the a an of to in on at for and or is are was were be by with as that this it its from into about fact what which who whose".split())
def names(t, inter): return any(re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", t.lower()) for w in inter)
def cw(it):
    drop = set(re.findall(r"[a-z0-9]+", " ".join(it["intermediates"] + [it["target"]]).lower()))
    return {w for w in re.findall(r"[a-z0-9]+", it["prompt"].lower()) if w not in STOP and w not in drop and len(w) > 1}
def rec(t, c): return len(c & set(re.findall(r"[a-z0-9]+", t.lower()))) / len(c)
def boot(x):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
CONDS = ["full", "J1024", "N1024", "R1024", "RN1024", "P1024", "PN1024"]
L = [open(D / "pca_job.log").read().split("PCA fit")[1].splitlines()[0].join(["PCA fit", ""])] if (D / "pca_job.log").exists() else []
S = {}
for rd, fs in (("oracle", ["olens_split", "olens_pca"]), ("NLA", ["nla_split", "nla_pca"])):
    rows = [json.loads(l) for f in fs for l in open(D / f"{f}.jsonl")]
    L.append(f"\n{rd.upper()} (items x samples: {len({r['id'] for r in rows})} x {len(rows[0]['samples'])}): part | share of squared norm | J-Lens top-10 names bridge | samples naming bridge [95%] | prompt recovery [95%]")
    for c in CONDS:
        rs = [r for r in rows if r["cond"] == c]
        nm = {r["id"]: np.mean([names(s, bank[r["id"]]["intermediates"]) for s in r["samples"]]) for r in rs}
        rc = {r["id"]: np.mean([rec(s, cw(bank[r["id"]])) for s in r["samples"]]) for r in rs}
        S[(rd, c)] = (nm, rc); a, b = boot(list(nm.values())), boot(list(rc.values()))
        jb = sum(names(" ".join(r["tokens"]), bank[r["id"]]["intermediates"]) for r in rs) if "tokens" in rs[0] else None
        L.append(f"  {c}: {np.mean([r['share'] for r in rs]):.2f} | {jb if jb is not None else '-'}/{len(rs)} | {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] | {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}]")
    L.append("  paired differences in naming the bridge, mean over items [95% bootstrap]:")
    for x, y in (("J1024", "full"), ("J1024", "N1024"), ("N1024", "RN1024"), ("J1024", "R1024"), ("R1024", "RN1024"), ("P1024", "PN1024"), ("J1024", "P1024"), ("N1024", "PN1024")):
        A, B = S[(rd, x)][0], S[(rd, y)][0]; m, lo, hi = boot([A[i] - B[i] for i in A])
        L.append(f"    {x} - {y}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
open(D / "analysis.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
