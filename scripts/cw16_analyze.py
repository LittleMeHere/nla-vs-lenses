"""CW-16 analysis (designs/CW-16_reconstructor.md). Usage: python scripts/cw16_analyze.py -> runs/cw16/analysis.txt, analysis.json
Rebuilt vectors and targets are scaled to norm sqrt(5120), as the reconstructor's loss does. mu = mean of the 70 scaled targets."""
import json, math, re
from pathlib import Path
import numpy as np, torch
R = Path(__file__).parent.parent / "runs"; D = R / "cw16"; SC = math.sqrt(5120)
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
rec = torch.load(D / "recon.pt"); meta = json.load(open(D / "meta.json")); partner = meta["partner"]
cells = torch.load(R / "cw14/resid_split.pt") + torch.load(R / "cw14/resid_ext.pt")
sc = lambda v: v.float() / v.float().norm().clamp_min(1e-9) * SC
G = {c["id"]: sc(c["h"]) for c in cells if c["cond"] == "full"}; ids = sorted(G)
PART = {(c["cond"], c["id"]): sc(c["h"]) for c in cells if c["cond"] != "full"}
mu = torch.stack([G[i] for i in ids]).mean(0)
Q = torch.load(R / "cw12/jbasis.pt")["Q"].float()[:, :1024]
pj = lambda v: Q @ (Q.T @ v)
def metrics(p, g):
    p = sc(p); d = p - g; gj, dj = pj(g - mu), pj(d); gr, dr = (g - mu) - gj, d - dj
    cos = torch.nn.functional.cosine_similarity
    return dict(cos=float(cos(p, g, dim=0)), fve=1 - float(d.pow(2).sum() / (g - mu).pow(2).sum()),
                fve_J=1 - float(dj.pow(2).sum() / gj.pow(2).sum()), fve_rest=1 - float(dr.pow(2).sum() / gr.pow(2).sum()),
                cosc=float(cos(p - mu, g - mu, dim=0)), cosc_J=float(cos(pj(p - mu), gj, dim=0)), cosc_rest=float(cos((p - mu) - pj(p - mu), gr, dim=0)),
                share_J=float(pj(p).pow(2).sum() / p.pow(2).sum()))
def boot(x):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
per = {}   # version -> id -> list of metric dicts
for k, p in rec.items():
    f = k.split("|"); v, i = f[0], f[1]
    if v.startswith("part:"): continue
    per.setdefault(v, {}).setdefault(i, []).append(metrics(p, G[i]))
KEYS = ["cos", "fve", "fve_J", "fve_rest", "cosc", "cosc_J", "cosc_rest", "share_J"]
agg = lambda dd: {m: boot([np.mean([x[m] for x in v]) for v in dd.values()]) for m in KEYS}
res = {}; L = [f"targets: 70 prompts; share of the target's squared norm in the J part {np.mean([float(pj(G[i]).pow(2).sum()/G[i].pow(2).sum()) for i in ids]):.3f}; cosine of a target with the mean {np.mean([float(torch.nn.functional.cosine_similarity(G[i], mu, dim=0)) for i in ids]):.3f}",
     "", "version (prompts) | cosine | FVE | FVE in J part | FVE in rest | centred cosine: all / J part / rest | share of rebuilt norm in J part"]
NAMES = [("orig", "original write-up"), ("shuf", "another prompt's write-up"), ("prompt", "the prompt itself"), ("factual", "false statements removed"), ("content", "content only"), ("form", "form only"), ("swap", "hidden step swapped"), ("olens", "oracle lens write-up")]
for v, lab in NAMES:
    if v not in per: continue
    a = agg(per[v]); res[v] = a
    L.append(f"  {lab} ({len(per[v])}): {a['cos'][0]:.3f} | {a['fve'][0]:+.3f} [{a['fve'][1]:+.3f}, {a['fve'][2]:+.3f}] | {a['fve_J'][0]:+.3f} [{a['fve_J'][1]:+.3f}, {a['fve_J'][2]:+.3f}] | {a['fve_rest'][0]:+.3f} [{a['fve_rest'][1]:+.3f}, {a['fve_rest'][2]:+.3f}] | {a['cosc'][0]:.3f} / {a['cosc_J'][0]:.3f} / {a['cosc_rest'][0]:.3f} | {a['share_J'][0]:.3f}")
L.append("\npaired differences against the original write-up (mean over prompts [95% bootstrap]): FVE | FVE in J part | FVE in rest")
for v, lab in NAMES[1:]:
    if v not in per: continue
    common = [i for i in per[v] if i in per["orig"]]; row = []
    for m in ("fve", "fve_J", "fve_rest"):
        d = boot([np.mean([x[m] for x in per[v][i]]) - np.mean([x[m] for x in per["orig"][i]]) for i in common]); row.append(f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    L.append(f"  {lab} ({len(common)}): " + " | ".join(row))
# hidden step swapped: does the J part move toward the partner prompt?
sw = {}
for k, p in rec.items():
    f = k.split("|")
    if f[0] != "swap": continue
    i = f[1]; o = sc(rec[f"orig|{i}|{f[2]}"]); p = sc(p); cos = torch.nn.functional.cosine_similarity
    own, par = pj(G[i] - mu), pj(G[partner[i]] - mu)
    sw.setdefault(i, []).append((float(cos(pj(o - mu), own, dim=0)), float(cos(pj(p - mu), own, dim=0)), float(cos(pj(o - mu), par, dim=0)), float(cos(pj(p - mu), par, dim=0))))
if sw:
    M = np.array([np.mean(v, axis=0) for v in sw.values()])
    L.append(f"\nhidden step swapped ({len(sw)} prompts), centred cosine inside the J part: with own target, original {M[:,0].mean():.3f} -> swapped {M[:,1].mean():.3f}; with the partner prompt's target, original {M[:,2].mean():.3f} -> swapped {M[:,3].mean():.3f}")
    res["swap_J"] = {"own_orig": boot(M[:, 0]), "own_swap": boot(M[:, 1]), "partner_orig": boot(M[:, 2]), "partner_swap": boot(M[:, 3])}
# write-ups made from parts: which target do they rebuild?
L.append("\nwrite-ups the NLA made from one part (70 prompts): centred cosine of the rebuilt vector with the whole target, inside the J part / inside the rest | share of rebuilt norm in J part")
pp = {}
for k, p in rec.items():
    f = k.split("|")
    if not f[0].startswith("part:"): continue
    pp.setdefault(f[0][5:], {}).setdefault(f[1], []).append(metrics(p, G[f[1]]))
for c, lab in (("J1024", "J part"), ("N1024", "rest"), ("R1024", "random part"), ("R1024s1", "random part 2"), ("RN1024", "all but random"), ("P1024", "PCA part"), ("PN1024", "all but PCA")):
    if c in pp:
        a = agg(pp[c]); res["part:" + c] = a
        L.append(f"  from {lab}: {a['cosc_J'][0]:.3f} [{a['cosc_J'][1]:.3f}, {a['cosc_J'][2]:.3f}] / {a['cosc_rest'][0]:.3f} [{a['cosc_rest'][1]:.3f}, {a['cosc_rest'][2]:.3f}] | {a['share_J'][0]:.3f}   (FVE {a['fve'][0]:+.3f})")
open(D / "analysis.txt", "w").write("\n".join(L) + "\n"); json.dump(res, open(D / "analysis.json", "w"), indent=1); print("\n".join(L))
