"""CW-11 analysis (designs/CW-11_cue_removal.md). Usage: python scripts/cw11_analyze.py -> runs/cw11/analysis.txt"""
import json, re
from pathlib import Path
import numpy as np
R = Path(__file__).parent.parent / "runs"; D = R / "cw11"
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
STOP = set("the a an of to in on at for and or is are was were be by with as that this it its from into about fact "
           "what which who whose".split())
def names(t, inter): return any(re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", t.lower()) for w in inter)
def cw(it):
    drop = set(re.findall(r"[a-z0-9]+", " ".join(it["intermediates"] + [it["target"]]).lower()))
    return {w for w in re.findall(r"[a-z0-9]+", it["prompt"].lower()) if w not in STOP and w not in drop and len(w) > 1}
def rec(t, c): return len(c & set(re.findall(r"[a-z0-9]+", t.lower()))) / len(c)
def load(*fs):
    by = {}
    for f in fs:
        for l in open(D / f):
            r = json.loads(l); by.setdefault(r["cond"], {})[r["id"]] = r
    return by
def boot(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float); k = len(d)
    bs = d[rng.integers(0, k, (B, k))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
CONDS = ["intact", "rand", "cue", "cue_orth", "bridge", "both"]
L = []
J = load("jlens_cw11.jsonl", "jlens_cw11c.jsonl"); ids = sorted(J["intact"])
L.append(f"items {len(ids)}; bridge dirs mean {np.mean([J['intact'][i]['n_bridge_dirs'] for i in ids]):.1f}; cue dirs mean {np.mean([J['intact'][i]['n_cue_dirs'] for i in ids]):.1f}")
L.append("\nCHECKS ON THE REMOVALS (J-Lens): cond | bridge in top-10 | bridge J score | cue words in top-50 | norm ratio")
for c in CONDS:
    v = J[c]
    L.append(f"  {c}: {sum(names(' '.join(t.replace('Ġ', ' ') for t in v[i]['tokens']), bank[i]['intermediates']) for i in ids)}/50 | "
             f"{np.mean([v[i]['bridge_score'] for i in ids]):.2f} | {np.mean([v[i]['cue_in_top50'] for i in ids]):.2f} | {np.mean([v[i]['norm_ratio'] for i in ids]):.3f}")
L.append(f"  cue_orth vs intact bridge score: max abs diff {max(abs(J['cue_orth'][i]['bridge_score'] - J['intact'][i]['bridge_score']) for i in ids):.4f}")
res = {}
for reader, fs in (("oracle", ("olens_cw11.jsonl", "olens_cw11c.jsonl")), ("NLA", ("nla_cw11.jsonl", "nla_cw11c.jsonl"))):
    by = load(*fs)
    rate = {c: np.array([np.mean([names(s, bank[i]["intermediates"]) for s in by[c][i]["samples"]]) for i in ids]) for c in CONDS}
    rc = {c: np.array([np.mean([rec(s, cw(bank[i])) for s in by[c][i]["samples"]]) for i in ids]) for c in CONDS}
    anyh = {c: int(sum(any(names(s, bank[i]["intermediates"]) for s in by[c][i]["samples"]) for i in ids)) for c in CONDS}
    res[reader] = (rate, rc)
    L.append(f"\n{reader.upper()} (50 items x {len(by['intact'][ids[0]]['samples'])} samples): cond | share of samples naming the bridge | items with any sample naming | prompt recovery")
    for c in CONDS:
        L.append(f"  {c}: {rate[c].mean():.3f} | {anyh[c]} | {rc[c].mean():.3f}")
    L.append("  paired differences over items, mean [95% bootstrap]:")
    for a, b, tag in (("cue_orth", "rand", "MAIN naming"), ("cue", "rand", "naming"), ("bridge", "rand", "naming"), ("both", "bridge", "naming"), ("both", "rand", "naming")):
        m, lo, hi = boot(rate[a] - rate[b]); L.append(f"    {tag}: {a} - {b} = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for a, b in (("cue_orth", "rand"), ("cue", "rand"), ("bridge", "rand")):
        m, lo, hi = boot(rc[a] - rc[b]); L.append(f"    recovery: {a} - {b} = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # items whose prompt was hidden most by cue_orth: does naming fall there?
    drop = rc["rand"] - rc["cue_orth"]; top = np.argsort(-drop)[:17]
    m, lo, hi = boot((rate["cue_orth"] - rate["rand"])[top])
    L.append(f"    third of items with the largest recovery drop under cue_orth (n 17, mean drop {drop[top].mean():.2f}): naming cue_orth - rand = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    L.append(f"    corr over items (recovery change, naming change), cue_orth vs rand: {np.corrcoef(rc['cue_orth'] - rc['rand'], rate['cue_orth'] - rate['rand'])[0, 1]:+.2f}")
    # CW-10 relation with naming as a rate: intact
    L.append(f"    intact: corr over items (mean recovery, naming rate) {np.corrcoef(rc['intact'], rate['intact'])[0, 1]:+.2f}; after bridge removal {np.corrcoef(rc['bridge'], rate['bridge'])[0, 1]:+.2f}")
    # sample-to-sample agreement
    ag = [np.mean([names(s, bank[i]['intermediates']) for s in by['intact'][i]['samples']]) for i in ids]
    L.append(f"    intact: items where the 4 samples all agree {sum(a in (0.0, 1.0) for a in ag)}/50")
open(D / "analysis.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
