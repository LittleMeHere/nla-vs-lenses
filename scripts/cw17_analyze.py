"""CW-17 analysis (designs/CW-17_poetry_split.md). Usage: python scripts/cw17_analyze.py -> runs/cw17/analysis.txt
Prompts complete for all 4 conditions are used, per reader. Chance = the same score against another prompt's rhyme word."""
import json, random, re
from pathlib import Path
import numpy as np
R = Path(__file__).parent.parent / "runs"; D = R / "cw17"
bank = {i["name"]: i for i in json.load(open(D / "poetry_items.json"))["items"]}
def names(t, ws): return any(re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", t.lower()) for w in ws)
def boot(x):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1); return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def perm(d, B=100000):
    d = np.asarray(d, float); s = np.random.default_rng(1).choice([-1.0, 1.0], (B, len(d))); return (1 + np.sum(np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12)) / (B + 1)
CONDS = ["full", "J1024", "N1024", "R1024"]; LAB = {"full": "whole", "J1024": "J part", "N1024": "rest", "R1024": "random part"}; L = []
for rd, files in (("ORACLE LENS (layer 44)", ["olens_poetry_a", "olens_poetry_b"]), ("NLA (layer 42)", ["nla_poetry_a", "nla_poetry_b"])):
    rows = [json.loads(l) for f in files if (D / f"{f}.jsonl").exists() for l in open(D / f"{f}.jsonl")]
    if not rows: continue
    assert len({r["layer"] for r in rows}) == 1, "mixed layers"
    by = {}
    for r in rows: by.setdefault(r["id"], {})[r["cond"]] = r
    ids = sorted(i for i in by if set(by[i]) >= set(CONDS)); rng = random.Random(0)
    while True:
        perm_ids = ids[:]; rng.shuffle(perm_ids)
        if all(a != b and bank[a]["intermediates"] != bank[b]["intermediates"] for a, b in zip(ids, perm_ids)): break
    other = dict(zip(ids, perm_ids))
    S = {c: {i: np.mean([names(s, bank[i]["intermediates"]) for s in by[i][c]["samples"]]) for i in ids} for c in CONDS}
    Ch = {c: {i: np.mean([names(s, bank[other[i]]["intermediates"]) for s in by[i][c]["samples"]]) for i in ids} for c in CONDS}
    L.append(f"\n{rd}: {len(ids)} prompts x {len(rows[0]['samples'])} samples, read layer {rows[0]['layer']}")
    L.append("  part | share of squared norm | names the rhyme word [95%] | chance (another prompt's word) | J-Lens top-10 has it")
    for c in CONDS:
        m, lo, hi = boot(list(S[c].values())); jl = sum(names(" ".join(by[i][c]["tokens"]), bank[i]["intermediates"]) for i in ids)
        L.append(f"  {LAB[c]}: {np.mean([by[i][c]['share'] for i in ids]):.2f} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {np.mean(list(Ch[c].values())):.3f} | {jl}/{len(ids)}")
    def diff(x, y, sel, tag):
        d = [S[x][i] - S[y][i] for i in sel]; m, lo, hi = boot(d)
        L.append(f"  {tag}{LAB[x]} - {LAB[y]} (n={len(sel)}): {m:+.3f} [{lo:+.3f}, {hi:+.3f}], sign-flip p = {perm(d):.5f}, prompts +/0/-: {sum(v > 0 for v in d)}/{sum(v == 0 for v in d)}/{sum(v < 0 for v in d)}")
    L.append("  pre-set tests:"); diff("J1024", "N1024", ids, "  "); diff("J1024", "R1024", ids, "  ")
    gated = [i for i in ids if S["full"][i] == 1.0]; L.append(f"  gated (both whole-activation samples name it): {len(gated)} prompts")
    if len(gated) >= 5:
        L.append("    " + ", ".join(f"{LAB[c]} {np.mean([S[c][i] for i in gated]):.3f}" for c in CONDS)); diff("J1024", "N1024", gated, "    "); diff("J1024", "R1024", gated, "    ")
open(D / "analysis.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
