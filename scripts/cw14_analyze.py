"""CW-14 analysis (designs/CW-14_split_confirmation.md). Usage: python scripts/cw14_analyze.py -> runs/cw14/analysis.txt
Only prompts complete (all 4 conditions) for both readers are used."""
import json, re
from pathlib import Path
import numpy as np
R = Path(__file__).parent.parent / "runs"; D = R / "cw14"
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
def names(t, inter): return any(re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", t.lower()) for w in inter)
def boot(x):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def perm(d, B=100000):
    d = np.asarray(d, float); s = np.random.default_rng(1).choice([-1.0, 1.0], (B, len(d)))
    return (1 + np.sum(np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12)) / (B + 1)
CONDS = ["full", "J1024", "N1024", "R1024"]; LAB = {"full": "whole", "J1024": "J part", "N1024": "rest", "R1024": "random part"}
EXT = ["RN1024", "R1024s1", "P1024", "PN1024"]; LAB.update({"RN1024": "all but random", "R1024s1": "random part (seed 1)", "P1024": "PCA part", "PN1024": "all but PCA"})
def load(fs): return [json.loads(l) for f in fs if (D / f"{f}.jsonl").exists() for l in open(D / f"{f}.jsonl")]
rows = {"oracle": load(["olens_split", "olens_ext"]), "NLA": load(["nla_split", "nla_ext"])}
have_ext = all({r["cond"] for r in rs if r["id"] == rs[0]["id"]} >= set(EXT) for rs in rows.values())
done = {rd: {i for i in {r["id"] for r in rs} if {r["cond"] for r in rs if r["id"] == i} >= set(CONDS)} for rd, rs in rows.items()}
ids = sorted(done["oracle"] & done["NLA"])
if have_ext:  # extension is analysed on prompts complete for all 8 conditions and both readers; the primary tests use `ids`
    ext_ids = [i for i in ids if all({r["cond"] for r in rs if r["id"] == i} >= set(CONDS + EXT) for rs in rows.values())]
L = [f"prompts complete for both readers: {len(ids)} (oracle {len(done['oracle'])}, NLA {len(done['NLA'])})"]
S = {}
for rd, rs in rows.items():
    L.append(f"\n{rd.upper()} ({len(ids)} prompts x {len(rs[0]['samples'])} samples): part | share of squared norm | samples naming the hidden step [95% bootstrap]")
    for c in CONDS:
        cr = {r["id"]: r for r in rs if r["cond"] == c and r["id"] in ids}
        S[(rd, c)] = {i: np.mean([names(s, bank[i]["intermediates"]) for s in cr[i]["samples"]]) for i in ids}
        m, lo, hi = boot(list(S[(rd, c)].values()))
        L.append(f"  {LAB[c]}: {np.mean([cr[i]['share'] for i in ids]):.2f} | {m:.3f} [{lo:.3f}, {hi:.3f}]")
def diff(rd, x, y, tag):
    d = [S[(rd, x)][i] - S[(rd, y)][i] for i in ids]; m, lo, hi = boot(d)
    L.append(f"  {tag} {rd}: {LAB[x]} - {LAB[y]} = {m:+.3f} [{lo:+.3f}, {hi:+.3f}], sign-flip p = {perm(d):.5f}, SD {np.std(d, ddof=1):.3f}, prompts +/0/-: {sum(v > 0 for v in d)}/{sum(v == 0 for v in d)}/{sum(v < 0 for v in d)}")
L.append("\nPRIMARY (confirmed at p < 0.005 with the CW-12 sign; not confirmed if the interval includes 0)")
diff("oracle", "J1024", "N1024", "1."); diff("NLA", "J1024", "N1024", "2."); diff("oracle", "J1024", "R1024", "3.")
L.append("\nSECONDARY")
diff("NLA", "J1024", "R1024", "-"); diff("oracle", "J1024", "full", "-"); diff("NLA", "J1024", "full", "-"); diff("oracle", "R1024", "N1024", "-"); diff("NLA", "R1024", "N1024", "-")
o, n = S[("oracle", "full")], S[("NLA", "full")]
L.append(f"  whole activation, any sample names the step: both {sum(o[i] > 0 and n[i] > 0 for i in ids)}, oracle only {sum(o[i] > 0 and n[i] == 0 for i in ids)}, NLA only {sum(o[i] == 0 and n[i] > 0 for i in ids)}, neither {sum(o[i] == 0 and n[i] == 0 for i in ids)}")
if have_ext:
    L.append(f"\nEXTENSION (secondary; {len(ext_ids)} prompts complete for all 8 conditions)")
    main_ids, ids = ids, ext_ids
    for rd, rs in rows.items():
        for c in CONDS + EXT:
            cr = {r["id"]: r for r in rs if r["cond"] == c and r["id"] in ids}
            S[(rd, c)] = {i: np.mean([names(s, bank[i]["intermediates"]) for s in cr[i]["samples"]]) for i in ids}
            if c in EXT:
                m, lo, hi = boot(list(S[(rd, c)].values())); L.append(f"  {rd} {LAB[c]}: {np.mean([cr[i]['share'] for i in ids]):.2f} | {m:.3f} [{lo:.3f}, {hi:.3f}]")
    for rd in rows:
        for x, y in (("N1024", "RN1024"), ("J1024", "R1024s1"), ("R1024", "R1024s1"), ("J1024", "P1024"), ("N1024", "PN1024"), ("P1024", "PN1024")): diff(rd, x, y, "-")
    ids = main_ids
jl = {c: sum(names(" ".join(r["tokens"]), bank[r["id"]]["intermediates"]) for r in rows["oracle"] if r["cond"] == c and r["id"] in ids) for c in CONDS}
L.append("  J-Lens top-10 names the step: " + ", ".join(f"{LAB[c]} {jl[c]}/{len(ids)}" for c in CONDS))
open(D / "analysis.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
