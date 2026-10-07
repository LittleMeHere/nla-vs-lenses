"""CW-19 control: judge passes for the real target vs 3 other rhyme words (runs/cw19/ctrl)."""
import collections
import json
import random

R = "runs/cw19"
alt = {r["name"]: r for r in json.load(open(f"{R}/ctrl/alt_rhymes.json"))}


def passes(d):
    o = collections.defaultdict(dict)
    for l in open(f"{d}/cells.jsonl"):
        x = json.loads(l)
        k, L = x["key"].split("|")
        o[L][k] = bool((x.get("result") or {}).get("expressed"))
    return o


for r in ["olens", "nla"]:
    real = passes(f"{R}/judged_{r}")
    ctrl = [passes(f"{R}/ctrl/judged_{r}_ctrl{k}") for k in range(3)]
    for L in sorted(real):
        print(r, L, "target", sum(real[L].values()), "controls", [sum(c[L].values()) for c in ctrl], "n", len(real[L]))
    anyr = {i for L in real for i, v in real[L].items() if v}
    anyc = [{i for L in c for i, v in c[L].items() if v} for c in ctrl]
    print(r, "any layer: target", len(anyr), "controls", [len(a) for a in anyc])
    # paired, per prompt: target pass minus mean control pass, bootstrap over prompts
    ids = sorted(alt)
    d = [(i in anyr) - sum(i in a for a in anyc) / 3 for i in ids]
    random.seed(0)
    bs = sorted(sum(random.choices(d, k=len(d))) / len(d) for _ in range(10000))
    print(r, "target minus mean control: %.3f [%.3f, %.3f]" % (sum(d) / len(d), bs[250], bs[9750]))
