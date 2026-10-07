"""CW-19: judge pass counts per reader and layer, and where the NLA's passes come from.
Reads runs/cw19/ (readouts and the benchmark judge's cells) and runs/cw17/poetry_items.json."""
import collections
import json
import re

R = "runs/cw19"
items = {i["name"]: i for i in json.load(open("runs/cw17/poetry_items.json"))["items"]}
for r in ["olens", "nla"]:
    c, anyl = collections.Counter(), set()
    for l in open(f"{R}/judged_{r}/cells.jsonl"):
        d = json.loads(l)
        k, L = d["key"].split("|")
        c[L, "n"] += 1
        c[L, "yes"] += bool(d["result"]["expressed"])
        if d["result"]["expressed"]:
            anyl.add(k)
    print(r, dict(c), "pass at any layer:", len(anyl))

nla = {json.loads(l)["id"]: json.loads(l)["samples"][0] for l in open(f"{R}/nla_poetry.jsonl")}
judged = {json.loads(l)["key"].split("|")[0]: json.loads(l)["result"] for l in open(f"{R}/judged_nla/cells.jsonl")}
c = collections.Counter()
for k, s in nla.items():
    it = items[k]
    W = re.findall(r"[A-Za-z']+", it["prompt"].split("\n")[1])[-1].lower()
    T = [t.lower() for t in it["intermediates"]]
    # last word of each line the write-up quotes as line one (word, comma, then newline or closing quote)
    ends = [w.lower() for w in re.findall(r"([A-Za-z']+),\s*\\?n?\s*[\"”]", s)] + [w.lower() for w in re.findall(r"([A-Za-z']+),\n", s)]
    c["n"] += 1
    c["write-up restates a line"] += bool(ends)
    c["a restated line ends in the real last word"] += any(e == W for e in ends)
    c["a restated line ends in the target"] += any(e in T for e in ends)
    c["restated lines end in neither"] += bool(ends) and not any(e == W or e in T for e in ends)
    if judged[k]["expressed"]:
        c["judge pass"] += 1
        c["judge pass, target is the end of a restated line"] += any(e in T for e in ends)
for k, v in c.items():
    print(f"{k}: {v}")
