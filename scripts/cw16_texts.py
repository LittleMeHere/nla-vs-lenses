"""CW-16: build the list of texts for the reconstructor (designs/CW-16_reconstructor.md). -> runs/cw16/texts.jsonl, meta.json"""
import json, random, re
from pathlib import Path
R = Path(__file__).parent.parent / "runs"; D = R / "cw16"
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
nla = [json.loads(l) for f in ("nla_split", "nla_ext") for l in open(R / f"cw14/{f}.jsonl")]
ol = [json.loads(l) for l in open(R / "cw14/olens_split.jsonl")]
full = {r["id"]: r["samples"] for r in nla if r["cond"] == "full"}; ids = sorted(full)
rng = random.Random(0)
while True:   # derangement with a different hidden step
    perm = ids[:]; rng.shuffle(perm)
    if all(a != b and set(map(str.lower, bank[a]["intermediates"])).isdisjoint(map(str.lower, bank[b]["intermediates"])) for a, b in zip(ids, perm)): break
partner = dict(zip(ids, perm)); rows = []; nswap = 0
def swap(t, a, b):
    out = t
    for w in sorted(bank[a]["intermediates"], key=len, reverse=True):
        out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", bank[b]["intermediates"][0], out, flags=re.I)
    return out
for i in ids:
    rows.append({"key": f"prompt|{i}", "text": bank[i]["prompt"].strip()})
    for k, s in enumerate(full[i]):
        rows.append({"key": f"orig|{i}|{k}", "text": s}); rows.append({"key": f"shuf|{i}|{k}", "text": full[partner[i]][k]})
        sw = swap(s, i, partner[i])
        if sw != s: rows.append({"key": f"swap|{i}|{k}", "text": sw}); nswap += 1
for r in nla:
    if r["cond"] != "full":
        for k, s in enumerate(r["samples"]): rows.append({"key": f"part:{r['cond']}|{r['id']}|{k}", "text": s})
for r in ol:
    if r["cond"] == "full":
        for k, s in enumerate(r["samples"]): rows.append({"key": f"olens|{r['id']}|{k}", "text": s})
nrw = {"factual": 0, "content": 0, "form": 0}
for l in open(D / "rewrites.jsonl"):
    d = json.loads(l)
    if "error" in d: continue
    for v in nrw:
        if isinstance(d.get(v), str) and len(d[v].strip()) >= 20: rows.append({"key": f"{v}|{d['key']}", "text": d[v].strip()}); nrw[v] += 1
with open(D / "texts.jsonl", "w") as fh:
    for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"partner": partner, "n_rows": len(rows), "n_swap": nswap, "n_rewrites": nrw}, open(D / "meta.json", "w"), indent=1)
print(len(rows), "texts; swapped", nswap, "; rewrites", nrw)
