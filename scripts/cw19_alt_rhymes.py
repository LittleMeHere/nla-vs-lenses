"""CW-19 control: for each poetry item, 3 other words that rhyme with line one's last word,
closest in word frequency to the item's target. Writes one altered bank per control set."""
import json
import re
import sys

import pronouncing
from wordfreq import zipf_frequency

src, outdir = sys.argv[1], sys.argv[2]
bank = json.load(open(src))
sets = [json.loads(json.dumps(bank)) for _ in range(3)]
log, short = [], 0
for n, it in enumerate(bank["items"]):
    W = re.findall(r"[A-Za-z']+", it["prompt"].split("\n")[1])[-1].lower()
    T = it["intermediates"][0].lower()
    zt = zipf_frequency(T, "en")
    cand = {r for r in pronouncing.rhymes(W) if r.isalpha() and r not in (W, T)
            and not r.startswith(T) and not T.startswith(r) and zipf_frequency(r, "en") >= 3.5}
    cand = sorted(cand, key=lambda r: (abs(zipf_frequency(r, "en") - zt), r))[:3]
    if len(cand) < 3:
        short += 1
    for k in range(3):
        sets[k]["items"][n]["intermediates"] = [cand[k]] if k < len(cand) else ["zzzunused"]
    log.append({"name": it["name"], "last_word": W, "target": T, "target_zipf": zt,
                "controls": cand, "control_zipf": [zipf_frequency(c, "en") for c in cand]})
for k in range(3):
    json.dump(sets[k], open(f"{outdir}/items_ctrl{k}.json", "w"), indent=1)
json.dump(log, open(f"{outdir}/alt_rhymes.json", "w"), indent=1)
print("items", len(log), "with fewer than 3 controls", short)
for r in log[:6]:
    print(r["name"], r["last_word"], r["target"], r["controls"])
