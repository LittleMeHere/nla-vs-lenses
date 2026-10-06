"""CW-16b: build text versions from the span labels (runs/cw16b/labels.jsonl). Deterministic.
Versions: orig; corrected (LLM, wrong spans rewritten in place); drop_<group> for each group of labels; and for each drop,
3 random-deletion controls that remove random spans of about the same number of characters. -> runs/cw16b/texts.jsonl"""
import json, random, re
from pathlib import Path
D = Path(__file__).parent.parent / "runs/cw16b"
GROUPS = {"content_wrong": {"content_wrong"}, "restate_wrong": {"restate_wrong"}, "all_wrong": {"content_wrong", "restate_wrong"},
          "restate": {"restate_ok", "restate_wrong"}, "format": {"format"}, "content": {"content_ok", "content_wrong"}}
def join(spans): return re.sub(r"[ \t]+", " ", " ".join(s["text"].strip() for s in spans if s["text"].strip())).strip()
rows = []; stats = {g: [] for g in GROUPS}
for l in open(D / "labels.jsonl"):
    d = json.loads(l)
    if "error" in d: continue
    sp = d["spans"]; key = d["key"]; n = sum(len(s["text"]) for s in sp)
    rows.append({"key": f"orig|{key}", "text": join(sp)}); rows.append({"key": f"corrected|{key}", "text": d["corrected"].strip()})
    for g, labs in GROUPS.items():
        kept = [s for s in sp if s["label"] not in labs]; removed = n - sum(len(s["text"]) for s in kept)
        if removed == 0 or not kept: continue
        rows.append({"key": f"drop_{g}|{key}", "text": join(kept)})
        for seed in range(3):
            rng = random.Random(f"{key}|{g}|{seed}"); order = list(range(len(sp))); rng.shuffle(order); gone, tot = set(), 0
            for j in order:
                L = len(sp[j]["text"])
                if abs(tot + L - removed) < abs(tot - removed) and len(gone) < len(sp) - 1: gone.add(j); tot += L
            ctrl = [s for j, s in enumerate(sp) if j not in gone]
            rows.append({"key": f"rand_{g}_{seed}|{key}", "text": join(ctrl)}); stats[g].append((removed / n, tot / n))
with open(D / "texts.jsonl", "w") as fh:
    for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "texts")
for g, v in stats.items():
    if v: print(f"  drop_{g}: removes {100*sum(a for a,_ in v)/len(v):.0f}% of characters; its random controls remove {100*sum(b for _,b in v)/len(v):.0f}%")
