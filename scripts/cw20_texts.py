"""CW-20: build text versions of the oracle lens write-ups from the span labels (runs/cw20/labels.jsonl). Deterministic.
Versions: orig; corrected; drop_<group>; and for each drop, 3 random-removal controls of about the same number of
characters. Line breaks between bullets are kept; a bullet left with nothing but its list mark is dropped.
MODE whole -> one row per version; MODE bullet -> one row per remaining bullet (key suffix |k), list mark stripped
when STRIP=1. Usage: cw20_texts.py whole|bullet [STRIP] -> runs/cw20/texts.jsonl"""
import json
import random
import re
import sys
from pathlib import Path

D = Path(__file__).parent.parent / "runs/cw20"
MODE = sys.argv[1]
STRIP = len(sys.argv) > 2 and sys.argv[2] == "1"
GROUPS = {"answer": {"answer_ok", "answer_wrong"}, "wrong": {"answer_wrong", "about_wrong"}, "about": {"about_ok", "about_wrong"},
          "invented": {"invented"}, "format": {"format"}}


def join(spans):
    t = "".join(s["text"] for s in spans)
    lines = [re.sub(r"[ \t]+", " ", l).rstrip() for l in t.split("\n")]
    return "\n".join(l for l in lines if l.strip() and l.strip() != "-")


def emit(rows, key, text):
    if MODE == "whole":
        rows.append({"key": key, "text": text})
    else:
        for k, l in enumerate(text.split("\n")):
            l = l[2:] if STRIP and l.startswith("- ") else l
            if l.strip():
                rows.append({"key": f"{key}|{k}", "text": l})


rows, stats = [], {g: [] for g in GROUPS}
for l in open(D / "labels.jsonl"):
    d = json.loads(l)
    if "error" in d:
        continue
    sp, key = d["spans"], d["key"]
    # list marks and line breaks are structure, never removed: only "format" spans without a line break or list mark count
    def removable(s, labs):
        return s["label"] in labs and not (s["label"] == "format" and ("\n" in s["text"] or s["text"].strip() in ("-", "")))
    n = sum(len(s["text"]) for s in sp)
    emit(rows, f"orig|{key}", join(sp))
    emit(rows, f"corrected|{key}", "\n".join(x.rstrip() for x in d["corrected"].strip().split("\n") if x.strip()))
    for g, labs in GROUPS.items():
        kept = [s for s in sp if not removable(s, labs)]
        removed = n - sum(len(s["text"]) for s in kept)
        if removed == 0 or not join(kept).strip():
            continue
        emit(rows, f"drop_{g}|{key}", join(kept))
        cand = [j for j, s in enumerate(sp) if not (s["label"] == "format" and ("\n" in s["text"] or s["text"].strip() in ("-", "")))]
        for seed in range(3):
            rng = random.Random(f"{key}|{g}|{seed}")
            order = cand[:]
            rng.shuffle(order)
            gone, tot = set(), 0
            for j in order:
                L = len(sp[j]["text"])
                if abs(tot + L - removed) < abs(tot - removed) and len(gone) < len(cand) - 1:
                    gone.add(j)
                    tot += L
            emit(rows, f"rand_{g}_{seed}|{key}", join([s for j, s in enumerate(sp) if j not in gone]))
            stats[g].append((removed / n, tot / n))
with open(D / "texts.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "texts", MODE, "strip" if STRIP else "")
for g, v in stats.items():
    if v:
        print(f"  drop_{g}: {len(v)//3} write-ups; removes {100*sum(a for a,_ in v)/len(v):.0f}% of characters; its random controls remove {100*sum(b for _,b in v)/len(v):.0f}%")
