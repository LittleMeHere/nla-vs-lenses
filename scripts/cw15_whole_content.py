"""CW-15 part 2: what each reader's write-up is made of, on whole activations (100 multihop prompts, 1 write-up each;
NLA at L42, oracle lens at L42 and at L44, its trained layer). No GPU, no LLM.
Every content word of a write-up (Latin letters/digits, stopwords dropped) goes in one bucket, first match wins:
  hidden step | answer | prompt word | format word (quiz, sentence, token, ...) | other
Usage: python scripts/cw15_whole_content.py -> runs/cw15/whole_content.txt"""
import json, re, collections
from pathlib import Path
import numpy as np
R = Path(__file__).parent.parent / "runs"; OUT = R / "cw15"; OUT.mkdir(exist_ok=True)
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
STOP = set("the a an of to in on at for and or is are was were be by with as that this it its from into about fact what which who whose not no yes also than then there their they them these those has have had will would can could should may might must do does did if so such more most other one two only very often called known since because while where when why how you your we our i he she his her".split())
META = set("format prompt question sentence clause token tokens completion complete completing answer answers trivia riddle quiz structure pattern phrase setup continuation predicate declarative fill blank template instruction style register statement noun verb grammatical syntactic punctuation word words tone factual text final fragment specific expected next naming name requires requiring establishing established signals demands narrative building toward likely almost certainly strongly implies implying context educational encyclopedic correct incomplete ends ending follows following begins beginning article proper identify identified".split())
def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3: return w[: -len(suf)]
    return w
def toks(s): return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]
def like(w, pool): 
    a = stem(w); return any(a == b or (len(a) >= 4 and len(b) >= 4 and (a in b or b in a)) for b in pool)
MS = {stem(w) for w in META}
def buckets(text, it):
    B = {stem(w) for x in it["intermediates"] for w in toks(x)}; T = {stem(w) for w in toks(it["target"])}; P = {stem(w) for w in toks(it["prompt"])}
    c = collections.Counter(); oth = []
    for w in toks(text):
        if like(w, B): c["hidden step"] += 1
        elif like(w, T): c["answer"] += 1
        elif like(w, P): c["prompt word"] += 1
        elif stem(w) in MS: c["format word"] += 1
        else: c["other"] += 1; oth.append(w)
    return c, oth
def boot(x):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
SETS = [("NLA, L42", "nla.multihop.jsonl", 42), ("Oracle lens, L42 (off its trained layers)", "olens_L42,44.multihop.jsonl", 42), ("Oracle lens, L44 (trained layer)", "olens_L42,44.multihop.jsonl", 44)]
K = ["hidden step", "answer", "prompt word", "format word", "other"]; L = ["Share of a write-up's content words in each bucket (mean over 100 prompts [95% bootstrap]); then length and the commonest 'other' words"]
for lab, f, layer in SETS:
    rows = [r for r in map(json.loads, open(R / "multihop100" / f)) if r["layer"] == layer]
    sh = {k: [] for k in K}; n = []; zh = []; allo = collections.Counter(); doc = collections.Counter()
    for r in rows:
        t = " ".join(r["samples"]); c, oth = buckets(t, bank[r["id"]]); tot = sum(c.values()) or 1
        for k in K: sh[k].append(c[k] / tot)
        n.append(tot); zh.append(len(re.findall(r"[一-鿿]", t)) / max(len(t), 1)); allo.update(oth); doc.update(set(oth))
    L.append(f"\n{lab}: {len(rows)} write-ups, {np.mean(n):.0f} content words each, {100*np.mean(zh):.1f}% of characters Chinese")
    for k in K: L.append("  %-12s %.2f [%.2f, %.2f]" % ((k,) + boot(sh[k])))
    L.append("  commonest other words (write-ups containing it): " + ", ".join(f"{w} {d}" for w, d in doc.most_common(28)))
open(OUT / "whole_content.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
