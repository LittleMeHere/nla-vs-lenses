"""Validation of the span labels (CW-16b NLA labels, CW-20 oracle lens labels).
1. second labeller: Gemini 3.8 Flash (OpenRouter) labels the SAME spans of 30 random write-ups per reader, blind to
   the first labels -> runs/cw20/validate/second_<reader>.jsonl, agreement and Cohen's kappa.
2. hand-check sheet: 20 random spans per reader, stratified over labels -> to_read/label_check_blind.md (+ key file).
Usage: cw20_validate.py second|sheet|score"""
import collections
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent
R = ROOT / "runs"
V = R / "cw20/validate"
V.mkdir(parents=True, exist_ok=True)
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
DEFS = {
    "nla": ("DESCRIPTION of the model's internal state at the last word of the PROMPT", {
        "format": "says what kind of text this is or how it is laid out, without restating the wording",
        "restate_ok": "quotes or paraphrases the PROMPT's own wording or structure, or says which word it ends on, correctly in meaning",
        "restate_wrong": "the same kind of thing, but it changes the meaning (wrong entity, wrong relation, invented context)",
        "content_ok": "a statement about the topic, entities or expected answer that is true of the PROMPT or its correct continuation",
        "content_wrong": "a statement about the topic, entities or expected answer that the PROMPT contradicts or does not support",
        "other": "stray markup or text that fits none of these"}),
    "olens": ("READOUT: a few bullets, each a guess at how the text continues after the PROMPT", {
        "answer_ok": "gives the correct continuation of the PROMPT (also in another language, or in brackets)",
        "answer_wrong": "gives a wrong answer, or an unfilled placeholder such as [State Name]",
        "about_ok": "restates the PROMPT's question, or says something true about the PROMPT's own subject and entities",
        "about_wrong": "the same kind of thing, but false or changed in meaning",
        "invented": "new questions, entities or scenarios that the PROMPT does not contain",
        "format": "list marks, markup and framing with no content"})}
SRC = {"nla": R / "cw16b/labels.jsonl", "olens": R / "cw20/labels.jsonl"}


def load(reader):
    return [d for d in (json.loads(l) for l in open(SRC[reader])) if "error" not in d]


def ask(reader, d):
    it = bank[d["key"].split("|")[0]]
    what, defs = DEFS[reader]
    spans = [s for s in d["spans"] if s["text"].strip() and s["text"].strip() != "-"]
    p = (f"A language model was reading the PROMPT below. An interpretability tool wrote the {what}. The correct continuation of the PROMPT is: {it['target']}\n\n"
         "The text has been cut into numbered spans. Give each span one label:\n" + "\n".join(f'- "{k}": {v}' for k, v in defs.items()) +
         '\n\nReturn ONLY JSON: {"labels": ["...", ...]} with one label per span, in order.\n\nPROMPT:\n' + it["prompt"] + "\n\nSPANS:\n" +
         "\n".join(f"{i + 1}. {json.dumps(s['text'], ensure_ascii=False)}" for i, s in enumerate(spans)))
    key = os.environ["OPENROUTER_API_KEY"]
    for _ in range(6):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                         data=json.dumps({"model": "google/gemini-3.8-flash", "temperature": 0, "messages": [{"role": "user", "content": p}]}).encode())
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
            labs = json.loads(re.search(r"\{.*\}", out, re.S).group(0))["labels"]
            assert len(labs) == len(spans) and all(x in defs for x in labs)
            return {"key": d["key"], "first": [s["label"] for s in spans], "second": labs, "chars": [len(s["text"]) for s in spans]}
        except Exception as e:
            err = str(e)[:120]
            time.sleep(20)
    return {"key": d["key"], "error": err}


def kappa(a, b, w=None):
    w = w or [1] * len(a)
    n = sum(w)
    po = sum(x for x, p, q in zip(w, a, b) if p == q) / n
    ca, cb = collections.Counter(), collections.Counter()
    for x, p, q in zip(w, a, b):
        ca[p] += x
        cb[q] += x
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return po, (po - pe) / (1 - pe)


mode = sys.argv[1]
if mode == "second":
    for reader in (sys.argv[2:] or ["nla", "olens"]):
        rows = load(reader)
        random.Random(0).shuffle(rows)
        with ThreadPoolExecutor(3) as ex:
            res = list(ex.map(lambda d: ask(reader, d), rows[:30]))
        with open(V / f"second_{reader}.jsonl", "w") as fh:
            for r in res:
                fh.write(json.dumps(r) + "\n")
        ok = [r for r in res if "error" not in r]
        a = [x for r in ok for x in r["first"]]
        b = [x for r in ok for x in r["second"]]
        w = [x for r in ok for x in r["chars"]]
        po, k = kappa(a, b)
        pow_, kw = kappa(a, b, w)
        print(f"{reader}: {len(ok)} of 30 write-ups, {len(a)} spans. agreement {po:.2f} (kappa {k:.2f}); by characters {pow_:.2f} (kappa {kw:.2f})")
        conf = collections.Counter(zip(a, b))
        for lab in DEFS[reader][1]:
            n = sum(v for (p, q), v in conf.items() if p == lab)
            if n:
                top = sorted(((v, q) for (p, q), v in conf.items() if p == lab), reverse=True)[:3]
                print(f"   first labeller {lab:14} n {n:4}: second says " + ", ".join(f"{q} {100 * v / n:.0f}%" for v, q in top))
elif mode == "sheet":
    rng = random.Random(8)
    items, key = [], []
    for reader in ("nla", "olens"):
        rows = load(reader)
        pool = collections.defaultdict(list)
        for d in rows:
            for j, s in enumerate(d["spans"]):
                if len(s["text"].strip()) >= 25 and s["label"] != "other":
                    pool[s["label"]].append((d, j))
        labs = [l for l in DEFS[reader][1] if pool[l]]
        for n in range(20):
            lab = labs[n % len(labs)]
            d, j = pool[lab].pop(rng.randrange(len(pool[lab])))
            items.append((reader, d, j))
    rng.shuffle(items)
    out = ["# Label check (blind)\n", "40 pieces of reader text, 20 from each reader, drawn at random with every label represented (pieces of at least 25 characters).\n\nJudge **only the piece between the >>> <<< marks**. The grey text around it is context so you can see where it sits; it has its own labels and does not count. If the marked piece is right and the text after it is wrong, the piece is still right.\n\nReply with a list like `1 b, 2 a, 3 d`. The letters mean different things for the two readers, so use the options printed under each item.\n",
           "**NLA pieces** (the NLA describes the model's state):\n\n" + "\n".join(f"- **{chr(97 + i)}** {k}: {v}" for i, (k, v) in enumerate(DEFS['nla'][1].items()) if k != "other"),
           "\n**Oracle lens pieces** (each bullet is a guess at how the text continues):\n\n" + "\n".join(f"- **{chr(97 + i)}** {k}: {v}" for i, (k, v) in enumerate(DEFS['olens'][1].items())), ""]
    for n, (reader, d, j) in enumerate(items, 1):
        it = bank[d["key"].split("|")[0]]
        sp = d["spans"]
        before = "".join(s["text"] for s in sp[max(0, j - 2):j])[-160:]
        after = "".join(s["text"] for s in sp[j + 1:j + 3])[:120]
        opts = [k for k in DEFS[reader][1] if k != "other"]
        out.append(f"## {n}. {'NLA' if reader == 'nla' else 'Oracle lens'}\n\nPrompt: *{it['prompt'].strip()}* → **{it['target']}** (hidden step: {', '.join(it['intermediates'])})\n\n> …{before.replace(chr(10), ' / ')} >>> **{sp[j]['text'].strip().replace(chr(10), ' / ')}** <<< {after.replace(chr(10), ' / ')}…\n\nOptions: " + "  ".join(f"**{chr(97 + i)}** {k}" for i, k in enumerate(opts)) + "\n")
        key.append({"n": n, "reader": reader, "key": d["key"], "span": j, "label": sp[j]["label"], "letter": chr(97 + opts.index(sp[j]["label"]))})
    (ROOT / "to_read/label_check_blind.md").write_text("\n".join(out), encoding="utf-8")
    json.dump(key, open(V / "sheet_key.json", "w"), indent=1)
    print("sheet written", len(items))
