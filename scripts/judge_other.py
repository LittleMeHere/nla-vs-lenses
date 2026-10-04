"""Judge the NLA concepts left in the 'other' bucket by classify_claims.py. One `claude -p` call per
item: given the exact prompt and that item's 'other' concepts, label each concept:
  contradicted  - asserts something about this prompt or its context that the prompt rules out
                  (wrong format claim, wrong number, wrong entity) -> hallucination
  restates      - says what the prompt says, in other words
  form          - describes the prompt's form or the expected answer type, consistent with it
  unstated      - content not stated in the prompt and not contradicted by it (a fact, entity or
                  inference the model might have computed)
Usage: python judge_other.py ITEMS.json CLAIMS.jsonl OUT.jsonl"""
import json, re, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path
PROMPT = """You label short concepts extracted from an interpretability tool's description of a language model's internal state while it read the prompt below. For EACH concept choose exactly one label:
- "contradicted": it asserts something about this prompt or its context that the prompt rules out (e.g. claims a multiple-choice format, an instruction, a number or an entity that is not there or is different).
- "restates": it says what the prompt itself says, possibly in other words.
- "form": it describes the prompt's form, genre or the type of answer expected, and is consistent with the prompt.
- "unstated": content that the prompt does not state and does not rule out (a fact, entity, answer candidate or inference).
Return ONLY JSON: {"labels": [{"concept": <exact concept>, "label": <label>}, ...]} with one entry per concept, in order.

PROMPT:
<<<%s>>>

CONCEPTS:
%s"""
def main(items_p, claims_p, out_p):
    items = {it["name"]: it for it in json.load(open(items_p))["items"]}
    by = defaultdict(list)
    for l in open(claims_p):
        r = json.loads(l)
        if r["bucket"] == "other": by[r["id"]].append(r["concept"])
    done = {json.loads(l)["id"] for l in open(out_p)} if Path(out_p).exists() else set()
    with open(out_p, "a") as fh:
        for i, cs in by.items():
            if i in done: continue
            listing = "\n".join(f"{k+1}. {c}" for k, c in enumerate(cs))
            raw = subprocess.run(["claude", "-p", PROMPT % (items[i]["prompt"], listing), "--output-format", "text"],
                                 capture_output=True, text=True, timeout=300).stdout
            m = re.search(r"\{.*\}", raw, re.S)
            labs = json.loads(m.group(0))["labels"] if m else []
            fh.write(json.dumps({"id": i, "concepts": cs, "labels": labs, "raw_ok": bool(m)}, ensure_ascii=False) + "\n"); fh.flush()
    c = Counter(x["label"] for l in open(out_p) for x in json.loads(l)["labels"])
    n = sum(c.values()); print({k: f"{v} ({v/n:.0%})" for k, v in c.most_common()}, "n =", n)
if __name__ == "__main__":
    main(*sys.argv[1:4])
