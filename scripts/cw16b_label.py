"""CW-16b: label every span of an NLA write-up, and write one minimally corrected version.
One `claude -p --model sonnet` call per write-up; cached in runs/cw16b/labels.jsonl. Spans must reproduce the write-up
exactly (whitespace aside), or the call is retried. Usage: python scripts/cw16b_label.py [N_PROMPTS]"""
import json, random, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R = Path(__file__).parent.parent / "runs"; OUT = R / "cw16b/labels.jsonl"
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
INSTR = """A language model was reading the PROMPT below. An interpretability tool wrote the DESCRIPTION of the model's internal state at the last word of the PROMPT. The correct continuation of the PROMPT is: %s

Task 1. Cut the DESCRIPTION into consecutive spans and label each one. The spans, joined in order, must reproduce the DESCRIPTION exactly, character for character (keep punctuation, quotes and line breaks inside the spans). Cut inside sentences where a sentence mixes kinds. Labels:
- "format": says what kind of text this is or how it is laid out (quiz, Q&A, fill-in-the-blank, list pattern), without restating the wording.
- "restate_ok": quotes or paraphrases the PROMPT's own wording or sentence structure, or says which word/phrase it ends on, and gets it right in meaning (small wording differences are fine).
- "restate_wrong": the same kind of thing, but it changes the meaning (wrong entity, wrong relation, invented context such as "in this image").
- "content_ok": a statement about the topic, the entities involved or the expected answer that is true of the PROMPT or its correct continuation.
- "content_wrong": a statement about the topic, entities or expected answer that the PROMPT contradicts or does not support.
- "other": stray markup or text that fits none of these.

Task 2. Write "corrected": the whole DESCRIPTION with every restate_wrong and content_wrong span minimally rewritten so that it is true of the PROMPT (fix the wrong entity or relation, replace an invented quote by the PROMPT's real wording). Change nothing else. Keep the length about the same.

Return ONLY JSON: {"spans": [{"text": "...", "label": "..."}, ...], "corrected": "..."}

PROMPT:
%s

DESCRIPTION:
%s
"""
norm = lambda s: re.sub(r"\s+", "", s)
LABELS = {"format", "restate_ok", "restate_wrong", "content_ok", "content_wrong", "other"}
def call(job):
    key, it, text = job; last = ""
    for _ in range(3):
        out = subprocess.run(["claude", "-p", INSTR % (it["target"], it["prompt"], text), "--model", "sonnet", "--output-format", "text"], capture_output=True, text=True, timeout=400).stdout
        m = re.search(r"\{.*\}", out, re.S); last = out[:200]
        try:
            d = json.loads(m.group(0)); sp = d["spans"]
            assert all(s["label"] in LABELS for s in sp) and norm("".join(s["text"] for s in sp)) == norm(text) and isinstance(d["corrected"], str)
            return {"key": key, "spans": sp, "corrected": d["corrected"]}
        except Exception: continue
    return {"key": key, "error": last}
rows = [json.loads(l) for l in open(R / "cw14/nla_split.jsonl")]
ids = sorted({r["id"] for r in rows}); random.Random(0).shuffle(ids)
ids = ids[: int(sys.argv[1])] if len(sys.argv) > 1 else ids
jobs = [(f"{r['id']}|{k}", bank[r["id"]], s) for r in rows if r["cond"] == "full" and r["id"] in ids for k, s in enumerate(r["samples"])]
done = {json.loads(l)["key"] for l in open(OUT) if '"error"' not in l} if OUT.exists() else set()
jobs = [j for j in jobs if j[0] not in done]
with ThreadPoolExecutor(8) as ex, open(OUT, "a") as fh:
    for res in ex.map(call, jobs):
        fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush(); print(res["key"], "ERR" if "error" in res else "ok", flush=True)
