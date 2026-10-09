"""CW-20: label every span of an oracle lens write-up (layer 44), and write one minimally corrected version.
One `claude -p --model sonnet` call per write-up; cached in runs/cw20/labels.jsonl. Spans must reproduce the write-up
exactly (whitespace aside), or the call is retried. Usage: python scripts/cw20_label.py [N_PROMPTS]"""
import json, random, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R = Path(__file__).parent.parent / "runs"; OUT = R / "cw20/labels.jsonl"
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
INSTR = """A language model was reading the PROMPT below. An interpretability tool then wrote the READOUT: a few bullets, each a guess at how the text continues after the PROMPT. The correct continuation of the PROMPT is: %s

Task 1. Cut the READOUT into consecutive spans and label each one. The spans, joined in order, must reproduce the READOUT exactly, character for character (keep the "- " list marks, punctuation, quotes and line breaks inside the spans). Cut inside a bullet where it mixes kinds. Labels:
- "answer_ok": gives the correct continuation of the PROMPT (also in another language, or inside brackets).
- "answer_wrong": gives a wrong answer, or an unfilled placeholder such as "[State Name]".
- "about_ok": restates the PROMPT's question, or says something true about the PROMPT's own subject and entities.
- "about_wrong": the same kind of thing, but false or changed in meaning.
- "invented": new questions, entities or scenarios that the PROMPT does not contain.
- "format": list marks, markup and framing with no content ("- ", "Answer:", "(Clue:").

Task 2. Write "corrected": the whole READOUT with every answer_wrong and about_wrong span minimally rewritten so that it is true (put the correct answer in place of a wrong one or a placeholder). Change nothing else. Keep the line breaks and keep the length about the same.

Return ONLY JSON: {"spans": [{"text": "...", "label": "..."}, ...], "corrected": "..."}

PROMPT:
%s

READOUT:
%s
"""
norm = lambda s: re.sub(r"\s+", "", s)
LABELS = {"answer_ok", "answer_wrong", "about_ok", "about_wrong", "invented", "format"}
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
rows = [json.loads(l) for l in open(R / "cw20/olens_L44.jsonl")]
ids = sorted({r["id"] for r in rows}); random.Random(0).shuffle(ids)
ids = ids[: int(sys.argv[1])] if len(sys.argv) > 1 else ids
jobs = [(f"{r['id']}|{k}", bank[r["id"]], s) for r in rows if r["id"] in ids for k, s in enumerate(r["samples"])]
done = {json.loads(l)["key"] for l in open(OUT) if '"error"' not in l} if OUT.exists() else set()
jobs = [j for j in jobs if j[0] not in done]
with ThreadPoolExecutor(8) as ex, open(OUT, "a") as fh:
    for res in ex.map(call, jobs):
        fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush(); print(res["key"], "ERR" if "error" in res else "ok", flush=True)
