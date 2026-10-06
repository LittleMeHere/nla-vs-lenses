"""CW-16: LLM rewrites of the NLA's whole-activation write-ups (CW-14): hallucinations removed, content only, form only.
One `claude -p --model sonnet` call per write-up, cached in runs/cw16/rewrites.jsonl. The call sees the prompt and the write-up.
Usage: python scripts/cw16_rewrites.py [N]"""
import json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R = Path(__file__).parent.parent / "runs"; OUT = R / "cw16/rewrites.jsonl"
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
INSTR = """Below is a PROMPT that a language model was reading, and a DESCRIPTION that an interpretability tool wrote about the model's internal state at the last word of that prompt. The description may contain false statements.

Produce three edited versions of the DESCRIPTION. Edit minimally: delete or trim, keep the original wording and order wherever you keep something, and do not add new information.

1. "factual": keep only statements that are true of the PROMPT or of its correct continuation. Remove anything the PROMPT contradicts or does not support (wrong entities, wrong format claims, invented quotes).
2. "content": from the ORIGINAL description, keep only what says what the text is about: topics, entities, facts, the expected answer. Remove everything about how the text is written (format, genre, sentence structure, which token or phrase it ends on).
3. "form": from the ORIGINAL description, keep only what says how the text is written: format, genre, sentence structure, where it stops, what kind of word comes next. Remove the specific topics, entities and facts (replace a named thing by nothing, not by a placeholder).

A version may be an empty string if nothing qualifies. Return ONLY JSON: {"factual": "...", "content": "...", "form": "..."}

PROMPT:
%s

DESCRIPTION:
%s
"""
def call(job):
    key, prompt, text = job
    for _ in range(3):
        out = subprocess.run(["claude", "-p", INSTR % (prompt, text), "--model", "sonnet", "--output-format", "text"], capture_output=True, text=True, timeout=300).stdout
        m = re.search(r"\{.*\}", out, re.S)
        try:
            d = json.loads(m.group(0)); assert set(d) >= {"factual", "content", "form"}
            return {"key": key, **{k: d[k] for k in ("factual", "content", "form")}}
        except Exception: continue
    return {"key": key, "error": out[:300]}
rows = [json.loads(l) for l in open(R / "cw14/nla_split.jsonl")]
jobs = [(f"{r['id']}|{k}", bank[r["id"]]["prompt"], s) for r in rows if r["cond"] == "full" for k, s in enumerate(r["samples"])]
done = {json.loads(l)["key"] for l in open(OUT)} if OUT.exists() else set()
jobs = [j for j in jobs if j[0] not in done][: int(sys.argv[1]) if len(sys.argv) > 1 else None]
with ThreadPoolExecutor(6) as ex, open(OUT, "a") as fh:
    for res in ex.map(call, jobs):
        fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush(); print(res["key"], "ERR" if "error" in res else "ok", flush=True)
