"""Role-aware claim judge for brew readouts, a prompt-blind judge (it
that only sees the readout and the item's candidate colours). For each readout: which colour does the
text put in the START role, the INTERMEDIATE role (colour after one stir), and the ANSWER role
(colour after two stirs)? Each role: a colour, "none", "cannot tell" or "conflicting".
Judge: JUDGE=claude (`claude -p`) or JUDGE=gemini (Gemini 3.8 Flash on Vertex AI). Not validated against human labels.
Usage: python judge_roles.py ITEMS.json READOUTS.jsonl OUT.jsonl [text_field=samples]"""
import json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path
PROMPT = """You read text produced by an interpretability tool that describes what a language model was thinking while solving a colour puzzle: a potion changes colour by fixed rules each time it is stirred, it is stirred twice, and the model must give the final colour.
The puzzle's candidate colours are: %s.
For the text below, say which colour the text puts in each role:
- start: the colour the potion starts as
- intermediate: the colour after the first stir
- answer: the colour after the second stir (the final answer)
For each role give one candidate colour, or "none" (the text makes no claim about that role), "cannot tell" (it is unclear which colour it means), or "conflicting" (it claims more than one colour for that role).
Return ONLY JSON: {"start": ..., "intermediate": ..., "answer": ...}

TEXT:
<<<%s>>>"""
JUDGE = os.environ.get("JUDGE", "claude")   # "claude" (claude -p) or "gemini" (Vertex AI, gemini-3.8-flash; project from GCP_PROJECT or gcloud config)
_tok = {"t": None, "at": 0.0}
def ask(prompt):
    if JUDGE == "claude":
        return subprocess.run(["claude", "-p", prompt, "--output-format", "text"], capture_output=True, text=True, timeout=300).stdout
    if time.time() - _tok["at"] > 1800:
        _tok["t"] = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True).stdout.strip(); _tok["at"] = time.time()
    project = os.environ.get("GCP_PROJECT") or subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True).stdout.strip()
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/gemini-3.8-flash:generateContent"
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Authorization": "Bearer " + _tok["t"], "Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=120))
    return d["candidates"][0]["content"]["parts"][0]["text"]
def main(items_p, readouts_p, out_p):
    items = {it["name"]: it for it in json.load(open(items_p))["items"]}
    done = set()
    if Path(out_p).exists():
        done = {(r["id"], r.get("cond"), r["k"]) for r in map(json.loads, open(out_p))}
    with open(out_p, "a") as fh:
        for r in map(json.loads, open(readouts_p)):
            it = items[r["id"]]
            cands = sorted(set(it["options_adjacent"][0]) | {it["start"], it["answer"], *it["intermediates"]})
            for k, text in enumerate(r["samples"]):
                key = (r["id"], r.get("cond"), k)
                if key in done: continue
                lab = None
                for attempt in range(3):  # retry on timeout, empty or unparseable replies
                    try:
                        raw = ask(PROMPT % (", ".join(cands), text[:4000]))
                        m = re.search(r"\{.*\}", raw, re.S)
                        lab = json.loads(m.group(0)) if m else None
                    except Exception:
                        lab = None
                    if lab: break
                fh.write(json.dumps({"id": r["id"], "cond": r.get("cond"), "partner": r.get("partner"), "k": k,
                                     "labels": lab, "truth": {"start": it["start"], "intermediate": it["intermediates"][0], "answer": it["answer"]}}) + "\n"); fh.flush()
if __name__ == "__main__":
    main(*sys.argv[1:4])
