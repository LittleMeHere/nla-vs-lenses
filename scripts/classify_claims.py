"""Classify what a prose readout (NLA or oracle lens) says that a token lens (J-Lens top-10) at
the same cell does not. Stage A: split each readout into concepts (LLM). Then every concept gets
one bucket, cheapest test first:
  jlens      - concept appears (word-level, inflection-tolerant) among the J-Lens top-10 tokens
  intermed   - concept names a bank intermediate (workspace content, by construction)
  past_ctx   - concept's content words occur in the prompt text up to the read position
  next_tok   - concept's content words are in the model's top-10 next tokens at that position
  format_meta- mostly words about the prompt's form (format, clause, riddle...), not content
  other      - none of the above (candidate for hallucination or present-but-unused; judged later)
Deterministic tests only here; the LLM is used for Stage A alone (via `claude -p`, no API key).
Usage: python classify_claims.py BANK_ITEMS.json PROSE.jsonl JLENS.jsonl [NEXT.jsonl] OUT.jsonl"""
import json, re, subprocess, sys
from collections import Counter
from pathlib import Path

STAGE_A = ("You are given text produced by an interpretability lens that verbalizes a language "
           "model's internal state. Break the text into its UNIQUE concepts: short noun phrases "
           "naming each distinct thing, property, action or entity the text asserts. Merge "
           "duplicates and near-duplicates. Return ONLY JSON: {\"concepts\": [string, ...]}.\n\nText:\n")
STOP = set("the a an of to in on at for and or is are was were be by with as that this it its from into about".split())

def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]

def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w

def overlap(concept, pool):
    """pool: a string or list of strings; both sides word-normalized and stemmed; a concept word
    matches a pool word if either stem contains the other (brazil ~ brazilian)."""
    pw = {stem(w) for w in words(pool if isinstance(pool, str) else " ".join(pool))}
    cw = {stem(w) for w in words(concept)}
    return any(a == b or (len(a) >= 4 and len(b) >= 4 and (a in b or b in a)) for a in cw for b in pw)

META = set("format prompt question sentence clause token completion answer trivia riddle quiz structure "
           "pattern phrase setup continuation predicate declarative fill blank template instruction "
           "style register statement noun verb grammatical syntactic punctuation word tone".split())

def is_meta(concept):
    cw = {stem(w) for w in words(concept)}
    return bool(cw) and len(cw & {stem(w) for w in META}) / len(cw) >= 0.5

def concepts_of(text, cache):
    if text in cache:
        return cache[text]
    out = subprocess.run(["claude", "-p", STAGE_A + text, "--output-format", "text"],
                         capture_output=True, text=True, timeout=180).stdout
    m = re.search(r"\{.*\}", out, re.S)
    cache[text] = json.loads(m.group(0))["concepts"] if m else []
    return cache[text]

def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def main(bank, prose, jlens, nxt, out):
    items = {it["name"]: it for it in json.load(open(bank))["items"]}
    jl = {(r["id"], r["pos"]): r["tokens"] for r in rows(jlens)}  # one reference layer per file
    nx = {(r["id"], r["pos"]): r["tokens"] for r in rows(nxt)} if nxt else {}
    cache, tally, lines = {}, Counter(), []
    prior = None  # --rebucket: reuse Stage A concepts from an existing OUT file (no LLM calls)
    if "--rebucket" in sys.argv and Path(out).exists():
        prior = {}
        for x in rows(out):
            prior.setdefault((x["id"], x["layer"], x["pos"]), []).append(x["concept"])
    for r in rows(prose):
        key = (r["id"], r["layer"], r["pos"]); it = items[r["id"]]
        prompt_words = words(it["prompt"]); jtoks = [t.strip() for t in jl.get((r["id"], r["pos"]), [])]
        for c in (prior[key] if prior else concepts_of(" ".join(r["samples"]), cache)):
            if overlap(c, it.get("intermediates", []) + [it.get("target", "")]): b = "intermed"
            elif overlap(c, jtoks): b = "jlens"
            elif overlap(c, it["prompt"]): b = "past_ctx"
            elif nx and overlap(c, nx.get((r["id"], r["pos"]), [])): b = "next_tok"
            elif is_meta(c): b = "format_meta"
            else: b = "other"
            tally[b] += 1; lines.append({"id": r["id"], "layer": r["layer"], "pos": r["pos"], "concept": c, "bucket": b})
    Path(out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n")
    n = sum(tally.values()); print({k: f"{v} ({v/n:.0%})" for k, v in tally.most_common()}, "n =", n)

if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], a[2], a[3] if len(a) == 5 else None, a[-1])
