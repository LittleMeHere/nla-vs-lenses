"""CW-15: what each reader writes from each part of the activation (CW-14's saved readouts; no GPU, no LLM).
Per readout, deterministic measures:
  bridge / target      : names the hidden step / the answer (word match, as CW-12)
  prompt words         : share of the prompt's content words found in the readout (bridge and answer words excluded)
  last word (NLA only) : the readout quotes a final token/word/phrase/fragment and it ends in the prompt's last word
  foreign names        : capitalised words (not sentence-initial) that are not in the prompt, bridge or target; a proxy for
                         made-up entities. Latin script only.
Usage: python scripts/cw15_readout_content.py -> runs/cw15/content.txt, runs/cw15/content.json"""
import json, re
from pathlib import Path
import numpy as np
R = Path(__file__).parent.parent / "runs"; D = R / "cw14"; OUT = R / "cw15"; OUT.mkdir(exist_ok=True)
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
STOP = set("the a an of to in on at for and or is are was were be by with as that this it its from into about fact what which who whose".split())
def names(t, ws): return any(re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", t.lower()) for w in ws)
def cw(it):
    drop = set(re.findall(r"[a-z0-9]+", " ".join(it["intermediates"] + [it["target"]]).lower()))
    return {w for w in re.findall(r"[a-z0-9]+", it["prompt"].lower()) if w not in STOP and w not in drop and len(w) > 1}
def rec(t, c): return len(c & set(re.findall(r"[a-z0-9]+", t.lower()))) / len(c)
GENERIC = set("the a an in it this that what which who why how if is are was were for final here note answer question fact trivia quiz correct true false none all one two three first second no yes and or but so because unlike however since when where q".split())
def foreign(t, it):
    known = set(re.findall(r"[a-z]+", (it["prompt"] + " " + " ".join(it["intermediates"]) + " " + it["target"]).lower()))
    out = set()
    for m in re.finditer(r"(?<![.!?:*\-/\n\"“(]\s)(?<!^)(?<![A-Za-z])([A-Z][a-z]{2,})", t):
        w = m.group(1).lower()
        if w not in known and w not in GENERIC and not any(w.startswith(k[:5]) for k in known if len(k) >= 5): out.add(w)
    return len(out)
def lastword(t, it):
    qs = re.findall(r"[Ff]inal (?:token|word|phrase|fragment|clause|words)[^\"“]{0,12}[\"“]([^\"”]{1,200})[\"”]", t)
    if not qs: return None
    pw = re.findall(r"[a-z0-9]+", it["prompt"].lower())[-1]
    return float(any((re.findall(r"[a-z0-9]+", q.lower()) or [""])[-1] == pw for q in qs))
def boot(x):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
CONDS = [("full", "whole"), ("J1024", "J part"), ("N1024", "rest"), ("R1024", "random part"), ("R1024s1", "random part 2"), ("RN1024", "all but random"), ("P1024", "PCA part"), ("PN1024", "all but PCA")]
res = {}; L = []
for rd, fs in (("oracle", ["olens_split", "olens_ext"]), ("NLA", ["nla_split", "nla_ext"])):
    rows = [json.loads(l) for f in fs for l in open(D / f"{f}.jsonl")]
    L.append(f"\n{rd.upper()} (70 prompts x 2 samples; mean over prompts [95% bootstrap]): part | names bridge | names answer | prompt words | foreign names per readout | chars" + (" | quotes a final word (share of readouts) | of those, it is the prompt's last word" if rd == "NLA" else ""))
    for c, lab in CONDS:
        rs = [r for r in rows if r["cond"] == c]; per = {}
        for r in rs:
            it = bank[r["id"]]; S = r["samples"]
            lw = [lastword(s, it) for s in S]
            per[r["id"]] = dict(bridge=np.mean([names(s, it["intermediates"]) for s in S]), target=np.mean([names(s, [it["target"]]) for s in S]),
                                prompt=np.mean([rec(s, cw(it)) for s in S]), foreign=np.mean([foreign(s, it) for s in S]), chars=np.mean([len(s) for s in S]),
                                quotes=np.mean([x is not None for x in lw]), last=(np.mean([x for x in lw if x is not None]) if any(x is not None for x in lw) else np.nan))
        agg = {k: boot([v[k] for v in per.values() if not np.isnan(v[k])]) for k in ("bridge", "target", "prompt", "foreign", "chars", "quotes", "last")}
        res[f"{rd}|{c}"] = {k: v for k, v in agg.items()}
        f = lambda k: f"{agg[k][0]:.2f} [{agg[k][1]:.2f}, {agg[k][2]:.2f}]"
        L.append(f"  {lab}: {f('bridge')} | {f('target')} | {f('prompt')} | {f('foreign')} | {agg['chars'][0]:.0f}" + (f" | {agg['quotes'][0]:.2f} | {f('last')}" if rd == "NLA" else ""))
open(OUT / "content.txt", "w").write("\n".join(L) + "\n"); json.dump(res, open(OUT / "content.json", "w"), indent=1); print("\n".join(L))
