"""Build a self-contained page of the reader-content and rebuild results with highlighted write-ups (CW-15, CW-16b).
Usage: python scripts/cw16b_page.py OUT.html"""
import html, json, math, random, sys
from pathlib import Path
import numpy as np, torch
R = Path(__file__).parent.parent / "runs"; SC = math.sqrt(5120); sc = lambda v: v.float() / v.float().norm() * SC
rec = torch.load(R / "cw16b/recon.pt"); G = {c["id"]: sc(c["h"]) for c in torch.load(R / "cw14/resid_split.pt") if c["cond"] == "full"}; mu = torch.stack(list(G.values())).mean(0)
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
T = {json.loads(l)["key"]: json.loads(l)["text"] for l in open(R / "cw16b/texts.jsonl")}
LAB = {json.loads(l)["key"]: json.loads(l) for l in open(R / "cw16b/labels.jsonl") if '"error"' not in l}
def fve(k):
    g = G[k.split("|")[1]]; return 100 * (1 - float((sc(rec[k]) - g).pow(2).sum() / (g - mu).pow(2).sum()))
NAMES = {"format": "format", "restate_ok": "restates the prompt, right", "restate_wrong": "restates the prompt, inexact", "content_ok": "claim, right", "content_wrong": "claim, wrong", "other": "other"}
keys = sorted(k for k in LAB if k.endswith("|0") and f"corrected|{k}" in rec); random.Random(3).shuffle(keys); pick = keys[:5]
E = html.escape; S = []
for k in pick:
    it = bank[k.split("|")[0]]; spans = "".join(f'<span class="sp {s["label"]}" title="{NAMES[s["label"]]}">{E(s["text"])}</span>' for s in LAB[k]["spans"])
    rows = "".join(f"<tr><td>{lab}</td><td class=num>{fve(f'{v}|{k}'):+.0f}%</td></tr>" for v, lab in (("orig", "Original write-up"), ("corrected", "Wrong parts corrected in place"), ("drop_content_wrong", "Wrong claims removed"), ("drop_restate_wrong", "Inexact restatements removed"), ("drop_restate", "All restatements removed")) if f"{v}|{k}" in rec)
    S.append(f'<article class=sample><p class=prompt>{E(it["prompt"].strip())}</p><p class=meta>Hidden step: <b>{E(", ".join(it["intermediates"]))}</b> · Answer: <b>{E(it["target"])}</b></p><div class=two><div class=writeup>{spans}</div><table class=mini><thead><tr><th>Text given to the reconstructor</th><th class=num>Rebuilt</th></tr></thead><tbody>{rows}</tbody></table></div><details><summary>Corrected version</summary><div class=writeup>{E(T[f"corrected|{k}"])}</div></details></article>')
BARS = [("All restatements of the prompt", -51, -59, -43), ("Only the inexact restatements", -18, -24, -13), ("Claims about the topic and answer", -11, -16, -7), ("Only the wrong claims", -2, -3, 0), ("Statements about the format", 10, 6, 15)]
bars = "".join(f'<div class=bar><span class=bl>{l}</span><span class=track><i class="{"neg" if m < 0 else "pos"}" style="{"right" if m < 0 else "left"}:{"25%" if m < 0 else "75%"};width:{abs(m) * 1.25:.1f}%"></i></span><span class=bv>{m:+d} <small>[{lo:+d}, {hi:+d}]</small></span></div>' for l, m, lo, hi in BARS)
page = f"""<title>Reader Content Notes</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700&family=IBM+Plex+Mono&display=swap">
<style>
:root {{ --bg:#fbfbf8; --surface:#ffffff; --fg:#14213d; --fg2:#4c5566; --line:#dfe2e8; --accent:#2a78d6; --neg:#eb6834; --pos:#0f8f83;
  --format:#e6e8ee; --rok:#cfe6d8; --rwrong:#ffe2b8; --cok:#d6e4fb; --cwrong:#f9cfc9; --other:#eeeeee; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#101318; --surface:#181c23; --fg:#f2f4f8; --fg2:#aab2c0; --line:#2b313c; --accent:#7fb3f2; --neg:#f08a5d; --pos:#4cc4b6;
  --format:#343a46; --rok:#1f4a36; --rwrong:#5c4310; --cok:#1f3a66; --cwrong:#63261f; --other:#2a2a2a; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#101318; --surface:#181c23; --fg:#f2f4f8; --fg2:#aab2c0; --line:#2b313c; --accent:#7fb3f2; --neg:#f08a5d; --pos:#4cc4b6;
  --format:#343a46; --rok:#1f4a36; --rwrong:#5c4310; --cok:#1f3a66; --cwrong:#63261f; --other:#2a2a2a; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.55 "Public Sans", system-ui, sans-serif; }}
main {{ max-width:820px; margin:0 auto; padding-inline:16px; padding-block:28px 64px; display:grid; gap:20px; }}
h1 {{ font-size:1.7rem; line-height:1.2; margin:0; text-wrap:balance; }} h2 {{ font-size:1.2rem; margin:14px 0 0; text-wrap:balance; }}
p {{ margin:0; max-width:70ch; }} .sub {{ color:var(--fg2); }} .eyebrow {{ font:12px "IBM Plex Mono", monospace; letter-spacing:.05em; text-transform:uppercase; color:var(--fg2); }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }} th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }} th {{ color:var(--fg2); font-weight:600; font-size:13px; }} .num {{ text-align:right; }}
.bar {{ display:grid; grid-template-columns:minmax(120px, 230px) 1fr 110px; gap:10px; align-items:center; padding:5px 0; }} .bl {{ font-size:14px; }} .bv {{ font:13px "IBM Plex Mono", monospace; }} .bv small {{ color:var(--fg2); }}
.track {{ position:relative; height:16px; background:linear-gradient(to right, transparent calc(75% - 1px), var(--fg2) calc(75% - 1px), var(--fg2) 75%, transparent 75%); }} .track i {{ position:absolute; top:2px; bottom:2px; border-radius:2px; }} .neg {{ background:var(--neg); }} .pos {{ background:var(--pos); }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px; font-size:13px; }} .legend span {{ padding:2px 8px; border-radius:3px; }}
.sample {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:14px; display:grid; gap:10px; min-width:0; }} .prompt {{ font-weight:600; }} .meta {{ color:var(--fg2); font-size:13.5px; }}
.two {{ display:grid; grid-template-columns:minmax(0, 1.6fr) minmax(0, 1fr); gap:16px; align-items:start; }} @media (max-width:640px) {{ .two {{ grid-template-columns:1fr; }} .bar {{ grid-template-columns:1fr; gap:2px; }} }}
.writeup {{ font:13px/1.7 "IBM Plex Mono", monospace; white-space:pre-wrap; overflow-wrap:anywhere; min-width:0; }} .sp {{ padding:1px 0; border-radius:2px; }}
.format {{ background:var(--format); }} .restate_ok {{ background:var(--rok); }} .restate_wrong {{ background:var(--rwrong); }} .content_ok {{ background:var(--cok); }} .content_wrong {{ background:var(--cwrong); }} .other {{ background:var(--other); }}
.mini td, .mini th {{ font-size:13px; padding:4px 6px; }} details summary {{ cursor:pointer; color:var(--accent); font-size:13px; }} ul {{ margin:0; padding-left:20px; max-width:70ch; }} li {{ margin:3px 0; }}
</style>
<main>
<div class=eyebrow>Qwen3.6-27B · WorkspaceBench multihop and poetry · layer 42 · exploratory · 7 Oct 2026</div>
<h1>What the NLA writes, and what its reconstructor uses</h1>
<p>Three checks on the NLA's write-ups for multihop prompts, following the TA's suggestions. The NLA's reconstructor is the one in <code>ceselder/qwen3.6-27b-nla-rl</code> (<code>ar_reconstructor/</code>); the verbalizer is step 400. Our loader gives the same vectors as EasyNLA's own loader on all 3,376 texts (cosine at least 0.999999).</p>

<h2>1. Removing the NLA's restatements of the prompt hurts the rebuild most</h2>
<p class=sub>70 prompts, 2 write-ups each. Each row removes one kind of text and rebuilds the activation from what is left. The number is the change in variance explained, in points, compared with removing random pieces of the same length (3 random draws per row). Brackets: 95% bootstrap over prompts.</p>
<div>{bars}</div>
<div class=wrap><table><thead><tr><th>Text given to the reconstructor</th><th class=num>Variance explained</th><th class=num>In the J part</th><th class=num>In the rest</th></tr></thead><tbody>
<tr><td>Original write-up</td><td class=num>41%</td><td class=num>60%</td><td class=num>31%</td></tr>
<tr><td>Wrong parts corrected in place (same length)</td><td class=num>47%</td><td class=num>62%</td><td class=num>38%</td></tr>
<tr><td>Another prompt's write-up</td><td class=num>−105%</td><td class=num></td><td class=num></td></tr></tbody></table></div>
<p>Correcting the wrong parts gains 5 points [3, 8], mostly outside the J part (+7 against +2). Variance explained is measured against the mean of the 70 target activations.</p>

<h2>2. What a write-up is made of</h2>
<p class=sub>Share of characters by kind, first 20 write-ups labelled.</p>
<div class=legend><span class=content_ok>claim, right 29%</span><span class=format>format 26%</span><span class=restate_wrong>restates the prompt, inexact 24%</span><span class=restate_ok>restates the prompt, right 11%</span><span class=content_wrong>claim, wrong 9%</span></div>
<p>Compared with the oracle lens on the 100-prompt run (whole activation, one write-up each): both carry about the same task content (4 to 5 words on the hidden step and answer, 12 to 14 prompt words). The NLA adds about 33 format words per write-up; 41% of its content words are format words against 4% for the oracle lens (word-list count).</p>

<h2>3. From outside the J part the NLA keeps the form and loses the content</h2>
<div class=wrap><table><thead><tr><th>NLA given</th><th class=num>Names the hidden step</th><th class=num>Names the answer</th><th class=num>Quoted last word is right</th></tr></thead><tbody>
<tr><td>Whole activation</td><td class=num>85%</td><td class=num>89%</td><td class=num>98%</td></tr><tr><td>J part (16% of squared norm)</td><td class=num>88%</td><td class=num>89%</td><td class=num>83%</td></tr><tr><td>The rest (84%)</td><td class=num>23%</td><td class=num>9%</td><td class=num>98%</td></tr><tr><td>Random part</td><td class=num>49%</td><td class=num>36%</td><td class=num>65%</td></tr></tbody></table></div>
<p class=sub>The last column counts write-ups that quote a final word or phrase (about 80% do). The 70 prompts end in "is" (49), "the" (20) or "of" (1); scored against another prompt's last word the rate is 50 to 60%.</p>

<h2>4. Poetry: the NLA names the rhyme word more than other rhyme words, which does not show it reads a plan</h2>
<p class=sub>WorkspaceBench poetry, all 100 prompts, read at the newline ending line one. The target is the word the model ends line two with. Oracle lens through the benchmark's producer at layers 40 and 44; NLA at layer 42, one write-up per prompt. Scored by the benchmark's judge. No J part / rest split yet.</p>
<div class=wrap><table><thead><tr><th>Reader</th><th class=num>Names the target</th><th class=num>Names another rhyme word (3 sets)</th><th class=num>Difference</th></tr></thead><tbody>
<tr><td>NLA, layer 42</td><td class=num>30 of 100</td><td class=num>6, 8, 10</td><td class=num>+0.22 [0.14, 0.30]</td></tr>
<tr><td>Oracle lens, layer 40</td><td class=num>4 of 100</td><td class=num>3, 1, 6</td><td class=num></td></tr>
<tr><td>Oracle lens, layer 44</td><td class=num>11 of 100</td><td class=num>4, 2, 6</td><td class=num></td></tr>
<tr><td>Oracle lens, either layer</td><td class=num>15 of 100</td><td class=num>7, 3, 10</td><td class=num>+0.08 [0.01, 0.16]</td></tr></tbody></table></div>
<p>The other rhyme words are three words per prompt that rhyme with line one's last word and are closest to the target in word frequency, scored by the same judge on the same write-ups. Brackets: 95% bootstrap over prompts.</p>
<p>Every NLA write-up quotes a version of line one. In 43 of 100 a quoted line ends in the real last word. In 23 of 100 a quoted line ends in the target instead (the real line ends "wheat", the write-up quotes a line ending "heat"); that is 23 of the NLA's 30 passes.</p>
<p>What is not shown: that either reader reads a planned word. The read position comes before line two, so the target can only be a likely rhyme there, and a reader that knows the last word could prefer the same rhyme the model does. The control words are matched on frequency, not on how natural a rhyme they are. The benchmark's text-only floor for poetry is 71% [62, 80], measured at an older read position; it has not been measured at this one. The control was chosen after seeing the scores.</p>

<h2>Five write-ups, picked at random</h2>
<p class=sub>Whole-activation NLA write-ups with each span coloured by its label. The table beside each shows how much of the activation the reconstructor rebuilds from each version of that write-up.</p>
<div class=legend><span class=format>format</span><span class=restate_ok>restates the prompt, right</span><span class=restate_wrong>restates the prompt, inexact</span><span class=content_ok>claim, right</span><span class=content_wrong>claim, wrong</span></div>
{"".join(S)}

<h2>How it was done, and limits</h2>
<ul><li>Spans were cut and labelled by a model (one call per write-up; the spans must reproduce the write-up exactly; 138 of 140 passed). A few were read by hand; the labels are not validated beyond that.</li>
<li>Versions are built from the labels by code. Random-removal controls match the removed length within 1 to 2 points.</li>
<li>Sections 1 to 3: one prompt family, one model, one reconstructor. Removing spans leaves broken sentences, which the controls share.</li>
<li>Code and data: <code>scripts/cw15_*</code>, <code>scripts/cw16b_*</code>, <code>runs/cw15</code>, <code>runs/cw16b</code> in github.com/LittleMeHere/nla-vs-lenses. The poetry files (<code>scripts/cw19_*</code>, <code>runs/cw19</code>) are not in that repo yet.</li></ul>
</main>"""
open(sys.argv[1], "w", encoding="utf-8").write(page); print("written", len(page))
