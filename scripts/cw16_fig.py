"""CW-16 figure and examples. -> figs/cw16_rebuild_by_text.png, to_read/rebuild_examples_5_random.md"""
import json, math, random
from pathlib import Path
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = Path(__file__).parent.parent; R = ROOT / "runs"; SC = math.sqrt(5120)
for f in (ROOT / "scripts/fonts").glob("*.ttf"): fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Public Sans", "font.size": 11})
INK, MUTED, GRID, BG = "#14213d", "#5b6472", "#e3e6ec", "#fbfbf8"
sc = lambda v: v.float() / v.float().norm() * SC
rec = {**torch.load(R / "cw16/recon.pt"), **torch.load(R / "cw16/recon_ctrl.pt")}
T = {json.loads(l)["key"]: json.loads(l)["text"] for f in ("texts.jsonl", "texts_ctrl.jsonl") for l in open(R / "cw16" / f)}
G = {c["id"]: sc(c["h"]) for c in torch.load(R / "cw14/resid_split.pt") if c["cond"] == "full"}; mu = torch.stack(list(G.values())).mean(0)
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
fve = lambda k: 1 - float((sc(rec[k]) - G[k.split("|")[1]]).pow(2).sum() / (G[k.split("|")[1]] - mu).pow(2).sum())
def boot(x):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(1); return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)
ROWS = [("orig", "The NLA's write-up, unchanged", "#eb6834"), ("prompt", "The prompt itself", "#9aa3b2"), ("factual", "False statements removed", "#eb6834"),
        ("randdel_factual", "Random sentences removed\n(same length, control)", "#9aa3b2"), ("content", "Content sentences only", "#eb6834"), ("form", "Form sentences only", "#eb6834"), ("shuf", "Another prompt's write-up", "#9aa3b2")]
fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG); fig.subplots_adjust(left=0.30, right=0.95, top=0.76, bottom=0.13)
for n, (v, lab, col) in enumerate(ROWS):
    per = {}
    for k in rec:
        if k.split("|")[0] == v: per.setdefault(k.split("|")[1], []).append(fve(k))
    m, lo, hi = boot([np.mean(x) for x in per.values()]); y = len(ROWS) - 1 - n
    ax.barh(y, m, height=0.6, color=col, zorder=3); ax.plot([lo, hi], [y, y], color=INK, lw=1.3, zorder=4)
    ax.text(hi + 3 if m >= 0 else lo - 3, y, f"{m:+.0f}%", va="center", ha="left" if m >= 0 else "right", fontsize=11, color=INK)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[1] for r in ROWS][::-1], fontsize=11.5, color=INK); ax.axvline(0, color=INK, lw=1, zorder=2)
ax.set_xlim(-150, 70); ax.set_xticks([-100, -50, 0, 50]); ax.set_xticklabels(["−100%", "−50%", "0 (as good as guessing\nthe average activation)", "+50%"], color=MUTED, fontsize=9.5)
for x in (-100, -50, 50): ax.axvline(x, color=GRID, lw=0.9, zorder=0)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
fig.text(0.04, 0.925, "Removing false statements hurts the rebuild no more than removing random sentences", fontsize=14.5, fontweight="bold", color=INK)
fig.text(0.04, 0.868, "How much of the real layer-42 activation the NLA's reconstructor rebuilds from each text, beyond the average activation. 70 multihop prompts,\n2 write-ups each. Edits by an LLM, not validated; the control was added after seeing the first results. Lines: 95% bootstrap over prompts.", fontsize=9.5, color=MUTED, va="top", linespacing=1.45)
fig.savefig(ROOT / "figs/cw16_rebuild_by_text.png", facecolor=BG)
ids = sorted(G); random.seed(0); pick = random.sample(ids, 5); L = ["# The rebuild experiment: 5 random prompts", "",
"For each prompt: the NLA's original write-up, then edited versions of it. The number after each is how much of the real activation the reconstructor rebuilt from that text, beyond the average activation (higher is better, 0 means no better than the average, negative means worse).", "",
"What to look for: are the 'false statements removed' versions sensible? Do they look like they lost something useful, or just got shorter?", ""]
for n, i in enumerate(pick, 1):
    it = bank[i]; L += [f"## {n}. {it['prompt']}", "", f"Hidden step: **{', '.join(it['intermediates'])}**. Answer: **{it['target']}**.", ""]
    for v, lab in (("orig", "Original write-up"), ("factual", "False statements removed"), ("randdel_factual", "Random sentences removed (control)"), ("content", "Content only"), ("form", "Form only")):
        k = f"{v}|{i}|0"
        if k in T: L += [f"**{lab}** ({100*fve(k):+.0f}%)", "", "> " + T[k].strip().replace("\n\n", "\n").replace("\n", "\n> "), ""]
open(ROOT / "to_read/rebuild_examples_5_random.md", "w", encoding="utf-8").write("\n".join(L)); print("saved")
