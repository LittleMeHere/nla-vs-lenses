"""CW-13 figures. Usage: python scripts/cw13_fig.py  -> figs/cw13_bridge_rank.png, figs/cw13_where_is_the_bridge.png"""
import json, re
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = Path(__file__).parent.parent; R = ROOT / "runs"
for f in (ROOT / "scripts/fonts").glob("*.ttf"): fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Public Sans", "font.size": 11})
INK, MUTED, GRID, BG = "#14213d", "#5b6472", "#e3e6ec", "#fbfbf8"
C = {"jlens": "#0f8f83", "tl": "#7a5aa6", "oracle": "#2a78d6", "nla": "#eb6834"}
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
def names(t, inter): return any(re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", t.lower()) for w in inter)
jr = json.load(open(R / "jrank/ranks.json")); tr = json.load(open(R / "jrank/template_ranks.json"))
# ---------- figure 1: how far down J-Lens's list is the bridge ----------
ids = list(jr["whole50"]); n = len(ids)
jl = np.array([jr["whole50"][i]["rank"] or 10**9 for i in ids]); tl = np.array([tr["whole50"][i]["rank"] or 10**9 for i in ids])
def rate(f):
    rows = [json.loads(l) for l in open(f)]; rows = [r for r in rows if r["cond"] == "intact"]
    return np.mean([np.mean([names(s, bank[r["id"]]["intermediates"]) for s in r["samples"]]) for r in rows])
ro, rn = rate(R / "cw11/olens_cw11.jsonl"), rate(R / "cw11/nla_cw11.jsonl")
ks = np.unique(np.round(np.logspace(0, 4, 200)).astype(int))
fig, ax = plt.subplots(figsize=(11, 5.9), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.plot(ks, [100 * (jl <= k).mean() for k in ks], color=C["jlens"], lw=2.6, zorder=4)
ax.plot(ks, [100 * (tl <= k).mean() for k in ks], color=C["tl"], lw=2.6, zorder=4)
for v, c, lab in ((ro, C["oracle"], "Oracle lens names it"), (rn, C["nla"], "NLA names it")):
    ax.axhline(100 * v, color=c, lw=1.6, ls=(0, (5, 3)), zorder=3)
ax.text(1.25, 100 * ro + 1.6, f"Oracle lens names it in one readout: {100*ro:.0f}%", color=C["oracle"], ha="left", fontsize=10.5, fontweight="semibold")
ax.text(1.25, 100 * rn - 4.6, f"NLA names it in one readout: {100*rn:.0f}%", color=C["nla"], ha="left", fontsize=10.5, fontweight="semibold")
ax.axvline(10, color=MUTED, lw=1, ls=":", zorder=2); ax.text(10.8, 3, "top 10\n(the benchmark's cutoff)", color=MUTED, fontsize=9.5, va="bottom")
for k in (10, 50, 200):
    y = 100 * (jl <= k).mean(); ax.scatter([k], [y], color=C["jlens"], s=42, zorder=5, edgecolor=BG, linewidth=1.5)
    ax.text(k * 1.15, y - 5.5, f"{y:.0f}%", color=C["jlens"], fontsize=11, fontweight="semibold")
ax.plot([1.25, 1.9], [93, 93], color=C["jlens"], lw=2.6); ax.text(2.1, 93, "J-Lens", color=C["jlens"], fontsize=12, fontweight="bold", va="center")
ax.plot([1.25, 1.9], [86, 86], color=C["tl"], lw=2.6); ax.text(2.1, 86, "Template lens", color=C["tl"], fontsize=12, fontweight="bold", va="center")
ax.set_xscale("log"); ax.set_xlim(1, 12000); ax.set_ylim(0, 100)
ax.set_xticks([1, 10, 50, 200, 1000, 10000]); ax.set_xticklabels(["1", "10", "50", "200", "1,000", "10,000"], color=MUTED)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], color=MUTED)
ax.set_xlabel("how far down the lens's ranked word list you look", color=INK, labelpad=8)
ax.set_ylabel("prompts where the bridge is within that many words", color=INK, labelpad=8)
ax.grid(color=GRID, lw=0.8, zorder=0); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
fig.text(0.02, 0.955, "J-Lens has the bridge more often than its top 10 shows", fontsize=17, color=INK, fontweight="bold", va="top")
fig.text(0.02, 0.895, "Within the top 10 it has the bridge for 30% of prompts. Within the top 200 it reaches 70%, level with the two writers.", fontsize=11.5, color=MUTED, va="top")
fig.text(0.02, 0.025, f"Qwen3.6-27B, layer 42, {n} two-step questions. J-Lens ranks 248,320 tokens; the template lens ranks 13,174 words (the bridge is in its vocabulary for 41). Writers: 4 samples per prompt, word match.", fontsize=8.6, color=MUTED)
fig.subplots_adjust(left=0.085, right=0.975, top=0.82, bottom=0.14); fig.savefig(ROOT / "figs/cw13_bridge_rank.png", facecolor=BG); print("wrote cw13_bridge_rank.png")
# ---------- figure 2: where is the bridge, by part ----------
def boot(x):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (5000, len(x)))].mean(1); return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)
def reader(files, cond):
    rows = [json.loads(l) for f in files for l in open(R / "cw12" / f)]
    return [np.mean([names(s, bank[r["id"]]["intermediates"]) for s in r["samples"]]) for r in rows if r["cond"] == cond]
PARTS = [("full", "Whole activation"), ("J1024", "J part\n(16% of it)"), ("N1024", "The rest\n(84% of it)")]
SER = [("jlens", "J-Lens: bridge in its top 50", lambda c: [float(v <= 50) for v in jr["parts30"][c].values() if v]),
       ("tl", "Template lens: bridge in its top 50", lambda c: [float(v <= 50) for v in tr["parts30"][c].values() if v]),
       ("oracle", "Oracle lens: names the bridge", lambda c: reader(["olens_split.jsonl"], c)),
       ("nla", "NLA: names the bridge", lambda c: reader(["nla_split.jsonl"], c))]
fig, ax = plt.subplots(figsize=(11, 5.9), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG); w = 0.19
for gi, (cond, lab) in enumerate(PARTS):
    for si, (key, _, fn) in enumerate(SER):
        m, lo, hi = boot(fn(cond)); x = gi + (si - 1.5) * (w + 0.015)
        ax.bar(x, m, w, color=C[key], zorder=3); ax.plot([x, x], [lo, hi], color=INK, lw=1, alpha=0.5, zorder=4)
        ax.text(x, hi + 2, f"{m:.0f}%", ha="center", fontsize=10.5, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in PARTS], fontsize=12, color=INK)
ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
for si, (key, lab, _) in enumerate(SER):
    fig.patches.append(matplotlib.patches.Rectangle((0.09 + 0.225 * si, 0.795), 0.012, 0.02, transform=fig.transFigure, color=C[key]))
    fig.text(0.106 + 0.225 * si, 0.805, lab, fontsize=9.8, color=INK, va="center")
fig.text(0.02, 0.955, "The bridge is still in the rest of the activation. The two writers barely read it there.", fontsize=16, color=INK, fontweight="bold", va="top")
fig.text(0.02, 0.895, "The template lens, a plain word-direction lens, finds the bridge in the rest as often as in the J part. J-Lens cannot, by construction.", fontsize=11, color=MUTED, va="top")
fig.text(0.02, 0.025, "Qwen3.6-27B, layer 42, 30 two-step questions (template lens: the 25 whose bridge is in its vocabulary). Writers: 2 samples, word match. Lines: 95% bootstrap over items.", fontsize=8.6, color=MUTED)
fig.subplots_adjust(left=0.07, right=0.985, top=0.75, bottom=0.15); fig.savefig(ROOT / "figs/cw13_where_is_the_bridge.png", facecolor=BG); print("wrote cw13_where_is_the_bridge.png")
