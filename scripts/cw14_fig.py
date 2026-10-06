"""CW-14 figure: share of samples naming the hidden step, per part and reader, 70 new prompts. -> figs/cw14_split_70_new_prompts.png"""
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
COL = {"oracle": "#2a78d6", "nla": "#eb6834"}; NAME = {"oracle": "Oracle lens", "nla": "NLA"}
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
def names(t, inter): return any(re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", t.lower()) for w in inter)
def boot(x, B=10000):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)
data = {"oracle": [json.loads(l) for l in open(R / "cw14/olens_split.jsonl")], "nla": [json.loads(l) for l in open(R / "cw14/nla_split.jsonl")]}
ROWS = [("full", "Whole activation", 4.0), ("J1024", "J part", 2.9), ("N1024", "The rest", 1.9), ("R1024", "Random part\ncontrol", 0.6)]
n = len({r["id"] for r in data["oracle"]})
fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
fig.subplots_adjust(left=0.30, right=0.95, top=0.76, bottom=0.13)
h = 0.34
for cond, label, y in ROWS:
    sh = 100 * np.mean([r["share"] for r in data["oracle"] if r["cond"] == cond])
    ax.text(-3, y + 0.10, label.split("\n")[0], ha="right", va="center", fontsize=12.5, color=INK, fontweight="semibold" if cond in ("J1024", "N1024") else "regular")
    ax.text(-3, y - 0.22, (label.split("\n")[1] + " · " if "\n" in label else "") + f"{sh:.0f}% of the activation", ha="right", va="center", fontsize=9.5, color=MUTED)
    for j, rd in enumerate(("oracle", "nla")):
        v = [np.mean([names(s, bank[r["id"]]["intermediates"]) for s in r["samples"]]) for r in data[rd] if r["cond"] == cond]
        m, lo, hi = boot(v); yy = y + (h / 2 + 0.02) * (1 if j == 0 else -1)
        ax.barh(yy, m, height=h, color=COL[rd], zorder=3); ax.plot([lo, hi], [yy, yy], color=INK, lw=1.3, zorder=4)
        ax.text(hi + 1.5, yy, f"{m:.0f}%", va="center", fontsize=11, color=INK)
        if cond == "full": ax.text(m - 2, yy, NAME[rd], va="center", ha="right", fontsize=10.5, color="white", fontweight="semibold", zorder=5)
ax.set_xlim(0, 108); ax.set_ylim(0.0, 4.6); ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=10)
for x in (0, 25, 50, 75, 100): ax.axvline(x, color=GRID, lw=0.9, zorder=0)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0); ax.set_xlabel("Share of write-ups that name the hidden step", color=MUTED, fontsize=10.5, labelpad=8)
fig.text(0.04, 0.925, "Both readers name the hidden step from the J part, and rarely from the rest", fontsize=15.5, fontweight="bold", color=INK)
fig.text(0.04, 0.868, f"{n} multihop prompts not used in the first run. Tests and cutoff fixed before the run. J part minus rest: oracle lens +71 points,\nNLA +65 points, both p < 0.0001. Qwen3.6-27B, layer 42, 2 write-ups per prompt, word match. Lines: 95% bootstrap over prompts.", fontsize=9.8, color=MUTED, va="top", linespacing=1.45)
fig.savefig(ROOT / "figs/cw14_split_70_new_prompts.png", facecolor=BG); print("saved")
