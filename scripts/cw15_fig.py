"""CW-15 figure: what survives in the J part and in the rest, per reader. -> figs/cw15_content_by_part.png"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = Path(__file__).parent.parent
for f in (ROOT / "scripts/fonts").glob("*.ttf"): fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Public Sans", "font.size": 11})
INK, MUTED, GRID, BG = "#14213d", "#5b6472", "#e3e6ec", "#fbfbf8"; CJ, CR = "#0f8f83", "#9aa3b2"
res = json.load(open(ROOT / "runs/cw15/content.json"))
ROWS = [("bridge", "Names the hidden step"), ("target", "Names the answer"), ("prompt", "Prompt's content words\nit repeats"), ("last", "Quotes the right last word\nof the prompt")]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=170, sharey=True); fig.patch.set_facecolor(BG)
fig.subplots_adjust(left=0.20, right=0.97, top=0.74, bottom=0.12, wspace=0.08)
for ax, rd, name in zip(axes, ("oracle", "NLA"), ("Oracle lens", "NLA")):
    ax.set_facecolor(BG)
    for k, (key, lab) in enumerate(ROWS):
        y = len(ROWS) - 1 - k
        if rd == "oracle" and key == "last":
            ax.text(3, y, "not applicable: it does not quote the prompt", va="center", fontsize=9.5, color=MUTED, style="italic"); continue
        for j, (c, col) in enumerate((("J1024", CJ), ("N1024", CR))):
            m, lo, hi = [100 * v for v in res[f"{rd}|{c}"][key]]; yy = y + (0.19 if j == 0 else -0.19)
            ax.barh(yy, m, height=0.34, color=col, zorder=3); ax.plot([lo, hi], [yy, yy], color=INK, lw=1.2, zorder=4)
            ax.text(hi + 1.5, yy, f"{m:.0f}%", va="center", fontsize=10.5, color=INK)
            if k == 0: ax.text(m - 2 if m > 30 else hi + 12, yy, "J part" if j == 0 else "The rest", va="center", ha="right" if m > 30 else "left", fontsize=10, color="white" if m > 30 else MUTED, fontweight="semibold", zorder=5)
        w = 100 * res[f"{rd}|full"][key][0]; ax.plot([w, w], [y - 0.42, y + 0.42], color=INK, lw=1.4, ls=(0, (2, 2)), zorder=5)
        if k == 0: ax.text(w, y + 0.5, "whole activation", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xlim(0, 112); ax.set_ylim(-0.6, len(ROWS) - 0.25); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], color=MUTED, fontsize=9.5)
    for x in (0, 25, 50, 75, 100): ax.axvline(x, color=GRID, lw=0.9, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.set_title(name, loc="left", fontsize=12.5, color=INK, fontweight="semibold", pad=8)
axes[0].set_yticks(range(len(ROWS))); axes[0].set_yticklabels([l for _, l in ROWS][::-1], fontsize=11.5, color=INK)
fig.text(0.03, 0.93, "The J part carries the content; the rest still tells the NLA where the sentence ends", fontsize=15, fontweight="bold", color=INK)
fig.text(0.03, 0.875, "Same 70 multihop prompts and write-ups as the split run (2 per prompt, Qwen3.6-27B, layer 42). Word-level measures, no judge; chosen after reading\nthe write-ups, so exploratory. J part = 16% of squared norm, the rest = 84%. Lines: 95% bootstrap over prompts. Chance for the last word is about 55%.", fontsize=9.5, color=MUTED, va="top", linespacing=1.45)
fig.savefig(ROOT / "figs/cw15_content_by_part.png", facecolor=BG); print("saved")
