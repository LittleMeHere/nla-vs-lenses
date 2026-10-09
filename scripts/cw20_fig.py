"""CW-20 figure: effect of removing each kind of text from oracle lens write-ups, beyond removing random text of the
same length. Reads runs/cw20/analysis.txt. -> figs/cw20_olens_rebuild_by_kind.png"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).parent.parent
for f in (ROOT / "scripts/fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Public Sans", "font.size": 11})
INK, MUTED, GRID, BG = "#14213d", "#5b6472", "#e3e6ec", "#fbfbf8"
txt = (ROOT / "runs/cw20/analysis.txt").read_text()
vals = {}
for m in re.finditer(r"(\w+) - (\w+) \((\d+)\): ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", txt):
    vals[m.group(1)] = (100 * float(m.group(4)), 100 * float(m.group(5)), 100 * float(m.group(6)), int(m.group(3)))
ROWS = [("drop_answer", "The answer"), ("drop_about", "Statements about the prompt's subject"), ("drop_invented", "Invented follow-on questions"),
        ("drop_wrong", "Only the wrong pieces"), ("drop_format", "Markup")]
fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=170)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
fig.subplots_adjust(left=0.36, right=0.95, top=0.70, bottom=0.17)
for n, (k, lab) in enumerate(ROWS):
    m, lo, hi, cnt = vals[k]
    y = len(ROWS) - 1 - n
    col = "#eb6834" if hi < 0 else ("#0f8f83" if lo > 0 else "#838c9b")
    ax.barh(y, m, height=0.58, color=col, zorder=3)
    ax.plot([lo, hi], [y, y], color=INK, lw=1.3, zorder=4)
    ax.text(lo - 0.4 if m < 0 else hi + 0.4, y, ("0.0" if abs(m) < 0.05 else f"{m:+.1f}".replace("-", "−")), va="center", ha="right" if m < 0 else "left", fontsize=11, color=INK)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[1] for r in ROWS][::-1], fontsize=11.5, color=INK)
ax.axvline(0, color=INK, lw=1, zorder=2)
ax.set_xlim(-12, 6)
ax.set_xticks([-10, -5, 0, 5])
ax.set_xticklabels(["−10", "−5", "0", "+5"], color=MUTED, fontsize=9.5)
for x in (-10, -5, 5):
    ax.axvline(x, color=GRID, lw=0.9, zorder=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_xlabel("Change in rebuilt variance, in points, compared with removing random pieces of the same length\n(left = this kind of text mattered more than average)", color=MUTED, fontsize=9.8, labelpad=8)
c = vals["corrected"]
fig.text(0.035, 0.93, "Rebuilt variance after removing one kind of text from oracle lens write-ups", fontsize=14.5, fontweight="bold", color=INK, va="top")
fig.text(0.035, 0.862, "Qwen3.6-27B, layer 44, 70 WorkspaceBench multihop prompts, 2 write-ups each. Each bullet is rebuilt by the oracle lens's reconstructor and the\n"
         "predictions averaged; whitened FVE around the corpus mean, one global scale. Pieces labelled by a model (agreement with a second model 89%).\n"
         "3 random controls per row. Correcting the wrong pieces in place: " + f"{c[0]:+.1f} points [{c[1]:+.1f}, {c[2]:+.1f}]".replace("-", "−") + ". Lines: 95% bootstrap over prompts.",
         fontsize=9.5, color=MUTED, va="top", linespacing=1.45)
fig.savefig(ROOT / "figs/cw20_olens_rebuild_by_kind.png", facecolor=BG)
print("saved", vals)
