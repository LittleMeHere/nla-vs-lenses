"""CW-16b figure: effect of removing each kind of text, beyond removing random text of the same length. -> figs/cw16b_what_the_rebuild_needs.png"""
import json, math
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
rec = torch.load(R / "cw16b/recon.pt"); G = {c["id"]: sc(c["h"]) for c in torch.load(R / "cw14/resid_split.pt") if c["cond"] == "full"}; mu = torch.stack(list(G.values())).mean(0)
fve = lambda k: 1 - float((sc(rec[k]) - G[k.split("|")[1]]).pow(2).sum() / (G[k.split("|")[1]] - mu).pow(2).sum())
per = {}
for k in rec:
    v, i, s = k.split("|"); per.setdefault(v, {}).setdefault(i, []).append(fve(k))
def item(v, i):
    vs = [v] if not v.startswith("rand_") else [f"{v}_{s}" for s in range(3)]; xs = [x for u in vs if u in per and i in per[u] for x in per[u][i]]
    return np.mean(xs) if xs else None
def diff(a, b):
    d = [(item(a, i), item(b, i)) for i in per["orig"]]; d = np.array([x - y for x, y in d if x is not None and y is not None])
    bs = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1); return 100 * d.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5), len(d)
ROWS = [("drop_restate", "rand_restate", "All restatements of the prompt"), ("drop_restate_wrong", "rand_restate_wrong", "Only the inexact restatements"), ("drop_content", "rand_content", "Claims about the topic and answer"),
        ("drop_content_wrong", "rand_content_wrong", "Only the wrong claims"), ("drop_format", "rand_format", "Statements about the format")]
fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG); fig.subplots_adjust(left=0.33, right=0.95, top=0.74, bottom=0.16)
for n, (a, b, lab) in enumerate(ROWS):
    m, lo, hi, k = diff(a, b); y = len(ROWS) - 1 - n; col = "#eb6834" if hi < 0 else ("#0f8f83" if lo > 0 else "#9aa3b2")
    ax.barh(y, m, height=0.58, color=col, zorder=3); ax.plot([lo, hi], [y, y], color=INK, lw=1.3, zorder=4)
    ax.text(lo - 2 if m < 0 else hi + 2, y, f"{m:+.0f}", va="center", ha="right" if m < 0 else "left", fontsize=11, color=INK)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[2] for r in ROWS][::-1], fontsize=11.5, color=INK); ax.axvline(0, color=INK, lw=1, zorder=2)
ax.set_xlim(-72, 25); ax.set_xticks([-60, -40, -20, 0, 20]); ax.set_xticklabels(["−60", "−40", "−20", "0", "+20"], color=MUTED, fontsize=9.5)
for x in (-60, -40, -20, 20): ax.axvline(x, color=GRID, lw=0.9, zorder=0)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0); ax.set_xlabel("Change in how much of the activation is rebuilt, in points, compared with\nremoving random pieces of the same length (left = this kind of text mattered more than average)", color=MUTED, fontsize=9.8, labelpad=8)
c = diff("corrected", "orig")
fig.text(0.04, 0.925, "The rebuild depends on the NLA's restatement of the prompt, even when it is inexact", fontsize=14.5, fontweight="bold", color=INK)
fig.text(0.04, 0.868, f"Each row removes one kind of text from the NLA's write-up and rebuilds the layer-42 activation from what is left. 70 multihop prompts, 2 write-ups each;\npieces labelled by an LLM (not validated); 3 random controls per row. Correcting the wrong pieces in place instead: {c[0]:+.0f} points [{c[1]:+.0f}, {c[2]:+.0f}]. Lines: 95% bootstrap.", fontsize=9.5, color=MUTED, va="top", linespacing=1.45)
fig.savefig(ROOT / "figs/cw16b_what_the_rebuild_needs.png", facecolor=BG); print("saved")
