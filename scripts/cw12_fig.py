"""CW-12 figure: what each reader reports from each part of the activation.
Usage: python scripts/cw12_fig.py OUT.png "Title" "Subtitle" oracle=FILE.jsonl [nla=FILE.jsonl]"""
import json, re, sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
ROOT = Path(__file__).parent.parent; R = ROOT / "runs"
for f in (ROOT / "scripts/fonts").glob("*.ttf"): fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Public Sans", "font.size": 11})
INK, MUTED, GRID, BG, SHARE = "#14213d", "#5b6472", "#e3e6ec", "#fbfbf8", "#a9b1bf"
COL = {"oracle": "#2a78d6", "nla": "#eb6834"}; NAME = {"oracle": "Oracle lens", "nla": "NLA"}
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
STOP = set("the a an of to in on at for and or is are was were be by with as that this it its from into about fact what which who whose".split())
def names(t, inter): return any(re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", t.lower()) for w in inter)
def cw(it):
    drop = set(re.findall(r"[a-z0-9]+", " ".join(it["intermediates"] + [it["target"]]).lower()))
    return {w for w in re.findall(r"[a-z0-9]+", it["prompt"].lower()) if w not in STOP and w not in drop and len(w) > 1}
def rec(t, c): return len(c & set(re.findall(r"[a-z0-9]+", t.lower()))) / len(c)
def boot(x, B=5000):
    x = np.asarray(x, float); b = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)
out, title, sub = sys.argv[1], sys.argv[2], sys.argv[3]
data = {}
for a in sys.argv[4:]:
    k, f = a.split("=", 1); rows = [json.loads(l) for ff in f.split(",") for l in open(ff)]
    done = {r["id"] for r in rows if r["cond"] == "RN1024"}; data[k] = [r for r in rows if r["id"] in done]
readers = list(data); first = data[readers[0]]; n_items = len({r["id"] for r in first}); n_s = len(first[0]["samples"])
has_pca = any(r["cond"] == "P1024" for r in first)
ROWS = [("full", "Whole activation", 6.3), ("J1024", "J part", 4.7), ("N1024", "The rest", 3.7),
        ("R1024", "Random 1,024 directions", 2.0), ("RN1024", "Everything except those", 1.0)]
HEAD = [(5.42, "SPLIT BY J-LENS  ·  J part = the 1,024 directions J-lens uses most"), (2.72, "CONTROL  ·  the same split with random directions")]
YLO = 0.3
if has_pca:
    ROWS += [("P1024", "Top 1,024 PCA directions", -0.7), ("PN1024", "Everything except those", -1.7)]
    HEAD += [(0.02, "COMPARISON  ·  the 1,024 directions where activations vary most (PCA)")]; YLO = -2.4
fig = plt.figure(figsize=(12, 8.2 if has_pca else 6.4), dpi=170); fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.25, 1.25], left=0.215, right=0.975, top=(0.835 if has_pca else 0.79), bottom=(0.09 if has_pca else 0.115), wspace=0.14)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
PANELS = ["Share of the activation", "Names the bridge", "Prompt words it rebuilds"]
for ax, t in zip(axes, PANELS):
    ax.set_facecolor(BG); ax.set_xlim(0, 118); ax.set_ylim(YLO, 7.0); ax.set_yticks([])
    ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0", "50%", "100%"], color=MUTED, fontsize=9.5)
    for x in (0, 50, 100): ax.axvline(x, color=GRID, lw=0.9, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.set_title(t, loc="left", fontsize=11.5, color=INK, fontweight="semibold", pad=10)
h = 0.34 if len(readers) > 1 else 0.5
for cond, label, y in ROWS:
    strong = cond in ("J1024", "N1024")
    fig.text(0.205, axes[0].transData.transform((0, y))[1] / fig.bbox.height, label, ha="right", va="center",
             fontsize=12, color=INK, fontweight="semibold" if strong else "regular")
    shr = [r["share"] for rd in readers for r in data[rd] if r["cond"] == cond]
    sh = 100 * np.mean(shr)
    axes[0].barh(y, sh, height=0.5, color=SHARE, zorder=3); axes[0].text(sh + 2.5, y, f"{sh:.0f}%", va="center", fontsize=11, color=INK)
    for j, rd in enumerate(readers):
        rs = [r for r in data[rd] if r["cond"] == cond]; yy = y + (h / 2 + 0.02) * (1 if j == 0 else -1) * (len(readers) > 1)
        for ax, vals in ((axes[1], [np.mean([names(s, bank[r["id"]]["intermediates"]) for s in r["samples"]]) for r in rs]),
                         (axes[2], [np.mean([rec(s, cw(bank[r["id"]])) for s in r["samples"]]) for r in rs])):
            m, lo, hi = boot(vals)
            ax.barh(yy, m, height=h, color=COL[rd], zorder=3)
            ax.plot([lo, hi], [yy, yy], color=INK, lw=1.0, alpha=0.55, zorder=4, solid_capstyle="butt")
            ax.text(hi + 2.5, yy, f"{m:.0f}%", va="center", fontsize=11, color=INK)
for y, t in HEAD:
    fig.text(0.02, axes[0].transData.transform((0, y))[1] / fig.bbox.height, t, ha="left", va="center", fontsize=8.6, color=MUTED, fontweight="medium")
fig.text(0.02, 0.965 if has_pca else 0.955, title, fontsize=17, color=INK, fontweight="bold", va="top")
fig.text(0.02, 0.918 if has_pca else 0.895, sub, fontsize=11.5, color=MUTED, va="top")
if len(readers) > 1:
    for j, rd in enumerate(readers):
        fig.patches.append(matplotlib.patches.Rectangle((0.80 + 0.09 * j, 0.948 if has_pca else 0.928), 0.012, 0.014 if has_pca else 0.018, transform=fig.transFigure, color=COL[rd]))
        fig.text(0.816 + 0.09 * j, 0.955 if has_pca else 0.937, NAME[rd], fontsize=10.5, color=INK, va="center")
who = " and ".join(NAME[r] for r in readers) if len(readers) > 1 else NAME[readers[0]] + " only"
fig.text(0.02, 0.022 if has_pca else 0.03, f"Qwen3.6-27B, layer 42, {n_items} two-step questions, {n_s} samples per cell, {who}. Bridge scored by word match. Lines: 95% bootstrap over items.",
         fontsize=8.8, color=MUTED)
fig.savefig(out, facecolor=BG); print("wrote", out)
