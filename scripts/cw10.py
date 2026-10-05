"""CW-10 (designs/CW-10_prompt_recovery_and_j_any_layer.md): J-any-layer check and prompt recovery. No model calls.
Usage: python scripts/cw10.py  -> runs/cw10/analysis.txt"""
import json, re, ast
from pathlib import Path
import numpy as np
R = Path(__file__).parent.parent / "runs"
bank = {i["name"]: i for i in json.load(open(R / "multihop100/items.json"))["items"]}
STOP = set("the a an of to in on at for and or is are was were be by with as that this it its from into about fact "
           "what which who whose".split())

def rows(p):
    out = []
    for l in open(p):
        r = json.loads(l)
        for k in ("samples", "tokens"):
            if isinstance(r.get(k), str):
                r[k] = ast.literal_eval(r[k])
        out.append(r)
    return out

def names(text, inter):                     # the pilot's rule: exact word match of a bank intermediate, case-insensitive
    return any(re.search(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])", text.lower()) for w in inter)

def jhit(tokens, inter):
    return names(" ".join(t.replace("Ġ", " ").replace("▁", " ") for t in tokens[:10]), inter)

def content(prompt, drop):
    dw = set(re.findall(r"[a-z0-9]+", " ".join(drop).lower()))
    return {w for w in re.findall(r"[a-z0-9]+", prompt.lower()) if w not in STOP and w not in dw and len(w) > 1}

def recovery(text, cw):
    tw = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(cw & tw) / len(cw) if cw else np.nan

def auc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return np.nan
    p, n = s[y], s[~y]
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / (len(p) * len(n)))

def auc_ci(score, y, B=10000, seed=0):
    rng = np.random.default_rng(seed); s, y = np.asarray(score, float), np.asarray(y, bool); k = len(y)
    bs = [auc(s[i], y[i]) for i in (rng.integers(0, k, k) for _ in range(B))]
    return auc(s, y), np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)

L = []
# ---------- reproduce CW-1 ----------
J = {}
for r in rows(R / "multihop100/jlens.multihop.jsonl"):
    J.setdefault(r["id"], {})[int(r["layer"])] = jhit(r["tokens"], bank[r["id"]]["intermediates"])
O = {r["id"]: " ".join(r["samples"]) for r in rows(R / "multihop100/olens_L42,44.multihop.jsonl") if str(r["layer"]) == "42"}
N = {r["id"]: " ".join(r["samples"]) for r in rows(R / "multihop100/nla.multihop.jsonl")}
ids = list(bank)
J42 = {r["id"]: jhit(r["tokens"], bank[r["id"]]["intermediates"]) for r in rows(R / "multihop100/jlens_L42.multihop.jsonl")
       if str(r["layer"]) == "42"}                     # L42 is in its own file; the 11-layer file has 20..60 step 4
j42 = {i: J42[i] for i in ids}; o = {i: names(O[i], bank[i]["intermediates"]) for i in ids}
n = {i: names(N[i], bank[i]["intermediates"]) for i in ids}
layers = sorted({l for i in ids for l in J[i]})
L.append(f"check against CW-1: J-Lens L42 {sum(j42.values())} (39), oracle {sum(o.values())} (80), NLA {sum(n.values())} (75); "
         f"J any of {len(layers)} layers {sum(any(J[i].values()) for i in ids)} (76); J layers {layers}")
# ---------- Part A ----------
L.append("\nPART A: J-Lens at any layer, by L42 agreement group")
groups = {"oracle+NLA, not J (prose-only)": [i for i in ids if o[i] and n[i] and not j42[i]],
          "oracle only": [i for i in ids if o[i] and not n[i] and not j42[i]],
          "NLA only": [i for i in ids if n[i] and not o[i] and not j42[i]],
          "none": [i for i in ids if not (o[i] or n[i] or j42[i])]}
for g, m in groups.items():
    anyl = [i for i in m if any(J[i].values())]
    first = [min(l for l, h in J[i].items() if h) for i in anyl]
    L.append(f"  {g}: n {len(m)}; J-Lens names the bridge at some layer in {len(anyl)}; first layers {sorted(first)}")
po = groups["oracle+NLA, not J (prose-only)"]
L.append("  prose-only items J-Lens never names: " + ", ".join(i for i in po if not any(J[i].values())))
L.append("  per layer, prose-only items named by J-Lens: " + ", ".join(f"L{l}:{sum(J[i].get(l, False) for i in po)}" for l in layers))
# ---------- Part B ----------
L.append("\nPART B: prompt recovery (share of the prompt's content words, bridge and answer words excluded, found in the readout)")
cw = {i: content(bank[i]["prompt"], bank[i]["intermediates"] + [bank[i]["target"]]) for i in ids}
L.append(f"  content words per prompt: mean {np.mean([len(cw[i]) for i in ids]):.1f}, min {min(len(cw[i]) for i in ids)}")
def block(title, texts, base_texts=None):
    m = [i for i in texts if i in bank]
    y = [names(texts[i], bank[i]["intermediates"]) for i in m]
    rec = [recovery(texts[i], cw[i]) for i in m]
    oth = [recovery(texts[i], cw[m[(k + 1) % len(m)]]) for k, i in enumerate(m)]
    ln = [len(texts[i].split()) for i in m]
    a = auc_ci(rec, y); b = auc_ci(ln, y)
    s = (f"  {title}: n {len(m)}, names bridge {sum(y)}; recovery mean {np.mean(rec):.2f} (other item's prompt {np.mean(oth):.2f}); "
         f"recovery when named {np.mean([r for r, t in zip(rec, y) if t]):.2f} vs not {np.mean([r for r, t in zip(rec, y) if not t]):.2f}; "
         f"AUC {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]; length-only AUC {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}]")
    if base_texts is not None:
        c = auc_ci([recovery(base_texts[i], cw[i]) for i in m], y)
        s += f"; AUC using the intact L42 readout's recovery {c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]"
    L.append(s)
for reader, f, base in (("oracle", "olens_abl2.jsonl", O), ("NLA", "nla_resid_abl2.jsonl", N)):
    rr = rows(R / "abl3" / f)
    for cond in sorted({r["cond"] for r in rr}):
        block(f"{reader}, abl3 cond={cond}", {r["id"]: " ".join(r["samples"]) for r in rr if r["cond"] == cond},
              base if cond == "strong" else None)
for reader, f in (("oracle", "olens_L16,28,36.multihop.jsonl"), ("NLA", "nla_L16,28,36.multihop.jsonl")):
    rr = rows(R / "sweep50" / f)
    for lay in ("16", "28", "36"):
        block(f"{reader}, L{lay}", {r["id"]: " ".join(r["samples"]) for r in rr if str(r["layer"]) == lay})
block("oracle, intact L42", O); block("NLA, intact L42", N)
(R / "cw10").mkdir(exist_ok=True); open(R / "cw10/analysis.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
