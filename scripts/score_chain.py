"""Score chain_intermediates readouts: which numbers each readout states (digits, English number
words, Chinese numerals; J-Lens tokens byte-decoded first), against the item's intermediates, with a
chance baseline = expected hits if the readout were scored against another item's intermediates."""
import json, re, collections, sys
P = sys.argv[1] if len(sys.argv) > 1 else "runs/chain60"
items = {it["name"]: it for it in json.load(open(f"{P}/items.json"))["items"]}
def _b2u():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1)); cs = bs[:]; k = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + k); k += 1
    return {chr(c): b for b, c in zip(bs, cs)}
U = _b2u()
def dec(t):
    try: return bytes(U[ch] for ch in t).decode("utf-8")
    except Exception: return t
CN = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
def cn_num(s):
    if not s or any(ch not in CN and ch != "十" for ch in s): return None
    if "十" in s:
        a, _, b = s.partition("十")
        if len(a) > 1 or len(b) > 1: return None
        return (CN[a] if a else 1) * 10 + (CN[b] if b else 0)
    return CN[s] if len(s) == 1 else None
W = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
def word(n): return W[n] if n < 20 else TENS[n // 10] + ("" if n % 10 == 0 else "-" + W[n % 10])
def nums_in_text(text):
    out = {int(x) for x in re.findall(r"(?<![\d.])[0-9]+(?![0-9])", text)}
    low = text.lower()
    for n in range(100):
        if re.search(r"\b" + word(n).replace("-", "[- ]") + r"\b", low): out.add(n)
    for m in re.findall(r"[零一二两三四五六七八九十]+", text):
        v = cn_num(m)
        if v is not None: out.add(v)
    return out
def nums_in_tokens(toks):
    out = set()
    for t in toks:
        s = dec(t).strip()
        if re.fullmatch(r"[0-9]+", s): out.add(int(s))
        if s.lower() in W: out.add(W.index(s.lower()))
        v = cn_num(s)
        if v is not None: out.add(v)
    return out
def load(f): return [json.loads(l) for l in open(f"{P}/{f}")]
if __name__ == "__main__":
    nla = {r["id"]: nums_in_text(r["samples"][0]) for r in load("nla_L42.chain.jsonl")}
    ol = {r["id"]: nums_in_text(" ".join(r["samples"])) for r in load("olens_L42.chain.jsonl")}
    jl = collections.defaultdict(dict)
    for r in load("jlens.chain.jsonl"): jl[r["id"]][r["layer"]] = nums_in_tokens(r["tokens"])
    ids = sorted(nla); n = len(ids)
    I = {i: {int(x) for x in items[i]["intermediates"]} for i in ids}
    def rate(S): return sum(bool(S[i] & I[i]) for i in ids)
    def null(S): return sum(sum(bool(S[i] & I[j]) for j in ids if j != i) / (n - 1) for i in ids)
    jlany = {i: set().union(*jl[i].values()) for i in ids}
    print(f"n={n}")
    for name, S in (("NLA L42", nla), ("oracle lens L42", ol), ("J-Lens L42", {i: jl[i][42] for i in ids}),
                    ("J-Lens L56", {i: jl[i][56] for i in ids}), ("J-Lens any of 12 layers", jlany)):
        print(f"{name:24s} names an intermediate {rate(S):2d}/{n}   chance {null(S):5.1f}")
    print("J-Lens by layer:", {L: sum(bool(jl[i][L] & I[i]) for i in ids) for L in sorted(jl[ids[0]])})
    for name, S in (("NLA", nla), ("oracle", ol)):
        st = {i: items[i]["start"] in S[i] for i in ids}; hit = {i: bool(S[i] & I[i]) for i in ids}
        tab = collections.Counter((st[i], hit[i]) for i in ids)
        print(f"{name}: states start {sum(st.values())}; start&inter {tab[(True, True)]}, inter-without-start {tab[(False, True)]}, "
              f"start-only {tab[(True, False)]}, neither {tab[(False, False)]}; names answer {sum(items[i]['answer'] in S[i] for i in ids)}; "
              f"mean distinct numbers per readout {sum(len(S[i]) for i in ids)/n:.1f}")
