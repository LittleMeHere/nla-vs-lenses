"""CW-20: rebuild layer-44 activations from texts with the oracle lens's reconstructor, through the collaborator's
loader imported unchanged (Causal-Concordance scripts/011_olens_swaps/ar_vectors.py). Runs on the pod, own process.
  python cw20_ar.py CC_DIR resid_L44.pt texts.jsonl out.pt
texts.jsonl rows: {"key": ..., "text": ...}. out.pt: {"keys", "raw" [n,5120] f32 (ln_pre read, raw output),
"white" [n,5120] f32 = W(raw - mu)}, plus once "targets": {id: W(h - mu)} and the capture check."""
import json
import sys
import time

import torch

cc, resid_path, texts_path, out_path = sys.argv[1:5]
sys.path.insert(0, f"{cc}/scripts/common")
sys.path.insert(0, f"{cc}/scripts/011_olens_swaps")
import ar_vectors as AV  # noqa: E402
import registry as R  # noqa: E402

L = 44
log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
torch.set_grad_enabled(False)
model = R.load_model()
tok = R.make_tokenizer()
caps = torch.load(resid_path)
# check 1: the saved capture is the block-L output this loader's convention targets
blocks = R.blocks(model)
cos = []
for c in caps:
    keep = {}
    hk = blocks[L].register_forward_hook(lambda m, i, o: keep.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
    try:
        model(torch.tensor([c["ids"]], device=model.device), use_cache=False)
    finally:
        hk.remove()
    cos.append(float(torch.nn.functional.cosine_similarity(keep["h"][c["pos"]], c["h"].float(), dim=0)))
log(f"capture check: block-{L} output vs saved, cosine min {min(cos):.5f} mean {sum(cos)/len(cos):.5f} (n {len(cos)})")
w = AV.whitener(L)
targets = {c["id"]: ((c["h"].double() - w["mu"]) @ w["W"].T).float() for c in caps}
pt, head, heads = AV.attach_ar(model, log)
rows = [json.loads(l) for l in open(texts_path)]
spans = [tok(r["text"], add_special_tokens=False).input_ids for r in rows]
keep_i = [i for i, s in enumerate(spans) if len(s) > 0]
log(f"{len(rows)} texts, {len(rows) - len(keep_i)} empty, token length max {max(len(s) for s in spans)}")
order = sorted(keep_i, key=lambda i: len(spans[i]))
raw = torch.zeros(len(rows), 5120)
for j in range(0, len(order), 256):
    ch = order[j:j + 256]
    o = AV.ar_forward(pt, head, heads, [spans[i] for i in ch], L, bs=16)["ln_pre"]
    raw[ch] = o.float()
    log(f"{j + len(ch)}/{len(order)}")
white = ((raw.double() - w["mu"]) @ w["W"].T).float()
torch.save({"keys": [r["key"] for r in rows], "empty": [i for i in range(len(rows)) if i not in set(keep_i)], "raw": raw,
            "white": white, "targets": targets, "capture_cos": cos, "cc_commit": "b8d458e"}, out_path)
log(f"saved {out_path}")
