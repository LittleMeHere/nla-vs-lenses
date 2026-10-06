# CW-14 add-on (base worker): generic mean at L42 from wikitext (positions after the first 16), then the oracle lens on
# the task-mean-centred cells. The generic-centred cells are built on CPU from the saved mean and run in a later job.
import json, time, zlib
from datasets import load_dataset
OUT = Path("/workspace/out/cw14"); b = p.backend; tok = b.tokenizer
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
texts = [t.strip() for t in ds["text"] if len(t.strip()) > 400 and not t.strip().startswith("=")][:200]
s = torch.zeros(5120, dtype=torch.float64); n = 0
for t in texts:
    ids = tok.encode(t, add_special_tokens=False)[:128]
    if len(ids) <= 20: continue
    v = b.capture(ids, [42], list(range(16, len(ids))))[42]; s += v.double().sum(0); n += len(v)
torch.save({"mean": (s / n).float(), "n_vectors": n, "n_passages": len(texts), "source": "Salesforce/wikitext wikitext-103-raw-v1 validation, first 200 paragraphs over 400 chars, 128 tokens each, positions 16+"}, OUT / "generic_mean_L42.pt")
print("GENERIC_MEAN_SAVED", n, len(texts), float((s / n).norm()), flush=True)
def run(cells, name):
    p.use("olens"); p.method.sampling.k = 2
    with open(OUT / name, "w") as fh:
        for c in cells:
            torch.manual_seed(zlib.crc32(f"{c['id']}|{c['cond']}|olens".encode())); t = time.time(); r = p.method.read(c["h"], 42)
            fh.write(json.dumps({"id": c["id"], "cond": c["cond"], "share": c["share"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
            print(f"{c['id']} {c['cond']} {time.time()-t:.1f}s", flush=True)
run(torch.load("/workspace/resid_ctask.pt"), "olens_ctask.jsonl"); print("CTASK_OLENS_DONE", flush=True)
