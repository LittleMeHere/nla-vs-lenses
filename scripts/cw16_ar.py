"""CW-16 (pod): run the NLA reconstructor on a list of texts. Forward pass follows EasyNLA nla/utils/critic.py:
pred = value_head(normalize(last-block hidden at the last token, sqrt(d))), final norm and lm_head removed.
Usage: uv run python cw16_ar.py TEXTS.jsonl OUT.pt     TEXTS rows: {"key": str, "text": str}"""
import glob, json, math, sys, time, torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
root = glob.glob("/workspace/hf/hub/models--ceselder--qwen3.6-27b-nla-rl/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(f"{root}/av_base"); t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(f"{root}/ar_reconstructor", dtype=torch.bfloat16, device_map="cuda")
inner = m.model; assert len(inner.layers) == 43, len(inner.layers)
inner.norm = torch.nn.Identity()
W = load_file(f"{root}/ar_reconstructor/value_head.safetensors"); print({k: tuple(v.shape) for k, v in W.items()}, flush=True)
head = W["weight"].float().cuda(); SCALE = math.sqrt(5120)
TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
print(f"loaded {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
rows = [json.loads(l) for l in open(sys.argv[1])]; out = {}; skipped = 0
enc = []
for r in rows:
    ids = tok.encode(TEMPLATE.format(explanation=r["text"]), add_special_tokens=False)
    if 0 < len(ids) <= 1024: enc.append((r["key"], ids))
    else: skipped += 1
enc.sort(key=lambda x: len(x[1])); B = 16; pad = tok.eos_token_id
for i in range(0, len(enc), B):
    ch = enc[i:i + B]; L = max(len(x[1]) for x in ch)
    x = torch.full((len(ch), L), pad, dtype=torch.long, device="cuda"); a = torch.zeros((len(ch), L), dtype=torch.long, device="cuda")
    for j, (_, ids) in enumerate(ch): x[j, :len(ids)] = torch.tensor(ids); a[j, :len(ids)] = 1
    h = inner(input_ids=x, attention_mask=a).last_hidden_state
    last = h[torch.arange(len(ch)), a.sum(1) - 1].float()
    last = last / last.norm(dim=1, keepdim=True).clamp_min(1e-9) * SCALE
    pred = last @ head.T
    for j, (k, _) in enumerate(ch): out[k] = pred[j].cpu()
    if (i // B) % 20 == 0: print(i, len(enc), f"{time.time()-t0:.0f}s", flush=True)
torch.save(out, sys.argv[2]); print("AR_DONE", len(out), "skipped", skipped, flush=True)
