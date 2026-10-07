"""CW-16 check (pod): the same texts through EasyNLA's own loader and forward pass (github.com/asherps/EasyNLA:
nla.models.NLACriticModel.from_pretrained, nla.utils.critic.critic_predict), to compare with scripts/cw16_ar.py.
Usage: PYTHONPATH=/workspace/EasyNLA uv run python cw16_ar_official.py TEXTS.jsonl OUT.pt"""
import glob, json, math, sys, time, torch
from transformers import AutoTokenizer
from nla.models import NLACriticModel
from nla.utils.critic import critic_predict
torch.set_grad_enabled(False)
root = glob.glob("/workspace/hf/hub/models--ceselder--qwen3.6-27b-nla-rl/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(f"{root}/av_base"); t0 = time.time()
critic = NLACriticModel.from_pretrained(f"{root}/ar_reconstructor", torch_dtype=torch.bfloat16, device_map="cuda").eval()
SCALE = math.sqrt(5120); TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
print(f"loaded {time.time()-t0:.0f}s", flush=True)
rows = [json.loads(l) for l in open(sys.argv[1])]; enc = [(r["key"], tok.encode(TEMPLATE.format(explanation=r["text"]), add_special_tokens=False)) for r in rows]
enc = sorted([e for e in enc if 0 < len(e[1]) <= 1024], key=lambda x: len(x[1])); out = {}; B = 16; pad = tok.eos_token_id
for i in range(0, len(enc), B):
    ch = enc[i:i + B]; L = max(len(x[1]) for x in ch)
    x = torch.full((len(ch), L), pad, dtype=torch.long, device="cuda"); a = torch.zeros((len(ch), L), dtype=torch.long, device="cuda")
    for j, (_, ids) in enumerate(ch): x[j, :len(ids)] = torch.tensor(ids); a[j, :len(ids)] = 1
    pred = critic_predict(critic, x, a, SCALE)
    for j, (k, _) in enumerate(ch): out[k] = pred[j].cpu()
    if (i // B) % 40 == 0: print(i, len(enc), f"{time.time()-t0:.0f}s", flush=True)
torch.save(out, sys.argv[2]); print("AR_OFFICIAL_DONE", len(out), flush=True)
