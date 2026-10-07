# CW-19 (NLA worker, K=1 = the benchmark's default): the NLA at L42 on the whole poetry activations captured by the
# benchmark's capture (runs/cw17/resid_poetry_L42.pt, cond "full"), rows in the benchmark's readout format.
import json
OUT = Path("/workspace/out/cw19"); OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "nla_poetry.jsonl", "w") as fh:
    for c in torch.load("/workspace/resid_poetry_L42.pt"):
        if c["cond"] != "full": continue
        r = m.read(c["h"], 42)
        fh.write(json.dumps({"id": c["id"], "layer": 42, "pos": c["pos"], "token": c["token"], "samples": r.samples}, ensure_ascii=False) + "\n"); fh.flush()
print("CW19_NLA_DONE", flush=True)
