# CW-18 (base worker): capture each family's read position(s) at every 4th layer, to see where its content is readable.
from wsbench import readplan
from wsbench.produce.producer import _positions
from wsbench.produce.render import render
OUT = Path("/workspace/out/cw18"); OUT.mkdir(parents=True, exist_ok=True); tok = p.backend.tokenizer; LAYERS = list(range(20, 61, 4)) + [42, 62]
for fam in ("multihop", "association", "typo", "multilingual", "basic_readout"):
    rows = []
    try: specs = readplan.plan(fam)[:100]
    except Exception as e: print("skip family", fam, repr(e)[:200], flush=True); continue
    for spec in specs:
        try:
            r = render(spec, tok); pos = _positions(spec.positions, r.tokens); idx = [q % len(r) for q in pos][:12]
            acts = p.backend.capture(r.ids, LAYERS, idx)
            rows.append({"id": spec.id, "pos": idx, "n_tokens": len(r), "tokens": [r.decoded[j] for j in idx], "h": {L: acts[L].half() for L in LAYERS}})
        except Exception as e: print("skip", fam, getattr(spec, "id", "?"), repr(e)[:150], flush=True)
    torch.save(rows, OUT / f"layers_{fam}.pt"); print("SAVED", fam, len(rows), "positions per item", sorted({len(x["pos"]) for x in rows}), flush=True)
print("CW18_DONE", flush=True)
