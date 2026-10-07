# CW-17 follow-up (base worker): capture the poetry read position at every 4th layer, to find where the rhyme word is readable.
from wsbench import readplan
from wsbench.produce.producer import _positions
from wsbench.produce.render import render
OUT = Path("/workspace/out/cw17"); tok = p.backend.tokenizer; LAYERS = list(range(20, 61, 4)) + [42, 62]; rows = []
for spec in readplan.plan("poetry")[:100]:
    r = render(spec, tok); pos = _positions(spec.positions, r.tokens); idx = [q % len(r) for q in pos]
    last = len(r) - 1   # also the final prompt token (the space before the rhyme word), the bank's earlier read position
    acts = p.backend.capture(r.ids, LAYERS, [idx[0], last])
    rows.append({"id": spec.id, "pos": [idx[0], last], "tokens": [r.decoded[idx[0]], r.decoded[last]], "h": {L: acts[L].half() for L in LAYERS}})
torch.save(rows, OUT / "resid_poetry_layers.pt"); print("POETRY_LAYERS_SAVED", len(rows), flush=True)
