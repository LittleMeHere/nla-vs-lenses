"""CW-14 add-on: centred activations. c = h - (h.u)u with u the unit direction of a mean.
Usage: python scripts/cw14_centre_cells.py task OUT.pt      (leave-one-out mean of the other 99 multihop items, CPU)
       python scripts/cw14_centre_cells.py generic OUT.pt MEAN.pt   (mean captured on the pod from wikitext)"""
import sys, torch
from pathlib import Path
R = Path(__file__).parent.parent / "runs"
full = {c["id"]: c["h"].float() for c in torch.load(R / "cw14/resid_split.pt") if c["cond"] == "full"}
allh = dict(full)
for c in torch.load(R / "abl50/resid_L42.pt"):
    if c["layer"] == 42 and c["id"] not in allh and len(allh) < 100: allh[c["id"]] = c["h"].float()
assert len(allh) == 100, len(allh)
tot = torch.stack(list(allh.values())).sum(0); out = []; sh = []
gen = torch.load(sys.argv[3])["mean"].float() if sys.argv[1] == "generic" else None
for i, h in full.items():
    m = gen if gen is not None else (tot - h) / 99
    u = m / m.norm(); c = h - (h @ u) * u; sh.append(float((c.norm() / h.norm()) ** 2))
    out.append({"id": i, "cond": "C_gen" if gen is not None else "C_task", "h": c, "layer": 42, "share": sh[-1]})
torch.save(out, sys.argv[2]); print("cells", len(out), "mean share of squared norm kept %.3f" % (sum(sh) / len(sh)))
