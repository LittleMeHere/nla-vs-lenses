"""CW-14 extension: build all-but-random (seed 0), a second random part (seed 1), PCA part and all-but-PCA from the saved
whole activations. CPU. Usage: python scripts/cw14_ext_cells.py RESID_SPLIT.pt OUT.pt"""
import sys, torch
from pathlib import Path
R = Path(__file__).parent.parent / "runs"; K = 1024
cells = [c for c in torch.load(sys.argv[1]) if c["cond"] == "full"]
def rand(seed):
    g = torch.Generator(device="cpu").manual_seed(seed); Q, _ = torch.linalg.qr(torch.randn(5120, K, generator=g)); return Q
R0, R1 = rand(0), rand(1); P = torch.load(R / "cw12/pcabasis.pt")["P"].float()
old = {c["id"]: c["h"] for c in torch.load(sys.argv[1]) if c["cond"] == "R1024"}
out = []
for c in cells:
    h = c["h"].float(); r0 = R0 @ (R0.T @ h); r1 = R1 @ (R1.T @ h); hp = P @ (P.T @ h)
    assert torch.nn.functional.cosine_similarity(r0, old[c["id"]].float(), dim=0) > 0.9999, "seed-0 random part differs from the pod's"
    for cond, v in (("RN1024", h - r0), ("R1024s1", r1), ("P1024", hp), ("PN1024", h - hp)):
        out.append({"id": c["id"], "cond": cond, "h": v, "layer": 42, "share": float((v.norm() / h.norm()) ** 2)})
torch.save(out, sys.argv[2]); print("cells", len(out), "items", len(cells))
