"""Top-1,024 principal directions of the unit-normalised unembedding rows (no Jacobian), and their overlap with the
J basis and the PCA basis. CPU. Output: runs/cw14/ubasis.pt, runs/cw14/ubasis_overlap.txt"""
import numpy as np, torch
from pathlib import Path
R = Path(__file__).parent.parent / "runs"; K = 1024
W = np.load(R / "jrank/lm_head_fp16.npy", mmap_mode="r"); C = torch.zeros(5120, 5120, dtype=torch.float64)
for i in range(0, W.shape[0], 16384):
    w = torch.from_numpy(np.asarray(W[i:i + 16384])).float(); w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-9); C += (w.T @ w).double()
ev, V = torch.linalg.eigh(C); U = V[:, -K:].flip(1).float(); share = float(ev[-K:].sum() / ev.sum())
Q = torch.load(R / "cw12/jbasis.pt")["Q"].float()[:, :K]; P = torch.load(R / "cw12/pcabasis.pt")["P"].float()
ov = lambda A, B: float(((A.T @ B) ** 2).sum() / K)
cells = [c for c in torch.load(R / "cw14/resid_split.pt") if c["cond"] == "full"]
H = torch.stack([c["h"].float() for c in cells]); e = lambda B: float((((H @ B) ** 2).sum(1) / (H ** 2).sum(1)).mean())
L = [f"unembedding basis: top {K} directions hold {share:.3f} of the unit-row variance",
     f"overlap (1 = same subspace, {K/5120:.2f} = unrelated): unembedding vs J {ov(U, Q):.3f}; unembedding vs PCA {ov(U, P):.3f}; J vs PCA {ov(Q, P):.3f}",
     f"share of activation squared norm (70 CW-14 prompts, L42): unembedding part {e(U):.3f}; J part {e(Q):.3f}; PCA part {e(P):.3f}"]
torch.save({"U": U.half(), "share": share}, R / "cw14/ubasis.pt"); open(R / "cw14/ubasis_overlap.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
