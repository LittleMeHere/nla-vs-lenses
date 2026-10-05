"""How jbasis_L42_qwen36_27b.pt was made (Qwen3.6-27B, layer 42, neuronpedia J-Lens, Salesforce-wikitext n1000).

J-Lens scores token t on an activation h as  unit(d_t) . h,  with  d_t = W_U[t] @ J   (J = the layer-42 Jacobian).
The basis is the principal directions of all unit-normalised d_t over the whole vocabulary:
    G = sum_t unit(d_t) unit(d_t)^T          (5120 x 5120)
    eigenvectors of G, largest eigenvalue first  ->  Q
The "J part" of h for a given k is  Q[:, :k] @ (Q[:, :k].T @ h);  the rest is  h - that.  I used k = 1024.
Share of the J-direction variance in the top k:  16: 0.43,  64: 0.53,  256: 0.73,  1024: 0.93,  2560: 0.99.

File contents:  {"Q": float16 [5120, 2560] (columns = directions, orthonormal),  "ev": float64 [5120] eigenvalues}

Load:
    b = torch.load("jbasis_L42_qwen36_27b.pt"); Q = b["Q"].float()[:, :1024]
    share_inside = ((v @ Q) ** 2).sum() / (v ** 2).sum()      # how much of a vector v lies in the J part
"""
import torch
from wsbench.produce.producer import Producer
from wsbench.produce.methods import _load_jacobians

p = Producer.load("Qwen/Qwen3.6-27B", "jlens")
dev = p.backend.device
J = _load_jacobians(p.method.repo, p.method.filename, "cpu")[42].to(dev)      # [5120, 5120]
W = p.backend.unembed.float()                                                # [V, 5120]
G = torch.zeros(W.shape[1], W.shape[1], device=dev, dtype=torch.float64)
for i in range(0, W.shape[0], 16384):
    D = W[i:i + 16384] @ J
    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-9)
    G += (D.T @ D).double()
ev, Q = torch.linalg.eigh(G)
ev, Q = ev.flip(0), Q.flip(1).float()
torch.save({"Q": Q[:, :2560].half().cpu(), "ev": ev.cpu()}, "jbasis_L42_qwen36_27b.pt")
