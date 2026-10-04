"""Patch wsbench JLens for one 80GB GPU with the 27B base loaded: keep the 64-layer Jacobian stack
on CPU and move one layer at a time; keep W_U in bf16; compute the per-layer denominator in
32k-row chunks (the original keeps a 6.7GB fp32 stack on GPU and makes a 5GB fp32 temporary).
Top-10 tokens are the same as fp32 in spot checks; scores lose fp32 precision. Run on the pod."""
import sys
p = sys.argv[1]; s = open(p).read()
old1 = """        self._jac = _load_jacobians(self.repo, self.filename, backend.device)
        self._w_u = backend.unembed.float()
"""
new1 = """        self._jac = _load_jacobians(self.repo, self.filename, "cpu")  # patch_jlens_mem.py
        self._w_u = backend.unembed  # bf16
"""
old2 = """        if layer not in self._denom:
            self._denom[layer] = (w_u @ self._jac[layer]).norm(dim=1).clamp_min(1e-9)
        scores = ((h.to(b.device) @ self._jac[layer].T) @ w_u.T) / self._denom[layer]
"""
new2 = """        J = self._jac[layer].to(b.device, dtype=w_u.dtype)
        print(f"[jlens] layer {layer} allocated {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
        if layer not in self._denom:
            parts = [(w_u[i : i + 32768] @ J).float().norm(dim=1) for i in range(0, w_u.shape[0], 32768)]
            self._denom[layer] = torch.cat(parts).clamp_min(1e-9)
        scores = ((h.to(b.device).to(w_u.dtype) @ J.T) @ w_u.T).float() / self._denom[layer]
"""
assert old1 in s and old2 in s, "patch anchors not found"
old3 = "        x = b.final_norm(h.to(b.device) @ self._jac[layer].T)\n"
new3 = "        x = b.final_norm(h.to(b.device) @ self._jac[layer].to(b.device).T)  # patch_jlens_mem.py\n"
old4 = "        vals, ids = torch.topk(x @ self._w_u.T, self.k)\n"
new4 = "        vals, ids = torch.topk((x.to(self._w_u.dtype) @ self._w_u.T).float(), self.k)  # patch_jlens_mem.py\n"
assert old3 in s and old4 in s, "rlens anchors not found"
open(p, "w").write(s.replace(old1, new1).replace(old2, new2).replace(old3, new3).replace(old4, new4)); print("patched", p)
