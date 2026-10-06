# CW-14 add-on (base worker): patch the L42 read-position activation with one part and read the model's next token.
import json, time
from contextlib import nullcontext
from wsbench import readplan
from wsbench.produce.producer import _positions
from wsbench.produce.render import render
OUT = Path("/workspace/out/cw14"); K = 1024; b = p.backend; tok, dev = b.tokenizer, b.device
def rand(seed):
    g = torch.Generator(device="cpu").manual_seed(seed); Q, _ = torch.linalg.qr(torch.randn(5120, K, generator=g)); return Q.to(dev)
B = {"J": torch.load("/workspace/jbasis.pt")["Q"].float().to(dev)[:, :K], "R0": rand(0), "R1": rand(1),
     "P": torch.load("/workspace/pcabasis.pt")["P"].float().to(dev), "U": torch.load("/workspace/ubasis.pt")["U"].float().to(dev)}
def fwd(ids, pos, v=None):
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if v is not None: h[0, pos, :] = v.to(h.dtype)
        else: store["h"] = h[0, pos, :].detach().float().clone()
    store = {}; hd = b.blocks[42].register_forward_hook(hook)
    plain = b.model.disable_adapter() if hasattr(b.model, "disable_adapter") else nullcontext()
    try:
        with torch.no_grad(), plain:
            lg = b.model(torch.tensor([ids], device=dev)).logits[0, -1].float()
    finally: hd.remove()
    return torch.log_softmax(lg, -1), store.get("h")
t0 = time.time()
with open(OUT / "modeluse.jsonl", "w") as fh:
    for n, spec in enumerate(readplan.plan("multihop")[:100]):
        r = render(spec, tok); pos = _positions(spec.positions, r.tokens)[0] % len(r)
        lp0, h = fwd(r.ids, pos); top = int(lp0.argmax())
        lpc, _ = fwd(r.ids, pos, h)   # patching the activation back in must change nothing
        row = {"id": spec.id, "n": n, "pos_from_end": pos - len(r), "clean_top": tok.decode([top]), "clean_lp": float(lp0[top]),
               "selfpatch_maxdiff": float((lpc - lp0).abs().max()), "conds": {}}
        for name, Q in B.items():
            part = Q @ (Q.T @ h)
            for cond, v in ((f"{name}_only", part), (f"all_but_{name}", h - part)):
                lp, _ = fwd(r.ids, pos, v); t5 = lp.topk(5)
                row["conds"][cond] = {"share": float((v.norm() / h.norm()) ** 2), "lp_clean_top": float(lp[top]), "same_top": int(lp.argmax()) == top,
                                      "kl": float((lp0.exp() * (lp0 - lp)).sum()), "top5": [tok.decode([i]) for i in t5.indices.tolist()]}
        fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
        if n % 10 == 0: print(n, spec.id, row["clean_top"], row["selfpatch_maxdiff"], f"{time.time()-t0:.0f}s", flush=True)
print("MODELUSE_DONE", flush=True)
