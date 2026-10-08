#!/usr/bin/env python3
"""Debug 12: clean isolation — eq_lib eval vs ref eval on IDENTICAL windows,
with per-layer instrumentation inside the same script."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/tmp/ref")
sys.path.insert(0, os.path.dirname(__file__))
from model import ModelArgs, Transformer
import eq_lib as L  # noqa: E402

DEV = "cuda"
DATA = os.path.join(L.ROOT, "data")

ck = torch.load(os.path.join(DATA, "stories260K.pt"), map_location="cpu", weights_only=False)
sd0, ma0, _ = L.load_state_dict()
sd = L.to_device(sd0, DEV)
ma = dict(ma0)
args = ModelArgs(**ma0)
ref = Transformer(args)
ref.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=False)
ref = ref.eval().to(DEV)

tokens = np.load(os.path.join(DATA, "valid_tokens.npy")).astype(np.int64)[: 384 * 512]
win = torch.from_numpy(tokens).view(384, 512)

@torch.no_grad()
def eval_both(batch=32):
    res = {}
    for tag in ("mine", "ref"):
        nll = 0.0
        per_win = []
        for i in range(0, 384, batch):
            b = win[i:i + batch].to(DEV)
            if tag == "mine":
                lg = L.forward(sd, b, ma)
            else:
                lg = ref(b, targets=b)
            lg = lg[:, :-1]
            tg = b[:, 1:]
            nll_b = -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).squeeze(-1)
            nll += nll_b.sum().item()
            per_win.append(nll_b.mean(-1).cpu())
        res[tag] = torch.cat(per_win)
        print(f"[{tag:4s}] loss {res['mine' if tag=='mine' else tag].mean():.4f}")
    d = res["mine"] - res["ref"]
    print("per-window mine-ref: mean", float(d.mean()), "std", float(d.std()))
    worst = d.abs().argmax().item()
    print("worst window idx:", worst, "mine", res["mine"][worst].item(), "ref", res["ref"][worst].item())
    # instrument the worst window per layer
    bw = win[worst:worst + 1].to(DEV)
    caches = []
    hooks = [l.register_forward_hook(lambda m, i, o: caches.append(o.detach().clone())) for l in ref.layers]
    with torch.no_grad():
        caches.clear()
        ref(bw)
        for hk in hooks:
            hk.remove()
        h = sd["tok_embeddings.weight"][bw]
        print("emb diff:", float((h - ref.tok_embeddings(bw)).abs().max()))
        for li in range(ma["n_layers"]):
            p = f"layers.{li}."
            h = L.rmsnorm(h, sd[p + "attention_norm.weight"])
            B, T = b.shape
            hd = ma["dim"] // ma["n_heads"]
            q = (h @ sd[p + "attention.wq.weight"].T).view(B, T, ma["n_heads"], hd)
            k = (h @ sd[p + "attention.wk.weight"].T).view(B, T, ma["n_kv_heads"], hd)
            v = (h @ sd[p + "attention.wv.weight"].T).view(B, T, ma["n_kv_heads"], hd)
            pos = torch.arange(T, device=DEV)
            q, k = L.rope(q, pos), L.rope(k, pos)
            k = k.repeat_interleave(ma["n_heads"] // ma["n_kv_heads"], 2)
            v = v.repeat_interleave(ma["n_heads"] // ma["n_kv_heads"], 2)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            mask = torch.full((T, T), float("-inf"), device=DEV).triu(1)
            att = (q @ k.transpose(-2, -1)) / hd**0.5 + mask
            o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, ma["dim"])
            h = h + o @ sd[p + "attention.wo.weight"].T
            hn = L.rmsnorm(h, sd[p + "ffn_norm.weight"])
            g = hn @ sd[p + "feed_forward.w1.weight"].T
            h = h + (g * torch.sigmoid(g) * (hn @ sd[p + "feed_forward.w3.weight"].T)) \
                @ sd[p + "feed_forward.w2.weight"].T
            print(f"blk{li} diff:", float((h - caches[li]).abs().max()))
        hf = L.rmsnorm(h, sd["norm.weight"])
        lg = hf @ sd["output.weight"].T

eval_both()