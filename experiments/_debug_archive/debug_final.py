#!/usr/bin/env python3
"""Debug 7 (final): (a) reference model + correctly-shifted eval on my windows;
(b) per-layer numerical diff of my eq_lib.forward vs llama2.c reference."""
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
N = 384

ck = torch.load(os.path.join(DATA, "stories260K.pt"), map_location="cpu", weights_only=False)
args = ModelArgs(**ck["model_args"])
ref = Transformer(args)
ref.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=False)
ref = ref.eval().to(DEV)

tokens = np.load(os.path.join(DATA, "valid_tokens.npy")).astype(np.int64)[: N * 512]
win = torch.from_numpy(tokens).view(N, 512).to(DEV)

# (a) reference with correct next-token pairing
nll = 0
with torch.no_grad():
    for i in range(0, N, 32):
        b = win[i:i + 32]
        lg = ref(b, targets=b)
        full = lg[..., :-1, :]  # logits full (targets given) -> logits[:, :-1] predicts b[:,1:]
        tgt = b[:, 1:]
        nll += -torch.log_softmax(full, -1).gather(-1, tgt[..., None]).sum().item()
        if i == 0:
            acc0 = (full.argmax(-1) == tgt).float().mean().item()
loss = nll / (N * 511)
print(f"[ref] loss {loss:.4f} (ppl {np.exp(loss):.2f}), window0 next-token acc {acc0:.3f}")

# (b) diff my forward vs reference per layer on window 0
mine_sd, ma, _ = L.load_state_dict()
mine_sd = L.to_device(mine_sd, DEV)
b0 = win[:1]
with torch.no_grad():
    caches = []
    hooks = [l.register_forward_hook(lambda m, i, o: caches.append(o.detach().clone())) for l in ref.layers]
    caches.clear()
    ref(b0)
    for hk in hooks:
        hk.remove()
    h_mine = mine_sd["tok_embeddings.weight"][b0]
    print(f"[diff] emb            maxabs {float((caches[0] - h_mine).abs().max()):.2e}")
    for li, lref in enumerate(ref.layers):
        # run my single-layer step from h_mine
        p = f"layers.{li}."
        h = h_mine
        h = L.rmsnorm(h, mine_sd[p + "attention_norm.weight"])
        B, T = h.shape[:2]
        hd = args.dim // args.n_heads
        q = (h @ mine_sd[p + "attention.wq.weight"].T).view(B, T, args.n_heads, hd)
        k = (h @ mine_sd[p + "attention.wk.weight"].T).view(B, T, args.n_kv_heads, hd)
        v = (h @ mine_sd[p + "attention.wv.weight"].T).view(B, T, args.n_kv_heads, hd)
        pos = torch.arange(T, device=DEV)
        q, k = L.rope(q, pos), L.rope(k, pos)
        k = k.repeat_interleave(args.n_heads // args.n_kv_heads, 2)
        v = v.repeat_interleave(args.n_heads // args.n_kv_heads, 2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        mask = torch.full((T, T), float("-inf"), device=DEV).triu(1)
        att = (q @ k.transpose(-2, -1)) / hd**0.5 + mask
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, args.dim)
        h = h + o @ mine_sd[p + "attention.wo.weight"].T
        hn = L.rmsnorm(h, mine_sd[p + "ffn_norm.weight"])
        g = hn @ mine_sd[p + "feed_forward.w1.weight"].T
        ff = (g * torch.sigmoid(g) * (hn @ mine_sd[p + "feed_forward.w3.weight"].T))
        h = h + ff @ mine_sd[p + "feed_forward.w2.weight"].T
        print(f"[diff] block{li} ref-out vs my-out maxabs {float((caches[li] - h).abs().max()):.2e}")