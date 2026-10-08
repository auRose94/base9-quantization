#!/usr/bin/env python3
"""Debug 11: minimal repro — L.forward vs stepwise, per-layer intermediates."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/tmp/ref")
sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

DEV = "cuda"
DATA = os.path.join(L.ROOT, "data")
sd0, ma0, _ = L.load_state_dict()
sd = L.to_device(sd0, DEV)
ma = dict(ma0)
print("ma:", ma)

tokens = np.load(os.path.join(DATA, "valid_tokens.npy")).astype(np.int64)
b = torch.from_numpy(tokens[:512].astype(np.int64))[None].to(DEV)


def stepwise(sd, idx, ma):
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    rep = nh // nkv
    h = sd["tok_embeddings.weight"][idx]
    pos = torch.arange(T, device=idx.device)
    mask = torch.full((T, T), float("-inf"), device=idx.device).triu(1)
    snaps = [h.clone()]
    for li in range(ma["n_layers"]):
        p = f"layers.{li}."
        h = L.rmsnorm(h, sd[p + "attention_norm.weight"])
        q = (h @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k = (h @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v = (h @ sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q, k = L.rope(q, pos), L.rope(k, pos)
        k = k.repeat_interleave(rep, 2)
        v = v.repeat_interleave(rep, 2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / hd**0.5 + mask
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, dim)
        h = h + o @ sd[p + "attention.wo.weight"].T
        hn = L.rmsnorm(h, sd[p + "ffn_norm.weight"])
        g = hn @ sd[p + "feed_forward.w1.weight"].T
        h = h + (g * torch.sigmoid(g) * (hn @ sd[p + "feed_forward.w3.weight"].T)) \
            @ sd[p + "feed_forward.w2.weight"].T
        snaps.append(h.clone())
    h = L.rmsnorm(h, sd["norm.weight"])
    lg = h @ sd["output.weight"].T
    snaps.append(h.clone())
    return lg, snaps


with torch.no_grad():
    lg2, s2 = stepwise(sd, b, ma)
    # my forward, instrumented by re-running its lines verbatim with captures
    B, T = b.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    h = sd["tok_embeddings.weight"][b]
    pos = torch.arange(T, device=b.device)
    mask = torch.full((T, T), float("-inf"), device=b.device).triu(1)
    verbatim = [h.clone()]
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        h = L.rmsnorm(h, sd[p + "attention_norm.weight"])
        q = (h @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k = (h @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v = (h @ sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q = L.rope(q, pos)
        k = L.rope(k, pos)
        rep = nh // nkv
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / hd**0.5 + mask
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, dim)
        h = h + (o @ sd[p + "attention.wo.weight"].T)
        hn = L.rmsnorm(h, sd[p + "ffn_norm.weight"])
        w1out = hn @ sd[p + "feed_forward.w1.weight"].T
        ff = (w1out * torch.sigmoid(w1out) * (hn @ sd[p + "feed_forward.w3.weight"].T))
        h = h + (ff @ sd[p + "feed_forward.w2.weight"].T)
        verbatim.append(h.clone())
    h = L.rmsnorm(h, sd["norm.weight"])
    lg3 = h @ sd["output.weight"].T
    verbatim.append(h.clone())

print("L.forward logits vs stepwise:", float((lg3 - lg2).abs().max()))
for i, (a_, b_) in enumerate(zip(s2, verbatim)):
    tag = "emb" if i == 0 else ("fin" if i == len(s2) - 1 else f"blk{i-1}")
    print(f"  {tag:4s} stepwise-vs-verbatim maxabs {float((a_ - b_).abs().max()):.3e}")