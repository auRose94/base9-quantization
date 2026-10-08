#!/usr/bin/env python3
"""Debug: which forward variant reproduces the checkpoint? (val loss + sample)"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

DEV = "cuda"
sd0, ma, ck = L.load_state_dict()
SDD = L.to_device(sd0, DEV)
tokens = L.valid_tokens(32 * 512 + 512).astype(np.int64)
N = 32


def fwd(idx, rope_mode="interleaved"):
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    h = SDD["tok_embeddings.weight"][idx]
    pos = torch.arange(T, device=idx.device)
    if rope_mode == "interleaved":
        inv = 10000.0 ** (-torch.arange(0, hd, 2, device=DEV) / hd)
        ang = pos[:, None] * inv[None, :]
        cos, sin = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
        r = lambda x: torch.stack([x[..., 0::2] * cos - x[..., 1::2] * sin,
                                   x[..., 0::2] * sin + x[..., 1::2] * cos], -1).flatten(-2)
    elif rope_mode == "half":
        inv = 10000.0 ** (-torch.arange(0, hd, 2, device=DEV) / hd)
        ang = pos[:, None] * inv[None, :]
        hc, hs = ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]
        cos, sin = torch.cat([hc, hc], -1), torch.cat([hs, hs], -1)   # (1,T,1,D)
        rot = lambda x: torch.cat([-x[..., hd // 2:], x[..., :hd // 2]], -1)
        r = lambda x: x * cos + rot(x) * sin
    else:
        r = lambda x: x
    mask = torch.full((T, T), float("-inf"), device=DEV).triu(1)
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        h = L.rmsnorm(h, SDD[p + "attention_norm.weight"])
        q = (h @ SDD[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k = (h @ SDD[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v = (h @ SDD[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q, k = r(q), r(k)
        k, v = k.repeat_interleave(nh // nkv, 2), v.repeat_interleave(nh // nkv, 2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / hd**0.5 + mask
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, dim)
        h = h + o @ SDD[p + "attention.wo.weight"].T
        hn = L.rmsnorm(h, SDD[p + "ffn_norm.weight"])
        g = hn @ SDD[p + "feed_forward.w1.weight"].T
        u = hn @ SDD[p + "feed_forward.w3.weight"].T
        h = h + (g * torch.sigmoid(g) * u) @ SDD[p + "feed_forward.w2.weight"].T
    return L.rmsnorm(h, SDD["norm.weight"]) @ SDD["output.weight"].T


@torch.no_grad()
def loss_for(rope_mode, order="attn-first"):
    win = torch.from_numpy(tokens[: (N + 1) * 512]).view(N + 1, 512).to(DEV)
    b = win[:N]
    lg = fwd(b, rope_mode)[:, :-1]
    tg = b[:, 1:]
    nll = -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).squeeze(-1)
    return nll.mean().item()


@torch.no_grad()
def gen(rope_mode, seed=1, n=120):
    torch.manual_seed(seed)
    sp = L.get_sp()
    ids = [sp.bos_id()] + sp.encode("Once upon a time,")
    for _ in range(n):
        lg = fwd(torch.tensor([ids[-512:]], device=DEV), rope_mode)[0, -1]
        v, ix = torch.topk(lg, 50)
        ids.append(int(ix[torch.multinomial(torch.softmax(v / 0.8, -1), 1)]))
    return sp.decode(ids)


if __name__ == "__main__":
    for mode in ("interleaved", "half", "none"):
        los = loss_for(mode)
        print(f"== rope={mode:12s} loss {los:6.3f} (ppl {np.exp(los):8.2f})")
        print("   ", gen(mode).replace("\n", " ")[:180])