#!/usr/bin/env python3
"""Debug 8: diff sub-steps inside block0 (my math vs llama2.c modules)."""
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
args = ModelArgs(**ck["model_args"])
ref = Transformer(args)
ref.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=False)
ref = ref.eval().to(DEV)
sd0, ma, _ = L.load_state_dict()
sd = L.to_device(sd0, DEV)

tokens = np.load(os.path.join(DATA, "valid_tokens.npy")).astype(np.int64)
b = torch.from_numpy(tokens[:512].astype(np.int64))[None].to(DEV)
T = 512

# reference sub-outputs
outs = {}
def grab(name):
    def hook(m, i, o):
        outs[name] = (o.detach() if not isinstance(o, tuple) else o[0].detach())
    return hook
h1 = ref.layers[0].attention.register_forward_hook(grab("attn"))
h2 = ref.layers[0].feed_forward.register_forward_hook(grab("ffn"))
h3 = ref.layers[0].attention_norm.register_forward_hook(grab("norm1"))
h4 = ref.layers[0].ffn_norm.register_forward_hook(grab("norm2"))
with torch.no_grad():
    x0 = ref.tok_embeddings(b)
    n1_ref = ref.layers[0].attention_norm(x0)
    attn_out = ref.layers[0].attention(n1_ref, ref.freqs_cos[:T], ref.freqs_sin[:T])
    h1r = x0 + attn_out
    n2_ref = ref.layers[0].ffn_norm(h1r)
    ffn_out = ref.layers[0].feed_forward(n2_ref)
    h2r = h1r + ffn_out
for hh in (h1, h2, h3, h4):
    hh.remove()

# ---- mine ----
p = "layers.0."
hd = args.dim // args.n_heads
pos = torch.arange(T, device=DEV)
mask = torch.full((T, T), float("-inf"), device=DEV).triu(1)
h = sd["tok_embeddings.weight"][b]
print(f"[n1 ] {float((L.rmsnorm(h, sd[p + 'attention_norm.weight']) - n1_ref).abs().max()):.3e}")
hh = L.rmsnorm(h, sd[p + "attention_norm.weight"])
q = (hh @ sd[p + "attention.wq.weight"].T).view(1, T, args.n_heads, hd)
k = (hh @ sd[p + "attention.wk.weight"].T).view(1, T, args.n_kv_heads, hd)
v = (hh @ sd[p + "attention.wv.weight"].T).view(1, T, args.n_kv_heads, hd)
q, k = L.rope(q, pos), L.rope(k, pos)
k = k.repeat_interleave(args.n_heads // args.n_kv_heads, 2)
v = v.repeat_interleave(args.n_heads // args.n_kv_heads, 2)
qq, kk, vv = (t.transpose(1, 2) for t in (q, k, v))
att = (qq @ kk.transpose(-2, -1)) / hd**0.5 + mask
o = (att.softmax(-1) @ vv).transpose(1, 2).reshape(1, T, args.dim)
attn_mine = o @ sd[p + "attention.wo.weight"].T
print(f"[attn] mine vs ref: {float((attn_mine - attn_out).abs().max()):.3e}")

# also compare via flash SDPA with is_causal
attn_flash = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
attn_flash = attn_flash.transpose(1, 2).reshape(1, T, args.dim) @ sd[p + "attention.wo.weight"].T
print(f"[attn-flash] mine vs ref: {float((attn_flash - attn_out).abs().max()):.3e}")

h1m = h + attn_mine
print(f"[n2 ] {float((L.rmsnorm(h1m, sd[p + 'ffn_norm.weight']) - n2_ref).abs().max()):.3e}")
g = L.rmsnorm(h1m, sd[p + "ffn_norm.weight"]) @ sd[p + "feed_forward.w1.weight"].T
u = L.rmsnorm(h1m, sd[p + "ffn_norm.weight"]) @ sd[p + "feed_forward.w3.weight"].T
ffn_mine = (g * torch.sigmoid(g) * u) @ sd[p + "feed_forward.w2.weight"].T
print(f"[ffn ] mine vs ref: {float((ffn_mine - ffn_out).abs().max()):.3e}")
print(f"[blk] {float((h1m + ffn_mine - h2r).abs().max()):.3e}")

# rope cross-check against frequence values on q at position 7, head 0
ang = pos[:8].float()[:, None] * 10000.0 ** (-torch.arange(0, 8, 2) / 8)[None]
print("[rope] angles pos0-7:", ang[0].tolist())