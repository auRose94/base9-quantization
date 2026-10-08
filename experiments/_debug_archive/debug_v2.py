#!/usr/bin/env python3
"""Debug 10: stepwise diff for ALL layers, device-clean, plus eq_lib source."""
import inspect
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
sd, ma, _ = L.load_state_dict()
sd = L.to_device(sd, DEV)

tokens = np.load(os.path.join(DATA, "valid_tokens.npy")).astype(np.int64)
b = torch.from_numpy(tokens[:512].astype(np.int64))[None].to(DEV)
T = b.shape[1]
hd = args.dim // args.n_heads
rep = args.n_heads // args.n_kv_heads
pos = torch.arange(T, device=DEV)
mask = torch.full((T, T), float("-inf"), device=DEV).triu(1)

# reference per-layer outputs
caches = []
hooks = [l.register_forward_hook(lambda m, i, o: caches.append(o.detach().clone())) for l in ref.layers]
with torch.no_grad():
    h_emb_ref = ref.tok_embeddings(b).clone()
ref(b)
for hk in hooks:
    hk.remove()
with torch.no_grad():
    x = ref.tok_embeddings(b)
    for l in ref.layers:
        x = l(x, ref.freqs_cos[:T], ref.freqs_sin[:T])
    hfin_ref = ref.norm(x)
    logits_ref = ref.output(hfin_ref)
    caches_full = list(caches)
caches.clear()

# ---- my stepwise ----
h = sd["tok_embeddings.weight"][b]
print(f"emb   diff vs ref: {float((h - h_emb_ref).abs().max()):.3e}")
for li in range(args.n_layers):
    p = f"layers.{li}."
    hn = L.rmsnorm(h, sd[p + "attention_norm.weight"])
    q = (hn @ sd[p + "attention.wq.weight"].T).view(1, T, args.n_heads, hd)
    k = (hn @ sd[p + "attention.wk.weight"].T).view(1, T, args.n_kv_heads, hd)
    v = (hn @ sd[p + "attention.wv.weight"].T).view(1, T, args.n_kv_heads, hd)
    q, k = L.rope(q, pos), L.rope(k, pos)
    k = k.repeat_interleave(rep, 2)
    v = v.repeat_interleave(rep, 2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    att = (q @ k.transpose(-2, -1)) / hd**0.5 + mask
    o = (att.softmax(-1) @ v).transpose(1, 2).reshape(1, T, args.dim)
    h = h + o @ sd[p + "attention.wo.weight"].T
    hn2 = L.rmsnorm(h, sd[p + "ffn_norm.weight"])
    g = hn2 @ sd[p + "feed_forward.w1.weight"].T
    h = h + (g * torch.sigmoid(g) * (hn2 @ sd[p + "feed_forward.w3.weight"].T)) \
        @ sd[p + "feed_forward.w2.weight"].T
    print(f"blk{li}  diff vs ref: {float((h - caches_full[li]).abs().max()):.3e}")
hf = L.rmsnorm(h, sd["norm.weight"])
print(f"final-norm diff: {float((hf - hfin_ref).abs().max()):.3e}")
lg = hf @ sd["output.weight"].T
print(f"logits  diff: {float((lg - logits_ref).abs().max()):.3e}")

# and eq_lib.forward on the same input
lg2 = L.forward(sd, b, {"dim": 64, "n_layers": 5, "n_heads": 8, "n_kv_heads": 4,
                        "vocab_size": 512, "max_seq_len": 512})
print(f"L.forward logits diff vs ref: {float((lg2 - logits_ref).abs().max()):.3e}")
print(f"L.forward logits diff vs my stepwise: {float((lg2 - lg).abs().max()):.3e}")
print("\n--- eq_lib.forward source ---")
print(inspect.getsource(L.forward))