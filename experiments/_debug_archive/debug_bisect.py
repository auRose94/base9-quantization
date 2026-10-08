#!/usr/bin/env python3
"""Debug 4: bisect (a) my forward vs llama2.c reference — per-layer activation
diff on the same input; (b) my valid_tokens stream — decode/roundtrip/bos-eos
stats + per-position accuracy of the reference model on window 0."""
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

# ---------------- (b) eval data sanity ----------------
sp = L.get_sp()
tok = np.load(os.path.join(DATA, "valid_tokens.npy"))
print(f"[data] {len(tok):,} tokens; bos {float((tok==1).mean()):.4f}, eos {float((tok==2).mean()):.4f}, "
      f"unk {float((tok==0).mean()):.4f}")
print("[data] tokens[:40]:", tok[:40].tolist())
rt = sp.decode(tok[:120].tolist())
print("[data] window0 decode[:220]:", repr(rt[:220]))

# roundtrip check on a fresh story
txt = open(os.path.join(DATA, "TinyStories-valid.txt"), encoding="utf-8").read(200_000)
story = txt.split("_________________________________________")[1].strip()[:400]
ids = sp.encode(story)
print(f"[data] roundtrip identity for fresh story: {sp.decode(ids) == story}")

# per-position top-1 accuracy of the REFERENCE model on window 0 (positions 1..)
ck = torch.load(os.path.join(DATA, "stories260K.pt"), map_location="cpu", weights_only=False)
args = ModelArgs(**ck["model_args"])
ref = Transformer(args)
ref.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=False)
ref = ref.eval().to(DEV)
b0 = torch.from_numpy(tok[:512].astype(np.int64))[None].to(DEV)
with torch.no_grad():
    lg = ref(b0, targets=b0)
    last_loss = ref.last_loss.item()
    # full logits via targets path above; recompute per-position:
    lg = ref(b0, targets=b0)
    pred = lg.argmax(-1)
    acc_pos = (pred[0] == b0[0]).float()
print(f"[data] ref-model window0 mean NLL {last_loss:.4f}  per-pos acc "
      f"first50 {acc_pos[:50].mean():.3f} last50 {acc_pos[-50:].mean():.3f}")

# what does the model PREDICT at these positions vs the ids we see?
tgt = b0[0]
print("[data] pred vs tok (first 30):")
print("  ref pred:", sp.decode(pred[0][:30].tolist()))
print("  our tok :", sp.decode(tgt[:30].tolist()))

# ---------------- (a) my forward vs reference, per-layer diff ----------------
x = b0
with torch.no_grad():
    # reference hidden states via hooks
    caches = []
    hooks = []
    for l in ref.layers:
        hooks.append(l.register_forward_hook(lambda m, i, o: caches.append(o.detach().clone())))
    h_ref = ref.tok_embeddings(x)
    caches.clear()
    _ = ref(x)
    my = L.forward(L.to_device({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, DEV),
                   x, {"dim": args.dim, "n_layers": args.n_layers, "n_heads": args.n_heads,
                       "n_kv_heads": args.n_kv_heads, "vocab_size": args.vocab_size,
                       "max_seq_len": args.max_seq_len})
sd_mine = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
h = sd_mine["tok_embeddings.weight"][x]
for li, l in enumerate(ref.layers):
    n = li * 2
    print(f"[diff] embedding      {torch.maxdiff if False else 0}", end="\r")
    print(f"[diff] ref-block{li} hidden vs my running state: "
          f"maxabs {float((caches[li] - h).abs().max()):.3e}")
    # advance my copy through this layer same as reference did
    p = f"layers.{li}."
    h = L.rmsnorm(h, sd_mine[p + "attention_norm.weight"])
    B, T = x.shape
    hd = args.dim // args.n_heads
    q = (h @ sd_mine[p + "attention.wq.weight"].T).view(B, T, args.n_heads, hd)
    k = (h @ sd_mine[p + "attention.wk.weight"].T).view(B, T, args.n_kv_heads, hd)
    v = (h @ sd_mine[p + "attention.wv.weight"].T).view(B, T, args.n_kv_heads, hd)
    q = L.rope(q, torch.arange(T, device=DEV))
    k = L.rope(k, torch.arange(T, device=DEV))
    k = k.repeat_interleave(args.n_heads // args.n_kv_heads, 2)
    v = v.repeat_interleave(args.n_heads // args.n_kv_heads, 2)
    q0, k0, v0 = (t.transpose(1, 2) for t in (q, k, v))
    mask = torch.full((T, T), float("-inf"), device=DEV).triu(1)
    att = (q0 @ k0.transpose(-2, -1)) / hd**0.5 + mask
    o = (att.softmax(-1) @ v0).transpose(1, 2).reshape(B, T, args.dim)
    h = h + o @ sd_mine[p + "attention.wo.weight"].T
    hn = L.rmsnorm(h, sd_mine[p + "ffn_norm.weight"])
    g = hn @ sd_mine[p + "feed_forward.w1.weight"].T
    ff = (g * torch.sigmoid(g) * (hn @ sd_mine[p + "feed_forward.w3.weight"].T))
    h = h + ff @ sd_mine[p + "feed_forward.w2.weight"].T
print("[diff] final-norm logits maxabs:",
      f"{float(((L.rmsnorm(h, sd_mine['norm.weight']) @ sd_mine['output.weight'].T) - my).abs().max()):.3e}")