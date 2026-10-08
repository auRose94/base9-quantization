#!/usr/bin/env python3
"""Debug 9: A/B my eq_lib eval vs reference eval — same windows, same pairing."""
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
mine_sd, ma, _ = L.load_state_dict()
mine_sd = L.to_device(mine_sd, DEV)

tokens = np.load(os.path.join(DATA, "valid_tokens.npy")).astype(np.int64)

@torch.no_grad()
def mine_loss(t_windows):
    losses = []
    for b in t_windows:
        lg = L.forward(mine_sd, b, {"dim": 64, "n_layers": 5, "n_heads": 8, "n_kv_heads": 4,
                                    "vocab_size": 512, "max_seq_len": 512})[:, :-1]
        tg = b[:, 1:]
        nll = -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).squeeze(-1)
        losses.append(nll.mean().item())
    return float(np.mean(losses))

@torch.no_grad()
def ref_loss_windows(t_windows):
    losses = []
    for b in t_windows:
        lg = ref(b, targets=b)
        full = lg[..., :-1, :]
        tg = b[:, 1:]
        nll = -torch.log_softmax(full, -1).gather(-1, tg[..., None]).squeeze(-1)
        losses.append(nll.mean().item())
    return float(np.mean(losses))

ws = [torch.from_numpy(tokens[i * 512:(i + 1) * 512].reshape(1, 512)).to(DEV) for i in range(48)]
ws_b = torch.cat(ws, 0)
print("[A/B] mine  48 windows:", round(mine_loss(ws), 4))
print("[A/B] ref   48 windows:", round(ref_loss_windows(ws), 4))

# batch consistency: single window0 vs batched
b0 = ws_b[:1]
b2 = ws_b[:2]
lg0 = L.forward(mine_sd, b0, {"dim": 64, "n_layers": 5, "n_heads": 8, "n_kv_heads": 4,
                              "vocab_size": 512, "max_seq_len": 512})
lg2 = L.forward(mine_sd, b2, {"dim": 64, "n_layers": 5, "n_heads": 8, "n_kv_heads": 4,
                              "vocab_size": 512, "max_seq_len": 512})
print("mine: batched logits[0]==solo logits[0]:",
      float((lg2[0] - lg0[0]).abs().max()))
r0 = ref(b0, targets=b0)[..., :-1, :]
r2 = ref(b2, targets=b2)[..., :-1, :]
print("ref : batched logits[0]==solo logits[0]:", float((r2[0] - r0[0]).abs().max()))
print("mine vs ref solo window0 logits maxabs:", float((lg0[0, :-1] - r0[0]).abs().max()))
print("mine vs ref solo window0 nll diff:",
      float((-torch.log_softmax(lg0[0, :-1], -1).gather(-1, b0[0, 1:, None])
             - -torch.log_softmax(r0[0], -1).gather(-1, b0[0, 1:, None])).abs().max()))