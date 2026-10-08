#!/usr/bin/env python3
"""Debug 3: run llama2.c's OWN model.py on this checkpoint + my eval data.
If it scores ~1.30 loss and reproduces the readme's greedy text, my forward
has a ported bug; if it also fails, the eval data prep is wrong."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/tmp/ref")
sys.path.insert(0, os.path.dirname(__file__))
from model import ModelArgs, Transformer  # ground truth from karpathy/llama2.c
import eq_lib as L  # noqa: E402

DEV = "cuda"
DATA = os.path.join(L.ROOT, "data")

ck = torch.load(os.path.join(DATA, "stories260K.pt"), map_location="cpu", weights_only=False)
args = ModelArgs(**ck["model_args"])
model = Transformer(args)
sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
model.load_state_dict(sd, strict=False)
model = model.eval().to(DEV)
print("freqs buffer:", model.freqs_cos.shape, "hidden_dim computed:", None)

N = 384
tokens = L.valid_tokens(N * args.max_seq_len).astype(np.int64)
win = torch.from_numpy(tokens[: N * 512]).view(N, 512).to(DEV)
nll = ntok = acc = ntok2 = 0
with torch.no_grad():
    for i in range(0, N, 32):
        b = win[i:i + 32]
        model(b, targets=b)
        lo = model.last_loss
        nll += lo.item() * b.numel()
        acc += 0
        ntok2 += b.numel()
loss = nll / ntok2
print(f"[llama2c-model.py] loss on my windows: {loss:.4f} (ppl {np.exp(loss):.2f}), acc {acc / ntok2:.3f}")

# greedy sample — compare against the readme's golden text
torch.manual_seed(0)
x = torch.tensor([[1, *L.get_sp().encode("Once upon a time,")]], device=DEV)
ys = x
with torch.no_grad():
    for _ in range(40):
        lg = model(ys[:, -512:])[0, -1]
        nxt = int(lg.argmax())
        ys = torch.cat([ys, torch.tensor([[nxt]], device=DEV)], 1)
sp = L.get_sp()
print("[llama2c-model.py] greedy:", sp.decode(ys[0].tolist()))
print("README golden: Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. ...")