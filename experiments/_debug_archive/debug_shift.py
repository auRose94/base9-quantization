#!/usr/bin/env python3
"""Debug 5: is the eval token stream systematically shifted/wrong-mapped?
(1) read ref-model predictions vs my ids on a window; (2) +-shift scan; (3) try
TinyStoriesV2-GPT4-valid as the corpus; (4) vocab-table sanity."""
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

sp = L.get_sp()
tok = np.load(os.path.join(DATA, "valid_tokens.npy"))

# (1) predictions vs ids — read both
b0 = torch.from_numpy(tok[:512].astype(np.int64))[None].to(DEV)
with torch.no_grad():
    ref(b0, targets=b0)
    lg = None
    # need full logits: reimpl with targets (returns only loss); use manual pass
    # -> easiest: forward with targets=None gives last pos only; so do per-position hack:
    # instead: compare top-20 predicted ids at position 200 vs actual ids around there
    h = None
    from model import RMSNorm  # not needed; just do a second model call trick
# full logits: temporarily disable the last-position shortcut by patching targets path
# model returns logits full-size when targets given. Use CE-free call:
with torch.no_grad():
    logits_full = ref(b0, targets=b0)  # logits (1,512,512)
    pred = logits_full.argmax(-1)[0]
print("OUR TOKENS   :", sp.decode(tok[100:180].tolist()))
print("MODEL PREDICT:", sp.decode(pred[100:180].tolist()))
print("our ids[100:120] :", tok[100:120].tolist())
print("pred ids[100:120]:", pred[100:120].tolist())

# (2) shift scan: does +-k shift of the stream recover the model?
win = torch.from_numpy(tok[: 48 * 512].astype(np.int64)).view(48, 512)
base = {}
with torch.no_grad():
    for sh in range(-4, 5):
        t = torch.from_numpy((tok[: 32 * 512] + sh).astype(np.int64)).view(32, 512).to(DEV)
        t = t.clamp(0, 511)
        ref(t, targets=t)
        base[sh] = ref.last_loss.item()
print("[shift] loss per shift:", {k: round(v, 3) for k, v in sorted(base.items())})

# (3) vocab table sanity: what pieces live at which ids
print("[vocab] pieces 0,1,2,100,295,301,350,427,500:", [sp.id_to_piece(i) for i in
      [0, 1, 2, 100, 295, 301, 350, 427, 500]])
print("[vocab] id of 'the','a','was','Once','upon',' girl':",
      [sp.piece_to_id(p) for p in ["the", "a", "was", "▁Once", "▁upon", "▁girl"]])