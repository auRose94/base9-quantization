#!/usr/bin/env python3
"""Debug 6: settle the tokenizer mismatch. Dump the piece table; eval ref-model
loss under (a) shipped tok512.model ids, (b) self-trained tok512-clone ids."""
import os
import sys

import numpy as np
import sentencepiece as spm
import torch

sys.path.insert(0, "/tmp/ref")
sys.path.insert(0, os.path.dirname(__file__))
from model import ModelArgs, Transformer
import eq_lib as L  # noqa: E402

DEV = "cuda"
DATA = os.path.join(L.ROOT, "data")

# ---- piece table ----
sp = L.get_sp()
pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
print("[vocab] pieces 256..320:", pieces[256:320])
for s in ("▁the", "▁a", "▁Once", "▁once", "▁upon", "▁time", "▁girl", "▁said", "▁was",
          "the", "▁Once ▁upon", "▁story"):
    print(f"[vocab] id({s!r}) = {sp.piece_to_id(s)}")

# ---- ref model ----
ck = torch.load(os.path.join(DATA, "stories260K.pt"), map_location="cpu", weights_only=False)
args = ModelArgs(**ck["model_args"])
ref = Transformer(args)
ref.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=False)
ref = ref.eval().to(DEV)


@torch.no_grad()
def ref_loss(ids, n_win=48):
    t = torch.from_numpy(np.asarray(ids, dtype=np.int64)[: n_win * 512]).view(n_win, 512).to(DEV)
    ref(t, targets=t)
    return ref.last_loss.item()


# (a) shipped tok512 on valid-text stories (regex split, no EOS like training prep)
import re
txt = open(os.path.join(DATA, "TinyStories-valid.txt"), encoding="utf-8").read()
docs = [d.strip() for d in re.split(r"<END_OF_DOC_TAG>|_{8,}", txt) if d.strip()]
ids = []
for d in docs:
    ids += [1] + sp.encode(d)          # bos=True, eos=False (llama2.c prep)
print(f"[a] shipped tok512: {len(ids):,} tokens, ref loss {ref_loss(ids):.4f}")

# (b) self-trained 512-BPE clone on the SAME text, karpathy settings
sp2_path = os.path.join(DATA, "tok512_self.model")
tiny = os.path.join(DATA, "_tiny_valid.txt")
with open(tiny, "w", encoding="utf-8") as f:
    for d in docs[:1200]:
        f.write(d + "\n")
spm.SentencePieceTrainer.train(
    input=tiny, model_prefix=sp2_path[:-6], model_type="bpe", vocab_size=512,
    self_test_sample_size=0, input_format="text", character_coverage=1.0,
    split_digits=True, allow_whitespace_only_pieces=True, byte_fallback=True,
    unk_surface=r" ⁇ ", normalization_rule_name="identity",
    num_threads=os.cpu_count())
sp2 = spm.SentencePieceProcessor(model_file=sp2_path)
ids2 = []
for d in docs:
    ids2 += [1] + sp2.encode(d)
print(f"[b] self-trained tok512: {len(ids2):,} tokens, ref loss {ref_loss(ids2):.4f}")

# (c) sanity: model on ITS OWN greedily generated text stream (should be memorized)
sp3 = sp
ids3 = [1] + sp3.encode("Once upon a time, there was a little girl named Lily. She loved"
                        " to play outside in the park. One day, she saw a big, red ball.")
t = torch.tensor([ids3], device=DEV)
ref(t, targets=t)
print(f"[c] ref loss on readme-story prompt: {ref.last_loss.item():.4f}")

# both tokenizers on the same 80 chars — show boundary disagreement
demo = docs[0][:80]
print("[demo]", repr(demo))
print("[demo] tok512  :", sp.encode(demo))
print("[demo] tok512self:", sp2.encode(demo))