#!/usr/bin/env python3
"""Debug 2: map the llm.c state dict into HF LlamaForCausalLM and eval.
Decides between (a) my forward is wrong -> HF wins, (b) eval data prep is
wrong -> HF also fails."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

DEV = "cuda"

sd0, ma, ck = L.load_state_dict()
cfg = LlamaConfig(vocab_size=ma["vocab_size"], hidden_size=ma["dim"],
                  num_hidden_layers=ma["n_layers"], num_attention_heads=ma["n_heads"],
                  num_key_value_heads=ma["n_kv_heads"],
                  intermediate_size=ma["hidden_dim"] if "hidden_dim" in ma else 172,
                  max_position_embeddings=ma["max_seq_len"],
                  rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False,
                  attention_bias=False, mlp_bias=False)
model = LlamaForCausalLM(cfg)
m = model.model
sd = {}
sd["model.embed_tokens.weight"] = sd0["tok_embeddings.weight"]
for i in range(ma["n_layers"]):
    src = f"layers.{i}"
    dst = f"model.layers.{i}"
    sd[f"{dst}.self_attn.q_proj.weight"] = sd0[f"{src}.attention.wq.weight"]
    sd[f"{dst}.self_attn.k_proj.weight"] = sd0[f"{src}.attention.wk.weight"]
    sd[f"{dst}.self_attn.v_proj.weight"] = sd0[f"{src}.attention.wv.weight"]
    sd[f"{dst}.self_attn.o_proj.weight"] = sd0[f"{src}.attention.wo.weight"]
    sd[f"{dst}.mlp.gate_proj.weight"] = sd0[f"{src}.feed_forward.w1.weight"]
    sd[f"{dst}.mlp.down_proj.weight"] = sd0[f"{src}.feed_forward.w2.weight"]
    sd[f"{dst}.mlp.up_proj.weight"] = sd0[f"{src}.feed_forward.w3.weight"]
    sd[f"{dst}.input_layernorm.weight"] = sd0[f"{src}.attention_norm.weight"]
    sd[f"{dst}.post_attention_layernorm.weight"] = sd0[f"{src}.ffn_norm.weight"]
sd["model.norm.weight"] = sd0["norm.weight"]
sd["lm_head.weight"] = sd0["output.weight"]
missing, unexpected = model.load_state_dict(sd, strict=True), None
model = model.to(DEV).eval()

N = 384
tokens = L.valid_tokens(N * 512).astype(np.int64)
win = torch.from_numpy(tokens[: N * 512]).view(N, 512).to(DEV)
acc = nll = ntok = 0
with torch.no_grad():
    for i in range(0, N, 32):
        b = win[i:i + 32]
        lg = model(b).logits[:, :-1]
        tg = b[:, 1:]
        lp = torch.log_softmax(lg, -1)
        sel = lp.gather(-1, tg[..., None]).squeeze(-1)
        nll += -sel.sum().item()
        acc += (lg.argmax(-1) == tg).sum().item()
        ntok += tg.numel()
        if i == 0:
            first = (lg.argmax(-1) == tg).float().mean().item()
loss = nll / ntok
print(f"[hf] loss {loss:.4f} (ppl {np.exp(loss):.2f}), top-1 acc {acc / ntok:.3f}, "
      f"batch0 acc {first:.3f}")
print("[hf] sample:", end=" ")
out = model.generate(input_ids=torch.tensor([[1] + L.get_sp().encode("Once upon a time,")],
                                            device=DEV), max_new_tokens=90, do_sample=True,
                     top_k=50, temperature=0.8)
print(L.get_sp().decode(out[0].tolist()))