#!/usr/bin/env python
"""Qwen3.5-family QLoRA fit probe: bnb-NF4 + LoRA + one fwd/bwd on the 7900 XT.

Probes the two named risks for the Qwen3.8-27B plan:
  1. FLA/triton Gated-DeltaNet kernels under ROCm/RDNA3 (hybrid qwen3_5 arch)
  2. bnb-NF4 + PEFT round-trip (gradients flowing through quantized layers)
Peak-VRAM printout decides the 27B training fit. Reused with --model on the 27B.
"""
import argparse
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3.5-9B")
p.add_argument("--seq", type=int, default=2048)
p.add_argument("--skip-vision", action="store_true")
a = p.parse_args()

from transformers import AutoTokenizer, BitsAndBytesConfig

skip = ["vision_tower", "multi_modal_projector", "visual"] if a.skip_vision else None
qc = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=skip,
)

t0 = time.time()
kw = dict(quantization_config=qc, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
try:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
except Exception as e:
    print(f"[probe] AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:160]})", flush=True)
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
print(f"[probe] load {time.time() - t0:.0f}s class={type(model).__name__}", flush=True)
print(f"[probe] after-load VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
agg = {}
for n, x in model.named_parameters():
    k = ".".join(n.replace("base_model.model.", "").split(".")[:2])
    agg[k] = agg.get(k, 0) + x.numel() * x.element_size()
top = sorted(agg.items(), key=lambda kv: -kv[1])
print("[probe] breakdown: " + ", ".join(f"{k}={v / 2**30:.2f}G" for k, v in top[:8]), flush=True)

tok = AutoTokenizer.from_pretrained(a.model)
from peft import LoraConfig, get_peft_model

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()

tc = getattr(model.config, "text_config", model.config)
V = int(getattr(tc, "vocab_size", 248320))
lats = sum(1 for l in getattr(tc, "layer_types", []) if "linear" in str(getattr(l, "value", l)))
print(f"[probe] layers={getattr(tc, 'num_hidden_layers', '?')} linear_attn={lats} vocab={V}", flush=True)

import torch.nn.functional as F

ids = torch.randint(0, V, (1, a.seq), device=0)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
# phase2/train_qlora.py's validated chunked_loss: trunk without logits, then
# 256-position chunks through checkpoint so fp32 logits never coexist.
body = model.get_base_model().model
head = model.get_base_model().lm_head
hidden = body(input_ids=ids).last_hidden_state
shift = ids[:, 1:]


def part(h, lab, w):
    logits = w(h).float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           lab.reshape(-1), reduction="sum")


total = None
for c in range(0, ids.shape[1] - 1, 256):
    e = min(c + 256, ids.shape[1] - 1)
    s = torch.utils.checkpoint.checkpoint(part, hidden[:, c:e], shift[:, c:e],
                                          head, use_reentrant=False)
    total = s if total is None else total + s
loss = total / (ids.shape[1] - 1)
print(f"[probe] fwd+chunked-CE {time.time() - t0:.1f}s loss={loss.item():.3f} "
      f"(peak after fwd {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)", flush=True)
t0 = time.time()
try:
    loss.backward()
except torch.OutOfMemoryError:
    torch.cuda.synchronize()
    print(f"[probe] OOM in backward; peak at failure = "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB alloc / "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved", flush=True)
    raise SystemExit(3)
torch.cuda.synchronize()
print(f"[probe] bwd {time.time() - t0:.1f}s", flush=True)
print(f"[probe] peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB alloc / "
      f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved", flush=True)
print("[probe] PASS")