#!/usr/bin/env python
"""K9-digit / rANS KV-cache simulation on qwen3_5 hybrid models.

mode sim:    loads the model (bnb-NF4 weights = the serving config), hooks the
             full-attention layers, and quantizes the REAL cache-write tensors:
             K after RoPE (module-level apply_rotary_pos_emb — used only by
             Qwen3_5Attention) and V at the v_proj output. Arms: k9_4 = k15
             symmetric odd grid, 4-bit code + fp16 scale per 64 (4.25 b/elem);
             k9_6 = k63, 6-bit (6.25 b/elem); q4sim = exact ggml q4_0 grid
             (4-bit block32, fp16 scale — apparatus control vs the GPU-swept
             llama.cpp row); none = KV passthrough reference under the same
             weights. PPL from chunked-CE windows over the house wikitext
             corpus, non-overlapping windows, comparable segmentation to
             llama-perplexity.

mode density: quantize captured real K/V tensors to a k-family grid and run
             the REAL rANS codec (experiments/k9.py) on the code stream ->
             measured b/elem (the "rANS on cache" number; accuracy is
             bit-identical to the fixed-rate digit container by construction).
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

P = argparse.ArgumentParser()
P.add_argument("--mode", choices=["sim", "density"], default="sim")
P.add_argument("--model", default="Qwen/Qwen3.5-9B")
P.add_argument("--corpus", default=ROOT / "results/wiki2_raw_test.txt")
P.add_argument("--depths", default="2048,8192")
P.add_argument("--arms", default="none,k9_6,k9_4,q4sim")
P.add_argument("--max-chunks", type=int, default=0, help="0 = all windows")
P.add_argument("--out", default=str(ROOT / "results/kv_digit_sim_9b.csv"))
P.add_argument("--capture", default="/mnt/matrix/Work/models/kv_capture")
P.add_argument("--capture-layers", default="7,31")
P.add_argument("--capture-pass", action="store_true",
               help="with mode sim: stash unquantized K/V of --capture-layers during window 0")
P.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa avoids eager O(T^2) buffers)")
P.add_argument("--quant-head", action="store_true",
               help="NF4-quantize lm_head (llm_int8_skip_modules=[]) — frees ~1.8 GiB on 27B")
P.add_argument("--cpu-embed", action="store_true",
               help="place embed_tokens on CPU (device_map) — frees ~2.4 GiB GPU")
a = P.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        llm_int8_skip_modules=[] if a.quant_head else None,
                        llm_int8_enable_fp32_cpu_offload=True)
t0 = time.time()
kw = dict(quantization_config=qc, dtype=torch.bfloat16,
          low_cpu_mem_usage=True, attn_implementation=a.attn,
          device_map={"": 0} if not a.cpu_embed else
          {"model.embed_tokens": "cpu", "": 0})
try:
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
except Exception:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
tok = AutoTokenizer.from_pretrained(a.model)
model.eval()
tc = getattr(model.config, "text_config", model.config)
V = int(tc.vocab_size)
layer_types = list(getattr(tc, "layer_types"))
full_attn = [getattr(l, "self_attn", None) for l in model.model.layers]
full_attn_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
print(f"[sim] {type(model).__name__} layers={len(layer_types)} full_attn={full_attn_idx} "
      f"vocab={V} load={time.time() - t0:.0f}s", flush=True)

STATE = {"mode": "none", "capture": False}


def k9_qdq(x, k, g=64):
    xf = x.float().reshape(-1, g)
    m = xf.abs().amax(-1, keepdim=True)
    step = 2.0 * m / (k - 1)
    ic = torch.clamp(torch.round((xf + m) / step), 0, k - 1)
    return ((ic * step) - m).reshape(x.shape).to(x.dtype)


def q4_sim(x, g=32):
    xf = x.float().reshape(-1, g)
    d = (xf.abs().amax(-1, keepdim=True) / 8.0).half().float()
    q = torch.clamp(torch.round(xf / d) + 8, 0, 15)
    return ((q - 8) * d).reshape(x.shape).to(x.dtype)


def qdq(x, arm):
    return {"k9_4": lambda t: k9_qdq(t, 15), "k9_5": lambda t: k9_qdq(t, 27),
            "k9_6": lambda t: k9_qdq(t, 63), "k9_7": lambda t: k9_qdq(t, 99),
            "q4sim": q4_sim, "none": lambda t: t}[arm](x)


import transformers.models.qwen3_5.modeling_qwen3_5 as m35  # noqa: E402

ORIG_ROT = m35.apply_rotary_pos_emb


def rot_wrap(q, k, cos, sin, unsqueeze_dim=1):
    qq, kk = ORIG_ROT(q, k, cos, sin, unsqueeze_dim)
    if STATE["mode"] != "none":
        kk = qdq(kk, STATE["mode"])
    return qq, kk


m35.apply_rotary_pos_emb = rot_wrap


class Wrap(torch.nn.Module):
    """Passthrough wrapper applying KV quant (v_proj) or tagged capture (k_norm)."""

    def __init__(self, inner, tag=None, quantize=False):
        super().__init__()
        self.inner = inner
        self.tag = tag
        self.quantize = quantize

    def forward(self, x):
        out = self.inner(x)
        if self.quantize and STATE["mode"] != "none":
            out = qdq(out, STATE["mode"])
        if self.tag is not None and STATE["capture"]:
            Path(a.capture).mkdir(parents=True, exist_ok=True)
            np.save(f"{a.capture}/{self.tag}.npy", out.detach().float().cpu().numpy())
        return out


STATE["capture_layers"] = {int(x) for x in a.capture_layers.split(",")}
for i in full_attn_idx:
    cap = i in STATE["capture_layers"]
    sa = model.model.layers[i].self_attn
    sa.v_proj = Wrap(sa.v_proj, tag=f"L{i}_v" if cap else None, quantize=True)
    sa.k_norm = Wrap(sa.k_norm, tag=f"L{i}_k" if cap else None)

if a.mode == "sim":
    text = Path(a.corpus).read_text()
    ids_all = tok(text, add_special_tokens=True)["input_ids"]
    n = len(ids_all)
    print(f"[sim] corpus tokens {n}", flush=True)
    done = set()
    try:
        with open(a.out) as f:
            for row in csv.DictReader(f):
                if float(row.get("ppl") or 0) > 0:
                    done.add((row["arm"], int(row["ctx"])))
    except FileNotFoundError:
        pass
    new = not Path(a.out).exists()
    cf = open(a.out, "a", newline="")
    w = csv.DictWriter(cf, fieldnames=["arm", "ctx", "ppl", "chunks", "nan", "seconds"])
    if new:
        w.writeheader()
        cf.flush()
    head = model.lm_head
    body = model.model
    with torch.inference_mode():
        for arm in a.arms.split(","):
            for ctx in (int(d) for d in a.depths.split(",")):
                if (arm, ctx) in done:
                    print(f"[sim] skip {arm}@{ctx}", flush=True)
                    continue
                STATE["mode"] = arm
                nnan = [0]
                nll = ntok = 0
                n_chunk = 0
                t0 = time.time()
                try:
                    # warmup: the fla triton autotuner benches a STATEFUL
                    # kernel on the real buffers, multiplying the DeltaNet
                    # recurrence state -> NaN on the first call(s) at each new
                    # shape. Discarded passes absorb it before measurement.
                    xw = torch.tensor([ids_all[:ctx]], device=0)
                    for _ in range(2):
                        body(input_ids=xw)
                    for w0 in range(0, n - ctx, ctx):
                        if a.max_chunks and n_chunk >= a.max_chunks:
                            break
                        STATE["capture"] = (a.capture_pass and STATE["mode"] == "none"
                                            and n_chunk == 0)
                        x = torch.tensor([ids_all[w0:w0 + ctx]], device=0)
                        h = body(input_ids=x).last_hidden_state
                        tot = None
                        for c in range(0, ctx - 1, 256):
                            e = min(c + 256, ctx - 1)
                            lg = head(h[:, c:e]).float()
                            s = F.cross_entropy(lg.reshape(-1, V), x[:, c + 1:e + 1].reshape(-1),
                                                reduction="sum")
                            tot = s if tot is None else tot + s
                        if not torch.isfinite(tot):
                            nnan[0] += 1
                            print(f"[sim] {arm}@{ctx}: NaN window at pos {w0} "
                                  f"(nan count {nnan[0]})", flush=True)
                            continue
                        nll += tot.item()
                        ntok += ctx - 1
                        n_chunk += 1
                except torch.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    print(f"[sim] {arm}@{ctx}: OOM at chunk {n_chunk}: {str(e)[:220]}", flush=True)
                if ntok == 0:
                    continue
                ppl = math.exp(nll / ntok)
                secs = time.time() - t0
                w.writerow({"arm": arm, "ctx": ctx, "ppl": f"{ppl:.4f}", "chunks": n_chunk,
                            "nan": nnan[0], "seconds": f"{secs:.0f}"})
                cf.flush()
                print(f"[sim] {arm}@{ctx}: ppl={ppl:.4f} {n_chunk}ch {secs:.0f}s", flush=True)
        STATE["capture"] = False
    cf.close()

elif a.mode == "density":
    sys.path.insert(0, str(ROOT / "experiments"))
    import k9  # noqa: E402

    ks = [15, 27, 63, 99]
    print(f"[dens] file; k; b/elem; scales included", flush=True)
    for f in sorted(Path(a.capture).glob("*.npy")):
        arr = np.load(f)
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        W = torch.from_numpy(arr)
        r, c = W.shape
        out = [Path(f).name, f"{W.numel()}"]
        for k in ks:
            Wc = W.float().reshape(r, c // 64, 64)
            m = Wc.abs().amax(dim=2)
            idx = torch.clamp(torch.round((Wc + m.unsqueeze(-1)) /
                                          (2.0 * m / (k - 1)).unsqueeze(-1)), 0, k - 1).long()
            rec = k9.encode_tensor(idx.reshape(r, c).numpy().astype(np.uint8), k,
                                   m.numpy().astype(np.float32), "ent8")
            rans_b = (len(rec["digits"]) + len(rec["freq"]) +
                      len(rec["scales"]) + len(rec["perm"]))
            fixed_b = W.numel() * (4.0 if k == 15 else 6.0) / 8 + 2.0 * (r * (c // 64))
            out.append(f"k{k}:rans={8 * rans_b / W.numel():.3f}b:fixed={8 * fixed_b / W.numel():.3f}b")
        print(f"[dens] {'; '.join(out)}", flush=True)