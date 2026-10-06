#!/usr/bin/env python3
"""K9-quantize a model into .k9 container files — CPU-only, exp24's writer.

Same quantize-in-place -> entropy-bits -> rANS pipeline as
`experiments/exp24_qwen7b.py` (symmetric odd grid, per-row group scales, ent8,
C coder with parallel encode), but it loads a local model dir — the merged,
tuned model rather than the HF hub ID — and does no ppl evaluation, since the
quality question here is answered by the 24-task GDScript eval, not perplexity.
Runs on CPU so it can overlap with GPU work; ~24 GB RAM.

    python3 quantize_k9.py --model out/merged_7b --out results/qwen7b_tuned_k15_embed99.k9 \
        --k-body 15 --k-embed 99
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

import numpy as np
import torch
import k9
from rans import entropy_bits

GROUP = 64
THREADS = min(24, os.cpu_count() or 1)


def body_layers(model):
    """exp18's definition, replicated locally to keep this import-light:
    every 2-D Linear except lm_head."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and m.weight.dim() == 2
            and not n.endswith("lm_head")]


def quantize_inplace(W, k, g=GROUP, row_chunk=2048):
    """exp24's routine verbatim: symmetric odd grid, row-chunked, written back
    into W. Returns digit indices (CPU uint8) and per-(row, group) max scales."""
    r, c = W.shape
    idx = torch.empty((r, c), dtype=torch.uint8, device="cpu")
    scales = torch.empty((r, c // g), dtype=torch.float32, device="cpu")
    for a in range(0, r, row_chunk):
        b = min(r, a + row_chunk)
        Wc = W[a:b].float().reshape(b - a, c // g, g)
        m = Wc.abs().amax(dim=2)
        step = 2.0 * m / (k - 1)
        ic = torch.clamp(torch.round((Wc + m.unsqueeze(-1)) / step.unsqueeze(-1)),
                         0, k - 1).long()
        deq = (-m.unsqueeze(-1) + ic * step.unsqueeze(-1)).reshape(b - a, c)
        idx[a:b] = ic.reshape(b - a, c).to(torch.uint8).cpu()
        scales[a:b] = m.cpu()
        W[a:b] = deq.to(W.dtype)
        del Wc, m, step, ic, deq
    return idx, scales


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model dir to quantize")
    ap.add_argument("--out", required=True, help="output .k9 path")
    ap.add_argument("--k-body", type=int, default=15)
    ap.add_argument("--k-embed", type=int, default=99)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t_all = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    n_total = sum(p.numel() for p in model.parameters())
    n_body = sum(m.weight.numel() for _, m in body_layers(model))
    print(f"{a.model}: {n_total:,} params (body {n_body:,})", flush=True)
    print(f"k9 CPU quantize: k_body={a.k_body} k_embed={a.k_embed} "
          f"| threads {THREADS}", flush=True)

    tensors = []
    dig_bits = sc_bytes = 0.0
    t0 = time.time()
    for name, m in body_layers(model):
        W = m.weight.data
        idx, scales_t = quantize_inplace(W, a.k_body)
        idx_u8 = idx.numpy()
        dig_bits += entropy_bits(idx_u8.reshape(-1), a.k_body) * W.numel()
        s_np = scales_t.numpy()
        sc_bytes += len(k9.encode_scales(s_np, "ent8")) + 2 * a.k_body
        rec = k9.encode_tensor(idx_u8, a.k_body, s_np, "ent8")
        del idx_u8, idx, scales_t, s_np
        tensors.append(dict(name=name, shape=tuple(W.shape), group=GROUP,
                            scale_mode="ent8", k=a.k_body, rec=rec))

    embed = model.get_submodule("model.embed_tokens")
    head = model.get_submodule("lm_head")
    for name, mod in (("__embed__", embed), ("__lm_head__", head)):
        W = mod.weight.data
        idx, scales_t = quantize_inplace(W, a.k_embed)
        idx_u8 = idx.numpy()
        dig_bits += entropy_bits(idx_u8.reshape(-1), a.k_embed) * W.numel()
        s_np = scales_t.numpy()
        sc_bytes += len(k9.encode_scales(s_np, "ent8")) + 2 * a.k_embed
        rec = k9.encode_tensor(idx_u8, a.k_embed, s_np, "ent8")
        del idx_u8, idx, scales_t, s_np
        tensors.append(dict(name=name, shape=tuple(W.shape), group=GROUP,
                            scale_mode="ent8", k=a.k_embed, rec=rec))
    print(f"quantized+encoded {len(tensors)} tensors in {time.time()-t0:.0f}s "
          f"(est {dig_bits/8e6 + sc_bytes/1e6:.1f} MB from entropy)", flush=True)

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    size = k9.write_k9(out, tensors, threads=THREADS)
    print(f"wrote {out.name}: {size/1e6:.1f} MB = {8*size/n_total:.3f} b/param "
          f"({size/n_total:.3f} B/param) in {time.time()-t_all:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())