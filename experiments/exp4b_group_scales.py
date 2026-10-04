#!/usr/bin/env python3
"""exp4b — per-GROUP scales (g=64) on TinyStories-33M: does group scaling fix
the uniform-9 vs uniform-8 perplexity inversion seen with per-row scales in exp4?

Per-row max-scaling concentrates outliers in the outermost level; transformer
outlier channels sit inside rows, not whole rows. Group scales along the input
dimension (64 weights per scale, GGUF-style) localize the outliers.

SIDE-INFO ACCOUNTING (same rule as exp4): every fp32 scalar the decoder needs
is counted. Uniform grids: 1 scale per group = 32/64 = 0.5 b/param. Lloyd–Max
groups: 9 fitted center floats per group = 9·32/64 = 4.5 b/param (a per-group
codebook is expensive side info — fp16 centers would halve it; estimates shown
separately, without re-eval).

Schemes: int4 (15), 8-level, 9-level ninths, 9-level Lloyd–Max, all g=64.

Out: results/exp4b_group_scales.{csv,md}
"""
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch
from rans import rans_bits, entropy_bits
import exp4_real_model_ptq as e4

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
GROUP = 64


def q_uniform_group(W, k, g=GROUP):
    r, c = W.shape
    assert c % g == 0, f"col dim {c} not divisible by {g}"
    Wg = W.reshape(r, c // g, g)
    m = Wg.abs().amax(dim=2, keepdim=True)
    step = 2.0 * m / (k - 1)
    idx = torch.clamp(torch.round((Wg + m) / step), 0, k - 1).long()
    deq = (-m + idx * step).float().reshape(r, c)
    return deq, idx.reshape(r, c)


def q_lloyd_group(W, k, g=GROUP, iters=80):
    r, c = W.shape
    assert c % g == 0, f"col dim {c} not divisible by {g}"
    Wg = W.reshape(r, c // g, g)
    m = Wg.abs().amax(dim=2, keepdim=True)
    centers = torch.linspace(-1, 1, k, device=W.device).view(1, 1, k) * m
    ones = torch.ones_like(Wg)
    scale = float(m.max().item()) + 1e-30
    for _ in range(iters):
        b = (centers[..., 1:] + centers[..., :-1]) / 2.0
        idx = torch.searchsorted(b, Wg).clamp(0, k - 1).long()
        ssum = torch.zeros_like(centers).scatter_add_(-1, idx, Wg)
        cnt = torch.zeros_like(centers).scatter_add_(-1, idx, ones)
        new = torch.where(cnt > 0, ssum / cnt.clamp(min=1e-30), centers)
        done = (new - centers).abs().max().item() < 1e-6 * scale
        centers = new
        if done:
            break
    b = (centers[..., 1:] + centers[..., :-1]) / 2.0
    idx = torch.searchsorted(b, Wg).clamp(0, k - 1).long()
    deq = centers.gather(-1, idx).float()
    return deq.reshape(r, c), idx.reshape(r, c)


# (quantizer, args, side-info floats per group): uniform = 1 scale (m);
# Lloyd–Max = 9 fitted centers (no separate scale; centers define the grid).
SCHEMES = [
    ("int4 (15) g64", q_uniform_group, (15,), 1),
    ("8-level g64", q_uniform_group, (8,), 1),
    ("9-level ninths g64 (uniform)", q_uniform_group, (9,), 1),
    ("9-level Lloyd-Max g64 (fitted)", q_lloyd_group, (9,), 9),
]


def main():
    dev = e4.device_setup()
    model, tok = e4.load_model(dev)
    layers = e4.quantized_layers(model)
    n_lin = sum(m.weight.numel() for _, m in layers)
    n_groups = 0
    for _, m in layers:
        r, c = (m.weight.t() if type(m).__name__ == "Conv1D" else m.weight).shape
        n_groups += r * (c // GROUP)
    saved = {name: m.weight.data.detach().clone() for name, m in layers}
    blocks = e4.get_blocks(tok, dev)
    ppl0 = e4.perplexity(model, blocks)
    print(f"fp32 baseline (sanity): {ppl0:.3f}; {n_lin:,} linear params, "
          f"{n_groups:,} group scales")

    rows = []
    t0 = time.time()
    for scheme, qfn, args, side_per_group in SCHEMES:
        t1 = time.time()
        streams = []
        serr = wsum2 = wsum = 0.0
        n_side = 0
        for name, m in layers:
            W = m.weight.data
            tr = type(m).__name__ == "Conv1D"
            Wq = W.t() if tr else W
            deq, idx = qfn(Wq, *args)
            serr += ((Wq - deq) ** 2).sum().item()
            wsum2 += (Wq * Wq).sum().item()
            wsum += Wq.sum().item()
            m.weight.data = (deq.t() if tr else deq).contiguous()
            streams.append(idx.detach().cpu().numpy().reshape(-1))
            n_side += idx.shape[0] * (idx.shape[1] // GROUP) * side_per_group
        sym = np.concatenate(streams)
        k = int(sym.max()) + 1
        ent = entropy_bits(sym, k)
        rb_digits = rans_bits(sym, k)
        side32 = 32.0 * n_side / n_lin
        side16 = 16.0 * n_side / n_lin
        eff32 = rb_digits + side32
        ppl = e4.perplexity(model, blocks)
        rel_mse = serr / max(wsum2 - len(sym) * (wsum / len(sym)) ** 2, 1e-30)
        rows.append(dict(scheme=scheme, levels=k, digits_entropy=ent,
                         digits_rans=rb_digits, side32=side32, side16=side16,
                         eff32=eff32, ppl=ppl, mse=rel_mse,
                         note=f"{time.time()-t1:.0f}s"))
        print(f"{scheme:<32} ppl {ppl:8.3f} | digits rANS {rb_digits:.3f} + side "
              f"{side32:.2f} (fp32) = {eff32:.3f} eff | mse {rel_mse:.5f}")
        for name, m in layers:
            m.weight.data.copy_(saved[name])

    with open(RESULTS / "exp4b_group_scales.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scheme", "levels", "digits_entropy",
                                           "digits_rans", "side32", "side16",
                                           "eff32", "ppl", "mse", "note"])
        w.writeheader()
        w.writerows(rows)

    lines = [
        f"# exp4b — per-group scales (g={GROUP}) on {e4.MODEL_ID}",
        "",
        f"Group scales along the input dim ({GROUP} weights/scale). Side info",
        "counted: uniform grids store 1 fp32 scale per group (0.50 b/param);",
        "Lloyd–Max stores 9 fp32 fitted centers per group (4.50 b/param — ",
        "fp16 centers would be 2.25, estimates without re-eval).",
        f"Companion to exp4 (per-row); same eval ({e4.EVAL_BLOCKS}x{e4.BLOCK} tokens),",
        f"fp32 baseline {ppl0:.3f} (matches exp4 exactly).",
        "",
        "| scheme | levels | digit rANS | side fp32 | eff (fp32 side) | ppl | lin rel MSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['scheme']} | {r['levels']} | {r['digits_rans']:.3f} |"
                     f" {r['side32']:.2f} | {r['eff32']:.3f} | {r['ppl']:.3f} |"
                     f" {r['mse']:.5f} |")
    lines += [
        "",
        "fp16-side estimates (no re-eval): "
        + "; ".join(f"{r['scheme']} {r['digits_rans'] + r['side16']:.2f} b/param"
                    for r in rows),
        "",
        f"wall {time.time()-t0:.0f}s", "",
    ]
    (RESULTS / "exp4b_group_scales.md").write_text("\n".join(lines))
    print("wrote results/exp4b_group_scales.md")


if __name__ == "__main__":
    main()