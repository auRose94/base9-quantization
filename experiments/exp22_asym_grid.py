#!/usr/bin/env python3
"""exp22 — asymmetric (min+scale) grids vs symmetric absmax at the low-bit points.

Question: does an affine per-group grid `w = lo + d*step` (GGUF Q4_K style)
beat the symmetric odd grid `w = m*(d−H)/H` used by K9 — and does it lift the
low-bit points (641 MB / 809 MB rows)?

Variants, all with group=64 scales:
  sym      symmetric absmax                         1 scale/group
  symclip  symmetric, range = q-quantile of |w|      1 scale/group
  asym     affine over [min, max]                   2 scales/group (lo fp16 + step ent8)

Digits use the same odd alphabets (k = 9/15/27). Byte cost is measured with the
K9 scale codec (ent8). Evaluation: the shared 8-window harness, paired vs fp32,
so results are directly comparable to exp18-21.

Run: python3 exp22_asym_grid.py            # RTN sweep, ~6 min
"""
import csv
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch

import k9
import eval_harness as eh
import exp18_qwen_gptq_noise as e18
from rans import entropy_bits

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
BLOCK, GROUP, N_WINDOWS = 1024, 64, 8
KS = (9, 15, 27)
CLIP_Q = 0.9995


# ----------------------------------------------------------- quantizers ----
def q_sym(W, k, g=GROUP):
    r, c = W.shape
    Wg = W.reshape(r, c // g, g)
    m = Wg.abs().amax(dim=2).float()
    step = 2.0 * m / (k - 1)
    idx = torch.clamp(torch.round((Wg + m.unsqueeze(-1)) / step.unsqueeze(-1)),
                       0, k - 1).long()
    deq = (-m.unsqueeze(-1) + idx * step.unsqueeze(-1)).reshape(r, c)
    return deq, idx.reshape(r, c), dict(m=m)


def q_symclip(W, k, g=GROUP, q=CLIP_Q):
    r, c = W.shape
    Wg = W.reshape(r, c // g, g)
    m = Wg.abs().quantile(q, dim=2).float().clamp_min(1e-12)
    step = 2.0 * m / (k - 1)
    idx = torch.clamp(torch.round((Wg + m.unsqueeze(-1)) / step.unsqueeze(-1)),
                       0, k - 1).long()
    deq = (-m.unsqueeze(-1) + idx * step.unsqueeze(-1)).reshape(r, c)
    return deq, idx.reshape(r, c), dict(m=m)


def q_asym(W, k, g=GROUP):
    r, c = W.shape
    Wg = W.reshape(r, c // g, g)
    lo = Wg.amin(dim=2).float()
    hi = Wg.amax(dim=2).float()
    step = ((hi - lo) / (k - 1)).clamp_min(1e-12)
    idx = torch.clamp(torch.round((Wg - lo.unsqueeze(-1)) / step.unsqueeze(-1)),
                       0, k - 1).long()
    deq = (lo.unsqueeze(-1) + idx * step.unsqueeze(-1)).reshape(r, c)
    return deq, idx.reshape(r, c), dict(lo=lo, step=step)


QUANT = {"sym": q_sym, "symclip": q_symclip, "asym": q_asym}


def scale_bytes(variant, sc):
    """Coded bytes for one tensor's scales (lo is signed -> fp16; step/m -> ent8)."""
    if variant == "asym":
        return 2 * sc["lo"].numel() + len(
            k9.encode_scales(sc["step"].cpu().numpy(), "ent8"))
    return len(k9.encode_scales(sc["m"].cpu().numpy(), "ent8"))


def apply_all(body, embed, k_body, k_embed, variant):
    """Quantize body at k_body and embed at k_embed; return (bits, scale_bytes)."""
    fn = QUANT[variant]
    n = 0
    dig_bits = 0.0
    sc_bytes = 0
    for name, m in body + [("__embed__", embed)]:
        W = m.weight.data
        k = k_body if name != "__embed__" else k_embed
        deq, idx, sc = fn(W, k)
        dig_bits += entropy_bits(idx.to(torch.uint8).cpu().numpy().reshape(-1), k) * W.numel()
        sc_bytes += scale_bytes(variant, sc)
        W.copy_(deq)
    return dig_bits, sc_bytes


def main():
    t_all = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = e18.load_model(dev)
    body = e18.body_layers(model)
    embed = model.model.embed_tokens
    n_total = sum(p.numel() for p in model.parameters())

    wiki = eh.get_windows(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                          "test", N_WINDOWS, BLOCK, label="wiki")
    code = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean-valid", None,
                          "train", N_WINDOWS, BLOCK, label="code")
    master = {n: m.weight.data.detach().to("cpu").clone() for n, m in body}
    master_embed = embed.weight.data.detach().to("cpu").clone()

    def restore():
        for n, m in body:
            m.weight.data.copy_(master[n].to(dev))
        embed.weight.data.copy_(master_embed.to(dev))
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("fp32 reference...")
    pw0 = eh.ppl_per_window(model, wiki)
    pc0 = eh.ppl_per_window(model, code)
    print(f"  code {np.mean(pc0):.3f} | wiki {np.mean(pw0):.3f}")
    restore()

    variants = [("sym", q_sym), ("symclip", q_symclip), ("asym", q_asym)]
    rows, store = [], {}
    rest_mb = 0.0   # everything is quantized; fp32 remainder is tiny
    for k in KS:
        for variant, _ in variants:
            t0 = time.time()
            dig_bits, sc_b = apply_all(body, embed, k, 99, variant)
            pw = eh.ppl_per_window(model, wiki)
            pc = eh.ppl_per_window(model, code)
            key = f"{variant} k={k}"
            store[key] = (pw, pc)
            mb = (dig_bits / 8 + sc_b) / 1e6 + rest_mb
            rows.append(dict(variant=variant, k=k, digits_bpp=dig_bits / n_total,
                             scale_bpp=8 * sc_b / n_total, MB=mb,
                             code=float(np.mean(pc)), code_se=float(np.std(pc, ddof=1) / math.sqrt(len(pc))),
                             wiki=float(np.mean(pw))))
            print(f"{key:<12} {mb:7.1f} MB | digits {dig_bits/n_total:5.3f} + scales"
                  f" {8*sc_b/n_total:5.3f} b/p | code {np.mean(pc):7.3f} |"
                  f" {time.time()-t0:.0f}s")
            restore()

    for r in rows:
        pw, pc = store[f"{r['variant']} k={r['k']}"]
        r["d_code"], r["d_code_se"] = eh.paired_delta(pc, pc0)
        r["d_wiki"], _ = eh.paired_delta(pw, pw0)

    cols = list(rows[0].keys())
    with open(RESULTS / "exp22_asym_grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# exp22 — asymmetric (min+scale) grids vs symmetric absmax",
        "",
        "Qwen2.5-Coder-1.5B-Instruct, body k and embedding k=99, group=64.",
        "`sym` = current K9 (symmetric absmax, 1 scale/group); `symclip` = symmetric",
        f"with the range clipped at the {CLIP_Q:g} quantile of |w|; `asym` = affine",
        "min..max (2 scales/group: lo fp16 + step ent8). Digits on the same odd",
        "alphabets. Bytes measured with the K9 ent8 scale codec.",
        "",
        "| variant | k | digits b/p | scales b/p | MB | code ppl | Δcode vs fp32 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['k']} | {r['digits_bpp']:.3f} |"
                     f" {r['scale_bpp']:.3f} | {r['MB']:.1f} | {r['code']:.3f}"
                     f" | {r['d_code']:+.3f} ± {r['d_code_se']:.3f} |")
    by = {(r["variant"], r["k"]): r for r in rows}
    lines += [
        "",
        "## Reading",
        "",
        "- Compare variants **at the same k** (digits ~equal): the difference is",
        "  the grid shape. Then compare at matched **MB** (asym pays 2 scales/group).",
        "",
        f"wall {time.time()-t_all:.0f}s", "",
    ]
    (RESULTS / "exp22_asym_grid.md").write_text("\n".join(lines))
    print("wrote results/exp22_asym_grid.{csv,md}")


if __name__ == "__main__":
    main()
