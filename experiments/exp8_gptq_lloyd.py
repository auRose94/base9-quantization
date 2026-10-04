#!/usr/bin/env python3
"""exp8 — compensated FITTED codebooks: GPTQ + Lloyd–Max on the same pass.

Open item from session 3: combine error compensation with fitted codebooks —
expected to be the best accuracy-per-bit point on the curve. Also re-measures
(actually evaluates, not estimates) the fp16-codebook variant whose ppl was an
estimate in exp4b.

Method: per (row, group-of-64) Lloyd–Max fitted centers from the ORIGINAL W
(9 centers, sorted). RTN = nearest-center assignment (cross-check: must
reproduce exp4b's 40.534). GPTQ = exp6 machinery, column-wise compensated
rounding onto the static fitted codebook. fp16 codebook variants round the
STORED centers to float16 before quantization/dequant — side info honestly
fp32 = 4.5 b/param, fp16 = 2.25 b/param (both counted).

Pre-registered: P11 — fp16-codebook ppl within noise (±0.3) of fp32-codebook
ppl in both RTN and GPTQ phases.

Out: results/exp8_gptq_lloyd.{csv,md}
"""
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch
from rans import rans_bits
import exp4_real_model_ptq as e4
import exp4b_group_scales as e4b
import exp6_gptq_grids as e6

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
GROUP = 64
K = 9
CODEC_K = 12


def lloyd_centers_group(W, k=K, g=GROUP, iters=80):
    """Lloyd–Max per (row, group); returns sorted centers (r, c/g, k)."""
    r, c = W.shape
    Wg = W.reshape(r, c // g, g)
    m = Wg.abs().amax(dim=2, keepdim=True)
    centers = torch.linspace(-1, 1, k, device=W.device).view(1, 1, k) * m
    ones = torch.ones_like(Wg)
    scale = float(m.max().item()) + 1e-30
    for _ in range(iters):
        bnd = ((centers[..., 1:] + centers[..., :-1]) / 2.0).contiguous()
        idx = torch.searchsorted(bnd, Wg).clamp(0, k - 1).long()
        ssum = torch.zeros_like(centers).scatter_add_(-1, idx, Wg)
        cnt = torch.zeros_like(centers).scatter_add_(-1, idx, ones)
        new = torch.where(cnt > 0, ssum / cnt.clamp(min=1e-30), centers)
        done = (new - centers).abs().max().item() < 1e-6 * scale
        centers = new
        if done:
            break
    centers = torch.sort(centers, dim=-1).values  # safety against reordering
    bnd = ((centers[..., 1:] + centers[..., :-1]) / 2.0).contiguous()
    idx = torch.searchsorted(bnd, Wg).clamp(0, k - 1).long()
    deq = centers.gather(-1, idx).float()
    return centers, deq.reshape(r, c), idx.reshape(r, c)


def quantize_to_centers(W1_col, C):
    """Nearest fitted center per row for one compensated column."""
    bnd = ((C[:, 1:] + C[:, :-1]) / 2.0).contiguous()
    idx = torch.searchsorted(bnd, W1_col.unsqueeze(1)).clamp(0, C.shape[-1] - 1)
    deq = C.gather(1, idx).squeeze(1)
    return deq, idx.squeeze(1)


def gptq_quantize_centers_h(W, H, centers, compensate=True, g=GROUP):
    """Column-wise pass over W: quantize each (compensated) column to the
    static fitted `centers` (r, c/g, k). compensate=True runs the GPTQ update
    with the dampered inverse-Hessian factor Hinv = cholesky(H^{-1}, upper)."""
    r, c = W.shape
    Hinv = e6.gptq_hinv(H, W.device) if compensate else None
    W1 = W.clone()
    IDX = torch.zeros_like(W, dtype=torch.long)
    Q = torch.empty_like(W)
    for i in range(c):
        grp = i // g
        C = centers[:, grp]
        w = W1[:, i]
        deq, s_idx = quantize_to_centers(w, C)
        IDX[:, i] = s_idx
        Q[:, i] = deq
        if compensate and Hinv is not None and i + 1 < c:
            err = (w - deq) / Hinv[i, i]
            W1[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    return Q, IDX


PHASES = [
    ("RTN fp32-codebook", False, False, 32),
    ("GPTQ fp32-codebook", True, False, 32),
    ("RTN fp16-codebook", False, True, 16),
    ("GPTQ fp16-codebook", True, True, 16),
]


def main():
    dev = e4.device_setup()
    model, tok = e4.load_model(dev)
    layers = e4.quantized_layers(model)
    n_lin = sum(m.weight.numel() for _, m in layers)
    saved = {name: m.weight.data.detach().clone() for name, m in layers}
    cal_all = e6.get_cal_blocks(tok, dev)
    cal_batches = [cal_all[i * 8:(i + 1) * 8] for i in range(e6.N_CAL // 8)]
    blocks = e4.get_blocks(tok, dev)
    ppl0 = e4.perplexity(model, blocks)
    print(f"fp32 baseline (sanity): {ppl0:.3f}")

    rows = []
    t_start = time.time()
    for label, use_gptq, use_fp16, side_bits_per_center in PHASES:
        t1 = time.time()
        streams = []
        serr = wsum2 = wsum = 0.0
        n_side = 0
        for name, m in layers:
            tr = type(m).__name__ == "Conv1D"
            W = m.weight.data.t() if tr else m.weight.data
            centers, deq_rtn, idx_rtn = lloyd_centers_group(W)
            stored = centers.half().float() if use_fp16 else centers
            if use_gptq:
                H = e6.collect_hessian(model, m, cal_batches)
                Q, IDX = gptq_quantize_centers_h(W, H, stored, compensate=True)
            else:
                Q, IDX = deq_rtn, idx_rtn
                if use_fp16:
                    # re-assign to the STORED (fp16-rounded) centers — honest dequant
                    Q, IDX = gptq_quantize_centers_h(W, None, stored, compensate=False)
            serr += ((W - Q) ** 2).sum().item()
            wsum2 += (W * W).sum().item()
            wsum += W.sum().item()
            r_, c_ = W.shape
            n_side += r_ * (c_ // GROUP) * K
            m.weight.data = (Q.t().contiguous() if tr else Q.contiguous())
            streams.append(IDX.detach().cpu().numpy().reshape(-1))
        sym = np.concatenate(streams)
        rb = rans_bits(sym, K, CODEC_K)
        ppl = e4.perplexity(model, blocks)
        rel_mse = serr / max(wsum2 - len(sym) * (wsum / len(sym)) ** 2, 1e-30)
        side = side_bits_per_center * n_side / n_lin
        eff = rb + side
        rows.append(dict(phase=label, ppl=ppl, digits_rans=rb, eff=eff,
                         mse=rel_mse, note=f"{time.time()-t1:.0f}s"))
        print(f"{label:<22} ppl {ppl:8.3f} | digits {rb:.3f} + codebook "
              f"{side:.3f} = {eff:.3f} eff b/p | mse {rel_mse:.5f}")
        for name, m in layers:
            m.weight.data.copy_(saved[name])

    with open(RESULTS / "exp8_gptq_lloyd.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["phase", "ppl", "digits_rans", "eff",
                                           "mse", "note"])
        w.writeheader()
        w.writerows(rows)

    rtn32 = next(r for r in rows if r["phase"] == "RTN fp32-codebook")["ppl"]
    p11 = all(abs(r["ppl"] - rtn32) < 0.3 or r["phase"] == "RTN fp32-codebook"
              for r in rows if "fp16" in r["phase"])
    lines = [
        f"# exp8 — compensated fitted codebooks (GPTQ + Lloyd–Max, 9 centers, g={GROUP})",
        "",
        f"Cross-check: 'RTN fp32-codebook' must equal exp4b's LM9-g64 = 40.534.",
        f"fp32 baseline {ppl0:.3f}; noise ±0.3 ppl. Codebook side info:",
        f"{K} centers/group of {GROUP} → fp32 = {32*K/GROUP:.2f} b/param,",
        f"fp16 = {16*K/GROUP:.2f} b/param (counted). Pre-registered P11:",
        "fp16-codebook ppl within noise of fp32-codebook ppl.",
        "",
        "| phase | ppl | digit rANS | eff b/param | lin rel MSE | wall |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['phase']} | {r['ppl']:.3f} | {r['digits_rans']:.3f} |"
                     f" {r['eff']:.3f} | {r['mse']:.5f} | {r['note']} |")
    lines += ["", f"P11 (fp16 within ±0.3 of fp32): **{'PASS' if p11 else 'FAIL'}**",
              "", f"wall {time.time()-t_start:.0f}s", ""]
    (RESULTS / "exp8_gptq_lloyd.md").write_text("\n".join(lines))
    print("wrote results/exp8_gptq_lloyd.md")


if __name__ == "__main__":
    main()