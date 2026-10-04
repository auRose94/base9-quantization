#!/usr/bin/env python3
"""exp6 — "more math on the GPU", quality side: GPTQ-style Hessian error
compensation applied to the ninths/integer grids of exp4b.

Method (Frantar et al., "GPTQ", arXiv:2210.17323, ICLR 2023):
  - calibration: N_CAL blocks of the TinyStories TRAIN split (never the eval
    window), pushed through the model — with already-quantized earlier layers,
    so Hessians see realistic quantized inputs;
  - per layer: H = 2·XᵀX over calibration activations; damper 1% mean diag;
    Hinv = upper Cholesky of H⁻¹ (CPU float64 — cusolver on this Blackwell
    GPU + torch 2.14 errors in cusolverDnXpotrs);
  - columns of W (input dim) in order: quantize with the SAME per-(row,
    group-of-64-input-columns) grid as exp4b (static scales from the
    ORIGINAL W), then propagate the rounding error to later columns.

Each scheme runs twice — RTN (compensation off; cross-validated against
exp4b's numbers) and GPTQ (compensation on) — so the table isolates the
compensation effect. Grids/scales/side-info identical to exp4b throughout.

Out: results/exp6_gptq_grids.{csv,md}
"""
import csv
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch
from rans import rans_bits, entropy_bits
import exp4_real_model_ptq as e4
import exp4b_group_scales as e4b

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
GROUP = 64
N_CAL = 64
SCALE_BITS = 32.0 / GROUP


def get_cal_blocks(tok, dev):
    """Calibration = train split (no eval leakage), N_CAL blocks of 1024."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    eos = tok.eos_token_id
    ids = []
    for row in ds:
        ids.extend(tok(row["text"]).input_ids)
        ids.append(eos)
        if len(ids) >= N_CAL * e4.BLOCK:
            break
    arr = np.asarray(ids[: N_CAL * e4.BLOCK], dtype=np.int64).reshape(N_CAL, e4.BLOCK)
    print(f"calibration: {N_CAL} train blocks ({arr.size:,} tokens)")
    return torch.from_numpy(arr).to(dev)


@torch.no_grad()
def collect_hessian(model, module, cal_batches):
    """H = 2·XᵀX averaged over calibration batches (scaled)."""
    acc = {}

    def hook(_mod, inputs):
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        xt_x = x.T @ x
        if "H" not in acc:
            acc["H"] = xt_x
        else:
            acc["H"] += xt_x

    handle = module.register_forward_pre_hook(hook)
    for xb in cal_batches:
        model(xb)
    handle.remove()
    return 2.0 * acc["H"] / (len(cal_batches) * cal_batches[0].shape[0])


def gptq_hinv(H, device):
    """Dampered upper-Cholesky inverse Hessian (GPTQ).

    cusolver on this machine (Blackwell GB206, torch 2.14) errors inside
    cusolverDnXpotrs regardless of damping — backend issue, not conditioning —
    so the chain runs on CPU in float64 (H ≤ 3072², milliseconds)."""
    Hc = H.detach().to("cpu", torch.float64)
    damp = 0.01 * float(torch.diagonal(Hc).mean()) + 1e-8
    eye = torch.eye(Hc.shape[0], dtype=torch.float64)
    last = None
    for mult in (1.0, 10.0, 100.0, 1000.0):
        try:
            Hd = Hc + mult * damp * eye
            L = torch.linalg.cholesky(Hd)
            Hi = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hi, upper=True)
            return Hinv.to(device=device, dtype=torch.float32)
        except Exception as e:
            last = e
    raise RuntimeError(f"Hessian not invertible: {last}")


def gptq_quantize(W, H, kind, k, g=GROUP, compensate=True):
    """Column-wise pass over W (r, c). Static per-(row, group) scales from the
    ORIGINAL W (exp4b semantics). compensate=False reproduces exp4b RTN exactly
    (same grid, same rounding) — used as the internal baseline/cross-check."""
    r, c = W.shape
    Wg = W.reshape(r, c // g, g)
    if kind == "ternary":
        m_g = Wg.abs().mean(dim=2).unsqueeze(-1)
    else:
        m_g = Wg.abs().amax(dim=2).unsqueeze(-1)
    step_g = 2.0 * m_g / (k - 1)

    Hinv = gptq_hinv(H, W.device) if compensate else None
    W1 = W.clone()
    IDX = torch.zeros_like(W, dtype=torch.long)
    Q = torch.empty_like(W)
    for i in range(c):
        grp = i // g
        m_r = m_g[:, grp, 0]
        st_r = step_g[:, grp, 0]
        w = W1[:, i]
        if kind == "ternary":
            t = torch.where(w > 0.5 * m_r, torch.ones_like(w),
                            torch.where(w < -0.5 * m_r, -torch.ones_like(w),
                                        torch.zeros_like(w)))
            deq = t * m_r
            idx = (t + 1).long()
        else:
            idx = torch.clamp(torch.round((w + m_r) / st_r), 0, k - 1).long()
            deq = -m_r + idx * st_r
        IDX[:, i] = idx
        Q[:, i] = deq
        if compensate and i + 1 < c:
            err = (w - deq) / Hinv[i, i]
            W1[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    return Q, IDX


SCHEMES = [
    ("int4-15 g64", 15, "grid"),
    ("8-level g64", 8, "grid"),
    ("9-level ninths g64", 9, "grid"),
    ("ternary absmean g64", 3, "ternary"),
]


def rtn_baseline_lookup(base, name):
    key = name.strip()
    for bkey, v in base.items():
        if bkey.startswith(key) or key.startswith(bkey):
            return v
    return None


def main():
    dev = e4.device_setup()
    model, tok = e4.load_model(dev)
    layers = e4.quantized_layers(model)
    n_lin = sum(m.weight.numel() for _, m in layers)
    saved = {name: m.weight.data.detach().clone() for name, m in layers}

    base = {}
    with open(RESULTS / "exp4b_group_scales.csv") as fh:
        for row in csv.DictReader(fh):
            base[row["scheme"]] = float(row["ppl"])

    cal_all = get_cal_blocks(tok, dev)
    cal_batches = [cal_all[i * 8:(i + 1) * 8] for i in range(N_CAL // 8)]
    blocks = e4.get_blocks(tok, dev)
    ppl0 = e4.perplexity(model, blocks)
    print(f"fp32 baseline (sanity): {ppl0:.3f}")

    def run_scheme(kind, k, compensate):
        streams = []
        serr = wsum2 = wsum = 0.0
        for name, m in layers:
            tr = type(m).__name__ == "Conv1D"
            W = m.weight.data.t() if tr else m.weight.data
            if compensate:
                H = collect_hessian(model, m, cal_batches)
                Q, IDX = gptq_quantize(W, H, kind, k, compensate=True)
            else:
                Q, IDX = gptq_quantize(W, None, kind, k, compensate=False)
            serr += ((W - Q) ** 2).sum().item()
            wsum2 += (W * W).sum().item()
            wsum += W.sum().item()
            m.weight.data = (Q.t().contiguous() if tr else Q.contiguous())
            streams.append(IDX.detach().cpu().numpy().reshape(-1))
        sym = np.concatenate(streams)
        kk = int(sym.max()) + 1
        ent = entropy_bits(sym, kk)
        rb = rans_bits(sym, kk)
        ppl = e4.perplexity(model, blocks)
        rel_mse = serr / max(wsum2 - len(sym) * (wsum / len(sym)) ** 2, 1e-30)
        for name, m in layers:
            m.weight.data.copy_(saved[name])
        return dict(levels=kk, entropy=ent, rans=rb, eff=rb + SCALE_BITS,
                    ppl=ppl, mse=rel_mse)

    rows = []
    t_start = time.time()
    for scheme, k, kind in SCHEMES:
        t1 = time.time()
        rtn = run_scheme(kind, k, compensate=False)
        gq = run_scheme(kind, k, compensate=True)
        exp4b_ppl = rtn_baseline_lookup(base, scheme)
        cross = "matches exp4b" if (exp4b_ppl is not None and
                                    abs(rtn["ppl"] - exp4b_ppl) < 0.5) else \
                (f"exp4b {exp4b_ppl:.3f} MISMATCH?" if exp4b_ppl is not None
                 else "no exp4b row (new baseline)")
        rows.append(dict(scheme=scheme, levels=k, rtn_eff=rtn["eff"],
                         rtn_ppl=rtn["ppl"], gq_ppl=gq["ppl"],
                         gq_eff=gq["eff"], delta=gq["ppl"] - rtn["ppl"],
                         mse_gq=gq["mse"], cross=cross, note=f"{time.time()-t1:.0f}s"))
        print(f"{scheme:<22} RTN {rtn['ppl']:8.3f} @ {rtn['eff']:.2f} b/p | "
              f"GPTQ {gq['ppl']:8.3f} @ {gq['eff']:.2f} b/p | Δ {gq['ppl']-rtn['ppl']:+8.3f}"
              f" | cross-check: {cross}")

    with open(RESULTS / "exp6_gptq_grids.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scheme", "levels", "rtn_eff", "rtn_ppl",
                                           "gq_eff", "gq_ppl", "delta", "mse_gq",
                                           "cross", "note"])
        w.writeheader()
        w.writerows(rows)

    lines = [
        f"# exp6 — GPTQ-style Hessian compensation on the exp4b grids ({e4.MODEL_ID})",
        "",
        f"Calibration: {N_CAL} TRAIN-split blocks (1024 tok), sequential per-layer",
        f"Hessians over already-quantized predecessors. Grids/scales/side-info",
        f"identical to exp4b (g={GROUP}, static scales from original W; "
        f"+{SCALE_BITS:.2f} b/param). Only compensation differs. fp32 baseline"
        f" {ppl0:.3f}; noise floor ±0.3 ppl.",
        "",
        "| scheme | levels | RTN ppl | GPTQ ppl | Δ ppl | eff b/param | GPTQ eff | lin rel MSE (GPTQ) | RTN cross-check |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['scheme']} | {r['levels']} | {r['rtn_ppl']:.3f} |"
                     f" {r['gq_ppl']:.3f} | {r['delta']:+.2f} | {r['rtn_eff']:.3f} |"
                     f" {r['gq_eff']:.3f} | {r['mse_gq']:.5f} | {r['cross']} |")
    lines += ["", f"wall {time.time()-t_start:.0f}s", ""]
    (RESULTS / "exp6_gptq_grids.md").write_text("\n".join(lines))
    print("wrote results/exp6_gptq_grids.md")


if __name__ == "__main__":
    main()