#!/usr/bin/env python3
"""eq4 — 'the model lives on the GPU as a program.'

The GPU receives ONLY small variable tables; weight buffers are SYNTHESIZED
on device; the model then runs from those buffers. Neither weight tensors nor
basis matrices are shipped:

  path A  uniform-6   variables = 6-bit levels + per-tensor scales
          (the quality-preserving point from eq1: +4.2% paired loss, 5.3x smaller)
  path B  dct-topk    variables = (index, 8-bit level, scale) per kept DCT
          coefficient; the DCT-II basis is BUILT ON DEVICE from the cosine
          formula (the 'mathematical constants' of the README mapping)

Checks: synthesis matches the CPU reference reconstruction (P5: max|Δ| < 2e-4);
a story is sampled end-to-end from synthesized weights on both paths.
Reports: variable bytes vs fp32 weight bytes, synthesis wall-time, P5 verdict.
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

DEV = "cuda"
P5_TOL = 2e-4


def gpu_dct_matrix(N):
    """DCT-II basis built ON DEVICE from the formula (no shipped table)."""
    n = torch.arange(N, device=DEV, dtype=torch.float32)
    k = torch.arange(N, device=DEV, dtype=torch.float32)
    C = torch.cos(math.pi / N * (n[None, :] + 0.5) * k[:, None]) * math.sqrt(2.0 / N)
    C[0] /= math.sqrt(2.0)
    return C


def synth_from_coef_tables(tables, device):
    """tables: {name: (idx int64 gpu, levels int16 gpu, scale, (n, m))} → state dict."""
    out = {}
    for name, (idx, lev, scale, shape) in tables.items():
        n, m = shape
        if n == 0:
            continue
        Cn, Cm = gpu_dct_matrix(n), gpu_dct_matrix(m)
        Xhat = torch.zeros(n * m, device=device, dtype=torch.float32)
        Xhat[idx] = lev.float() * scale
        Wh = (Cn.T @ Xhat.view(n, m)) @ Cm
        out[name] = Wh
    return out


def main():
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    w2d = {k: v for k, v in uniq.items() if v.dim() == 2}
    w1d = {k: v for k, v in uniq.items() if v.dim() == 1}
    P = sum(v.numel() for v in uniq.values())
    W_BYTES = P * 4
    print(f"[eq4] unique params {P:,}; fp32 weight bytes {W_BYTES:,}")

    # ---------------- path A: uniform-6 dequant-from-variables ----------------
    q, s = {}, {}
    for k, W in w2d.items():
        q[k], s[k] = L.quantize_sym(W.double(), 6)
    a_vars_bytes = sum(q[k].numel() for k in q)  # int8 levels
    a_vars_bytes += 4 * (len(q) + len(w1d)) + sum(v.numel() * 2 for v in w1d.values())
    lev = {k: q[k].to(torch.int8).to(DEV) for k in q}  # 6-bit levels fit int8
    sc = {k: float(s[k]) for k in s}
    sdA = {k: (lev[k].float() * sc[k]).float() for k in q}
    for k, W in w1d.items():
        sdA[k] = W.half().float().to(DEV)
    # fp32 path also needs alias (uniq keeps only the first tied key):
    emb = sdA.get("output.weight", sdA.get("tok_embeddings.weight"))
    sdA["output.weight"] = sdA["tok_embeddings.weight"] = emb

    # ---------------- path B: DCT variables (topk 0.05, b=8) -----------------
    t0 = time.perf_counter()
    tables, cpu_recon = {}, {}
    for k, W in w2d.items():
        X = L.dct2(W.double())
        numel = X.numel()
        kk = max(1, round(0.05 * numel))
        idx = torch.topk(X.abs().flatten(), kk).indices
        vals = X.flatten()[idx]
        lev, sc_ = L.quantize_sym(vals, 8)
        tables[k] = (idx.to(DEV), lev.to(torch.int16).to(DEV), float(sc_), tuple(W.shape))
        Xh = torch.zeros_like(X.flatten())
        Xh[idx] = L.dequant(lev, sc_)
        cpu_recon[k] = L.idct2(Xh.reshape(W.shape)).float()
    b_vars_bytes = sum(lev.numel() * 2 + idx.numel() * 4 + 4
                       for idx, lev, sc_, (n, m) in tables.values())
    b_vars_bytes += sum(v.numel() * 2 for v in w1d.values())
    synth_t0 = time.perf_counter()
    sdB = synth_from_coef_tables(tables, DEV)
    emb = sdB.get("output.weight", sdB.get("tok_embeddings.weight"))
    sdB["output.weight"] = sdB["tok_embeddings.weight"] = emb
    for k, W in w1d.items():
        sdB[k] = W.half().float().to(DEV)
    synth_s = time.perf_counter() - synth_t0
    fit_s = time.perf_counter() - t0
    print(f"[eq4] variables: pathA {a_vars_bytes:,} B ({a_vars_bytes/W_BYTES:.2f}x fp32) | "
          f"pathB {b_vars_bytes:,} B ({b_vars_bytes/W_BYTES:.2f}x fp32)")

    # ---------------- P5: GPU synthesis == CPU reconstruction ----------------
    dmaxB, dmaxA = 0.0, 0.0
    for k in w2d:
        dmaxB = max(dmaxB, float((sdB[k] - cpu_recon[k].to(DEV)).abs().max()))
        dmaxA = max(dmaxA, float((sdA[k] - (w2d[k].to(DEV))).abs().max()))
    print(f"[eq4] synthesis vs CPU recon: pathB max|Δ| {dmaxB:.2e} | "
          f"pathA dequant exact-ish max|Δ| {dmaxA:.2e} | fit {fit_s:.2f}s synth {synth_s:.3f}s "
          f"({fit_s:.2f}s on CPU incl. fitting)")
    v5 = dict(prediction="P5", pathB_maxabs=dmaxB, tol=P5_TOL,
              verdict="PASS" if dmaxB < P5_TOL else "FAIL")
    print(f"[eq4] P5 ({v5['verdict']}): < 2e-4" if v5["verdict"] == "PASS"
          else f"[eq4] P5 ({v5['verdict']}): max|Δ| {dmaxB:.2e} >= 2e-4 — analyze")

    # ---------------- stories from synthesized models ----------------
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    evA = L.eval_loss(sdA, ma, tokens, DEV, 384)
    evB = L.eval_loss(sdB, ma, tokens, DEV, 384)
    print(f"[eq4] val loss from SYNTHESIZED buffers: pathA {evA[0]:.4f}±{evA[1]:.4f} "
          f"(eq1: 2.0548) | pathB {evB[0]:.4f}±{evB[1]:.4f} (eq1 dct-topk 0.05: 7.5189)")
    for tag, sdm in (("A(uniform6)", sdA), ("B(dct-topk-0.05)", sdB)):
        txt = L.sample_story(sdm, ma, DEV, n_new=160, seed=7)
        print(f"[eq4] story from GPU-synthesized weights [{tag}]: {txt[:110]!r}")

    import json
    with open(os.path.join(L.RESULTS, "eq4_verdicts.json"), "w") as f:
        json.dump([v5, dict(pathA_vars_bytes=a_vars_bytes, pathB_vars_bytes=b_vars_bytes,
                            fp32_weight_bytes=W_BYTES, pathA_val_loss=evA[0],
                            pathB_val_loss=evB[0], fit_s=fit_s, synth_s=synth_s)], f, indent=2)
    print("[eq4] wrote results/eq4_verdicts.json")


if __name__ == "__main__":
    main()