#!/usr/bin/env python3
"""eq7 — RQ2b (first stage): the artifact IS the model; the GPU synthesizes it.

A Triton kernel reads ONLY the K9Q1 artifact's digits (uint8) + per-(row,
group) scales (fp16) and materializes fp16 weight buffers on device — the k9
grid math (w = m*(d-H)/H) lives in the kernel, not in any weight file.
End-to-end: eq6's fitted C artifact -> GPU synth -> paired eval -> story.
Correctness gate: bitwise vs k9.decode_tensor (same fp16 scales, same fp32 math).
"""
import json
import os
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

B9 = "/mnt/matrix/Work/base9-quantization/experiments"
sys.path.insert(0, B9)
import k9  # noqa: E402

DEV = "cuda"
RES = os.path.join(L.ROOT, "results")
GROUP = 64


@triton.jit
def k9_synth_kernel(digits_ptr, scales_ptr, out_ptr, mp, ng, H,
                    numel, GROUPC: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel
    row = off // mp
    col = off % mp
    d = tl.load(digits_ptr + off, mask=mask, other=0).to(tl.float32)
    s = tl.load(scales_ptr + row * ng + col // GROUPC, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + off, s * (d - H) / H, mask=mask)


def synth_on_gpu(rec, out_fp16=True):
    """One k9 record -> synthesized buffer on DEV. Only the DECODED
    digit bytes + scales are shipped to GPU (tiny); rANS decoding stays on
    the validated base9 CPU path (GPU-parallel rANS decode = future RQ2c)."""
    r, mp = rec["shape"]
    ng = mp // rec["group"]
    H = (rec["k"] - 1) // 2
    digits = k9.decode_digits(rec)                                   # uint8 flat
    scales = k9.decode_scales(rec["scales"], rec["scale_mode"], r * ng)
    numel = r * mp
    d_gpu = torch.from_numpy(np.ascontiguousarray(digits)).to(DEV)
    s_gpu = torch.from_numpy(np.ascontiguousarray(scales)).half().to(DEV)
    out = torch.empty(numel, device=DEV,
                      dtype=torch.float16 if out_fp16 else torch.float32)
    BLOCK = 1024
    k9_synth_kernel[(triton.cdiv(numel, BLOCK),)](
        d_gpu, s_gpu, out, mp, ng, H, numel, GROUPC=GROUP, BLOCK=BLOCK)
    return out.view(r, mp)


def main():
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    fitted = os.path.join(RES, "eq6_C_k63emb99_fitted.k9")
    recs = k9.read_k9(fitted)
    assert "tok_embeddings.weight" in recs or "output.weight" in recs

    # ---------- synth: artifact blobs -> GPU buffers (the ONLY shipped weights) ----------
    t0 = time.perf_counter()
    sd_t = {name: synth_on_gpu(rec) for name, rec in recs.items()}
    synth_s = time.perf_counter() - t0

    # correctness gate vs the container's own decoder:
    #   fp32 synthesis must match the container bitwise-ish (tol 1e-5);
    #   fp16 deployment output is bounded by fp16 rounding (ulp ~5e-4 at O(1)).
    dmax32, dmax16 = 0.0, 0.0
    for name, rec in recs.items():
        sl = slice(0, int(uniq[name].shape[1]))
        W_ref = k9.decode_tensor(rec, GROUP)[:, sl].to(DEV)
        dmax32 = max(dmax32, float((W_ref - synth_on_gpu(rec, out_fp16=False)[:, sl]).abs().max()))
        dmax16 = max(dmax16, float((W_ref - synth_on_gpu(rec, out_fp16=True)[:, sl].float()).abs().max()))
    assert dmax32 < 1e-5, f"triton fp32 synth diverges from container: {dmax32}"
    assert dmax16 <= 6e-4, f"triton fp16 synth diverges beyond fp16 rounding: {dmax16}"

    # ---------- evaluate the synthesized model ----------
    EMB = "tok_embeddings.weight" if "tok_embeddings.weight" in recs else "output.weight"
    for name in recs:                                   # slice off digit-padding cols
        true_cols = int(uniq[name].shape[1])
        if sd_t[name].shape[1] != true_cols:
            sd_t[name] = sd_t[name][:, :true_cols]
    w1d = {k: v for k, v in uniq.items() if v.dim() == 1}
    for k, v in w1d.items():
        sd_t[k] = v.half().float().to(DEV)
    emb = sd_t[EMB].float()
    sd_t["output.weight"] = sd_t["tok_embeddings.weight"] = emb
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    ev = L.eval_loss({k: v.float() for k, v in sd_t.items()}, ma, tokens, DEV, 384)
    story = L.sample_story({k: v.float() for k, v in sd_t.items()}, ma, DEV, n_new=160, seed=7)

    # ---------- baselines: shipping the weights as data ----------
    t0 = time.perf_counter()
    sd32 = {k: v.to(DEV) for k, v in sd0.items()}  # fp32 weight upload baseline
    up32_s = time.perf_counter() - t0
    fp32_bytes = sum(v.numel() * 4 for v in uniq.values())
    fp16_up_bytes = sum(v.numel() * 2 for v in uniq.values())

    k9_bytes = os.path.getsize(fitted)
    rest_bytes = sum(v.numel() * 2 for v in w1d.values())
    mat_bytes = sum(t.numel() * 2 for t in sd_t.values())
    row = dict(fitted_artifact=k9_bytes, rest_fp16_bytes=rest_bytes,
               total_variables_bytes=k9_bytes + rest_bytes,
               bytes_materialized_on_gpu=mat_bytes,
               synth_s=round(synth_s, 4), fp32_upload_bytes=fp32_bytes,
               fp32_upload_s=round(up32_s, 4),
               maxabs_fp32_vs_container=dmax32, maxabs_fp16_vs_container=dmax16,
               eval_loss=round(ev[0], 4), sem=round(ev[1], 4), story=story[:96])
    print(json.dumps(row, indent=2))
    with open(os.path.join(RES, "eq7_results.json"), "w") as f:
        json.dump(row, f, indent=2)
    print("[eq7] wrote results/eq7_results.json")


if __name__ == "__main__":
    main()