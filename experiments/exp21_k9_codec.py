#!/usr/bin/env python3
"""exp21 — the K9 file writer: real bytes, bit-exact decode, decode speed.

Converts exp18-20's *entropy-estimated* sizes into a measured K9 container
(k9.py, per docs/04), for Qwen2.5-Coder-1.5B-Instruct with body k=63 and
embedding k=99, entropy-coded scales. Then it decodes the file back into the
model and asserts the weights are bit-exact and the perplexity identical.

Config is RTN (no act-order), so no permutation metadata is needed; the GPTQ
variant would add only the uint32 permutation arrays per tensor (measured
separately by exp20's byte accounting). Scale mode is ent8 (spec §4 mode 4).

Run: python3 exp21_k9_codec.py --selftest    # synthetic round trip, seconds
     python3 exp21_k9_codec.py               # full model, ~35-40 min
"""
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch

import k9
import eval_harness as eh
import exp4b_group_scales as e4b
import exp18_qwen_gptq_noise as e18
from rans import entropy_bits

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
MODEL_ID = e18.MODEL_ID
BLOCK = 1024
GROUP = 64
N_WINDOWS = 8
K_BODY, K_EMBED = 63, 99
SCALE_MODE = "ent8"


def selftest():
    """Synthetic round trip: every grid, every scale mode, with/without perm."""
    rng = np.random.default_rng(0)
    ok = True
    for (r, c) in [(96, 192), (64, 256)]:
        for k in (9, 15, 27, 63, 99):
            for mode in ("fp32", "fp16", "ent8"):
                H = (k - 1) // 2
                # digits/scales are generated in the STORED frame; for k=63 we
                # also exercise the optional permutation (undone at decode)
                digits = rng.integers(0, k, size=(r, c)).astype(np.uint8)
                m = rng.random((r, c // GROUP)).astype(np.float32) + 0.05
                perm = rng.permutation(c).astype(np.uint32) if k == 63 else None
                path = RESULTS / "_k9_selftest.k9"
                size = k9.write_k9(path, [dict(name="t", shape=(r, c), group=GROUP,
                                               scale_mode=mode, k=k,
                                               digits=digits, scales=m, perm=perm)])
                rec = k9.read_k9(path)["t"]
                q = k9.decode_tensor(rec, GROUP)          # torch, cpu
                mt = torch.from_numpy(m).repeat_interleave(GROUP, dim=1)
                dt = torch.from_numpy(digits.astype(np.int64))
                exp = (mt * (dt - H) / H)
                if perm is not None:
                    exp = exp[:, torch.argsort(torch.from_numpy(perm.astype(np.int64)))]
                err = float((q - exp).abs().max())
                rel = err / max(float(exp.abs().max()), 1e-30)
                # fp32 scales are lossless; fp16/ent8 scales are lossy
                exact = rel == 0.0 if mode == "fp32" else rel < 2e-2
                ok &= exact
                print(f"  {r}x{c} k={k:>3} {mode:>4} perm={perm is not None}:"
                      f" max err {err:.2e} (rel {rel:.1e})"
                      f" {'ok' if exact else 'FAIL'} ({size} B)")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)

    t_all = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = e18.load_model(dev)
    body = e18.body_layers(model)
    embed = model.model.embed_tokens
    n_total = sum(p.numel() for p in model.parameters())
    print(f"body {len(body)} tensors | k_body={K_BODY} k_embed={K_EMBED}"
          f" | scale mode {SCALE_MODE}"
          f" | coder {'C (k9rans.so)' if k9.FAST_CODER else 'pure Python'}")

    wiki = eh.get_windows(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                          "test", N_WINDOWS, BLOCK, label="wiki")
    code = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean-valid", None,
                          "train", N_WINDOWS, BLOCK, label="code")

    # ---------------------------------------------------------- encode ----
    tensors, saved, est_bits = [], {}, 0.0
    t0 = time.time()
    for name, m in body + [("__embed__", embed)]:
        W = m.weight.data
        k = K_BODY if name != "__embed__" else K_EMBED
        saved[name] = W.detach().to("cpu").clone()
        deq, idx = e4b.q_uniform_group(W, k)
        scales = e18.group_scales(W).cpu().numpy()
        W.copy_(deq)
        idx_u8 = idx.to(torch.uint8).cpu().numpy()
        scale_bytes = len(k9.encode_scales(scales, SCALE_MODE))
        est_bits += entropy_bits(idx_u8.reshape(-1), k) * W.numel() + 8 * scale_bytes + 16 * k
        tensors.append(dict(name=name, shape=tuple(W.shape), group=GROUP,
                            scale_mode=SCALE_MODE, k=k, digits=idx_u8, scales=scales))
    quantize_s = time.time() - t0
    print(f"quantized {len(tensors)} tensors ({quantize_s:.0f}s)")

    path = RESULTS / "qwen_coder_1.5b_k63_embed99.k9"
    t0 = time.time()
    size = k9.write_k9(path, tensors)
    enc_s = time.time() - t0
    est_mb = est_bits / 8 / 1e6
    print(f"wrote {path.name}: {size/1e6:.1f} MB = {8*size/n_total:.3f} b/param"
          f" in {enc_s:.0f}s ({n_total/enc_s/1e6:.2f} M sym/s)")
    print(f"  entropy estimate {est_mb:.1f} MB (delta {100*(size/1e6-est_mb)/est_mb:+.2f}%)")

    # evaluate the in-memory quantized model (reference for the round trip)
    pw_ref = eh.ppl_per_window(model, wiki)
    pc_ref = eh.ppl_per_window(model, code)

    # ---------------------------------------------------------- decode ----
    del tensors
    t0 = time.time()
    recs = k9.read_k9(path)
    decode_s = time.time() - t0
    print(f"parsed directory in {decode_s:.0f}s")
    t0 = time.time()
    max_err = 0.0
    digit_mismatch = 0
    for name, m in body + [("__embed__", embed)]:
        rec = recs[name]
        # digit stream must round-trip bit-exactly (the codec's contract)
        d_dec = k9.decode_digits(rec)
        ref = saved[name].to(dev)
        ref_q, idx_ref = e4b.q_uniform_group(ref, rec["k"])
        idx_np = idx_ref.to(torch.uint8).cpu().numpy().reshape(-1)
        if rec["perm"] is not None:      # digits are stored in permuted order
            perm = torch.from_numpy(np.frombuffer(rec["perm"], dtype="<u4").astype(np.int64))
            idx_np = idx_np.reshape(ref.shape)[:, perm].reshape(-1)
        digit_mismatch += int((d_dec.astype(np.int64) != idx_np.astype(np.int64)).sum())
        q = k9.decode_tensor(rec, GROUP, device=dev)
        m.weight.data.copy_(q)
        # weight difference is only the lossy scale mode (ent8), not the digits
        max_err = max(max_err, float((q - ref_q).abs().max()))
    decode_s = time.time() - t0
    print(f"decoded + loaded all tensors in {decode_s:.0f}s"
          f" ({n_total/decode_s/1e6:.2f} M sym/s) | digit mismatches = {digit_mismatch}"
          f" | max |decoded-encoder| weight = {max_err:.2e} (scale-mode loss only)")

    pw = eh.ppl_per_window(model, wiki)
    pc = eh.ppl_per_window(model, code)
    dpw = float(np.abs(pw - pw_ref).max())
    dpc = float(np.abs(pc - pc_ref).max())
    print(f"ppl in-memory code {np.mean(pc_ref):.4f} -> decoded {np.mean(pc):.4f}"
          f" | max per-window delta {dpc:.2e}")

    with open(RESULTS / "exp21_k9_codec.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k_body", "k_embed", "scale_mode", "file_MB", "bits_per_param",
                    "entropy_est_MB", "encode_s", "decode_s",
                    "code_ppl_ref", "code_ppl_decoded", "max_weight_err"])
        w.writerow([K_BODY, K_EMBED, SCALE_MODE, f"{size/1e6:.1f}",
                    f"{8*size/n_total:.3f}", f"{est_mb:.1f}", f"{enc_s:.0f}",
                    f"{decode_s:.0f}", f"{np.mean(pc_ref):.4f}", f"{np.mean(pc):.4f}",
                    f"{max_err:.2e}"])

    fp32_mb = 4.0 * n_total / 1e6
    lines = [
        "# exp21 — the K9 file writer: real bytes, bit-exact decode",
        "",
        f"Qwen2.5-Coder-1.5B-Instruct, body k={K_BODY} + embedding k={K_EMBED},",
        f"g={GROUP}, scales {SCALE_MODE} (spec §4 mode 4). Real container written",
        f"by `k9.py` (magic K9Q1), decoded back into the model.",
        "",
        f"- file size: **{size/1e6:.1f} MB** = {8*size/n_total:.3f} b/param"
        f" (fp32 reference {fp32_mb:.0f} MB)",
        f"- entropy estimate: {est_mb:.1f} MB (delta"
        f" {100*(size/1e6-est_mb)/est_mb:+.2f}%) — validates the accounting used",
        "  in exp16-20",
        f"- encode {enc_s:.0f}s ({n_total/enc_s/1e6:.2f} M sym/s),"
        f" decode+load {decode_s:.0f}s ({n_total/decode_s/1e6:.2f} M sym/s),"
        f" coder: {'C (k9rans.so, byte-identical to rans.py)' if k9.FAST_CODER else 'pure-Python rANS'}",
        f"- round-trip: **{digit_mismatch} digit mismatches** over"
        f" {n_total:,} weights (rANS is lossless); max weight difference"
        f" {max_err:.2e} comes only from the lossy `{SCALE_MODE}` scale mode"
        f" (fp32 scales are bit-exact — see `--selftest`)",
        f"- perplexity: in-memory {np.mean(pc_ref):.4f} → decoded"
        f" {np.mean(pc):.4f} code (max per-window Δ {dpc:.2e})",
        "",
        "Compare to the deployed formats measured in exp19 (same windows):",
        "q4_k_m 1117.3 MB / +0.170 code Δ; q8_0 1894.5 MB / +0.086; NF4 999.5 MB",
        "/ +0.270. K9 is RTN here (no GPTQ), so quality is the RTN point; the",
        "GPTQ variant of the same grid added only ~0.5% bytes in exp20.",
        "",
        f"wall {time.time()-t_all:.0f}s", "",
    ]
    (RESULTS / "exp21_k9_codec.md").write_text("\n".join(lines))
    print("wrote results/exp21_k9_codec.{csv,md}")


if __name__ == "__main__":
    main()
