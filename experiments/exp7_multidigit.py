#!/usr/bin/env python3
"""exp7 — MULTI-DIGIT repeating grids: the b^L−1 family on TinyStories-33M.

Law (verified below): an odd symmetric grid of k = 2H+1 levels is
    w = (b − H)·m / H,   b ∈ {0..k−1} (per (row, group-of-64) scale m)
and whenever k = bᴸ − 1, the block b IS an L-digit repeating decimal in
base b:  b/k = 0.bbbb... (L-digit period), giving the dequant identity
    w = m·((k/H)·0.b̄ − 1)          — affine in one repeating decimal
(L=1: w = m·(9/4·0.s̄ − 1) = m·(s−4)/4 — exp5's printed equation).

Families tested (all per-group g=64, entropy-coded with M=4096 freq tables):
  decimal:  9 (L=1 — the original ninths/2-trit grid) · 99 (L=2) · 999 (L=3)
  binary:   7 (L=3) · 15 (L=4 — exp4's champion int4!) · 63 · 127 · 255 · 1023
  (base-3 multi-digit, 3²−1 = 8 levels, is an EVEN grid — no zero level —
   and is skipped; documented in docs/01.)

Pre-registered predictions (before running):
  P6  digit entropy < log2(levels) on every grid
  P7  ppl strictly decreasing in level count across the RTN grids
  P8  GPTQ gain shrinks with grid fineness (|Δ(99)| < |Δ(9)|), stays < 0
  P9  multi-digit (L=2) print "0.(43)" → parse round trip is bit-exact,
      and w = m·((99/49)·0.b̄ − 1) holds to float precision
  P10 the 99-level (eff-bits, ppl) point lies below the linear interpolation
      of the 63- and 127-level points (decimal family is frontier-efficient)

Out: results/exp7_multidigit.{csv,md}, results/multidigit_print_sample.txt
"""
import csv
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch
from rans import rans_bits, entropy_bits
import exp4_real_model_ptq as e4
import exp4b_group_scales as e4b
import exp6_gptq_grids as e6

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
GROUP = 64
SCALE_BITS = 32.0 / GROUP
CODEC_K = 12  # M=4096 frequency table — resolves up to ~4096-symbol alphabets

RTN_GRIDS = [(7, "2^3-1"), (9, "10^1-1"), (15, "2^4-1"), (63, "2^6-1"),
             (99, "10^2-1"), (127, "2^7-1"), (255, "2^8-1"),
             (999, "10^3-1"), (1023, "2^10-1")]
GPTQ_GRIDS = [9, 15, 99]


def base_power_family(k):
    for b in (2, 10):
        L = 1
        while b ** L - 1 < k:
            L += 1
        if b ** L - 1 == k:
            return f"{b}^{L}-1 (base-{b} L={L})"
    return "— (not b^L-1)"


def run_rtn(layers, k):
    streams = []
    serr = wsum2 = wsum = 0.0
    for name, m in layers:
        tr = type(m).__name__ == "Conv1D"
        Wq = m.weight.data.t() if tr else m.weight.data
        deq, idx = e4b.q_uniform_group(Wq, k)
        serr += ((Wq - deq) ** 2).sum().item()
        wsum2 += (Wq * Wq).sum().item()
        wsum += Wq.sum().item()
        m.weight.data = (deq.t() if tr else deq).contiguous()
        streams.append(idx.detach().cpu().numpy().reshape(-1))
    sym = np.concatenate(streams)
    return dict(sym=sym, entropy=entropy_bits(sym, k), rans=rans_bits(sym, k, CODEC_K),
                mse=serr / max(wsum2 - len(sym) * (wsum / len(sym)) ** 2, 1e-30))


def run_gptq(model, layers, k, cal_batches):
    streams = []
    serr = wsum2 = wsum = 0.0
    for name, m in layers:
        tr = type(m).__name__ == "Conv1D"
        W = m.weight.data.t() if tr else m.weight.data
        H = e6.collect_hessian(model, m, cal_batches)
        Q, IDX = e6.gptq_quantize(W, H, "grid", k, g=GROUP, compensate=True)
        serr += ((W - Q) ** 2).sum().item()
        wsum2 += (W * W).sum().item()
        wsum += W.sum().item()
        m.weight.data = (Q.t().contiguous() if tr else Q.contiguous())
        streams.append(IDX.detach().cpu().numpy().reshape(-1))
    sym = np.concatenate(streams)
    return dict(sym=sym, rans=rans_bits(sym, k, CODEC_K),
                mse=serr / max(wsum2 - len(sym) * (wsum / len(sym)) ** 2, 1e-30))


def print_roundtrip_test(layers, k=99):
    """P9: print one row-group as L-digit repeating decimals, re-parse, verify."""
    L = 2
    name, m = layers[0]
    W = m.weight.data.detach().cpu().numpy().astype(np.float32)
    r = W.shape[0]
    mgs = np.abs(W.reshape(r, -1, GROUP)).max(axis=2).astype(np.float32)  # (r, groups)
    H = (k - 1) // 2
    step = (2.0 * mgs / (k - 1)).astype(np.float32)
    # per-element quantization with each column's own group scale
    Wg = W.reshape(r, -1, GROUP)
    b3 = np.clip(np.round((Wg + mgs[:, :, None]) / step[:, :, None]), 0, k - 1).astype(np.int64)
    row0 = b3[0, 0]
    line = f"m={repr(float(mgs[0, 0]))} | " + "".join(f"0.({v:0{L}d})" for v in row0)
    (RESULTS / "multidigit_print_sample.txt").write_text(
        "# multi-digit print sample (99-level grid = decimal 2-digit repeating)\n"
        "# weight w = (b - 49)·m/49 = m·((99/49)·0.(bb) - 1); b printed as 0.(bb)\n"
        + line + "\n")
    back = np.array([int(t) for t in re.findall(r"0\.\((\d{2})\)", line)], dtype=np.int64)
    assert np.array_equal(back, row0), "digit roundtrip mismatch"
    mg = np.float32(float(line.split(" | ")[0][2:]))
    assert mg == mgs[0, 0], "scale roundtrip mismatch"
    step_p = (2.0 * mg / (k - 1)).astype(np.float32)
    deq_p = (-mg + back * step_p).astype(np.float32)
    deq_o = (-mgs[0, 0] + row0 * step[0, 0]).astype(np.float32)
    assert np.array_equal(deq_p, deq_o), "dequant roundtrip mismatch"
    ident = mg * ((k / H) * (back / float(2 * H + 1)) - 1.0)
    assert np.allclose(deq_o, ident, atol=1e-6), "repeating-decimal identity w = m·((k/H)·0.b̄ − 1) failed"
    return len(line)


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

    def run_and_eval(fn, k):
        out = fn(layers, k)
        ppl = e4.perplexity(model, blocks)
        for name, m in layers:
            m.weight.data.copy_(saved[name])
        out["ppl"] = ppl
        return out

    rows = []
    t_start = time.time()
    for k, fam in RTN_GRIDS:
        t1 = time.time()
        o = run_and_eval(lambda ls, kk: run_rtn(ls, kk), k)
        rows.append(dict(grid=f"{k}-level", family=base_power_family(k),
                         levels=k, entropy=o["entropy"], rans=o["rans"],
                         eff=o["rans"] + SCALE_BITS, ppl=o["ppl"], mse=o["mse"],
                         note=f"{time.time()-t1:.0f}s"))
        print(f"{k:>4}-level RTN ({fam:<9}) ppl {o['ppl']:8.3f} | "
              f"eff {o['rans']+SCALE_BITS:.3f} b/p | mse {o['mse']:.5f}")
    gq = {}
    for k in GPTQ_GRIDS:
        t1 = time.time()
        o = run_and_eval(lambda ls, kk: run_gptq(model, ls, kk, cal_batches), k)
        gq[k] = o
        print(f"{k:>4}-level GPTQ            ppl {o['ppl']:8.3f} | "
              f"eff {o['rans']+SCALE_BITS:.3f} b/p | mse {o['mse']:.5f}")

    nbytes = print_roundtrip_test(layers)
    print(f"P9 multi-digit print round trip: bit-exact ({nbytes} chars for 64 weights)")

    rows_out = []
    for r in rows:
        rows_out.append(r)
    with open(RESULTS / "exp7_multidigit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["grid", "family", "levels", "entropy",
                                           "rans", "eff", "ppl", "mse", "note"])
        w.writeheader()
        w.writerows(rows_out)
        for k, o in gq.items():
            w.writerow(dict(grid=f"{k}-level GPTQ", family="", levels=k,
                            entropy="", rans=o["rans"], eff=o["rans"] + SCALE_BITS,
                            ppl=o["ppl"], mse=o["mse"], note=""))

    by_k = {int(r["levels"]): r for r in rows}
    verdicts = {
        "P6 (entropy < log2 levels)": all(r["entropy"] < math.log2(int(r["levels"])) for r in rows),
        "P7 (ppl strictly decreasing in levels)": all(
            by_k[a]["ppl"] > by_k[b]["ppl"]
            for a, b in zip([7, 9, 15, 63, 99, 127, 255, 999, 1023],
                            [9, 15, 63, 99, 127, 255, 999, 1023])),
        "P8 (|Δ99| < |Δ9|, Δ99 < 0)": (abs(gq[99]["ppl"] - by_k[99]["ppl"]) <
                                       abs(gq[9]["ppl"] - by_k[9]["ppl"]) and
                                       gq[99]["ppl"] < by_k[99]["ppl"]),
        "P9 (multi-digit print round trip bit-exact + identity)": True,  # assertion above
        "P10 (99-g64 below 63↔127 interpolation)": None,
    }
    b63, b99, b127 = by_k[63], by_k[99], by_k[127]
    interp = b63["ppl"] + (b127["ppl"] - b63["ppl"]) * \
        (b99["eff"] - b63["eff"]) / (b127["eff"] - b63["eff"])
    verdicts["P10 (99-g64 below 63↔127 interpolation)"] = bool(b99["ppl"] < interp)
    print(f"P10: 99-level ppl {b99['ppl']:.2f} vs interpolated {interp:.2f} -> {verdicts['P10 (99-g64 below 63↔127 interpolation)']}")

    lines = [
        f"# exp7 — multi-digit repeating grids: the b^L−1 family ({e4.MODEL_ID})",
        "",
        f"Odd symmetric grids, per-(row, group-of-64) scales (+{SCALE_BITS:.2f} b/p side),",
        f"entropy-coded with M=4096 freq tables; eval {e4.EVAL_BLOCKS}x{e4.BLOCK}",
        f"TinyStories-val tokens; fp32 baseline {ppl0:.3f}, noise ±0.3.",
        "Law: 2H+1 = b^L−1 levels ⇔ weights are L-digit repeating decimals",
        "(w = m·(2·0.b̄ − 1)); exp4's champion int4-15 IS the binary L=4 grid.",
        "",
        "| grid | family | digit entropy | rANS b/p | eff b/p | ppl | rel MSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['grid']} | {r['family']} | {r['entropy']:.3f} |"
                     f" {r['rans']:.3f} | {r['eff']:.3f} | {r['ppl']:.3f} |"
                     f" {r['mse']:.5f} |")
    lines += ["", "## GPTQ-compensated grids", "",
              "| grid | RTN ppl | GPTQ ppl | Δ ppl | GPTQ eff b/p |", "|---|---|---|---|---|"]
    for k in GPTQ_GRIDS:
        lines.append(f"| {k}-level | {by_k[k]['ppl']:.3f} | {gq[k]['ppl']:.3f} |"
                     f" {gq[k]['ppl']-by_k[k]['ppl']:+.2f} | {gq[k]['rans']+SCALE_BITS:.3f} |")
    lines += ["", "## Pre-registered verdicts", ""]
    lines += [f"- {k}: **{'PASS' if v else 'FAIL'}**" for k, v in verdicts.items()]
    lines += ["", f"Multi-digit print sample: results/multidigit_print_sample.txt",
              f"(64 weights = {nbytes} chars; extrapolated L=2 text ~"
              f"{nbytes / 64 * n_lin / 1e6:.0f} MB for all linear weights — gzip/rANS", "reclaim it, same as exp5.)", "",
              f"wall {time.time()-t_start:.0f}s", ""]
    (RESULTS / "exp7_multidigit.md").write_text("\n".join(lines))
    print("wrote results/exp7_multidigit.md")


if __name__ == "__main__":
    main()