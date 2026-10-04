#!/usr/bin/env python3
"""exp9b — cure test for the 127-level anomaly (follow-up to exp9).

exp9 established the inversion is real (reproduced in 3 disjoint windows) and
global (both attn and mlp scopes). This test asks a targeted mechanistic
question: does a FITTED (Lloyd–Max) codebook cure what the UNIFORM 127-level
grid breaks? If fitted-127 ≤ fitted-99 + noise, the anomaly is an artifact of
uniform-grid endpoint/step structure; if not, it is something deeper.

Pre-registered: P16 — fitted 127-level ppl ≤ fitted 99-level ppl + 0.3 on the
same window.

Out: results/exp9b_fitted_127.{csv,md}
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import exp4_real_model_ptq as e4
import exp8_gptq_lloyd as e8
import exp9_anomaly_check as e9

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def main():
    dev = e4.device_setup()
    model, tok = e4.load_model(dev)
    layers = e4.quantized_layers(model)
    saved = {name: m.weight.data.detach().clone() for name, m in layers}
    blocks = e9.get_many_blocks(tok, dev, e4.EVAL_BLOCKS)  # window A, identical eval
    ppl0 = e4.perplexity(model, blocks)
    print(f"fp32 baseline window A: {ppl0:.3f}")

    rows = []
    t0 = time.time()
    for k in (99, 127):
        serr = wsum2 = wsum = 0.0
        for name, m in layers:
            tr = type(m).__name__ == "Conv1D"
            W = m.weight.data.t() if tr else m.weight.data
            _, deq, _ = e8.lloyd_centers_group(W, k=k)
            serr += ((W - deq) ** 2).sum().item()
            wsum2 += (W * W).sum().item()
            wsum += W.sum().item()
            m.weight.data = (deq.t() if tr else deq).contiguous()
        n = sum(W.numel() for _, m in layers)
        mse = serr / max(wsum2 - n * (wsum / n) ** 2, 1e-30)
        ppl = e4.perplexity(model, blocks)
        rows.append(dict(grid=k, ppl=ppl, mse=mse, note=f"{time.time()-t0:.0f}s"))
        print(f"fitted {k}-level (Lloyd, RTN): ppl {ppl:.3f} | mse {mse:.5f}")
        for name, m in layers:
            m.weight.data.copy_(saved[name])

    p99, p127 = rows[0]["ppl"], rows[1]["ppl"]
    p16 = p127 <= p99 + 0.3
    uniform_ref = {99: 40.565, 127: 40.978}  # exp7/exp9 window A
    with open(RESULTS / "exp9b_fitted_127.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["grid", "ppl", "mse", "note"])
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# exp9b — cure test: fitted (Lloyd–Max) codebooks on the 127-level anomaly",
        "",
        f"fp32 baseline window A {ppl0:.3f}; uniform-RTN references "
        f"(exp7/exp9): 99 → 40.565, 127 → 40.978.",
        "",
        "| grid | fitted ppl | fitted rel MSE |", "|---|---|---|",
    ] + [f"| {r['grid']} | {r['ppl']:.3f} | {r['mse']:.5f} |" for r in rows]
    lines += [
        "",
        f"P16 (fitted-127 ≤ fitted-99 + 0.3): **{'PASS' if p16 else 'FAIL'}**",
        "- If PASS: the anomaly is a uniform-grid structural artifact that fitted",
        "  codebooks cure (outlier/endpoint placement of the ±m-matched step).",
        "- If FAIL: the 127 loss is not endpoint-driven; flag stays open.",
        "",
        f"wall {time.time()-t0:.0f}s", "",
    ]
    (RESULTS / "exp9b_fitted_127.md").write_text("\n".join(lines))
    print("wrote results/exp9b_fitted_127.md")


if __name__ == "__main__":
    main()