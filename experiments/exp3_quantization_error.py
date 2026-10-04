#!/usr/bin/env python3
"""exp3 — quantization error vs alphabet size: where does the 9-level grid sit?

Uniform symmetric grids linspace(-m, m, k) for k in {3..9, 12, 15, 16, 27, 31, 32}
on N(0,1) weights (same weights as exp1, seed 42). Reports:
  raw_bits   log2(k) alphabet bits (fixed-width needs ceil — noted)
  rel MSE    mean((w - deq)^2) / var(w)
  entropy    empirical entropy of the k symbols

Plus the fair ceiling for k=9: Lloyd–Max (MSE-optimal scalar quantizer for the
weight distribution), seeded from the uniform ninths grid.

Note: linspace(-m, m, 9) IS the ninths grid {0,±m/8·2..} = k·m/4 — identical
to exp1's "nine" scheme, so the two experiments cross-check.

Outputs results/exp3_quantization_mse.{csv,md}
"""
import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

N = 200_000
SEED = 42
LOG = math.log2


def quantize_uniform(w, m, k):
    centers = np.linspace(-m, m, k)
    step = centers[1] - centers[0]
    idx = np.clip(np.round((w - centers[0]) / step), 0, k - 1).astype(np.int64)
    return idx, centers


def quantize_lloyd_max(w, k, iters=300):
    """1-D Lloyd–Max: nearest-center assignment + centroid update."""
    m = float(np.max(np.abs(w)))
    centers = np.linspace(-m, m, k)
    for _ in range(iters):
        boundaries = (centers[1:] + centers[:-1]) / 2.0
        idx = np.searchsorted(boundaries, w)
        new = centers.copy()
        for j in range(k):
            cell = w[idx == j]
            if len(cell):
                new[j] = cell.mean()
        if np.allclose(new, centers, atol=1e-9):
            centers = new
            break
        centers = new
    boundaries = (centers[1:] + centers[:-1]) / 2.0
    idx = np.searchsorted(boundaries, w)
    return idx, centers


def entropy_bits(idx, k):
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    rng = np.random.default_rng(SEED)
    w = rng.standard_normal(N)
    var = float(np.var(w))
    m = float(np.max(np.abs(w)))
    rows = []
    for k in (3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 27, 31, 32):
        idx, centers = quantize_uniform(w, m, k)
        deq = centers[idx]
        rows.append(dict(
            grid=f"uniform {k}", levels=k, raw_bits=LOG(k),
            mse=float(np.mean((w - deq) ** 2) / var),
            entropy=entropy_bits(idx, k),
            centers=",".join(f"{c:.3f}" for c in centers) if k == 9 else "",
            note="= ninths grid k·m/4" if k == 9 else ""))
    for k in (3, 4, 8, 9, 16):
        idx, centers = quantize_lloyd_max(w, k)
        rows.append(dict(
            grid=f"Lloyd–Max {k}", levels=k, raw_bits=LOG(k),
            mse=float(np.mean((w - centers[idx]) ** 2) / var),
            entropy=entropy_bits(idx, k),
            centers=",".join(f"{c:.3f}" for c in centers) if k == 9 else "",
            note="MSE-optimal k levels (fitted to the distribution)"))

    with open(RESULTS / "exp3_quantization_mse.csv", "w", newline="") as fh:
        cwr = csv.DictWriter(fh, fieldnames=["grid", "levels", "raw_bits", "mse",
                                             "entropy", "centers", "note"])
        cwr.writeheader()
        cwr.writerows(rows)

    lines = ["# exp3 — quantization error vs alphabet size", "",
             "Uniform grids on [-m, m], Gaussian weights (seed 42), rel MSE"
             " = mean sq err / variance. raw_bits = log2(levels) alphabet bits"
             " (a fixed-width code needs ceil — entropy coding reaches the"
             " fractional figure).", "",
             "| grid | levels | log2 k | rel MSE | entropy | note |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['grid']} | {r['levels']} | {r['raw_bits']:.3f} |"
                     f" {r['mse']:.6f} | {r['entropy']:.3f} | {r['note']} |")
    lines.append("")
    (RESULTS / "exp3_quantization_mse.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()