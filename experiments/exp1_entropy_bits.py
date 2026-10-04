#!/usr/bin/env python3
"""exp1 — bits-per-parameter for standard vs ternary vs 9-level ninths schemes.

Weights: n = 200,000 standard-normal values (seed 42), scale-max `m`.

Schemes
  int8   symmetric, 255-level step m/127, 8-bit indices
  int4   symmetric, 15-level  step m/7,   4-bit indices
  tern   BitNet b1.58-style absmean ternary in {-1,0,+1}
  nine   9-level ninths grid {0,±1..4} with step m/4  (dequant k·m/4)

For tern and nine, three lossless-storage measurements beyond the raw index:
  entropy_bits   empirical symbol entropy (bits/param)
  pair_entropy   ternary only: joint entropy of ADJACENT PAIRS mapped to one
                 base-9 digit (3·a + b), reported per parameter
  rans_bits      ACTUAL bits/param from a roundtrip-asserted rANS coder
                 (frequency table shared as metadata, excluded from counts)

Outputs results/exp1_entropy_bits.csv and results/exp1_entropy_bits.md
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


# --------------------------------------------------------------- rANS ----
def normalize_freqs(symbols, n_syms, M=256):
    counts = np.bincount(symbols, minlength=n_syms).astype(np.int64)
    total = int(counts.sum())
    assert total > 0
    f = np.zeros(n_syms, dtype=np.int64)
    for s in range(n_syms):
        if counts[s]:
            f[s] = max(1, int(round(counts[s] * M / total)))
    diff = int(f.sum()) - M
    order = list(np.argsort(-counts))
    i = 0
    while diff != 0:
        assert i < 10_000
        s = order[i % len(order)]
        i += 1
        if counts[s] == 0:
            continue
        if diff > 0 and f[s] > 1:
            f[s] -= 1
            diff -= 1
        elif diff < 0:
            f[s] += 1
            diff += 1
    assert int(f.sum()) == M
    return f


def _tables(f, k):
    cumul = np.zeros(len(f) + 1, dtype=np.int64)
    cumul[1:] = np.cumsum(f)
    M = 1 << k
    inv = np.empty(M, dtype=np.int64)
    for s in range(len(f)):
        inv[cumul[s]:cumul[s] + f[s]] = s
    return cumul, inv


def rans_encode(symbols, f, k=8):
    """Duda rANS, 32-bit state, byte renorm. State invariant x in [2^24, 2^32).

    Renorm-before-write keeps x < f·2^24 entering each step; the encode map
    then restores x in [2^24, 2^32). Symbols processed in reverse, so the
    decoder recovers the original order. Final 4 bytes = end state (big-endian).
    """
    cumul, _ = _tables(f, k)
    x = 1 << 24
    out = bytearray()
    f_ = f.tolist()
    c_ = cumul.tolist()
    for s in reversed(symbols.tolist()):
        fs = f_[s]
        cs = c_[s]
        while x >= (fs << 24):
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // fs) << k) + (x % fs) + cs
    assert (1 << 24) <= x < (1 << 32)
    return bytes(out) + x.to_bytes(4, "big")


def rans_decode(stream, f, n, k=8):
    cumul, inv = _tables(f, k)
    x = int.from_bytes(stream[-4:], "big")
    pos = len(stream) - 5
    f_ = f.tolist()
    c_ = cumul.tolist()
    inv_ = inv.tolist()
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = inv_[x & 255]
        x = f_[s] * (x >> k) + (x & 255) - c_[s]
        while x < (1 << 24):
            x = (x << 8) | stream[pos]
            pos -= 1
        out[i] = s
    return out


def rans_bits(symbols, n_syms, k=8):
    f = normalize_freqs(symbols, n_syms, 1 << k)
    enc = rans_encode(symbols, f, k)
    back = rans_decode(enc, f, len(symbols), k)
    assert np.array_equal(back, symbols), "rANS roundtrip failed"
    return 8.0 * (len(enc) + 4) / len(symbols)  # +4 shared end-state bytes


# -------------------------------------------------------------- quant ----
def rel_mse(w, deq):
    return float(np.mean((w - deq) ** 2) / np.var(w))


def entropy_bits(symbols, n_syms):
    counts = np.bincount(symbols, minlength=n_syms).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    rng = np.random.default_rng(SEED)
    w = rng.standard_normal(N)
    m = float(np.max(np.abs(w)))
    rows = []

    # int8
    scale = m / 127
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int64)
    rows.append(dict(scheme="int8 symmetric", levels=255, raw_bits=8.0,
                     alphabet_bits=LOG(255), entropy=entropy_bits(q + 127, 255),
                     rans=rans_bits(q + 127, 255), mse=rel_mse(w, q * scale),
                     note="GGUF Q8-style baseline"))

    # int4 (15 levels, symmetric)
    scale = m / 7
    q = np.clip(np.round(w / scale), -7, 7).astype(np.int64)
    rows.append(dict(scheme="int4 symmetric (15 levels)", levels=15, raw_bits=4.0,
                     alphabet_bits=LOG(15), entropy=entropy_bits(q + 7, 15),
                     rans=rans_bits(q + 7, 15), mse=rel_mse(w, q * scale),
                     note="GGUF Q4-style baseline"))

    # ternary, BitNet b1.58-style absmean rule
    gamma = float(np.mean(np.abs(w)))
    q = np.where(w > 0.5 * gamma, 1, np.where(w < -0.5 * gamma, -1, 0)).astype(np.int64)
    s = q + 1  # symbols {0,1,2}
    s_pair = 3 * s[0::2] + s[1::2]  # one base-9 digit per pair of trits
    pair_H = entropy_bits(s_pair, 9) / 2.0        # per parameter (2 params/pair)
    pair_rans = rans_bits(s_pair, 9) / 2.0
    rows.append(dict(scheme="ternary absmean (BitNet b1.58-style)", levels=3,
                     raw_bits=2.0, alphabet_bits=LOG(3), entropy=entropy_bits(s, 3),
                     rans=rans_bits(s, 3), mse=rel_mse(w, q * gamma),
                     note=f"ternary pair stream (2 trits -> 1 base-9 digit):"
                          f" joint entropy {pair_H:.4f} b/param, rANS on pair"
                          f" stream {pair_rans:.4f} b/param"))

    # 9-level ninths grid, step m/4, digits {-4..4}; scale = 9·m/4 so step s/9 = m/4
    scale = 9.0 * m / 4.0
    q = np.clip(np.round(9 * w / scale), -4, 4).astype(np.int64)
    rows.append(dict(scheme="9-level ninths grid (step m/4)", levels=9,
                     raw_bits=4.0, alphabet_bits=LOG(9), entropy=entropy_bits(q + 4, 9),
                     rans=rans_bits(q + 4, 9), mse=rel_mse(w, q * scale / 9.0),
                     note="k/9·(9m/4); k/9 = 0.kkk... repeating in base 10"))

    # report
    cols = ["scheme", "levels", "raw_bits", "alphabet_bits", "entropy",
            "rans", "mse", "note"]
    with open(RESULTS / "exp1_entropy_bits.csv", "w", newline="") as fh:
        cwr = csv.DictWriter(fh, fieldnames=cols)
        cwr.writeheader()
        cwr.writerows(rows)

    lines = [
        "# exp1 — bits per parameter (n = 200000 Gaussian weights, seed 42)",
        "",
        "All bit figures are bits/parameter; `entropy` is the empirical symbol",
        "entropy, `rans` is actual coder output (roundtrip-asserted), `mse` is",
        "relative MSE (mean sq err / weight variance).",
        "",
        "| scheme | levels | raw index | alphabet | entropy | rANS | rel MSE |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rans = f"{r['rans']:.3f}" if r["rans"] is not None else "—"
        lines.append(
            f"| {r['scheme']} | {r['levels']} | {r['raw_bits']:.3f} |"
            f" {r['alphabet_bits']:.3f} | {r['entropy']:.3f} | {rans} |"
            f" {r['mse']:.6f} |")
    lines += ["", "## Ternary pair (base-9 digit) block",
              "", "```", *[r["note"] for r in rows if "pair stream" in r["note"]],
              "```", ""]
    (RESULTS / "exp1_entropy_bits.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()