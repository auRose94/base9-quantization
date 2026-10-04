"""Shared rANS entropy coder (Duda) + helpers for all experiments.

rANS, byte renorm, 32-bit state domain [2^24, 2^32). Frequency-table size
M = 1 << scale_bits; every stream encoded with this module is roundtrip-
asserted by `rans_bits`. Frequency tables are shared metadata and excluded
from bit counts (standard practice; at most ~M bytes).

Renorm invariant (general M): before each encode step, push while
x >= f·2^32/M; the step then restores x in [2^24, 2^32) for ANY M ≤ 2^8·2^24·?
— for M=256 this reduces to the original `f << 24` condition, byte-identical
to exp1–6 behavior. Larger M (e.g. 4096) just resolves wide alphabets better.

exp1 predates this file and still carries its own byte-identical twin of the
coder; this module is the canonical copy for new experiments (exp4, exp5...).
"""
import numpy as np


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
    """Duda rANS, byte renorm. State invariant x in [2^24, 2^32).

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
        while x >= (fs << 32) // (1 << k):  # = f·2^32/M; f·2^24 at M=256
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // fs) << k) + (x % fs) + cs
    assert (1 << 24) <= x < (1 << 32)
    return bytes(out) + x.to_bytes(4, "big")


def rans_decode(stream, f, n, k=8):
    cumul, inv = _tables(f, k)
    mask = (1 << k) - 1
    x = int.from_bytes(stream[-4:], "big")
    pos = len(stream) - 5
    f_ = f.tolist()
    c_ = cumul.tolist()
    inv_ = inv.tolist()
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = inv_[x & mask]
        x = f_[s] * (x >> k) + (x & mask) - c_[s]
        while x < (1 << 24):
            x = (x << 8) | stream[pos]
            pos -= 1
        out[i] = s
    return out


def rans_bits(symbols, n_syms, k=8):
    """Compress a symbol stream, assert exact roundtrip, return bits/symbol."""
    f = normalize_freqs(symbols, n_syms, 1 << k)
    enc = rans_encode(symbols, f, k)
    back = rans_decode(enc, f, len(symbols), k)
    assert np.array_equal(back, symbols), "rANS roundtrip failed"
    return 8.0 * (len(enc) + 4) / len(symbols)  # +4 shared end-state bytes


def entropy_bits(symbols, n_syms):
    counts = np.bincount(symbols, minlength=n_syms).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())