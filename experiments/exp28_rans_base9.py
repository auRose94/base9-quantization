#!/usr/bin/env python3
"""exp28 — RQ11: the base-9 codec. Decode AND encode on the grid (docs/09).

The deployed coder (rans.py/rans_fast.c) is rANS with byte (base-256)
renormalization over a stream *sourced* from base-9 digits — the state
arithmetic is binary. This experiment builds the grid-native variant:

  * frequency table M = 9^n (per-symbol values sum to an exact power of 9);
  * state invariant x in [L, 9L) with L = 9^m, renorm emits BASE-9 digits
    (emit x % 9, x /= 9) while x >= fs * 9^(m+1-n);
  * symbol slot = x % M = the low n base-9 digits of the state — symbol
    lookup IS digit inspection;
  * encode/decode are exact inverses (same one state-update op) — the
    "decode implies encode" property, now on the grid;
  * output stream = base-9 digits, packed 169 digits -> 67 bytes
    (9^169 < 2^536; waste 0.285 bits/block) -> whole artifact stays base-9
    end to end.

Gates (docs/09 P41-P43):
  P41 rate parity: base-9 coder rate <= byte coder rate + 0.01 b/digit on
      the same real digit streams (eq13 33M QAT tensors + eq9 260K QAT).
  P42 throughput: C core <= 2x slower than the byte coder (enc and dec).
  P43 round-trip: 0 mismatches over >= 1e9 symbols (C, synthetic iid9).
Plus: Python<->C output parity (the rans_fast house bar), adversarial
round-trips (single-symbol, f=1-rare, skewed streams), M in {9^4,9^5,9^6}
ablation, and the op census of the state machine (renorm ops are digit ops).

CPU-only.
"""
import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rans  # noqa: E402  (byte-renorm reference + normalize_freqs)
import rans_fast  # noqa: E402  (byte coder, C; HAVE_FAST gate)

B9_M4 = 9 ** 4
B9_L4 = 9 ** 12
B9_BLOCK, B9_BYTES = 169, 67          # 9^169 < 2^536 = 2^67.0 exactly? 2^536 = 2^(67*8) ✓
LOG2_9 = float(np.log2(9.0))


def b9_tables(f, M):
    cumul = np.zeros(len(f) + 1, dtype=np.int64)
    cumul[1:] = np.cumsum(f)
    inv = np.zeros(M, dtype=np.int64)
    for s in range(len(f)):
        inv[cumul[s]: cumul[s] + f[s]] = s
    return cumul, inv


def b9_encode(syms, f, m_exp=12, n_exp=4):
    """Python reference. Returns (emitted digits list, end state)."""
    M, L = 9 ** n_exp, 9 ** m_exp
    push = m_exp + 1 - n_exp
    cumul, _ = b9_tables(f, M)
    x = L
    emits = []
    c_, f_ = cumul.tolist(), f.tolist()
    nine = 9
    for s in reversed(syms.tolist()):
        fs = f_[s]; cs = c_[s]
        th = fs * 9 ** push
        while x >= th:
            emits.append(x % nine)
            x //= nine
        x = (x // fs) * M + (x % fs) + cs
    assert L <= x < 9 * L, f"state invariant broken: {x}"
    return emits, x


def b9_decode(emits, endx, f, n_out, m_exp=12, n_exp=4):
    M, L = 9 ** n_exp, 9 ** m_exp
    cumul, inv = b9_tables(f, M)
    x = endx
    i_ = len(emits) - 1
    out = np.empty(n_out, dtype=np.int64)
    for i in range(n_out):
        slot = x % M
        s = int(inv[slot])
        x = f[s] * (x // M) + slot - int(cumul[s])
        while x < L:
            x = x * 9 + emits[i_]
            i_ -= 1
        out[i] = s
    assert i_ == -1, "digit stack not fully consumed"
    return out


def b9_pack(emits):
    """169-digit blocks (67 bytes) + tightly packed tail (ceil bits -> bytes)."""
    import math
    out = bytearray()
    p9 = [9 ** r for r in range(B9_BLOCK)]
    n = len(emits)
    b = 0
    while n - b >= B9_BLOCK:
        I = 0
        for r, dgt in enumerate(emits[b:b + B9_BLOCK]):
            I += dgt * p9[r]
        out += I.to_bytes(B9_BYTES, "big")
        b += B9_BLOCK
    tail = emits[b:]
    if tail:
        I = 0
        for r, dgt in enumerate(tail):
            I += dgt * p9[r]
        out += I.to_bytes(math.ceil(len(tail) * LOG2_9 / 8), "big")
    return bytes(out)


def b9_unpack(buf, n_digits):
    import math
    p9 = [9 ** r for r in range(B9_BLOCK)]
    digits = []
    b = 0
    while len(digits) < n_digits:
        rem = n_digits - len(digits)
        if rem >= B9_BLOCK:
            I = int.from_bytes(buf[b:b + B9_BYTES], "big")
            b += B9_BYTES
            take = B9_BLOCK
        else:
            nb = math.ceil(rem * LOG2_9 / 8)
            I = int.from_bytes(buf[b:b + nb], "big")
            b += nb
            take = rem
        for r in range(take):
            digits.append((I // p9[r]) % 9)
    assert b == len(buf), "pack/unpack length mismatch"
    return digits


# ------------------------------------------------------------------ C bridge

def build_c():
    src = os.path.join(HERE, "exp28_rans_base9.c")
    so = os.path.join(HERE, "exp28_rans_base9.so")
    subprocess.run(["gcc", "-O3", "-shared", "-fPIC", "-o", so, src], check=True)
    lib = ctypes.CDLL(so)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.b9_rans_encode.argtypes = [u8p, ctypes.c_size_t, u64p, u64p,
                                   u8p, ctypes.c_size_t, u64p]
    lib.b9_rans_encode.restype = ctypes.c_size_t
    lib.b9_rans_decode.argtypes = [u8p, ctypes.c_size_t, ctypes.c_uint64,
                                   ctypes.c_size_t, u64p, u64p, u32p, u8p]
    lib.b9_rans_decode.restype = None
    return lib


def c_encode(lib, syms, f, cap):
    cumul, _ = b9_tables(f, B9_M4)
    f_ = np.ascontiguousarray(f, dtype=np.uint64)
    c_ = np.ascontiguousarray(cumul, dtype=np.uint64)
    out = np.zeros(cap, dtype=np.uint8)
    end = ctypes.c_uint64(0)
    n = lib.b9_rans_encode(syms.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                           len(syms), f_.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                           c_.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                           out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                           cap, ctypes.byref(end))
    if n == ctypes.c_size_t(-1).value or n > cap:
        raise RuntimeError("b9 encode digit cap too small")
    return out[:n], end.value


def c_decode(lib, digits, end, f, n):
    cumul, inv = b9_tables(f, B9_M4)
    f_ = np.ascontiguousarray(f, dtype=np.uint64)
    c_ = np.ascontiguousarray(cumul, dtype=np.uint64)
    inv_ = np.ascontiguousarray(inv, dtype=np.uint32)
    out = np.zeros(n, dtype=np.uint8)
    lib.b9_rans_decode(digits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                       len(digits), end, n,
                       f_.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                       c_.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                       inv_.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return out


# ------------------------------------------------------------------ streams

def real_streams():
    import k9  # noqa: E402
    streams = []
    work = os.path.dirname(os.path.dirname(HERE))
    a9 = os.path.join(work, "weights-as-equations", "results")
    q260 = os.path.join(a9, "eq9_QAT_k9emb99.k9")
    m33 = os.path.join(a9, "eq13_s4000_lr1e4.k9")
    for path in (q260, m33):
        if os.path.exists(path):
            for name, rec in k9.read_k9(path).items():
                d = k9.decode_digits(rec)
                if len(d) >= 512:
                    streams.append((f"{os.path.basename(path)}:{name}", d, rec["k"]))
    return streams


def main():
    lib = build_c() if shutil_which("gcc") else None
    have_c = lib is not None
    print(f"[exp28] C coder: {'compiled' if have_c else 'MISSING (gcc?)'}")

    # ------------------------------------------------ correctness + parity
    rng = np.random.default_rng(0)
    streams = real_streams()
    print(f"[exp28] real digit streams loaded: {len(streams)} tensors, "
          f"{sum(len(d) for _, d, _ in streams):,} digits")
    synth = [(
        "synth:iid9", rng.integers(0, 9, 2_000_000).astype(np.uint8), 9),
        ("synth:skewed100:1", np.where(rng.random(2_000_000) < 0.01,
                                       rng.integers(0, 8, 2_000_000), 8).astype(np.uint8), 9),
        ("synth:constant", np.full(1_000_000, 3, dtype=np.uint8), 9),
        ("synth:rare-f1", np.full(1_000_000, 4, dtype=np.uint8), 9)]
    all_streams = streams + synth

    # per-stream roundtrip (Python reference, M = 9^6) + C parity
    rt_fail = parity_fail = 0
    rate_rows = []
    for name, d, ks_n in all_streams:
        f9 = rans.normalize_freqs(d.astype(np.int64), ks_n, B9_M4)
        if len(d) <= 3_000_000:                     # python reference path
            emits, end = b9_encode(d, f9)
            back = b9_decode(emits, end, f9, len(d))
            rt_ok = bool(np.array_equal(back, d.astype(np.int64)))
            rt_fail += (not rt_ok)
            if have_c:
                cd, cend = c_encode(lib, d.astype(np.uint8), f9, 8 * len(d) + 64)
                cback = c_decode(lib, cd, cend, f9, len(d))
                if not np.array_equal(cback, d.astype(np.uint8)):
                    rt_fail += 1
                if int(cend) != int(end) or not np.array_equal(cd, np.asarray(emits, np.uint8)):
                    parity_fail += 1
        else:                                       # big streams: C roundtrip only
            cd, cend = c_encode(lib, d.astype(np.uint8), f9, 4 * len(d) + 64)
            cback = c_decode(lib, cd, cend, f9, len(d))
            rt_ok = bool(np.array_equal(cback, d.astype(np.uint8)))
            rt_fail += (not rt_ok)
            emits, end = cd, cend
        # byte coder rate on the same stream
        if len(d) <= 30_000_000:
            f12 = rans.normalize_freqs(d.astype(np.int64), ks_n, 4096)
            enc = rans_fast.encode_fast(d, f12, 12)
        else:
            f12 = rans.normalize_freqs(d[:30_000_000].astype(np.int64), ks_n, 4096)
            enc = rans_fast.encode_fast(d[:30_000_000], f12, 12)
        rate_byte = 8.0 * (len(enc) + 4) / min(len(d), 30_000_000)
        # base-9 rate: exact output length = full 169-digit blocks (536 bits
        # each) + tightly packed tail + 6-byte end state (state needs 41.25 bits)
        nb_full, tail = divmod(len(emits), B9_BLOCK)
        tail_bits = int(np.ceil(tail * LOG2_9 / 8)) * 8 if tail else 0
        stream_bits = nb_full * (8 * B9_BYTES) + tail_bits + 48
        rate_b9 = stream_bits / len(d)
        packed = b9_pack(emits) if len(emits) <= 3_000_000 else None
        if packed is not None:
            up = np.asarray(b9_unpack(packed, len(emits)))
            assert np.array_equal(up, np.asarray(emits).astype(np.int64)), \
                "pack/unpack mismatch"
        rate_rows.append(dict(stream=name, n=len(d), rate_byte=round(rate_byte, 5),
                              rate_b9=round(rate_b9, 5),
                              emit_per_sym=round(len(emits) / len(d), 4)))
        print(f"  {name}: n={len(d):>9,} byte {rate_byte:.4f} | b9 {rate_b9:.4f} "
              f"emits/sym {len(emits)/len(d):.3f} [{'rt-ok' if rt_ok else 'RT-FAIL'}]")

    # small-stream pack roundtrip at container level (already asserted above)
    max_delta = max(r["rate_b9"] - r["rate_byte"] for r in rate_rows)
    v41 = dict(prediction="P41", max_delta_b_per_digit=round(max_delta, 5), bar=0.01,
               verdict="PASS" if max_delta <= 0.01 else "FAIL")
    print(f"[exp28] P41 ({v41['verdict']}): max rate delta b9-vs-byte = {max_delta:.5f} b/digit")

    # ------------------------------------------------ throughput (C cores)
    thr = {}
    if have_c:
        body = [d for _, d, ks_n in streams if ks_n == 9]
        real = np.concatenate(body).astype(np.uint8)
        real = np.tile(real, int(np.ceil(10**8 / len(real))))[: 10**8]
        iid = np.tile(rng.integers(0, 9, 2_000_000).astype(np.uint8), 50)[: 10**8]
        for tag, s in (("real33m", real), ("iid9", iid)):
            # per-coder tables from a fixed 4M prefix (metadata excluded from rates)
            f9 = rans.normalize_freqs(s[:4_000_000].astype(np.int64), 9, B9_M4)
            f12 = rans.normalize_freqs(s[:4_000_000].astype(np.int64), 9, 4096)
            for coder in ("byte", "b9"):
                t0 = time.perf_counter()
                if coder == "b9":
                    em, end = c_encode(lib, s, f9, 4 * len(s))
                    t1 = time.perf_counter()
                    assert np.array_equal(c_decode(lib, em, end, f9, len(s)), s), "c rt"
                    t2 = time.perf_counter()
                else:
                    eb = rans_fast.encode_fast(s, f12, 12)
                    t1 = time.perf_counter()
                    assert np.array_equal(
                        rans_fast.decode_fast(eb, f12, len(s), 12).astype(np.uint8), s), "rt"
                    t2 = time.perf_counter()
                thr[f"{tag}:{coder}"] = dict(enc_sym_s=round(len(s) / (t1 - t0) / 1e6, 1),
                                             dec_sym_s=round(len(s) / (t2 - t1) / 1e6, 1))
        print(f"[exp28] throughput (M sym/s): {json.dumps(thr, indent=1)}")
        b9_slowdown_enc = thr["iid9:byte"]["enc_sym_s"] / thr["iid9:b9"]["enc_sym_s"]
        b9_slowdown_dec = thr["iid9:byte"]["dec_sym_s"] / thr["iid9:b9"]["dec_sym_s"]
        v42 = dict(prediction="P42", slowdown_enc=round(b9_slowdown_enc, 2),
                   slowdown_dec=round(b9_slowdown_dec, 2), bar=2.0,
                   verdict="PASS" if max(b9_slowdown_enc, b9_slowdown_dec) <= 2.0 else "FAIL")
    else:
        v42 = dict(prediction="P42", verdict="SKIP (no gcc)")
    print(f"[exp28] P42 ({v42.get('verdict')})")

    # ------------------------------------------------ P43: 1e9 roundtrip
    if have_c:
        total, mism = 0, 0
        t0 = time.perf_counter()
        base = rng.integers(0, 9, 10_000_000).astype(np.uint8)
        for c in range(10):
            s = np.tile(base, 10) if c % 2 == 0 else \
                (np.tile(base, 10) + c) % 9
            s = s.astype(np.uint8)
            f9 = rans.normalize_freqs(base.astype(np.int64), 9, B9_M4)
            em, end = c_encode(lib, s, f9, 2 * len(s) + 64)
            back = c_decode(lib, em, end, f9, len(s))
            mism += int((back != s).sum())
            total += len(s)
        dt = time.perf_counter() - t0
        assert mism == 0, f"P43 roundtrip mismatches: {mism}"
        v43 = dict(prediction="P43", symbols=total, mismatches=mism,
                   sym_s=round(total / dt / 1e6, 1),
                   verdict="PASS" if mism == 0 and total >= 10**9 else "FAIL")
    else:
        v43 = dict(prediction="P43", verdict="SKIP (no gcc)")
    print(f"[exp28] P43 ({v43['verdict']}): {v43.get('symbols', 0):,} symbols, "
          f"{v43.get('mismatches', '?')} mismatches")

    # ------------------------------------------------ M ablation (rate)
    abl = {}
    probe_idx = min(5, len(streams) - 1)
    probe, ks_n_probe = streams[probe_idx][1], streams[probe_idx][2]
    for n_exp in (4, 5, 6):
        M = 9 ** n_exp; m_exp = n_exp + 8           # thresh = fs * 9^9 always
        f9 = rans.normalize_freqs(probe.astype(np.int64), ks_n_probe, M)
        emits, end = b9_encode(probe, f9, m_exp=m_exp, n_exp=n_exp)
        nb_full, tail = divmod(len(emits), B9_BLOCK)
        tail_bits = int(np.ceil(tail * LOG2_9 / 8)) * 8 if tail else 0
        rate = (nb_full * 8 * B9_BYTES + tail_bits + 48) / len(probe)
        abl[f"9^{n_exp}"] = round(rate, 5)
    print(f"[exp28] M ablation (b/digit): {abl}")

    if parity_fail or rt_fail:
        print(f"[exp28] !!! roundtrip failures: {rt_fail}, C-parity failures: {parity_fail}")
    out = dict(rate_rows=rate_rows, P41=v41, P42=v42, P43=v43, thr=thr, abl=abl,
               parity_fail=parity_fail, rt_fail=rt_fail,
               op_census=dict(renorm_emit_ops="x % 9, x //= 9 (base-9 digit ops)",
                              byte_coder_ops="x & 0xFF, x >>= 8",
                              symbol_slot="x % 9^4 = low 4 base-9 digits of state"))
    with open(os.path.join(RES := os.path.join(os.path.dirname(HERE), "results"),
                           "exp28_rans_base9.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("[exp28] wrote results/exp28_rans_base9.json")


def shutil_which(x):
    import shutil
    return shutil.which(x)


if __name__ == "__main__":
    main()