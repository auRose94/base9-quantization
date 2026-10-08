#!/usr/bin/env python3
"""Ninths native resident containers (stage 2) — bit-exact packer/unpacker.

Format name: ninths; K9 is the secondary shorthand (and the GGML type prefix
K9_4/K9_6/K9_7). The research/rANS container keeps the .k9 file extension.
Reference dequant is normative in k9.py:  w = m * (d - H) / H,  H = (k-1)//2.

The native container bakes H and /H out of the runtime path at export time:

    code = d - H         signed integer, |code| <= 49 (2.6x headroom in int8)
    s    = fp16(m / H)   one fp16 scale per 64-column group, round-nearest-even
    w    = s * code      exact in fp32 (f16 scale x code <= 49 stays < 24 bits)

So runtime kernels see plain (code, scale) operands and no ninths math at all,
and the type carries no k: K9_4/K9_6/K9_7 are bit-width containers (k<=15/63/99).

Frozen layout (docs/08), superblock = 256 codes along the stored row = GGUF
ne[0], holding 4 x 64-groups:

  type   fields                                  bytes   bits/param
  k9_4   sc[8]  ql[128]                          136     4.25
  k9_6   sc[8]  ql[128]  qh[64]                  200     6.25
  k9_7   sc[8]  cs[256]                          264     8.25

  sc  one superblock's 4 little-endian fp16 scales (64-groups in order)
  ql  two codes per byte, first in the low nibble, second in the high nibble,
      sequential; nibble value = code + 8 (k9_4), or the low half of the
      unsigned value u6 = code + 32 (k9_6)
  qh  (k9_6 only) byte j holds the 2-bit high halves of u6 for codes
      (4j .. 4j+3) in bit pairs 0-1, 2-3, 4-5, 6-7
  cs  (k9_7) signed int8 = code

The sequential packing is deliberately NOT the k-quant SIMD interleave: the
kernels belong to this project, and numpy packing/unpacking stays vectorizable.

Numerics equivalence to stage 1 (docs/05 §2): the fork's loader materialized
Q8_0 with the same fp16(m/H) scale and the same int8 codes, so
unpack(pack(digits, k, m)) is bitwise equal to the C materializer's Q8_0
blocks — the docs/05 ppl table carries over unchanged. Gate A proves this
end-to-end against the materialized GGUF (ninths_gate_a.py).
"""
import numpy as np

NBLK = 256                       # codes per superblock == QK_K
GROUP = 64                       # codes per fp16 scale
BLOCK_BYTES = {"k9_4": 8 + 128, "k9_6": 8 + 128 + 64, "k9_7": 8 + 256}
MAX_K = {"k9_4": 15, "k9_6": 63, "k9_7": 99}
GGML_IDS = {"k9_4": 44, "k9_6": 45, "k9_7": 46}   # fork-provisional values


def type_for_k(k):
    k = int(k)
    if k % 2 == 0 or k < 3:
        raise ValueError(f"k must be an odd grid size >= 3, got {k}")
    for tname, cap in MAX_K.items():
        if k <= cap:
            return tname
    raise ValueError(f"k={k} exceeds the largest container (k9_7, k<=99)")


def bytes_for(tname, ne0, ne1):
    """Tensor blob size: ne1 rows x (ne0/256) superblocks x block bytes."""
    return ne1 * (ne0 // NBLK) * BLOCK_BYTES[tname]


def pack(digits, k, scales, chunk_rows=8192):
    """Pack one stored-frame tensor into its native blob.

    digits : uint8 (rows, cols), row-major stored (out, in) frame, cols % 256 == 0
    scales : per-(row, 64-group) scale values m, shape (rows, cols // 64)
    Returns flat uint8 laid out for GGUF (ne[0] = cols contiguous):
    rows x superblocks x block bytes.
    """
    tname = type_for_k(k)
    H = (k - 1) // 2
    digits = np.asarray(digits, dtype=np.uint8)
    r, c = digits.shape
    m = np.asarray(scales, dtype=np.float32).reshape(r, c // GROUP)
    if c % NBLK != 0:
        raise ValueError(f"cols {c} not divisible by {NBLK} (superblock)")
    if c % GROUP != 0:
        raise ValueError(f"cols {c} not divisible by {GROUP}")
    s16 = (m / np.float32(H)).astype(np.float16)                    # (r, c//64)
    nb = c // NBLK
    out = np.empty((r, nb, BLOCK_BYTES[tname]), dtype=np.uint8)
    step = chunk_rows if chunk_rows else r
    for a in range(0, r, step):
        b = min(r, a + step)
        code = digits[a:b].astype(np.int16) - H
        if tname == "k9_4":
            u = (code + 8).astype(np.uint8)
            if u.max() > 15:
                raise ValueError(f"k9_4 overflow: code {u.max() - 8} outside [-8, 7]")
            payload = (u[:, 0::2] | (u[:, 1::2] << 4)).reshape(b - a, nb, 128)
        elif tname == "k9_6":
            u = (code + 32).astype(np.uint8)
            if u.max() > 63:
                raise ValueError(f"k9_6 overflow: code {u.max() - 32} outside [-32, 31]")
            lo15 = u & 15
            lo = (lo15[:, 0::2] | (lo15[:, 1::2] << 4)).reshape(b - a, nb, 128)
            hi = (u >> 4).reshape(b - a, c // 4, 4)
            qh = hi[..., 0] | (hi[..., 1] << 2) | (hi[..., 2] << 4) | (hi[..., 3] << 6)
            payload = np.concatenate([lo, qh.reshape(b - a, nb, 64)], axis=2)
        else:  # k9_7
            payload = code.astype(np.int8).view(np.uint8).reshape(b - a, nb, 256)
        out[a:b, :, :8] = s16[a:b].reshape(b - a, nb, 4).astype("<f2").view(np.uint8)
        out[a:b, :, 8:] = payload
    return out.reshape(r, nb * BLOCK_BYTES[tname]).reshape(-1)


def unpack(blob, ne0, ne1, tname):
    """Inverse of pack. Returns (weights f32 flat, codes int16 (ne1, ne0),
    scales f32 (ne1, ne0 // 64)) with weights = s * code (bit-exact)."""
    if ne0 % NBLK != 0:
        raise ValueError(f"ne0 {ne0} not divisible by {NBLK} (superblock)")
    arr = np.asarray(blob, dtype=np.uint8)
    if arr.size != bytes_for(tname, ne0, ne1):
        raise ValueError(f"blob size {arr.size} != {bytes_for(tname, ne0, ne1)}")
    arr = arr.reshape(ne1, ne0 // NBLK, BLOCK_BYTES[tname])
    s = arr[:, :, :8].view("<f2").astype(np.float32)                # (r, nb, 4)
    if tname == "k9_4":
        ql = arr[:, :, 8:]
        u = np.empty((ne1, ne0 // NBLK, NBLK), dtype=np.uint8)
        u[..., 0::2] = ql & 15
        u[..., 1::2] = ql >> 4
        code = u.astype(np.int16) - 8
    elif tname == "k9_6":
        ql = arr[:, :, 8:136]
        qh = arr[:, :, 136:]
        hi = np.empty((ne1, ne0 // NBLK, NBLK), dtype=np.uint8)
        hi[..., 0::4] = qh & 3
        hi[..., 1::4] = (qh >> 2) & 3
        hi[..., 2::4] = (qh >> 4) & 3
        hi[..., 3::4] = (qh >> 6) & 3
        lo = np.empty_like(hi)
        lo[..., 0::2] = ql & 15
        lo[..., 1::2] = ql >> 4
        code = (lo | (hi << 4)).astype(np.int16) - 32
    else:  # k9_7
        code = arr[:, :, 8:].view(np.int8).astype(np.int16)
    w = (s[:, :, :, None] * code.reshape(ne1, ne0 // NBLK, 4, 64).astype(np.float32))
    return w.reshape(ne1, ne0).reshape(-1), code.reshape(ne1, ne0), s.reshape(ne1, ne0 // GROUP)


# ------------------------------------------------- stage-1 Q8_0 helpers ----
def unpack_q8_0(raw):
    """Dequantize stage-1 materialized Q8_0 bytes (vectorized mirror of
    k9_llamacpp_gate's replica: per 32-code block two scale bytes, fp16)."""
    raw = np.asarray(raw, dtype=np.uint8)
    if raw.size % 34 != 0:
        raise ValueError(f"Q8_0 byte size {raw.size} not a multiple of 34")
    raw = raw.reshape(-1, 34)
    scales = raw[:, :2].view("<f2").astype(np.float32).reshape(-1)
    codes = raw[:, 2:].view(np.int8).astype(np.float32)
    return (codes * scales[:, None]).reshape(-1)


def pack_q8_0(digits, k, scales, group=64, chunk_rows=8192):
    """numpy replica of the exact stage-1 loader Q8_0 materialization
    (byte-for-byte; validated against the C decoder in k9_llamacpp_gate.py)."""
    H = (k - 1) // 2
    digits = np.asarray(digits, dtype=np.uint8)
    r, c = digits.shape
    m = np.asarray(scales, dtype=np.float32).reshape(r, c // group)
    if c % group != 0:
        raise ValueError(f"cols {c} not divisible by group {group}")
    s16 = (m / np.float32(H)).astype(np.float16)                    # (r, c//group)
    out = np.empty((r * (c // 32), 34), dtype=np.uint8)
    nb = c // 32
    step = chunk_rows if chunk_rows else r
    for a in range(0, r, step):
        b = min(r, a + step)
        codes = (digits[a:b].astype(np.int16) - H).astype(np.int8)
        out[a * nb:b * nb, :2] = np.repeat(s16[a:b], 2, axis=1).reshape(-1).view(np.uint8).reshape(-1, 2)
        out[a * nb:b * nb, 2:] = codes.reshape(b * (c // 32), 32).view(np.uint8)
    return out.reshape(-1)


# ---------------------------------------------------------------- selftest ----
def _selftest():
    rng = np.random.default_rng(9)
    cases = ([(k, "k9_4") for k in (9, 11, 13, 15)]
             + [(k, "k9_6") for k in (17, 27, 63)]
             + [(k, "k9_7") for k in (99,)])
    n_fail = 0
    for k, want_t in cases:
        for r, c in ((1, 256), (3, 512), (17, 768)):
            tname = type_for_k(k)
            assert tname == want_t
            H = (k - 1) // 2
            m = (rng.random((r, c // GROUP)) * 1e-2).astype(np.float64) + 1e-5
            d = rng.integers(0, k, (r, c)).astype(np.uint8)
            blob = pack(d, k, m, chunk_rows=4)
            got_w, got_code, got_s = unpack(blob, c, r, tname)

            code_ref = d.astype(np.int16) - H
            roundtrip = np.array_equal(got_code, code_ref.reshape(r, c))
            # native dequant is exact: s * code with fp16 s (same cast chain as pack)
            m32 = m.astype(np.float32)
            s_ref = np.float32(np.float16(m32 / np.float32(H)))
            w_ref = (s_ref.reshape(r, c // NBLK, 4)[:, :, :, None]
                     * code_ref.reshape(r, c // NBLK, 4, 64).astype(np.float32)).reshape(r, c).reshape(-1)
            exact = np.array_equal(
                got_w.view(np.uint32), w_ref.view(np.uint32))
            # stage-1 equivalence: bitwise equal to Q8_0 replica dequant
            q8 = pack_q8_0(d, k, m)
            st1 = unpack_q8_0(q8)
            equiv = np.array_equal(got_w.view(np.uint32), st1.view(np.uint32))
            nbytes = bytes_for(tname, c, r)
            bpc = nbytes * 8 / (r * c)
            status = "ok" if (roundtrip and exact and equiv) else "FAIL"
            n_fail += 0 if (roundtrip and exact and equiv) else 1
            print(f"  k={k:2d} {tname} ({r:2d},{c:4d}): bytes={nbytes:6d} "
                  f"({bpc:5.2f} b/param) roundtrip={roundtrip} exact={exact} "
                  f"stage1-equal={equiv} -> {status}")
    print("SELFTEST:", "PASS" if n_fail == 0 else f"FAIL ({n_fail})")
    return n_fail == 0


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)