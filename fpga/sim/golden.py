"""Golden models for Gate-0.

k96_partials replicates ggml_vec_dot_k9_6_q8_0_generic (fork branch k9,
ggml/src/ggml-cpu/quants.c) op-for-op: IEEE fp32, separate mul+add, no FMA
contraction, same accumulation tree (per 64-group: two int32 block sums,
each scaled by its own fp16 activation scale, then one fp16 weight-scale
product added sequentially; groups then superblocks chained left-to-right).
Vector builders share layouts with the RTL ports (docs/08 §2).
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import ninths_native as nn  # weight packer (docs/08, Gate A validated)


def f32b(x) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def b32f(v: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", v), dtype="<f4")[0]


def k96_partials(wblks, act_blocks):
    """Bit-exact replication of the C kernel: one running fp32 sumf across
    superblocks; returns the cumulative sumf bits after each superblock
    (matching C's state after each i-loop iteration) + the final."""
    sumf = np.float32(0.0)
    cums = []
    for i, wblk in enumerate(wblks):
        w = np.frombuffer(wblk, dtype=np.uint8)
        sc = w[:8].view("<f2").astype(np.float32)
        for g in range(4):
            d0 = sc[g]
            qs = w[8 + 32 * g : 8 + 32 * g + 32]
            hh = w[136 + 16 * g : 136 + 16 * g + 16]
            sumi = np.float32(0.0)
            for k in (0, 1):
                yb = np.frombuffer(act_blocks[8 * i + 2 * g + k], dtype=np.uint8)
                d1 = yb[:2].view("<f2").astype(np.float32)[0]
                qy = yb[2:34].view(np.int8)
                sblk = 0
                for b in range(16):
                    c0 = 32 * k + 2 * b
                    c1 = c0 + 1
                    lb = int(qs[16 * k + b])
                    h0 = (int(hh[c0 >> 2]) >> ((c0 & 3) * 2)) & 3
                    h1 = (int(hh[c1 >> 2]) >> ((c1 & 3) * 2)) & 3
                    u0 = ((lb & 0x0F) | (h0 << 4)) - 32
                    u1 = ((lb >> 4) | (h1 << 4)) - 32
                    sblk += u0 * int(qy[2 * b]) + u1 * int(qy[2 * b + 1])
                sumi = np.float32(sumi + np.float32(d1) * np.float32(sblk))
            sumf = np.float32(sumf + d0 * sumi)
        cums.append(f32b(sumf))
    return cums, f32b(sumf)


# ------------------------------------------------------------- vector gen ----
def gen_superblocks(rng, count, k=63, m=None):
    """Random weight superblocks via the Gate-A packer; m overrides scales."""
    if m is None:
        m = (10.0 ** rng.uniform(-5, 1.0, (count, 4))).astype(np.float32)
    d = rng.integers(0, k, (count, 256)).astype(np.uint8)
    blob = nn.pack(d, k, m)
    return [bytes(b) for b in blob.reshape(count, 200)]


def gen_act_blocks(rng, count, d16=None, codes=None):
    """Random-ish Q8_0 blocks (34 B each = fp16 scale LE + 32 int8 codes)."""
    out = []
    for i in range(count):
        if d16 is None:
            h = np.float16(rng.uniform(0.25, 4.0))
        else:
            h = np.float16(d16)
        if codes is None:
            q = rng.integers(-128, 128, 32).astype(np.uint8)
        else:
            q = np.full(32, codes & 0xFF, dtype=np.uint8)
        out.append(struct.pack("<e", h) + q.tobytes())
    return out


def act_flat_int(blocks):
    """Pack Q8_0 blocks into the 272-byte little-endian act port value."""
    return int.from_bytes(b"".join(blocks), "little")