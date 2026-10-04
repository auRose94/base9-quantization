#!/usr/bin/env python3
"""k9 — reference implementation of the K9 container codec (docs/04).

Standalone file format for weights stored on the odd b^L−1 grids (9/15/27/63/99
…), entropy-coded with order-0 rANS:

    [header] [tensor directory] [per-tensor blobs: freq table | digits | scales | perm]

Digits are uint8 (k <= 255); each tensor carries its own normalized frequency
table (uint16 x k, M = 4096) and its per-(row, group) scales in one of several
modes. Act-order GPTQ builds groups in a permuted column order, so an optional
uint32 permutation is stored and undone at decode.

Dequantization (normative):  w = m * (d - H) / H,  H = (k-1)//2.

This module is deliberately simple and self-asserting; it reuses the validated
pure-Python rANS in rans.py (so it is slow — a production codec needs a
vectorized/native coder, see docs/04 §13).
"""
import struct

import numpy as np

from rans import normalize_freqs, rans_encode, rans_decode, _tables

try:                                    # optional compiled coder (byte-identical)
    import rans_fast
    FAST_CODER = rans_fast.HAVE_FAST
except Exception:                       # no compiler -> pure-Python fallback
    rans_fast = None
    FAST_CODER = False


def _enc(flat_u8, f, k):
    if FAST_CODER:
        return rans_fast.encode_fast(flat_u8, f, k)
    return rans_encode(np.asarray(flat_u8, dtype=np.int64), f, k)


def _dec(stream, f, n, k):
    if FAST_CODER:
        return rans_fast.decode_fast(stream, f, n, k)
    return rans_decode(stream, f, n, k)

MAGIC = b"K9Q1"
VERSION = 1
SCALE_BITS = 12                      # M = 4096
SCALE_MODES = {"fp32": 0, "fp16": 1, "ent8": 4}
_HEADER = "<4sHHIIQQ"                # magic, version, flags, n, scale_bits, dir_off, data_off
_HEADER_SIZE = struct.calcsize(_HEADER)
_SCALE_HDR = struct.calcsize("<ffH")  # lo, hi, table_len for scale mode 4


# ------------------------------------------------------------- scales ----
def _mode_id(mode):
    """Accept either the mode name ('fp32'/'fp16'/'ent8') or its integer id."""
    return SCALE_MODES[mode] if isinstance(mode, str) else int(mode)


def encode_scales(sc, mode):
    mode = _mode_id(mode)
    sc = np.ascontiguousarray(np.asarray(sc, dtype=np.float32).reshape(-1))
    if mode == 0:                                            # fp32
        return sc.tobytes()
    if mode == 1:                                            # fp16
        return sc.astype(np.float16).tobytes()
    if mode == 4:                                            # entropy-coded 8-bit log
        x = np.log2(np.maximum(sc.astype(np.float64), 1e-30))
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            hi = lo + 1e-30
        q = np.clip(((x - lo) / (hi - lo) * 255.0).round().astype(np.int64), 0, 255)
        f = normalize_freqs(q, 256, 1 << SCALE_BITS)
        body = _enc(q, f, SCALE_BITS)
        return (struct.pack("<ffH", lo, hi, 256)
                + f.astype("<u2").tobytes() + body)
    raise NotImplementedError(f"scale mode {mode}")


def decode_scales(buf, mode, n):
    mode = _mode_id(mode)
    if mode == 0:
        return np.frombuffer(buf, dtype="<f4", count=n).copy()
    if mode == 1:
        return np.frombuffer(buf, dtype="<f2", count=n).astype(np.float32)
    if mode == 4:
        lo, hi, k = struct.unpack_from("<ffH", buf, 0)
        f = np.frombuffer(buf, dtype="<u2", count=k,
                          offset=_SCALE_HDR).astype(np.int64)
        q = _dec(buf[_SCALE_HDR + 2 * k:], f, n, SCALE_BITS)
        return (2.0 ** (lo + (hi - lo) * q / 255.0)).astype(np.float32)
    raise NotImplementedError(f"scale mode {mode}")


# ------------------------------------------------------------ tensors ----
def encode_tensor(digits_u8, k, scales, scale_mode, perm=None):
    flat = np.ascontiguousarray(np.asarray(digits_u8, dtype=np.uint8).reshape(-1))
    f = normalize_freqs(flat.astype(np.int64), k, 1 << SCALE_BITS)
    return dict(k=int(k), freq=f.astype("<u2").tobytes(),
                digits=_enc(flat, f, SCALE_BITS),
                scales=encode_scales(scales, scale_mode),
                perm=b"" if perm is None else np.asarray(perm, dtype="<u4").tobytes())


def decode_digits(rec):
    """Decode a record's rANS digit stream (uint8, flat, stored order)."""
    f = np.frombuffer(rec["freq"], dtype="<u2").astype(np.int64)
    r, c = rec["shape"]
    return _dec(rec["digits"], f, r * c, SCALE_BITS).astype(np.uint8)


def load_into(module, rec, group, device="cpu", row_chunk=0):
    """Dequantize a record straight into `module.weight`, optionally in row
    chunks so the full fp32 tensor never materialises on the device.

    `row_chunk=0` builds the whole tensor at once (as decode_tensor does);
    `row_chunk=4096` bounds the device allocation to (4096, c) floats — this is
    what keeps the embed (151936x1536 fp32 ~ 0.9 GB) from OOMing on a small GPU.
    """
    import torch
    r, c = rec["shape"]
    k = rec["k"]
    H = (k - 1) // 2
    d = torch.from_numpy(decode_digits(rec).astype(np.int64)).view(r, c // group, group)
    m = torch.from_numpy(
        decode_scales(rec["scales"], rec["scale_mode"], r * (c // group))
    ).view(r, c // group, 1)
    inv = None
    if rec["perm"] is not None:
        perm = torch.from_numpy(np.frombuffer(rec["perm"], dtype="<u4").astype(np.int64))
        inv = torch.argsort(perm).to(device)
    W = module.weight.data
    step = max(1, row_chunk) if row_chunk else r
    for a in range(0, r, step):
        b = min(r, a + step)
        q = (m[a:b].to(device) * (d[a:b].to(device).float() - H) / H)
        if inv is not None:
            q = q[:, inv]
        W[a:b].copy_(q.reshape(b - a, c))
        del q
    return r * c


def decode_tensor(rec, group, device="cpu"):
    """Reconstruct the (r, c) float32 dequantized weight tensor from a record."""
    import torch
    r, c = rec["shape"]
    k = rec["k"]
    H = (k - 1) // 2
    d = decode_digits(rec)
    n_scales = (r * (c // group))
    m = decode_scales(rec["scales"], rec["scale_mode"], n_scales)
    dt = torch.from_numpy(d.astype(np.int64)).view(r, c // group, group).float()
    mt = torch.from_numpy(m).view(r, c // group, 1)
    q = (mt * (dt - H) / H).reshape(r, c)
    if rec["perm"] is not None:
        perm = torch.from_numpy(np.frombuffer(rec["perm"], dtype="<u4").astype(np.int64))
        q = q[:, torch.argsort(perm)]
    return q.to(device)


# ------------------------------------------------------------ container ----
def _encode_one(t):
    # a tensor may carry a pre-encoded `rec` (lets a caller encode on the fly
    # and drop the raw digits, which matters for 7B+ where digits are ~8 GB)
    rec = t.get("rec")
    if rec is None:
        rec = encode_tensor(t["digits"], t["k"], t["scales"], t["scale_mode"],
                            t.get("perm"))
    out = dict(rec)
    out["name"] = t["name"]
    out["shape"] = tuple(int(x) for x in t["shape"])
    out["group"] = int(t["group"])
    out["mode_id"] = SCALE_MODES[t["scale_mode"]]
    return out


def write_k9(path, tensors, threads=0):
    """tensors: iterable of dicts with name, shape, group, scale_mode, k,
    digits (uint8 array), scales, perm (optional). Returns bytes written.

    `threads > 1` encodes tensors concurrently: tensors are independent, so
    this is embarrassingly parallel, and the ctypes call into the C coder
    releases the GIL. Output bytes are identical either way.
    """
    tensors = list(tensors)
    if threads and threads > 1 and len(tensors) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=threads) as ex:
            recs = list(ex.map(_encode_one, tensors))
    else:
        recs = [_encode_one(t) for t in tensors]

    def hdr_fmt(name, shape):
        return f"<H{len(name)}sB{len(shape)}IBBBBBBIIIIQ"

    dir_off = _HEADER_SIZE
    dir_size = sum(struct.calcsize(hdr_fmt(r["name"].encode(), r["shape"]))
                   for r in recs)
    offset = dir_off + dir_size
    with open(path, "wb") as fh:
        fh.write(struct.pack(_HEADER, MAGIC, VERSION, 0, len(recs), SCALE_BITS,
                             dir_off, offset))
        for r in recs:                                   # directory
            name = r["name"].encode()
            fh.write(struct.pack(
                hdr_fmt(name, r["shape"]), len(name), name, len(r["shape"]),
                *r["shape"], r["k"], r["group"], 0, r["mode_id"], 0,
                1 if len(r["perm"]) else 0,
                len(r["freq"]), len(r["digits"]), len(r["scales"]), len(r["perm"]),
                offset))
            offset += (len(r["freq"]) + len(r["digits"])
                        + len(r["scales"]) + len(r["perm"]))
        for r in recs:                                   # blobs
            fh.write(r["freq"])
            fh.write(r["digits"])
            fh.write(r["scales"])
            fh.write(r["perm"])
        return fh.tell()


def _parse_directory(buf, pos, n):
    """Parse n directory records -> list of metadata dicts (no blobs)."""
    inv_mode = {v: k for k, v in SCALE_MODES.items()}
    out = []
    for _ in range(n):
        (name_len,) = struct.unpack_from("<H", buf, pos)
        pos += 2
        name = buf[pos:pos + name_len].decode()
        pos += name_len
        (rank,) = struct.unpack_from("<B", buf, pos)
        pos += 1
        shape = struct.unpack_from(f"<{rank}I", buf, pos)
        pos += 4 * rank
        k, group, codec, scale_mode, pair, pflags = struct.unpack_from(
            "<BBBBBB", buf, pos)
        pos += 6
        tbl, dig, sca, plen = struct.unpack_from("<IIII", buf, pos)
        pos += 16
        (off,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        out.append(dict(name=name, shape=shape, k=k, group=group,
                        scale_mode=inv_mode[scale_mode], tbl=tbl, dig=dig,
                        sca=sca, plen=plen, off=off))
    return out


def read_k9(path):
    """Return {name: record} with keys shape, k, group, scale_mode, freq,
    digits, scales, perm. Loads the whole file (see K9File for streaming)."""
    with open(path, "rb") as fh:
        buf = fh.read()
    magic, version, flags, n, sb, dir_off, data_off = struct.unpack_from(
        _HEADER, buf, 0)
    assert magic == MAGIC and version == VERSION, "not a K9 file"
    assert sb == SCALE_BITS
    out = {}
    for m in _parse_directory(buf, dir_off, n):
        blob = buf[m["off"]:m["off"] + m["tbl"] + m["dig"] + m["sca"] + m["plen"]]
        out[m["name"]] = dict(
            shape=m["shape"], k=m["k"], group=m["group"],
            scale_mode=m["scale_mode"],
            freq=blob[:m["tbl"]], digits=blob[m["tbl"]:m["tbl"] + m["dig"]],
            scales=blob[m["tbl"] + m["dig"]:m["tbl"] + m["dig"] + m["sca"]],
            perm=(blob[m["tbl"] + m["dig"] + m["sca"]:] if m["plen"] else None))
    return out


class K9File:
    """Random-access reader: parses the header/directory once, then reads a
    single tensor's blob on demand, so a caller can stream the model in
    tensor-by-tensor without ever holding the whole file or the whole fp32
    model (peak memory ~ the largest tensor)."""

    def __init__(self, path):
        self.path = path
        self.fh = open(path, "rb")
        magic, version, flags, n, sb, dir_off, data_off = struct.unpack(
            _HEADER, self.fh.read(_HEADER_SIZE))
        assert magic == MAGIC and version == VERSION, "not a K9 file"
        assert sb == SCALE_BITS
        self.fh.seek(dir_off)
        self.meta = _parse_directory(self.fh.read(data_off - dir_off), 0, n)
        self.total_bytes = data_off - dir_off

    def names(self):
        return [m["name"] for m in self.meta]

    def record(self, name):
        m = next(x for x in self.meta if x["name"] == name)
        self.fh.seek(m["off"])
        blob = self.fh.read(m["tbl"] + m["dig"] + m["sca"] + m["plen"])
        return dict(shape=m["shape"], k=m["k"], group=m["group"],
                    scale_mode=m["scale_mode"],
                    freq=blob[:m["tbl"]], digits=blob[m["tbl"]:m["tbl"] + m["dig"]],
                    scales=blob[m["tbl"] + m["dig"]:m["tbl"] + m["dig"] + m["sca"]],
                    perm=(blob[m["tbl"] + m["dig"] + m["sca"]:] if m["plen"] else None))

    def iter_tensors(self):
        for m in self.meta:
            yield m["name"], self.record(m["name"])

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
