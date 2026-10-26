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
def lane_plan(nbytes, rows, want_lanes):
    """docs/11 rule: L = max(1, min(want, bytes>>21, rows))."""
    return max(1, min(int(want_lanes), (int(nbytes) + (1 << 21) - 1) >> 21, int(rows)))


def encode_tensor(digits_u8, k, scales, scale_mode, perm=None, lanes=1, shape=None):
    flat = np.ascontiguousarray(np.asarray(digits_u8, dtype=np.uint8).reshape(-1))
    f = normalize_freqs(flat.astype(np.int64), k, 1 << SCALE_BITS)
    L = 1
    if shape is not None and lanes and lanes > 1:
        r0, c0 = int(shape[0]), int(shape[1])
        L = lane_plan(flat.size, r0, lanes)
    if L <= 1 or shape is None:
        return dict(k=int(k), lanes=1, freq=f.astype("<u2").tobytes(),
                    digits=_enc(flat, f, SCALE_BITS),
                    scales=encode_scales(scales, scale_mode),
                    perm=b"" if perm is None else np.asarray(perm, dtype="<u4").tobytes())
    # v2: row-aligned lane chunks, one rANS state per lane, shared freq table.
    # lane_off[L+1] are u32 offsets relative to the sub-stream area; each lane
    # stream is exactly one v1 stream (renorm bytes + trailing BE state).
    rows, cols = int(shape[0]), int(shape[1])
    rows_per = (rows + L - 1) // L
    bounds = [min(rows, i * rows_per) for i in range(L + 1)]
    chunks = [flat[a * cols: min(rows, bnd) * cols] if bnd > a else flat[:0]
              for a, bnd in zip(bounds, bounds[1:])]
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=min(L, 32)) as ex:
        streams = list(ex.map(lambda ch: _enc(ch, f, SCALE_BITS), chunks))
    offs = [0]
    for st in streams:
        offs.append(offs[-1] + len(st))
    head = struct.pack("<%uI" % (L + 1), *offs)
    return dict(k=int(k), lanes=L, freq=f.astype("<u2").tobytes(),
                digits=head + b"".join(streams),
                scales=encode_scales(scales, scale_mode),
                perm=b"" if perm is None else np.asarray(perm, dtype="<u4").tobytes())


def decode_digits(rec):
    """Decode a record's rANS digit stream (uint8, flat, stored order)."""
    f = np.frombuffer(rec["freq"], dtype="<u2").astype(np.int64)
    r, c = rec["shape"]
    n = r * c
    L = int(rec.get("lanes", 1) or 1)
    if L <= 1:
        return _dec(rec["digits"], f, n, SCALE_BITS).astype(np.uint8)
    buf = rec["digits"]
    otab = 4 * (L + 1)
    offs = np.frombuffer(buf[:otab], dtype="<u4")
    assert int(offs[0]) == 0 and int(offs[-1]) == len(buf) - otab, "k9 v2 lane offsets broken"
    base = otab
    out = np.empty(n, dtype=np.uint8)
    rows_per = (r + L - 1) // L
    for li in range(L):
        a = min(r, li * rows_per)
        b = min(r, (li + 1) * rows_per)
        if a >= b:
            break
        seg = buf[base + int(offs[li]): base + int(offs[li + 1])]
        cnt = (b - a) * c
        out[a * c: b * c] = _dec(seg, f, cnt, SCALE_BITS)
    return out


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


# ------------------------------------------------- iq4_nl materialization ----
IQ4NL_VALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                          1,   13,  25,  38,  53,  69,  89, 113], dtype=np.float32)


def materialize_iq4_nl(digits, scales, rows, cols, H, group=64, chunk=768):
    """Vectorized mirror of the fork's k9 iq4_nl materialization branch.
    Per 32-weight block: d = amax/113 (f16), codes = first-strict-minimum
    codebook match, qs[j] = L[j] | (L[j+16] << 4). Row-chunked so peak
    memory stays ~hundreds of MB even for expert tensors."""
    flat = np.asarray(digits, dtype=np.uint8).reshape(-1)
    sc = np.asarray(scales, dtype=np.float32).reshape(-1)
    vals_tab = IQ4NL_VALUES.astype(np.float32)
    out = np.empty(rows * (cols // 32) * 18, dtype=np.uint8)
    Hf = np.float32(H)
    for a in range(0, rows, chunk):
        b = min(rows, a + chunk)
        seg = flat[a * cols: b * cols]
        mrep = np.repeat(sc[a * (cols // group): b * (cols // group)], group)
        v = (mrep * (seg.astype(np.float32) - Hf)) / Hf
        blk = v.reshape(-1, 32)
        amax = np.abs(blk).max(axis=1)
        d32 = np.where(amax < np.float32(1e-8), np.float32(0),
                       amax / np.float32(113.0)).astype(np.float32)
        d16 = d32.astype(np.float16)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(d32 > 0,
                           np.float32(1.0) / np.float32(d32), np.float32(0))
        tgt = (blk * inv[:, None]).astype(np.float32)
        # (blocks, 32, 16) argmin in one shot; np.argmin keeps the FIRST
        # minimum on ties, same as the C strict-less scan
        diff = np.abs(tgt[:, :, None] - vals_tab[None, None, :])
        L = np.argmin(diff, axis=2).astype(np.uint8)
        del diff, tgt, blk, v, mrep
        nb2 = (b - a) * (cols // 32)
        outb = out[a * (cols // 32) * 18: b * (cols // 32) * 18].reshape(-1, 18)
        outb[:, :2] = d16.view(np.uint8).reshape(nb2, 2)
        outb[:, 2:] = L[:, :16] | (L[:, 16:] << np.uint8(4))
    return out.tobytes()


def decode_digits_par(rec, workers=16):
    """Lane-parallel decode of a (possibly v2) record: each lane decodes its
    row-aligned chunk independently (ctypes releases the GIL)."""
    import concurrent.futures as cf
    L = int(rec.get("lanes", 1) or 1)
    if L <= 1:
        return decode_digits(rec)
    f = np.frombuffer(rec["freq"], dtype="<u2").astype(np.int64)
    r, c = rec["shape"]
    buf = rec["digits"]
    otab = 4 * (L + 1)
    offs = np.frombuffer(buf[:otab], dtype="<u4")
    base = otab
    out = np.empty(r * c, dtype=np.uint8)
    rows_per = (r + L - 1) // L

    def lane(li):
        a = min(r, li * rows_per)
        b = min(r, (li + 1) * rows_per)
        if a >= b:
            return None
        seg = buf[base + int(offs[li]): base + int(offs[li + 1])]
        cnt = (b - a) * c
        return a * c, _dec(seg, f, cnt, SCALE_BITS)

    with cf.ThreadPoolExecutor(max_workers=min(L, workers)) as ex:
        for res in ex.map(lane, range(L)):
            if res is None:
                continue
            a0, arr = res
            out[a0: a0 + arr.size] = arr
    return out


def materialize_from(digits, scales, r, c, k, mode="q8_0", as_bytes=True):
    """Materialize decoded digits+scales into GGML block bytes (q8_0 exact /
    iq4_nl RTN). digits: (r*c,) uint8; scales: (r*(c/64),) f32."""
    Hh = (k - 1) // 2
    if mode == "q8_0":
        d16 = (np.asarray(scales, dtype=np.float32).reshape(r, c // 64) / np.float32(Hh)).astype(np.float16)
        # uint8 wraparound then int8 view: byte-exact to (digit - H) as int8,
        # without materializing an int64 intermediate (OOM-safe at 1B+ params)
        codes = (np.asarray(digits, dtype=np.uint8).reshape(-1) - np.uint8(Hh)).astype(np.int8)
        canvas = np.empty((r * (c // 32), 34), dtype=np.uint8)
        canvas[:, :2] = np.repeat(d16, 2, axis=1).view(np.uint8).reshape(-1, 2)
        canvas[:, 2:34] = codes.reshape(-1, 32).view(np.uint8)
        return bytes(canvas.reshape(-1)) if as_bytes else canvas.reshape(-1)
    if mode == "iq4_nl":
        try:
            out = rans_fast.pack_iq4nl(digits, scales, r, c, k, threads=16)
            return out if as_bytes else np.frombuffer(out, dtype=np.uint8)
        except (AttributeError, RuntimeError):
            pass
        return materialize_iq4_nl(digits, scales, r, c, H=Hh)
    raise ValueError(f"unsupported materialize target {mode}")


def materialize(rec, mode="q8_0"):
    """Materialize a record (decodes digits + scales first)."""
    r, c = rec["shape"]
    sm = rec.get("scale_mode", 4)  # 4 == ent8 (the only mode we emit today)
    digits = decode_digits_par(rec)
    ms = decode_scales(rec["scales"], sm, r * (c // 64))
    return materialize_from(digits, ms, r, c, rec["k"], mode)


class ChunkedLaneDecoder:
    """Band-wise decode of a (v2-lane) record with bounded memory. Each band
    re-decodes the needed lane prefixes (rANS is forward-sequential, so a
    band costs up to one lane-length of extra symbols — noise vs the pack)."""

    def __init__(self, rec, band=33554432, workers=16):
        self.rec = rec
        self.r, self.c = rec["shape"]
        self.L = int(rec.get("lanes", 1) or 1)
        self.band = max(1024, band)
        self.workers = workers
        self.f = np.frombuffer(rec["freq"], dtype="<u2").astype(np.int64)
        self._full = None
        if self.L > 1:
            buf = rec["digits"]
            self.otab = 4 * (self.L + 1)
            self.offs = np.frombuffer(buf[:self.otab], dtype="<u4")
            self.sub = buf[self.otab:]
        else:
            self.stream = rec["digits"]

    def row_bands(self):
        p = 0
        while p < self.r:
            yield p, min(self.r, p + self.band)
            p += self.band

    def decode_band(self, a0, b0):
        n = (b0 - a0) * self.c
        out = np.empty(n, dtype=np.uint8)
        if self.L <= 1:
            if self._full is None:
                self._full = _dec(self.stream, self.f, self.r * self.c, SCALE_BITS)
            out[:] = self._full[a0 * self.c: b0 * self.c]
            return out
        rows_per = (self.r + self.L - 1) // self.L
        jobs = []
        for li in range(self.L):
            la = min(self.r, li * rows_per)
            lb = min(self.r, (li + 1) * rows_per)
            x0, x1 = max(a0, la), min(b0, lb)
            if x0 >= x1:
                continue
            seg = self.sub[int(self.offs[li]): int(self.offs[li + 1])]
            jobs.append((x0 - a0, seg, (x1 - la) * self.c, (x1 - x0) * self.c))

        import concurrent.futures as cf

        def run(job):
            dst_off, seg, need, want = job
            arr = _dec(seg, self.f, need, SCALE_BITS)
            return dst_off, arr[-want:] if want else arr[:0]

        with cf.ThreadPoolExecutor(max_workers=min(self.workers, len(jobs) or 1)) as ex:
            for dst_off, arr in ex.map(run, jobs):
                out[dst_off: dst_off + arr.size] = arr
        return out
