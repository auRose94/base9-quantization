#!/usr/bin/env python3
"""glm_k9_build — GLM-4.5-Air (glm4moe) → K9 GGUF, streaming two-pass builder.

The .k9 container path (write_k9) holds every encoded blob in RAM before
writing — fine at 7B, impossible at 103B params (~73 GB) on a 31 GB machine.
This builder never holds more than one tensor's working set:

  pass1  read the f16 reference GGUF (mmap), RTN-quantize each weight tensor
         (symmetric odd grid, group=64, ent8 scales; k=63 body / k=99 embed,
         lm_head and nextn embeddings), rANS-encode with the compiled C coder,
         append the blob to a spill file, record (name, 2-D frame shape, blob
         offset/length) in a JSON manifest. Rank-3 expert tensors are stored
         as their flattened 2-D frame (out*n_expert, in) — byte-identical to
         the 3-D layout because ggml is ne[0]-contiguous (see the fork's
         llama-model-loader.cpp K9 block and experiments/ninths_rank3_gate.py).
  pass2  assemble the K9 GGUF: copy all KV from the reference, add
         ninths.directory (2-D frame records) + ninths.materialize=q8_0,
         emit each quantized tensor as a 1-D K9 byte blob, copy every other
         reference tensor verbatim (norms/biases/router stay f16/f32).
         --mini emits only the manifest tensors (for llama-k9probe gates).
  gate   llama-k9probe a tensor of the built GGUF and compare digits/scales/
         materialized Q8_0 bytes against a fresh Python encode from the f16
         reference — byte-exactness against the C++ loader, on real weights.

Usage:
  glm_k9_build.py pass1 --ref REF --spill PREFIX [--only RE] [--k-body 63 --k-aux 99]
  glm_k9_build.py pass2 --ref REF --spill PREFIX --out OUT.gguf [--mini]
  glm_k9_build.py gate  --gguf OUT.gguf --ref REF --tensor NAME
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf  # noqa: E402  (fork's gguf-py on PYTHONPATH)
import k9  # noqa: E402
import rans_fast  # noqa: E402

GROUP = 64
SCALE_MODE = "ent8"
SCALE_MODE_ID = k9.SCALE_MODES[SCALE_MODE]
K_BODY, K_AUX = 63, 99
CHUNK_ROWS = 16384          # RTN rows per chunk (~268 MB f32 working set)
HIST_CHUNK = 1 << 24        # histogram symbols per chunk (16M)
PROBE = "/home/rose/Work/llama.cpp/build-cpu/bin/llama-k9probe"

AUX_EXACT = {"token_embd.weight", "output.weight"}


def plan_k(name: str, ne) -> int | None:
    """k for this GGUF tensor, or None to copy it verbatim from the reference."""
    ne = tuple(int(x) for x in ne)
    if len(ne) < 2 or ne[0] % GROUP != 0:
        return None                       # 1-D norms/biases, odd widths
    n = int(np.prod(ne))
    if n < 1_000_000:
        return None                       # router (ffn_gate_inp) etc. stay dense
    if name in AUX_EXACT or name.endswith("nextn.embed_tokens.weight") \
            or "shared_head_head" in name:
        return K_AUX
    return K_BODY


def frame_f32(t) -> np.ndarray:
    """(rows, cols) fp32 materialization of a reference tensor view.

    Reference GGUFs are F16 or BF16; the reader hands F16 back as float16 but
    BF16 as a raw uint8 blob (numpy has no bf16 dtype) — reinterpret pairs and
    bit-shift into f32 (exact)."""
    ne0 = int(t.shape[0])
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        bytes_ = np.asarray(t.data, dtype=np.uint8)
        u16 = bytes_.view(np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(-1, ne0)
    return np.asarray(t.data, dtype=np.float32).reshape(-1, ne0)


def rtn_frame(frame_f16, k: int):
    """RTN on the (rows, cols) f16 frame view → flat digits u8 + flat scales f32.

    Grid identical to exp24's quantize_inplace: idx = round((w+m)·H/m), m the
    per-(row,64) |max|, H=(k-1)//2; decode is w = m·(idx-H)/H."""
    rows, cols = frame_f16.shape
    ng = cols // GROUP
    H = (k - 1) // 2
    digits = np.empty(rows * cols, dtype=np.uint8)
    scales = np.empty(rows * ng, dtype=np.float32)
    for a in range(0, rows, CHUNK_ROWS):
        b = min(rows, a + CHUNK_ROWS)
        Wc = np.ascontiguousarray(frame_f16[a:b], dtype=np.float32)
        Wc = Wc.reshape(b - a, ng, GROUP)
        m = np.abs(Wc).max(axis=2)
        msafe = np.maximum(m, 1e-30)
        idx = np.rint((Wc + m[:, :, None]) * (float(H) / msafe)[:, :, None])
        np.clip(idx, 0, k - 1, out=idx)
        digits[a * cols:b * cols] = idx.reshape(-1).astype(np.uint8)
        scales[a * ng:b * ng] = m.reshape(-1)
    return digits, scales


def normalize_from_counts(counts: np.ndarray, M: int):
    """rans.normalize_freqs with the histogram precomputed (avoids the full
    int64 symbol copy k9.encode_tensor would make). Same counts → same f."""
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    assert total > 0
    f = np.zeros(len(counts), dtype=np.int64)
    for s in range(len(counts)):
        if counts[s]:
            f[s] = max(1, int(round(counts[s] * M / total)))
    diff = int(f.sum()) - M
    order = list(np.argsort(-counts))
    i = 0
    while diff != 0:
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
    return f


LANE_MAX = 16
LANE_BYTES = 1 << 21


def lane_for(rows: int, cols: int) -> int:
    """The docs/11 v2 rule, evaluated with the pre-encode digit count."""
    return k9.lane_plan(rows * cols, rows, LANE_MAX)


def encode_stream(digits: np.ndarray, k: int, scales: np.ndarray,
                  lanes: int = 1, shape=None) -> dict:
    """Byte-identical to k9.encode_tensor(digits, k, scales, 'ent8',
    lanes=lanes, shape=shape) — the gate cross-checks it. lanes > 1 writes
    the v2 lane-interleaved digit area (row-aligned chunks, shared table)."""
    assert rans_fast.HAVE_FAST, "compiled rANS coder required for the full model"
    counts = np.zeros(k, dtype=np.int64)
    for a in range(0, digits.size, HIST_CHUNK):
        counts += np.bincount(digits[a:a + HIST_CHUNK], minlength=k)
    f = normalize_from_counts(counts, 1 << k9.SCALE_BITS)
    L = k9.lane_plan(digits.size, shape[0] if shape else 1, lanes) if shape else 1
    if L <= 1:
        out = dict(k=int(k), freq=f.astype("<u2").tobytes(),
                   digits=rans_fast.encode_fast(digits, f, k9.SCALE_BITS),
                   scales=k9.encode_scales(scales, SCALE_MODE), perm=b"")
        out["lanes"] = 1
        return out
    r, c = shape
    rows_per = (r + L - 1) // L
    bounds = [min(r, i * rows_per) for i in range(L + 1)]
    chunks = [digits[a * c:b * c] for a, b in zip(bounds, bounds[1:]) if b > a]
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=min(L, 32)) as ex:
        streams = list(ex.map(lambda ch: rans_fast.encode_fast(ch, f, k9.SCALE_BITS), chunks))
    offs = [0]
    for st in streams:
        offs.append(offs[-1] + len(st))
    out = dict(k=int(k), lanes=L, freq=f.astype("<u2").tobytes(),
               digits=struct.pack("<%uI" % (L + 1), *offs) + b"".join(streams),
               scales=k9.encode_scales(scales, SCALE_MODE), perm=b"")
    return out


def cmd_pass1(a):
    assert rans_fast.HAVE_FAST, "refusing the pure-Python coder at 103B params"
    r = gguf.GGUFReader(a.ref)
    only = re.compile(a.only) if a.only else None
    n_q = n_cop = 0
    p_q = p_cop = blob_bytes = 0
    plan = []
    for t in r.tensors:
        k = plan_k(t.name, t.shape)
        if (k is not None) and (only is None or only.search(t.name)):
            plan.append((t.name, k))
            n_q += 1
            p_q += int(np.prod(t.shape))
        else:
            n_cop += 1
            p_cop += int(np.prod(t.shape))
    print(f"plan: quantize {n_q} tensors ({p_q/1e9:.2f} B params), "
          f"copy {n_cop} ({p_cop/1e6:.1f} M params)", flush=True)

    t0 = time.time()
    qmap = dict(plan)
    with open(a.spill + ".bin", "wb") as sp:
        manifest, off = [], 0
        for t in r.tensors:
            if t.name not in qmap:
                continue
            k = qmap[t.name]
            ne0 = int(t.shape[0])
            frame = frame_f32(t)
            assert frame.shape[1] == ne0
            digits, scales = rtn_frame(frame, k)
            lanes = lane_for(frame.shape[0], ne0)
            rec = encode_stream(digits, k, scales, lanes=lanes,
                                shape=(frame.shape[0], ne0))
            blob = rec["freq"] + rec["digits"] + rec["scales"]
            sp.write(blob)
            manifest.append(dict(name=t.name, r=frame.shape[0], c=ne0, k=k,
                                 group=GROUP, mode=SCALE_MODE_ID,
                                 tbl=len(rec["freq"]), dig=len(rec["digits"]),
                                 sca=len(rec["scales"]), plen=0,
                                 lanes=int(lanes), off=off, n=len(blob)))
            off += len(blob)
            blob_bytes += len(blob)
            del digits, scales, rec, blob, frame
            m = manifest[-1]
            print(f"  {t.name:44s} k={k:2d} frame=({m['r']},{m['c']}) "
                  f"blob={m['n']/1e6:8.2f} MB  {m['n']*8.0/(m['r']*m['c']):.3f} b/p "
                  f"| cum {blob_bytes/1e9:6.2f} GB  {time.time()-t0:6.0f}s", flush=True)
    Path(a.spill + ".json").write_text(json.dumps(
        dict(ref=a.ref, k_body=a.k_body, k_aux=a.k_aux, records=manifest)))
    print(f"pass1 done: {len(manifest)} records, {blob_bytes/1e9:.2f} GB blobs "
          f"({blob_bytes*8.0/max(p_q,1):.3f} b/param quantized), "
          f"wall {time.time()-t0:.0f}s", flush=True)


def build_directory(records, version: int = 2) -> bytes:
    """v2 (docs/11): u16 version + u16 count, then v1 records plus a u8
    lanes field. Old readers fail loudly on the version instead of
    misdecoding lane streams."""
    out = bytearray()
    out += struct.pack("<HH", version, len(records))
    for m in records:
        name = m["name"].encode()
        out += struct.pack("<H", len(name)) + name
        out += struct.pack("<BI", 2, m["r"]) + struct.pack("<I", m["c"])
        out += struct.pack("<5B", m["k"], m["group"], m["mode"],
                           int(m.get("perm", 0)), int(m.get("lanes", 1)))
        out += struct.pack("<4I", m["tbl"], m["dig"], m["sca"], m["plen"])
    return bytes(out)


def read_lane_map(gguf_path) -> dict:
    """Read the built GGUF's directory KV and return {tensor: lanes}."""
    import gguf as _g
    rr = _g.GGUFReader(gguf_path)
    fld = None
    for key in ("ninths.directory2", "ninths.directory", "k9.directory"):
        fld = rr.get_field(key)
        if fld is not None:
            break
    if fld is None:
        return {}
    # ReaderField.parts = [offset, name, type, subtype, count, *payload-bytes]
    data = b"".join(bytes(np.asarray(part, dtype=np.uint8)) for part in fld.parts[5:])
    pos = 0
    ver, n = struct.unpack_from("<HH", data, pos); pos += 4
    lanes_map = {}
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", data, pos); pos += 2
        name = data[pos:pos + nl].decode(); pos += nl
        rank = data[pos]; pos += 1
        pos += 4 * rank
        lanes = data[pos + 4]
        pos += 5 + 16
        lanes_map[name] = lanes if ver == 2 else 1
    return lanes_map


def cmd_pass2(a):
    manifest = json.loads(Path(a.spill + ".json").read_text())["records"]
    by_name = {m["name"]: m for m in manifest}
    dir_bytes = build_directory(manifest)

    if a.mini:
        w = gguf.GGUFWriter(a.out, arch="glm4moe")
    else:
        r = gguf.GGUFReader(a.ref)
        arch = str(bytes(r.get_field("general.architecture").parts[-1]), "utf-8")
        w = gguf.GGUFWriter(a.out, arch=arch)
        for field in r.fields.values():
            if field.name.startswith("GGUF.") or field.name == "general.architecture":
                continue
            val_type = field.types[0]
            sub = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(field.name, field.contents(), val_type, sub_type=sub)
    w.add_key_value("ninths.directory2", list(np.frombuffer(dir_bytes, dtype=np.uint8)),
                    gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT8)
    w.add_key_value("ninths.materialize", "q8_0", gguf.GGUFValueType.STRING)

    t0 = time.time()
    if a.mini:
        for m in manifest:
            w.add_tensor_info(m["name"], (m["n"],), np.dtype(np.uint8), m["n"],
                              raw_dtype=gguf.GGMLQuantizationType.K9)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
        with open(a.spill + ".bin", "rb") as sp:
            for m in manifest:
                sp.seek(m["off"])
                w.write_tensor_data(np.frombuffer(sp.read(m["n"]), dtype=np.uint8))
        w.close()
        print(f"mini GGUF: {len(manifest)} K9 tensors -> {a.out} ({time.time()-t0:.0f}s)")
        return

    # full build: keep the reference's tensor order for info AND data
    n_copied = 0
    total = 0
    for t in r.tensors:
        m = by_name.get(t.name)
        if m:
            w.add_tensor_info(t.name, (m["n"],), np.dtype(np.uint8), m["n"],
                              raw_dtype=gguf.GGMLQuantizationType.K9)
            total += m["n"]
        else:
            # reader shape is ne order; the writer reverses numpy-style shapes,
            # so reverse back or 2-D copies (e.g. the router) store transposed ne
            w.add_tensor_info(t.name, tuple(int(x) for x in reversed(t.shape)),
                              t.data.dtype, t.data.nbytes)
            total += t.data.nbytes
            n_copied += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    with open(a.spill + ".bin", "rb") as sp:
        for t in r.tensors:
            m = by_name.get(t.name)
            if m:
                sp.seek(m["off"])
                w.write_tensor_data(np.frombuffer(sp.read(m["n"]), dtype=np.uint8))
            else:
                w.write_tensor_data(t.data)
    w.close()
    got = os.path.getsize(a.out)
    print(f"stock GGUF: {n_copied} copied + {len(by_name)} K9 tensors, "
          f"{got/2**30:.2f} GiB written (header+{total:,}B expected), {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d_ref = "/home/rose/glm45air_ref_f16.gguf"
    d_spill = str(HERE.parent / "results" / "glm_k9_spill")

    p1 = sub.add_parser("pass1")
    p1.add_argument("--ref", default=d_ref)
    p1.add_argument("--spill", default=d_spill)
    p1.add_argument("--only", default=None, help="regex: quantize only matching tensors")
    p1.add_argument("--k-body", type=int, default=K_BODY)
    p1.add_argument("--k-aux", type=int, default=K_AUX)
    p1.set_defaults(fn=cmd_pass1)

    p2 = sub.add_parser("pass2")
    p2.add_argument("--ref", default=d_ref)
    p2.add_argument("--spill", default=d_spill)
    p2.add_argument("--out", required=True)
    p2.add_argument("--mini", action="store_true")
    p2.set_defaults(fn=cmd_pass2)

    p3 = sub.add_parser("gate")
    p3.add_argument("--gguf", required=True)
    p3.add_argument("--ref", default=None,
                    help="f16 reference (default: the ninths GLM ref)")
    p3.add_argument("--tensor", required=True)
    p3.add_argument("--dst", default=None, choices=[None, "q8_0", "f16", "iq4_nl"])
    p3.set_defaults(fn=cmd_gate)

    p4 = sub.add_parser("materialize")
    p4.add_argument("--k9gguf", required=True)
    p4.add_argument("--ref", default=None,
                    help="f16 reference GGUF (source of true tensor shapes)")
    p4.add_argument("--out", required=True)
    p4.add_argument("--dst", default="iq4_nl", choices=["q8_0", "iq4_nl"])
    p4.add_argument("--threads", type=int, default=16)
    p4.set_defaults(fn=cmd_materialize)

    a = ap.parse_args()
    raise SystemExit(a.fn(a) or 0)




def cmd_gate(a):
    ref_path = a.ref or "/home/rose/glm45air_ref_f16.gguf"
    rdr = gguf.GGUFReader(ref_path)
    t = next(x for x in rdr.tensors if x.name == a.tensor)
    k = plan_k(t.name, t.shape)
    assert k is not None, f"{a.tensor} is not a quantized tensor"
    ne0 = int(t.shape[0])
    frame = frame_f32(t)
    assert frame.shape[1] == ne0
    rows, cols = frame.shape
    lanes = read_lane_map(a.gguf).get(a.tensor, 1)
    print(f"  lanes  : {lanes}")
    digits, scales = rtn_frame(frame, k)
    rec = encode_stream(digits, k, scales, lanes=lanes, shape=(rows, cols))
    rec["shape"] = (rows, cols)

    if a.tensor == "blk.1.attn_q.weight":     # one-time codec cross-check
        ref_rec = k9.encode_tensor(digits, k, scales, SCALE_MODE,
                                   lanes=lanes, shape=(rows, cols))
        for part in ("freq", "digits", "scales"):
            assert rec[part] == ref_rec[part], f"stream encoder != k9.encode_tensor ({part})"
        assert rec.get("lanes", 1) == ref_rec.get("lanes", 1)
        print("  stream encoder byte-identical to k9.encode_tensor", flush=True)

    ok = True
    dpath = Path("/tmp/glm_gate_digits.bin")
    pp = subprocess.run([PROBE, a.gguf, a.tensor, "digits", str(dpath)],
                        capture_output=True, text=True)
    if pp.returncode != 0:
        raise RuntimeError(f"probe digits failed: {pp.stdout} {pp.stderr}")
    got = np.fromfile(dpath, dtype=np.uint8)
    ref = k9.decode_digits(rec)
    nmis = int(np.count_nonzero(got != ref.reshape(-1)))
    print(f"  digits : {got.size} values, mismatches={nmis}")
    ok &= nmis == 0

    spath = Path("/tmp/glm_gate_scales.bin")
    pp = subprocess.run([PROBE, a.gguf, a.tensor, "scales", str(spath)],
                        capture_output=True, text=True)
    if pp.returncode != 0:
        raise RuntimeError(f"probe scales failed: {pp.stdout} {pp.stderr}")
    got_s = np.fromfile(spath, dtype=np.float32)
    rt_scales = k9.decode_scales(rec["scales"], SCALE_MODE, rows * (cols // GROUP))
    d64 = np.abs(rt_scales.astype(np.float64) - got_s.astype(np.float64))
    ulp = float(np.max(d64 / np.maximum(np.abs(rt_scales.astype(np.float64)), 1e-30) / 1.19e-7))
    print(f"  scales : max ulp diff = {ulp:.3f}")
    ok &= ulp <= 2.0

    wpath = Path("/tmp/glm_gate_weights.bin")
    dst = getattr(a, "dst", None) or "q8_0"
    pp = subprocess.run([PROBE, a.gguf, a.tensor, "weights", str(wpath)]
                        + ([dst] if dst in ("f16", "iq4_nl") else []),
                        capture_output=True, text=True)
    if pp.returncode != 0:
        raise RuntimeError(f"probe weights failed: {pp.stdout} {pp.stderr}")
    rec["shape"] = (rows, cols)
    canvas = k9.materialize(rec, dst)
    probed = np.fromfile(wpath, dtype=np.uint8)
    nmis = int(np.count_nonzero(probed != np.frombuffer(canvas, dtype=np.uint8)))
    print(f"  weights: {dst} {probed.size} bytes vs python materialize, mismatches={nmis}")
    ok &= nmis == 0
    if dst == "iq4_nl" and nmis == 0:
        blocks = np.frombuffer(canvas, dtype=np.uint8).reshape(-1, 18)
        d16v = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float64).ravel()
        Lb = np.ascontiguousarray(blocks[:, 2:])
        num = 0.0
        den = 0.0
        CH = 262144
        for a2 in range(0, rows, CH):
            e0 = min(rows, a2 + CH)
            wex = (np.repeat(rt_scales[a2 * (cols // GROUP): e0 * (cols // GROUP)], GROUP)
                   * (digits[a2 * cols: e0 * cols].astype(np.int64) - (k - 1) // 2)
                   / ((k - 1) // 2)).astype(np.float64)
            Lc = Lb[a2 * (cols // 32): e0 * (cols // 32)]
            wq = (np.repeat(d16v[a2 * (cols // 32): e0 * (cols // 32)], 32)
                  * k9.IQ4NL_VALUES[np.concatenate([Lc & 0x0F, Lc >> 4], axis=1).ravel()])
            num += float(np.abs(wq - wex).sum())
            den += float(np.abs(wex).sum())
        print(f"  iq4_nl mean rel err vs k63 grid: {100 * num / den:.2f}%")

    print(f"GATE {a.tensor}:", "PASS" if ok else "FAIL")
    return 0 if ok else 1




def parse_dir2_from_gguf(gguf_path):
    """Full directory parse (v1/v2) straight from a built GGUF's KV array."""
    import gguf as _g
    rr = _g.GGUFReader(gguf_path)
    fld = None
    ver = 1
    for key, v in (("ninths.directory2", 2), ("ninths.directory", 1),
                   ("k9.directory", 1)):
        fl = rr.get_field(key)
        if fl is not None:
            fld, ver = fl, v
            break
    if fld is None:
        return {}
    data = b"".join(bytes(np.asarray(x, dtype=np.uint8)) for x in fld.parts[5:])
    pos = 0
    if ver == 2:
        _, n = struct.unpack_from("<HH", data, pos); pos += 4
    else:
        (n,) = struct.unpack_from("<H", data, pos); pos += 2
    recs = {}
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", data, pos); pos += 2
        name = data[pos:pos + nl].decode(); pos += nl
        rank = data[pos]; pos += 1
        assert rank == 2
        dims = struct.unpack_from("<2I", data, pos); pos += 8
        if ver == 2:
            kk, group, mode, perm, lanes = (data[pos], data[pos + 1],
                                            data[pos + 2], data[pos + 3],
                                            data[pos + 4])
            pos += 5
        else:
            kk, group, mode, perm = data[pos], data[pos + 1], data[pos + 2], data[pos + 3]
            lanes = 1
            pos += 4
        tbl, dig, sca, plen = struct.unpack_from("<4I", data, pos); pos += 16
        recs[name] = dict(shape=dims, k=kk, group=group, scale_mode=mode,
                          perm=perm, lanes=lanes, tbl=tbl, dig=dig, sca=sca,
                          plen=plen)
    return recs


def cmd_materialize(a):
    """Export a stock-compatible GGUF: decode every K9 stream tensor and
    write regular GGML-typed weights (no ninths.* KV in the output)."""
    assert rans_fast.HAVE_FAST
    recs = parse_dir2_from_gguf(a.k9gguf)
    rr = gguf.GGUFReader(a.k9gguf)
    ne_map = {}
    if a.ref:
        rr_ref = gguf.GGUFReader(a.ref)
        for t in rr_ref.tensors:
            ne_map[t.name] = tuple(int(x) for x in t.shape)
    arch = str(bytes(rr.get_field("general.architecture").parts[-1]), "utf-8")
    w = gguf.GGUFWriter(a.out, arch=arch)
    for field in rr.fields.values():
        nm = field.name
        if nm.startswith("GGUF.") or nm == "general.architecture" \
                or nm.startswith("ninths.") or nm.startswith("k9."):
            continue
        val_type = field.types[0]
        sub = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(nm, field.contents(), val_type, sub_type=sub)

    def dst_for(mm):
        return "q8_0" if (a.dst == "iq4_nl" and mm["k"] >= 99) else a.dst

    per_tb = {}
    for t in sorted(rr.tensors, key=lambda x: x.name):
        m = recs.get(t.name)
        if m is None:
            per_tb[t.name] = None  # verbatim copy
            w.add_tensor_info(t.name, tuple(int(x) for x in reversed(t.shape)),
                              t.data.dtype, t.data.nbytes)
            continue
        dt = dst_for(m)
        r_, c_ = m["shape"]
        n_out = r_ * (c_ // 32) * (34 if dt == "q8_0" else 18)
        per_tb[t.name] = (m, dt, n_out)
        ne_t = ne_map.get(t.name) or (c_, r_)
        assert ne_t[0] == c_ and int(np.prod(ne_t[1:])) == r_, (t.name, ne_t, (c_, r_))
        # byte-shape = reversed(ne) with the fastest dim replaced by row
        # bytes; gguf-py converts back and reverses again -> ne_target
        bshape = tuple(int(x) for x in reversed(ne_t))[:-1] + (n_out // r_,)
        w.add_tensor_info(t.name, bshape, np.dtype(np.uint8), n_out,
                          raw_dtype=gguf.GGMLQuantizationType.IQ4_NL
                          if dt == "iq4_nl" else gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()

    t0 = time.time()
    n_q = n_c = 0
    total = 0
    for t in sorted(rr.tensors, key=lambda x: x.name):
        entry = per_tb[t.name]
        if entry is None:
            arr = np.asarray(t.data, dtype=t.data.dtype)
            w.write_tensor_data(arr)
            n_c += 1
            total += arr.nbytes
            del arr
            continue
        m, dt, n_out = entry
        r_, c_ = m["shape"]
        blob = np.asarray(t.data, dtype=np.uint8)
        rec = {"name": t.name, "shape": (r_, c_), "k": m["k"],
               "freq": np.asarray(blob[:m["tbl"]]),
               "digits": np.asarray(blob[m["tbl"]:m["tbl"] + m["dig"]]),
               "scales": np.asarray(blob[m["tbl"] + m["dig"]:m["tbl"] + m["dig"] + m["sca"]]),
               "scale_mode": m["scale_mode"], "lanes": m["lanes"]}
        digits = k9.decode_digits_par(rec, workers=a.threads)
        del rec
        ms = k9.decode_scales(np.asarray(t.data, dtype=np.uint8)[m["tbl"] + m["dig"]:m["tbl"] + m["dig"] + m["sca"]],
                              m["scale_mode"], r_ * (c_ // m["group"]))
        cnv = k9.materialize_from(digits, ms, r_, c_, m["k"], dt, as_bytes=False)
        w.write_tensor_data(cnv)
        n_q += 1
        total += cnv.nbytes
        del cnv, digits, ms
        if n_q % 120 == 0:
            print(f"  materialized {n_q} / {len(per_tb)}  {time.time()-t0:6.0f}s",
                  flush=True)
    w.close()
    got = os.path.getsize(a.out)
    print(f"stock GGUF: {n_q} materialized + {n_c} copied, "
          f"{got/2**30:.2f} GiB written (chain-end expected = header+{total:,}B), "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
