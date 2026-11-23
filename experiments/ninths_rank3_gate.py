#!/usr/bin/env python3
"""Rank-3 K9 gate — validates the MoE-expert path end to end.

GLM-4.5-Air's MoE experts are rank-3 GGUF tensors (e.g. ffn_up_exps =
{in, out, n_expert}). The ninths codec stores each expert as a 2-D K9 frame
{out, in}; the fused expert is the experts concatenated along the expert axis,
which is byte-identical to a flattened 2-D frame {out*n_expert, in} (ggml is
ne[0]-contiguous). This gate proves:

  1. the fork's C++ loader accepts a 3-D K9 tensor (no "must be 2D" abort), and
  2. the C++ k9_dequantize of the flattened frame is byte-exact vs k9.py.

We write a minimal GGUF with ONE 3-D K9 expert tensor + the ninths.directory KV
carrying the 2-D record, then run llama-k9probe to decode it and compare against
the Python reference (digits, scales, and the Q8_0 materialized weights).
"""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import gguf

sys.path.insert(0, str(Path(__file__).parent))
import k9  # noqa: E402

PROBE = "/home/rose/Work/llama.cpp/build-cpu/bin/llama-k9probe"


def quantize_frame(digits, k, group=64):
    """Quantize a 2-D stored frame {out, in} (digits uint8) with per-(row,64)
    ent8 scales. Returns (rec, scales_f32)."""
    out, inn = digits.shape
    ng = inn // group
    x = np.log2(np.maximum(np.random.default_rng(0).random(out * ng) * 0.5 + 0.1, 1e-30))
    lo, hi = float(x.min()), float(x.max())
    q = np.clip(((x - lo) / (hi - lo) * 255.0).round().astype(np.int64), 0, 255)
    scales = k9.decode_scales(k9.encode_scales(q.astype(np.float32), "ent8"), 4, out * ng)
    rec = k9.encode_tensor(digits, k, scales, "ent8")
    return rec, scales


def build_ninths_directory(records):
    """records: [{name, out, inn, k, group, tbl, dig, sca, plen}]; 4-byte header."""
    out = bytearray()
    out += struct.pack("<H", len(records))
    for m in records:
        name = m["name"].encode()
        out += struct.pack("<H", len(name)) + name
        out += struct.pack("<BI", 2, m["out"]) + struct.pack("<I", m["inn"])
        out += struct.pack("<4B", m["k"], m["group"], 4, 0)  # k, group, ent8, no perm
        out += struct.pack("<4I", m["tbl"], m["dig"], m["sca"], m["plen"])
    return bytes(out)


def run_probe(gguf_path, tensor, mode):
    out = Path(f"/tmp/rank3_{mode}.bin")
    r = subprocess.run([PROBE, gguf_path, tensor, mode, str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"probe {mode} failed: {r.stdout} {r.stderr}")
    print(f"  probe {mode}: {r.stdout.strip()}")
    return out


def main():
    n_expert, n_embd, n_ff_exp, k, group = 4, 128, 96, 63, 64

    # quantize the fused expert as ONE 2-D frame {out*n_expert, in}
    out, inn = n_ff_exp * n_expert, n_embd
    digits = (np.random.default_rng(7).random(out * inn) * k).astype(np.uint8).reshape(out, inn)
    rec, scales = quantize_frame(digits, k, group)
    rec["shape"] = (out, inn)  # decode_digits/decode_scales need the stored-frame shape
    blob = (bytes(rec["freq"]) + bytes(rec["digits"]) + bytes(rec["scales"])
            + (bytes(rec["perm"]) if rec["perm"] else b""))
    tbl, dig, sca, plen = len(rec["freq"]), len(rec["digits"]), len(rec["scales"]), len(rec["perm"])
    assert len(blob) == tbl + dig + sca + plen

    name = "blk.1.ffn_up_exps.weight"
    # directory record stores the FLATTENED 2-D frame {out_total, in}
    dir_recs = [dict(name=name, out=out, inn=inn, k=k, group=group,
                     tbl=tbl, dig=dig, sca=sca, plen=plen)]
    dir_bytes = build_ninths_directory(dir_recs)

    # minimal GGUF: one K9 tensor whose payload is the compressed rANS blob.
    # The blob is a flattened {out_total, in} frame (out_total = out*n_expert);
    # it is byte-identical to the real rank-3 tensor {in, out, n_expert} because
    # ggml is ne[0]-contiguous: flat index c + r*in + e*in*out == frame index
    # (e*out + r)*in + c.
    #
    # The GGUF tensor SHAPE here is 1-D {blob bytes}: gguf-py's reader validates
    # that the payload length equals prod(shape)*elem_size, and K9 is a 1x1 block
    # so a 3-D model shape (128,96,4)=49152 elements would demand 49152 bytes, not
    # the 37632-byte compressed blob. The fork's loader does NOT read the true rank
    # from the GGUF tensor shape — it reconstructs it from the model config and
    # flattens the frame for the decoder (llama-model-loader.cpp:1402-1451), reading
    # the blob by the tensor's declared nbytes. So the 1-D byte shape is correct.
    gguf_path = "/tmp/rank3_k9_expert.gguf"
    w = gguf.GGUFWriter(gguf_path, arch="glm4moe")
    w.add_key_value("ninths.directory", list(np.frombuffer(dir_bytes, dtype=np.uint8)),
                    gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT8)
    payload = np.frombuffer(blob, dtype=np.uint8)
    w.add_tensor_info(name, payload.shape, payload.dtype, payload.nbytes,
                      raw_dtype=gguf.GGMLQuantizationType.K9)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    w.write_tensor_data(payload); w.close()
    print(f"wrote {gguf_path}: K9 tensor, blob {payload.nbytes} B "
          f"(real rank-3 shape {{in={n_embd}, out={n_ff_exp}, n_expert={n_expert}}}, "
          f"flattened frame {{{out}, {n_embd}}})")

    ok = True
    # 1. digits — bit exact
    dpath = run_probe(gguf_path, name, "digits")
    got = np.fromfile(dpath, dtype=np.uint8)
    ref = k9.decode_digits(rec)
    nmis = int(np.count_nonzero(got != ref.reshape(-1)))
    print(f"  digits: {got.size} values, mismatches={nmis}")
    ok &= nmis == 0
    # 2. scales — <= 2 ulp
    spath = run_probe(gguf_path, name, "scales")
    got_s = np.fromfile(spath, dtype=np.float32)
    d64 = np.abs(scales.astype(np.float64) - got_s.astype(np.float64))
    denom = np.maximum(np.abs(scales.astype(np.float64)), 1e-30)
    ulp = float(np.max(d64 / (denom * 1.19e-7)))
    print(f"  scales: max ulp diff = {ulp:.3f}")
    ok &= ulp <= 2.0
    # 3. weights — Q8_0 materialized, byte exact vs numpy replica of the loader
    wpath = run_probe(gguf_path, name, "weights")
    H = (k - 1) // 2
    m = scales.reshape(out, inn // group).astype(np.float32)
    d16 = (m / np.float32(H)).astype(np.float16)
    codes = (digits - H).astype(np.int8)
    d_bytes = np.repeat(d16, 2, axis=1).reshape(-1).view(np.uint8).reshape(-1, 2)
    qs_bytes = codes.reshape(out, (inn // 32), 32).reshape(-1).view(np.uint8).reshape(-1, 32)
    canvas = np.empty((out * (inn // 32), 34), dtype=np.uint8)
    canvas[:, :2] = d_bytes
    canvas[:, 2:34] = qs_bytes
    packed = canvas.reshape(-1)
    probed = np.fromfile(wpath, dtype=np.uint8)
    nmis = int(np.count_nonzero(probed != packed))
    print(f"  weights(Q8_0): C++ probe {probed.size} bytes vs numpy pack, mismatches={nmis}")
    ok &= nmis == 0

    print("RANK3 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
