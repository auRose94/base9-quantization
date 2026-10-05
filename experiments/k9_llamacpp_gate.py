#!/usr/bin/env python3
"""Bit-exactness gate: llama.cpp fork's C++ K9 decoder vs experiments/k9.py.

For each (gguf, tensor) case, runs the fork's k9probe binary to dump
digits / scales / materialized weights and compares them against the Python
reference. Gates:
  - digits: bit-exact (integer stream decode)
  - scales: f32 values within <= 2 ulp (exp2 implementation may differ by ulps)
  - weights (q8_0, RTN tensors): byte-identical vs a numpy replica of the
    exact pack (d = fp16(m/H), qs = digit - H); mismatches must be 0
  - weights (f16, permuted GPTQ tensors): dequantized f16 within
    ~4.9e-4 * max|W| of the Python f32 decode (fp16 rounding of values)
"""
import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import gguf

sys.path.insert(0, str(Path(__file__).parent))
import k9  # noqa: E402

PROBE = "/home/rose/Work/llama.cpp/build-cpu/bin/llama-k9probe"


def unpack_q8_0(raw: np.ndarray) -> np.ndarray:
    n = raw.size // 34
    out = np.empty(n * 32, dtype=np.float32)
    for b in range(n):
        blk = raw[b * 34:(b + 1) * 34]
        d = struct.unpack_from("<e", blk, 0)[0]
        out[b * 32:(b + 1) * 32] = np.frombuffer(blk[2:], dtype=np.int8).astype(np.float32) * np.float32(np.float16(d))
    return out


def run_probe(gguf_path: Path, tensor: str, mode: str) -> np.ndarray:
    out = Path("/tmp") / f"k9probe_{tensor.replace('.', '_')}_{mode}.bin"
    out.unlink(missing_ok=True)
    r = subprocess.run([PROBE, str(gguf_path), tensor, mode, str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"probe failed ({r.returncode}): {r.stdout} {r.stderr}")
    print(f"  probe: {r.stdout.strip()}")
    dtype = {"digits": np.uint8, "scales": np.float32}.get(mode)
    if dtype is None:  # raw weights, interpret per call site
        return np.fromfile(out, dtype=np.uint8)
    return np.fromfile(out, dtype=dtype)


def check(gguf_path: Path, tensor: str, k9path: Path, k9name: str) -> bool:
    ok = True
    with k9.K9File(str(k9path)) as k9f:
        rec = k9f.record(k9name)
        r, c = rec["shape"]
        Hn = (rec["k"] - 1) // 2
        group = rec["group"]
        print(f"[{tensor}] {k9name}: shape (r,c)=({r},{c}) k={rec['k']} mode={rec['scale_mode']}")

        # digits — bit-exact
        ref_digits = k9.decode_digits(rec)
        got_digits = run_probe(gguf_path, tensor, "digits")
        assert got_digits.dtype == np.uint8 and got_digits.size == r * c
        nd = int(np.count_nonzero(ref_digits != got_digits))
        print(f"  digits: {r*c} values, mismatches={nd}")
        ok &= nd == 0

        # scales — <= 2 ulp
        ref_sc = k9.decode_scales(rec["scales"], rec["scale_mode"], r * (c // group))
        got_sc = run_probe(gguf_path, tensor, "scales")
        assert got_sc.dtype == np.float32 and got_sc.size == r * (c // group)
        d64 = np.abs(ref_sc.astype(np.float64) - got_sc.astype(np.float64))
        denom = np.maximum(np.abs(ref_sc.astype(np.float64)), 1e-30)
        ulp_err = float(np.max(d64 / (denom * 1.19e-7)))
        print(f"  scales: max ulp diff = {ulp_err:.3f}")
        ok &= ulp_err <= 2.0

        # weights — materialized raw bytes as the loader produces them
        if rec["perm"] is None:
            # numpy replica of the exact q8_0 pack, byte-for-byte:
            # per row: two 32-blocks per 64-group, d = fp16(m/H), qs = digit - H
            m = ref_sc.reshape(r, c // group).astype(np.float32)
            d16 = (m / np.float32(Hn)).astype(np.float16)                     # (r, ng)
            codes = (ref_digits - Hn).astype(np.int8)                         # (r, c) stored frame
            d_bytes = np.repeat(d16, 2, axis=1).reshape(-1).view(np.uint8).reshape(-1, 2)
            qs_bytes = codes.reshape(r, (c // 32), 32).reshape(-1).view(np.uint8).reshape(-1, 32)
            canvas = np.empty((r * (c // 32), 34), dtype=np.uint8)
            canvas[:, :2] = d_bytes
            canvas[:, 2:34] = qs_bytes
            packed = canvas.reshape(-1)
            probed = run_probe(gguf_path, tensor, "weights")
            nb = r * (c // 32) * 34
            assert probed.size == nb, (probed.size, nb)
            nmis = int(np.count_nonzero(probed != packed))
            print(f"  weights(q8_0): byte-exact vs numpy pack, mismatches={nmis}")
            ok &= nmis == 0
        else:
            probed = run_probe(gguf_path, tensor, "weights")
            # dest F16 raw is i0-fastest with i0 = stored columns (GGUF transposed
            # layout): viewing as (r, c) gives the stored frame in original
            # column order directly
            f16 = probed.view(np.float16).astype(np.float32).reshape(r, c).reshape(-1)
            ref_w = k9.decode_tensor(rec, rec["group"]).numpy().reshape(-1)
            bound = 4.9e-4 * float(np.abs(ref_w).max())
            delta = np.abs(ref_w - f16)
            print(f"  weights(f16): max|delta|={delta.max():.3e} (bound {bound:.3e})")
            ok &= delta.max() <= bound
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--k9", required=True, type=Path)
    ap.add_argument("--tensor", required=True, action="append",
                    help="gguf-tensor-name=k9-name, e.g. blk.0.attn_q.weight=model.layers.0.self_attn.q_proj")
    args = ap.parse_args()

    all_ok = True
    for pair in args.tensor:
        tname, kname = pair.split("=", 1)
        all_ok &= check(args.gguf, tname, args.k9, kname)
    print("\nGATE:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()