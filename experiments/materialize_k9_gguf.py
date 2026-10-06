#!/usr/bin/env python3
"""Materialize a K9 GGUF into a plain Q8_0/F16 GGUF, leaving no K9 content.

This is the "same weights, stock loader" reference arm for runtime gates
(e.g. the koboldcpp K9 port): the K9 loader decodes each sentinel blob into
its exact destination bytes on load, and this script produces exactly those
bytes with the Python reference (experiments/k9.py) whose byte layout was
verified in k9_llamacpp_gate.py (digits bit-exact, scales <=2 ulp, q8_0
canvas byte-identical to the fork's k9probe weights dump).

Non-K9 tensors (norms, biases, tokenizer-carrying reference KV/tensors) are
copied verbatim from the f16 skeleton GGUF, as in export_k9_gguf.py. The
k9.directory / k9.materialize KV keys are dropped: the output is an ordinary
GGUF that any runtime can load.

Requires the fork's gguf-py for raw_dtype values (PYTHONPATH=<llama.cpp>/gguf-py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import gguf
from gguf import GGUFValueType

sys.path.insert(0, str(Path(__file__).parent))
import k9  # noqa: E402
from export_k9_gguf import gguf_name  # noqa: E402


def pack_q8_0(digits: np.ndarray, scales: np.ndarray, rows: int, cols: int,
              group: int, H: int) -> np.ndarray:
    """Exact loader-identical q8_0 pack of one K9 tensor (k9_llamacpp_gate.py recipe)."""
    m = scales.reshape(rows, -1).astype(np.float32)                          # (r, c/group)
    d16 = (m / np.float32(H)).astype(np.float16)
    codes = (digits - H).astype(np.int8)                                     # stored frame
    per_group = group // 32
    d_bytes = np.repeat(d16, per_group, axis=1).reshape(-1).view(np.uint8).reshape(-1, 2)
    qs_bytes = codes.reshape(rows, cols // 32, 32).reshape(-1).view(np.uint8).reshape(-1, 32)
    assert d_bytes.shape[0] == qs_bytes.shape[0] == rows * (cols // 32)
    canvas = np.empty((rows * (cols // 32), 34), dtype=np.uint8)
    canvas[:, :2] = d_bytes
    canvas[:, 2:34] = qs_bytes
    return canvas.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k9", required=True, type=Path, help="input .k9 artifact")
    ap.add_argument("--ref", required=True, type=Path, help="reference GGUF from convert_hf_to_gguf.py")
    ap.add_argument("--out", required=True, type=Path, help="output plain Q8_0/F16 GGUF")
    ap.add_argument("--mode", default="q8_0", choices=["q8_0", "f16"],
                    help="materialize destination (q8_0 = the loader's default)")
    args = ap.parse_args()

    reader = gguf.GGUFReader(str(args.ref))
    arch = str(bytes(reader.get_field("general.architecture").parts[-1]), "utf-8")
    ref_shape = {t.name: tuple(int(x) for x in t.shape) for t in reader.tensors}

    dst = gguf.GGMLQuantizationType.Q8_0 if args.mode == "q8_0" else gguf.GGMLQuantizationType.F16
    payload_dtype = np.uint8

    writer = gguf.GGUFWriter(str(args.out), arch=arch)

    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type,
                             sub_type=sub_type if val_type == GGUFValueType.ARRAY else None)

    total_bytes = 0
    copy_list = []
    n_q8, n_f16, n_copy = 0, 0, 0
    with k9.K9File(str(args.k9)) as k9f:
        records = k9f.meta
        mapped = {}
        for m in records:
            gname = gguf_name(m["name"])
            if gname not in ref_shape:
                raise SystemExit(f"k9 tensor {m['name']} maps to {gname}, absent from reference")
            mapped[gname] = m

        for t in reader.tensors:
            if t.name in mapped:
                continue
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes)
            copy_list.append(t.data)
            total_bytes += t.data.nbytes
            n_copy += 1
        for m in records:
            gname = gguf_name(m["name"])
            rec = k9f.record(m["name"])
            r, c = rec["shape"]
            if (ref_shape[gname][0], ref_shape[gname][1]) != (c, r):
                raise SystemExit(f"shape mismatch for {gname}: k9 (r,c)=({r},{c}) vs "
                                 f"reference (ne0,ne1)={ref_shape[gname]}")
            if rec["perm"] is not None or args.mode == "f16":
                w = k9.decode_tensor(rec, rec["group"]).numpy().reshape(-1)
                raw = w.astype(np.float16).tobytes()   # i0-fastest stored frame == GGUF f16 order
                nbytes = len(raw)
                # byte-shape (rows, cols*2): gguf-py converts to element dims (ne0=cols)
                writer.add_tensor_info(gname, (r, c * 2), np.dtype(np.uint8), nbytes,
                                       raw_dtype=gguf.GGMLQuantizationType.F16)
                copy_list.append(np.frombuffer(raw, dtype=np.uint8))
                n_f16 += 1
            else:
                ref_digits = k9.decode_digits(rec)
                ref_sc = k9.decode_scales(rec["scales"], rec["scale_mode"], r * (c // rec["group"]))
                assert ref_sc.size == r * (c // rec["group"]), (ref_sc.size, r * (c // rec["group"]))
                raw = pack_q8_0(ref_digits, ref_sc, r, c, rec["group"], (rec["k"] - 1) // 2)
                nbytes = int(raw.size)
                # byte-shape (rows, cols/32*34): converts to element dims (ne0=cols)
                writer.add_tensor_info(gname, (r, c // 32 * 34), np.dtype(np.uint8), nbytes,
                                       raw_dtype=dst)
                copy_list.append(raw)
                n_q8 += 1
            del rec
            total_bytes += nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for item in copy_list:
        writer.write_tensor_data(item)
    writer.close()

    print(f"wrote {args.out}: {total_bytes / 2**30:.2f} GiB "
          f"({n_q8} q8_0-packed + {n_f16} f16 + {n_copy} copied from reference)")


if __name__ == "__main__":
    main()