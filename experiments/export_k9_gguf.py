#!/usr/bin/env python3
"""Export a .k9 artifact into a llama.cpp-loadable GGUF.

The .k9 blob areas (freq table | rANS digits | scales | perm) are re-embedded
verbatim as GGML_TYPE_K9 sentinel tensors — one variable-length payload per
weight tensor, named with the llama.cpp tensor name. Everything else
(architecture KV, tokenizer, per-layer norms/biases) is copied from a
reference GGUF produced by convert_hf_to_gguf.py. No requantization happens:
the llama.cpp loader decodes on load (exact Q8_0 by default; F16 is forced
inside the loader for act-order-permuted GPTQ tensors, whose scattered scales
cannot fit q8_0 blocks).

Requires the fork's gguf-py (pip install -e <llama.cpp>/gguf-py) so that
GGMLQuantizationType.K9 (43, block 1x1) exists.

Usage:
  export_k9_gguf.py --k9 results/qwen_coder_1.5b_k63_embed99.k9 \
                    --ref results/ref_qwen_1.5b_f16.gguf \
                    --out results/qwen_1.5b_k63.gguf
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import gguf
from gguf import GGUFValueType

sys.path.insert(0, str(Path(__file__).parent))
from k9 import K9File, SCALE_MODES  # noqa: E402

# HF module paths (k9 record names) -> llama.cpp GGUF tensor names
_LINEAR = {
    ("self_attn", "q_proj"): "attn_q",
    ("self_attn", "k_proj"): "attn_k",
    ("self_attn", "v_proj"): "attn_v",
    ("self_attn", "o_proj"): "attn_output",
    ("mlp", "gate_proj"): "ffn_gate",
    ("mlp", "up_proj"): "ffn_up",
    ("mlp", "down_proj"): "ffn_down",
}


def gguf_name(name: str) -> str:
    if name in ("__embed__", "__lm_head__"):
        return {"__embed__": "token_embd.weight", "__lm_head__": "output.weight"}[name]
    parts = name.split(".")
    # model.layers.<i>.<submod>.<proj>
    if len(parts) == 5 and parts[0] == "model" and parts[1] == "layers" and parts[2].isdigit():
        key = (parts[3], parts[4])
        if key in _LINEAR:
            return f"blk.{parts[2]}.{_LINEAR[key]}.weight"
    raise ValueError(f"unmapped k9 tensor name: {name}")


def build_directory(records) -> bytes:
    """k9.directory KV payload (little-endian; see the fork's src/k9.h):
    u16 count, then per record: u16 name_len, name, u8 rank, u32 dims[rank],
    u8 k, u8 group, u8 scale_mode, u8 perm_flag, u32 tbl/dig/sca/plen.
    The name stored is the llama.cpp GGUF tensor name (the loader keys on it)."""
    out = bytearray()
    out += struct.pack("<H", len(records))
    for m in records:
        name = gguf_name(m["name"]).encode()
        r, c = m["shape"]
        out += struct.pack("<H", len(name)) + name
        out += struct.pack("<BI", 2, r) + struct.pack("<I", c)
        out += struct.pack("<4B", m["k"], m["group"], SCALE_MODES[m["scale_mode"]],
                           1 if m["plen"] else 0)
        out += struct.pack("<4I", m["tbl"], m["dig"], m["sca"], m["plen"])
    return bytes(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-embed a .k9 artifact as a llama.cpp GUF with K9 stream tensors",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k9", required=True, type=Path, help="input .k9 artifact")
    ap.add_argument("--ref", required=True, type=Path, help="reference GGUF from convert_hf_to_gguf.py (f16)")
    ap.add_argument("--out", required=True, type=Path, help="output GGUF")
    ap.add_argument("--materialize", default="q8_0", choices=["q8_0", "f16"],
                    help="decode-on-load destination type (default: q8_0, exact for odd grids)")
    args = ap.parse_args()

    reader = gguf.GGUFReader(str(args.ref))
    arch = str(bytes(reader.get_field("general.architecture").parts[-1]), "utf-8")
    ref_shape = {t.name: tuple(int(x) for x in t.shape) for t in reader.tensors}
    ref_names = set(ref_shape)

    with K9File(str(args.k9)) as k9f:
        records = k9f.meta
        mapped = {}
        for m in records:
            gname = gguf_name(m["name"])
            if gname not in ref_names:
                raise SystemExit(f"k9 tensor {m['name']} maps to {gname}, absent from reference")
            r, c = m["shape"]
            if (ref_shape[gname][0], ref_shape[gname][1]) != (c, r):
                raise SystemExit(f"shape mismatch for {gname}: k9 (r,c)=({r},{c}) vs "
                                 f"reference (ne0,ne1)={ref_shape[gname]}")
            mapped[gname] = m
        print(f"k9 records: {len(records)}; replaced reference tensors: {len(mapped)}")

        writer = gguf.GGUFWriter(str(args.out), arch=arch)

        for field in reader.fields.values():
            # the writer manages its own copy of these keys
            if field.name.startswith("GGUF.") or field.name == "general.architecture":
                continue
            val_type = field.types[0]
            sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
            writer.add_key_value(field.name, field.contents(), val_type,
                                 sub_type=sub_type if val_type == GGUFValueType.ARRAY else None)

        dir_bytes = build_directory(records)
        # plain list: gguf-py's array packing requires an abc.Sequence
        # (ndarray does not qualify) and the directory is only a few KiB
        writer.add_key_value("k9.directory", list(np.frombuffer(dir_bytes, dtype=np.uint8)),
                             GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT8)
        writer.add_key_value("k9.materialize", args.materialize, GGUFValueType.STRING)

        total_bytes = 0
        copy_list = []      # (name, data) from the reference, in order
        for t in reader.tensors:
            if t.name in mapped:
                continue
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes)
            copy_list.append(t.data)
            total_bytes += t.data.nbytes
        for m in records:
            gname = gguf_name(m["name"])
            rec = k9f.record(m["name"])
            payload = bytes(rec["freq"]) + bytes(rec["digits"]) + bytes(rec["scales"]) \
                + (bytes(rec["perm"]) if rec["perm"] else b"")
            assert len(payload) == m["tbl"] + m["dig"] + m["sca"] + m["plen"]
            del rec
            payload = np.frombuffer(payload, dtype=np.uint8)
            writer.add_tensor_info(gname, payload.shape, payload.dtype, payload.nbytes,
                                   raw_dtype=gguf.GGMLQuantizationType.K9)
            copy_list.append(payload)
            total_bytes += payload.nbytes

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for item in copy_list:
            writer.write_tensor_data(item)
        print(f"wrote {args.out}: {total_bytes / 2**30:.2f} GiB payload")

    writer.close()


if __name__ == "__main__":
    main()