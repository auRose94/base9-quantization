#!/usr/bin/env python3
"""Gate A — native ninths containers (stage 2): packer + numerics identity.

Cases (all assertions are bit-exact float32 equality):
  selftest   synthetic roundtrip + fp16-scale exactness + stage-1 Q8_0
             equivalence (dispatched into ninths_native._selftest)
  1p5b       every tensor of the 1.5B k63+embed99 .k9 packed to its native
             container (K9_6 body / K9_7 embed) must dequantize bitwise equal
             to the C materializer's Q8_0 blocks in the materialized GGUF —
             which makes the docs/05 ppl rows carry over unchanged
  7b-k15     tuned-7B k15 .k9 -> K9_4/K9_7: bitwise equal to the numpy Q8_0
             replica (the layout the session-24 gate proved byte-identical to
             the C decoder); pass --gguf of a freshly materialized Q8_0 GGUF
             to also check against that file arm

Exit 0 iff everything tested PASSES.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import gguf

sys.path.insert(0, str(Path(__file__).parent))
import k9  # noqa: E402
import ninths_native as nn  # noqa: E402
from export_k9_gguf import gguf_name  # noqa: E402


def q8_0_weights(raw: np.ndarray) -> np.ndarray:
    """Vectorized Q8_0 dequant of raw materialized bytes (n*34 layout)."""
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1, 34)
    scales = raw[:, :2].view("<f2").astype(np.float32).reshape(-1)
    codes = raw[:, 2:].view(np.int8).astype(np.float32)
    return (codes * scales[:, None]).reshape(-1)


def check(k9_path: Path, gguf_path: Path | None) -> bool:
    ok_all = True
    reader = gguf.GGUFReader(str(gguf_path)) if gguf_path else None
    gg_tensors = {t.name: t for t in reader.tensors} if reader else {}
    with k9.K9File(str(k9_path)) as kf:
        for m in kf.meta:
            try:
                gname = gguf_name(m["name"])
            except ValueError:
                print(f"  [{m['name']}] SKIP: no llama.cpp name mapping")
                continue
            if reader is not None and gname not in gg_tensors:
                print(f"  [{gname}] SKIP: absent from {gguf_path.name}")
                continue
            r, c = m["shape"]
            if m["group"] != nn.GROUP:
                print(f"  [{gname}] SKIP: native pins group={nn.GROUP}, record uses {m['group']}")
                continue
            if c % nn.NBLK != 0:
                print(f"  [{gname}] SKIP: native requires ne0 % {nn.NBLK} == 0, got {c}")
                continue
            rec = kf.record(m["name"])
            tname = nn.type_for_k(m["k"])
            digits = k9.decode_digits(rec).reshape(r, c)
            mvals = k9.decode_scales(rec["scales"], rec["scale_mode"], r * (c // nn.GROUP))
            blob = nn.pack(digits, m["k"], mvals)
            w_n, _, _ = nn.unpack(blob, c, r, tname)
            if reader is not None:
                w_c = q8_0_weights(gg_tensors[gname].data)
                arm = "C-materialized"
            else:
                w_c = q8_0_weights(nn.pack_q8_0(digits, m["k"], mvals))
                arm = "numpy-replica"
            del digits, rec
            if w_c.size != r * c:
                print(f"  [{gname}] FAIL: reference size {w_c.size} != {r * c}")
                ok_all = False
                continue
            eq = np.array_equal(w_n.view(np.uint32), w_c.view(np.uint32))
            nmis = 0 if eq else int(np.count_nonzero(w_n.view(np.uint32) != w_c.view(np.uint32)))
            print(f"  [{gname}] k={m['k']:2d} -> {tname}: {len(blob)/2**20:8.1f} MiB blob, "
                  f"{arm} bitwise equal={eq}{'' if eq else f' (mismatches={nmis})'}")
            ok_all &= eq
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    choices=["selftest", "1p5b", "7b-k15"])
    ap.add_argument("--k9", type=Path, default=None, help=".k9 artifact (default: case's tuned model)")
    ap.add_argument("--gguf", type=Path, default=None)
    args = ap.parse_args()

    res = Path(__file__).resolve().parent.parent / "results"

    if args.case == "selftest":
        ok = nn._selftest()
    elif args.case == "1p5b":
        ok = check(args.k9 or res / "qwen_coder_1.5b_k63_embed99.k9",
                   args.gguf or res / "base1p5b_k63_materialized_q8_0.gguf")
    else:  # 7b-k15
        ok = check(args.k9 or res / "qwen7b_tuned_k15_embed99.k9", args.gguf)

    print(f"GATE A [{args.case}]:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()