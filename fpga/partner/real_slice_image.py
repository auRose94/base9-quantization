#!/usr/bin/env python3
"""Build the REAL-TENSOR Gate-0 image: K9_6 superblocks sliced from a real
served model container (the 1.5B coder, k63+embed99 .k9 artifact), crossed
with a real activation vector, packed in the exact 472-byte record shape the
gate0sd_top FPGA already consumes.

This is the "real tensors" rung (fpga/README next-rungs): the card payload
stops being rng-seeded and becomes actual model arithmetic —

  record i = one 256-weight superblock of  out_proj(L).W  row R
             (K9_6 bytes packed by ninths_native.pack, the Gate-A blob)
             x Q8_0(h[256j : 256j+256])                    (8 x 34 B blocks)

  cases    = chains of records whose cumulative fp32 sumf after each
             superblock is the exp word the FPGA compares against.

Activation sourcing, in increasing realness:
  --act embd:TOK   h = token_embd row TOK (f32, runtime-materialization chain
                   fp16(m/H) x codes, decoded from this .k9)
  --h-file f.npy   h = a real forward-pass hidden state (e.g. the fork's
                   llama-embedding last-token state of a served prompt)
The WEIGHT digits are always 100% real model container data either way.

Closure, all bit-exact before anything is written: numpy golden
(sim/golden.k96_partials) AND the verbatim fork C kernel
(crosscheck/k9_c_harness) must agree on every cumulative state; the cocotb
full-top sim (tb_gate0sd with NIN_SD_IMG/NIN_SD_NPZ overrides) then closes
RTL over the same 21 records.

Outputs:
  partner/ninths_sd_real.bin        the 20-LBA card payload (same layout)
  ../synth/pay_real.hex             embedded-payload hex for the real top
  /tmp/gate0_real_vecs.npz          rec{i}_wblk/act/gold/case vectors
"""
import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO / "fpga" / "sim"))
import k9                            # noqa: E402  (container decoder)
import ninths_native as nn           # noqa: E402  (Gate-A block packer)
from golden import b32f, f32b, k96_partials, k97_partials  # noqa: E402

K9_PATH = REPO / "results" / "qwen_coder_1.5b_k63_embed99.k9"
SB_TYPE = {"k9_6": 6, "k9_7": 7}
HARNESS = REPO / "fpga" / "crosscheck" / "k9_c_harness"
N_REC = 21
OUT_IMG = HERE / "ninths_sd_real.bin"
OUT_HEX = REPO / "fpga" / "synth" / "pay_real.hex"
OUT_HEX2 = HERE / "pay_real.hex"   # sim cwd copy (real_top's $readmemh resolves here)
OUT_NPZ = Path("/tmp/gate0_real_vecs.npz")

# Each chain = one real fp32 dot product (row * h) of the real model.
PLAN_K9_6 = [("l0", "model.layers.0.self_attn.o_proj", 0, 6),
             ("l1", "model.layers.1.self_attn.o_proj", 0, 6),
             ("l2", "model.layers.2.self_attn.o_proj", 0, 6),
             ("l0b", "model.layers.0.self_attn.o_proj", 1, 3)]
# k9_7 records: the 1.5B is tied-embed, so __embed__ rows ARE the unembed;
# each chain = one real logit-class dot product row(0|1) · h.
PLAN_K9_7 = [("u0", "__embed__", 0, 6),
             ("u1", "__embed__", 1, 6)]
assert sum(n for *_, n in PLAN_K9_6) == N_REC


def q8_0_blocks(x):
    """Exact ggml Q8_0 per 32-block: fp16(float32 amax/127) LE then 32
    codes, rounded half-away-from-zero (roundf) using the float32 scale --
    the runtime's own recipe."""
    out = []
    for i in range(0, len(x), 32):
        b = np.asarray(x[i:i + 32], dtype=np.float32)
        amax = float(np.max(np.abs(b)))
        d = np.float32(amax / 127.0)
        if d == 0:
            q = np.zeros(32, dtype=np.int8)
        else:
            q = np.floor(np.abs(b) / d + 0.5) * np.sign(b)   # roundf = half away
            q = np.clip(q, -127, 127).astype(np.int8)
        out.append(struct.pack("<e", np.float16(d)) + q.tobytes())
    return out


def decode_record(k9f, name, cache, kind="k9_6"):
    """Decode one .k9 tensor: (digits (r,c), scales f32 (r, c//64), k)."""
    if name in cache:
        return cache[name]
    rec = k9f.record(name)
    cap = nn.MAX_K["k9_6"] if kind == "k9_6" else nn.MAX_K["k9_7"]
    assert rec["k"] <= cap, f"{name}: k={rec['k']} > {kind}"
    assert rec["perm"] is None, f"{name}: permuted (GPTQ act-order) container"
    r, c = rec["shape"]
    g = rec["group"]
    assert g == nn.GROUP and c % 256 == 0 and c % g == 0, (name, g, c)
    dig = k9.decode_digits(rec)
    sc = k9.decode_scales(rec["scales"], rec["scale_mode"], r * (c // g))
    cache[name] = (dig.reshape(r, c), sc.reshape(r, c // g), rec["k"])
    return cache[name]


def embd_row(k9f, tok):
    """Runtime-chain dequant of one embed row: s = fp16(m/H), w = s * (d-H),
    exactly the loader's materialization arithmetic (k9_llamacpp_gate recipe)."""
    rec = k9f.record("__embed__")
    assert rec["k"] <= nn.MAX_K["k9_7"], rec["k"]
    r, c = rec["shape"]          # (vocab, n_embd), row-major stored
    g = rec["group"]
    H = (rec["k"] - 1) // 2
    d = k9.decode_digits(rec).reshape(r, c).astype(np.int16)[tok] - H
    sc = k9.decode_scales(rec["scales"], rec["scale_mode"], r * (c // g)).reshape(r, c // g)[tok]
    s = np.repeat(np.float32(np.float16(sc / np.float32(H))), g)
    return (s * d.astype(np.float32)).astype(np.float32)


def build_records(k9f, act_h, plan, kind):
    """PLAN case ids are strings; the npz/image protocol wants integer case ids
    (the tb/crosscheck convention), names keep living in the meta string."""
    recs, table = [], []
    cache = {}
    for pi, (case_id, name, row, n_sb) in enumerate(plan):
        dig, sc, k = decode_record(k9f, name, cache, kind)
        nb = nn.BLOCK_BYTES[nn.type_for_k(k)]
        blob = nn.pack(dig[row:row + 1], k, sc[row:row + 1]).reshape(-1)
        for j in range(n_sb):
            a0, a1 = 256 * j, 256 * (j + 1)
            acts = q8_0_blocks(act_h[a0:a1])
            recs.append({"case": pi, "wblk": blob[nb * j: nb * (j + 1)], "act": acts})
            table.append((len(recs) - 1, pi, name, row, j))
    return recs, table


def gold_and_crosscheck(recs, kind="k9_6"):
    """Three-level bit-exact closure (partials replica + harness type are
    kind-aware: k9_6 = 200 B/type-6, k9_7 = 264 B/type-7 kernels).

    1. per-CALL: numpy singleton partial (k96_partials over ONE superblock =
       the fork kernel's exact per-call semantics) must equal the C harness
       fed the same singleton (N=1 chain).
    2. row-tree: numpy's row-kernel tree (k96_partials over the whole chain,
       the C kernel's own per-group accumulation) must equal the C harness on
       the same N-superblock chain -- validates both replicas of the row
       kernel on real data. NOTE: this tree is 1-ulp-class different from the
       gate's chain below on real magnitudes (proven by the record dump).
    3. gate tree (the image's exp): each record's partial chained left-to-
       right with running fp32 adds, case-reset at flags -- the exact
       sequence the FPGA tops run. Deterministic IEEE: identical operands in
       identical order, so the board's compare holds by construction, not
       luck (the synthetic gate's old C-tree exps passed 21/21 only because
       18/21 records were singleton chains).
    """
    by_case = {}
    for idx, r in enumerate(recs):
        by_case.setdefault(r["case"], []).append(idx)
    cums_all = [None] * len(recs)
    for cid, idxs in sorted(by_case.items()):
        acc = np.float32(0.0)
        for k_, i in enumerate(idxs):
            wblk = bytes(recs[i]["wblk"])
            acts = recs[i]["act"]
            # level 1: python singleton == C singleton
            one_np, _ = _replica(kind)([wblk], acts)
            inb = (SB_TYPE[kind]).to_bytes(4, "little") + (1).to_bytes(4, "little") \
                + wblk + b"".join(acts)
            fin, fout = HERE / f"_real_single_{i}.bin", HERE / f"_real_single_out_{i}.bin"
            fin.write_bytes(inb)
            subprocess.run([str(HARNESS), str(fin), str(fout)], check=True)
            one_c = int(np.frombuffer(fout.read_bytes(), dtype="<u4")[0])
            fin.unlink(); fout.unlink()
            assert one_c == int(one_np[0]), f"rec {i}: numpy singleton != C singleton"
            # level 3: the gate's chain tree -- convert BIT PATTERNS to fp32
            # values before adding (b32f); adding the raw int = the bug that
            # produced ~3e9 garbage exps in the first real run.
            acc = np.float32(acc + b32f(int(one_np[0])))
            cums_all[i] = int(f32b(acc))
        # level 2: row kernel over the whole chain, python vs verbatim C
        wblks = [bytes(recs[i]["wblk"]) for i in idxs]
        acts = [b for i in idxs for b in recs[i]["act"]]
        one_r = _replica(kind)
        cums_row, final = one_r(wblks, acts)
        inb = (SB_TYPE[kind]).to_bytes(4, "little") + len(idxs).to_bytes(4, "little")
        inb += b"".join(wblks) + b"".join(acts)
        fin, fout = HERE / f"_real_row{cid}.bin", HERE / f"_real_row_out{cid}.bin"
        fin.write_bytes(inb)
        subprocess.run([str(HARNESS), str(fin), str(fout)], check=True)
        cums_c = np.frombuffer(fout.read_bytes(), dtype="<u4")
        fin.unlink(); fout.unlink()
        assert np.array_equal(cums_c, np.array(cums_row, dtype=np.uint32)), \
            f"case {cid}: numpy row-tree != C row-tree"
        val = np.frombuffer(struct.pack("<I", final), "<f4")[0]
        print(f"  case {cid}: singleton+row closures bit-exact, "
              f"row-final {final:#010x} ({val:+.6f})")
    return cums_all


def _replica(kind):
    return k97_partials if kind == "k9_7" else k96_partials


def assemble_image(recs, cums):
    seen, flags = set(), []
    for r in recs:
        first = r["case"] not in seen
        seen.add(r["case"])
        flags.append(1 if first else 0)
    buf = bytearray()
    for r in recs:
        blob = bytes(r["wblk"]) + b"".join(r["act"])
        assert len(blob) == 472, len(blob)
        buf += blob
    assert len(buf) == 2478 * 4
    for w in cums:
        buf += struct.pack("<I", w & 0xFFFFFFFF)
    for f in flags:
        buf += struct.pack("<I", int(f))
    assert len(buf) == 10080, len(buf)
    return bytes(buf) + bytes(20 * 512 - len(buf)), flags


NPZ_PATHS = {"k9_6": Path("/tmp/gate0_real_vecs.npz"), "k9_7": Path("/tmp/gate0_k97_real_vecs.npz")}


def write_outputs(payload, recs, cums, table, act_desc, kind):
    if kind == "k9_7":
        # k97 has no card image/top yet: npz only (sim tb_k97 consumes it)
        npz = NPZ_PATHS[kind]
        d = {"rec_meta": np.frombuffer(
            "; ".join(f"rec{i}:case{c}:{name}:row{row}+sb{j}"
                      for i, (_, c, name, row, j) in enumerate(table)).encode(), np.uint8)}
        for i, r in enumerate(recs):
            d[f"rec{i}_wblk"] = np.asarray(r["wblk"], np.uint8)
            d[f"rec{i}_act"] = np.frombuffer(b"".join(r["act"]), np.uint8)
            d[f"rec{i}_gold"] = cums[i]
            d[f"rec{i}_case"] = r["case"]
        np.savez(npz, **d)
        print(f"wrote {npz} ({len(recs)} records)")
        return
    OUT_IMG.write_bytes(payload)
    OUT_IMG.write_bytes(payload)
    words_txt = "\n".join(
        "%08X" % int.from_bytes(payload[4 * j:4 * j + 4], "little")
        for j in range(2560)) + "\n"
    OUT_HEX.write_text(words_txt)
    OUT_HEX2.write_text(words_txt)
    meta = "; ".join(f"rec{i}:case{c}:{name}:row{row}+sb{j}"
                     for i, (_, c, name, row, j) in enumerate(table))
    d = {"rec_meta": np.frombuffer(meta.encode(), np.uint8)}
    for i, r in enumerate(recs):
        d[f"rec{i}_wblk"] = np.asarray(r["wblk"], np.uint8)
        d[f"rec{i}_act"] = np.frombuffer(b"".join(r["act"]), np.uint8)
        d[f"rec{i}_gold"] = cums[i]
        d[f"rec{i}_case"] = r["case"]
    np.savez(NPZ_PATHS[kind], **d)
    print(f"wrote {OUT_IMG.name} ({len(payload)} B), {OUT_HEX.name}, {NPZ_PATHS[kind]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k9", type=Path, default=K9_PATH)
    ap.add_argument("--kind", choices=["k9_6", "k9_7"], default="k9_6",
                    help="container kind: k9_6 = the 200-B gate image "
                         "(o_proj plan), k9_7 = 264-B npz only for sim "
                         "(__embed__ row plan: the tied unembed)")
    ap.add_argument("--act", default="embd:2048",
                    help="h source: 'embd:TOK' = the __embed__ record's row "
                         "TOK decoded from --k9 (runtime fp16-scale chain), or "
                         "'file:PATH' = real f32 hidden state (.npy, 1536)")
    args = ap.parse_args()

    spec = args.act.split(":", 1)
    if spec[0] == "file":
        h = np.load(spec[1]).astype(np.float32)
        act_desc = f"hidden state from {spec[1]}"
    else:
        assert spec[0] == "embd" and spec[1].lstrip("-").isdigit(), args.act
        h = None

    plan = PLAN_K9_6 if args.kind == "k9_6" else PLAN_K9_7
    n_rec_plan = sum(n for *_, n in plan)
    with k9.K9File(str(args.k9)) as k9f:
        if h is None:
            tok = int(spec[1])
            h = embd_row(k9f, tok)
            act_desc = f"embd row (runtime chain) of token {tok}"
        assert h.shape == (1536,), h.shape
        recs, table = build_records(k9f, h, plan, args.kind)
        cums = gold_and_crosscheck(recs, args.kind)
        if args.kind == "k9_6":
            payload, flags = assemble_image(recs, cums)
            write_outputs(payload, recs, cums, table, act_desc, args.kind)
        else:
            write_outputs(None, recs, cums, table, act_desc, args.kind)

    print("\nmodel container:", args.k9.name)
    print("activation h:", act_desc, "(1536-dim, Q8_0 per 32-block)")
    print(f"plan ({args.kind}; each chain = one real fp32 dot product, "
          f"{n_rec_plan} records):")
    for case, name, row, n_sb in plan:
        print(f"  case '{case}': {name} row {row} x {n_sb} superblocks")
    print("\nFPGA expectation: boot -> card path -> run -> 'G0 PASS' + 5 lit LEDs")
    print("(led[5] + all four index LEDs lit, led[4] dark -- same telegraph as "
          "the synthetic gate)")


if __name__ == "__main__":
    main()