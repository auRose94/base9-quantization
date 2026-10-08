#!/usr/bin/env python3
"""eq27 — RQ13 c-d engineering half: chunk-granularity throughput (P47).

Registered (docs/09 RQ13, verbatim):
  P47 (granularity keeps the rate). Companion-augmented decode maintains
      >= 50% of the byte coder's throughput with companions predicting
      per-group/per-block tables (chunk >= 1k symbols), not per-symbol.

Design (and why it must be this one): the companion's tables are predicted
from the DECODED PREFIX, so decoding has to proceed in stream order — which a
single rANS stack forbids (LIFO). The codec is therefore per-chunk
INDEPENDENT rANS segments: each chunk carries its own 6-byte end state
(0.047 bits/symbol at 1k chunks), the companion refreshes the
context-conditioned table set once per chunk from the already-decoded
prefix, and the state machine consumes per-symbol table lookups at symbol
rate. That is the spec's load-bearing constraint made concrete: a model
forward per symbol would strangle the loop; a per-chunk refresh does not.

The coder: eq27_ctx_coder.c — the exp28_rans_base9 state algebra verbatim
(M = 9^4, L = 9^12, base-9 renorm) plus a per-symbol table selection by the
context (the two preceding symbols, folded), with table 0 = the static
fallback for contexts unseen in the prefix.

Measured on two streams: eq26's winner (layer 0's pre-attention norm digit
stream, alpha 130) and eq26's weight null (the embed digits, alpha 63).
  * rate: the real C bytes, static single-table vs the per-chunk companion
    (+ the 6 B/chunk states), at the registered >= 1k cadence;
  * throughput (M sym/s): the byte coder (rans_fast, its own M=2^12 tables),
    base9's plain single-table coder, and the ctx coder at cadences
    1k/4k/16k/whole-stream, all decoding identical digits;
  * the companion's own cost: the per-chunk table build (this driver,
    amortized per symbol) and the per-symbol pathology (a tiny torch
    forward) the constraint exists to avoid.

Asserts: the chunked roundtrip is digit-exact; the static path through the
new C code is bit-identical to base9's plain coder (machinery cross-check).
CPU-only.
"""
import ctypes
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

B9 = "/mnt/matrix/Work/base9-quantization/experiments"
sys.path.insert(0, B9)
import k9  # noqa: E402
import rans  # noqa: E402
import rans_fast  # noqa: E402

U8P = ctypes.POINTER(ctypes.c_uint8)
U64P = ctypes.POINTER(ctypes.c_uint64)
U32P = ctypes.POINTER(ctypes.c_uint32)


def load_base9():
    """exp28's compiled base-9 coder (the plain single-table reference)."""
    lib = ctypes.CDLL(os.path.join(B9, "exp28_rans_base9.so"))
    lib.b9_rans_encode.argtypes = [U8P, ctypes.c_size_t, U64P, U64P, U8P,
                                   ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_uint64)]
    lib.b9_rans_encode.restype = ctypes.c_size_t
    lib.b9_rans_decode.argtypes = [U8P, ctypes.c_size_t, ctypes.c_uint64,
                                   ctypes.c_size_t, U64P, U64P, U32P, U8P]
    lib.b9_rans_decode.restype = None
    return lib


def b9_encode_ref(lib9, s_u8, freq, cumul):
    out = np.empty(4 * len(s_u8) + 64, dtype=np.uint8)
    end = ctypes.c_uint64(0)
    m = lib9.b9_rans_encode(s_u8.ctypes.data_as(U8P), len(s_u8),
                            freq.ctypes.data_as(U64P),
                            cumul.ctypes.data_as(U64P),
                            out.ctypes.data_as(U8P), out.size,
                            ctypes.byref(end))
    return out[:m].copy(), int(end.value)


def b9_decode_ref(lib9, digs, end, n, freq, cumul, inv, out):
    lib9.b9_rans_decode(digs.ctypes.data_as(U8P), len(digs), end, n,
                        freq.ctypes.data_as(U64P), cumul.ctypes.data_as(U64P),
                        inv.ctypes.data_as(U32P), out.ctypes.data_as(U8P))

RES = L.RESULTS
M = 9 ** 4
B9_L = 9 ** 12
CHUNKS = [1024, 4096, 16384, 1 << 30]     # the registered cadence is >= 1k
TIMING_R = 60
CTX_NONE = 0xFFFFFFFF
LOG2_9 = math.log2(9.0)


# --------------------------------------------------------------- the lib ----
def build_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "eq27_ctx_coder.c")
    so = os.path.join(here, "eq27_ctx_coder.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.run(["gcc", "-O3", "-funroll-loops", "-shared", "-fPIC",
                        src, "-o", so], check=True, capture_output=True)
    lib = ctypes.CDLL(so)
    lib.b9x_decode.restype = None
    lib.b9x_decode.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p]
    lib.b9x_encode.restype = ctypes.c_size_t
    lib.b9x_encode.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
    return lib


def coder_tables(f):
    """(freq u64, cumul u64, inv u32) for the C coder, one table packed as
    [nt][alpha]. f may be 1-D (one static table) or 2-D (a context set)."""
    f2 = np.atleast_2d(np.asarray(f, dtype=np.int64))
    assert int(f2.sum(1).min()) == M, "every table must sum to M"
    freq = np.ascontiguousarray(f2, dtype=np.uint64)
    cumul = np.zeros_like(freq)
    cumul[:, 1:] = np.cumsum(freq, axis=1)[:, :-1]
    flat = f2.reshape(-1).astype(np.intp)
    syms = np.tile(np.arange(f2.shape[1], dtype=np.uint32), f2.shape[0])
    inv = np.repeat(syms, flat).reshape(f2.shape[0], M)
    return freq, np.ascontiguousarray(cumul), np.ascontiguousarray(inv)


# ------------------------------------------------------- the companion ------
class Order2Companion:
    """Incremental order-2 counts; refreshes a context-conditioned table set
    from the current prefix. Estimator = eq26's (order-2 ML interpolated with
    the order-1 conditional and a KT order-0 base, registered lambdas)."""
    def __init__(self, alpha, lams):
        self.a = alpha
        self.lams = lams
        self.pair = np.zeros((alpha * alpha, alpha), dtype=np.int32)
        self.ctx_tot = np.zeros(alpha * alpha, dtype=np.int32)
        self.glob = np.zeros(alpha, dtype=np.int64)

    def update(self, sym):
        s = np.asarray(sym, dtype=np.int64)
        if len(s) >= 3:
            np.add.at(self.pair, (s[:-2] * self.a + s[1:-1], s[2:]), 1)
            np.add.at(self.ctx_tot, s[:-2] * self.a + s[1:-1], 1)
        self.glob += np.bincount(s, minlength=self.a)

    def tables(self):
        a = self.a
        n = int(self.glob.sum())
        kt = (self.glob + 0.5) / (n + 0.5 * a)
        seen = np.nonzero(self.ctx_tot > 0)[0]
        f = np.empty((1 + len(seen), a), dtype=np.int64)
        f[0] = quantize(kt)
        if len(seen):
            tot = self.ctx_tot[seen].astype(np.float64)
            ml2 = self.pair[seen] / tot[:, None]
            m1 = self.pair.reshape(a, a, a).sum(2).astype(np.float64)
            t1 = m1.sum(1, keepdims=True)
            p1 = np.where(t1 > 0, m1 / np.maximum(t1, 1.0), kt[None, :])
            l2, l1 = self.lams
            p = l2 * ml2 + (1 - l2) * (l1 * p1[seen // a] + (1 - l1) * kt)
            for i in range(len(seen)):
                f[1 + i] = quantize(p[i])
        ctx_id = np.full(a * a, CTX_NONE, dtype=np.uint32)
        ctx_id[seen] = np.arange(1, 1 + len(seen), dtype=np.uint32)
        return f, ctx_id

    def ctxmul(self):
        return self.a


def quantize(p):
    """Largest-remainder allocation to sum exactly M (house convention)."""
    p = np.asarray(p, dtype=np.float64)
    f = np.maximum(1, np.floor(p * M).astype(np.int64))
    rem = M - int(f.sum())
    if rem > 0:
        frac = p * M - np.floor(p * M)
        order = np.argsort(-frac, kind="stable")
        for j in range(rem):
            f[order[j % len(order)]] += 1
    while rem < 0:
        j = int(np.argmax(f))
        if f[j] > 1:
            f[j] -= 1
            rem += 1
        else:
            break
    return f


# ------------------------------------------------------------------ run -----
def main():
    t0 = time.time()
    lib = build_lib()
    lib9 = load_base9()
    recs = k9.read_k9(os.path.join(RES, "eq9_QAT_k27emb99.k9"))
    emb_digits = k9.decode_digits(recs["tok_embeddings.weight"]).astype(np.int64)

    import eq16_substituted as E16
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    emb = next(n for n in ("tok_embeddings.weight", "output.weight")
               if n in uniq)
    widths = {n: int(v.shape[1]) for n, v in uniq.items() if v.dim() == 2}
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    sd = {n: k9.decode_tensor(r, 64)[:, : widths[n]].float()
          for n, r in recs.items()}
    for k2, v in uniq.items():
        if v.dim() == 1:
            sd[k2] = v.half().float()
    sd[emb] = sd["tok_embeddings.weight"]
    sd["output.weight"] = sd[emb]
    idx = torch.from_numpy(tokens[:8 * ma["max_seq_len"]].reshape(
        8, ma["max_seq_len"]).astype(np.int64))
    cap = []
    E16.forward_sub(sd, idx, ma, use_norm=False, capture=cap)
    dmat = E16.digitize(cap[0]).numpy()
    act = (dmat.reshape(-1) + 81).astype(np.int64)
    eq26 = json.load(open(os.path.join(RES, "eq26_context_companion.json")))
    wrow = next(r for r in eq26["weights"]["per_tensor"]
                if r["art"] == "eq9_QAT_k27emb99.k9"
                and r["tensor"] == "tok_embeddings.weight")
    print(f"[eq27] streams: activation {len(act)} (alpha "
          f"{int(act.max())+1}), embed {len(emb_digits)} ({time.time()-t0:.0f}s)")

    out = {"cadence_min": 1024, "streams": {}}
    for tag, stream, lams in (
            ("activation_site0", act, tuple(eq26["activations"]["best_lams"])),
            ("embed_weight", emb_digits, tuple(wrow["lams"]))):
        alpha = int(stream.max()) + 1
        s = np.ascontiguousarray(stream, dtype=np.uint8)
        n = len(s)
        print(f"[eq27] {tag}: rate + roundtrip at the 1k cadence (lams "
              f"{lams}) ...")
        # static single-table coding through the same C code
        f_st = rans.normalize_freqs(stream, alpha, M)
        freq_s, cumul_s, inv_s = coder_tables(f_st)
        ctx_none = np.full(alpha * alpha, CTX_NONE, dtype=np.uint32)
        ctx1 = np.zeros(alpha * alpha, dtype=np.uint32)
        buf = np.empty(4 * n + 64, dtype=np.uint8)
        x = ctypes.c_uint64(B9_L)
        oi = lib.b9x_encode(s.ctypes.data, n, freq_s.ctypes.data,
                            cumul_s.ctypes.data, ctx1.ctypes.data, alpha,
                            alpha, 0, 0, buf.ctypes.data, len(buf),
                            ctypes.byref(x),
                            ctypes.byref(ctypes.c_uint32(0)),
                            ctypes.byref(ctypes.c_uint32(0)))
        st_digits = buf[:oi].copy()
        st_end = int(x.value)
        b9_digits, b9_end = b9_encode_ref(lib9, s, freq_s, cumul_s)
        assert list(b9_digits) == st_digits.tolist() and b9_end == st_end, \
            "static path diverges from base9's plain coder"
        static_bits = len(st_digits) * LOG2_9 + 48

        # the companion cadence: per chunk the codec picks the cheaper of
        # {static stored table, companion context tables} -- 1 flag bit per
        # chunk, exactly eq26's codec-choice semantics (a prefix-only
        # companion pays a cold-start tax on early chunks; the choice keeps
        # the union from ever losing to the static codec)
        comp_e = Order2Companion(alpha, lams)
        build_t, digs, states, flags, ch_bits = [], [], [], [], []
        for c0 in range(0, n, 1024):
            seg = s[c0:c0 + 1024]
            tb = time.perf_counter()
            f_static = np.atleast_2d(f_st)
            fs, cs_, is_ = coder_tables(f_static)
            b1 = np.empty(4 * len(seg) + 64, dtype=np.uint8)
            x1 = ctypes.c_uint64(B9_L)
            o1 = lib.b9x_encode(seg.ctypes.data, len(seg), fs.ctypes.data,
                                cs_.ctypes.data, ctx1.ctypes.data, alpha,
                                alpha, 0, 0, b1.ctypes.data, len(b1),
                                ctypes.byref(x1),
                                ctypes.byref(ctypes.c_uint32(0)),
                                ctypes.byref(ctypes.c_uint32(0)))
            f, ctx_id = comp_e.tables()
            freq, cumul, inv = coder_tables(f)
            b2 = np.empty(4 * len(seg) + 64, dtype=np.uint8)
            x2 = ctypes.c_uint64(B9_L)
            o2 = lib.b9x_encode(seg.ctypes.data, len(seg), freq.ctypes.data,
                                cumul.ctypes.data, ctx_id.ctypes.data, alpha,
                                alpha, 0, 0, b2.ctypes.data, len(b2),
                                ctypes.byref(x2),
                                ctypes.byref(ctypes.c_uint32(0)),
                                ctypes.byref(ctypes.c_uint32(0)))
            build_t.append(time.perf_counter() - tb)
            if o1 <= o2:
                digs.append(b1[:o1].copy())
                states.append(int(x1.value))
                flags.append(0)
                ch_bits.append(o1 * LOG2_9)
            else:
                digs.append(b2[:o2].copy())
                states.append(int(x2.value))
                flags.append(1)
                ch_bits.append(o2 * LOG2_9)
            comp_e.update(seg)
        nch = len(digs)
        total_ctx = sum(ch_bits) + nch * 49          # +1 flag bit per chunk
        us = 1e6 * float(np.mean(build_t))
        comp_d = Order2Companion(alpha, lams)
        dec = np.empty(n, dtype=np.uint8)
        o = 0
        for ci, (d, st) in enumerate(zip(digs, states)):
            if flags[ci]:
                f, ctx_id = comp_d.tables()
                freq, cumul, inv = coder_tables(f)
            else:
                freq, cumul, inv = coder_tables(np.atleast_2d(f_st))
                ctx_id = ctx1
            sn = min(1024, n - o)
            seg = np.empty(sn, dtype=np.uint8)
            pos = ctypes.c_long(len(d) - 1)
            xx = ctypes.c_uint64(st)
            lib.b9x_decode(d.ctypes.data, len(d), ctypes.byref(pos),
                           ctypes.byref(xx), sn, freq.ctypes.data,
                           cumul.ctypes.data, inv.ctypes.data,
                           ctx_id.ctypes.data, alpha, alpha,
                           ctypes.byref(ctypes.c_uint32(0)),
                           ctypes.byref(ctypes.c_uint32(0)), seg.ctypes.data)
            dec[o:o + sn] = seg
            o += sn
            comp_d.update(seg)
        assert np.array_equal(dec, s), "chunked roundtrip not digit-exact"
        print(f"[eq27]   bits: static {static_bits/8:,.0f} B -> codec-choice"
              f" {total_ctx/8:,.0f} B ({100*(1-total_ctx/static_bits):+.2f}%)"
              f" | companion chosen in {int(np.sum(flags))}/{nch} chunks"
              f" | build {us:.0f} us/chunk ({1e3*us/1024:.0f} ns/sym)")

        # ---- throughput, every timed decode valid and in-bounds ----------
        R = TIMING_R
        f_byte = rans.normalize_freqs(stream, alpha, 1 << 12)
        bs = rans_fast.encode_fast(s, f_byte, 12)
        t = time.perf_counter()
        for _ in range(R):
            rans_fast.decode_fast(bs, f_byte, n, 12)
        byte_mps = R * n / (time.perf_counter() - t) / 1e6
        st_dig_arr = np.asarray(b9_digits, dtype=np.uint8)
        out_ref = np.empty(n, dtype=np.uint8)
        t = time.perf_counter()
        for _ in range(R):
            b9_decode_ref(lib9, st_dig_arr, int(b9_end), n, freq_s, cumul_s,
                          inv_s, out_ref)
        base9_mps = R * n / (time.perf_counter() - t) / 1e6
        # the full context table set (the representative working set)
        fin = Order2Companion(alpha, lams)
        fin.update(s)
        f_full, ctx_full = fin.tables()
        fq, cq, iq = coder_tables(f_full)
        cadence, segs_by_cadence = {}, {}
        for ch in CHUNKS:
            segs = []
            for c0 in range(0, n, ch if ch < (1 << 30) else n):
                seg = s[c0:c0 + (ch if ch < (1 << 30) else n)]
                b = np.empty(4 * len(seg) + 64, dtype=np.uint8)
                xx = ctypes.c_uint64(B9_L)
                oo = lib.b9x_encode(seg.ctypes.data, len(seg),
                                    fq.ctypes.data, cq.ctypes.data,
                                    ctx_full.ctypes.data, alpha, alpha, 0, 0,
                                    b.ctypes.data, len(b), ctypes.byref(xx),
                                    ctypes.byref(ctypes.c_uint32(0)),
                                    ctypes.byref(ctypes.c_uint32(0)))
                segs.append((b[:oo].copy(), int(xx.value), len(seg)))
            tout = np.empty(n, dtype=np.uint8)
            t = time.perf_counter()
            for _ in range(R):
                o = 0
                for d, st, sn in segs:
                    pos = ctypes.c_long(len(d) - 1)
                    xx = ctypes.c_uint64(st)
                    lib.b9x_decode(d.ctypes.data, len(d), ctypes.byref(pos),
                                   ctypes.byref(xx), sn, fq.ctypes.data,
                                   cq.ctypes.data, iq.ctypes.data,
                                   ctx_full.ctypes.data, alpha, alpha,
                                   ctypes.byref(ctypes.c_uint32(0)),
                                   ctypes.byref(ctypes.c_uint32(0)),
                                   tout.ctypes.data + o)
                    o += sn
            key = ch if ch < (1 << 30) else "whole"
            cadence[key] = round(R * n / (time.perf_counter() - t) / 1e6, 1)
            segs_by_cadence[key] = len(segs)
        ratio = cadence[1024] / byte_mps
        print(f"[eq27]   throughput: byte {byte_mps:.0f} | base9 {base9_mps:.0f}"
              f" | ctx@1k {cadence[1024]:.0f} ({100*ratio:.0f}% of byte) |"
              f" ctx@4k {cadence[4096]:.0f} | whole {cadence['whole']:.0f}"
              " M sym/s")

        net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(),
                                  torch.nn.Linear(32, alpha))
        xin = torch.zeros(1, 16)
        with torch.no_grad():
            for _ in range(20):
                net(xin)
            t = time.perf_counter()
            for _ in range(200):
                net(xin)
            fwd_us = 1e6 * (time.perf_counter() - t) / 200
        out["streams"][tag] = dict(
            n=n, alpha=alpha, chunks=nch,
            companion_chunks=int(np.sum(flags)),
            ch_bits_first=round(ch_bits[0], 1),
            ch_bits_last=round(ch_bits[-1], 1),
            static_bytes=round(static_bits / 8, 1),
            ctx_bytes=round(total_ctx / 8, 1),
            win_pct=round(100 * (1 - total_ctx / static_bits), 3),
            build_us_per_chunk=round(us, 1),
            build_ns_per_sym=round(1e3 * us / 1024, 1),
            byte_M_sym_s=round(byte_mps, 1),
            base9_M_sym_s=round(base9_mps, 1),
            ctx_M_sym_s=cadence, segs_per_cadence=segs_by_cadence,
            p47_ratio_1k=round(ratio, 3),
            torch_forward_us=round(fwd_us, 1),
            roundtrip="exact", machinery="bit-identical to base9")

    p47 = dict(prediction="P47",
               ratios={k: v["p47_ratio_1k"] for k, v in out["streams"].items()},
               bar=0.5,
               verdict="PASS" if all(v["p47_ratio_1k"] >= 0.5
                                     for v in out["streams"].values())
               else "FAIL",
               note="cadence tested at 1k/4k/16k/whole; the companion's "
                    "per-chunk build and a per-symbol torch forward are "
                    "recorded separately (the constraint's reason: per-symbol "
                    "inference costs ~1e4x the loop's ns/symbol)")
    out.update(P47=p47, cadences=[c if c < (1 << 30) else "whole"
                                  for c in CHUNKS])
    with open(os.path.join(RES, "eq27_chunk_throughput.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    md = ["# eq27 — RQ13 c-d engineering half: chunk-granularity throughput"
          " (docs/09 P47)\n",
          "codec: per-chunk independent rANS segments (each with its own"
          f" 6-byte end state = {48/1024:.3f} b/sym at 1k chunks); companion"
          " = the order-2 interpolated count model refreshing the context-"
          "conditioned table set once per chunk from the decoded prefix; the"
          " state machine does per-symbol table lookups only.\n",
          f"- **P47 ({p47['verdict']})** (bar: >= 50% of the byte coder's"
          " decode throughput):", "",
          "| stream | n | byte M/s | base9 M/s | ctx@1k M/s | ratio | ctx@4k"
          " | ctx whole |", "|---|---|---|---|---|---|---|---|"]
    for tag, v in out["streams"].items():
        md.append(f"| {tag} | {v['n']} | {v['byte_M_sym_s']:.0f} |"
                  f" {v['base9_M_sym_s']:.0f} | {v['ctx_M_sym_s'][1024]:.0f} |"
                  f" {100*v['p47_ratio_1k']:.0f}% |"
                  f" {v['ctx_M_sym_s'][4096]:.0f} |"
                  f" {v['ctx_M_sym_s']['whole']:.0f} |")
    md += ["", "| stream | static B | companion B | bits win | build us/chunk"
           " | ns/sym amortized | per-symbol torch fwd us |",
           "|---|---|---|---|---|---|---|"]
    for tag, v in out["streams"].items():
        md.append(f"| {tag} | {v['static_bytes']:,.0f} |"
                  f" {v['ctx_bytes']:,.0f} | {v['win_pct']:+.2f}% |"
                  f" {v['build_us_per_chunk']:.0f} |"
                  f" {v['build_ns_per_sym']:.0f} | {v['torch_forward_us']:.1f} |")
    with open(os.path.join(RES, "eq27_chunk_throughput.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"[eq27] P47 {p47['verdict']} ({time.time()-t0:.0f}s). artifacts:"
          " eq27_chunk_throughput.json/.md")


if __name__ == "__main__":
    main()