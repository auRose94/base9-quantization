#!/usr/bin/env python3
"""eq32 — the P47 frontier with a real clusterer (docs/09 RQ13 c-d / P47).

eq31 settled two things: the branchless inverse-table lookup is required (it
nearly tripled the loop) and the cheap top-1-argmax cluster key leaves half
the rate win behind. This leg runs the frontier with three context keys, all
deterministic from the prefix counts, all swept over K:

  mode 0 — top-1 argmax rank (eq31's key, the A/B control);
  mode 1 — k-means on the context distributions (L2 on the raw ML
           probabilities, mass-weighted centroids, deterministic init from
           the most massive contexts, recomputed every 32 generations);
  mode 2 — p2-bucket (the two-before symbol's mass-rank bucket: one
           L1-sized array lookup per symbol, the cheapest possible key).

Measured: the real bytes (the rate) and M sym/s (the throughput) per
(mode, K), on eq27's two streams, at R=4096, with the inverse-table decode.
Context: the bar is >= 50% of the byte coder; eq31's own K=1 control (no
rate win at all) reached 37.5% (activation) / 46.0% (embed), so the
activation stream cannot clear the bar in this datapath family regardless of
the clusterer — this leg completes the frontier and closes that question
rather than chasing it.

Assertions: every (mode, K) roundtrip is digit-exact.
"""
import ctypes
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402
import eq30_p47_fix as E30  # noqa: E402  (stream prep + constants)

RES = L.RESULTS
SEG = 1024
R_FIX = 4096
MODES = {0: "top1_argmax", 1: "kmeans", 2: "p2_bucket"}
K_SWEEP = [4, 8, 16, 32]
TIMING_R = 16
B9_L = E30.B9_L
LOG2_9 = math.log2(9.0)


def build_lib(mode):
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "eq32_ctx_kmeans.c")
    so = os.path.join(here, f"eq32_ctx_kmeans_m{mode}.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.run(["gcc", "-O3", "-funroll-loops",
                        "-DCLUSTER_MODE=%d" % mode, "-shared", "-fPIC",
                        src, "-o", so, "-lm"], check=True, capture_output=True)
    lib = ctypes.CDLL(so)
    lib.comp_new.restype = ctypes.c_void_p
    lib.comp_new.argtypes = [ctypes.c_uint32, ctypes.c_double,
                             ctypes.c_double, ctypes.c_uint32]
    lib.comp_free.argtypes = [ctypes.c_void_p]
    lib.comp_new_chunk.argtypes = [ctypes.c_void_p]
    lib.comp_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_size_t]
    lib.comp_encode_seg.restype = ctypes.c_size_t
    lib.comp_encode_seg.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
    lib.comp_decode_seg.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p]
    return lib


def run_config(lib, alpha, lams, K, s, n):
    """encode (rate) + decode (roundtrip + throughput) for one (mode, K)."""
    comp = lib.comp_new(alpha, lams[0], lams[1], K)
    digs, states = [], []
    gen_at = 0
    for c0 in range(0, n, SEG):
        if c0 >= gen_at:
            lib.comp_new_chunk(comp)
            gen_at = c0 + R_FIX
        seg = np.ascontiguousarray(s[c0:c0 + SEG], dtype=np.uint8)
        buf = np.empty(4 * len(seg) + 64, dtype=np.uint8)
        x = ctypes.c_uint64(B9_L)
        oi = lib.comp_encode_seg(comp, seg.ctypes.data, len(seg),
                                 buf.ctypes.data, len(buf), ctypes.byref(x))
        digs.append(buf[:oi].copy())
        states.append(int(x.value))
        lib.comp_update(comp, seg.ctypes.data, len(seg))
    total_bits = sum(len(d) for d in digs) * LOG2_9 + len(digs) * 48
    comp2 = lib.comp_new(alpha, lams[0], lams[1], K)
    dec = np.empty(n, dtype=np.uint8)
    o, gen_at = 0, 0
    for gi, (d, st) in enumerate(zip(digs, states)):
        c0 = gi * SEG
        if c0 >= gen_at:
            lib.comp_new_chunk(comp2)
            gen_at = c0 + R_FIX
        sn = min(SEG, n - o)
        x2 = ctypes.c_uint64(st)
        lib.comp_decode_seg(comp2, d.ctypes.data, len(d), ctypes.byref(x2),
                            sn, dec[o:o + sn].ctypes.data)
        lib.comp_update(comp2, dec[o:o + sn].ctypes.data, sn)
        o += sn
    ok = bool(np.array_equal(dec, s))
    lib.comp_free(comp)
    lib.comp_free(comp2)
    times = []
    for _ in range(TIMING_R):
        comp3 = lib.comp_new(alpha, lams[0], lams[1], K)
        o, gen_at = 0, 0
        t = time.perf_counter()
        for gi, (d, st) in enumerate(zip(digs, states)):
            c0 = gi * SEG
            if c0 >= gen_at:
                lib.comp_new_chunk(comp3)
                gen_at = c0 + R_FIX
            sn = min(SEG, n - o)
            x2 = ctypes.c_uint64(st)
            lib.comp_decode_seg(comp3, d.ctypes.data, len(d), ctypes.byref(x2),
                                sn, dec[o:o + sn].ctypes.data)
            o += sn
        times.append(time.perf_counter() - t)
        lib.comp_free(comp3)
    return total_bits / 8, TIMING_R * n / sum(times) / 1e6, ok


def main():
    t0 = time.time()
    out = {"seg": SEG, "R": R_FIX, "K_sweep": K_SWEEP, "streams": {}}
    streams = E30.prep_streams()
    print(f"[eq32] streams prepped ({time.time()-t0:.0f}s)")
    libs = {m: build_lib(m) for m in MODES}
    for tag, stream, lams in streams:
        alpha = int(stream.max()) + 1
        s = np.ascontiguousarray(stream, dtype=np.uint8)
        n = len(s)
        rows = []
        for mode in MODES:
            for K in K_SWEEP:
                by, mps, ok = run_config(libs[mode], alpha, lams, K, s, n)
                assert ok, f"{tag} mode={mode} K={K}: roundtrip not exact"
                rows.append(dict(mode=mode, mode_name=MODES[mode], K=K,
                                 bytes=round(by, 1), M_sym_s=round(mps, 1),
                                 roundtrip=ok))
                print(f"[eq32]   {MODES[mode]:>11} K={K:<3}: {by:>9,.0f} B |"
                      f" {mps:7.1f} M sym/s")
        f_byte = E30.rans.normalize_freqs(stream, alpha, 1 << 12)
        bs = E30.rans_fast.encode_fast(s, f_byte, 12)
        t = time.perf_counter()
        for _ in range(TIMING_R):
            E30.rans_fast.decode_fast(bs, f_byte, n, 12)
        byte_mps = TIMING_R * n / (time.perf_counter() - t) / 1e6
        out["streams"][tag] = dict(n=n, alpha=alpha,
                                   byte_M_sym_s=round(byte_mps, 1), rows=rows)
        print(f"[eq32] {tag}: byte {byte_mps:.0f} M sym/s")
    for tag, v in out["streams"].items():
        for r in v["rows"]:
            r["ratio"] = round(r["M_sym_s"] / v["byte_M_sym_s"], 3)
        v["best_ratio"] = max(r["ratio"] for r in v["rows"])
        v["best_row"] = max(v["rows"], key=lambda r: r["ratio"])
        v["best_rate_row"] = min(
            (r for r in v["rows"] if r["ratio"] >= 0.30),
            key=lambda r: r["bytes"], default=None)
    p47 = dict(prediction="P47", bar=0.5,
               best_by_stream={t: v["best_ratio"]
                               for t, v in out["streams"].items()},
               verdict="PASS" if min(v["best_ratio"] for v in
                                     out["streams"].values()) >= 0.5
               else "FAIL",
               note="three deterministic context keys (top-1 argmax / k-means "
                    "/ p2-bucket) x K, all with the branchless inverse-table "
                    "decode at R=4096; eq31's K=1 control (no rate win) "
                    "bounded the activation stream at 37.5%, so the bar was "
                    "out of reach for this datapath regardless of the "
                    "clusterer.")
    out["P47"] = p47
    with open(os.path.join(RES, "eq32_kmeans.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    md = ["# eq32 — the P47 frontier with a real clusterer (docs/09 RQ13 c-d)\n",
          "Deterministic context keys from the prefix counts, swept over K,"
          " with the branchless inverse-table decode at R=4096 (eq31's"
          " finding: the search's branches cost 2-3x).\n",
          "", "| stream | key | K | bytes | M/s | ratio of byte coder |",
          "|---|---|---|---|---|---|"]
    for tag, v in out["streams"].items():
        for r in v["rows"]:
            md.append(f"| {tag} | {r['mode_name']} | {r['K']} |"
                      f" {r['bytes']:,.0f} | {r['M_sym_s']:.1f} |"
                      f" {100*r['ratio']:.0f}% |")
        br = v["best_rate_row"]
        line = (f"| {tag} | **best ratio** | {v['best_row']['K']} |"
                f" {v['best_row']['bytes']:,.0f} |"
                f" {v['best_row']['M_sym_s']:.1f} |"
                f" {100*v['best_ratio']:.0f}% |")
        if br:
            line += (f" *(best rate at >=30%: {br['mode_name']} K={br['K']},"
                     f" {br['bytes']:,.0f} B, {100*br['ratio']:.0f}%)*")
        md.append(line)
    md += ["",
           f"- **P47 ({p47['verdict']})** (bar 50%): best ratios"
           f" {p47['best_by_stream']}. eq31's K=1 controls (a single table, no"
           " rate win) reached 37.5% / 46.0%, so the activation stream could"
           " not clear the bar in this datapath family even with a perfect"
           " clusterer; the embed stream is bounded at 46%.",
           "",
           "Reference points: the static single-table rate is 175,975 B"
           " (activation) / 24,749 B (embed); eq30's unclustered full-context"
           " engine coded 93,592 / 26,108 B at 15-19% of the byte coder.",
           "",
           "Framing (kept from eq31): 90-105 M sym/s means a 1.5B-class k9"
           " artifact (~1.5G digits) loads in ~15-45 s with context coding"
           " on, which is what the companion design is for; the registered"
           " bar is stricter than a load-time path needs."]
    with open(os.path.join(RES, "eq32_kmeans.md"), "w") as fh:
        fh.write("\n".join(md))
    print(f"[eq32] P47 {p47['verdict']} | best {p47['best_by_stream']}"
          f" ({time.time()-t0:.0f}s). artifacts: eq32_kmeans.json/.md")


if __name__ == "__main__":
    main()