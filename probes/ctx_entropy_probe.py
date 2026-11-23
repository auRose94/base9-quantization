#!/usr/bin/env python3
"""ctx_entropy_probe — how much rate would a CONTEXT model save over the
order-0 rANS the K9 container pays today?

The container entropy-codes two symbol streams per tensor with static
per-tensor tables (docs/12 §3-4.2): digits (one per weight) and ent8 scale
levels (one per 64-weight group). Order-0 rANS pays the marginal entropy
H(X); a decoder that conditions on what it already holds pays H(X | ctx).
The gap is the whole lever — the coder itself is already at entropy
(exp21: +0.00%). exp12 closed digit *adjacency* on a QAT 33M model
(0.0-0.2%). This measures the cases never tested, on deployed PTQ
containers, CPU only, read-only:

  digits  D0    H(d)                            paid today
          D1    H(d | previous digit in row)    exp12's test, deployed container
          D2    H(d | own group's scale code)   scales are a separate stream, so
                the decoder holds them before the digits (16 and 256 contexts)
          D3    H(d | own scale bucket, row-mean scale bucket)
  scales  S0    H(q)                            paid today
          S*    residual q - predictor for {left, top, avg(l,t), avg(l,t,tl)},
                clamped to +-RES with one escape symbol (+ raw 8-bit code) — a
                delta stream an order-0 coder carries as-is
          Sx    per-context tables, ctx = (left>>6, top>>6), 25 x 256

Every conditional table is fitted on even rows and scored on odd rows, so
the reported bits are what a static-table coder would actually pay.

Pre-registered before the first run: D1 gain < 1% of digit bits; D2 gain in
[0.5%, 5%]; best scale context saves >= 25% of scale bits.

Usage:
  ctx_entropy_probe.py CONTAINER [--every N] [--max-rows R] [--include RE]
CONTAINER is a .k9 file or a GGUF carrying `ninths.directory2` (v2 lanes).
Out: results/ctx_entropy_probe.csv (+ a table on stdout).
"""
import argparse
import csv
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import k9  # noqa: E402  (normative codec)
from k9 import SCALE_BITS, _SCALE_HDR, _dec, _mode_id  # noqa: E402

CHUNK_ELEMS = 1 << 22                  # digits per histogram chunk (bounds temp arrays)
MAX_ELEMS = 1 << 26                    # digits probed per tensor
RES = 31                               # scale residual clamp; 64-symbol alphabet
SCALE_FIXED = _SCALE_HDR + 2 * 256     # ent8 header + rANS table, per tensor
MODE_NAMES = {0: "fp32", 1: "fp16", 4: "ent8"}
GGUF_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
               7: "?", 10: "Q", 11: "q", 12: "d"}
S_PRED = ("left", "top", "avg_lt", "avg_ltt", "ctx_lt")
S_KEYS = tuple("S_" + p for p in S_PRED)


# ---------------------------------------------------------------- entropy ----
def h_of(counts):
    c = np.asarray(counts, dtype=np.float64).ravel()
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def xent_of(fit_counts, test_counts):
    """Cross-entropy (bits/symbol) of `test_counts` under the table built
    from `fit_counts` (+0.5 smoothing)."""
    p = np.asarray(fit_counts, dtype=np.float64).ravel() + 0.5
    p /= p.sum()
    t = np.asarray(test_counts, dtype=np.float64).ravel()
    n = t.sum()
    if n <= 0:
        return 0.0
    nz = t > 0
    return float(-(t[nz] * np.log2(p[nz])).sum() / n)


# ----------------------------------------------------------------- readers ----
def records_k9(path):
    """Yield (name, rec) for a standalone .k9 container (normative reader)."""
    with k9.K9File(path) as kf:
        for name in kf.names():
            rec = kf.record(name)
            rec["perm_flag"] = rec["perm"] is not None
            yield name, rec


def records_gguf(path):
    """Yield (name, rec) for a GGUF whose K9 blobs are described by
    `ninths.directory2`. Minimal v3 header parser: only KVs and tensor
    offsets; no tensor data other than the K9 blobs is touched."""
    with open(path, "rb") as fh:
        magic, version, n_t, n_kv = struct.unpack("<4sIQQ", fh.read(24))
        assert magic == b"GGUF" and version == 3, "not a GGUF v3 file"

        def rd(fmt):
            return struct.unpack("<" + fmt, fh.read(struct.calcsize("<" + fmt)))[0]

        def skip(t, count=1):
            if t == 8:                      # strings are variable length: one by one
                for _ in range(count):
                    fh.seek(rd("Q"), 1)
            elif t == 9:
                skip(rd("I"), rd("Q"))
            else:
                fh.seek(count * struct.calcsize("<" + GGUF_SCALAR[t]), 1)

        def value(t):
            if t == 8:
                return fh.read(rd("Q")).decode("utf-8")
            if t == 9:
                et, n = rd("I"), rd("Q")
                raw = fh.read(n * struct.calcsize("<" + GGUF_SCALAR[et]))
                return np.frombuffer(raw, dtype="<" + GGUF_SCALAR[et])
            return rd(GGUF_SCALAR[t])

        align, directory = 32, None
        for _ in range(n_kv):
            key = value(8)
            t = rd("I")
            if key == "general.alignment":
                align = value(t)
            elif key in ("ninths.directory2", "k9.directory2"):
                directory = bytes(value(t))       # bytes, not a u8 ndarray: the
                # record walk below indexes it, and numpy scalars would wrap pos
            else:
                skip(t)
        assert directory is not None, "no ninths.directory2 in this GGUF"
        tensors = {}
        for _ in range(n_t):
            name = value(8)
            n_dims = rd("I")
            dims = struct.unpack("<%dQ" % n_dims, fh.read(8 * n_dims))
            tensors[name] = (dims, rd("I"), rd("Q"))
        data_start = (fh.tell() + align - 1) // align * align

        ver, n = struct.unpack_from("<HH", directory, 0)
        assert ver == 2, "directory2 version %d" % ver
        pos = 4
        for _ in range(n):
            (nl,) = struct.unpack_from("<H", directory, pos)
            pos += 2
            name = bytes(directory[pos:pos + nl]).decode()
            pos += nl
            rank = directory[pos]
            pos += 1
            dims = struct.unpack_from("<%dI" % rank, directory, pos)
            pos += 4 * rank
            k, group, smode, pflag, lanes = struct.unpack_from("<BBBBB", directory, pos)
            pos += 5
            tbl, dig, sca, plen = struct.unpack_from("<IIII", directory, pos)
            pos += 16
            gdims, gtype, goff = tensors[name]
            blob_len = tbl + dig + sca + plen          # a K9 tensor is a 1-D blob
            assert gtype == 43 and int(np.prod(gdims)) == blob_len, \
                "%s: GGUF tensor %d bytes, directory says %d" % (name, int(np.prod(gdims)), blob_len)
            rows, cols = int(np.prod(dims[:-1])), dims[-1]
            fh.seek(data_start + goff)
            blob = fh.read(blob_len)
            yield name, dict(shape=(rows, cols), k=k, group=group,
                             scale_mode=MODE_NAMES[smode], perm_flag=bool(pflag),
                             lanes=lanes, freq=blob[:tbl],
                             digits=blob[tbl:tbl + dig],
                             scales=blob[tbl + dig:tbl + dig + sca],
                             perm=blob[tbl + dig + sca:] if plen else None)


# ------------------------------------------------------------------ streams ----
def digit_prefix(rec, rows):
    """First `rows` rows of the digit canvas, stored (row-major) layout."""
    f = np.frombuffer(rec["freq"], dtype="<u2").astype(np.int64)
    r, c = rec["shape"]
    L = int(rec.get("lanes", 1) or 1)
    n = rows * c
    if L <= 1:
        out = _dec(rec["digits"], f, n, SCALE_BITS)
    else:                                  # v2: lane i owns rows [i*rpl, (i+1)*rpl)
        buf = rec["digits"]
        otab = 4 * (L + 1)
        offs = np.frombuffer(buf[:otab], dtype="<u4")
        rpl = (r + L - 1) // L
        out = np.empty(n, dtype=np.uint8)
        for li in range((rows + rpl - 1) // rpl):
            a, b = li * rpl, min(rows, (li + 1) * rpl)
            seg = buf[otab + int(offs[li]): otab + int(offs[li + 1])]
            out[a * c: b * c] = _dec(seg, f, (b - a) * c, SCALE_BITS)
    d = np.asarray(out).astype(np.uint8).reshape(rows, c)
    assert d.max() < rec["k"], "digit outside [0, k)"
    return d


def scale_codes(rec, n):
    """ent8 level indices exactly as stored (q in 2**(lo+(hi-lo)*q/255))."""
    if _mode_id(rec["scale_mode"]) != 4:
        return None
    buf = rec["scales"]
    lo, hi, k = struct.unpack_from("<ffH", buf, 0)
    f = np.frombuffer(buf, dtype="<u2", count=k, offset=_SCALE_HDR).astype(np.int64)
    q = np.asarray(_dec(buf[_SCALE_HDR + 2 * k:], f, n, SCALE_BITS)).astype(np.int64)
    assert q.size == n and q.min() >= 0 and q.max() <= 255, "scale codes out of range"
    ref = k9.decode_scales(buf, rec["scale_mode"], n).astype(np.float64)
    assert np.array_equal(ref, (2.0 ** (lo + (hi - lo) * q / 255.0)).astype(np.float32)), \
        "scale codes != decode_scales"
    return q


def digit_stats(d, qg, k):
    """Bits/digit for each digit context. d: (rows, c) stored-layout digits;
    qg: (rows, c//64) scale codes of the same rows, or None. Tables are fitted
    on even rows and scored on odd rows."""
    rows, c = d.shape
    step = max(2, (CHUNK_ELEMS // c) // 2 * 2)      # even, so row parity is global
    cd = np.zeros(k, np.int64)
    j1, cprev = np.zeros(k * k, np.int64), np.zeros(k, np.int64)
    nc = 256 if qg is not None else 0
    z = lambda n: np.zeros(n * k, np.int64)
    fit = [z(nc), z(16), z(256)]          # D2 (own code), D2_16, D3 (16 x 16 contexts)
    test = [z(nc), z(16), z(256)]
    for a in range(0, rows, step):
        m = d[a:a + step]
        cd += np.bincount(m.ravel(), minlength=k)
        p64 = m[:, :-1].astype(np.int64)
        j1 += np.bincount((p64 * k + m[:, 1:]).ravel(), minlength=k * k)
        cprev += np.bincount(p64.ravel(), minlength=k)
        if qg is None:
            continue
        for par in (0, 1):                          # 0 = even rows (fit), 1 = odd (test)
            ms, g = m[par::2], qg[a:a + step][par::2].astype(np.int64)
            ctx = np.repeat(g, 64, axis=1)         # own group's scale code, per digit
            b16 = ctx >> 4
            rmb = (g.mean(axis=1) / 16).astype(np.int64)          # 16 row-level buckets
            ctx3 = b16 * 16 + np.repeat(rmb[:, None], c, axis=1)
            tgt = fit if par == 0 else test       # even rows fit, odd rows score
            tgt[0] += np.bincount((ctx * k + ms).ravel(), minlength=nc * k)
            tgt[1] += np.bincount((b16 * k + ms).ravel(), minlength=16 * k)
            tgt[2] += np.bincount((ctx3 * k + ms).ravel(), minlength=256 * k)
    d1 = h_of(j1) - h_of(cprev)
    if qg is None:
        return h_of(cd), d1, float("nan"), float("nan"), float("nan")

    def cond(j, nctx):                        # xent(joint) - xent(ctx), even-row tables
        jf, jt = j[0].reshape(nctx, k), j[1].reshape(nctx, k)
        return xent_of(jf, jt) - xent_of(jf.sum(1), jt.sum(1))
    d2 = cond([fit[0], test[0]], nc)
    d2_16 = cond([fit[1], test[1]], 16)
    d3 = cond([fit[2], test[2]], 256)
    return h_of(cd), d1, d2_16, d2, d3


def scale_stats(q):
    """q: (R, G) scale codes, raster order. Returns S0 and {name: bits}."""
    R = q.shape[0]
    left = np.full_like(q, -1); left[:, 1:] = q[:, :-1]
    top = np.full_like(q, -1); top[1:, :] = q[:-1, :]
    tl = np.full_like(q, -1); tl[1:, 1:] = q[:-1, :-1]
    even = (np.arange(R) % 2) == 0
    out = {}

    def resid(pred):
        ok = pred >= 0
        e = np.where(ok, q - pred, 0)
        far = ok & (np.abs(e) > RES)
        sym = np.where(ok & ~far, e + RES + 1, 0).astype(np.int64)   # 0 = escape
        f = np.bincount(sym[even].ravel(), minlength=2 * RES + 2)
        t = np.bincount(sym[~even].ravel(), minlength=2 * RES + 2)
        return xent_of(f, t) + 8.0 * float(far[~even].mean())

    def avg(*nb):
        s, n = np.zeros(q.shape, np.int64), np.zeros(q.shape, np.int64)
        for x in nb:
            ok = x >= 0
            s += np.where(ok, x, 0)
            n += ok
        safe = np.maximum(n, 1)
        return np.where(n > 0, (s + n // 2) // safe, -1)

    out["left"] = resid(left)
    out["top"] = resid(top)
    out["avg_lt"] = resid(avg(left, top))
    out["avg_ltt"] = resid(avg(left, top, tl))

    def cb(x):
        return np.where(x >= 0, x >> 6, 4)
    ctx = cb(left) * 5 + cb(top)                       # 25 contexts x 256 levels
    f = np.bincount((ctx[even] * 256 + q[even]).ravel(), minlength=25 * 256)
    t = np.bincount((ctx[~even] * 256 + q[~even]).ravel(), minlength=25 * 256)
    out["ctx_lt"] = xent_of(f, t)
    return h_of(np.bincount(q.ravel(), minlength=256)), out


# -------------------------------------------------------------------- probe ----
def probe(name, rec, max_rows):
    r, c = rec["shape"]
    k, g = rec["k"], rec["group"]
    rows = min(r, max_rows, max(2, MAX_ELEMS // c))     # >= 2 rows: the top neighbour
    n_s = rows * (c // g)
    q = scale_codes(rec, n_s)
    d = digit_prefix(rec, rows)

    permuted = bool(rec.get("perm_flag"))
    if permuted and rec.get("perm") is not None:
        # act-order: digits are stored in permuted column order. Spatial
        # context needs the real layout, which a decode-in-stored-order coder
        # does NOT have — D1 is informational only for these tensors, and the
        # scales belong to stored groups, so no real-space scale grid exists.
        perm = np.frombuffer(rec["perm"], dtype="<u4").astype(np.int64)
        assert perm.size == c, "perm width != cols"
        d = d[:, perm]
        q = None

    qg = q.reshape(rows, c // g) if q is not None else None
    D0, D1, D216, D2, D3 = digit_stats(d, qg, k)
    row = dict(tensor=name, rows=r, cols=c, k=k, perm=int(permuted),
               lanes=int(rec.get("lanes", 1) or 1), n_probe=rows * c, ns_probe=n_s,
               dig_bytes=len(rec["digits"]), sca_bytes=len(rec["scales"]),
               dig_paid=8.0 * len(rec["digits"]) / (r * c), D0=D0, D1=D1,
               D2_16=D216, D2=D2, D3=D3,
               S0=float("nan"), S_left=float("nan"), S_top=float("nan"),
               S_avg_lt=float("nan"), S_avg_ltt=float("nan"), S_ctx_lt=float("nan"))
    if rows == r:                       # prefix rows need not match the whole stream
        assert abs(D0 - row["dig_paid"]) / max(D0, 1e-9) < 2e-3, \
            "%s: H(d) %.4f vs paid %.4f" % (name, D0, row["dig_paid"])
    if q is not None:
        S0, res = scale_stats(qg)
        row.update(S0=S0, S_left=res["left"], S_top=res["top"],
                   S_avg_lt=res["avg_lt"], S_avg_ltt=res["avg_ltt"], S_ctx_lt=res["ctx_lt"])
    return row


def wavg(rows, key, wkey="n_probe"):
    sel = [x for x in rows if x[key] == x[key]]
    return sum(x[key] * x[wkey] for x in sel) / max(1, sum(x[wkey] for x in sel)), len(sel)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("container")
    ap.add_argument("--every", type=int, default=1, help="take every Nth record")
    ap.add_argument("--max-rows", type=int, default=32768, help="rows per tensor")
    ap.add_argument("--include", default=r"embed|embd|output|head",
                    help="regex; matching records are always taken")
    ap.add_argument("--quiet", action="store_true", help="summary only")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "ctx_entropy_probe.csv"))
    a = ap.parse_args()
    path = Path(a.container)
    with open(path, "rb") as fh:
        magic = fh.read(4)
    reader = records_gguf if magic == b"GGUF" else records_k9
    want = re.compile(a.include)

    rows, skipped, t_all = [], [], time.time()
    for i, (name, rec) in enumerate(reader(path)):
        if i % a.every and not want.search(name):
            continue
        r, c = rec["shape"]
        if min(r, c // rec["group"]) < 2 or _mode_id(rec["scale_mode"]) != 4:
            skipped.append(name)
            continue
        t0 = time.time()
        row = probe(name, rec, a.max_rows)
        row["sec"] = round(time.time() - t0, 2)
        rows.append(row)
        if not a.quiet:
            best = min(x for x in ("D1", "D2_16", "D2", "D3") if row[x] == row[x])
            print("%-40s k=%-2d perm=%d | paid %.4f D1 %.4f D2 %.4f D3 %.4f | "
                  "sca paid %.3f best %.3f" %
                  (name[:40], row["k"], row["perm"], row["dig_paid"], row["D1"],
                   row["D2"], row["D3"], row["S0"], min(row[s] for s in S_KEYS
                                                       if row[s] == row[s])), flush=True)
    if not rows:
        print("no records probed; skipped: %s" % skipped[:10])
        return

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def agg(sel, label):
        if not sel:
            return
        n = sum(x["n_probe"] for x in sel)
        ns = sum(x["ns_probe"] for x in sel if x["S0"] == x["S0"])
        d0, _ = wavg(sel, "D0")
        dd = {key: wavg(sel, key)[0] for key in ("D1", "D2_16", "D2", "D3")}
        dbest = min(dd.values())
        line = "%-14s %3d tensors | digits %.4f -> prev %.4f (%+.2f%%) scale %.4f (%+.2f%%) " \
               "scale+row %.4f (%+.2f%%)" % (
                   label, len(sel), d0, dd["D1"], 100 * (dd["D1"] - d0) / d0,
                   dd["D2"], 100 * (dd["D2"] - d0) / d0, dd["D3"], 100 * (dd["D3"] - d0) / d0)
        if ns:
            s0 = wavg(sel, "S0", "ns_probe")[0]
            sbest = min(wavg(sel, s, "ns_probe")[0] for s in S_KEYS)
            line += " | scales %.3f -> %.3f (%+.1f%%)" % (s0, sbest, 100 * (sbest - s0) / s0)
        else:
            sbest = float("nan")
        print(line)

    print("\n%s  (%d tensors probed, %.0fs)" % (path.name, len(rows), time.time() - t_all))
    kinds = {}
    for x in rows:
        kinds.setdefault(x["tensor"].split(".")[-1], []).append(x)
    for kind in sorted(kinds):
        agg(kinds[kind], kind)
    agg(rows, "ALL")
    n = sum(x["n_probe"] for x in rows)
    ns = sum(x["ns_probe"] for x in rows if x["S0"] == x["S0"])
    d0 = wavg(rows, "D0")[0]
    dbest = min(wavg(rows, key)[0] for key in ("D1", "D2_16", "D2", "D3") if wavg(rows, key)[1])
    s0 = wavg(rows, "S0", "ns_probe")[0] if ns else float("nan")
    sbest = min(wavg(rows, s, "ns_probe")[0] for s in S_KEYS) if ns else float("nan")
    save_d = n * max(0.0, d0 - dbest) / 8
    save_s = ns * max(0.0, s0 - sbest) / 8
    tot = sum(x["dig_bytes"] + x["sca_bytes"] for x in rows)
    sc_d = sum(x["dig_bytes"] for x in rows)
    sc_s = sum(x["sca_bytes"] for x in rows if x["S0"] == x["S0"])
    print("best digit context %.4f, best scale context %.3f bits/scale" % (dbest, sbest))
    print("digits: %d MB probed-equivalent, scales: %d MB; context would save ~%.1f MB "
          "(digits %.2f%%, scales %.2f%%, together %.2f%% of the two streams)" % (
              sc_d / 1e6, sc_s / 1e6, (save_d + save_s) / 1e6,
              100 * save_d / max(sc_d, 1), 100 * save_s / max(sc_s, 1),
              100 * (save_d + save_s) / max(tot, 1)))
    if skipped:
        print("skipped (%d, too small or non-ent8): %s" % (len(skipped), skipped[:6]))


if __name__ == "__main__":
    main()
