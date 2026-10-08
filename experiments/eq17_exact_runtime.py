#!/usr/bin/env python3
"""eq17 — RQ10 stage B, leg 2: the exact runtime (docs/09 P37 widths + P38).

Mixed design, declared. EXACT track = every linear computation (embed lookup,
all matmuls, the mean-abs norm's algebra, residual adds, output head); the
continuous ops (RoPE, attention, silu) run fp-mixed with a DECLARED 2^30
fixed-point intake boundary — full rational closure of the continuous ops is
the registered follow-up.

State (the scaling-alphabet runtime): activation tensor =
  (num int64 (n, dim), den int64 (n, 1), E int64 (n, 1), J int64 (n, 1)),
value_il = num_il · 9^{E_i} · 2^{J_i} / den_i.

  * weights: value = Wnum·2^{Jw} / (den_s·H) — one scalar each; the fp16
    group scales decoded BIT-EXACTLY (fp16 mantissa ≈ 11 bits; a frexp-in-
    fp64 decode would waste ~42 bits), groups aligned to the minimal
    2-exponent (measured spread ~2 bits).
  * matmul: int64 num product-sum (magnitude-guarded → OverflowSite is the
    measured saturation point), den ×H_s scalar, tracks pass through.
  * mean-abs norm (exact): Σ_l|x_il| = Σ_l|num_il|·9^{E}·2^{J}/den — the
    (E, J, den) algebra CANCELS: norm = num·(dim·λ_num) / (λ_den·Σ_l|num_l|)
    with E, J untouched; λ an exact dyadic (λ_num, 2^10) from eq16.
  * residual add: per-row exact alignment to min(E), min(J) (multiplying
    by 9^k/2^k is always exact), lcm-weighted den, then reduce.
  * renorm after every site: gg = gcd(row nums ∪ row den) exact-divided;
    then den's 9^b / 2^a parts pulled into (E −= b, J −= a) — exact, and
    the reason mantissas stay near their significance width instead of
    compounding scale-magnitude.

Gates (docs/09):
  P37 (pure-exact widths): per-site num/den bit widths in base-9 digits
      (bits / log2 9) across the 5 layers vs the registered ≤20-digit bar.
      The census measured the regridded design (11.2–14.2 digits, that
      reading PASS); this measures whether the pure-exact track — the
      actual "no rounding anywhere" runtime — fits the bar. An overflow or
      over-bar width = recorded finding: exactness vs bounded state, with
      the registered fallback (regrid-per-layer) as the compliant form.
  P38: bit-exact across accumulation-order permutation (each matmul's sum
      split, halves accumulated in reversed order): identical width tracks
      site-by-site and identical logits when reached (CPU only; the int64-
      GPU device axis deferred).
  fp-agreement readout: exact-track logits vs the fp32 forward of the SAME
      substituted semantics (eq16 joint) — reported, not gated.

Subject: eq9_QAT_k27emb99.k9 + fp16 source norms (eq14's declared proxy).
CPU-only, 2 × 512 tokens.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402
from eq16_substituted import forward_sub, calibrate_lams  # noqa

B9 = "/mnt/matrix/Work/base9-quantization/experiments"
sys.path.insert(0, B9)
import k9  # noqa: E402

RES = L.RESULTS
GROUP = 64
LOG2_9 = math.log2(9.0)
I64MAX = 1 << 63
FIXED = 30


class OverflowSite(Exception):
    def __init__(self, tag, num_bits, den_bits):
        self.tag = tag
        self.num_bits = num_bits
        self.den_bits = den_bits
        super().__init__(f"int64 exact track saturates at {tag}: "
                         f"num {num_bits} b, den {den_bits} b")



def exact_rest(w32):
    """1-D rest weight (fp32 source) -> (Waligned (dim,) int64, J scalar int):
    value_c = W1c · 2^{J}. fp32 mantissa decoded bit-exactly (24 bits)."""
    bits = np.frombuffer(np.ascontiguousarray(w32.detach().numpy().astype(np.float32),
                                              dtype=np.float32).tobytes(),
                         dtype=np.uint32)
    exp_biased = ((bits >> np.uint32(23)) & np.uint32(255)).astype(np.int64)
    mant = (bits & np.uint32((1 << 23) - 1)).astype(np.int64)
    normal = exp_biased != 0
    e = np.where(normal, exp_biased - 150, np.int64(-149))   # a·2^e, a ≤ 2^24
    a = np.where(normal, mant | (np.int64(1) << 23), mant).astype(np.int64)
    e0 = int(e.min())
    assert e0 >= -40, f"rest 2-exponent too wide ({e0})"
    shift = np.clip(e - e0, 0, 60).astype(np.int64)
    Wal = a << shift
    assert int(np.abs(Wal).max()) < I64MAX, "rest overflow"
    return Wal.astype(np.int64), int(e0)        # value = Wal·2^{e0} raw

def fp16_rational(m16):
    """fp16 -> exact (a, e): m = a·2^e with a the 11-bit mantissa int."""
    bits = np.frombuffer(np.ascontiguousarray(m16, dtype=np.float16).tobytes(),
                         dtype=np.uint16)
    exp_biased = ((bits >> np.uint16(10)) & np.uint16(31)).astype(np.int64)
    mant = (bits & np.uint16(1023)).astype(np.int64)
    normal = exp_biased != 0
    e = np.where(normal, exp_biased - 25, np.int64(-24))   # a·2^{-24} subnormal
    a = np.where(normal, mant | np.int64(1024), mant).astype(np.int64)
    got = (a * np.exp2(e.astype(np.float64)))
    assert (got.astype(np.float32) == np.asarray(m16, np.float32)).all(), \
        "fp16 decode mismatch"
    return a, e.astype(np.int64)


def exact_weight(rec, true_cols, group=GROUP):
    """record -> (Wnum (r, c_pad) int64, Jw scalar int, Hs scalar int,
    true_cols): value_jl = Wnum_jl · 2^{Jw} / Hs. Padded width is sliced to
    true_cols (the record stores padding; padding digits = H = exact 0)."""
    H = (rec["k"] - 1) // 2
    d = k9.decode_digits(rec).reshape(rec["shape"][0], -1, group).astype(np.int64) - H
    r, ng, g = d.shape
    m = k9.decode_scales(rec["scales"], rec["scale_mode"], r * ng).astype(np.float16)
    a16, e16 = fp16_rational(m.reshape(-1))
    a16 = a16.reshape(r, ng)
    e16 = e16.reshape(r, ng)
    e0 = int(e16.min())
    shift = np.clip(e16 - e0, 0, 40).astype(np.int64)
    Wnum = ((a16 << shift))[:, :, None] * d                # (r, ng, g) exact
    Wnum = Wnum.reshape(r, -1)
    assert int(np.abs(Wnum).max()) * H < I64MAX, "Wnum overflow"
    Jw = int(e0) + 20                                       # the shared 2-part
    Wd = int(H) * (1 << 20)                                 # matches the Jw offset
    return Wnum, Jw, Wd, true_cols


def vextract(den):
    """Pull den's 9-powers and 2-powers out exactly: returns (den', ΔE, ΔJ)."""
    dE = np.zeros_like(den)
    dJ = np.zeros_like(den)
    t = den.copy()
    while (t % 9 == 0).any():
        m9 = t % 9 == 0
        dE[m9] += 1
        t[m9] //= 9
    while (t % 2 == 0).any():
        m2 = t % 2 == 0
        dJ[m2] += 1
        t[m2] //= 2
    return t, dE, dJ


def _gcd_obj(num):
    from math import gcd
    out = np.empty((num.shape[0], 1), dtype=object)
    for i in range(num.shape[0]):
        g = 0
        for v in num[i]:
            v = abs(int(v))
            if v == 0:
                continue
            g = gcd(g, v)
            if g == 1:
                break
        out[i, 0] = g or 1
    return out


def renorm(num, den, E, J, tag):
    """Exact: gg = per-row gcd(|num| ∪ den); divide; pull den's 9/2 parts
    into the tracks. value preserved. Works on int64 and object carriers."""
    from math import gcd
    obj = num.dtype == object
    if not obj:
        gg = np.gcd.reduce(np.abs(num), axis=1, dtype=np.int64).reshape(-1, 1)
        gg = np.gcd(gg, den)
        gg = np.where(gg == 0, 1, gg)
    else:
        gg = np.maximum(_gcd_obj(num), 1)
        gg = np.array([[gcd(int(g), abs(int(d)))
                        for g, d in zip(gg.reshape(-1), den.reshape(-1))]
                       ], dtype=object).reshape(den.shape)
    num2 = num // gg
    den2 = den // gg
    obj2 = den2.dtype == object
    dE = np.zeros(den2.shape, dtype=object if obj2 else np.int64)
    dJ = np.zeros(den2.shape, dtype=object if obj2 else np.int64)
    t = den2.copy()
    if not obj2:
        while (t % 9 == 0).any():
            m9 = t % 9 == 0
            dE[m9] += 1
            t[m9] //= 9
        while (t % 2 == 0).any():
            m2 = t % 2 == 0
            dJ[m2] += 1
            t[m2] //= 2
    else:
        for i in range(t.shape[0]):
            v = int(t[i, 0])
            while v % 9 == 0:
                dE[i, 0] += 1
                v //= 9
            while v % 2 == 0:
                dJ[i, 0] += 1
                v //= 2
            t[i, 0] = v
    E2 = E - dE
    J2 = J - dJ
    return num2, t, E2, J2


def matmul_exact(num, den, E, J, Wnum, Jw, Wd, tag, order=None):
    """Exact; on int64 saturation, widens to object (big-int) and continues
    (the widening events are recorded in the audit — they are the P37 data)."""
    obj = num.dtype == object or Wnum.dtype == object
    if not obj:
        mx = int(np.abs(num).max()) if num.size else 1
        mw = int(np.abs(Wnum).max())
        if mx and mx * mw * num.shape[1] >= I64MAX:
            if MATMUL_POLICY == "int64":
                raise OverflowSite(tag, mx.bit_length(), mw.bit_length())
            MATMUL_WIDENED.append(tag)
            num = num.astype(object) if num.dtype != object else num
        else:
            if order is None:
                ynum = num @ Wnum.T
            else:
                h = num.shape[1] // 2
                ynum = (num[:, h:] @ Wnum[:, h:].T) + (num[:, :h] @ Wnum[:, :h].T)
            return ynum, den * Wd, E, J + np.int64(Jw)
    # object path: exact python ints
    Wt = Wnum.T.astype(object)
    if order is None:
        ynum = (num[:, :, None] * Wt[None, :, :]).sum(1)
    else:
        h = num.shape[1] // 2
        ynum = ((num[:, h:, None] * Wt[None, h:, :]).sum(1)
                + (num[:, :h, None] * Wt[None, :h, :]).sum(1))
    return (ynum, den.astype(object) * Wd, E.astype(object),
            J.astype(object) + Jw)


def to_fp(num, den, E, J):
    """Correctly-rounded float view of the exact state. The object path goes
    through Fraction -> float; the exact nums outgrow fp64 (206 base-9 digits
    by L4) and any direct f64 cast of num overflows to inf."""
    if num.dtype == object:
        from fractions import Fraction
        val = np.empty(num.shape, dtype=np.float64)
        for i in range(num.shape[0]):
            scale = Fraction(3) ** (2 * int(E[i, 0])) * Fraction(2) ** int(J[i, 0])
            d0 = Fraction(int(den[i, 0]))
            for j in range(num.shape[1]):
                val[i, j] = float(Fraction(int(num[i, j])) / d0 * scale)
        return torch.from_numpy(val)
    val = (num.astype(np.float64) / den.astype(np.float64)
           * np.power(9.0, E.astype(np.float64))
           * np.exp2(J.astype(np.float64)))
    return torch.from_numpy(val)


ADD_WIDENED = []          # log of widening events (audit)
MATMUL_WIDENED = []       # log of widening events (audit)
MATMUL_POLICY = "wide"   # int64 would raise OverflowSite; wide falls back to object


def add_exact(num1, den1, E1, J1, num2, den2, E2, J2, tag):
    obj = num1.dtype == object or num2.dtype == object
    if not obj:
        Em = np.minimum(E1, E2)
        Jm = np.minimum(J1, J2)
        g = np.gcd(den1, den2)
        l = (den1 // g) * den2
        p9a = np.power(np.int64(9), np.clip(E1 - Em, 0, 40))
        p9b = np.power(np.int64(9), np.clip(E2 - Em, 0, 40))
        p2a = np.exp2(np.clip(J1 - Jm, 0, 60).astype(np.float64))
        p2b = np.exp2(np.clip(J2 - Jm, 0, 60).astype(np.float64))
        bound = (int(np.abs(num1).max()) * int(np.abs(l // den1).max())
                 * int(p9a.max()) * int(p2a.max())
                 + int(np.abs(num2).max()) * int(np.abs(l // den2).max())
                 * int(p9b.max()) * int(p2b.max()))
        if int(l.max()) < I64MAX and bound < I64MAX:
            num = (num1 * (l // den1) * p9a * p2a.astype(np.int64)
                   + num2 * (l // den2) * p9b * p2b.astype(np.int64))
            return renorm(num, l, Em, Jm, tag)
        # first saturation: widen both sides and continue (recorded)
        ADD_WIDENED.append(tag)
        (num1, den1, E1, J1) = (num1.astype(object), den1.astype(object),
                                E1.astype(object), J1.astype(object))
        (num2, den2, E2, J2) = (num2.astype(object), den2.astype(object),
                                E2.astype(object), J2.astype(object))
    # object path (exact python ints; slow — wide/width-probe runs only)
    from math import gcd
    n = den1.shape[0]
    Em = np.minimum(E1.astype(object), E2.astype(object)) \
        if E1.dtype == object or E2.dtype == object else np.minimum(E1, E2)
    Jm = np.minimum(J1, J2) if J1.dtype != object else \
        np.array([min(int(a), int(b)) for a, b in
                  zip(J1.reshape(-1), J2.reshape(-1))],
                 dtype=object).reshape(J1.shape)
    out_num = np.empty((n, num1.shape[1]), dtype=object)
    out_den = np.empty((n, 1), dtype=object)
    out_E = np.empty((n, 1), dtype=object)
    out_J = np.empty((n, 1), dtype=object)
    for i in range(n):
        d1 = int(den1[i, 0]); d2 = int(den2[i, 0])
        g = gcd(d1, d2)
        lcm = (d1 // g) * d2
        e1 = int(E1[i, 0]); e2 = int(E2[i, 0])
        j1 = int(J1[i, 0]); j2 = int(J2[i, 0])
        Em_i = min(e1, e2); Jm_i = min(j1, j2)
        t1 = int(num1[i, 0] if num1.ndim == 2 and num1.shape[1] == 1 else 0)
        num_sum = np.zeros(num1.shape[1], dtype=object)
        s1 = int(num1[i].sum()) if False else None
        # vectorized row ops (row i):
        a1 = (lcm // d1) * (9 ** (e1 - Em_i) if e1 > Em_i else 1) * \
            (2 ** (j1 - Jm_i) if j1 > Jm_i else 1)
        a2 = (lcm // d2) * (9 ** (e2 - Em_i) if e2 > Em_i else 1) * \
            (2 ** (j2 - Jm_i) if j2 > Jm_i else 1)
        row = [int(x) for x in num1[i]] 
        num_sum = np.array([a1 * int(num1[i, k]) + a2 * int(num2[i, k])
                            for k in range(num1.shape[1])], dtype=object)
        out_num[i] = num_sum
        out_den[i, 0] = lcm
        out_E[i, 0] = Em_i
        out_J[i, 0] = Jm_i
    return renorm(out_num, out_den, out_E, out_J, tag)


def tracksite(track, tag, num, den):
    nb = int(np.abs(num).max()).bit_length() if num.size else 0
    db = int(den.max()).bit_length() if den.size else 0
    if tag not in track or nb > track[tag][0]:
        track[tag] = (nb, db)


@torch.no_grad()
def exact_forward(arts, rests, idx, ma, lam_pairs, order=None, track=None):
    audit = dict(fp_intakes=[])
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    idx_np = idx.numpy().reshape(-1)

    We, Jwe, Wde, we_cols = arts["tok_embeddings.weight"]
    num = We[idx_np][:, :we_cols]
    den = np.full((num.shape[0], 1), Wde, dtype=np.int64)
    E = np.zeros((num.shape[0], 1), dtype=np.int64)
    J = np.full((num.shape[0], 1), Jwe, dtype=np.int64)
    num, den, E, J = renorm(num, den, E, J, "embed")
    if track is not None:
        tracksite(track, "embed", num, den)

    neg_mask = torch.full((T, T), float("-inf")).triu(1)
    resid_keep = None
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        resid_keep = (num.copy(), den.copy(), E.copy(), J.copy())
        # norm output goes to a SEPARATE carrier; the residual stays the
        # pre-norm stream (the residual-rebind pattern) — restored at res1
        lna, lda = lam_pairs[2 * l]
        W1a, Jw1a, = rests[f"{p}attention_norm.weight"]
        anum = num * (dim * lna) * W1a[None, :]
        aden = lda * np.abs(num).sum(1, keepdims=True)
        num, den, E, J = renorm(anum, aden,
                                np.zeros_like(E), np.full_like(E, Jw1a),
                                f"{p}attnorm")
        if track is not None:
            tracksite(track, f"L{l}.attnorm", num, den)
        qn, Jq, Wdq, qc = arts[p + "attention.wq.weight"]
        kn, Jk, Wdk, kc = arts[p + "attention.wk.weight"]
        vn, Jv, Wdv, vc = arts[p + "attention.wv.weight"]
        qnum, qden, qE, qJ = matmul_exact(num, den, E, J, qn[:, :qc], Jq, Wdq,
                                          f"{p}wq", order)
        knum, kden, kE, kJ = matmul_exact(num, den, E, J, kn[:, :kc], Jk, Wdk,
                                          f"{p}wk", order)
        vnum, vden, vE, vJ = matmul_exact(num, den, E, J, vn[:, :vc], Jv, Wdv,
                                          f"{p}wv", order)
        if track is not None:
            tracksite(track, f"L{l}.qkv",
                      np.concatenate([qnum, knum, vnum], 1),
                      np.concatenate([qden, kden, vden], 1))
        # fp-mixed RoPE + attention (declared; deterministic on CPU)
        qf = to_fp(qnum, qden, qE, qJ).reshape(B, T, nh, hd).float()
        kf = to_fp(knum, kden, kE, kJ).reshape(B, T, nkv, hd).float()
        vf = to_fp(vnum, vden, vE, vJ).reshape(B, T, nkv, hd).float()
        qf = L.rope(qf, torch.arange(T))
        kf = L.rope(kf, torch.arange(T))
        rep = nh // nkv
        kk = kf.repeat_interleave(rep, dim=2).transpose(1, 2)
        vv = vf.repeat_interleave(rep, dim=2).transpose(1, 2)
        s = (qf.transpose(1, 2) @ kk.transpose(-2, -1)) / hd ** 0.5 + neg_mask
        o = (s.softmax(-1) @ vv).transpose(1, 2).reshape(B, T, dim)
        o_flat = o.reshape(-1, dim).numpy().astype(np.float64)
        onum = np.rint(o_flat * (1 << FIXED)).astype(np.int64)
        audit["fp_intakes"].append(f"{p}attention")
        on, Jo, Wdo, oc = arts[p + "attention.wo.weight"]
        ones = np.full((onum.shape[0], 1), 1, dtype=np.int64)
        oEz = np.zeros((onum.shape[0], 1), dtype=np.int64)
        oJz = np.full((onum.shape[0], 1), -FIXED, dtype=np.int64)
        onum, oden, oE, oJ = matmul_exact(onum, ones, oEz, oJz, on[:, :oc],
                                          Jo, Wdo, f"{p}wo", order)
        oE = np.zeros_like(oden)
        oJ = np.full_like(oden, -FIXED + Jo)
        num, den, E, J = add_exact(resid_keep[0], resid_keep[1],
                                   resid_keep[2], resid_keep[3],
                                   onum, oden, oE, oJ, f"{p}res1")
        if track is not None:
            tracksite(track, f"L{l}.res_mid", num, den)
        # second residual carrier (for res2)
        resid2_keep = (num.copy(), den.copy(), E.copy(), J.copy())
        # mean-abs norm into a separate carrier; (E, J) cancel algebraically
        lnum, lden = lam_pairs[2 * l + 1]
        W1n, Jw1e, = rests[f"{p}ffn_norm.weight"]
        num, den, E, J = renorm(num * (dim * lnum) * W1n[None, :],
                                lden * np.abs(num).sum(1, keepdims=True),
                                np.zeros_like(resid2_keep[2]),
                                np.full_like(resid2_keep[2], Jw1e), f"{p}norm")
        if track is not None:
            tracksite(track, f"L{l}.norm", num, den)
        w1n, Jw1, Wdw1, wc1 = arts[p + "feed_forward.w1.weight"]
        w3n, Jw3, Wdw3, wc3 = arts[p + "feed_forward.w3.weight"]
        w1num, w1den, w1E, w1J = matmul_exact(num, den, E, J, w1n[:, :wc1],
                                              Jw1, Wdw1, f"{p}w1", order)
        w3num, w3den, w3E, w3J = matmul_exact(num, den, E, J, w3n[:, :wc3],
                                              Jw3, Wdw3, f"{p}w3", order)
        if track is not None:
            tracksite(track, f"L{l}.w1w3",
                      np.concatenate([w1num, w3num], 1),
                      np.concatenate([w1den, w3den], 1))
        sil = to_fp(w1num, w1den, w1E, w1J).float()
        siln = np.rint(((sil * (1.0 + sil / (sil.abs() + 1.0)) * 0.5)
                        .numpy().astype(np.float64)) * (1 << FIXED)).astype(np.int64)
        audit["fp_intakes"].append(f"{p}silu")
        prod_num = siln * w3num
        prod_den = w3den
        prod_E, prod_J = w3E, w3J - FIXED
        w2n, Jw2, Wdw2, wc2 = arts[p + "feed_forward.w2.weight"]
        o2num, o2den, o2E, o2J = matmul_exact(prod_num, prod_den, prod_E,
                                              prod_J, w2n[:, :wc2], Jw2,
                                              Wdw2, f"{p}w2", order)
        num, den, E, J = add_exact(resid2_keep[0], resid2_keep[1],
                                   resid2_keep[2], resid2_keep[3],
                                   o2num, o2den, o2E, o2J, f"{p}res2")
        if track is not None:
            tracksite(track, f"L{l}.res_out", num, den)

    lnum, lden = lam_pairs[2 * ma["n_layers"]]
    W1f, Jw1f, = rests["norm.weight"]
    num_n = num * (dim * lnum) * W1f[None, :]
    den_n = lden * np.abs(num).sum(1, keepdims=True)
    num, den, E, J = renorm(num_n, den_n,
                            np.zeros_like(E), np.full_like(E, Jw1f), "normF")
    hn, JwH, WdH, hc = arts["output.weight"]
    out_num, out_den, out_E, out_J = matmul_exact(num, den, E, J, hn[:, :hc],
                                                  JwH, WdH, "head", order)
    if track is not None:
        tracksite(track, "logits", out_num, out_den)
    logits = to_fp(out_num, out_den, out_E, out_J).reshape(B, T, -1).float()
    return logits, audit


# ------------------------------------------------------------------ main

def main():
    t0 = time.time()
    torch.manual_seed(0)
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    EMB = next(n for n in ("tok_embeddings.weight", "output.weight") if n in uniq)
    tokens = L.valid_tokens(8 * ma["max_seq_len"])
    idx = torch.from_numpy(tokens[: 2 * ma["max_seq_len"]]
                           .reshape(2, ma["max_seq_len"]).astype(np.int64))

    recs = k9.read_k9(os.path.join(RES, "eq9_QAT_k27emb99.k9"))
    widths = {kk: int(v.shape[1]) for kk, v in uniq.items() if v.dim() == 2}

    def width_of(name):
        if name in widths:
            return widths[name]
        alt = "tok_embeddings.weight" if name == "output.weight" \
            else "output.weight"
        return widths[alt]

    arts, fp32_sd = {}, {}
    for n, rec in recs.items():
        cols = width_of(n)
        W, Jw, Wd, cc = exact_weight(rec, cols)
        arts[n] = (W, Jw, Wd, cc)
        val = W[:, :cc].astype(np.float64) * 2.0 ** Jw / Wd
        fp32_sd[n] = torch.from_numpy(val.astype(np.float32))
    for k2, v in uniq.items():
        if v.dim() == 1:
            fp32_sd[k2] = v.half().float()
    fp32_sd[EMB] = fp32_sd["tok_embeddings.weight"]
    fp32_sd["output.weight"] = fp32_sd[EMB]
    arts["output.weight"] = arts[EMB]                 # the tied tie
    print(f"[eq17] exact weight prep done ({time.time()-t0:.0f}s)")

    lams = calibrate_lams(fp32_sd, ma)
    lam_pairs = [(int(round(x * (1 << 10))), 1 << 10) for x in lams]
    rests = {}
    for kk, v in uniq.items():
        if v.dim() == 1:
            if os.environ.get("EQ17_REST_BITS"):
                rb = int(os.environ["EQ17_REST_BITS"])
                w32 = v.detach().numpy().astype(np.float64)
                # uniform-power grid rest: value = Wal·2^{−rb} (exact, no
                # per-column alignment) — a defined rounding of the rest
                Wal = np.rint(w32 * (1 << rb)).astype(np.int64)
                rests[kk] = (Wal, -rb)
            else:
                Wal, Jr = exact_rest(v)
                rests[kk] = (Wal, Jr)
    print(f"[eq17] λ dyadic (2^-10): {[round(a / 1024, 3) for a, _ in lam_pairs]}")
    # the fp reference of the EXACT track's OWN mixed-op semantics: standard
    # attention (use_att=False = softmax path), mean-abs norm + silu-r.
    ref = forward_sub(fp32_sd, idx, ma, use_norm=True, use_att=False,
                      use_silu=True, lams=lams)

    tracks, logits_of, sat = {}, {}, {}
    for order in (None, "split_rev"):
        tag = order or "natural"
        ADD_WIDENED.clear()
        MATMUL_WIDENED.clear()
        track = {}
        try:
            logits, _ = exact_forward(arts, rests, idx, ma, lam_pairs,
                                      order=order, track=track)
            logits_of[tag] = logits
            sat[tag] = dict(widen_events=list(ADD_WIDENED) + list(MATMUL_WIDENED)) \
                if (ADD_WIDENED or MATMUL_WIDENED) else None
        except OverflowSite as e:
            sat[tag] = dict(site=e.tag, num_bits=e.num_bits, den_bits=e.den_bits)
            logits_of[tag] = None
            print(f"[eq17] {tag}: OVERFLOW at {e.tag} "
                  f"(num {e.num_bits} b, den {e.den_bits} b)")
        tracks[tag] = track
        print(f"[eq17] order={tag}: ran {len(track)} tracked sites "
              f"({time.time()-t0:.0f}s)")

    if logits_of["natural"] is not None and logits_of["split_rev"] is not None:
        agree = torch.equal(logits_of["natural"], logits_of["split_rev"])
        tracks_agree = (set(tracks["natural"]) == set(tracks["split_rev"]) and
                        all(tracks["natural"][k] == tracks["split_rev"][k]
                            for k in tracks["natural"]))
        sat_agree = (sat["natural"] is None and sat["split_rev"] is None) or (
            sat["natural"] and sat["split_rev"] and
            sat["natural"].get("widen_events") == sat["split_rev"].get("widen_events"))
        drift = float((logits_of["natural"] - ref).abs().max())
        p38 = dict(prediction="P38", order_identity=bool(agree),
                   width_tracks_identical=bool(tracks_agree),
                   widen_events_identical=bool(sat_agree),
                   max_logit_delta_vs_fp32ref=drift,
                   device_axis="cpu only (int64 GPU kernels deferred)",
                   verdict="PASS" if (agree and tracks_agree and sat_agree) else "FAIL")
        msg = (f"order-identical logits={agree}; width tracks identical="
               f"{tracks_agree}; widening events identical={sat_agree}; "
               f"max|Δlogits−fp32ref| = {drift:.3e}")
    else:
        s1, s2 = sat.get("natural"), sat.get("split_rev")
        tracks_agree = (set(tracks["natural"]) == set(tracks["split_rev"]) and
                        all(tracks["natural"][k] == tracks["split_rev"][k]
                            for k in tracks["natural"]))
        p38_ok = bool(s1 and s2 and s1.get("site") == s2.get("site") and tracks_agree)
        p38 = dict(prediction="P38", order_identity=None,
                   same_saturation=p38_ok, width_tracks_identical=tracks_agree,
                   saturated_at=s1.get("site") if s1 else None,
                   note="exactness proven on the int64-fitting prefix; integer "
                        "arithmetic is order-exact by construction",
                   verdict="PASS" if p38_ok else "FAIL")
        msg = f"both runs saturate at {s1.get('site') if s1 else '?'}"
    print(f"[eq17] P38 ({p38['verdict']}): {msg}")

    digits = {tag: dict(num_digits=round(nb / LOG2_9, 1),
                        den_digits=round(db / LOG2_9, 1))
              for tag, (nb, db) in tracks["natural"].items()}
    worst_num = max(v["num_digits"] for v in digits.values())
    worst_den = max(v["den_digits"] for v in digits.values())
    widened = sat.get("natural") is not None
    widen_events = (sat["natural"].get("widen_events")
                    if widened and "widen_events" in sat["natural"] else
                    [sat["natural"].get("site")] if widened else [])
    p37 = dict(prediction="P37", worst_num_digits=worst_num,
               worst_den_digits=worst_den, bar=20, u128_digits=40.4,
               widened_at=widen_events,
               verdict=("FAIL" if (worst_num > 20 or worst_den > 20)
                        else "PASS"),
               note="pure-exact track vs the ≤20-digit registered bar; the "
                    "regrid-per-layer design (registered fallback; census "
                    "11.2–14.2 digits) is the compliant bounded-state form. "
                    "Exactness-vs-bounded-state = the recorded finding.")
    print(f"[eq17] P37 ({p37['verdict']}): worst num {worst_num} / den "
          f"{worst_den} base-9 digits (bar 20, u128 = 40.4)"
          + (f", int64 widening at {', '.join(widen_events)}" if widened else ""))
    for tag, d in digits.items():
        print(f"   {tag:>10}: num {d['num_digits']:>6} | den {d['den_digits']:>6}")

    out = dict(P37=p37, P38=p38, widths=digits, saturation=sat,
               lams10=[round(a / 1024, 4) for a, _ in lam_pairs],
               rest_bits=os.environ.get("EQ17_REST_BITS", "fp32-exact"),
               note="EQ17_REST_BITS set -> rests are uniform-power grid "
                    "roundings (defined rounding of the 1-D constants); "
                    "unset -> bit-exact fp32 rests (aligned per column)")
    suffix = ("_" + os.environ["EQ17_REST_BITS"]) if os.environ.get("EQ17_REST_BITS") else ""
    with open(os.path.join(RES, f"eq17_exact_runtime{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    md = ["# eq17 — RQ10 stage B leg 2: the exact runtime (docs/09)\n",
          "subject: eq9_QAT_k27emb99.k9, 2×512 tokens; state = (num, den, E, J)\n"
          "per row with (E, J) exponent tracks absorbing scale; exact per-row\n"
          "reduction/renorm after every site; fp-mixed RoPE/attention/silu\n"
          "with declared 2^30 fixed-point intakes; λ dyadic 2^-10.\n",
          f"- P38 ({p38['verdict']}): {msg}.",
          f"- P37 ({p37['verdict']}): worst num {worst_num} digits, worst den "
          f"{worst_den} digits vs the ≤20-digit bar"
          + (f"; int64 widened at {widen_events}." if widened else "."),
          "", "| site | num digits | den digits |", "|---|---|---|"]
    for tag, d in digits.items():
        md.append(f"| {tag} | {d['num_digits']} | {d['den_digits']} |")
    with open(os.path.join(RES, f"eq17_exact_runtime{suffix}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[eq17] wrote results/eq17_exact_runtime.{{json,md}} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()