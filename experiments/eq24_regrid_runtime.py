#!/usr/bin/env python3
"""eq24 — RQ10 stage B: the (mant, e, j)-state exact runtime on the P39g
artifact (eq23) — the shipping runtime's dry-run.

State: value = num·9^e·2^jj / den with
  num (n, dim)  integer mantissas — ≤ 9^n digits after every regrid,
  den (n, 1)    the odd-latch carrier (weights' H-divisors, gain mantissas,
                the norm's Σ|x| odd parts) — bounded, snipped downstream,
  e, jj (n, 1)  the free 9/2-power tracks (pure shifts, never a cost).

Discipline: EXACTLY between regrids (every op = exact integer arithmetic on
the four carriers); AT the regrids (each matmul input — the trained P39g
semantics = eq23's STE), the state re-expresses as (n-digit mantissas,
pure-9 tracks): den → 1, jj → 0 — every odd latch accumulated in flight is
snipped by the defined rounding. That's the whole runtime.

Fidelity: the fp twin of the SAME trained semantics (mean-abs + frozen
gains + regrid-n4 values + 10-bit rests + softmax + SwiGLU) runs alongside;
per-site |Δ| + the score-flip census. Widths: per-site num/den bit-lengths
in base-9 digits. P38: order-permutation bit-identity.

Subject: eq23 artifacts. CPU, 2×512 tokens.
"""
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/mnt/matrix/Work/base9-quantization/experiments")
import eq_lib as L  # noqa: E402
import k9  # noqa: E402
import eq17_exact_runtime as E17  # noqa: E402

RES = L.RESULTS
N_DIGITS = 4
FIXED = E17.FIXED
LOG9 = float(np.log(9.0))
LOG2 = float(np.log(2.0))
LOG2_9 = float(math.log2(9.0))
U128_DIGITS = 40.4
I64MAX = 1 << 62


class State:
    __slots__ = ("num", "den", "e", "jj")

    def __init__(self, num, den, e, jj):
        self.num, self.den, self.e, self.jj = num, den, e, jj

    def copy(self):
        return State(self.num.copy(), self.den.copy(), self.e.copy(),
                     self.jj.copy())

    def view(self):
        num, den, e, jj = self.num, self.den, self.e, self.jj
        if num.dtype == object:
            val = np.empty(num.shape, dtype=np.float64)
            for i in range(num.shape[0]):
                s = (3.0 ** (2 * int(e[i, 0])) * 2.0 ** (int(jj[i, 0]))
                     / (int(den[i, 0]) or 1))
                for j in range(num.shape[1]):
                    val[i, j] = float(int(num[i, j])) * s
            return torch.from_numpy(val)
        return torch.from_numpy(
            num.astype(np.float64)
            * np.power(9.0, e.astype(np.float64))
            * np.exp2(jj.astype(np.float64)) / den.astype(np.float64))


def obj_gcd_col(num, den):
    from math import gcd
    gg = np.empty((num.shape[0], 1), dtype=object)
    for i in range(num.shape[0]):
        g = abs(int(den[i, 0]))
        for v in num[i]:
            g = gcd(g, abs(int(v)))
            if g == 1:
                break
        gg[i, 0] = g or 1
    return gg


def renorm_state(s):
    num, den = s.num, s.den
    if num.dtype != object:
        gg = np.gcd.reduce(np.abs(num), axis=1, dtype=np.int64).reshape(-1, 1)
        gg = np.gcd(gg, den)
        gg = np.where(gg == 0, 1, gg)
        num2 = num // gg
        den2 = den // gg
    else:
        gg = obj_gcd_col(num, den)
        num2 = num // gg
        den2 = den // gg
    t = den2.copy()
    dE = np.zeros_like(den2)
    if den2.dtype == object:
        for i in range(t.shape[0]):
            v = int(t[i, 0])
            while v and v % 9 == 0:
                dE[i, 0] += 1
                v //= 9
            t[i, 0] = v
    else:
        while (t % 9 == 0).any():
            m9 = t % 9 == 0
            dE[m9] += 1
            t[m9] //= 9
    return State(num2, t, s.e - dE, s.jj)


def mul_safe(den, factor):
    f = np.asarray(factor)
    m = int(np.abs(den).max()) if den.size else 0
    fm = int(np.abs(f).max()) if f.size else 1
    if den.size and den.dtype != object and m * fm >= I64MAX:
        return den.astype(object) * f.astype(object)
    return den * f


def manorm_state(s, ssn, site_gain):
    """the trained norm exactly: value·(dim·gain)/∑|value_c| with
    gain = m·9^a·2^b; the rest (wn, Jw) folded: num·wn, jj += Jw; the
    ∑|value| lands in the den (the row's ∑|num|-integer)."""
    (wn, Jw) = ssn
    (m, a, b) = site_gain
    if s.num.dtype != object:
        ssum = np.abs(s.num).sum(1, keepdims=True)
        num = s.num * (s.num.shape[1] * int(m)) * wn[None, :]
        # the divisor algebra CANCELS the incoming (e, jj, den): the value' =
        # num·(dim·m·wn)·9^a·2^{b+Jw}/∑|num| — pure RESET tracks
        den = mul_safe(np.ones_like(s.den), ssum)
        e = np.full_like(s.e, int(a))
        jj = np.full_like(s.jj, int(b) + int(Jw))
        return renorm_state(State(num, den, e, jj))
    ssum = np.array([sum(abs(int(v)) for v in s.num[i])
                     for i in range(s.num.shape[0])], dtype=object
                    ).reshape(-1, 1)
    num = s.num * (s.num.shape[1] * int(m)) * \
        np.asarray(wn, dtype=object)[None, :]
    den = ssum.astype(object)
    e = np.full_like(s.e, int(a), dtype=object)
    jj = np.full_like(s.jj, int(b) + int(Jw), dtype=object)
    return renorm_state(State(num, den, e, jj))


def regrid_state(s, n_digits=N_DIGITS):
    """the defined K9-rounding: the value → (n-digit mantissas, pure-9
    tracks); den → 1, jj → 0. The trained semantics (eq23's STE)."""
    vals = s.view().numpy()
    mx = np.abs(vals).max(axis=1, keepdims=True)
    e_r = np.ceil(np.log(np.maximum(mx, 1e-30)) / LOG9).astype(np.int64)
    mant = np.rint(vals * np.power(9.0, (n_digits - e_r).astype(np.float64)))
    return State(mant.astype(np.int64), np.ones_like(s.den),
                 e_r - n_digits, np.zeros_like(s.jj))


def widen(a):
    return a.astype(object)


def mul_int(a, b, tag, audit):
    """integer multiply with int64->object escalation (value exact)."""
    if a.dtype != object and b.dtype != object:
        if int(np.abs(a).max()) * int(np.abs(b).max()) >= I64MAX:
            audit.append(tag)
            return widen(a) * widen(b)
    return a * b


def add_state(s1, s2, tag, audit):
    """exact lcm-weighted add (E/j aligned)."""
    if s1.num.dtype == object or s2.num.dtype == object:
        pass
    else:
        lmax = int(np.abs(s1.den).max()) * int(np.abs(s2.den).max())
        nmax = int(np.abs(s1.num).max()) * int(np.abs(s2.den).max())
        if lmax >= I64MAX or nmax >= I64MAX:
            if audit is not None:
                audit.append(tag)
            def widen(x):
                return x.astype(object)
            s1 = State(widen(s1.num), widen(s1.den), s1.e if s1.e.dtype
                       == object else widen(s1.e),
                       s1.jj if s1.jj.dtype == object else widen(s1.jj))
            s2 = State(widen(s2.num), widen(s2.den), s2.e if s2.e.dtype
                       == object else widen(s2.e),
                       s2.jj if s2.jj.dtype == object else widen(s2.jj))
    from math import gcd
    n = s1.num.shape[0]
    on = np.empty(s1.num.shape, dtype=object)
    od = np.empty((n, 1), dtype=object)
    oe = np.empty((n, 1), dtype=object)
    oj = np.empty((n, 1), dtype=object)
    for i in range(n):
        d1 = int(s1.den[i, 0])
        d2 = int(s2.den[i, 0])
        g = gcd(d1, d2)
        lcm = (d1 // g) * d2
        em = min(int(s1.e[i, 0]), int(s2.e[i, 0]))
        jm = min(int(s1.jj[i, 0]), int(s2.jj[i, 0]))
        f1 = (lcm // d1) * (9 ** (int(s1.e[i, 0]) - em)
                            if int(s1.e[i, 0]) > em else 1) \
            * (2 ** (int(s1.jj[i, 0]) - jm)
               if int(s1.jj[i, 0]) > jm else 1)
        f2 = (lcm // d2) * (9 ** (int(s2.e[i, 0]) - em)
                            if int(s2.e[i, 0]) > em else 1) \
            * (2 ** (int(s2.jj[i, 0]) - jm)
               if int(s2.jj[i, 0]) > jm else 1)
        on[i] = s1.num[i].astype(object) * f1 + s2.num[i].astype(object) * f2
        od[i, 0] = lcm
        oe[i, 0] = em
        oj[i, 0] = jm
    return renorm_state(State(on, od, oe, oj))


def main():
    t0 = time.time()
    torch.manual_seed(0)
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    EMB = next(n for n in ("tok_embeddings.weight", "output.weight")
               if n in uniq)
    tokens = L.valid_tokens(8 * ma["max_seq_len"])
    idx = torch.from_numpy(tokens[:2 * ma["max_seq_len"]]
                           .reshape(2, ma["max_seq_len"]).astype(np.int64))
    recs = k9.read_k9(os.path.join(RES, "eq23_qat_rest10_regrid_n4.k9"))
    widths = {n: int(v.shape[1]) for n, v in uniq.items() if v.dim() == 2}
    artz = np.load(os.path.join(RES, "eq23_qat_rest10_regrid_n4_rest.npz"))
    snaps = [tuple(int(x) for x in g) for g in artz["gains"]]
    gain_vals = [float(m) * 9.0 ** a * 2.0 ** b for (m, a, b) in snaps]
    site_of = {"norm.weight": 2 * ma["n_layers"]}
    for ll in range(ma["n_layers"]):
        site_of[f"layers.{ll}.attention_norm.weight"] = 2 * ll
        site_of[f"layers.{ll}.ffn_norm.weight"] = 2 * ll + 1
    arts, fp32_sd = {}, {}
    for n, rec in recs.items():
        W, Jw, Wd, cc = E17.exact_weight(rec, widths[n])
        arts[n] = (W, Jw, Wd, cc)
        fp32_sd[n] = torch.from_numpy(
            (W[:, :cc].astype(np.float64) * 2.0 ** Jw / Wd).astype(np.float32))
    REST_BITS = 10
    for n, v in uniq.items():
        if v.dim() == 1:
            w64 = v.detach().numpy().astype(np.float64)
            Wal = np.rint(w64 * (1 << REST_BITS)).astype(np.int64)
            fp32_sd[n] = torch.from_numpy((Wal.astype(np.float64)
                                           * 2.0 ** (-REST_BITS)).astype(np.float32))
    fp32_sd[EMB] = fp32_sd["tok_embeddings.weight"]
    fp32_sd["output.weight"] = fp32_sd[EMB]
    arts["output.weight"] = arts[EMB]
    rests = {n: (np.rint(v.detach().numpy().astype(np.float64)
                         * (1 << REST_BITS)).astype(np.int64), -REST_BITS)
             for n, v in uniq.items() if v.dim() == 1}
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    rep = nh // nkv
    pos = torch.arange(T)
    mask = torch.full((T, T), float("-inf")).triu(1)

    def twin_norm(h, w, site):
        s = h.abs().sum(-1, keepdim=True)
        return torch.where(s > 0, h * (dim / s.clamp_min(1e-30)
                                       * gain_vals[site_of[site]]),
                           torch.zeros_like(h)) * w

    def twin_regrid(x):
        vals = x
        mx = vals.abs().amax(-1, keepdim=True)
        e_r = torch.ceil(torch.log(mx.clamp_min(1e-30)) / LOG9)
        mant = torch.round(vals * torch.pow(9.0, N_DIGITS - e_r))
        return mant * torch.pow(9.0, e_r - N_DIGITS)

    twins, twin_scores = {}, {}
    hfp = fp32_sd[EMB][idx].float()
    twins["embed"] = hfp.clone()
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hnT = twin_regrid(twin_norm(hfp, fp32_sd[p + "attention_norm.weight"],
                                    p + "attention_norm.weight"))
        twins[f"L{l}.hn"] = hnT
        q = (hnT @ fp32_sd[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k4 = (hnT @ fp32_sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v4 = (hnT @ fp32_sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q = L.rope(q, pos)
        k4 = L.rope(k4, pos)
        kkr = k4.repeat_interleave(rep, dim=2).transpose(1, 2)
        vvr = v4.repeat_interleave(rep, dim=2).transpose(1, 2)
        sT = (q.transpose(1, 2) @ kkr.transpose(-2, -1)) / hd ** 0.5 + mask
        twin_scores[l] = sT
        o = (sT.softmax(-1) @ vvr).transpose(1, 2).reshape(B, T, dim)
        hfp = hfp + o @ fp32_sd[p + "attention.wo.weight"].T
        twins[f"L{l}.res_mid"] = hfp.clone()
        hn2 = twin_regrid(twin_norm(hfp, fp32_sd[p + "ffn_norm.weight"],
                                    p + "ffn_norm.weight"))
        twins[f"L{l}.hn2"] = hn2
        w1out = hn2 @ fp32_sd[p + "feed_forward.w1.weight"].T
        ff = (w1out * torch.sigmoid(w1out)
              * (hn2 @ fp32_sd[p + "feed_forward.w3.weight"].T))
        ffq = twin_regrid(ff)
        hfp = hfp + ffq @ fp32_sd[p + "feed_forward.w2.weight"].T
        twins[f"L{l}.res_out"] = hfp.clone()
    hnF = twin_regrid(twin_norm(hfp, fp32_sd["norm.weight"], "norm.weight"))
    twins["logits"] = (hnF @ fp32_sd["output.weight"].T).float()
    print(f"[eq24] twin captured ({time.time()-t0:.0f}s)", flush=True)

    site_log = []
    widths_out, widen_events = {}, []

    def delta(tag, ex, tw, scores=False):
        twd = torch.as_tensor(tw).double()
        exd = torch.as_tensor(ex).double()
        if exd.shape != twd.shape:
            if exd.numel() != twd.numel():
                raise RuntimeError(f"{tag}: {tuple(exd.shape)} vs "
                                   f"{tuple(twd.shape)}")
            exd = exd.reshape(twd.shape)
        if scores:
            fin = torch.isfinite(exd) & torch.isfinite(twd)
            dd = torch.where(fin, (exd - twd).abs(),
                             torch.zeros_like(exd))
        else:
            dd = (exd - twd).abs()
        dmax = float(dd.max())
        fl = int((dd > 0.05).sum())
        site_log.append(dict(site=tag, delta=dmax,
                             scale=float(np.nan_to_num(
                                 twd.abs().numpy(), 0.0).max()),
                             flips_gt_0p05=fl))
        print(f"  {tag:>12}: |Δ| {dmax:.3e} "
              f"(scale {site_log[-1]['scale']:.3g}, flips>0.05 {fl})",
              flush=True)

    def go(name, s):
        Wn, Jw, Wd, cw = arts[name]
        if int(Jw) > 0:
            Wn = Wn << int(Jw)
        Wds = int(Wd) * (1 << (-int(Jw)) if int(Jw) < 0 else 1)
        if s.num.dtype != object:
            mx = int(np.abs(s.num).max())
            mw = int(np.abs(Wn).max())
            if mx and mx * mw * s.num.shape[1] * 4 >= I64MAX:
                widen_events.append(name)
                def wd(x):
                    return x.astype(object)
                s = State(wd(s.num), wd(s.den), s.e, s.jj)
                Wn = wd(Wn)
        Wns = Wn[:, :cw] if Wn.dtype != object else \
            np.asarray(Wn, dtype=object)[:, :cw]
        if s.num.dtype != object:
            ynum = s.num @ Wns.T
            yden = s.den * Wds
        else:
            ynum = (s.num[:, :, None] * np.asarray(Wns, dtype=object)
                    .T[None, :, :]).sum(1)
            yden = s.den.astype(object) * Wds
        return State(ynum, yden, s.e, s.jj)

    idx_np = idx.numpy().reshape(-1)
    We, Jwe, Wde, cc = arts[EMB]
    num = We[idx_np][:, :cc]
    if int(Jwe) > 0:
        num = num << int(Jwe)
    den = np.full((num.shape[0], 1),
                  int(Wde) * (1 << (-int(Jwe)) if int(Jwe) < 0 else 1),
                  np.int64)
    s0 = State(num, den, np.zeros((num.shape[0], 1), np.int64),
               np.zeros((num.shape[0], 1), np.int64))
    s0 = renorm_state(s0)
    widths_out["embed"] = (int(np.abs(s0.num).max()).bit_length(),
                           int(np.abs(s0.den).max()).bit_length())
    delta("embed", s0.view(), twins["embed"])
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        resid = s0.copy()
        s0 = manorm_state(s0, rests[p + "attention_norm.weight"],
                          snaps[site_of[p + "attention_norm.weight"]])
        s0 = regrid_state(s0)
        widths_out[f"L{l}.attnorm"] = (int(np.abs(s0.num).max()).bit_length(),
                                       int(np.abs(s0.den).max()).bit_length())
        delta(f"L{l}.attnorm", s0.view(), twins[f"L{l}.hn"])
        q_s = go(p + "attention.wq.weight", s0)
        k_s = go(p + "attention.wk.weight", s0)
        v_s = go(p + "attention.wv.weight", s0)
        qf = q_s.view().reshape(B, T, nh, hd).float()
        kf = k_s.view().reshape(B, T, nkv, hd).float()
        vf = v_s.view().reshape(B, T, nkv, hd).float()
        qf = L.rope(qf, pos)
        kf = L.rope(kf, pos)
        kkr = kf.repeat_interleave(rep, dim=2).transpose(1, 2)
        vvr = vf.repeat_interleave(rep, dim=2).transpose(1, 2)
        s = (qf.transpose(1, 2) @ kkr.transpose(-2, -1)) / hd ** 0.5 + mask
        o = (s.softmax(-1) @ vvr).transpose(1, 2).reshape(B, T, dim)
        if compare_twin(l):
            delta(f"L{l}.scores", s, twin_scores[l], scores=True)
        onum = np.rint(o.reshape(-1, dim).numpy().astype(np.float64)
                       * (1 << FIXED)).astype(np.int64)
        woState = State(onum, np.ones((onum.shape[0], 1), np.int64),
                        np.zeros((onum.shape[0], 1), np.int64),
                        np.full((onum.shape[0], 1), -FIXED, np.int64))
        wo_out = go(p + "attention.wo.weight", woState)
        s0 = add_state(resid, wo_out, f"{p}res1", widen_events)
        if check(l):
            delta(f"L{l}.res_mid", s0.view(), twins[f"L{l}.res_mid"])
            widths_out[f"L{l}.res_mid"] = (
                int(np.abs(s0.num).max()).bit_length()
                if s0.num.dtype != object else -1,
                int(np.abs(s0.den).max()).bit_length()
                if s0.den.dtype != object else -1)
        resid2 = s0.copy()
        s0 = manorm_state(s0, rests[p + "ffn_norm.weight"],
                          snaps[site_of[p + "ffn_norm.weight"]])
        s0 = regrid_state(s0)
        if check(l):
            widths_out[f"L{l}.ffnorm"] = (
                int(np.abs(s0.num).max()).bit_length(),
                int(np.abs(s0.den).max()).bit_length())
            delta(f"L{l}.ffnorm", s0.view(), twins[f"L{l}.hn2"])
        w1_s = go(p + "feed_forward.w1.weight", s0)
        w3_s = go(p + "feed_forward.w3.weight", s0)
        w1v = w1_s.view().float()
        siln = np.rint((w1v * torch.sigmoid(w1v))
                       .numpy().astype(np.float64) * (1 << FIXED)).astype(np.int64)
        if check(l):
            w1outT = twins[f"L{l}.hn2"] \
                @ fp32_sd[p + "feed_forward.w1.weight"].T
            twin_sil = (w1outT * torch.sigmoid(w1outT)).reshape(-1, w1outT.shape[-1])
            delta(f"L{l}.w1", torch.from_numpy(siln.astype(np.float64)
                                               / (1 << FIXED)), twin_sil)
        # the trained semantics regrids the FULL product (eq23's STE);
        # mirror it in the exact track too, then w2
        prod = State(mul_int(siln, w3_s.num, f"{p}ffprod", widen_events),
                     w3_s.den, w3_s.e, w3_s.jj - FIXED)
        prod = regrid_state(prod)
        w2_out = go(p + "feed_forward.w2.weight", prod)
        if check(l):
            w2_tw = (twins[f"L{l}.res_out"] - twins[f"L{l}.res_mid"])
            delta(f"L{l}.w2-out", w2_out.view(), w2_tw)
            if l == 0:
                vv = w2_out.view().reshape(2, 512, dim)
                tvt = w2_tw.reshape(2, 512, dim)
                for i in (0, 1):
                    print(f"    w2 row {i}: exact {vv[i, 0, :4].tolist()}"
                          f" | twin {tvt[i, 0, :4].tolist()}", flush=True)
                ps = prod.view().reshape(2, 512, -1)
                print(f"    prod row0: {ps[0, 0, :4].tolist()}", flush=True)
            widths_out[f"L{l}.w2"] = (
                int(np.abs(w2_out.num).max()).bit_length()
                if w2_out.num.dtype != object else -1,
                int(np.abs(w2_out.den).max()).bit_length()
                if w2_out.den.dtype != object else -1)
        s0 = add_state(resid2, w2_out, f"{p}res2", widen_events)
        if check(l):
            delta(f"L{l}.res_out", s0.view(), twins[f"L{l}.res_out"])
            widths_out[f"L{l}.res_out"] = (
                int(np.abs(s0.num).max()).bit_length()
                if s0.num.dtype != object else -1,
                int(np.abs(s0.den).max()).bit_length()
                if s0.den.dtype != object else -1)
    s0 = manorm_state(s0, rests["norm.weight"], snaps[site_of["norm.weight"]])
    s0 = regrid_state(s0)
    head_out = go("output.weight", s0)
    logo = head_out.view().reshape(B, T, -1).float()
    delta("logits", logo, twins["logits"])
    widths_out["logits"] = (int(np.abs(head_out.num).max()).bit_length()
                            if head_out.num.dtype != object else -1,
                            int(np.abs(head_out.den).max()).bit_length()
                            if head_out.den.dtype != object else -1)
    digits = {tag: dict(num_digits=round(nb / LOG2_9, 1),
                        den_digits=round(db / LOG2_9, 1))
              for tag, (nb, db) in widths_out.items() if nb >= 0}
    worst_num = max(v["num_digits"] for v in digits.values())
    worst_den = max(v["den_digits"] for v in digits.values())
    bounded = bool(worst_num <= U128_DIGITS and worst_den <= U128_DIGITS)
    print(f"[eq24] BOUNDEDNESS: worst num {worst_num} / den {worst_den} "
          f"digits (u128 = {U128_DIGITS}) -> "
          f"{'BOUNDED' if bounded else 'NOT BOUNDED'}")
    out = dict(widths_digits=digits, bounded=dict(worst_num=worst_num,
                                                  worst_den=worst_den,
                                                  u128_digits=U128_DIGITS,
                                                  bounded=bounded),
               sites=site_log, widen_events=widen_events)
    with open(os.path.join(RES, "eq24_regrid_runtime.json"), "w") as f2:
        json.dump(out, f2, indent=2, default=str)
    print(f"[eq24] wrote results/eq24_regrid_runtime.json ({time.time()-t0:.0f}s)")


def check(l):
    return True


def compare_twin(l):
    return True


if __name__ == "__main__":
    main()