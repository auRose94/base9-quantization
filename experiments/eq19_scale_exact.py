#!/usr/bin/env python3
"""eq19 — RQ10 stage B: the (num, den, E) hybrid carrier with the scale-norm.

Measured sequence (all logged 2026-10-07):
  * (num, den, E, J) with full 2/9 track-extraction: the negative-J drift
    forces num ~ value·den·2^{|J|} — num-carriers ~138 digits (inflated).
  * plain (num, den): the scale-norm's 9^K multiplies land in the den and
    the adds lcm the odd parts — carriers ~228 digits.
  * THIS design — the hybrid: the 9-parts (the scale-norm's K-shifts and the
    dens' 3-powers) ride a per-row E-track (free: pure shifts); the 2-parts
    stay in the den as plain integers (gcd-reduce after every site shares
    them); no J track. Expected: both carriers bounded near value-scale +
    the LCM'd odd-denominator product.

State: value = num·9^{E} / den  (num (n,dim), den (n,1), E (n,1)), ints.
  matmul: ynum = Σ num·Wnum; yden = den·Wdscale; Jw>0 folds into Wnum
      (exact int shift), Jw<0 into Wdscale.
  scale-norm: value·9^{−K} → E −= K (K = floor(log9 max_c |value_c|));
      per-channel rest = num·wn, den·2^6.
  attention/silu: fp-mixed with declared 2^30 intakes.
  adds: exact lcm per row (9-alignment via the E-track); renorm: per-row
      gcd + the den's 9-powers pulled into E.

Fidelity vs the fp twin (same semantics) + the score-flip census resolves
the old drift; P38 = order-permutation bit-identity.
Subject: eq9_QAT_k27emb99.k9, 2×512 tokens, CPU.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/mnt/matrix/Work/base9-quantization/experiments")
import eq_lib as L  # noqa: E402
import k9  # noqa: E402
import eq17_exact_runtime as E17  # noqa: E402

RES = L.RESULTS
REST_BITS = 6
FIXED = E17.FIXED
LOG9 = float(np.log(9.0))
LOG2_9 = float(math.log2(9.0))
U128_DIGITS = 40.4
I64MAX = 1 << 62


def widen(a):
    return a.astype(object)


def view2(num, den, E):
    if num.dtype == object:
        val = np.empty(num.shape, dtype=np.float64)
        for i in range(num.shape[0]):
            s = 3.0 ** (2 * int(E[i, 0])) / (int(den[i, 0]) or 1)
            for j in range(num.shape[1]):
                val[i, j] = float(int(num[i, j])) * s
        return torch.from_numpy(val)
    return torch.from_numpy(num.astype(np.float64)
                            * np.power(9.0, E.astype(np.float64))
                            / den.astype(np.float64))


def renorm3(num, den, E):
    """per-row gcd + the den's 9-powers pulled into E (value preserved)."""
    if num.dtype != object:
        gg = np.gcd.reduce(np.abs(num), axis=1, dtype=np.int64).reshape(-1, 1)
        gg = np.gcd(gg, den)
        gg = np.where(gg == 0, 1, gg)
        num2 = num // gg
        den2 = den // gg
    else:
        from math import gcd
        gg = np.empty((num.shape[0], 1), dtype=object)
        for i in range(num.shape[0]):
            g = abs(int(den[i, 0]))
            for v in num[i]:
                g = gcd(g, abs(int(v)))
                if g == 1:
                    break
            gg[i, 0] = g or 1
        num2 = num // gg
        den2 = den // gg
    t = den2.copy()
    dE = np.zeros_like(den2)
    if den2.dtype != object:
        while (t % 9 == 0).any():
            m9 = t % 9 == 0
            dE[m9] += 1
            t[m9] //= 9
    else:
        for i in range(t.shape[0]):
            v = int(t[i, 0])
            while v and v % 9 == 0:
                dE[i, 0] += 1
                v //= 9
            t[i, 0] = v
    return num2, t, E - dE


def mul_safe(den, factor):
    f = np.asarray(factor)
    m = int(np.abs(den).max()) if den.size else 0
    fm = int(np.abs(f).max()) if f.size else 1
    if den.size and den.dtype != object and m * fm >= I64MAX:
        return den.astype(object) * f.astype(object)
    return den * f


def matmul3(num, den, E, Wnum, Wdscale, tag, order=None, audit=None):
    Wn = widen(Wnum) if Wnum.dtype == object else Wnum
    if num.dtype != object:
        mx = int(np.abs(num).max())
        mw = int(np.abs(Wn).max())
        if mx and mx * mw * num.shape[1] * 4 >= I64MAX:
            (num, den, Wn, Wdscale) = (widen(num), widen(den), widen(Wn),
                                       int(Wdscale))
            if audit is not None:
                audit.append(tag)
    if order is None:
        if num.dtype != object:
            ynum = num @ Wn.T
        else:
            ynum = (num[:, :, None] * Wn.T.astype(object)[None, :, :]).sum(1)
    else:
        h = num.shape[1] // 2
        if num.dtype != object:
            ynum = (num[:, h:] @ Wn[:, h:].T) + (num[:, :h] @ Wn[:, :h].T)
        else:
            Wt = Wn.T.astype(object)
            ynum = ((num[:, h:, None] * Wt[None, h:, :]).sum(1)
                    + (num[:, :h, None] * Wt[None, :h, :]).sum(1))
    return ynum, mul_safe(den, Wdscale), E


def add3(num1, den1, E1, num2, den2, E2, tag, audit):
    if num1.dtype != object and num2.dtype != object:
        lmax = int(np.abs(den1).max()) * int(np.abs(den2).max())
        if lmax >= I64MAX:
            audit.append(tag)
            (num1, den1, E1) = (widen(num1), widen(den1), widen(E1))
            (num2, den2, E2) = (widen(num2), widen(den2), widen(E2))
    from math import gcd
    n = num1.shape[0]
    out_num = np.empty(num1.shape, dtype=object)
    out_den = np.empty((n, 1), dtype=object)
    out_E = np.empty((n, 1), dtype=object)
    for i in range(n):
        d1 = int(den1[i, 0])
        d2 = int(den2[i, 0])
        g = gcd(d1, d2)
        lcm = (d1 // g) * d2
        Em = min(int(E1[i, 0]), int(E2[i, 0]))
        a1 = (lcm // d1) * (9 ** (int(E1[i, 0]) - Em) if int(E1[i, 0]) > Em
                            else 1)
        a2 = (lcm // d2) * (9 ** (int(E2[i, 0]) - Em) if int(E2[i, 0]) > Em
                            else 1)
        row1 = num1[i].astype(object)
        row2 = num2[i].astype(object)
        out_num[i] = row1 * a1 + row2 * a2
        out_den[i, 0] = lcm
        out_E[i, 0] = Em
    return renorm3(out_num, out_den, out_E)


def scalenorm_exact(num, den, E, wn, Jw):
    """exact scale-norm: E −= K (per-position 9-reset; free), then the
    per-channel 6-bit grid rest (num·wn, den·2^6) and renorm."""
    vals = view2(num, den, E).numpy()
    mx = np.abs(vals).max(axis=1, keepdims=True)
    K = np.floor(np.log(np.maximum(mx, 1e-30)) / LOG9).astype(np.int64)
    E = E - K
    num = num * (wn.astype(object)[None, :] if num.dtype == object
                 else wn[None, :])
    den = mul_safe(den, np.full_like(den, 1 << REST_BITS))
    return renorm3(num, den, E)


def main():
    t0 = time.time()
    torch.manual_seed(0)
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    EMB = next(n for n in ("tok_embeddings.weight", "output.weight") if n in uniq)
    tokens = L.valid_tokens(8 * ma["max_seq_len"])
    idx = torch.from_numpy(tokens[:2 * ma["max_seq_len"]]
                           .reshape(2, ma["max_seq_len"]).astype(np.int64))
    recs = k9.read_k9(os.path.join(RES, "eq9_QAT_k27emb99.k9"))
    widths = {n: int(v.shape[1]) for n, v in uniq.items() if v.dim() == 2}
    arts, fp32_sd = {}, {}
    for n, rec in recs.items():
        W, Jw, Wd, cc = E17.exact_weight(rec, widths[n])
        arts[n] = (W, Jw, Wd, cc)
        fp32_sd[n] = torch.from_numpy(
            (W[:, :cc].astype(np.float64) * 2.0 ** Jw / Wd).astype(np.float32))
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

    def scalenorm_fp(h, w):
        mx = h.abs().amax(-1, keepdim=True)
        K = torch.floor(torch.log(mx.clamp_min(1e-30)) / LOG9)
        return h * torch.pow(9.0, -K) * w

    def silr(x):
        return x * (1.0 + x / (x.abs() + 1.0)) * 0.5

    twins, twin_scores = {}, {}
    hfp = fp32_sd[EMB][idx].float()
    twins["embed"] = hfp.clone()
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hnT = scalenorm_fp(hfp, fp32_sd[p + "attention_norm.weight"])
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
        hn2 = scalenorm_fp(hfp, fp32_sd[p + "ffn_norm.weight"])
        twins[f"L{l}.hn2"] = hn2
        w1out = hn2 @ fp32_sd[p + "feed_forward.w1.weight"].T
        ff = silr(w1out) * (hn2 @ fp32_sd[p + "feed_forward.w3.weight"].T)
        hfp = hfp + ff @ fp32_sd[p + "feed_forward.w2.weight"].T
        twins[f"L{l}.res_out"] = hfp.clone()
    hnF = scalenorm_fp(hfp, fp32_sd["norm.weight"])
    twins["logits"] = (hnF @ fp32_sd["output.weight"].T).float()
    print(f"[eq19] twin captured ({time.time()-t0:.0f}s)", flush=True)

    site_log = []

    def delta(tag, ex, tw, scores=False):
        twd = torch.nan_to_num(torch.as_tensor(tw).double(), nan=0.0,
                               posinf=1.7e308, neginf=-1.7e308)
        exd = torch.nan_to_num(torch.as_tensor(ex).double(), nan=0.0,
                               posinf=1.7e308, neginf=-1.7e308)
        if exd.shape != twd.shape:
            if exd.numel() != twd.numel():
                raise RuntimeError(f"{tag}: {tuple(exd.shape)} vs {tuple(twd.shape)}")
            exd = exd.reshape(twd.shape)
        d = (exd - twd).abs()
        dmax = float(d.max())
        fl = int((d > 0.05).sum())
        site_log.append(dict(site=tag, delta=dmax, scale=float(twd.abs().max()),
                             flips_gt_0p05=fl))
        print(f"  {tag:>12}: |Δ| {dmax:.3e} (scale {site_log[-1]['scale']:.3g}, "
              f"flips>0.05 {fl})", flush=True)

    widths_nat, widen_all = {}, []

    def go(name, num_, den_, E_):
        Wn, Jw, Wd, cw = arts[name]
        if int(Jw) > 0:
            Wn = Wn << int(Jw)
        Wds = int(Wd) * (1 << (-int(Jw)) if int(Jw) < 0 else 1)
        return matmul3(num_, den_, E_, Wn[:, :cw], Wds, name, order=None,
                       audit=widen_all)

    def exact_run(order=None, compare_twin=True):
        widths_out = {}
        idx_np = idx.numpy().reshape(-1)
        We, Jwe, Wde, cc = arts[EMB]
        num = We[idx_np][:, :cc]
        if int(Jwe) > 0:
            num = num << int(Jwe)
        den = np.full((num.shape[0], 1),
                      int(Wde) * (1 << (-int(Jwe)) if int(Jwe) < 0 else 1),
                      np.int64)
        E = np.zeros((num.shape[0], 1), np.int64)
        num, den, E = renorm3(num, den, E)
        widths_out["embed"] = (int(np.abs(num).max()).bit_length(),
                               int(np.abs(den).max()).bit_length())
        if compare_twin:
            delta("embed", view2(num, den, E).reshape(B, T, dim), twins["embed"])
        for l in range(ma["n_layers"]):
            p = f"layers.{l}."
            resid = (num.copy(), den.copy(), E.copy())
            wn, Jw = rests[p + "attention_norm.weight"]
            num, den, E = scalenorm_exact(num, den, E, wn, Jw)
            if compare_twin:
                delta(f"L{l}.attnorm", view2(num, den, E), twins[f"L{l}.hn"])
            qn2, qd2, qE2 = go(p + "attention.wq.weight", num, den, E)
            kn2, kd2, kE2 = go(p + "attention.wk.weight", num, den, E)
            vn2, vd2, vE2 = go(p + "attention.wv.weight", num, den, E)
            qf = view2(qn2, qd2, qE2).reshape(B, T, nh, hd).float()
            kf = view2(kn2, kd2, kE2).reshape(B, T, nkv, hd).float()
            vf = view2(vn2, vd2, vE2).reshape(B, T, nkv, hd).float()
            qf = L.rope(qf, pos)
            kf = L.rope(kf, pos)
            kkr = kf.repeat_interleave(rep, dim=2).transpose(1, 2)
            vvr = vf.repeat_interleave(rep, dim=2).transpose(1, 2)
            s = (qf.transpose(1, 2) @ kkr.transpose(-2, -1)) / hd ** 0.5 + mask
            o = (s.softmax(-1) @ vvr).transpose(1, 2).reshape(B, T, dim)
            if compare_twin:
                delta(f"L{l}.scores", s, twin_scores[l], scores=True)
            onum = np.rint(o.reshape(-1, dim).numpy().astype(np.float64)
                           * (1 << FIXED)).astype(np.int64)
            # the intake's own state: (o·2^30)/(2^30) — a CONSTANT den, NOT
            # the stream's den (that leak was the missing o·W term)
            oren = np.full((onum.shape[0], 1), 1 << FIXED, np.int64)
            oE = np.zeros_like(E)
            onn, odn, oEn = go(p + "attention.wo.weight", onum, oren, oE)
            num, den, E = add3(resid[0], resid[1], resid[2], onn, odn, oEn,
                               f"{p}res1", widen_all)
            if compare_twin:
                delta(f"L{l}.res_mid", view2(num, den, E), twins[f"L{l}.res_mid"])
                widths_out[f"L{l}.res_mid"] = (
                    int(np.abs(num).max()).bit_length()
                    if num.dtype != object else -1,
                    int(np.abs(den).max()).bit_length()
                    if den.dtype != object else -1)
            resid2 = (num.copy(), den.copy(), E.copy())
            wn, Jw = rests[p + "ffn_norm.weight"]
            num, den, E = scalenorm_exact(num, den, E, wn, Jw)
            if compare_twin:
                delta(f"L{l}.ffnorm", view2(num, den, E), twins[f"L{l}.hn2"])
            w1n2, w1d2, w1E2 = go(p + "feed_forward.w1.weight", num, den, E)
            w3n2, w3d2, w3E2 = go(p + "feed_forward.w3.weight", num, den, E)
            sil = view2(w1n2, w1d2, w1E2).float()
            siln = np.rint((sil * (1.0 + sil / (sil.abs() + 1.0)) * 0.5)
                           .numpy().astype(np.float64) * (1 << FIXED)).astype(np.int64)
            w2n, Jw2, Wd2, wc2 = arts[p + "feed_forward.w2.weight"]
            prod = siln * (w3n2 if w3n2.dtype != object
                           else w3n2.astype(object))
            prod_den = mul_safe(w3d2, np.full_like(w3d2, 1 << FIXED))
            o2n, o2d, o2E = go_w2(w2n, Jw2, Wd2, wc2, prod, prod_den, w3E2,
                                  order=order)
            num, den, E = add3(resid2[0], resid2[1], resid2[2], o2n, o2d,
                               o2E, f"{p}res2", widen_all)
            if compare_twin:
                delta(f"L{l}.res_out", view2(num, den, E), twins[f"L{l}.res_out"])
                widths_out[f"L{l}.res_out"] = (
                    int(np.abs(num).max()).bit_length()
                    if num.dtype != object else -1,
                    int(np.abs(den).max()).bit_length()
                    if den.dtype != object else -1)
        wn, Jw = rests["norm.weight"]
        num, den, E = scalenorm_exact(num, den, E, wn, Jw)
        on2, od2, oE2 = go("output.weight", num, den, E)
        logo = view2(on2, od2, oE2).reshape(B, T, -1).float()
        if compare_twin:
            delta("logits", logo, twins["logits"])
        return logo, widths_out

    def go_w2(Wn, Jw, Wd, cw, num_, den_, E_, order=None):
        if int(Jw) > 0:
            Wn = Wn << int(Jw)
        Wds = int(Wd) * (1 << (-int(Jw)) if int(Jw) < 0 else 1)
        return matmul3(num_, den_, E_, Wn[:, :cw], Wds, "w2", order, widen_all)

    def go2(Wn, Jw, Wd, cw, num_, den_, E_):
        if int(Jw) > 0:
            Wn = Wn << int(Jw)
        Wds = int(Wd) * (1 << (-int(Jw)) if int(Jw) < 0 else 1)
        return matmul3(num_, den_, E_, Wn[:, :cw], Wds, "mm",
                       order, widen_all)

    logo_nat, widths_nat = exact_run(None, compare_twin=True)
    logo_rev, widths_rev = exact_run("split_rev", compare_twin=False)
    agree = torch.equal(torch.nan_to_num(logo_nat), torch.nan_to_num(logo_rev))

    digits = {tag: dict(num_digits=round(nb / LOG2_9, 1),
                        den_digits=round(db / LOG2_9, 1))
              for tag, (nb, db) in widths_nat.items() if nb >= 0}
    worst_num = max(v["num_digits"] for v in digits.values())
    worst_den = max(v["den_digits"] for v in digits.values())
    bounded = bool(worst_num <= U128_DIGITS and worst_den <= U128_DIGITS)
    print(f"[eq19] BOUNDEDNESS: worst num {worst_num} / den {worst_den} digits "
          f"(u128 = {U128_DIGITS}) -> {'BOUNDED' if bounded else 'NOT BOUNDED'}; "
          f"order-identical: {agree}")

    out = dict(widths_digits=digits,
               bounded=dict(worst_num=worst_num, worst_den=worst_den,
                            u128_digits=U128_DIGITS, bounded=bounded),
               sites=site_log, order_identical=bool(agree),
               wide_events=widen_all,
               wall_s=round(time.time() - t0, 1))
    with open(os.path.join(RES, "eq19_scale_exact.json"), "w") as f2:
        json.dump(out, f2, indent=2, default=str)
    md = ["# eq19 — the (num, den, E) hybrid carrier with the scale-norm\n",
          f"subject: eq9_QAT_k27emb99.k9, 2×512; order-identical: {agree}; "
          f"widen events: {widen_all or 'none'}\n",
          f"BOUNDEDNESS: worst num {worst_num} / den {worst_den} digits "
          f"(u128 = {U128_DIGITS}) — "
          + ("BOUNDED." if bounded else "not bounded."),
          "", "| site | |Δ| vs twin | scale | flips>0.05 |", "|---|---|---|---|"]
    for s in site_log:
        md.append(f"| {s['site']} | {s['delta']:.3e} | {s['scale']:.3g} | "
                  f"{s['flips_gt_0p05']} |")
    with open(os.path.join(RES, "eq19_scale_exact.md"), "w") as f3:
        f3.write("\n".join(md) + "\n")
    print(f"[eq19] wrote results/eq19_scale_exact.{{json,md}} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()