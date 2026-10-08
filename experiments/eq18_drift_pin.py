#!/usr/bin/env python3
"""eq18 — RQ10 stage B open item 1: pin the exact-vs-fp32ref logits drift.

The 2026-10-07 readout: |Δlogits| ≈ 21–24 (ref scale ~21) while L0 is
faithful site-by-site (1e-3–1e-4, fp32 class). This walk mirrors the
module's `eq17_exact_forward` op-for-op with capture and, at EVERY site,
compares the state's float view against the fp twin's matching tensor:

  per-site |Δ| table (embed → attnorm → q/k/v(rope) → attention-out →
  wo intake → res1 → ffn-norm → w1/w3 → silu-prod → w2 + res2 → normF →
  logits), plus:
  * attention flip census: |Δs| > 0.05 count per layer with the
    first-incident layer/position (the chaos discriminator);
  * the drift's growth signature: if |Δ| stays fp-class until some site and
    then jumps O(1), the jump's site name pins the defect/op; if it creeps
    ×(few)/layer everywhere, it is compounding sensitivity and the
    fp-mixed boundary is (harmlessly) the seed.

Verdict target (not pre-registered — this is the registered open item):
a named site + mechanism, written into the log.
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
from eq16_substituted import calibrate_lams  # noqa
import eq17_exact_runtime as E17  # noqa: E402

RES = L.RESULTS
GROUP = 64


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
            fp32_sd[n] = v.half().float()
    fp32_sd[EMB] = fp32_sd["tok_embeddings.weight"]
    fp32_sd["output.weight"] = fp32_sd[EMB]
    arts["output.weight"] = arts[EMB]
    lams = calibrate_lams(fp32_sd, ma)
    lam_pairs = [(int(round(x * (1 << 10))), 1 << 10) for x in lams]
    rests = {n: E17.exact_rest(v) for n, v in uniq.items() if v.dim() == 1}

    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    rep = nh // nkv
    pos = torch.arange(T)
    mask = torch.full((T, T), float("-inf")).triu(1)

    def normsub(x, w, lam):
        s = x.abs().sum(-1, keepdim=True)
        return torch.where(s > 0, x * (dim / s.clamp_min(1e-30) * lam),
                           torch.zeros_like(x)) * w

    def silr(x):
        return x * (1.0 + x / (x.abs() + 1.0)) * 0.5

    # ---------------- parallel walks ----------------
    idx_np = idx.numpy().reshape(-1)
    We, Jwe, Wde, _ = arts[EMB]
    num = We[idx_np][:, :cc]
    den = np.full((num.shape[0], 1), Wde, np.int64)
    E = np.zeros((num.shape[0], 1), np.int64)
    J = np.full((num.shape[0], 1), Jwe, np.int64)
    num, den, E, J = E17.renorm(num, den, E, J, "embed")
    h_exp = torch.from_numpy(E17.to_fp(num, den, E, J).numpy())
    h_fpp = fp32_sd[EMB][idx].float()
    table = []
    flips = []

    def delta(tag, a_ex, a_fp):
        """shape-safe |Δ| record; broadcast-flattened defensively."""
        if a_ex.shape != a_fp.shape:
            # try the flatten-compare only when sizes match
            assert a_ex.numel() == a_fp.numel(), f"{tag}: {a_ex.shape} vs {a_fp.shape}"
            a_ex = a_ex.reshape(a_fp.shape)
        d = (a_ex.double() - a_fp.double()).abs()
        dmax = float(np.nanmax(d.numpy())) if not torch.isnan(d).all() else float("nan")
        sc = float(np.nanmax(a_fp.abs().numpy())) or 1.0
        sc_ex = float(np.nanmax(a_ex.abs().numpy())) or 1.0
        table.append(dict(site=tag, delta=dmax, scale=sc, scale_exact=sc_ex,
                          rel=round(dmax / sc, 6),
                          nan_in_exact=int(torch.isnan(a_ex).sum())))
        print(f"  {tag:>12}: |Δ| {dmax:.4e} (fp {sc:.3f}, exact {sc_ex:.3f}, "
              f"rel {dmax/sc:.2e})", flush=True)
        return d

    delta("embed", h_exp, h_fpp)
    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        resid_keep = (num.copy(), den.copy(), E.copy(), J.copy())
        # attnorm into a separate carrier: the residual stays the pre-norm
        # stream (the rebind pattern)
        lna, lda = lam_pairs[2 * l]
        W1a, Jw1a = rests[p + "attention_norm.weight"]
        num, den, E, J = E17.renorm(num * (64 * lna) * W1a[None, :],
                                    lda * np.abs(num).sum(1, keepdims=True),
                                    np.zeros_like(E), np.full_like(E, Jw1a),
                                    f"{p}an")
        hn_ex = torch.from_numpy(E17.to_fp(num, den, E, J).numpy()).float()
        hn_fp = normsub(h_fpp, fp32_sd[p + "attention_norm.weight"], lams[2 * l])
        delta(f"L{l}.attnorm", hn_ex, hn_fp)
        # q/k/v (fp twin)
        q_fp = L.rope((hn_fp @ fp32_sd[p + "attention.wq.weight"].T)
                      .view(B, T, nh, hd), pos)
        k_fpv = L.rope((hn_fp @ fp32_sd[p + "attention.wk.weight"].T)
                       .view(B, T, nkv, hd), pos).repeat_interleave(rep, dim=2)
        v_fpv = (hn_fp @ fp32_sd[p + "attention.wv.weight"].T
                 ).view(B, T, nkv, hd).repeat_interleave(rep, dim=2)
        # attention scores + out
        qe4 = E17.to_fp(*E17.matmul_exact(num, den, E, J,
                                          arts[p + "attention.wq.weight"][0][:, :64],
                                          arts[p + "attention.wq.weight"][1],
                                          arts[p + "attention.wq.weight"][2], "q")
                        ).reshape(B, T, nh, hd).float()
        qe4 = L.rope(qe4, pos).transpose(1, 2)
        ke4 = L.rope(E17.to_fp(*E17.matmul_exact(
            num, den, E, J, arts[p + "attention.wk.weight"][0][:, :64],
            arts[p + "attention.wk.weight"][1],
            arts[p + "attention.wk.weight"][2], "k")
        ).reshape(B, T, nkv, hd).float(), pos).repeat_interleave(rep, dim=2).transpose(1, 2)
        ve4 = E17.to_fp(*E17.matmul_exact(
            num, den, E, J, arts[p + "attention.wv.weight"][0][:, :64],
            arts[p + "attention.wv.weight"][1],
            arts[p + "attention.wv.weight"][2], "v")
        ).reshape(B, T, nkv, hd).float().repeat_interleave(rep, dim=2).transpose(1, 2)
        s_ex = (qe4 @ ke4.transpose(-2, -1)) / hd ** 0.5 + mask
        s_fp = (q_fp.transpose(1, 2) @ k_fpv.transpose(1, 2).transpose(-2, -1)) \
            / hd ** 0.5 + mask
        d_scores = (s_ex - s_fp).abs()
        flips.append(dict(layer=l,
                          flips_gt_0p05=int((d_scores > 0.05).sum()),
                          max_dscore=float(np.nanmax(d_scores.numpy())),
                          mean_dscore=float(np.nanmean(np.abs(s_fp.numpy() - s_ex.numpy())))))
        o_ex = (s_ex.softmax(-1) @ ve4).transpose(1, 2).reshape(B, T, dim)
        o_fp = (s_fp.softmax(-1) @ v_fpv.transpose(1, 2)).transpose(1, 2).reshape(B, T, dim)
        delta(f"L{l}.att-out", o_ex, o_fp)
        # wo + residual
        onum = np.rint(o_ex.reshape(-1, dim).numpy().astype(np.float64)
                       * (1 << E17.FIXED)).astype(np.int64)
        ones = np.full((onum.shape[0], 1), 1, np.int64)
        oEz = np.zeros((onum.shape[0], 1), np.int64)
        oJz = np.full((onum.shape[0], 1), -E17.FIXED, np.int64)
        on, Jw, Wd, cw = arts[p + "attention.wo.weight"]
        onn, odn, oEn, oJn = E17.matmul_exact(onum, ones, oEz, oJz, on[:, :cw],
                                              Jw, Wd, f"{p}wo")
        o_view = E17.to_fp(onn, odn, oEn, oJn)
        wo_fpv = (o_fp @ fp32_sd[p + "attention.wo.weight"].T)
        delta(f"L{l}.wo", o_view.reshape(-1, dim), wo_fpv.reshape(-1, dim))
        num, den, E, J = E17.add_exact(resid_keep[0], resid_keep[1],
                                       resid_keep[2], resid_keep[3],
                                       onn, odn, oEn, oJn, f"{p}res1")
        vres = torch.from_numpy(E17.to_fp(num, den, E, J).numpy()).float()
        h_fpp = h_fpp + wo_fpv.reshape(B, T, dim)
        delta(f"L{l}.res_mid", vres.reshape(B, T, dim), h_fpp)
        resid2 = (num.copy(), den.copy(), E.copy(), J.copy())
        # ffn norm into a separate carrier; res2 restores the residual
        # ffn
        ln_, ld_ = lam_pairs[2 * l + 1]
        W1n, Jw1 = rests[p + "ffn_norm.weight"]
        num, den, E, J = E17.renorm(num * (64 * ln_) * W1n[None, :],
                                    ld_ * np.abs(num).sum(1, keepdims=True),
                                    np.zeros_like(E), np.full_like(E, Jw1),
                                    f"{p}fn")
        hn2_ex = torch.from_numpy(E17.to_fp(num, den, E, J).numpy()).float()
        hn2_fp = normsub(h_fpp, fp32_sd[p + "ffn_norm.weight"], lams[2 * l + 1])
        delta(f"L{l}.ffnorm", hn2_ex, hn2_fp)
        w1n, Jw1i, Wd1, wc1 = arts[p + "feed_forward.w1.weight"]
        w3n, Jw3, Wd3, wc3 = arts[p + "feed_forward.w3.weight"]
        w1n2, w1d2, w1E2, w1J2 = E17.matmul_exact(num, den, E, J, w1n[:, :wc1],
                                                  Jw1i, Wd1, f"{p}w1")
        w3n2, w3d2, w3E2, w3J2 = E17.matmul_exact(num, den, E, J, w3n[:, :wc3],
                                                  Jw3, Wd3, f"{p}w3")
        sil_ex = torch.from_numpy(E17.to_fp(w1n2, w1d2, w1E2, w1J2).numpy())
        sil_fp = w1out = hn2_fp @ fp32_sd[p + "feed_forward.w1.weight"].T
        delta(f"L{l}.w1", sil_ex, sil_fp)
        prod_ex = (silr(sil_ex) * (hn2_ex @ fp32_sd[p + "feed_forward.w3.weight"].T))
        prod_fp = silr(sil_fp) * (hn2_fp @ fp32_sd[p + "feed_forward.w3.weight"].T)
        delta(f"L{l}.ffprod", prod_ex, prod_fp)
        w2n, Jw2, Wd2, wc2 = arts[p + "feed_forward.w2.weight"]
        siln = np.rint(silr(sil_ex).numpy().astype(np.float64)
                       * (1 << E17.FIXED)).astype(np.int64)
        o2n, o2d, o2E, o2J = E17.matmul_exact(siln * w3n2, w3d2, w3E2,
                                              w3J2 - E17.FIXED, w2n[:, :wc2],
                                              Jw2, Wd2, f"{p}w2")
        num, den, E, J = E17.add_exact(resid2[0], resid2[1], resid2[2],
                                       resid2[3], o2n, o2d, o2E, o2J,
                                       f"{p}res2")
        vres = torch.from_numpy(E17.to_fp(num, den, E, J).numpy()).float()
        ff_fp = silr(sil_fp) * (hn2_fp @ fp32_sd[p + "feed_forward.w3.weight"].T)
        h_fpp = h_fpp + (ff_fp @ fp32_sd[p + "feed_forward.w2.weight"].T)
        delta(f"L{l}.res_out", vres.reshape(B, T, dim), h_fpp)

    # final norm + head
    lnf, ldf = lam_pairs[2 * ma["n_layers"]]
    Wfn, Jwf = rests["norm.weight"]
    num, den, E, J = E17.renorm(num * (64 * lnf) * Wfn[None, :],
                                ldf * np.abs(num).sum(1, keepdims=True),
                                np.zeros_like(E), np.full_like(E, Jwf), "normF")
    hnn, JwH, WdH, hc = arts["output.weight"]
    on2, od2, oE2, oJ2 = E17.matmul_exact(num, den, E, J, hnn[:, :hc], JwH,
                                          WdH, "head")
    logo = E17.to_fp(on2, od2, oE2, oJ2).float()
    hnF = normsub(h_fpp, fp32_sd["norm.weight"], lams[2 * ma["n_layers"]])
    ref = hnF @ fp32_sd["output.weight"].T
    delta("logits", logo.reshape(ref.shape), ref)

    out = dict(table=table, flips=flips)
    with open(os.path.join(RES, "eq18_drift_walk.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[eq18] wrote results/eq18_drift_walk.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()