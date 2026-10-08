#!/usr/bin/env python3
"""eq8 — the high-K tail of the odd grid: body k=255 (container-direct, the
8-bit class) and body k=1023 (their multi-digit law: binary L=2 base-32, two
5-bit digit planes), both + embed k99, both with the eq6 scale-only functional
fit, both as REAL K9Q1 bytes.

rose's framing under test: k255 = quality-per-size point; k1023 = denser data
(10 b/p body) that composes as two 5-bit planes — the "turned into math" end
of the family (P14 measures the decomposition's coding cost).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402
from eq6_func_fit import quant_rtn_odd, train_tokens, GROUP as G64, UNIQUE_PARAMS  # noqa

B9 = "/mnt/matrix/Work/base9-quantization/experiments"
sys.path.insert(0, B9)
import k9  # noqa: E402

DEV = "cuda"
RES = os.path.join(L.ROOT, "results")
STEPS, LR, BATCH, BLOCK = 400, 3e-3, 8, 512


def fit_scales(m_params, digits_store, rest1d, EMB, H_map, ma, fit_tok, steps=STEPS):
    def build_sd(mp):
        sd = {}
        for name, st in digits_store.items():
            m = torch.exp(mp[name]).double()
            Srep = torch.repeat_interleave(m, GROUP(st), dim=1)
            w = Srep * (st["D"].double() - H_map[name]) / H_map[name]
            sd[name] = w[:, : st["m_true"]].float().to(DEV)
        for n_, v in rest1d.items():
            sd[n_] = v.half().float().to(DEV)
        emb = sd.pop(EMB)
        sd["output.weight"] = sd["tok_embeddings.weight"] = emb
        return sd

    opt = torch.optim.Adam(m_params.values(), lr=LR)
    for t in range(steps):
        starts = torch.randint(0, len(fit_tok) - BLOCK - 1, (BATCH,))
        b = torch.stack([torch.from_numpy(fit_tok[s:s + BLOCK].astype(np.int64))
                         for s in starts]).to(DEV)
        lg = L.forward(build_sd(m_params), b, ma)
        nll = -torch.log_softmax(lg[:, :-1], -1).gather(-1, b[:, 1:, None]).mean()
        opt.zero_grad(); nll.backward(); opt.step()
    return build_sd(m_params), m_params


def GROUP(_):
    return G64


def write_k1023(name, W, path_stubs):
    """Two base-32 digit planes; scales ride on hi; lo gets a (r,1) zero stub.
    Returns (hi_digits, lo_digits, S, m_true, D_composite)."""
    r, mcols = W.shape
    D, S, m_true = quant_rtn_odd(W.double().numpy(), 1023, dtype=np.uint16)
    hi = (D // 32).astype(np.uint8)
    lo = (D % 32).astype(np.uint8)
    return hi, lo, S, m_true, D


def main():
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    w2d = {k: v for k, v in uniq.items() if v.dim() == 2}
    EMB = "tok_embeddings.weight" if "tok_embeddings.weight" in uniq else "output.weight"
    rest1d = {n: v for n, v in uniq.items() if v.dim() == 1}
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    fit_tok = train_tokens(2_600_000)
    fp32 = L.eval_loss(L.to_device(sd0, DEV), ma, tokens, DEV, 384)
    print(f"[eq8] fp32 anchor {fp32[0]:.4f}")

    out_rows, verdict_bits = [], []

    # ---------------- variant 1: body k=255 (container-direct) ----------------
    dst255 = {}
    tensors = []
    for name, W in w2d.items():
        k = 99 if name == EMB else 255
        D, S, m_true = quant_rtn_odd(W.double().numpy(), k)
        dst255[name] = dict(D=torch.from_numpy(D.astype(np.int64)).to(DEV),
                            m0=torch.from_numpy(S.astype(np.float64)).to(DEV),
                            k=k, m_true=m_true)
        tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]), group=G64,
                            scale_mode="fp16", k=k, digits=D,
                            scales=S.astype(np.float32), perm=None))
    H255 = {name: (st["k"] - 1) // 2 for name, st in dst255.items()}
    p_k255 = os.path.join(RES, "eq8_k255emb99.k9")
    k9.write_k9(p_k255, tensors)
    b255 = os.path.getsize(p_k255) + sum(v.numel() * 2 for v in rest1d.values())
    mp = {name: torch.log(st["m0"]).float().to(DEV).requires_grad_(True)
          for name, st in dst255.items()}
    pre255 = L.eval_loss(fit_scales(mp, dst255, rest1d, EMB, H255, ma, fit_tok, steps=0)[0],
                         ma, tokens, DEV, 384)
    sd_f255, _ = fit_scales(mp, dst255, rest1d, EMB, H255, ma, fit_tok)
    post255 = L.eval_loss(sd_f255, ma, tokens, DEV, 384)
    out_rows.append(dict(variant="k255+emb99", bytes=b255,
                         bpp=round(b255 * 8 / UNIQUE_PARAMS, 3),
                         pre=round(pre255[0], 4), post=round(post255[0], 4),
                         ratio_post=round(post255[0] / fp32[0], 4)))

    # ---------------- variant 2: body k=1023 as two base-32 planes ----------------
    dst1023, k1023_records = {}, []
    for name, W in w2d.items():
        if name == EMB:
            k = 99
            D, S, m_true = quant_rtn_odd(W.double().numpy(), k)
            k1023_records.append(dict(name=name, shape=(D.shape[0], D.shape[1]),
                                      group=G64, scale_mode="fp16", k=k, digits=D,
                                      scales=S.astype(np.float32), perm=None))
            dst1023[name] = dict(D=torch.from_numpy(D.astype(np.int64)).to(DEV),
                                 m0=torch.from_numpy(S.astype(np.float64)).to(DEV),
                                 k=k, m_true=m_true)
            continue
        hi, lo, S, m_true, Dcomp = write_k1023(name, W, None)
        r, mp_cols = hi.shape
        k1023_records += [
            dict(name=name + ".hi", shape=(r, mp_cols), group=G64, scale_mode="fp16",
                 k=32, digits=hi, scales=S.astype(np.float32), perm=None),
            dict(name=name + ".lo", shape=(r, mp_cols), group=mp_cols, scale_mode="fp16",
                 k=32, digits=lo, scales=np.zeros((r, 1), np.float32), perm=None),
        ]
        mdev = torch.from_numpy(S.astype(np.float64)).to(DEV)
        dst1023[name] = dict(D=torch.from_numpy(Dcomp.astype(np.int64)).to(DEV),
                             m0=mdev, k=1023, m_true=m_true, hi=hi, lo=lo)
    H1023 = {name: (st["k"] - 1) // 2 for name, st in dst1023.items()}
    p_k1023 = os.path.join(RES, "eq8_k1023emb99.k9")
    k9.write_k9(p_k1023, k1023_records)
    b1023 = os.path.getsize(p_k1023) + sum(v.numel() * 2 for v in rest1d.values())
    mp2 = {name: torch.log(st["m0"]).float().to(DEV).requires_grad_(True)
           for name, st in dst1023.items()}
    pre_k = L.eval_loss(fit_scales(mp2, dst1023, rest1d, EMB, H1023, ma, fit_tok, steps=0)[0],
                        ma, tokens, DEV, 384)
    sd_f1023, _ = fit_scales(mp2, dst1023, rest1d, EMB, H1023, ma, fit_tok)
    post_k = L.eval_loss(sd_f1023, ma, tokens, DEV, 384)
    out_rows.append(dict(variant="k1023(2x5bit)+emb99", bytes=b1023,
                         bpp=round(b1023 * 8 / UNIQUE_PARAMS, 3),
                         pre=round(pre_k[0], 4), post=round(post_k[0], 4),
                         ratio_post=round(post_k[0] / fp32[0], 4)))

    # P14: plane coding cost of the k1023 body
    # P14: plane coding cost of the k1023 body (all its hi/lo records)
    body_params = sum(st["D"].numel() for n, st in dst1023.items() if n != EMB)
    plane_bytes = 0
    for rec in k1023_records:
        if rec["name"].endswith(".hi") or rec["name"].endswith(".lo"):
            one = os.path.join(RES, "_eq8_plane_probe.k9")
            k9.write_k9(one, [rec])
            plane_bytes += os.path.getsize(one)
    plane_bpp = plane_bytes * 8 / body_params
    print(f"[eq8] P14: k1023 body as 2×5-bit planes rANS = {plane_bytes:,} B = "
          f"{plane_bpp:.3f} b/p (10 raw bits; embed99 record excluded)")

    for r_ in out_rows:
        print(f"[eq8] {r_['variant']}: {r_['bytes']:,} B = {r_['bpp']:.2f} b/p | "
              f"pre {r_['pre']} -> post {r_['post']} ({r_['ratio_post']:.3f}x anchor)")
    story255 = L.sample_story(sd_f255, ma, DEV, n_new=160, seed=7)
    story1023 = L.sample_story(sd_f1023, ma, DEV, n_new=160, seed=7)
    print(f"[eq8] story k255-fitted: {story255[:90]!r}")
    print(f"[eq8] story k1023-fitted: {story1023[:90]!r}")

    # ---------------- verdicts ----------------
    ref = dict(k63=1.4397, k255=out_rows[0]["post"], k1023=out_rows[1]["post"])
    v11 = dict(prediction="P11", bpp=out_rows[0]["bpp"], ratio=out_rows[0]["ratio_post"],
               verdict="PASS" if (out_rows[0]["bpp"] <= 8.6
                                  and out_rows[0]["ratio_post"] <= 1.01) else "FAIL")
    v12 = dict(prediction="P12", losses=ref,
               verdict="PASS" if ref["k1023"] <= ref["k255"] <= ref["k63"] else "FAIL")
    v13 = dict(prediction="P13", d_hi_to_lo=round(abs(ref["k1023"] - ref["k255"]), 4),
               step_lo=round(abs(ref["k255"] - ref["k63"]), 4),
               verdict="PASS" if abs(ref["k1023"] - ref["k255"]) <=
               0.5 * abs(ref["k255"] - ref["k63"]) else "FAIL")
    v14 = dict(prediction="P14", body_plane_bpp=round(plane_bpp, 3), bound=10.4,
               verdict="PASS" if plane_bpp <= 10.4 else "FAIL")
    for v in (v11, v12, v13, v14):
        print(f"[eq8] {v['prediction']} ({v['verdict']}): " +
              json.dumps({k: v[k] for k in v if k not in ("prediction", "verdict")}))
    with open(os.path.join(RES, "eq8_results.json"), "w") as f:
        json.dump([v11, v12, v13, v14, dict(rows=out_rows,
                   story_k255=story255[:160], story_k1023=story1023[:160])], f, indent=2)
    print("[eq8] wrote results/eq8_results.json (+ eq8_*.k9 artifacts)")


if __name__ == "__main__":
    main()