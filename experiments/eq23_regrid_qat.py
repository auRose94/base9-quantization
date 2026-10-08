#!/usr/bin/env python3
"""eq23 — P39g: the shipping-design arm — mean-abs + trained per-site gains
+ activation-regrid, all trained in; digit-STE weights; 6-bit rests.

The norm matrix (eq21/eq22) said: quality wants the RMS-class norms (whose
exact divisors latch exponential carriers); bounded carriers want the
free-divisor buckets. P39g composes the honest middle:

  * norm = mean-abs, NO frozen λ — instead a TRAINED per-site gain, any
    real during training (t_l, log9-parameterized, initialized from the
    source calibration), snapped to (m_l · 9^{a_l} · 2^{b_l}) in the
    artifact with m_l a small integer (≤ 2 digits): the carrier law then
    latches a BOUNDED odd set (the m's + the H's per site), not a running
    product;
  * ACTIVATION-REGRID: before every matmul, activations are rounded onto
    the K9-native scaled alphabet with n digits and a per-row 9-exponent
    (the runtime's defined rounding, trained in by STE) — n ∈ {2, 3};
  * quadratic attention/silu are NOT in this arm (eq22's ma-arm used the
    original ops; +0.91% — keep those gains stacked);
  * digit-STE weights (k27 body, k99 embeds) + 6-bit uniform-power rests.

Gates: P39g — trained loss ≤ eq9's original-ops arm + 2% (1.3406) at n=3
and n=2; the artifact identity after the gain-snap (≤ 1e-2, reported).
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
from eq6_func_fit import quant_rtn_odd, train_tokens, GROUP as G64, \
    UNIQUE_PARAMS  # noqa
from eq9_qat import dequant_grid, lr_at  # noqa
from eq20_qat_substituted import quant_from_store  # noqa
from eq22_qat_freenorm import _lams_of  # noqa

RES = L.RESULTS
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG9 = float(np.log(9.0))
STEPS = int(os.environ.get("EQ20_STEPS", "2000"))
BATCH, BLOCK = 8, 512
REST_BITS = 6
EMB_KEY_NAMES = ("tok_embeddings.weight", "output.weight")


def rest_swap(w, bits=REST_BITS):
    return torch.round(w * (1 << bits)) / (1 << bits)


def regrid(x, n_digits):
    """the runtime's defined rounding: per row, value → mant·9^{e−n} with
    mant = round(x·9^{n−e}), e = ceil(log9 max|x_c|). Straight-through grad."""
    with torch.no_grad():
        mx = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-30)
        e = torch.ceil(torch.log(mx) / LOG9)
        mant = torch.round(x * torch.pow(9.0, n_digits - e)).clamp(
            -(9 ** n_digits), 9 ** n_digits)
    return x + (mant * torch.pow(9.0, e - n_digits) - x.detach())


def forward_rg(sd, idx, ma, site_of, gains, n_digits, snaps=None,
               regrid_on=True):
    """mean-abs + trained 9-gains + activation-regrid; quadratic/softmax per
    the source's own ops (the ma-arm kept them); silu original."""
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    rep = nh // nkv
    pos = torch.arange(T, device=idx.device)
    mask = torch.full((T, T), float("-inf"), device=idx.device).triu(1)
    h = sd["tok_embeddings.weight"][idx]
    if snaps is not None:
        lam_of = [float(m) * 9.0 ** a * 2.0 ** b for (m, a, b) in snaps]
    else:
        lam_of = gains

    def lam(site):
        lv = lam_of[site_of[site]]
        if isinstance(lv, torch.Tensor):
            return torch.pow(9.0, lv)
        return torch.tensor(float(lv), device=idx.device)

    def manorm(x, w, site):
        s = x.abs().sum(-1, keepdim=True)
        return torch.where(s > 0, x * (x.shape[-1] / s.clamp_min(1e-30) * lam(site)),
                           torch.zeros_like(x)) * w

    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hn = (regrid(manorm(h, sd[p + "attention_norm.weight"],
                            p + "attention_norm.weight"), n_digits)
              if regrid_on else
              manorm(h, sd[p + "attention_norm.weight"],
                     p + "attention_norm.weight"))
        q = (hn @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k4 = (hn @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v4 = (hn @ sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q = L.rope(q, pos)
        k4 = L.rope(k4, pos)
        kkr = k4.repeat_interleave(rep, dim=2).transpose(1, 2)
        vvr = v4.repeat_interleave(rep, dim=2).transpose(1, 2)
        s = (q.transpose(1, 2) @ kkr.transpose(-2, -1)) / hd ** 0.5 + mask
        o = (s.softmax(-1) @ vvr).transpose(1, 2).reshape(B, T, dim)
        h = h + (o @ sd[p + "attention.wo.weight"].T)
        hn2 = (regrid(manorm(h, sd[p + "ffn_norm.weight"],
                             p + "ffn_norm.weight"), n_digits)
               if regrid_on else
               manorm(h, sd[p + "ffn_norm.weight"], p + "ffn_norm.weight"))
        w1out = hn2 @ sd[p + "feed_forward.w1.weight"].T
        ff = w1out * torch.sigmoid(w1out) * \
            (hn2 @ sd[p + "feed_forward.w3.weight"].T)
        prod = (regrid(ff, n_digits) if regrid_on else ff)
        h = h + (prod @ sd[p + "feed_forward.w2.weight"].T)
    hnF = (regrid(manorm(h, sd["norm.weight"], "norm.weight"), n_digits)
           if regrid_on else manorm(h, sd["norm.weight"], "norm.weight"))
    return hnF @ sd["output.weight"].T


@torch.no_grad()
def eval_rg(sd, ma, tokens, n_windows, site_of, gains, n_digits, snaps=None,
            regrid_on=True):
    block = ma["max_seq_len"]
    win = torch.from_numpy(tokens[: n_windows * block].astype(np.int64)
                           ).view(n_windows, block)
    losses = []
    for i in range(0, n_windows, 16):
        b = win[i:i + 16].to(DEV)
        lg = forward_rg(sd, b, ma, site_of, gains, n_digits, snaps=snaps,
                        regrid_on=regrid_on)[:, :-1]
        tg = b[:, 1:]
        nll = -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).squeeze(-1)
        losses.append(nll.mean(-1).cpu())
    lt = torch.cat(losses)
    return float(lt.mean()), float(lt.std(unbiased=True) / 19)


def main():
    sd0, ma, _ = L.load_state_dict()
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    fit_tok = train_tokens(2_600_000)
    uniq0 = L.unique_parameters(sd0)
    lams_src = _lams_of(sd0, ma)               # log9-gain inits
    t_inits = [float(np.log(max(λ, 1e-30)) / LOG9) for λ in lams_src]
    print(f"[eq23] dev {DEV}; gain-inits (log9): "
          f"{[round(t, 3) for t in t_inits]}", flush=True)
    results = []
    if os.environ.get("EQ23_FINAL"):
        specs = [(4, 27, True, "qat_rest10_regrid_n4", True)]
    else:
        specs = [(4, 27, True, "qat_regrid_n4", False),
                 (5, 27, True, "qat_regrid_n5", False),
                 (3, 27, True, "qat_regrid_n3", False),
                 (3, 27, False, "qat_gains_only", False),
                 (4, 27, True, "qat_frozen_regrid_n4", True),
                 (4, 27, False, "qat_rest6_only", True)]
    for n_digits, body_k, regrid_on, tag, frozen in specs:
        if tag == "qat_regrid_n3" and (skip := os.environ.get("EQ23_SKIP_N3")):
            continue
        RBits = int(os.environ.get("EQ23_REST_BITS", str(REST_BITS)))
        torch.manual_seed(42)
        uniq = L.unique_parameters(sd0)
        w2d = {kk: v for kk, v in uniq.items() if v.dim() == 2}
        if frozen:
            gains = [torch.tensor(t_i, device=DEV, dtype=torch.float32)
                     for t_i in t_inits]          # NEVER optimized
        else:
            gains = [torch.nn.Parameter(torch.tensor(t_i, device=DEV,
                                                     dtype=torch.float32))
                     for t_i in t_inits]
        EMB = next(n for n in EMB_KEY_NAMES if n in uniq)
        resting = {kk: torch.nn.Parameter(v.to(DEV)) for kk, v in uniq.items()
                   if v.dim() == 1}
        site_names = ([f"layers.{ll}.{nn}_norm.weight"
                       for ll in range(ma["n_layers"])
                       for nn in ("attention", "ffn")] + ["norm.weight"])
        leaves, latents = {}, []
        for name, W in w2d.items():
            kk = None if body_k is None else (99 if name == EMB else body_k)
            val, _ = dequant_grid(W.to(DEV), kk)
            p = torch.nn.Parameter(val, requires_grad=True)
            lat = torch.nn.Parameter(W.to(DEV).clone())
            leaves[name] = (p, lat, kk)
            latents.append(lat)
        trainable = latents + list(resting.values())
        if not frozen:
            trainable = trainable + gains
        opt = torch.optim.AdamW(trainable, lr=1.0, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        rng = np.random.default_rng(42)
        t0 = time.time()
        for step in range(STEPS):
            pos_ids = rng.integers(0, len(fit_tok) - BLOCK, size=BATCH)
            xb = torch.stack([torch.from_numpy(fit_tok[pos:pos + BLOCK]
                                               .astype(np.int64))
                              for pos in pos_ids]).to(DEV)
            with torch.no_grad():
                for name, (p, lat, kk) in leaves.items():
                    val, _ = dequant_grid(lat, kk)
                    p.data.copy_(val)
            sd_now = {n_: pv for n_, (pv, _, _) in leaves.items()} | \
                {kk: rest_swap(resting[kk], RBits) for kk in resting} | \
                {EMB_KEY_NAMES[0]: leaves[EMB][0],
                 EMB_KEY_NAMES[1]: leaves[EMB][0]}
            site_of = {"norm.weight": 2 * ma["n_layers"]}
            for ll in range(ma["n_layers"]):
                site_of[f"layers.{ll}.attention_norm.weight"] = 2 * ll
                site_of[f"layers.{ll}.ffn_norm.weight"] = 2 * ll + 1
            lg = forward_rg(sd_now, xb, ma, site_of, gains,
                            20 if body_k is None else n_digits,
                            regrid_on=regrid_on)
            loss = -torch.log_softmax(lg[:, :-1], -1).gather(
                -1, xb[:, 1:, None]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                for name, (p, lat, kk) in leaves.items():
                    if p.grad is not None:
                        lat.grad = p.grad
                        p.grad = None
            opt.step()
            sch.step()
            if (step + 1) % 500 == 0:
                evq = eval_rg(sd_now, ma, tokens, 64, site_of, gains, n_digits)
                print(f"  {tag} step {step+1}: fit {loss.item():.4f} | "
                      f"quick {evq[0]:.4f} | {time.time()-t0:.0f}s", flush=True)
        # freeze + snaps
        t_final = [float(g.detach()) for g in gains]
        snaps = []
        ms = np.arange(1, 6562)                       # 4-digit odd latches
        lg = np.log(ms) / LOG9                        # their base-9 logs
        for t in t_final:
            # value = m·9^a·2^b: a = floor(t − lg), then the remainder to the
            # nearest 2-power. numpy-vectorized over all m.
            a_rem = t - lg
            a = np.floor(a_rem)
            frac2 = a_rem - a                          # ∈ [0,1): the 2-part's log9
            b = np.round(frac2 * (LOG9 / float(np.log(2.0))))
            resid = frac2 - b * float(np.log(2.0)) / LOG9
            i = int(np.abs(resid).argmin())
            snaps.append((int(ms[i]), int(a[i]), int(b[i])))
        ev_free = eval_rg(sd_now, ma, tokens, 384, site_of, gains, n_digits,
                          snaps=None, regrid_on=regrid_on)
        ev_snap = eval_rg(sd_now, ma, tokens, 384, site_of, gains, n_digits,
                          snaps=snaps, regrid_on=regrid_on)
        tensors, final_state = [], {}
        for name, (p, lat, kk) in leaves.items():
            D, S, m_true = quant_rtn_odd(lat.detach().double().cpu().numpy(), kk)
            tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]),
                                group=G64, scale_mode="fp16", k=kk, digits=D,
                                scales=S.astype(np.float32), perm=None))
            final_state[name] = torch.from_numpy(
                quant_from_store(D, S, kk, m_true).astype(np.float32))
        path = os.path.join(RES, f"eq23_{tag}.k9")
        k9.write_k9(path, tensors)
        rest_np = {kk: rest_swap(resting[kk], RBits).detach().float().cpu().numpy()
                   for kk in resting}
        np.savez(os.path.join(RES, f"eq23_{tag}_rest.npz"),
                 gains=np.array([[m, a, b] for m, a, b in snaps],
                                dtype=np.int64), gains_t=np.array(t_final))
        nbytes = os.path.getsize(path) + sum(v.nbytes for v in rest_np.values())
        sd_art = {n_: t.to(DEV) for n_, t in final_state.items()} | \
            {kk: torch.from_numpy(v.astype(np.float32)).to(DEV)
             for kk, v in rest_np.items()}
        sd_art[EMB_KEY_NAMES[0]] = sd_art[EMB_KEY_NAMES[1]] = sd_art[EMB]
        ev_art = eval_rg(sd_art, ma, tokens, 384, site_of, gains, n_digits,
                         regrid_on=regrid_on)
        print(f"  {tag} FINAL: trained(free-t) {ev_free[0]:.4f} | "
              f"snapped {ev_snap[0]:.4f} | artifact {ev_art[0]:.4f} | "
              f"{nbytes:,} B", flush=True)
        results.append(dict(tag=tag, n_digits=n_digits,
                            trained=float(ev_free[0]),
                            snapped=float(ev_snap[0]),
                            artifact=float(ev_art[0]), bytes=nbytes,
                            snaps=snaps,
                            t_final=[round(t, 4) for t in t_final]))
    ARM = 1.3143247365951538
    rows = []
    for r in results:
        vs = round(100 * (r["trained"] / ARM - 1), 2)
        rows.append(dict(**r, vs_arm_pct=vs,
                         verdict="PASS" if vs <= 2.0 else "FAIL"))
        print(f"[eq23] {r['tag']}: {r['trained']:.4f} ({vs:+.2f}% vs arm) "
              f"| snapped {r['snapped']:.4f} -> {'PASS' if vs <= 2.0 else 'FAIL'}",
              flush=True)
    with open(os.path.join(RES, "eq23_regrid_qat.json"), "w") as f2:
        json.dump([ARM, dict(arms=rows)], f2, indent=2, default=str)
    print("[eq23] wrote results/eq23_regrid_qat.json", flush=True)


if __name__ == "__main__":
    main()