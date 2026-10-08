#!/usr/bin/env python3
"""eq22 — P39e-v2: the FREE-DIVISOR norm, trained in.

The scale-norm's +74% (eq21) came from unnormalized per-row magnitude. But
TWO divisor families are free in the exact algebra — 9-powers (the E-track)
and 2-powers (the den's integers / the J-track) — so the norm can normalize
magnitude WITHOUT leaving the free families:

    norm(x)_c = x_c · 9^{−K} · 2^{−j} · w_c
      K = floor(log9 max_c |x_c|)      → the max lands in [9^0, 9^1)
      j = floor(log2 rms(x·9^{−K}))    → the per-row rms lands in a 2-band

Both K and j are per-position INTEGER buckets (detached; scale-STE); every
divisor is from the free families; the per-channel rest w is the 6-bit
uniform-power grid (trained, swap-STE). Compared to RMS: the same intent
(magnitude normalization), but the divisor algebra is exactly
track-expressible — the carrier law never sees an odd denominator from the
norm.

This arm trains: free-norm + quadratic rational attention (c=1.0; eq21's
trained-in result: +2.78%) + silu-r (+0.19%) + digit-STE k27 + 6-bit rests.
Gate: P39e — the trained loss ≤ eq9's original-ops k27 arm + 2% (1.3406).
Follow-up: eq23 runs the exact track on THIS artifact to measure the
trained model's carriers.
"""
import json
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
from eq20_qat_substituted import eval_sub, quant_from_store  # noqa

RES = L.RESULTS
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG9 = float(np.log(9.0))
LOG2 = float(np.log(2.0))
STEPS = int(os.environ.get("EQ20_STEPS", "2000"))
BATCH, BLOCK = 8, 512
C_ATT = 1.0
REST_BITS = 6
EMB_KEY_NAMES = ("tok_embeddings.weight", "output.weight")


def rest_swap(w, bits=REST_BITS):
    return torch.round(w * (1 << bits)) / (1 << bits)


def freenorm(x, w):
    """the free-divisor norm: ·9^{−K} (max→[1,9)) then ·2^{−j} (rms→2-band)."""
    mx = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-30)
    K = torch.floor(torch.log(mx) / LOG9)
    h = x * torch.pow(9.0, -K)
    ms = h.detach().pow(2).mean(-1, keepdim=True).clamp_min(1e-30)
    j = torch.floor(torch.log(ms.sqrt()) / LOG2)
    return h * torch.exp2(-j) * w


def forward_free(sd, idx, ma):
    """free-norm + quadratic rational attention + silu-r + rope (fp)."""
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    rep = nh // nkv
    pos = torch.arange(T, device=idx.device)
    mask = torch.full((T, T), float("-inf"), device=idx.device).triu(1)
    h = sd["tok_embeddings.weight"][idx]

    def silr(x):
        return x * (1.0 + x / (x.abs() + 1.0)) * 0.5

    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hn = freenorm(h, sd[p + "attention_norm.weight"])
        q = (hn @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k4 = (hn @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v4 = (hn @ sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q = L.rope(q, pos)
        k4 = L.rope(k4, pos)
        kkr = k4.repeat_interleave(rep, dim=2).transpose(1, 2)
        vvr = v4.repeat_interleave(rep, dim=2).transpose(1, 2)
        s = (q.transpose(1, 2) @ kkr.transpose(-2, -1)) / hd ** 0.5 + mask
        wq = torch.relu(1.0 + s / C_ATT) ** 2
        wq = wq / wq.sum(-1, keepdim=True).clamp_min(1e-30)
        o = (wq @ vvr).transpose(1, 2).reshape(B, T, dim)
        h = h + (o @ sd[p + "attention.wo.weight"].T)
        hn2 = freenorm(h, sd[p + "ffn_norm.weight"])
        w1out = hn2 @ sd[p + "feed_forward.w1.weight"].T
        ff = silr(w1out) * (hn2 @ sd[p + "feed_forward.w3.weight"].T)
        h = h + (ff @ sd[p + "feed_forward.w2.weight"].T)
    hnF = freenorm(h, sd["norm.weight"])
    return hnF @ sd["output.weight"].T


@torch.no_grad()
def eval_free(sd, ma, tokens, n_windows=384, batch=16):
    block = ma["max_seq_len"]
    win = torch.from_numpy(tokens[: n_windows * block].astype(np.int64)
                           ).view(n_windows, block)
    losses = []
    for i in range(0, n_windows, batch):
        b = win[i:i + batch].to(DEV)
        lg = forward_free(sd, b, ma)[:, :-1]
        tg = b[:, 1:]
        nll = -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).squeeze(-1)
        losses.append(nll.mean(-1).cpu())
    lt = torch.cat(losses)
    return float(lt.mean()), float(lt.std(unbiased=True) / 19)


def run_arm(tag, body_k, sd0, ma, fit_tok, tokens, seed=42):
    torch.manual_seed(seed)
    uniq = L.unique_parameters(sd0)
    w2d = {kk: v for kk, v in uniq.items() if v.dim() == 2}
    EMB = next(n for n in EMB_KEY_NAMES if n in uniq)
    resting = {kk: torch.nn.Parameter(v.to(DEV)) for kk, v in uniq.items()
               if v.dim() == 1}
    rest_leaf = {kk: torch.nn.Parameter(rest_swap(v.to(DEV)))
                 for kk, v in uniq.items() if v.dim() == 1}
    leaves, latents = {}, []
    for name, W in w2d.items():
        kk = None if body_k is None else (99 if name == EMB else body_k)
        val, _ = dequant_grid(W.to(DEV), kk)
        p = torch.nn.Parameter(val, requires_grad=True)
        lat = torch.nn.Parameter(W.to(DEV).clone())
        leaves[name] = (p, lat, kk)
        latents.append(lat)
    opt = torch.optim.AdamW(latents + list(resting.values()), lr=1.0,
                            weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.default_rng(seed)
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
            for kk in rest_leaf:
                rest_leaf[kk].data.copy_(rest_swap(resting[kk]))
        sd_now = {n_: pv for n_, (pv, _, _) in leaves.items()} | rest_leaf | \
            {EMB_KEY_NAMES[0]: leaves[EMB][0],
             EMB_KEY_NAMES[1]: leaves[EMB][0]}
        lg = forward_free(sd_now, xb, ma)
        loss = -torch.log_softmax(lg[:, :-1], -1).gather(
            -1, xb[:, 1:, None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(latents
                                           + list(resting.values()), 1.0)
            for name, (p, lat, kk) in leaves.items():
                if p.grad is not None:
                    lat.grad = p.grad
                    p.grad = None
            for kk in rest_leaf:
                if rest_leaf[kk].grad is not None:
                    resting[kk].grad = rest_leaf[kk].grad
                    rest_leaf[kk].grad = None
        opt.step()
        sch.step()
        if (step + 1) % 500 == 0:
            sd_now = {n_: pv for n_, (pv, _, _) in leaves.items()} | \
                {kk: rest_swap(resting[kk]) for kk in resting} | \
                {EMB_KEY_NAMES[0]: leaves[EMB][0],
                 EMB_KEY_NAMES[1]: leaves[EMB][0]}
            evq = eval_free(sd_now, ma, tokens, 64)
            print(f"  {tag} step {step+1}: fit {loss.item():.4f} | "
                  f"quick {evq[0]:.4f} | {time.time()-t0:.0f}s", flush=True)
    tensors, final_state = [], {}
    from eq6_func_fit import quant_rtn_odd
    for name, (p, lat, kk) in leaves.items():
        D, S, m_true = quant_rtn_odd(lat.detach().double().cpu().numpy(), kk)
        tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]),
                            group=G64, scale_mode="fp16", k=kk, digits=D,
                            scales=S.astype(np.float32), perm=None))
        final_state[name] = torch.from_numpy(
            quant_from_store(D, S, kk, m_true).astype(np.float32))
    path = os.path.join(RES, f"eq22_{tag}.k9")
    k9.write_k9(path, tensors)
    rest_np = {kk: rest_swap(resting[kk]).detach().float().cpu().numpy()
               for kk in resting}
    np.savez(os.path.join(RES, f"eq22_{tag}_rest.npz"), **rest_np)
    nbytes = os.path.getsize(path) + sum(v.nbytes for v in rest_np.values())
    sd_art = {n_: t.to(DEV) for n_, t in final_state.items()} | \
        {kk: torch.from_numpy(v.astype(np.float32)).to(DEV)
         for kk, v in rest_np.items()}
    sd_art[EMB_KEY_NAMES[0]] = sd_art[EMB_KEY_NAMES[1]] = sd_art[EMB]
    ev_fin = eval_free(sd_now, ma, tokens, 384)
    ev_art = eval_free(sd_art, ma, tokens, 384)
    print(f"  {tag} FINAL: trained {ev_fin[0]:.4f} | artifact {ev_art[0]:.4f} "
          f"| {nbytes:,} B ({nbytes*8/UNIQUE_PARAMS:.2f} b/p)", flush=True)
    return dict(arm=tag, loss=ev_fin[0], artifact_loss=ev_art[0], bytes=nbytes)


_LAMS_CACHE = {}


def run_arm_ma(tag, body_k, sd0, ma, fit_tok, tokens, seed=42):
    """mean-abs norm + per-site λ (frozen from the source calibration)
    trained in — the map's quality-norm cell."""
    import eq20_qat_substituted as E20
    from eq20_qat_substituted import forward_sub
    torch.manual_seed(seed)
    uniq = L.unique_parameters(sd0)
    w2d = {k2: v for k2, v in uniq.items() if v.dim() == 2}
    EMB = next(n for n in EMB_KEY_NAMES if n in uniq)
    resting = {kk: torch.nn.Parameter(v.to(DEV)) for kk, v in uniq.items()
               if v.dim() == 1}
    leaves, latents = {}, []
    for name, W in w2d.items():
        kk = None if body_k is None else (99 if name == EMB else body_k)
        val, _ = dequant_grid(W.to(DEV), kk)
        p = torch.nn.Parameter(val, requires_grad=True)
        lat = torch.nn.Parameter(W.to(DEV).clone())
        leaves[name] = (p, lat, kk)
        latents.append(lat)
    # λ sites measured on the train slice (frozen per eq16's recipe)
    lams = _lams_of(sd0, ma)
    opt = torch.optim.AdamW(latents + list(resting.values()), lr=1.0,
                            weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    def build():
        sd = {n_: (pv if pv.dim() != 2 else pv) for n_, (pv, _, _) in leaves.items()}
        sd.update(resting)
        sd[EMB_KEY_NAMES[0]] = sd[EMB_KEY_NAMES[1]] = sd[EMB]
        return sd

    def ma_norm(x, w, site):
        s = x.abs().sum(-1, keepdim=True)
        return torch.where(s > 0, x * (x.shape[-1] / s.clamp_min(1e-30) * lams[site]),
                           torch.zeros_like(x)) * w

    site_of = {"norm.weight": 2 * ma["n_layers"]}
    for ll in range(ma["n_layers"]):
        site_of[f"layers.{ll}.attention_norm.weight"] = 2 * ll
        site_of[f"layers.{ll}.ffn_norm.weight"] = 2 * ll + 1
    for step in range(STEPS):
        pos_ids = rng.integers(0, len(fit_tok) - BLOCK, size=BATCH)
        xb = torch.stack([torch.from_numpy(fit_tok[pos:pos + BLOCK]
                                           .astype(np.int64))
                          for pos in pos_ids]).to(DEV)
        with torch.no_grad():
            for name, (p, lat, kk) in leaves.items():
                val, _ = dequant_grid(lat, kk)
                p.data.copy_(val)
        sd_now = build()
        lg = _forward_ma(sd_now, xb, ma, site_of, lams)
        loss = -torch.log_softmax(lg[:, :-1], -1).gather(-1, xb[:, 1:, None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(latents + list(resting.values()), 1.0)
            for name, (p, lat, kk) in leaves.items():
                if p.grad is not None:
                    lat.grad = p.grad
                    p.grad = None
        opt.step()
        sch.step()
        if (step + 1) % 500 == 0:
            sd_now = build()
            evq = _eval_ma(sd_now, ma, tokens, 64, site_of, lams)
            print(f"  {tag} step {step+1}: fit {loss.item():.4f} | "
                  f"quick {evq[0]:.4f} | {time.time()-t0:.0f}s", flush=True)
    sd_fin = {n_: pv.detach() for n_, (pv, _, _) in leaves.items()} |         {kk: v.detach() for kk, v in resting.items()}
    sd_fin[EMB_KEY_NAMES[0]] = sd_fin[EMB_KEY_NAMES[1]] = sd_fin[EMB]
    ev_fin = _eval_ma(sd_fin, ma, tokens, 384, site_of, lams)
    tensors, final_state = [], {}
    from eq6_func_fit import quant_rtn_odd
    for name, (p, lat, kk) in leaves.items():
        D, S, m_true = quant_rtn_odd(lat.detach().double().cpu().numpy(), kk)
        tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]),
                            group=G64, scale_mode="fp16", k=kk, digits=D,
                            scales=S.astype(np.float32), perm=None))
        final_state[name] = torch.from_numpy(
            quant_from_store(D, S, kk, m_true).astype(np.float32))
    path = os.path.join(RES, f"eq22_{tag}.k9")
    k9.write_k9(path, tensors)
    rest_np = {kk: v.detach().float().cpu().numpy() for kk, v in resting.items()}
    np.savez(os.path.join(RES, f"eq22_{tag}_rest.npz"), **rest_np)
    nbytes = os.path.getsize(path) + sum(v.nbytes for v in rest_np.values())
    sd_art = {n_: t.to(DEV) for n_, t in final_state.items()} | \
        {kk: torch.from_numpy(v.astype(np.float32)).to(DEV) for kk, v in rest_np.items()}
    sd_art[EMB_KEY_NAMES[0]] = sd_art[EMB_KEY_NAMES[1]] = sd_art[EMB]
    ev_art = _eval_ma(sd_art, ma, tokens, 384, site_of, lams)
    print(f"  {tag} FINAL: trained {ev_fin[0]:.4f} | artifact {ev_art[0]:.4f} "
          f"| {nbytes:,} B", flush=True)
    return dict(arm=tag, loss=ev_fin[0], artifact_loss=ev_art[0], bytes=nbytes)


def _lams_of(sd0, ma):
    """eq16's calibration: rms(manorm)/rms(rmsnorm) per site, train slice."""
    from eq16_substituted import calibrate_lams
    uniq = L.unique_parameters(sd0)
    fp32_sd = {n_: v for n_, v in uniq.items()}
    for n_, v in fp32_sd.items():
        if v.dim() == 1:
            fp32_sd[n_] = v.half().float()
    EMB = "tok_embeddings.weight"
    fp32_sd[EMB] = fp32_sd["tok_embeddings.weight"]
    fp32_sd["output.weight"] = fp32_sd[EMB]
    return calibrate_lams(
        {**fp32_sd, "tok_embeddings.weight": fp32_sd[EMB],
         "output.weight": fp32_sd[EMB]}, ma)


def _forward_ma(sd, idx, ma, site_of, lams):
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    rep = nh // nkv
    pos = torch.arange(T, device=idx.device)
    mask = torch.full((T, T), float("-inf"), device=idx.device).triu(1)
    h = sd["tok_embeddings.weight"][idx]

    def manorm(x, w, name):
        s = x.abs().sum(-1, keepdim=True)
        lam = lams[int(site_of[name])]
        return torch.where(s > 0, x * (x.shape[-1] / s.clamp_min(1e-30) * lam),
                           torch.zeros_like(x)) * w

    def silr(x):
        return x * (1.0 + x / (x.abs() + 1.0)) * 0.5

    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hn = manorm(h, sd[p + "attention_norm.weight"], p + "attention_norm.weight")
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
        hn2 = manorm(h, sd[p + "ffn_norm.weight"], p + "ffn_norm.weight")
        w1out = hn2 @ sd[p + "feed_forward.w1.weight"].T
        ff = silr(w1out) * (hn2 @ sd[p + "feed_forward.w3.weight"].T)
        h = h + (ff @ sd[p + "feed_forward.w2.weight"].T)
    hnF = manorm(h, sd["norm.weight"], "norm.weight")
    return hnF @ sd["output.weight"].T


@torch.no_grad()
def _eval_ma(sd, ma, tokens, n_windows, site_of, lams):
    block = ma["max_seq_len"]
    win = torch.from_numpy(tokens[: n_windows * block].astype(np.int64)).view(n_windows, block)
    losses = []
    for i in range(0, n_windows, 16):
        b = win[i:i + 16].to(DEV)
        lg = _forward_ma(sd, b, ma, site_of, lams)[:, :-1]
        tg = b[:, 1:]
        nll = -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).squeeze(-1)
        losses.append(nll.mean(-1).cpu())
    lt = torch.cat(losses)
    return float(lt.mean()), float(lt.std(unbiased=True) / 19)


def main():
    sd0, ma, _ = L.load_state_dict()
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    fit_tok = train_tokens(2_600_000)
    print(f"[eq22] dev {DEV}; steps {STEPS}", flush=True)
    out = [run_arm("qat_free_k27", 27, sd0, ma, fit_tok, tokens)]
    # the map's missing cell: mean-abs (calibrated, eq16's frozen λ) trained in
    out.append(run_arm_ma("qat_ma_k27", 27, sd0, ma, fit_tok, tokens))
    ARM = 1.3143247365951538
    v39e = dict(prediction="P39e", loss=out[0]["loss"], arm=ARM,
                verdict="PASS" if out[0]["loss"] <= ARM * 1.02 else "FAIL")
    dv = abs(out[0]["artifact_loss"] - out[0]["loss"])
    v22 = dict(prediction="eq22-identity", delta=round(dv, 5), bar=5e-3,
               verdict="PASS" if dv <= 5e-3 else "FAIL")
    for v in (v39e, v22):
        print(f"[eq22] {v['prediction']} ({v['verdict']}): "
              f"{json.dumps({kk: v[kk] for kk in v if kk not in ('prediction', 'verdict')})}",
              flush=True)
    with open(os.path.join(RES, "eq22_qat_freenorm.json"), "w") as f2:
        json.dump([v39e, v22, dict(arms=out)], f2, indent=2, default=str)
    print("[eq22] wrote results/eq22_qat_freenorm.json (+ .k9/_rest.npz)",
          flush=True)


if __name__ == "__main__":
    main()