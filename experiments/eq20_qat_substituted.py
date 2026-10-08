#!/usr/bin/env python3
"""eq20 — P39d: substitution-aware QAT (docs/09). The exp11/eq9 digit-swap-STE
recipe with the EXACT-CAPABLE semantics trained IN:

  * scale-norm (value·9^{−K}, K = floor(log9 max_c|x_c|) — detached per step,
    a scale-STE) + the per-channel rest weights, trained as 6-bit
    uniform-power grid values (rint(w·64)·2^{−6} swap per step, same STE);
  * quadratic rational attention w = relu(1 + s/c)² / Σ, c = 1.0 (the eq16
    kernel family, mask handled naturally by the clamp: masked s = −inf ⇒
    weight exactly 0 — the relu-on-(1+s/c) form, NOT the mask-erasing
    log1p(relu(·)) form);
  * silu-r gate x·(1 + x/(|x|+1))/2;
  * rope stays fp (transcendental; registered follow-up);
  * weights: digit-STE on the odd k-grid (eq9's dequant_grid, matched to the
    source init, same seeded batches), per-(row,group) amax scales.

Gate (registered in logs/docs): P39d — the sub-QAT k27 arm's paired loss lands
within +2% of eq9's original-ops QAT-k27 (1.3143), i.e. ≤ 1.3406, with the
matched control 1.2962 reused (same recipe, same seed-42 batches, same windows).

Also closes the data-integrity gap: the trained rests are saved alongside the
K9Q1 artifact (eq20_*.npz), which eq9-13 never did.
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

RES = L.RESULTS
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG9 = float(np.log(9.0))
STEPS = int(os.environ.get('EQ20_STEPS', '2000'))
WARMUP = 100
BATCH, BLOCK = 8, 512
C_ATT = 1.0
REST_BITS = 6
EMB_KEY_NAMES = ("tok_embeddings.weight", "output.weight")


def rest_swap(w, bits=REST_BITS):
    """6-bit uniform-power grid swap (STE-friendly: values only)."""
    return torch.round(w * (1 << bits)) / (1 << bits)


def forward_sub(sd, idx, ma, use_sc=True, use_qatt=True, use_silur=True):
    """substituted-semantics forward (grad-flowing; fp32); flags isolate the
    three substitutions for the ablation arms."""
    B, T = idx.shape
    nh, nkv, dim = ma["n_heads"], ma["n_kv_heads"], ma["dim"]
    hd = dim // nh
    rep = nh // nkv
    pos = torch.arange(T, device=idx.device)
    mask = torch.full((T, T), float("-inf"), device=idx.device).triu(1)
    h = sd["tok_embeddings.weight"][idx]

    def scalenorm(x, w, on):
        if not on:
            return L.rmsnorm(x, w)
        mx = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-30)
        K = torch.floor(torch.log(mx) / LOG9)
        return x * torch.pow(9.0, -K) * w

    def silr(x):
        return x * (1.0 + x / (x.abs() + 1.0)) * 0.5

    def gate(x, on):
        return silr(x) if on else x * torch.sigmoid(x)

    for l in range(ma["n_layers"]):
        p = f"layers.{l}."
        hn = scalenorm(h, sd[p + "attention_norm.weight"], use_sc)
        q = (hn @ sd[p + "attention.wq.weight"].T).view(B, T, nh, hd)
        k4 = (hn @ sd[p + "attention.wk.weight"].T).view(B, T, nkv, hd)
        v4 = (hn @ sd[p + "attention.wv.weight"].T).view(B, T, nkv, hd)
        q = L.rope(q, pos)
        k4 = L.rope(k4, pos)
        kkr = k4.repeat_interleave(rep, dim=2).transpose(1, 2)
        vvr = v4.repeat_interleave(rep, dim=2).transpose(1, 2)
        s = (q.transpose(1, 2) @ kkr.transpose(-2, -1)) / hd ** 0.5 + mask
        if use_qatt:
            wq = torch.relu(1.0 + s / C_ATT) ** 2
            wq = wq / wq.sum(-1, keepdim=True).clamp_min(1e-30)
            o = wq @ vvr
        else:
            o = s.softmax(-1) @ vvr
        o = o.transpose(1, 2).reshape(B, T, dim)
        h = h + (o @ sd[p + "attention.wo.weight"].T)
        hn2 = scalenorm(h, sd[p + "ffn_norm.weight"], use_sc)
        w1out = hn2 @ sd[p + "feed_forward.w1.weight"].T
        ff = gate(w1out, use_silur) * (hn2 @ sd[p + "feed_forward.w3.weight"].T)
        h = h + (ff @ sd[p + "feed_forward.w2.weight"].T)
    hnF = scalenorm(h, sd["norm.weight"], use_sc)
    return hnF @ sd["output.weight"].T


@torch.no_grad()
def eval_sub(sd, ma, tokens, n_windows=384, batch=16, use_sc=True,
             use_qatt=True, use_silur=True):
    block = ma["max_seq_len"]
    win = torch.from_numpy(tokens[: n_windows * block].astype(np.int64)
                           ).view(n_windows, block)
    losses = []
    for i in range(0, n_windows, batch):
        b = win[i:i + batch].to(DEV)
        lg = forward_sub(sd, b, ma, use_sc=use_sc, use_qatt=use_qatt,
                         use_silur=use_silur)[:, :-1]
        tg = b[:, 1:]
        nll = -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).squeeze(-1)
        losses.append(nll.mean(-1).cpu())
    losses = torch.cat(losses)
    return float(losses.mean()), float(losses.std(unbiased=True) / 19)


def run_arm(tag, body_k, sd0, ma, fit_tok, tokens, seed=42):
    torch.manual_seed(seed)
    uniq = L.unique_parameters(sd0)
    w2d = {kk: v for kk, v in uniq.items() if v.dim() == 2}
    EMB = next(n for n in EMB_KEY_NAMES if n in uniq)
    resting = {kk: torch.nn.Parameter(v.to(DEV)) for kk, v in uniq.items()
               if v.dim() == 1}     # trained latents (float, pre-swap)
    rest_leaf = {kk: torch.nn.Parameter(rest_swap(v.to(DEV)))
                 for kk, v in uniq.items() if v.dim() == 1}  # graph leaves
    leaves, latents = {}, []
    for name, W in w2d.items():
        kk = None if body_k is None else (99 if name == EMB else body_k)
        val, _ = dequant_grid(W.to(DEV), kk)
        p = torch.nn.Parameter(val, requires_grad=True)
        lat = torch.nn.Parameter(W.to(DEV).clone())
        leaves[name] = (p, lat, kk)
        latents.append(lat)

    def build_sd():
        sd = {name: pv for name, (pv, _, _) in leaves.items()}
        sd.update(rest_leaf)
        sd[EMB_KEY_NAMES[0]] = sd[EMB_KEY_NAMES[1]] = sd[EMB]
        return sd

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
        lg = forward_sub(build_sd(), xb, ma)
        loss = -torch.log_softmax(lg[:, :-1], -1).gather(
            -1, xb[:, 1:, None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(latents + list(resting.values()), 1.0)
            for name, (p, lat, kk) in leaves.items():
                if p.grad is not None:
                    lat.grad = p.grad
                    p.grad = None
            for kk in rest_leaf:
                if rest_leaf[kk].grad is not None:
                    resting[kk].grad = rest_leaf[kk].grad   # swap STE: identity
                    rest_leaf[kk].grad = None
        opt.step()
        sch.step()
        if (step + 1) % 500 == 0:
            ev = eval_sub(build_sd(), ma, tokens, 64)
            print(f"  {tag} step {step+1}: fit {loss.item():.4f} | "
                  f"quick {ev[0]:.4f} | {time.time()-t0:.0f}s", flush=True)

    # freeze: RTN the final latents -> the K9Q1 artifact (+ rests as npz!)
    tensors, final_state = [], {}
    for name, (p, lat, kk) in leaves.items():
        D, S, m_true = quant_rtn_odd(lat.detach().double().cpu().numpy(), kk)
        tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]),
                            group=G64, scale_mode="fp16", k=kk, digits=D,
                            scales=S.astype(np.float32), perm=None))
        final_state[name] = torch.from_numpy(
            quant_from_store(D, S, kk, m_true).astype(np.float32))
    path = os.path.join(RES, f"eq20_{tag}.k9")
    k9.write_k9(path, tensors)
    rest_np = {kk: rest_swap(resting[kk]).detach().float().cpu().numpy()
               for kk in resting}
    np.savez(os.path.join(RES, f"eq20_{tag}_rest.npz"), **rest_np)
    nbytes = os.path.getsize(path) + sum(v.nbytes for v in rest_np.values())
    sd_art = {n_: t.to(DEV) for n_, t in final_state.items()}
    sd_art.update({kk: torch.from_numpy(v.astype(np.float32)).to(DEV)
                   for kk, v in rest_np.items()})
    sd_art[EMB_KEY_NAMES[0]] = sd_art[EMB_KEY_NAMES[1]] = sd_art[EMB]
    ev_art = eval_sub(sd_art, ma, tokens, 384)
    ev_fin = eval_sub(build_sd(), ma, tokens, 384)
    print(f"  {tag} FINAL: trained {ev_fin[0]:.4f} | artifact {ev_art[0]:.4f} "
          f"| {nbytes:,} B ({nbytes*8/UNIQUE_PARAMS:.2f} b/p)", flush=True)
    return dict(arm=tag, bytes=int(nbytes), trained_loss=ev_fin[0],
                artifact_loss=ev_art[0], identity=abs(ev_art[0] - ev_fin[0]))


def quant_from_store(D, S, kk, m_true):
    H = (kk - 1) // 2
    m = np.repeat(S, G64, axis=1)[:, :m_true]
    return m * (D[:, :m_true].astype(np.float64) - H) / H


def main():
    sd0, ma, _ = L.load_state_dict()
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    fit_tok = train_tokens(2_600_000)
    print(f"[eq20] dev = {DEV}; steps = {STEPS}", flush=True)
    out = [run_arm("qat_sub_k27", 27, sd0, ma, fit_tok, tokens)]
    k27 = out[0]
    ARM = 1.3143247365951538       # eq9's original-ops QAT-k27
    CTRL = 1.2962031364440918      # eq9's matched control
    v39d = dict(prediction="P39d", loss=k27["trained_loss"], arm=ARM,
                ctrl=CTRL, bar="≤ ARM·1.02",
                verdict="PASS" if k27["trained_loss"] <= ARM * 1.02 else "FAIL")
    v20_id = dict(prediction="eq20-artifact-identity",
                  delta=round(k27["identity"], 5), bar=5e-3,
                  verdict="PASS" if k27["identity"] <= 5e-3 else "FAIL")
    for v in (v39d, v20_id):
        print(f"[eq20] {v['prediction']} ({v['verdict']}): "
              f"{json.dumps({kk: v[kk] for kk in v if kk not in ('prediction', 'verdict')})}",
              flush=True)
    with open(os.path.join(RES, "eq20_qat_sub.json"), "w") as f:
        json.dump([v39d, v20_id, dict(arms=out)], f, indent=2, default=str)
    print("[eq20] wrote results/eq20_qat_sub.json + eq20_*.k9/_rest.npz",
          flush=True)


if __name__ == "__main__":
    main()