#!/usr/bin/env python3
"""eq21 — the P39d decomposition: which substitution carries the trained-in
cost? Three matched arms (each ~100 s on the 5060 Ti), everything else kept
at the original ops:
  normQAT_k27   scale-norm only
  attQAT_k27    quadratic rational attention only
  siluQAT_k27   silu-r only
Gate per arm: within +2% of eq9's original-ops QAT-k27 (1.3143). Rests stay
trained fp32 in the ablations (the 6-bit rest swap rides only in the joint
arm — measured separately at 5e-4 class, negligible)."""
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
from eq6_func_fit import train_tokens, GROUP as G64, UNIQUE_PARAMS  # noqa
from eq9_qat import dequant_grid, lr_at  # noqa
import eq20_qat_substituted as E20  # noqa: E402
from eq20_qat_substituted import eval_sub, quant_from_store  # noqa

RES = L.RESULTS
DEV = E20.DEV
STEPS = int(os.environ.get("EQ20_STEPS", "2000"))
EMB_KEY_NAMES = E20.EMB_KEY_NAMES


def run_arm(tag, body_k, sd0, ma, fit_tok, tokens, seed=42,
            use_sc=False, use_qatt=False, use_silur=False):
    torch.manual_seed(seed)
    uniq = L.unique_parameters(sd0)
    w2d = {kk: v for kk, v in uniq.items() if v.dim() == 2}
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
    opt = torch.optim.AdamW(latents + list(resting.values()), lr=1.0,
                            weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    def fwd(sd, xb):
        return E20.forward_sub(sd, xb, ma, use_sc=use_sc,
                               use_qatt=use_qatt, use_silur=use_silur)

    def ev(n_windows):
        return eval_sub(sd_now_holder[0], ma, tokens, n_windows,
                        use_sc=use_sc, use_qatt=use_qatt,
                        use_silur=use_silur)
    sd_now_holder = [None]

    for step in range(STEPS):
        pos_ids = rng.integers(0, len(fit_tok) - E20.BLOCK, size=E20.BATCH)
        xb = torch.stack([torch.from_numpy(fit_tok[pos:pos + E20.BLOCK]
                                           .astype(np.int64))
                          for pos in pos_ids]).to(DEV)
        with torch.no_grad():
            for name, (p, lat, kk) in leaves.items():
                val, _ = dequant_grid(lat, kk)
                p.data.copy_(val)
        lg = fwd({n_: pv for n_, (pv, _, _) in leaves.items()}
                 | resting | {EMB_KEY_NAMES[0]: leaves[EMB][0],
                              EMB_KEY_NAMES[1]: leaves[EMB][0]}, xb)
        loss = -torch.log_softmax(lg[:, :-1], -1).gather(
            -1, xb[:, 1:, None]).mean()
        sd_now_holder[0] = None
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
            sd_now = {n_: pv for n_, (pv, _, _) in leaves.items()} | \
                {kk: rest_swap_now(v) for kk, v in resting.items()}
            sd_now_holder[0] = sd_now | {EMB_KEY_NAMES[0]: sd_now[EMB],
                                         EMB_KEY_NAMES[1]: sd_now[EMB]}
            evq = ev(64)
            print(f"  {tag} step {step+1}: fit {loss.item():.4f} | "
                  f"quick {evq[0]:.4f} | {time.time()-t0:.0f}s", flush=True)
    sd_fin = {n_: pv.detach() for n_, (pv, _, _) in leaves.items()} | \
        {kk: v.detach() for kk, v in resting.items()}
    sd_fin[EMB_KEY_NAMES[0]] = sd_fin[EMB_KEY_NAMES[1]] = sd_fin[EMB]
    sd_now_holder[0] = sd_fin
    ev_fin = ev(384)
    tensors, final_state = [], {}
    for name, (p, lat, kk) in leaves.items():
        from eq6_func_fit import quant_rtn_odd
        D, S, m_true = quant_rtn_odd(lat.detach().double().cpu().numpy(), kk)
        tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]),
                            group=G64, scale_mode="fp16", k=kk, digits=D,
                            scales=S.astype(np.float32), perm=None))
        final_state[name] = torch.from_numpy(
            quant_from_store(D, S, kk, m_true).astype(np.float32))
    path = os.path.join(RES, f"eq21_{tag}.k9")
    k9.write_k9(path, tensors)
    sd_art = {n_: t.to(DEV) for n_, t in final_state.items()} | \
        {kk: v.to(DEV) for kk, v in sd_fin.items() if v.dim() == 1}
    sd_art[EMB_KEY_NAMES[0]] = sd_art[EMB_KEY_NAMES[1]] = sd_art[EMB]
    sd_now_holder[0] = sd_art
    ev_art = ev(384)
    nbytes = os.path.getsize(path)
    print(f"  {tag} FINAL: trained {ev_fin[0]:.4f} | artifact {ev_art[0]:.4f} | "
          f"{nbytes:,} B", flush=True)
    return dict(arm=tag, loss=ev_fin[0], artifact_loss=ev_art[0], bytes=nbytes)


def rest_swap_now(v):
    return torch.round(v * 64) / 64


def main():
    sd0, ma, _ = L.load_state_dict()
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    fit_tok = train_tokens(2_600_000)
    ARM, CTRL = 1.3143247365951538, 1.2962031364440918
    specs = [("normQAT_k27", dict(use_sc=True)),
             ("attQAT_k27", dict(use_qatt=True)),
             ("siluQAT_k27", dict(use_silur=True))]
    arms = [run_arm(t, 27, sd0, ma, fit_tok, tokens, **kw)
            for t, kw in specs]
    rows = []
    for a in arms:
        vs = round(100 * (a["loss"] / ARM - 1), 2)
        rows.append(dict(arm=a["arm"], loss=round(a["loss"], 4),
                         vs_eq9arm_pct=vs, artifact=round(a["artifact_loss"], 4),
                         verdict="PASS" if vs <= 2.0 else "FAIL"))
        print(f"[eq21] {a['arm']}: {a['loss']:.4f} ({vs:+.2f}% vs eq9-arm) "
              f"-> {'PASS' if vs <= 2.0 else 'FAIL'}", flush=True)
    with open(os.path.join(RES, "eq21_ablation_qat.json"), "w") as f:
        json.dump([ARM, CTRL, dict(arms=rows)], f, indent=2, default=str)
    print("[eq21] wrote results/eq21_ablation_qat.json", flush=True)


if __name__ == "__main__":
    main()
