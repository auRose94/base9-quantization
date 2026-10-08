#!/usr/bin/env python3
"""eq9 — QAT on the odd grid with digit-swap STE vs matched fp32 control.

Faithful port of base9 exp11's swap-STE recipe onto our eq_lib forward:
  * optimized objects = fp32 LATENTS (one per 2-D unique tensor) + the 1-D
    rests (RMSNorm weights) trained in fp32;
  * every step the state-dict tensors (graph leaves) receive the hard
    dequantization of the latents: per-(row,group) amax scale m, digits
    round(H*lat/m)+H, value m*(d-H)/H — the container's normative math;
  * backward populates the graph-leaf grads; lat.grad = leaf.grad verbatim
    (swap-form STE); leaf grads cleared;
  * arms start from the ORIGINAL fp32 source (matched init), identical seeded
    batches; 2000 steps x 8x512; AdamW base-lr 1 with LambdaLR warmup-100 +
    cosine 1e-3->1e-5, wd 0.01, clip 1.0.

Post-training each arm re-encodes a K9Q1 artifact (digits + amax scales via
the padding-aware RTN path) at its class size; the reloaded artifact must
reproduce the trained-state paired loss (P19).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402
from eq6_func_fit import quant_rtn_odd, train_tokens, GROUP as G64, UNIQUE_PARAMS  # noqa
from eq5_fusion import dequant_from_store  # noqa: E402

B9 = "/mnt/matrix/Work/base9-quantization/experiments"
sys.path.insert(0, B9)
import k9  # noqa: E402

DEV = "cuda"
RES = os.path.join(L.ROOT, "results")
STEPS, WARMUP, LR = 2000, 100, 1e-3
BATCH, BLOCK = 8, 512
EMB_KEY_NAMES = ("tok_embeddings.weight", "output.weight")


def dequant_grid(lat, k, group=G64):
    """Hard dequantization (true width), container-normative math."""
    H = (k - 1) // 2
    r, c = lat.shape
    mp = ((c - 1) // group + 1) * group
    pad = mp - c
    latp = torch.nn.functional.pad(lat, (0, pad))
    m = latp.reshape(r, mp // group, group).abs().amax(dim=2)                 # (r, ng)
    mrep = torch.repeat_interleave(m, group, dim=1)[:, :c]
    digits = torch.clamp(torch.round(H * lat / mrep), -H, H).long() + H
    return mrep * (digits - H) / H, digits


def lr_at(step):
    if step < WARMUP:
        return max(LR * (step + 1) / WARMUP, 1e-6)
    p = (step - WARMUP) / max(STEPS - WARMUP, 1)
    return 1e-5 + (LR - 1e-5) * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))


def run_arm(tag, body_k, sd0, ma, fit_tok, tokens, steps=None, seed=42):
    global STEPS
    STEPS = STEPS if steps is None else steps
    WARMUP_L = WARMUP                                     # closure copy (read-only)
    uniq = L.unique_parameters(sd0)
    w2d = {k: v for k, v in uniq.items() if v.dim() == 2}
    EMB = next(n for n in EMB_KEY_NAMES if n in uniq)
    rest = {k: torch.nn.Parameter(v.to(DEV))
            for k, v in uniq.items() if v.dim() == 1}
    leaves, latents = {}, []
    for name, W in w2d.items():
        k = None if body_k is None else (99 if name == EMB else body_k)
        if k is None:                                     # CONTROL: fp32 param, no swap
            p = torch.nn.Parameter(W.to(DEV).clone(), requires_grad=True)
            leaves[name] = (p, None, None)
            latents.append(p)
            continue
        val, _ = dequant_grid(W.to(DEV), k)               # RTN init swap
        p = torch.nn.Parameter(val, requires_grad=True)   # graph leaf (holds dequant)
        lat = torch.nn.Parameter(W.to(DEV).clone())       # optimized latent
        leaves[name] = (p, lat, k)
        latents.append(lat)

    def build_sd():
        sd = {name: pv for name, (pv, _, _) in leaves.items()}
        sd.update(rest)
        sd[EMB_KEY_NAMES[0]] = sd[EMB_KEY_NAMES[1]] = sd[EMB]
        return sd

    opt = torch.optim.AdamW(latents + list(rest.values()), lr=1.0, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(STEPS):
        pos = rng.integers(0, len(fit_tok) - BLOCK, size=BATCH)
        xb = torch.stack([torch.from_numpy(fit_tok[p:p + BLOCK].astype(np.int64))
                          for p in pos]).to(DEV)
        with torch.no_grad():                             # the swap-in
            for name, (p, lat, k) in leaves.items():
                if k is None:
                    continue                              # p IS the fp32 latent
                val, _ = dequant_grid(lat, k)
                p.data.copy_(val)
        lg = L.forward(build_sd(), xb, ma)
        loss = -torch.log_softmax(lg[:, :-1], -1).gather(-1, xb[:, 1:, None]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(latents + list(rest.values()), 1.0)
            for name, (p, lat, k) in leaves.items():
                if lat is not None and p.grad is not None:
                    lat.grad = p.grad                     # swap-form STE
                    p.grad = None
        opt.step()
        sch.step()
        if (step + 1) % 500 == 0:
            ev = L.eval_loss(build_sd(), ma, tokens, DEV, n_windows=64)
            print(f"  {tag} step {step+1}: fit-loss {loss.item():.4f} | "
                  f"quick-loss {ev[0]:.4f} | {time.time()-t0:.0f}s")

    # freeze: padding-aware RTN of the final latents -> K9Q1 artifact (grid arms only)
    tensors, final_state = [], {}
    for name, (p, lat, k) in leaves.items():
        if k is None:
            final_state[name] = p.detach()
            continue
        D, S, m_true = quant_rtn_odd(lat.detach().double().cpu().numpy(), k)
        tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]), group=G64,
                            scale_mode="fp16", k=k, digits=D,
                            scales=S.astype(np.float32), perm=None))
        final_state[name] = torch.from_numpy(
            dequant_from_store(D, S, k)[:, : m_true].astype(np.float32))
    path = os.path.join(RES, f"eq9_{tag}.k9")
    k9.write_k9(path, tensors)
    nbytes = os.path.getsize(path) + sum(v.numel() * 2 for v in rest.values())

    recs = k9.read_k9(path)
    sd_rest = {k: v.detach() for k, v in rest.items()}
    ev_final = L.eval_loss(L.to_device(final_state | sd_rest
                                       | {EMB_KEY_NAMES[0]: final_state[EMB],
                                          EMB_KEY_NAMES[1]: final_state[EMB]}, DEV),
                           ma, tokens, DEV, 384)
    sd_art = {n: k9.decode_tensor(r, G64)[:, : int(uniq[n].shape[1])] for n, r in recs.items()}
    if not sd_art:                                        # CONTROL: no grid records
        ev_art = ev_final
        story = L.sample_story(L.to_device(final_state | sd_rest
                                           | {EMB_KEY_NAMES[0]: final_state[EMB],
                                              EMB_KEY_NAMES[1]: final_state[EMB]}, DEV),
                               ma, DEV, n_new=160, seed=7)
    else:
        for k, v in rest.items():
            sd_art[k] = v.detach()
        sd_art[EMB_KEY_NAMES[0]] = sd_art[EMB_KEY_NAMES[1]] = sd_art[EMB]
        ev_art = L.eval_loss(L.to_device(sd_art, DEV), ma, tokens, DEV, 384)
        story = L.sample_story(L.to_device(sd_art, DEV), ma, DEV, n_new=160, seed=7)
    print(f"  {tag} FINAL: trained {ev_final[0]:.4f} | artifact {ev_art[0]:.4f} | "
          f"{nbytes:,} B ({nbytes*8/UNIQUE_PARAMS:.2f} b/p)")
    print(f"    story: {story[:88]!r}")
    return dict(arm=tag, bytes=int(nbytes), trained_loss=ev_final[0],
                artifact_loss=ev_art[0], story=story[:120])


def main():
    sd0, ma, _ = L.load_state_dict()
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    fit_tok = train_tokens(2_600_000)
    fp32 = L.eval_loss(L.to_device(sd0, DEV), ma, tokens, DEV, 384)
    print(f"[eq9] fp32 source anchor {fp32[0]:.4f}")
    out = [run_arm("CONTROL_fp32", None, sd0, ma, fit_tok, tokens),
           run_arm("QAT_k27emb99", 27, sd0, ma, fit_tok, tokens),
           run_arm("QAT_k9emb99", 9, sd0, ma, fit_tok, tokens)]
    ctrl = out[0]["trained_loss"]
    k27 = next(r for r in out if r["arm"] == "QAT_k27emb99")
    k9r = next(r for r in out if r["arm"] == "QAT_k9emb99")
    v16 = dict(prediction="P16", control=ctrl, verdict="PASS" if ctrl <= 1.60 else "FAIL")
    v17 = dict(prediction="P17", gap_k27=round(k27["trained_loss"] - ctrl, 4),
               verdict="PASS" if k27["trained_loss"] <= ctrl + 0.10 else "FAIL")
    v18 = dict(prediction="P18", gap_k9=round(k9r["trained_loss"] - ctrl, 4),
               beats_scalefit=bool(k9r["trained_loss"] <= 1.85),
               verdict="PASS" if (k9r["trained_loss"] <= ctrl + 0.25
                                  and k9r["trained_loss"] <= 1.85) else "FAIL")
    v19 = dict(prediction="P19", identity=[round(r["artifact_loss"] - r["trained_loss"], 5)
                                           for r in out],
               verdict="PASS" if all(abs(r["artifact_loss"] - r["trained_loss"]) <= 5e-3
                                     for r in out) else "FAIL")
    for v in (v16, v17, v18, v19):
        print(f"[eq9] {v['prediction']} ({v['verdict']}): " +
              json.dumps({k: v[k] for k in v if k not in ("prediction", "verdict")}))
    with open(os.path.join(RES, "eq9_results.json"), "w") as f:
        json.dump([v16, v17, v18, v19, dict(anchor=fp32[0], arms=out)], f, indent=2)
    print("[eq9] wrote results/eq9_results.json (+ eq9_*.k9 artifacts)")


if __name__ == "__main__":
    main()