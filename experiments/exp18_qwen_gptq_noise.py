#!/usr/bin/env python3
"""exp18 — GPTQ on the grids at 1.5B, with a real noise floor (exp17 follow-up).

Does two things the first pass (exp17) could not:

  1. REDUCES NOISE — every number is a mean +/- SE over 8 disjoint windows per
     corpus, and every scheme is compared PAIRED against the fp32 baseline on
     identical windows (shared eval_harness.py).

  2. ADDS GPTQ — Hessian error compensation (exp6 machinery) applied to the
     8/9/15-level grids on Qwen2.5-Coder-1.5B, which was the single biggest
     accuracy lever on TinyStories (exp6: -11 ppl on coarse grids at matched
     bits).

Deviation from exp6, stated for honesty: Hessians are collected ONCE on the
fp32 model (one pass, all layers) rather than sequentially on the partially
quantized model. This is the standard "calibrate on unquantized activations"
simplification; it trades a little accuracy for a 196x cheaper calibration.
Calibration is 32 x 1024 in-domain code blocks (train split); eval is
disjoint (WikiText-103 test + codeparrot-clean-valid).

Run:  python3 exp18_qwen_gptq_noise.py            # full (~30-45 min, GPU)
      python3 exp18_qwen_gptq_noise.py --smoke    # harness + RTN only, fast
"""
import csv
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch

import eval_harness as eh
import exp4b_group_scales as e4b
import exp6_gptq_grids as e6
from rans import entropy_bits
from exp16_codec_accounting import scale_codec_bytes

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BLOCK = 1024
GROUP = 64
SMOKE = "--smoke" in sys.argv
GPTQ_ONLY = "--gptq-only" in sys.argv
N_WINDOWS = 2 if SMOKE else 8
# GPTQ knobs (env-overridable): calibration size, damping fraction, act-order.
# exp18 v1 (32 blocks, damp 1%, no act-order) made GPTQ WORSE than RTN on a
# 1.5B model — ill-conditioned Hessians for the 8960-dim layers. These are the
# standard remedies.
N_CAL = int(os.environ.get("K9_NCAL", 4 if SMOKE else 128))
DAMP = float(os.environ.get("K9_DAMP", 0.01))
ACT = os.environ.get("K9_ACT", "0" if SMOKE else "1") == "1"

if SMOKE:
    SCHEMES = [("fp32 baseline", None, False, None),
               ("9-ninths g64 RTN", 9, False, None)]
elif GPTQ_ONLY:
    SCHEMES = [("fp32 baseline", None, False, None),
               ("9-ninths g64 RTN", 9, False, None),
               ("9-ninths g64 GPTQ", 9, True, None)]
else:
    SCHEMES = [("fp32 baseline", None, False, None),
               ("8-level g64 RTN", 8, False, None),
               ("9-ninths g64 RTN", 9, False, None),
               ("int4-15 g64 RTN", 15, False, None),
               ("99-level g64 RTN", 99, False, None),
               ("9-ninths g64 GPTQ", 9, True, None),
               ("int4-15 g64 GPTQ", 15, True, None),
               ("K9 full (body9-GPTQ + embed99)", 9, True, 99)]


def load_model(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32).to(dev).eval()
    return model, tok


def body_layers(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and m.weight.dim() == 2
            and not n.endswith("lm_head")]


def group_scales(W):
    r, c = W.shape
    return W.reshape(r, c // GROUP, GROUP).abs().amax(dim=2).float()


@torch.no_grad()
def collect_all_hessians(model, body, cal_batches):
    """H = 2*X^T X per Linear, one fp32 pass, accumulated on CPU (low VRAM)."""
    accs = {}

    def mk(name):
        def hook(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            xtx = (x.T @ x).cpu()
            accs[name] = xtx if name not in accs else accs[name] + xtx
        return hook

    handles = [m.register_forward_pre_hook(mk(n)) for n, m in body]
    for xb in cal_batches:
        model(xb)
    for h in handles:
        h.remove()
    n = sum(xb.shape[0] for xb in cal_batches)
    return {name: 2.0 * accs[name] / n for name, _ in body}


def h_inv(H, device, damp_frac):
    """Dampered upper-Cholesky inverse Hessian (GPTQ), CPU float64."""
    Hc = H.detach().to("cpu", torch.float64)
    damp = damp_frac * float(torch.diagonal(Hc).mean()) + 1e-8
    eye = torch.eye(Hc.shape[0], dtype=torch.float64)
    last = None
    for mult in (1.0, 10.0, 100.0, 1000.0):
        try:
            L = torch.linalg.cholesky(Hc + mult * damp * eye)
            Hi = torch.cholesky_inverse(L)
            return torch.linalg.cholesky(Hi, upper=True).to(
                device=device, dtype=torch.float32)
        except Exception as e:
            last = e
    raise RuntimeError(f"Hessian not invertible: {last}")


def gptq_quantize(W, H, k, g=GROUP, damp=0.01, act_order=True, compensate=True):
    """GPTQ column-wise quantization on the odd grid (exp6 core + act-order).

    act_order permutes columns by descending diag(H) before the pass — the
    standard remedy for ill-conditioned H on wide layers. Groups are then
    contiguous in permuted order (the stored scale order would follow it);
    the byte/entropy accounting is unaffected because digits are permuted.
    """
    r, c = W.shape
    perm = None
    if act_order:
        perm_cpu = torch.argsort(torch.diagonal(H), descending=True)
        H = H[perm_cpu][:, perm_cpu]
        perm = perm_cpu.to(W.device)
        W = W[:, perm]
    Wg = W.reshape(r, c // g, g)
    m_g = Wg.abs().amax(dim=2).unsqueeze(-1)
    step_g = 2.0 * m_g / (k - 1)
    Hinv = h_inv(H, W.device, damp) if compensate else None
    W1 = W.clone()
    IDX = torch.zeros_like(W, dtype=torch.long)
    Q = torch.empty_like(W)
    for i in range(c):
        grp = i // g
        m_r, st_r = m_g[:, grp, 0], step_g[:, grp, 0]
        w = W1[:, i]
        idx = torch.clamp(torch.round((w + m_r) / st_r), 0, k - 1).long()
        deq = -m_r + idx * st_r
        IDX[:, i] = idx
        Q[:, i] = deq
        if compensate and i + 1 < c:
            err = (w - deq) / Hinv[i, i]
            W1[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
    if perm is not None:
        inv = torch.argsort(perm)
        Q, IDX = Q[:, inv], IDX[:, inv]
    return Q, IDX


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(dev)
    body = body_layers(model)
    embed = model.model.embed_tokens
    n_body = sum(m.weight.numel() for _, m in body)
    n_embed = embed.weight.numel()
    ptrs, n_total = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in ptrs:
            ptrs.add(p.data_ptr())
            n_total += p.numel()
    n_rest = n_total - n_body - n_embed
    print(f"body {len(body)} Linear {n_body:,} | embed {n_embed:,} | rest {n_rest:,}")

    master = {n: m.weight.data.detach().to("cpu").clone() for n, m in body}
    master_embed = embed.weight.data.detach().to("cpu").clone()

    def restore_body():
        for n, m in body:
            m.weight.data.copy_(master[n].to(dev))
        embed.weight.data.copy_(master_embed.to(dev))

    # ------------------------------------------------------------- data ----
    print("calibration + eval windows")
    cal = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean", None, "train",
                         N_CAL, BLOCK, gap=0, label="calib(python train)")
    cal_batches = [cal[i:i + 8] for i in range(0, N_CAL, 8)]
    wiki = eh.get_windows(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                          "test", N_WINDOWS, BLOCK, label="wiki")
    code = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean-valid", None,
                          "train", N_WINDOWS, BLOCK, label="code(python valid)")

    # ------------------------------------------------ GPTQ Hessians (once) ----
    gptq_cache = {}
    hess = {}

    def ensure_gptq(k):
        if k in gptq_cache:
            return
        t0 = time.time()
        if "H" not in hess:
            print(f"GPTQ: collecting Hessians ({N_CAL} blocks, one fp32 pass)...")
            hess["H"] = collect_all_hessians(model, body, cal_batches)
        H = hess["H"]
        print(f"GPTQ k={k}: damp={DAMP} act_order={ACT}")
        streams, scales, qs = [], [], {}
        for i, (n, m) in enumerate(body):
            W = m.weight.data
            Q, IDX = gptq_quantize(W, H[n], k, g=GROUP, damp=DAMP,
                                   act_order=ACT, compensate=True)
            streams.append(IDX.to(torch.uint8).cpu().numpy().reshape(-1))
            scales.append(group_scales(W).cpu().numpy())
            qs[n] = Q.detach().to("cpu")
            if (i + 1) % 40 == 0:
                print(f"  k={k}: {i+1}/{len(body)} layers, {time.time()-t0:.0f}s")
        gptq_cache[k] = (qs, streams, scales)
        print(f"GPTQ k={k} done in {time.time()-t0:.0f}s")

    # ------------------------------------------------------- eval loop ----
    rows = []
    ppl_store = {}
    t_all = time.time()
    for name, kb, use_gptq, ke in SCHEMES:
        t0 = time.time()
        restore_body()
        if kb is not None and use_gptq:
            ensure_gptq(kb)
            qs, streams, scales = gptq_cache[kb]
            for n, m in body:
                m.weight.data.copy_(qs[n].to(dev))
        elif kb is not None:
            streams, scales = [], []
            for _, m in body:
                W = m.weight.data
                deq, idx = e4b.q_uniform_group(W, kb)
                streams.append(idx.to(torch.uint8).cpu().numpy().reshape(-1))
                scales.append(group_scales(W).cpu().numpy())
                m.weight.data = deq

        if ke is not None:
            deq, idx = e4b.q_uniform_group(embed.weight.data, ke)
            streams.append(idx.to(torch.uint8).cpu().numpy().reshape(-1))
            scales.append(group_scales(embed.weight.data).cpu().numpy())
            embed.weight.data = deq

        pw = eh.ppl_per_window(model, wiki)
        pc = eh.ppl_per_window(model, code)
        ppl_store[name] = (pw, pc)
        if kb is None:
            dig_bpp, n_quant, scale_mb = 0.0, 0, 0.0
        else:
            n_quant = sum(s.size for s in streams)
            dig_bpp = sum(entropy_bits(s, kb) * s.size for s in streams) / n_quant
            scale_mb = sum(scale_codec_bytes(s) for s in scales) / 1e6
        fp32_params = (n_rest + (n_embed if ke is None else 0)
                       + (n_body if kb is None else 0))
        total_mb = dig_bpp * n_quant / 8 / 1e6 + scale_mb + 4.0 * fp32_params / 1e6
        mw, sew = eh.summarize(pw)
        mc, sec = eh.summarize(pc)
        rows.append(dict(scheme=name, k=kb or 0, gptq=int(use_gptq),
                         digit_bpp=dig_bpp, total_mb=total_mb,
                         wiki=mw, wiki_se=sew, code=mc, code_se=sec,
                         secs=time.time() - t0))
        print(f"{name:<30} wiki {mw:7.3f}±{sew:.3f}  code {mc:7.3f}±{sec:.3f}"
              f" | {dig_bpp:5.3f} b/p {total_mb:6.1f} MB | {time.time()-t0:.0f}s")

    # paired deltas vs fp32 on the same windows
    fp_w, fp_c = ppl_store["fp32 baseline"]
    for r in rows:
        pw, pc = ppl_store[r["scheme"]]
        r["d_wiki"], r["d_wiki_se"] = eh.paired_delta(pw, fp_w)
        r["d_code"], r["d_code_se"] = eh.paired_delta(pc, fp_c)

    cols = list(rows[0].keys())
    with open(RESULTS / "exp18_qwen_gptq_noise.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# exp18 — GPTQ on the grids at 1.5B, with a noise floor",
        "",
        f"Qwen2.5-Coder-1.5B-Instruct; {N_WINDOWS} disjoint windows x {BLOCK} tok",
        f"per corpus (WikiText-103 test, codeparrot-clean-valid); calibration",
        f"{N_CAL} x {BLOCK} in-domain Python (train); GPTQ damping {DAMP:g},",
        f"act_order={ACT}. Mean +/- SE over windows;",
        "deltas are PAIRED against fp32 on identical windows. Hessians are",
        "collected once on the fp32 model (see docstring).",
        "",
        "| scheme | k | GPTQ | digits b/p | MB | wiki ppl | code ppl | Δwiki | Δcode |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scheme']} | {r['k'] or '—'} | {'yes' if r['gptq'] else '—'}"
            f" | {r['digit_bpp']:.3f} | {r['total_mb']:.1f}"
            f" | {r['wiki']:.3f} ± {r['wiki_se']:.3f}"
            f" | {r['code']:.3f} ± {r['code_se']:.3f}"
            f" | {r['d_wiki']:+.3f} ± {r['d_wiki_se']:.3f}"
            f" | {r['d_code']:+.3f} ± {r['d_code_se']:.3f} |")
    lines += [
        "",
        "A delta whose magnitude is within ~2 SE of zero is parity, not a win.",
        "The GPTQ rows are the test of whether compensation closes the gap to",
        "int4 at matched bits on a real code model.",
        "",
        f"wall {time.time()-t_all:.0f}s", "",
    ]
    (RESULTS / "exp18_qwen_gptq_noise.md").write_text("\n".join(lines))
    print("wrote results/exp18_qwen_gptq_noise.{csv,md}")


if __name__ == "__main__":
    main()
