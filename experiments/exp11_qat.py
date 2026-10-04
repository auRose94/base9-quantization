#!/usr/bin/env python3
"""exp11 — RQ8: can we CREATE a new model with this research? QAT-train new
checkpoints natively ON the repeating-decimal grids, vs properly-matched
controls, and measure gain vs loss.

FORMAT CONFOUND (found in exp11 v1, honestly logged): the pretrained
checkpoint's quick-ppl in our eval format was ~46 while its train-pool ppl is
~4 — our EOS-packed 1024-block format penalizes the pretrained weights. Any
QAT run BOTH adapts to the format AND learns quantization, so a naive
"QAT 3.8 vs fp32 40.7" comparison would overstate the gain ~10×.

Therefore THREE training arms (identical data/steps/schedule):
  CONTROL  — fp32 weights, trained (format-adapted fp32 model of this config)
  QAT-RECIPE — body → 9-level ninths g64; wte/wpe → 99-level g64 (STE swap)
  QAT-TERN   — body → ternary with learnable per-row gamma; embeds → 99
Baselines measured fresh on window A (asserts anchor: fp32-pretrained ==
40.663, PTQ body9 == 54.679); plus a train-pool diagnostic (the pretrained
model's ppl on 100 training-format windows).

Pre-registered (v2, before running — P19 re-anchored to the adapted control
after the v1 confound):
  P17 QAT-RECIPE beats PTQ embed99+body9 by ≥ 2 ppl        (PTQ → QAT gain)
  P18 QAT-TERN ppl ≤ 200 → the base-9 pair-codec payload (~1.6 b/param)
      becomes viable                                        (payload test)
  P19 QAT-RECIPE ≤ fp32-CONTROL ppl + 25%   (the honest quantization LOSS
      at matched training)

Out: results/exp11_qat.{csv,md},
     results/qat_digits_{control,recipe,tern}.npz
"""
import csv
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch
from rans import rans_bits, entropy_bits
import exp4_real_model_ptq as e4
import exp4b_group_scales as e4b
import exp9_anomaly_check as e9

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
GROUP = 64
CODEC_K = 12
STEPS = 2000
WARMUP = 100
LR = 2e-4
POOL_TOKENS = 40_000_000
QUICK_BLOCKS = 100
POOL_CACHE = RESULTS / "train_pool.npy"


# ------------------------------------------------------------- data ----
def get_pool(tok):
    if POOL_CACHE.exists():
        pool = np.load(POOL_CACHE)
        print(f"pool (cached): {pool.size:,} train tokens")
        return pool
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    eos = tok.eos_token_id
    ids = []
    t0 = time.time()
    for row in ds:
        tk = tok(row["text"]).input_ids
        tk.append(eos)
        ids.extend(tk)
        if len(ids) >= POOL_TOKENS:
            break
    pool = np.asarray(ids[:POOL_TOKENS], dtype=np.int64)
    np.save(POOL_CACHE, pool)
    print(f"pool: {pool.size:,} train tokens in {time.time()-t0:.0f}s")
    return pool


# --------------------------------------------------- QAT machinery ----
def quant_deq(W, k, g=GROUP):
    """Hard dequantization per (row, group) amax scale — the exp4b formula."""
    r, c = W.shape
    Wg = W.reshape(r, c // g, g)
    m = Wg.abs().amax(dim=2, keepdim=True)
    step = 2.0 * m / (k - 1)
    idx = torch.clamp(torch.round((Wg + m) / step), 0, k - 1).long()
    return (-m + idx * step).reshape(r, c), idx.reshape(r, c)


def ternary_parts(latent, gamma):
    t = torch.where(latent > 0.5 * gamma, torch.ones_like(latent),
                    torch.where(latent < -0.5 * gamma, -torch.ones_like(latent),
                                torch.zeros_like(latent)))
    return t, t * gamma


def swap_in(param, latent, kind, k, gamma=None):
    with torch.no_grad():
        if kind == "fp32":
            deq = latent
        elif kind == "grid":
            deq, _ = quant_deq(latent, k)
        else:
            _, deq = ternary_parts(latent, gamma)
        param.data.copy_(deq)


def lr_at(step):
    if step < WARMUP:
        return max(LR * (step + 1) / WARMUP, 1e-6)
    p = (step - WARMUP) / max(STEPS - WARMUP, 1)
    return 1e-5 + (LR - 1e-5) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def eval_ppl(model, blocks, batch=4):
    with torch.no_grad():
        nll, ntok = 0.0, 0
        for i in range(0, blocks.size(0), batch):
            x = blocks[i:i + batch]
            logits = model(x).logits
            nll += torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1), reduction="sum").item()
            ntok += x[:, 1:].numel()
    return math.exp(nll / ntok)


def train_qat(model, spec, pool, rest_params, quick_val, label, batch, npz_path):
    """spec: list of (param, kind, k). Returns (hist, gammas, streams)."""
    dev = next(model.parameters()).device
    latents, gamma_params = [], {}
    for param, kind, k in spec:
        lat = torch.nn.Parameter(param.detach().clone())
        latents.append((param, lat, kind, k))
        if kind == "ternary":
            gamma_params[param] = torch.nn.Parameter(
                param.detach().abs().mean(dim=1, keepdim=True))
    lat_list = [lat for _, lat, _, _ in latents]
    gam_list = list(gamma_params.values())
    # base lr = 1.0; LambdaLR multiplies by lr_at(step) → effective = lr_at
    opt_q = torch.optim.AdamW(lat_list + gam_list, lr=1.0, weight_decay=0.01)
    opt_r = torch.optim.AdamW(rest_params, lr=1.0, weight_decay=0.01)
    sch_q = torch.optim.lr_scheduler.LambdaLR(opt_q, lr_at)
    sch_r = torch.optim.lr_scheduler.LambdaLR(opt_r, lr_at)
    rng = np.random.default_rng(42)

    hist = []
    t0 = time.time()
    for step in range(STEPS):
        pos = rng.integers(0, pool.size - e4.BLOCK, size=batch)
        xb = torch.from_numpy(np.stack([pool[p:p + e4.BLOCK] for p in pos])).to(dev)
        for param, lat, kind, k in latents:
            swap_in(param, lat, kind, k,
                    gamma_params[param] if kind == "ternary" else None)
        loss = model(xb, labels=xb).loss
        loss.backward()
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(lat_list + gam_list, 1.0)
        with torch.no_grad():
            for param, lat, kind, k in latents:
                if param.grad is None:
                    continue
                if kind == "ternary":
                    gamma = gamma_params[param]
                    t, _ = ternary_parts(lat, gamma)
                    gamma.grad = (param.grad * t).sum(dim=1, keepdim=True)
                lat.grad = param.grad
                param.grad = None
        opt_q.step()
        opt_r.step()
        sch_q.step()
        sch_r.step()
        opt_q.zero_grad(set_to_none=True)
        opt_r.zero_grad(set_to_none=True)
        if (step + 1) % 500 == 0 or step == 0:
            ppl = eval_ppl(model, quick_val, batch)
            hist.append((step + 1, float(loss), ppl))
            print(f"  {label} step {step+1}: loss {float(loss):.3f} | "
                  f"quick ppl {ppl:.2f} | {time.time()-t0:.0f}s")
    # freeze + dump digit streams from the FROZEN state (before any restore)
    dig = {}
    for idx_s, (param, lat, kind, k) in enumerate(latents):
        if kind == "fp32":
            swap_in(param, lat, kind, k)
            continue  # fp32 latents need no digit coding
        if kind == "grid":
            _, d_idx = quant_deq(lat.detach(), k)
            swap_in(param, lat, kind, k)
        else:
            gamma = gamma_params[param].detach()
            d_idx, deq = ternary_parts(lat.detach(), gamma)
            d_idx = (d_idx + 1).long()
            swap_in(param, lat, kind, k, gamma_params[param])
        dig[f"{idx_s:02d}_k{k}"] = d_idx.detach().cpu().numpy()
    np.savez(npz_path, **{kk: v.reshape(-1) for kk, v in dig.items()})
    streams = [(int(kk.split("_k")[1]), v.reshape(-1)) for kk, v in dig.items()]
    return hist, {p: g.detach().clone() for p, g in gamma_params.items()}, streams


def main():
    dev = e4.device_setup()
    batch = 8
    if dev == "cuda":
        free, _ = torch.cuda.mem_get_info()
        print(f"GPU free {free/1e9:.1f} GB")
        if free < 8e9:
            batch = 4
            print("shared GPU is loaded — batch 4")
    model, tok = e4.load_model(dev)
    body = e4.quantized_layers(model)
    wte, wpe = model.transformer.wte, model.transformer.wpe
    tied = model.lm_head.weight.data_ptr() == wte.weight.data_ptr()
    print(f"lm_head tied: {tied}")

    blocks = e4.get_blocks(tok, dev)      # val window A — same as exp4–10
    quick_val = blocks[:QUICK_BLOCKS]

    body_params = [m.weight for _, m in body]
    embed_params = [wte.weight, wpe.weight]
    body_ids = {id(p) for p in body_params}
    rest_params = [p for n, p in model.named_parameters()
                   if id(p) not in body_ids and p is not wte.weight
                   and p is not wpe.weight and p.requires_grad]

    ppl0 = eval_ppl(model, blocks, batch)
    print(f"fp32 baseline (pretrained, window A): {ppl0:.3f}")
    assert abs(ppl0 - 40.663) < 0.05, f"window-A fp32 regression broken: {ppl0}"
    saved = {p: p.data.detach().clone() for p in body_params + embed_params}

    def restore():
        for p, v in saved.items():
            p.data.copy_(v)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pool = get_pool(tok)
    # diagnostic: the pretrained model's ppl on TRAINING-format windows
    rng = np.random.default_rng(0)
    pos = rng.integers(0, pool.size - e4.BLOCK, size=QUICK_BLOCKS)
    train_diag = torch.from_numpy(
        np.stack([pool[p:p + e4.BLOCK] for p in pos])).to(dev)
    diag_ppl = eval_ppl(model, train_diag, batch)
    print(f"pretrained model on TRAIN-format windows: {diag_ppl:.3f}")

    # fresh PTQ baselines (assert-anchored where exp7 pinned them)
    base = {}
    for p in body_params:
        deq, _ = quant_deq(p.data, 9)
        p.data.copy_(deq)
    base["PTQ body9"] = eval_ppl(model, blocks, batch)
    assert abs(base["PTQ body9"] - 54.679) < 0.05
    print(f"PTQ body9: {base['PTQ body9']:.3f}")
    restore()

    for p in body_params:
        deq, _ = quant_deq(p.data, 9)
        p.data.copy_(deq)
    for p in embed_params:
        deq, _ = quant_deq(p.data, 99)
        p.data.copy_(deq)
    base["PTQ embed99+body9"] = eval_ppl(model, blocks, batch)
    print(f"PTQ embed99+body9: {base['PTQ embed99+body9']:.3f}")
    restore()

    for p in body_params:
        g = p.data.abs().mean(dim=1, keepdim=True)
        _, deq = ternary_parts(p.data, g)
        p.data.copy_(deq)
    for p in embed_params:
        deq, _ = quant_deq(p.data, 99)
        p.data.copy_(deq)
    base["PTQ ternary-g64+embed99"] = eval_ppl(model, blocks, batch)
    print(f"PTQ ternary-g64+embed99: {base['PTQ ternary-g64+embed99']:.3f}")
    restore()

    t_start = time.time()
    spec_all_fp32 = ([(p, "fp32", None) for p in body_params]
                     + [(p, "fp32", None) for p in embed_params])
    spec_recipe = ([(p, "grid", 9) for p in body_params]
                   + [(p, "grid", 99) for p in embed_params])
    spec_tern = ([(p, "ternary", 3) for p in body_params]
                 + [(p, "grid", 99) for p in embed_params])

    print("=== CONTROL (fp32, format-adapted) ===")
    _, _, s0 = train_qat(model, spec_all_fp32, pool, rest_params, quick_val,
                         "control", batch, RESULTS / "qat_digits_control.npz")
    ppl_ctrl = eval_ppl(model, blocks, batch)
    print(f"CONTROL final ppl: {ppl_ctrl:.3f}")
    restore()

    print("=== QAT-RECIPE (body9-ninths + embed99) ===")
    _, _, s1 = train_qat(model, spec_recipe, pool, rest_params, quick_val,
                         "recipe", batch, RESULTS / "qat_digits_recipe.npz")
    ppl1 = eval_ppl(model, blocks, batch)
    print(f"QAT-RECIPE final ppl: {ppl1:.3f}")
    restore()

    print("=== QAT-TERN (ternary body + embed99) ===")
    _, _, s2 = train_qat(model, spec_tern, pool, rest_params, quick_val,
                         "tern", batch, RESULTS / "qat_digits_tern.npz")
    ppl2 = eval_ppl(model, blocks, batch)
    print(f"QAT-TERN final ppl: {ppl2:.3f}")
    restore()

    # ternary payload cost: code the trained ternary body stream + base-9 pairs
    tern_syms = np.concatenate([v for k, v in s2 if k == 3])
    pair = 3 * tern_syms[0::2] + tern_syms[1::2]
    tern_bits = rans_bits(tern_syms, 3, CODEC_K)
    pair_bits = rans_bits(pair, 9, CODEC_K) / 2.0
    print(f"QAT-ternary body digits: rANS {tern_bits:.3f} b/param; "
          f"base-9 PAIRS: {pair_bits:.3f} b/param")

    with open(RESULTS / "exp11_qat.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "ppl", "note"])
        w.writerow(["fp32 pretrained", f"{ppl0:.3f}", "window A reference"])
        w.writerow(["fp32 pretrained, TRAIN-format windows", f"{diag_ppl:.3f}",
                    "format diagnostic"])
        for name, v in base.items():
            w.writerow([name, f"{v:.3f}", "PTQ baseline"])
        w.writerow(["CONTROL (fp32 adapted)", f"{ppl_ctrl:.3f}", f"{STEPS} steps"])
        w.writerow(["QAT-RECIPE body9+embed99", f"{ppl1:.3f}", f"{STEPS} steps"])
        w.writerow(["QAT-TERN ternary+embed99", f"{ppl2:.3f}", f"{STEPS} steps"])
        w.writerow(["QAT-TERN pair-codec b/param", f"{pair_bits:.3f}",
                    f"single-stream {tern_bits:.3f}"])

    tok_m = STEPS * batch * e4.BLOCK / 1e6
    p17 = ppl1 <= base["PTQ embed99+body9"] - 2.0
    p18 = ppl2 <= 200.0
    p19 = ppl1 <= ppl_ctrl * 1.25

    lines = [
        "# exp11 — QAT: creating a new model ON the repeating-decimal grids",
        "",
        f"Eval: val window A (same as exp4–10; fp32-pretrained {ppl0:.3f},",
        f"TRAIN-format diagnostic {diag_ppl:.3f} — the format confound v1 found).",
        f"Arms (identical {STEPS}-step schedule, {tok_m:.0f} M train tokens,",
        f"batch {batch}×512, AdamW + cosine, weight-swap STE): CONTROL fp32 →",
        "QAT-RECIPE (body 9-level ninths + embeds 99) → QAT-TERN (ternary body",
        "with learnable gamma + embeds 99).",
        "",
        "| variant | ppl (window A) | vs CONTROL |", "|---|---|---|",
        f"| fp32 pretrained | {ppl0:.3f} | {ppl0 - ppl_ctrl:+.2f} |",
    ]
    for name, v in base.items():
        lines.append(f"| {name} (PTQ) | {v:.3f} | {v - ppl_ctrl:+.2f} |")
    lines += [
        f"| CONTROL (fp32 adapted) | {ppl_ctrl:.3f} | — |",
        f"| QAT-RECIPE (body9 + embed99) | {ppl1:.3f} | {ppl1 - ppl_ctrl:+.2f} |",
        f"| QAT-TERN (ternary + embed99) | {ppl2:.3f} | {ppl2 - ppl_ctrl:+.2f} |",
        "",
        f"Ternary body payload: rANS {tern_bits:.3f} b/param, base-9 pairs "
        f"{pair_bits:.3f} b/param.",
        "",
        "## Pre-registered verdicts (v2 — P19 re-anchored to the adapted",
        "control after v1 exposed the format confound)",
        "",
        f"- P17 (QAT-RECIPE ≥2 ppl better than its PTQ): **{'PASS' if p17 else 'FAIL'}** "
        f"({base['PTQ embed99+body9']:.2f} → {ppl1:.2f})",
        f"- P18 (QAT-TERN ≤ 200 → base-9 pair payload viable): **{'PASS' if p18 else 'FAIL'}** "
        f"({base['PTQ ternary-g64+embed99']:.2f} → {ppl2:.2f})",
        f"- P19 (QAT-RECIPE ≤ CONTROL×1.25 = {ppl_ctrl*1.25:.2f}): **{'PASS' if p19 else 'FAIL'}**",
        "",
        "Gain = QAT vs its PTQ and (the honest one) QAT vs the trained control;",
        "loss = QAT vs the trained control. Digit artifacts:",
        "results/qat_digits_{control,recipe,tern}.npz.",
        "",
        f"wall {time.time()-t_start:.0f}s", "",
    ]
    (RESULTS / "exp11_qat.md").write_text("\n".join(lines))
    print("wrote results/exp11_qat.md")


if __name__ == "__main__":
    main()