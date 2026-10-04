#!/usr/bin/env python3
"""exp14b — per-arm runner for exp14's remaining arms (resilience after a
CUDA error 719 driver fault killed exp14 mid-recipe-arm). Each invocation
runs ONE arm in a fresh CUDA context:

    python3 exp14b_arm_runner.py recipe
    python3 exp14b_arm_runner.py tern

exp14's CONTROL arm @4000 steps completed (ppl 3.895, measured; its npz
exists). The tern invocation finalizes: merges both arms + the measured
control, computes exp14's verdicts (P20/P21/P22) and the ternary payload,
and writes results/exp14_longer_qat.{csv,md}.

Pre-registered (unchanged from exp14):
  P20 recipe rel loss-to-control @4k ≤ its 2000-step loss (6.4%)
  P21 tern rel loss-to-control @4k < 20% (was 30%)
  P22 control improves with 2× tokens: control@4k (3.895) < control@2k (4.591)

Out: results/arm_{recipe,tern}_4000.json → results/exp14_longer_qat.{csv,md}
"""
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch
from rans import rans_bits
import exp4_real_model_ptq as e4
import exp11_qat as e11

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

REF_2000 = {"control": 4.591, "recipe": 4.886, "tern": 5.980}
CTRL_4K = 3.895          # measured: exp14's completed control arm
STEPS_4K = 4000


def main():
    arm = sys.argv[1]
    assert arm in ("recipe", "tern")
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
    blocks = e4.get_blocks(tok, dev)
    quick_val = blocks[:e11.QUICK_BLOCKS]

    body_params = [m.weight for _, m in body]
    embed_params = [wte.weight, wpe.weight]
    body_ids = {id(p) for p in body_params}
    rest_params = [p for n, p in model.named_parameters()
                   if id(p) not in body_ids and p is not wte.weight
                   and p is not wpe.weight and p.requires_grad]
    ppl0 = e11.eval_ppl(model, blocks, batch)
    print(f"fp32 baseline: {ppl0:.3f}")
    assert abs(ppl0 - 40.663) < 0.05, f"window-A regression broken: {ppl0}"
    saved = {p: p.data.detach().clone() for p in body_params + embed_params}

    pool = e11.get_pool(tok)
    spec = ([(p, "grid", 9) for p in body_params]
            + [(p, "grid", 99) for p in embed_params]) if arm == "recipe" else \
           ([(p, "ternary", 3) for p in body_params]
            + [(p, "grid", 99) for p in embed_params])

    print(f"=== {arm} @ {STEPS_4K} steps (fresh CUDA context) ===")
    t0 = time.time()
    _, _, streams = e11.train_qat(model, spec, pool, rest_params, quick_val,
                                  arm, batch, RESULTS / f"qat_digits_{arm}_4k.npz")
    ppl = e11.eval_ppl(model, blocks, batch)
    print(f"{arm} final ppl @4k: {ppl:.3f}")
    (RESULTS / f"arm_{arm}_4000.json").write_text(
        json.dumps(dict(arm=arm, ppl=ppl, steps=STEPS_4K, wall=t0 and time.time() - t0)))
    for p, v in saved.items():
        p.data.copy_(v)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"saved results/arm_{arm}_4000.json")

    if arm != "tern":
        return  # remaining work happens in the tern invocation

    # ------- finalize exp14 -------
    with open(RESULTS / "arm_recipe_4000.json") as fh:
        rcp = json.load(fh)["ppl"]
    trn = ppl
    loss_rcp = (rcp - CTRL_4K) / CTRL_4K
    loss_trn = (trn - CTRL_4K) / CTRL_4K
    p20 = loss_rcp <= 0.064
    p21 = loss_trn < 0.20
    p22 = CTRL_4K < REF_2000["control"]

    tern_npz = np.load(RESULTS / "qat_digits_tern_4k.npz")
    tern_syms = np.concatenate([tern_npz[kk].reshape(-1) for kk in tern_npz.files
                                if str(kk).endswith("_k3")])
    recipe_npz = np.load(RESULTS / "qat_digits_recipe_4k.npz")
    pair = 3 * tern_syms[0::2] + tern_syms[1::2]
    tern_bits = rans_bits(tern_syms, 3, 12)
    pair_bits = rans_bits(pair, 9, 12) / 2.0
    body_digits_bits = rans_bits(
        np.concatenate([recipe_npz[kk].reshape(-1) for kk in recipe_npz.files
                        if str(kk).endswith("_k9")]), 9, 12)

    with open(RESULTS / "exp14_longer_qat.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "ppl_4000", "ppl_2000_ref", "rel_loss_to_control"])
        w.writerow(["control", f"{CTRL_4K:.3f}", f"{REF_2000['control']:.3f}", "—"])
        w.writerow(["recipe", f"{rcp:.3f}", f"{REF_2000['recipe']:.3f}", f"{loss_rcp:.3f}"])
        w.writerow(["tern", f"{trn:.3f}", f"{REF_2000['tern']:.3f}", f"{loss_trn:.3f}"])
        w.writerow(["tern pair-codec b/param", f"{pair_bits:.4f}",
                    f"single-stream {tern_bits:.4f}"])
        w.writerow(["recipe body9 digits b/param", f"{body_digits_bits:.4f}", ""])

    lines = [
        f"# exp14 — longer QAT ({STEPS_4K} steps × 3 arms)",
        "",
        f"fp32-pretrained reference {ppl0:.3f}; exp11 (2000-step) refs: control "
        f"{REF_2000['control']:.3f}, recipe {REF_2000['recipe']:.3f}, tern "
        f"{REF_2000['tern']:.3f}. Control@4k (3.895) was completed by exp14 before",
        "a CUDA driver fault (error 719); the recipe/tern arms re-ran in fresh",
        "CUDA contexts (exp14b). Matched-control losses are the metrics.",
        "",
        "| arm | ppl @4000 | ppl @2000 | rel loss vs control@4000 |",
        "|---|---|---|---|",
        f"| control (fp32) | {CTRL_4K:.3f} | {REF_2000['control']:.3f} | — |",
        f"| recipe (body9+embed99) | {rcp:.3f} | {REF_2000['recipe']:.3f} | {100*loss_rcp:.1f}% |",
        f"| tern (ternary+embed99) | {trn:.3f} | {REF_2000['tern']:.3f} | {100*loss_trn:.1f}% |",
        "",
        f"Ternary payload @4000: rANS {tern_bits:.3f} b/param, base-9 pairs "
        f"**{pair_bits:.3f} b/param**; recipe body9 digits {body_digits_bits:.3f}.",
        "",
        "## Pre-registered verdicts",
        "",
        f"- P20 (recipe rel loss ≤ 6.4% @4000): **{'PASS' if p20 else 'FAIL'}** ({100*loss_rcp:.1f}%)",
        f"- P21 (tern rel loss < 20% @4000; was 30%): **{'PASS' if p21 else 'FAIL'}** ({100*loss_trn:.1f}%)",
        f"- P22 (control improves with 2× tokens): **{'PASS' if p22 else 'FAIL'}** "
        f"({REF_2000['control']:.3f} → {CTRL_4K:.3f})",
        "",
        f"Ops note: CUDA_ERROR_LAUNCH_FAILED (719) mid-run on the shared GPU;",
        "per-arm processes isolate future hits — rerun one arm, keep the rest.",
    ]
    (RESULTS / "exp14_longer_qat.md").write_text("\n" . join(lines))
    print("wrote results/exp14_longer_qat.md")


if __name__ == "__main__":
    main()