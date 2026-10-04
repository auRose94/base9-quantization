#!/usr/bin/env python3
"""exp14 — longer QAT: does more training shrink the quantization loss?

exp11's 2000-step arms were a lower bound; BitNet's literature says ternary
gap shrinks with training. This runs the SAME three arms at 4000 steps
(~16.4 M tokens each, same pool/schedule/LR — control · recipe (body
9-ninths + embeds 99) · tern (ternary + learnable gamma + embeds 99)) and
compares matched-control losses against exp11's 2000-step numbers.

Pre-registered:
  P20  recipe's relative loss-to-control at 4000 steps ≤ its 2000-step loss
       (6.4%) — adaptation improves with training
  P21  tern's relative loss-to-control at 4000 steps < 20% (was 30%)
  P22  control-4000 ppl < control-2000 ppl (4.591) — more tokens help fp32 too

Out: results/exp14_longer_qat.{csv,md}, refreshed digit npz artifacts.
"""
import csv
import os
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

REF_2000 = {"control": 4.591, "recipe": 4.886, "tern": 5.980}  # exp11 finals
STEPS_4K = 4000


def main():
    e11.STEPS = STEPS_4K  # train_qat/lr_at read the module global
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

    def restore():
        for p, v in saved.items():
            p.data.copy_(v)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pool = e11.get_pool(tok)
    arms = {
        "control": [(p, "fp32", None) for p in body_params]
                   + [(p, "fp32", None) for p in embed_params],
        "recipe":  [(p, "grid", 9) for p in body_params]
                   + [(p, "grid", 99) for p in embed_params],
        "tern":    [(p, "ternary", 3) for p in body_params]
                   + [(p, "grid", 99) for p in embed_params],
    }

    finals = {}
    t_start = time.time()
    for arm, spec in arms.items():
        print(f"=== {arm} @ {STEPS_4K} steps ===")
        hist, _, _ = e11.train_qat(model, spec, pool, rest_params, quick_val,
                                   arm, batch, RESULTS / f"qat_digits_{arm}_4k.npz")
        ppl = e11.eval_ppl(model, blocks, batch)
        finals[arm] = ppl
        print(f"{arm} final ppl @4k: {ppl:.3f}")
        restore()

    # ternary payload at 4000 steps
    tern_syms = np.concatenate([v.reshape(-1) for kk, v in
                                np.load(RESULTS / "qat_digits_tern_4k.npz").items()
                                if str(kk).endswith("_k3")])
    recipe_npz = np.load(RESULTS / "qat_digits_recipe_4k.npz")
    pair = 3 * tern_syms[0::2] + tern_syms[1::2]
    tern_bits = rans_bits(tern_syms, 3, 12)
    pair_bits = rans_bits(pair, 9, 12) / 2.0
    body_digits_bits = rans_bits(
        np.concatenate([recipe_npz[kk].reshape(-1) for kk in recipe_npz.files
                        if str(kk).endswith("_k9")]), 9, 12)
    print(f"tern payload @4k: single {tern_bits:.3f}, pairs {pair_bits:.3f}; "
          f"recipe body9 digits {body_digits_bits:.3f} b/param")

    ctrl, rcp, trn = finals["control"], finals["recipe"], finals["tern"]
    loss_rcp = (rcp - ctrl) / ctrl
    loss_trn = (trn - ctrl) / ctrl
    p20 = loss_rcp <= 0.064
    p21 = loss_trn < 0.20
    p22 = ctrl < REF_2000["control"]

    with open(RESULTS / "exp14_longer_qat.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "ppl_4000", "ppl_2000_ref", "rel_loss_to_control"])
        w.writerow(["control", f"{ctrl:.3f}", f"{REF_2000['control']:.3f}", "—"])
        w.writerow(["recipe", f"{rcp:.3f}", f"{REF_2000['recipe']:.3f}", f"{loss_rcp:.3f}"])
        w.writerow(["tern", f"{trn:.3f}", f"{REF_2000['tern']:.3f}", f"{loss_trn:.3f}"])
        w.writerow(["tern pair-codec b/param", f"{pair_bits:.3f}",
                    f"single {tern_bits:.3f}"])

    lines = [
        f"# exp14 — longer QAT ({STEPS_4K} steps × 3 arms, ~{STEPS_4K*batch*e4.BLOCK/1e6:.0f} M tokens/arm)",
        "",
        f"fp32-pretrained reference {ppl0:.3f}; exp11 (2000-step) refs:",
        f"control {REF_2000['control']:.3f}, recipe {REF_2000['recipe']:.3f},",
        f"tern {REF_2000['tern']:.3f}. Matched-control losses are the metrics.",
        "",
        "| arm | ppl @4000 | ppl @2000 | rel loss vs control@4000 |",
        "|---|---|---|---|",
        f"| control (fp32) | {ctrl:.3f} | {REF_2000['control']:.3f} | — |",
        f"| recipe (body9+embed99) | {rcp:.3f} | {REF_2000['recipe']:.3f} | {100*loss_rcp:.1f}% |",
        f"| tern (ternary+embed99) | {trn:.3f} | {REF_2000['tern']:.3f} | {100*loss_trn:.1f}% |",
        "",
        f"Ternary payload @4000: rANS {tern_bits:.3f} b/param, base-9 pairs "
        f"**{pair_bits:.3f} b/param**; recipe body9 digits {body_digits_bits:.3f}.",
        "",
        "## Pre-registered verdicts",
        "",
        f"- P20 (recipe loss ≤ 6.4% @4000): **{'PASS' if p20 else 'FAIL'}** ({100*loss_rcp:.1f}%)",
        f"- P21 (tern loss < 20% @4000; was 30%): **{'PASS' if p21 else 'FAIL'}** ({100*loss_trn:.1f}%)",
        f"- P22 (control improves with 2× tokens): **{'PASS' if p22 else 'FAIL'}** "
        f"({REF_2000['control']:.3f} → {ctrl:.3f})",
        "",
        f"wall {time.time()-t_start:.0f}s", "",
    ]
    (RESULTS / "exp14_longer_qat.md").write_text("\n".join(lines))
    print("wrote results/exp14_longer_qat.md")


if __name__ == "__main__":
    main()