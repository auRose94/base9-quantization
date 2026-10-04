#!/usr/bin/env python3
"""exp13 — the DEPLOYABLE full-model point: embed99 + body9(+GPTQ).

Combines the two best PTQ components measured so far:
  body  — 9-level ninths grid, g=64, GPTQ-compensated (exp6 machinery;
          calibration = 64 train-split blocks; low-bit champion 43.64 ppl
          when embeddings stayed fp32)
  embed — wte+wpe per-row g=64 on the 99-level grid (exp10: only +0.1 ppl
          over body-only when the body was RTN'd)

Cross-checks (asserts): embed99+body9-RTN == exp10's 54.781.
Pre-registered: P23 — the full-model GPTQ artifact lands ≤ 50 ppl.
Byte table: per-component rANS digits + fp32 scales + fp32 rest.
Artifact: results/full_model_digits.npz — per-matrix digits (2-D) + per-(row,
group) scales + name inventory; plus per-row m for wte/wpe. Reload recipe:
dequant per cell w = −m + b·2m/(k−1) with b = stored digits, k = 9 (body) or
99 (embeddings); embeds and body reassembled by name from the inventory.

Out: results/exp13_full_model.{csv,md}, results/full_model_digits.npz
"""
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch
from rans import rans_bits, entropy_bits
import exp4_real_model_ptq as e4
import exp4b_group_scales as e4b
import exp6_gptq_grids as e6

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
CODEC_K = 12
BGRID, EGRID, GROUP = 9, 99, 64


def main():
    dev = e4.device_setup()
    batch = 8
    if dev == "cuda":
        free, _ = torch.cuda.mem_get_info()
        print(f"GPU free {free/1e9:.1f} GB")
        if free < 8e9:
            batch = 2
            print("shared GPU is loaded — eval batch 2")
    model, tok = e4.load_model(dev)
    body = e4.quantized_layers(model)
    embed_modules = [model.transformer.wte, model.transformer.wpe]
    tied = model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
    print(f"lm_head tied: {tied}")

    n_body = sum(m.weight.numel() for _, m in body)
    n_embed = sum(m_.weight.numel() for m_ in embed_modules)
    ptrs = set()
    n_total = 0
    for p in model.parameters():
        if p.data_ptr() not in ptrs:
            ptrs.add(p.data_ptr())
            n_total += p.numel()
    n_other = n_total - n_body - n_embed
    saved = {n_: m.weight.data.detach().clone() for n_, m in body}
    saved_emb = [m_.weight.data.detach().clone() for m_ in embed_modules]

    blocks = e4.get_blocks(tok, dev)
    ppl0 = e4.perplexity(model, blocks, batch)
    print(f"fp32 baseline: {ppl0:.3f} | fp32 model {4.0*n_total/1e6:.1f} MB")

    cal_all = e6.get_cal_blocks(tok, dev)
    cal_batches = [cal_all[i * 8:(i + 1) * 8] for i in range(e6.N_CAL // 8)]

    def restore():
        for n_, m in body:
            m.weight.data.copy_(saved[n_])
        for md_, v in zip(embed_modules, saved_emb):
            md_.weight.data.copy_(v)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    art = {}
    results = {}
    t0 = time.time()
    for variant, use_gptq in (("RTN", False), ("GPTQ", True)):
        tag = f"full_{variant}"
        bstreams, b_m, bside = {}, {}, 0
        serr = wsum2 = wsum = 0.0
        for name, m in body:
            tr = type(m).__name__ == "Conv1D"
            W = m.weight.data.t() if tr else m.weight.data
            r, c = W.shape
            scales = W.reshape(r, c // GROUP, GROUP).abs().amax(2).float()
            scales_t = scales.clone()
            if use_gptq:
                H = e6.collect_hessian(model, m, cal_batches)
                Q, IDX = e6.gptq_quantize(W, H, "grid", BGRID, g=GROUP,
                                          compensate=True)
            else:
                Q, IDX = e4b.q_uniform_group(W, BGRID)
            serr += ((W - Q) ** 2).sum().item()
            wsum2 += (W * W).sum().item()
            wsum += W.sum().item()
            m.weight.data = (Q.t().contiguous() if tr else Q.contiguous())
            bside += r * (c // GROUP)
            bstreams[name] = IDX.detach().cpu().numpy().reshape(-1)
            art[f"{tag}_body_{name.replace('.', '|')}_idx"] = IDX.detach().cpu().numpy()
            art[f"{tag}_body_{name.replace('.', '|')}_m"] = scales_t.cpu().numpy()
        bsym = np.concatenate(list(bstreams.values()))
        b_rans = rans_bits(bsym, BGRID, CODEC_K)
        b_marg = entropy_bits(bsym, BGRID)
        body_mse = serr / max(wsum2 - len(bsym) * (wsum / len(bsym)) ** 2, 1e-30)

        estreams, e_m, eside = {}, {}, 0
        for nm, md_ in zip(("wte", "wpe"), embed_modules):
            W = md_.weight.data
            r, c = W.shape
            scales = W.reshape(r, c // GROUP, GROUP).abs().amax(2).float()
            deq, idx = e4b.q_uniform_group(W, EGRID)
            md_.weight.data = deq.contiguous()
            eside += r * (c // GROUP)
            estreams[nm] = idx.detach().cpu().numpy().reshape(-1)
            art[f"{tag}_embed_{nm}_idx"] = idx.detach().cpu().numpy()
            art[f"{tag}_embed_{nm}_m"] = scales.cpu().numpy()
        esym = np.concatenate(list(estreams.values()))
        e_rans = rans_bits(esym, EGRID, CODEC_K)
        e_marg = entropy_bits(esym, EGRID)

        ppl = e4.perplexity(model, blocks, batch)
        body_mb = (b_rans * n_body / 8.0 + 4.0 * bside) / 1e6
        embed_mb = (e_rans * n_embed / 8.0 + 4.0 * eside) / 1e6
        rest_mb = 4.0 * n_other / 1e6
        results[variant] = dict(ppl=ppl, body_rans=b_rans, body_marg=b_marg,
                                embed_rans=e_rans, embed_marg=e_marg,
                                body_mse=body_mse,
                                total=body_mb + embed_mb + rest_mb,
                                body=body_mb, embed=embed_mb, rest=rest_mb)
        print(f"{variant}: ppl {ppl:.3f} | body {b_rans:.3f} (H {b_marg:.3f}) +"
              f" embed {e_rans:.3f} (H {e_marg:.3f}) b/param | total "
              f"{body_mb + embed_mb + rest_mb:.1f} MB | body mse {body_mse:.5f}")
        restore()

    out_rtn, out_gptq = results["RTN"], results["GPTQ"]
    assert abs(out_rtn["ppl"] - 54.781) < 0.05, \
        f"RTN cross-check vs exp10 broken: {out_rtn['ppl']}"
    p23 = out_gptq["ppl"] <= 50.0

    with open(RESULTS / "exp13_full_model.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "ppl", "total_MB", "body_MB", "embed_MB",
                    "rest_MB", "body_digits_rans", "body_mse"])
        for lbl, o in (("RTN", out_rtn), ("GPTQ", out_gptq)):
            w.writerow([lbl, f"{o['ppl']:.3f}", f"{o['total']:.1f}",
                        f"{o['body']:.1f}", f"{o['embed']:.1f}",
                        f"{o['rest']:.2f}", f"{o['body_rans']:.3f}",
                        f"{o['body_mse']:.5f}"])

    art["inventory"] = np.array(
        sorted(kk for kk in art if kk.endswith("_idx")), dtype=object)
    np.savez(RESULTS / "full_model_digits.npz", **art)
    print(f"saved results/full_model_digits.npz "
          f"({(RESULTS / 'full_model_digits.npz').stat().st_size/1e6:.1f} MB)")

    lines = [
        "# exp13 — the deployable full-model point: embed99 + body9(+GPTQ)",
        "",
        f"fp32 baseline {ppl0:.3f} | fp32 model {4.0*n_total/1e6:.1f} MB | eval",
        f"window A, batch {batch}; body = 9-ninths g=64 (GPTQ where labeled),",
        "embeddings = 99-level grid; ALL side info (scales) counted.",
        "",
        "| variant | ppl | total MB | body MB | embed MB | fp32 rest | body digits rANS | body rel MSE |",
        "|---|---|---|---|---|---|---|---|",
        f"| RTN | {out_rtn['ppl']:.3f} | {out_rtn['total']:.1f} | {out_rtn['body']:.1f} |"
        f" {out_rtn['embed']:.1f} | {out_rtn['rest']:.2f} | {out_rtn['body_rans']:.3f} | {out_rtn['body_mse']:.5f} |",
        f"| GPTQ | {out_gptq['ppl']:.3f} | {out_gptq['total']:.1f} | {out_gptq['body']:.1f} |"
        f" {out_gptq['embed']:.1f} | {out_gptq['rest']:.2f} | {out_gptq['body_rans']:.3f} | {out_gptq['body_mse']:.5f} |",
        "",
        f"P23 (GPTQ full-model ≤ 50 ppl): **{'PASS' if p23 else 'FAIL'}** "
        f"({out_gptq['ppl']:.3f})",
        "",
        f"Deployable artifact: results/full_model_digits.npz — per-matrix digits",
        f"+ per-(row,group) scales for both variants (inventory inside).",
        "Context: exp11's TRAINED arms sit at 4.89/5.98 ppl (they continued",
        "training); the PTQ routes here keep the pretrained weights.",
        "",
        f"wall {time.time()-t0:.0f}s", "",
    ]
    (RESULTS / "exp13_full_model.md").write_text("\n".join(lines))
    print("wrote results/exp13_full_model.md")


if __name__ == "__main__":
    main()