#!/usr/bin/env python3
"""exp10 — quantize the EMBEDDINGS (open item: they are >half the model bytes).

All quantization so far left wte (50257×768, tied to lm_head), wpe (1024×768)
fp32 — at 33M-class scale that's ~160 MB of the ~271 MB model. This experiment
quantizes them per-row (group-of-64, same rule as the body) on the ninths (9)
and decimal-2-digit (99) grids, with the body at 9-level ninths g64 RTN
(cross-check must equal exp7's 54.679).

Schemes:
  embed9 + fp32 body   (isolates the ppl cost of embedding quantization)
  embed9 + body9       (combined; full byte accounting)
  embed99 + body9      (fine grid for embeddings)

Full-model byte accounting per scheme (dedup params): body digits (rANS) +
body scales; embedding digits (rANS) + embedding scales; rest fp32.

Pre-registered: P14 — embedding quantization alone costs ≤ 10% relative ppl.
P15 — combined scheme saves ≥ 40% of total model bytes vs fp32.

Out: results/exp10_embeddings.{csv,md}
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
from rans import rans_bits, entropy_bits
import exp4_real_model_ptq as e4
import exp4b_group_scales as e4b

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
GROUP = 64
CODEC_K = 12


def quant_stream(model, modules, k):
    """Quantize given modules per row (g=64); return (digit stream, side floats)."""
    streams = []
    serr = wsum2 = wsum = 0.0
    n_side = 0
    for m in modules:
        tr = type(m).__name__ == "Conv1D"
        W = m.weight.data.t() if tr else m.weight.data
        deq, idx = e4b.q_uniform_group(W, k)
        serr += ((W - deq) ** 2).sum().item()
        wsum2 += (W * W).sum().item()
        wsum += W.sum().item()
        m.weight.data = (deq.t() if tr else deq).contiguous()
        streams.append(idx.detach().cpu().numpy().reshape(-1))
        n_side += W.shape[0] * (W.shape[1] // GROUP)
    sym = np.concatenate(streams)
    n = sym.size
    return dict(sym=sym, entropy=entropy_bits(sym, k), rans=rans_bits(sym, k, CODEC_K),
                side_floats=n_side,
                mse=serr / max(wsum2 - n * (wsum / n) ** 2, 1e-30))


def main():
    dev = e4.device_setup()
    model, tok = e4.load_model(dev)
    body = e4.quantized_layers(model)
    wte = model.transformer.wte
    wpe = model.transformer.wpe
    tied = model.lm_head.weight.data_ptr() == wte.weight.data_ptr()
    print(f"lm_head tied to wte: {tied}")

    body_w = [m.weight.numel() for _, m in body]
    n_body = sum(body_w)
    n_wte = wte.weight.numel()
    n_wpe = wpe.weight.numel()
    ptrs = set()
    n_total = 0
    for p in model.parameters():
        if p.data_ptr() not in ptrs:
            ptrs.add(p.data_ptr())
            n_total += p.numel()
    saved = {name: m.weight.data.detach().clone() for name, m in body}
    saved_wte = wte.weight.data.detach().clone()
    saved_wpe = wpe.weight.data.detach().clone()

    EVAL_BATCH = 2  # the GPU is shared with other processes; batch 8 can OOM
    blocks = e4.get_blocks(tok, dev)
    ppl0 = e4.perplexity(model, blocks, batch=EVAL_BATCH)
    print(f"fp32 baseline: {ppl0:.3f}; params: body {n_body:,} + "
          f"wte {n_wte:,} + wpe {n_wpe:,} (lm_head {'tied' if tied else 'untied'})")

    def restore_all():
        for name, m in body:
            m.weight.data.copy_(saved[name])
        wte.weight.data.copy_(saved_wte)
        wpe.weight.data.copy_(saved_wpe)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []
    t0 = time.time()
    exp7_body9_ref = 54.679
    for label, body_k, embed_k in [("fp32 embeds + body9", 9, None),
                                   ("embed9 + fp32 body", None, 9),
                                   ("embed9 + body9", 9, 9),
                                   ("embed99 + body9", 9, 99)]:
        t1 = time.time()
        comp = {}
        if body_k is not None:
            comp["body"] = (n_body, quant_stream(model, [m for _, m in body], body_k))
        if embed_k is not None:
            comp["embed"] = (n_wte + n_wpe, quant_stream(model, [wte, wpe], embed_k))
        ppl = e4.perplexity(model, blocks, batch=EVAL_BATCH)

        def bytes_for(parts):
            total = 0.0
            for n_params, o in parts:
                side = 4.0 * o["side_floats"]
                total += (o["rans"] * n_params / 8.0 + side) / 1e6
            return total

        quant_mb = bytes_for(comp.values())
        fp32_part_mb = 4.0 * (n_total - sum(p[0] for p in comp.values())) / 1e6
        total_mb = quant_mb + fp32_part_mb
        if label == "fp32 embeds + body9":
            assert abs(ppl - exp7_body9_ref) < 0.05, \
                f"body9 regression vs exp7 broken: {ppl:.3f} vs {exp7_body9_ref}"
        rows.append(dict(scheme=label, ppl=ppl, total_mb=total_mb,
                         quant_mb=quant_mb,
                         note=f"fp32-rest {fp32_part_mb:.1f} MB | {time.time()-t1:.0f}s"))
        print(f"{label:<20} ppl {ppl:8.3f} | model {total_mb:7.1f} MB "
              f"(quantized {quant_mb:.1f} + fp32 rest {fp32_part_mb:.1f})")
        restore_all()

    mb_fp32 = 4.0 * n_total / 1e6
    best = min(rows, key=lambda r: r["total_mb"])
    by_label = {r["scheme"]: r for r in rows}
    p14 = by_label["embed9 + fp32 body"]["ppl"] <= 1.10 * ppl0
    p15 = best["total_mb"] <= 0.60 * mb_fp32
    emb_ppl = by_label["embed9 + fp32 body"]["ppl"]

    with open(RESULTS / "exp10_embeddings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scheme", "ppl", "total_mb", "quant_mb", "note"])
        w.writeheader()
        w.writerows(rows)

    lines = [
        f"# exp10 — quantizing the embeddings ({e4.MODEL_ID})",
        "",
        f"fp32 baseline {ppl0:.3f} | fp32 model {mb_fp32:.1f} MB | body-quant rows",
        "use the 9-level ninths grid (g=64 RTN; cross-check vs exp7: 54.679);",
        "embeddings quantized per-row (g=64) on 9- and 99-level grids.",
        "",
        "| scheme | ppl | total model MB | quantized MB | fp32 rest MB |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        rest = r["total_mb"] - r["quant_mb"]
        lines.append(f"| {r['scheme']} | {r['ppl']:.3f} | {r['total_mb']:.1f} |"
                     f" {r['quant_mb']:.1f} | {rest:.1f} |")
    lines += [
        "",
        f"P14 (embedding quant ≤ +10% relative ppl): **{'PASS' if p14 else 'FAIL'}** "
        f"(embed9+fp32body: {emb_ppl:.2f} vs {ppl0:.2f})",
        f"P15 (combined saves ≥ 40% bytes): **{'PASS' if p15 else 'FAIL'}** "
        f"({best['scheme']}: {best['total_mb']:.1f} vs {mb_fp32:.1f} MB)",
        "",
        f"wall {time.time()-t0:.0f}s", "",
    ]
    (RESULTS / "exp10_embeddings.md").write_text("\n".join(lines))
    print("wrote results/exp10_embeddings.md")


if __name__ == "__main__":
    main()