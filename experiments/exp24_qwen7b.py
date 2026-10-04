#!/usr/bin/env python3
"""exp24 — K9 at 7B: Qwen2.5-Coder-7B-Instruct.

The scale test. 7.6 B params, 28 layers, hidden 3584, GQA (28 q / 4 kv heads),
vocab 152064, **untied** lm_head (so embed + lm_head are 1.09 B params, ~14% of
the model and both need quantizing).

Memory reality on a 16 GB card: 7B in fp32 (30 GB) does not fit, so the model is
loaded in **bfloat16** (15.2 GB) and everything is compared against that bf16
baseline — i.e. this measures the marginal cost of K9 *on top of* bf16, not on
top of fp32 as in the 1.5 B study. Noted in the output.

To avoid holding ~7.6 GB of raw digits, each tensor is quantized, encoded
immediately with the K9 codec, and the digits dropped; only the compressed blobs
are kept until the file is written.

Run: python3 exp24_qwen7b.py            # ~25-40 min, 16 GB GPU
"""
import csv
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

import k9
import eval_harness as eh
import exp4b_group_scales as e4b
import exp18_qwen_gptq_noise as e18
from rans import entropy_bits

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
GROUP = 64
BLOCK = 1024                     # forwards are memory-bound, not count-bound
N_WINDOWS = 8
CONFIGS = [(15, 99), (9, 99)]    # (body k, embed+lm_head k)
THREADS = min(24, os.cpu_count() or 1)


def load_bf16(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa").to(dev).eval()
    return model, tok


def quantize_inplace(W, k, g=GROUP, row_chunk=512):
    """Symmetric odd grid, computed in row chunks and written back INTO W, so
    neither the fp32 working set nor the digit buffer is ever full-size on the
    device (a 7B tensor in fp32 is 272 MB, and the embed's digits alone would be
    545 MB, on a card with ~0.9 GiB spare). Returns digits (CPU) and scales."""
    r, c = W.shape
    idx = torch.empty((r, c), dtype=torch.uint8, device="cpu")
    scales = torch.empty((r, c // g), dtype=torch.float32, device="cpu")
    for a in range(0, r, row_chunk):
        b = min(r, a + row_chunk)
        Wc = W[a:b].float().reshape(b - a, c // g, g)
        m = Wc.abs().amax(dim=2)
        step = 2.0 * m / (k - 1)
        ic = torch.clamp(torch.round((Wc + m.unsqueeze(-1)) / step.unsqueeze(-1)),
                          0, k - 1).long()
        deq = (-m.unsqueeze(-1) + ic * step.unsqueeze(-1)).reshape(b - a, c)
        idx[a:b] = ic.reshape(b - a, c).to(torch.uint8).cpu()
        scales[a:b] = m.cpu()
        W[a:b] = deq.to(W.dtype)
        del Wc, m, step, ic, deq
    return idx, scales


def main():
    t_all = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_bf16(dev)
    if dev == "cuda":
        print(f"loaded bf16, GPU peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{MODEL_ID}: {n_total:,} params")

    body = e18.body_layers(model)
    embed = model.model.embed_tokens
    lm_head = model.lm_head
    n_body = sum(m.weight.numel() for _, m in body)
    n_aux = embed.weight.numel() + lm_head.weight.numel()
    print(f"body {len(body)} Linear {n_body:,} | embed+lm_head {n_aux:,}")

    wiki = eh.get_windows(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                          "test", N_WINDOWS, BLOCK, label="wiki")
    code = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean-valid", None,
                          "train", N_WINDOWS, BLOCK, label="code(python valid)")

    print("bf16 reference...")
    pw0 = eh.ppl_per_window(model, wiki)
    pc0 = eh.ppl_per_window(model, code)
    print(f"  code {np.mean(pc0):.3f} | wiki {np.mean(pw0):.3f}")
    saved = {n: m.weight.data.detach().to("cpu").clone() for n, m in body}
    saved_embed = embed.weight.data.detach().to("cpu").clone()
    saved_head = lm_head.weight.data.detach().to("cpu").clone()

    def restore():
        # chunked: a full-size .to(dev) needs a temporary the card cannot spare
        def put(param, master):
            for a in range(0, param.shape[0], 1024):
                b = min(param.shape[0], a + 1024)
                param[a:b].copy_(master[a:b].to(dev))
        with torch.no_grad():
            for n, m in body:
                put(m.weight.data, saved[n])
            put(embed.weight.data, saved_embed)
            put(lm_head.weight.data, saved_head)
        if dev == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for ci, (k_body, k_aux) in enumerate(CONFIGS):
        if ci:
            restore()
        tensors = []
        dig_bits = sc_bytes = 0.0
        t0 = time.time()
        for name, m in body + [("__embed__", embed), ("__lm_head__", lm_head)]:
            W = m.weight.data
            k = k_body if name not in ("__embed__", "__lm_head__") else k_aux
            idx, scales_t = quantize_inplace(W, k)
            idx_u8 = idx.cpu().numpy()
            dig_bits += entropy_bits(idx_u8.reshape(-1), k) * W.numel()
            s_np = scales_t.cpu().numpy()
            sc_bytes += len(k9.encode_scales(s_np, "ent8")) + 2 * k
            rec = k9.encode_tensor(idx_u8, k, s_np, "ent8")       # encode now,
            del idx_u8, idx, scales_t, s_np                       # drop digits
            tensors.append(dict(name=name, shape=tuple(W.shape), group=GROUP,
                                scale_mode="ent8", k=k, rec=rec))
            if dev == "cuda":
                torch.cuda.empty_cache()
        print(f"k_body={k_body}: quantized+encoded in {time.time()-t0:.0f}s")

        path = RESULTS / f"qwen7b_k{k_body}_embed{k_aux}.k9"
        t0 = time.time()
        size = k9.write_k9(path, tensors, threads=THREADS)
        print(f"  wrote {path.name}: {size/1e6:.1f} MB ="
              f" {8*size/n_total:.3f} b/param in {time.time()-t0:.0f}s")
        del tensors

        pw = eh.ppl_per_window(model, wiki)
        pc = eh.ppl_per_window(model, code)
        d_code, d_code_se = eh.paired_delta(pc, pc0)
        d_wiki, _ = eh.paired_delta(pw, pw0)
        rows.append(dict(k_body=k_body, k_aux=k_aux, MB=size / 1e6,
                         bits_per_param=8 * size / n_total,
                         est_MB=(dig_bits / 8 + sc_bytes) / 1e6,
                         code=float(np.mean(pc)), code_se=float(np.std(pc, ddof=1) / math.sqrt(len(pc))),
                         wiki=float(np.mean(pw)), d_code=d_code,
                         d_code_se=d_code_se, d_wiki=d_wiki))
        print(f"  code {np.mean(pc):.3f} (±{rows[-1]['code_se']:.3f}) vs bf16"
              f" {np.mean(pc0):.3f} → Δ {d_code:+.3f} ± {d_code_se:.3f}"
              f" | est {rows[-1]['est_MB']:.1f} MB vs real {size/1e6:.1f} MB")
        if dev == "cuda":
            torch.cuda.empty_cache()

    with open(RESULTS / "exp24_qwen7b.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# exp24 — K9 at 7B (Qwen2.5-Coder-7B-Instruct)",
        "",
        f"{n_total:,} params; body {n_body:,} (k=body), embed+lm_head {n_aux:,}",
        f"(k=embed, untied). {N_WINDOWS} windows x {BLOCK} tok per corpus.",
        "**Baseline is bf16, not fp32** (7B fp32 = 30 GB does not fit a 16 GB",
        "card), so these deltas are the marginal cost of K9 on top of bf16.",
        "Bytes are the real K9 file written by k9.py (C coder, parallel encode).",
        "",
        "| k body | k embed | MB | b/param | code ppl | Δcode vs bf16 | wiki ppl |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['k_body']} | {r['k_aux']} | {r['MB']:.1f} |"
                     f" {r['bits_per_param']:.3f} | {r['code']:.3f}"
                     f" ± {r['code_se']:.3f} | {r['d_code']:+.3f}"
                     f" ± {r['d_code_se']:.3f} | {r['wiki']:.3f} |")
    lines += [
        "",
        "## Reading",
        "",
        "- Compare against the published GGUF sizes for this model (q2_k / q4_k_m",
        "  / q8_0) and the 1.5B frontier: the question is whether the K9 advantage",
        "  holds, shrinks, or grows with scale.",
        "- Estimated vs real bytes confirm the entropy accounting at 7B.",
        "",
        f"wall {time.time()-t_all:.0f}s", "",
    ]
    (RESULTS / "exp24_qwen7b.md").write_text("\n".join(lines))
    print("wrote results/exp24_qwen7b.{csv,md}")


if __name__ == "__main__":
    main()
