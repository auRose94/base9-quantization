#!/usr/bin/env python3
"""exp17 — K9 PTQ on a modern code LLM (Qwen2.5-Coder-1.5B-Instruct).

First real-model compression test on the enhancement plan: a modern coder
(Qwen2ForCausalLM: 28 layers, hidden 1536, GQA 12q/2kv, SwiGLU, RMSNorm, RoPE,
vocab 151,936, TIED embeddings) instead of TinyStories-33M's GPT-Neo.

Body = every nn.Linear except lm_head; embeddings = embed_tokens (tied to
lm_head). Grids are symmetric odd, per-(row, group-of-64) scales (input dim).
Rate accounting follows exp16: empirical digit entropy (per-tensor rANS tracks
it within ~0.01 b/param — verified here on a layer subset) + entropy-coded
log scales + fp32 rest (RMSNorm weights and Qwen2 attention biases, 144,896
params — not negligible in count, negligible in bytes).

Eval: WikiText-103 test (general) and a held-out Python slice (in-domain),
streamed and packed into 1024-token blocks. Small by design for a first pass
(noise is noted); the point is the ordering and the bytes.

Run:  python3 exp17_qwen_k9_ptq.py            # full first pass
      python3 exp17_qwen_k9_ptq.py --smoke    # 2 schemes, tiny eval
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
from rans import rans_bits, entropy_bits
import exp4b_group_scales as e4b
from exp16_codec_accounting import scale_codec_bytes

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BLOCK = 1024
GROUP = 64
CODEC_K = 12
SMOKE = "--smoke" in sys.argv
N_EVAL = 8 if SMOKE else 24
EVAL_BATCH = 2

SCHEMES = ([("fp32 baseline", None, None), ("9-ninths g64 (K9 body)", 9, None)]
           if SMOKE else
           [("fp32 baseline", None, None),
            ("8-level g64 (body)", 8, None),
            ("9-ninths g64 (K9 body)", 9, None),
            ("int4-15 g64 (body)", 15, None),
            ("99-level g64 (body)", 99, None),
            ("9-level full (body9+embed9)", 9, 9),
            ("K9 full (body9+embed99)", 9, 99),
            ("int4 full (body15+embed15)", 15, 15)])


def device_setup():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no CUDA — this will be very slow on a 1.5B model")
    return dev


def load_model(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32).to(dev).eval()
    return model, tok


def body_layers(model):
    out = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.weight.dim() == 2 \
                and not name.endswith("lm_head"):
            out.append((name, m))
    return out


def get_blocks(tok, dev, dataset, config, split, blocks):
    """Stream text, tokenize, pack into `blocks` x BLOCK token chunks."""
    from datasets import load_dataset
    ds = (load_dataset(dataset, config, split=split, streaming=True)
          if config else load_dataset(dataset, split=split, streaming=True))
    feats = list(ds.features)
    field = next((f for f in ("text", "content", "code", "body") if f in feats), None)
    if field is None:
        raise SystemExit(f"no text field in {dataset}: {feats}")
    print(f"  {dataset}: using field '{field}'")
    ids = []
    for row in ds:
        t = row[field]
        if not t or not t.strip():
            continue
        ids.extend(tok(t, add_special_tokens=False).input_ids)
        if len(ids) >= blocks * BLOCK:
            break
    arr = np.asarray(ids[: blocks * BLOCK], dtype=np.int64).reshape(blocks, BLOCK)
    return torch.from_numpy(arr).to(dev)


@torch.no_grad()
def perplexity(model, blocks, batch=EVAL_BATCH):
    nll, ntok = 0.0, 0
    for i in range(0, blocks.size(0), batch):
        xb = blocks[i:i + batch]
        logits = model(xb).logits
        nll += torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            xb[:, 1:].reshape(-1), reduction="sum").item()
        ntok += xb[:, 1:].numel()
    return math.exp(nll / ntok)


def group_scales(W):
    r, c = W.shape
    return W.reshape(r, c // GROUP, GROUP).abs().amax(dim=2).float()


def apply_scheme(body, embed, k_body, k_embed):
    """Quantize in place; return digit streams, scale arrays, param counts."""
    streams, scales = [], []
    n_body = 0
    for _, m in body:
        W = m.weight.data
        if k_body is not None:
            deq, idx = e4b.q_uniform_group(W, k_body)
            scales.append(group_scales(W).cpu().numpy())
            streams.append(idx.to(torch.uint8).cpu().numpy().reshape(-1))
            m.weight.data = deq
        n_body += W.numel()
    n_embed = embed.weight.numel()
    if k_embed is not None:
        W = embed.weight.data
        deq, idx = e4b.q_uniform_group(W, k_embed)
        scales.append(group_scales(W).cpu().numpy())
        streams.append(idx.to(torch.uint8).cpu().numpy().reshape(-1))
        embed.weight.data = deq
    return streams, scales, n_body, n_embed


def rate_bits(streams, k):
    """Size-weighted entropy (b/param) over per-tensor streams."""
    n = sum(s.size for s in streams)
    return sum(entropy_bits(s, k) * s.size for s in streams) / n, n


def rans_check(streams, k, n_tensors=3):
    """Verify per-tensor rANS == entropy on a few tensors (this model)."""
    tot_e = tot_r = 0.0
    n = 0
    for s in streams[:n_tensors]:
        e = entropy_bits(s, k)
        r = rans_bits(s, k, CODEC_K)
        tot_e += e * s.size
        tot_r += r * s.size
        n += s.size
    return tot_e / n, tot_r / n


def main():
    dev = device_setup()
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
    print(f"body {len(body)} Linear layers, {n_body:,} params | embed {n_embed:,}"
          f" | rest(fp32) {n_rest:,} | total {n_total:,}")

    master = {name: m.weight.data.detach().to("cpu").clone() for name, m in body}
    master_embed = embed.weight.data.detach().to("cpu").clone()

    def restore():
        for name, m in body:
            m.weight.data.copy_(master[name].to(dev))
        embed.weight.data.copy_(master_embed.to(dev))
        if dev == "cuda":
            torch.cuda.empty_cache()

    wiki = get_blocks(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                      "test", N_EVAL)
    code = get_blocks(tok, dev, "codeparrot/codeparrot-clean-valid", None,
                      "train", N_EVAL)
    print(f"eval: wiki {wiki.numel():,} tok | code {code.numel():,} tok")

    rows = []
    t_all = time.time()
    for name, kb, ke in SCHEMES:
        t0 = time.time()
        streams, scales, nb, ne = apply_scheme(body, embed, kb, ke)
        ppl_w = perplexity(model, wiki)
        ppl_c = perplexity(model, code)
        if kb is None:
            dig_bpp = scale_mb = 0.0
            n_quant = 0
        else:
            dig_bpp, n_quant = rate_bits(streams, kb)
            scale_mb = sum(scale_codec_bytes(s) for s in scales) / 1e6
        dig_mb = dig_bpp * n_quant / 8 / 1e6
        # anything not quantized stays fp32: rest always, embed if k_embed is
        # None, and the whole body if k_body is None (fp32 baseline row)
        fp32_params = (n_rest + (n_embed if ke is None else 0)
                       + (n_body if kb is None else 0))
        rest_mb = 4.0 * fp32_params / 1e6
        total_mb = dig_mb + scale_mb + rest_mb
        rows.append(dict(scheme=name, k_body=kb or 0, k_embed=ke or 0,
                         digit_bpp=dig_bpp, digit_mb=dig_mb, scale_mb=scale_mb,
                         rest_mb=rest_mb, total_mb=total_mb, ppl_wiki=ppl_w,
                         ppl_code=ppl_c, secs=time.time() - t0))
        print(f"{name:<26} ppl wiki {ppl_w:8.3f} code {ppl_c:8.3f} | digits"
              f" {dig_bpp:5.3f} b/p + scales {scale_mb:6.2f} MB + fp32 {rest_mb:7.1f}"
              f" MB = {total_mb:7.1f} MB | {time.time()-t0:.0f}s")
        restore()

    # rANS vs entropy check on the K9 body (re-quantize once)
    kb9 = [r for r in rows if r["k_body"] == 9][0]
    streams, _, _, _ = apply_scheme(body, embed, 9, None)
    e_b, r_b = rans_check(streams, 9)
    restore()
    print(f"rANS check (K9 body, 3 tensors): entropy {e_b:.4f} vs rANS {r_b:.4f} b/param")

    cols = list(rows[0].keys())
    with open(RESULTS / "exp17_qwen_k9_ptq.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    base = rows[0]
    lines = [
        "# exp17 — K9 PTQ on Qwen2.5-Coder-1.5B-Instruct",
        "",
        f"Modern coder LLM ({n_total:,} dedup params; body {n_body:,} Linear,",
        f"tied embedding {n_embed:,}, fp32 rest {n_rest:,}). Eval: {N_EVAL}x{BLOCK}",
        "tokens of WikiText-103 test and a held-out Python slice.",
        "Rate = per-tensor digit entropy + entropy-coded log scales + fp32 rest",
        f"(RMSNorm weights + Qwen2 attention biases). rANS == entropy on this"
        f" model to {abs(r_b-e_b):.4f} b/param (3-tensor check).",
        "",
        "| scheme | k body | k embed | digits b/p | model MB | ppl wiki | ppl code |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['scheme']} | {r['k_body'] or '—'} | {r['k_embed'] or 'fp32'}"
                     f" | {r['digit_bpp']:.3f} | {r['total_mb']:.1f} |"
                     f" {r['ppl_wiki']:.3f} | {r['ppl_code']:.3f} |")
    body_k = {r["k_body"] for r in rows if r["k_embed"] == 0}
    frontier = ""
    if {8, 9, 15} <= body_k:
        r8 = next(r for r in rows if r["k_body"] == 8 and r["k_embed"] == 0)
        r9 = next(r for r in rows if r["k_body"] == 9 and r["k_embed"] == 0)
        r15 = next(r for r in rows if r["k_body"] == 15 and r["k_embed"] == 0)
        y = r8["ppl_code"] + (r15["ppl_code"] - r8["ppl_code"]) * \
            (r9["digit_bpp"] - r8["digit_bpp"]) / (r15["digit_bpp"] - r8["digit_bpp"])
        below = r9["ppl_code"] < y
        frontier = (f"**Frontier check (body, code ppl):** 9-ninths at "
                    f"{r9['digit_bpp']:.3f} b/p gives {r9['ppl_code']:.3f}; the "
                    f"8<->15 linear interpolation at the same rate gives {y:.3f} — "
                    f"the odd middle alphabet is {'BELOW' if below else 'ABOVE'} it "
                    f"({'frontier-efficient (P10-style)' if below else 'not special'}).")

    lines += [
        "",
        frontier,
        "",
        f"fp32 footprint {4.0*n_total/1e6:.1f} MB. Body-only schemes leave the",
        "tied embedding fp32 (its 0.23B params dominate the MB column — the exp10",
        "lesson at a new scale); the full rows quantize it too.",
        "",
        f"wall {time.time()-t_all:.0f}s", "",
    ]
    (RESULTS / "exp17_qwen_k9_ptq.md").write_text("\n".join(lines))
    print("wrote results/exp17_qwen_k9_ptq.{csv,md}")


if __name__ == "__main__":
    main()
