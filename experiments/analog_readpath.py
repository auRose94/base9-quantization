"""Phase B: the analog READ path — DAC/MVM-noise/ADC on activations.

Phase A (analog_channel.py) modeled what the array does to WEIGHTS between
program and hold. This models what the array does to SIGNALS on every pass:

  DAC   (pre-hook, per token vector): input activations quantized to
        `dac_bits` levels against the token's own full scale — the input
        DAC driving the crossbar columns.
  col   (pre-hook, fixed per layer): per-input-channel gain error ~N(1, col_gamma)
        drawn ONCE per linear — systematic column mismatch (the DAC/column
        spread that calibration would remove).
  MVM   (post-hook, per token vector): additive noise sigma_readpath * the
        token's output full scale — sense-amp + thermal noise on the
        partial sums.
  ADC   (post-hook): output quantized to `adc_bits` levels per token.

Body linears only; the lm_head stays digital so degradation is attributable
to the body.  Compose with the write path via --sigma-w (weight noise
applied at load through analog_channel, group reference).

The payoff A/B: base model vs the E3 mixed-sigma-trained merge
(results/e3_noiseaware/mixed/merged) — does noise-aware training survive the
read path too?

Run: .venv-rocm/bin/python (ROCm torch). ppl is torch-NLL over wiki2 blocks
(seq 512, first N blocks — NOT llama-perplexity; compare within this table
only).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))
from analog_channel import AnalogParams, analog_channel, tensor_rng  # noqa: E402

BASE = REPO / "results/e3_noiseaware/base_model"
MIXED = REPO / "results/e3_noiseaware/mixed/merged"
CORPUS = REPO / "results/wiki2_raw_test.txt"
OUT = REPO / "results/analog/readpath.csv"


def quantize_per_row(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return x
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    step = (2 * amax / (2 ** bits - 1))
    return torch.round(x / step) * step


def build_hooks(model, dac_bits, adc_bits, col_gamma, sigma, seed):
    handles = []
    n_lin = 0

    def make(name):
        rng = torch.Generator(device="cpu").manual_seed(
            abs(hash(name)) % (2 ** 31) + seed)

        def pre(module, args):
            x = args[0]
            if col_gamma > 0:
                g = (1.0 + col_gamma * torch.randn(
                    x.shape[-1], generator=rng).clamp(-4, 4)
                ).to(device=x.device, dtype=x.dtype)
                x = x * g
            x = quantize_per_row(x, dac_bits)
            return (x,) + tuple(args[1:])

        def post(module, args, output):
            y = output
            if sigma > 0:
                ref = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
                noise = torch.randn(y.shape, device=y.device, dtype=torch.float32)
                y = y + (sigma * ref).float() * noise
            return quantize_per_row(y.to(y.dtype), adc_bits).to(output.dtype)

        return pre, post

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            pre, post = make(name)
            handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))
            n_lin += 1
    return handles, n_lin


@torch.no_grad()
def eval_ppl(model, blocks, device):
    tot, ntok = 0.0, 0
    for i in range(0, blocks.shape[0], 4):
        b = blocks[i:i + 4].to(device)
        out = model(input_ids=b, labels=b)
        tot += float(out.loss) * b.numel()
        ntok += b.numel()
    return math.exp(tot / ntok)


def tokenize(tok, seq, n_blocks):
    ids = []
    eos = tok.eos_token_id or 151643
    for line in open(CORPUS, encoding="utf-8", errors="ignore"):
        ids.extend(tok(line, add_special_tokens=False)["input_ids"])
        ids.append(eos)
        if len(ids) >= seq * n_blocks:
            break
    n = (len(ids) // seq) * seq
    return torch.tensor(ids[:n], dtype=torch.long).view(-1, seq)


def load_model(path, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    m.to(device).eval()
    m.config.use_cache = False
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="base",
                    help="comma list of base[,mixed] — local HF dirs")
    ap.add_argument("--sigmas", default="0,0.0025,0.005,0.01,0.02")
    ap.add_argument("--dac-bits", type=int, default=0)
    ap.add_argument("--adc-bits", type=int, default=0)
    ap.add_argument("--col-gamma", type=float, default=0.0)
    ap.add_argument("--sigma-w", type=float, default=0.0,
                    help="compose write-path weight noise (group ref)")
    ap.add_argument("--blocks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.blocks, args.sigmas = 4, "0,0.01"

    device = "cuda:0"
    paths = {"base": BASE, "mixed": MIXED}
    models = {m: paths[m] for m in args.models.split(",")}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(BASE))
    blocks = tokenize(tok, 512, args.blocks)
    print(f"corpus blocks: {blocks.shape[0]} x 512")

    sigmas = [float(s) for s in args.sigmas.split(",")]
    rows = []
    for label, path in models.items():
        model = load_model(path, device)
        if args.sigma_w > 0:
            for n, p in model.named_parameters():
                if p.numel() % 64 == 0 and p.dim() >= 1:
                    w = p.data.detach().float().cpu().numpy()
                    out = analog_channel(w, AnalogParams(
                        sigma_prog=args.sigma_w, scale_ref="group"),
                        tensor_rng(args.seed, n))
                    p.data.copy_(torch.from_numpy(out).to(p.dtype).to(device))
            print(f"  [{label}] write-path sigma_w={args.sigma_w} applied")
        for sigma in sigmas:
            handles, n_lin = build_hooks(
                model, args.dac_bits, args.adc_bits, args.col_gamma, sigma,
                args.seed)
            t0 = time.time()
            ppl = eval_ppl(model, blocks, device)
            for h in handles:
                h.remove()
            row = dict(model=label, sigma_readpath=sigma, dac_bits=args.dac_bits,
                       adc_bits=args.adc_bits, col_gamma=args.col_gamma,
                       sigma_w=args.sigma_w, blocks=args.blocks, seed=args.seed,
                       ppl=f"{ppl:.4f}", n_linears=n_lin,
                       secs=f"{time.time()-t0:.1f}")
            rows.append(row)
            print(f"[{label}] sigma_r={sigma:g} dac={args.dac_bits} "
                  f"adc={args.adc_bits} colg={args.col_gamma:g} "
                  f"sw={args.sigma_w:g} -> ppl {ppl:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fresh = not OUT.exists()
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if fresh:
            w.writeheader()
        w.writerows(rows)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
