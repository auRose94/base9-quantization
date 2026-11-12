"""E3: noise-aware training — adapters ON a noisy analog substrate.

The analog array is modeled as the FROZEN base model's weights, re-noised on
every use (a real array's weights are noisy at every read); LoRA adapters are
the digital part trained around it.  Each step:

    add_noise(sigma)   # save base params, add group-64-referenced Gaussian
    loss = forward(batch)
    loss.backward()
    remove_noise()     # restore base params exactly
    optimizer.step()   # LoRA-only update

Control run = --sigma 0 (identical recipe, no perturbation).  Comparing the
survival curves of control-vs-noise merges answers both open questions at
once: does noise-aware training flatten the analog cliff (Result B/D
mechanism), and does it cost or help clean quality (noise-as-regularizer)?

Run env: .venv-rocm (torch rocm7.2, transformers 5.15, peft, pandas).
House rules: capped unit for the real runs; XT stays free of servers.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA = REPO / "phase2/data/gdscript_train.parquet"
VAL = REPO / "phase2/data/gdscript_validation.parquet"
OUTROOT = REPO / "results/e3_noiseaware"
GROUP = 64


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class SubstrateNoise:
    """Group-64-referenced Gaussian noise + optional stuck-at cells on frozen
    base params, applied per use (a real array's stuck cells are fixed; the
    per-step fresh mask here is the conservative worst case — every step sees
    a different 1% of cells dead)."""

    def __init__(self, model, sigma: float, stuck_p: float = 0.0):
        self.sigma = sigma
        self.stuck_p = stuck_p
        self.saved: dict[str, torch.Tensor] = {}
        self.params = [(n, p) for n, p in model.named_parameters()
                       if not p.requires_grad]

    def add(self) -> None:
        if self.sigma <= 0.0:
            return
        for n, p in self.params:
            self.saved[n] = p.data.clone()
            t = p.data
            if t.numel() % GROUP == 0:
                flat = t.view(-1, GROUP)
            else:
                flat = t.view(1, -1)
            ref = flat.abs().amax(dim=-1, keepdim=True)  # group-64 reference
            noise = torch.randn(flat.shape, device=t.device,
                                dtype=torch.float32).to(t.dtype)
            flat.add_(noise * (self.sigma * ref))
            if self.stuck_p > 0.0:
                mask = torch.rand(flat.shape, device=t.device) < self.stuck_p
                if mask.any():
                    rail = torch.where(torch.rand(flat.shape, device=t.device) < 0.5,
                                       ref, -ref).to(t.dtype)
                    flat[mask] = rail[mask]

    def remove(self) -> None:
        if not self.saved:
            return
        for n, p in self.params:
            p.data.copy_(self.saved.pop(n))


def build_dataset(tok, df: pd.DataFrame, seq: int, limit: int | None = None):
    col = "text" if "text" in df.columns else None
    if col is None:
        col = df.columns[0]
    texts = df[col].astype(str).tolist()
    if limit:
        texts = texts[:limit]
    ids: list[int] = []
    eos = tok.eos_token_id or 151643
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(eos)
    n = (len(ids) // seq) * seq
    blocks = torch.tensor(ids[:n], dtype=torch.long).view(-1, seq)
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sigma", type=float, default=0.01,
                    help="substrate noise, fraction of group-64 amax (0 = control)")
    ap.add_argument("--sigma-max", type=float, default=0.0,
                    help=">0 = MIXED-sigma training: draw sigma ~ U(0, sigma-max) "
                         "fresh each step (one model robust across the band — "
                         "the appliance's real operating condition)")
    ap.add_argument("--stuck-p", type=float, default=0.0,
                    help="fraction of cells stuck at a rail each step (fresh mask)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap training samples (smoke)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.batch, args.limit = 30, 4, 200
    tag = args.tag or (f"e3_sigma{args.sigma:g}_s{args.steps}")
    out = OUTROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    dev = "cuda:0"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to(dev)
    model.config.use_cache = False

    lconf = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                       bias="none", task_type="CAUSAL_LM",
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

    noise = SubstrateNoise(model.get_base_model(), args.sigma, stuck_p=args.stuck_p)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"substrate params (frozen): {n_frozen/1e6:.0f}M; sigma={args.sigma:g}")

    train_df = pd.read_parquet(DATA)
    blocks = build_dataset(tok, train_df, args.seq, args.limit)
    print(f"train blocks: {blocks.shape[0]} x {args.seq}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 60) *
        max(0.05, 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    losses = []
    for step in range(args.steps):
        idx = rng.integers(0, blocks.shape[0], size=args.batch)
        batch = blocks[idx].to(dev)
        if args.sigma_max > 0.0:
            noise.sigma = float(rng.uniform(0.0, args.sigma_max))
        noise.add()
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        noise.remove()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if step % 50 == 0 or step == args.steps - 1:
            dt = time.time() - t0
            print(f"step {step:5d}  loss {loss.item():.4f}  "
                  f"avg50 {sum(losses[-50:])/len(losses[-50:]):.4f}  "
                  f"({dt:.0f}s, {dt/(step+1):.2f}s/step)", flush=True)

    # clean validation loss (noise OFF — the model's own quality)
    model.eval()
    val_df = pd.read_parquet(VAL)
    vblocks = build_dataset(tok, val_df, args.seq, 256)[:64]
    tot = 0.0
    with torch.no_grad():
        for i in range(0, vblocks.shape[0], 8):
            vb = vblocks[i:i + 8].to(dev)
            tot += float(model(input_ids=vb, labels=vb).loss) * vb.shape[0]
    val_loss = tot / vblocks.shape[0]
    print(f"clean val loss: {val_loss:.4f} (per-step avg {math.exp(val_loss):.2f} nll-ppl)")

    model.save_pretrained(out / "adapters")
    tok.save_pretrained(out / "adapters")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    (out / "run_config.txt").write_text(
        f"sigma={args.sigma} sigma_max={args.sigma_max} stuck_p={args.stuck_p} "
        f"steps={args.steps} batch={args.batch} seq={args.seq} "
        f"lr={args.lr} lora_r={args.lora_r} seed={args.seed} "
        f"final_avg50={sum(losses[-50:])/50:.4f} clean_val_loss={val_loss:.4f}\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
