#!/usr/bin/env python3
"""QLoRA SFT on the verified Godot-4 corpus.

Prompt tokens are masked (-100); the loss is on the assistant's GDScript only,
so the adapter learns to *write* Godot 4 rather than to memorize file names.
4-bit NF4 base + LoRA on every projection, so a 1.5B trains alongside the
running chat server in a few GB, and a 7B would fit a 16 GB card.

Held out by project: any project whose name hashes into the holdout bucket is
excluded from training entirely, so validation loss is on projects the model
never saw (files within a project are near-duplicates of each other, so a
per-file split would leak).

    python3 train_qlora.py --data out/godot4_sft.jsonl --max-samples 300 --epochs 1
    python3 train_qlora.py --data out/godot4_sft.jsonl --epochs 2 --out out/qlora_1.5b
    python3 train_qlora.py --model Qwen/Qwen2.5-Coder-7B-Instruct --out out/qlora_7b
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch                                                      # noqa: E402
import torch.nn.functional as F                                   # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,    # noqa: E402
                          BitsAndBytesConfig)

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]


def chunked_loss(model, input_ids, attention_mask, labels, chunk=256):
    """The ordinary HF CE loss, but the vocab logits are built per position
    chunk inside a checkpoint, so the fp32 upcast over a 152k vocab never
    materializes full-size (a batch-8 x 2048 logits+CE spike is ~15 GB; this
    path's transient is ~0.4 GB). Mathematically the same mean-over-masked
    cross-entropy HF's ForCausalLMLoss computes; the chunked softmax is
    recomputed in each chunk's backward (lm_head is ~7% of model FLOPs)."""
    body = model.get_base_model().model          # Qwen2Model (LoRA-wrapped target
    head = model.get_base_model().lm_head        # Linear modules live inside it)
    hidden = body(input_ids=input_ids,
                  attention_mask=attention_mask).last_hidden_state
    shift = labels[:, 1:]
    total = None
    for c in range(0, input_ids.shape[1] - 1, chunk):
        e = min(c + chunk, input_ids.shape[1] - 1)

        def part(h, lab_chunk, w=None):          # w pins head for the closure
            logits = w(h).float()
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                lab_chunk.reshape(-1), ignore_index=-100, reduction="sum")

        s = torch.utils.checkpoint.checkpoint(part, hidden[:, c:e, :], shift[:, c:e],
                                              head, use_reentrant=False)
        total = s if total is None else total + s
    n = int((shift != -100).sum())
    return total / max(n, 1)


def holdout(project: str, mod: int, bucket: int = 0) -> bool:
    """Deterministic project-level split, shared with the evaluator."""
    return int(hashlib.sha1(project.encode()).hexdigest()[:8], 16) % mod == bucket


def encode(tok, msgs, seq_len):
    """input_ids + labels with the prompt masked."""
    prefix = tok.apply_chat_template(msgs[:2], tokenize=False,
                                     add_generation_prompt=True)
    body = msgs[2]["content"]
    pid = tok(prefix, add_special_tokens=False)["input_ids"]
    bid = tok(body, add_special_tokens=False)["input_ids"]
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    if len(pid) >= seq_len - 8:
        return None                      # prompt alone eats the window
    room = seq_len - len(pid)
    bid = bid[:room - len(eos)]
    ids = pid + bid + eos
    labels = [-100] * len(pid) + bid + eos
    return ids, labels


def pad_batch(batch, pad_id):
    n = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(x)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids, lab, att


def batches(data, bs, shuffle=True, seed=0):
    idx = list(range(len(data)))
    if shuffle:
        import random
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [data[j] for j in idx[i:i + bs]]


@torch.no_grad()
def evaluate(model, data, bs, pad_id, device, limit=8):
    model.eval()
    tot = n = 0.0
    for batch in batches(data[:limit * bs], bs, shuffle=False):
        ids, lab, att = pad_batch(batch, pad_id)
        out = model(input_ids=ids.to(device), attention_mask=att.to(device),
                    labels=lab.to(device))
        tot += float(out.loss)
        n += 1
    model.train()
    return tot / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/godot4_sft.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--out", default="out/qlora_1.5b")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--only-clean", action="store_true",
                    help="train only on files that parsed outright (drop context-only)")
    ap.add_argument("--holdout-mod", type=int, default=50)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="",
                    help="adapter dir from an interrupted run to continue from")
    ap.add_argument("--start-step", type=int, default=0,
                    help="batch position already trained when interrupted "
                         "(steps are batches here, batch=1; make it a multiple "
                         "of grad-accum so accumulation stays aligned)")
    ap.add_argument("--vram-cap", type=float, default=0.0,
                    help="0<f<=1: torch.cuda memory-fraction cap, so a trainer "
                         "sharing a card with the desktop can't evict kwin")
    ap.add_argument("--shrink-frozen", action="store_true",
                    help="recast peft's fp32 upcast of the frozen embed/lm_head "
                         "back to bf16 (needed to fit 16 GB cards)")
    ap.add_argument("--ce-chunk", type=int, default=256,
                    help="token-chunk of the checkpointed CE; smaller halves "
                         "the vocab-logits transients, loss value unchanged")
    ap.add_argument("--no-4bit", action="store_true",
                    help="bf16 LoRA instead of QLoRA (more VRAM, faster)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    if a.vram_cap:
        torch.cuda.set_per_process_memory_fraction(a.vram_cap)

    path = Path(a.data) if Path(a.data).is_absolute() else HERE / a.data
    rows, bad = [], 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1          # tolerate a partially written corpus
    if bad:
        print(f"  skipped {bad} unparsable line(s) in {path.name}")
    if a.only_clean:
        rows = [r for r in rows if r.get("verdict", "clean") == "clean"]
        print(f"  only-clean filter: {len(rows):,} rows")
    train_rows = [r for r in rows if not holdout(r["project"], a.holdout_mod)]
    val_rows = [r for r in rows if holdout(r["project"], a.holdout_mod)]
    if a.max_samples:
        train_rows = train_rows[:a.max_samples]
    print(f"data {path.name}: {len(rows):,} rows | train {len(train_rows):,} "
          f"| held-out projects {len(val_rows):,} rows", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    enc_train, enc_val, dropped = [], [], 0
    for r in train_rows:
        e = encode(tok, r["messages"], a.seq_len)
        if e is None:
            dropped += 1
        else:
            enc_train.append(e)
    for r in val_rows:
        e = encode(tok, r["messages"], a.seq_len)
        if e is not None:
            enc_val.append(e)
    print(f"encoded: train {len(enc_train):,} (dropped {dropped}) "
          f"| val {len(enc_val):,}", flush=True)
    if not enc_train:
        print("nothing to train on")
        return 1

    qcfg = None if a.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=qcfg,
        dtype=torch.bfloat16 if a.no_4bit else None,
        low_cpu_mem_usage=True, attn_implementation="sdpa")
    if dev == "cuda" and a.no_4bit:
        model = model.to(dev)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if qcfg is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        if a.shrink_frozen:
            # peft blanket-casts every frozen bf16 tensor up to fp32; on a 16 GB
            # card that is +3.1 GB each for embed/lm_head and the run OOMs. Cast
            # them back — they never train (LoRA targets the projections), and
            # the frozen-quantized body was computed in bf16 anyway.
            for getter in ("get_input_embeddings", "get_output_embeddings"):
                mod = getattr(model, getter)()
                if mod is not None and mod.weight.dtype == torch.float32:
                    mod.weight.data = mod.weight.data.to(torch.bfloat16)
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha,
                      lora_dropout=a.lora_dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=TARGETS)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.config.use_cache = False

    params = [p for p in model.parameters() if p.requires_grad]
    base_vl = None
    if a.resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        adir = Path(a.resume) if Path(a.resume).is_absolute() else HERE / a.resume
        set_peft_model_state_dict(model, load_file(adir / "adapter_model.safetensors"))
        # fresh LoRA starts with B=0, so nonzero B proves the load actually landed
        n_b = sum(1 for n, _ in model.named_parameters() if "lora_B" in n)
        live_b = sum(1 for n, p in model.named_parameters()
                     if "lora_B" in n and float(p.abs().sum()) > 0)
        print(f"resume: {adir} → {live_b}/{n_b} lora_B tensors alive", flush=True)
        if live_b < 0.5 * n_b:
            print("  adapter failed to load (lora_B still zero) — aborting")
            return 1
        # the load's transient copies sit in the allocator's cache; release
        # them so the first batch's dequant spike has a clean card (the
        # 2026-10-06 empty-card OOMs missed by <0.1 GB at this exact moment)
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # LoRA's B matrix is zero-initialised, so right now the wrapped model is
        # numerically the base model — a free "before" number on held-out projects.
        base_vl = evaluate(model, enc_val, a.batch, tok.pad_token_id,
                           dev) if enc_val else None
    if base_vl is not None:
        print(f"base val loss (held-out projects): {base_vl:.4f}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    steps_per_epoch = math.ceil(len(enc_train) / (a.batch * a.grad_accum))
    total = max(1, int(steps_per_epoch * a.epochs))
    if a.max_steps:
        total = min(total, a.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=total, pct_start=min(0.05, a.warmup / max(total, 1)),
        anneal_strategy="cos", div_factor=10.0, final_div_factor=20.0)
    print(f"{total} optimizer steps ({steps_per_epoch}/epoch, "
          f"batch {a.batch} x accum {a.grad_accum} = "
          f"{a.batch*a.grad_accum} samples/step)", flush=True)

    outdir = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    log = open(outdir / "train_log.jsonl", "w")
    hist, step, t0 = [], 0, time.time()
    if a.start_step:
        # the scheduler only advances on optimizer steps (one per grad-accum
        # batches), so replay it — not the data — to the interruption point
        for _ in range(a.start_step // a.grad_accum):
            sched.step()
        step = a.start_step
        print(f"resumed at batch {step}: lr back to "
              f"{sched.get_last_lr()[0]:.2e}", flush=True)
    stop = False
    for ep in range(math.ceil(a.epochs)):
        skip = a.start_step * a.batch if ep == 0 else 0
        for bi, batch in enumerate(batches(enc_train, a.batch, seed=a.seed + ep)):
            if stop:
                break
            if bi < skip:
                continue
            ids, lab, att = pad_batch(batch, tok.pad_token_id)
            loss = chunked_loss(model, ids.to(dev), att.to(dev), lab.to(dev),
                                chunk=a.ce_chunk)
            (loss / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0 or step + 1 >= total:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0 or step == 1 or step >= total:
                rec = dict(step=step, total=total, loss=round(float(loss.detach()), 4),
                           lr=round(sched.get_last_lr()[0], 8),
                           sec=round(time.time() - t0, 1))
                hist.append(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"  step {step:5d}/{total} loss {rec['loss']:.4f} "
                      f"lr {rec['lr']:.2e} {rec['sec']:.0f}s", flush=True)
            if step % a.save_every == 0:
                model.save_pretrained(outdir / "adapter")
                print(f"  saved adapter @ step {step}", flush=True)
            if step >= total:
                stop = True
                break

    model.save_pretrained(outdir / "adapter")
    tok.save_pretrained(outdir / "adapter")
    vl = evaluate(model, enc_val, a.batch, tok.pad_token_id, dev) if enc_val else None
    base_vl = None
    summary = dict(model=a.model, data=str(path), rows=len(rows),
                   train_samples=len(enc_train), val_samples=len(enc_val),
                   epochs=a.epochs, steps=step, lora_r=a.lora_r,
                   seq_len=a.seq_len, batch=a.batch, grad_accum=a.grad_accum,
                   lr=a.lr, final_loss=hist[-1]["loss"] if hist else None,
                   first_loss=hist[0]["loss"] if hist else None,
                   val_loss=round(vl, 4) if vl else None,
                   minutes=round((time.time() - t0) / 60, 1))
    if base_vl is not None:
        summary["val_loss_base"] = round(base_vl, 4)
    (outdir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nadapter → {outdir/'adapter'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
