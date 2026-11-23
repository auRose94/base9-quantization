#!/usr/bin/env python3
"""A1: train a tiny llama-arch char model (the resident tier's first LM).

Architecture mirrors the ninths stage-2 container family so ONE transformer
block RTL serves this resident demo AND the SmolLM2-135M streamed tier:
RMSNorm (pre) + GQA-parameterizable attention (RoPE, theta 1e4) + SwiGLU MLP,
tied embeddings. fp32 CPU, deterministic; no dropout, no autocast.

Output (results dir, default ./out):
  conf.json        architecture config (the packer + board RTL read this)
  state.safetensors  fp32 weights
  loss.csv         iter,loss
  eval_<iter>.txt  greedy sample at checkpoints
  final_loss.txt

Run:  python3 partner/char9_train.py --iters 1500 [--out DIR]
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── model ────────────────────────────────────────────────────────────────────
class RMS(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x):                      # (b,t,d) fp32
        var = (x * x).mean(-1, keepdim=True)   # plain mean — matches board plan
        x = x * torch.rsqrt(var + self.eps)
        return x * self.w


def rope_theta(dim, theta=1e4, seq=128, device=None):
    i = torch.arange(dim // 2, device=device, dtype=torch.float64)
    inv = 1.0 / theta ** (2.0 * i / dim)
    t = torch.arange(seq, device=device, dtype=torch.float64)
    f = torch.outer(t, inv)                    # (seq, dh)
    return f.cos().float(), f.sin().float()


def apply_rope(x, cos, sin):                   # x (b,h,t,hh) half-pair split
    x1, x2 = x[..., 0::2], x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, d, heads, kv_heads, max_seq):
        super().__init__()
        self.h, self.kv, self.dh = heads, kv_heads, d // heads
        self.q = nn.Linear(d, heads * self.dh, bias=False)
        self.k = nn.Linear(d, kv_heads * self.dh, bias=False)
        self.v = nn.Linear(d, kv_heads * self.dh, bias=False)
        self.o = nn.Linear(heads * self.dh, d, bias=False)
        self.max_seq = max_seq

    def forward(self, x, cos, sin, mask):
        b, t, _ = x.shape
        q = apply_rope(self.q(x).view(b, t, self.h, self.dh).transpose(1, 2), cos, sin)
        k = apply_rope(self.k(x).view(b, t, self.kv, self.dh).transpose(1, 2), cos, sin)
        v = self.v(x).view(b, t, self.kv, self.dh).transpose(1, 2)
        if self.kv != self.h:
            rep = self.h // self.kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        att = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        att = att.masked_fill(mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        return self.o((att @ v).transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.hidden = hidden                     # divisible by 256/128 mults of 128

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, heads, kv_heads, hidden, max_seq):
        super().__init__()
        self.n1, self.n2 = RMS(d), RMS(d)
        self.attn = Attn(d, heads, kv_heads, max_seq)
        self.mlp = SwiGLU(d, hidden)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.n1(x), cos, sin, mask)
        return x + self.mlp(self.n2(x))


class TinyLlama(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        d, n = conf["d"], conf["n_layers"]
        self.embed = nn.Embedding(conf["vocab"], d)
        self.blocks = nn.ModuleList([
            Block(d, conf["n_heads"], conf["n_kv_heads"], conf["mlp"], conf["seq"])
            for _ in range(n)])
        self.norm = RMS(d)
        self.lm_head = nn.Linear(d, conf["vocab"], bias=False)   # tied below
        cos, sin = rope_theta(d // conf["n_heads"], conf["rope_theta"], conf["seq"] * 2)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("mask",
                             torch.tril(torch.ones(conf["seq"] * 2, conf["seq"] * 2)).view(
                                 1, 1, conf["seq"] * 2, conf["seq"] * 2),
                             persistent=False)

    def forward(self, idx):
        pos = self._pos
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x, self.cos[pos:pos + idx.shape[1]], self.sin[pos:pos + idx.shape[1]], self.mask)
        return self.lm_head(self.norm(x))

    # position offset for cached generation; training always starts at 0
    _pos = 0

    def tie(self):
        self.lm_head.weight = self.embed.weight


# ── data ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "char9_out")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data/input.txt")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--resume", type=Path, default=None,
                    help="state.safetensors to warm-start (val-guarded mode)")
    ap.add_argument("--val-patience", type=int, default=0,
                    help=">0: hold out last 10%% as val; early-stop when val "
                         "loss stops improving for this many checks")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    conf = dict(vocab=65, d=256, n_layers=4, n_heads=8, n_kv_heads=8,
                mlp=1024, seq=128, rope_theta=1_0000, tie=True, norm_eps=1e-5,
                corpus=str(args.data.name), seed=7)

    torch.manual_seed(conf["seed"])
    text = args.data.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    model = TinyLlama(conf)
    model.tie()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params={nparams:,}  ({4.25*1.024*nparams/2**20:.2f} MiB @ K9_4)", flush=True)

    if args.resume is not None:
        from safetensors.torch import load_file
        state = load_file(str(args.resume))
        model.load_state_dict(
            {**state, "lm_head.weight": state["embed.weight"]}, strict=True)
        print(f"resumed from {args.resume}", flush=True)

    # optional val split (last 10%): the "keep training" mode early-stops on it
    n_val = int(len(data) * 0.10) if args.val_patience > 0 else 0
    data_train, data_val = (data[:len(data) - n_val], data[len(data) - n_val - conf["seq"]:]) \
        if n_val else (data, None)
    print(f"train tokens {len(data_train):,}", f"val tokens {len(data_val):,}"
          if data_val is not None else "", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.03)

    itoi = {i: c for c, i in stoi.items()}

    def sample(prefix="O", n=220, temp=0.0):
        model.eval()
        idx = torch.tensor([[stoi[c] for c in prefix]], dtype=torch.long)
        model._pos = 0
        for _ in range(n):
            x = idx[:, -conf["seq"]:]
            model._pos = 0 if idx.shape[1] <= conf["seq"] else idx.shape[1] - conf["seq"]
            logits = model(x)
            nxt = torch.argmax(logits[0, -1]) if temp == 0 else None
            idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
        model.train()
        return "".join(itoi[i] for i in idx[0].tolist())

    loss_csv = out / "loss.csv"
    loss_csv.write_text("iter,loss\n")
    best_val, stale = float("inf"), 0
    t0 = time.time()

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            losses = []
            for i in range(0, len(data_val) - conf["seq"] - 1, conf["seq"] * 8):
                ix = data_val[i:i + conf["seq"]]
                xb = ix[None, :-1]
                yb = data_val[i + 1:i + conf["seq"]]
                model._pos = 0
                losses.append(F.cross_entropy(
                    model(xb).reshape(-1, conf["vocab"]), yb).item())
        model.train()
        return sum(losses) / len(losses)

    for it in range(1, args.iters + 1):
        stop = False
        ix = torch.randint(len(data_train) - conf["seq"] - 1, (args.batch,))
        xb = torch.stack([data_train[i:i + conf["seq"]] for i in ix])
        yb = torch.stack([data_train[i + 1:i + 1 + conf["seq"]] for i in ix])
        loss = F.cross_entropy(model(xb).reshape(-1, conf["vocab"]), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        log_loss = loss.item()
        if it % 25 == 0 or it == 1:
            with loss_csv.open("a") as f:
                f.write(f"{it},{log_loss:.5f}\n")
            print(f"iter {it:5d}  loss {log_loss:.4f}  "
                  f"({time.time()-t0:6.0f}s)", flush=True)
        if args.val_patience > 0 and (it % args.eval_every == 0 or it == args.iters):
            v = val_loss()
            print(f"iter {it:5d}  VAL {v:.5f}  (best {best_val:.5f}, stale {stale})",
                  flush=True)
            if v < best_val - 0.1:           # rose's rule: progress = 0.1000+
                best_val, stale = v, 0
                from safetensors.torch import save_file
                bstate = {kk: vv.to(torch.float32) for kk, vv in
                          model.state_dict().items() if kk != "lm_head.weight"}
                save_file(bstate, str(out / "state_best.safetensors"))
                if best_val < 1.0:
                    print("MILESTONE: val loss under 1.0", flush=True)
            else:
                stale += 1
                if stale >= args.val_patience:
                    print(f"early stop: val gains < 0.1000 for {stale} rounds",
                          flush=True)
                    stop = True
        if it % args.eval_every == 0 or it == args.iters or stop:
            (out / f"eval_{it}.txt").write_text(sample())
        (out / "final_loss.txt").write_text(f"{log_loss:.5f}\n")
        if stop:
            break
    (out / "eval_prompt.txt").write_text("O\n")
    from safetensors.torch import save_file
    save_file({k: v.to(torch.float32) for k, v in
               model.state_dict().items() if k != "lm_head.weight"},
              str(out / "state.safetensors"))   # FINAL state, always
    if args.val_patience > 0 and (out / "state_best.safetensors").exists():
        (out / "state_best.safetensors").replace(out / "state.safetensors")
    conf_v = {**conf, "val_guard": args.val_patience > 0}
    (out / "conf.json").write_text(json.dumps(conf_v, indent=2))
    print("saved to", out, flush=True)


if __name__ == "__main__":
    main()