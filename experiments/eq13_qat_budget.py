#!/usr/bin/env python3
"""eq13 — closing the 33M QAT protocol set (session 5).

Fixes vs eq12: fresh model per arm (cold init always), schedule passed as a
parameter factory (no module-captured LR). Axes: steps (4000/8000 @ lr 1e-4)
and LR (2e-4 @ 2000) against the eq12 cold baseline (1.5227 @ 2000@1e-4) and
the converged control (1.2817).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402
from eq6_func_fit import quant_rtn_odd  # noqa: E402
import eq11_scaleup as E  # noqa: E402  (dequant_grid, tokens_for, load_docs, make_spec)

B9 = "/mnt/matrix/Work/base9-quantization"
os.environ.setdefault("HF_HOME", str(E.Path(B9) / ".hf-cache"))
sys.path.insert(0, os.path.join(B9, "experiments"))
import k9  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DEV = "cuda"
RES = E.RES
BATCH, BLOCK = 8, 512


def make_lr_fn(lr, warmup=100, steps=None):
    def lr_at(step):
        if step < warmup:
            return max(lr * (step + 1) / warmup, 1e-6)
        p = (step - warmup) / max((steps or 2000) - warmup, 1)
        return 1e-5 + (lr - 1e-5) * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))
    return lr_at


def run_arm(tag, steps, lr, spec0, model, tok, valid, train, tokens):
    spec = E.make_spec(model, spec0)   # fresh model per arm
    in_spec = set(id(p) for _, p, _ in spec)
    rest = [p for p in model.parameters() if id(p) not in in_spec]
    lr_fn = make_lr_fn(lr, steps=steps)
    rng = np.random.default_rng(42)
    latents, leaves = [], []
    for name, p, k in spec:
        val, _ = E.dequant_grid(p.data, k)
        leaf = torch.nn.Parameter(val, requires_grad=True)
        lat = torch.nn.Parameter(p.data.clone())
        leaves.append((name, p, leaf, lat, k))
        latents.append(lat)
    opt = torch.optim.AdamW(latents + rest, lr=1.0, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    t0 = time.time()
    for step in range(steps):
        pos = rng.integers(0, len(train) - BLOCK, size=BATCH)
        xb = torch.from_numpy(np.stack([train[p:p + BLOCK] for p in pos])).long().to(DEV)
        with torch.no_grad():
            for name, p, leaf, lat, k in leaves:
                val, _ = E.dequant_grid(lat, k)
                leaf.data.copy_(val)
                p.data.copy_(leaf.data)
        loss = model(xb, labels=xb).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(latents + rest, 1.0)
            for name, p, leaf, lat, k in leaves:
                if p.grad is not None:
                    lat.grad = p.grad
                    p.grad = None
        opt.step()
        sch.step()
        if (step + 1) % 1000 == 0:
            print(f"  {tag} step {step+1}: {loss.item():.4f} | {time.time()-t0:.0f}s",
                  flush=True)
    with torch.no_grad():
        for name, p, leaf, lat, k in leaves:
            val, _ = E.dequant_grid(lat, k)
            p.data.copy_(val)
    nll = 0.0
    with torch.no_grad():
        for i in range(0, 384, 16):
            b = torch.from_numpy(tokens[i * 512:(i + 16) * 512]).view(-1, 512).to(DEV)
            lg = model(b).logits[:, :-1]
            tg = b[:, 1:]
            nll += -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).sum().item()
    fin = nll / (384 * 511)
    tensors = []
    for name, p, leaf, lat, k in leaves:
        D, S, m_true = quant_rtn_odd(lat.detach().double().cpu().numpy(), k)
        tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]), group=64,
                            scale_mode="fp16", k=k, digits=D,
                            scales=S.astype(np.float32), perm=None))
    path = os.path.join(RES, f"eq13_{tag}.k9")
    k9.write_k9(path, tensors)
    total = os.path.getsize(path) + sum(p.numel() * 2 for p in rest)
    print(f"  {tag} FINAL loss {fin:.4f} | artifact {total/1e6:.1f} MB", flush=True)
    return dict(arm=tag, loss=round(fin, 4), artifact_MB=round(total / 1e6, 2))


def main():
    tok = AutoTokenizer.from_pretrained(E.MODEL_ID)
    valid = E.tokens_for(tok, E.load_docs("TinyStories-valid.txt"),
                         os.path.join(L.DATA, "valid_tokens_33m.npy"))
    train = E.tokens_for(tok, E.load_docs("TinyStories-train-slice.txt"),
                         os.path.join(L.DATA, "train_tokens_33m.npy"))
    tokens = valid[: 384 * BLOCK].astype(np.int64)
    ref_cold = 1.5227         # eq12 cold recipe @2000@1e-4
    ctrl = 1.2817             # eq12/eq11 converged CONTROL

    arms = [("s4000_lr1e4", 4000, 1e-4), ("s8000_lr1e4", 8000, 1e-4),
            ("s2000_lr2e4", 2000, 2e-4)]
    runs = []
    for tag, steps, lr in arms:
        print(f"[eq13] === {tag} ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(E.MODEL_ID).to(DEV).eval()
        tied = model.get_output_embeddings().weight.data_ptr() == \
            model.get_input_embeddings().weight.data_ptr()
        runs.append(dict(**run_arm(tag, steps, lr, tied, model, tok, valid, train, tokens)))
        del model
        torch.cuda.empty_cache()

    g4k = runs[0]["loss"] - ctrl
    g8k = runs[1]["loss"] - ctrl
    g2k = runs[2]["loss"] - ctrl
    v31 = dict(prediction="P31", gap_4000=round(g4k, 4), gap_8000=round(g8k, 4),
               verdict="PASS" if (g4k <= 0.15 and g8k <= 0.10) else "FAIL")
    v32 = dict(prediction="P32", gap_lr2e4=round(g2k, 4), gap_lr1e4_ref=round(runs[0]["loss"] - ctrl if False else 0.241, 4),
               verdict="PASS" if g2k <= 0.241 else "FAIL")
    v33 = dict(prediction="P33", MB=[r["artifact_MB"] for r in runs], in_class=[43, 45],
               verdict="PASS" if all(43 <= r["artifact_MB"] <= 45 for r in runs) else "FAIL")
    for v in (v31, v32, v33):
        print(f"[eq13] {v['prediction']} ({v['verdict']}): " +
              json.dumps({kk: v[kk] for kk in v if kk not in ("prediction", "verdict")}))
    with open(os.path.join(RES, "eq13_results.json"), "w") as f:
        json.dump([v31, v32, v33, dict(ctrl=ctrl, ref_cold=ref_cold, runs=runs)], f, indent=2)
    print("[eq13] wrote results/eq13_results.json")


if __name__ == "__main__":
    main()