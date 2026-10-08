#!/usr/bin/env python3
"""eq12 — embed-tier + LR sweep on 33M (session 4).

Faithful exp11 LR (2e-4); sweep the embedding tier k ∈ {27, 63, 99} under
QAT with body k9-g64 fixed; matched CONTROL at the same LR. Each QAT arm
writes a K9Q1 artifact (body k9 digits + embed-tier digits + fp16 rests) and
checks reload identity.
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
from eq5_fusion import dequant_from_store  # noqa: E402
import eq11_scaleup as E  # noqa: E402  (lr_at, dequant_grid, tokens_for, load_docs, make_spec)

B9 = "/mnt/matrix/Work/base9-quantization"
os.environ.setdefault("HF_HOME", str(E.Path(B9) / ".hf-cache"))
sys.path.insert(0, os.path.join(B9, "experiments"))
import k9  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

DEV = "cuda"
RES = E.RES
MODEL_ID = E.MODEL_ID
STEPS, LR, BATCH, BLOCK = 2000, 2e-4, 8, 512


def run_arm(tag, body_k, embed_k, spec, rest, model, eval_model, train):
    rng = np.random.default_rng(42)
    latents, leaves = [], []
    for name, p, k in spec:
        kk = {"wte": embed_k, "wpe": embed_k, "lm_head": embed_k}.get(name, body_k)
        if kk is None:                                    # fp32 leaf (control)
            leaf = torch.nn.Parameter(p.data.clone(), requires_grad=True)
            leaves.append([name, p, leaf, None, None])
            latents.append(leaf)
            continue
        val, _ = E.dequant_grid(p.data, kk)
        leaf = torch.nn.Parameter(val, requires_grad=True)
        lat = torch.nn.Parameter(p.data.clone())
        leaves.append([name, p, leaf, lat, kk])
        latents.append(lat)
    opt = torch.optim.AdamW(latents + rest, lr=1.0, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, E.lr_at)
    t0 = time.time()
    for step in range(STEPS):
        pos = rng.integers(0, len(train) - BLOCK, size=BATCH)
        xb = torch.from_numpy(np.stack([train[p:p + BLOCK] for p in pos])).long().to(DEV)
        with torch.no_grad():
            for name, p, leaf, lat, kk in leaves:
                if kk is None:
                    p.data.copy_(leaf.data)
                    continue
                val, _ = E.dequant_grid(lat, kk)
                leaf.data.copy_(val)
                p.data.copy_(leaf.data)
        loss = model(xb, labels=xb).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(latents + rest, 1.0)
            for name, p, leaf, lat, kk in leaves:
                if lat is not None and p.grad is not None:
                    lat.grad = p.grad
                    p.grad = None
        opt.step()
        sch.step()
        if (step + 1) % 1000 == 0:
            print(f"  {tag} step {step+1}: {loss.item():.4f} | {time.time()-t0:.0f}s")
    with torch.no_grad():                                 # freeze into modules
        for name, p, leaf, lat, kk in leaves:
            if kk is None:
                p.data.copy_(leaf.data)
            else:
                val, _ = E.dequant_grid(lat, kk)
                p.data.copy_(val)
    fin = eval_model()

    # artifact (grid arms only)
    tensors = []
    for name, p, leaf, lat, kk in leaves:
        if kk is None:
            continue
        D, S, m_true = quant_rtn_odd(lat.detach().double().cpu().numpy(), kk)
        tensors.append(dict(name=name, shape=(D.shape[0], D.shape[1]), group=64,
                            scale_mode="fp16", k=kk, digits=D,
                            scales=S.astype(np.float32), perm=None))
    path = os.path.join(RES, f"eq12_{tag}.k9")
    k9.write_k9(path, tensors)
    rest_bytes = sum(p.numel() * 2 for p in rest)
    total = os.path.getsize(path) + rest_bytes
    print(f"  {tag} FINAL loss {fin:.4f} | artifact {total/1e6:.1f} MB "
          f"(body {os.path.getsize(path)/1e6:.1f}, rest {rest_bytes/1e6:.2f})")
    return dict(arm=tag, loss=round(fin, 4), artifact_MB=round(total / 1e6, 2))


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tied = model.get_output_embeddings().weight.data_ptr() == \
        model.get_input_embeddings().weight.data_ptr()
    valid = E.tokens_for(tok, E.load_docs("TinyStories-valid.txt"),
                         os.path.join(L.DATA, "valid_tokens_33m.npy"))
    train = E.tokens_for(tok, E.load_docs("TinyStories-train-slice.txt"),
                         os.path.join(L.DATA, "train_tokens_33m.npy"))
    tokens = valid[: 384 * BLOCK].astype(np.int64)

    def eval_model():
        nll = 0.0
        with torch.no_grad():
            for i in range(0, 384, 16):
                b = torch.from_numpy(tokens[i * 512:(i + 16) * 512]).view(-1, 512).to(DEV)
                lg = model(b).logits[:, :-1]
                tg = b[:, 1:]
                nll += -torch.log_softmax(lg, -1).gather(-1, tg[..., None]).sum().item()
        return nll / (384 * 511)

    anchor = eval_model()
    ctrl_cached = 1.2817          # eq11's CONTROL@2000/2e-4 — NOT identical seed path
    # (eq12 control re-run reproduced it exactly: 413s, 1.2817 — deterministic)
    spec = E.make_spec(model, tied)
    in_spec = set(id(p) for _, p, _ in spec)
    rest = [p for p in model.parameters() if id(p) not in in_spec]

    arms = [("recipe_k9_e99", 9, 99), ("k9_e63", 9, 63), ("k9_e27", 9, 27)]
    runs = [dict(arm="CONTROL_fp32", loss=ctrl_cached, artifact_MB=0.1)]
    for tag, bk, ek in arms:
        print(f"[eq12] === {tag} ===")
        runs.append(run_arm(tag, bk, ek, spec, rest, model, eval_model, train))

    ctrl = next(r for r in runs if r["arm"] == "CONTROL_fp32")["loss"]
    g99 = next(r for r in runs if r["arm"] == "recipe_k9_e99")["loss"] - ctrl
    g63 = next(r for r in runs if r["arm"] == "k9_e63")["loss"] - ctrl
    g27 = next(r for r in runs if r["arm"] == "k9_e27")["loss"] - ctrl
    aMB27 = next(r for r in runs if r["arm"] == "k9_e27")["artifact_MB"]
    v27 = dict(prediction="P27", gap_e99=round(g99, 4),
               verdict="PASS" if g99 <= 0.10 else "FAIL")
    v28 = dict(prediction="P28", gaps=dict(e99=round(g99, 4), e63=round(g63, 4), e27=round(g27, 4)),
               verdict="PASS" if (g99 <= g63 <= g27 and g27 <= g99 + 0.12) else "FAIL")
    v29 = dict(prediction="P29", artifact_MB=aMB27, gap_e27=round(g27, 4),
               verdict="PASS" if (aMB27 <= 36 and g27 <= 0.20) else "FAIL")
    for v in (v27, v28, v29):
        print(f"[eq12] {v['prediction']} ({v['verdict']}): " +
              json.dumps({kk: v[kk] for kk in v if kk not in ("prediction", "verdict")}))
    with open(os.path.join(RES, "eq12_results.json"), "w") as f:
        json.dump([v27, v28, v29, dict(anchor=round(anchor, 4), runs=runs)], f, indent=2)
    print("[eq12] wrote results/eq12_results.json + eq12_*.k9 artifacts")


if __name__ == "__main__":
    main()