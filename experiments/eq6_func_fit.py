#!/usr/bin/env python3
"""eq6 — RQ3a: the variable table is trainable.

The K9 grid scales (per-(row,group) m) are the first "variables" of the
representation; the dequant w = m*(d-H)/H is LINEAR in m, so their gradient is
exact (no STE needed on the scale path). Fit them with Adam against
train-split next-token loss (digits + fp16 rests frozen), then:

  * re-encode the artifact with the FITTED scales (fp16 scale mode) — bytes
    must be unchanged (P9's deployment claim);
  * paired eval on the same 384 valid windows before/after (P9: recovers >=
    50% of the RTN->anchor gap, per variant);
  * story sample after fit.

Runs on BOTH eq5 variant A (k9-g64 low-bit, the +57% failure case — the best
test of functional compensation) and variant C (k63-g64, the +1.1% case —
does the fit recover even the last bit?).
"""
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import eq_lib as L  # noqa: E402

B9 = "/mnt/matrix/Work/base9-quantization/experiments"
sys.path.insert(0, B9)
import k9  # noqa: E402

DEV = "cuda"
RES = os.path.join(L.ROOT, "results")
GROUP = 64
UNIQUE_PARAMS = 260032
STEPS = 400
LR = 3e-3
BATCH, BLOCK = 8, 512


def train_tokens(n_tokens):
    cache = os.path.join(L.DATA, "train_tokens.npy")
    if os.path.exists(cache):
        tok = np.load(cache)
    else:
        sp = L.get_sp()
        txt = open(os.path.join(L.DATA, "TinyStories-train-slice.txt"),
                   encoding="utf-8").read()
        docs = [d.strip() for d in re.split(r"<END_OF_DOC_TAG>|_{8,}", txt) if d.strip()]
        ids = []
        for d in docs:
            ids += [sp.bos_id()] + sp.encode(d)
        tok = np.array(ids, dtype=np.int32)
        np.save(cache, tok)
    return tok[:n_tokens]


def quant_rtn_odd(W, k, group=GROUP, dtype=np.uint8):
    n, m = W.shape
    H = (k - 1) // 2
    g = min(group, m)
    pad = (-m) % g
    Wp = np.pad(W, ((0, 0), (0, pad)))
    ng = Wp.shape[1] // g
    S = np.maximum(np.abs(Wp.reshape(n, ng, g)).max(axis=2), 1e-12)
    D = np.rint(H * Wp / np.repeat(S, g, axis=1)).clip(-H, H).astype(dtype) + H
    return D.reshape(n, ng * g), S, m


def main():
    sd0, ma, _ = L.load_state_dict()
    uniq = L.unique_parameters(sd0)
    w2d = {k: v for k, v in uniq.items() if v.dim() == 2}
    EMB = "tok_embeddings.weight" if "tok_embeddings.weight" in uniq else "output.weight"
    rest1d = {n: v for n, v in uniq.items() if v.dim() == 1}
    tokens = L.valid_tokens(384 * ma["max_seq_len"])
    fit_tok = train_tokens(2_600_000)

    fp32 = L.eval_loss(L.to_device(sd0, DEV), ma, tokens, DEV, 384)
    print(f"[eq6] fp32 anchor {fp32[0]:.4f}")

    variants = {"A_k9emb99": ({n: 9 for n in w2d if n != EMB}, {EMB: 99}),
                "B_k27emb99": ({n: 27 for n in w2d if n != EMB}, {EMB: 99}),
                "C_k63emb99": ({n: 63 for n in w2d if n != EMB}, {EMB: 99})}
    out = []
    for tag, (body_spec, emb_spec) in variants.items():
        digits_store = {}
        for name, W in w2d.items():
            k = (emb_spec if name == EMB else body_spec).get(name)
            if k is None:
                continue
            D, S, m_true = quant_rtn_odd(W.double().numpy(), k)
            digits_store[name] = dict(D=torch.from_numpy(D.astype(np.int64)).to(DEV),
                                      m0=torch.from_numpy(S.astype(np.float64)).to(DEV),
                                      k=k, m_true=m_true, shape=W.shape)
        H_pad = {name: (st["k"] - 1) // 2 for name, st in digits_store.items()}

        def build_sd(m_params):
            sd = {}
            for name, st in digits_store.items():
                m = torch.exp(m_params[name]).double()          # (rows, ngroups)
                Srep = torch.repeat_interleave(m, GROUP, dim=1)
                w = Srep * (st["D"].double() - H_pad[name]) / H_pad[name]
                sd[name] = w[:, : st["m_true"]].float().to(DEV)
            for n_, v in rest1d.items():
                sd[n_] = v.half().float().to(DEV)
            emb = sd.pop(EMB)
            sd["output.weight"] = sd["tok_embeddings.weight"] = emb
            return sd

        m_params = {name: torch.log(st["m0"]).float().to(DEV).requires_grad_(True)
                    for name, st in digits_store.items()}
        pre = L.eval_loss(build_sd(m_params), ma, tokens, DEV, 384)
        opt = torch.optim.Adam(m_params.values(), lr=LR)
        for t in range(STEPS):
            starts = torch.randint(0, len(fit_tok) - BLOCK - 1, (BATCH,))
            b = torch.stack([torch.from_numpy(fit_tok[s:s + BLOCK].astype(np.int64))
                             for s in starts]).to(DEV)
            sd = build_sd(m_params)
            lg = L.forward(sd, b, ma)
            nll = -torch.log_softmax(lg[:, :-1], -1).gather(-1, b[:, 1:, None]).mean()
            opt.zero_grad(); nll.backward(); opt.step()
            if t % 100 == 0 or t == STEPS - 1:
                print(f"[eq6] {tag} step {t:4d} fit-loss {nll.item():.4f}")
        post = L.eval_loss(build_sd(m_params), ma, tokens, DEV, 384)

        # re-encode artifact with FITTED scales (fp16 scale mode) — bytes claim
        fitted_path = os.path.join(RES, f"eq6_{tag}_fitted.k9")
        tensors = []
        for name, st in digits_store.items():
            m_fit = torch.exp(m_params[name].detach()).double().cpu().numpy()
            D_cpu = st["D"].cpu().numpy().astype(np.uint8)
            tensors.append(dict(name=name, shape=(D_cpu.shape[0], D_cpu.shape[1]),
                                group=GROUP, scale_mode="fp16", k=st["k"],
                                digits=D_cpu, scales=m_fit.astype(np.float32), perm=None))
        k9.write_k9(fitted_path, tensors)
        nbytes = os.path.getsize(fitted_path) + sum(v.numel() * 2 for v in rest1d.values())
        ev_file = L.eval_loss(L.to_device(reload(fitted_path, digits_store, rest1d, EMB),
                                          DEV), ma, tokens, DEV, 384)
        story = L.sample_story(build_sd(m_params), ma, DEV, n_new=160, seed=7)
        gap = pre[0] - fp32[0]
        rec = (pre[0] - post[0]) / gap
        res = dict(variant=tag, bytes=int(nbytes), pre_loss=pre[0], post_loss=post[0],
                   post_loss_artifact=ev_file[0], gap_recovery=rec,
                   target_half=fp32[0] + 0.5 * gap, story=story[:80])
        out.append(res)
        print(f"[eq6] {tag}: {pre[0]:.4f} -> {post[0]:.4f} (anchor {fp32[0]:.4f}) "
              f"gap-recovery {rec:.1%} | fitted-file {nbytes:,} B loss {ev_file[0]:.4f}")
        print(f"      story: {story[:96]!r}")

    v9 = dict(prediction="P9", results=[dict(variant=r["variant"], recovery=r["gap_recovery"])
                                        for r in out],
              verdict="PASS" if all(r["gap_recovery"] >= 0.5 for r in out) else "FAIL")
    print(f"[eq6] P9 ({v9['verdict']}): " +
          ", ".join(f"{r['variant']} {r['gap_recovery']:.1%}" for r in out))
    with open(os.path.join(RES, "eq6_results.json"), "w") as f:
        json.dump([v9, dict(results=out)], f, indent=2)
    print("[eq6] wrote results/eq6_results.json + eq6_*_fitted.k9")


def reload(path, digits_store, rest1d, EMB):
    recs = k9.read_k9(path)
    sd = {}
    widths = {name: st["m_true"] for name, st in digits_store.items()}
    for name, rec in recs.items():
        q = k9.decode_tensor(rec, GROUP)
        sd[name] = q[:, : widths[name]] if q.dim() == 2 else q
    for name, v in rest1d.items():
        sd[name] = v.half().float()
    sd["output.weight"] = sd["tok_embeddings.weight"] = sd[EMB]
    return sd


if __name__ == "__main__":
    main()