#!/usr/bin/env python3
"""exp20 — K9 k-palette: mixed-precision grid allocation, to move up the
deployed frontier.

exp19 put K9-full at 641.9 MB / +0.657 code ppl — the smallest point on the
curve, but ~3.8x worse quality-per-byte than NF4. This experiment spends bytes
where they matter: assign the grid size k per tensor (9 / 15 for the body,
9 / 99 for the tied embedding) instead of one global k, and measure where K9
lands against q4_k_m (1117.3 MB, +0.170) and NF4 (999.5 MB, +0.270).

Allocation. For every (tensor, k) we compute the Hessian-diagonal-weighted
quantization error (the standard GPTQ-style sensitivity proxy) and the exact
coded byte cost (digit entropy + entropy-coded scales + table). Starting from
all-k9, we greedily promote the tensor with the largest error reduction per
byte until a total-byte budget is met. This is cheap because GPTQ for a given
(layer, k) is independent of other layers' k, so cached tensors compose into
any palette.

Evaluation. Body tensors are GPTQ'd (our best recipe, cached per (layer, k));
embeddings RTN. Same 8-window harness as exp17-19, paired vs fp32, so the
result is directly comparable to the external baselines.

Run:  python3 exp20_k_palette.py             # ~15 min on a 5060 Ti
      python3 exp20_k_palette.py --smoke     # fast: base + one promote
"""
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))

import numpy as np
import torch

import eval_harness as eh
import exp4b_group_scales as e4b
import exp18_qwen_gptq_noise as e18
from rans import entropy_bits
from exp16_codec_accounting import scale_codec_bytes

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
MODEL_ID = e18.MODEL_ID
BLOCK = 1024
GROUP = 64
N_WINDOWS = 2 if "--smoke" in sys.argv else 8
N_CAL = 8 if "--smoke" in sys.argv else 128
DAMP = 0.05
ACT = True
def _ks(name, default):
    v = os.environ.get(name)
    return [int(x) for x in v.split(",")] if v else default


K_BODY = _ks("K9_KBODY", [9, 15])
K_EMBED = _ks("K9_KEMBED", [9, 99])
BUDGETS_MB = _ks("K9_BUDGETS", [700, 850, 1000, 1150])
EMB = "__embed__"


def tensor_cost(idx, scales, k, numel):
    """Coded bytes: digit entropy + entropy-coded scales + k*2 table bytes."""
    flat = idx.reshape(-1)
    return (entropy_bits(flat, k) * numel / 8.0
            + scale_codec_bytes(scales) + 2 * k)


def greedy(base_assign, embed_k, cost, err, budget_bytes):
    """Multi-level greedy knapsack: repeatedly take the single best marginal
    upgrade (largest error reduction per coded byte) until the budget is hit.
    Works for any number of body/embed levels in K_BODY / K_EMBED."""
    assign = dict(base_assign)
    ek = embed_k
    total = sum(cost[(n, assign[n])] for n in assign) + cost[(EMB, ek)]

    def upgrades():
        ups = []
        for n in assign:
            cur = assign[n]
            for k in K_BODY:
                if k > cur:
                    db = cost[(n, k)] - cost[(n, cur)]
                    de = err[(n, cur)] - err[(n, k)]
                    if db > 0 and de > 0:
                        ups.append((de / db, "body", n, k, db))
        for k in K_EMBED:
            if k > ek:
                db = cost[(EMB, k)] - cost[(EMB, ek)]
                de = err[(EMB, ek)] - err[(EMB, k)]
                if db > 0 and de > 0:
                    ups.append((de / db, "embed", EMB, k, db))
        return ups

    while True:
        ups = upgrades()
        ups.sort(reverse=True)
        for _, kind, name, k, db in ups:
            if total + db <= budget_bytes:
                total += db
                if kind == "body":
                    assign[name] = k
                else:
                    ek = k
                break
        else:
            break
    return assign, ek, total


def main():
    t_all = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = e18.load_model(dev)
    body = e18.body_layers(model)
    embed = model.model.embed_tokens
    n_body = sum(m.weight.numel() for _, m in body)
    n_embed = embed.weight.numel()
    ptrs, n_total = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in ptrs:
            ptrs.add(p.data_ptr())
            n_total += p.numel()
    n_rest = n_total - n_body - n_embed
    rest_mb = 4.0 * n_rest / 1e6
    print(f"body {len(body)} Linear {n_body:,} | embed {n_embed:,} | rest {n_rest:,}")

    cal = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean", None, "train",
                         N_CAL, BLOCK, gap=0, label="calib")
    cal_batches = [cal[i:i + 8] for i in range(0, N_CAL, 8)]
    wiki = eh.get_windows(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                          "test", N_WINDOWS, BLOCK, label="wiki")
    code = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean-valid", None,
                          "train", N_WINDOWS, BLOCK, label="code(python valid)")

    # fp32 reference FIRST, then drop the fp32 weights: every palette overwrites
    # every tensor, so no restore is needed and no fp32 master copy is kept
    # (the box is shared with a ~30 GB dotnet process; memory is the bottleneck).
    print("fp32 reference...")
    pw0 = eh.ppl_per_window(model, wiki)
    pc0 = eh.ppl_per_window(model, code)
    print(f"  fp32 code {np.mean(pc0):.3f} | wiki {np.mean(pw0):.3f}")

    t0 = time.time()
    print("collecting Hessians (one fp32 pass)...")
    hess = e18.collect_all_hessians(model, body, cal_batches)
    print(f"  done {time.time()-t0:.0f}s")

    cost, err, cidx, cm = {}, {}, {}, {}

    def rebuild(key):
        """Reconstruct dequantized weights from cached digits + group scales
        (the codec representation: uint8 digits, not float weights). With
        act-order the digits/scales live in the permuted frame; undo it here."""
        idx, (sc, shape, perm) = cidx[key], cm[key]
        r, c = shape
        k = key[1]
        mm = torch.from_numpy(sc).to(dev)
        step = 2.0 * mm / (k - 1)
        it = torch.from_numpy(idx).to(dev).view(r, c // GROUP, GROUP).float()
        q = (-mm.unsqueeze(-1) + it * step.unsqueeze(-1)).reshape(r, c)
        if perm is not None:
            q = q[:, torch.argsort(perm.to(q.device))]
        return q

    # embeddings first (cheap; a self-check here catches storage bugs in
    # seconds rather than after the ~16 min body pass)
    W = embed.weight.data
    eshape = tuple(W.shape)
    for k in K_EMBED:
        deq, idx = e4b.q_uniform_group(W, k)
        sc = e18.group_scales(W).cpu().numpy()
        cidx[(EMB, k)] = idx.to(torch.uint8).cpu().numpy().reshape(-1)
        cm[(EMB, k)] = (sc, eshape, None)
        cost[(EMB, k)] = tensor_cost(cidx[(EMB, k)], sc, k, W.numel())
        err[(EMB, k)] = float(((W - deq) ** 2).sum())
        assert torch.allclose(rebuild((EMB, k)), deq, atol=1e-6), \
            f"embed rebuild mismatch k={k}"
        del deq

    t0 = time.time()
    for i, (n, m) in enumerate(body):
        W = m.weight.data
        shape = tuple(W.shape)
        d = torch.diagonal(hess[n]).to(dev)
        perm = (torch.argsort(torch.diagonal(hess[n]), descending=True)
                if ACT else None)
        for k in K_BODY:
            Q, IDX = e18.gptq_quantize(W, hess[n], k, g=GROUP, damp=DAMP,
                                       act_order=ACT, compensate=True)
            if perm is not None:
                # act-order: gptq_quantize builds groups in the PERMUTED column
                # order, so store digits + scales in that frame and undo the
                # permutation at rebuild time (scales from the original order
                # would disagree with the digits).
                p = perm.to(dev)
                sc = e18.group_scales(W[:, p]).cpu().numpy()
                cidx[(n, k)] = IDX[:, p].to(torch.uint8).cpu().numpy().reshape(-1)
            else:
                sc = e18.group_scales(W).cpu().numpy()
                cidx[(n, k)] = IDX.to(torch.uint8).cpu().numpy().reshape(-1)
            cm[(n, k)] = (sc, shape, perm)
            cost[(n, k)] = tensor_cost(cidx[(n, k)], sc, k, W.numel())
            err[(n, k)] = float(((W - Q) ** 2 * d.unsqueeze(0)).sum())
            if i == 0:
                assert torch.allclose(rebuild((n, k)), Q, atol=1e-5), \
                    f"rebuild mismatch at {n} k={k}"
            del Q
        del hess[n]          # free each Hessian as soon as it is consumed
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(body)} tensors, {time.time()-t0:.0f}s")
    print(f"body GPTQ cached {time.time()-t0:.0f}s")

    base_assign = {n: 9 for n, _ in body}
    kmax = K_BODY[-1]
    kmid = K_BODY[1] if len(K_BODY) > 1 else K_BODY[-1]
    palettes = [("all-k9 (base)", dict(base_assign), 9, None),
                ("all-k9 + embed99", dict(base_assign), 99, None)]
    if "--smoke" not in sys.argv:
        for b in BUDGETS_MB:
            a, ek, tot = greedy(base_assign, 9, cost, err, b * 1e6)
            palettes.append((f"greedy <= {b} MB", a, ek, None))
        palettes.append((f"all-k{kmax} + embed99",
                         {n: kmax for n, _ in body}, 99, None))
        palettes.append((f"attn{kmax}/mlp{kmid} + embed99",
                         {n: (kmax if ".self_attn." in n else kmid)
                          for n, _ in body}, 99, None))

    def apply_palette(assign, ek):
        for n, m in body:
            m.weight.data.copy_(rebuild((n, assign[n])))
        embed.weight.data.copy_(rebuild((EMB, ek)))

    rows, store = [], {}
    for name, assign, ek, _ in palettes:
        apply_palette(assign, ek)
        pw = eh.ppl_per_window(model, wiki)
        pc = eh.ppl_per_window(model, code)
        store[name] = (pw, pc)
        n_hi = sum(1 for v in assign.values() if v == kmax)
        mb = (sum(cost[(n, assign[n])] for n in assign) + cost[(EMB, ek)]) / 1e6 + rest_mb
        mw, sw = eh.summarize(pw)
        mc, sc_ = eh.summarize(pc)
        rows.append(dict(palette=name, MB=mb, n_kmax=n_hi, embed_k=ek,
                         wiki=mw, wiki_se=sw, code=mc, code_se=sc_))
        print(f"{name:<22} {mb:7.1f} MB (k{kmax} x{n_hi:3d}, emb{ek:3d})"
              f" | wiki {mw:7.3f} | code {mc:7.3f} ± {sc_:.3f}")

    for r in rows:
        r["d_code"], r["d_code_se"] = eh.paired_delta(store[r["palette"]][1], pc0)
        r["d_wiki"], r["d_wiki_se"] = eh.paired_delta(store[r["palette"]][0], pw0)
    print(f"{'fp32':<22} {4.0*n_total/1e6:7.1f} MB | wiki {np.mean(pw0):7.3f}"
          f" | code {np.mean(pc0):7.3f}")

    cols = list(rows[0].keys())
    with open(RESULTS / "exp20_k_palette.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# exp20 — K9 k-palette (mixed-precision grid allocation)",
        "",
        "Qwen2.5-Coder-1.5B-Instruct, same 8-window harness as exp17-19. Body",
        "tensors GPTQ'd (damp 0.05, act-order), embeddings RTN. Allocation is",
        "greedy on Hessian-diagonal-weighted quantization error per coded byte.",
        "",
        "External reference points (exp19, same windows): **q4_k_m 1117.3 MB,",
        "code +0.170**; NF4 999.5 MB, +0.270; q2_k 752.9 MB, +1.426; fp32",
        f"{np.mean(pc0):.3f} code.",
        "",
        "| palette | MB | k=max tensors | embed k | code ppl | Δcode vs fp32 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['palette']} | {r['MB']:.1f} | {r['n_kmax']} | {r['embed_k']}"
                     f" | {r['code']:.3f} ± {r['code_se']:.3f}"
                     f" | {r['d_code']:+.3f} ± {r['d_code_se']:.3f} |")
    lines += [
        "",
        "## Reading",
        "",
        "- Compare each palette against **q4_k_m (1117.3 MB, +0.170)**: any row",
        "  that is both smaller and lower Δcode dominates the most-used quant.",
        "- `all-k9 (base)` should reproduce exp17's uniform body9+embed9 point;",
        "  the greedy rows show whether allocation buys quality efficiently.",
        "",
        f"wall {time.time()-t_all:.0f}s", "",
    ]
    (RESULTS / "exp20_k_palette.md").write_text("\n".join(lines))
    print("wrote results/exp20_k_palette.{csv,md}")


if __name__ == "__main__":
    main()
