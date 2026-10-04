#!/usr/bin/env python3
"""exp9 — the 127-level anomaly (open flag from exp7): is the ppl inversion
(127-level 40.98 > 99-level 40.57 > 63-level 40.65, despite monotone MSE)
real or eval noise?

Two tests:
  A. INDEPENDENT EVAL WINDOWS — quantize 63/99/127 (g64 RTN) and evaluate on
     windows B (blocks 300–600) and C (600–900), disjoint from exp7's window
     A. If the inversion reproduces in both new windows, it's real.
  B. SCOPE ABLATION on window A — quantize ONLY attention matrices vs ONLY
     mlp matrices, for 99 vs 127: where does 127 lose to 99?

Pre-registered: P12 — the inversion (127 ppl > max(99, 63-level ppl) within
its window) reproduces in BOTH independent windows, or it is noise.
P13 — the inversion, if real, is localized (attn vs mlp differ).

Out: results/exp9_anomaly_check.{csv,md}
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import exp4_real_model_ptq as e4
import exp4b_group_scales as e4b
import exp6_gptq_grids as e6

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

GRIDS = [63, 99, 127]
N_WIN = 3          # windows A/B/C of 300 blocks each


def get_many_blocks(tok, dev, n):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    ids = []
    for row in ds:
        ids.extend(tok(row["text"]).input_ids)
        ids.append(tok.eos_token_id)
        if len(ids) >= n * e4.BLOCK:
            break
    arr = np.asarray(ids[: n * e4.BLOCK], dtype=np.int64).reshape(n, e4.BLOCK)
    return torch.from_numpy(arr).to(dev)


def quant_all(model, layers, k):
    for name, m in layers:
        tr = type(m).__name__ == "Conv1D"
        W = m.weight.data.t() if tr else m.weight.data
        deq, idx = e4b.q_uniform_group(W, k)
        m.weight.data = (deq.t() if tr else deq).contiguous()


def main():
    dev = e4.device_setup()
    model, tok = e4.load_model(dev)
    layers_all = e4.quantized_layers(model)
    attn_layers = [(n, m) for n, m in layers_all if ".attn." in n]
    mlp_layers = [(n, m) for n, m in layers_all if ".mlp." in n]
    assert len(attn_layers) + len(mlp_layers) == len(layers_all)
    saved = {name: m.weight.data.detach().clone() for name, m in layers_all}

    big = get_many_blocks(tok, dev, N_WIN * e4.EVAL_BLOCKS)
    windows = {"A": big[: e4.EVAL_BLOCKS], "B": big[e4.EVAL_BLOCKS: 2 * e4.EVAL_BLOCKS],
               "C": big[2 * e4.EVAL_BLOCKS:]}
    ppl0 = e4.perplexity(model, windows["A"])
    print(f"fp32 baseline window A: {ppl0:.3f}")

    def restore():
        for name, m in layers_all:
            m.weight.data.copy_(saved[name])

    rows = []
    t0 = time.time()
    # Test A: full-quant ppl on three disjoint windows
    exp7_ref = {63: 40.646, 99: 40.565, 127: 40.978}  # exp7's window-A RTN values
    for k in GRIDS:
        quant_all(model, layers_all, k)
        for wname, w in windows.items():
            ppl = e4.perplexity(model, w)
            if wname == "A":
                assert abs(ppl - exp7_ref[k]) < 0.05, \
                    f"window-A regression vs exp7 broken: {ppl:.3f} vs {exp7_ref[k]}"
            rows.append(dict(test="A-windows", grid=k, scope="all",
                             window=wname, ppl=ppl, mse=None, note=""))
            print(f"A  {k}-level all-layers window {wname}: ppl {ppl:.3f}")
        restore()

    # Test B: scope ablation on window A (MSE captured in the same pass)
    for k in (99, 127):
        for scope_name, scope in (("attn-only", attn_layers), ("mlp-only", mlp_layers)):
            serr = wsum2 = wsum = 0.0
            n = 0
            for name, m in scope:
                tr = type(m).__name__ == "Conv1D"
                W = m.weight.data.t() if tr else m.weight.data
                deq, _ = e4b.q_uniform_group(W, k)
                serr += ((W - deq) ** 2).sum().item()
                wsum2 += (W * W).sum().item()
                wsum += W.sum().item()
                n += W.numel()
                m.weight.data = (deq.t() if tr else deq).contiguous()
            ppl = e4.perplexity(model, windows["A"])
            mse = serr / max(wsum2 - n * (wsum / n) ** 2, 1e-30)
            rows.append(dict(test="B-scope", grid=k, scope=scope_name,
                             window="A", ppl=ppl, mse=mse, note=""))
            print(f"B  {k}-level {scope_name}: ppl {ppl:.3f} mse {mse:.5f}")
            restore()

    with open(RESULTS / "exp9_anomaly_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["test", "grid", "scope", "window",
                                           "ppl", "mse", "note"])
        w.writeheader()
        w.writerows(rows)

    # P12: inversion per window: 127 > max(99, 63)?
    inv = {}
    for wname in ("A", "B", "C"):
        get = lambda g: next(r["ppl"] for r in rows
                             if r["test"] == "A-windows" and r["grid"] == g and r["window"] == wname)
        inv[wname] = get(127) > max(get(99), get(63))
    a_inv = {w: f"{w}:{'inverted' if v else 'ordered'}" for w, v in inv.items()}
    p12 = inv["B"] and inv["C"] and inv["A"]

    lines = [
        "# exp9 — the 127-level anomaly: independent windows + scope ablation",
        "",
        f"fp32 baseline window A {ppl0:.3f}; window A values reproduce exp7 RTN.",
        "",
        "## Test A — three disjoint eval windows (g=64 RTN, all layers)",
        "",
        "| grid | window | ppl |", "|---|---|---|",
    ] + [f"| {r['grid']} | {r['window']} | {r['ppl']:.3f} |"
         for r in rows if r["test"] == "A-windows"]
    lines += [
        "",
        "## Test B — scope ablation on window A",
        "",
        "| grid | scope | ppl | rel MSE (scoped) |", "|---|---|---|---|",
    ] + [f"| {r['grid']} | {r['scope']} | {r['ppl']:.3f} | {r['mse']:.5f} |"
         for r in rows if r["test"] == "B-scope"]
    lines += [
        "",
        "## Verdicts",
        "",
        f"- P12 (inversion reproduces in A, B and C): **{'PASS' if p12 else 'FAIL'}** — {a_inv}",
        f"- P13 (localization): see Test B — if 127 loses to 99 in one scope",
        "  only, the anomaly is structural; if both, it is global.",
        "",
        f"wall {time.time()-t0:.0f}s", "",
    ]
    (RESULTS / "exp9_anomaly_check.md").write_text("\n".join(lines))
    print("wrote results/exp9_anomaly_check.md")


if __name__ == "__main__":
    main()