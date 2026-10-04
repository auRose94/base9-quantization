#!/usr/bin/env python3
"""exp26 — 7B + GPTQ: does compensation tip K9 past q4_k_m at scale?

exp24/25 left the 7B story as a trade: K9 k=15 (3740 MB, +0.117 code vs bf16) is
20% smaller than GGUF q4_k_m (4683 MB, +0.070) but slightly worse. At 1.5B, GPTQ
with act-order bought -0.45 code ppl (8%) on the same grid. This applies the same
compensation at 7B.

Memory design (16 GB card, 15.3 GB model):
  * GPTQ is applied to every body matrix with in_features <= 4096 — q/k/v/o
    (3584) and gate/up (3584) — which is ~71% of the body parameters and
    includes the attention projections that exp20 identified as the sensitive
    type. Hessians there are <= 3584^2 (51 MB fp32).
  * **down_proj is left RTN**: its in_features is 18944, so its Hessian is
    1.4 GB fp32 (2.9 GB fp64) and does not fit alongside the resident model.
  * embed and lm_head are RTN at k=99 (no Hessian needed).
  * Hessians are collected per transformer block (sequential: upstream layers are
    already quantized when the next block's activations are measured), accumulated
    on CPU, and freed after the block.

Run: python3 exp26_qwen7b_gptq.py      # ~30-45 min, 16 GB GPU
"""
import csv
import math
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

import k9
import eval_harness as eh
import exp18_qwen_gptq_noise as e18
import exp24_qwen7b as e24
from rans import entropy_bits

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
GROUP, BLOCK, N_WINDOWS = 64, 1024, 8
N_CAL, CAL_BATCH = 64, 4
DAMP, ACT = 0.05, True
CONFIGS = [(15, 99)]          # the decisive config: k=15 is what could beat q4_k_m
GPTQ_MAX_IN = 4096          # down_proj (18944) stays RTN

# K9_CONFIGS="9,99" runs a probe config and writes exp26_..._k9.{md,csv} so the
# default k=15 record is untouched.
_cfg = os.environ.get("K9_CONFIGS")
if _cfg:
    CONFIGS = [tuple(int(x) for x in part.split(",")) for part in _cfg.split(":")]
SUFFIX = "" if CONFIGS == [(15, 99)] else "_" + "_".join(f"k{k}" for k, _ in CONFIGS)
CSV_PATH = RESULTS / f"exp26_qwen7b_gptq{SUFFIX}.csv"
MD_PATH = RESULTS / f"exp26_qwen7b_gptq{SUFFIX}.md"


def collect_hessians(model, modules, cal_batches):
    """H = 2 X^T X per module, accumulated on CPU, freed by the caller."""
    accs = {}

    def mk(name):
        def hook(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            xtx = (x.T @ x).cpu()
            accs[name] = xtx if name not in accs else accs[name] + xtx
            del x, xtx
        return hook

    handles = [m.register_forward_pre_hook(mk(n)) for n, m in modules]
    with torch.no_grad():
        for xb in cal_batches:
            model.model(xb)                     # transformer only: no lm_head/logits
    for h in handles:
        h.remove()
    n = sum(xb.shape[0] for xb in cal_batches)
    return {name: 2.0 * accs[name] / n for name, _ in modules}


def gptq_apply(W, H, k, g=GROUP, damp=DAMP, act_order=True, row_chunk=4096):
    """GPTQ in row blocks; updates W in place and returns digits (uint8, CPU),
    scales (fp32, CPU) and the act-order permutation.

    The rank-1 compensation is row-independent, so one Hinv drives every block
    and the device working set stays ~row_chunk x c floats instead of the full
    r x c (272 MB fp32) plus an int64 digit buffer (543 MB) — neither of which
    fits beside a 15.3 GB model.
    """
    r, c = W.shape
    if act_order:
        perm_cpu = torch.argsort(torch.diagonal(H), descending=True)
        Hp = H[perm_cpu][:, perm_cpu]
        perm = perm_cpu.to(W.device)
        inv = torch.argsort(perm)
    else:
        Hp, perm, inv = H, None, None
    Hinv = e18.h_inv(Hp, W.device, damp)
    idx_out = torch.empty((r, c), dtype=torch.uint8, device="cpu")
    sc_out = torch.empty((r, c // g), dtype=torch.float32, device="cpu")
    for a in range(0, r, row_chunk):
        b = min(r, a + row_chunk)
        Wb = W[a:b].float()
        if perm is not None:
            Wb = Wb[:, perm]
        Wg = Wb.reshape(b - a, c // g, g)
        m_g = Wg.abs().amax(dim=2)
        step_g = 2.0 * m_g / (k - 1)
        W1 = Wb.clone()
        Qb = torch.empty_like(Wb)
        idxb = torch.empty((b - a, c), dtype=torch.uint8, device=W.device)
        for i in range(c):
            m_r, st_r = m_g[:, i // g], step_g[:, i // g]
            w = W1[:, i]
            idx = torch.clamp(torch.round((w + m_r) / st_r), 0, k - 1).long()
            deq = -m_r + idx * st_r
            idxb[:, i] = idx.to(torch.uint8)
            Qb[:, i] = deq
            if i + 1 < c:
                err = (w - deq) / Hinv[i, i]
                W1[:, i + 1:] -= err.unsqueeze(1) * Hinv[i, i + 1:].unsqueeze(0)
        idx_out[a:b] = idxb.cpu()
        sc_out[a:b] = m_g.cpu()
        if inv is not None:
            Qb = Qb[:, inv]
        for aa in range(0, b - a, 512):
            bb = min(b - a, aa + 512)
            W[a + aa:a + bb].copy_(Qb[aa:bb].to(W.dtype))
        del Wb, Wg, m_g, step_g, W1, Qb, idxb
    return idx_out, sc_out, (perm.cpu().numpy() if perm is not None else None)


def main():
    t_all = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa").to(dev).eval()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{n_total:,} params | bf16 GPU peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    body = e18.body_layers(model)
    embed, lm_head = model.model.embed_tokens, model.lm_head
    n_gptq = sum(m.weight.numel() for _, m in body if m.weight.shape[1] <= GPTQ_MAX_IN)
    print(f"body {len(body)} | GPTQ-eligible {n_gptq:,} params"
          f" ({100*n_gptq/sum(m.weight.numel() for _, m in body):.0f}% of body);"
          f" down_proj + embed/lm_head are RTN")

    wiki = eh.get_windows(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                          "test", N_WINDOWS, BLOCK, label="wiki")
    code = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean-valid", None,
                          "train", N_WINDOWS, BLOCK, label="code")
    cal = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean", None, "train",
                         N_CAL, BLOCK, gap=0, label="calib")
    cal_batches = [cal[i:i + CAL_BATCH] for i in range(0, N_CAL, CAL_BATCH)]

    print("bf16 reference...")
    pw0 = eh.ppl_per_window(model, wiki)
    pc0 = eh.ppl_per_window(model, code)
    print(f"  code {np.mean(pc0):.3f} | wiki {np.mean(pw0):.3f}")

    saved = {n: m.weight.data.detach().to("cpu").clone() for n, m in body}
    saved_embed = embed.weight.data.detach().to("cpu").clone()
    saved_head = lm_head.weight.data.detach().to("cpu").clone()

    def restore():
        def put(param, master):
            for a in range(0, param.shape[0], 1024):
                b = min(param.shape[0], a + 1024)
                param[a:b].copy_(master[a:b].to(dev))
        with torch.no_grad():
            for n, m in body:
                put(m.weight.data, saved[n])
            put(embed.weight.data, saved_embed)
            put(lm_head.weight.data, saved_head)
        if dev == "cuda":
            torch.cuda.empty_cache()

    # group the Linears of each transformer block (name: model.layers.<i>.<...>)
    blocks = {}
    for name, m in body:
        i = int(name.split(".")[2])
        blocks.setdefault(i, []).append((name, m))

    rows = []
    for ci, (k_body, k_aux) in enumerate(CONFIGS):
        if ci:
            restore()
        tensors = []
        dig_bits = sc_bytes = 0.0
        t0 = time.time()
        lm_head.to("cpu")               # free ~1.1 GB for the fp32 GPTQ working set
        if dev == "cuda":
            torch.cuda.empty_cache()
        for i in sorted(blocks):
            mods = blocks[i]
            eligible = [(n, m) for n, m in mods if m.weight.shape[1] <= GPTQ_MAX_IN]
            H = collect_hessians(model, eligible, cal_batches) if eligible else {}
            for name, m in mods:
                W = m.weight.data
                r, c = W.shape
                if name in H:
                    idx_t, sc_t, perm_np = gptq_apply(W, H[name], k_body)
                    idx_u8, s_np = idx_t.numpy(), sc_t.numpy()
                    rec = k9.encode_tensor(idx_u8, k_body, s_np, "ent8",
                                           perm=perm_np)
                    del idx_t, sc_t
                else:
                    idx_t, sc_t = e24.quantize_inplace(W, k_body)
                    idx_u8, s_np = idx_t.numpy(), sc_t.numpy()
                    rec = k9.encode_tensor(idx_u8, k_body, s_np, "ent8")
                    del idx_t, sc_t
                dig_bits += entropy_bits(idx_u8.reshape(-1), k_body) * r * c
                sc_bytes += len(k9.encode_scales(s_np, "ent8")) + 2 * k_body
                tensors.append(dict(name=name, shape=(r, c), group=GROUP,
                                    scale_mode="ent8", k=k_body, rec=rec))
                del idx_u8, s_np, rec
                if dev == "cuda":
                    torch.cuda.empty_cache()
            del H
            if i % 7 == 0:
                print(f"  k={k_body} block {i}: {time.time()-t0:.0f}s"
                      f" | GPU {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # embed (RTN), then bring lm_head back and quantize it (RTN)
        for nm, mod in (("__embed__", embed), ("__lm_head__", lm_head)):
            if nm == "__lm_head__":
                mod.to(dev)
                if dev == "cuda":
                    torch.cuda.empty_cache()
            W = mod.weight.data
            r, c = W.shape
            idx_t, sc_t = e24.quantize_inplace(W, k_aux)
            idx_u8, s_np = idx_t.numpy(), sc_t.numpy()
            dig_bits += entropy_bits(idx_u8.reshape(-1), k_aux) * r * c
            sc_bytes += len(k9.encode_scales(s_np, "ent8")) + 2 * k_aux
            tensors.append(dict(name=nm, shape=(r, c), group=GROUP,
                                scale_mode="ent8", k=k_aux,
                                rec=k9.encode_tensor(idx_u8, k_aux, s_np, "ent8")))
            del idx_t, sc_t, idx_u8, s_np
        print(f"k_body={k_body}: GPTQ+RTN done in {time.time()-t0:.0f}s")

        path = RESULTS / f"qwen7b_gptq_k{k_body}_embed{k_aux}.k9"
        size = k9.write_k9(path, tensors, threads=min(24, os.cpu_count() or 1))
        del tensors
        print(f"  wrote {path.name}: {size/1e6:.1f} MB = {8*size/n_total:.3f} b/param"
              f" | est {dig_bits/8/1e6 + sc_bytes/1e6:.1f} MB")

        pw = eh.ppl_per_window(model, wiki)
        pc = eh.ppl_per_window(model, code)
        d_code, d_se = eh.paired_delta(pc, pc0)
        d_wiki, _ = eh.paired_delta(pw, pw0)
        rows.append(dict(k_body=k_body, k_aux=k_aux, MB=size / 1e6,
                         bits_per_param=8 * size / n_total,
                         code=float(np.mean(pc)),
                         code_se=float(np.std(pc, ddof=1) / math.sqrt(len(pc))),
                         wiki=float(np.mean(pw)), d_code=d_code, d_code_se=d_se,
                         d_wiki=d_wiki))
        print(f"  code {np.mean(pc):.3f} vs bf16 {np.mean(pc0):.3f}"
              f" → Δ {d_code:+.3f} ± {d_se:.3f}")
        if dev == "cuda":
            torch.cuda.empty_cache()

    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# exp26 — 7B + GPTQ: K9 vs q4_k_m at scale",
        "",
        f"{MODEL_ID} ({n_total:,} params). GPTQ (act-order, damp {DAMP},"
        f" {N_CAL}x{BLOCK} calib) on every body matrix with in <= {GPTQ_MAX_IN}"
        f" ({100*n_gptq/sum(m.weight.numel() for _, m in body):.0f}% of body params);",
        "down_proj (in=18944) is RTN — its Hessian does not fit the card — and so",
        "are embed/lm_head (k=99). Same 8x1024 windows as exp24/25; bf16 reference",
        f"code {np.mean(pc0):.3f}.",
        "",
        "| k body | k embed | MB | b/param | code ppl | Δcode vs bf16 | wiki ppl |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['k_body']} | {r['k_aux']} | {r['MB']:.1f} |"
                     f" {r['bits_per_param']:.3f} | {r['code']:.3f}"
                     f" | {r['d_code']:+.3f} ± {r['d_code_se']:.3f} | {r['wiki']:.3f} |")
    lines += [
        "",
        "Reference points (exp24 RTN / exp25 GGUF): K9 k=15 RTN 3740.2 MB / +0.117;",
        "K9 k=9 RTN 3103.7 MB / +0.374; GGUF q4_k_m 4683.1 MB / +0.070;",
        "q2_k 3015.9 MB / +0.561.",
        "",
        "## Reading",
        "",
        *(["- If GPTQ moves k=15 from +0.117 toward <= +0.070, K9 beats q4_k_m on",
           "  both axes at 7B (20% smaller AND better) — the clean form of the result."]
          if (15, 99) in CONFIGS else
          ["- The k=9 probe: at 1.5B the largest GPTQ gain was on this coarse grid",
           "  (-0.449 code ppl), so it is the one setting where compensation still has",
           "  error to recover at 7B. RTN reference is 3103.7 MB / +0.374 (exp24)."]),
        "",
        f"wall {time.time()-t_all:.0f}s", "",
    ]
    MD_PATH.write_text("\n".join(lines))
    print(f"wrote {CSV_PATH.name} + {MD_PATH.name}")


if __name__ == "__main__":
    main()
