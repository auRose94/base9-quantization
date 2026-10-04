#!/usr/bin/env python3
"""exp25 — GGUF 7B baselines for the exp24 comparison.

Dequantizes the official Qwen2.5-Coder-7B-Instruct GGUF quants into the bf16 HF
model **tensor by tensor** (a full fp32 state dict would be ~30 GB) and evaluates
them on the same 8x1024 windows as exp24, so K9's 3740 MB / 3104 MB points can be
placed against the deployed frontier at 7B, not just at 1.5B.

Run: python3 exp25_gguf7b.py            # ~10 min (+ download)
"""
import csv
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".hf-cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import gguf

import eval_harness as eh

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
SNAP = (HERE.parent / ".hf-cache/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct-GGUF"
        / "snapshots")
BLOCK, N_WINDOWS = 1024, 8

_BLK = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.bias": "self_attn.v_proj.bias",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def gguf_to_hf(name):
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    if m and m.group(2) in _BLK:
        return f"model.layers.{m.group(1)}.{_BLK[m.group(2)]}"
    return None


def _param(model, key):
    """Resolve a state-dict key to the actual tensor (weight OR bias)."""
    mod = model.get_submodule(key.rsplit(".", 1)[0])
    return getattr(mod, key.rsplit(".", 1)[1])


def load_gguf_streaming(model, path, dev):
    r = gguf.GGUFReader(str(path))
    targets = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    loaded = 0
    skipped = []
    for t in r.tensors:
        key = gguf_to_hf(t.name)
        if key is None or key not in targets:
            skipped.append(t.name)
            continue
        arr = (np.asarray(t.data, dtype=np.float32) if t.tensor_type == 0
               else np.asarray(gguf.quants.dequantize(t.data, t.tensor_type),
                               dtype=np.float32))
        if tuple(arr.shape) != targets[key]:
            if tuple(arr.shape) == tuple(reversed(targets[key])):
                arr = arr.T
            else:
                skipped.append(f"{t.name}{arr.shape}!={targets[key]}")
                continue
        param = _param(model, key)
        src = torch.from_numpy(np.ascontiguousarray(arr))
        with torch.no_grad():                      # chunked: no full-size temp
            for a in range(0, param.shape[0], 512):
                b = min(param.shape[0], a + 512)
                param[a:b].copy_(src[a:b].to(dev))
        del arr, src
        loaded += 1
        if dev == "cuda":
            torch.cuda.empty_cache()
    return path.stat().st_size, loaded, skipped


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

    snap = next(SNAP.iterdir())
    wiki = eh.get_windows(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                          "test", N_WINDOWS, BLOCK, label="wiki")
    code = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean-valid", None,
                          "train", N_WINDOWS, BLOCK, label="code")
    master = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}

    print("bf16 reference (same process, for paired deltas)...")
    pw0 = eh.ppl_per_window(model, wiki)
    pc0 = eh.ppl_per_window(model, code)
    print(f"  code {np.mean(pc0):.3f} | wiki {np.mean(pw0):.3f}")

    def restore():
        with torch.no_grad():
            for k, v in master.items():
                param = _param(model, k)          # weight OR bias
                for a in range(0, param.shape[0], 512):
                    b = min(param.shape[0], a + 512)
                    param[a:b].copy_(v[a:b].to(dev))   # v is bf16 already
        if dev == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for q in ("q4_k_m", "q2_k"):
        path = snap / f"qwen2.5-coder-7b-instruct-{q}.gguf"
        if not path.exists():
            print(f"missing {path.name} — skipping")
            continue
        t0 = time.time()
        size, loaded, skipped = load_gguf_streaming(model, path, dev)
        pw = eh.ppl_per_window(model, wiki)
        pc = eh.ppl_per_window(model, code)
        d_code, d_code_se = eh.paired_delta(pc, pc0)
        d_wiki, _ = eh.paired_delta(pw, pw0)
        rows.append(dict(quant=q, MB=size / 1e6, bits_per_param=8 * size / n_total,
                         code=float(np.mean(pc)),
                         code_se=float(np.std(pc, ddof=1) / np.sqrt(len(pc))),
                         wiki=float(np.mean(pw)), d_code=d_code,
                         d_code_se=d_code_se, d_wiki=d_wiki, loaded=loaded,
                         skipped=len(skipped)))
        print(f"GGUF {q}: {size/1e6:.1f} MB ({8*size/n_total:.2f} b/p) |"
              f" code {np.mean(pc):.3f} (Δ {d_code:+.3f} ± {d_code_se:.3f}) |"
              f" wiki {np.mean(pw):.3f} | {loaded} tensors, {len(skipped)} skipped"
              f" | {time.time()-t0:.0f}s")
        restore()

    with open(RESULTS / "exp25_gguf7b.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# exp25 — GGUF 7B baselines (dequantized into bf16, same windows)",
        "",
        f"{MODEL_ID}, {N_WINDOWS}x{BLOCK} tokens per corpus — identical windows to",
        "exp24, so these are directly comparable to the K9 7B points.",
        "",
        "| quant | MB | b/param | code ppl | Δcode vs bf16 (paired) | wiki ppl |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['quant']} | {r['MB']:.1f} | {r['bits_per_param']:.2f}"
                     f" | {r['code']:.3f} | {r['d_code']:+.3f} ± {r['d_code_se']:.3f}"
                     f" | {r['wiki']:.3f} |")
    lines += [
        "",
        "K9 7B for comparison (exp24, same windows; bf16 reference code 3.493):",
        "k=15 + embed99 = **3740.2 MB, code 3.610 (+0.117)**;",
        "k=9 + embed99 = **3103.7 MB, code 3.867 (+0.374)**.",
        "",
        f"wall {time.time()-t_all:.0f}s", "",
    ]
    (RESULTS / "exp25_gguf7b.md").write_text("\n".join(lines))
    print("wrote results/exp25_gguf7b.{csv,md}")


if __name__ == "__main__":
    main()
