#!/usr/bin/env python3
"""Perplexity of a merged tuned model vs its K9 files, over the exp24 windows.

    python3 ppl_check_k9.py                             # merged_7b on cuda
    python3 ppl_check_k9.py --model out/merged_14b --cpu --prefix qwen14b_tuned
"""
import argparse
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
import numpy as np, torch, k9, eval_harness as eh

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "phase2/out/merged_7b"))
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--prefix", default="qwen7b_tuned")
a = ap.parse_args()
RESULTS = ROOT / "results"
dev = "cpu" if a.cpu else "cuda"
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    attn_implementation="sdpa").to(dev).eval()
wiki = eh.get_windows(tok, dev, "Salesforce/wikitext", "wikitext-103-raw-v1",
                      "test", 8, 1024, label="wiki")
code = eh.get_windows(tok, dev, "codeparrot/codeparrot-clean-valid", None,
                      "train", 8, 1024, label="code")
pw0, pc0 = eh.ppl_per_window(model, wiki), eh.ppl_per_window(model, code)
print(f"bf16: code {np.mean(pc0):.3f} | wiki {np.mean(pw0):.3f}", flush=True)
for tag, path in [("k15", f"{a.prefix}_k15_embed99.k9"),
                  ("k9", f"{a.prefix}_k9_embed99.k9"),
                  ("k63", f"{a.prefix}_k63_embed99.k9")]:
    p = RESULTS / path
    if not p.exists():
        print(f"{tag}: (no file {path} — skipped)", flush=True)
        continue
    with k9.K9File(str(p)) as f:
        for name, rec in f.iter_tensors():
            mod = (model.model.embed_tokens if name == "__embed__"
                   else model.lm_head if name == "__lm_head__"
                   else model.get_submodule(name))
            k9.load_into(mod, rec, rec["group"], device=dev, row_chunk=4096)
    pw, pc = eh.ppl_per_window(model, wiki), eh.ppl_per_window(model, code)
    dc, dcs = eh.paired_delta(pc, pc0)
    dw, dws = eh.paired_delta(pw, pw0)
    print(f"{tag}: code {np.mean(pc):.3f} (Δ{dc:+.3f}±{dcs:.3f}) | "
          f"wiki {np.mean(pw):.3f} (Δ{dw:+.3f}±{dws:.3f})", flush=True)