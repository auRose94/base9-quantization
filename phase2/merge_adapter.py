#!/usr/bin/env python3
"""Merge a QLoRA adapter into bf16 weights and save a standalone model dir.

The adapter is a set of low-rank deltas, so applying it does not need the 4-bit
base it was trained against — merging into a plain bf16 model is the standard
export path and yields something any HF/llama.cpp/chat loader can consume.

    python3 merge_adapter.py --adapter out/qlora_1.5b_v2/adapter \
        --out out/merged_1.5b_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

import torch                                                      # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    a = ap.parse_args()

    dt = dict(bfloat16=torch.bfloat16, float16=torch.float16,
              float32=torch.float32)[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=dt, low_cpu_mem_usage=True, attn_implementation="sdpa")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
    model.config.use_cache = True
    out = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    n = sum(p.numel() for p in model.parameters())
    meta = dict(base=a.base, adapter=a.adapter, dtype=a.dtype, params=n,
                bytes=int(n * (2 if dt != torch.float32 else 4)))
    (out / "merge_info.json").write_text(json.dumps(meta, indent=2))
    print(f"merged → {out}  ({n/1e9:.2f} B params, "
          f"{meta['bytes']/1e9:.1f} GB {a.dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
