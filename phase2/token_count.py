#!/usr/bin/env python3
"""Exact token counts for the GDScript corpus (dumps → Qwen tokenizer).

The chars/3.5 estimate in build_corpus --stats is only a placeholder; this is
the number a training budget needs. Also reports file-length percentiles so we
can see how much of the corpus is long enough to be a useful SFT sample.

Usage: python3 token_count.py [--model Qwen/Qwen2.5-Coder-7B-Instruct]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_corpus as bc

HERE = Path(__file__).resolve().parent


def pct(vals: list, q: float) -> int:
    if not vals:
        return 0
    return vals[min(len(vals) - 1, int(len(vals) * q))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--min-block", type=int, default=0,
                    help="ignore files shorter than this many chars")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    projects = bc.load_all()
    per_ver: dict = {}
    for p in projects:
        v = bc.major(p.version)
        files = [c for c in p.gd().values() if len(c) >= a.min_block]
        per_ver.setdefault(v, {"projects": 0, "files": 0, "chars": 0, "tokens": 0,
                               "lens": []})
        d = per_ver[v]
        d["projects"] += 1
        d["files"] += len(files)
        d["chars"] += sum(len(c) for c in files)
        for c in files:
            n = len(tok(c, add_special_tokens=False)["input_ids"])
            d["tokens"] += n
            d["lens"].append(n)
        if d["files"] and d["files"] % 20000 < len(files):
            print(f"  godot {v}: {d['files']:,} files, {d['tokens']:,} tok", flush=True)

    tot = 0
    print(f"\nmodel: {a.model} | min_block {a.min_block}")
    for v in sorted(per_ver, key=lambda x: (x == "?", x)):
        d = per_ver[v]
        s = sorted(d["lens"])
        tot += d["tokens"] if v == "4" else 0
        print(f"  godot {v:>2s}: {d['projects']:5d} projects {d['files']:7d} files"
              f" {d['chars']:13,} chars {d['tokens']:13,} tokens"
              f" ({d['tokens']/max(d['chars'],1):.3f} tok/char)")
        print(f"           file tokens: p50 {pct(s,.5):5d} p90 {pct(s,.9):6d}"
              f" p99 {pct(s,.99):7d} max {s[-1] if s else 0:8d}"
              f" | >=256 tok: {sum(1 for x in s if x>=256):,}")
    print(f"\nGodot-4 training tokens (exact): {tot:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
