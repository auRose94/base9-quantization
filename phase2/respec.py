#!/usr/bin/env python3
"""Rewrite the SFT user prompts from the stored targets — no Godot re-scan.

The corpus build kept each assistant message (the real file content), so a
prompt-wording change is a cheap text transform rather than a 15-minute
rescan. Used to align the training prompt with the evaluation prompt.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import build_sft as bs

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
with open(dst, "w") as fo:
    for line in open(src):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        code = r["messages"][2]["content"]
        r["messages"][1]["content"] = bs.spec(code, r["path"])
        fo.write(json.dumps(r) + "\n")
        n += 1
print(f"rewrote {n:,} prompts → {dst}")
