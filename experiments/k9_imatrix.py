#!/usr/bin/env python3
"""Track A helper: produce the npz importance matrix consumed by
experiments/glm_k9_build.py --imatrix.

This script does NOT compute importance itself; it packages an importance
matrix produced by upstream llama.cpp's `llama-imatrix` (or any compatible
tool) into the (tensor-name -> (cols,) float32) npz that glm_k9_build pass1
expects. Keeping conversion separate so the exact source/provenance of the
importance numbers stays visible and reproducible.

Usage:
  python experiments/k9_imatrix.py --in matrix.dat --out matrix.npz

The input format expected is the same one llama.cpp's `llama-imatrix` writes:
a gguf-ish text? In practice we support a JSON sidecar:

  { "tensor.name": [float, ... per column], ... }

i.e. if you have a script that extracts per-column importance from whatever
tool you use, dump it as JSON and convert here. This deliberately does NOT
fake importance from weight magnitudes: that would hide the calibration
lever behind a proxy and defeat track A's purpose (docs/13 §1.2 item 4).
"""
import argparse
import json
import numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True,
                    help="JSON file: {tensor_name: [per-column importance...]}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.loads(Path(a.src).read_text())
    out = {}
    for name, vals in data.items():
        arr = np.asarray(vals, dtype=np.float32).reshape(-1)
        # importance matrices are often squared; sqrt keeps scales comparable
        # to |w| magnitudes (matches upstream's sqrt(imatrix) convention).
        arr = np.sqrt(np.maximum(arr, 0.0))
        out[name] = arr
    np.savez(a.out, **out)
    print(f"wrote {len(out)} tensors -> {a.out}")


if __name__ == "__main__":
    main()
