#!/usr/bin/env python3
"""Convert the paired golden files to 16-bit flat files for tb_tile_ser."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
SHIFT = int(sys.argv[1]) if len(sys.argv) > 1 else 11
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
NB = (ROWS + 3) // 4

def rd(p):
    out = []
    for line in open(HERE / p):
        if line.strip():
            out.extend(int(t, 16) for t in line.split())
    return out

W = rd("golden_w.memh")
with open(HERE / "golden_w16.memh", "w") as fh:
    for v in W: fh.write(f"{v & 0xFFFF:04x}\n")
X = rd("golden_tokens.hex")
with open(HERE / "golden_tokens16.memh", "w") as fh:
    for v in X: fh.write(f"{v & 0xFFFF:04x}\n")
Y = rd("golden_expected.hex")
with open(HERE / "golden_yq16.memh", "w") as fh:
    for v in Y: fh.write(f"{(v >> SHIFT) & 0xFFFF:04x}\n")
S = rd("golden_spikes.hex")
with open(HERE / "golden_spikes16.memh", "w") as fh:
    for v in S: fh.write(f"{v & ((1 << (NB*4)) - 1):0{NB}x}\n")
print(f"flat16 written (shift {SHIFT}, spike width {NB} nibbles)")
