"""Analyze the tb_gate0sd NIN_SD_DUMP json against the real-tensor npz:
locates the first record where the RTL's value departs from the goldens and
classifies it (partial-level vs chain-level vs exp-quirk)."""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "sim"))
from golden import f32b, k96_partials   # noqa: E402

dump = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gate0sd_real_dump.json").read_text())
npz = np.load("/tmp/gate0_real_vecs.npz")
import re
n = 1 + max(int(re.match(r"rec(\d+)_wblk", k).group(1))
            for k in npz.files if re.match(r"rec(\d+)_wblk", k))

print(f"{'i':>3} {'cs':>4} {'partial':>12} {'npP':>12} {'pn':>3} {'chainN':>12} {'exp':>12} {'cn==exp':>7}")
first = None
for d in dump:
    i = d["i"]
    if i >= n:
        break
    wblk = npz[f"rec{i}_wblk"]
    acts = [bytes(npz[f"rec{i}_act"][34 * j: 34 * (j + 1)]) for j in range(8)]
    g1, _ = k96_partials([bytes(wblk)], acts)
    npP = int(g1[0])
    pn = "=OK" if npP == d["partial"] else "DIF"
    cn = "=OK" if d["chain_next"] == d["exp"] else "DIF"
    if cn == "DIF" and first is None:
        first = (i, d, npP)
    print(f"{i:3d} {d['cs_r']:4d} {d['partial']:12x} {npP:12x} {pn:>3} "
          f"{d['chain_next']:12x} {d['exp']:12x} {cn:>7}")

if first:
    i, d, npP = first
    wblk = bytes(npz[f"rec{i}_wblk"])
    acts = [bytes(npz[f"rec{i}_act"][34 * j: 34 * (j + 1)]) for j in range(8)]
    print(f"\nFIRST divergence at rec {i} (cs={d['cs_r']}):")
    print(f"  rtl partial      {d['partial']:08x} -> {struct.unpack('<f', struct.pack('<I', d['partial']))[0]:r}")
    print(f"  numpy partial    {npP:08x} -> {struct.unpack('<f', struct.pack('<I', npP))[0]:r}")
    print(f"  rtl chain next   {d['chain_next']:08x}")
    print(f"  exp              {d['exp']:08x}")
    # per-group replay of the C order with the SAME bytes, printing intermediates
    w = np.frombuffer(wblk, dtype=np.uint8)
    sc = w[:8].view("<f2").astype(np.float32)
    print("  wblk scales f16->f32:", [f"{x:.8g}" for x in sc])
    for g in range(4):
        sumi_parts = []
        for k in (0, 1):
            yb = np.frombuffer(acts[8 * 0 + 2 * g + k], dtype=np.uint8)
            d1 = yb[:2].view("<f2").astype(np.float32)[0]
            qy = yb[2:34].view(np.int8)
            sblk = 0
            for b in range(16):
                c0 = 32 * k + 2 * b
                lb = int(w[8 + 32 * g + 16 * k + b])
                h0 = (int(w[136 + 16 * g + ((c0 >> 2) & 15)]) >> ((c0 & 3) * 2)) & 3
                h1 = (int(w[136 + 16 * g + ((c0 + 1 >> 2) & 15)]) >> (((c0 + 1) & 3) * 2)) & 3
                u0 = ((lb & 0x0F) | (h0 << 4)) - 32
                u1 = ((lb >> 4) | (h1 << 4)) - 32
                sblk += u0 * int(qy[2 * b]) + u1 * int(qy[2 * b + 1])
            sumi_parts.append((k, d1, sblk))
        print(f"  g{g}: d0={sc[g]:.8g} blocks={sumi_parts}")
else:
    print("\nno chain divergence found in the dump; compare the LAST chain against npz gold directly")