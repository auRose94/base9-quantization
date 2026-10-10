"""Generate BRAM init files for the on-silicon Gate-0 (Tang Nano 20K).

stim.hex: 21 Gate-0 records x 118 words (main BRAM, 4096 entries, 32-bit wide).
  Words 0..49  = wblk bytes 0..199  (word j = byte 4j | 4j+1<<8 | ...; matches
                 RTL port indexing wblk[8b +: 8], byte 0 at LSB)
  Words 50..117 = act bytes 0..271  (same rule)
exp.hex: 21 words = expected cumulative fp32 sumf bits after each superblock
  (per record, C/golden chain state; the silicon top chains partials itself).
"""
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORDS_PER_REC = 118          # 50 wblk + 68 act
N_RECS = 21
DEPTH = 4096                 # power of 2 >= N_RECS * WORDS_PER_REC


def main():
    data = np.load("/tmp/gate0_vecs.npz")
    n = 1 + max(int(k.split("_")[0][3:]) for k in data.files if k.startswith("rec"))
    assert n == N_RECS, f"expected {N_RECS} records, npz has {n}"
    mem = []
    exp = []
    flags = []
    prev_case = -1
    for i in range(n):
        wblk = data[f"rec{i}_wblk"].tobytes()
        act = data[f"rec{i}_act"].tobytes()
        assert len(wblk) == 200 and len(act) == 272
        blob = wblk + act
        for j in range(WORDS_PER_REC):
            w = int.from_bytes(blob[4 * j: 4 * j + 4], "little")
            mem.append(f"{w:08x}")
        exp.append(f"{int(data[f'rec{i}_gold']):08x}")
        cid = int(data[f'rec{i}_case'])
        flags.append("1" if i == 0 or cid != prev_case else "0")
        prev_case = cid
    while len(mem) < DEPTH:
        mem.append("00000000")
    (HERE / "stim.hex").write_text("\n".join(mem) + "\n")
    (HERE / "exp.hex").write_text("\n".join(exp) + "\n")
    (HERE / "flags.hex").write_text("\n".join(flags) + "\n")
    print(f"stim.hex: {DEPTH} words; exp.hex: {n} entries; flags.hex: {n} case-reset bits")
    case_ids = [int(data[f'rec{i}_case']) for i in range(n)]
    print("case chain starts at records:",
          [i for i, c in enumerate(case_ids) if i == 0 or c != case_ids[i - 1]])


if __name__ == "__main__":
    main()