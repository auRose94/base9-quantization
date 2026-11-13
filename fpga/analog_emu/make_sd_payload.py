#!/usr/bin/env python3
"""Build the analog_emu SD payload (pay_analog.hex) from the goldens.

Runs golden.py (params MUST match the RTL tile constants — see
emu_sd_top.sv), packs the image + tokens + expected results into the G0
SD-gate format: 32-bit words, little-endian bytes, one word per line, 128
words per LBA.  Layout (word offsets):

    0    ..2047   weights W  (4096 x u16 LE, two per word)
    2048..2063    gains G    (64 x u8, four per word)
    2064..2319    tokens     (NT x 64 x i16 LE, two per word)
    2320..2575    expected yq (NT x 64 x i16 LE, two per word)
    2576..2591    expected spike words (NT x u64 LE, two words each)
    2592..2655    expected membranes (64 x i32 LE)
    2656..2687    zero pad -> 2688 words = 21 LBAs
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = COLS = 64
NT = 8
OUT_SHIFT = 14
THRESH = 96
SIGMA_LSB = 12
STUCK_P = 0.01
COL_GAMMA = 0.03
RN_POW = 3
DRIFT_SHIFT = 6
DRIFT_EVERY = 2
LEAK_SHIFT = 3
LFSR_SEED = 0xACE1
SEED = 5


def read_words(path: str):
    out = []
    for line in open(HERE / path):
        line = line.strip()
        if line:
            out.extend(int(t, 16) for t in line.split())
    return out


def main() -> None:
    subprocess.run([sys.executable, str(HERE / "golden.py"),
                    "--rows", str(ROWS), "--cols", str(COLS),
                    "--tokens", str(NT), "--seed", str(SEED),
                    "--out-shift", str(OUT_SHIFT), "--thresh", str(THRESH),
                    "--sigma-lsb", str(SIGMA_LSB), "--stuck-p", str(STUCK_P),
                    "--col-gamma", str(COL_GAMMA), "--rn-pow", str(RN_POW),
                    "--drift-shift", str(DRIFT_SHIFT),
                    "--drift-every", str(DRIFT_EVERY),
                    "--leak-shift", str(LEAK_SHIFT),
                    "--lfsr-seed", str(LFSR_SEED)], check=True, cwd=HERE)

    W = read_words("golden_w.memh")        # 4096 u16
    G = read_words("golden_g.memh")        # 64 u8
    X = read_words("golden_tokens.hex")    # NT*64 i16
    Y = read_words("golden_expected.hex")  # NT*64 s40 = yq << OUT_SHIFT
    S = read_words("golden_spikes.hex")    # NT u64
    V = read_words("golden_vmem.hex")      # 64 s32
    assert len(W) == ROWS * COLS and len(X) == NT * COLS

    yq = [(y >> OUT_SHIFT) & 0xFFFF for y in Y]

    words = []
    # weights: two u16 per word, LE
    for k in range(ROWS * COLS // 2):
        words.append((W[2*k] & 0xFFFF) | ((W[2*k+1] & 0xFFFF) << 16))
    # gains: four u8 per word
    for k in range(COLS // 4):
        words.append((G[4*k] | (G[4*k+1] << 8) | (G[4*k+2] << 16)
                      | (G[4*k+3] << 24)))
    tok_w = (X[2*k] & 0xFFFF) | ((X[2*k+1] & 0xFFFF) << 16)
    words += [((X[2*k] & 0xFFFF) | ((X[2*k+1] & 0xFFFF) << 16))
              for k in range(NT * COLS // 2)]
    words += [((yq[2*k]) | (yq[2*k+1] << 16)) for k in range(NT * ROWS // 2)]
    for t in range(NT):
        words.append(S[t] & 0xFFFFFFFF)
        words.append((S[t] >> 32) & 0xFFFFFFFF)
    words += [v & 0xFFFFFFFF for v in V]
    while len(words) < 2688:
        words.append(0)
    assert len(words) == 2688, f"payload size {len(words)}"

    with open(HERE / "pay_analog.hex", "w") as fh:
        for w in words:
            fh.write(f"{w:08x}\n")
    # raw sectors for the host-side dd: the SPI reader assembles words
    # BIG-ENDIAN from the sector bytes (first byte on the wire = MSB)
    import struct
    with open(HERE / "pay_analog.bin", "wb") as fh:
        for w in words:
            fh.write(struct.pack(">I", w))
    print(f"pay_analog.hex: {len(words)} words = {len(words)//128} LBAs "
          f"(LBA 2048..{2048 + len(words)//128 - 1})")


if __name__ == "__main__":
    main()
