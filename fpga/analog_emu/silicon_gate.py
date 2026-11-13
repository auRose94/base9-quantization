#!/usr/bin/env python3
"""analog_emu silicon gate — Tang Nano 20K vs the python goldens, over a wire.

Flow: regenerate goldens (same params as the bitstream), convert to the UART
protocol, talk to the board (FT2232 channel B = /dev/ttyUSB1 typically),
diff every yq word, spike word, and membrane.  GATE PASS = silicon and
python agree bit-for-bit on the degraded array.

Usage:
  .venv-glm/bin/python silicon_gate.py            # auto-find the serial port
  .venv-glm/bin/python silicon_gate.py --port /dev/ttyUSB1 --tokens 8
Flash first:  openFPGALoader --board tangnano20k synth/emu_top.fs
"""
from __future__ import annotations

import argparse
import glob
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = COLS = 64
WW = 16
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


def read_words(path: str, width_nibbles: int):
    out = []
    for line in open(HERE / path):
        line = line.strip()
        if line:
            out.extend(int(t, 16) for t in line.split())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    # 1) goldens — identical params to the RTL
    subprocess.run([sys.executable, str(HERE / "golden.py"),
                    "--rows", str(ROWS), "--cols", str(COLS),
                    "--tokens", str(args.tokens), "--seed", str(args.seed),
                    "--out-shift", str(OUT_SHIFT), "--thresh", str(THRESH),
                    "--sigma-lsb", str(SIGMA_LSB), "--stuck-p", str(STUCK_P),
                    "--col-gamma", str(COL_GAMMA), "--rn-pow", str(RN_POW),
                    "--drift-shift", str(DRIFT_SHIFT),
                    "--drift-every", str(DRIFT_EVERY),
                    "--leak-shift", str(LEAK_SHIFT),
                    "--lfsr-seed", str(LFSR_SEED)],
                   check=True, cwd=HERE)
    W = read_words("golden_w.memh", 4)      # 4096 u16 (two's complement)
    G = read_words("golden_g.memh", 2)      # 64 u8
    X = read_words("golden_tokens.hex", 4)  # NTOK*64 i16
    Y = read_words("golden_expected.hex", 10)  # NTOK*64 s40 (yq << OUT_SHIFT)
    S = read_words("golden_spikes.hex", ((ROWS + 3) // 4) * 4)
    V = read_words("golden_vmem.hex", 8)
    NT = args.tokens
    yq_exp = [[(y >> OUT_SHIFT) & 0xFFFF for y in Y[t*ROWS:(t+1)*ROWS]]
              for t in range(NT)]

    # 2) serial
    port = args.port
    if not port:
        cands = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
        if not cands:
            raise SystemExit("no serial ports found — is the board plugged in?")
        port = cands[-1]   # FT2232: ch A (JTAG) first, ch B (UART) last
    import serial
    ser = serial.Serial(port, args.baud, timeout=3)
    print(f"port: {port}")

    def expect_k():
        b = ser.read(1)
        if b != b"K":
            raise SystemExit(f"expected 'K', got {b!r} — check rxd pin 70?")

    def cmd_id():
        ser.reset_input_buffer()
        ser.write(b"I")
        time.sleep(0.2)
        return ser.read(32)

    # find the right port: probe each candidate with 'I'
    if not args.port:
        for cand in cands:
            try:
                ser = serial.Serial(cand, args.baud, timeout=1)
                ser.write(b"I")
                time.sleep(0.15)
                r = ser.read(16)
                if r.startswith(b"analog-emu"):
                    print(f"board found on {cand}: {r.decode(errors='replace').strip()}")
                    ser.timeout = 3
                    break
                ser.close()
            except Exception:
                continue
        else:
            raise SystemExit("no board answered 'I' on any port — flash the "
                             "bitstream and check the rxd pin (70)")
    else:
        ser.write(b"I")
        time.sleep(0.2)
        r = ser.read(16)
        print(f"board: {r.decode(errors='replace').strip()}")

    # 3) program image
    payload = b"P"
    for w in W[:ROWS*COLS]:
        payload += struct.pack("<H", w & 0xFFFF)
    payload += bytes(g & 0xFF for g in G[:COLS])
    ser.write(payload)
    expect_k()
    print(f"programmed: {ROWS*COLS} weights + {COLS} gains")

    # 4) tokens (+ drift sweeps on the golden schedule)
    n_mis = 0
    for t in range(NT):
        ser.write(b"X")
        for x in X[t*COLS:(t+1)*COLS]:
            ser.write(struct.pack("<H", x & 0xFFFF))
        buf = ser.read(ROWS*2 + 8)
        if len(buf) != ROWS*2 + 8:
            raise SystemExit(f"token {t}: short read ({len(buf)} B) — RX path?")
        expect_k()
        got_yq = struct.unpack(f"<{ROWS}h", buf[:ROWS*2])
        got_sp = struct.unpack("<Q", buf[ROWS*2:])[0]
        for r in range(ROWS):
            if got_yq[r] != yq_exp[t][r]:
                n_mis += 1
                if n_mis <= 8:
                    print(f"YQ MISMATCH tok {t} row {r}: "
                          f"exp {yq_exp[t][r]} got {got_yq[r]}")
        if got_sp != S[t]:
            n_mis += 1
            print(f"SPIKE MISMATCH tok {t}: exp {S[t]:#x} got {got_sp:#x}")
        if (t + 1) % DRIFT_EVERY == 0 and t + 1 < NT:
            ser.write(b"D")
            expect_k()
        print(f"token {t+1}/{NT} done ({n_mis} mismatches so far)", flush=True)

    # 5) membranes
    ser.write(b"V")
    vbuf = ser.read(ROWS*4)
    expect_k()
    got_v = struct.unpack(f"<{ROWS}i", vbuf)
    for r in range(ROWS):
        if got_v[r] != V[r]:
            n_mis += 1
            if n_mis <= 8:
                print(f"VMEM MISMATCH row {r}: exp {V[r]} got {got_v[r]}")

    if n_mis == 0:
        print(f"GATE PASS — silicon == python, bit-exact "
              f"({NT} tokens x {ROWS} rows)")
    else:
        print(f"GATE FAIL — {n_mis} mismatches")
        sys.exit(1)


if __name__ == "__main__":
    main()
