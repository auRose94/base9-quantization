#!/usr/bin/env python3
"""analog_emu golden generator — pure-stdlib integer reference for the tile.

Builds a k9-grid weight image (digits + per-group scales -> Q4.12), bakes
write-path degradation into the image (uniform +-sigma LSB programming noise,
stuck rails, column gain), generates tokens, runs the reference datapath
(exactly the contract in README.md: floor shifts, symmetric-odd quantizers,
no division), and writes:

    golden_w.memh        ROWS*COLS x 4-hex  (tile loads this under SIM_LOAD)
    golden_g.memh        COLS x 2-hex       (column gains, Q0.7)
    golden_tokens.hex    NTOK*COLS x 4-hex  (input tokens, Q4.12)
    golden_expected.hex  NTOK*ROWS x 10-hex (expected 40-bit outputs)
    sim_flags.txt        the -G parameter line for the simulators

Every arithmetic op here has a defined RTL twin; the gate is bit-exact.
"""
from __future__ import annotations

import argparse
import math
import random

WW = 16
FRAC = 12            # Q4.12
K = 9                # ninths grid (H = 4): the k9-as-memory-map story
H = (K - 1) // 2
GROUP = 64


def clip(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dacb", type=int, default=8)
    ap.add_argument("--adcb", type=int, default=8)
    ap.add_argument("--sigma-lsb", type=int, default=12,
                    help="programming noise, +-LSBs of the Q4.12 weight word")
    ap.add_argument("--stuck-p", type=float, default=0.01)
    ap.add_argument("--col-gamma", type=float, default=0.03)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    R, C = args.rows, args.cols
    L_IN = (1 << (args.dacb - 1)) - 1
    L_ADC = (1 << (args.adcb - 1)) - 1
    SHR_IN = WW - args.dacb

    # --- weight image: k9 grid -> Q4.12 -> baked degradation ----------------
    W = []
    n_stuck = 0
    for r in range(R):
        row = []
        for g0 in range(0, C, GROUP):
            gsz = min(GROUP, C - g0)
            grp = [rng.gauss(0, 0.3) for _ in range(gsz)]
            m = max(abs(w) for w in grp) or 1.0
            for w in grp:
                d = clip(int(round(w / m * H)) + H, 0, K - 1)
                wq = math.floor(m * (d - H) / H * (1 << FRAC))   # Q4.12
                wq += rng.randint(-args.sigma_lsb, args.sigma_lsb)
                if rng.random() < args.stuck_p:
                    wq = 32760 if w >= 0 else -32760
                    n_stuck += 1
                row.append(clip(wq, -32760, 32760))
        W.append(row)

    # --- column gains (the per-group calibration the findings demand) ------
    G = [clip(int(round(256 * (1 + rng.gauss(0, args.col_gamma)))), 1, 255)
         for _ in range(C)]

    # --- tokens -------------------------------------------------------------
    X = [[clip(int(round(rng.gauss(0, 0.5) * (1 << FRAC))), -16000, 16000)
          for _ in range(C)] for _ in range(args.tokens)]

    # --- reference datapath (pure int, floor semantics) ---------------------
    outs = []
    maxacc = 0
    for tok in X:
        accs = []
        for r in range(R):
            acc = 0
            for c in range(C):
                xq = clip(tok[c] >> SHR_IN, -L_IN, L_IN) << SHR_IN
                xg = (xq * G[c]) >> 7
                acc += W[r][c] * xg
            accs.append(acc)
            maxacc = max(maxacc, abs(acc))
        outs.append(accs)

    shift = 4
    while (L_ADC << shift) < maxacc:
        shift += 1
    assert maxacc < (1 << 39), "accumulator overflow — shrink the model"

    Y = [[clip(a >> shift, -L_ADC, L_ADC) << shift for a in accs]
         for accs in outs]

    # --- emit ----------------------------------------------------------------
    with open("golden_w.memh", "w") as fh:
        for r in range(R):
            fh.write(" ".join(f"{v & 0xFFFF:04x}" for v in W[r]) + "\n")
    with open("golden_g.memh", "w") as fh:
        fh.write(" ".join(f"{v:02x}" for v in G) + "\n")
    with open("golden_tokens.hex", "w") as fh:
        for tok in X:
            fh.write(" ".join(f"{v & 0xFFFF:04x}" for v in tok) + "\n")
    with open("golden_expected.hex", "w") as fh:
        for accs in Y:
            fh.write(" ".join(f"{v & 0xFFFFFFFFFFFF:010x}" for v in accs) + "\n")
    with open("sim_flags.txt", "w") as fh:
        fh.write(f"-GROWS={R} -GCOLS={C} -GWW={WW} -GDACB={args.dacb} "
                 f"-GADCB={args.adcb} -GOUT_SHIFT={shift} -GNTOK={args.tokens}\n")

    print(f"image: {R}x{C} k{K} grid, Q4.12; sigma_lsb={args.sigma_lsb} "
          f"stuck={n_stuck} gamma={args.col_gamma}")
    print(f"maxacc={maxacc} -> OUT_SHIFT={shift} (ADC rails "
          f"+-{L_ADC << shift})")
    print("golden files written; run `make gate`")


if __name__ == "__main__":
    main()
