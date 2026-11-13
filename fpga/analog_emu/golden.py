#!/usr/bin/env python3
"""analog_emu golden generator v2 — pure-stdlib integer reference for the tile.

Builds a k9-grid weight image (digits + per-group scales -> Q4.12), bakes
write-path degradation into the image (uniform +-sigma LSB programming noise,
stuck rails, column gain), then runs the reference datapath with python twins
of every RTL mechanism:

  read noise   xorshift16(7,9,8) LFSR, drawn fresh per row per token,
               symmetric-odd +-2^(k-1)-1 ADC-LSBs (RTL twin: on-tile LFSR)
  drift        W <- W - (W >> DRIFT_SHIFT) per cell, swept between token
               batches on the TB's schedule (exponential relaxation)
  LIF          cand = (V >> LEAK_SHIFT) + yq; spike if cand >= THRESH;
               V <- 0 on spike (BrainScaleS-style trigger, per row)

Writes golden_{w,g,tokens,expected,spikes,vmem} + sim_flags.txt (--sweep
additionally prints the tile-level survival sweep: output RMS vs read-noise
power, the primitive-level version of the docs/15 curves).

Every arithmetic op here has a defined RTL twin; the gate is bit-exact.
"""
from __future__ import annotations

import argparse
import math
import random

WW = 16
FRAC = 12            # Q4.12
K = 9                # ninths grid (H = 4)
H = (K - 1) // 2
GROUP = 64


def clip(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def xorshift16(x: int) -> int:
    x ^= (x << 7) & 0xFFFF
    x ^= x >> 9
    x ^= (x << 8) & 0xFFFF
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dacb", type=int, default=8)
    ap.add_argument("--adcb", type=int, default=8)
    ap.add_argument("--sigma-lsb", type=int, default=12)
    ap.add_argument("--stuck-p", type=float, default=0.01)
    ap.add_argument("--col-gamma", type=float, default=0.03)
    ap.add_argument("--rn-pow", type=int, default=3,
                    help="read noise k: +-(2^(k-1)-1) ADC-LSBs, k<2 = off")
    ap.add_argument("--drift-shift", type=int, default=6)
    ap.add_argument("--drift-every", type=int, default=2)
    ap.add_argument("--leak-shift", type=int, default=3)
    ap.add_argument("--thresh", type=int, default=384)
    ap.add_argument("--no-lif", action="store_true")
    ap.add_argument("--lfsr-seed", type=int, default=0xACE1)
    ap.add_argument("--out-shift", type=int, default=0,
                    help="override ADC rail calibration (physical: rails near "
                         "the signal, outliers clip)")
    ap.add_argument("--sweep", action="store_true",
                    help="print the tile-level read-noise survival sweep")
    args = ap.parse_args()
    lif = not args.no_lif
    rng = random.Random(args.seed)

    R, C = args.rows, args.cols
    NT = args.tokens
    L_IN = (1 << (args.dacb - 1)) - 1
    L_ADC = (1 << (args.adcb - 1)) - 1
    SHR_IN = WW - args.dacb
    RN_MSK = (1 << args.rn_pow) - 1
    RN_OFF = (1 << (args.rn_pow - 1)) - 1

    # --- weight image: k9 grid -> Q4.12 -> baked degradation ----------------
    W0 = []
    n_stuck = 0
    for r in range(R):
        row = []
        for g0 in range(0, C, GROUP):
            gsz = min(GROUP, C - g0)
            grp = [rng.gauss(0, 0.3) for _ in range(gsz)]
            m = max(abs(w) for w in grp) or 1.0
            for w in grp:
                d = clip(int(round(w / m * H)) + H, 0, K - 1)
                wq = math.floor(m * (d - H) / H * (1 << FRAC))
                wq += rng.randint(-args.sigma_lsb, args.sigma_lsb)
                if rng.random() < args.stuck_p:
                    wq = 32760 if w >= 0 else -32760
                    n_stuck += 1
                row.append(clip(wq, -32760, 32760))
        W0.append(row)

    G = [clip(int(round(256 * (1 + rng.gauss(0, args.col_gamma)))), 1, 255)
         for _ in range(C)]
    X = [[clip(int(round(rng.gauss(0, 0.5) * (1 << FRAC))), -16000, 16000)
          for _ in range(C)] for _ in range(NT)]

    # --- pipeline (python twin of the RTL) ----------------------------------
    def run(rn_pow: int):
        W = [row[:] for row in W0]
        V = [0] * R
        lfsr = args.lfsr_seed & 0xFFFF
        ys, spks = [], []
        for ti in range(NT):
            tok = X[ti]
            yq_list = []
            for r in range(R):
                acc = 0
                for c in range(C):
                    xq = clip(tok[c] >> SHR_IN, -L_IN, L_IN) << SHR_IN
                    xg = (xq * G[c]) >> 7
                    acc += W[r][c] * xg
                n = 0
                if rn_pow >= 2:
                    n = (lfsr & ((1 << rn_pow) - 1)) - ((1 << (rn_pow - 1)) - 1)
                    lfsr = xorshift16(lfsr)
                yq = clip((acc >> args_ashift) + n, -L_ADC, L_ADC)
                yq_list.append(yq)
            spkword = 0
            for r in range(R):
                cand = (V[r] >> args.leak_shift) + yq_list[r]
                spk = lif and cand >= args.thresh
                V[r] = 0 if spk else cand
                if spk:
                    spkword |= 1 << r
            ys.append([yq << args_ashift for yq in yq_list])
            spks.append(spkword)
            if (ti + 1) % args.drift_every == 0 and ti + 1 < NT:
                for r in range(R):
                    for c in range(C):
                        W[r][c] = W[r][c] - (W[r][c] >> args.drift_shift)
        return ys, spks, V

    # OUT_SHIFT: rails >= max clean accumulator (calibration, like the ADC
    # range select on a real board). Drift only shrinks weights, so the
    # clean pass bounds all later passes.
    maxacc = 0
    for r in range(R):
        for c in range(C):
            pass
    for ti in range(NT):
        for r in range(R):
            acc = 0
            for c in range(C):
                xq = clip(X[ti][c] >> SHR_IN, -L_IN, L_IN) << SHR_IN
                acc += W0[r][c] * ((xq * G[c]) >> 7)
            maxacc = max(maxacc, abs(acc))
    assert maxacc < (1 << 39), "accumulator overflow — shrink the model"
    args_ashift = args.out_shift or 4
    if not args.out_shift:
        while (L_ADC << args_ashift) < maxacc and args_ashift <= 40:
            args_ashift += 1

    ys, spks, V = run(args.rn_pow)

    # --- emit ----------------------------------------------------------------
    with open("golden_w.memh", "w") as fh:
        for r in range(R):
            fh.write(" ".join(f"{v & 0xFFFF:04x}" for v in W0[r]) + "\n")
    with open("golden_g.memh", "w") as fh:
        fh.write(" ".join(f"{v:02x}" for v in G) + "\n")
    with open("golden_tokens.hex", "w") as fh:
        for tok in X:
            fh.write(" ".join(f"{v & 0xFFFF:04x}" for v in tok) + "\n")
    with open("golden_expected.hex", "w") as fh:
        for accs in ys:
            fh.write(" ".join(f"{v & 0xFFFFFFFFFFFF:010x}" for v in accs) + "\n")
    nb = (R + 3) // 4 * 4 // 4
    with open("golden_spikes.hex", "w") as fh:
        for s in spks:
            fh.write(f"{s:0{nb}x}\n")
    with open("golden_vmem.hex", "w") as fh:
        for v in V:
            fh.write(f"{v & 0xFFFFFFFF:08x}\n")
    with open("sim_flags.txt", "w") as fh:
        fh.write(f"-GROWS={R} -GCOLS={C} -GWW={WW} -GDACB={args.dacb} "
                 f"-GADCB={args.adcb} -GOUT_SHIFT={args_ashift} "
                 f"-GNTOK={NT} -GRN_POW={args.rn_pow} "
                 f"-GDRIFT_SHIFT={args.drift_shift} -GLIF_EN={int(lif)} "
                 f"-GLEAK_SHIFT={args.leak_shift} -GTHRESH={args.thresh} "
                 f"-GLFSR_SEED={args.lfsr_seed} -GDRIFT_EVERY={args.drift_every}\n")

    yq_all = [y >> args_ashift for tok in ys for y in tok]
    yq_nz = sum(1 for v in yq_all if v != 0)
    n_spikes = sum(bin(s).count("1") for s in spks)
    print(f"image: {R}x{C} k{K} grid Q4.12; sigma_lsb={args.sigma_lsb} "
          f"stuck={n_stuck} gamma={args.col_gamma}")
    print(f"read noise k={args.rn_pow} (+-{(1 << max(args.rn_pow,1) - 1) - 1} "
          f"ADC-LSB), drift 1/2^{args.drift_shift} every {args.drift_every}, "
          f"LIF={'on' if lif else 'off'} (leak 1/2^{args.leak_shift}, "
          f"thresh {args.thresh})")
    print(f"maxacc={maxacc} -> OUT_SHIFT={args_ashift}; yq nonzero: "
          f"{yq_nz}/{len(yq_all)}, |yq|max={max(abs(v) for v in yq_all)}; "
          f"spikes fired: {n_spikes} over {NT} tokens")
    if args.sweep:
        base, _, _ = run(1)
        rails = L_ADC << args_ashift
        print("\ntile-level read-noise survival sweep (vs k=1 = noiseless):")
        print("  k   rms_err/rails   spikes")
        for k in range(1, 7):
            yk, sk, _ = run(k)
            mse = sum((yk[t][r] - base[t][r]) ** 2
                      for t in range(NT) for r in range(R))
            rms = math.sqrt(mse / (NT * R)) / rails
            ns = sum(bin(s).count("1") for s in sk)
            print(f"  {k}   {rms:12.5f}   {ns}")

    print("golden files written; run `make gate`")


if __name__ == "__main__":
    main()
