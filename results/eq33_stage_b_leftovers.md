# eq33 — stage-B leftovers: the P38 device axis + the fp64 twin (docs/09 RQ10)

## A. the device axis (int64 on the GPU)

- A1 order identity on cuda: **True** (ascending vs descending vs split-reversed, bit-identical), spot-exact vs bigint True; 26.66 ms per 256x512x256 int64 accumulation pass.
- A2 the device wraps int64 at 2^63 exactly like the CPU: -9223372036709301616 vs exact 9223372037000250000 (wrap=True, bigint escalation exact=True) -- so eq24's ff-product escalation is required on the device too.
- A3 torch int64 matmul on device: supported=False (unavailable) -- no int64 tensor-core path.

## B. the fp64 twin (noise vs chaos)

| site | twin fp32-vs-fp64 max|d| | scale | eq24 exact-vs-fp32 | eq24 scale | ratio |
|---|---|---|---|---|---|
| embed | 0.000e+00 | 1.33 | 2.919e-08 | 1.32 | 29194111683672470978560.00 |
| L0.attnorm | 9.537e-07 | 7.13 | 1.372e-03 | 5.63 | 1438.39 |
| L0.scores | 1.831e-04 | 267 | 7.862e-03 | 1.8e+308 | 42.94 |
| L0.res_mid | 1.463e-05 | 1.72 | 3.138e-05 | 1.56 | 2.14 |
| L0.ffnorm | 1.788e-05 | 2.95 | 1.372e-03 | 2.45 | 76.71 |
| L0.silu | 2.289e-05 | 5.67 | — | — | — |
| L0.res_out | 1.585e-05 | 6.25 | 1.259e-03 | 4.07 | 79.42 |
| logits | 6.027e-04 | 28.4 | 5.013e-02 | 20.7 | 83.18 |

**Verdict:** the recorded exact-vs-fp32 deltas EXCEED the twin's own fp32-vs-fp64 spread by 3-3e4x at the shared sites: the wobble is the exact track's own defined rounding (the per-layer regrid), not the fp32 boundary's softness -- the registered noise-vs-chaos guess is split the other way