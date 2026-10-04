# exp6 — GPTQ-style Hessian compensation on the exp4b grids (roneneldan/TinyStories-33M)

Calibration: 64 TRAIN-split blocks (1024 tok), sequential per-layer
Hessians over already-quantized predecessors. Grids/scales/side-info
identical to exp4b (g=64, static scales from original W; +0.50 b/param). Only compensation differs. fp32 baseline 40.663; noise floor ±0.3 ppl.

| scheme | levels | RTN ppl | GPTQ ppl | Δ ppl | eff b/param | GPTQ eff | lin rel MSE (GPTQ) | RTN cross-check |
|---|---|---|---|---|---|---|---|---|
| int4-15 g64 | 15 | 42.334 | 42.119 | -0.22 | 3.984 | 3.987 | 0.01660 | no exp4b row (new baseline) |
| 8-level g64 | 8 | 58.213 | 46.843 | -11.37 | 3.018 | 3.028 | 0.06614 | matches exp4b |
| 9-level ninths g64 | 9 | 54.679 | 43.644 | -11.04 | 3.201 | 3.209 | 0.05054 | matches exp4b |
| ternary absmean g64 | 3 | 1029.694 | 4027.256 | +2997.56 | 2.083 | 2.076 | 0.35734 | no exp4b row (new baseline) |

wall 275s
