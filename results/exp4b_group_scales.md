# exp4b — per-group scales (g=64) on roneneldan/TinyStories-33M

Group scales along the input dim (64 weights/scale). Side info
counted: uniform grids store 1 fp32 scale per group (0.50 b/param);
Lloyd–Max stores 9 fp32 fitted centers per group (4.50 b/param — 
fp16 centers would be 2.25, estimates without re-eval).
Companion to exp4 (per-row); same eval (300x1024 tokens),
fp32 baseline 40.663 (matches exp4 exactly).

| scheme | levels | digit rANS | side fp32 | eff (fp32 side) | ppl | lin rel MSE |
|---|---|---|---|---|---|---|
| int4 (15) g64 | 15 | 3.484 | 0.50 | 3.984 | 42.334 | 0.01206 |
| 8-level g64 | 8 | 2.518 | 0.50 | 3.018 | 58.213 | 0.04844 |
| 9-level ninths g64 (uniform) | 9 | 2.701 | 0.50 | 3.201 | 54.679 | 0.03690 |
| 9-level Lloyd-Max g64 (fitted) | 9 | 2.770 | 4.50 | 7.270 | 40.534 | 0.02472 |

fp16-side estimates (no re-eval): int4 (15) g64 3.73 b/param; 8-level g64 2.77 b/param; 9-level ninths g64 (uniform) 2.95 b/param; 9-level Lloyd-Max g64 (fitted) 5.02 b/param

wall 55s
