# exp7 — multi-digit repeating grids: the b^L−1 family (roneneldan/TinyStories-33M)

Odd symmetric grids, per-(row, group-of-64) scales (+0.50 b/p side),
entropy-coded with M=4096 freq tables; eval 300x1024
TinyStories-val tokens; fp32 baseline 40.663, noise ±0.3.
Law: 2H+1 = b^L−1 levels ⇔ weights are L-digit repeating decimals
(w = m·(2·0.b̄ − 1)); exp4's champion int4-15 IS the binary L=4 grid.

| grid | family | digit entropy | rANS b/p | eff b/p | ppl | rel MSE |
|---|---|---|---|---|---|---|
| 7-level | 2^3-1 (base-2 L=3) | 2.308 | 2.308 | 2.808 | 76.283 | 0.06548 |
| 9-level | 10^1-1 (base-10 L=1) | 2.701 | 2.701 | 3.201 | 54.679 | 0.03690 |
| 15-level | 2^4-1 (base-2 L=4) | 3.484 | 3.484 | 3.984 | 42.334 | 0.01206 |
| 63-level | 2^6-1 (base-2 L=6) | 5.598 | 5.598 | 6.098 | 40.646 | 0.00062 |
| 99-level | 10^2-1 (base-10 L=2) | 6.249 | 6.250 | 6.750 | 40.565 | 0.00025 |
| 127-level | 2^7-1 (base-2 L=7) | 6.607 | 6.607 | 7.107 | 40.978 | 0.00015 |
| 255-level | 2^8-1 (base-2 L=8) | 7.604 | 7.605 | 8.105 | 40.452 | 0.00004 |
| 999-level | 10^3-1 (base-10 L=3) | 9.549 | 9.560 | 10.060 | 40.697 | 0.00000 |
| 1023-level | 2^10-1 (base-2 L=10) | 9.582 | 9.595 | 10.095 | 40.662 | 0.00000 |

## GPTQ-compensated grids

| grid | RTN ppl | GPTQ ppl | Δ ppl | GPTQ eff b/p |
|---|---|---|---|---|
| 9-level | 54.679 | 43.644 | -11.04 | 3.209 |
| 15-level | 42.334 | 42.119 | -0.22 | 3.986 |
| 99-level | 40.565 | 40.727 | +0.16 | 6.750 |

## Pre-registered verdicts

- P6 (entropy < log2 levels): **PASS**
- P7 (ppl strictly decreasing in levels): **FAIL**
- P8 (|Δ99| < |Δ9|, Δ99 < 0): **FAIL**
- P9 (multi-digit print round trip bit-exact + identity): **PASS**
- P10 (99-g64 below 63↔127 interpolation): **PASS**

Multi-digit print sample: results/multidigit_print_sample.txt
(64 weights = 409 chars; extrapolated L=2 text ~181 MB for all linear weights — gzip/rANS
reclaim it, same as exp5.)

wall 323s
