# exp1 — bits per parameter (n = 200000 Gaussian weights, seed 42)

All bit figures are bits/parameter; `entropy` is the empirical symbol
entropy, `rans` is actual coder output (roundtrip-asserted), `mse` is
relative MSE (mean sq err / weight variance).

| scheme | levels | raw index | alphabet | entropy | rANS | rel MSE |
|---|---|---|---|---|---|---|
| int8 symmetric | 255 | 8.000 | 7.994 | 6.715 | 7.440 | 0.000129 |
| int4 symmetric (15 levels) | 15 | 4.000 | 3.907 | 2.564 | 2.594 | 0.042466 |
| ternary absmean (BitNet b1.58-style) | 3 | 2.000 | 1.585 | 1.583 | 1.584 | 0.264413 |
| 9-level ninths grid (step m/4) | 9 | 4.000 | 3.170 | 1.814 | 1.831 | 0.129681 |

## Ternary pair (base-9 digit) block

```
ternary pair stream (2 trits -> 1 base-9 digit): joint entropy 1.5833 b/param, rANS on pair stream 1.5836 b/param
```
