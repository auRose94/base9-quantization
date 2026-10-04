# exp3 — quantization error vs alphabet size

Uniform grids on [-m, m], Gaussian weights (seed 42), rel MSE = mean sq err / variance. raw_bits = log2(levels) alphabet bits (a fixed-width code needs ceil — entropy coding reaches the fractional figure).

| grid | levels | log2 k | rel MSE | entropy | note |
|---|---|---|---|---|---|
| uniform 3 | 3 | 1.585 | 0.958626 | 0.112 |  |
| uniform 4 | 4 | 2.000 | 1.115029 | 1.011 |  |
| uniform 5 | 5 | 2.322 | 0.491570 | 0.960 |  |
| uniform 6 | 6 | 2.585 | 0.336101 | 1.270 |  |
| uniform 7 | 7 | 2.807 | 0.230334 | 1.460 |  |
| uniform 8 | 8 | 3.000 | 0.169834 | 1.648 |  |
| uniform 9 | 9 | 3.170 | 0.129681 | 1.814 | = ninths grid k·m/4 |
| uniform 12 | 12 | 3.585 | 0.068778 | 2.233 |  |
| uniform 15 | 15 | 3.907 | 0.042466 | 2.564 |  |
| uniform 16 | 16 | 4.000 | 0.036883 | 2.661 |  |
| uniform 27 | 27 | 4.755 | 0.012290 | 3.436 |  |
| uniform 31 | 31 | 4.954 | 0.009235 | 3.640 |  |
| uniform 32 | 32 | 5.000 | 0.008633 | 3.687 |  |
| Lloyd–Max 3 | 3 | 1.585 | 0.190561 | 1.535 | MSE-optimal k levels (fitted to the distribution) |
| Lloyd–Max 4 | 4 | 2.000 | 0.117890 | 1.910 | MSE-optimal k levels (fitted to the distribution) |
| Lloyd–Max 8 | 8 | 3.000 | 0.034807 | 2.821 | MSE-optimal k levels (fitted to the distribution) |
| Lloyd–Max 9 | 9 | 3.170 | 0.028026 | 2.978 | MSE-optimal k levels (fitted to the distribution) |
| Lloyd–Max 16 | 16 | 4.000 | 0.009611 | 3.762 | MSE-optimal k levels (fitted to the distribution) |
