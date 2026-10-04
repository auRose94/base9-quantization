# exp15 — alphabet sweep + the EC-SQ bound

Sources: Gaussian(0,1) and standardized Student-t(nu=4), n=200,000.
`uniform` is exp3's linspace(-m,m,k); **k=9 is the ninths grid**. `LloydMax`
is the fixed-rate MSE-optimal grid. `EC-SQ` is the entropy-constrained
scalar quantizer (min D s.t. H<=R, optimized over level count) — the true
scalar ceiling once entropy coding (rANS) is allowed. `eff = MSE / EC-SQ`
at the same entropy; 1.0 = rate-optimal, larger = rate wasted.

## gaussian

| k | uniform MSE | uniform H | Lloyd–Max MSE | LM H | EC-SQ @H_uni | eff uniform | eff LM |
|---|---|---|---|---|---|---|---|
| 3 | 0.958626 | 0.112 | 0.190561 | 1.535 | 0.916310 | 1.05 | 1.09 |
| 4 | 1.115029 | 1.011 | 0.117890 | 1.910 | 0.359854 | 3.10 | 1.13 |
| 5 | 0.491570 | 0.960 | 0.080256 | 2.203 | 0.382423 | 1.29 | 1.18 |
| 6 | 0.336101 | 1.270 | 0.058208 | 2.442 | 0.244821 | 1.37 | 1.17 |
| 7 | 0.230334 | 1.460 | 0.044330 | 2.645 | 0.193970 | 1.19 | 1.20 |
| 8 | 0.169834 | 1.648 | 0.034807 | 2.821 | 0.147674 | 1.15 | 1.18 |
| 9 | 0.129681 | 1.814 | 0.028026 | 2.978 | 0.117051 | 1.11 | 1.23 |
| 10 | 0.102647 | 1.966 | 0.023147 | 3.123 | 0.096606 | 1.06 | 1.21 |
| 11 | 0.083031 | 2.107 | 0.019360 | 3.247 | 0.077634 | 1.07 | 1.20 |
| 12 | 0.068778 | 2.233 | 0.016497 | 3.373 | 0.065680 | 1.05 | 1.24 |
| 13 | 0.057713 | 2.353 | 0.014211 | 3.471 | 0.056627 | 1.02 | 1.23 |
| 14 | 0.049204 | 2.463 | 0.012359 | 3.577 | 0.048061 | 1.02 | 1.21 |
| 15 | 0.042466 | 2.564 | 0.010842 | 3.675 | 0.040706 | 1.04 | 1.23 |
| 16 | 0.036883 | 2.661 | 0.009611 | 3.762 | 0.036322 | 1.02 | 1.24 |
| 17 | 0.032514 | 2.750 | 0.008547 | 3.842 | 0.032537 | 1.00 | 1.24 |
| 18 | 0.028701 | 2.834 | 0.007692 | 3.930 | 0.028918 | 0.99 | 1.25 |
| 19 | 0.025701 | 2.915 | 0.006960 | 4.009 | 0.025455 | 1.01 | 1.25 |
| 20 | 0.023026 | 2.991 | 0.006354 | 4.059 | 0.022329 | 1.03 | 1.23 |
| 21 | 0.020829 | 3.063 | 0.005767 | 4.119 | 0.020613 | 1.01 | 1.22 |
| 22 | 0.018814 | 3.133 | 0.005282 | 4.178 | 0.018921 | 0.99 | 1.20 |
| 23 | 0.017166 | 3.199 | 0.004832 | 4.236 | 0.017295 | 0.99 | 1.18 |
| 24 | 0.015774 | 3.261 | 0.004449 | 4.309 | 0.015770 | 1.00 | 1.20 |
| 25 | 0.014399 | 3.322 | 0.004109 | 4.372 | 0.014280 | 1.01 | 1.22 |
| 26 | 0.013330 | 3.381 | 0.003812 | 4.425 | 0.013116 | 1.02 | 1.23 |
| 27 | 0.012290 | 3.436 | 0.003571 | 4.460 | 0.012158 | 1.01 | 1.20 |
| 28 | 0.011422 | 3.490 | 0.003329 | 4.505 | 0.011306 | 1.01 | 1.19 |
| 29 | 0.010574 | 3.542 | 0.003135 | 4.540 | 0.010623 | 1.00 | 1.17 |
| 30 | 0.009900 | 3.592 | 0.002930 | 4.585 | 0.009964 | 0.99 | 1.17 |
| 31 | 0.009235 | 3.640 | 0.002757 | 4.624 | 0.009287 | 0.99 | 1.15 |
| 32 | 0.008633 | 3.687 | 0.002604 | 4.659 | 0.008603 | 1.00 | 1.14 |

## student_t4

Linear-uniform grids are **not heavy-tail-robust**: with the raw
absmax rule k=4 gives rel MSE 88.96, and even
with a p99.99 clipped range it is 10.26. The uniform columns are omitted here;
the meaningful comparison on this source is the fitted grid (Lloyd–Max)
against the entropy-constrained envelope.

| k | Lloyd–Max MSE | LM H | EC-SQ @H_LM | eff LM |
|---|---|---|---|---|
| 3 | 0.318312 | 1.393 | 0.170677 | 1.86 |
| 4 | 0.224064 | 1.678 | 0.117714 | 1.90 |
| 5 | 0.167185 | 1.895 | 0.088009 | 1.90 |
| 6 | 0.130430 | 2.100 | 0.066072 | 1.97 |
| 7 | 0.113918 | 2.113 | 0.065091 | 1.75 |
| 8 | 0.090359 | 2.306 | 0.050001 | 1.81 |
| 9 | 0.073839 | 2.442 | 0.041132 | 1.80 |
| 10 | 0.060880 | 2.572 | 0.035175 | 1.73 |
| 11 | 0.051510 | 2.700 | 0.029297 | 1.76 |
| 12 | 0.044027 | 2.826 | 0.024918 | 1.77 |
| 13 | 0.038035 | 2.913 | 0.022445 | 1.69 |
| 14 | 0.033256 | 3.009 | 0.019711 | 1.69 |
| 15 | 0.033184 | 3.036 | 0.018884 | 1.76 |
| 16 | 0.029770 | 3.038 | 0.018824 | 1.58 |
| 17 | 0.025920 | 3.129 | 0.016672 | 1.55 |
| 18 | 0.025676 | 3.128 | 0.016698 | 1.54 |
| 19 | 0.022521 | 3.239 | 0.014864 | 1.52 |
| 20 | 0.022520 | 3.241 | 0.014829 | 1.52 |
| 21 | 0.020237 | 3.304 | 0.013795 | 1.47 |
| 22 | 0.018860 | 3.337 | 0.013389 | 1.41 |
| 23 | 0.019027 | 3.323 | 0.013542 | 1.41 |
| 24 | 0.018858 | 3.337 | 0.013391 | 1.41 |
| 25 | 0.018777 | 3.343 | 0.013325 | 1.41 |
| 26 | 0.016713 | 3.421 | 0.012486 | 1.34 |
| 27 | 0.016773 | 3.420 | 0.012503 | 1.34 |
| 28 | 0.014943 | 3.492 | 0.011602 | 1.29 |
| 29 | 0.014620 | 3.494 | 0.011576 | 1.26 |
| 30 | 0.013090 | 3.576 | 0.010715 | 1.22 |
| 31 | 0.013106 | 3.593 | 0.010568 | 1.24 |
| 32 | 0.011881 | 3.663 | 0.009696 | 1.23 |

## Pre-registered verdicts (Gaussian)

- P25 (odd-k zero-level sawtooth at coarse k, overtaken at fine k): **PASS** (uni3 0.9586 < uni4 1.1150; uni9 0.1297 < uni8 0.1698; uni16 0.0369 < uni15 0.0425)
- P26 (uniform-9 not anomalous: eff within ±0.02 of k=7,11): **FAIL** (eff 7/9/11 = 1.187/1.108/1.070)
- P27 (uniform-9 > 2x worse than EC-SQ at its own entropy): **FAIL** (eff 1.11)
- P28 (Lloyd–Max-9 within 15% of EC-SQ envelope): **FAIL** (eff LM9 1.231)

Note: `eff` far above 1.0 means the empirical grid wastes rate versus an
EC-SQ of the same output entropy — i.e. the win available is *grid design
under an entropy constraint*, not the base-9 digit identity.

## Reading (Gaussian unless stated)

1. **Uniform ninths is near rate-optimal at its own entropy.** At H=1.81 b the uniform grid sits only 11% above the EC-SQ bound (eff 1.11); P27's 2x prediction fails. The grid *placement* is not what limits the 9-level operating point — its low rate (1.81 b) is.
2. **For k>=17 uniform + entropy coding is essentially optimal** (eff 1.03 at k=20): the classical asymptotic optimality of uniform scalar quantization, now on a measured entropy axis.
3. **Fixed-rate Lloyd-Max is not the right ceiling.** At matched entropy it sits 23% above the EC-SQ envelope at k=9 on Gaussian and 80% on the heavy tail — because at a given *entropy* the best grid is a more non-uniform design than the MSE-optimal fixed-rate one. P28 fails.
4. **The lever is entropy-constrained grid design, not the base-9 identity.** Fit the 9-level codebook to minimize D subject to H <= R (a parametric or
 shared codebook is a cheap approximation) rather than D at fixed rate —
 that is where the 14-25% (Gaussian) / 80% (heavy tail) sits.
5. **Heavy tails punish linear grids.** Coarse even-k uniform grids collapse (no zero level, plus tail mass beyond the range): k=4 rel MSE 89 under raw absmax scaling. This is *why* the project's per-(row,group) scales matter on real weights.
