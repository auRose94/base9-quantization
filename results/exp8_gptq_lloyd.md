# exp8 — compensated fitted codebooks (GPTQ + Lloyd–Max, 9 centers, g=64)

Cross-check: 'RTN fp32-codebook' must equal exp4b's LM9-g64 = 40.534.
fp32 baseline 40.663; noise ±0.3 ppl. Codebook side info:
9 centers/group of 64 → fp32 = 4.50 b/param,
fp16 = 2.25 b/param (counted). Pre-registered P11:
fp16-codebook ppl within noise of fp32-codebook ppl.

| phase | ppl | digit rANS | eff b/param | lin rel MSE | wall |
|---|---|---|---|---|---|
| RTN fp32-codebook | 40.534 | 2.769 | 7.269 | 0.02472 | 16s |
| GPTQ fp32-codebook | 41.089 | 2.790 | 7.290 | 0.03427 | 56s |
| RTN fp16-codebook | 40.622 | 2.769 | 5.019 | 0.02472 | 17s |
| GPTQ fp16-codebook | 41.417 | 2.790 | 5.040 | 0.03432 | 56s |

P11 (fp16 within ±0.3 of fp32): **FAIL**

wall 144s
