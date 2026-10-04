# exp19 — external bytes-matched baselines (GGUF + NF4)

Qwen2.5-Coder-1.5B-Instruct, same 8-window harness as exp17/exp18.
External formats are dequantized into the fp32 HF model and evaluated
on identical windows; bytes are the authoritative file/packed sizes.
K9 reference rows (same windows, exp18): body9-GPTQ+embed99 = 641.9 MB,
code ppl 5.149 (+0.657±0.106); body9+embed99 RTN = 1387.0 MB body-only.

| format | MB | b/param | wiki ppl | code ppl | Δcode vs fp32 (paired) |
|---|---|---|---|---|---|
| fp32 baseline | 6174.9 | 32.00 | 14.064 ± 2.336 | 4.492 ± 0.366 | +0.000 ± 0.000 |
| GGUF q8_0 | 1894.5 | 9.82 | 14.419 ± 2.389 | 4.578 ± 0.377 | +0.086 ± 0.014 |
| GGUF q5_k_m | 1285.5 | 6.66 | 14.581 ± 2.426 | 4.602 ± 0.375 | +0.110 ± 0.017 |
| GGUF q4_k_m | 1117.3 | 5.79 | 14.800 ± 2.460 | 4.662 ± 0.381 | +0.170 ± 0.027 |
| GGUF q4_0 | 1066.2 | 5.53 | 14.979 ± 2.497 | 4.735 ± 0.368 | +0.243 ± 0.037 |
| GGUF q2_k | 752.9 | 3.90 | 20.085 ± 3.512 | 5.918 ± 0.458 | +1.426 ± 0.178 |
| bitsandbytes NF4 | 999.5 | 5.18 | 15.028 ± 2.564 | 4.761 ± 0.392 | +0.270 ± 0.053 |
| **K9 full (body9-GPTQ+embed99)** *(exp18, same windows)* | **641.9** | **3.23** | 19.603 ± 3.642 | **5.149** ± 0.402 | **+0.657 ± 0.106** |

## Reading

1. **K9 dominates the smallest deployed quant.** q2_k is 752.9 MB at
   +1.426 code ppl; K9-full is smaller (641.9 MB) AND better (+0.657).
   That is the first head-to-head win over a shipped format.
2. **K9 is 1.74x smaller than q4_k_m** (641.9 vs 1117.3 MB) at +0.657 vs
   +0.170 — K9 is the smallest operating point on the curve, not the best
   quality per byte. NF4 (999.5 MB, +0.270) is more byte-efficient.
3. **Quality-per-byte (Δcode/MB):** q4_k_m 1.5e-4 < NF4 2.7e-4 < K9
   1.0e-3 < q2_k 1.9e-3 — K9 sits between NF4 and q2_k, i.e. the low-bit
   end of the deployed frontier.
4. **q8_0 validates the harness**: it reproduces fp32 within ~2%
   (Δcode +0.086), so the GGUF name/shape mapping is correct.

Caveat: external formats are dequantized into the fp32 HF model and
evaluated on our windows (isolates the quantization effect); this is not
the llama.cpp runtime, though q8_0's near-parity checks the mapping.

wall 95s
