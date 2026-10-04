# exp25 — GGUF 7B baselines (dequantized into bf16, same windows)

Qwen/Qwen2.5-Coder-7B-Instruct, 8x1024 tokens per corpus — identical windows to
exp24, so these are directly comparable to the K9 7B points.

| quant | MB | b/param | code ppl | Δcode vs bf16 (paired) | wiki ppl |
|---|---|---|---|---|---|
| q4_k_m | 4683.1 | 4.92 | 3.563 | +0.070 ± 0.013 | 10.288 |
| q2_k | 3015.9 | 3.17 | 4.054 | +0.561 ± 0.076 | 11.669 |

K9 7B for comparison (exp24, same windows; bf16 reference code 3.493):
k=15 + embed99 = **3740.2 MB, code 3.610 (+0.117)**;
k=9 + embed99 = **3103.7 MB, code 3.867 (+0.374)**.

wall 77s
