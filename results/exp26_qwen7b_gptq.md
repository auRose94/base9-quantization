# exp26 — 7B + GPTQ: K9 vs q4_k_m at scale

Qwen/Qwen2.5-Coder-7B-Instruct (7,615,616,512 params). GPTQ (act-order, damp 0.05, 64x1024 calib) on every body matrix with in <= 4096 (71% of body params);
down_proj (in=18944) is RTN — its Hessian does not fit the card — and so
are embed/lm_head (k=99). Same 8x1024 windows as exp24/25; bf16 reference
code 3.493.

| k body | k embed | MB | b/param | code ppl | Δcode vs bf16 | wiki ppl |
|---|---|---|---|---|---|---|
| 15 | 99 | 3760.2 | 3.950 | 3.608 | +0.115 ± 0.020 | 10.484 |

Reference points (exp24 RTN / exp25 GGUF): K9 k=15 RTN 3740.2 MB / +0.117;
K9 k=9 RTN 3103.7 MB / +0.374; GGUF q4_k_m 4683.1 MB / +0.070;
q2_k 3015.9 MB / +0.561.

## Reading

- If GPTQ moves k=15 from +0.117 toward <= +0.070, K9 beats q4_k_m on both
  axes at 7B (20% smaller AND better) — the clean form of the result.

wall 1219s
