# exp26 — 7B + GPTQ: K9 vs q4_k_m at scale

Qwen/Qwen2.5-Coder-7B-Instruct (7,615,616,512 params). GPTQ (act-order, damp 0.05, 64x1024 calib) on every body matrix with in <= 4096 (71% of body params);
down_proj (in=18944) is RTN — its Hessian does not fit the card — and so
are embed/lm_head (k=99). Same 8x1024 windows as exp24/25; bf16 reference
code 3.493.

| k body | k embed | MB | b/param | code ppl | Δcode vs bf16 | wiki ppl |
|---|---|---|---|---|---|---|
| 9 | 99 | 3129.5 | 3.287 | 3.826 | +0.333 ± 0.034 | 11.743 |

Reference points (exp24 RTN / exp25 GGUF): K9 k=15 RTN 3740.2 MB / +0.117;
K9 k=9 RTN 3103.7 MB / +0.374; GGUF q4_k_m 4683.1 MB / +0.070;
q2_k 3015.9 MB / +0.561.

## Reading

- The k=9 probe: at 1.5B the largest GPTQ gain was on this coarse grid
  (-0.449 code ppl), so it is the one setting where compensation still has
  error to recover at 7B. RTN reference is 3103.7 MB / +0.374 (exp24).

wall 1255s
