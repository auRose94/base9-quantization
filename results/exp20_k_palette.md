# exp20 — K9 k-palette (mixed-precision grid allocation)

Qwen2.5-Coder-1.5B-Instruct, same 8-window harness as exp17-19. Body
tensors GPTQ'd (damp 0.05, act-order), embeddings RTN. Allocation is
greedy on Hessian-diagonal-weighted quantization error per coded byte.

External reference points (exp19, same windows): **q4_k_m 1117.3 MB,
code +0.170**; NF4 999.5 MB, +0.270; q2_k 752.9 MB, +1.426; fp32
4.492 code.

| palette | MB | k=max tensors | embed k | code ppl | Δcode vs fp32 |
|---|---|---|---|---|---|
| all-k9 (base) | 538.3 | 0 | 9 | 5.447 ± 0.443 | +0.956 ± 0.137 |
| all-k9 + embed99 | 641.8 | 0 | 99 | 5.149 ± 0.402 | +0.657 ± 0.106 |
| greedy <= 700 MB | 700.6 | 19 | 9 | 4.972 ± 0.410 | +0.481 ± 0.058 |
| greedy <= 850 MB | 850.5 | 126 | 9 | 4.835 ± 0.409 | +0.343 ± 0.048 |
| greedy <= 1000 MB | 1000.4 | 191 | 9 | 4.745 ± 0.405 | +0.253 ± 0.043 |
| greedy <= 1150 MB | 1114.5 | 196 | 99 | 4.504 ± 0.367 | +0.012 ± 0.004 |
| greedy <= 1350 MB | 1114.5 | 196 | 99 | 4.504 ± 0.367 | +0.012 ± 0.004 |
| all-k63 + embed99 | 1114.5 | 196 | 99 | 4.504 ± 0.367 | +0.012 ± 0.004 |
| attn63/mlp15 + embed99 | 809.4 | 112 | 99 | 4.608 ± 0.363 | +0.116 ± 0.031 |

## Reading

- Compare each palette against **q4_k_m (1117.3 MB, +0.170)**: any row
  that is both smaller and lower Δcode dominates the most-used quant.
- `all-k9 (base)` should reproduce exp17's uniform body9+embed9 point;
  the greedy rows show whether allocation buys quality efficiently.

wall 1182s
