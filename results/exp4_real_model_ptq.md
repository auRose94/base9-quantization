# exp4 — real-model PTQ: roneneldan/TinyStories-33M

28,311,552 linear params quantized per output row; embeddings/wpe/lm_head/biases/norms stay fp32 (40,202,496 params side info). Eval = 300x1024 tokens from TinyStories validation.

Effective b/param = rANS digits + fp32 side info (row scales; for
Lloyd–Max, 9 fitted center floats per row — codebook cost counted).

| scheme | levels | raw b/param | entropy | digits rANS | side info | eff total | ppl | lin rel MSE | model MB |
|---|---|---|---|---|---|---|---|---|---|
| fp32 baseline | — | 32.0 | — | — | 0.000 | — | 40.663 | 0.00000 | 274.1 |
| int8 (255 sym) | 255 | 8.0 | 7.204 | 7.990 | 0.031 | 8.021 | 40.628 | 0.00007 | 189.2 |
| int4 (15 sym) | 15 | 4.0 | 3.041 | 3.047 | 0.031 | 3.078 | 45.008 | 0.02341 | 171.7 |
| 16-level uniform | 16 | 4.0 | 3.138 | 3.144 | 0.031 | 3.176 | 48.041 | 0.02042 | 172.0 |
| 8-level uniform | 8 | 3.0 | 2.087 | 2.089 | 0.031 | 2.120 | 81.003 | 0.09523 | 168.3 |
| ternary absmean (3) | 3 | 2.0 | 1.583 | 1.583 | 0.031 | 1.615 | 992.279 | 0.26842 | 166.5 |
| 9-level ninths (uniform) | 9 | 4.0 | 2.265 | 2.267 | 0.031 | 2.299 | 93.333 | 0.07102 | 168.9 |
| 9-level Lloyd-Max (fitted) | 9 | 4.0 | 2.866 | 2.866 | 0.281 | 3.147 | 47.164 | 0.03034 | 171.9 |
| 27-level uniform | 27 | 5.0 | 3.923 | 3.939 | 0.031 | 3.970 | 42.927 | 0.00679 | 174.9 |

## Ternary pair (base-9 digit) coder rows

- ternary absmean (3) | ternary PAIRS -> base-9 digits: entropy 1.583, rANS 1.584 b/param

Perplexity noise floor at this eval size is roughly ±0.3 ppl;
differences below that are read as parity.

total wall 115s
