# exp24 — K9 at 7B (Qwen2.5-Coder-7B-Instruct)

7,615,616,512 params; body 6,525,288,448 (k=body), embed+lm_head 1,089,994,752
(k=embed, untied). 8 windows x 1024 tok per corpus.
**Baseline is bf16, not fp32** (7B fp32 = 30 GB does not fit a 16 GB
card), so these deltas are the marginal cost of K9 on top of bf16.
Bytes are the real K9 file written by k9.py (C coder, parallel encode).

| k body | k embed | MB | b/param | code ppl | Δcode vs bf16 | wiki ppl |
|---|---|---|---|---|---|---|
| 15 | 99 | 3740.2 | 3.929 | 3.610 ± 0.248 | +0.117 ± 0.015 | 10.354 |
| 9 | 99 | 3103.7 | 3.260 | 3.867 ± 0.263 | +0.374 ± 0.043 | 11.268 |

## Reading

- Compare against the published GGUF sizes for this model (q2_k / q4_k_m
  / q8_0) and the 1.5B frontier: the question is whether the K9 advantage
  holds, shrinks, or grows with scale.
- Estimated vs real bytes confirm the entropy accounting at 7B.

wall 145s
