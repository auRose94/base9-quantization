# eq17 — RQ10 stage B leg 2: the exact runtime (docs/09)

subject: eq9_QAT_k27emb99.k9, 2×512 tokens; state = (num, den, E, J)
per row with (E, J) exponent tracks absorbing scale; exact per-row
reduction/renorm after every site; fp-mixed RoPE/attention/silu
with declared 2^30 fixed-point intakes; λ dyadic 2^-10.

- P38 (PASS): order-identical logits=True; width tracks identical=True; widening events identical=True; max|Δlogits−fp32ref| = 3.802e+01.
- P37 (FAIL): worst num 790.9 digits, worst den 784.9 digits vs the ≤20-digit bar; int64 widened at ['layers.0.wq', 'layers.0.wk', 'layers.0.w1', 'layers.0.w3'].

| site | num digits | den digits |
|---|---|---|
| embed | 5.4 | 1.9 |
| L0.attnorm | 12.9 | 3.5 |
| L0.qkv | 18.9 | 11.0 |
| L0.res_mid | 17.4 | 3.2 |
| L0.norm | 17.7 | 16.7 |
| L0.w1w3 | 23.7 | 24.3 |
| L0.res_out | 47.6 | 21.1 |
| L1.attnorm | 58.4 | 48.6 |
| L1.qkv | 64.0 | 55.8 |
| L1.res_mid | 47.6 | 21.1 |
| L1.norm | 58.0 | 48.6 |
| L1.w1w3 | 63.4 | 56.2 |
| L1.res_out | 96.2 | 69.4 |
| L2.attnorm | 106.9 | 97.5 |
| L2.qkv | 113.3 | 104.7 |
| L2.res_mid | 96.2 | 69.4 |
| L2.norm | 106.3 | 97.2 |
| L2.w1w3 | 111.7 | 104.7 |
| L2.res_out | 193.1 | 166.2 |
| L3.attnorm | 203.2 | 194.3 |
| L3.qkv | 209.5 | 201.9 |
| L3.res_mid | 193.1 | 166.2 |
| L3.norm | 203.5 | 194.3 |
| L3.w1w3 | 208.8 | 201.9 |
| L3.res_out | 387.7 | 359.6 |
| L4.attnorm | 397.8 | 388.7 |
| L4.qkv | 403.8 | 396.2 |
| L4.res_mid | 387.7 | 359.6 |
| L4.norm | 397.8 | 388.7 |
| L4.w1w3 | 403.2 | 396.2 |
| L4.res_out | 775.4 | 748.0 |
| logits | 790.9 | 784.9 |
