# eq17 — RQ10 stage B leg 2: the exact runtime (docs/09)

subject: eq9_QAT_k27emb99.k9, 2×512 tokens; state = (num, den, E, J)
per row with (E, J) exponent tracks absorbing scale; exact per-row
reduction/renorm after every site; fp-mixed RoPE/attention/silu
with declared 2^30 fixed-point intakes; λ dyadic 2^-10.

- P38 (PASS): order-identical logits=True; width tracks identical=True; widening events identical=True; max|Δlogits−fp32ref| = 3.500e+01.
- P37 (FAIL): worst num 718.3 digits, worst den 716.7 digits vs the ≤20-digit bar; int64 widened at ['layers.0.w1', 'layers.0.w3'].

| site | num digits | den digits |
|---|---|---|
| embed | 5.4 | 1.9 |
| L0.attnorm | 8.5 | 3.5 |
| L0.qkv | 14.5 | 11.0 |
| L0.res_mid | 17.4 | 3.2 |
| L0.norm | 17.7 | 16.7 |
| L0.w1w3 | 23.7 | 24.3 |
| L0.res_out | 42.9 | 20.8 |
| L1.attnorm | 49.5 | 44.2 |
| L1.qkv | 55.2 | 51.4 |
| L1.res_mid | 42.9 | 20.8 |
| L1.norm | 48.9 | 44.2 |
| L1.w1w3 | 54.6 | 51.7 |
| L1.res_out | 87.4 | 64.7 |
| L2.attnorm | 93.7 | 88.6 |
| L2.qkv | 100.0 | 95.9 |
| L2.res_mid | 87.4 | 64.7 |
| L2.norm | 93.1 | 88.6 |
| L2.w1w3 | 98.4 | 96.2 |
| L2.res_out | 176.0 | 152.7 |
| L3.attnorm | 181.4 | 176.7 |
| L3.qkv | 187.7 | 184.2 |
| L3.res_mid | 176.0 | 152.7 |
| L3.norm | 181.7 | 177.3 |
| L3.w1w3 | 187.1 | 184.5 |
| L3.res_out | 353.0 | 329.0 |
| L4.attnorm | 359.3 | 354.3 |
| L4.qkv | 365.6 | 361.8 |
| L4.res_mid | 353.0 | 329.0 |
| L4.norm | 359.3 | 354.3 |
| L4.w1w3 | 364.7 | 361.8 |
| L4.res_out | 707.3 | 680.5 |
| logits | 718.3 | 716.7 |
